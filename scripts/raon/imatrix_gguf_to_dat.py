# GGUF imatrix -> legacy ik .dat, with per-expert counts (mean per expert; unused experts get the
# layer's mean over used experts so quantization stays well-defined)
import struct, sys
sys.path.insert(1, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'gguf-py'))
import numpy as np
import gguf

src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
sums, counts = {}, {}
for t in r.tensors:
    d = np.array(t.data, dtype=np.float32)
    if t.name.endswith('.in_sum2'):
        sums[t.name[:-8]] = d
    elif t.name.endswith('.counts'):
        counts[t.name[:-7]] = d.reshape(-1)
chunks = 0
for k, f in r.fields.items():
    if k == 'imatrix.chunk_count':
        chunks = int(f.parts[f.data[0]][0])
n_unused = 0
with open(dst, 'wb') as out:
    out.write(struct.pack('<i', len(sums)))
    for name, s in sums.items():
        c = counts[name]
        s = s.reshape(len(c), -1) if c.size > 1 else s.reshape(1, -1)
        used = c > 0
        v = np.zeros_like(s)
        v[used] = s[used] / c[used, None]
        if (~used).any():
            n_unused += int((~used).sum())
            v[~used] = v[used].mean(axis=0) if used.any() else 1.0
        v = v.reshape(-1).astype(np.float32)
        nb = name.encode()
        out.write(struct.pack('<i', len(nb))); out.write(nb)
        out.write(struct.pack('<ii', 1, v.size)); v.tofile(out)
    ds = b'imatrix-raon-from-q4km'
    out.write(struct.pack('<ii', chunks, len(ds))); out.write(ds)
print(f'{len(sums)} entries, {chunks} chunks, {n_unused} unused expert slots filled')
