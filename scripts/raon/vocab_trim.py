"""Trim Raon's output head to an English keep-list (a JSON {"keep": [token ids]}), before quantizing.

  vocab_trim.py --input Q8_0.gguf --keep vocab-keep.json --output Q8_0-vt.gguf

Slices output.weight rows [n_embd, n_vocab] -> [n_embd, n_keep] (row-aligned, works on quantized data) and adds
<arch>.output_token_ids (uint32, ascending): head row i scores token output_token_ids[i]. The tokenizer, token_embd
and vocab_size stay full, so input text tokenizes and embeds as before; the engine scatters the n_keep logits back
into the full vocab with -inf for dropped tokens (local raon1 patch). The imatrix is indexed by n_embd, so the
existing one still applies when quantizing the result.

The input may itself be trimmed (has <arch>.output_token_ids): the keep list must then be a subset of its ids,
and the new head is a row subset of the old one, so every tensor but output.weight stays byte-identical.
"""
import argparse
import json

import gguf
import numpy as np
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    keep = np.asarray(sorted(json.load(open(args.keep))["keep"]), dtype=np.uint32)
    reader = gguf.GGUFReader(args.input)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    n_vocab = reader.fields[f"{arch}.vocab_size"].contents()
    assert keep[-1] < n_vocab and len(np.unique(keep)) == len(keep)
    ids_key = f"{arch}.output_token_ids"
    rows = keep                                    # head rows to take from the input
    if ids_key in reader.fields:
        old = np.asarray(reader.fields[ids_key].contents(), dtype=np.int64)
        pos = np.full(n_vocab, -1, np.int64); pos[old] = np.arange(len(old))
        rows = pos[keep]
        assert (rows >= 0).all(), f"{int((rows < 0).sum())} kept ids are not in the input's trimmed head"
    for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id"):
        if k in reader.fields:
            assert reader.fields[k].contents() in set(keep.tolist()), f"{k} would be dropped"

    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)
    if gguf.Keys.General.ALIGNMENT in reader.fields:
        writer.data_alignment = reader.fields[gguf.Keys.General.ALIGNMENT].contents()
    for field in reader.fields.values():
        if field.name in (gguf.Keys.General.ARCHITECTURE, ids_key) or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    writer.add_array(f"{arch}.output_token_ids", keep.tolist())

    datas = []
    for t in reader.tensors:
        d = t.data
        if t.name == "output.weight":
            n_in = n_vocab if ids_key not in reader.fields else len(reader.fields[ids_key].contents())
            assert d.shape[0] == n_in, f"output.weight vocab axis is not outermost: {d.shape}"
            d = d[rows]
        datas.append(d)
        writer.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for d in tqdm(datas, desc="writing"):
        try:
            writer.write_tensor_data(np.ascontiguousarray(d), tensor_endianess=reader.endianess)
        except TypeError:  # ik's older gguf-py (needed for ik quant types, e.g. trimming an IQ3_KS file)
            writer.write_tensor_data(np.ascontiguousarray(d))
    writer.close()
    print(f"wrote {args.output}: output head -> {len(keep)} rows (vocab {n_vocab})")


if __name__ == "__main__":
    main()
