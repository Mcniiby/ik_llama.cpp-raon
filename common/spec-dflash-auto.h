#pragma once

// Acceptance-gated draft length for DFlash (--spec-type dflash:auto=1).
// Picks n in [0, n_cap] maximising gain(n) = (1 + E[accepted | n]) * t_plain / t(n), from EMAs of the
// per-depth conditional acceptance and of the step time per n. Below min_gain drafting is switched off
// for a cooldown that doubles on repeated switch-offs; every explore_every steps it drafts one deeper.

#include <algorithm>
#include <cstdint>
#include <vector>

struct common_spec_auto_ctl {
    // config
    bool   enabled        = false;
    int    n_cap          = 7;     // configured n_max
    double min_gain       = 1.03;  // draft only if the expected speedup beats this
    int    cooldown_base  = 32;    // target-only steps after a switch-off
    int    cooldown_max   = 512;
    int    plain_refresh  = 256;   // one target-only step this often, to track t_plain at depth
    int    explore_every  = 16;
    int    min_on_steps   = 8;     // drafted steps before the controller may switch off
    int    low_needed     = 4;     // consecutive below-threshold judgements before switching off
    double prior_p        = 0.7;
    double alpha_p_min    = 0.05;  // EMA floors
    double alpha_t_min    = 0.1;

    // state
    std::vector<double> p;     // [n_cap] conditional acceptance per depth
    std::vector<int>    p_obs;
    std::vector<double> t;     // [n_cap+1] step time EMA (us) by drafted length, t[0] = plain
    std::vector<int>    t_obs;
    int    cooldown       = 0;
    int    cooldown_len   = 0;
    int    good_streak    = 0;
    int    low_streak     = 0;
    int    since_plain    = 0;
    int    on_steps       = 0;
    uint64_t n_steps      = 0;
    double acc_ema        = 0.0;   // accepted tokens per drafted step
    double last_gain      = 0.0;
    int    last_pick      = 0;

    // stats
    uint64_t n_off_steps = 0, n_on_steps = 0, n_switch_off = 0, n_explore = 0;

    void init(int cap) {
        n_cap = std::max(1, cap);
        p.assign(n_cap, prior_p);
        p_obs.assign(n_cap, 0);
        t.assign(n_cap + 1, 0.0);
        t_obs.assign(n_cap + 1, 0);
        cooldown = 0;
        cooldown_len = cooldown_base;
        good_streak = 0;
        low_streak = 0;
        since_plain = 0;
        on_steps = 0;
        n_steps = 0;
        acc_ema = 0.0;
    }

    double expected_accepted(int n) const {
        double e = 0.0, run = 1.0;
        for (int i = 0; i < n && i < n_cap; ++i) {
            run *= p[i];
            e += run;
        }
        return e;
    }

    // step time for n drafted tokens: measured, else interpolated from the nearest measured lengths
    double step_time(int n) const {
        if (t_obs[n] >= 2) return t[n];
        int lo = -1, hi = -1;
        for (int i = n - 1; i >= 1; --i) if (t_obs[i] >= 2) { lo = i; break; }
        for (int i = n + 1; i <= n_cap; ++i) if (t_obs[i] >= 2) { hi = i; break; }
        if (lo < 0 && hi < 0) return -1.0;
        int a = lo >= 0 ? lo : hi;
        int b = -1;
        if (lo >= 0 && hi >= 0) b = hi;
        else {
            for (int i = 1; i <= n_cap; ++i) if (i != a && t_obs[i] >= 2) { b = i; break; }
        }
        if (b < 0) {
            if (t_obs[0] < 1) return t[a];
            const double slope = (t[a] - t[0]) / a; // pessimistic for n > a
            return t[a] + slope * (n - a);
        }
        const double slope = (t[b] - t[a]) / (b - a);
        return std::max(t[0] > 0 ? t[0] : 1.0, t[a] + slope * (n - a));
    }

    // best length and its gain over plain decoding; 0 when nothing can be judged yet
    int best(double * gain_out) const {
        int best_n = n_cap;
        double best_g = -1.0;
        if (t_obs[0] < 1) { if (gain_out) *gain_out = -1.0; return n_cap; }
        for (int n = 1; n <= n_cap; ++n) {
            const double tn = step_time(n);
            if (tn <= 0) continue;
            const double g = (1.0 + expected_accepted(n)) * t[0] / tn;
            if (g > best_g) { best_g = g; best_n = n; }
        }
        if (gain_out) *gain_out = best_g;
        return best_g < 0 ? n_cap : best_n;
    }

    // draft length for the next step; 0 = target-only step
    int decide() {
        if (!enabled) return n_cap;
        n_steps++;
        if (t_obs[0] < 2) { n_off_steps++; return 0; }
        if (cooldown > 0) { cooldown--; n_off_steps++; return 0; }
        if (++since_plain >= plain_refresh) { since_plain = 0; n_off_steps++; return 0; }

        double g = 0.0;
        int n = best(&g);
        last_gain = g;
        if (g >= 0 && on_steps >= min_on_steps) {
            if (g < min_gain && ++low_streak >= low_needed) {
                low_streak = 0;
                cooldown = cooldown_len - 1;
                cooldown_len = std::min(cooldown_max, cooldown_len * 2);
                good_streak = 0;
                on_steps = 0;  // re-measure after the cooldown before judging again
                n_switch_off++;
                n_off_steps++;
                return 0;
            }
            if (g >= min_gain) {
                low_streak = 0;
                if (++good_streak >= 4 * cooldown_base) cooldown_len = cooldown_base;
            }
        }
        if (explore_every > 0 && n_steps % explore_every == 0 && n < n_cap) {
            n++;
            n_explore++;
        }
        last_pick = n;
        n_on_steps++;
        return n;
    }

    // feedback for the step decide() was asked for: drafted tokens, accepted tokens, wall time
    void observe(int n_drafted, int n_accepted, double step_us) {
        if (!enabled || step_us <= 0) return;
        n_drafted = std::min(n_drafted, n_cap);
        n_accepted = std::max(0, std::min(n_accepted, n_drafted));
        auto ema = [](double & v, int & obs, double x, double floor) {
            const double a = std::max(floor, 1.0 / (obs + 2));
            v = obs == 0 ? x : v + a * (x - v);
            obs++;
        };
        ema(t[n_drafted], t_obs[n_drafted], step_us, alpha_t_min);
        if (n_drafted == 0) return;
        on_steps++;
        for (int i = 0; i < n_accepted; ++i) {
            double a = std::max(alpha_p_min, 1.0 / (p_obs[i] + 2));
            p[i] += a * (1.0 - p[i]);
            p_obs[i]++;
        }
        if (n_accepted < n_drafted) {
            double a = std::max(alpha_p_min, 1.0 / (p_obs[n_accepted] + 2));
            p[n_accepted] += a * (0.0 - p[n_accepted]);
            p_obs[n_accepted]++;
        }
        acc_ema += 0.1 * (n_accepted - acc_ema);
    }
};
