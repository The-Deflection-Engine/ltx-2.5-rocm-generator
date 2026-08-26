#!/usr/bin/env python3
"""Self-check for the VRAM self-calibration fit.

    python tests/test_vram_calibration.py

No GPU and no model needed: `fit_vram_model` is a pure function of the recorded
points, which is why it is separated from the file I/O around it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ltx_engine as eng


def pts(profile, pairs):
    return [{"profile": profile, "tokens": t, "peak_gb": g} for t, g in pairs]


def test_recovers_a_known_line():
    # peak = 8.0 + 2e-4 * tokens, sampled exactly.
    base, slope = 8.0, 2e-4
    p = pts("offload", [(t, base + slope * t) for t in (4000, 12000, 30000)])
    got = eng.fit_vram_model(p, "offload")
    assert got is not None, "should fit a clean 3-point line"
    assert abs(got[0] - base) < 1e-6, got
    assert abs(got[1] - slope) < 1e-12, got


def test_rejects_draft_sized_runs_only():
    # Good spread and a clean line, but every point is a small draft. Trusting
    # this would extrapolate ~10x past anything actually observed.
    p = pts("offload", [(400, 6.6), (1100, 6.9), (3100, 7.3)])
    assert max(x["tokens"] for x in p) < eng.VRAM_CALIB_MIN_MAX_TOKENS
    assert eng.fit_vram_model(p, "offload") is None
    # One real-size run alongside them is enough to make it usable.
    p += pts("offload", [(29000, 13.9)])
    assert eng.fit_vram_model(p, "offload") is not None


def test_ignores_other_profiles():
    p = pts("offload", [(4000, 8.8), (12000, 10.4), (30000, 14.0)])
    p += pts("resident", [(4000, 25.0), (12000, 26.0), (30000, 28.0)])
    off = eng.fit_vram_model(p, "offload")
    res = eng.fit_vram_model(p, "resident")
    assert off and res
    # The resident profile's intercept is far higher; if the two were pooled
    # neither line would come out near either truth.
    assert res[0] > off[0] + 10, (off, res)


def test_rejects_too_few_points():
    assert eng.fit_vram_model(pts("offload", [(4000, 8.8), (30000, 14.0)]), "offload") is None


def test_rejects_no_leverage():
    # Three runs at nearly the same size can't separate slope from intercept.
    p = pts("offload", [(10000, 9.8), (10500, 9.9), (11000, 10.0)])
    assert eng.fit_vram_model(p, "offload") is None


def test_rejects_insane_fit():
    # Descending peak vs tokens -> negative slope -> must be refused.
    p = pts("offload", [(4000, 14.0), (12000, 10.0), (30000, 8.0)])
    assert eng.fit_vram_model(p, "offload") is None


def test_unknown_profile_is_empty_not_crash():
    p = pts("offload", [(4000, 8.8), (12000, 10.4), (30000, 14.0)])
    assert eng.fit_vram_model(p, "resident") is None


def test_model_falls_back_to_shipped_constants():
    base, per_token, source = eng.vram_model(config={})
    assert source in ("shipped", "calibrated:offload", "calibrated:resident")
    assert base > 0 and per_token > 0


def test_config_override_wins():
    base, per_token, source = eng.vram_model(
        config={"vram_base_gb": 3.5, "vram_gb_per_token": 1e-4})
    assert (base, per_token, source) == (3.5, 1e-4, "config")


def test_estimate_is_monotonic():
    a = eng.estimate_vram_gb(1000, config={})
    b = eng.estimate_vram_gb(50000, config={})
    assert b > a


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
