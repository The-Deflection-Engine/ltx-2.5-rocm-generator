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


def _rec(threshold):
    return eng.recommended_defaults({"token_warn_threshold": threshold})


def test_defaults_never_choose_the_consent_knobs():
    # Every quality knob is the user's call at every card size, however big:
    # defaults pick a resolution and nothing else.
    for thr in (2000, 30000, 100000, 5_000_000):
        r = _rec(thr)
        for key in ("upscale", "stg_mode", "cfg_mode", "modality_scale"):
            assert key not in r, f"{key} must never be auto-enabled ({thr})"


def test_defaults_stay_inside_their_budget():
    for thr in (2000, 10000, 33837, 200000):
        r = _rec(thr)
        cost = eng.latent_tokens(r["width"], r["height"], 121, False)
        # The floor is allowed to exceed it -- there is nothing smaller.
        floor = eng.DEFAULT_RESOLUTIONS[-1]
        if (r["width"], r["height"]) != floor:
            assert cost <= thr * eng.DEFAULT_BUDGET_FRACTION, (thr, r, cost)


def test_defaults_are_monotonic_in_headroom():
    """More VRAM must never select a leaner preset than less VRAM."""
    order = {res: i for i, res in enumerate(eng.DEFAULT_RESOLUTIONS)}
    prev_rank = None
    for thr in (2000, 5000, 15000, 35000, 70000, 200000):
        r = _rec(thr)
        rank = order[(r["width"], r["height"])]
        if prev_rank is not None:
            assert rank <= prev_rank, f"threshold {thr} picked a leaner preset"
        prev_rank = rank


def test_tiny_card_gets_the_floor_not_a_crash():
    r = _rec(1)
    assert (r["width"], r["height"]) == eng.DEFAULT_RESOLUTIONS[-1]


def test_apply_is_a_noop_when_a_config_exists():
    base = {"width": 111, "height": 222, "stg_mode": False,
            "modality_scale": 1.0, "frames": 121}
    out = eng.apply_recommended_defaults(dict(base), config_exists=True)
    assert out == base, "an existing config must never be overridden"


def test_apply_seeds_a_fresh_config():
    base = {"width": 111, "height": 222, "stg_mode": False,
            "modality_scale": 1.0, "frames": 121}
    out = eng.apply_recommended_defaults(dict(base), config_exists=False)
    assert (out["width"], out["height"]) != (111, 222)
    # Seeding must not switch on anything that needs the user's consent.
    assert out["stg_mode"] is False
    assert out["modality_scale"] == 1.0
    assert out.get("upscale", False) is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
