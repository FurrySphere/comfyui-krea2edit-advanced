"""The hold: the falling early pull (step weights and the hold knob), the edit mask filters (grow, shrink, feather) and what the patch node installs.

Run standalone:  python tests/test_hold.py
Or via pytest:   pytest tests/test_hold.py
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stubs import _FakeModelPatcher, _load_pack, _node   # noqa: E402

SIG = [1.0, 0.89, 0.775, 0.657, 0.535, 0.408, 0.277, 0.141, 0.0]
ALIGNED = "aligned (trained)"


# ---------------------------------------------------------------- the early pull falls with sigma

def _fall(sigma, sigma0, crossover, decay):
    c = crossover * sigma0
    return ((sigma - c) / (sigma0 - c)) ** decay


def test_decay_zero_is_the_constant_early_pull():
    mod = _load_pack()
    assert mod.step_weights(SIG, 0.9, 0.5, 4.0, 0.8, 0.0) == mod.step_weights(SIG, 0.9, 0.5, 4.0, 0.8)


def test_the_early_pull_falls_from_full_at_the_first_step_to_zero_at_the_crossover():
    mod = _load_pack()
    flat = mod.step_weights(SIG, 0.9, 0.5, 4.0, 0.8, 0.0)
    for decay in (1.0, 2.0, 2.857):
        plan = mod.step_weights(SIG, 0.9, 0.5, 4.0, 0.8, decay)
        for (early, late), (flat_early, flat_late), sigma in zip(plan, flat, SIG):
            assert math.isclose(late, flat_late)                                        # the late pull is untouched
            expected = flat_early * (_fall(sigma, SIG[0], 0.5, decay) if sigma > 0.5 else 1.0)
            assert math.isclose(early, expected, abs_tol=1e-12), (decay, sigma)
        assert math.isclose(plan[0][0], flat[0][0])                                     # full at the first step
        assert all(a[0] >= b[0] for a, b in zip(plan, plan[1:3]))                       # and falling
        assert all(e == 0 for e, _ in plan[5:])                                         # nothing after the crossover


def test_a_larger_decay_pulls_less_at_every_step():
    mod = _load_pack()
    plans = [mod.step_weights(SIG, 0.9, 0.5, 4.0, 0.8, d) for d in (0.5, 1.0, 2.0, 4.0)]
    for lighter, heavier in zip(plans[1:], plans):
        assert all(a[0] <= b[0] + 1e-12 for a, b in zip(lighter, heavier))


# ---------------------------------------------------------------- the hold knob: peak min(X, 1), decay 2 / X

def _hold(mod, x, **kw):
    return mod.Guide(mod.GuideConfig(early_hold=x, **kw), SIG).hold


def test_knob_one_is_peak_one_and_decay_two():
    mod = _load_pack()
    assert _hold(mod, 1.0) == [e for e, _ in mod.step_weights(SIG, 0.9, 0.5, 10.0, 1.0, 2.0)]


def test_knob_above_one_keeps_the_peak_and_flattens_the_fall():
    mod = _load_pack()
    for x in (2.0, 4.0):
        assert _hold(mod, x) == [e for e, _ in mod.step_weights(SIG, 0.9, 0.5, 10.0, 1.0, 2.0 / x)]
    assert math.isclose(_hold(mod, 1000.0)[1], _hold(mod, 1000.0)[0], rel_tol=1e-3)     # practically flat


def test_knob_below_one_lowers_the_peak_and_steepens_the_fall():
    mod = _load_pack()
    assert _hold(mod, 0.5) == [e for e, _ in mod.step_weights(SIG, 0.9, 0.5, 10.0, 0.5, 4.0)]
    assert math.isclose(_hold(mod, 0.5)[0], 0.45)                                       # peak 0.5 of the 0.9 pull


def test_knob_zero_adds_no_hold():
    mod = _load_pack()
    assert _hold(mod, 0.0) == [0.0] * (len(SIG) - 1)


def test_the_hold_grows_with_the_knob_at_every_step():
    mod = _load_pack()
    holds = [_hold(mod, x) for x in (0.1, 0.25, 0.5, 0.7, 1.0, 1.5, 4.0, 10.0)]
    for lighter, heavier in zip(holds, holds[1:]):
        assert all(a <= b + 1e-12 for a, b in zip(lighter, heavier))


def test_the_hold_acts_inside_the_edit_area_and_the_flat_pull_outside_it():
    """Early weight = m * flat + (1 - m) * hold with m the hold mask (1 = keep): linear in the gray value."""
    mod = _load_pack()
    x0, ref = torch.zeros(1, 4, 8, 8), torch.ones(1, 4, 8, 8)
    flat, hold = mod.Guide(mod.GuideConfig(early_hold=0.7), SIG).plan[0][0], mod.Guide(mod.GuideConfig(early_hold=0.7), SIG).hold[0]
    for m in (0.0, 0.25, 0.5, 1.0):
        out = mod.Guide(mod.GuideConfig(early_hold=0.7), SIG).apply(SIG[0], x0, ref, torch.full((1, 1, 8, 8), m))
        assert torch.allclose(out, torch.full_like(out, m * flat + (1 - m) * hold)), m


# ---------------------------------------------------------------- the mask filters

def _disc(h, w, cy, cx, r):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).float()[None, None]


def test_grown_pixel_is_a_disc():
    mod = _load_pack()
    for r in (1, 5, 10, 24):
        m = torch.zeros(1, 1, 101, 101)
        m[..., 50, 50] = 1
        assert torch.equal(mod._disc_max(m, r), _disc(101, 101, 50, 50, r)), r


def test_no_radius_changes_nothing():
    mod = _load_pack()
    m = torch.rand(1, 1, 20, 30)
    assert torch.equal(mod._disc_max(m, 0), m)


def test_a_gray_mask_grows_by_taking_the_brightest_value_in_reach():
    mod = _load_pack()
    m = torch.zeros(1, 1, 31, 31)
    m[..., 15, 15] = 0.4
    m[..., 15, 20] = 0.9
    out = mod._disc_max(m, 3)
    assert math.isclose(float(out[0, 0, 15, 17]), 0.9, abs_tol=1e-6) and math.isclose(float(out[0, 0, 15, 13]), 0.4, abs_tol=1e-6) and float(out[0, 0, 15, 10]) == 0.0


def test_large_radii_are_close_to_the_exact_disc():
    mod = _load_pack()
    m = torch.zeros(1, 1, 301, 301)
    m[..., 150, 150] = 1
    for r in (60, 100):
        out = mod._disc_max(m, r)
        exact = _disc(301, 301, 150, 150, r)
        assert abs(float(out.sum()) - float(exact.sum())) / float(exact.sum()) < 0.05, r
        assert (out[exact == 1] > 0.5).float().mean() > 0.9


def test_shrink_is_the_grow_of_the_background_and_leaves_the_image_border_alone():
    mod = _load_pack()
    left = torch.zeros(1, 40, 60)
    left[:, :, :30] = 1.0                                             # white from the left border to the middle
    out = mod._edit_mask(left, (40, 60), -10.0, 0.0)                   # 10% of the short side = 4 px
    assert (out[0, :, 0] == 1).all()                                   # the border is not a mask edge
    assert float(out[0, 20, 25]) == 1.0 and float(out[0, 20, 26]) == 0.0
    assert (mod._edit_mask(None, (40, 60), -100.0, 0.0) == 1).all()    # no mask stays all white, however far it shrinks


def test_grow_moves_the_edge_by_the_percentage_of_the_short_side():
    mod = _load_pack()
    box = torch.zeros(1, 100, 200)
    box[:, 40:60, 80:120] = 1.0
    out = mod._edit_mask(box, (100, 200), 10.0, 0.0)                   # 10 px
    assert float(out[0, 50, 129]) == 1.0 and float(out[0, 50, 130]) == 0.0
    assert float(out[0, 30, 100]) == 1.0 and float(out[0, 29, 100]) == 0.0


def test_feather_blurs_the_edge_and_keeps_the_values_in_range():
    mod = _load_pack()
    left = torch.zeros(1, 80, 80)
    left[:, :, :40] = 1.0
    out = mod._edit_mask(left, (80, 80), 0.0, 5.0)                     # sigma 4 px
    row = out[0, 40]
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert 0.45 < float(row[39]) < 0.7 and float(row[30]) > 0.97 and float(row[50]) < 0.03
    assert bool((row[1:] <= row[:-1] + 1e-6).all())                    # one smooth ramp


def test_a_mask_of_another_resolution_is_resized_to_the_pixel_grid():
    mod = _load_pack()
    small = torch.zeros(1, 10, 10)
    small[:, :, 5:] = 1.0
    out = mod._edit_mask(small, (80, 120), 0.0, 0.0)
    assert out.shape == (1, 80, 120) and float(out[0, 40, 10]) < 0.1 and float(out[0, 40, 110]) > 0.9


# ---------------------------------------------------------------- what the node installs

class _Patcher(_FakeModelPatcher):
    def clone(self):
        c = _Patcher()
        c.model = self.model
        return c


def _run(mod, **kw):
    """Run the node with add_guidance replaced by a recorder: returns (the call, the node's outputs)."""
    calls = []
    mod.add_guidance = lambda m, cfg, source, mask: calls.append((cfg, source, mask))
    out = mod.Krea2EditAdvanced.execute(_Patcher(), {"samples": torch.zeros(1, 4, 8, 12)}, **kw)
    return calls, out


def _aligned(**kw):
    return {"ref_position_a": ALIGNED, "source_hold": True, "hold_strength": 0.7, "grow_percent": 0.0, "feather_percent": 0.0, **kw}


def test_the_hold_is_installed_with_the_fixed_settings_and_the_knob():
    mod = _load_pack()
    calls, _ = _run(mod, ref_position_a=_aligned(hold_strength=1.3))
    (cfg, _, _), = calls
    assert cfg.early_hold == 1.3
    assert cfg == mod.GuideConfig(early_hold=1.3)


def test_the_hold_is_off_when_the_switch_is_off_or_the_source_is_not_aligned():
    mod = _load_pack()
    assert _run(mod, ref_position_a=_aligned(source_hold=False))[0] == []
    for position in ("beside (right of target)", "bag of features"):
        calls, (model, mask) = _run(mod, ref_position_a={"ref_position_a": position})
        assert calls == [] and mask.shape == (1, 64, 96) and (mask == 1).all()


def test_without_a_mask_everything_is_the_edit_area_and_nothing_is_held_by_the_mask():
    mod = _load_pack()
    calls, (_, mask) = _run(mod, ref_position_a=_aligned())
    (_, _, hold), = calls
    assert mask.shape == (1, 64, 96) and (mask == 1).all() and (hold == 0).all()       # source is 8 x 12 latent cells = 64 x 96 pixels


def test_the_hold_mask_is_the_complement_of_the_grown_and_feathered_edit_mask_that_is_also_returned():
    mod = _load_pack()
    drawn = torch.zeros(1, 32, 48)
    drawn[:, 8:24, 12:36] = 1.0
    calls, (_, mask) = _run(mod, ref_position_a=_aligned(edit_mask=drawn, grow_percent=10.0, feather_percent=5.0))
    (_, _, hold), = calls
    assert torch.equal(mask, mod._edit_mask(drawn, (64, 96), 10.0, 5.0)) and torch.allclose(hold, 1 - mask)


def test_the_reference_is_the_source_fitted_to_the_sampled_grid():
    mod = _load_pack()
    calls, _ = _run(mod, ref_position_a=_aligned())
    (_, source, hold), = calls
    placed, weight = mod._place_ref(source, hold, 8, 12)
    assert placed.shape == (1, 4, 8, 12) and weight.shape == (1, 1, 8, 12)
    placed, _ = mod._place_ref(source, hold, 8, 8)                     # a different sampled size: the same crop the attention path uses
    assert placed.shape == (1, 4, 8, 8)


def test_the_real_hold_registers_its_sampler_hooks():
    mod = _load_pack()
    import comfy.patcher_extension as pe
    import test_guide_hooks as hooks
    opts = hooks._Opts()
    mod.add_guidance(opts, mod.GuideConfig(), torch.ones(1, 4, 8, 8), torch.ones(1, 64, 64))
    assert "sampler_post_cfg_function" in opts.model_options
    assert "krea2_edit_advanced_guide" in opts.model_options["transformer_options"]["wrappers"][pe.WrappersMP.SAMPLER_SAMPLE]


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:                               # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
