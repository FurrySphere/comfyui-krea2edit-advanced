"""Source hold math: the per-step schedule, the x0 pull and its distance gate.

These are properties of the method, not tuned values, so they stay valid when defaults change.

Run standalone:  python tests/test_guidance.py
Or via pytest:   pytest tests/test_guidance.py
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stubs import _load_pack   # noqa: E402

guidance = _load_pack()

Guide, GuideConfig, step_weights = guidance.Guide, guidance.GuideConfig, guidance.step_weights

H = W = 16


# ---------------------------------------------------------------- schedules

def _simple(n, start=1.0, shift=1.15):
    t = torch.linspace(start, 0, n + 1, dtype=torch.float64)
    return (shift * t / (1 + (shift - 1) * t)).tolist()


def _karras(n, start=1.0, lo=0.02, rho=7.0):
    i = torch.arange(n, dtype=torch.float64) / (n - 1)
    return ((start ** (1 / rho) + i * (lo ** (1 / rho) - start ** (1 / rho))) ** rho).tolist() + [0.0]


def _exponential(n, start=1.0, lo=0.02):
    return torch.linspace(math.log(start), math.log(lo), n, dtype=torch.float64).exp().tolist() + [0.0]


SCHEDULES = {"simple": _simple, "karras": _karras, "exponential": _exponential}


def _early_left(sigmas, plan):
    """Fraction of the initial deviation left after the early phase (each step pulls by w * (1 - s'/s))."""
    left = 1.0
    for (early, _), s, sn in zip(plan, sigmas[:-1], sigmas[1:]):
        left *= 1 - early * (1 - sn / s)
    return left


def test_zero_guidance_has_zero_weights():
    for late_gain in (0.0, 1.0, 5.0):
        assert all(e == 0 and l == 0 for e, l in step_weights(_simple(8), 0.0, 0.5, late_gain))


def test_weights_stay_in_unit_range_and_phases_are_ordered():
    for name, make in SCHEDULES.items():
        for n in (8, 20, 40):
            for g in (0.25, 0.5, 1.0):
                for c in (0.3, 0.5, 0.7):
                    for late_gain in (0.0, 1.0, 4.0):
                        sig = make(n)
                        plan = step_weights(sig, g, c, late_gain)
                        assert len(plan) == n
                        for e, l in plan:
                            assert 0 <= e <= g + 1e-12 and 0 <= l <= 1 + 1e-12, (name, n, g, c)
                        early, late = [e for e, _ in plan], [l for _, l in plan]
                        assert early == sorted(early, reverse=True) and late == sorted(late), (name, n, g, c)


def test_early_pull_matches_the_continuous_limit_for_any_schedule_and_step_count():
    """The remaining deviation after the early phase is ~ crossover ** g, whatever the scheduler or step count."""
    for name, make in SCHEDULES.items():
        for n in (8, 12, 20, 40):
            for g in (0.25, 0.5, 1.0):
                for c in (0.3, 0.5, 0.7):
                    sig = make(n)
                    left = _early_left(sig, step_weights(sig, g, c, 1.0))
                    assert abs(left - c ** g) < 0.05, (name, n, g, c, left, c ** g)


def test_crossover_is_relative_to_the_start_sigma():
    """denoise < 1 starts lower; the early phase must still exist and pull by the same amount."""
    for start in (1.0, 0.8, 0.6, 0.4):
        sig = _simple(12, start=start)
        plan = step_weights(sig, 0.5, 0.5, 1.0)
        assert plan[0][0] > 0, start
        assert abs(_early_left(sig, plan) - 0.5 ** 0.5) < 0.05, start


def test_straddling_step_blends_instead_of_snapping_to_a_whole_step():
    sig = _simple(8)
    plan = step_weights(sig, 1.0, 0.5, 0.0)
    straddling = [i for i, (e, l) in enumerate(plan) if 0 < e < 1.0]
    assert len(straddling) == 1
    # moving the crossover slightly must move the result smoothly (no jump from step quantization)
    a = _early_left(sig, step_weights(sig, 1.0, 0.50, 0.0))
    b = _early_left(sig, step_weights(sig, 1.0, 0.51, 0.0))
    assert abs(a - b) < 0.02


def test_late_gain_controls_the_late_phase_only():
    sig = _simple(12)
    early_only = step_weights(sig, 0.5, 0.5, 0.0)
    constant = step_weights(sig, 0.5, 0.5, 1.0)
    saturated = step_weights(sig, 0.5, 0.5, 100.0)
    assert [e for e, _ in early_only] == [e for e, _ in constant] == [e for e, _ in saturated]
    assert all(l == 0 for _, l in early_only)
    assert all(abs(l - 0.5) < 1e-12 for e, l in constant if e == 0)
    assert all(abs(l - 1.0) < 1e-12 for e, l in saturated if e == 0)


def test_early_gain_scales_only_the_early_phase():
    sig = _simple(12)
    base = step_weights(sig, 0.8, 0.5, 4.0)
    for early_gain in (0.0, 0.25, 0.5):
        plan = step_weights(sig, 0.8, 0.5, 4.0, early_gain)
        assert [l for _, l in plan] == [l for _, l in base]                   # the late phase does not move
        assert all(abs(e - early_gain * e0) < 1e-12 for (e, _), (e0, _) in zip(plan, base))
    assert step_weights(sig, 0.8, 0.5, 4.0, 1.0) == base                      # the default reproduces the old behavior


def test_late_only_guidance_leaves_the_early_steps_alone():
    x0, ref = _fields()
    g = _guide(ref_guide=0.8, late_gain=10.0, early_gain=0.0)
    assert torch.equal(g.apply(EARLY, x0, ref, ONES, VALID), x0)
    assert torch.allclose(g.apply(LATE, x0, ref, ONES, VALID), ref)                   # late strength saturates at 1
    full_early = _guide(ref_guide=0.8, late_gain=10.0, early_gain=1.0).apply(EARLY, x0, ref, ONES, VALID)
    assert torch.allclose(full_early, x0 + 0.8 * (ref - x0), atol=1e-6)


def test_degenerate_schedules_do_not_break():
    assert step_weights([1.0, 0.0], 0.5, 0.5, 1.0) == [(0.25, 0.25)]      # one step: half its pull above the crossover
    plan = step_weights(torch.tensor(_simple(4)), 0.5, 0.5, 1.0)       # tensor input
    assert len(plan) == 4


# ---------------------------------------------------------------- the pull

def _fields(seed=0, c=4):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, c, H, W, generator=g), torch.randn(1, c, H, W, generator=g)


def _guide(sigmas=None, **cfg):
    return Guide(GuideConfig(**cfg), sigmas or _simple(10))


ONES = torch.ones(1, 1, H, W)
ZEROS = torch.zeros(1, 1, H, W)
VALID = torch.ones(1, 1, H, W)
LATE = 0.05          # a sigma well inside the late phase of _simple(10)
EARLY = 0.95


def test_zero_guidance_returns_the_estimate_untouched():
    x0, ref = _fields()
    assert torch.equal(_guide(ref_guide=0.0).apply(LATE, x0, ref, ONES, VALID), x0)


def test_full_guidance_copies_the_reference():
    x0, ref = _fields()
    for sigma in (EARLY, LATE):
        out = _guide(ref_guide=1.0, early_gain=1.0).apply(sigma, x0, ref, ONES, VALID)
        assert torch.allclose(out, ref), sigma


def test_guidance_is_proportional_to_the_mask_and_the_strength():
    x0, ref = _fields()
    mask = torch.full((1, 1, H, W), 0.5)
    out = _guide(ref_guide=0.5, early_gain=1.0).apply(EARLY, x0, ref, mask, VALID)
    assert torch.allclose(out, x0 + 0.25 * (ref - x0), atol=1e-6)


def test_the_edit_area_is_left_alone_in_the_early_phase():
    x0, ref = _fields()
    mask = torch.ones(1, 1, H, W)
    mask[..., :8] = 0
    out = _guide(ref_guide=1.0, early_gain=1.0).apply(EARLY, x0, ref, mask, VALID)
    assert torch.equal(out[..., :8], x0[..., :8])
    assert torch.allclose(out[..., 8:], ref[..., 8:])


def test_output_keeps_the_estimate_dtype():
    x0, ref = _fields()
    out = _guide(ref_guide=0.5).apply(EARLY, x0.bfloat16(), ref.bfloat16(), ONES, VALID)
    assert out.dtype == torch.bfloat16


def test_nothing_is_pulled_where_the_reference_shows_no_source():
    x0, ref = _fields()
    valid = VALID.clone()
    valid[..., 8:] = 0
    for sigma in (EARLY, LATE):
        for mask in (ONES, ZEROS):                                           # held or edit area: no source means no pull either way
            out = _guide(ref_guide=1.0, early_gain=1.0, dist_tau=0.0, early_hold=1.0).apply(sigma, x0, ref, mask, valid)
            assert torch.equal(out[..., 8:], x0[..., 8:]), (sigma, float(mask.max()))
            assert not torch.equal(out[..., :8], x0[..., :8])


def test_the_distance_scale_comes_from_the_cells_that_show_the_source():
    x0, ref = _fields()
    ref[..., 8:] = 0.0
    valid = VALID.clone()
    valid[..., 8:] = 0
    g = _guide(ref_guide=0.5, late_gain=1.0, dist_tau=1.0)
    g.apply(LATE, x0, ref, ZEROS, valid)
    assert math.isclose(float(g.ref_std), float(ref[..., :8].std()), rel_tol=1e-5)


# ---------------------------------------------------------------- distance gate

def _two_zone(dev):
    """Reference with a pattern; estimate equals it on the left half and is `dev` away on the right half."""
    ref = torch.zeros(1, 4, H, W)
    ref[:, :, ::2, :] = 1.0                                                  # non-trivial reference (std ~0.5)
    x0 = ref.clone()
    x0[..., 8:] += dev
    return x0, ref


def _pull_weight(dev, mask_value=0.0, **cfg):
    """Effective pull weight (out - x0) / (ref - x0) in the late phase when the estimate is `dev` away from the reference."""
    ref = torch.zeros(1, 4, H, W)
    ref[:, :, ::2, :] = 1.0                                                  # reference std = 0.5
    x0 = ref + dev
    out = _guide(**{"smooth": 0.0, "late_gain": 1.0, "dist_pow": 2.0, **cfg}).apply(LATE, x0, ref, torch.full((1, 1, H, W), mask_value), VALID)
    return float(((out - x0) / (ref - x0)).mean())


def test_distance_gate_down_weights_deviating_tokens_robustly():
    """weight = g * tau^2 / (d^2 + tau^2) with d in reference-std units and tau = tau0 * g / (1 - g)."""
    g, tau0 = 0.6, 1.0
    tau = tau0 * g / (1 - g)
    weights = [_pull_weight(dev, ref_guide=g, dist_tau=tau0) for dev in (0.05, 0.25, 0.5, 1.0, 2.0, 4.0)]
    for dev, w in zip((0.05, 0.25, 0.5, 1.0, 2.0, 4.0), weights):
        d = dev / 0.5
        assert abs(w - g * tau ** 2 / (d ** 2 + tau ** 2)) < 1e-3, (dev, w)
    assert all(a > b for a, b in zip(weights, weights[1:]))                  # more deviation, less pull
    assert weights[0] > 0.55 and weights[-1] < 0.05                           # near-agreeing: ~full g; far: ~free


def test_the_gate_is_confined_to_the_edit_area():
    assert abs(_pull_weight(4.0, 1.0, ref_guide=0.6, dist_tau=1.0) - 0.6) < 1e-5       # keep area: pulled at the full strength even when far
    assert _pull_weight(2.0, 0.0, ref_guide=0.6, dist_tau=1.0) < 0.6 - 1e-3            # edit area: gated


def test_the_freedom_is_linear_in_the_mask():
    w0, wh, w1 = (_pull_weight(2.0, m, ref_guide=0.6, dist_tau=1.0) for m in (0.0, 0.5, 1.0))
    assert w0 < wh < w1 and abs(wh - (w0 + w1) / 2) < 1e-5


def test_gate_steepness_sharpens_the_transition_around_the_tolerance():
    g, tau0 = 0.6, 1.0
    tau = tau0 * g / (1 - g)
    for n in (2.0, 4.0, 8.0):
        for dev in (0.25, 0.75, 1.5):
            d = dev / 0.5
            w = _pull_weight(dev, ref_guide=g, dist_tau=tau0, dist_pow=n)
            assert abs(w - g * tau ** n / (d ** n + tau ** n)) < 1e-3, (n, dev, w)
    below, above = 0.6, 0.9375                                               # d = 0.8 tau and 1.25 tau
    assert _pull_weight(below, ref_guide=g, dist_tau=tau0, dist_pow=8.0) > _pull_weight(below, ref_guide=g, dist_tau=tau0, dist_pow=2.0)
    assert _pull_weight(above, ref_guide=g, dist_tau=tau0, dist_pow=8.0) < _pull_weight(above, ref_guide=g, dist_tau=tau0, dist_pow=2.0)


def test_distance_gate_is_off_at_zero_tau_and_in_the_early_phase():
    x0, ref = _two_zone(3.0)
    ref = ref + 1.0
    off = _guide(ref_guide=0.5, late_gain=1.0, dist_tau=0.0).apply(LATE, x0, ref, ZEROS, VALID)
    assert torch.allclose(off, x0 + 0.5 * (ref - x0), atol=1e-6)
    gated_early = _guide(ref_guide=0.5, dist_tau=1.0, early_hold=1.0).apply(EARLY, x0, ref, ZEROS, VALID)
    plain_early = _guide(ref_guide=0.5, dist_tau=0.0, early_hold=1.0).apply(EARLY, x0, ref, ZEROS, VALID)
    assert torch.allclose(gated_early, plain_early)


def test_distance_gate_tolerance_grows_with_the_guidance_strength():
    x0, ref = _two_zone(2.0)
    ref = ref + 1.0
    pulls = []
    for g in (0.3, 0.6, 0.9, 1.0):
        out = _guide(ref_guide=g, late_gain=1.0, dist_tau=1.0, smooth=0.0).apply(LATE, x0, ref, ZEROS, VALID)
        pulls.append(float(((out - x0) / (ref - x0))[..., 8:].mean()) / g)     # pull per unit of strength
    assert all(a < b for a, b in zip(pulls, pulls[1:])) and abs(pulls[-1] - 1.0) < 1e-5, pulls


def test_distance_memory_ignores_the_early_phase():
    """Early estimates are blurry, so they must not leak into the gate's memory for the late phase."""
    x0, ref = _two_zone(0.0)
    ref = ref + 1.0
    g = _guide(ref_guide=0.5, dist_tau=0.5, smooth=0.5, early_hold=1.0)
    g.apply(EARLY, x0 + 5.0, ref, ZEROS, VALID)
    remembered = g.apply(LATE, x0, ref, ZEROS, VALID)
    fresh = _guide(ref_guide=0.5, dist_tau=0.5, smooth=0.5, early_hold=1.0).apply(LATE, x0, ref, ZEROS, VALID)
    assert torch.equal(remembered, fresh)


def test_distance_smoothing_averages_across_steps():
    x0, ref = _two_zone(0.0)
    ref = ref + 1.0
    cfg = dict(ref_guide=0.5, late_gain=1.0, dist_tau=0.5, dist_pow=2.0, smooth=0.5)
    g = _guide(**cfg)
    first = g.apply(LATE, x0, ref, ZEROS, VALID)
    far = x0 + 5.0
    smoothed = g.apply(LATE, far, ref, ZEROS, VALID)
    fresh = _guide(**cfg).apply(LATE, far, ref, ZEROS, VALID)
    assert (smoothed - far).abs().mean() > (fresh - far).abs().mean()         # remembers that it was close before
    assert first.shape == x0.shape


# ---------------------------------------------------------------- run isolation

SIG = _simple(10)


def test_each_run_has_its_own_state():
    a, b = _guide(ref_guide=0.5, dist_tau=1.0), _guide(ref_guide=0.5, dist_tau=1.0)
    x0, ref = torch.zeros(1, 4, H, W), torch.ones(1, 4, H, W)
    a.apply(LATE, x0, ref, ZEROS, VALID)
    assert a.dist is not None and b.dist is None


def test_step_is_found_from_the_current_sigma_even_as_a_batch_tensor():
    g = Guide(GuideConfig(ref_guide=0.5), SIG)
    assert g.step_of(torch.tensor([SIG[4], SIG[4]])) == 4
    assert g.step_of(SIG[4] * 1.00001) == 4


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
