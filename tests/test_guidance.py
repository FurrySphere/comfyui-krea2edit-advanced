"""Reference guidance math (guidance.py): the per-step schedule, the x0 pull and its gates.

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
    assert torch.equal(g.apply(EARLY, x0, ref, ONES, ONES), x0)
    assert torch.allclose(g.apply(LATE, x0, ref, ONES, ONES), ref)             # late strength saturates at 1
    ungated_early = _guide(ref_guide=0.8, late_gain=10.0, early_gain=1.0).apply(EARLY, x0, ref, ONES, ONES)
    assert torch.allclose(ungated_early, x0 + 0.8 * (ref - x0), atol=1e-6)


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
LATE = 0.05          # a sigma well inside the late phase of _simple(10)
EARLY = 0.95


def test_zero_guidance_returns_the_estimate_untouched():
    x0, ref = _fields()
    out = _guide().apply(LATE, x0, ref, ONES, ONES)
    assert torch.equal(out, x0)


def test_full_guidance_with_open_gates_copies_the_reference():
    x0, ref = _fields()
    for sigma in (EARLY, LATE):
        out = _guide(ref_guide=1.0, dist_tau=1.0).apply(sigma, x0, ref, ONES, ONES)
        assert torch.allclose(out, ref), sigma


def test_guidance_is_proportional_to_the_mask_and_the_strength():
    x0, ref = _fields()
    mask = torch.full((1, 1, H, W), 0.5)
    out = _guide(ref_guide=0.5).apply(EARLY, x0, ref, mask, ONES)
    assert torch.allclose(out, x0 + 0.25 * (ref - x0), atol=1e-6)


def test_mask_zero_and_coverage_zero_leave_the_estimate_alone():
    x0, ref = _fields()
    mask = torch.ones(1, 1, H, W)
    mask[..., :8] = 0
    cov = torch.ones(1, 1, H, W)
    cov[..., 12:] = 0
    for feather in (0.0, 0.1):
        out = _guide(ref_guide=1.0, feather=feather).apply(EARLY, x0, ref, mask, cov)
        assert torch.equal(out[..., 12:], x0[..., 12:]), feather          # no reference there: never touched, even feathered
        if feather == 0:
            assert torch.equal(out[..., :8], x0[..., :8])
            assert torch.allclose(out[..., 8:12], ref[..., 8:12])


def test_feather_softens_mask_edges_monotonically():
    x0, ref = _fields()
    mask = torch.zeros(1, 1, H, W)
    mask[..., 8:] = 1
    out = _guide(ref_guide=1.0, feather=0.2).apply(EARLY, x0, ref, mask, ONES)
    w = ((out - x0) / (ref - x0)).mean(1)[0, 8]                              # weight along one row
    assert (w[1:] >= w[:-1] - 1e-5).all() and 0 < w[6] < 1 and w[0] < 0.2 and w[-1] > 0.8


def test_inside_floor_pulls_the_masked_region_at_that_fraction():
    x0, ref = _fields()
    out = _guide(ref_guide=1.0, inside=0.3).apply(EARLY, x0, ref, torch.zeros(1, 1, H, W), ONES)
    assert torch.allclose(out, x0 + 0.3 * (ref - x0), atol=1e-6)


def test_inside_floor_ramps_linearly_up_to_full_strength():
    x0, ref = _fields()
    for m in (0.0, 0.25, 0.5, 1.0):
        out = _guide(ref_guide=1.0, inside=0.2).apply(EARLY, x0, ref, torch.full((1, 1, H, W), m), ONES)
        assert torch.allclose(out, x0 + (0.2 + 0.8 * m) * (ref - x0), atol=1e-6), m


def test_inside_floor_scales_with_the_strength_and_never_touches_uncovered_cells():
    x0, ref = _fields()
    cov = torch.ones(1, 1, H, W)
    cov[..., 12:] = 0
    out = _guide(ref_guide=0.5, inside=0.4).apply(EARLY, x0, ref, torch.zeros(1, 1, H, W), cov)
    assert torch.allclose(out[..., :12], x0[..., :12] + 0.2 * (ref[..., :12] - x0[..., :12]), atol=1e-6)
    assert torch.equal(out[..., 12:], x0[..., 12:])


def test_zero_inside_floor_is_the_plain_mask():
    x0, ref = _fields()
    mask = torch.rand(1, 1, H, W)
    assert torch.equal(_guide(ref_guide=0.7, inside=0.0).apply(EARLY, x0, ref, mask, ONES),
                       _guide(ref_guide=0.7).apply(EARLY, x0, ref, mask, ONES))


def test_output_keeps_the_estimate_dtype():
    x0, ref = _fields()
    out = _guide(ref_guide=0.5).apply(EARLY, x0.bfloat16(), ref.bfloat16(), ONES, ONES)
    assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------- distance gate

def _two_zone(dev):
    """Reference zeros; estimate equals it on the left half and is `dev` away on the right half."""
    ref = torch.zeros(1, 4, H, W)
    ref[:, :, ::2, :] = 1.0                                                  # non-trivial reference (std ~0.5)
    x0 = ref.clone()
    x0[..., 8:] += dev
    return x0, ref


def _pull_weight(dev, **cfg):
    """Effective pull weight (out - x0) / (ref - x0) when the estimate is `dev` away from the reference."""
    ref = torch.zeros(1, 4, H, W)
    ref[:, :, ::2, :] = 1.0                                                  # reference std = 0.5
    x0 = ref + dev
    out = _guide(smooth=0.0, **cfg).apply(LATE, x0, ref, ONES, ONES)
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


def _pull_weight_masked(dev, mask_value, **cfg):
    ref = torch.zeros(1, 4, H, W)
    ref[:, :, ::2, :] = 1.0
    x0 = ref + dev
    out = _guide(smooth=0.0, **cfg).apply(LATE, x0, ref, torch.full((1, 1, H, W), mask_value), ONES)
    return float(((out - x0) / (ref - x0)).mean())


def test_gate_in_mask_pulls_unconditionally_where_the_mask_is_one():
    w = _pull_weight_masked(4.0, 1.0, ref_guide=0.6, dist_tau=1.0, gate_in_mask=True)
    assert abs(w - 0.6) < 1e-5                                               # a far estimate is pulled at the full strength


def test_gate_in_mask_gates_like_the_plain_gate_where_the_mask_is_zero():
    w = _pull_weight_masked(2.0, 0.0, ref_guide=0.6, dist_tau=1.0, gate_in_mask=True, inside=1.0)
    assert abs(w - _pull_weight(2.0, ref_guide=0.6, dist_tau=1.0)) < 1e-5 and w < 0.6 - 1e-3


def test_gate_in_mask_freedom_is_linear_in_the_mask():
    cfg = dict(ref_guide=0.6, dist_tau=1.0, gate_in_mask=True, inside=1.0)
    w0, wh, w1 = (_pull_weight_masked(2.0, m, **cfg) for m in (0.0, 0.5, 1.0))
    assert w0 < wh < w1 and abs(wh - (w0 + w1) / 2) < 1e-5


def test_gate_acts_everywhere_unless_confined_to_the_mask():
    assert _pull_weight_masked(2.0, 1.0, ref_guide=0.6, dist_tau=1.0) < 0.6 - 1e-3


def test_gate_steepness_sharpens_the_transition_around_the_tolerance():
    g, tau0 = 0.6, 1.0
    tau = tau0 * g / (1 - g)
    for n in (2.0, 4.0, 8.0):
        for dev in (0.25, 0.75, 1.5):
            d = dev / 0.5
            w = _pull_weight(dev, ref_guide=g, dist_tau=tau0, dist_pow=n)
            assert abs(w - g * tau ** n / (d ** n + tau ** n)) < 1e-3, (n, dev, w)
    below, above = 0.6, 0.9375                                               # d = 0.8 tau and 1.25 tau
    assert _pull_weight(below, ref_guide=g, dist_tau=tau0, dist_pow=8.0) > _pull_weight(below, ref_guide=g, dist_tau=tau0)
    assert _pull_weight(above, ref_guide=g, dist_tau=tau0, dist_pow=8.0) < _pull_weight(above, ref_guide=g, dist_tau=tau0)


def _pulls(first, second, **cfg):
    """Pull weight of a second late call after a first one on the same run (separate sigmas, no smoothing)."""
    ref = torch.zeros(1, 4, H, W)
    ref[:, :, ::2, :] = 1.0
    g = _guide(smooth=0.0, ref_guide=0.6, dist_tau=1.0, **cfg)
    g.apply(LATE, ref + first, ref, ONES, ONES)
    x0 = ref + second
    return float(((g.apply(0.2, x0, ref, ONES, ONES) - x0) / (ref - x0)).mean())


def test_latch_keeps_a_freed_token_free():
    plain = _pulls(3.0, 0.01)
    latched = _pulls(3.0, 0.01, free_latch=True)
    assert plain > 0.55 and latched < 0.3 * plain                          # a token that once deviated stays free


def test_latch_changes_nothing_when_no_token_deviated():
    assert abs(_pulls(0.01, 0.01, free_latch=True) - _pulls(0.01, 0.01)) < 1e-5


def _boundary_weights(**cfg):
    ref = torch.zeros(1, 4, H, W)
    ref[:, :, ::2, :] = 1.0
    x0 = ref + 0.01
    x0[..., 8:] += 3.0                                                       # the object: right half, far from the reference
    out = _guide(smooth=0.0, ref_guide=0.6, dist_tau=1.0, **cfg).apply(LATE, x0, ref, ONES, ONES)
    return ((out - x0) / (ref - x0)).mean(1)[0, 8]


def test_free_grow_spreads_the_freedom_to_the_neighbours_of_the_object():
    plain, grown = _boundary_weights(), _boundary_weights(free_grow=2)
    assert plain[5] > 0.55 and grown[5] < 0.5 * plain[5]                     # next to the object: freed
    assert abs(float(grown[0]) - float(plain[0])) < 1e-4                     # far away: untouched
    assert (grown[:8] <= plain[:8] + 1e-6).all()                             # never pulls more than before


def _calibrated_weight(bg, obj, **cfg):
    gen = torch.Generator().manual_seed(1)
    ref = torch.zeros(1, 4, H, W)
    ref[:, :, ::2, :] = 1.0
    x0 = ref + bg * 0.05 * torch.randn(1, 4, H, W, generator=gen)
    x0[..., 8:] += obj                                                       # right half is inside the mask and holds the object
    mask = torch.ones(1, 1, H, W)
    mask[..., 8:] = 0
    g = _guide(smooth=0.0, ref_guide=0.9, gate_in_mask=True, inside=1.0, dist_calib=3.0, **cfg)
    out = g.apply(LATE, x0, ref, mask, ONES)
    return float(((out - x0) / (ref - x0)).mean(1)[0, :, 8:].mean())


def test_calibrated_tolerance_does_not_depend_on_the_overall_deviation_scale():
    a, b = _calibrated_weight(1.0, 1.0), _calibrated_weight(3.0, 3.0)
    assert abs(a - b) < 1e-3, (a, b)


def test_calibrated_tolerance_follows_the_background_deviation_outside_the_mask():
    quiet, noisy = _calibrated_weight(1.0, 0.5), _calibrated_weight(4.0, 0.5)
    assert noisy > quiet + 0.05                                              # a noisier background tolerates more, so the same deviation is pulled more


def test_distance_blur_makes_a_single_deviating_token_count_less():
    ref = torch.zeros(1, 4, H, W)
    ref[:, :, ::2, :] = 1.0
    x0 = ref + 0.01
    x0[..., 4:6, 4:6] += 3.0                                                 # one token
    weights = []
    for blur_tokens in (0.0, 1.0):
        out = _guide(smooth=0.0, ref_guide=0.6, dist_tau=1.0, dist_blur=blur_tokens).apply(LATE, x0, ref, ONES, ONES)
        weights.append(float(((out - x0) / (ref - x0)).mean(1)[0, 4, 4]))
    assert weights[1] > 3 * weights[0]


def test_distance_gate_is_off_at_zero_tau_and_in_the_early_phase():
    x0, ref = _two_zone(3.0)
    ref = ref + 1.0
    off = _guide(ref_guide=0.5, dist_tau=0.0).apply(LATE, x0, ref, ONES, ONES)
    assert torch.allclose(off, x0 + 0.5 * (ref - x0), atol=1e-6)
    gated_early = _guide(ref_guide=0.5, dist_tau=1.0).apply(EARLY, x0, ref, ONES, ONES)
    plain_early = _guide(ref_guide=0.5, dist_tau=0.0).apply(EARLY, x0, ref, ONES, ONES)
    assert torch.allclose(gated_early, plain_early)


def test_distance_gate_tolerance_grows_with_the_guidance_strength():
    x0, ref = _two_zone(2.0)
    ref = ref + 1.0
    pulls = []
    for g in (0.3, 0.6, 0.9, 1.0):
        out = _guide(ref_guide=g, dist_tau=1.0, smooth=0.0).apply(LATE, x0, ref, ONES, ONES)
        pulls.append(float(((out - x0) / (ref - x0))[..., 8:].mean()) / g)     # pull per unit of strength
    assert all(a < b for a, b in zip(pulls, pulls[1:])) and abs(pulls[-1] - 1.0) < 1e-5, pulls


def test_distance_memory_ignores_the_early_phase():
    """Early estimates are blurry, so they must not leak into the gate's memory for the late phase."""
    x0, ref = _two_zone(0.0)
    ref = ref + 1.0
    g = _guide(ref_guide=0.5, dist_tau=0.5, smooth=0.5)
    g.apply(EARLY, x0 + 5.0, ref, ONES, ONES)
    remembered = g.apply(LATE, x0, ref, ONES, ONES)
    fresh = _guide(ref_guide=0.5, dist_tau=0.5, smooth=0.5).apply(LATE, x0, ref, ONES, ONES)
    assert torch.equal(remembered, fresh)


def test_distance_smoothing_averages_across_steps():
    x0, ref = _two_zone(0.0)
    ref = ref + 1.0
    g = _guide(ref_guide=0.5, dist_tau=0.5, smooth=0.5)
    first = g.apply(LATE, x0, ref, ONES, ONES)
    far = x0 + 5.0
    smoothed = g.apply(LATE, far, ref, ONES, ONES)
    fresh = _guide(ref_guide=0.5, dist_tau=0.5, smooth=0.5).apply(LATE, far, ref, ONES, ONES)
    assert (smoothed - far).abs().mean() > (fresh - far).abs().mean()         # remembers that it was close before
    assert first.shape == x0.shape


# ---------------------------------------------------------------- edit-map gate

SIG = _simple(10)
IN_WINDOW = SIG[3]       # about 0.7 of the start sigma, inside the default window [0.9, 0.5]


def _cond_uncond(strength=1.0):
    uncond = torch.zeros(1, 4, H, W)
    cond = uncond.clone()
    cond[..., :6, :6] += strength                                             # the instruction changes the top-left corner
    return cond, uncond


def test_edit_map_requests_the_unconditional_pass_only_inside_its_window():
    g = Guide(GuideConfig(ref_guide=0.5, edit_thr=0.5), SIG)
    flags = [g.needs_uncond(s) for s in SIG[:-1]]
    assert any(flags) and not all(flags)
    first, last = flags.index(True), len(flags) - 1 - flags[::-1].index(True)
    assert all(flags[first:last + 1])
    assert not any(Guide(GuideConfig(ref_guide=0.5), SIG).needs_uncond(s) for s in SIG)
    assert not any(Guide(GuideConfig(edit_thr=0.5), SIG).needs_uncond(s) for s in SIG)      # no guidance, no pass


def _edit_weight(g, strength=1.0, sigma=LATE):
    cond, uncond = _cond_uncond(strength)
    x0 = torch.zeros(1, 4, H, W)
    ref = torch.ones(1, 4, H, W)
    g.apply(IN_WINDOW, x0, ref, ONES, ONES, cond_x0=cond, uncond_x0=uncond)
    out = g.apply(sigma, x0, ref, ONES, ONES)
    return (out - x0)[0, 0]


def test_edit_map_frees_the_region_the_instruction_changes():
    w = _edit_weight(Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0), SIG))
    assert w[:4, :4].mean() < 0.05 and w[10:, 10:].mean() > 0.45


def test_edit_map_is_scale_invariant():
    a = _edit_weight(Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0), SIG), strength=1.0)
    b = _edit_weight(Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0), SIG), strength=10.0)
    assert torch.allclose(a, b, atol=1e-4)


def test_edit_map_is_inert_when_the_instruction_changes_nothing():
    g = Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0), SIG)
    x0, ref = torch.zeros(1, 4, H, W), torch.ones(1, 4, H, W)
    same = torch.zeros(1, 4, H, W)
    g.apply(IN_WINDOW, x0, ref, ONES, ONES, cond_x0=same, uncond_x0=same)
    out = g.apply(LATE, x0, ref, ONES, ONES)
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_edit_map_only_grows_and_freezes_after_its_window():
    g = Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0), SIG)
    x0, ref = torch.zeros(1, 4, H, W), torch.ones(1, 4, H, W)
    cond, uncond = _cond_uncond()
    g.apply(IN_WINDOW, x0, ref, ONES, ONES, cond_x0=cond, uncond_x0=uncond)
    before = g.edit_acc.clone()
    moved = cond.roll(8, dims=-1)                                              # the instruction now "changes" another region
    g.apply(IN_WINDOW, x0, ref, ONES, ONES, cond_x0=moved, uncond_x0=uncond)
    assert (g.edit_acc >= before - 1e-6).all() and g.edit_acc[..., 8:].max() > before[..., 8:].max()
    frozen = g.edit_acc.clone()
    g.apply(SIG[8], x0, ref, ONES, ONES, cond_x0=torch.ones_like(cond), uncond_x0=uncond)   # outside the window
    assert torch.equal(g.edit_acc, frozen)


def test_edit_gate_without_the_unconditional_prediction_stays_off():
    g = Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4), SIG)
    x0, ref = torch.zeros(1, 4, H, W), torch.ones(1, 4, H, W)
    out = g.apply(LATE, x0, ref, ONES, ONES)
    assert torch.allclose(out, torch.full_like(out, 0.5))


def _weight_of(g, cond, uncond, x0=None, ref=None):
    x0 = torch.zeros(1, 4, H, W) if x0 is None else x0
    ref = torch.ones(1, 4, H, W) if ref is None else ref
    g.apply(IN_WINDOW, x0, ref, ONES, ONES, cond_x0=cond, uncond_x0=uncond)
    return (g.apply(LATE, x0, ref, ONES, ONES) - x0).mean(1)[0]


def test_edit_softness_widens_the_transition_around_the_threshold():
    uncond = torch.zeros(1, 4, H, W)
    cond = uncond.clone()
    cond[..., :6, :6] += 1.0                                                   # strongly changed
    cond[..., 6:12, :6] += 0.5                                                 # half as strong
    pulls = [float(_weight_of(Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0, edit_soft=s), SIG), cond, uncond)[8, 2])
             for s in (0.1, 0.3, 1.0)]
    assert pulls[0] < pulls[1] < pulls[2] and pulls[0] < 0.01                  # the half-strength token is freed sharply or only partly


def test_edit_normalisation_quantile_decides_how_noise_is_read_when_the_object_is_small():
    gen = torch.Generator().manual_seed(0)
    uncond = torch.zeros(1, 4, H, W)
    cond = 0.01 * torch.randn(1, 4, H, W, generator=gen)
    cond[..., :2, :2] += 1.0                                                   # one changed token among 64
    pulls = [float(_weight_of(Guide(GuideConfig(ref_guide=0.5, edit_thr=0.5, smooth=0.0, edit_q=q), SIG), cond, uncond)[:, 8:].mean())
             for q in (0.95, 0.99)]
    assert pulls[1] > pulls[0] + 0.2                                           # the higher quantile keeps the noise from being read as an edit


def test_edit_region_normalisation_reads_the_edit_relative_to_the_masked_region():
    gen = torch.Generator().manual_seed(0)
    uncond = torch.zeros(1, 4, H, W)
    cond = 0.01 * torch.randn(1, 4, H, W, generator=gen)
    cond[..., :2, :2] += 1.0                                                   # a small object inside the masked corner
    mask = torch.ones(1, 1, H, W)
    mask[..., :8, :8] = 0                                                      # the masked region: 16 of the 64 tokens
    x0, ref = torch.zeros(1, 4, H, W), torch.ones(1, 4, H, W)
    pulls = {}
    for region in (False, True):
        g = Guide(GuideConfig(ref_guide=0.5, edit_thr=0.5, smooth=0.0, gate_in_mask=True, inside=1.0, edit_region_norm=region), SIG)
        g.apply(IN_WINDOW, x0, ref, mask, ONES, cond_x0=cond, uncond_x0=uncond)
        out = g.apply(LATE, x0, ref, mask, ONES)
        pulls[region] = float((out - x0).mean(1)[0, 4:8, 4:8].mean())          # masked cells away from the object
    assert pulls[True] > pulls[False] + 0.2                                    # whole-image normalisation reads the noise there as an edit


def test_edit_region_normalisation_falls_back_to_the_whole_map_without_a_masked_region():
    cond, uncond = _cond_uncond()
    a = _weight_of(Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0), SIG), cond, uncond)
    b = _weight_of(Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0, edit_region_norm=True), SIG), cond, uncond)   # mask is all ones
    assert torch.equal(a, b)


def test_edit_grow_spreads_the_freedom_to_the_neighbours():
    cond, uncond = _cond_uncond()
    plain = _weight_of(Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0), SIG), cond, uncond)
    grown = _weight_of(Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0, edit_grow=1), SIG), cond, uncond)
    assert plain[7, 2] > 0.4 and grown[7, 2] < 0.5 * plain[7, 2]               # next to the changed corner
    assert (grown <= plain + 1e-6).all() and abs(float(grown[15, 15] - plain[15, 15])) < 1e-6


def _two_gate_weights(**cfg):
    """Distance gate frees the right half, edit gate the top half; returns the pull in the four quadrants."""
    ref = torch.zeros(1, 4, H, W)
    ref[:, :, ::2, :] = 1.0
    x0 = ref + 0.01
    x0[..., 8:] += 3.0
    uncond = torch.zeros(1, 4, H, W)
    cond = uncond.clone()
    cond[..., :8, :] += 1.0
    g = Guide(GuideConfig(ref_guide=0.6, dist_tau=1.0, edit_thr=0.4, smooth=0.0, **cfg), SIG)
    g.apply(IN_WINDOW, x0, ref, ONES, ONES, cond_x0=cond, uncond_x0=uncond)
    w = ((g.apply(LATE, x0, ref, ONES, ONES) - x0) / (ref - x0)).mean(1)[0]
    return {"top_left": float(w[:8, :8].mean()), "top_right": float(w[:8, 8:].mean()),
            "bottom_left": float(w[8:, :8].mean()), "bottom_right": float(w[8:, 8:].mean())}


def test_two_gates_free_the_union_by_default_and_the_intersection_when_both_must_agree():
    union, both = _two_gate_weights(), _two_gate_weights(gate_and=True)
    assert union["bottom_left"] > 0.5 and both["bottom_left"] > 0.5            # neither gate frees it
    assert union["top_left"] < 0.1 and union["bottom_right"] < 0.1             # one gate is enough
    assert both["top_left"] > 0.5 and both["bottom_right"] > 0.5               # one gate is not enough
    assert union["top_right"] < 0.1 and both["top_right"] < 0.1                # both gates free it


def test_gate_and_with_only_the_edit_gate_active_is_the_edit_gate():
    cond, uncond = _cond_uncond()
    a = _weight_of(Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0), SIG), cond, uncond)
    b = _weight_of(Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0, gate_and=True), SIG), cond, uncond)
    assert torch.equal(a, b)


def test_gate_gamma_shapes_how_fast_the_freedom_fades_toward_the_mask_edge():
    free = 1 - _pull_weight(2.0, ref_guide=0.6, dist_tau=1.0) / 0.6            # freedom where the mask is 0
    for gamma in (0.5, 1.0, 2.0):
        w = _pull_weight_masked(2.0, 0.5, ref_guide=0.6, dist_tau=1.0, gate_in_mask=True, inside=1.0, gate_gamma=gamma)
        assert abs(w - 0.6 * (1 - free * 0.5 ** gamma)) < 1e-4, gamma


def test_early_outside_keeps_the_early_pull_off_the_masked_region_only():
    x0, ref = _fields()
    zero, one = torch.zeros(1, 1, H, W), torch.ones(1, 1, H, W)
    cfg = dict(ref_guide=1.0, inside=1.0)
    assert not torch.equal(_guide(**cfg).apply(EARLY, x0, ref, zero, ONES), x0)                      # by default the floor applies in the early phase too
    assert torch.equal(_guide(early_outside=True, **cfg).apply(EARLY, x0, ref, zero, ONES), x0)       # now the mask region is left alone early
    assert torch.equal(_guide(early_outside=True, **cfg).apply(EARLY, x0, ref, one, ONES), _guide(**cfg).apply(EARLY, x0, ref, one, ONES))
    assert torch.equal(_guide(early_outside=True, **cfg).apply(LATE, x0, ref, zero, ONES), _guide(**cfg).apply(LATE, x0, ref, zero, ONES))


# ---------------------------------------------------------------- run isolation

def test_each_run_has_its_own_state():
    a = Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, dist_tau=1.0), SIG)
    b = Guide(GuideConfig(ref_guide=0.5, edit_thr=0.4, dist_tau=1.0), SIG)
    cond, uncond = _cond_uncond()
    x0, ref = torch.zeros(1, 4, H, W), torch.ones(1, 4, H, W)
    a.apply(IN_WINDOW, x0, ref, ONES, ONES, cond_x0=cond, uncond_x0=uncond)
    assert a.edit_acc is not None and b.edit_acc is None


def test_debug_mode_reports_without_changing_the_result():
    import contextlib
    import io
    import tempfile
    x0, ref = _fields()
    plain = _guide(ref_guide=0.5, dist_tau=1.0).apply(LATE, x0, ref, ONES, ONES)
    log, cwd = io.StringIO(), os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            with contextlib.redirect_stdout(log):
                out = _guide(ref_guide=0.5, dist_tau=1.0, debug=True).apply(LATE, x0, ref, ONES, ONES)
            assert any(f.endswith("_dist.png") for _, _, files in os.walk(tmp) for f in files)
        finally:
            os.chdir(cwd)
    assert torch.equal(out, plain) and "guide step" in log.getvalue() and "dist" in log.getvalue()


def test_default_config_is_off():
    assert GuideConfig().ref_guide == 0.0
    x0, ref = _fields()
    for sigma in SIG[:-1]:
        assert torch.equal(Guide(GuideConfig(), SIG).apply(sigma, x0, ref, ONES, ONES), x0)


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
