"""Wiring of reference guidance: where the reference lands on the target grid, and the sampler hooks.

The placement must agree with the attention path (the RoPE ids krea2_edit_forward gives the reference), the
mask must follow the reference through the same crop/resize, and the hooks run through ComfyUI's own
cfg_function and wrapper executor.

Run standalone:  python tests/test_guide_hooks.py
Or via pytest:   pytest tests/test_guide_hooks.py
"""
import contextlib
import io
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import comfy.model_patcher                                                    # noqa: E402
import comfy.patcher_extension as pe                                          # noqa: E402
import comfy.samplers                                                         # noqa: E402

from test_mask_alignment import LATENT_CASES, RECTS, TOL, _MarkerVAE, _image, _rect   # noqa: E402
from stubs import _load_pack                                  # noqa: E402


# ---------------------------------------------------------------- placement

def test_reference_fills_the_grid_and_odd_targets_are_cropped():
    mod = _load_pack()
    ref = torch.randn(1, 4, 64, 64)
    placed, cover, weight = mod._place_ref(ref, None, 64, 64)
    assert torch.equal(placed, ref) and (cover == 1).all() and (weight == 1).all()
    placed, cover, weight = mod._place_ref(torch.randn(1, 4, 63, 63), None, 63, 63)
    assert placed.shape == (1, 4, 63, 63) and cover.shape == weight.shape == (1, 1, 63, 63) and (cover == 1).all()


def test_reference_of_another_shape_is_fitted_like_the_attention_path():
    mod = _load_pack()
    ref = torch.randn(1, 4, 64, 96)
    placed, cover, _ = mod._place_ref(ref, None, 64, 64)
    assert torch.allclose(placed, mod._fit_src(ref, 64, 64)) and (cover == 1).all()


def test_mask_weight_is_the_mask_averaged_onto_the_grid():
    mod = _load_pack()
    mask = torch.zeros(1, 32, 32)
    mask[:, :16] = 1.0
    _, _, weight = mod._place_ref(torch.randn(1, 4, 64, 64), mask, 64, 64)
    assert torch.allclose(weight, F.interpolate(mask[None], size=(64, 64), mode="area"))


def test_without_a_mask_the_weight_is_one_everywhere():
    mod = _load_pack()
    assert (mod._place_ref(torch.randn(1, 4, 20, 30), None, 40, 64)[2] == 1).all()


# ---------------------------------------------------------------- the mask follows the reference

def _alignment_error(ref, cover, weight):
    return float(((weight * cover) - ref[:, :1]).abs().mean())


def test_latent_mask_lands_on_the_content_it_marks():
    mod = _load_pack()
    for ih, iw, H, W in LATENT_CASES:
        for rect in RECTS:
            lat = _MarkerVAE().encode(_image(ih, iw, rect))
            ref, cover, weight = mod._place_ref(lat, _rect(ih, iw, rect), H, W)
            assert _alignment_error(ref, cover, weight) < TOL, ((ih, iw, H, W), rect)


# ---------------------------------------------------------------- hooks

SIG = [1.0, 0.89, 0.775, 0.657, 0.535, 0.408, 0.277, 0.141, 0.0]
EARLY, LATE = SIG[0], SIG[6]
SHAPE = (1, 4, 16, 16)


class _Opts:
    """A model patcher that just keeps what the guidance installs, with ComfyUI's own option setters."""

    def __init__(self):
        self.model_options = {}

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options = comfy.model_patcher.set_model_options_post_cfg_function(self.model_options, fn, disable_cfg1_optimization)

    def set_model_sampler_calc_cond_batch_function(self, fn):
        self.model_options["sampler_calc_cond_batch_function"] = fn


class _Guider:
    def __init__(self, cfg):
        self.cfg = cfg


class _Sampler:
    def __init__(self, body):
        self.body = body

    def sample(self, guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar):
        return self.body(extra_args["model_options"])


def _run(opts, body, cfg=1.0, sigmas=SIG):
    """One sampling call as ComfyUI's inner_sample makes it: through the SAMPLER_SAMPLE wrappers."""
    wrappers = comfy.patcher_extension.get_all_wrappers(pe.WrappersMP.SAMPLER_SAMPLE, opts.model_options, is_model_options=True)
    sampler = _Sampler(body)
    executor = pe.WrapperExecutor.new_class_executor(sampler.sample, sampler, wrappers)
    return executor.execute(_Guider(cfg), torch.tensor(sigmas), {"model_options": opts.model_options}, None, None, None, None, False)


def _fields(seed=0, shape=SHAPE):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)


def _provider(ref, calls=None):
    h, w = ref.shape[-2:]

    def provider(H, W, mask):
        if calls is not None:
            calls.append((H, W))
        return ref, torch.ones(1, 1, h, w), torch.ones(1, 1, h, w)
    return provider


def _guided(mod, opts, sigma, cond, uncond=None, scale=1.0, **cfg):
    """What the sampler would produce for one step: ComfyUI's cfg_function with our post-cfg hook inside it."""
    return comfy.samplers.cfg_function(None, cond, torch.zeros_like(cond) if uncond is None else uncond, scale, cond,
                                       torch.tensor([sigma]), model_options=opts.model_options, cond=None, uncond=uncond)


def test_guidance_pulls_the_sampled_estimate_toward_the_reference_inside_a_run():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(ref_guide=1.0), _provider(ref), None)
    out = _run(opts, lambda mo: _guided(mod, opts, EARLY, x0))
    assert torch.allclose(out, ref)
    out = _run(opts, lambda mo: _guided(mod, opts, LATE, x0))
    assert torch.allclose(out, ref)


def test_no_guidance_outside_a_sampling_run():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(ref_guide=1.0), _provider(ref), None)
    assert torch.equal(_guided(mod, opts, LATE, x0), x0)


def test_state_is_dropped_when_the_run_ends():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(ref_guide=1.0), _provider(ref), None)
    _run(opts, lambda mo: _guided(mod, opts, LATE, x0))
    assert torch.equal(_guided(mod, opts, LATE, x0), x0)


def test_state_is_dropped_even_if_the_sampler_raises():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(ref_guide=1.0), _provider(ref), None)

    def boom(mo):
        raise RuntimeError("sampler failed")
    try:
        _run(opts, boom)
    except RuntimeError:
        pass
    assert torch.equal(_guided(mod, opts, LATE, x0), x0)


def test_reference_is_placed_once_per_size_per_run():
    mod = _load_pack()
    x0, ref = _fields()
    calls, opts = [], _Opts()
    mod.add_guidance(opts, mod.GuideConfig(ref_guide=0.5), _provider(ref, calls), None)
    _run(opts, lambda mo: [_guided(mod, opts, s, x0) for s in SIG[:-1]])
    assert calls == [(16, 16)]
    _run(opts, lambda mo: _guided(mod, opts, LATE, x0))
    assert calls == [(16, 16), (16, 16)]


def test_batches_and_five_dimensional_latents_keep_their_shape():
    mod = _load_pack()
    ref = torch.randn(1, 4, 16, 16)
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(ref_guide=1.0), _provider(ref), None)
    batch = torch.randn(3, 4, 16, 16)
    out = _run(opts, lambda mo: _guided(mod, opts, LATE, batch))
    assert out.shape == batch.shape and all(torch.allclose(out[i], ref[0]) for i in range(3))
    video = torch.randn(1, 4, 1, 16, 16)
    out = _run(opts, lambda mo: _guided(mod, opts, LATE, video))
    assert out.shape == video.shape and torch.allclose(out[:, :, 0], ref)


class _Movable:
    """Stands in for a tensor the provider built elsewhere (the reference is encoded on whatever device the VAE used)."""

    def __init__(self, tensor, log):
        self.tensor, self.log = tensor, log

    def to(self, device):
        self.log.append(device)
        return self.tensor


def test_placed_tensors_are_moved_to_the_estimates_device_once_per_run():
    mod = _load_pack()
    x0, ref = _fields()
    ones = torch.ones(1, 1, 16, 16)
    log, opts = [], _Opts()
    mod.add_guidance(opts, mod.GuideConfig(ref_guide=0.5), lambda H, W, mask: tuple(_Movable(t, log) for t in (ref, ones, ones)), None)
    _run(opts, lambda mo: [_guided(mod, opts, s, x0) for s in SIG[:-1]])
    assert log == [x0.device] * 3, log                                          # reference, coverage and weight, once


def test_guidance_works_when_the_reference_lives_on_another_device_than_the_estimate():
    if not torch.cuda.is_available():
        return                                                                  # CPU-only environment: nothing to mismatch
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(ref_guide=1.0, dist_tau=1.0, edit_thr=0.4), _provider(ref), None)   # reference stays on the CPU
    out = _run(opts, lambda mo: _guided(mod, opts, LATE, x0.cuda(), uncond=torch.zeros(SHAPE).cuda()))
    assert out.device.type == "cuda" and torch.allclose(out.cpu(), ref)


def test_unaligned_reference_leaves_the_estimate_alone_and_warns_once():
    mod = _load_pack()
    x0, _ = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(ref_guide=1.0), lambda H, W, mask: None, None)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        outs = _run(opts, lambda mo: [_guided(mod, opts, s, x0) for s in SIG[:-1]])
    assert all(torch.equal(o, x0) for o in outs)
    assert log.getvalue().count("needs ref_position_a 'aligned'") == 1


def test_runs_do_not_share_the_edit_map():
    mod = _load_pack()
    ref = torch.ones(1, 4, 16, 16)
    cond = torch.zeros(1, 4, 16, 16)
    uncond = torch.zeros(1, 4, 16, 16)
    uncond[..., :8, :] = 4.0                                                   # the instruction changes the top half
    cfg = mod.GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0)
    opts = _Opts()
    mod.add_guidance(opts, cfg, _provider(ref), None)

    def steps(mo):
        for s in SIG[:-1]:
            comfy.samplers.cfg_function(None, cond, uncond, 1.0, cond, torch.tensor([s]), model_options=mo, cond=None, uncond=[{}])
        return comfy.samplers.cfg_function(None, cond, uncond, 1.0, cond, torch.tensor([LATE]), model_options=mo, cond=None, uncond=[{}])
    first = _run(opts, steps)
    assert first[..., :8, :].mean() < 0.1 and first[..., 8:, :].mean() > 0.4     # edited half is free, the rest pulled
    second = _run(opts, lambda mo: _guided(mod, opts, LATE, cond))               # a fresh run starts without that map
    assert torch.allclose(second, torch.full_like(second, 0.5))


def test_edit_gate_without_a_negative_conditioning_does_nothing_and_does_not_fail():
    mod = _load_pack()
    ref = torch.ones(1, 4, 16, 16)
    cond = torch.full((1, 4, 16, 16), 3.0)
    cond[..., :6, :6] = -3.0                                                    # differs from the dropped (zero) unconditional pass
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(ref_guide=0.5, edit_thr=0.4, smooth=0.0), _provider(ref), None)
    out = _run(opts, lambda mo: [_guided(mod, opts, s, cond, uncond=None) for s in SIG[:-1]][-1])
    assert torch.allclose(out, cond + 0.5 * (ref - cond))                       # plain pull everywhere, no phantom edit map


# ---------------------------------------------------------------- the forced unconditional pass

def _calc_cond_recorder(seen):
    real = comfy.samplers.calc_cond_batch

    def fake(model, conds, x_in, timestep, model_options):
        seen.append([c is not None for c in conds])
        return [torch.zeros_like(x_in) for _ in conds]
    comfy.samplers.calc_cond_batch = fake
    return real


def test_unconditional_pass_is_forced_only_when_the_edit_gate_is_on():
    mod = _load_pack()
    ref = torch.ones(1, 4, 16, 16)
    for cfg, forced in ((mod.GuideConfig(ref_guide=0.5), False), (mod.GuideConfig(ref_guide=0.5, dist_tau=1.0), False),
                        (mod.GuideConfig(ref_guide=0.5, edit_thr=0.4), True)):
        opts = _Opts()
        mod.add_guidance(opts, cfg, _provider(ref), None)
        assert bool(opts.model_options.get("disable_cfg1_optimization")) is forced
        assert ("sampler_calc_cond_batch_function" in opts.model_options) is forced


def _calls_per_step(mod, scale):
    ref = torch.ones(1, 4, 16, 16)
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(ref_guide=0.5, edit_thr=0.4), _provider(ref), None)
    seen = []
    real = _calc_cond_recorder(seen)
    try:
        def body(mo):
            x = torch.zeros(1, 4, 16, 16)
            for s in SIG[:-1]:
                args = {"conds": [[{}], [{}]], "input": x, "sigma": torch.tensor([s]), "model": None, "model_options": mo}
                mo["sampler_calc_cond_batch_function"](args)
        _run(opts, body, cfg=scale)
    finally:
        comfy.samplers.calc_cond_batch = real
    return seen


def test_at_cfg_1_the_unconditional_pass_runs_only_inside_the_edit_window():
    mod = _load_pack()
    seen = _calls_per_step(mod, 1.0)
    guide = mod.Guide(mod.GuideConfig(ref_guide=0.5, edit_thr=0.4), SIG)
    assert seen == [[True, guide.needs_uncond(s)] for s in SIG[:-1]]
    assert any(c[1] for c in seen) and not all(c[1] for c in seen)


def test_above_cfg_1_the_unconditional_pass_always_runs():
    mod = _load_pack()
    assert all(c == [True, True] for c in _calls_per_step(mod, 3.0))


def test_dropping_the_unconditional_pass_leaves_the_conditional_result_at_cfg_1():
    """What cfg_function makes of the dropped pass (zeros): exactly the conditional prediction."""
    cond = torch.randn(1, 4, 8, 8)
    out = comfy.samplers.cfg_function(None, cond, torch.zeros_like(cond), 1.0, cond, torch.tensor([0.5]), model_options={})
    assert torch.equal(out, cond)


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
