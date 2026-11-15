"""Wiring of the source hold: where the reference lands on the target grid, and the sampler hooks.

The placement must agree with the attention path (the RoPE ids krea2_edit_forward gives the reference), the
mask must follow the reference through the same crop/resize, and the hooks run through ComfyUI's own
cfg_function and wrapper executor.

Run standalone:  python tests/test_guide_hooks.py
Or via pytest:   pytest tests/test_guide_hooks.py
"""
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
    placed, weight = mod._place_ref(ref, torch.ones(1, 64, 64), 64, 64)
    assert torch.equal(placed, ref) and (weight == 1).all()
    placed, weight = mod._place_ref(torch.randn(1, 4, 63, 63), torch.ones(1, 63, 63), 63, 63)
    assert placed.shape == (1, 4, 63, 63) and weight.shape == (1, 1, 63, 63)


def test_reference_of_another_shape_is_fitted_like_the_attention_path():
    mod = _load_pack()
    ref = torch.randn(1, 4, 64, 96)
    placed, _ = mod._place_ref(ref, torch.ones(1, 64, 96), 64, 64)
    assert torch.allclose(placed, mod._fit_src(ref, 64, 64))


def test_mask_weight_is_the_mask_averaged_onto_the_grid():
    mod = _load_pack()
    mask = torch.zeros(1, 32, 32)
    mask[:, :16] = 1.0
    _, weight = mod._place_ref(torch.randn(1, 4, 64, 64), mask, 64, 64)
    assert torch.allclose(weight, F.interpolate(mask[None], size=(64, 64), mode="area"))


# ---------------------------------------------------------------- the mask follows the reference

def _alignment_error(ref, weight):
    return float((weight - ref[:, :1]).abs().mean())


def test_latent_mask_lands_on_the_content_it_marks():
    mod = _load_pack()
    for ih, iw, H, W in LATENT_CASES:
        for rect in RECTS:
            lat = _MarkerVAE().encode(_image(ih, iw, rect))
            ref, weight = mod._place_ref(lat, _rect(ih, iw, rect), H, W)
            assert _alignment_error(ref, weight) < TOL, ((ih, iw, H, W), rect)


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


FULL = dict(ref_guide=1.0, early_gain=1.0)                                   # copies the reference wherever the mask holds
HOLD_ALL = torch.ones(1, 16, 16)


def _guided(mod, opts, sigma, cond, uncond=None, scale=1.0, **cfg):
    """What the sampler would produce for one step: ComfyUI's cfg_function with our post-cfg hook inside it."""
    return comfy.samplers.cfg_function(None, cond, torch.zeros_like(cond) if uncond is None else uncond, scale, cond,
                                       torch.tensor([sigma]), model_options=opts.model_options, cond=None, uncond=uncond)


def test_guidance_pulls_the_sampled_estimate_toward_the_reference_inside_a_run():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(**FULL), ref, HOLD_ALL)
    out = _run(opts, lambda mo: _guided(mod, opts, EARLY, x0))
    assert torch.allclose(out, ref)
    out = _run(opts, lambda mo: _guided(mod, opts, LATE, x0))
    assert torch.allclose(out, ref)


def test_no_guidance_outside_a_sampling_run():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(**FULL), ref, HOLD_ALL)
    assert torch.equal(_guided(mod, opts, LATE, x0), x0)


def test_state_is_dropped_when_the_run_ends():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(**FULL), ref, HOLD_ALL)
    _run(opts, lambda mo: _guided(mod, opts, LATE, x0))
    assert torch.equal(_guided(mod, opts, LATE, x0), x0)


def test_state_is_dropped_even_if_the_sampler_raises():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(**FULL), ref, HOLD_ALL)

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
    calls, real = [], mod._place_ref
    mod._place_ref = lambda *a: calls.append(a[2:]) or real(*a)
    try:
        opts = _Opts()
        mod.add_guidance(opts, mod.GuideConfig(ref_guide=0.5), ref, HOLD_ALL)
        _run(opts, lambda mo: [_guided(mod, opts, s, x0) for s in SIG[:-1]])
        assert calls == [(16, 16)]
        _run(opts, lambda mo: _guided(mod, opts, LATE, x0))
        assert calls == [(16, 16), (16, 16)]
    finally:
        mod._place_ref = real


def test_batches_and_five_dimensional_latents_keep_their_shape():
    mod = _load_pack()
    ref = torch.randn(1, 4, 16, 16)
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(**FULL), ref, HOLD_ALL)
    batch = torch.randn(3, 4, 16, 16)
    out = _run(opts, lambda mo: _guided(mod, opts, LATE, batch))
    assert out.shape == batch.shape and all(torch.allclose(out[i], ref[0]) for i in range(3))
    video = torch.randn(1, 4, 1, 16, 16)
    out = _run(opts, lambda mo: _guided(mod, opts, LATE, video))
    assert out.shape == video.shape and torch.allclose(out[:, :, 0], ref)


class _Movable:
    """Stands in for a tensor built elsewhere (the source is encoded on whatever device the VAE used)."""

    def __init__(self, tensor, log):
        self.tensor, self.log = tensor, log

    def to(self, device):
        self.log.append(device)
        return self.tensor


def test_placed_tensors_are_moved_to_the_estimates_device_once_per_run():
    mod = _load_pack()
    x0, ref = _fields()
    log, real = [], mod._place_ref
    mod._place_ref = lambda *a: tuple(_Movable(t, log) for t in real(*a))
    try:
        opts = _Opts()
        mod.add_guidance(opts, mod.GuideConfig(ref_guide=0.5), ref, HOLD_ALL)
        _run(opts, lambda mo: [_guided(mod, opts, s, x0) for s in SIG[:-1]])
    finally:
        mod._place_ref = real
    assert log == [x0.device] * 2, log                                          # reference and weight, once


def test_guidance_works_when_the_reference_lives_on_another_device_than_the_estimate():
    if not torch.cuda.is_available():
        return                                                                  # CPU-only environment: nothing to mismatch
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(**FULL, dist_tau=1.0), ref, HOLD_ALL)   # reference stays on the CPU
    out = _run(opts, lambda mo: _guided(mod, opts, LATE, x0.cuda(), uncond=torch.zeros(SHAPE).cuda()))
    assert out.device.type == "cuda" and torch.allclose(out.cpu(), ref)


def test_cfg_1_optimization_stays_on():
    mod = _load_pack()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(), torch.ones(1, 4, 16, 16), HOLD_ALL)
    assert not opts.model_options.get("disable_cfg1_optimization")


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
