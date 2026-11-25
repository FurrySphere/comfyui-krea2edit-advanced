"""Wiring of the source hold: where the reference lands on the target grid, and the sampler hooks.

The placement must agree with the attention path (the RoPE ids krea2_edit_forward gives the reference), the
mask must follow the reference through the same crop/resize, and the hooks run through ComfyUI's own
cfg_function and wrapper executor.

Run standalone:  python tests/test_guide_hooks.py
Or via pytest:   pytest tests/test_guide_hooks.py
"""
import functools
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

def _place(mod, ref, mask, size, x=50.0, y=50.0, grow=0.0, feather=0.0):
    return mod._place_ref(ref, mask, x, y, grow, feather, size)


def test_reference_fills_the_grid_and_odd_targets_are_cropped():
    mod = _load_pack()
    ref = torch.randn(1, 4, 64, 64)
    placed, edit, valid = _place(mod, ref, torch.ones(1, 64, 64), (64, 64))
    assert torch.equal(placed, ref) and (valid == 1).all() and edit.shape == (1, 512, 512)
    placed, edit, valid = _place(mod, torch.randn(1, 4, 63, 63), torch.ones(1, 63, 63), (63, 63))
    assert placed.shape == (1, 4, 63, 63) and valid.shape == (1, 1, 63, 63) and edit.shape == (1, 504, 504) and (valid == 1).all()


def test_reference_larger_than_the_grid_is_cut_at_its_own_scale():
    mod = _load_pack()
    ref = torch.randn(1, 4, 64, 96)
    for x, c0 in ((50.0, 16), (0.0, 0), (100.0, 32)):
        placed, _, valid = _place(mod, ref, torch.ones(1, 64, 96), (64, 64), x=x)
        assert torch.equal(placed, ref[..., c0:c0 + 64]) and (valid == 1).all(), x


def test_reference_smaller_than_the_grid_keeps_its_size_and_sits_where_the_percentages_say():
    mod = _load_pack()
    ref = torch.randn(1, 4, 40, 64)
    for x, y, rows, cols in ((50.0, 50.0, 12, 0), (0.0, 0.0, 0, 0), (100.0, 100.0, 24, 0), (50.0, 25.0, 6, 0)):
        placed, edit, valid = _place(mod, ref, torch.zeros(1, 40, 64), (64, 80), x=x, y=y)
        cols = round(x / 100 * 16)
        assert torch.equal(placed[..., rows:rows + 40, cols:cols + 64], ref), (x, y)
        assert float(valid.sum()) == 40 * 64 and (valid[..., rows:rows + 40, cols:cols + 64] == 1).all(), (x, y)
        assert (edit[0, rows * 8:(rows + 40) * 8, cols * 8:(cols + 64) * 8] == 0).all() and float(edit.sum()) == 512 * 640 - 320 * 512


def test_edit_mask_weight_is_the_drawn_mask_on_the_grid():
    mod = _load_pack()
    mask = torch.zeros(1, 32, 32)
    mask[:, :16] = 1.0
    _, edit, _ = _place(mod, torch.randn(1, 4, 64, 64), mask, (64, 64))
    got, want = F.interpolate(edit[None], size=(64, 64), mode="area"), F.interpolate(mask[None], size=(64, 64), mode="area")
    assert torch.allclose(got[..., :30, :], want[..., :30, :]) and torch.allclose(got[..., 34:, :], want[..., 34:, :])      # a mask of lower resolution is soft across its own edge
    assert float((got - want).abs().mean()) < 0.01


# ---------------------------------------------------------------- the mask follows the reference

def _alignment_error(ref, weight):
    return float((weight - ref[:, :1]).abs().mean())


def test_latent_mask_lands_on_the_content_it_marks():
    mod = _load_pack()
    for ih, iw, H, W in LATENT_CASES:
        for rect in RECTS:
            for x, y in ((50.0, 50.0), (0.0, 100.0), (100.0, 0.0)):       # the mask follows the cut wherever it is made
                lat = _MarkerVAE().encode(_image(ih, iw, rect))
                ref, edit, _ = _place(mod, lat, _rect(ih, iw, rect), (H, W), x=x, y=y)
                weight = F.interpolate(edit[None], size=(H, W), mode="area")
                assert _alignment_error(ref, weight) < TOL, ((ih, iw, H, W), rect, x, y)


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
BLACK = torch.zeros(1, 16, 16)                                               # a mask that keeps the whole picture


def _hold_all(mod, ref):
    """The placement of a source the size of the sampled picture, kept everywhere."""
    return functools.partial(mod._place_ref, ref, BLACK, 50.0, 50.0, 0.0, 0.0)


def _guided(mod, opts, sigma, cond, uncond=None, scale=1.0, **cfg):
    """What the sampler would produce for one step: ComfyUI's cfg_function with our post-cfg hook inside it."""
    return comfy.samplers.cfg_function(None, cond, torch.zeros_like(cond) if uncond is None else uncond, scale, cond,
                                       torch.tensor([sigma]), model_options=opts.model_options, cond=None, uncond=uncond)


def test_guidance_pulls_the_sampled_estimate_toward_the_reference_inside_a_run():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(**FULL), _hold_all(mod, ref))
    out = _run(opts, lambda mo: _guided(mod, opts, EARLY, x0))
    assert torch.allclose(out, ref)
    out = _run(opts, lambda mo: _guided(mod, opts, LATE, x0))
    assert torch.allclose(out, ref)


def test_no_guidance_outside_a_sampling_run():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(**FULL), _hold_all(mod, ref))
    assert torch.equal(_guided(mod, opts, LATE, x0), x0)


def test_state_is_dropped_when_the_run_ends():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(**FULL), _hold_all(mod, ref))
    _run(opts, lambda mo: _guided(mod, opts, LATE, x0))
    assert torch.equal(_guided(mod, opts, LATE, x0), x0)


def test_state_is_dropped_even_if_the_sampler_raises():
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(**FULL), _hold_all(mod, ref))

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
    mod._place_ref = lambda *a: calls.append(a[-1]) or real(*a)
    try:
        opts = _Opts()
        mod.add_guidance(opts, mod.GuideConfig(ref_guide=0.5), _hold_all(mod, ref))
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
    mod.add_guidance(opts, mod.GuideConfig(**FULL), _hold_all(mod, ref))
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
    mod._place_ref = lambda *a: (lambda ref, edit, valid: (_Movable(ref, log), edit, _Movable(valid, log)))(*real(*a))
    try:
        opts = _Opts()
        mod.add_guidance(opts, mod.GuideConfig(ref_guide=0.5), _hold_all(mod, ref))
        _run(opts, lambda mo: [_guided(mod, opts, s, x0) for s in SIG[:-1]])
    finally:
        mod._place_ref = real
    assert log == [x0.device] * 2, log                                          # reference and valid cells, once


def test_guidance_works_when_the_reference_lives_on_another_device_than_the_estimate():
    if not torch.cuda.is_available():
        return                                                                  # CPU-only environment: nothing to mismatch
    mod = _load_pack()
    x0, ref = _fields()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(**FULL, dist_tau=1.0), _hold_all(mod, ref))   # reference stays on the CPU
    out = _run(opts, lambda mo: _guided(mod, opts, LATE, x0.cuda(), uncond=torch.zeros(SHAPE).cuda()))
    assert out.device.type == "cuda" and torch.allclose(out.cpu(), ref)


def test_cfg_1_optimization_stays_on():
    mod = _load_pack()
    opts = _Opts()
    mod.add_guidance(opts, mod.GuideConfig(), _hold_all(mod, torch.ones(1, 4, 16, 16)))
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
