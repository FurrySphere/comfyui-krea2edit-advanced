"""With `vae` + `source_image_a` the pixel path owns the references: the latent inputs
(`source_latent_a`, `source_latent_b`) must contribute nothing, and the working resolution
comes from `target_latent` (pre-encode) or the sampled latent, never from a source latent.

Run standalone:  python tests/test_pixel_path.py
Or via pytest:   pytest tests/test_pixel_path.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import comfy.patcher_extension as pe                                          # noqa: E402

from test_pre_encode import _latent, _load_pack                              # noqa: E402
from test_ref_controls import _capture, _img, _src, _val                     # noqa: E402


class _SpyInner:
    """Records every tensor that goes through process_latent_in."""

    def __init__(self):
        self.processed = []

    def process_latent_in(self, x):
        self.processed.append(x)
        return x


class _SpyPatcher:
    def __init__(self, inner):
        self.model = inner
        self.model_options = {}

    def clone(self):
        c = _SpyPatcher(self.model)
        c.model_options = dict(self.model_options)
        return c


class _RecordingVAE:
    """Encodes to a latent filled with the pixel mean and remembers the pixel resolutions asked for."""

    def __init__(self):
        self.pixels = []

    def encode(self, pixels):
        b, h, w, _ = pixels.shape
        self.pixels.append((h, w))
        return torch.full((b, 4, h // 8, w // 8), float(pixels.mean()))


# Latent that is wrong in every way: NaN, another size, another channel count.
POISON = {"samples": torch.full((1, 16, 3, 5), float("nan"))}


def _wide(v, h, w):
    """An image with the target's aspect ratio, so `fit` fills the target grid exactly."""
    return torch.full((1, h, w, 3), float(v))


def _patch(mod, source_latent_a, inner=None, **kw):
    inner = inner or _SpyInner()
    (m,) = mod.Krea2EditModelPatch().patch(_SpyPatcher(inner), source_latent_a, **kw)
    return m.model_options["transformer_options"]["wrappers"][pe.WrappersMP.DIFFUSION_MODEL]["krea2_edit"][0], inner


def _run(wrapper, h=64, w=64, n=2):
    for _ in range(n):
        pe.WrapperExecutor(original=lambda *a, **k: None, class_obj=object(), wrappers=[wrapper], idx=0).execute(
            torch.zeros(1, 4, h, w), None, None, None, {})


def _forwarded(mod, source_latent_a, source_latent_b=None, h=64, w=64, **kw):
    seen = _capture(mod)
    wrapper, _ = _patch(mod, source_latent_a, source_latent_b=source_latent_b, vae=_RecordingVAE(),
                        source_image_a=_img(0.25), source_image_b=_img(0.5), target_latent=_latent(h, w), **kw)
    _run(wrapper, h, w, n=1)
    return seen[-1][0]


def test_pixel_path_result_does_not_depend_on_source_latent_a_or_b():
    mod = _load_pack()
    clean = _forwarded(mod, {"samples": _src(1, 64)}, {"samples": _src(2, 64)})
    poisoned = _forwarded(mod, POISON, POISON)
    assert [_val(s) for s in clean] == [0.25, 0.5]
    assert all(torch.equal(c, p) for c, p in zip(clean, poisoned))


def test_pixel_path_resolution_comes_from_target_latent():
    mod = _load_pack()
    seen = _capture(mod)
    vae = _RecordingVAE()
    wrapper, _ = _patch(mod, {"samples": _src(1, 8)}, vae=vae, source_image_a=_wide(0.25, 384, 640),
                        source_image_b=_wide(0.5, 384, 640), target_latent=_latent(48, 80))
    assert vae.pixels == [(384, 640), (384, 640)], "encoded at the target_latent size, not the source latent's"
    _run(wrapper, 48, 80)
    assert vae.pixels == [(384, 640), (384, 640)], "sampling at the same size reuses the pre-encode"
    assert seen[-1][0][0].shape[-2:] == (48, 80)


def test_pixel_path_resolution_without_target_latent_comes_from_the_sampled_latent():
    mod = _load_pack()
    seen = _capture(mod)
    vae = _RecordingVAE()
    wrapper, _ = _patch(mod, POISON, vae=vae, source_image_a=_wide(0.25, 320, 448))
    assert vae.pixels == []
    _run(wrapper, 40, 56, n=1)
    assert vae.pixels == [(320, 448)]
    assert seen[-1][0].shape[-2:] == (40, 56)


def test_pixel_path_never_processes_the_source_latents():
    """Nothing may even be computed from them: process_latent_in only sees the image encodes."""
    mod = _load_pack()
    _capture(mod)
    wrapper, inner = _patch(mod, POISON, source_latent_b=POISON, vae=_RecordingVAE(),
                            source_image_a=_img(0.25), source_image_b=_img(0.5), target_latent=_latent())
    _run(wrapper, n=1)
    assert not any(t is POISON["samples"] for t in inner.processed), "a source latent was processed"
    assert len(inner.processed) == 2 and all(not torch.isnan(t).any() for t in inner.processed)


def test_source_latent_a_can_be_omitted_on_the_pixel_path():
    mod = _load_pack()
    with_latent = _forwarded(mod, {"samples": _src(1, 64)})
    without = _forwarded(mod, None)
    assert [_val(s) for s in without] == [0.25, 0.5]
    assert all(torch.equal(a, b) for a, b in zip(with_latent, without))


def test_source_latent_a_is_required_without_the_pixel_path():
    mod = _load_pack()
    for kw in ({}, {"source_image_a": _img(0.25)}, {"vae": _RecordingVAE()},
               {"source_latent_b": {"samples": _src(2, 64)}}):
        try:
            _patch(mod, None, **kw)
        except ValueError as e:
            assert "source_latent_a" in str(e) and "source_image_a" in str(e)
        else:
            raise AssertionError(f"expected ValueError for {list(kw)}")


def test_latent_path_still_processes_both_latents():
    mod = _load_pack()
    a, b = {"samples": _src(1, 64)}, {"samples": _src(2, 64)}
    _, inner = _patch(mod, a, source_latent_b=b)
    assert inner.processed == [a["samples"], b["samples"]]


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
