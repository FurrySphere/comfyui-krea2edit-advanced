"""A ref_boost_mask is drawn on the reference image as the user supplied it, but the model sees that
image after it was center-cropped / letterboxed / resized to the target grid. The boost must land on
the tokens that show the masked content, whatever the transform.

Method: the reference is a bright rectangle on black, the mask marks the same rectangle. A fake VAE
keeps that brightness as the latent, so the content of every reference token is observable in what the
model's `first` layer receives. The boost weight the forward pass puts on each reference token must
match that content.

Run standalone:  python tests/test_mask_alignment.py
Or via pytest:   pytest tests/test_mask_alignment.py
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import comfy.patcher_extension as pe                                          # noqa: E402

from stubs import _FakeModelPatcher, _load_pack           # noqa: E402
from test_ref_controls import TXT, _FakeDiT                                   # noqa: E402

BOOST = 3.0


class _MarkerVAE:
    """Latent = 8x-pooled red channel, in all four channels."""

    def encode(self, pixels):
        m = F.avg_pool2d(pixels[..., 0][:, None], 8)
        return m.expand(-1, 4, -1, -1).contiguous()


class _RecordingDiT(_FakeDiT):
    def __init__(self):
        super().__init__()
        self.token_inputs = []
        self.first = self._first

    def _first(self, tokens):
        self.token_inputs.append(tokens)
        return torch.zeros(*tokens.shape[:2], 8)


def _rect(h, w, rect):
    y0, y1, x0, x1 = rect
    m = torch.zeros(1, h, w)
    m[:, round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)] = 1.0
    return m


def _image(ih, iw, rect):
    img = torch.zeros(1, ih, iw, 3)
    img[..., 0] = _rect(ih, iw, rect)
    return img


def _run(mod, H, W, ref_index=0, **patch_kw):
    """Run the real forward through the wrapper; returns (content of ref tokens, boost map of ref tokens)."""
    dm = _RecordingDiT()
    (m,) = mod.Krea2EditAdvanced.execute(_FakeModelPatcher(), **patch_kw)
    wrapper = m.model_options["transformer_options"]["wrappers"][pe.WrappersMP.DIFFUSION_MODEL]["krea2_edit_advanced"][0]
    ex = pe.WrapperExecutor(original=lambda *a, **k: None, class_obj=dm, wrappers=[wrapper], idx=0)
    ex.execute(torch.zeros(1, 4, H, W), torch.tensor([0.5]), torch.zeros(1, TXT, 8), None, {})
    bias = dm.block_inputs[0][1]
    toks = [t.mean(-1)[0] for t in dm.token_inputs[1:]]            # per ref: brightness per token
    off, maps = TXT, []
    for t in toks:
        n = t.shape[0]
        w = torch.exp(bias[0, 0, -1, off:off + n])                  # a target row's weight on this ref's tokens
        maps.append((w - 1) / (BOOST - 1))                          # back to the mask value in 0..1
        off += n
    return toks[ref_index], maps[ref_index]


def _err(content, boost_map):
    return float((content - boost_map).abs().mean())


TOL = 0.06
RECTS = [(0.10, 0.50, 0.60, 0.95), (0.55, 0.90, 0.05, 0.40), (0.30, 0.70, 0.30, 0.70)]

LATENT_CASES = [(512, 512, 64, 64), (512, 768, 64, 64), (768, 512, 64, 64), (640, 1024, 48, 80)]


def test_latent_path_mask_follows_the_reference_crop():
    mod = _load_pack()
    for ih, iw, H, W in LATENT_CASES:
        for rect in RECTS:
            lat = _MarkerVAE().encode(_image(ih, iw, rect))
            content, boost = _run(mod, H, W, source_latent_a={"samples": lat}, ref_boost_a=BOOST,
                                  ref_boost_mask_a=_rect(ih, iw, rect))
            assert _err(content, boost) < TOL, ((ih, iw, H, W), rect, _err(content, boost))


def test_each_reference_gets_its_own_transform():
    """a and b differ in aspect ratio, so each mask needs a different crop."""
    mod = _load_pack()
    ra, rb = RECTS[0], RECTS[1]
    kw = dict(source_latent_a={"samples": _MarkerVAE().encode(_image(512, 768, ra))},
              source_latent_b={"samples": _MarkerVAE().encode(_image(768, 512, rb))},
              ref_boost_a=BOOST, ref_boost_b=BOOST, ref_boost_mask_a=_rect(512, 768, ra), ref_boost_mask_b=_rect(768, 512, rb))
    for idx in (0, 1):
        content, boost = _run(mod, 64, 64, ref_index=idx, **kw)
        assert _err(content, boost) < TOL, (idx, _err(content, boost))


def test_unmasked_boost_is_unaffected_by_geometry():
    mod = _load_pack()
    _, boost = _run(mod, 64, 64, source_latent_a={"samples": _MarkerVAE().encode(_image(512, 768, RECTS[0]))}, ref_boost_a=BOOST)
    assert torch.allclose(boost, torch.ones_like(boost), atol=1e-5)


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
