"""Pins how a reference image / source latent is cropped and resized to the target grid.

Each golden is (encoded height, encoded width, first/last source row kept, first/last source column
kept), read back from a ramp image whose channels encode their own coordinates. The reference mask
has to follow exactly this window, so this geometry must not drift.

Run standalone:  python tests/test_fit_geometry.py
Or via pytest:   pytest tests/test_fit_geometry.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from test_pre_encode import _load_pack   # noqa: E402


class _RecordingVAE:
    def encode(self, pixels):
        self.pixels = pixels
        b, h, w, _ = pixels.shape
        return torch.zeros(b, 4, h // 8, w // 8)


def _ramp(ih, iw):
    """Image whose channel 0 is the column and channel 1 the row, both normalized to 0..1."""
    img = torch.zeros(1, ih, iw, 3)
    img[..., 0] = torch.linspace(0, 1, iw)[None, None, :]
    img[..., 1] = torch.linspace(0, 1, ih)[None, :, None]
    return img


def _window(c0, c1, ih, iw):
    return (round(float(c1.min()) * (ih - 1)), round(float(c1.max()) * (ih - 1)),
            round(float(c0.min()) * (iw - 1)), round(float(c0.max()) * (iw - 1)))


# (image h, image w, target latent h, target latent w, mode) -> (nh, nw, row0, row1, col0, col1)
IMAGE_GOLDENS = {
    (512, 512, 64, 64, "crop"): (512, 512, 0, 511, 0, 511),
    (512, 512, 64, 64, "fit"): (512, 512, 0, 511, 0, 511),
    (512, 768, 64, 64, "crop"): (512, 512, 0, 511, 128, 639),
    (512, 768, 64, 64, "fit"): (336, 512, 4, 507, 0, 767),
    (768, 512, 64, 64, "crop"): (512, 512, 128, 639, 0, 511),
    (768, 512, 64, 64, "fit"): (512, 336, 0, 767, 4, 507),
    (500, 508, 64, 64, "crop"): (512, 512, 0, 499, 4, 503),
    (500, 508, 64, 64, "fit"): (512, 512, 0, 499, 4, 503),           # near-matched AR: minimal crop, fills the grid
    (1024, 640, 48, 80, "crop"): (384, 640, 320, 703, 0, 639),
    (1024, 640, 48, 80, "fit"): (384, 240, 1, 1022, 1, 638),
    (600, 1000, 64, 48, "crop"): (512, 384, 0, 599, 275, 724),
    (600, 1000, 64, 48, "fit"): (224, 384, 9, 589, 1, 998),
    (333, 777, 40, 56, "crop"): (320, 448, 0, 332, 155, 620),
    (333, 777, 40, 56, "fit"): (192, 448, 0, 332, 0, 776),
}

# (source latent h, w, target latent h, w) -> (row0, row1, col0, col1) of the source kept; output is always the target size
LATENT_GOLDENS = {
    (64, 96, 64, 64): (0, 63, 16, 79),
    (96, 64, 64, 64): (16, 79, 0, 63),
    (48, 80, 64, 64): (0, 47, 16, 63),
    (64, 64, 64, 64): (0, 63, 0, 63),
}


def test_image_fit_geometry_is_pinned():
    mod = _load_pack()
    for (ih, iw, H, W, mode), (nh, nw, *win) in IMAGE_GOLDENS.items():
        vae = _RecordingVAE()
        mod._fit_encode_image(_ramp(ih, iw), vae, H, W, {}, ("a", H, W), mode)
        p = vae.pixels
        assert tuple(p.shape[1:3]) == (nh, nw), (ih, iw, H, W, mode)
        got = _window(p[..., 0], p[..., 1], ih, iw)
        assert all(abs(g - e) <= 1 for g, e in zip(got, win)), ((ih, iw, H, W, mode), got, win)


def test_latent_fit_geometry_is_pinned():
    mod = _load_pack()
    for (sh, sw, H, W), win in LATENT_GOLDENS.items():
        src = torch.zeros(1, 4, sh, sw)
        src[:, 0] = torch.linspace(0, 1, sw)[None, None, :]
        src[:, 1] = torch.linspace(0, 1, sh)[None, :, None]
        out = mod._fit_src(src, H, W)
        assert tuple(out.shape[-2:]) == (H, W)
        got = _window(out[:, 0], out[:, 1], sh, sw)
        assert all(abs(g - e) <= 1 for g, e in zip(got, win)), ((sh, sw, H, W), got, win)


def test_encode_is_cached_per_target_and_mode():
    mod = _load_pack()
    vae, cache = _RecordingVAE(), {}
    n = []
    real = vae.encode
    vae.encode = lambda px: (n.append(1), real(px))[1]
    img = _ramp(512, 768)
    for _ in range(3):
        mod._fit_encode_image(img, vae, 64, 64, cache, ("a", 64, 64), "fit")
    mod._fit_encode_image(img, vae, 64, 64, cache, ("a", 64, 64), "crop")
    assert len(n) == 2


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
