"""Pins how a source latent is cropped and resized to the target grid.

Each golden is (first/last source row kept, first/last source column kept), read back from a ramp whose channels encode their own coordinates. The reference mask
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

from stubs import _load_pack   # noqa: E402


def _window(c0, c1, ih, iw):
    return (round(float(c1.min()) * (ih - 1)), round(float(c1.max()) * (ih - 1)),
            round(float(c0.min()) * (iw - 1)), round(float(c0.max()) * (iw - 1)))


# (source latent h, w, target latent h, w) -> (row0, row1, col0, col1) of the source kept; output is always the target size
LATENT_GOLDENS = {
    (64, 96, 64, 64): (0, 63, 16, 79),
    (96, 64, 64, 64): (16, 79, 0, 63),
    (48, 80, 64, 64): (0, 47, 16, 63),
    (64, 64, 64, 64): (0, 63, 0, 63),
}


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
