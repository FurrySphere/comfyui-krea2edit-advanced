"""Pins which part of a source latent is used for a picture and where it sits (see _window).

A source keeps its size. One that fits sits at x / y percent of the free space; a larger one is cut to the picture at the
same percentages of the surplus. Latent and masks follow this one window, so it must not drift.

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

# (source latent h, w, picture h, w, x percent, y percent) -> ((first row, last row, first col, last col) of the source that is used, (down, right) offset in the picture)
GOLDENS = {
    (64, 96, 64, 64, 50, 50): ((0, 63, 16, 79), (0, 0)),
    (64, 96, 64, 64, 0, 50): ((0, 63, 0, 63), (0, 0)),
    (64, 96, 64, 64, 100, 50): ((0, 63, 32, 95), (0, 0)),
    (96, 64, 64, 64, 50, 50): ((16, 79, 0, 63), (0, 0)),
    (96, 64, 64, 64, 50, 0): ((0, 63, 0, 63), (0, 0)),
    (128, 128, 64, 64, 50, 50): ((32, 95, 32, 95), (0, 0)),
    (128, 128, 64, 64, 100, 100): ((64, 127, 64, 127), (0, 0)),
    (48, 80, 64, 64, 50, 50): ((0, 47, 8, 71), (8, 0)),            # shorter than the picture: placed; wider: cut
    (48, 80, 64, 64, 0, 100): ((0, 47, 0, 63), (16, 0)),
    (32, 40, 64, 64, 50, 50): ((0, 31, 0, 39), (16, 12)),
    (32, 40, 64, 64, 100, 0): ((0, 31, 0, 39), (0, 24)),
    (64, 64, 64, 64, 50, 50): ((0, 63, 0, 63), (0, 0)),
}


def test_window_geometry_is_pinned():
    mod = _load_pack()
    for (sh, sw, H, W, x, y), (rows_cols, offset) in GOLDENS.items():
        (y0, x0, h, w), got_offset = mod._window((sh, sw), (H, W), x, y)
        assert (y0, y0 + h - 1, x0, x0 + w - 1) == rows_cols and got_offset == offset, ((sh, sw, H, W, x, y), (y0, y0 + h - 1, x0, x0 + w - 1), got_offset)


def test_the_latent_is_used_at_its_own_scale():
    """No resizing: every used cell of the picture holds the source cell the window names, unchanged."""
    mod = _load_pack()
    for (sh, sw, H, W, x, y), ((r0, r1, c0, c1), (oy, ox)) in GOLDENS.items():
        src = torch.zeros(1, 4, sh, sw)
        src[:, 0] = torch.arange(sw, dtype=torch.float32)[None, None, :]
        src[:, 1] = torch.arange(sh, dtype=torch.float32)[None, :, None]
        placed, _, valid = mod._place_ref(src, torch.zeros(1, sh, sw), float(x), float(y), 0.0, 0.0, (H, W))
        used = placed[..., oy:oy + (r1 - r0 + 1), ox:ox + (c1 - c0 + 1)]
        assert torch.equal(used[0, 0], src[0, 0, r0:r1 + 1, c0:c1 + 1]) and torch.equal(used[0, 1], src[0, 1, r0:r1 + 1, c0:c1 + 1]), (sh, sw, H, W, x, y)
        assert float(valid.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1)


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
