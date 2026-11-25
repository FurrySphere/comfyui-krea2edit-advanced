"""A source that fits inside the picture keeps its size and sits where ref_x_percent / ref_y_percent say; the picture around it is drawn freely.

The attention path (RoPE ids of the reference tokens) and the hold (the cells the source covers) have to agree on where the source is.

Run standalone:  python tests/test_placement.py
Or via pytest:   pytest tests/test_placement.py
"""
import os
import random
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stubs import _load_pack, _run_steps   # noqa: E402
from test_ref_controls import TXT, _capture, _FakeDiT, _forward, _src, _wrapper   # noqa: E402


def _ids(m, first, count):
    return m.pos[0, first:first + count]


def test_a_smaller_source_keeps_its_size_and_is_centered_on_the_rope_grid():
    mod = _load_pack()
    m, shape, _ = _forward(mod, [torch.zeros(1, 4, 8, 12)])               # 4 x 6 tokens in an 8 x 8 token picture
    assert shape == (1, TXT + 24 + 64, 8)
    ids = _ids(m, TXT, 24)
    assert (ids[:, 0] == 1).all()
    assert sorted(set(ids[:, 1].tolist())) == [2.0, 3.0, 4.0, 5.0] and sorted(set(ids[:, 2].tolist())) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_the_percentages_move_the_source_through_the_free_space():
    mod = _load_pack()
    for x, y, rows, cols in ((0.0, 0.0, 0.0, 0.0), (100.0, 100.0, 4.0, 2.0), (0.0, 100.0, 4.0, 0.0), (100.0, 0.0, 0.0, 2.0), (50.0, 25.0, 1.0, 1.0)):
        m, _, _ = _forward(mod, [torch.zeros(1, 4, 8, 12)], ref_x_percent=x, ref_y_percent=y)
        ids = _ids(m, TXT, 24)
        assert ids[:, 1].min().item() == rows and ids[:, 2].min().item() == cols, (x, y)
        assert ids[:, 1].max().item() == rows + 3 and ids[:, 2].max().item() == cols + 5, (x, y)


def test_a_source_as_large_as_the_picture_is_not_moved_by_the_percentages():
    mod = _load_pack()
    for x, y in ((0.0, 0.0), (100.0, 100.0)):
        m, _, _ = _forward(mod, [_src(1)], ref_x_percent=x, ref_y_percent=y)
        ids = _ids(m, TXT, 64)
        assert ids[:, 1].min().item() == 0 and ids[:, 1].max().item() == 7 and ids[:, 2].min().item() == 0 and ids[:, 2].max().item() == 7


def test_the_percentages_act_on_source_a_and_the_second_source_is_centered():
    mod = _load_pack()
    m, _, _ = _forward(mod, [torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8)], ref_x_percent=0.0, ref_y_percent=0.0)
    a, b = _ids(m, TXT, 16), _ids(m, TXT + 16, 16)
    assert (a[:, 0] == 1).all() and a[:, 1].min().item() == 0 and a[:, 2].min().item() == 0
    assert (b[:, 0] == 2).all() and b[:, 1].min().item() == 2 and b[:, 2].min().item() == 2


def test_an_axis_is_placed_when_it_fits_and_cut_when_it_is_larger():
    mod = _load_pack()
    assert mod._axis(8, 16, 50.0) == (0, 8, 4) and mod._axis(8, 16, 0.0) == (0, 8, 0) and mod._axis(8, 16, 100.0) == (0, 8, 8)
    assert mod._axis(24, 16, 50.0) == (4, 16, 0) and mod._axis(24, 16, 0.0) == (0, 16, 0) and mod._axis(24, 16, 100.0) == (8, 16, 0)
    assert mod._axis(16, 16, 30.0) == (0, 16, 0)                           # as large as the picture: nothing to move
    assert mod._axis(7, 8, 50.0) == (0, 7, 0) and mod._axis(9, 8, 50.0) == (0, 8, 0)       # padded to whole tokens: 7 -> 8 fits, 9 does not
    for pct in range(0, 101, 7):
        for s, t in ((8, 16), (24, 16), (15, 16), (17, 16)):
            first, used, offset = mod._axis(s, t, float(pct))
            assert isinstance(first, int) and isinstance(offset, int) and first >= 0 and offset >= 0 and first + used <= s and offset + used <= t + 1


def test_an_odd_source_is_placed_by_its_padded_size_but_holds_only_its_own_cells():
    mod = _load_pack()
    ref = torch.randn(1, 4, 7, 11)                                       # padded to 8 x 12 by the forward
    placed, edit, valid = mod._place_ref(ref, torch.zeros(1, 7, 11), 50.0, 50.0, 0.0, 0.0, (16, 16))
    assert float(valid.sum()) == 77 and valid[0, 0, 4:11, 2:13].min() == 1 and valid[0, 0, 4:11, 2:13].sum() == 77     # one cell less than the tokens cover
    assert torch.equal(placed[..., 4:11, 2:13], ref) and (placed[..., 11:, :] == 0).all()
    assert float((edit == 0).sum()) == 77 * 64
    m, _, _ = _forward(mod, [ref])
    ids = _ids(m, TXT, 24)
    assert ids[:, 1].min().item() == 2 and ids[:, 2].min().item() == 1     # the tokens sit at the same cell offset


def test_the_rope_offset_and_the_holds_cells_agree_for_any_sizes():
    """Odd latent sizes of the source and of the picture included, larger sources too: the model and the hold must see the same cells."""
    mod = _load_pack()
    rng = random.Random(0)
    for _ in range(200):
        H, W = rng.randint(6, 40), rng.randint(6, 40)
        hs, ws = rng.randint(3, H + 12), rng.randint(3, W + 12)                # a source that fits, or one that is larger
        x, y = rng.choice([0.0, 50.0, 100.0, rng.uniform(0, 100)]), rng.choice([0.0, 50.0, 100.0, rng.uniform(0, 100)])
        src = torch.randn(1, 4, hs, ws)
        m = _FakeDiT()
        mod.krea2_edit_forward(m, torch.zeros(1, 4, H, W), torch.tensor([0.5]), torch.zeros(1, TXT, 8), src, {}, ref_x_percent=x, ref_y_percent=y)
        placed, edit, valid = mod._place_ref(src, torch.zeros(1, hs, ws), x, y, 0.0, 0.0, (H, W))
        assert placed.shape[-2:] == (H, W) and valid.shape[-2:] == (H, W) and edit.shape[-2:] == (H * 8, W * 8)
        n_ref = m.block_inputs[0][0][1] - TXT - (-(-H // 2)) * (-(-W // 2))
        ids = _ids(m, TXT, n_ref)
        cells = valid[0, 0].nonzero()
        assert (ids[:, 1].min().item() * 2, ids[:, 2].min().item() * 2) == (cells[:, 0].min().item(), cells[:, 1].min().item()), (H, W, hs, ws, x, y)
        (y0, x0, h, w), (oy, ox) = mod._window((hs, ws), (H, W), x, y)
        assert torch.equal(placed[..., oy:oy + min(h, H - oy), ox:ox + min(w, W - ox)], src[..., y0:y0 + min(h, H - oy), x0:x0 + min(w, W - ox)])


def test_a_larger_source_is_cut_to_the_picture_and_the_tokens_follow():
    mod = _load_pack()
    m, shape, _ = _forward(mod, [torch.zeros(1, 4, 16, 24)], ref_x_percent=100.0)       # 8 x 12 tokens, only 8 x 8 are inside
    assert shape == (1, TXT + 64 + 64, 8)
    ids = _ids(m, TXT, 64)
    assert ids[:, 1].min().item() == 0 and ids[:, 1].max().item() == 7 and ids[:, 2].min().item() == 0 and ids[:, 2].max().item() == 7


def test_the_boost_mask_follows_the_cut_wherever_it_is_made():
    mod = _load_pack()
    left = torch.zeros(1, 16, 24)
    left[..., :12] = 1.0                                                                  # white on the left half of the source
    for x, boosted_columns in ((0.0, 6), (100.0, 2), (50.0, 4)):                          # tokens of the picture that show white: cells 0..11 / 8..11 / 4..11
        _, _, bias = _forward(mod, [torch.zeros(1, 4, 16, 24)], ref_boost_a=3.0, ref_boost_mask_a=left, ref_x_percent=x)
        row = torch.exp(bias[0, 0, -1, TXT:TXT + 64]).reshape(8, 8)
        assert torch.allclose(row[:, :boosted_columns], torch.full((8, boosted_columns), 3.0), atol=1e-4), x
        assert torch.allclose(row[:, boosted_columns:], torch.ones(8, 8 - boosted_columns), atol=1e-4), x


def test_the_edit_mask_output_follows_the_cut():
    mod = _load_pack()
    drawn = torch.zeros(1, 128, 192)
    drawn[..., 96:] = 1.0                                                                 # drawn at the source's pixel size: white from cell 12 on
    for x, white_from in ((0.0, 96), (50.0, 64), (100.0, 32)):                            # the picture shows cells 0..15 / 4..19 / 8..23 of the 24
        _, mask = _run_node(mod, drawn, x)
        assert mask.shape == (1, 128, 128)
        assert (mask[..., white_from:] == 1).all() and (mask[..., :white_from] == 0).all(), x


def _run_node(mod, drawn, x):
    from stubs import _FakeModelPatcher

    class P(_FakeModelPatcher):
        def clone(self):
            c = P()
            c.model = self.model
            return c
    mod.add_guidance = lambda m, cfg, place: None
    pos = {"ref_position_a": "aligned (trained)", "source_hold": True, "hold_strength": 0.7, "grow_percent": 0.0, "feather_percent": 0.0,
           "edit_mask": drawn, "ref_x_percent": x, "ref_y_percent": 50.0}
    return mod.Krea2EditAdvanced.execute(P(), {"samples": torch.zeros(1, 4, 16, 24)}, pos, target_latent={"samples": torch.zeros(1, 4, 16, 16)}).args


def test_a_source_that_is_larger_on_one_axis_and_smaller_on_the_other_follows_each_rule():
    """Tall and narrow in a square picture: the rows are cut, the columns are placed with gaps on both sides."""
    mod = _load_pack()
    src = torch.randn(1, 4, 24, 8)
    (y0, x0, h, w), (oy, ox) = mod._window((24, 8), (16, 16), 50.0, 50.0)
    assert (y0, h, oy) == (4, 16, 0) and (x0, w, ox) == (0, 8, 4)
    placed, edit, valid = mod._place_ref(src, torch.zeros(1, 24, 8), 50.0, 50.0, 0.0, 0.0, (16, 16))
    assert torch.equal(placed[..., :, 4:12], src[..., 4:20, :]) and (placed[..., :, :4] == 0).all() and (placed[..., :, 12:] == 0).all()
    assert float(valid.sum()) == 16 * 8 and (valid[..., :, 4:12] == 1).all()
    assert (edit[0, :, :32] == 1).all() and (edit[0, :, 96:] == 1).all() and (edit[0, :, 32:96] == 0).all()      # free gaps either side, kept source between
    m, shape, _ = _forward(mod, [src])
    assert shape == (1, TXT + 32 + 64, 8)                                                    # 8 x 4 tokens of source, none outside the picture
    ids = _ids(m, TXT, 32)
    assert ids[:, 1].min().item() == 0 and ids[:, 1].max().item() == 7 and ids[:, 2].min().item() == 2 and ids[:, 2].max().item() == 5


def test_masks_follow_each_rule_in_a_mixed_case():
    mod = _load_pack()
    top = torch.zeros(1, 24, 8)
    top[:, :12] = 1.0                                                                       # white on the upper half of the tall source (rows 0..11)
    for y, boosted_rows in ((0.0, 6), (50.0, 4), (100.0, 2)):                              # token rows showing white: source rows 0..11 / 4..11 / 8..11
        _, _, bias = _forward(mod, [torch.zeros(1, 4, 24, 8)], ref_boost_a=3.0, ref_boost_mask_a=top, ref_y_percent=y)
        row = torch.exp(bias[0, 0, -1, TXT:TXT + 32]).reshape(8, 4)
        assert torch.allclose(row[:boosted_rows], torch.full((boosted_rows, 4), 3.0), atol=1e-4), y
        assert torch.allclose(row[boosted_rows:], torch.ones(8 - boosted_rows, 4), atol=1e-4), y
    drawn = torch.zeros(1, 192, 64)                                                         # the edit mask, drawn at the source's pixel size: white in its top half
    drawn[:, :96] = 1.0
    for y, white_rows in ((0.0, 96), (50.0, 64), (100.0, 32)):                             # the picture shows rows 0..15 / 4..19 / 8..23 of the 24 cells: white px rows
        _, mask = _run_node_tall(mod, drawn, y)
        assert mask.shape == (1, 128, 128)
        assert (mask[:, :white_rows, 32:96] == 1).all() and (mask[:, white_rows:, 32:96] == 0).all(), y
        assert (mask[:, :, :32] == 1).all() and (mask[:, :, 96:] == 1).all(), y            # the gaps either side are always free


def _run_node_tall(mod, drawn, y):
    from stubs import _FakeModelPatcher

    class P(_FakeModelPatcher):
        def clone(self):
            c = P()
            c.model = self.model
            return c
    mod.add_guidance = lambda m, cfg, place: None
    pos = {"ref_position_a": "aligned (trained)", "source_hold": True, "hold_strength": 0.7, "grow_percent": 0.0, "feather_percent": 0.0,
           "edit_mask": drawn, "ref_x_percent": 50.0, "ref_y_percent": y}
    return mod.Krea2EditAdvanced.execute(P(), {"samples": torch.zeros(1, 4, 24, 8)}, pos, target_latent={"samples": torch.zeros(1, 4, 16, 16)}).args


def test_the_node_passes_the_placement_to_the_forward():
    mod = _load_pack()
    seen = _capture(mod)
    w = _wrapper(mod, _src(1), ref_position_a={"ref_position_a": "aligned (trained)", "source_hold": False, "ref_x_percent": 20.0, "ref_y_percent": 80.0})
    _run_steps(w, n=1)
    assert (seen[-1][1]["ref_x_percent"], seen[-1][1]["ref_y_percent"]) == (20.0, 80.0)
    seen.clear()
    _run_steps(_wrapper(mod, _src(1)), n=1)                               # not given: centered
    assert (seen[-1][1]["ref_x_percent"], seen[-1][1]["ref_y_percent"]) == (50.0, 50.0)


def test_the_mask_follows_the_placed_source_into_the_boost_bias():
    """A boost mask drawn on the source is not cropped or resized when the source fits: it covers the source's own tokens."""
    mod = _load_pack()
    top = torch.zeros(1, 8, 12)
    top[:, :4] = 1.0
    _, _, bias = _forward(mod, [torch.zeros(1, 4, 8, 12)], ref_boost_a=3.0, ref_boost_mask_a=top)
    row = torch.exp(bias[0, 0, -1, TXT:TXT + 24]).reshape(4, 6)
    assert torch.allclose(row[:2], torch.full((2, 6), 3.0), atol=1e-4) and torch.allclose(row[2:], torch.ones(2, 6), atol=1e-4)


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
