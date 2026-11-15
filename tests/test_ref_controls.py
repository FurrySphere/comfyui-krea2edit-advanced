"""Tests for how reference inputs and their per-reference controls reach the forward pass.

Contract: every `*_a` input (source_latent_a, source_image_a, ref_boost_a, ref_boost_mask_a,
ref_position_a) acts on the first reference, every `*_b` input on the second. Which controls
are in play never depends on how many references are connected.

Run standalone:  python tests/test_ref_controls.py
Or via pytest:   pytest tests/test_ref_controls.py
"""
import math
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from test_pre_encode import _FakeModelPatcher, _load_pack, _run_steps   # noqa: E402

TXT, GRID, TOK = 3, 8, 64          # text tokens; target/ref grid (patch 2 on a 16x16 latent); tokens per image
LOG = math.log


class _ValueVAE:
    """Encodes an image to a latent filled with the image's (constant) pixel value."""

    def encode(self, pixels):
        b, h, w, _ = pixels.shape
        return torch.full((b, 4, h // 8, w // 8), float(pixels.mean()))


class _FakeDiT:
    """Just enough of SingleStreamDiT for krea2_edit_forward; records what the blocks and RoPE see."""
    patch, channels, tdim = 2, 4, 8

    def __init__(self, d=8):
        self.first = nn.Linear(16, d)
        self.tmlp = nn.Linear(8, d)
        self.tproj = nn.Linear(d, d)
        self._last = nn.Linear(d, 16)
        self.pos = None
        self.block_inputs = []
        self.blocks = [self._block]

    def _unpack_context(self, c): return c
    def txtfusion(self, c, mask=None, transformer_options=None): return c
    def txtmlp(self, c): return c
    def last(self, combined, t): return self._last(combined)

    def pe_embedder(self, pos):
        self.pos = pos
        return None

    def _block(self, combined, tvec, freqs, attn_bias, transformer_options=None):
        self.block_inputs.append((tuple(combined.shape), attn_bias))
        return combined


def _src(v, size=16):
    return torch.full((1, 4, size, size), float(v))


def _forward(mod, srcs, **kw):
    m = _FakeDiT()
    x = torch.zeros(1, 4, 16, 16)
    out = mod.krea2_edit_forward(m, x, torch.tensor([0.5]), torch.zeros(1, TXT, 8), srcs, {}, **kw)
    assert out.shape == x.shape
    shape, bias = m.block_inputs[0]
    return m, shape, bias


def _target_rows(bias, nref):
    return bias[0, 0, TXT + nref * TOK:]


def _ref_cols(bias, nref, i):
    """Distinct bias values target rows apply to ref i's key columns."""
    s = TXT + i * TOK
    return sorted(set(_target_rows(bias, nref)[:, s:s + TOK].flatten().tolist()))


def _ref_pos(m, i):
    s = TXT + i * TOK
    return m.pos[0, s:s + TOK]


# ---------------------------------------------------------------- node interface

def test_input_names_are_a_b_symmetric():
    mod = _load_pack()
    it = mod.Krea2EditModelPatch.INPUT_TYPES()
    assert list(it["required"]) == ["model"]
    assert list(it["optional"]) == [
        "source_latent_a", "source_latent_b", "ref_boost_a", "ref_boost_b", "fit_mode", "ref_boost_mask_a", "ref_boost_mask_b",
        "vae", "source_image_a", "source_image_b", "target_latent", "ref_position_a", "ref_position_b"]
    names = list(it["required"]) + list(it["optional"])
    for n in names:
        if n.endswith("_b"):
            assert n[:-1] + "a" in names, f"{n} has no _a counterpart"


def test_widget_order_matches_single_ref_workflows_saved_before_the_rename():
    """Widgets serialize positionally. Old order was [ref_boost, ref_boost_a, fit_mode, ref_position,
    ref_position_a]; in single-ref workflows slot 0/3 controlled the only image, which is now `a`."""
    mod = _load_pack()
    it = mod.Krea2EditModelPatch.INPUT_TYPES()
    widgets = [n for n, spec in it["optional"].items()
               if spec[0] in ("FLOAT", "INT", "STRING") or isinstance(spec[0], list)]
    assert widgets == ["ref_boost_a", "ref_boost_b", "fit_mode", "ref_position_a", "ref_position_b"]


# ---------------------------------------------------------------- sequence layout

def test_forward_sequence_and_rope_frames():
    mod = _load_pack()
    m, shape, _ = _forward(mod, [_src(1), _src(2)])
    assert shape == (1, TXT + 3 * TOK, 8)                       # [text | ref1 | ref2 | target]
    assert (m.pos[0, :TXT] == 0).all()
    assert (_ref_pos(m, 0)[:, 0] == 1).all()
    assert (_ref_pos(m, 1)[:, 0] == 2).all()
    assert (m.pos[0, TXT + 2 * TOK:, 0] == 0).all()             # target = frame 0


def test_single_tensor_source_equals_one_item_list():
    mod = _load_pack()
    m1, shape1, _ = _forward(mod, _src(1))
    m2, shape2, _ = _forward(mod, [_src(1)])
    assert shape1 == shape2 == (1, TXT + 2 * TOK, 8)
    assert torch.equal(m1.pos, m2.pos)


# ---------------------------------------------------------------- ref_boost: a -> first ref, b -> second ref

def test_no_boost_means_no_bias():
    mod = _load_pack()
    assert _forward(mod, [_src(1), _src(2)])[2] is None


def test_single_ref_ref_boost_a_hits_the_only_ref():
    mod = _load_pack()
    _, _, bias = _forward(mod, [_src(1)], ref_boost_a=4.0)
    assert set(_ref_cols(bias, 1, 0)) == {_f32(LOG(4.0))}
    assert (bias[0, 0, :TXT + TOK] == 0).all()                  # only target rows are biased


def test_single_ref_ref_boost_b_is_inert():
    """No b reference -> ref_boost_b has nothing to act on, not even a bias tensor is built."""
    mod = _load_pack()
    assert _forward(mod, [_src(1)], ref_boost_b=5.0)[2] is None


def test_single_ref_ref_boost_b_does_not_leak_onto_a():
    mod = _load_pack()
    _, _, bias = _forward(mod, [_src(1)], ref_boost_a=4.0, ref_boost_b=9.0)
    assert set(_ref_cols(bias, 1, 0)) == {_f32(LOG(4.0))}


def test_two_refs_boosts_follow_their_suffix():
    mod = _load_pack()
    _, _, bias = _forward(mod, [_src(1), _src(2)], ref_boost_a=2.0, ref_boost_b=4.0)
    assert set(_ref_cols(bias, 2, 0)) == {_f32(LOG(2.0))}
    assert set(_ref_cols(bias, 2, 1)) == {_f32(LOG(4.0))}
    assert (bias[0, 0, :TXT + 2 * TOK] == 0).all()
    assert (_target_rows(bias, 2)[:, :TXT] == 0).all()          # text columns untouched
    assert (_target_rows(bias, 2)[:, TXT + 2 * TOK:] == 0).all()  # target->target untouched

    _, _, bias = _forward(mod, [_src(1), _src(2)], ref_boost_a=4.0, ref_boost_b=2.0)   # swapped values
    assert set(_ref_cols(bias, 2, 0)) == {_f32(LOG(4.0))}
    assert set(_ref_cols(bias, 2, 1)) == {_f32(LOG(2.0))}


def test_two_refs_each_boost_alone_leaves_the_other_ref_unbiased():
    mod = _load_pack()
    _, _, bias = _forward(mod, [_src(1), _src(2)], ref_boost_b=4.0)
    assert set(_ref_cols(bias, 2, 0)) == {0.0} and set(_ref_cols(bias, 2, 1)) == {_f32(LOG(4.0))}
    _, _, bias = _forward(mod, [_src(1), _src(2)], ref_boost_a=4.0)
    assert set(_ref_cols(bias, 2, 0)) == {_f32(LOG(4.0))} and set(_ref_cols(bias, 2, 1)) == {0.0}


def test_boost_below_floor_is_clamped():
    mod = _load_pack()
    _, _, bias = _forward(mod, [_src(1)], ref_boost_a=0.0)
    assert set(_ref_cols(bias, 1, 0)) == {_f32(LOG(1e-4))}


# ---------------------------------------------------------------- ref_boost_mask: a -> first ref, b -> second ref

def _top_half_mask():
    m = torch.zeros(1, 16, 16)
    m[:, :8] = 1.0
    return m


def _top_half_only(bias, nref, i, boost):
    v = _target_rows(bias, nref)[0, TXT + i * TOK:TXT + (i + 1) * TOK].reshape(GRID, GRID)
    assert (v[:4] == _f32(LOG(boost))).all() and (v[4:] == 0).all()


def test_single_ref_mask_a_restricts_the_only_ref():
    mod = _load_pack()
    _, _, bias = _forward(mod, [_src(1)], ref_boost_a=3.0, ref_boost_mask_a=_top_half_mask())
    _top_half_only(bias, 1, 0, 3.0)


def test_single_ref_mask_b_is_ignored():
    mod = _load_pack()
    _, _, bias = _forward(mod, [_src(1)], ref_boost_a=3.0, ref_boost_mask_b=_top_half_mask())
    assert set(_ref_cols(bias, 1, 0)) == {_f32(LOG(3.0))}


def test_two_refs_each_mask_restricts_only_its_own_ref():
    mod = _load_pack()
    _, _, bias = _forward(mod, [_src(1), _src(2)], ref_boost_a=2.0, ref_boost_b=3.0,
                          ref_boost_mask_b=_top_half_mask())
    assert set(_ref_cols(bias, 2, 0)) == {_f32(LOG(2.0))}       # a: unmasked
    _top_half_only(bias, 2, 1, 3.0)

    _, _, bias = _forward(mod, [_src(1), _src(2)], ref_boost_a=2.0, ref_boost_b=3.0,
                          ref_boost_mask_a=_top_half_mask())
    _top_half_only(bias, 2, 0, 2.0)
    assert set(_ref_cols(bias, 2, 1)) == {_f32(LOG(3.0))}       # b: unmasked


def test_mask_has_no_effect_while_its_own_boost_is_neutral():
    mod = _load_pack()
    _, _, bias = _forward(mod, [_src(1), _src(2)], ref_boost_a=2.0, ref_boost_mask_b=_top_half_mask())
    assert set(_ref_cols(bias, 2, 0)) == {_f32(LOG(2.0))}
    assert set(_ref_cols(bias, 2, 1)) == {0.0}


def _soft(bias, i=0, nref=1):
    return _target_rows(bias, nref)[0, TXT + i * TOK:TXT + (i + 1) * TOK].reshape(GRID, GRID)


def test_mask_is_proportional_not_thresholded():
    """A mask value m scales the boost: the key's attention weight becomes 1 + m * (boost - 1)."""
    mod = _load_pack()
    for boost, m in ((3.0, 0.5), (3.0, 0.3), (3.0, 0.9), (0.5, 0.5), (8.0, 0.1)):
        _, _, bias = _forward(mod, [_src(1)], ref_boost_a=boost, ref_boost_mask_a=torch.full((1, 16, 16), m))
        assert torch.allclose(_soft(bias), torch.full((GRID, GRID), _f32(LOG(1 + m * (boost - 1)))), atol=1e-6), (boost, m)


def test_mask_is_averaged_over_each_token_before_weighting():
    mod = _load_pack()
    mask = torch.zeros(1, 16, 16)
    mask[:, ::2, ::2] = 1.0                                        # one of the four cells under every token
    _, _, bias = _forward(mod, [_src(1)], ref_boost_a=3.0, ref_boost_mask_a=mask)
    assert torch.allclose(_soft(bias), torch.full((GRID, GRID), _f32(LOG(1 + 0.25 * 2.0))), atol=1e-6)


def test_gradient_mask_gives_a_gradient_of_weights():
    mod = _load_pack()
    ramp = torch.linspace(0, 1, 16)[None, None, :].expand(1, 16, 16).contiguous()
    _, _, bias = _forward(mod, [_src(1)], ref_boost_a=3.0, ref_boost_mask_a=ramp)
    row = _soft(bias)[0]
    assert (row[1:] > row[:-1]).all() and row[0] > 0 and row[-1] < _f32(LOG(3.0))


def test_zero_and_one_mask_values_match_off_and_full_boost():
    mod = _load_pack()
    _, _, bias = _forward(mod, [_src(1)], ref_boost_a=3.0, ref_boost_mask_a=torch.zeros(1, 16, 16))
    assert (_soft(bias) == 0).all()
    _, _, bias = _forward(mod, [_src(1)], ref_boost_a=3.0, ref_boost_mask_a=torch.ones(1, 16, 16))
    assert (_soft(bias) == _f32(LOG(3.0))).all()


def test_mask_accepts_2d_and_batched_shapes():
    mod = _load_pack()
    for mask in (_top_half_mask()[0], _top_half_mask().repeat(2, 1, 1)):
        _, _, bias = _forward(mod, [_src(1)], ref_boost_a=3.0, ref_boost_mask_a=mask)
        _top_half_only(bias, 1, 0, 3.0)


# ---------------------------------------------------------------- ref_position: a -> first ref, b -> second ref

def _w(m, i):
    return _ref_pos(m, i)[:, 2]


BESIDE = "beside (right of target)"


def test_aligned_refs_sit_on_the_target_grid():
    mod = _load_pack()
    m, _, _ = _forward(mod, [_src(1), _src(2)])
    for i in (0, 1):
        assert _w(m, i).min() == 0 and _w(m, i).max() == GRID - 1
    assert torch.equal(_ref_pos(m, 0)[:, 1:], _ref_pos(m, 1)[:, 1:])


def test_single_ref_ref_position_a_moves_the_only_ref():
    mod = _load_pack()
    m, _, _ = _forward(mod, [_src(1)], ref_position_a=BESIDE)
    assert _w(m, 0).min() == GRID and _w(m, 0).max() == 2 * GRID - 1


def test_single_ref_ref_position_b_is_ignored():
    mod = _load_pack()
    m, _, _ = _forward(mod, [_src(1)], ref_position_b=BESIDE)
    assert _w(m, 0).min() == 0


def test_two_refs_positions_follow_their_suffix():
    mod = _load_pack()
    m, _, _ = _forward(mod, [_src(1), _src(2)], ref_position_a=BESIDE)
    assert _w(m, 0).min() == GRID and _w(m, 1).min() == 0

    m, _, _ = _forward(mod, [_src(1), _src(2)], ref_position_b=BESIDE)
    assert _w(m, 0).min() == 0 and _w(m, 1).min() == GRID


def test_two_beside_refs_stack_to_the_right_in_order():
    mod = _load_pack()
    m, _, _ = _forward(mod, [_src(1), _src(2)], ref_position_a=BESIDE, ref_position_b=BESIDE)
    assert _w(m, 0).min() == GRID and _w(m, 1).min() == 2 * GRID


def test_bag_of_features_collapses_h_w_to_the_target_centre():
    mod = _load_pack()
    for kw, bag, kept in ((dict(ref_position_b="bag of features"), 1, 0),
                          (dict(ref_position_a="bag of features"), 0, 1)):
        m, _, _ = _forward(mod, [_src(1), _src(2)], **kw)
        p = _ref_pos(m, bag)
        assert (p[:, 1] == (GRID - 1) / 2).all() and (p[:, 2] == (GRID - 1) / 2).all()
        assert (p[:, 0] == bag + 1).all()                        # frame axis untouched
        assert _w(m, kept).max() == GRID - 1                     # the other ref still aligned


def test_unknown_ref_position_raises():
    mod = _load_pack()
    try:
        _forward(mod, [_src(1)], ref_position_a="nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------------------------------------------------------------- patch(): source selection & routing

def _capture(mod):
    seen = []
    mod.krea2_edit_forward = lambda dm, x, t, ctx, src, to, **k: seen.append((src, k))
    return seen


def _wrapper(mod, latent_a, **kw):
    import comfy.patcher_extension as pe
    (m,) = mod.Krea2EditModelPatch().patch(_FakeModelPatcher(), {"samples": latent_a}, **kw)
    return m.model_options["transformer_options"]["wrappers"][pe.WrappersMP.DIFFUSION_MODEL]["krea2_edit"][0]


def _img(v, px=512):
    return torch.full((1, px, px, 3), float(v))


def _val(t):
    return round(float(t.mean()), 2)


def test_latent_path_single_and_pair():
    mod = _load_pack()
    seen = _capture(mod)
    _run_steps(_wrapper(mod, _src(1)), n=1)
    assert _val(seen[-1][0]) == 1.0 and torch.is_tensor(seen[-1][0])

    _run_steps(_wrapper(mod, _src(1), source_latent_b={"samples": _src(2)}), n=1)
    src = seen[-1][0]
    assert [_val(s) for s in src] == [1.0, 2.0]


def test_pixel_path_replaces_latents_entirely():
    mod = _load_pack()
    seen = _capture(mod)
    w = _wrapper(mod, _src(1), source_latent_b={"samples": _src(2)}, vae=_ValueVAE(),
                 source_image_a=_img(0.25), source_image_b=_img(0.5))
    _run_steps(w, n=1)
    assert [_val(s) for s in seen[-1][0]] == [0.25, 0.5]


def test_pixel_path_without_second_image_drops_the_second_latent():
    mod = _load_pack()
    seen = _capture(mod)
    w = _wrapper(mod, _src(1), source_latent_b={"samples": _src(2)}, vae=_ValueVAE(),
                 source_image_a=_img(0.25))
    _run_steps(w, n=1)
    src = seen[-1][0]
    assert torch.is_tensor(src) and _val(src) == 0.25


def test_second_image_without_first_image_is_ignored():
    mod = _load_pack()
    seen = _capture(mod)
    w = _wrapper(mod, _src(1), source_latent_b={"samples": _src(2)}, vae=_ValueVAE(),
                 source_image_b=_img(0.5))
    _run_steps(w, n=1)
    assert [_val(s) for s in seen[-1][0]] == [1.0, 2.0]


def test_images_without_vae_use_the_latent_path():
    mod = _load_pack()
    seen = _capture(mod)
    _run_steps(_wrapper(mod, _src(1), source_image_a=_img(0.25)), n=1)
    assert _val(seen[-1][0]) == 1.0


def test_controls_are_forwarded_unchanged():
    mod = _load_pack()
    seen = _capture(mod)
    ma, mb = torch.ones(1, 16, 16), torch.zeros(1, 16, 16)
    w = _wrapper(mod, _src(1), ref_boost_a=4.0, ref_boost_b=2.0, ref_boost_mask_a=ma, ref_boost_mask_b=mb,
                 ref_position_a=BESIDE, ref_position_b="bag of features")
    _run_steps(w, n=1)
    k = seen[-1][1]
    assert (k["ref_boost_a"], k["ref_boost_b"]) == (4.0, 2.0)
    assert k["ref_boost_mask_a"] is ma and k["ref_boost_mask_b"] is mb
    assert (k["ref_position_a"], k["ref_position_b"]) == (BESIDE, "bag of features")


def test_fit_geometry_flags_follow_pixel_path_and_fit_mode():
    mod = _load_pack()
    seen = _capture(mod)
    pix = dict(vae=_ValueVAE(), source_image_a=_img(0.25))
    for kw, native, pos in (
            (dict(**pix, fit_mode="fit"), True, "stride1"),
            (dict(**pix, fit_mode="crop (legacy)"), False, "anchor"),
            (dict(fit_mode="fit"), False, "stride1")):          # no pixel path: pos_mode still follows fit_mode
        _run_steps(_wrapper(mod, _src(1), **kw), n=1)
        k = seen[-1][1]
        assert (k["ref_native"], k["pos_mode"]) == (native, pos), kw.keys()


# ---------------------------------------------------------------- Krea2EditGroundedEncode: image_a / image_b

class _FakeCLIP:
    def tokenize(self, prompt, **kw):
        return {"prompt": prompt, **kw}

    def encode_from_tokens_scheduled(self, tokens):
        return tokens


def _encode(mod, **kw):
    return mod.Krea2EditGroundedEncode().encode(_FakeCLIP(), "edit it", **kw)[0]


def test_encode_inputs_follow_the_a_b_naming():
    mod = _load_pack()
    it = mod.Krea2EditGroundedEncode.INPUT_TYPES()
    assert list(it["optional"])[:2] == ["image_a", "image_b"]


def test_encode_without_images_is_text_only():
    mod = _load_pack()
    tokens = _encode(mod)
    assert "images" not in tokens and "llama_template" not in tokens


def test_encode_one_and_two_images_use_matching_vision_blocks():
    mod = _load_pack()
    a, b = _img(0.25, 64), _img(0.5, 64)
    one = _encode(mod, image_a=a)
    two = _encode(mod, image_a=a, image_b=b)
    assert len(one["images"]) == 1 and one["llama_template"].count("<|image_pad|>") == 1
    assert len(two["images"]) == 2 and two["llama_template"].count("<|image_pad|>") == 2
    assert _val(two["images"][0]) == 0.25 and _val(two["images"][1]) == 0.5      # order: a, b


def test_encode_image_b_alone_falls_back_to_text_only():
    mod = _load_pack()
    assert "images" not in _encode(mod, image_b=_img(0.5, 64))


def test_encode_grounding_px_caps_the_longest_side():
    mod = _load_pack()
    capped = _encode(mod, image_a=_img(0.25, 1024), grounding_px=256)["images"][0]
    assert capped.shape[1:3] == (256, 256)
    native = _encode(mod, image_a=_img(0.25, 1024), grounding_px=0)["images"][0]
    assert native.shape[1:3] == (1024, 1024)


def test_old_input_names_fail_loudly_instead_of_dropping_the_references():
    """A pre-rename workflow still sends these; the node must say so, not run without references."""
    mod = _load_pack()
    for old, new in (("source_latent", "source_latent_a"), ("source_image", "source_image_a"),
                     ("ref_boost", "ref_boost_a"), ("ref_boost_mask", "ref_boost_mask_a"),
                     ("ref_position", "ref_position_a")):
        try:
            _wrapper(mod, _src(1), **{old: 1})
        except ValueError as e:
            assert old in str(e) and new in str(e), str(e)
        else:
            raise AssertionError(f"{old} was silently accepted")
    try:
        mod.Krea2EditGroundedEncode().encode(_FakeCLIP(), "p", image=_img(0.25, 64), image_b=_img(0.5, 64))
    except ValueError as e:
        assert "image -> image_a" in str(e)
    else:
        raise AssertionError("old `image` input was silently accepted")


def test_unknown_inputs_only_warn():
    mod = _load_pack()
    _capture(mod)
    _wrapper(mod, _src(1), some_future_input=1)


def _f32(x):
    return float(torch.tensor(x, dtype=torch.float32))


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
