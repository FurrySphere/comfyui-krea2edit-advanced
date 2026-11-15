"""Contract and regression snapshot of the nodes: the input list with types, defaults and limits, the outputs, the registration, and a
snapshot of what krea2_edit_forward builds (sequence layout, RoPE ids, attention bias) for the main input combinations.

These pin the behavior of the node layer so that it can be rebuilt (new schema, new inputs) without the engine changing under it. A deliberate
change of the node interface updates the frozen tables below in the same commit.

Run standalone:  python tests/test_node_contract.py        (add --print to print the snapshot table)
Or via pytest:   pytest tests/test_node_contract.py
"""
import asyncio
import contextlib
import io
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stubs import _FakeModelPatcher, _load_pack   # noqa: E402
from test_ref_controls import TXT, _FakeCLIP, _FakeDiT, _src, _top_half_mask   # noqa: E402

POSITIONS = ["aligned (trained)", "beside (right of target)", "bag of features"]
FLOAT_BOOST = {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001}

HOLD_INPUTS = {          # the inputs that only exist while reference a is aligned (the dynamic combo option)
    "source_hold": ("required", "BOOLEAN", {"default": True}),
    "hold_strength": ("required", "FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.05}),
    "grow_percent": ("required", "FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.5}),
    "feather_percent": ("required", "FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5}),
    "edit_mask": ("optional", "MASK", {}),
}
PATCH_INPUTS = {          # name -> (group, type, options without the tooltip), in declaration order
    "model": ("required", "MODEL", {}),
    "source_latent_a": ("required", "LATENT", {}),
    "ref_position_a": ("required", {POSITIONS[0]: HOLD_INPUTS, POSITIONS[1]: {}, POSITIONS[2]: {}}, {}),
    "source_latent_b": ("optional", "LATENT", {}),
    "ref_boost_a": ("optional", "FLOAT", FLOAT_BOOST),
    "ref_boost_b": ("optional", "FLOAT", FLOAT_BOOST),
    "ref_boost_mask_a": ("optional", "MASK", {}),
    "ref_boost_mask_b": ("optional", "MASK", {}),
    "ref_position_b": ("optional", POSITIONS, {"default": POSITIONS[0]}),
}
ENCODE_INPUTS = {
    "clip": ("required", "CLIP", {}),
    "prompt": ("required", "STRING", {"multiline": True, "default": ""}),
    "image_a": ("optional", "IMAGE", {}),
    "image_b": ("optional", "IMAGE", {}),
    "grounding_px": ("optional", "INT", {"default": 768, "min": 0, "max": 4096, "step": 64}),
    "system_prompt": ("optional", "STRING", {"multiline": True, "default": ""}),
}

# krea2_edit_forward on a 16x16 target latent with the fake DiT: output shape, sequence shape, RoPE id sum / shape, attention bias (shape, sum, min, max) or None
SNAPSHOT = {
    'single': {'out': (1, 4, 16, 16), 'seq': (1, 131, 8), 'pos': 960.0, 'pos_shape': (1, 131, 3), 'bias': None},
    'pair': {'out': (1, 4, 16, 16), 'seq': (1, 195, 8), 'pos': 1536.0, 'pos_shape': (1, 195, 3), 'bias': None},
    'single_boost_masked': {'out': (1, 4, 16, 16), 'seq': (1, 131, 8), 'pos': 960.0, 'pos_shape': (1, 131, 3), 'bias': ((1, 1, 131, 131), 1419.565, 0.0, 0.693)},
    'pair_boosts': {'out': (1, 4, 16, 16), 'seq': (1, 195, 8), 'pos': 1536.0, 'pos_shape': (1, 195, 3), 'bias': ((1, 1, 195, 195), 5919.481, 0.0, 1.099)},
    'pair_beside_bag': {'out': (1, 4, 16, 16), 'seq': (1, 195, 8), 'pos': 2048.0, 'pos_shape': (1, 195, 3), 'bias': None},
    'source_larger_than_target': {'out': (1, 4, 16, 16), 'seq': (1, 131, 8), 'pos': 960.0, 'pos_shape': (1, 131, 3), 'bias': ((1, 1, 131, 131), 2839.131, 0.0, 0.693)},
    'batch_of_two': {'out': (2, 4, 16, 16), 'seq': (2, 131, 8), 'pos': 1920.0, 'pos_shape': (2, 131, 3), 'bias': None},
    'five_d_target': {'out': (1, 4, 1, 16, 16), 'seq': (1, 131, 8), 'pos': 960.0, 'pos_shape': (1, 131, 3), 'bias': None},
}


def _group(inputs):
    """{name: (group, type, options without the tooltip)}; a combo is its option list, a dynamic combo {option key: its inputs}."""
    out = {}
    for group in ("required", "optional"):
        for name, spec in inputs.get(group, {}).items():
            typ, opts = spec[0], (dict(spec[1]) if len(spec) > 1 else {})
            opts.pop("tooltip", None)
            if typ == "COMBO":                      # V3 reports a combo as COMBO plus its options; V1 had the option list as the type
                typ = list(opts.pop("options"))
                opts.pop("multiselect")
            elif typ == "COMFY_DYNAMICCOMBO_V3":
                typ = {o["key"]: _group(o["inputs"]) for o in opts.pop("options")}
            out[name] = (group, typ, opts)
    return out


def _contract(cls):
    return _group(cls.INPUT_TYPES())


def _snapshot(mod, srcs, x=None, **kw):
    torch.manual_seed(0)
    m = _FakeDiT()
    x = torch.zeros(1, 4, 16, 16) if x is None else x
    with contextlib.redirect_stdout(io.StringIO()):
        out = mod.krea2_edit_forward(m, x, torch.full((x.shape[0],), 0.5), torch.zeros(x.shape[0], TXT, 8), srcs, {}, **kw)
    shape, bias = m.block_inputs[0]
    return {"out": tuple(out.shape), "seq": tuple(shape), "pos": round(float(m.pos.sum()), 3), "pos_shape": tuple(m.pos.shape),
            "bias": None if bias is None else (tuple(bias.shape), round(float(bias.sum()), 3), round(float(bias.min()), 3), round(float(bias.max()), 3))}


def _cases():
    top = _top_half_mask()
    return {
        "single": ((_src(1),), {}),
        "pair": (([_src(1), _src(2)],), {}),
        "single_boost_masked": ((_src(1),), dict(ref_boost_a=2.0, ref_boost_mask_a=top)),
        "pair_boosts": (([_src(1), _src(2)],), dict(ref_boost_a=2.0, ref_boost_b=3.0, ref_boost_mask_a=top)),
        "pair_beside_bag": (([_src(1), _src(2)],), dict(ref_position_a=POSITIONS[1], ref_position_b=POSITIONS[2])),
        "source_larger_than_target": ((torch.full((1, 4, 24, 32), 1.0),), dict(ref_boost_a=2.0, ref_boost_mask_a=torch.ones(48, 64))),
        "batch_of_two": ((_src(1),), dict(x=torch.zeros(2, 4, 16, 16))),
        "five_d_target": ((_src(1),), dict(x=torch.zeros(1, 4, 1, 16, 16))),
    }


def _snapshots(mod):
    return {name: _snapshot(mod, *args, **kw) for name, (args, kw) in _cases().items()}


# ---------------------------------------------------------------- the node interface

def test_patch_node_inputs_are_frozen():
    mod = _load_pack()
    assert _contract(mod.Krea2EditAdvanced) == PATCH_INPUTS
    assert list(_contract(mod.Krea2EditAdvanced)) == list(PATCH_INPUTS)           # order = socket order = widget order
    schema = mod.Krea2EditAdvanced.GET_SCHEMA()
    assert (schema.node_id, schema.display_name, schema.category) == ("Krea2EditAdvanced", "Krea2 Edit Advanced (source patch)", "krea2edit")
    assert list(mod.Krea2EditAdvanced.RETURN_TYPES) == ["MODEL", "MASK"]


def test_encode_node_inputs_are_frozen():
    mod = _load_pack()
    assert _contract(mod.Krea2EditAdvancedGroundedEncode) == ENCODE_INPUTS
    assert list(_contract(mod.Krea2EditAdvancedGroundedEncode)) == list(ENCODE_INPUTS)
    schema = mod.Krea2EditAdvancedGroundedEncode.GET_SCHEMA()
    assert (schema.node_id, schema.display_name, schema.category) == ("Krea2EditAdvancedGroundedEncode", "Krea2 Edit Advanced (grounded encode)", "krea2edit")
    assert list(mod.Krea2EditAdvancedGroundedEncode.RETURN_TYPES) == ["CONDITIONING"]


def test_the_extension_lists_both_nodes():
    mod = _load_pack()
    nodes = asyncio.run(asyncio.run(mod.comfy_entrypoint()).get_node_list())
    assert nodes == [mod.Krea2EditAdvanced, mod.Krea2EditAdvancedGroundedEncode]


def _tooltips(inputs):
    for group in ("required", "optional"):
        for name, spec in inputs.get(group, {}).items():
            if name not in ("model", "clip"):
                assert len(spec) > 1 and spec[1].get("tooltip"), f"{name} has no tooltip"
            if spec[0] == "COMFY_DYNAMICCOMBO_V3":
                for option in spec[1]["options"]:
                    _tooltips(option["inputs"])


def test_every_input_has_a_tooltip():
    mod = _load_pack()
    for cls in (mod.Krea2EditAdvanced, mod.Krea2EditAdvancedGroundedEncode):
        _tooltips(cls.INPUT_TYPES())


# ---------------------------------------------------------------- what the forward builds

def test_forward_snapshot_matches_the_frozen_table():
    mod = _load_pack()
    assert _snapshots(mod) == SNAPSHOT


def test_patchify_orders_tokens_by_position_and_channels_by_pixel():
    mod = _load_pack()
    x = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
    tokens = mod._patchify(x, 2)
    assert tokens.shape == (2, 2 * 3, 3 * 2 * 2)
    assert torch.equal(tokens[1, 4], x[1, :, 2:4, 2:4].flatten())            # token (row 1, column 1), channels then pixel rows then columns


def test_batch_and_temporal_targets_keep_their_shape():
    mod = _load_pack()
    snap = _snapshots(mod)
    assert snap["batch_of_two"]["out"] == (2, 4, 16, 16)
    assert snap["five_d_target"]["out"] == (1, 4, 1, 16, 16)


if __name__ == "__main__":
    if "--print" in sys.argv:
        for k, v in _snapshots(_load_pack()).items():
            print(f"    {k!r}: {v!r},")
        sys.exit(0)
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
