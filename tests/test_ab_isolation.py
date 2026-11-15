"""No cross-talk between the `_a` and `_b` references.

Complements test_ref_controls.py: instead of asserting expected values, these tests perturb
only one side and require the other side to be byte-identical, then check the routing
from node inputs to the forward pass.

Run standalone:  python tests/test_ab_isolation.py
Or via pytest:   pytest tests/test_ab_isolation.py
"""
import inspect
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stubs import _FakeModelPatcher, _latent, _load_pack, _run_steps   # noqa: E402
from test_ref_controls import (BESIDE, TOK, TXT, _FakeCLIP, _capture, _forward, _img,   # noqa: E402
                               _src, _val, _wrapper)

BAG = "bag of features"


def _half_mask():
    m = torch.zeros(1, 16, 16)
    m[:, :8] = 1.0
    return m


def _slots(m, bias):
    """Per-reference view of everything a control can change: RoPE ids and the target->ref bias."""
    out = []
    for i in range(2):
        s = TXT + i * TOK
        rows = bias[0, 0, TXT + 2 * TOK:, s:s + TOK] if bias is not None else None
        out.append((m.pos[0, s:s + TOK].clone(), None if rows is None else rows.clone()))
    return out


def _bias_equal(x, y):
    """A missing bias and an all-zero bias are the same thing."""
    if x is None or y is None:
        other = y if x is None else x
        return other is None or bool((other == 0).all())
    return torch.equal(x, y)


def _same(x, y):
    return torch.equal(x[0], y[0]) and _bias_equal(x[1], y[1])


# One non-default setting per control, as (side-independent name, kwargs builder).
SETTINGS = {
    "boost": lambda side: {f"ref_boost_{side}": 3.0},
    "position_beside": lambda side: {f"ref_position_{side}": BESIDE},
    "position_bag": lambda side: {f"ref_position_{side}": BAG},
    "mask": lambda side: {f"ref_boost_{side}": 3.0, f"ref_boost_mask_{side}": _half_mask()},
}


def _snapshot(mod, **kw):
    m, _, bias = _forward(mod, [_src(1), _src(2)], **kw)
    return _slots(m, bias)


def test_changing_only_a_never_touches_b_and_vice_versa():
    mod = _load_pack()
    base = _snapshot(mod)
    for name, make in SETTINGS.items():
        changed_a = _snapshot(mod, **make("a"))
        changed_b = _snapshot(mod, **make("b"))
        assert _same(changed_a[1], base[1]), f"{name}: an _a setting changed slot b"
        assert _same(changed_b[0], base[0]), f"{name}: a _b setting changed slot a"
        assert not _same(changed_a[0], base[0]), f"{name}: _a setting had no effect on slot a (vacuous)"
        assert not _same(changed_b[1], base[1]), f"{name}: _b setting had no effect on slot b (vacuous)"


def test_a_and_b_settings_compose_independently():
    """Setting both sides equals the per-slot results of setting each side alone."""
    mod = _load_pack()
    for name_a, make_a in SETTINGS.items():
        for name_b, make_b in SETTINGS.items():
            both = _snapshot(mod, **make_a("a"), **make_b("b"))
            only_a = _snapshot(mod, **make_a("a"))
            only_b = _snapshot(mod, **make_b("b"))
            # RoPE stacking: two `beside` refs shift each other, so compare ids only when that cannot apply
            stacking = name_a == "position_beside" and name_b == "position_beside"
            if not stacking:
                assert torch.equal(both[0][0], only_a[0][0]), (name_a, name_b)
                assert torch.equal(both[1][0], only_b[1][0]), (name_a, name_b)
            for slot, alone in ((0, only_a), (1, only_b)):
                assert _bias_equal(both[slot][1], alone[slot][1]), (name_a, name_b, slot)


def test_values_are_routed_by_suffix_not_by_position_in_the_call():
    """Same two settings, kwargs given in opposite orders, land on the same slots."""
    mod = _load_pack()
    one = _snapshot(mod, ref_boost_a=2.0, ref_boost_b=5.0, ref_position_b=BESIDE)
    two = _snapshot(mod, ref_position_b=BESIDE, ref_boost_b=5.0, ref_boost_a=2.0)
    assert _same(one[0], two[0]) and _same(one[1], two[1])


def test_distinct_sentinels_reach_the_forward_unmixed():
    mod = _load_pack()
    seen = _capture(mod)
    ma, mb = torch.ones(1, 16, 16), torch.zeros(1, 16, 16)
    w = _wrapper(mod, _src(1), source_latent_b={"samples": _src(2)},
                 ref_boost_a=1.5, ref_boost_b=2.5, ref_boost_mask_a=ma, ref_boost_mask_b=mb,
                 ref_position_a=BESIDE, ref_position_b=BAG)
    _run_steps(w, n=1)
    src, k = seen[-1]
    assert [_val(s) for s in src] == [1.0, 2.0]
    assert (k["ref_boost_a"], k["ref_boost_b"]) == (1.5, 2.5)
    assert k["ref_boost_mask_a"] is ma and k["ref_boost_mask_b"] is mb
    assert (k["ref_position_a"], k["ref_position_b"]) == (BESIDE, BAG)


def test_forward_and_patch_signatures_carry_matching_pairs():
    mod = _load_pack()
    for fn in (mod.krea2_edit_forward, mod.Krea2EditAdvanced.execute):
        names = set(inspect.signature(fn).parameters)
        for n in names:
            if n.endswith("_b"):
                assert n[:-1] + "a" in names, f"{fn.__name__}: {n} has no _a counterpart"
            if n.endswith("_a"):
                assert n[:-1] + "b" in names or n.startswith("source_"), f"{fn.__name__}: {n} has no _b counterpart"


# ---------------------------------------------------------------- optional inputs

def test_every_b_input_is_optional_and_unneeded():
    mod = _load_pack()
    for node in (mod.Krea2EditAdvanced, mod.Krea2EditAdvancedGroundedEncode):
        it = node.INPUT_TYPES()
        b_names = [n for sect in it.values() for n in sect if n.endswith("_b")]
        assert b_names
        assert not [n for n in it["required"] if n.endswith("_b")]
    # and the nodes really run with none of them connected
    seen = _capture(mod)
    _run_steps(_wrapper(mod, _src(1)), n=1)
    assert torch.is_tensor(seen[-1][0])
    tokens = mod.Krea2EditAdvancedGroundedEncode.execute(_FakeCLIP(), "p", image_a=_img(0.25, 64))[0]
    assert len(tokens["images"]) == 1


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
