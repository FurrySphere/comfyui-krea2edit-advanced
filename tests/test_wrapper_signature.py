"""Regression test: the DIFFUSION_MODEL wrapper must accept the exact positional
call shape that comfy/ldm/krea2/model.py uses.

Original test contributed by @akashzeno (GitHub PR #9, thank you). The second test
here is adapted to our fix strategy: rather than mirroring forward's signature
name-for-name, our wrapper absorbs trailing positionals with ``*wargs`` and picks
``transformer_options`` as the trailing dict, so it tolerates future argument
insertions too. So test 2 checks the *behavior* (the right dict is picked up)
instead of the signature shape.

Upstream commit c9602625 inserted a new ``ref_latents`` positional argument between
``attention_mask`` and ``transformer_options`` in SingleStreamDiT.forward:

    .execute(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

WrapperExecutor forwards these positionally as ``wrapper(self, x, timesteps, context,
attention_mask, ref_latents, transformer_options)`` -> 7 positional args. A wrapper that
predates the change only has 6 slots and dies with:

    TypeError: wrapper() takes from 4 to 6 positional arguments but 7 were given

These tests reproduce that call path with a stubbed forward so they stay fast and
model-free, and guard against the signature drifting out of sync again.

Run standalone:  python tests/test_wrapper_signature.py
Or via pytest:   pytest tests/test_wrapper_signature.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stubs import _FakeModelPatcher, _load_pack, _node   # noqa: E402


def _get_wrapper(mod):
    import comfy.patcher_extension as pe

    m, _ = _node(mod, _FakeModelPatcher(), {"samples": torch.zeros(1, 4, 8, 8)})
    to = m.model_options["transformer_options"]
    return to["wrappers"][pe.WrappersMP.DIFFUSION_MODEL]["krea2_edit_advanced"][0]


def test_wrapper_accepts_upstream_positional_call():
    """The crash path: new-style 6-positional call (with ref_latents) must not raise."""
    import comfy.patcher_extension as pe

    mod = _load_pack()
    wrapper = _get_wrapper(mod)

    sentinel = object()
    mod.krea2_edit_forward = lambda *a, **k: sentinel

    executor = pe.WrapperExecutor(
        original=lambda *a, **k: None,
        class_obj=object(),   # stands in for the SingleStreamDiT instance
        wrappers=[wrapper],
        idx=0,
    )

    result = executor.execute(
        torch.zeros(1, 4, 8, 8),   # x
        None,   # timesteps
        None,   # context
        None,   # attention_mask
        None,   # ref_latents  <-- the argument added by c9602625
        {},     # transformer_options
    )
    assert result is sentinel


def test_transformer_options_picked_up_under_both_signatures():
    """Behavioral guard: transformer_options must bind to the trailing dict, never to
    ref_latents, under both the old (5-arg) and new (6-arg) positional shapes."""
    import comfy.patcher_extension as pe

    mod = _load_pack()

    captured = {}
    mod.krea2_edit_forward = lambda dm, x, t, ctx, src, to, **k: captured.__setitem__("to", to)

    marker_old = {"tag": "old"}
    marker_new = {"tag": "new"}

    for shape, marker in (("old", marker_old), ("new", marker_new)):
        wrapper = _get_wrapper(mod)
        executor = pe.WrapperExecutor(
            original=lambda *a, **k: None, class_obj=object(),
            wrappers=[wrapper], idx=0,
        )
        if shape == "old":
            executor.execute(torch.zeros(1, 4, 8, 8), None, None, None, marker)              # ...attention_mask, t_opts
        else:
            executor.execute(torch.zeros(1, 4, 8, 8), None, None, None, ["ref"], marker)     # ...ref_latents, t_opts
        assert captured["to"] is marker, f"{shape}: got {captured['to']}, expected the trailing dict"


def test_a_source_of_another_size_than_the_sampled_latent_warns_once():
    import contextlib
    import io

    import comfy.patcher_extension as pe

    mod = _load_pack()
    mod.krea2_edit_forward = lambda *a, **k: None
    for sampled, warns in ((8, 0), (6, 1)):
        wrapper = _get_wrapper(mod)
        executor = pe.WrapperExecutor(original=lambda *a, **k: None, class_obj=object(), wrappers=[wrapper], idx=0)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            for _ in range(3):
                executor.execute(torch.zeros(1, 4, sampled, sampled), None, None, None, {})
        assert log.getvalue().count("WARNING: source_latent_a is 64x64px but the picture is drawn at") == warns, sampled


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
