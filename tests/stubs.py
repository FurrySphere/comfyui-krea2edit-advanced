"""Helpers shared by the tests: loads the pack's __init__.py (the folder name has a hyphen) from a ComfyUI checkout and stands in for the
few objects the node touches.

The tests import comfy and comfy_api, so they run with the pack inside a ComfyUI checkout (custom_nodes/comfyui-krea2edit).
"""
import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_COMFY_ROOT = os.path.dirname(os.path.dirname(_PACK))   # .../ComfyUI

if _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)


def _load_pack():
    spec = importlib.util.spec_from_file_location("comfyui_krea2edit_test", os.path.join(_PACK, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeInner:
    def process_latent_in(self, x):
        return x


class _FakeModelPatcher:
    """Minimal stand-in for ModelPatcher: only what the node touches."""

    def __init__(self):
        self.model = _FakeInner()
        self.model_options = {}

    def clone(self):
        c = _FakeModelPatcher()
        c.model = self.model      # share the inner model, like the real clone
        return c


def _latent(h=64, w=64):
    return {"samples": torch.zeros(1, 4, h, w)}


def _run_steps(wrapper, n=3, h=64, w=64):
    """Drive the wrapper the way a sampler would: same latent shape, n times."""
    import comfy.patcher_extension as pe

    for _ in range(n):
        executor = pe.WrapperExecutor(
            original=lambda *a, **k: None, class_obj=object(), wrappers=[wrapper], idx=0,
        )
        executor.execute(torch.zeros(1, 4, h, w), None, None, None, {})


def _node(mod, model, source_latent_a, ref_position_a="aligned (trained)", **kw):
    """Run the patch node with a plain position string; the hold stays off unless a dict (the dynamic combo value) is given."""
    if isinstance(ref_position_a, str):
        ref_position_a = {"ref_position_a": ref_position_a, **({"source_hold": False} if ref_position_a.startswith("aligned") else {})}
    return mod.Krea2EditAdvanced.execute(model, source_latent_a, ref_position_a, **kw)
