"""The example workflows in workflows/ (ComfyUI API format) against the node schemas: the inputs exist, the required ones are
present, the links resolve, and the chain passes the latent from one edit to the next.

Run standalone:  python tests/test_example_workflows.py
Or via pytest:   pytest tests/test_example_workflows.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stubs import _load_pack   # noqa: E402
from test_node_contract import _contract   # noqa: E402

WORKFLOWS = os.path.join(os.path.dirname(_HERE), "workflows")
SINGLE, CHAIN = "krea2_edit_advanced_api.json", "krea2_edit_advanced_chain_api.json"


def _load(name):
    with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as f:
        return json.load(f)


def _links(graph):
    for node_id, node in graph.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                yield node_id, name, value[0]


def _check_pack_nodes(graph):
    mod = _load_pack()
    classes = {"Krea2EditAdvanced": mod.Krea2EditAdvanced, "Krea2EditAdvancedGroundedEncode": mod.Krea2EditAdvancedGroundedEncode}
    for node_id, node in graph.items():
        if node["class_type"] not in classes:
            continue
        spec = _contract(classes[node["class_type"]])
        given = node["inputs"]
        for name in given:
            assert name.split(".")[0] in spec, f"{node_id}: {name} is not an input of {node['class_type']}"
        for name, (group, typ, _) in spec.items():
            if group == "required":
                assert name in given, f"{node_id}: required input {name} is missing"
            if isinstance(typ, dict):                         # a dynamic combo: the nested inputs of the chosen option
                nested = typ[given[name]]
                for sub, (sub_group, _, _) in nested.items():
                    assert (f"{name}.{sub}" in given) or sub_group == "optional", f"{node_id}: {name}.{sub} is missing"
                assert {k for k in given if k.startswith(name + ".")} <= {f"{name}.{sub}" for sub in nested}


def test_the_workflows_only_link_to_existing_nodes():
    for name in (SINGLE, CHAIN):
        graph = _load(name)
        assert all(target in graph for _, _, target in _links(graph)), name


def test_the_workflows_use_the_node_inputs_as_defined():
    for name in (SINGLE, CHAIN):
        _check_pack_nodes(_load(name))


def test_the_single_edit_workflow_has_one_edit():
    graph = _load(SINGLE)
    kinds = [n["class_type"] for n in graph.values()]
    assert kinds.count("Krea2EditAdvanced") == 1 and kinds.count("KSampler") == 1 and kinds.count("SaveImage") == 1


def test_the_chain_passes_the_latent_from_one_edit_to_the_next():
    graph = _load(CHAIN)
    patches = [(i, n) for i, n in graph.items() if n["class_type"] == "Krea2EditAdvanced"]
    assert len(patches) == 6
    sources = [graph[n["inputs"]["source_latent_a"][0]]["class_type"] for _, n in patches]
    assert sources == ["VAEEncode"] + ["KSampler"] * 5
    assert sum(n["class_type"] == "VAEEncode" for n in graph.values()) == 1          # the source is encoded once, at the start
    for (previous, _), (_, n) in zip(patches, patches[1:]):                          # each edit starts from the previous edit's sampler output
        assert graph[n["inputs"]["source_latent_a"][0]]["inputs"]["model"][0] == previous


def test_the_decoded_pictures_only_feed_the_text_encoder_and_the_save():
    graph = _load(CHAIN)
    decoded = {i for i, n in graph.items() if n["class_type"] == "VAEDecode"}
    for node_id, _, target in _links(graph):
        if target in decoded:
            assert graph[node_id]["class_type"] in ("SaveImage", "Krea2EditAdvancedGroundedEncode"), graph[node_id]["class_type"]


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
