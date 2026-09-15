"""`skills/truecell/reference/api-map.md` shows each function's real signature.

Its first paragraph says so, and an agent writes calls from it. By the Frontiers
revision five entries had fallen behind the code (`run_tsne`, `glm_pca`,
`find_clusters`, `aggregate_expression` and the spatial loaders) and the header
still named 0.9.0. Nothing failed, because the skills guards in `test_docs.py`
ask whether a name is mentioned, not whether what is written about it is true.

Every call that starts a line of a ``python`` block is read as a signature and
compared with ``inspect.signature``: the same parameters, in the same order, with
the same defaults.
"""
import ast
import importlib
import inspect
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API_MAP = ROOT / "skills" / "truecell" / "reference" / "api-map.md"
#: Where a name in the map can live, tried in this order.
MODULES = ("truecell", "truecell.generics", "truecell.io", "truecell.datasets",
           "truecell.compat.anndata")
#: The map's convention: a first parameter called `seurat` or `obj` is the object.
OBJECT_NAMES = {"seurat", "obj", "object", "object_"}


def _calls(text: str):
    """Each ``name(...)`` starting a line of a python block, with its parameter text."""
    for block in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
        for match in re.finditer(r"(?m)^([A-Za-z_]\w*)\(", block):
            depth, end = 0, match.end() - 1
            for end in range(match.end() - 1, len(block)):
                if block[end] == "(":
                    depth += 1
                elif block[end] == ")":
                    depth -= 1
                    if depth == 0:
                        break
            yield match.group(1), block[match.end():end]


def _value(node: ast.expr):
    """A default as the map writes it: a literal, ``inf``, a tuple, or ``1/15``."""
    if isinstance(node, ast.Name) and node.id == "inf":
        return math.inf
    if isinstance(node, ast.Tuple):
        return tuple(_value(item) for item in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _value(node.left) / _value(node.right)
    return ast.literal_eval(node)


def _target(name: str):
    for module in MODULES:
        try:
            found = getattr(importlib.import_module(module), name, None)
        except ImportError:
            continue
        if found is not None:
            return found
    return None


SIGNATURES = list(_calls(API_MAP.read_text()))


def test_the_map_is_read_at_all():
    names = [name for name, _ in SIGNATURES]
    assert len(names) > 60, names
    assert {"find_markers", "load_cosmx", "read_10x", "from_anndata", "show_versions"} <= set(names)


@pytest.mark.parametrize("name,written", SIGNATURES, ids=[name for name, _ in SIGNATURES])
def test_each_signature_in_the_map_is_the_code_s(name, written):
    target = _target(name)
    assert target is not None, f"{name} is in the API map but in none of {MODULES}"

    args = ast.parse(f"def _({written}): pass").body[0].args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    written_params = list(zip((a.arg for a in positional), defaults))
    if args.vararg:
        written_params.append(("*" + args.vararg.arg, None))
    written_params += list(zip((a.arg for a in args.kwonlyargs), args.kw_defaults))
    if args.kwarg:
        written_params.append(("**" + args.kwarg.arg, None))

    code_params = [({inspect.Parameter.VAR_POSITIONAL: "*",
                     inspect.Parameter.VAR_KEYWORD: "**"}.get(p.kind, "") + p.name, p.default)
                   for p in inspect.signature(target).parameters.values()]

    written_names = [param for param, _ in written_params]
    code_names = [param for param, _ in code_params]
    if written_names and written_names[0] in OBJECT_NAMES and code_names[0] in OBJECT_NAMES:
        written_names[0] = code_names[0]
    assert written_names == code_names, (
        f"{name}: the map lists {written_names}, the code has {code_names}")

    for (param, written_default), (_, default) in zip(written_params, code_params):
        if written_default is None:
            assert default is inspect.Parameter.empty, (
                f"{name}({param}): the code defaults it to {default!r}, the map gives none")
        else:
            assert default is not inspect.Parameter.empty, (
                f"{name}({param}): the map gives a default the code does not have")
            assert _value(written_default) == default, (
                f"{name}({param}): the map says {ast.unparse(written_default)}, "
                f"the code {default!r}")
