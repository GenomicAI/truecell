"""Every tutorial that imports `tutorials.*` puts the checkout root on `sys.path` first.

`python tutorials/<name>.py` puts `tutorials/` on `sys.path`, not the checkout root, so
`from tutorials.bands import ...` resolves only where an editable install's `.pth` file
happens to add the root. Against an installed wheel, which is a reviewer's situation, it
raises `ModuleNotFoundError`, and `pbmc3k_de_tutorial.py` did exactly that with truecell
1.2.0. This suite runs from an editable install, where the import works either way, so
the scripts are read rather than run. CI's `wheel-tutorials` job runs two of them against
the built wheel.
"""
import ast
from pathlib import Path

import pytest

TUTORIALS = Path(__file__).resolve().parent.parent / "tutorials"


def _imports_tutorials(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.level == 0 and (node.module or "").split(".")[0] == "tutorials"
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == "tutorials" for alias in node.names)
    return False


def _extends_sys_path(node: ast.AST) -> bool:
    """`sys.path.insert(...)` or `sys.path.append(...)`."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("insert", "append")
            and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "path"
            and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "sys")


def _module_level(tree: ast.Module):
    """The nodes that run on import: everything outside function and class bodies."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _importers() -> dict:
    found = {}
    for path in sorted(TUTORIALS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = [node.lineno for node in ast.walk(tree) if _imports_tutorials(node)]
        if lines:
            found[path.name] = (tree, min(lines))
    return found


IMPORTERS = _importers()


def test_the_scan_finds_the_tutorials_that_import_each_other():
    """Guard the guard: a scan that matched nothing would pass every case below."""
    assert {"pbmc3k_de_tutorial.py", "generate_de_plots.py",
            "pbmc8k_subclustering_tutorial.py"} <= set(IMPORTERS)


@pytest.mark.parametrize("name", sorted(IMPORTERS))
def test_the_checkout_root_is_on_sys_path_before_tutorials_is_imported(name):
    tree, first_import = IMPORTERS[name]
    bootstraps = [node.lineno for node in _module_level(tree) if _extends_sys_path(node)]
    assert bootstraps and min(bootstraps) < first_import, (
        f"tutorials/{name} imports `tutorials` at line {first_import} without first putting "
        "the checkout root on sys.path at module level, so it fails against an installed "
        "wheel. Add the bootstrap the other tutorials use:\n"
        "    _ROOT = Path(__file__).parent.parent\n"
        "    if str(_ROOT) not in sys.path:\n"
        "        sys.path.insert(0, str(_ROOT))"
    )
