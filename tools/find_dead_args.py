"""Find parameters that are never read inside their own function body.

Mechanical detection only -- every hit needs triage, because several patterns
are legitimately "unused": abstract stubs, signature-compatible overrides,
callbacks with a fixed protocol, and params consumed by `locals()`.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "truecell"

SKIP_NAMES = {"self", "cls"}


class Scope:
    """Names loaded anywhere in a function body, including nested scopes."""

    def __init__(self, node):
        self.loaded: set[str] = set()
        self.uses_locals = False
        self.has_star_kwargs = bool(node.args.kwarg)
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                self.loaded.add(child.id)
            elif isinstance(child, ast.Call):
                f = child.func
                if isinstance(f, ast.Name) and f.id in ("locals", "vars", "eval", "exec"):
                    self.uses_locals = True
            # A nonlocal/global reference counts as a use.
            elif isinstance(child, (ast.Nonlocal, ast.Global)):
                self.loaded.update(child.names)


def params_of(node):
    a = node.args
    out = []
    for group in (a.posonlyargs, a.args, a.kwonlyargs):
        out.extend((p.arg, p) for p in group)
    if a.vararg:
        out.append((a.vararg.arg, a.vararg))
    if a.kwarg:
        out.append((a.kwarg.arg, a.kwarg))
    return out


def body_is_stub(node) -> bool:
    """`...`, `pass`, a bare docstring, or `raise NotImplementedError`."""
    body = [s for s in node.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                    and isinstance(s.value.value, str))]
    if not body:
        return True
    if len(body) == 1:
        s = body[0]
        if isinstance(s, ast.Pass):
            return True
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis:
            return True
        if isinstance(s, ast.Raise):
            return True
    return False


def decorator_names(node):
    out = []
    for d in node.decorator_list:
        if isinstance(d, ast.Name):
            out.append(d.id)
        elif isinstance(d, ast.Attribute):
            out.append(d.attr)
        elif isinstance(d, ast.Call):
            f = d.func
            out.append(f.id if isinstance(f, ast.Name) else getattr(f, "attr", "?"))
    return out


findings = []
for path in sorted(ROOT.rglob("*.py")):
    if "__pycache__" in str(path):
        continue
    tree = ast.parse(path.read_text(), filename=str(path))
    # Map each function to its enclosing class, for triage.
    parent_class = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    parent_class[child] = node.name

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if body_is_stub(node):
            continue
        scope = Scope(node)
        if scope.uses_locals:
            continue
        for name, arg in params_of(node):
            if name in SKIP_NAMES or name.startswith("_"):
                continue
            if name in scope.loaded:
                continue
            findings.append({
                "file": str(path.relative_to(ROOT.parent)),
                "line": node.lineno,
                "func": node.name,
                "cls": parent_class.get(node, ""),
                "param": name,
                "public_fn": not node.name.startswith("_"),
                "decorators": decorator_names(node),
                "has_kwargs": scope.has_star_kwargs,
            })

print(f"{len(findings)} unused parameters\n")
pub = [f for f in findings if f["public_fn"] and not f["cls"]]
meth = [f for f in findings if f["public_fn"] and f["cls"]]
priv = [f for f in findings if not f["public_fn"]]

for title, group in (("PUBLIC MODULE-LEVEL FUNCTIONS", pub),
                     ("PUBLIC METHODS", meth),
                     ("PRIVATE / HELPERS", priv)):
    print(f"=== {title} ({len(group)}) ===")
    for f in group:
        where = f"{f['cls']}.{f['func']}" if f["cls"] else f["func"]
        extra = []
        if f["has_kwargs"]:
            extra.append("**kwargs present")
        if f["decorators"]:
            extra.append("@" + ",@".join(f["decorators"]))
        print(f"  {f['file']}:{f['line']:>5}  {where}({f['param']})"
              + (f"   [{'; '.join(extra)}]" if extra else ""))
    print()
