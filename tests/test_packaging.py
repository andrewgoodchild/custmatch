"""The package declares what it imports: a clean `pip install .` once gave a `custmatch` command that
crashed on start (networkx, duckdb and jellyfish were used but not declared), and the suite passed
only because the development environment happened to have them."""
import ast
import sys
from pathlib import Path

try:
    import tomllib                                  # Python 3.11+
except ModuleNotFoundError:                         # 3.10: the same parser, as a package
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
IMPORT_TO_DIST = {"sklearn": "scikit-learn", "yaml": "pyyaml"}


def _declared():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    name = lambda spec: spec.split(";")[0].split("[")[0].split("<")[0].split(">")[0].split("=")[0].split("~")[0].strip().lower()  # noqa: E731
    core = {name(d) for d in meta["dependencies"]}
    extras = {name(d) for group in meta.get("optional-dependencies", {}).values() for d in group}
    return core, extras


def _imports():
    """Top-level third-party modules imported by custmatch: (module, imported at module level)."""
    found = {}
    for path in (ROOT / "custmatch").glob("*.py"):
        tree = ast.parse(path.read_text())
        top = {id(n) for n in tree.body}
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for n in names:
                mod = n.split(".")[0]
                if mod in sys.stdlib_module_names or mod == "custmatch":
                    continue
                found[mod] = found.get(mod, False) or id(node) in top
    return found


def test_every_import_is_declared():
    core, extras = _declared()
    missing = sorted(m for m in _imports() if IMPORT_TO_DIST.get(m, m).lower() not in core | extras)
    assert not missing, f"imported by custmatch but not in pyproject.toml: {missing}"


def test_module_level_imports_are_core_dependencies():
    """A module-level import of an optional package breaks `import custmatch` without that extra."""
    core, _ = _declared()
    top = sorted(m for m, at_top in _imports().items() if at_top and IMPORT_TO_DIST.get(m, m).lower() not in core)
    assert not top, f"imported at module level but only optional: {top}"
