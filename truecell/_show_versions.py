"""``show_versions()``: what a report about a crash needs.

A segmentation fault in compiled code ends the process before Python can say
where it was, and the traceback ``faulthandler`` rescues names a Python line,
not a cause. For numba-parallel code, which ``run_umap`` reaches through
umap-learn and pynndescent, the cause is usually in the environment: which
threading layer numba picked, and how many OpenMP runtimes were loaded. Two
copies of LLVM's OpenMP runtime in one process is the crash scanpy's users hit
on Apple Silicon (scverse/scanpy#3507, #4026). This module reports both, with
the versions and the platform details, Rosetta 2 and free-threading, that decide
which wheels can install at all.
"""
from __future__ import annotations

import errno
import faulthandler
import json
import os
import platform
import signal
import subprocess
import sys
import sysconfig
import textwrap
import time
from importlib import metadata
from pathlib import Path
from typing import Any

# Distribution names in the order a report reads: the core, the numba path
# `run_umap` takes, then the other extras. `torch` is not a dependency. It is
# listed because it ships its own libomp, the second runtime in scanpy#4026.
_DEPENDENCIES = (
    "numpy", "scipy", "pandas", "packaging",
    "numba", "llvmlite", "umap-learn", "pynndescent", "scikit-learn", "threadpoolctl",
    "igraph", "leidenalg", "statsmodels", "matplotlib", "seaborn", "scikit-misc",
    "anndata", "harmonypy", "pydeseq2",
    "torch",
)
# The `analysis` extra requires `python-igraph`, a stub that installs `igraph`.
_ALIASES = {"igraph": ("python-igraph",)}

# Printed only when set: the thread settings of numba and the BLAS libraries,
# the library paths that decide which runtime a `dlopen` finds, and
# KMP_DUPLICATE_LIB_OK. scikit-learn and threadpoolctl both set that one to True
# when they are imported, unless it is already set, to let more than one OpenMP
# runtime load. It is on in nearly every truecell session, so it is reported and
# not warned about.
_ENVIRONMENT = (
    "NUMBA_THREADING_LAYER", "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT", "NUMBA_CACHE_DIR",
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "KMP_DUPLICATE_LIB_OK", "PYTHONFAULTHANDLER",
    "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD",
    "CONDA_PREFIX", "VIRTUAL_ENV",
)

_LAYER_MODULES = {
    "tbb": "numba.np.ufunc.tbbpool",
    "omp": "numba.np.ufunc.omppool",
    "workqueue": "numba.np.ufunc.workqueue",
}

# Both probes run in a child interpreter, never in the caller's. Starting
# numba's threads fixes the threading layer for the life of a process, and
# importing a layer loads the runtime it links, so probing in place would change
# the session being reported on, and a crash would take that session down. The
# first probe imports scikit-learn before it runs a parallel function because
# truecell's pipeline loads them in that order: scikit-learn's OpenMP runtime is
# already in the process when `run_umap` starts numba's threads.
_FRESH_PROCESS = """
import json
try:
    import sklearn  # noqa: F401
except ImportError:
    pass
import numba
from numba.core import config

@numba.njit(parallel=True)
def parallel_sum(n):
    total = 0.0
    for i in numba.prange(n):
        total += i
    return total

parallel_sum(64)
out = {
    "layer": numba.threading_layer(),
    "configured": config.THREADING_LAYER,
    "priority": list(config.THREADING_LAYER_PRIORITY),
    "threads": numba.get_num_threads(),
}
try:
    from threadpoolctl import threadpool_info
except ImportError:
    out["threadpools"] = None
else:
    out["threadpools"] = threadpool_info()
print(json.dumps(out))
"""

# One child per layer, so a layer whose import crashes cannot hide the others.
_IMPORT_MODULE = """
import importlib, json, sys
try:
    importlib.import_module(sys.argv[1])
except Exception as exc:
    print(json.dumps(f"{type(exc).__name__}: {exc}"))
else:
    print(json.dumps(None))
"""


def show_versions(as_dict: bool = False) -> dict[str, Any] | None:
    """Print what a bug report about a crash needs.

    Run it in the environment that crashed, with
    ``python -c "import truecell; truecell.show_versions()"``, and paste the
    output into the issue beside the traceback from ``python -X faulthandler``.

    It reports where the running truecell lives and how it was installed; the
    interpreter, platform and processor, including whether Python runs under
    Rosetta 2 or is a free-threaded build; the version of each package in the
    scientific stack and the installer that put it there (``pip``, ``uv``,
    ``conda``); numba's threading layer; the OpenMP and BLAS runtimes
    ``threadpoolctl`` finds loaded; and the environment variables that change
    any of those. It ends with a warning for each known cause of a crash it
    sees, such as two copies of LLVM's OpenMP runtime in one process.

    numba is examined in separate Python processes: one imports scikit-learn
    and runs a numba parallel function, and one per threading layer tries to
    load that layer. Calling this therefore cannot choose a layer or load a
    runtime in your own session, and a probe that crashes is reported rather
    than raised. The probes take a few seconds.

    Parameters
    ----------
    as_dict : return the report instead of printing it

    Returns
    -------
    ``None`` once the report is printed, or the report itself with
    ``as_dict=True``.
    """
    report: dict[str, Any] = {
        "truecell": _truecell(),
        "system": _system(),
        "dependencies": {name: _distribution(name) for name in _DEPENDENCIES},
    }
    report["numba"], fresh_pools = _numba()
    report["threadpools"] = {"this_process": _threadpools(), "fresh_process": fresh_pools}
    report["environment"] = _environment()
    report["warnings"] = _warnings(report)
    if as_dict:
        return report
    print(_format(report))
    return None


def _distribution(name: str) -> dict[str, Any] | None:
    """Version, installer and editable flag of an installed distribution, or None."""
    for candidate in (name, *_ALIASES.get(name, ())):
        try:
            dist = metadata.distribution(candidate)
        except metadata.PackageNotFoundError:
            continue
        installer = (dist.read_text("INSTALLER") or "").strip() or None
        try:
            direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
        except json.JSONDecodeError:
            direct_url = {}
        editable = isinstance(direct_url, dict) and bool(
            direct_url.get("dir_info", {}).get("editable"))
        return {"version": dist.version, "installer": installer, "editable": editable}
    return None


def _truecell() -> dict[str, Any]:
    """The installed distribution, and the directory the running code came from.

    The two can disagree: a checkout on ``sys.path`` shadows an installed wheel.
    """
    return {"path": str(Path(__file__).resolve().parent), **(_distribution("truecell") or {})}


def _system() -> dict[str, Any]:
    if (Path(sys.prefix) / "conda-meta").is_dir():
        environment = "conda"
    elif sys.prefix != sys.base_prefix:
        environment = "venv"
    else:
        environment = "system"
    return {
        "python": " ".join(sys.version.split()),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu(),
        "cpu_count": os.cpu_count(),
        "rosetta": _rosetta(),
        "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "environment": environment,
        "prefix": sys.prefix,
        "faulthandler": faulthandler.is_enabled(),
    }


def _cpu() -> str | None:
    if sys.platform == "darwin":
        try:
            done = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() or None
    if sys.platform.startswith("linux"):
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
        except OSError:
            return None
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
        return None
    return platform.processor() or None


def _rosetta() -> bool | None:
    """Whether this interpreter is x86_64 code translated by Rosetta 2; None off macOS.

    ``platform.machine()`` says ``x86_64`` on an Intel Mac too, so this asks the
    kernel, with ``sysctlbyname("sysctl.proc_translated")`` as Apple documents.
    """
    if sys.platform != "darwin":
        return None
    import ctypes
    import ctypes.util

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        status = libc.sysctlbyname(b"sysctl.proc_translated", ctypes.byref(value),
                                   ctypes.byref(size), None, ctypes.c_size_t(0))
    except (OSError, AttributeError):
        return None
    if status != 0:
        # An Intel Mac has no such key, because it has no Rosetta to translate with.
        return False if ctypes.get_errno() == errno.ENOENT else None
    return value.value == 1


def _threadpools() -> list[dict[str, Any]] | None:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    return threadpool_info()


def _environment() -> dict[str, str]:
    return {name: os.environ[name] for name in _ENVIRONMENT if name in os.environ}


def _numba(timeout: float = 300.0) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """numba's threading layer here and in a fresh process, and that process's runtimes."""
    if _distribution("numba") is None:
        return None, None
    deadline = time.monotonic() + timeout
    fresh = _start(_FRESH_PROCESS)
    children = {name: _start(_IMPORT_MODULE, module) for name, module in _LAYER_MODULES.items()}

    result, error = _finish(fresh, deadline)
    pools = result.pop("threadpools", None) if isinstance(result, dict) else None
    layers = {}
    for name, child in children.items():
        import_error, failure = _finish(child, deadline)
        problem = failure if failure is not None else import_error
        layers[name] = {"loads": problem is None, "error": problem}

    module = sys.modules.get("numba")
    this_process = None
    if module is not None:
        try:
            this_process = module.threading_layer()
        except ValueError:  # raised until the first parallel function runs
            this_process = None
    return {
        "imported": module is not None,
        "this_process": this_process,
        "fresh_process": result,
        "fresh_process_error": error,
        "layers": layers,
    }, pools


def _start(code: str, *args: str) -> subprocess.Popen[str] | str:
    """A child interpreter running ``code``, or the reason one could not start."""
    if not sys.executable:
        return "sys.executable is empty, so no child interpreter can start"
    try:
        return subprocess.Popen([sys.executable, "-X", "faulthandler", "-c", code, *args],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"could not start a child interpreter: {exc}"


def _finish(child: subprocess.Popen[str] | str, deadline: float) -> tuple[Any, str | None]:
    """The JSON a child printed last, or the reason there is none."""
    if isinstance(child, str):
        return None, child
    try:
        stdout, stderr = child.communicate(timeout=max(deadline - time.monotonic(), 1.0))
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        return None, "did not finish in time"
    if child.returncode < 0:
        try:
            name = signal.Signals(-child.returncode).name
        except ValueError:
            name = f"signal {-child.returncode}"
        return None, "\n".join([f"crashed with {name}", *stderr.strip().splitlines()[-20:]])
    lines = stdout.strip().splitlines()
    if child.returncode != 0 or not lines:
        errors = stderr.strip().splitlines()
        return None, errors[-1] if errors else f"exited with status {child.returncode}"
    try:
        return json.loads(lines[-1]), None
    except json.JSONDecodeError:
        return None, f"printed something other than a result: {lines[-1][:200]}"


def _llvm_openmp_copies(pools: list[dict[str, Any]] | None) -> list[str]:
    """The distinct files of LLVM's or Intel's OpenMP runtime among loaded thread pools.

    GNU's libgomp is left out: a second copy of it is not the reported crash.
    """
    return sorted({
        os.path.realpath(pool["filepath"]) for pool in pools or ()
        if pool.get("user_api") == "openmp" and pool.get("prefix") in ("libomp", "libiomp")
    })


def _warnings(report: dict[str, Any]) -> list[str]:
    """One sentence for each known cause of a crash that the report shows."""
    found = []
    if report["system"]["rosetta"]:
        found.append(
            "This Python is x86_64 code running under Rosetta 2 on Apple Silicon. numba "
            "stopped publishing Intel macOS wheels at 0.63 and llvmlite at 0.46, so "
            "truecell[analysis] is not supported here. Use an arm64 Python.")
    if report["system"]["free_threaded"]:
        found.append(
            "This is a free-threaded Python build, which truecell does not support: "
            "truecell[analysis] could not be installed on 3.14t.")
    for where in ("this_process", "fresh_process"):
        copies = _llvm_openmp_copies(report["threadpools"][where])
        if len(copies) > 1:
            found.append(
                f"{len(copies)} copies of the OpenMP runtime are loaded in "
                f"{where.replace('_', ' ')}: {', '.join(copies)}. Two in one process is "
                "the known cause of segmentation faults in numba-parallel code on Apple "
                "Silicon.")
    numba = report["numba"]
    if numba is not None and numba["fresh_process_error"] is not None:
        first = numba["fresh_process_error"].splitlines()[0]
        found.append(f"A fresh Python process could not run a numba parallel function: {first}.")
    return found


def _format(report: dict[str, Any]) -> str:
    lines: list[str] = []

    def row(label: str, value: object) -> None:
        lines.append(f"  {label:<15}{value}")

    def more(value: object) -> None:
        lines.append(f"  {'':<15}{value}")

    def installed(dist: dict[str, Any]) -> str:
        editable = ", editable" if dist["editable"] else ""
        return f"{dist['installer'] or 'installer unknown'}{editable}"

    tc = report["truecell"]
    if "version" in tc:
        lines.append(f"truecell {tc['version']} ({installed(tc)})")
    else:
        lines.append("truecell (no installed distribution; imported from a source tree)")
    row("path", tc["path"])

    system = report["system"]
    lines += ["", "System"]
    row("python", system["python"])
    row("executable", system["executable"])
    row("platform", system["platform"])
    cpu = system["cpu"] or "unknown"
    row("cpu", f"{cpu}, {system['cpu_count']} logical cores" if system["cpu_count"] else cpu)
    if system["rosetta"] is not None:
        row("rosetta 2", "yes" if system["rosetta"] else "no")
    row("free-threaded", "yes" if system["free_threaded"] else "no")
    row("environment", f"{system['environment']}, {system['prefix']}")
    row("faulthandler", "on" if system["faulthandler"] else "off")

    lines += ["", "Dependencies"]
    for name, dist in report["dependencies"].items():
        row(name, f"{dist['version']:<12}{installed(dist)}" if dist else "not installed")

    lines += ["", "numba threading layer"]
    numba = report["numba"]
    if numba is None:
        lines.append("  numba is not installed")
    else:
        if numba["this_process"] is not None:
            row("this process", numba["this_process"])
        elif numba["imported"]:
            row("this process", "not chosen yet: no parallel function has run")
        else:
            row("this process", "not chosen: numba is not imported")
        fresh = numba["fresh_process"]
        if fresh is not None:
            row("fresh process", f"{fresh['layer']}, {fresh['threads']} threads "
                                 f"(NUMBA_THREADING_LAYER={fresh['configured']}, "
                                 f"tried in order: {', '.join(fresh['priority'])})")
        else:
            first, *rest = numba["fresh_process_error"].splitlines()
            row("fresh process", f"failed: {first}")
            for line in rest:
                more(line)
        for layer, probe in numba["layers"].items():
            row(layer, "loads" if probe["loads"]
                else f"does not load: {probe['error'].splitlines()[0]}")

    lines += ["", "OpenMP and BLAS runtimes (threadpoolctl)"]
    pools = report["threadpools"]
    if pools["this_process"] is None:
        lines.append("  threadpoolctl is not installed")
    for where in ("this_process", "fresh_process"):
        if pools[where] is None:
            continue
        label = where.replace("_", " ")
        if not pools[where]:
            row(label, "none loaded")
        for i, pool in enumerate(pools[where]):
            name = " ".join(str(part) for part in (
                pool.get("prefix") or pool.get("internal_api"), pool.get("version")) if part)
            text = f"{name}, {pool.get('num_threads')} threads, {pool.get('filepath')}"
            row(label, text) if i == 0 else more(text)

    lines += ["", "Environment variables"]
    if report["environment"]:
        for name, value in report["environment"].items():
            note = ("  (scikit-learn and threadpoolctl set this when they are imported)"
                    if name == "KMP_DUPLICATE_LIB_OK" else "")
            lines.append(f"  {name}={value}{note}")
    else:
        lines.append("  none of the threading or library-path variables are set")

    lines += ["", "Warnings"]
    if report["warnings"]:
        lines += [textwrap.fill(warning, width=88, initial_indent="  - ",
                                subsequent_indent="    ") for warning in report["warnings"]]
    else:
        lines.append("  none")
    return "\n".join(lines)
