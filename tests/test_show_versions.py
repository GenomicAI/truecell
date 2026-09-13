"""`show_versions()` reports what a crash report needs, and leaves the session alone.

Its numba and OpenMP probes run in child interpreters. These tests check the
report against the environment running them, and that nothing the probes do
reaches the caller: a threading layer, once chosen, stays for the life of a
process, which is exactly what a diagnostic must not decide for the session it
is diagnosing.
"""
import importlib.util
import json
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import truecell
from truecell import _show_versions as sv

HEADINGS = ("System", "Dependencies", "numba threading layer",
            "OpenMP and BLAS runtimes (threadpoolctl)", "Environment variables", "Warnings")


@pytest.fixture(scope="module")
def report():
    return truecell.show_versions(as_dict=True)


def _pool(path, prefix="libomp"):
    return {"user_api": "openmp", "internal_api": "openmp", "prefix": prefix,
            "filepath": str(path), "version": None, "num_threads": 8}


def _bare_report(**overrides):
    """The fields `_warnings` reads, set to an environment with nothing wrong in it."""
    report = {"system": {"rosetta": False, "free_threaded": False},
              "threadpools": {"this_process": [], "fresh_process": []},
              "environment": {}, "numba": None}
    report.update(overrides)
    return report


def test_the_versions_are_the_ones_imported(report):
    import numpy
    import pandas
    import scipy

    deps = report["dependencies"]
    assert deps["numpy"]["version"] == numpy.__version__
    assert deps["scipy"]["version"] == scipy.__version__
    assert deps["pandas"]["version"] == pandas.__version__
    assert report["truecell"]["version"] == truecell.__version__
    path = Path(truecell.__file__).resolve().parent
    assert report["truecell"]["path"] == str(path)
    # An editable install runs from the checkout, a wheel from site-packages.
    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    assert report["truecell"]["editable"] is (site not in path.parents)


def test_a_package_that_is_not_installed_is_none_and_an_alias_is_followed(monkeypatch):
    import numpy

    assert sv._distribution("truecell-no-such-distribution") is None
    monkeypatch.setitem(sv._ALIASES, "numpy-by-another-name", ("numpy",))
    assert sv._distribution("numpy-by-another-name")["version"] == numpy.__version__


def test_the_report_is_plain_json(report):
    assert json.loads(json.dumps(report)) == report


def test_the_fresh_process_runs_numba_on_a_layer_that_loads(report):
    pytest.importorskip("numba")
    numba = report["numba"]
    assert numba["fresh_process_error"] is None
    fresh = numba["fresh_process"]
    assert fresh["layer"] in {"tbb", "omp", "workqueue"}
    assert fresh["priority"] and fresh["threads"] >= 1
    # workqueue is numba's own pool, built into every wheel.
    assert numba["layers"]["workqueue"] == {"loads": True, "error": None}
    assert numba["layers"][fresh["layer"]]["loads"], "numba chose a layer the probe says cannot load"
    if importlib.util.find_spec("sklearn") and importlib.util.find_spec("threadpoolctl"):
        # scikit-learn's wheel links an OpenMP runtime, and the probe imports it
        # before numba starts its threads, the order truecell's pipeline uses.
        assert any(pool["user_api"] == "openmp" for pool in report["threadpools"]["fresh_process"])


def test_the_probes_leave_the_callers_process_alone(tmp_path):
    """No threading layer is chosen and no layer is loaded in the calling interpreter.

    Run in a fresh interpreter, because earlier tests in this session may have
    started numba's threads already.
    """
    pytest.importorskip("numba")
    code = (
        "import json, sys, truecell\n"
        "truecell.show_versions(as_dict=True)\n"
        "import numba\n"
        "try:\n"
        "    chosen = numba.threading_layer()\n"
        "except ValueError:\n"
        "    chosen = None\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('numba.np.ufunc.')\n"
        "                and m.rsplit('.', 1)[1] in ('tbbpool', 'omppool', 'workqueue'))\n"
        "print(json.dumps({'chosen': chosen, 'loaded': loaded}))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=tmp_path, timeout=600, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1]) == {"chosen": None, "loaded": []}


def test_the_probe_sees_the_callers_environment(monkeypatch):
    pytest.importorskip("numba")
    monkeypatch.setenv("NUMBA_THREADING_LAYER", "workqueue")
    numba, _ = sv._numba()
    assert numba["fresh_process"]["configured"] == "workqueue"
    assert numba["fresh_process"]["layer"] == "workqueue"
    assert sv._environment()["NUMBA_THREADING_LAYER"] == "workqueue"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_probe_that_crashes_is_reported_not_raised(monkeypatch):
    pytest.importorskip("numba")
    monkeypatch.setattr(sv, "_FRESH_PROCESS", "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n")
    report = truecell.show_versions(as_dict=True)
    error = report["numba"]["fresh_process_error"]
    assert report["numba"]["fresh_process"] is None
    assert error.startswith("crashed with SIGSEGV")
    assert "Fatal Python error: Segmentation fault" in error  # faulthandler's traceback came along
    assert report["threadpools"]["fresh_process"] is None
    assert any("could not run a numba parallel function" in w for w in report["warnings"])
    assert "failed: crashed with SIGSEGV" in sv._format(report)


def test_two_copies_of_the_openmp_runtime_are_counted_by_file(tmp_path):
    sklearn, torch = tmp_path / "sklearn-libomp.dylib", tmp_path / "torch-libomp.dylib"
    sklearn.touch()
    torch.touch()
    alias = tmp_path / "alias.dylib"
    alias.symlink_to(sklearn)

    assert sv._llvm_openmp_copies([_pool(sklearn), _pool(torch)]) == sorted(
        [os.path.realpath(sklearn), os.path.realpath(torch)])
    # The same file reported twice, or reached through a link, is one runtime.
    assert sv._llvm_openmp_copies([_pool(sklearn), _pool(sklearn), _pool(alias)]) == [
        os.path.realpath(sklearn)]
    # GNU's runtime is not the crash being looked for.
    assert sv._llvm_openmp_copies([_pool("/a/libgomp.so.1", "libgomp"),
                                   _pool("/b/libgomp-x.so.1", "libgomp")]) == []

    assert sv._warnings(_bare_report()) == []
    one = [_pool(sklearn), _pool(alias)]
    assert sv._warnings(_bare_report(threadpools={"this_process": one, "fresh_process": one})) == []
    two = _bare_report(threadpools={"this_process": [], "fresh_process": [_pool(sklearn), _pool(torch)]})
    [warning] = sv._warnings(two)
    assert warning.startswith("2 copies of the OpenMP runtime are loaded in fresh process")


def test_rosetta_and_free_threading_each_warn():
    [rosetta] = sv._warnings(_bare_report(system={"rosetta": True, "free_threaded": False}))
    assert "Rosetta 2" in rosetta and "arm64 Python" in rosetta
    [threaded] = sv._warnings(_bare_report(system={"rosetta": None, "free_threaded": True}))
    assert "free-threaded" in threaded


def test_kmp_duplicate_lib_ok_is_reported_but_not_warned_about(report):
    """threadpoolctl sets it on import, so a warning would fire in every session."""
    pytest.importorskip("threadpoolctl")
    assert report["environment"]["KMP_DUPLICATE_LIB_OK"] == os.environ["KMP_DUPLICATE_LIB_OK"]
    assert sv._warnings(_bare_report(environment={"KMP_DUPLICATE_LIB_OK": "True"})) == []
    assert "(scikit-learn and threadpoolctl set this when they are imported)" in sv._format(report)


def test_rosetta_is_asked_only_on_macos():
    value = sv._rosetta()
    if sys.platform != "darwin":
        assert value is None
    elif platform.machine() == "arm64":
        assert value is False  # a native arm64 interpreter is never translated
    else:
        assert value in (True, False)


def test_the_printed_report_has_every_section(report):
    text = sv._format(report)
    assert text.startswith(f"truecell {truecell.__version__} (")
    for heading in HEADINGS:
        assert f"\n{heading}\n" in text
    for name in sv._DEPENDENCIES:
        assert f"\n  {name} " in text
    if report["numba"] is not None:
        assert f"  fresh process  {report['numba']['fresh_process']['layer']}, " in text


def test_show_versions_prints_and_returns_none(monkeypatch, capsys):
    """A REPL would print a returned dict a second time under the report."""
    monkeypatch.setattr(sv, "_numba", lambda: (None, None))
    assert truecell.show_versions() is None
    out = capsys.readouterr().out
    assert out.startswith("truecell ") and "  numba is not installed" in out
