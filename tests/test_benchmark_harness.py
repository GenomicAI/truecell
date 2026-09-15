"""The benchmark harness runs on Linux as well as macOS.

Four places assumed a Mac: `sysctl` for the machine description, CRAN's macOS
path for Rscript, a `uv sync` checkout's `.venv` for Python, and `gzcat` for the
THP-1 counts. These tests run the harness's own helpers, not a benchmark, so
CI's Linux runners check them.
"""
import os
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "tutorials" / "benchmark"
sys.path.insert(0, str(HARNESS))

import run_benchmarks as rb


def test_the_machine_is_described_on_this_platform():
    machine = rb.machine()
    assert machine["cpu"] not in ("", "?"), machine
    assert int(machine["cores"]) == os.cpu_count()
    assert machine["memory_gb"] > 0.5
    assert machine["platform"]


def test_the_truecell_arm_runs_this_python_unless_told_otherwise(monkeypatch):
    monkeypatch.delenv("TRUECELL_BENCH_PYTHON", raising=False)
    assert rb.ARMS["truecell"]()[0] == sys.executable
    monkeypatch.setenv("TRUECELL_BENCH_PYTHON", "/opt/wheel-env/bin/python")
    assert rb.ARMS["truecell"]()[0] == "/opt/wheel-env/bin/python"


def test_the_seurat_arm_finds_rscript_on_path_or_the_override(monkeypatch, tmp_path):
    rscript = tmp_path / "Rscript"
    rscript.write_text("#!/bin/sh\n")
    rscript.chmod(0o755)
    monkeypatch.delenv("TRUECELL_BENCH_RSCRIPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert rb.ARMS["seurat"]()[0] == str(rscript)
    monkeypatch.setenv("TRUECELL_BENCH_RSCRIPT", "/opt/R/bin/Rscript")
    assert rb.ARMS["seurat"]()[0] == "/opt/R/bin/Rscript"


def test_without_rscript_the_error_says_how_to_name_one(monkeypatch, tmp_path):
    monkeypatch.delenv("TRUECELL_BENCH_RSCRIPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(SystemExit, match="TRUECELL_BENCH_RSCRIPT"):
        rb.ARMS["seurat"]()


def test_the_truecell_arm_s_blas_is_asked_of_its_own_interpreter(monkeypatch):
    import numpy

    monkeypatch.delenv("TRUECELL_BENCH_PYTHON", raising=False)
    blas = rb.arm_blas("truecell", rb.child_env(threads=1))
    assert blas["numpy"] == numpy.__version__
    assert blas["blas"] and blas["lapack"]
    assert isinstance(blas["threadpools"], list)
    # The pin reaches the child. threadpoolctl cannot see Accelerate, so on macOS
    # the list is empty; on Linux it holds NumPy's OpenBLAS.
    assert all(pool["num_threads"] == 1 for pool in blas["threadpools"])
    assert rb.blas_label(blas) == f"NumPy {numpy.__version__} on {blas['blas']}"


def test_a_probe_that_cannot_run_is_recorded_not_raised(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUECELL_BENCH_PYTHON", str(tmp_path / "no-such-python"))
    blas = rb.arm_blas("truecell")
    assert blas["error"].startswith("FileNotFoundError")
    assert rb.blas_label(blas).startswith("BLAS unknown (FileNotFoundError")
    assert rb.blas_label(None) == "BLAS not recorded"


def test_an_r_record_is_labelled_by_its_library_file():
    record = {"r": "R version 4.6.1 (2026-06-24)",
              "blas": "/usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3",
              "lapack": "/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so"}
    assert rb.blas_label(record) == "R version 4.6.1 (2026-06-24) on libblas.so.3"


def test_the_report_skips_the_tutorial_scripts_file(monkeypatch, tmp_path, capsys):
    """`scripts` writes a file of another shape into the same directory, and
    `report` used to stop at it with a KeyError."""
    (tmp_path / "tutorial_scripts.json").write_text('{"machine": {}, "tutorials": {}}')
    (tmp_path / "blas_probe.truecell.json").write_text(
        '{"machine": {}, "runs": [{"wall_seconds": 1.0, "process_peak_rss_mb": 10.0,'
        ' "baseline_rss_mb": 5.0, "steps": [{"step": "blas_gemm", "seconds": 0.5,'
        ' "peak_rss_mb": 9.0, "anchor": 1}]}]}')
    monkeypatch.setattr(rb, "RESULTS", tmp_path)
    assert rb.cmd_report(None) == 0
    out = capsys.readouterr().out
    assert "=== blas_probe.truecell  (1 repeats" in out
    assert "BLAS not recorded" in out
    assert "tutorial_scripts" not in out


def test_the_seurat_arm_decompresses_with_a_command_both_platforms_have():
    source = (HARNESS / "bench_seurat.R").read_text()
    assert 'fread(cmd = paste("gzip -dc",' in source
    assert '"gzcat"' not in source and '"zcat"' not in source
