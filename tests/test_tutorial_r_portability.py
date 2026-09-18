"""The R scripts run on Linux as shipped.

Two verify scripts read THP-1's counts through `gzcat`, the macOS name for
`gzip -dc`. A Linux shell has only `zcat`, so both stopped on Linux before any
Seurat code ran. The benchmark's R arm had already moved to `gzip -dc`, and
`test_benchmark_harness.py` pins that one file; this reads every R script under
`tutorials/`. Found by running all eighteen tutorials on Linux x86-64 before
2.0.0.
"""
import re
from pathlib import Path

import pytest

TUTORIALS = Path(__file__).resolve().parents[1] / "tutorials"
R_SCRIPTS = sorted(TUTORIALS.rglob("*.R"))

# A quoted command name, as `paste("gzcat", ...)` or `system("zcat ...")` passes
# it to the shell. Comments naming the commands in backticks do not match.
ONE_PLATFORM_DECOMPRESSOR = re.compile(r"""["'](gzcat|zcat)\b""")


def test_the_scan_finds_the_r_scripts():
    names = {p.name for p in R_SCRIPTS}
    assert {"thp1_mixscape_verify.R", "thp1_cellcycle_verify.R",
            "bench_seurat.R"} <= names
    assert len(R_SCRIPTS) >= 18


@pytest.mark.parametrize("path", R_SCRIPTS,
                         ids=lambda p: str(p.relative_to(TUTORIALS)))
def test_no_r_script_decompresses_with_a_command_one_platform_lacks(path):
    calls = [line.strip() for line in path.read_text().splitlines()
             if ONE_PLATFORM_DECOMPRESSOR.search(line)
             and not line.lstrip().startswith("#")]
    assert not calls, (
        f"{path.name} shells out to a decompressor only one platform has; "
        f"`gzip -dc` works on both: {calls}"
    )
