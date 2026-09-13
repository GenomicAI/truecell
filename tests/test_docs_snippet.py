"""`tools/run_docs_snippet.py` runs the home-page block the way the macOS canary needs.

The canary exists because a reviewer's run of that block died in `run_umap` with
a segmentation fault. For its runs to mean anything, three things must hold, and
each is checked here without the dataset: the block it runs is the published
one, an error is reported against the page's own line, and a native crash fails
the process with a faulthandler traceback that names that line.
"""
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_docs_snippet.py"
sys.path.insert(0, str(ROOT / "tools"))

import run_docs_snippet


def _page(tmp_path, code, prose_lines=37):
    """A page with a python block after some prose, and the line its code starts on."""
    lines = [f"Prose, line {i + 1}." for i in range(prose_lines)] + ["", "```python", *code, "```", ""]
    page = tmp_path / "page.md"
    page.write_text("\n".join(lines), encoding="utf-8")
    return page, prose_lines + 3


def test_the_home_page_has_one_block_and_it_calls_run_umap():
    first_line, code = run_docs_snippet.the_snippet(run_docs_snippet.HOME_PAGE)
    assert "truecell.run_umap(" in code
    page_lines = run_docs_snippet.HOME_PAGE.read_text(encoding="utf-8").splitlines()
    assert page_lines[first_line - 1] == code.splitlines()[0]


def test_an_error_names_the_line_of_the_page(tmp_path):
    page, first = _page(tmp_path, ["x = 1", "raise RuntimeError('raised on the page')"])
    done = subprocess.run([sys.executable, str(RUNNER), str(page)],
                          capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 1
    assert f'File "{page}", line {first + 1}' in done.stderr
    assert "RuntimeError: raised on the page" in done.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_segfault_fails_the_run_and_its_traceback_names_the_line(tmp_path):
    page, first = _page(tmp_path, ["import ctypes", "", "ctypes.string_at(0)"])
    done = subprocess.run([sys.executable, "-X", "faulthandler", str(RUNNER), str(page)],
                          capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode in (-signal.SIGSEGV, -signal.SIGBUS), done.returncode
    assert "Fatal Python error" in done.stderr
    assert f'File "{page}", line {first + 2}' in done.stderr


def test_a_page_with_two_blocks_is_refused(tmp_path):
    page = tmp_path / "two.md"
    page.write_text("```python\nx = 1\n```\n\ntext\n\n```python\ny = 2\n```\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="has 2 python blocks"):
        run_docs_snippet.the_snippet(page)
