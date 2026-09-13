"""Run the Python block on the documentation home page, exactly as published.

The macOS canary (`.github/workflows/macos-canary.yml`) runs this under
`python -X faulthandler`. The block is compiled under the page's own file name
and line numbers, so when a native library crashes, the traceback faulthandler
prints names the line of `docs/index.md` that was running. Reading the block
out of the page, rather than keeping a copy of it, means the canary always runs
what the site shows.

    python -X faulthandler tools/run_docs_snippet.py [PAGE] [--show-versions]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME_PAGE = ROOT / "docs" / "index.md"

_BLOCK = re.compile(r"^```python[ \t]*\n(?P<code>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def python_blocks(text: str) -> list[tuple[int, str]]:
    """Each fenced python block in a page, with the line its code starts on."""
    return [(text.count("\n", 0, match.start("code")) + 1, match.group("code"))
            for match in _BLOCK.finditer(text)]


def the_snippet(page: Path) -> tuple[int, str]:
    """The page's one python block. More than one is refused, never guessed between."""
    blocks = python_blocks(page.read_text(encoding="utf-8"))
    if len(blocks) != 1:
        raise SystemExit(f"{page} has {len(blocks)} python blocks; this runner needs exactly one")
    return blocks[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Python block on a docs page.")
    parser.add_argument("page", nargs="?", type=Path, default=HOME_PAGE)
    parser.add_argument("--show-versions", action="store_true",
                        help="then print truecell.show_versions() from the same process, "
                             "which reports the threading layer the snippet ran on")
    args = parser.parse_args(argv)

    first_line, code = the_snippet(args.page)
    # Blank lines in front put every statement on its own line of the page, so a
    # traceback reads `File "docs/index.md", line 34`.
    compiled = compile("\n" * (first_line - 1) + code, str(args.page), "exec")
    started = time.perf_counter()
    exec(compiled, {"__name__": "__main__"})  # noqa: S102 - running the published block is the point
    print(f"{args.page}: the snippet ran to the end in {time.perf_counter() - started:.1f} s",
          flush=True)
    if args.show_versions:
        import truecell

        truecell.show_versions()
    return 0


if __name__ == "__main__":
    sys.exit(main())
