# Troubleshooting

## Python crashed without a traceback

When Python stops with `Segmentation fault`, or Jupyter says the kernel died,
compiled code failed and Python had no chance to say where. Run the script again
with faulthandler on:

```bash
python -X faulthandler my_analysis.py
```

At the crash it prints the Python stack of every thread, so you can see which call
was running. In Jupyter, run `import faulthandler; faulthandler.enable()` in the
first cell. The traceback then goes to the terminal Jupyter was started from, not
to the notebook.

## What to put in a bug report

```bash
python -c "import truecell; truecell.show_versions()"
```

Run it in the environment that crashed. Paste its output into
[an issue](https://github.com/GenomicAI/truecell/issues), with the faulthandler
traceback and the commands that installed truecell. It reports:

- the platform, including whether Python runs under Rosetta 2 or is a free-threaded build;
- the version of each package in the scientific stack, and what installed it;
- numba's threading layer;
- the OpenMP and BLAS runtimes loaded.

It ends with a warning for each known cause of a crash it finds. The numba checks
run in separate processes, so running it cannot change the session it reports on,
or crash it.

## A segmentation fault in `run_umap`

`run_umap` calls umap-learn, whose nearest-neighbour search runs numba code on
parallel threads. numba runs them on one of three threading layers: the first that
loads out of `tbb`, `omp` (OpenMP) and `workqueue`, numba's own pool. Scanpy users
have reported the crash on Apple Silicon with the OpenMP layer. A second copy of
the OpenMP runtime was loaded in the same process, torch's beside scikit-learn's
([scanpy#4026](https://github.com/scverse/scanpy/issues/4026),
[#3507](https://github.com/scverse/scanpy/issues/3507)).

Try numba's own pool first:

```bash
export NUMBA_THREADING_LAYER=workqueue
```

Or set it in Python before anything imports numba:

```python
import os
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

import truecell
```

`show_versions()` shows whether it took: its "fresh process" line names the layer a
new process picks.

### What has been tried

A reviewer of the truecell paper hit this crash on an Apple M3 Pro, running the
block on the [home page](index.md). On an Apple M4 Pro with macOS 26.6, the same
block ran to the end in every configuration tried:

| Configuration | numba layer | Result |
|---|---|---|
| `pip install "truecell[analysis]"`, Python 3.12, 3.13 and 3.14 | workqueue | completes |
| `uv sync --locked` | workqueue | completes |
| torch installed and imported before truecell | workqueue | completes, with torch's and scikit-learn's OpenMP runtimes both loaded |
| numba's OpenMP layer made loadable from torch's runtime | omp | completes, with both runtimes loaded |

The two torch configurations also complete with `KMP_DUPLICATE_LIB_OK=FALSE`.
scikit-learn and threadpoolctl otherwise set that variable to `True` when they are
imported.

With pip's wheels, numba cannot load its OpenMP layer on macOS. The wheel's
`omppool` module links a `libomp.dylib` that nothing on the library path provides,
so numba falls back to workqueue. A conda-forge environment and an older macOS
were not tried. The [macOS canary](https://github.com/GenomicAI/truecell/blob/main/.github/workflows/macos-canary.yml)
runs the block every week on macOS 15 in four environments: pip, uv, conda-forge,
and conda-forge's numba beside pip for the rest.

## Conda environments

numba from conda-forge is linked against conda's own OpenMP runtime
(`llvm-openmp`), so its OpenMP layer can load. scikit-learn's pip wheel carries its
own copy of the runtime. Install numba with conda and the rest with pip, and one
process ends up with two OpenMP runtimes and an OpenMP layer that can load. That is
the combination behind the scanpy reports. Pick one package manager for the
numerical stack:

- **All conda.** Install the stack from conda-forge, then truecell alone with pip:

    ```bash
    conda install -c conda-forge numpy scipy pandas packaging numba llvmlite \
        scikit-learn umap-learn pynndescent statsmodels python-igraph leidenalg \
        matplotlib-base seaborn scikit-misc threadpoolctl
    pip install --no-deps truecell
    ```

- **No conda.** A venv with pip or uv for everything, which is what CI tests.

`show_versions()` lists every OpenMP runtime it finds loaded, and warns when LLVM's
or Intel's is loaded more than once.

## Apple Silicon: use an arm64 Python

numba stopped publishing Intel macOS wheels at 0.63, and llvmlite at 0.46, both in
December 2025. On an Apple Silicon Mac, an x86_64 Python cannot install them. That
means an Intel build of Python, or any Python started under Rosetta 2. pip tries to
build llvmlite from source, and the build fails. `[analysis]` needs numba, so it
cannot be installed. An Intel Mac has no wheels to install either.

Check which Python you have:

```bash
python -c "import platform; print(platform.machine())"
```

It should print `arm64`. For a translated interpreter, `show_versions()` reports
`rosetta 2  yes` and prints a warning.

## Free-threaded Python (3.14t)

truecell does not support the free-threaded build.
`pip install "truecell[analysis]"` fails on 3.14t: in September 2026, igraph and
leidenalg published no free-threaded wheels, and their source builds fail against
its headers. The regular 3.14 build installed and ran the home-page block, but CI
does not test 3.14 yet; see [Installation](installation.md#why-the-floor-is-312).
