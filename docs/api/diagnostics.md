# Diagnostics

A crash in compiled code often depends more on the environment than on the call
that set it off. `show_versions` collects the parts of the environment that have
mattered: the numba threading layer, the OpenMP runtimes loaded, and whether
Python runs under Rosetta 2 or is a free-threaded build. It is truecell's
counterpart to R's `sessionInfo()`. [Troubleshooting](../troubleshooting.md) covers
what to do with its output.

::: truecell._show_versions.show_versions
