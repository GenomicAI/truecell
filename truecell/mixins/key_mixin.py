from __future__ import annotations

import re


_KEY_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9]*_$")


def _validate_key(key: str) -> None:
    """Keys must be non-empty, start with a letter, contain only alphanumerics, and end with '_'."""
    if not _KEY_RE.match(key):
        raise ValueError(
            f"Invalid key '{key}'. Keys must start with a letter, contain only "
            "alphanumeric characters, and end with '_'."
        )


def update_key(key: str) -> str:
    """A valid key made from ``key``, by SeuratObject's ``UpdateKey`` rule.

    A string that already is a key comes back unchanged. Otherwise its runs of
    letters and digits are joined and ``_`` is appended. This is what
    ``Key(reduction.name)`` gives in SeuratObject 5.4.0: ``"umap"`` becomes
    ``"umap_"``, ``"wnn_umap"`` becomes ``"wnnumap_"`` and ``"ref.umap"`` becomes
    ``"refumap_"``. When nothing alphanumeric is left, SeuratObject makes up a
    key from three random letters. A random key cannot be reproduced, so this
    raises instead.
    """
    if _KEY_RE.fullmatch(key):
        return key
    new_key = "".join(re.findall(r"[A-Za-z0-9]+", key)) + "_"
    if new_key == "_":
        raise ValueError(
            f"Cannot make a key from {key!r}, which has no letters or digits; "
            "pass the key explicitly."
        )
    return new_key


class KeyMixin:
    """Mixin providing a validated 'key' slot, mirroring R's KeyMixin from keymixin.R."""

    __slots__ = ("_key",)

    @property
    def key(self) -> str:
        return self._key

    @key.setter
    def key(self, value: str) -> None:
        _validate_key(value)
        self._key = value
