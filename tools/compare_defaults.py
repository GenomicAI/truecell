"""Compare truecell's signature defaults with those of the Seurat functions it ports.

    python tools/compare_defaults.py            # list what differs, and why
    python tools/compare_defaults.py --refresh  # re-read Seurat's formals (needs R)

The reference is ``tests/data/seurat_formals.json``, dumped from a live Seurat by
``tools/compare_defaults.R``. ``tests/test_default_parity.py`` runs the same
comparison against it, so CI needs no R.

Why it exists: a default is the kind of detail that reads correctly in a diff and
that nobody re-derives. ``find_markers`` shipped Seurat 4's ``logfc_threshold``
and ``min_pct`` for months after Seurat 5 changed them, and it was the second
such miss to be found by accident rather than by a check.

Every default a truecell function shares with its Seurat counterpart must either
equal Seurat's or appear in one of two tables below, with the reason:
:data:`EQUIVALENT` for a value spelled differently that resolves to Seurat's at
call time, :data:`KNOWN_DIVERGENCES` for a real difference kept on purpose or not
yet fixed. An entry that stops describing a live difference is an error too, so
neither table can quietly outlive what it excuses.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "tests" / "data" / "seurat_formals.json"

S, O = "Seurat", "SeuratObject"

#: truecell function -> the R function(s) whose formals it mirrors. When a Seurat
#: generic splits its arguments between the object method and the default method,
#: both are listed and the first one to declare an argument wins.
FUNCTIONS: dict[str, tuple[str, ...]] = {
    "truecell.create_truecell_object": (f"{O}:::CreateSeuratObject.default",),
    "truecell.create_assay_object": (f"{O}::CreateAssayObject",),
    "truecell.create_assay5_object": (f"{O}::CreateAssay5Object",),
    "truecell.diet_truecell": (f"{S}::DietSeurat",),
    "truecell.percentage_feature_set": (f"{S}::PercentageFeatureSet",),
    "truecell.normalize_data": (f"{S}:::NormalizeData.Seurat", f"{S}:::NormalizeData.default"),
    "truecell.find_variable_features": (f"{S}:::FindVariableFeatures.Seurat",
                                        f"{S}:::FindVariableFeatures.default"),
    "truecell.scale_data": (f"{S}:::ScaleData.Seurat", f"{S}:::ScaleData.default"),
    "truecell.sctransform": (f"{S}:::SCTransform.Seurat", f"{S}:::SCTransform.default"),
    "truecell.prep_sct_find_markers": (f"{S}::PrepSCTFindMarkers",),
    "truecell.run_pca": (f"{S}:::RunPCA.Seurat", f"{S}:::RunPCA.default"),
    "truecell.run_ica": (f"{S}:::RunICA.Seurat", f"{S}:::RunICA.default"),
    "truecell.run_spca": (f"{S}:::RunSPCA.Seurat", f"{S}:::RunSPCA.default"),
    "truecell.run_tsne": (f"{S}:::RunTSNE.Seurat", f"{S}:::RunTSNE.matrix"),
    "truecell.run_umap": (f"{S}:::RunUMAP.Seurat", f"{S}:::RunUMAP.default"),
    "truecell.jack_straw": (f"{S}::JackStraw",),
    "truecell.score_jackstraw": (f"{S}:::ScoreJackStraw.Seurat", f"{S}:::ScoreJackStraw.DimReduc"),
    "truecell.find_neighbors": (f"{S}:::FindNeighbors.Seurat", f"{S}:::FindNeighbors.default"),
    "truecell.find_clusters": (f"{S}:::FindClusters.Seurat", f"{S}:::FindClusters.default"),
    "truecell.find_multi_modal_neighbors": (f"{S}::FindMultiModalNeighbors",),
    "truecell.integrate_layers": (f"{S}::IntegrateLayers",),
    "truecell.find_integration_anchors": (f"{S}::FindIntegrationAnchors",),
    "truecell.select_integration_features": (f"{S}::SelectIntegrationFeatures",),
    "truecell.integrate_data": (f"{S}::IntegrateData",),
    "truecell.integrate_embeddings": (f"{S}:::IntegrateEmbeddings.IntegrationAnchorSet",),
    "truecell.find_transfer_anchors": (f"{S}::FindTransferAnchors",),
    "truecell.transfer_data": (f"{S}::TransferData",),
    "truecell.map_query": (f"{S}::MapQuery",),
    "truecell.project_umap": (f"{S}:::ProjectUMAP.Seurat", f"{S}:::ProjectUMAP.default"),
    "truecell.add_module_score": (f"{S}::AddModuleScore",),
    "truecell.cell_cycle_scoring": (f"{S}::CellCycleScoring",),
    "truecell.hto_demux": (f"{S}::HTODemux",),
    "truecell.multiseq_demux": (f"{S}::MULTIseqDemux",),
    "truecell.calc_perturb_sig": (f"{S}::CalcPerturbSig",),
    "truecell.run_mixscape": (f"{S}::RunMixscape",),
    "truecell.mixscape_lda": (f"{S}::MixscapeLDA",),
    "truecell.find_markers": (f"{S}:::FindMarkers.Seurat", f"{S}:::FindMarkers.default"),
    "truecell.find_all_markers": (f"{S}::FindAllMarkers",),
    "truecell.find_conserved_markers": (f"{S}::FindConservedMarkers", f"{S}:::FindMarkers.Seurat",
                                        f"{S}:::FindMarkers.default"),
    "truecell.aggregate_expression": (f"{S}::AggregateExpression",),
    "truecell.average_expression": (f"{S}::AverageExpression",),
    "truecell.load_xenium": (f"{S}::LoadXenium",),
    "truecell.load_visium": (f"{S}::Load10X_Spatial",),
    "truecell.load_cosmx": (f"{S}::LoadNanostring",),
    "truecell.load_merscope": (f"{S}::LoadVizgen",),
    "truecell.build_niche_assay": (f"{S}::BuildNicheAssay",),
    "truecell.find_spatially_variable_features": (f"{S}:::FindSpatiallyVariableFeatures.Seurat",
                                                  f"{S}:::FindSpatiallyVariableFeatures.default"),
    "truecell.leverage_score": (f"{S}:::LeverageScore.Seurat", f"{S}:::LeverageScore.default"),
    "truecell.sketch_data": (f"{S}::SketchData",),
    "truecell.project_data": (f"{S}::ProjectData",),
    "truecell.dim_plot": (f"{S}::DimPlot",),
    "truecell.feature_plot": (f"{S}::FeaturePlot",),
    "truecell.vln_plot": (f"{S}::VlnPlot",),
    "truecell.do_heatmap": (f"{S}::DoHeatmap",),
    "truecell.ridge_plot": (f"{S}::RidgePlot",),
    "truecell.dot_plot": (f"{S}::DotPlot",),
    "truecell.elbow_plot": (f"{S}::ElbowPlot",),
    "truecell.feature_scatter": (f"{S}::FeatureScatter",),
    "truecell.variable_feature_plot": (f"{S}::VariableFeaturePlot",),
    "truecell.viz_dim_loadings": (f"{S}::VizDimLoadings",),
    "truecell.dim_heatmap": (f"{S}::DimHeatmap",),
    "truecell.image_dim_plot": (f"{S}::ImageDimPlot",),
    "truecell.image_feature_plot": (f"{S}::ImageFeaturePlot",),
    "truecell.spatial_dim_plot": (f"{S}::SpatialDimPlot",),
    "truecell.spatial_feature_plot": (f"{S}::SpatialFeaturePlot",),
    "truecell.plot_perturb_score": (f"{S}::PlotPerturbScore",),
    "truecell.mixscape_heatmap": (f"{S}::MixscapeHeatmap",),
    "truecell.io.read_10x": (f"{S}::Read10X",),
}

#: R argument -> truecell argument, where the rename is not the mechanical
#: ``a.b`` -> ``a_b``. Applied to every function.
RENAMES: dict[str, str] = {"slot": "layer", "seed.use": "seed", "npcs": "n_pcs"}

#: Renames that hold for one function only, where the same R name means
#: something else elsewhere.
FUNCTION_RENAMES: dict[str, dict[str, str]] = {
    # MixscapeLDA's `assay` is the DE assay and `pc.assay` the signature assay;
    # truecell spells them `de_assay` and `assay`.
    "truecell.mixscape_lda": {"assay": "de_assay", "pc.assay": "assay"},
}

#: Arguments that change what is printed, never what is computed.
IGNORED: frozenset[str] = frozenset({"verbose"})

#: (function, argument) -> why the two defaults are the same value spelled
#: differently. Only for a default that resolves to Seurat's at call time.
EQUIVALENT: dict[tuple[str, str], str] = {
    ("truecell.find_markers", "layer"):
        "None reads the data layer (markers._get_expression_matrix), Seurat's default.",
    ("truecell.find_all_markers", "layer"):
        "Passed through to find_markers, where None reads the data layer.",
    ("truecell.find_conserved_markers", "layer"):
        "Passed through to find_markers, where None reads the data layer.",
    ("truecell.find_markers", "max_cells_per_ident"):
        "None downsamples nothing, which is what Seurat's Inf means.",
    ("truecell.find_all_markers", "max_cells_per_ident"):
        "None downsamples nothing, which is what Seurat's Inf means.",
    ("truecell.feature_plot", "layer"):
        "None reads the data layer (plotting._get_data_matrix), Seurat's default.",
    ("truecell.ridge_plot", "layer"):
        "None reads the data layer (plotting._get_data_matrix), Seurat's default.",
    ("truecell.feature_scatter", "layer"):
        "None reads the data layer (plotting._get_data_matrix), Seurat's default.",
    ("truecell.spatial_feature_plot", "layer"):
        "None reads the data layer (plotting._get_data_matrix), Seurat's default.",
    ("truecell.do_heatmap", "group_by"):
        "None groups by the active identities, which is what Seurat's 'ident' names.",
    ("truecell.load_visium", "image"):
        "Seurat's NULL reads the bundle's tissue image; truecell's True does the same.",
}

#: (function, argument) -> why truecell's default differs from Seurat's.
KNOWN_DIVERGENCES: dict[tuple[str, str], str] = {
    # -- Kept on purpose.
    ("truecell.find_integration_anchors", "reference"):
        "Seurat's NULL integrates every pair along a guide tree (BuildSampleTree), "
        "which truecell does not port; for two datasets both defaults take the one pair.",
    ("truecell.sctransform", "seed"):
        "R's random-number stream cannot be reproduced in Python, so no seed makes the "
        "cell subsample match Seurat's; 42 is truecell's convention.",
    ("truecell.run_tsne", "seed"):
        "R's random-number stream cannot be reproduced in Python, so no seed makes the "
        "embedding match Seurat's; 42 is truecell's convention.",
    ("truecell.hto_demux", "nstarts"):
        "Only read when kfunc='kmeans', and the default kfunc, 'clara', never uses it; "
        "truecell keeps that k-means path at 10 restarts.",
    ("truecell.sctransform", "assay"):
        "Seurat names 'RNA' outright; truecell's None is the active assay, which is "
        "'RNA' unless it was changed.",
    ("truecell.find_conserved_markers", "assay"):
        "Seurat names 'RNA' outright; truecell's None is the active assay, which is "
        "'RNA' unless it was changed.",
    ("truecell.create_assay_object", "key"):
        "Seurat's NULL leaves the key to be set from the assay name when the assay is "
        "added to an object; truecell's factory takes the key up front.",
    ("truecell.calc_perturb_sig", "assay"):
        "Seurat's NULL means DefaultAssay(object); truecell names 'RNA', the vignette's.",
    ("truecell.calc_perturb_sig", "num_neighbors"):
        "Seurat's NULL stops with 'Please specify number of nearest neighbors to "
        "consider'; truecell supplies the vignette's 20.",
    ("truecell.mixscape_lda", "de_assay"):
        "Seurat's assay = NULL means DefaultAssay(object); truecell names 'RNA'.",
    ("truecell.map_query", "reference_reduction"):
        "Seurat reads it from the anchor set's command log, which truecell's "
        "TransferAnchors does not keep; 'pca' is the default reduction.",
    ("truecell.map_query", "reduction_model"):
        "Seurat's NULL skips the UMAP projection; truecell projects onto 'umap' "
        "unless told otherwise.",

    # -- Arguments Seurat gives a default and truecell requires.
    ("truecell.find_markers", "ident_1"):
        "Seurat's NULL fails ValidateCellGroups, so it is required in practice in both.",
    ("truecell.run_spca", "graph"):
        "Seurat's NULL stops with 'Graph is not provided'; required in both.",
    ("truecell.mixscape_heatmap", "ident_1"):
        "Seurat's NULL fails inside FindMarkers; required in both.",
    ("truecell.plot_perturb_score", "target_gene_ident"):
        "Seurat's NULL only prints 'Please provide name of target gene class to "
        "plot'; truecell makes it required.",
    ("truecell.percentage_feature_set", "pattern"):
        "Seurat takes a feature list or a pattern; truecell ports only the pattern "
        "form, so the pattern is required.",
    ("truecell.do_heatmap", "features"):
        "Seurat's NULL falls back to VariableFeatures(object); truecell requires "
        "the features.",

    # -- Plotting choices, kept by decision (2026-09-12). ggplot2 sizes points in
    #    millimetres and matplotlib in points squared, so no size compares.
    ("truecell.dim_plot", "pt_size"):
        "Units differ: ggplot2 millimetres from AutoPointSize, matplotlib points squared.",
    ("truecell.dim_plot", "label"):
        "Plotting choice: truecell labels the groups unless told not to.",
    ("truecell.dim_plot", "label_size"):
        "Units differ: None scales with the theme; Seurat's 4 is ggplot2 millimetres.",
    ("truecell.dim_plot", "alpha"):
        "Plotting choice: points at 0.7 opacity show where they overlap.",
    ("truecell.dim_plot", "reduction"):
        "Seurat's NULL takes umap, then tsne, then pca; truecell defaults to 'umap'.",
    ("truecell.feature_plot", "pt_size"):
        "Units differ: ggplot2 millimetres, matplotlib points squared.",
    ("truecell.feature_plot", "order"):
        "Plotting choice: truecell draws the highest-expressing cells last, so they "
        "are not hidden.",
    ("truecell.feature_plot", "reduction"):
        "Seurat's NULL takes umap, then tsne, then pca; truecell defaults to 'umap'.",
    ("truecell.feature_scatter", "pt_size"):
        "Units differ: ggplot2 millimetres, matplotlib points squared.",
    ("truecell.image_dim_plot", "size"):
        "Units differ: ggplot2 millimetres, matplotlib points squared.",
    ("truecell.image_feature_plot", "size"):
        "Units differ: ggplot2 millimetres, matplotlib points squared.",
    ("truecell.spatial_dim_plot", "alpha"):
        "Seurat's c(1, 1) is a range for alpha scaled by value; truecell takes one alpha.",
    ("truecell.spatial_feature_plot", "alpha"):
        "Seurat's c(1, 1) is a range for alpha scaled by value; truecell takes one alpha.",
    ("truecell.variable_feature_plot", "log"):
        "Seurat's NULL uses a log axis only for vst or SCT variance columns; truecell "
        "always does.",
}


# ---------------------------------------------------------------------------
# Reading the two sides
# ---------------------------------------------------------------------------

_MISSING = object()


@dataclass(frozen=True)
class Unparsed:
    """A default this comparison does not evaluate, such as ``1:10`` or ``ncol(x)``."""

    text: str


@dataclass(frozen=True)
class Choice:
    """``c("a", "b")``: an argument resolved by ``match.arg``, whose default is ``"a"``."""

    first: str


def _split_top_level(text: str) -> list[str]:
    """Split ``text`` on the commas that are not inside brackets or quotes."""
    parts, depth, current, quote = [], 0, "", None
    for ch in text:
        if quote:
            current += ch
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            current += ch
        elif ch in "([":
            depth += 1
            current += ch
        elif ch in ")]":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def r_value(text: str | None):
    """A deparsed R default as a Python value, or :class:`Unparsed`."""
    if text is None:
        return _MISSING
    text = text.strip()
    literals = {"TRUE": True, "T": True, "FALSE": False, "F": False, "NULL": None,
                "Inf": math.inf, "-Inf": -math.inf}
    if text in literals:
        return literals[text]
    if (m := re.fullmatch(r'"(.*)"', text)):
        return m.group(1)
    if re.fullmatch(r"-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?L?", text):
        return float(text.rstrip("L"))
    if (m := re.fullmatch(r"(-?[\d.]+)\s*/\s*([\d.]+)", text)):
        return float(m.group(1)) / float(m.group(2))
    if (m := re.fullmatch(r"c\((.*)\)", text)):
        items = [r_value(x) for x in _split_top_level(m.group(1))]
        if items and all(isinstance(x, str) for x in items):
            return Choice(items[0])
        if items and all(isinstance(x, (bool, float)) for x in items):
            return tuple(items)
    return Unparsed(text)


def py_value(default):
    """A Python signature default in the same terms as :func:`r_value`."""
    if default is inspect.Parameter.empty:
        return _MISSING
    if default is None or isinstance(default, (bool, str)):
        return default
    if isinstance(default, (int, float)):
        return float(default)
    if isinstance(default, (tuple, list)) and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in default):
        return tuple(float(x) for x in default)
    return Unparsed(repr(default))


def _same(r, py) -> bool:
    if isinstance(r, Choice):
        r = r.first
    if isinstance(r, float) and isinstance(py, float):
        return r == py or math.isclose(r, py, rel_tol=1e-12, abs_tol=0.0)
    if isinstance(r, tuple) and isinstance(py, tuple):
        return len(r) == len(py) and all(_same(a, b) for a, b in zip(r, py))
    return type(r) is type(py) and r == py


def _python_function(dotted: str):
    module, name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mismatch:
    function: str
    argument: str
    r_default: str
    py_default: str
    kind: str          # "different" | "python requires it"

    @property
    def key(self) -> tuple[str, str]:
        return (self.function, self.argument)

    def __str__(self) -> str:
        return (f"{self.function.split('.')[-1]}({self.argument}): "
                f"Seurat {self.r_default}, truecell {self.py_default} [{self.kind}]")


def load_reference(path: Path = REFERENCE) -> dict:
    return json.loads(Path(path).read_text())


def compare(reference: dict) -> tuple[list[Mismatch], dict[str, int]]:
    """Every shared argument whose defaults differ, plus counts of what was skipped.

    Not a mismatch: an argument only one side has, a default truecell supplies
    where R has none (a call that runs in R already passes it), and a default
    either side spells in a form this does not evaluate.
    """
    formals = reference["functions"]
    mismatches: list[Mismatch] = []
    stats = {"functions": 0, "compared": 0, "unparsed": 0}
    for dotted, specs in FUNCTIONS.items():
        params = inspect.signature(_python_function(dotted)).parameters
        merged: dict[str, str | None] = {}
        for spec in specs:
            declared = formals.get(spec)
            if isinstance(declared, dict):
                for name, text in declared.items():
                    merged.setdefault(name, text)
        stats["functions"] += 1
        renames = {**RENAMES, **FUNCTION_RENAMES.get(dotted, {})}
        target = {name: renames.get(name, name.replace(".", "_").lower()) for name in merged}
        for r_name, text in merged.items():
            if r_name in ("...", "object") or r_name in IGNORED:
                continue
            py_name = target[r_name]
            if py_name not in params:
                continue
            if r_name in renames and any(
                    other not in renames and target[other] == py_name for other in merged):
                # A renamed alias beside the formal it renames, as the deprecated
                # `slot` sits beside `layer`: the real formal holds the default.
                continue
            r, py = r_value(text), py_value(params[py_name].default)
            if isinstance(r, Unparsed) or isinstance(py, Unparsed):
                stats["unparsed"] += 1
                continue
            if r is _MISSING:
                continue
            stats["compared"] += 1
            py_text = repr(params[py_name].default)
            if py is _MISSING:
                mismatches.append(Mismatch(dotted, py_name, text, "(required)", "python requires it"))
            elif not _same(r, py):
                mismatches.append(Mismatch(dotted, py_name, text, py_text, "different"))
    return mismatches, stats


def missing_functions(reference: dict) -> list[str]:
    """R functions the reference could not find, and truecell functions that do not exist."""
    problems = [spec for specs in FUNCTIONS.values() for spec in specs
                if not isinstance(reference["functions"].get(spec), dict)]
    for dotted in FUNCTIONS:
        try:
            _python_function(dotted)
        except (ImportError, AttributeError):
            problems.append(dotted)
    return problems


def unexplained(mismatches: list[Mismatch]) -> list[Mismatch]:
    return [m for m in mismatches if m.key not in EQUIVALENT and m.key not in KNOWN_DIVERGENCES]


def stale(mismatches: list[Mismatch]) -> list[tuple[str, str]]:
    live = {m.key for m in mismatches}
    return sorted(k for k in (*EQUIVALENT, *KNOWN_DIVERGENCES) if k not in live)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def refresh(path: Path = REFERENCE) -> None:
    specs = sorted({spec for specs in FUNCTIONS.values() for spec in specs})
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(specs, fh)
    subprocess.run(["Rscript", str(ROOT / "tools" / "compare_defaults.R"), fh.name, str(path)],
                   check=True)
    Path(fh.name).unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true",
                        help="re-dump Seurat's formals into tests/data (needs R)")
    parser.add_argument("--all", action="store_true",
                        help="also list the explained differences")
    args = parser.parse_args(argv)
    if args.refresh:
        refresh()
    reference = load_reference()
    mismatches, stats = compare(reference)
    print(f"Seurat {reference['seurat']} / SeuratObject {reference['seurat_object']}: "
          f"{stats['functions']} functions, {stats['compared']} defaults compared, "
          f"{stats['unparsed']} not evaluated")
    problems = missing_functions(reference)
    for problem in problems:
        print(f"  NOT FOUND   {problem}")
    for m in mismatches:
        if m.key in EQUIVALENT or m.key in KNOWN_DIVERGENCES:
            if args.all:
                why = EQUIVALENT.get(m.key) or KNOWN_DIVERGENCES[m.key]
                print(f"  explained   {m}\n              {why}")
        else:
            print(f"  UNEXPLAINED {m}")
    for key in stale(mismatches):
        print(f"  STALE       {key[0].split('.')[-1]}({key[1]}) no longer differs; remove its entry")
    bad = len(unexplained(mismatches)) + len(stale(mismatches)) + len(problems)
    print(f"{len(mismatches)} differences, {len(mismatches) - len(unexplained(mismatches))} explained, "
          f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
