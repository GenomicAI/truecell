"""Write the synthetic spatial bundles that `tests/test_loaders_vs_seurat.py` reads.

Each bundle is laid out the way the platform's own software writes it, small
enough to check by eye, with the cases Seurat's readers treat specially planted
in it. `make_loader_r_reference.R` loads the same files with `LoadNanostring`,
`LoadVizgen` and `LoadXenium` and records what R built, so the test compares
truecell's loaders with Seurat's without needing R in CI.

    python tests/data/make_loader_bundles.py        # needs pyarrow and h5py
    Rscript tests/data/make_loader_r_reference.R

These are not real runs. They check that truecell reads each layout the way
Seurat does: cell and feature names, which rows count as cells, coordinates, and
the image it builds. Whether a platform's current software still writes that
layout is a question the bundles cannot answer.

Coordinates are multiples of 1/8, so R's and pandas' CSV readers both parse them
exactly and the comparison can be exact too.
"""
from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent / "loaders"


def eighths(rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
    return np.round(rng.uniform(low, high, n) * 8) / 8


def write_gzip(path: Path, text: str) -> None:
    """Gzip with no timestamp in the header, so a rerun writes the same bytes."""
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                               mtime=0) as fh:
        fh.write(text.encode())


# ---------------------------------------------------------------------------
# NanoString CosMx
# ---------------------------------------------------------------------------

def cosmx(root: Path) -> None:
    """exprMat, metadata, polygons and transcripts CSVs, as AtoMx exports them.

    Planted:

    * the `cell_ID` 0 row each FOV carries, for transcripts outside any cell;
    * a cell with no counts at all;
    * a cell in the expression matrix with no metadata row, and a metadata row
      with no expression;
    * cell ID 12 in FOV 2, so a name read the wrong way round cannot pass;
    * a gene with an underscore, and a negative probe counted only outside cells.
    """
    rng = np.random.default_rng(20260915)
    genes = ["CD3D", "MS4A1", "LYZ", "HLA_DRA", "NegPrb1", "NegPrb2"]
    ids = [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 12),
           (3, 0), (3, 1), (3, 2), (3, 3)]
    expr = pd.DataFrame(ids, columns=["fov", "cell_ID"])
    counts = rng.poisson(4.0, size=(len(ids), len(genes))) + 1
    background = (expr["cell_ID"] == 0).to_numpy()
    counts[~background, genes.index("NegPrb2")] = 0
    empty = ((expr["fov"] == 1) & (expr["cell_ID"] == 3)).to_numpy()
    counts[empty] = 0
    for j, gene in enumerate(genes):
        expr[gene] = counts[:, j]
    expr.to_csv(root / "Run1_exprMat_file.csv", index=False)

    no_metadata = ((expr["fov"] == 3) & (expr["cell_ID"] == 3)).to_numpy()
    cells = pd.concat([expr.loc[~background & ~no_metadata, ["fov", "cell_ID"]],
                       pd.DataFrame({"fov": [2], "cell_ID": [13]})])
    cells = cells.sort_values(["fov", "cell_ID"]).reset_index(drop=True)
    n = len(cells)
    offset_x = cells["fov"].map({1: 0.0, 2: 5472.0, 3: 0.0}).to_numpy()
    offset_y = cells["fov"].map({1: 0.0, 2: 0.0, 3: 3648.0}).to_numpy()
    local_x, local_y = eighths(rng, n, 100, 5000), eighths(rng, n, 100, 3500)
    meta = cells.assign(
        Area=rng.integers(800, 4000, n), Width=rng.integers(20, 90, n),
        Height=rng.integers(20, 90, n),
        CenterX_local_px=local_x, CenterY_local_px=local_y,
        CenterX_global_px=local_x + offset_x, CenterY_global_px=local_y + offset_y)
    meta.to_csv(root / "Run1_metadata_file.csv", index=False)

    vertices = []
    for rec in meta.itertuples(index=False):
        for dx, dy in ((-20, -20), (20, -20), (20, 20), (-20, 20)):
            vertices.append((rec.fov, rec.cell_ID, rec.CenterX_local_px + dx,
                             rec.CenterY_local_px + dy, rec.CenterX_global_px + dx,
                             rec.CenterY_global_px + dy))
    pd.DataFrame(vertices, columns=["fov", "cellID", "x_local_px", "y_local_px",
                                    "x_global_px", "y_global_px"]).to_csv(
        root / "Run1-polygons.csv", index=False)

    molecules = []
    for rec in expr.itertuples(index=False):
        for k in range(2):
            x, y = eighths(rng, 1, 100, 5000)[0], eighths(rng, 1, 100, 3500)[0]
            molecules.append((rec.fov, rec.cell_ID, f"c_1_{rec.fov}_{rec.cell_ID}", x, y,
                              x, y, k, genes[(rec.cell_ID + k) % 4],
                              "None" if rec.cell_ID == 0 else "Nuclear"))
    pd.DataFrame(molecules, columns=["fov", "cell_ID", "cell", "x_local_px", "y_local_px",
                                     "x_global_px", "y_global_px", "z", "target",
                                     "CellComp"]).to_csv(root / "Run1_tx_file.csv", index=False)


# ---------------------------------------------------------------------------
# Vizgen MERSCOPE
# ---------------------------------------------------------------------------

def merscope(root: Path) -> None:
    """cell_by_gene.csv, cell_metadata.csv and HDF5 cell boundaries.

    Planted:

    * `Blank-` controls, and genes starting `blank-` or `Blank` with no hyphen,
      which `LoadVizgen`'s case-sensitive `^Blank-` filter keeps;
    * the metadata's unnamed leading id column, and 13-digit cell ids;
    * a cell with no counts, which `LoadVizgen` keeps;
    * a cell with no metadata row, and a metadata row with no counts;
    * a gene with an underscore.
    """
    rng = np.random.default_rng(20260916)
    genes = ["Gad1", "Sst", "Slc17a7", "Pvalb_2", "Blank-1", "Blank-2", "blank-3", "BlankLike"]
    cells = ["3141592653589", "2718281828459", "1618033988749", "1414213562373",
             "1732050807568", "2236067977499", "2645751311064"]
    counts = rng.poisson(3.0, size=(len(cells), len(genes))) + 1
    counts[3] = 0
    by_gene = pd.DataFrame(counts, columns=genes)
    by_gene.insert(0, "cell", cells)
    by_gene.to_csv(root / "cell_by_gene.csv", index=False)

    in_metadata = cells[:-1] + ["1123581321345"]
    fov = [0, 0, 1, 1, 1, 2, 2]
    n = len(in_metadata)
    center_x, center_y = eighths(rng, n, 1000, 9000), eighths(rng, n, 1000, 9000)
    meta = pd.DataFrame(
        {"fov": fov, "volume": eighths(rng, n, 200, 900), "center_x": center_x,
         "center_y": center_y, "min_x": center_x - 5, "max_x": center_x + 5,
         "min_y": center_y - 5, "max_y": center_y + 5},
        index=pd.Index(in_metadata))
    meta.to_csv(root / "cell_metadata.csv")

    boundaries = root / "cell_boundaries"
    boundaries.mkdir()
    for f in sorted(set(fov)):
        with h5py.File(boundaries / f"feature_data_{f}.hdf5", "w") as h5:
            for cell, x, y, cell_fov in zip(in_metadata, center_x, center_y, fov):
                if cell_fov != f:
                    continue
                ring = np.array([[x - 5, y - 5], [x + 5, y - 5], [x + 5, y + 5],
                                 [x - 5, y + 5], [x - 5, y - 5]])
                h5.create_dataset(f"featuredata/{cell}/zIndex_3/p_0/coordinates", data=ring)


# ---------------------------------------------------------------------------
# 10x Xenium
# ---------------------------------------------------------------------------

#: Xenium Onboard Analysis 1.x feature types, and those of 2.0 and later.
XOA1_FEATURES = [
    ("ENSMUSG00000035783", "Acta2", "Gene Expression"),
    ("ENSMUSG00000070570", "Slc17a7", "Gene Expression"),
    ("ENSMUSG00000070880", "Gad1", "Gene Expression"),
    ("ENSMUSG00000026581", "Sell_v2", "Gene Expression"),
    ("NegControlProbe_00004", "NegControlProbe_00004", "Negative Control Probe"),
    ("NegControlCodeword_0500", "NegControlCodeword_0500", "Negative Control Codeword"),
    ("BLANK_0006", "BLANK_0006", "Blank Codeword"),
]
XOA2_FEATURES = XOA1_FEATURES[:6] + [
    ("GenomicControl_00001", "GenomicControl_00001", "Genomic Control"),
    ("UnassignedCodeword_0001", "UnassignedCodeword_0001", "Unassigned Codeword"),
    ("DeprecatedCodeword_0317", "DeprecatedCodeword_0317", "Deprecated Codeword"),
]


def xenium(root: Path, cells_format: str) -> None:
    """`cell_feature_matrix/` and a cells table, with no molecules.

    ``cells_format`` is ``"csv"`` (Xenium Onboard Analysis 1.x: integer cell ids
    in `cells.csv.gz`), ``"parquet"`` (XOA 2.0 and later: string ids and a
    `segmentation_method` column in `cells.parquet`), or ``"parquet_binary"``,
    the same with `cell_id` stored as bytes, which Seurat's `ReadXenium` decodes.
    Only one cells table is written, so neither reader can fall back to the other.
    Planted: every feature type the version writes, and a gene with an underscore.
    """
    rng = np.random.default_rng(20260917)
    xoa1 = cells_format == "csv"
    features = XOA1_FEATURES if xoa1 else XOA2_FEATURES
    ids = ([str(i) for i in range(1, 7)] if xoa1 else
           [f"{p}-1" for p in ("aaabinlp", "aaacaemi", "aaadbhjo", "aaagnpkm",
                               "aaahmhhj", "aaaiccfk")])
    counts = rng.poisson(2.0, size=(len(features), len(ids)))

    matrix = root / "cell_feature_matrix"
    matrix.mkdir()
    write_gzip(matrix / "barcodes.tsv.gz", "".join(f"{c}\n" for c in ids))
    write_gzip(matrix / "features.tsv.gz", "".join(f"{i}\t{n}\t{t}\n" for i, n, t in features))
    entries = [f"{i + 1} {j + 1} {counts[i, j]}" for j in range(counts.shape[1])
               for i in range(counts.shape[0]) if counts[i, j]]
    write_gzip(matrix / "matrix.mtx.gz",
               "%%MatrixMarket matrix coordinate integer general\n"
               f"{counts.shape[0]} {counts.shape[1]} {len(entries)}\n"
               + "".join(f"{e}\n" for e in entries))

    def total(kind: str) -> np.ndarray:
        rows = [k for k, (_, _, t) in enumerate(features) if t == kind]
        return counts[rows].sum(axis=0)

    n = len(ids)
    table = {"x_centroid": eighths(rng, n, 500, 7000), "y_centroid": eighths(rng, n, 500, 5000),
             "transcript_counts": total("Gene Expression"),
             "control_probe_counts": total("Negative Control Probe")}
    if xoa1:
        table["control_codeword_counts"] = (total("Negative Control Codeword")
                                            + total("Blank Codeword"))
    else:
        table["genomic_control_counts"] = total("Genomic Control")
        table["control_codeword_counts"] = total("Negative Control Codeword")
        table["unassigned_codeword_counts"] = total("Unassigned Codeword")
        table["deprecated_codeword_counts"] = total("Deprecated Codeword")
    table["total_counts"] = counts.sum(axis=0)
    table["cell_area"] = eighths(rng, n, 40, 400)
    table["nucleus_area"] = eighths(rng, n, 10, 40)
    if xoa1:
        frame = pd.DataFrame({"cell_id": [int(c) for c in ids], **table})
        write_gzip(root / "cells.csv.gz", frame.to_csv(index=False))
        return
    table["nucleus_count"] = rng.integers(0, 3, n)
    table["segmentation_method"] = ["Boundary Stain", "Interior Stain",
                                    "Imaging Based Nuclear Expansion"] * 2
    cell_id = (pa.array([c.encode() for c in ids], type=pa.binary())
               if cells_format == "parquet_binary" else pa.array(ids, type=pa.string()))
    pq.write_table(pa.table({"cell_id": cell_id, **table}), root / "cells.parquet")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    writers = {
        "cosmx": cosmx,
        "merscope": merscope,
        "xenium_csv": lambda root: xenium(root, "csv"),
        "xenium_parquet": lambda root: xenium(root, "parquet"),
        "xenium_parquet_binary": lambda root: xenium(root, "parquet_binary"),
    }
    for name, write in writers.items():
        root = OUT / name
        root.mkdir(parents=True)
        write(root)
        print(f"wrote {root}")


if __name__ == "__main__":
    main()
