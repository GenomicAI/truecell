# AnnData, Scanpy and SpatialData

truecell converts to and from [AnnData](https://anndata.readthedocs.io), so an object
can move into Scanpy, Squidpy or SpatialData and back. The two functions live on
`truecell.compat.anndata`, not on the top-level package, and need the `anndata` extra.

```bash
pip install "truecell[anndata]"
```

```python
from truecell.compat.anndata import as_anndata, from_anndata

adata = as_anndata(obj)                    # Truecell → AnnData
obj = from_anndata(adata, assay="RNA")     # AnnData → Truecell
```

truecell stores features × cells, as Seurat does, and AnnData stores cells × features.
The conversion transposes for you, so don't transpose again by hand.

## What goes where

| truecell | AnnData |
|---|---|
| the assay's default layer | `X` |
| its other layers | `layers` |
| `meta_data`, and the active identity | `obs`, and `obs["ident"]` |
| the assay's feature metadata | `var` |
| a reduction's embeddings and loadings | `obsm["X_<name>"]` and `varm["<NAME>s"]` |
| `graphs` | `obsp` |
| `misc` | `uns` |
| each cell's position in `images` | `obsm["spatial"]` |
| the image each cell was placed from | `obs["fov"]` |
| a `VisiumV2`'s tissue image and scale factors | `uns["spatial"][image]` |

`from_anndata` reverses the table, with `X` as the counts layer. Pass the assay name
on the way back, because it is not read from `uns`.

## Space

The spatial fields follow the layout Scanpy and Squidpy read.

- **`obsm["spatial"]`** holds one (x, y) per cell, row for row with `obs_names`, from
  the first image that places the cell. The point comes from the image's `Centroids`.
  An image with only a `Segmentation` gives each cell the centroid of its ring, the
  point SeuratObject's `as(segmentation, "Centroids")` gives, which truecell has as
  `Segmentation.as_centroids()`. A cell that no image places is NaN.
- **`obs["fov"]`** is a categorical naming each cell's image, in the order of
  `obj.images`. When `meta_data` already has an FOV column that names the images, as
  CosMx and MERSCOPE objects do, it is written the same way.
- **`uns["spatial"][image]`** holds a Visium image's `images`, under the resolution it
  was loaded at, and its `scalefactors`, under `scalefactors_json.json`'s keys.

Visium spot positions stay in full-resolution pixels, column first. On 10x's V1
mouse-brain slide `as_anndata(load_visium(...))` and `scanpy.read_visium` agree for all
2,695 spots: the same `obsm["spatial"]` to the bit and in dtype, the same scale factors
and the same lowres image. Scanpy's spatial plots therefore work unchanged:

```python
import scanpy as sc
import truecell

adata = as_anndata(truecell.load_visium(path))
sc.pl.spatial(adata, img_key="lowres", color="Hpca")
```

The other direction works too. `from_anndata` turns a file `scanpy.read_visium` wrote
into a `VisiumV2` named after its library, and on that slide its coordinates, image and
scale factors match `load_visium`'s. When the file has both images, `image_resolution=`
chooses which one to keep.

```python
obj = from_anndata(sc.read_visium(path), assay="Spatial")
truecell.normalize_data(obj)
truecell.spatial_feature_plot(obj, feature="Hpca")
```

AnnData has nowhere to put cell polygons or transcript molecules, so those stay
behind. So does a second image of cells another image already placed, such as a crop,
and a radius other than the automatic one for the cells placed, which comes back as
the automatic one. Everything else survives: objects from `load_xenium`, `load_visium`,
`load_cosmx` and `load_merscope` return from an `.h5ad` file with their coordinates
bit-identical, cells and images in order, and Visium's image and scale factors
unchanged.

CosMx and MERSCOPE objects need one extra argument. Their loaders build one image, as
Seurat's do, and keep the platform's own `fov` column in `meta_data`, which does not
name that image. `as_anndata` keeps the column under `obs["fov"]` and warns, so pass
`fov_key="image"`, or any free name, to both calls:

```python
adata = as_anndata(truecell.load_cosmx(path), fov_key="image")
obj = from_anndata(adata, assay="Nanostring", fov_key="image")
```

## SpatialData

SpatialData keeps shapes and images as elements, with the expression in a table that
annotates one of them. truecell has no `to_spatialdata`, which would pull spatialdata's
dependency stack into the package. The recipes below build the same objects from an
`as_anndata` result. They were run with spatialdata 0.8.0 on the Xenium mouse-brain
section (36,602 cells) and the Visium slide above, written to zarr and read back. The
coordinates, polygons, expression and tissue image all came back identical.

### Cells as circles

```python
import numpy as np
import pandas as pd
import spatialdata as sd
from spatialdata.models import ShapesModel, TableModel

from truecell.compat.anndata import as_anndata

adata = as_anndata(obj)
adata = adata[~np.isnan(adata.obsm["spatial"]).any(axis=1)].copy()   # placed cells only
radius = next(iter(obj.images.values())).boundaries["centroids"].radius()
cells = ShapesModel.parse(adata.obsm["spatial"], geometry=0,
                          radius=np.full(adata.n_obs, radius),
                          index=adata.obs_names.to_numpy())
adata.obs["region"] = pd.Categorical(["cells"] * adata.n_obs)
adata.obs["instance_id"] = adata.obs_names.to_numpy()
table = TableModel.parse(adata, region="cells", region_key="region",
                         instance_key="instance_id")
sdata = sd.SpatialData(shapes={"cells": cells}, tables={"table": table})
sdata.write("section.zarr")
```

With several images, the cells of all of them go into the one shapes element, and
`obs["fov"]` still says which image each cell came from.

### Cell polygons

```python
import geopandas as gpd
import shapely

segmentation = obj.images["fov"].boundaries["segmentation"]
rings = segmentation.get_tissue_coordinates().reset_index(drop=True)
polygons = gpd.GeoDataFrame(
    geometry=[shapely.Polygon(ring[["x", "y"]].to_numpy())
              for _, ring in rings.groupby("cell", sort=False)],
    index=segmentation.cells(),
)
boundaries = ShapesModel.parse(polygons)
```

`boundaries` goes into `shapes=` next to the circles.

### Visium spots over the tissue image

```python
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

adata = as_anndata(obj)
slide = obj.images["slice1"]
f = slide.scale_factor()           # full-resolution pixels → pixels of the stored image
tissue = Image2DModel.parse(
    np.moveaxis(slide.get_image(), -1, 0), dims=("c", "y", "x"),
    transformations={"global": Scale([1 / f, 1 / f], axes=("y", "x"))},
)
spots = ShapesModel.parse(adata.obsm["spatial"], geometry=0,
                          radius=np.full(adata.n_obs, slide.radius()),
                          index=adata.obs_names.to_numpy())
adata.obs["region"] = pd.Categorical(["spots"] * adata.n_obs)
adata.obs["instance_id"] = adata.obs_names.to_numpy()
table = TableModel.parse(adata, region="spots", region_key="region",
                         instance_key="instance_id")
sdata = sd.SpatialData(images={"tissue": tissue}, shapes={"spots": spots},
                       tables={"table": table})
```

The spots stay in full-resolution pixels and the `Scale` moves the image into the
same space. On the slide above, every spot lands exactly where
`slide.scale_coordinates()` puts it on the image. `slide.radius()` is half of
`spot_diameter_fullres`, a true radius where Seurat stores the diameter; the
[Spatial](api/spatial.md) reference explains why.
