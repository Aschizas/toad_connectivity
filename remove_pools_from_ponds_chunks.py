import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize, shapes
import geopandas as gpd
from scipy import ndimage
from shapely.geometry import shape

ponds_path     = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\eauxcalmes_etangs_ch_resized_suitable.tif"
pools_path     = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\piscines_swissTLM3D.shp"
out_path       = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\output.tif"
buildings_path = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\zones_baties_ch.tif"

TILE_ROWS    = 2000
THRESHOLD_M  = 50

# ── filter_pools: vectorize buildings in chunks ───────────────────────────────
def filter_pools():
    print("Vectorizing buildings in chunks...")
    geoms = []

    with rasterio.open(buildings_path) as src:
        crs       = src.crs
        H, W      = src.height, src.width
        transform = src.transform

        for row_off in range(0, H, TILE_ROWS):
            tile_h = min(TILE_ROWS, H - row_off)
            win    = Window(0, row_off, W, tile_h)

            tile      = src.read(1, window=win, out_dtype='uint8')
            tile      = np.where(np.isnan(tile.astype(float)), 0, tile).astype(np.uint8)
            win_transform = src.window_transform(win)

            for geom, val in shapes(tile, transform=win_transform):
                if val == 1:
                    geoms.append(shape(geom))

            print(f"  row {row_off + tile_h} / {H}", end="\r")

    print(f"\nExtracted {len(geoms)} building polygons")
    buildings_vec = gpd.GeoDataFrame(geometry=geoms, crs=crs)

    all_basins = gpd.read_file(pools_path).to_crs(crs)
    basins     = all_basins[all_basins["OBJEKTART"] == 2]

    buildings_buffered          = buildings_vec.copy()
    buildings_buffered["geometry"] = buildings_vec.buffer(THRESHOLD_M)

    joined = gpd.sjoin(basins, buildings_buffered[["geometry"]], how="left", predicate="intersects")
    joined = joined[~joined.index.duplicated(keep="first")]
    basins = basins.copy()
    basins["label"] = np.where(joined["index_right"].notna(), "pool", "pond")

    ponds = basins[basins["label"] == "pond"]
    ponds.to_file(r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\ponds.gpkg", driver="GPKG")

    all_basins_without_ponds = all_basins[~all_basins.index.isin(ponds.index)]
    print(f"{len(all_basins)} total basins, {len(all_basins_without_ponds)} after removing ponds")
    return all_basins_without_ponds


# ── remove_pools: label + mask in chunks ─────────────────────────────────────
def remove_pools(pools):
    print("Removing pools from pond raster (chunked)...")

    with rasterio.open(ponds_path) as src:
        profile   = src.profile.copy()
        transform = src.transform
        H, W      = src.height, src.width

    pools_buffered          = pools.copy()
    pools_buffered["geometry"] = pools.buffer(3)

    profile.update(dtype=rasterio.uint8, nodata=0)
    total_removed = 0

    with rasterio.open(ponds_path) as pond_src, \
         rasterio.open(out_path, 'w', **profile) as dst:

        for row_off in range(0, H, TILE_ROWS):
            tile_h = min(TILE_ROWS, H - row_off)
            win    = Window(0, row_off, W, tile_h)
            win_transform = pond_src.window_transform(win)

            ponds_tile = pond_src.read(1, window=win).astype(bool)

            pools_mask = rasterize(
                [(geom, 1) for geom in pools_buffered.geometry],
                out_shape=ponds_tile.shape,
                transform=win_transform,
                fill=0,
                dtype=np.uint8
            ).astype(bool)

            labeled, _ = ndimage.label(ponds_tile)
            hit_labels = np.unique(labeled[pools_mask])
            hit_labels = hit_labels[hit_labels > 0]

            result = ponds_tile.copy()
            result[np.isin(labeled, hit_labels)] = False

            total_removed += len(hit_labels)
            dst.write(result.astype(np.uint8)[np.newaxis, ...], window=win)
            print(f"  row {row_off + tile_h} / {H}  — {total_removed} blobs removed so far", end="\r")

    print(f"\nDone. Removed {total_removed} pond blobs overlapping pools.")


if __name__ == "__main__":
    print("Sorting pools versus artificial ponds...")
    pools = filter_pools()

    print("Removing pools from input file...")
    remove_pools(pools)