import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
import geopandas as gpd
from scipy import ndimage
from shapely.geometry import shape

ponds_path = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\eauxcalmes_etangs_ch_resized_suitable.tif"
pools_path = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\piscines_swissTLM3D.shp"
out_path   = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\output.tif"
buildings_path = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\zones_baties_ch.tif"


def filter_pools():
    """
    Method to only keep basins meeting certain criteria in order to try and keep man-made ponds
    and discriminate swimming pools and retention basins
    """
    with rasterio.open(buildings_path) as src:
        buildings = np.nan_to_num(src.read(1), nan=0).astype(np.uint8)
        transform = src.transform
        crs = src.crs

    geoms = [shape(s) for s, v in shapes(buildings, transform=transform) if v == 1]
    buildings_vec = gpd.GeoDataFrame(geometry=geoms, crs=crs)

    all_basins = gpd.read_file(pools_path).to_crs(crs)
    basins = all_basins[all_basins["OBJEKTART"] == 2]

    THRESHOLD_M = 50
    buildings_buffered = buildings_vec.copy()
    buildings_buffered["geometry"] = buildings_vec.buffer(THRESHOLD_M)

    joined = gpd.sjoin(basins, buildings_buffered[["geometry"]], how="left", predicate="intersects")
    joined = joined[~joined.index.duplicated(keep="first")]
    basins["label"] = np.where(joined["index_right"].notna(), "pool", "pond")

    pools = basins[basins["label"] == "pool"]
    ponds = basins[basins["label"] == "pond"]

    ponds.to_file(r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\ponds.gpkg", driver="GPKG")

    all_basins_without_ponds = all_basins[~all_basins.index.isin(ponds.index)]
    print(len(all_basins), len(all_basins_without_ponds))
    return all_basins_without_ponds

def remove_pools(pools):

    with rasterio.open(ponds_path) as src:
        ponds = src.read(1).astype(bool)
        profile = src.profile
        transform = src.transform

    pools['geometry'] = pools.buffer(3)

    # Rasterize pools onto the pond grid
    pools_mask = rasterize(
        [(geom, 1) for geom in pools.geometry],
        out_shape=ponds.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    ).astype(bool)

    # Label pond blobs, then remove any blob that touches a pool pixel
    labeled, _ = ndimage.label(ponds)
    hit_labels = np.unique(labeled[pools_mask])
    hit_labels = hit_labels[hit_labels > 0]          # drop background (0)

    result = ponds.copy()
    result[np.isin(labeled, hit_labels)] = False      # wipe entire blobs

    profile.update(dtype=rasterio.uint8, nodata=0)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(result.astype(np.uint8), 1)

    print(f"Removed {len(hit_labels)} pond blobs overlapping pools.")

if __name__ == "__main__":

    print("sorting pools versus artificial ponds...")
    pools = filter_pools()

    
    # pools = gpd.read_file(pools_path).to_crs(src.crs)
    print("removing pools from input file...")
    remove_pools(pools)
    