import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

INPUT  = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\HabitatMap_v1_1_50_TypoCH_resistancemap.tif"
OUTPUT = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\typoCH_resistancemap_filled.tif"

CHUNK    = 2048
OVERLAP  = 512
NODATA   = -9999
FILL_VAL = -1

def fill_chunk(data):
    """Replace -1 pixels with nearest pixel that is not -1 and not -9999."""
    valid    = (data != NODATA) & (data != FILL_VAL)  # trusted source: 0–100
    to_fill  = data == FILL_VAL

    if not to_fill.any():
        return data

    _, idx = distance_transform_edt(~valid, return_indices=True)
    filled = data.copy()
    filled[to_fill] = data[idx[0][to_fill], idx[1][to_fill]]
    return filled

with rasterio.open(INPUT) as src:
    height, width = src.height, src.width

    profile = src.profile.copy()
    profile.update(
        dtype=rasterio.int16,
        nodata=NODATA,
        compress='lzw',
        predictor=2,
        bigtiff='YES',
    )

    print(f"Raster: {width}×{height} px  |  chunks of {CHUNK} rows  |  overlap {OVERLAP} px")

    with rasterio.open(OUTPUT, 'w', **profile) as dst:
        for row_start in range(0, height, CHUNK):
            row_end = min(row_start + CHUNK, height)

            read_start = max(0, row_start - OVERLAP)
            read_end   = min(height, row_end + OVERLAP)

            window = Window(0, read_start, width, read_end - read_start)
            data = src.read(1, window=window).astype(np.int16)

            filled = fill_chunk(data)

            local_start = row_start - read_start
            local_end   = local_start + (row_end - row_start)
            chunk_out   = filled[local_start:local_end, :]

            write_window = Window(0, row_start, width, row_end - row_start)
            dst.write(chunk_out, 1, window=write_window)

            pct = 100 * row_end / height
            n_filled = int((data[local_start:local_end] == FILL_VAL).sum())
            print(f"  [{pct:5.1f}%]  rows {row_start:6d}–{row_end:6d}  |  -1 px filled: {n_filled}", flush=True)

print("Done.")