import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

PONDS_PATH    = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\eauxcalmes_etangs_ch_resized.tif"
FORESTS_PATH  = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\forests_ch.tif"

MAX_POND_SIZE_M   = 10000
MAX_FOREST_DIST_M = 2200
TILE_ROWS         = 512   # reduce further if still OOM

# ── metadata ──────────────────────────────────────────────────────────────────
with rasterio.open(PONDS_PATH) as src:
    profile    = src.profile.copy()
    pixel_size = abs(src.transform.a)
    H, W       = src.height, src.width
    nodata_p   = src.nodata

with rasterio.open(FORESTS_PATH) as src:
    nodata_f = src.nodata

max_pond_px   = MAX_POND_SIZE_M   / pixel_size
max_forest_px = MAX_FOREST_DIST_M / pixel_size
overlap       = int(np.ceil(max_forest_px))

profile.update(dtype=rasterio.uint8, nodata=0)
out_path = PONDS_PATH.replace(".tif", "_suitable.tif")

# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — per-tile labeling + size + forest proximity
#   Problem: ponds that cross tile boundaries get split into two labels.
#   For Bufo bufo ponds (small water bodies) this is acceptable since any
#   pond large enough to span TILE_ROWS pixels would exceed MAX_POND_SIZE_M.
# ─────────────────────────────────────────────────────────────────────────────
print("Pass 1: labeling + filtering tiles …")

# We write a temporary label raster (int32) tile by tile,
# and collect which labels survive both filters.
import tempfile, os
tmp_label_path = out_path.replace("_suitable.tif", "_tmp_labels.tif")
label_profile  = profile.copy()
label_profile.update(dtype=rasterio.int32, nodata=0)

global_label_offset = 0   # ensures labels are unique across tiles
surviving_labels    = set()

with rasterio.open(tmp_label_path, 'w', **label_profile) as lbl_dst, \
     rasterio.open(PONDS_PATH)   as pond_src, \
     rasterio.open(FORESTS_PATH) as forest_src:

    for row_off in range(0, H, TILE_ROWS):
        tile_h = min(TILE_ROWS, H - row_off)

        # --- read pond tile (no overlap needed for labeling) -----------------
        win       = Window(0, row_off, W, tile_h)
        ponds_raw = pond_src.read(1, window=win, out_dtype='uint8')
        valid     = (ponds_raw != nodata_p) if nodata_p is not None \
                    else np.ones(ponds_raw.shape, dtype=bool)
        ponds     = valid & ponds_raw.astype(bool)
        del ponds_raw, valid

        # --- label this tile -------------------------------------------------
        labeled, n = ndimage.label(ponds)          # int32 on small tile = fine
        labeled    = labeled.astype(np.int32)

        if n > 0:
            labeled[labeled > 0] += global_label_offset

            # size filter (tile-local, fast)
            counts = np.bincount(labeled.ravel())   # small tile, no OOM risk
            size_ok = set(
                np.where(counts[global_label_offset + 1:
                                global_label_offset + n + 1] <= max_pond_px)[0]
                + global_label_offset + 1
            )

            # forest proximity (expanded window for EDT accuracy) -------------
            r0 = max(0, row_off - overlap)
            r1 = min(H, row_off + tile_h + overlap)
            exp_win     = Window(0, r0, W, r1 - r0)
            forests_raw = forest_src.read(1, window=exp_win, out_dtype='uint8')
            fvalid      = (forests_raw != nodata_f) if nodata_f is not None \
                          else np.ones(forests_raw.shape, dtype=bool)
            forests     = fvalid & forests_raw.astype(bool)
            del forests_raw, fvalid

            dist        = ndimage.distance_transform_edt(~forests)
            del forests

            # crop dist back to the real tile rows
            real_start  = row_off - r0
            near_forest = dist[real_start: real_start + tile_h] <= max_forest_px
            del dist

            near_labels = set(np.unique(labeled[near_forest]).tolist())
            near_labels.discard(0)
            del near_forest

            surviving_labels |= (size_ok & near_labels)
            global_label_offset += n

        # write label tile (0 where no pond)
        lbl_dst.write(labeled[np.newaxis, ...], window=win)
        del labeled, ponds

        print(f"  row {row_off + tile_h} / {H}  —  {len(surviving_labels)} surviving so far", end="\r")

print(f"\nFound {global_label_offset} total pond components, "
      f"{len(surviving_labels)} survive both filters")

# ─────────────────────────────────────────────────────────────────────────────
# PASS 2 — read label raster tile by tile, write output
# ─────────────────────────────────────────────────────────────────────────────
print("Pass 2: writing output …")

surviving_arr = np.array(list(surviving_labels), dtype=np.int32)

with rasterio.open(tmp_label_path) as lbl_src, \
     rasterio.open(out_path, 'w', **profile) as out_dst:

    for row_off in range(0, H, TILE_ROWS):
        tile_h = min(TILE_ROWS, H - row_off)
        win    = Window(0, row_off, W, tile_h)

        labeled = lbl_src.read(1, window=win)
        result  = np.isin(labeled, surviving_arr).astype(np.uint8)
        out_dst.write(result[np.newaxis, ...], window=win)

        print(f"  row {row_off + tile_h} / {H}", end="\r")

os.remove(tmp_label_path)
print(f"\nSaved to: {out_path}")