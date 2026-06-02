import csv
import numpy as np
import rasterio
from tkinter import filedialog, Tk

# --- Hardcoded column names ---
FROM_COL = "value"
TO_COL   = "new_raster_value"
NODATA   = -9999

# --- File pickers ---
root = Tk(); root.withdraw()
csv_path    = filedialog.askopenfilename(title="Select CSV",    filetypes=[("CSV", "*.csv")])
raster_path = filedialog.askopenfilename(title="Select Raster", filetypes=[("GeoTIFF", "*.tif *.tiff")])
root.destroy()

# --- Load mapping ---
mapping = {}
with open(csv_path, newline='', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f, delimiter=';'):
        try:
            mapping[int(row[FROM_COL].strip())] = int(row[TO_COL].strip())
        except (ValueError, KeyError):
            pass
print(mapping)
print(f"Loaded {len(mapping)} mappings.")

# --- Read raster metadata ---
with rasterio.open(raster_path) as src:
    profile   = src.profile
    nodata_in = src.nodata

profile.update(dtype=rasterio.int16, nodata=NODATA, compress='deflate', predictor=2)

# --- Process in chunks ---
out_path = raster_path.replace(".tif", "_reclassified.tif")
with rasterio.open(raster_path) as src:
    with rasterio.open(out_path, 'w', **profile) as dst:
        windows = list(src.block_windows(1))
        total   = len(windows)
        for i, (_, window) in enumerate(windows):
            data   = src.read(1, window=window)
            result = np.full(data.shape, NODATA, dtype=np.int16)
            for old, new in mapping.items():
                result[data == np.uint16(old)] = new
            if nodata_in is not None:
                result[data == np.uint16(nodata_in)] = NODATA
            dst.write(result, 1, window=window)
            print(f"  {i+1}/{total} blocks", end='\r')

print(f"\nDone! Saved to: {out_path}")