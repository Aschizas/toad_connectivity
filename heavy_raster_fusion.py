from osgeo import gdal
import numpy as np

ref_path = r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\etangs_bassins_suitable.tif"
out_path = r"C:\Users\alexis.schizas\Desktop\output_test.tif"
layers = [
    r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\ibn_ch_resampled.tif",
    r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\etangs_bassins_suitable.tif",
    r"C:\Users\alexis.schizas\Documents\Projets QGIS\layers\reproduction_certaine_ch_resampled.tif",
]

TILE_ROWS = 512

ref  = gdal.Open(ref_path)
gt   = ref.GetGeoTransform()
proj = ref.GetProjection()
W, H = ref.RasterXSize, ref.RasterYSize
ref  = None

out  = gdal.GetDriverByName("GTiff").Create(out_path, W, H, 1, gdal.GDT_Byte,
         options=["COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
out.SetGeoTransform(gt)
out.SetProjection(proj)
out.GetRasterBand(1).SetNoDataValue(0)

srcs = [gdal.Open(p) for p in layers]

for row_off in range(0, H, TILE_ROWS):
    tile_h = min(TILE_ROWS, H - row_off)
    result = np.zeros((tile_h, W), dtype=np.uint8)

    for src in srcs:
        tile    = src.GetRasterBand(1).ReadAsArray(0, row_off, W, tile_h)
        result |= (tile > 0).astype(np.uint8)

    out.GetRasterBand(1).WriteArray(result, 0, row_off)
    print(f"  row {row_off + tile_h} / {H}", end="\r")

out.FlushCache()
out = None
srcs = None
print(f"\nSaved to {out_path}")