import numpy as np
import matplotlib.pyplot as plt
import datasets as ds
import glob
import warnings

warnings.filterwarnings('ignore')

map_slope = ds.Slope(name="Slope")
print(map_slope)
map_slope.load_asc_raster(file="../samples/map_slope.asc")
map_slope.view(show=True)



'''
series_lulc = ds.LULCSeries(name="Potiribu_LULC")
series_lulc.load_folder(
    folder="../samples/lulc",
    name_pattern="map_lulc_*",
    table_file="../samples/table_lulc.txt",
    talk=True
)

#series_lulc.view_series_areas(show=True)

series_lulcc = series_lulc.get_lulcc_series(by_lulc_id=24)
print(series_lulcc.catalog.to_string())

series_lulcc.export_views(
    show=False,
    folder="C:/output",
    specs={"legend_ncol": 1, "legend_x": 0.4}
)

'''