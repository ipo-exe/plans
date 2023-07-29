import numpy as np
import matplotlib.pyplot as plt
import datasets as ds
import glob
import warnings

warnings.filterwarnings('ignore')

series_lulc = ds.LULCSeries(name="Potiribu")
series_lulc.load_folder(
    folder="../samples/lulc",
    name_pattern="map_lulc_*",
    table_file="../samples/table_lulc.txt",
    talk=True
)
print(series_lulc.catalog.to_string())
print()
print(series_lulc.table.to_string())

series_lulc.view_series_areas()

'''

series_lulc = ds.LULCSeries(name="Potiribu")
series_lulc.load_folder(
    folder="../samples/lulc",
    name_pattern="map_lulc_*",
    table_file="../samples/table_lulc.txt",
    talk=True
)
print(series_lulc.catalog.to_string())


series_lulc.plot_series_areas()

series_ndvi = ds.NDVISeries(name="Potiribu")
series_ndvi.load_folder(
    folder="../samples/ndvi",
    name_pattern="map_ndvi_*",
    talk=True
)
print(series_ndvi.catalog.to_string())
'''


