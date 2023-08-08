import matplotlib.pyplot as plt
from plans.project import Project
from plans import datasets as ds
import warnings
import os

warnings.filterwarnings('ignore')

map_slope = ds.Slope(name="Slope")
map_dem = ds.Elevation(name="Elevation")

raster_coll = ds.NDVISeries(name="NDVISeries")

raster_coll.load_folder(
    folder="C:/plans/Demo/datasets/ndvi",
    name_pattern="map_ndvi_*",
    talk=True
)

print(raster_coll.catalog.to_string())

print()
print(raster_coll.get_series_stats().to_string())