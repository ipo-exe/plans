import matplotlib.pyplot as plt

import datasets as ds

# load slope layer
map_slope = ds.Slope(name="potiribu")
map_slope.load_asc_raster(file="C:/gis/potiribu/plans/slope.asc")
map_slope.view_raster(show=True)

# load AOI raster
map_aoi = ds.AOI(name="potiribu_aoi")
map_aoi.load_asc_raster(file="C:/gis/potiribu/plans/basin.asc")
map_aoi.view_qualiraster(show=True)

# set AOI to slope layer
map_slope.apply_aoi_mask(grid_aoi=map_aoi.grid, inplace=True)
map_slope.view_raster(show=True)

# create LULC dataset
map_lulc = ds.LULC(name="potiribu", date="2020-01-01")
s_file_raster = "C:/gis/potiribu/plans/lulc_simple_2020-01-01.asc"
map_lulc.load_asc_raster(file=s_file_raster)
s_file_table = "C:/gis/potiribu/lulc_simple/lulc_simple_table.txt"
map_lulc.load_table(file=s_file_table)
# apply AOI mask
map_lulc.apply_aoi_mask(grid_aoi=map_aoi.grid, inplace=True)

# view areas
df_areas = map_lulc.get_areas(merge=True)
print(df_areas.to_string())

# view LULC map
map_lulc.view_qualiraster(show=True)

# zonal stats
df_stats = map_lulc.get_zonal_stats(raster_sample=map_slope, merge=True, skip_count=True)
print(df_stats.to_string())

print(map_lulc.table.to_string())