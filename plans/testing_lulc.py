import matplotlib.pyplot as plt
import datasets as ds

# load slope layer
map_slope = ds.Slope(name="Sample")
map_slope.load_asc_raster(file="../samples/map_slope.asc")
#map_slope.view_raster(show=True)

# load AOI raster
map_aoi = ds.AOI(name="Sample")
map_aoi.load_asc_raster(file="../samples/map_aoi.asc")
#map_aoi.view_qualiraster(show=True)

# set AOI to slope layer
map_slope.apply_aoi_mask(grid_aoi=map_aoi.grid, inplace=True)
#map_slope.view_raster(show=True)


# create LULC dataset
map_lulc = ds.LULC(name="potiribu", date="1985-01-01")
s_file_raster = "../samples/lulc/map_lulc_1985-01-01.asc"
map_lulc.load_asc_raster(file=s_file_raster)
s_file_table = "../samples/table_lulc.txt"
map_lulc.load_table(file=s_file_table)

print(map_lulc.table.to_string())

# apply AOI mask
map_lulc.apply_aoi_mask(grid_aoi=map_aoi.grid, inplace=True)


# view LULC map
map_lulc.view_qualiraster(show=True)
'''
df_areas = map_lulc.get_areas()
print(df_areas.to_string())
# zonal stats
df_stats = map_lulc.get_zonal_stats(raster_sample=map_slope, merge=True, skip_count=True)
print(df_stats.to_string())

print(map_lulc.table.to_string())

maps_lulc = ds.LULCSeries(name="potiribu")
lst_maps = [
    "C:/gis/potiribu/plans/lulc_simple_2016-01-01.asc",
    "C:/gis/potiribu/plans/lulc_simple_2017-01-01.asc",
    "C:/gis/potiribu/plans/lulc_simple_2018-01-01.asc",
    "C:/gis/potiribu/plans/lulc_simple_2019-01-01.asc",
    "C:/gis/potiribu/plans/lulc_simple_2020-01-01.asc"
]
for f in lst_maps:
    s_date = f.split(".")[0][-10:]
    print(s_date)

    maps_lulc.load_asc_raster(
        name="lulc_{}".format(s_date),
        date=s_date,
        file=f,
        file_table=s_file_table
    )

maps_lulc.get_series_areas()'''