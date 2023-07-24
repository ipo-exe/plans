import glob
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
s_date = '1990-01-01'
map_lulc = ds.LULC(name="Potiribu_{}".format(s_date), date=s_date)
s_file_raster = "../samples/lulc/map_lulc_{}.asc".format(s_date)
map_lulc.load_asc_raster(file=s_file_raster)
s_file_table = "../samples/table_lulc.txt"
map_lulc.load_table(file=s_file_table)

# apply AOI mask
map_lulc.apply_aoi_mask(grid_aoi=map_aoi.grid, inplace=True)

# view LULC map
map_lulc.view_qualiraster(show=True, filename="lulc_{}".format(s_date), folder="C:/bin")

# compute LULC areas
df_areas = map_lulc.get_areas()
print(df_areas.to_string())

# zonal stats from slope map
df_stats = map_lulc.get_zonal_stats(
    raster_sample=map_slope,
    merge=True,
    skip_count=True
)
print(df_stats.to_string())

print(map_lulc.table.to_string())

# Series
s_file_table = "../samples/table_lulc.txt"
series_lulc = ds.LULCSeries(name="potiribu")
lst_maps = glob.glob(r"..\samples\lulc\*.asc")
print("loading")
for f in lst_maps:
    s_date = f.split("_")[-1].split(".")[0]
    series_lulc.load_asc_raster(
        name="Potiribu_{}".format(s_date),
        date=s_date,
        file=f,
        file_table=s_file_table
    )
# apply AOI to
for raster in series_lulc.collection:
    series_lulc.collection[raster].apply_aoi_mask(
        grid_aoi=map_aoi.grid,
        inplace=True
    )

print(series_lulc.catalog.to_string())
print(series_lulc.get_series_stats().to_string())
series_lulc.plot_series_stats(show=True)



for raster in series_lulc.collection:
    print(raster)
    series_lulc.collection[raster].view_qualiraster(
        show=True,
        filename=raster,
        folder="C:/bin"
    )
#maps_lulc.get_series_areas()
