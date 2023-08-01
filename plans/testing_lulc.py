import numpy as np
import matplotlib.pyplot as plt
import datasets as ds
import glob, copy
import warnings

warnings.filterwarnings('ignore')


map_aoi = ds.AOI()
map_aoi.load(
    asc_file="../samples/map_aoi.asc",
    prj_file="../samples/map_aoi.prj"
)

map_lito = ds.Litology(name="Lito")
map_lito.load(
    asc_file="../samples/map_lito.asc",
    prj_file="../samples/map_lito.prj",
    table_file="../samples/table_lito.txt"
)
map_lito.view()
map_lito.apply_aoi_mask(grid_aoi=map_aoi.grid)
map_lito.view()
map_lito.release_aoi_mask()
map_lito.view()

map_hand = ds.HAND()
map_hand.load(
    asc_file="../samples/map_hand.asc",
    prj_file="../samples/map_hand.prj"
)
#map_hand.view()

map_soils = ds.Soils()
map_soils.load(
    asc_file="../samples/map_soils.asc",
    prj_file="../samples/map_soils.prj",
    table_file="../samples/table_soils.txt"
)
#map_soils.get_hydro_soils(map_lito=map_lito, map_hand=map_hand)
map_soils.view()

#map_soils.export(folder="C:/output", filename="map_soils")

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