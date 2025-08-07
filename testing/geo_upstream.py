import os, pprint
from plans import geo
import geopandas as gpd
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    f = "./data/geo/inputs/_basins.gpkg"
    basins = gpd.read_file(f, layer="area_drenagem")
    start = 20916
    up = geo.get_upstream_features(
        features=basins,
        field_id="cotrecho",
        field_id_down="nutrjus",
        start_id=start,
        include_start=True
    )
    print(up)
    print(type(up))
    up.to_file(f, layer=f"output_{start}", driver="GPKG")
    up_diss = up.dissolve()
    up_diss.to_file(f, layer=f"output_{start}_dissolved", driver="GPKG")

    gauges = gpd.read_file(f, layer="points")
    print("\n")
    print(gauges)

    basins_up = geo.get_basins_by_gauges(
        basins=basins,
        gauges=gauges,
        field_gauge="cod_rain",
        field_basin="cotrecho",
        field_basin_down="nutrjus"
    )
    print("\n")
    print(basins_up)
    basins_up.to_file(f, layer=f"output_basins_gauges", driver="GPKG")
