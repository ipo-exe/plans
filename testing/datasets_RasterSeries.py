import os, pprint
from plans import datasets as ds
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    # read tifs
    folder = "./data/parsers/inputs"
    rs = ds.RasterSeries(
        name="random-series",
        varname="random",
        varalias="rd",
        units="na"
    )
    rs.load_folder(folder, name_pattern="_random_*", talk=True)
    print(rs.catalog.to_string())
    df = rs.get_collection_stats()
    print(df[["name", "mean", "min", "max"]].to_string())
    df = rs.get_series_stats()
    print(df[["name", "datetime", "mean", "min", "max"]].to_string())

    f1 = "./data/parsers/inputs/basins_400.tif"
    aoi = ds.AOI()
    aoi.load_data(file_data=f1)
    aoi.view()

    rs.apply_aoi_masks(grid_aoi=aoi.data, inplace=False)

    specs = {
        "range": (0, 200)
    }
    rs.get_views(show=True, specs=specs, talk=True)

    # export views
    rs.get_views(
        show=False,
        specs=specs,
        folder="./data/parsers/outputs",
        suffix="testing",
        talk=True
    )
    print(">>> passing")