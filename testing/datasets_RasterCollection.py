import os, pprint
from plans import datasets as ds
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    # read tif
    f1 = "./data/parsers/inputs/dem.tif"
    f2 = "./data/parsers/inputs/ldd.tif"

    rs = ds.RasterCollection()
    rs.load_data(name="DEM", file_data=f1)
    rs.load_data(name="LDD", file_data=f2)

    print(rs.catalog.to_string())

    rs.get_views(show=True, talk=True)
    rs.get_views(
        show=False,
        folder="./data/parsers/outputs",
        talk=True
    )
    print(">>> passing")