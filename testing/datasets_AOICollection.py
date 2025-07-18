import os, pprint
from plans import datasets as ds
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    # read tifs
    folder = "./data/parsers/inputs"
    rs = ds.AOICollection(
        name="aoi-collection",
    )
    # todo [bugfix]
    rs.load_folder(folder, name_pattern="basins_*", talk=True)
    print(rs.catalog.to_string())