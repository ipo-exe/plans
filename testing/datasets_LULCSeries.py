import os, pprint
from plans.datasets import LULCSeries
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    # read tif
    f1 = "./data/parsers/inputs/lulc_2019-01-01.tif"
    f2 = "../plans/data/lulc_conversion.csv"

    rs = LULCSeries(name="mapbiomas")
    rs.load_folder(
        folder="./data/parsers/inputs",
        file_table=f2,
        name_pattern="lulc_*",
        talk=True
    )
    print(rs.catalog.to_string())
    rs.get_views(show=True)
