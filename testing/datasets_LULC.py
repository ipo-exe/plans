import os, pprint
from plans.datasets import LULC
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    # read tif
    f1 = "./data/parsers/inputs/lulc_2019-01-01.tif"
    f2 = "../plans/data/lulc_conversion.csv"
    r = LULC(name="Mapbiomas", date="2019-01-01")
    r.load(file_raster=f1, file_table=f2)
    r.view_specs["style"] = "wien"
    r.view_specs["filename"] = "figure"
    r.view_specs["folder"] = "C:/data"
    r.view(show=True)

    f3 = "./data/parsers/inputs/basins_300.tif"
    r.load_aoi_mask(file_raster=f3, inplace=False)
    r.view(show=True)