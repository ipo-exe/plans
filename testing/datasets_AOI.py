import os, pprint
from plans.datasets import QualiRaster, AOI
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    # read tif
    f1 = "./data/parsers/inputs/basins_300.tif"
    r = AOI()
    r.load_data(file_data=f1, )
    r.view_specs["style"] = "wien"
    r.view_specs["filename"] = "figure"
    r.view_specs["folder"] = "C:/data"
    r.view(show=True)
    print(">>> passing")