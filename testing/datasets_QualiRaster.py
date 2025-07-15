import os, pprint
import numpy as np
import matplotlib.pyplot as plt
from plans.datasets import QualiRaster
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    # read tif
    f1 = "./data/parsers/inputs/lulc_2019-01-01.tif"
    f2 = "../plans/data/lulc_conversion.csv"
    r = QualiRaster()
    r.load(file_raster=f1, file_table=f2)
    r.view_specs["style"] = "wien"
    r.view_specs["filename"] = "figure"
    r.view_specs["folder"] = "C:/data"
    r.view(show=True)