import os, pprint
import numpy as np
import matplotlib.pyplot as plt
from plans.datasets import Raster
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    b_zoom = False
    b_export = False
    # read tif
    f = "./data/parsers/inputs/dem.tif"
    r = Raster()
    r.load(file_raster=f)


    if b_zoom:
        r.view_specs["zoom_window"] = {
            "i_min": 0,
            "i_max": 50,
            "j_min": 0,
            "j_max": 100,
        }

    #r.view_specs["ylim"] = (900, 1200)
    r.view_specs["style"] = "wien"
    r.view(show=True)
    r.reset_nodata(new_nodata=-99999)
    r.view(show=True)
    if b_export:
        r.export_tif(folder="C:/plans/testing", filename="dem_testing")