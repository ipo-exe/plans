import os, pprint
import numpy as np
import matplotlib.pyplot as plt
from plans.datasets import Raster
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    f = "./data/parsers/inputs/dem.tif"
    rs = Raster()
    rs.load(file_raster=f)
    rs.view(show=True)

    grd_aoi = 1.0 * (rs.data > 1000)

    rs.apply_aoi_mask(grid_aoi=grd_aoi, inplace=False)
    rs.view(show=True)

    rs.release_aoi_mask()
    rs.view(show=True)
