import os
import numpy as np
import matplotlib.pyplot as plt
from plans.datasets import Raster
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    #f = "./data/parsers/inputs/dem.tif"
    f = "C:/plans/testing/moderate/inputs/topo/dem.tif"
    dc = Raster.read_raster(
        file_input=f,
        dtype="float",
        metadata=True
    )
    grd = dc["grid"]
    mtd = dc["metadata"]
    print(mtd)
    plt.imshow(grd)
    plt.show()