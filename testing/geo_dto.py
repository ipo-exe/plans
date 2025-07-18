import os, pprint
import numpy as np
import matplotlib.pyplot as plt
from plans.datasets import LDD
from plans import geo
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    f = "./data/parsers/inputs/ldd.tif"
    ldd = LDD()
    ldd.load_data(file_data=f)
    plt.imshow(ldd.data)
    plt.show()
    dto = geo.distance_to_outlet(grd_ldd=ldd.data, n_res=30)
    plt.imshow(dto)
    plt.show()

