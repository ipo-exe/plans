import os
import pprint as pp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from plans.hydro import Global
from testing.testing_hydro_utils import copy_inputs, delete_inputs

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":

    # instantiate
    m = Global(name="Global", alias="GLB000")
    m.boot(bootfile="./data/bootfile_Global.csv")
    copy_inputs(dst_folder=m.folder_data)
    m.load()

    m.params["G0"]["value"] = 10
    m.params["S0"]["value"] = 0

    # set up simulation data
    m.setup()
    print(m.data_paths)
    print(m.unit_hydrograph["q"].sum())

    m.solve()
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode="river"
    )
    delete_inputs(dst_folder=m.folder_data)
    print("\n>>> OK. PASSING.\n")



