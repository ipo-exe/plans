import os
import pprint as pp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from plans.hydro import Global
from testing.hydro_utils import copy_inputs, delete_inputs

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def print_soil(m):
    df = m.data.copy()
    ls = m.vars_soil
    print(ls)
    df = df[ls]
    print(df.tail(20).round(3))
    #print(df.tail(20).round(3))


if __name__ == "__main__":
    # instantiate
    m = Global(name="Global", alias="GLB000")
    m.boot(bootfile="./data/bootfile_Global.csv")
    copy_inputs(dst_folder=m.folder_data)
    m.load()

    m.params["g0"]["value"] = 80

    # set controllers for testing
    m.shutdown_epot = False
    m.shutdown_qif = False
    m.shutdown_qbf = False

    # set up simulation data
    m.setup()

    # set some viewspecs
    m.view_specs["ymax_G"] = None

    # Forest
    m.name = "Global-Soil-forest"
    print(m.name)
    # set canopy params
    m.params["c_a"]["value"] = 15
    m.params["c_k"]["value"] = 2
    # surface activation
    m.params["s_of_a"]["value"] = 10 # mm, level for activation of surface channels
    m.params["s_uf_a"]["value"] = 1  # mm, level for activation of topsoil channels
    m.params["s_uf_cap"]["value"] = 6  # mm for topsoil capacity
    # surface fragmentation
    m.params["s_of_c"]["value"] = 6  # mm for overland flow coef reach 0.5
    m.params["s_uf_c"]["value"] = 4 # mm for underland flow coef reach 0.5
    # infiltration
    m.params["s_k"]["value"] = 2.5  # days, residence time

    # soil params
    m.params["k_v"]["value"] = 20
    m.params["d_et_a"]["value"] = 60
    # run
    m.solve()
    print_soil(m)

    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode="soil"
    )
    # cleaup testing
    delete_inputs(dst_folder=m.folder_data)
    print("\n>>> passing\n")








