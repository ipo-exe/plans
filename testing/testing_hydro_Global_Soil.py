import os
import pprint as pp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from plans.hydro import Global

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def print_soil(m):
    df = m.data.copy()
    ls = [m.dtfield] + m.vars_soil
    df = df[ls]
    print(df.round(3))


if __name__ == "__main__":
    # instantiate
    m = Global(name="Global", alias="GLB000")
    m.boot(bootfile="./data/bootfile_Global.csv")
    m.load()

    m.params["G0"]["value"] = 80

    # set controllers for testing
    m.shutdown_epot = False
    m.shutdown_qif = False
    m.shutdown_qbf = False
    m.shutdown_dt = False
    # set up simulation data
    m.setup()
    # set some viewspecs
    m.view_specs["ymax_G"] = 200

    # Forest
    m.name = "Global-Soil-forest"
    print(m.name)
    # set canopy params
    m.params["C_a"]["value"] = 15
    m.params["C_k"]["value"] = 2
    # surface activation
    m.params["S_of_a"]["value"] = 10 # mm, level for activation of surface channels
    m.params["S_uf_a"]["value"] = 1  # mm, level for activation of topsoil channels
    m.params["S_uf_cap"]["value"] = 6  # mm for topsoil capacity
    # surface fragmentation
    m.params["S_of_c"]["value"] = 6  # mm for overland flow coef reach 0.5
    m.params["S_uf_c"]["value"] = 4 # mm for underland flow coef reach 0.5
    # infiltration
    m.params["S_k"]["value"] = 2.5  # days, residence time

    # soil params
    m.params["K"]["value"] = 20
    # run
    m.solve()
    print_soil(m)

    '''
    plt.plot(m.data[m.dtfield], m.data["Q_vf"])
    plt.show()
    '''
    m.view(mode="canopy")
    m.view(mode="surface")
    m.view(mode="soil")
    '''
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode="soil"
    )
    '''








