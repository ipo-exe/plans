import os
import pprint as pp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from plans.hydro import Global

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def print_surface(m):
    df = m.data.copy()
    ls = [m.dtfield] + m.vars_surface
    df = df[ls]
    print(df.round(3))


if __name__ == "__main__":
    # instantiate
    m = Global(name="Global", alias="GLB000")
    m.boot(bootfile="./data/bootfile_Global.csv")
    m.load()
    # set controllers for testing
    m.shutdown_epot = False
    m.shutdown_qif = False
    # set up simulation data
    m.setup()
    # set some viewspecs
    m.view_specs["ymax_S"] = 40

    # Forest
    m.name = "Global-Surface-forest"
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
    m.params["S_k"]["value"] = 0.5  # days, residence time
    # run
    m.solve()
    print_surface(m)
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode="surface"
    )

    # Farm
    m.name = "Global-Surface-farm"
    print(m.name)
    # set canopy params
    m.params["C_a"]["value"] = 5
    m.params["C_k"]["value"] = 0.5
    # surface activation
    m.params["S_of_a"]["value"] = 5  # mm for activation of surface
    m.params["S_uf_a"]["value"] = 5  # mm for activation of topsoil (subsurface)
    m.params["S_uf_cap"]["value"] = 6  # mm for topsoil capacity
    # surface fragmentation
    m.params["S_of_c"]["value"] = 2  # mm for overland flow coef reach 0.5
    m.params["S_uf_c"]["value"] = 20  # mm for underland flow coef reach 0.5 (more frag)
    # infiltration
    m.params["S_k"]["value"] = 1.0  # days, residence time
    # run
    m.solve()
    print_surface(m)
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode="surface"
    )

    # Farm NbS
    m.name = "Global-Surface-farmNbS"
    print(m.name)
    # set canopy params
    m.params["C_a"]["value"] = 8
    m.params["C_k"]["value"] = 0.6
    # surface activation
    m.params["S_of_a"]["value"] = 13  # mm for activation of surface
    m.params["S_uf_a"]["value"] = 5  # mm for activation of topsoil (subsurface)
    m.params["S_uf_cap"]["value"] = 6  # mm for topsoil capacity
    # surface fragmentation
    m.params["S_of_c"]["value"] = 8  # mm for overland flow coef reach 0.5
    m.params["S_uf_c"]["value"] = 10  # mm for underland flow coef reach 0.5 (more frag)
    # infiltration
    m.params["S_k"]["value"] = 0.8  # days, residence time
    # run
    m.solve()
    print_surface(m)
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode="surface"
    )


    # Urban
    m.name = "Global-Surface-urban"
    print(m.name)
    # set canopy params
    m.params["C_a"]["value"] = 3
    m.params["C_k"]["value"] = 0.01
    # surface activation
    m.params["S_of_a"]["value"] = 0  # mm for activation of surface
    m.params["S_uf_a"]["value"] = 0  # mm for activation of topsoil (subsurface)
    m.params["S_uf_cap"]["value"] = 0  # mm for topsoil capacity
    # surface fragmentation
    m.params["S_of_c"]["value"] = 0.01  # mm for overland flow coef reach 0.5
    m.params["S_uf_c"]["value"] = 20  # mm for underland flow coef reach 0.5 (more frag)
    # infiltration
    m.params["S_k"]["value"] = 1000.0  # days, residence time
    # run
    m.solve()
    print_surface(m)
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode="surface"
    )

    '''

    # Med Ca and Hi Ck
    m.name = "Global-urban"
    m.params["C_a"]["value"] = 3
    m.params["C_k"]["value"] = 0.01
    m.solve()
    print_canopy(m)
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode=None
    )'''





