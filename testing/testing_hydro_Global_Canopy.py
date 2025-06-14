import os
import pprint as pp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from plans.hydro import Global

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def print_canopy(m):
    df = m.data.copy()
    ls = [m.dtfield] + m.vars_canopy + ["Q"]
    df = df[ls]
    print(df.round(3))


if __name__ == "__main__":
    # instantiate
    m = Global(name="Global", alias="GLB000")
    m.boot(bootfile="./data/bootfile_Global.csv")
    m.load()
    m.shutdown_epot = False
    m.setup()
    m.view_specs["ymax_C"] = 20

    # Hi Ca and Hi Ck
    m.name = "Global-Canopy-forest"
    m.params["C_a"]["value"] = 15
    m.params["C_k"]["value"] = 2
    m.solve()
    print_canopy(m)
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode=None
    )

    # Med Ca and Hi Ck
    m.name = "Global-Canopy-farm"
    m.params["C_a"]["value"] = 5
    m.params["C_k"]["value"] = 2
    m.solve()
    print_canopy(m)
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode=None
    )

    # Med Ca and Hi Ck
    m.name = "Global-Canopy-urban"
    m.params["C_a"]["value"] = 3
    m.params["C_k"]["value"] = 0.01
    m.solve()
    print_canopy(m)
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode=None
    )





