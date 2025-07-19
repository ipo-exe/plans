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

def print_canopy(m):
    df = m.data.copy()
    ls = m.vars_canopy + ["Q"]
    df = df[ls]
    print(df.round(3))


if __name__ == "__main__":
    # instantiate
    m = Global(name="Global", alias="GLB000")
    m.boot(bootfile="./data/bootfile_Global.csv")
    copy_inputs(dst_folder=m.folder_data)
    m.load()
    m.shutdown_epot = False
    m.setup()
    m.view_specs["ymax_C"] = 20

    # Hi Ca and Hi Ck
    m.name = "Global-Canopy-forest"
    m.params["c_a"]["value"] = 15
    m.params["c_k"]["value"] = 2
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
    m.params["c_a"]["value"] = 5
    m.params["c_k"]["value"] = 2
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
    m.params["c_a"]["value"] = 3
    m.params["c_k"]["value"] = 0.01
    m.solve()
    print_canopy(m)
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode=None
    )
    # cleaup testing
    delete_inputs(dst_folder=m.folder_data)
    print("\n>>> passing\n")





