import os
from plans.hydro import LSRRE
from testing.hydro_utils import copy_inputs, delete_inputs

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    # instantiate
    print("loading")
    m = LSRRE(name="LSRRE", alias="LSRRE01")
    m.boot(bootfile="./data/bootfile_LSRRE.csv")
    copy_inputs(dst_folder=m.folder_data)
    m.load()
    m.setup()
    m.view_specs["ymax_Q"] = 15
    m.view_specs["ymax_S"] = 100

    # simulate with epot
    print("simulate with epot")
    m.shutdown_epot = False
    m.name = "LSRRE-epot"
    m.run(setup_model=False)
    m.export(
        folder="./data/LSRRE/outputs",
        filename=m.name,
        views=True
    )

    # simulate with no epot
    print("simulate with no epot")
    m.shutdown_epot = True
    m.name = "LSRRE-noepot"
    m.run(setup_model=False)
    m.export(
        folder="./data/LSRRE/outputs",
        filename=m.name,
        views=True
    )
    delete_inputs(dst_folder=m.folder_data)
    print("\n>>> OK. PASSING.\n")

