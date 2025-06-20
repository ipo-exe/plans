import os, shutil
from pathlib import Path
from plans.hydro import LSRR
from testing.testing_hydro_utils import copy_inputs, delete_inputs

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    print("loading")
    # instantiate model and boot from file
    m0 = LSRR(name="LSRR", alias="LSRR01")
    # boot() is a form of light load of filepaths and some basic attributes
    m0.boot(bootfile="./data/bootfile_LSRR_0.csv")
    # copy inputs
    copy_inputs(dst_folder=m0.folder_data)
    # load() handles heavy-loading of data (parameter data and inputs data)
    m0.load()
    # setup() creates the variables that will hold the simulation (may be intensive!)
    m0.params["S0"]["value"] = 14
    m0.setup()

    # set view specs
    m0.view_specs["color_P"] = "magenta"
    m0.view_specs["ymax_Q"] = 40

    # testing helper
    m0.shutdown_epot = False

    # Change parameters on the fly
    print("simulate with low k")
    #
    # simulate with low k
    m0.name = "LSRR-loK"
    m0.params["k"]["value"] = 1.5
    m0.run(setup_model=False)
    m0.export(
        folder="./data/LSRR/outputs",
        filename=m0.name,
        views=True
    )

    #
    # simulate with hih k
    print("simulate with hih k")
    m0.name = "LSRR-hiK"
    m0.params["k"]["value"] = 10.5
    m0.run(setup_model=False)
    m0.export(
        folder="./data/LSRR/outputs",
        filename=m0.name,
        views=True
    )
    delete_inputs(dst_folder=m0.folder_data)

    # create a new model from other file

    #
    # simulate with hi dt
    print("simulate with hi dt")
    m1 = LSRR(name="LSRR", alias="LSRR01")
    m1.boot(bootfile="./data/bootfile_LSRR_1.csv")
    copy_inputs(dst_folder=m1.folder_data)
    m1.view_specs["ymax_Q"] = 40
    m1.load()
    m1.setup()
    m1.name = "LSRR-hiK-hiDT"
    m1.params["k"]["value"] = 10
    m1.run(setup_model=False)
    m1.export(
        folder="./data/LSRR/outputs",
        filename=m1.name,
        views=True
    )
    delete_inputs(dst_folder=m1.folder_data)
    print("\n>>> OK. PASSING.\n")





