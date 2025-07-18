import os
import pprint
from plans.hydro import LSFAS
from testing.hydro_utils import copy_inputs, delete_inputs

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    # instantiate
    m = LSFAS(name="LSFAS", alias="LSFAS000")
    m.boot(bootfile="./data/bootfile_LSFAS.csv")
    copy_inputs(dst_folder=m.folder_data)
    m.load()
    # setup
    m.setup()
    m.view_specs["ymax_Q"] = 50
    m.view_specs["ymax_S"] = 120


    # simulate with epot
    m.shutdown_epot = False
    m.shutdown_qb = False

    m.name = "LSFAS-urban"
    m.params["s_c"]["value"] = 0.0
    m.run(setup_model=False)
    print(m.data.head().to_string())
    m.view()
    m.export(
        folder="./data/LSFAS/outputs",
        filename=m.name,
        views=True
    )

    m.name = "LSFAS-farm"
    m.params["s_a"]["value"] = 50.0
    m.params["s_c"]["value"] = 10.0
    m.run(setup_model=False)
    print(m.data.head().to_string())
    m.export(
        folder="./data/LSFAS/outputs",
        filename=m.name,
        views=True
    )

    m.name = "LSFAS-farm-nbs"
    m.params["s_a"]["value"] = 100.0
    m.params["s_c"]["value"] = 10.0
    m.run(setup_model=False)
    print(m.data.head().to_string())
    m.export(
        folder="./data/LSFAS/outputs",
        filename=m.name,
        views=True
    )

    m.name = "LSFAS-forest"
    m.params["s_a"]["value"] = 80.0
    m.params["s_c"]["value"] = 100.0
    m.run(setup_model=False)
    print(m.data.head().to_string())
    m.export(
        folder="./data/LSFAS/outputs",
        filename=m.name,
        views=True
    )

    delete_inputs(dst_folder=m.folder_data)
    print("\n>>> OK. PASSING.\n")

