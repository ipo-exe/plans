import os, time
import matplotlib.pyplot as plt
from plans.hydro import Global
from testing.hydro_utils import copy_inputs, delete_inputs

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":

    # instantiate
    m = Global(name="Global", alias="GLB000")
    m.boot(bootfile="./data/bootfile_Global.csv")
    copy_inputs(dst_folder=m.folder_data)
    m.load()

    m.params["g0"]["value"] = 10
    m.params["s0"]["value"] = 0

    # set up simulation data
    time_s = time.time()
    print("\n\nsetting up data...")
    m.setup()
    time_e = time.time()
    time_delta = time_e - time_s
    print("Setup elapsed time: {:.2f}s\n".format(time_delta))
    time.sleep(2)
    print(m.data_pah)
    print(m.data_tah)
    print(m.data_guh)
    # solve simulation data
    time_s = time.time()
    print("\n\nsolving simulation...")
    m.solve()
    time_e = time.time()
    time_delta = time_e - time_s
    print("Simulation elapsed time: {:.2f}s\n".format(time_delta))
    time.sleep(2)
    print(m.data.info())
    print(m.data.round(3))
    time.sleep(2)
    m.view(mode="river")

