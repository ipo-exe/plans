import os, time
import matplotlib.pyplot as plt
from plans.hydro import Global
from testing.hydro_utils import copy_inputs, delete_inputs

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    b_solve = True
    b_view = True
    b_print = True
    # instantiate
    m = Global(name="Global", alias="GLB000")
    m.boot(bootfile="./data/bootfile_Global.csv")
    copy_inputs(dst_folder=m.folder_data)

    if b_print:
        print("\n\nMETADATA\n")
        print(m.get_metadata_df().to_string())

    # load simulation data
    time_s = time.time()
    print("\n\nLOADING DATA...")
    m.load()
    time_e = time.time()
    time_delta = time_e - time_s
    print("loading elapsed time: {:.2f}s\n".format(time_delta))
    time.sleep(3)

    m.params["g0"]["value"] = 10
    m.params["s0"]["value"] = 0

    if b_print:
        # inspect parameters
        print("\n\nPARAMETERS\n")
        print(m.get_params().to_string())

        # inspect loaded data
        #
        print("\n\nSERIES\n")
        print("\n\nClimate series\n")
        print(m.data_clim.head())
        print("\n\nObserved series\n")
        print(m.data_obs.head())

        print("\n\nPath Area Histogram\n")
        print(m.data_pah.head())

    # set up simulation data
    time_s = time.time()
    print("\n\nSETUP DATA...")
    m.setup()
    time_e = time.time()
    time_delta = time_e - time_s
    print("Setup elapsed time: {:.2f}s\n".format(time_delta))
    time.sleep(3)

    if b_print:
        print("\n\nTime Area Histogram\n")
        print(m.data_tah.head())
        print("\n\nUnit Hydrograph\n")
        print(m.data_guh.head())

    # solve simulation data
    if b_solve:
        time_s = time.time()
        print("\n\nSOLVE...")
        m.solve()
        time_e = time.time()
        time_delta = time_e - time_s
        print("Simulation elapsed time: {:.2f}s\n".format(time_delta))
        time.sleep(3)

    if b_print:
        print(m.data.info())
        print(m.data.round(3))

    if b_solve and b_view:
        m.view(mode="river")

    print("\n\nEXPORTING DATA...")
    # load data
    start_time = time.time()
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=False,
        mode=None
    )
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print the elapsed time
    print(f"Export elapsed time: {elapsed_time:.2f} seconds")
    time.sleep(3)



    delete_inputs(dst_folder=m.folder_data)
    print("\n>>> OK. PASSING.\n")

