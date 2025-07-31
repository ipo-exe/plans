import os, sys
import pprint
import matplotlib.pyplot as plt
import numpy as np
import plans.geo
from plans.hydro import Local
from plans.datasets import SciRaster
from testing.hydro_utils import copy_inputs, delete_inputs
import time
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


if __name__ == "__main__":
    waiter = 1
    b_view = False
    b_solve = False
    b_print = False
    # instantiate
    m = Local()
    m.boot(bootfile="./data/bootfile_Local.csv")
    m.use_g2g = True

    if b_print:
        print("\n\nMETADATA\n")
        print(m.get_metadata_df().to_string())
        print(m.folder_data_clim)
        print(m.folder_data_lulc)
        print(m.folder_output)

    print("\n\nLOADING DATA ...")
    # load data
    start_time = time.time()
    m.load()
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print the elapsed time
    print(f"Loading elapsed time: {elapsed_time:.2f} seconds")
    time.sleep(waiter)

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

        print("\n\nSOIL\n")
        print(m.data_soils_table.head())

        print("\n\nBASINS\n")
        print(m.data_basins.get_catalog(mode="short"))

        print("\n\nLULC\n")
        print(m.data_lulc_table.head())
        print(m.data_lulc.get_catalog(mode="short"))

    # get views
    if b_view:
        m.data_tsi.view()
        m.data_soils.view()

        # get lulc views
        for b in m.data_lulc.collection:
            m.data_lulc.collection[b].view()

        # get basins views
        for b in m.data_basins.collection:
            m.data_basins.collection[b].view()


    print("\n\nSETUP DATA ...")
    # load data
    start_time = time.time()
    m.setup()
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print the elapsed time
    print(f"Setup elapsed time: {elapsed_time:.2f} seconds")
    time.sleep(waiter)

    if b_print:
        print("\n\nTime Area Histogram\n")
        print(m.data_tah.head())
        print("\n\nUnit Hydrograph\n")
        print(m.data_guh.head())

    msg = """
    KEEP DEVELOPING THE SOLVE() METHOD FOR LOCAL    
    """
    print(msg)
    if b_solve:
        print("\n\nSOLVE ...")
        # load data
        start_time = time.time()
        m.solve()
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        # Print the elapsed time
        print(f"Solving elapsed time: {elapsed_time:.2f} seconds")
        time.sleep(waiter)


    print("\n\nEXPORTING DATA ...")
    # load data
    start_time = time.time()
    m.export(
        folder="./data/Local/outputs",
        filename=m.name,
        views=False,
        mode=None
    )
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print the elapsed time
    print(f"Export elapsed time: {elapsed_time:.2f} seconds")
    time.sleep(waiter)


    print("\n>>> OK. PASSING.\n")