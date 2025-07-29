import os
import pprint
import matplotlib.pyplot as plt
import numpy as np
from plans.hydro import Local
from testing.hydro_utils import copy_inputs, delete_inputs
import time
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


if __name__ == "__main__":
    b_view = False
    # instantiate
    m = Local(name="Local")
    m.boot(bootfile="./data/bootfile_Local.csv")
    print("\n\nMETADATA\n")
    print(m.get_metadata_df().to_string())
    print(m.folder_data_clim)
    print(m.folder_data_lulc)
    print(m.folder_output)

    print("\n\nLOADING DATA...")
    # load data
    start_time = time.time()
    m.load()
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print the elapsed time
    print(f"Loading elapsed time: {elapsed_time:.2f} seconds")

    # inspect parameters
    print("\n\nPARAMETERS\n")
    print(m.get_params().to_string())

    # inspect loaded data
    #
    print("\n\nSERIES\n")
    print(m.data_clim.head())
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

