import os
import pprint
import matplotlib.pyplot as plt
import numpy as np
from plans.hydro import Local
from testing.hydro_utils import copy_inputs, delete_inputs

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


if __name__ == "__main__":
    b_view = False
    # instantiate
    m = Local(name="Local")
    m.boot(bootfile="./data/bootfile_Local.csv")
    print(m.get_metadata_df().to_string())
    print(m.folder_data_clim)
    print(m.folder_data_lulc)
    print(m.folder_output)
    print("loading data...")

    m.load()

    print(m.data_clim.head())
    print(m.data_obs.head())
    print(m.data_tah_input.head())
    print(m.data_lulc_table.head())
    if b_view:
        m.data_tsi.view()
        m.data_soils.view()

    print(m.data_basins.get_catalog(mode="short"))

    if b_view:
        for b in m.data_basins.collection:
            m.data_basins.collection[b].view()

    print(m.params["dt"]["value"])
    print(m.get_params().to_string())

    if b_view:
        for b in m.data_lulc.collection:
            m.data_lulc.collection[b].view()

    print(m.data_lulc.get_catalog(mode="short"))