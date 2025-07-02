import os, pprint
import numpy as np
import matplotlib.pyplot as plt
from plans.datasets import Raster
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    handle_tif = False
    handle_asc = True
    if handle_tif:
        # read tif
        f = "./data/parsers/inputs/dem.tif"
        dc = Raster.read_tif(
            file_input=f,
            dtype="float32",
            metadata=True
        )
        grd = dc["data"]
        mtd = dc["metadata"]
        pprint.pprint(mtd)
        plt.imshow(grd)
        plt.show()

        # write tif
        fo = "./data/parsers/outputs/dem_anom.tif"
        grd_output = np.mean(grd) - grd
        # change no data
        mtd["NODATA_value"] = -9999
        Raster.write_tif(
            grid_output=grd_output,
            dc_metadata=mtd,
            file_output=fo,
            dtype="float32",
        )

        dc2 = Raster.read_tif(fo)
        print("\n\n")
        pprint.pprint(dc2["metadata"])
        plt.imshow(dc2["data"], cmap="jet", vmin=-100, vmax=100)
        plt.show()

    if handle_asc:
        print("\n\n")
        # read asc
        f = "./data/parsers/inputs/dem.asc"
        dc = Raster.read_asc(
            file_input=f,
            dtype="float32",
            metadata=True
        )
        grd = dc["data"]
        mtd = dc["metadata"]
        pprint.pprint(mtd)
        plt.imshow(grd)
        plt.show()

        # write asc
        fo = "./data/parsers/outputs/dem_anom.asc"
        grd_output = np.mean(grd) - grd
        # change no data
        mtd["NODATA_value"] = -9999
        Raster.write_asc(
            grid_output=grd_output,
            dc_metadata=mtd,
            file_output=fo,
            dtype="float32",
        )

        dc2 = Raster.read_asc(fo)
        print("\n\n")
        pprint.pprint(dc2["metadata"])
        plt.imshow(dc2["data"], cmap="jet", vmin=-100, vmax=100)
        plt.show()



