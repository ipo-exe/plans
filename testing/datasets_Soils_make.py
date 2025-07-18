import os
import pprint
import matplotlib.pyplot as plt
import numpy as np
from plans import datasets as ds
from plans import geo

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


if __name__ == "__main__":
    f1 = "./data/parsers/inputs/hnd.tif"
    f2 = "./data/parsers/inputs/slp.tif"

    hand_dc = ds.Raster().read_tif(f1)
    slope_dc = ds.Raster().read_tif(f2)
    soils = geo.soils(slope_dc["data"], hand_dc["data"])

    r = ds.QualiHard()
    #r.dtype = "uint8"
    r.set_data(grid=soils)
    r.set_raster_metadata(metadata=slope_dc["metadata"])
    r.raster_metadata["NODATA_value"] = ds.DC_NODATA[r.dtype]
    r.update()
    r.view()
    r.export_tif(folder="./data/parsers/inputs", filename="soils")
