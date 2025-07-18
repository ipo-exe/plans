import os
import numpy as np
from plans.datasets import TWI, HTWI, Hydrology
from plans import geo

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":

    f1 = "./data/parsers/inputs/slp.tif"
    f2 = "./data/parsers/inputs/dem.tif"
    f3 = "./data/parsers/inputs/twi.tif"

    r_twi = TWI()
    r_twi.load_data(file_data=f3)
    #r_twi.view()

    r_d = Hydrology(name="D", varalias="deficit", datetime="2012-01-01")
    r_d.set_raster_metadata(metadata=r_twi.raster_metadata)
    r_d.update()
    r_d.view_specs["range"] = (-4, 250)
    r_d.view_specs["folder"] = "./data/Local/outputs/downscaling"

    d_scale = 10
    ls_d = [200, 150, 100, 90, 80, 70, 60, 50, 25, 20, 15, 12, 10, 7, 5, 4, 3, 2, 1, 0]
    ls_d1 = list(np.linspace(0, 30, 30))
    ls_d2 = list(np.linspace(40, 200, 30))
    ls_d = ls_d1 + ls_d2
    ls_d.reverse()

    #ls_d = [10]
    for d_mu in ls_d:
        s_name = "deficit_{}".format(str(int(d_mu)).zfill(4))
        print(s_name)
        d_grid = geo.downscale_variance(
            mean=d_mu,
            scale_factor=d_scale,
            array_covar=r_twi.data,
            mirror=True,
            mode="compensate",
            min_value=0.0
        )
        r_d.set_data(grid=d_grid)
        r_d.view_specs["suptitle"] = f"Soil water deficit = {round(d_mu, 2)} mm"
        r_d.view_specs["filename"] = s_name
        r_d.view(show=False)