import os
import numpy as np
import pandas as pd

from plans.analyst import Univar

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    r = 1
    v = np.random.normal(loc=100 * r, scale=2 * r, size=500)
    uv = Univar()
    uv.data = pd.DataFrame(
        {
            uv.varfield: v
        }
    )
    uv.view_specs["subtitle_a"]
    uv.view_specs["style"] = "seaborn"
    uv.view_specs["bins"] = 50
    uv.view_specs["range"] = (90, 110)
    uv.view_specs["mode"] = "mini"
    uv.view()
    uv.view_specs["mode"] = "full"
    uv.view()
