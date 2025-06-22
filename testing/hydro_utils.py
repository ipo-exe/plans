import os, shutil
from pathlib import Path

def copy_inputs(dst_folder):
    print("-- copy inputs")
    shutil.copy(
        src="./data/clim.csv",
        dst=Path(f"{dst_folder}/clim.csv")
    )
    shutil.copy(
        src="./data/q_obs.csv",
        dst=Path(f"{dst_folder}/q_obs.csv")
    )
    return None

def delete_inputs(dst_folder):
    print("-- remove inputs")
    os.remove(Path(f"{dst_folder}/clim.csv"))
    os.remove(Path(f"{dst_folder}/q_obs.csv"))
    return None