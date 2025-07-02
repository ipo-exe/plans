import os, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio.v3 as iio
from plans.hydro import LSRR
from plans.datasets import Raster

def slice_cmap(cmap_name, start_slice, end_slice):
    """
    Slices a Matplotlib colormap accessed by its name.

    Parameters:
    cmap_name (str): The name of the colormap to slice (e.g., 'viridis', 'plasma').
    start_slice (float): The starting point of the slice (0.0 to 1.0).
    end_slice (float): The ending point of the slice (0.0 to 1.0).

    Returns:
    matplotlib.colors.Colormap: A new colormap representing the sliced portion.
    """
    # Access the colormap by its name
    original_cmap = plt.colormaps[cmap_name]

    # Get 256 colors from the specified slice of the original colormap
    sliced_colors = original_cmap(np.linspace(start_slice, end_slice, 256))
    # Create a new colormap from these colors
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_sliced_{start_slice:.2f}_{end_slice:.2f}", sliced_colors
    )
    return new_cmap

def plot_frame(df, specs, raster, folder, tsi, step=0, show=False, mode="pif"):
    dstamp = df["DateTime"].iloc[step]
    dstamp_n = str(dstamp.datetime()).replace(".0", "")
    tmin = df["DateTime"].min()
    tmax = df["DateTime"].max()
    ls_dates = [tmin, tmax]
    dates2 = pd.to_datetime(np.linspace(tmin.datetime(), tmax.datetime(), 5), unit='s')
    # If you need them as a list
    ls_dates2 = dates2.tolist()

    dt = 10 / (24 * 60)

    # Deploy figure
    fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
    gs = mpl.gridspec.GridSpec(
        6, 16, wspace=0.8, hspace=0.3, left=0.08, bottom=0.15, top=0.85, right=0.98
    )
    fig.suptitle(f"PLANS model simulation -tsi {tsi} | date-time: {dstamp}")

    # inputs
    ax = plt.subplot(gs[:2, :9])
    plt.title("Infiltration flow", loc="left")
    plt.plot(df["DateTime"], df["Q_if"] / dt, c="purple")
    plt.ylim(0, 150)
    plt.xlim(ls_dates)
    plt.xticks(ls_dates2)
    plt.ylabel("mm/d")
    #plt.xlabel("date-time")
    plt.vlines(
        x=df["DateTime"].values[step],
        ymin=0,
        ymax=150,
        colors="red"
    )
    plt.text(
        x=df["DateTime"].values[step],
        y=120,
        s=" {} mm/d".format(round(df["Q_if"].values[step] / dt, 2)),
        color="red"
    )

    # storage
    ax = plt.subplot(gs[3:, :9])
    plt.title("Soil water storage", loc="left")
    plt.plot(df["DateTime"], df["G"], c="blue")
    plt.ylim(0, 50)
    plt.xlim(ls_dates)
    plt.xticks(ls_dates2)
    plt.ylabel("mm")
    plt.xlabel("date-time")
    plt.vlines(
        x=df["DateTime"].values[step],
        ymin=0,
        ymax=50,
        colors="red"
    )
    plt.text(
        x=df["DateTime"].values[step],
        y=44,
        s=" {} mm".format(round(df["G"].values[step], 2)),
        color="red"
    )

    ax = plt.subplot(gs[:, 8:])
    plt.title("Infiltration potential", loc="left")
    if mode == "rw":
        plt.title("Riparian wetlands", loc="left")
    im = plt.imshow(raster,
       cmap=specs["cmap"],
       vmin=specs["vmin"],
       vmax=specs["vmax"],
       interpolation="none",
    )
    if mode == "rw":
        pass
    else:
        cbar = fig.colorbar(im, shrink=0.5)
        cbar.set_label('mm')
    #plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    if show:
        plt.show()
    else:
        plt.savefig(
            f"./data/Local/outputs/{folder}/model_{dstamp_n}.jpg", dpi=150
        )
    plt.close(fig=fig)
    del fig
    return None

def compute_s(downscaling_grid, glb_s):
    #return np.maximum(0.0, np.minimum(s_cap - (glb_d + downscaling_grid), s_cap))
    return glb_s - downscaling_grid

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":

    dc_tests = {
        "hand": ["fhand-70m", 100, 10],
        "twi": ["ftwi-30u", 200, 15],
        "htwi": ["htwi-70-20", 200, 15]
    }

    print("loading")
    f_series = "./data/Local/inputs/_series.csv"
    df = pd.read_csv(f_series, sep=";", parse_dates=["DateTime"])
    print(df)
    # litte hack
    s_scale = 1.6

    df["G"] = (df["G"] - 5) * s_scale
    df["Q_if"] = (df["Q_if"] - 0) * s_scale

    r = Raster()
    meta = {
        'ncols': 98,
        'nrows': 113,
        'xllcorner': 0,
        'yllcorner': 0,
        'cellsize': 30,
        'NODATA_value': -1
    }
    r.set_raster_metadata(metadata=meta)
    r.view_specs["cmap"] = "gist_earth"
    s_cap = 80
    # m_param = 80
    m_param = 100 #200


    dc_folders = {
        "pif": s_cap, "rw": 1}

    for tst in dc_tests:
        # todo change source dir to project
        r.load_image(file="C:/gis/_projects_/plans/{}.tif".format(dc_tests[tst][0]))
        scalar_grid = dc_tests[tst][1] * (np.mean(r.data) - r.data)

        for s in dc_folders:
            gif_folder = f"{tst}/{s}"
            view_specs = {
                "width": 10,
                "height": 4,
                "suptitle": "HeY",
                "cmap": slice_cmap("gist_earth", start_slice=0.1, end_slice=0.9),
                "vmin": 0,
                "vmax": dc_folders[s],
            }

            steps = np.arange(0, len(df), step=100)

            #indexes = [10, 15, 20, 33, 40, 50]
            indexes = [i for i in range(len(steps))]
            for ind in indexes:
                step = steps[ind]
                print(f"{tst} -- {s} -- step: {step}")
                s_glb = df["G"].values[step]
                s_grid = s_cap - compute_s(downscaling_grid=scalar_grid, glb_s=s_glb)
                if s == "rw":
                    s_grid = np.where(s_grid <=dc_tests[tst][2], 0, 1)
                plot_frame(df, view_specs, tsi=tst.upper(), raster=s_grid, step=step, folder=gif_folder, show=False, mode=s)

            make_gif = True
            if make_gif:
                png_dir = f"./data/Local/outputs/{gif_folder}"
                gif_filename = "./data/Local/outputs/{}_{}_animation.gif".format(tst, s)
                image_paths = glob.glob(f"{png_dir}/model_*.jpg")
                frames = [iio.imread(path) for path in image_paths]
                iio.imwrite(gif_filename, frames, loop=0, duration=80)
                # clean up
                for fim in image_paths:
                    os.remove(fim)