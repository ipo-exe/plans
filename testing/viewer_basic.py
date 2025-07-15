import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plans import viewer

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def plot_data(fig, data, gs):
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("testing", loc="left")
    plt.scatter(data["X"], data["Y"], marker=".", s=0.3 * viewer.MM_TO_PT)
    ax1.set_ylabel("Y label")
    ax1.set_xlabel("X label 2")
    ax1.set_ylim(45, 155)
    ax1.set_xlim(45, 155)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("testing", loc="left")
    plt.scatter(data["X"], data["Y"], marker=".", s=0.3 * viewer.MM_TO_PT)
    ax2.set_ylabel("Y label")
    ax2.set_xlabel("X label 2")
    ax2.set_ylim(45, 155)
    ax2.set_xlim(45, 155)
    return fig

if __name__ == "__main__":
    ndata = 1000
    x = np.linspace(0, 100, num=ndata)
    v1 = np.random.normal(loc=100, scale=10, size=ndata)
    v2 = np.random.normal(loc=100, scale=10, size=ndata)

    df = pd.DataFrame({"X": v1, "Y": v2})

    specs = {
        "style": "dark",
        "nrows": 2,
        "ncols": 3,
        "dpi": 300,
        "width": viewer.FIG_SIZES["M"]["w"],
        "height": viewer.FIG_SIZES["M"]["h"],
    }
    specs.update(viewer.GRID_SPECS)

    fig, gs = viewer.build_fig(specs=specs)
    fig = plot_data(fig, df, gs)
    viewer.ship_fig(
        fig=fig,
        show=False,
        file_output="data/misc/outputs/_fig2.jpg",
        dpi=specs["dpi"]
    )
    print("passed.")