import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from plans.analyst import Univar, GeoUnivar
from plans import viewer

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    folder_src = "C:/gis/_losalamos/B000/B007_climasano"
    file_db = Path(f"{folder_src}/climasano_db.gpkg")
    # load data
    print("loading...")
    # Load ROI Polygon
    roi = gpd.read_file(file_db, layer="massa_dagua_clip")
    # Load ADA Polygon
    ada = gpd.read_file(file_db, layer="ada_simple")
    # Load Hex Bins
    hex = gpd.read_file(file_db, layer="hexbins_data_pch_C01")
    # point data
    points = gpd.read_file(file_db, layer="points_data_pch_C01")


    uv = GeoUnivar(name="tss")
    uv.varfield = "tss"
    uv.view_specs["yvar_field"] = uv.varfield
    uv.data = points[[uv.varfield, uv.field_geometry]]
    print("\ndata")
    print(uv.data.info())
    print(uv.data)
    uv.view_specs["scheme_cmap"] = "quantiles"
    uv.view_specs["cmap"] = "viridis"
    uv.view_specs["zorder_map"] = 3
    uv.view_specs["bins"] = 50
    uv.view_specs["empty_map"] = True
    uv.view_specs["mode"] = "mini"
    uv.view_specs["colorize_scatter"] = False

    fig = uv.view(return_fig=True)

    # add new stuff
    hex_plot = hex[[uv.varfield, uv.field_geometry]].copy()
    # classify hex
    bins = uv.get_bins(data=uv.data[uv.varfield].values, n_bins=5, scheme="quantiles")
    hex_plot[f"{uv.varfield}_class"] = Univar.classify(data=hex_plot[uv.varfield].values, bins=bins)
    print(hex.head())

    all_axes = fig.get_axes()
    ax_map = all_axes[0]
    roi.plot(ax=ax_map, zorder=1, color="steelblue", alpha=0.4)
    ada.plot(ax=ax_map, zorder=2, color="brown", alpha=0.6)
    hex_plot.plot(ax=ax_map, zorder=3, column=f"{uv.varfield}_class", alpha=1, cmap=uv.view_specs["cmap"])
    x_ext = [-51.35, -51.0]
    y_ext = [-30.3, -29.95]
    ls_xticks = [round(v, 1) for v in np.linspace(x_ext[0], x_ext[1], num=5)]
    ax_map.set_xticks(ls_xticks)
    ls_yticks = [round(v, 1) for v in np.linspace(y_ext[0], y_ext[1], num=4)]
    ax_map.set_yticks(ls_yticks)
    ax_map.set_xlim(x_ext)
    ax_map.set_ylim(y_ext)

    plt.show()


