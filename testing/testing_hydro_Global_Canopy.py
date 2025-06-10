import os
import pprint as pp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from plans.hydro import Global

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def plot_canopy(df, n_dt):
    # start plot
    fig = plt.figure(figsize=(6, 6))  # Width, Height
    plt.suptitle("Canopy | Model Global")
    # grid
    gs = mpl.gridspec.GridSpec(
        3,
        1,
        wspace=0.5,
        hspace=0.5,
        left=0.15,
        right=0.85,
        bottom=0.1,
        top=0.9,
    )  # nrows, ncols

    first_date = df["DateTime"].values[0]
    last_date = df["DateTime"].values[-1]
    ls_dates = [first_date, last_date]
    n_flow_max = np.max([df["P"].max(), df["Q"].max()])

    # ------------ P plot ------------
    n_pmax = df["P"].max() / n_dt
    ax = fig.add_subplot(gs[0, 0])
    plt.plot(df["DateTime"], df["P"] / n_dt, color="tab:gray", zorder=0)
    plt.title("P ({} mm)".format(round(df["P"].sum(), 1)), loc="left")
    ax.set_xticks(ls_dates)
    plt.xlim(ls_dates)
    plt.ylim(0, 1.1 * n_pmax)
    plt.ylabel("mm/d")

    # ------------ E plot ------------
    e_sum = round(df["E_c"].sum(), 1)
    ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
    ax2.plot(df["DateTime"], df["E_c"] / n_dt, c="tab:red", zorder=1)
    e_max = df["E_c"].max()
    if e_max == 0:
        ax2.set_ylim(-1, 1)
    else:
        ax2.set_ylim(0, 1.2 * e_max / n_dt)
    ax2.set_title("$E_c$ ({} mm)".format(e_sum), loc="right")
    ax2.set_ylabel("mm/d")

    # ------------ P_s plot ------------
    n_qmax = df["P_s"].max()
    # n_qmax = df["Q"].max() / n_dt
    q_sum = round(df["P_s"].sum(), 1)
    ax = fig.add_subplot(gs[1, 0])
    plt.plot(df["DateTime"], df["P_s"] / n_dt, color="blue")
    plt.plot(df["DateTime"], df["P_sf"] / n_dt, color="green")
    plt.title("$P_s$ ({} mm)".format(q_sum), loc="left")
    ax.set_xticks(ls_dates)
    plt.xlim(ls_dates)
    #plt.ylim(0, 1.1 * n_qmax)
    plt.ylabel("mm/d")

    # ------------ C plot ------------
    ax = fig.add_subplot(gs[2, 0])
    plt.plot(df["DateTime"], df["C"], color="black")
    plt.title("C", loc="left")
    ax.set_xticks(ls_dates)
    plt.xlim(ls_dates)
    plt.ylabel("mm")
    plt.ylim(0, 1.1 * df["C"].max())
    plt.show()


def print_canopy(m):
    df = m.data.copy()
    ls = [m.dtfield] + m.vars_canopy
    df = df[ls]
    print(df.round(3))


if __name__ == "__main__":
    # instantiate
    m = Global(name="Global", alias="GLB000")
    m.boot(bootfile="./data/bootfile_Global.csv")
    m.load()
    m.setup()
    m.solve()
    #pp.pprint(m.params)
    print(m.vars.keys())
    print_canopy(m)

    plot_canopy(df=m.data, n_dt=m.params["dt"]["value"])

    '''
    # setup
    m.setup()
    # simulate with epot
    m.shutdown_epot = False
    m.shutdown_qb = False
    '''


