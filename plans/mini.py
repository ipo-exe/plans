import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plans.analyst import Bivar

def _timer(t_start, t_end, dt):
    # datetime setup
    dtindex = pd.date_range(t_start, t_end, freq="{}H".format(dt))
    # time steps for numerical solution
    t_nsol = dt * np.linspace(0, len(dtindex), len(dtindex))
    return dtindex, t_nsol

def linear_bucket(
    s1_t0=10,
    k2=1, # in hours
    t_start="2020-01-01 00:00:0.00",
    t_end="2020-01-02 00:00:0.00",
    dt=1 # in hours
):
    # datetime setup
    dtindex, t_nsol = _timer(t_start=t_start, t_end=t_end, dt=dt)

    # get numerical solution
    s_nsol = np.zeros(shape=t_nsol.shape)
    s_nsol[0] = s1_t0

    for t in range(1, len(t_nsol)):
        s_nsol[t] = s_nsol[t-1] - ((1/k2) * s_nsol[t-1] * dt)

    # get analytical solution
    s_asol = s1_t0 * np.exp(-t_nsol / k2)

    rmse = Bivar.rmse(pred=s_nsol, obs=s_asol)

    df_out = pd.DataFrame(
        {
            "t": t_nsol,
            "DateTime": dtindex,
            "S1_NSol": s_nsol,
            "S1_ASol": s_asol,
        }
    )

    dict_output = {
        "RMSE": rmse,
        "Data": df_out,
        "k2": k2,
        "S1_t0": s1_t0,
        "DT": dt,
    }

    return dict_output


def single_stock(
    s1_t0=10,
    k1=30,  # in hours
    m1=0,
    k2=0.9,  # in hours
    m2=0,
    t_start="2020-01-01 00:00:0.00",
    t_end="2020-01-02 00:00:0.00",
    dt=1  # in hours

):
    # datetime setup
    dtindex, t_nsol = _timer(t_start=t_start, t_end=t_end, dt=dt)

    # get numerical solution

    # vector setup
    s_nsol = np.zeros(shape=t_nsol.shape)
    i_nsol = np.zeros(shape=t_nsol.shape)
    o_nsol = np.zeros(shape=t_nsol.shape)

    # initial conditions
    s_nsol[0] = s1_t0

    for t in range(1, len(t_nsol)):
        i_nsol[t - 1] = (1 / k1) * np.exp(m1 * s_nsol[t - 1])  * s_nsol[t - 1]
        o_nsol[t - 1] = (1 / k2) * np.exp(m2 * s_nsol[t - 1])  * s_nsol[t - 1]
        s_nsol[t] = s_nsol[t - 1] + i_nsol[t - 1] * dt - o_nsol[t - 1] * dt

    df_out = pd.DataFrame(
        {
            "t": t_nsol[:-1],
            "DateTime": dtindex[:-1],
            "S": s_nsol[:-1],
            "I": i_nsol[:-1],
            "O": o_nsol[:-1],
        }
    )

    dict_output = {
        "Data": df_out,
        "s1_k1": k1,
        "k2": k2,
        "S1_t0": s1_t0,
        "DT": dt,
    }

    return dict_output



class SingleStock():

    def __init__(self):
        self.varalias = "S"
        self.name = "MySingleStock"
        self.s_t0 = 10
        # INPUT parameters
        self.b1 = True
        self.k1 = 5  # in hours
        self.m1 = 0.0 # in level^-1
        # OUTPUT parameters
        self.b2 = True
        self.k2 = 20 # in hours
        self.m2 = 0.0
        self.start = "2020-01-01 00:00:0.00"
        self.end = "2020-01-02 00:00:0.00"
        self.dt = 1 # in hours
        self.units = "Un."

        # mutables
        self.data = None

        # view specs
        self._set_view_specs()

    def solve(self):
        # datetime setup
        dtindex, t_nsol = _timer(
            t_start=self.start,
            t_end=self.end,
            dt=self.dt
        )

        # get numerical solution
        if self.b1:
            c1 = 1.0
        else:
            c1 = 0.0

        if self.b2:
            c2 = 1.0
        else:
            c2 = 0.0

        # vector setup
        s_nsol = np.zeros(shape=t_nsol.shape)
        i_nsol = np.zeros(shape=t_nsol.shape)
        o_nsol = np.zeros(shape=t_nsol.shape)

        # initial conditions
        s_nsol[0] = self.s_t0

        # Euler Method loop
        for t in range(1, len(t_nsol)):
            # compute input flow rate
            i_nsol[t - 1] = c1 * (1 / self.k1) * np.exp(self.m1 * s_nsol[t - 1]) * s_nsol[t - 1]
            # compute output flow rate
            o_nsol[t - 1] = c2 * (1 / self.k2) * np.exp(self.m2 * s_nsol[t - 1]) * s_nsol[t - 1]
            # apply balance equation
            s_nsol[t] = s_nsol[t - 1] + i_nsol[t - 1] * self.dt - o_nsol[t - 1] * self.dt

        self.data = pd.DataFrame(
            {
                "t": t_nsol[:-1],
                "DateTime": dtindex[:-1],
                "S": s_nsol[:-1],
                "I": i_nsol[:-1],
                "O": o_nsol[:-1],
            }
        )

    def _set_view_specs(self):
        self.view_specs = {
            "color_s": "blueviolet",
            "color_i": "blue",
            "color_o": "red",
            "suptitle": "{}".format(self.name),
            "width": 4.0, # 5 * 1.618,
            "height": 5.0,
            "ylim_s": [0, 100],
            "ylim_io": [0, 25],
        }
        return None

    def view(self,
        show=True,
        folder="./output",
        filename=None,
        dpi=300,
        fig_format="jpg",
        suff=""):
        # Deploy figure
        fig = plt.figure(figsize=(self.view_specs["width"], self.view_specs["height"]))  # Width, Height
        gs = mpl.gridspec.GridSpec(
            2, 1,
            wspace=0.2,
            hspace=0.5,
            left=0.15,
            bottom=0.15,
            top=0.85,
            right=0.95
        )


        fig.suptitle(self.view_specs["suptitle"])
        # plot phase
        plt.subplot(gs[0, :])
        # plt.title("S")
        plt.plot(self.data["t"], self.data["S"], color=self.view_specs["color_s"], label="$S$")
        plt.ylim(self.view_specs["ylim_s"])
        plt.ylabel("{}".format(self.units))
        plt.legend(loc="best")

        # plt.ylabel("$S$")

        # plot
        plt.subplot(gs[1, :])
        # plt.title("I and O")
        plt.plot(self.data["t"], self.data["I"], color=self.view_specs["color_i"], label="$I$")
        plt.plot(self.data["t"], self.data["O"], color=self.view_specs["color_o"], label="$O$")
        plt.legend(loc="best")
        plt.ylabel("{} / h".format(self.units))
        plt.ylim(self.view_specs["ylim_io"])
        # plt.ylabel("$I, O$")
        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "{}_{}".format(self.varalias, self.name)
            # save figure
            plt.savefig(
                "{}/{}{}.{}".format(folder, filename, suff, fig_format), dpi=dpi
            )
            plt.close(fig)
        return None


class DoubleStock():

    def __init__(self):
        self.varalias = "S"
        self.name = "MyDoubleStock"

        self.start = "2020-01-01 00:00:0.00"
        self.end = "2020-01-02 00:00:0.00"
        self.dt = 1  # in hours
        self.units = "Un."

        # S1 -- RECURSO
        self.s1_t0 = 80
        # S1 INPUT parameters
        self.s1_b1 = True
        self.s1_k1 = 7  # in hours
        self.s1_m1 = 0.0 # in level^-1
        # S1 OUTPUT parameters
        self.s1_b2 = True
        self.s1_k2 = 10.5 # in hours
        self.s1_m2 = 0.0

        # S2 -- CAPITAL
        self.s2_t0 = 5
        # S2 INPUT parameters
        self.s2_b1 = True
        self.s2_k1 = 5  # in hours
        self.s2_m1 = 0.0  # in level^-1

        # extraction
        self.b_e = True
        self.s2_ke = 1.0  # in hours
        self.s2_me = -0.002  # in hours

        # S2 OUTPUT parameters
        self.s2_b2 = True
        self.s2_k2 = 4.5  # in hours
        self.s2_m2 = 0.05

        # mutables
        self.data = None

        # view specs
        self._set_view_specs()

    def solve(self):
        # datetime setup
        dtindex, t_nsol = _timer(
            t_start=self.start,
            t_end=self.end,
            dt=self.dt
        )

        # get numerical solution

        # handle switches

        # switches on S1
        if self.s1_b1:
            s1_c1 = 1.0
        else:
            s1_c1 = 0.0
        if self.s1_b2:
            s1_c2 = 1.0
        else:
            s1_c2 = 0.0

        # switches on extraction
        if self.b_e:
            s1_ce = 1.0
        else:
            s1_ce = 0.0

        # switches on S2
        if self.s2_b1:
            s2_c1 = 1.0
        else:
            s2_c1 = 0.0
        if self.s2_b2:
            s2_c2 = 1.0
        else:
            s2_c2 = 0.0

        # vector setup
        # S1
        s1 = np.zeros(shape=t_nsol.shape)
        s1_i = np.zeros(shape=t_nsol.shape)
        s1_o = np.zeros(shape=t_nsol.shape)
        s1_e = np.zeros(shape=t_nsol.shape)

        # S2
        s2 = np.zeros(shape=t_nsol.shape)
        s2_i = np.zeros(shape=t_nsol.shape)
        s2_o = np.zeros(shape=t_nsol.shape)

        # initial conditions
        s1[0] = self.s1_t0
        s2[0] = self.s2_t0

        # Euler Method loop
        for t in range(1, len(t_nsol)):
            # ---------- compute input flows rate ---------- #
            s1_i[t - 1] = s1_c1 * (1 / self.s1_k1) * np.exp(self.s1_m1 * s1[t - 1]) * s1[t - 1]
            s2_i[t - 1] = s2_c1 * (1 / self.s2_k1) * np.exp(self.s2_m1 * s2[t - 1]) * s2[t - 1]

            # ---------- compute output flow rate ---------- #
            # ---- s1

            # potential outputs
            s1_o_pot = s1_c2 * (1 / self.s1_k2) * np.exp(self.s1_m2 * s1[t - 1]) * s1[t - 1]

            eco_fric = (s1[t - 1] - 25) / s1[t - 1]
            s1_e_pot = s1_ce * (1 / self.s2_ke) * np.exp(self.s2_me * s2[t - 1]) * s2[t - 1] * eco_fric

            # compute actual total output on S1
            s1_ot_pot = s1_o_pot + s1_e_pot
            s1_ot = np.min([s1_ot_pot, s1[t - 1]/self.dt])

            if s1_ot_pot > 0:
                # distribute
                s1_o[t - 1] = s1_ot * s1_o_pot / s1_ot_pot
                s1_e[t - 1] = s1_ot * s1_e_pot / s1_ot_pot

            # ---- s2
            s2_o[t - 1] = s2_c2 * (1 / self.s2_k2) * np.exp(self.s2_m2 * s2[t - 1]) * s2[t - 1]

            # apply balance equation
            s1[t] = s1[t - 1] + s1_i[t - 1] * self.dt - s1_e[t - 1] * self.dt - s1_o[t - 1] * self.dt
            s2[t] = s2[t - 1] + s2_i[t - 1] * self.dt + s1_e[t - 1] * self.dt - s2_o[t - 1] * self.dt

        self.data = pd.DataFrame(
            {
                "t": t_nsol[:-1],
                "DateTime": dtindex[:-1],
                "S1": s1[:-1],
                "I1": s1_i[:-1],
                "O1": s1_o[:-1],
                "E": s1_e[:-1],
                "S2": s2[:-1],
                "I2": s2_i[:-1],
                "O2": s2_o[:-1],
            }
        )

    def _set_view_specs(self):
        self.view_specs = {
            "color_s": "blueviolet",
            "color_i": "blue",
            "color_o": "red",
            "color_s_alt": "green",
            "color_i_alt": "dodgerblue",
            "color_o_alt": "orange",
            "suptitle": "{}".format(self.name),
            "width": 4.0, # 5 * 1.618,
            "height": 5.0,
            "ylim_s": [0, 100],
            "ylim_io": [0, 50]
        }
        return None

    def view(self,
        show=True,
        folder="./output",
        filename=None,
        dpi=300,
        fig_format="jpg",
        suff=""):
        # Deploy figure
        fig = plt.figure(figsize=(self.view_specs["width"], self.view_specs["height"]))  # Width, Height
        gs = mpl.gridspec.GridSpec(
            2, 1,
            wspace=0.2,
            hspace=0.5,
            left=0.15,
            bottom=0.15,
            top=0.85,
            right=0.95
        )

        fig.suptitle(self.view_specs["suptitle"])
        # plot phase
        plt.subplot(gs[0, :])
        # plt.title("S")
        plt.plot(self.data["t"], self.data["S1"], color=self.view_specs["color_s"], label="$S_1$")
        plt.plot(self.data["t"], self.data["S2"], color=self.view_specs["color_s_alt"], label="$S_2$")
        plt.ylim(self.view_specs["ylim_s"])
        plt.ylabel("{}".format(self.units))
        plt.legend(loc="best")

        # plot
        plt.subplot(gs[1, :])
        # plt.title("I and O")
        plt.plot(self.data["t"], self.data["I1"], color=self.view_specs["color_i"], label="$I_1$")
        plt.plot(self.data["t"], self.data["I2"], color=self.view_specs["color_i_alt"], label="$I_2$")
        plt.plot(self.data["t"], self.data["O1"], color=self.view_specs["color_o"], label="$O_1$")
        plt.plot(self.data["t"], self.data["O2"], color=self.view_specs["color_o_alt"], label="$O_2$")
        plt.plot(self.data["t"], self.data["E"], color="black", label="$E$")
        plt.legend(loc="best", ncol=3)
        plt.ylabel("{} / h".format(self.units))
        plt.ylim(self.view_specs["ylim_io"])
        # plt.ylabel("$I, O$")
        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "{}_{}".format(self.varalias, self.name)
            # save figure
            plt.savefig(
                "{}/{}{}.{}".format(folder, filename, suff, fig_format), dpi=dpi
            )
            plt.close(fig)
        return None



class MiniPlans():

    def __init__(self):
        self.varalias = "S"
        self.name = "MyMiniPlans"
        self.units = "mm"

        # ---- DEFAULT PARAMETERS -----
        self.k1 = 20 # h
        self.k2 = 50 # h
        self.k3 = 30 # h

        self.s_a = 40 # mm
        self.s_c = 30 # mm
        self.s_2max = 200 # mm

        # ---- DEFAULT INITIAL CONDITIONS -----
        self.s1_t0 = 60 # mm
        self.s2_t0 = 120  # mm
        self.s3_t0 = 80  # mm

        self.start = None # datetime string
        self.end = None # datetime string
        self.dt = None  # in hours

        self.data = None
        self._set_view_specs()
    def solve(self, df_input, inplace=True):
        print("")
        df = df_input.copy()

        # stock variables
        s1 = np.zeros(len(df))
        h1 = np.zeros(len(df))
        s2 = np.zeros(len(df))
        d2 = np.zeros(len(df))
        s3 = np.zeros(len(df))

        # exogenous flows
        p = df["P"].values
        e = df["E"].values

        # endogenous flows
        q1 = np.zeros(len(df))
        q1_pot = np.zeros(len(df))
        q2 = np.zeros(len(df))
        q2_pot = np.zeros(len(df))
        q3 = np.zeros(len(df))
        q3_pot = np.zeros(len(df))
        r = np.zeros(len(df))
        r_pot = np.zeros(len(df))

        e1 = np.zeros(len(df))
        e1_pot = np.zeros(len(df))
        e2 = np.zeros(len(df))
        e2_pot = np.zeros(len(df))

        # changing parameters
        c1 = np.zeros(len(df))

        # initial conditions
        s1[0] = self.s1_t0
        s2[0] = self.s2_t0
        d2[0] = self.s_2max - s2[0]
        s3[0] = self.s3_t0

        # integration loop:
        for t in range(1, len(df)):
            # local dt
            dt = df["dt"].values[t]
            # last t
            t0 = t - 1

            # local head
            h1[t0] = np.max([0, s1[t0] - self.s_a])
            c1[t0] = h1[t0] / ((h1[t0] + self.s_c)) # /dt

            # compute potential flows
            e2_pot[t0] = np.min([e[t0], s2[t0]])
            e1_pot[t0] = e[t0] - e2_pot[t0]
            r_pot[t0] = c1[t0] * h1[t0]
            q1_pot[t0] = np.min([(1/ self.k1) * s1[t0], d2[t0]])
            q2_pot[t0] = (1 / self.k2) * s2[t0]
            q3_pot[t0] = (1 / self.k3) * s3[t0]

            # total outputs in s1
            o1_pot = r_pot[t0] + q1_pot[t0] + e1_pot[t0]
            o1 = np.min([o1_pot, s1[t0]])

            # compute actual output flows partition in s1
            r[t0] = r_pot[t0] * o1 / o1_pot
            q1[t0] = q1_pot[t0] * o1 / o1_pot
            e1[t0] = e1_pot[t0] * o1 / o1_pot

            # total outputs in s2
            o2_pot = q2_pot[t0] + e2_pot[t0]
            o2 = np.min([o2_pot, s2[t0]])

            # compute actual output flows partition in s2
            q2[t0] = q2_pot[t0] * o2 / o2_pot
            e2[t0] = e2_pot[t0] * o2 / o2_pot

            q3[t0] = q3_pot[t0]

            # apply balance
            s1[t] = s1[t0] + ((p[t0]) * dt) - ((q1[t0] + r[t0] + e1[t0]) * dt )
            s2[t] = s2[t0] + (q1[t0] * dt) - ((q2[t0] + e2[t0]) * dt)
            d2[t] = self.s_2max - s2[t0]
            s3[t] = s3[t0] + ((q2[t0] + r[t0]) * dt) - (q3[t0] * dt)

        # prepare output object
        df_out = pd.DataFrame(
            {
                "DateTime": df["DateTime"].values[:-1],
                "t": df["dt"].values[0] * df.index[:-1],
                "P": p[:-1],
                "E": e[:-1],
                "S1": s1[:-1],
                "S2": s2[:-1],
                "S3": s3[:-1],
                "H1": h1[:-1],
                "D2": d2[:-1],
                "C": c1[:-1],
                "R": r[:-1],
                "Q1": q1[:-1],
                "Q2": q2[:-1],
                "Q3": q3[:-1],
                "E1": e1[:-1],
                "E2": e2[:-1],
            }
        )
        if inplace:
            self.data = df_out.copy()
        else:
            return df_out


    def get_default_inputs(self, dt=0.5, kind="constant"):
        self.start = "2020-01-01 00:00:0.00"
        self.end = "2020-01-08 00:00:0.00"
        self.dt = dt  # in hours

        # datetime setup
        dtindex, t_nsol = _timer(
            t_start=self.start,
            t_end=self.end,
            dt=self.dt
        )

        p = np.zeros(shape=t_nsol.shape)
        e = np.zeros(shape=t_nsol.shape)

        if kind == "constant":
            p = p + 5.0 # mm/h
            e = e + 2.0 # mm /h ~ 10 mm/d
        elif kind == "zero":
            pass
        elif kind == "basic":
            from scipy.ndimage import gaussian_filter
            max_v = 9.5
            size = len(dtindex)
            f_sigma = 50
            # Create the initial signal
            v = np.zeros(size)
            v[int(len(v) / 5)] = max_v
            # Apply Gaussian filter
            filtered_signal = gaussian_filter(v, sigma=len(v) / f_sigma)
            # Normalize the signal to have a maximum value of max_v
            normalized_signal = (filtered_signal / np.max(filtered_signal)) * max_v
            p = normalized_signal * (normalized_signal > 0.01)

            # Parameters for the sinusoidal function
            _df = pd.DataFrame(
                {
                    "DateTime": dtindex
                }
            )
            max_value = 0.5  # Adjust as needed
            min_value = 0  # Adjust as needed

            A = (max_value - min_value) / 2
            D = (max_value + min_value) / 2
            B = 2 * np.pi / 24
            C = 15  # Since we want the peak at 3 PM, which is 15 in 24-hour format

            # Convert DateTime to hours and apply the sinusoidal function
            _df['sin_value'] = _df['DateTime'].apply(lambda x: A * np.sin(B * (x.hour - C)) + D)
            e = _df['sin_value'].values
            print(np.sum(e))
            print(np.sum(e)/7)

        df = pd.DataFrame(
            {
                "DateTime": dtindex,
                "P": p,
                "E": e
            }
        )
        # Calculate the difference between each row and its predecessor
        df['DeltaTime'] = df['DateTime'].diff()
        # Convert the Timedelta to hours (float value)
        df['dt'] = df['DeltaTime'].dt.total_seconds() / 3600
        # fix void
        df["dt"].values[0] = df["dt"].values[1]
        # remove deltatime
        df = df.drop(columns=["DeltaTime"])
        # organize columns
        df = df[["DateTime", "dt", "P", "E"]]
        return df

    def _set_view_specs(self):
        self.view_specs = {
            "color_s1": "indigo",
            "color_s2": "teal",
            "color_s3": "dodgerblue",
            "color_i": "blue",
            "color_o": "red",
            "color_s_alt": "green",
            "color_i_alt": "dodgerblue",
            "color_o_alt": "orange",
            "suptitle": "{}".format(self.name),
            "width": 11.0, # 5 * 1.618,
            "height": 10.0,
            "ylim_s": [0, 1.2 * self.s_2max],
            "ylim_io": [0, 10],
            "legend_y": 1.35
        }
        return None

    def view(self,
        show=True,
        extra=False,
        folder="./output",
        filename=None,
        dpi=300,
        fig_format="jpg",
        suff=""):

        # Deploy figure
        fig = plt.figure(figsize=(self.view_specs["width"], self.view_specs["height"]))  # Width, Height
        gs = mpl.gridspec.GridSpec(
            5, 3,
            wspace=0.25,
            hspace=0.8,
            left=0.06,
            bottom=0.10,
            top=0.9,
            right=0.95
        )
        fig.suptitle(self.view_specs["suptitle"])

        # FIRST COLUMN

        plt.subplot(gs[0, 0])
        # Define the parameters (Name, Set Value, Min, Max)
        parameters = [
            ("$k_1$", self.k1, 0, 100),
            ("$k_2$", self.k2, 0, 200),
            ("$k_3$", self.k3, 0, 100),
            ("$s_a$", self.s_a, 0, 100),
            ("$s_c$", self.s_c, 0, 300),
            ("$s_{2max}$", self.s_2max, 0, 300)
        ]
        ax = plt.gca()
        for i, (name, set_val, min_val, max_val) in enumerate(parameters):
            # Normalize the values again for plotting
            normalized_set_val = (set_val - min_val) / (max_val - min_val)

            # Plot the range line
            ax.plot([i, i], [0, 1], color='dimgray', linewidth=2)

            # Plot the set value with a larger marker
            ax.plot(i, normalized_set_val, marker='o', color="navy", markersize=8)
            # Add set value label
            ax.text(i + 0.15, normalized_set_val, f'{set_val:.0f}', ha='left', va='bottom', color='blue', fontsize=8)
            # min value
            ax.text(i, -0.05, f'{min_val:.0f}', ha='center', va='top', color='tab:gray', fontsize=8)
            # parameter name
            ax.text(i, -0.2, name, ha='center', va='top')
            # Add max value label at the top of each slider
            ax.text(i, 1.01, f'{max_val:.0f}', ha='center', va='bottom', color='tab:gray', fontsize=8)
        # Set the limits and labels
        ax.set_ylim(0, 1)  # Normalized scale
        ax.set_xlim(-1, len(parameters))
        ax.set_xticks([])  # Using custom text labels instead of x-ticks
        # Hide y-axis and grid lines
        ax.get_yaxis().set_visible(False)
        ax.grid(False)

        if extra:
            plt.subplot(gs[1, 0])
            # plt.title("exogenous flows", loc="left")
            plt.plot(self.data["t"], self.data["P"], color="tab:grey", label="$P$")
            plt.plot(self.data["t"], self.data["E"], color="darkred", label="$E_p$")
            plt.ylim(self.view_specs["ylim_io"])
            plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
            plt.ylabel("$mm/h$")
            plt.xlabel("$h$")

            plt.subplot(gs[2, 0])
            # plt.title("water stocks", loc="left")
            plt.plot(self.data["t"], self.data["S1"], color=self.view_specs["color_s1"], label="$S_1$", zorder=3)
            plt.plot(self.data["t"], self.data["S2"], color=self.view_specs["color_s2"], label="$S_2$", zorder=1)
            plt.plot(self.data["t"], self.data["S3"], color=self.view_specs["color_s3"], label="$S_3$", zorder=2)
            plt.ylim([0, 120])
            plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
            plt.ylabel("$mm$")
            plt.xlabel("$h$")

            plt.subplot(gs[3, 0])
            # plt.title("fast flow (overland flow)", loc="left")
            plt.plot(self.data["t"], self.data["R"], color="darkviolet", label="$R$")
            plt.plot(self.data["t"], self.data["Q2"], color="teal", label="$Q_2$")
            plt.ylim(self.view_specs["ylim_io"])
            plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
            plt.ylabel("$mm/h$")
            plt.xlabel("$h$")

            plt.subplot(gs[4, 0])
            # plt.title("discharge", loc="left")
            plt.plot(self.data["t"], self.data["Q3"], color="navy", label="$Q_3$")
            plt.ylim(self.view_specs["ylim_io"])
            plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
            plt.ylabel("$mm/h$")
            plt.xlabel("$h$")

        # GRID
        plt.subplot(gs[0, 1])
        #plt.title("exogenous flows", loc="left")
        plt.plot(self.data["t"], self.data["P"], color="tab:grey", label="$P$")
        plt.plot(self.data["t"], self.data["E"], color="darkred", label="$E_p$")
        plt.ylim(self.view_specs["ylim_io"])
        plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
        plt.ylabel("$mm/h$")
        plt.xlabel("$h$")

        plt.subplot(gs[1, 1])
        # plt.title("infiltration", loc="left")
        plt.plot(self.data["t"], self.data["Q1"], color="navy", label="$Q_1$")
        plt.ylim(self.view_specs["ylim_io"])
        plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
        plt.ylabel("$mm/h$")
        plt.xlabel("$h$")

        plt.subplot(gs[0, 2])
        #plt.title("water stocks", loc="left")
        plt.plot(self.data["t"], self.data["S1"], color=self.view_specs["color_s1"], label="$S_1$")
        plt.plot(self.data["t"], self.data["S2"], color=self.view_specs["color_s2"], label="$S_2$")
        plt.plot(self.data["t"], self.data["S3"], color=self.view_specs["color_s3"], label="$S_3$")
        plt.ylim(self.view_specs["ylim_s"])
        plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
        plt.ylabel("$mm$")
        plt.xlabel("$h$")

        plt.subplot(gs[1, 2])
        #plt.title("fast flow (overland flow)", loc="left")
        plt.plot(self.data["t"], self.data["R"], color="darkviolet", label="$R$")
        plt.ylim(self.view_specs["ylim_io"])
        plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
        plt.ylabel("$mm/h$")
        plt.xlabel("$h$")

        plt.subplot(gs[2, 1])
        # plt.title("deficit", loc="left")
        plt.plot(self.data["t"], self.data["D2"], color="black", label="$D_2$")
        plt.ylim(self.view_specs["ylim_s"])
        plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
        plt.ylabel("$mm$")
        plt.xlabel("$h$")

        plt.subplot(gs[2, 2])
        #plt.title("runoff coef.", loc="left")
        plt.plot(self.data["t"], self.data["C"], color="black", label="$hc_colors$")
        plt.ylim([0, 1])
        plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
        plt.ylabel("$h^{-1}$")
        plt.xlabel("$h$")

        plt.subplot(gs[3, 1])
        #plt.title("slow flow (baseflow)", loc="left")
        plt.plot(self.data["t"], self.data["Q2"], color="teal", label="$Q_2$")
        plt.ylim(self.view_specs["ylim_io"])
        plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
        plt.ylabel("$mm/h$")
        plt.xlabel("$h$")

        plt.subplot(gs[3, 2])
        #plt.title("discharge", loc="left")
        plt.plot(self.data["t"], self.data["Q3"], color="navy", label="$Q_3$")
        plt.ylim(self.view_specs["ylim_io"])
        plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
        plt.ylabel("$mm/h$")
        plt.xlabel("$h$")

        plt.subplot(gs[4, 1])
        #plt.title("evapotranspiration", loc="left")
        plt.plot(self.data["t"], self.data["E1"], color="tab:orange", label="$E_1$")
        plt.plot(self.data["t"], self.data["E2"], color="tab:green", label="$E_2$")
        plt.ylim([0, 2])
        plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
        plt.ylabel("$mm/h$")
        plt.xlabel("$h$")

        plt.subplot(gs[4, 2])
        #plt.title("evapotranspiration total", loc="left")
        plt.plot(self.data["t"], self.data["E1"] + self.data["E2"], color="magenta", label="$E$")
        plt.plot(self.data["t"], self.data["E"], color="darkred", label="$E_p$")
        plt.ylim([0, 2])
        plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
        plt.ylabel("$mm/h$")
        plt.xlabel("$h$")




        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "{}_{}".format(self.varalias, self.name)
            # save figure
            plt.savefig(
                "{}/{}{}.{}".format(folder, filename, suff, fig_format), dpi=dpi
            )
            plt.close(fig)
        return None


    def plot_principles(self,
        show=True,
        folder="./output",
        filename=None,
        dpi=300,
        fig_format="jpg",
        suff=""):
        # Deploy figure
        fig = plt.figure(figsize=(6, 5))  # Width, Height
        gs = mpl.gridspec.GridSpec(
            2, 2,
            wspace=0.5,
            hspace=0.4,
            left=0.1,
            bottom=0.12,
            top=0.95,
            right=0.95
        )

        def get_c(s, sc):
            return s / (s + sc)

        def get_r(s, sc, sa):
            v = (s - sa) * get_c(s=(s - sa) * (s > sa), sc=sc)
            return v * (s > sa)

        def get_s(s0, t, k):
            return s0 * np.exp(-(1/k) * t)

        # data setup
        s = np.linspace(0, 100, 100)

        lst_sc = [10, 20, 40]
        lst_sc2 = [10, 20, 40]
        lst_sa = [10, 15, 30]
        lst_k = [10, 25, 50]
        lst_colors = ["blueviolet", "blue", "teal"]
        lst_colors2 = ["navy", "teal", "darkgreen"]

        plt.subplot(gs[1, 1])
        # plt.title("exogenous flows", loc="left")
        for i in range(len(lst_sc)):
            c = get_c(s, lst_sc[i])
            plt.plot(s, c, color=lst_colors[i], label="$s_c = {}$".format(lst_sc[i]))
        plt.legend(loc="best", frameon=True, fancybox=True, facecolor="white", framealpha=0.7)
        plt.hlines(y=1, xmin=0, xmax=np.max(s), color="tab:grey", linestyles="--")
        plt.ylim([0, 1.1])
        plt.xlim([0, np.max(s)])
        #plt.legend(loc='upper right', bbox_to_anchor=(1, self.view_specs["legend_y"]), ncol=3)
        plt.ylabel("$hc_colors$")
        plt.xlabel("$S_1 - s_a$ ($mm$)")

        plt.subplot(gs[1, 0])
        # plt.title("exogenous flows", loc="left")
        plt.plot([0, np.max(s)], [0, np.max(s)], color="tab:grey", linestyle="--")
        for i in range(len(lst_sc2)):
            r = get_r(s, lst_sc2[i], lst_sa[i])
            lbl = "$s_a = {}$, $s_c = {}$".format(lst_sa[i], lst_sc2[i])
            plt.plot(s, r, color=lst_colors[i], label=lbl)
        plt.legend(loc="best", frameon=True, fancybox=True, facecolor="white", framealpha=0.7)
        plt.ylim([0, 1.2 * np.max(s)])
        plt.xlim([0, np.max(s)])
        plt.ylabel("$R$ ($mm/h$)")
        plt.xlabel("$S_1$ ($mm$)")

        lst_colors3 = ["navy", "blue", "dodgerblue"]
        t = np.linspace(0, 24 * 7, 100)
        plt.subplot(gs[0, 1])
        # plt.title("exogenous flows", loc="left")
        for i in range(len(lst_k)):
            s1 = get_s(s0=10, k=lst_k[i], t=t)
            lbl = "$k = {}$".format(lst_k[i])
            plt.plot(t, s1, color=lst_colors3[i], label=lbl)
        plt.legend(loc="best", frameon=True, fancybox=True, facecolor="white", framealpha=0.7)
        plt.ylim([0, 12])
        plt.xlim([0, np.max(t)])
        plt.xlabel("$h$")
        plt.ylabel("$S$ ($mm$)")

        plt.subplot(gs[0, 0])
        # plt.title("exogenous flows", loc="left")
        for i in range(len(lst_k)):
            q = (1/lst_k[i]) * s
            lbl = "$k = {}$".format(lst_k[i])
            plt.plot(s, q, color=lst_colors3[i], label=lbl)
        plt.legend(loc="best", frameon=True, fancybox=True, facecolor="white", framealpha=0.7)
        plt.ylim([0, 25])
        plt.xlim([0, np.max(s)])
        plt.ylabel("$Q$ ($mm/h$)")
        plt.xlabel("$S$ ($mm$)")

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "{}_{}".format(self.varalias, self.name)
            # save figure
            plt.savefig(
                "{}/{}{}.{}".format(folder, filename, suff, fig_format), dpi=dpi
            )
            plt.close(fig)
        return None



