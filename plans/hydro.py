"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all hydrology functions of PLANS.

Copyright (C) 2022 Iporã Brito Possantti
"""

import os.path
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plans.root import DataSet
from plans.datasets import TimeSeries, RainSeries
from plans.analyst import Bivar


class Model(DataSet):
    """The core ``Model`` base object. Expected to hold one :class:`pandas.DataFrame` as simulation data and
    a dictionary as parameters. This is a dummy object to be developed downstream.

    """

    def __init__(self, name="MyHydroModel", alias="HM01"):
        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)

        # overwriters
        self.object_alias = "HM"

        # variables
        self._set_model_vars()

        # parameters
        self.file_params = None  # global parameters
        self._set_model_params()

        # simulation data
        self.data = None

        # observed data
        self.file_data_obs = None
        self.data_obs = None

        # defaults
        self.dtfield = "DateTime"

        # evaluation parameters (model metrics)
        self.rmse = None
        self.rsq = None
        self.bias = None

        # expected structure for folder data:
        """
        self.workplace = {
            "inputs": {
                "params": {
                    "filepath": "/inputs/params.csv",
                    "description": "Global model parameters"
                }
            },

        }
        """
        # run file
        self.file_model = None

    def _set_fields(self):
        """Set fields names.
        Expected to increment superior methods.

        """
        # ------------ call super ----------- #
        super()._set_fields()
        # Attribute fields
        self.field_file_params = "File Parameters"
        self.field_folder_data = "Folder Data"
        self.field_rmse = "RMSE"
        self.field_rsq = "R2"
        self.field_bias = "Bias"

        # ... continues in downstream objects ... #

    def _set_model_vars(self):
        self.vars = {
            "t": {
                "units": "{k}",
                "description": "Accumulated time",
                "kind": "time",
            },
            "S": {
                "units": "mm",
                "description": "Storage level",
                "kind": "level",
            },
            "Q": {
                "units": "mm/{dt_freq}",
                "description": "Outflow",
                "kind": "flow",
            },
            "S_an": {
                "units": "mm",
                "description": "Storage level (analytical solution)",
                "kind": "level",
            },
            "S_obs": {
                "units": "mm",
                "description": "Storage level (observed evidence)",
                "kind": "level",
            },
        }
        self.var_eval = "S"

    def _set_model_params(self):
        """Intenal method for setting up model parameters data

        :return: None
        :rtype: None
        """
        # model parameters
        self.params = {
            "k": {
                "value": None,
                "units": None,
                "dtype": np.float64,
                "description": "Residence time",
                "kind": "conceptual",
            },
            "S0": {
                "value": None,
                "units": "mm",  # default is mm
                "dtype": np.float64,
                "description": "Storage initial condition",
                "kind": "conceptual",
            },
            "dt": {
                "value": None,
                "units": None,
                "dtype": np.float64,
                "description": "Time Step in k units",
                "kind": "procedural",
            },
            "dt_freq": {
                "value": None,
                "units": "unitless",
                "dtype": str,
                "description": "Time Step frequency flag",
                "kind": "procedural",
            },
            "t0": {
                "value": None,
                "units": "timestamp",
                "dtype": str,
                "description": "Simulation start",
                "kind": "procedural",
            },
            "tN": {
                "value": None,
                "units": "timestamp",
                "dtype": str,
                "description": "Simulation end",
                "kind": "procedural",
            },
        }
        self.reference_dt_param = "k"
        return None

    def get_metadata(self):
        """Get a dictionary with object metadata.
        Expected to increment superior methods.

        .. note::

            Metadata does **not** necessarily inclue all object attributes.

        :return: dictionary with all metadata
        :rtype: dict
        """
        # ------------ call super ----------- #
        dict_meta = super().get_metadata()

        # remove useless fields
        del dict_meta[self.field_file_data]

        # customize local metadata:
        dict_meta_local = {
            self.field_rmse: self.rmse,
            self.field_rsq: self.rsq,
            self.field_bias: self.bias,
            self.field_file_params: self.file_params,
            self.field_folder_data: self.folder_data,
            # ... continue if necessary
        }

        # update
        dict_meta.update(dict_meta_local)
        return dict_meta

    def setter(self, dict_setter):
        """Set selected attributes based on an incoming dictionary.
        This is calling the superior method using load_data=False.

        :param dict_setter: incoming dictionary with attribute values
        :type dict_setter: dict
        :return: None
        :rtype: None
        """
        super().setter(dict_setter, load_data=False)

        # set new attributes
        self.file_params = Path(dict_setter[self.field_file_params])
        self.folder_data = Path(dict_setter[self.field_folder_data])

        # ... continues in downstream objects ... #
        return None

    def update(self):
        """Refresh all mutable attributes based on data (includins paths).
        Base method. This is overwrinting the superior method.

        :return: None
        :rtype: None
        """
        # refresh all mutable attributes

        # (re)set fields
        self._set_fields()

        # (re)set view specs
        self._set_view_specs()

        # update major attributes
        if self.data is not None:
            # data size (rows)
            self.size = len(self.data)

        # ... continues in downstream objects ... #
        return None

    def update_dt(self):
        """Update Time Step value, units and tag to match the model time parameters (like k)

        :return: None
        :rtype: None
        """
        # handle input dt
        s_dt_unit_tag = str(self.params["dt"]["value"]) + self.params["dt"]["units"]

        # handle all good condition
        if s_dt_unit_tag == "1" + self.params[self.reference_dt_param]["units"]:
            pass
        else:
            # compute time step in k units
            ft_aux = Model.get_timestep_factor(
                from_unit=s_dt_unit_tag,
                to_unit="1" + self.params[self.reference_dt_param]["units"],
            )
            # update
            self.params["dt"]["value"] = ft_aux
            self.params["dt"]["units"] = self.params[self.reference_dt_param]["units"][:]
            self.params["dt_freq"]["value"] = s_dt_unit_tag

        return None

    def load_params(self):
        """Load parameter data

        :return: None
        :rtype: None
        """
        # -------------- load parameter data -------------- #

        # >>> develop logic in downstream objects

        return None

    def load_data(self):
        """Load simulation data. Expected to overwrite superior methods.

        :return: None
        :rtype: None
        """

        # -------------- load simulation data -------------- #

        # >>> develop logic in downstream objects

        # -------------- update other mutables -------------- #
        self.update()

        # ... continues in downstream objects ... #

        return None

    def load(self):
        """Load parameters and data

        :return: None
        :rtype: None
        """
        self.load_params()
        self.load_data()
        return None

    def setup(self):
        """Set model simulation. Expected to be incremented downstream.

        .. warning::

            This method overwrites model data.


        :return: None
        :rtype: None
        """
        # ensure to update dt
        self.update_dt()

        # get timestep series
        vc_ts = Model.get_timestep_series(
            start_time=self.params["t0"]["value"],
            end_time=self.params["tN"]["value"],
            time_unit=self.params["dt_freq"]["value"],
        )
        vc_t = np.linspace(
            start=0,
            stop=len(vc_ts) * self.params["dt"]["value"],
            num=len(vc_ts),
            dtype=np.float64,
        )

        # set simulation dataframe
        self.data = pd.DataFrame(
            {
                "DateTime": vc_ts,
                "t": vc_t,
            }
        )

        # append variables to dataframe
        # >>> develop logic in downstream objects

        # set initial conditions
        # >>> develop logic in downstream objects

        return None

    def solve(self):
        """Solve the model for boundary and initial conditions by numerical methods.

        .. warning::

            This method overwrites model data.

        :return: None
        :rtype: None
        """
        # >>> develop logic in downstream objects
        return None

    def evaluate(self):
        """Evaluate model.

        :return: None
        :rtype: None
        """
        # >>> develop logic in downstream objects
        return None

    def run(self):
        """Simulate model.

        :return: None
        :rtype: None
        """
        self.setup()
        self.solve()
        self.update()
        return None

    def export(self, folder, filename):
        """Export object resources

        :param folder: path to folder
        :type folder: str
        :param filename: file name without extension
        :type filename: str
        :return: None
        :rtype: None
        """
        # export model simulation data
        super().export(folder, filename=filename)
        # export model observation data
        # >>> develop in downstream objects

        # ... continues in downstream objects ... #

    def save(self, folder):
        """Save to sourced files is not allowed for Model() family. Use .export() instead.
        This is overwriting superior methods.

        :return: None
        :rtype: None
        """
        return None

    def view(self, show=True):
        # >>> develop logic in downstream object
        return None

    @staticmethod
    def get_timestep_factor(from_unit: str, to_unit: str) -> float:
        """Calculates the conversion factor between two time units.

        For instance, to find out how many days are in a given number of '10min' intervals,
        this function provides the multiplier.

        :param from_unit: The starting time unit (e.g., '10min', '15s').
        :type from_unit: str
        :param to_unit: The desired time unit (e.g., 'days', 'min').
        :type to_unit: str
        :return: The factor to convert from the starting to the desired time unit.
        :rtype: float
        """
        from_duration = pd.Timedelta(from_unit)
        to_duration = pd.Timedelta(to_unit)
        factor = to_duration.total_seconds() / from_duration.total_seconds()
        return 1 / factor

    @staticmethod
    def get_timestep_series(
        start_time: str, end_time: str, time_unit: str
    ) -> pd.DatetimeIndex:
        """Generates a time series of timestamps.

        Creates a sequence of timestamps between a start and end time,
        with a specified frequency.

        :param start_time: The starting timestamp for the series (e.g., '2024-01-01', '2024-01-01 00:00:00').
        :type start_time: str
        :param end_time: The ending timestamp for the series (e.g., '2024-01-10', '2024-01-10 23:59:59').
        :type end_time: str
        :param time_unit: The frequency or interval between timestamps (e.g., 'D' for day, 'H' for hour, '10min').
        :type time_unit: str
        :return: A pandas DatetimeIndex representing the generated time series.
        :rtype: pd.DatetimeIndex
        """

        time_series = pd.date_range(start=start_time, end=end_time, freq=time_unit)
        return time_series

    @staticmethod
    def get_gaussian_signal(value_max, size, sigma=50, position_factor=5):
        """Generates a vector of normally (gaussian) distributed values. Useful for testing input inflows.

        :param value_max: actual maximum value in the vector
        :type value_max: float
        :param size: size of vector (time series size)
        :type size: int
        :param sigma: standard deviation for gaussian (normal) distribution.
        :type sigma: float
        :param position_factor: where to place the peak in the vector
        :type position_factor: float
        :return: vector of values
        :rtype: Numpy array
        """
        from scipy.ndimage import gaussian_filter

        # Create the initial signal
        v = np.zeros(size)
        # place peak
        v[int(len(v) / position_factor)] = value_max
        # Apply Gaussian filter
        filtered_signal = gaussian_filter(v, sigma=len(v) / sigma)
        # Normalize the signal to have a maximum value of max_v
        normalized_signal = (filtered_signal / np.max(filtered_signal)) * value_max
        v_norm = normalized_signal * (normalized_signal > 0.01)
        return v_norm


class LinearStorage(Model):
    """This is the Linear Storage Model. No inflow, only outflows. Simulates storage decay only."""

    def __init__(self, name="MyLinearStorage", alias="LS01"):
        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "LS"

    def _set_model_vars(self):
        self.vars = {
            "t": {
                "units": "{k}",
                "description": "Accumulated time",
                "kind": "time",
            },
            "S": {
                "units": "mm",
                "description": "Storage level",
                "kind": "level",
            },
            "Q": {
                "units": "mm/{dt_freq}",
                "description": "Outflow",
                "kind": "flow",
            },
            "S_an": {
                "units": "mm",
                "description": "Storage level (analytical solution)",
                "kind": "level",
            },
            "S_obs": {
                "units": "mm",
                "description": "Storage level (observed evidence)",
                "kind": "level",
            },
        }
        self.var_eval = "S"

    def _set_model_params(self):
        """Intenal method for setting up model parameters data

        :return: None
        :rtype: None
        """
        # model parameters
        self.params = {
            "k": {
                "value": None,
                "units": "D",
                "dtype": np.float64,
                "description": "Residence time",
                "kind": "conceptual",
            },
            "S0": {
                "value": None,
                "units": "mm",  # default is mm
                "dtype": np.float64,
                "description": "Storage initial condition",
                "kind": "conceptual",
            },
            "dt": {
                "value": None,
                "units": None,
                "dtype": np.float64,
                "description": "Time Step in k units",
                "kind": "procedural",
            },
            "dt_freq": {
                "value": None,
                "units": "unitless",
                "dtype": str,
                "description": "Time Step frequency flag",
                "kind": "procedural",
            },
            "t0": {
                "value": None,
                "units": "timestamp",
                "dtype": str,
                "description": "Simulation start",
                "kind": "procedural",
            },
            "tN": {
                "value": None,
                "units": "timestamp",
                "dtype": str,
                "description": "Simulation end",
                "kind": "procedural",
            },
        }
        return None

    def load_params(self):
        """Load parameter data

        :return: None
        :rtype: None
        """
        # load dataframe
        df_ = pd.read_csv(
            self.file_params, sep=self.file_csv_sep, encoding="utf-8", dtype=str
        )
        ls_input_params = set(df_["Parameter"])

        # parse to dict
        for k in self.params:
            if k in set(ls_input_params):
                self.params[k]["value"] = self.params[k]["dtype"](
                    df_.loc[df_["Parameter"] == k, "Value"].values[0]
                )
                self.params[k]["units"] = df_.loc[
                    df_["Parameter"] == k, "Units"
                ].values[0]

        # handle input dt
        self.update_dt()

        return None

    def load_data(self):
        """Load simulation data. Expected to overwrite superior methods.

        :return: None
        :rtype: None
        """
        # -------------- load observation data -------------- #
        file_qobs = Path(f"{self.folder_data}/S_obs.csv")
        self.data_obs = pd.read_csv(
            file_qobs,
            sep=self.file_csv_sep,
            encoding=self.file_encoding,
            parse_dates=[self.dtfield],
        )

        # -------------- update other mutables -------------- #
        self.update()

        # ... continues in downstream objects ... #

        return None

    def setup(self):
        """Set model simulation.

        .. warning::

            This method overwrites model data.


        :return: None
        :rtype: None
        """
        # setup superior object
        super().setup()

        # append variables to dataframe
        self.data["S"] = np.nan
        self.data["Q"] = np.nan
        self.data["S_a"] = np.nan

        # set initial conditions
        self.data["S"].values[0] = self.params["S0"]["value"]

        return None

    def solve(self):
        """Solve the model for input and initial conditions by numerical methods.

        .. warning::

            This method overwrites model data.

        :return: None
        :rtype: None
        """
        # make simpler variables for clarity
        k = self.params["k"]["value"]
        dt = self.params["dt"]["value"]
        df = self.data.copy()
        n_steps = len(df)

        # --- analytical solution
        df["S_a"].values[:] = self.params["S0"]["value"] * np.exp(-df["t"].values / k)

        # --- numerical solution
        # loop over (Euler Method)
        for t in range(n_steps - 1):
            # Qt = dt * St / k
            df["Q"].values[t] = df["S"].values[t] * dt / k
            df["S"].values[t + 1] = df["S"].values[t] - df["Q"].values[t]

        # reset data
        self.data = df.copy()

        return None

    def get_evaldata(self):
        # merge simulation and observation data based
        df = pd.merge(left=self.data, right=self.data_obs, on=self.dtfield, how="left")

        # remove other columns
        df = df[[self.dtfield, self.var_eval, "{}_obs".format(self.var_eval)]]

        # remove voids
        df.dropna(inplace=True)

        # handle scaling factor for flows (expected to be in k-units)
        factor = 1.0
        if self.vars[self.var_eval]["kind"] == "flow":
            factor = self.params["dt"]["value"]
        # scale the factor
        df["{}_obs".format(self.var_eval)] = df["{}_obs".format(self.var_eval)] * factor

        return df

    def evaluate(self):
        """Evaluate model metrics.

        :return: None
        :rtype: None
        """
        df = self.get_evaldata()

        # evaluate paired data
        vc_pred = df[self.var_eval].values
        vc_obs = df["{}_obs".format(self.var_eval)].values

        # compute evaluation metrics
        self.rmse = Bivar.rmse(pred=vc_pred, obs=vc_obs)
        self.rsq = Bivar.rsq(pred=vc_pred, obs=vc_obs)
        self.bias = Bivar.bias(pred=vc_pred, obs=vc_obs)

        return None

    def export(self, folder, filename, views=False):
        """Export object resources

        :param folder: path to folder
        :type folder: str
        :param filename: file name without extension
        :type filename: str
        :return: None
        :rtype: None
        """
        # export model simulation data
        super().export(folder, filename=filename + "_sim")

        # export model observation data
        df_obs = self.get_evaldata()
        fpath = Path(folder + "/" + filename + "_obs" + self.file_csv_ext)
        df_obs.to_csv(
            fpath, sep=self.file_csv_sep, encoding=self.file_encoding, index=False
        )

        # export views
        if views:
            self.view_specs["folder"] = folder
            self.view_specs["filename"] = filename
            self.view(show=False)

        print(filename)
        # ... continues in downstream objects ... #

    def view(self, show=True):
        specs = self.view_specs.copy()

        ts = TimeSeries(name=self.name, alias=self.alias)
        ts.set_data(input_df=self.data, input_dtfield=self.dtfield, input_varfield="S")
        ts.view_specs["title"] = "Linear Model - R2: {}".format(round(self.rsq, 2))
        fig = ts.view(show=False, return_fig=True)
        # obs series plot
        # access axes from fig
        axes = fig.get_axes()
        ax = axes[0]
        ax.plot(self.data_obs[self.dtfield], self.data_obs["S_obs"], ".", color="k")
        if show:
            plt.show()
        else:
            file_path = "{}/{}.{}".format(
                specs["folder"], specs["filename"], specs["fig_format"]
            )
            plt.savefig(file_path, dpi=specs["dpi"])
            plt.close(fig)


class LSPQ(LinearStorage):
    """This is a Rainfall-Runoff model based on a Linear Storage.
    Inflow (P) in a necessary data input.

    """

    def __init__(self, name="MyLSPQ", alias="LSPQ001"):
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "LSPQ"

        # input data
        self.data_input = None

    def _set_model_params(self):
        """Intenal method for setting up model parameters data

        :return: None
        :rtype: None
        """
        super()._set_model_params()
        # model parameters
        self.params.update(
            {
                "Q_max": {
                    "value": None,
                    "units": "mm/{dt_freq}",
                    "dtype": np.float64,
                    "description": "Downstream limiting control for Q",
                    "kind": "conceptual",
                },
            }
        )

        return None

    def _set_model_vars(self):
        self.vars = {
            "t": {
                "units": "{k}",
                "description": "Accumulated time",
                "kind": "time",
            },
            "P": {
                "units": "mm/{dt_freq}",
                "description": "Inflow",
                "kind": "flow",
            },
            "S": {
                "units": "mm",
                "description": "Storage level",
                "kind": "level",
            },
            "Q": {
                "units": "mm/{dt_freq}",
                "description": "Outflow",
                "kind": "flow",
            },
            "Q_obs": {
                "units": "mm/d",
                "description": "Outflow (observed evidence)",
                "kind": "flow",
            },
        }
        self.var_eval = "Q"
        self.var_inputs = ["P"]
        return None

    def load_data(self):
        """Load simulation data. Expected to overwrite superior methods.

        :return: None
        :rtype: None
        """
        # -------------- load input data -------------- #
        file_input = Path(f"{self.folder_data}/clim.csv")
        df_data_input = pd.read_csv(
            file_input,
            sep=self.file_csv_sep,
            encoding=self.file_encoding,
            parse_dates=[self.dtfield],
        )
        # Format the datetime column to string (e.g., 'YYYY-MM-DD HH:MM:SS')
        df_data_input["DT_str"] = df_data_input[self.dtfield].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        # set t0 and tN
        self.params["t0"]["value"] = df_data_input["DT_str"].values[0]
        self.params["tN"]["value"] = df_data_input["DT_str"].values[-1]
        # set the data input
        self.data_input = df_data_input[[self.dtfield] + self.var_inputs].copy()
        # -------------- load observation data -------------- #
        file_qobs = Path(f"{self.folder_data}/q_obs.csv")
        self.data_obs = pd.read_csv(
            file_qobs,
            sep=self.file_csv_sep,
            encoding=self.file_encoding,
            parse_dates=[self.dtfield],
        )

        # -------------- update other mutables -------------- #
        self.update()

        # ... continues in downstream objects ... #

        return None

    def setup(self):
        """Set model simulation.

        .. warning::

            This method overwrites model data.


        :return: None
        :rtype: None
        """
        # setup superior object
        super().setup()

        # drop superior meaningless columns
        self.data.drop(columns=["S_a"], inplace=True)

        # handle input P values
        rs = RainSeries()
        rs.set_data(
            input_df=self.data_input, input_varfield="P", input_dtfield="DateTime"
        )
        df_downscaled = rs.downscale(freq=self.params["dt_freq"]["value"])

        # set interpolated input variables
        self.data["P"] = df_downscaled["P"].values

        # organize table
        ls_vars = [self.dtfield, "t", "P", "Q", "S"]
        self.data = self.data[ls_vars].copy()

        return None

    def solve(self):
        """Solve the model for input and initial conditions by numerical methods.

        .. warning::

            This method overwrites model data.

        :return: None
        :rtype: None
        """
        # make simpler variables for clarity
        k = self.params["k"]["value"]
        qmax = self.params["Q_max"]["value"]
        dt = self.params["dt"]["value"]
        df = self.data.copy()
        n_steps = len(df)

        # --- numerical solution
        # loop over (Euler Method)
        for t in range(n_steps - 1):
            # Qt = dt * St / k
            df["Q"].values[t] = np.min([df["S"].values[t] * dt / k, qmax])
            # S(t + 1) = S(t) + P(t) - Q(t)
            df["S"].values[t + 1] = (
                df["S"].values[t] + df["P"].values[t] - df["Q"].values[t]
            )

        # reset data
        self.data = df.copy()

        return None

    def view(self, show=True, return_fig=False):
        specs = self.view_specs.copy()
        # start plot
        fig = plt.figure(figsize=(6, 6))  # Width, Height
        plt.suptitle("Model LSPQ - R2: {}".format(round(self.rsq, 4)))
        # grid
        gs = mpl.gridspec.GridSpec(
            3,
            1,
            wspace=0.5,
            hspace=0.5,
            left=0.2,
            right=0.9,
            bottom=0.1,
            top=0.9,
        )  # nrows, ncols

        first_date = self.data[self.dtfield].values[0]
        last_date = self.data[self.dtfield].values[-1]
        ls_dates = [first_date, last_date]
        n_flow_max = np.max([self.data["P"].max(), self.data["Q"].max()])

        # ------------ P plot ------------
        n_pmax = self.data["P"].max() / self.params["dt"]["value"]
        ax = fig.add_subplot(gs[0, 0])
        plt.plot(self.data[self.dtfield], self.data["P"] / self.params["dt"]["value"], color="tab:gray")
        plt.title("P", loc="left")
        ax.set_xticks(ls_dates)
        plt.xlim(ls_dates)
        plt.ylim(0, 1.1 * n_pmax)
        plt.ylabel("mm/{}".format(self.params["dt"]["units"].lower()))

        # ------------ Q plot ------------
        n_qmax = np.max([self.data["Q"].max() / self.params["dt"]["value"], self.data_obs["Q_obs"].max()])
        ax = fig.add_subplot(gs[1, 0])
        plt.plot(self.data[self.dtfield], self.data["Q"] / self.params["dt"]["value"], color="blue")
        plt.plot(
            self.data_obs[self.dtfield],
            self.data_obs["Q_obs"],
            ".",
            color="indigo",
        )
        plt.title("Q", loc="left")
        ax.set_xticks(ls_dates)
        plt.xlim(ls_dates)
        plt.ylim(0, 1.1 * n_qmax)
        plt.ylabel("mm/{}".format(self.params["dt"]["units"].lower()))

        # ------------ S plot ------------
        ax = fig.add_subplot(gs[2, 0])
        plt.plot(self.data[self.dtfield], self.data["S"], color="black")
        plt.title("S", loc="left")
        ax.set_xticks(ls_dates)
        plt.xlim(ls_dates)
        plt.ylabel("mm")
        plt.ylim(0, 1.1 * self.data["S"].max())

        # handle return
        if return_fig:
            return fig
        else:
            if show:
                plt.show()
            else:
                file_path = "{}/{}.{}".format(
                    specs["folder"], specs["filename"], specs["fig_format"]
                )
                plt.savefig(file_path, dpi=specs["dpi"])
                plt.close(fig)


class LSPQE(LSPQ):
    """This is a Rainfall-Runoff model based on a Linear Storage.
    Inflow (P) and External flow (E) are necessary data input.

    """

    def __init__(self, name="MyLSPQE", alias="LSPQE001"):
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "LSPQE"

    def _set_model_vars(self):
        super()._set_model_vars()
        self.vars.update(
            {
                "E": {
                    "units": "mm/{dt_freq}",
                    "description": "External Outflow",
                    "kind": "flow",
                },
                "E_pot": {
                    "units": "mm/{dt_freq}",
                    "description": "Maximal External Outflow",
                    "kind": "flow",
                },
            }
        )
        self.var_inputs = ["P", "E_pot"]

    def setup(self):
        """Set model simulation.

        .. warning::

            This method overwrites model data.


        :return: None
        :rtype: None
        """
        # setup superior object
        super().setup()

        # set E (actual)
        self.data["E"] = np.nan

        # handle input E_pot values
        rs = RainSeries() # todo best practice is to have a EvapoSeries() object
        rs.varfield = "E_pot"
        rs.varname = "E_pot"
        rs.set_data(
            input_df=self.data_input, input_varfield="E_pot", input_dtfield="DateTime"
        )
        df_downscaled = rs.downscale(freq=self.params["dt_freq"]["value"])

        # set interpolated input variables
        self.data["E_pot"] = df_downscaled["E_pot"].values

        # organize table
        ls_vars = [self.dtfield, "t", "P", "E_pot", "E", "Q", "S"]
        self.data = self.data[ls_vars].copy()

        return None

    def solve(self):
        """Solve the model for input and initial conditions by numerical methods.

        .. warning::

            This method overwrites model data.

        :return: None
        :rtype: None
        """
        # make simpler variables for clarity
        k = self.params["k"]["value"]
        qmax = self.params["Q_max"]["value"]
        dt = self.params["dt"]["value"]
        df = self.data.copy()
        n_steps = len(df)

        # --- numerical solution
        # loop over (Euler Method)
        for t in range(n_steps - 1):
            # Qt = dt * St / k
            q_pot = np.min([df["S"].values[t] * dt / k, qmax])

            # Et = Et_pot
            e_pot = df["E_pot"].values[t]

            # Potential outflow O_pot
            o_pot = q_pot + e_pot

            # Maximum outflow Omax = St
            o_max = df["S"].values[t]

            # actual outflow
            o_act = np.min([o_max, o_pot])

            # allocate Q and E
            df["Q"].values[t] = o_act * (q_pot / o_pot)
            df["E"].values[t] = o_act * (e_pot / o_pot)

            # S(t + 1) = S(t) + P(t) - Q(t) - E(t)
            df["S"].values[t + 1] = (
                df["S"].values[t]
                + df["P"].values[t]
                - df["Q"].values[t]
                - df["E"].values[t]
            )
        # reset data
        self.data = df.copy()

        return None

    def view(self, show=True, return_fig=False):
        specs = self.view_specs.copy()
        fig = super().view(show=False, return_fig=True)
        plt.suptitle("Model LSFAS - R2: {}".format(round(self.rsq, 4)))
        axes = fig.get_axes()
        # P
        ax = axes[0]
        ax.set_title("P and E", loc="left")
        ax.plot(self.data[self.dtfield], self.data["E"] / self.params["dt"]["value"], c="red")

        # handle return
        if return_fig:
            return fig
        else:
            if show:
                plt.show()
            else:
                file_path = "{}/{}.{}".format(
                    specs["folder"], specs["filename"], specs["fig_format"]
                )
                plt.savefig(file_path, dpi=specs["dpi"])
                plt.close(fig)



class LSFAS(LSPQE):
    """This is a Rainfall-Runoff model based on a Linear Storage with Fill-And-Spill (FAS) mechanism.
    Inflow Precipitation (P) and Potential Evapotranspiration (E_pot) are necessary data input.

    """

    def __init__(self, name="MyLSFAS", alias="LSFAS001"):
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "LSFAS"

    def _set_model_vars(self):
        super()._set_model_vars()
        self.vars.update(
            {
                "Q_s": {
                    "units": "mm/{dt_freq}",
                    "description": "Spill flow (outflow)",
                    "kind": "flow",
                },
                "Q_s_f": {
                    "units": "mm/mm",
                    "description": "Spill flow coefficient",
                    "kind": "constant",
                },
                "Q_b": {
                    "units": "mm/{dt_freq}",
                    "description": "Base flow (outflow)",
                    "kind": "flow",
                },
            }
        )
        return None

    def _set_model_params(self):
        """Intenal method for setting up model parameters data

        :return: None
        :rtype: None
        """
        super()._set_model_params()
        # clean useless model parameters from upstream
        del self.params["Q_max"]
        # add new params
        self.params.update(
            {
                "s_a": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Activation level for storage spill",
                    "kind": "conceptual",
                },
                "s_c": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Effective fragmentantion level of storage",
                    "kind": "conceptual",
                },
            }
        )
        self.reference_dt_param = "k"
        return None

    def setup(self):
        """Set model simulation.

        .. warning::

            This method overwrites model data.


        :return: None
        :rtype: None
        """
        # setup superior object
        super().setup()

        # add new columns
        self.data["Q_s"] = np.nan
        self.data["Q_b"] = np.nan
        self.data["Q_s_f"] = np.nan

        # organize table
        ls_vars = [self.dtfield, "t", "P", "E_pot", "E", "Q_s", "Q_s_f", "Q_b", "Q", "S"]
        self.data = self.data[ls_vars].copy()

        return None
    def solve(self):
        """Solve the model for input and initial conditions by numerical methods.

        .. warning::

            This method overwrites model data.

        :return: None
        :rtype: None
        """
        # make simpler variables for clarity
        k = self.params["k"]["value"]
        s_a = self.params["s_a"]["value"]
        s_c = self.params["s_c"]["value"]
        # dt is the fraction of 1 Day/(1 simulation time step)
        dt = self.params["dt"]["value"]
        df = self.data.copy()
        n_steps = len(df)

        # --- numerical solution
        # loop over (Euler Method)
        for t in range(n_steps - 1):
            # Compute dynamic spill storage capacity in mm/D
            ss_cap = np.max([0, df["S"].values[t] - s_a]) # spill storage = s - s_a

            # compute dynamic runoff capacity
            qs_cap = ss_cap * dt # multiply by dt (figure out why!)

            # Compute runoff coeficient
            qs_f = (ss_cap / (ss_cap + (s_c + s_a))) # effective s_c = s_c + s_a
            df["Q_s_f"].values[t] = qs_f

            # potential runoff -- scale by the coefficient
            qs_pot = qs_f * qs_cap

            # Compute potential base flow
            # Qt = dt * St / k
            qb_pot = df["S"].values[t] * dt / k

            # Compute potential evaporation flow
            # Et = E_pot
            e_pot = df["E_pot"].values[t]

            # Compute Potential outflow O_pot
            o_pot = qs_pot + qb_pot + e_pot

            # Compute Maximum outflow Omax = St
            o_max = df["S"].values[t]

            # Compute actual outflow
            o_act = np.min([o_max, o_pot])

            # Allocate outflow
            df["Q_s"].values[t] = o_act * (qs_pot / o_pot)
            df["Q_b"].values[t] = o_act * (qb_pot / o_pot)
            df["E"].values[t] = o_act * (e_pot / o_pot)

            # Compute full Q
            df["Q"].values[t] = df["Q_s"].values[t] + df["Q_b"].values[t]

            # Apply Water balance
            # S(t + 1) = S(t) + P(t) - Qs(t) - Qb(t) - E(t)
            df["S"].values[t + 1] = (
                df["S"].values[t]
                + df["P"].values[t]
                - df["Q_s"].values[t]
                - df["Q_b"].values[t]
                - df["E"].values[t]
            )
        # reset data
        self.data = df.copy()

        return None


    def view(self, show=True, return_fig=False):
        specs = self.view_specs.copy()
        fig = super().view(show=False, return_fig=True)
        plt.suptitle("Model LSFAS - R2: {}".format(round(self.rsq, 4)))
        axes = fig.get_axes()
        # Q
        ax = axes[1]
        ax.set_title("Q, Qb and Qobs", loc="left")
        ax.plot(self.data[self.dtfield], self.data["Q_b"] / self.params["dt"]["value"], c="navy")
        ax.set_ylim(0, 800)

        ax.set_title("Runoff C", loc="right")
        ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
        ax2.plot(self.data[self.dtfield], self.data["Q_s_f"], 'm--')
        ax2.set_ylim(0, 1)

        # handle return
        if return_fig:
            return fig
        else:
            if show:
                plt.show()
            else:
                file_path = "{}/{}.{}".format(
                    specs["folder"], specs["filename"], specs["fig_format"]
                )
                plt.savefig(file_path, dpi=specs["dpi"])
                plt.close(fig)

class Global(LSFAS):
    """This is the global Rainfall-Runoff model of PLANS. Simulates the the catchment
    as if is a global system. Precipitation (P) and Potential Evapotranspiration (E_pot) are necessary data input.

    """
    def __init__(self, name="MyGlobal", alias="Glo001"):
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "Global"

    def _set_model_vars(self):
        super()._set_model_vars()
        # clean some vars
        del self.vars["S"]
        del self.vars["Q"]
        del self.vars["Q_b"]
        del self.vars["Q_s_f"]
        del self.vars["Q_s"]
        # update
        self.new_vars = {
                #
                #
                # Canopy variables
                "C": {
                    "units": "mm",
                    "description": "Canopy storage",
                    "kind": "level",
                },
                "E_c": {
                    "units": "mm/{dt_freq}",
                    "description": "Canopy evaporation",
                    "kind": "flow",
                },
                "P_tf": {
                    "units": "mm/{dt_freq}",
                    "description": "Throughfall -- canopy spill water",
                    "kind": "flow",
                },
                "P_tf_f": {
                    "units": "mm/mm",
                    "description": "Throughfall coefficient",
                    "kind": "constant",
                },
                "P_sf": {
                    "units": "mm/{dt_freq}",
                    "description": "Stemflow -- canopy drained water",
                    "kind": "flow",
                },
                "P_c": {
                    "units": "mm/{dt_freq}",
                    "description": "Effective precipitation",
                    "kind": "flow",
                },
                #
                #
                # Surface variables
                "S": {
                    "units": "mm",
                    "description": "Surface storage",
                    "kind": "level",
                },
                "E_s": {
                    "units": "mm/{dt_freq}",
                    "description": "Surface evaporation",
                    "kind": "flow",
                },
                "Q_of": {
                    "units": "mm/{dt_freq}",
                    "description": "Surface spill flow -- surface overland flow",
                    "kind": "flow",
                },
                "Q_of_f": {
                    "units": "mm/mm",
                    "description": "Surface spill flow coefficient",
                    "kind": "constant",
                },
                "Q_uf": {
                    "units": "mm/{dt_freq}",
                    "description": "Subsurface spill flow -- lateral flow on topsoil",
                    "kind": "flow",
                },
                "Q_uf_f": {
                    "units": "mm/mm",
                    "description": "Surface spill flow coefficient",
                    "kind": "constant",
                },
                "Q_if": {
                    "units": "mm/{dt_freq}",
                    "description": "Infiltration flow -- surface drainage flow",
                    "kind": "flow",
                },
                #
                #
                # Vadose vars
                "V": {
                    "units": "mm",
                    "description": "Vadose zone storage",
                    "kind": "level",
                },
                "E_v": {
                    "units": "mm/{dt_freq}",
                    "description": "Vadose zone transpiration",
                    "kind": "flow",
                },
                "Q_v": {
                    "units": "mm",
                    "description": "Recharge flow",
                    "kind": "flow",
                },
                #
                #
                # Phreatic vars
                "G": {
                    "units": "mm",
                    "description": "Phreatic zone storage",
                    "kind": "level",
                },
                "D": {
                    "units": "mm",
                    "description": "Phreatic zone storage deficit",
                    "kind": "level",
                },
                "E_g": {
                    "units": "mm/{dt_freq}",
                    "description": "Phreatic zone transpiration",
                    "kind": "flow",
                },
                "Q_g": {
                    "units": "mm/{dt_freq}",
                    "description": "Baseflow from Groundwater/Phreatic zone",
                    "kind": "flow",
                },
                # routing
                "R": {
                    "units": "mm/{dt_freq}",
                    "description": "Slope runoff (surface, subsurface and baseflow)",
                    "kind": "flow",
                },
                "Q": {
                    "units": "mm/{dt_freq}",
                    "description": "River discharge at stage station",
                    "kind": "flow",
                },
            }
        self.vars.update(self.new_vars.copy())

        self.vars_canopy = [
            "P", "P_c", "P_sf", "P_tf", "P_tf_f", "E_pot", "E_c", "C"
        ]

        return None

    def _set_model_params(self):
        """Intenal method for setting up model parameters data

        :return: None
        :rtype: None
        """
        super()._set_model_params()
        # clean useless model parameters from upstream
        del self.params["k"]
        del self.params["s_a"]
        del self.params["s_c"]
        del self.params["S0"]

        # add new params
        self.params.update(
            {
                #
                #
                # initial conditions
                "C0": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Canopy storage initial conditions",
                    "kind": "conceptual",
                },
                "S0": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Surface storage initial conditions",
                    "kind": "conceptual",
                },
                "V0": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Vadose zone initial conditions",
                    "kind": "conceptual",
                },
                "G0": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Phreatic zone initial conditions",
                    "kind": "conceptual",
                },
                #
                #
                # Canopy parameters
                "C_k": {
                    "value": None,
                    "units": "D",
                    "dtype": np.float64,
                    "description": "Canopy residence time",
                    "kind": "conceptual",
                },
                "C_a": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Activation level for canopy spill",
                    "kind": "conceptual",
                },
                "C_c": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Fragmentantion level of canopy",
                    "kind": "conceptual",
                },
                #
                #
                # Surface parameters
                "S_k": {
                    "value": None,
                    "units": "D",
                    "dtype": np.float64,
                    "description": "Surface residence time",
                    "kind": "conceptual",
                },
                "Q_f_cap": {
                    "value": None,
                    "units": "mm/D",
                    "dtype": np.float64,
                    "description": "Infiltration capacity",
                    "kind": "conceptual",
                },
                # subsurface spill
                "S_au": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Activation level for subsurface spill",
                    "kind": "conceptual",
                },
                "S_cu": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Activation level for subsurface spill",
                    "kind": "conceptual",
                },
                # surface spill
                "S_ao": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Activation level for surface overland spill",
                    "kind": "conceptual",
                },
                "S_co": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Activation level for surface overland spill",
                    "kind": "conceptual",
                },
                # recharge
                "K": {
                    "value": None,
                    "units": "mm/D",
                    "dtype": np.float64,
                    "description": "Effective hydraulic conductivity",
                    "kind": "conceptual",
                },
                # baseflow
                "D_max": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Maximal Deficit for root transpiration (effective root depth)",
                    "kind": "level",
                },
                "G_k": {
                    "value": None,
                    "units": "D",
                    "dtype": np.float64,
                    "description": "Phreatic zone residence time",
                    "kind": "conceptual",
                },
                "G_cap": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Phreatic zone storage capacity",
                    "kind": "conceptual",
                },
            }
        )
        # reset reference
        self.reference_dt_param = "G_k"
        return None


    def setup(self):
        """Set model simulation.

        .. warning::

            This method overwrites model data.


        :return: None
        :rtype: None
        """
        # setup superior object
        super().setup()

        # clear deprecated columns
        self.data.drop(
            columns=["Q_s", "Q_s_f", "Q_b", "S"],
            inplace=True
        )

        # append variables
        for v in self.new_vars:
            self.data[v] = np.nan
            # set initial conditions on storages
            if self.vars[v]["kind"] == "level":
                if v == "D":
                    pass
                else:
                    self.data[v].values[0] = self.params["{}0".format(v)]["value"]
        # compute deficit initial condition
        self.data["D"].values[0] = self.params["G_cap"]["value"] - self.data["G"].values[0]

        # organize table?

        return None


    def solve(self):
        """Solve the model for input and initial conditions by numerical methods.

        .. warning::

            This method overwrites model data.

        :return: None
        :rtype: None
        """
        # --- simulation setup
        # dt is the fraction of 1 Day/(1 simulation time step)
        dt = self.params["dt"]["value"]
        # global processes data
        GB = self.data.copy()
        n_steps = len(GB)
        # todo remove hacks
        #n_steps = 10

        # --- parameter setup ---

        # [Canopy] canopy parameters
        C_k = self.params["C_k"]["value"]
        C_a = self.params["C_a"]["value"]
        C_c = self.params["C_c"]["value"]
        # [Canopy] Compute effective canopy activation level
        C_a_eff = C_a # same
        # [Canopy] Compute effective canopy fragmentation level
        C_c_eff = C_c  # same
        # [Surface] surface parameters
        Q_f_cap = self.params["Q_f_cap"]["value"]
        S_k = self.params["S_k"]["value"]
        S_ao = self.params["S_ao"]["value"]
        S_co = self.params["S_co"]["value"]
        S_au = self.params["S_au"]["value"]
        S_cu = self.params["S_cu"]["value"]
        # [Surface] Compute effective overland flow activation level
        S_ao_eff = S_au + S_ao  # incremental with underland (topsoil)
        # [Surface] Compute effective overland flow fragmentation level
        S_co_eff = S_co  # incremental with activation
        # [Surface] Compute effective underland flow activation level
        S_au_eff = S_au  # same
        # [Surface] Compute effective underland flow fragmentation level
        S_cu_eff = S_cu  #

        # --- numerical solution
        # loop over (Euler Method)
        for t in range(n_steps - 1):

            # [Evaporation] ---- get evaporation flows first ---- #

            # [Evaporation] canopy
            E_c_pot = GB["E_pot"].values[t]
            E_c_cap = GB["C"].values[t] * dt  # multiply by dt
            GB["E_c"].values[t] = 0.0# np.min([E_c_pot, E_c_cap])
            # apply discount on the fly
            GB["C"].values[t] = GB["C"].values[t] - GB["E_c"].values[t]

            # todo include root zone depth rationale
            # [Evaporation] phreatic zone
            E_g_pot = GB["E_pot"].values[t] - GB["E_c"].values[t]  # discount actual Ec
            E_g_cap = GB["G"].values[t] * dt  # multiply by dt
            GB["E_g"].values[t] = np.min([E_g_pot, E_g_cap])
            # apply discount on the fly
            GB["G"].values[t] = GB["G"].values[t] - GB["E_g"].values[t]

            # [Evaporation] vadose zone
            E_v_pot = GB["E_pot"].values[t] - GB["E_c"].values[t] - GB["E_g"].values[t]
            E_v_cap = GB["V"].values[t] * dt  # multiply by dt
            GB["E_v"].values[t] = np.min([E_v_pot, E_v_cap])
            # apply discount on the fly
            GB["V"].values[t] = GB["V"].values[t] - GB["E_v"].values[t]

            # [Evaporation] surface
            E_s_pot = GB["E_pot"].values[t] - GB["E_c"].values[t] - GB["E_g"].values[t] - GB["E_v"].values[t]
            E_s_cap = GB["S"].values[t] * dt  # multiply by dt
            GB["E_s"].values[t] = np.min([E_s_pot, E_s_cap])
            # apply discount on the fly
            GB["S"].values[t] = GB["S"].values[t] - GB["E_s"].values[t]

            # [Canopy] ---- Solve canopy water balance ---- #
            # todo resume here ---- REDO CANOPY MODEL
            # [Canopy] ---- Potential flows

            # [Canopy] -- Throughfall

            # [Canopy] Compute canopy spill storage capacity
            C_ss_cap = np.max([0, GB["C"].values[t] - C_a_eff])

            # [Canopy] Compute dynamic throughfall capacity
            P_tf_cap = C_ss_cap * dt  # multiply by dt

            # [Canopy] Compute throughfall coeficient
            P_tf_f = (C_ss_cap / (C_ss_cap + C_c_eff))
            GB["P_tf_f"].values[t] = P_tf_f

            # [Canopy] Compute potential throughfall -- scale by the coefficient
            P_tf_pot = P_tf_f * GB["P"].values[t] # todo P_tf_f * P_tf_cap

            # [Canopy] -- Stemflow
            # [Canopy] Compute potential stemflow -- spill storage does not contribute
            P_sf_pot = (GB["C"].values[t] - C_ss_cap) * dt / C_k  # Linear storage Q = dt * S / k

            # [Canopy] -- Full potential outflow
            C_out_pot = P_tf_pot + P_sf_pot # by pass + E_c_pot

            # [Canopy] ---- Actual flows

            # [Canopy] Compute canopy maximum outflow
            C_out_max = GB["C"].values[t]

            # [Canopy] Compute Actual outflow
            C_out_act = np.min([C_out_max, C_out_pot])

            # [Canopy] Allocate outflows
            with np.errstate(divide='ignore', invalid='ignore'):
                GB["P_tf"].values[t] = C_out_act * np.where(C_out_pot == 0, 0, P_tf_pot / C_out_pot)
                GB["P_sf"].values[t] = C_out_act * np.where(C_out_pot == 0, 0, P_sf_pot / C_out_pot)
                GB["P_c"].values[t] = GB["P_tf"].values[t] + GB["P_sf"].values[t]

            # [Canopy Water Balance] ---- Apply water balance
            # C(t+1) = C(t) + P(t) - Psf(t) - P(tf) >>> Ec(t) discounted earlier
            GB["C"].values[t + 1] = GB["C"].values[t] + GB["P"].values[t] - GB["P_sf"].values[t] - GB["P_tf"].values[t]

            # [Surface] ---- Solve surface water balance ---- #

            # [Surface] ---- Potential flows

            # [Surface -- overland] -- Overland flow

            # [Surface -- overland] Compute surface overland spill storage capacity
            Sof_ss_cap = np.max([0, GB["S"].values[t] - S_ao_eff])

            # [Surface -- overland] Compute dynamic overland flow capacity
            Q_of_cap = Sof_ss_cap * dt  # multiply by dt

            # [Surface -- overland] Compute overland flow coeficient
            Q_of_f = Sof_ss_cap / (Sof_ss_cap + S_co_eff)
            GB["Q_of_f"].values[t] = Q_of_f

            # [Surface -- overland] Compute potential overland -- scale by the coefficient
            Q_of_pot = Q_of_f * Q_of_cap

            # [Surface -- underland] -- Underland flow

            # [Surface -- underland] Compute surface underland spill storage capacity
            Suf_ss_cap = np.max([0, GB["S"].values[t] - S_au_eff])

            # [Surface -- underland] Compute dynamic underland flow capacity
            Q_uf_cap = Suf_ss_cap * dt  # multiply by dt

            # [Surface -- underland] Compute underland flow coeficient
            Q_uf_f = Suf_ss_cap / (Suf_ss_cap + S_cu_eff)
            GB["Q_uf_f"].values[t] = Q_uf_f

            # [Surface -- underland] Compute potential overland -- scale by the coefficient
            Q_uf_pot = Q_uf_f * Q_uf_cap

            # [Surface -- infiltration] -- Infiltration flow
            # linear store upper bounded by infiltration capacity
            # todo integrate with downstream soil water control
            Q_if_pot = np.min([GB["S"].values[t] * dt / S_k, Q_f_cap])

            # [Surface] -- Full potential outflow
            S_out_pot = Q_of_pot + Q_uf_pot + Q_if_pot

            # [Surface] ---- Actual flows
            # [Canopy] Compute surface maximum outflow
            S_out_max = GB["S"].values[t]

            # [Surface] Compute Actual outflow
            S_out_act = np.min([S_out_max, S_out_pot])

            # [Surface] Allocate outflows
            with np.errstate(divide='ignore', invalid='ignore'):
                GB["Q_of"].values[t] = S_out_act * np.where(S_out_pot == 0, 0, Q_of_pot / S_out_pot)
                GB["Q_uf"].values[t] = S_out_act * np.where(S_out_pot == 0, 0, Q_uf_pot / S_out_pot)
                GB["Q_if"].values[t] = S_out_act * np.where(S_out_pot == 0, 0, Q_if_pot / S_out_pot)

            # [Surface Water Balance] ---- Apply water balance
            GB["S"].values[t + 1] = GB["S"].values[t] + GB["P_c"].values[t] - GB["Q_of"].values[t] - GB["Q_uf"].values[t] - GB["Q_if"].values[t]

            # todo CONTINUE HERE FOR SURFACE SPILLS
            # Compute compounded flows
            GB["Q_g"].values[t] = 0.0

            # R - Hillslope runoff
            GB["R"].values[t] = GB["Q_of"].values[t] + GB["Q_uf"].values[t] + GB["Q_g"].values[t]
            # E - Total E
            GB["E"].values[t] = GB["E_c"].values[t] + GB["E_s"].values[t] + GB["E_v"].values[t] + GB["E_g"].values[t]

            # hacking schemes
            # todo remove hacks
            GB["V"].values[t + 1] = GB["V"].values[t]
            GB["G"].values[t + 1] = GB["G"].values[t]


        # reset data
        self.data = GB.copy()

        return None


    def print_canopy(self):
        df = self.data.copy()
        ls = [self.dtfield] + self.vars_canopy
        df = df[ls]
        print(df.round(3))

if __name__ == "__main__":
    print("Hi")
