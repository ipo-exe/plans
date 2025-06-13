"""
Description:
    The ``hydro`` module provides objects useful for hydrology simulation.

License:
    This software is released under the GNU General Public License v3.0 (GPL-3.0).
    For details, see: https://www.gnu.org/licenses/gpl-3.0.html

Overview
--------

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis.
Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Pellentesque habitant morbi tristique senectus
et netus et malesuada fames ac turpis egestas.

>>> from plans import hydro

Class aptent taciti sociosqu ad litora torquent per
conubia nostra, per inceptos himenaeos. Nulla facilisi. Mauris eget nisl
eu eros euismod sodales. Cras pulvinar tincidunt enim nec semper.

Example
--------

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis.

.. code-block:: python

    # todo [code example]
    print("hello world!)


Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl.

"""

import os.path
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import matplotlib.dates as mdates
from plans.root import DataSet
from plans.datasets import TimeSeries, RainSeries
from plans.analyst import Bivar


class Model(DataSet):
    # todo [optimize for DRY] move this class to root.py and make more abstract for all Models. Here can be a HydroModel(Model).
    """The core ``Model`` base object. Expected to hold one :class:`pandas.DataFrame` as simulation data and
    a dictionary as parameters. This is a dummy class to be developed downstream.

    """

    def __init__(self, name="MyModel", alias="HM01"):
        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)

        # defaults
        self.dtfield = "DateTime"

        # overwriters
        self.object_alias = "M"

        # variables
        self._set_model_vars()

        # parameters
        self.file_params = None  # global parameters
        self._set_model_params()

        # simulation data
        self.data = None

        # observed data
        self.filename_data_obs = "q_obs.csv"
        self.file_data_obs = None
        self.data_obs = None

        # evaluation parameters (model metrics)
        self.rmse = None
        self.rsq = None
        self.bias = None

        # run file
        self.file_model = None

        # flags
        self.update_dt_flag = True

        # testing helpers
        self.n_steps = None

        self._set_view_specs()

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

    def _set_view_specs(self):
        """Set view specifications.
        Expected to increment superior methods.

        :return: None
        :rtype: None
        """
        super()._set_view_specs()
        # cleanup useless entries
        dc_remove = {
            "xvar": "RM",
            "yvar": "TempDB",
            "xlabel": "RM",
            "ylabel": "TempDB",
            "color": self.color,
            "xmin": None,
            "xmax": None,
            "ymin": None,
            "ymax": None,
        }
        for k in dc_remove:
            del self.view_specs[k]
        # add new specs
        self.view_specs.update(
            {
                "width": 6,
                "height": 6,
            }
        )
        return None

    def get_params(self):
        # todo [Make Docstring]
        ls_param = [p for p in self.params]
        ls_values = [self.params[p]["value"] for p in self.params]
        ls_units = [self.params[p]["units"] for p in self.params]
        df = pd.DataFrame(
            {
                "Parameter": ls_param,
                "Value": ls_values,
                "Units": ls_units
            }
        )
        return df

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
        ## self._set_view_specs()

        # update major attributes
        if self.data is not None:
            # data size (rows)
            self.size = len(self.data)

        # ... continues in downstream objects ... #
        return None

    def update_dt(self):
        """Update Time Step value, units and tag to match the model reference time parameter (like k)

        :return: None
        :rtype: None
        """
        # this flag prevents revisting the function unintentionally
        # todo actually this is more like a bad bugfix so it would be nice
        #  to remove the flag and optimize this process.
        if self.update_dt_flag:
            # handle input dt
            s_dt_unit_tag = str(self.params["dt"]["value"]) + self.params["dt"]["units"]

            # handle all good condition
            if s_dt_unit_tag == "1" + self.params[self.reference_dt_param]["units"]:
                pass
            else:
                #
                #
                # compute time step in reference units
                ft_aux = Model.get_timestep_factor(
                    from_unit=s_dt_unit_tag,
                    to_unit="1" + self.params[self.reference_dt_param]["units"],
                )
                #
                #
                # update
                self.params["dt"]["value"] = ft_aux
                self.params["dt"]["units"] = self.params[self.reference_dt_param]["units"][:]
                self.params["dt_freq"]["value"] = s_dt_unit_tag
            # shut down
            self.update_dt_flag = False
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
        # first params
        self.load_params()
        # then data
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

    def run(self, setup_model=True):
        """Simulate model (full procedure).

        :param setup_model: flag for setting up data (default=True)
        :type setup_model: bool
        :return: None
        :rtype: None
        """
        if setup_model:  # by-passing setup may save computing time
            self.setup()
        self.solve()
        self.update()
        self.evaluate()
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

        # export model parameter file
        df_params = self.get_params()
        df_params.to_csv(f"{folder}/{filename}_params.csv", sep=";", index=False)

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
        raw_time_series = pd.date_range(start=start_time, end=end_time, freq=time_unit)
        time_series = pd.to_datetime(
            np.where(raw_time_series.microsecond > 0,
                     raw_time_series.floor('s') + pd.Timedelta(seconds=1),
                     raw_time_series)
        )
        return time_series

    @staticmethod
    def get_gaussian_signal(value_max, size, sigma=50, position_factor=5, reverse=False):
        """Generates a vector of normally (gaussian) distributed values. Useful for testing input inflows.

        :param value_max: actual maximum value in the vector
        :type value_max: float
        :param size: size of vector (time series size)
        :type size: int
        :param sigma: standard deviation for gaussian (normal) distribution.
        :type sigma: float
        :param position_factor: where to place the peak in the vector
        :type position_factor: float
        :param reverse: boolean to reverse position in order
        :type reverse: bool
        :return: vector of values
        :rtype: Numpy array
        """
        from scipy.ndimage import gaussian_filter

        # Create the initial signal
        v = np.zeros(size)
        # place peak
        v[int(len(v) / position_factor)] = value_max
        if reverse:
            v = v[::-1]
        # Apply Gaussian filter
        filtered_signal = gaussian_filter(v, sigma=len(v) / sigma)
        # Normalize the signal to have a maximum value of max_v
        normalized_signal = (filtered_signal / np.max(filtered_signal)) * value_max
        v_norm = normalized_signal * (normalized_signal > 0.01)
        return v_norm


class LinearStorage(Model):
    """Linear Storage Model. This is a Toy Model. No inflow, only outflows. Simulates storage decay only."""

    def __init__(self, name="MyLinearStorage", alias="LS01"):
        # todo [docstring]
        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "LS"
        # expected filenames
        self.filename_data_obs = "S_obs.csv"

    def _set_model_vars(self):
        # todo [docstring]
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
        self.reference_dt_param = "k"
        return None

    def load_params(self):
        """Load parameter data from parameter file (stored in `self.file_params`)

        .. warning::

            This method overwrites model data.

        :return: None
        :rtype: None
        """
        # load dataframe
        df_ = pd.read_csv(
            self.file_params, sep=self.file_csv_sep, encoding="utf-8", dtype=str
        )
        ls_input_params = set(df_["Parameter"])

        # parse to dict
        for p in self.params:
            if p in set(ls_input_params):
                # set value
                self.params[p]["value"] = self.params[p]["dtype"](
                    df_.loc[df_["Parameter"] == p, "Value"].values[0]
                )
                # units
                self.params[p]["units"] = df_.loc[
                    df_["Parameter"] == p, "Units"
                ].values[0]
                # min
                self.params[p]["min"] = self.params[p]["dtype"](df_.loc[
                    df_["Parameter"] == p, "Min"
                ].values[0])
                # max
                self.params[p]["max"] = self.params[p]["dtype"](df_.loc[
                    df_["Parameter"] == p, "Max"
                ].values[0])


        # handle input dt
        self.update_dt()

        return None

    def load_data(self):
        """Load simulation data from folder data (stored in `self.folder_data`). Overwrites superior methods.

        .. warning::

            This method overwrites model data.

        :return: None
        :rtype: None
        """
        # -------------- load observation data -------------- #
        self.file_data_obs = Path(f"{self.folder_data}/{self.filename_data_obs}")
        self.data_obs = pd.read_csv(
            self.file_data_obs,
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

        # simulation steps (this is useful for testing and debug)
        if self.n_steps is None:
            n_steps = len(df)
        else:
            n_steps = self.n_steps

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
        # todo [docstring]
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
        #df["{}_obs".format(self.var_eval)] = df["{}_obs".format(self.var_eval)] * factor
        df["{}".format(self.var_eval)] = df["{}".format(self.var_eval)] / factor
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
        """Export object resources.

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

        # ... continues in downstream objects ... #

    def view(self, show=True):
        # todo [docstring]
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
        # handle return
        if show:
            plt.show()
        else:
            file_path = "{}/{}.{}".format(
                specs["folder"], specs["filename"], specs["fig_format"]
            )
            plt.savefig(file_path, dpi=specs["dpi"])
            plt.close(fig)
            return None


class LSRR(LinearStorage):
    # todo [major docstring improvement]
    """Rainfall-Runoff model based on a Linear Storage. This is a Toy Model. Inflow rain (P) in a necessary data input.

    """

    def __init__(self, name="MyLSRR", alias="LSRR001"):
        # todo [docstring]
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "LSRR"

        # observation data
        self.filename_data_obs = "q_obs.csv"

        # input data
        self.data_clim = None
        self.file_data_clim = None
        self.filename_data_clim = "clim.csv"


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
                    "TeX": "$Q_{max}$"
                },
            }
        )

        return None

    def _set_model_vars(self):
        # todo [docstring]
        self.vars = {
            "t": {
                "units": "{k}",
                "description": "Accumulated time",
                "kind": "time",
                "TeX": "$t$"
            },
            "P": {
                "units": "mm/{dt_freq}",
                "description": "Inflow",
                "kind": "flow",
                "TeX": "$P$"
            },
            "S": {
                "units": "mm",
                "description": "Storage level",
                "kind": "level",
                "TeX": "$S$"
            },
            "Q": {
                "units": "mm/{dt_freq}",
                "description": "Outflow",
                "kind": "flow",
                "TeX": "$Q$"
            },
            "Q_obs": {
                "units": "mm/d",
                "description": "Outflow (observed evidence)",
                "kind": "flow",
                "TeX": "$Q_{obs}$"
            },
        }
        self.var_eval = "Q"
        self.var_inputs = ["P"]
        return None

    def _set_view_specs(self):
        """Set view specifications. Expected to increment superior methods.

        :return: None
        :rtype: None
        """
        super()._set_view_specs()
        # cleanup useless entries

        # add new specs
        self.view_specs.update(
            {
                # update vales
                "gs_wspace": 0.4,
                "gs_hspace": 0.7,
                "gs_left": 0.15,
                "gs_right": 0.85,
                "gs_bottom": 0.1,
                "gs_top": 0.9,
                # new params
                "color_P": "tab:gray",
                "color_Q": "blue",
                "color_Q_obs": "indigo",
                "color_S": "black",
                "ymax_P": None,
                "ymax_Q": None,
                "ymax_S": None,
            }
        )
        return None

    def load_data(self):
        """Load simulation data. Expected to overwrite superior methods.

        .. warning::

            This method overwrites model data.

        :return: None
        :rtype: None
        """

        # -------------- load climate input data -------------- #
        self.file_data_clim = Path(f"{self.folder_data}/{self.filename_data_clim}")
        df_data_input = pd.read_csv(
            self.file_data_clim,
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

        # set the data input for climate
        self.data_clim = df_data_input[[self.dtfield] + self.var_inputs].copy()

        # -------------- load observation data -------------- #
        self.file_data_obs = Path(f"{self.folder_data}/{self.filename_data_obs}")
        self.data_obs = pd.read_csv(
            self.file_data_obs,
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
            input_df=self.data_clim.copy(), input_varfield="P", input_dtfield="DateTime"
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

        # simulation steps (this is useful for testing and debug)
        if self.n_steps is None:
            n_steps = len(df)
        else:
            n_steps = self.n_steps

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
        # todo [docstring]
        specs = self.view_specs.copy()
        n_dt = self.params["dt"]["value"]
        # start plot
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        plt.suptitle("{} | Model LSRR - R2: {}".format(self.name, round(self.rsq, 3)))
        # grid
        gs = mpl.gridspec.GridSpec(
            3,
            1,
            wspace=specs["gs_wspace"],
            hspace=specs["gs_hspace"],
            left=specs["gs_left"],
            right=specs["gs_right"],
            bottom=specs["gs_bottom"],
            top=specs["gs_top"],
        )  # nrows, ncols

        first_date = self.data[self.dtfield].iloc[0]
        last_date = self.data[self.dtfield].iloc[-1]
        ls_dates = [first_date, last_date]
        n_flow_max = np.max([self.data["P"].max(), self.data["Q"].max()])

        # ------------ P plot ------------
        n_pmax = self.data["P"].max() / n_dt
        ax = fig.add_subplot(gs[0, 0])
        plt.plot(self.data[self.dtfield], self.data["P"] / n_dt , color=specs["color_P"], zorder=2)
        plt.title("$P$ ({} mm)".format(round(self.data["P"].sum(), 1)), loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_xticks(ls_dates)
        # normalize X axis
        plt.xlim(ls_dates)
        # normalize Y axis
        ymax_p = specs["ymax_P"]
        if ymax_p is None:
            ymax_p = 1.2 * n_pmax
        plt.ylim(0, ymax_p)
        plt.ylabel("mm/{}".format(self.params["dt"]["units"].lower()))

        # ------------ Q plot ------------
        n_qmax = np.max([self.data["Q"].max() / n_dt, self.data_obs["Q_obs"].max()])
        #n_qmax = self.data["Q"].max() / n_dt
        q_sum = round(self.data["Q"].sum(), 1)
        ax = fig.add_subplot(gs[1, 0])
        # titles
        plt.title("$Q$ ({} mm) ".format(q_sum), loc="left")
        plt.title("$Q_{obs}$" + " ({} mm)".format(round(self.data_obs["Q_obs"].sum(), 1)), loc="right")
        # plots
        plt.plot(self.data[self.dtfield], self.data["Q"] / n_dt, color=specs["color_Q"])
        plt.plot(
            self.data_obs[self.dtfield],
            self.data_obs["Q_obs"],
            ".",
            color=specs["color_Q_obs"],
        )
        # tic labels
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_xticks(ls_dates)
        # normalize X axis
        plt.xlim(ls_dates)
        # normalize Y axis
        ymax_q = specs["ymax_Q"]
        if ymax_q is None:
            ymax_q = 1.2 * n_qmax
        plt.ylim(0, 1.2 * ymax_q)
        plt.ylabel("mm/{}".format(self.params["dt"]["units"].lower()))

        # ------------ S plot ------------
        n_smax = self.data["S"].max()
        ax = fig.add_subplot(gs[2, 0])
        plt.title("$S$ ($\mu$ = {} mm)".format(round(self.data["S"].mean(), 1)), loc="left")
        plt.plot(self.data[self.dtfield], self.data["S"], color=specs["color_S"])
        # ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_xticks(ls_dates)
        # normalize X axis
        plt.xlim(ls_dates)
        plt.ylabel("mm")
        # normalize Y axis
        ymax_s = specs["ymax_S"]
        if ymax_s is None:
            ymax_s = 1.2 * n_smax
        plt.ylim(0, ymax_s)

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

class LSRRE(LSRR):
    # todo [major docstring improvement]
    """Rainfall-Runoff-Evaporation model based on a Linear Storage. This is a Toy Model.
    Inflow rain (P) and outflow potential evaporation (E_pot) are necessary data input.

    """

    def __init__(self, name="MyLSRRE", alias="LSRRE001"):
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "LSRRE"
        # controllers
        self.shutdown_epot = False

    def _set_model_vars(self):
        super()._set_model_vars()
        self.vars.update(
            {
                "E": {
                    "units": "mm/{dt_freq}",
                    "description": "External Outflow",
                    "kind": "flow",
                    "TeX": "$E$"

                },
                "E_pot": {
                    "units": "mm/{dt_freq}",
                    "description": "Maximal External Outflow",
                    "kind": "flow",
                    "TeX": "$E_{pot}$"
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
            input_df=self.data_clim, input_varfield="E_pot", input_dtfield="DateTime"
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

        # simulation steps (this is useful for testing and debug)
        if self.n_steps is None:
            n_steps = len(df)
        else:
            n_steps = self.n_steps

        if self.shutdown_epot:
            df["E_pot"] = 0.0

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
            with np.errstate(divide='ignore', invalid='ignore'):
                df["Q"].values[t] = o_act * np.where(o_pot == 0, 0, q_pot / o_pot)
                df["E"].values[t] = o_act * np.where(o_pot == 0, 0, e_pot / o_pot)

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
        n_dt = self.params["dt"]["value"]
        fig = super().view(show=False, return_fig=True)
        plt.suptitle("{} | Model LSRRE - R2: {}".format(self.name, round(self.rsq, 3)))
        axes = fig.get_axes()
        # ---- E plot
        ax = axes[0]
        e_sum = round(self.data["E"].sum(), 1)
        ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
        ax2.plot(self.data[self.dtfield], self.data["E"] / n_dt, c="tab:red", zorder=1)
        e_max = self.data["E"].max()
        if e_max == 0:
            ax2.set_ylim(-1, 1)
        else:
            ax2.set_ylim(0, 1.2 * e_max/ n_dt)
        ax2.set_title("$E$ ({} mm)".format(e_sum), loc="right")
        ax2.set_ylabel("mm/d")
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

class LSFAS(LSRRE):
    # todo [major docstring improvement]
    """Rainfall-Runoff-Evaporation model based on Linear Storage and Fill-And-Spill (FAS) mechanism.
    Inflow rain (P) and potential evapotranspiration (E_pot) are necessary data inputs.

    """

    def __init__(self, name="MyLSFAS", alias="LSFAS001"):
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "LSFAS"
        # controllers:
        self.shutdown_qb = False

    def _set_model_vars(self):
        super()._set_model_vars()
        # todo [refactor] update TeX symbols
        self.vars.update(
            {
                "SS": {
                    "units": "mm",
                    "description": "Spill storage",
                    "kind": "flow",
                    "TeX": "TeX"
                },
                "Q_s": {
                    "units": "mm/{dt_freq}",
                    "description": "Spill flow (outflow)",
                    "kind": "flow",
                    "TeX": "TeX"
                },
                "Q_s_f": {
                    "units": "mm/mm",
                    "description": "Spill flow fraction",
                    "kind": "constant",
                    "TeX": "TeX"
                },
                "Q_b": {
                    "units": "mm/{dt_freq}",
                    "description": "Base flow (outflow)",
                    "kind": "flow",
                    "TeX": "TeX"
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
                    "TeX": "$s_{a}$"
                },
                "s_c": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Effective fragmentantion level of storage",
                    "kind": "conceptual",
                    "TeX": "$s_{c}$"
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
        self.data["SS"] = np.nan

        # organize table
        ls_vars = [self.dtfield, "t", "P", "E_pot", "E", "Q_s", "Q_s_f", "Q_b", "Q", "S", "SS"]
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

        # simulation steps (this is useful for testing and debug)
        if self.n_steps is None:
            n_steps = len(df)
        else:
            n_steps = self.n_steps

        # remove E_pot (useful for testing)
        if self.shutdown_epot:
            df["E_pot"] = 0.0

        # --- numerical solution
        # loop over (Euler Method)
        for t in range(n_steps - 1):
            # Compute dynamic spill storage capacity in mm/D
            ss_cap = np.max([0, df["S"].values[t] - s_a]) # spill storage = s - s_a
            df["SS"].values[t] = ss_cap

            # Compute runoff fraction
            with np.errstate(divide='ignore', invalid='ignore'):
                qs_f = np.where((ss_cap + s_c) == 0, 0, (ss_cap / (ss_cap + s_c)))
            df["Q_s_f"].values[t] = qs_f

            # potential runoff -- scale by the fraction

            # compute dynamic runoff capacity

            # -- includes P in the Qs capacity
            qs_cap = (ss_cap * dt) + df["P"].values[t] # multiply by dt (figure out why!)

            # apply runoff fraction over Qs
            qs_pot = qs_f * qs_cap

            # Compute potential base flow
            # Qt = dt * St / k
            # useful for testing
            if self.shutdown_qb:
                qb_pot = 0.0
            else:
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

            # Allocate outflows
            with np.errstate(divide='ignore', invalid='ignore'):
                df["Q_s"].values[t] = o_act * np.where(o_pot == 0, 0, qs_pot / o_pot)
                df["Q_b"].values[t] = o_act * np.where(o_pot == 0, 0, qb_pot / o_pot)
                df["E"].values[t] = o_act * np.where(o_pot == 0, 0, e_pot / o_pot)

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
        plt.suptitle("{} | Model LSFAS - R2: {}".format(self.name, round(self.rsq, 1)))
        axes = fig.get_axes()
        # add Qb
        ax = axes[1]
        q_sum = round(self.data["Q"].sum(), 1)
        qb_sum = round(self.data["Q_b"].sum(), 1)
        ax.set_title("$Q$ ({} mm)".format(q_sum) + ", $Q_b$ ({} mm)".format(qb_sum) + " and $Q_{obs}$", loc="left")
        ax.plot(self.data[self.dtfield], self.data["Q_b"] / self.params["dt"]["value"], c="navy", zorder=1)
        ax.set_title("Runoff C", loc="right")
        ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
        ax2.plot(self.data[self.dtfield], self.data["Q_s_f"], '--', zorder=2)
        ax2.set_ylim(0, 1)

        ax = axes[2]
        ax.fill_between(
            x=self.data[self.dtfield],
            y1=self.data["S"] - self.data["SS"],
            y2=self.data["S"],
            color="lightsteelblue",
            label="Spill storage"
        )
        ax.legend()
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
                return None

class Global(LSFAS):
    """This is a global Rainfall-Runoff model. Simulates the the catchment
    as if is a global system. Expected inputs:

    - `clim.csv` for Precipitation (P) and Potential Evapotranspiration (E_pot).
    - `path_areas.csv` for deriving the geomorphic unit hydrograph.

    # todo [major docstring] examples

    """
    def __init__(self, name="MyGlobal", alias="Glo001"):
        # todo [docstring]
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "Global"

        # testing controllers
        self.shutdown_qif = False
        self.shutdown_qbf = False

        # path area file
        self.data_paths = None
        self.data_paths_input = None
        self.file_data_paths = None
        self.filename_data_paths = "path_areas.csv"
        self.unit_hydrograph = None

    def _set_model_vars(self):
        super()._set_model_vars()

        # clean some vars
        del self.vars["S"]
        del self.vars["Q"]
        del self.vars["Q_b"]
        del self.vars["Q_s_f"]
        del self.vars["Q_s"]
        del self.vars["SS"]

        # update old attributes
        for v in self.vars:
            self.vars[v].update(
                {
                    "canopy": True,
                    "surface": True,
                    "soil": True,
                }
            )
        self.vars["Q_obs"].update(
            {
                "canopy": False,
                "surface": False,
                "soil": False,
            }
        )
        # set new entries
        self.new_vars = {
                #
                #
                # Canopy variables
                "C": {
                    "units": "mm",
                    "description": "Canopy storage",
                    "kind": "level",
                    "TeX": "$C$",
                    "canopy": True,
                    "surface": False,
                    "soil": False,
                },
                "E_c": {
                    "units": "mm/{dt_freq}",
                    "description": "Canopy evaporation",
                    "kind": "flow",
                    "TeX": "$E_{c}$",
                    "canopy": True,
                    "surface": False,
                    "soil": False,
                },
                "P_tf": {
                    "units": "mm/{dt_freq}",
                    "description": "Throughfall -- canopy spill water",
                    "kind": "flow",
                    "TeX": "$P_{tf}$",
                    "canopy": True,
                    "surface": False,
                    "soil": False,
                },
                "P_tf_f": {
                    "units": "mm/mm",
                    "description": "Throughfall fraction",
                    "kind": "constant",
                    "TeX": "$P_{tf, f}$",
                    "canopy": True,
                    "surface": False,
                    "soil": False,
                },
                "P_sf": {
                    "units": "mm/{dt_freq}",
                    "description": "Stemflow -- canopy drained water",
                    "kind": "flow",
                    "TeX": "$P_{sf}$",
                    "canopy": True,
                    "surface": False,
                    "soil": False,

                },
                "P_c": {
                    "units": "mm/{dt_freq}",
                    "description": "Effective precipitation on canopy",
                    "kind": "flow",
                    "TeX": "$P_{c}$",
                    "canopy": True,
                    "surface": False,
                    "soil": False,
                },
                "P_s": {
                    "units": "mm/{dt_freq}",
                    "description": "Effective precipitation on surface",
                    "kind": "flow",
                    "TeX": "$P_{s}$",
                    "canopy": True,
                    "surface": True,
                    "soil": False,
                },
                #
                #
                # Surface variables
                "S": {
                    "units": "mm",
                    "description": "Surface storage",
                    "kind": "level",
                    "TeX": "$S$",
                    "canopy": False,
                    "surface": True,
                    "soil": False,
                },
                "E_s": {
                    "units": "mm/{dt_freq}",
                    "description": "Surface evaporation",
                    "kind": "flow",
                    "TeX": "$E_{s}$",
                    "canopy": False,
                    "surface": True,
                    "soil": False,
                },
                "Q_of": {
                    "units": "mm/{dt_freq}",
                    "description": "Overland flow  -- surface spill",
                    "kind": "flow",
                    "TeX": "$Q_{of}$",
                    "canopy": False,
                    "surface": True,
                    "soil": False,
                },
                "Q_of_f": {
                    "units": "mm/mm",
                    "description": "Overland flow fraction",
                    "kind": "constant",
                    "TeX": "$Q_{of, f}$",
                    "canopy": False,
                    "surface": True,
                    "soil": False,
                },
                "Q_uf": {
                    "units": "mm/{dt_freq}",
                    "description": "Underland flow -- lateral leak in topsoil",
                    "kind": "flow",
                    "TeX": "$Q_{uf}$",
                    "canopy": False,
                    "surface": True,
                    "soil": False,
                },
                "Q_uf_f": {
                    "units": "mm/mm",
                    "description": "Underland flow fraction",
                    "kind": "constant",
                    "TeX": "$Q_{uf, f}$",
                    "canopy": False,
                    "surface": True,
                    "soil": False,
                },
                "Q_if": {
                    "units": "mm/{dt_freq}",
                    "description": "Infiltration flow -- drainage to mineral vadoze zone",
                    "kind": "flow",
                    "TeX": "$Q_{if}$",
                    "canopy": False,
                    "surface": True,
                    "soil": True,
                },
                #
                #
                # Vadose vars
                "V": {
                    "units": "mm",
                    "description": "Vadose zone storage",
                    "kind": "level",
                    "TeX": "$V$",
                    "canopy": False,
                    "surface": False,
                    "soil": True,
                },
                "Q_vf": {
                    "units": "mm",
                    "description": "Recharge flow -- vertical flow to phreatic zone",
                    "kind": "flow",
                    "TeX": "$Q_{v}$",
                    "canopy": False,
                    "surface": False,
                    "soil": True,
                },
                "Q_vf_f": {
                    "units": "mm/mm",
                    "description": "Recharge flow fraction",
                    "kind": "constant",
                    "TeX": "$Q_{vf, f}$",
                    "canopy": False,
                    "surface": False,
                    "soil": True,
                },
                "D_v": {
                    "units": "mm",
                    "description": "Vadose zone storage deficit",
                    "kind": "level",
                    "TeX": "$D_{v}$",
                    "canopy": False,
                    "surface": False,
                    "soil": True,
                },
                #
                #
                # Phreatic vars
                "G": {
                    "units": "mm",
                    "description": "Phreatic zone storage",
                    "kind": "level",
                    "TeX": "$G$",
                    "canopy": False,
                    "surface": False,
                    "soil": True,
                },
                "D": {
                    "units": "mm",
                    "description": "Phreatic zone storage deficit",
                    "kind": "level",
                    "TeX": "$D$",
                    "canopy": False,
                    "surface": False,
                    "soil": True,
                },
                "E_t": {
                    "units": "mm/{dt_freq}",
                    "description": "Phreatic zone transpiration",
                    "kind": "flow",
                    "TeX": "$E_{t}$",
                    "canopy": False,
                    "surface": False,
                    "soil": True,
                },
                "E_t_f": {
                    "units": "mm/{dt_freq}",
                    "description": "Phreatic zone transpiration fraction",
                    "kind": "constant",
                    "TeX": "$E_{t, f}$",
                    "canopy": False,
                    "surface": False,
                    "soil": True,
                },
                "Q_gf": {
                    "units": "mm/{dt_freq}",
                    "description": "Groundwater flow (baseflow at hillslope)",
                    "kind": "flow",
                    "TeX": "$Q_{gf}$",
                    "canopy": False,
                    "surface": False,
                    "soil": True,
                },
                # routing
                "Q_hf": {
                    "units": "mm/{dt_freq}",
                    "description": "Hillslope flow (surface, subsurface and baseflow)",
                    "kind": "flow",
                    "TeX": "$Q_{hf}$",
                    "canopy": False,
                    "surface": True,
                    "soil": True,
                },
                "Q": {
                    "units": "mm/{dt_freq}",
                    "description": "Stream flow at stage station",
                    "kind": "flow",
                    "TeX": "$Q$",
                    "canopy": False,
                    "surface": True,
                    "soil": True,
                },
                "Q_bf": {
                    "units": "mm/{dt_freq}",
                    "description": "River baseflow (routed groundflow at river gauge)",
                    "kind": "flow",
                    "TeX": "$Q_{bf}$",
                    "canopy": False,
                    "surface": False,
                    "soil": True,
                },
            }
        # update new entries
        self.vars.update(self.new_vars.copy())
        # get useful lists
        self.vars_canopy = [self.dtfield] + [v for v in self.vars if self.vars[v]["canopy"]]
        self.vars_surface = [self.dtfield] + [v for v in self.vars if self.vars[v]["surface"]]
        self.vars_soil = [self.dtfield] + [v for v in self.vars if self.vars[v]["soil"]]
        self.vars_levels = [
            "C", "S", "V", "D_v", "G", "D"
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
                    "TeX": "$C_{0}$",
                },
                "S0": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Surface storage initial conditions",
                    "kind": "conceptual",
                    "TeX": "$S_{0}$",
                },
                "V0": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Vadose zone initial conditions",
                    "kind": "conceptual",
                    "TeX": "$V_{0}$",
                },
                "G0": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Phreatic zone initial conditions",
                    "kind": "conceptual",
                    "TeX": "$G_{0}$",
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
                    "TeX": "$C_{k}$",
                },
                "C_a": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Activation level for canopy spill",
                    "kind": "conceptual",
                    "TeX": "$C_{a}$",
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
                    "TeX": "$S_{k}$",
                },
                # subsurface spill
                "S_uf_cap": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Topsoil zone storage capacity",
                    "kind": "conceptual",
                    "TeX": "$S_{uf, cap}$",
                },
                "S_uf_a": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Activation level for subsurface spill",
                    "kind": "conceptual",
                    "TeX": "$S_{uf, a}$",
                },
                "S_uf_c": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Fragmentation level for subsurface spill",
                    "kind": "conceptual",
                    "TeX": "$S_{uf, c}$",
                },
                # surface spill
                "S_of_a": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Activation level for surface overland spill",
                    "kind": "conceptual",
                    "TeX": "$S_{of, a}$",
                },
                "S_of_c": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Fragmentation level for surface overland spill",
                    "kind": "conceptual",
                    "TeX": "$S_{of, c}$",
                },
                # recharge
                "K_V": {
                    "value": None,
                    "units": "mm/D",
                    "dtype": np.float64,
                    "description": "Effective hydraulic conductivity",
                    "kind": "conceptual",
                    "TeX": "$K_{V}$",
                },
                # baseflow
                "D_et_a": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Effective root depth for full evapotranspiration activation",
                    "kind": "level",
                    "TeX": "$D_{et, a}$",
                },
                "G_et_cap": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Evapotranspiration capacity at capilar zone",
                    "kind": "level",
                    "TeX": "$G_{et, cap}$",
                },
                "G_cap": {
                    "value": None,
                    "units": "mm",
                    "dtype": np.float64,
                    "description": "Phreatic zone storage capacity",
                    "kind": "conceptual",
                    "TeX": "$G_{cap}$",
                },
                "G_k": {
                    "value": None,
                    "units": "D",
                    "dtype": np.float64,
                    "description": "Phreatic zone residence time",
                    "kind": "conceptual",
                    "TeX": "$G_{k}$",
                },
                "K_Q": {
                    "value": None,
                    "units": "m/D",
                    "dtype": np.float64,
                    "description": "Channel water celerity",
                    "kind": "conceptual",
                    "TeX": "$K_{Q}$",
                },
            }
        )
        # reset reference
        self.reference_dt_param = "G_k"
        return None


    def _set_view_specs(self):
        """Set view specifications.
        Expected to increment superior methods.

        :return: None
        :rtype: None
        """
        super()._set_view_specs()
        # cleanup useless entries

        # add new specs
        self.view_specs.update(
            {
                # new values
                "width": 10,
                "height": 6,
                "gs_left": 0.1,
                "gs_right": 0.9,
                # new params
                "color_P": "deepskyblue",
                "color_P_s": "blue",
                "color_P_sf": "green",
                "color_C": "black",
                "color_V": "steelblue",
                "color_G": "navy",
                "color_Q_if": "blue",
                "color_Q_hf": "blue",
                "color_Q_bf": "darkblue",
                "color_Q": "blue",
                "ymax_C": None,
                "ymax_V": None,
                "ymax_G": None,
                "ymax_Q_if": None,
                "ymax_Q": None,
                "ymax_Q_hf": None,
            }
        )
        return None


    def load_data(self):
        """Load simulation data. Expected to increment superior methods.

        :return: None
        :rtype: None
        """
        super().load_data()

        # -------------- load path areas input data -------------- #
        self.file_data_paths = Path(f"{self.folder_data}/{self.filename_data_paths}")
        self.data_paths_input = pd.read_csv(
            self.file_data_paths,
            sep=self.file_csv_sep,
            encoding=self.file_encoding,
        )


        # -------------- update other mutables -------------- #
        # >>> evaluate using self.update()

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

        # clear deprecated parent variables
        self.data.drop(
            columns=["Q_s", "Q_s_f", "Q_b", "S"],
            inplace=True
        )
        # append new variables
        for v in self.new_vars:
            self.data[v] = np.nan

        #
        # ---------------- routing setup ----------------- #
        #
        from scipy.interpolate import CubicSpline

        # get vectors
        vct_paths = self.data_paths_input["Path (m)"].values
        vct_area = self.data_paths_input["Global (ha)"].values

        # handle not starting in zero
        if np.min(vct_paths) > 0:
            vct_paths = np.insert(vct_paths, 0, 0)
            vct_area = np.insert(vct_area, 0, 0)

        # compute delay
        vct_t = vct_paths / self.params["K_Q"]["value"]

        # set dataframe
        self.data_paths = pd.DataFrame(
            {
                "Delay (d)": vct_t,
                "Path (m)": vct_paths,
                "Global (ha)": vct_area,
            }
        )

        # compute unit hydrograph
        vct_t_src = self.data_paths["Delay (d)"].values
        vct_q_src = self.data_paths["Global (ha)"].values / self.data_paths["Global (ha)"].sum()

        # smoothing
        window_size = 8  # make it more or less smooth
        # Create a kernel for the moving average
        kernel = np.ones(window_size) / window_size
        # Apply the convolution
        vct_q_src = np.convolve(vct_q_src, kernel, mode='same')

        # clip data
        t_max = np.max(vct_t_src)
        df_q = self.data.query(f"t <= {t_max}")
        vct_t_clip = df_q["t"].values

        # Cubic spline interpolation (smoother)
        f_cubic = CubicSpline(vct_t_src, vct_q_src)
        vct_q_clip = f_cubic(vct_t_clip)

        # add zeros at the tail
        vct_q_new = np.concatenate([vct_q_clip, np.zeros(len(self.data) - len(vct_q_clip), dtype=np.float64)])
        # normalize to sum = 1
        vct_q_new = vct_q_new / np.sum(vct_q_new)

        # set unit hydrograph attribute
        self.unit_hydrograph = pd.DataFrame(
            {
                "t": self.data["t"].values,
                "q": vct_q_new
            }
        )
        return None


    def solve(self):
        """Solve the model for input and initial conditions by numerical methods.

        .. warning::

            This method overwrites model data.

        :return: None
        :rtype: None
        """

        #
        # ---------------- initial conditions ----------------
        #

        for v in self.vars:
            # set initial conditions on storages
            if self.vars[v]["kind"] == "level":
                if v == "D" or v == "D_v":
                    pass
                else:
                    self.data[v].values[0] = self.params["{}0".format(v)]["value"]

        # --- handle bad (unfeaseable) initial conditions

        # --- G0 storage must not exceed G_cap
        G0 = self.data["G"].values[0]
        if G0 > self.params["G_cap"]["value"]:
            self.data["G"].values[0] = self.params["G_cap"]["value"]

        # --- V0 storage must not exceed D (Gcap-G)
        V0 = self.data["V"].values[0]
        G0 = self.data["G"].values[0]
        G_cap = self.params["G_cap"]["value"]
        if V0 > (G_cap - G0):
            self.data["V"].values[0] = G_cap - G0

        # --- handle bad (unfeaseable) parameters

        # --- D_et_cap must not exceed G_cap
        G_cap = self.params["G_cap"]["value"]
        D_et_a = self.params["D_et_a"]["value"]
        if D_et_a > G_cap:
            self.params["D_et_a"]["value"] = G_cap

        #
        # ---------------- simulation setup ----------------
        #

        # dt is the fraction of 1 Day/(1 simulation time step)
        dt = self.params["dt"]["value"]
        # global processes data
        GB = self.data.copy()

        #
        # ---------------- parameters variables ----------------
        # this section is for improving code readability only

        # [Canopy] canopy parameters
        C_k = self.params["C_k"]["value"]
        C_a = self.params["C_a"]["value"]

        # [Surface] surface parameters
        S_k = self.params["S_k"]["value"]
        S_of_a = self.params["S_of_a"]["value"]
        S_of_c = self.params["S_of_c"]["value"]
        S_uf_a = self.params["S_uf_a"]["value"]
        S_uf_c = self.params["S_uf_c"]["value"]
        S_uf_cap = self.params["S_uf_cap"]["value"]

        # [Soil] soil parameters
        G_cap = self.params["G_cap"]["value"]
        K_V = self.params["K_V"]["value"]
        G_k = self.params["G_k"]["value"]
        G_et_cap = self.params["G_et_cap"]["value"]
        D_et_a = self.params["D_et_a"]["value"]


        #
        # ---------------- derived parameter variables ----------------
        #

        # [Surface] Conpute switch for underland flow
        S_uf_switch = np.where(S_uf_cap <= S_uf_a, 0, 1)

        # [Surface] Compute effective overland flow activation level
        r'''
        [Model Equation]
        $$$

        S_{of, a}^{*} = S_{uf, cap} + S_{of, a} 

        $$$
        '''
        S_of_a_eff = S_uf_cap + S_of_a  # incremental with underland capacity (topsoil)

        #
        # ---------------- testing features ----------------
        #

        # [Testing feature] shutdown E_pot
        if self.shutdown_epot:
            GB["E_pot"] = 0.0

        # [Testing feature] constrain simulation steps
        if self.n_steps is None:
            n_steps = len(GB)
        else:
            n_steps = self.n_steps

        #
        # ---------------- numerical solution ----------------
        #

        # loop over steps (Euler Method)
        for t in range(n_steps - 1):

            #
            # [Deficit] ---------- update deficits ---------- #
            #

            # [Deficit] Phreatic zone deficit
            r'''
            [Model Equation]
            Phreatic zone water deficit 
            $$$

            D(t) = G_{cap} + G(t) 

            $$$
            '''
            GB["D"].values[t] = G_cap - GB["G"].values[t]

            # [Deficit] Vadose zone deficit
            r'''
            [Model Equation]
            Vadose zone water deficit 
            $$$

            D_{v}(t) = D(t) - V(t) 

            $$$
            '''
            GB["D_v"].values[t] = GB["D"].values[t] - GB["V"].values[t]

            #
            # [Evaporation] ---------- get evaporation flows first ---------- #
            #

            # [Evaporation Canopy] ---- evaporation from canopy

            # [Evaporation - Canopy] compute potential flow and current capacity
            E_c_pot = GB["E_pot"].values[t]
            E_c_cap = GB["C"].values[t] * dt

            # [Evaporation - Canopy] compute actual flow
            GB["E_c"].values[t] = np.min([E_c_pot, E_c_cap])

            # [Evaporation Canopy] water balance -- apply discount a priori
            GB["C"].values[t] = GB["C"].values[t] - GB["E_c"].values[t]

            # [Evaporation - Soil] ---- transpiration from soil

            # [Evaporation - Soil] compute potential flow and current capacity
            E_t_pot = GB["E_pot"].values[t] - GB["E_c"].values[t]  # discount actual Ec

            # [Evaporation - Soil] Compute Et capacity

            # [Evaporation - Soil] Compute the root zone depth factor
            r'''
            [Model Equation]
            Potencial transpiration fraction
            $$$

            E_{t, f} = 1 - \frac{D_{v}}{D_{v} + D_{et, a}} 

            $$$
            '''
            GB["E_t_f"].values[t] = 1 - (GB["D_v"].values[t] / (GB["D_v"].values[t] + D_et_a))

            # [Evaporation - Soil] Compute the Et capacity -- bounded by G_et_cap (green water capacity)
            E_t_cap = GB["E_t_f"].values[t] * np.min([GB["G"].values[t], G_et_cap]) * dt

            # [Evaporation - Soil] compute actual flow
            GB["E_t"].values[t] = np.min([E_t_pot, E_t_cap])

            # [Evaporation - Soil] water balance -- apply discount a priori
            GB["G"].values[t] = GB["G"].values[t] - GB["E_t"].values[t]

            # [Evaporation - Surface] evaporation from surface

            # [Evaporation - Surface] compute potential flow and current capacity
            E_s_pot = GB["E_pot"].values[t] - GB["E_c"].values[t] - GB["E_t"].values[t]
            E_s_cap = GB["S"].values[t] * dt

            # [Evaporation - Surface] compute actual flow
            GB["E_s"].values[t] = np.min([E_s_pot, E_s_cap])

            # [Evaporation - Surface] water balance -- apply discount a priori
            GB["S"].values[t] = GB["S"].values[t] - GB["E_s"].values[t]

            #
            # [Canopy] ---------- Solve canopy water balance ---------- #
            #

            # [Canopy] ---- Potential flows

            # [Canopy] -- Throughfall

            # [Canopy] Compute canopy spill storage capacity
            C_ss_cap = np.max([0, GB["C"].values[t] - C_a]) * dt # check dt

            # [Canopy] Compute throughfall fraction
            r'''
            [Model Equation]
            throughfall fraction
            $$$

            P_{tf, f} = \text{MIN}(1,  C(t)/C_{a}) 

            $$$
            '''
            with np.errstate(divide='ignore', invalid='ignore'):
                P_tf_f = np.where(C_a <= 0, 1, np.min([1, GB["C"].values[t] / C_a]))
            GB["P_tf_f"].values[t] = P_tf_f

            # [Canopy] Compute throughfall capacity
            P_tf_cap = C_ss_cap + GB["P"].values[t]  # include P

            # [Canopy] Compute throughfall -- scale by the fraction
            GB["P_tf"].values[t] = P_tf_f * P_tf_cap

            # [Canopy] -- Stemflow

            # [Canopy] Compute potential stemflow -- only activated storage contributes
            C_sf_pot = np.min([GB["C"].values[t], C_a])

            # [Canopy] Compute actual stemflow
            r'''
            [Model Equation]
            Stemflow
            $$$

            P_{sf} = \frac{1}{C_{k}} * \text{MIN}(C(t), C_{a})

            $$$
            '''
            GB["P_sf"].values[t] = C_sf_pot * dt / C_k  # Linear storage Q = dt * S / k

            # [Canopy] Compute effective rain on surface
            GB["P_s"].values[t] = GB["P_tf"].values[t] + GB["P_sf"].values[t]
            # [Canopy] Compute effective rain on canopy
            GB["P_c"].values[t] = GB["P"].values[t] * (1 - GB["P_tf_f"].values[t])

            # [Canopy Water Balance] ---- Apply water balance
            r'''
            [Model Equation]
            Canopy water balance. $E_{c}$ is discounted earlier.
            $$$

            C(t + 1) = C(t) + P_{c}(t) - E_{c}(t) - P_{sf}(t) 

            $$$
            '''
            GB["C"].values[t + 1] = GB["C"].values[t] + GB["P_c"].values[t] - GB["P_sf"].values[t]

            #
            # [Surface] ---------- Solve surface water balance ---------- #
            #

            # [Surface] ---- Potential flows

            # [Surface -- overland] -- Overland flow

            # [Surface -- overland] Compute surface overland spill storage capacity
            Sof_ss_cap = np.max([0, GB["S"].values[t] - S_of_a_eff])

            # [Surface -- overland] Compute overland flow fraction
            r'''
            [Model Equation]
            Overland flow fraction
            $$$

            Q_{of, f}(t) = \frac{S(t) - S_{of, a}(t)}{S(t) - S_{of, a}(t) + S_{of, c}(t)}

            $$$
            '''
            with np.errstate(divide='ignore', invalid='ignore'):
                Q_of_f = np.where((Sof_ss_cap + S_of_c) == 0, 0, (Sof_ss_cap / (Sof_ss_cap + S_of_c)))
            GB["Q_of_f"].values[t] = Q_of_f

            # [Surface -- overland] Compute dynamic overland flow capacity
            Q_of_cap = (Sof_ss_cap * dt) + GB["P_s"].values[t]  # include P_s # check dt

            # [Surface -- overland] Compute potential overland -- scale by the fraction
            r'''
            [Model Equation]
            Underland flow
            $$$

            Q_{of}(t) = Q_{of, f}(t) * (\frac{(S(t) - S_{of, a}(t))}{\Delta t} + P_{s})

            $$$
            '''
            Q_of_pot = Q_of_f * Q_of_cap

            # [Surface -- underland] -- Underland flow

            # [Surface -- underland] Compute surface underland spill storage capacity
            Suf_ss_cap = np.max([0, GB["S"].values[t] - S_uf_a]) * S_uf_switch  # apply factor

            # [Surface -- underland] Compute underland flow fraction
            r'''
            [Model Equation]
            Underland flow fraction
            $$$

            Q_{uf, f}(t) = \frac{S(t) - S_{uf, a}(t)}{S(t) - S_{uf, a}(t) + S_{uf, c}(t)}

            $$$
            '''
            with np.errstate(divide='ignore', invalid='ignore'):
                Q_uf_f = np.where((Suf_ss_cap + S_uf_c) == 0, 0, (Suf_ss_cap / (Suf_ss_cap + S_uf_c)))
            GB["Q_uf_f"].values[t] = Q_uf_f

            # [Surface -- underland] Compute dynamic underland flow capacity
            Q_uf_cap = Suf_ss_cap * dt # check dt

            # [Surface -- underland] Compute potential underland -- scale by the fraction
            r'''
            [Model Equation]
            Underland flow
            $$$

            Q_{uf}(t) = Q_{uf, f}(t) * \frac{(S(t) - S_{uf, a}(t))}{\Delta t}

            $$$
            '''
            Q_uf_pot = Q_uf_f * Q_uf_cap

            # [Surface -- infiltration] -- Infiltration flow

            # [Surface -- infiltration] -- Potential infiltration from downstream (soil)
            Q_if_pot_down = (GB["D"].values[t] - GB["V"].values[t]) * dt  # check dt

            # [Surface -- infiltration] -- Potential infiltration from upstream (hydraulic head)
            r'''
            [Model Equation]
            Infiltration flow
            $$$

            Q_{if}(t) = \frac{1}{S_{k}} * S(t)

            $$$
            '''
            Q_if_pot_up = GB["S"].values[t] * dt / S_k

            # [Surface -- infiltration] -- Potential infiltration
            Q_if_pot = np.min([Q_if_pot_down, Q_if_pot_up])

            # [Testing feature]
            if self.shutdown_qif:
                Q_if_pot = 0.0

            # [Surface] -- Full potential outflow
            S_out_pot = Q_of_pot + Q_uf_pot + Q_if_pot

            # [Surface] ---- Actual flows

            # [Surface] Compute surface maximum outflow
            S_out_max = GB["S"].values[t]

            # [Surface] Compute Actual outflow
            S_out_act = np.min([S_out_max, S_out_pot])

            # [Surface] Allocate outflows
            with np.errstate(divide='ignore', invalid='ignore'):
                GB["Q_of"].values[t] = S_out_act * np.where(S_out_pot == 0, 0, Q_of_pot / S_out_pot)
                GB["Q_uf"].values[t] = S_out_act * np.where(S_out_pot == 0, 0, Q_uf_pot / S_out_pot)
                GB["Q_if"].values[t] = S_out_act * np.where(S_out_pot == 0, 0, Q_if_pot / S_out_pot)

            # [Surface Water Balance] ---- Apply water balance.
            r'''
            [Model Equation]
            Surface water balance. $E_{c}$ is discounted earlier in procedure.
            $$$

            S(t + 1) = S(t) + P_{s}(t) - E_{s}(t) - Q_{if}(t) - Q_{uf}(t) - Q_{of}(t)

            $$$
            '''
            GB["S"].values[t + 1] = GB["S"].values[t] + GB["P_s"].values[t] - GB["Q_of"].values[t] - GB["Q_uf"].values[t] - GB["Q_if"].values[t]

            #
            # [Soil] ---------- Solve soil water balance ---------- #
            #

            # [Soil Vadose Zone]

            # [Soil Vadose Zone] Get Recharge Fraction
            r'''
            [Model Equation]
            Vertical (Recharge) flow fraction
            $$$

            Q_{vf, f}(t) = \frac{V(t)}{D(t)}

            $$$
            '''
            with np.errstate(divide='ignore', invalid='ignore'):
                GB["Q_vf_f"].values[t] = np.where(GB["D"].values[t] <= 0, 1.0, (GB["V"].values[t] / GB["D"].values[t]))

            # [Soil Vadose Zone] Compute Potential Recharge
            r'''
            [Model Equation]
            Vertical (Recharge) flow
            $$$

            Q_{vf}(t) = Q_{vf, f}(t) * K

            $$$
            '''
            Q_vf_pot = GB["Q_vf_f"].values[t] * K_V * dt

            # [Soil Vadose Zone] Compute Maximal Recharge
            Q_vf_max = GB["V"].values[t] * dt # check dt

            # [Soil Vadose Zone] Compute Actual Recharge
            GB["Q_vf"].values[t] = np.min([Q_vf_pot, Q_vf_max])

            # [Vadose Water Balance] ---- Apply water balance
            r'''
            [Model Equation]
            Vadose zone water balance
            $$$

            V(t + 1) = V(t) + Q_{if}(t) - Q_{vf}(t)

            $$$
            '''
            GB["V"].values[t + 1] = GB["V"].values[t] + GB["Q_if"].values[t] - GB["Q_vf"].values[t]

            # [Soil Phreatic Zone]

            # [Soil Phreatic Zone] Compute Base flow (blue water -- discount on green water)
            r'''
            [Model Equation]
            Baseflow
            $$$

            Q_{gf}(t) = \frac{1}{G_{k}} * (G(t) - G_{et, cap})

            $$$
            '''
            GB["Q_gf"].values[t] = (GB["G"].values[t] - G_et_cap) * dt / G_k

            # [Testing feature]
            if self.shutdown_qbf:
                GB["Q_gf"].values[t] = 0.0

            # [Phreatic Water Balance] ---- Apply water balance
            r'''
            [Model Equation]
            Phreatic zone water balance. $E_{t}$ is discounted earlier in the procedure.
            $$$

            G(t + 1) = G(t) + Q_{vf}(t) - E_{t}(t) - Q_{bf}(t) 

            $$$
            '''
            GB["G"].values[t + 1] = GB["G"].values[t] + GB["Q_vf"].values[t] - GB["Q_gf"].values[t]

        #
        # [Streamflow] ---------- Solve flow routing to gauge station ---------- #
        #

        # [Streamflow] Compute Hillslope flow
        r'''
        [Model Equation]
        Hillslope flow
        $$$

        Q_{hf}(t) = Q_{bf}(t) + Q_{uf}(t) + Q_{of}(t)

        $$$
        '''
        GB["Q_hf"] = GB["Q_gf"] + GB["Q_uf"] + GB["Q_of"]

        # [Baseflow] Compute river base flow
        GB["Q_bf"] = self.propagate_inflow(
            inflow=GB["Q_gf"].values,
            unit_hydrograph=self.unit_hydrograph["q"].values,
        )

        # [Fast Streamflow] Compute Streamflow
        # todo [feature] evaluate to split into more components
        Q_fast = self.propagate_inflow(
            inflow=GB["Q_uf"].values + GB["Q_of"].values,
            unit_hydrograph=self.unit_hydrograph["q"].values,
        )
        GB["Q"] = GB["Q_bf"].values + Q_fast

        #
        # [Total Flows] ---------- compute total flows ---------- #
        #

        # [Total Flows] Total E
        r'''
        [Model Equation]
        Total evaporation
        $$$

        E(t) =  E_{c}(t) + E_{s}(t) + E_{t}(t)

        $$$
        '''
        GB["E"] = GB["E_c"] + GB["E_s"] + GB["E_t"]

        # reset data
        self.data = GB.copy()

        return None

    def export(self, folder, filename, views=False, mode=None):
        """Export object resources

        :param folder: path to folder
        :type folder: str
        :param filename: file name without extension
        :type filename: str
        :return: None
        :rtype: None
        """
        # export model simulation data with views=False
        super().export(folder, filename=filename + "_sim", views=False)

        # handle views
        if views:
            self.view_specs["folder"] = folder
            self.view_specs["filename"] = filename
            # handle view mode
            self.view(show=False, mode=mode)

        # handle to export paths and unit hydrograph
        fpath = Path(folder + "/" + filename + "_path_areas" + self.file_csv_ext)
        self.data_paths.to_csv(
            fpath, sep=self.file_csv_sep, encoding=self.file_encoding, index=False
        )

        # ... continues in downstream objects ... #

    def view(self, show=True, return_fig=False, mode=None):
        # todo [docstring]
        # todo [optimize for DRY]
        specs = self.view_specs.copy()
        n_dt = self.params["dt"]["value"]
        # start plot
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        plt.suptitle("{} | Model Global".format(self.name))
        # grid
        gs = mpl.gridspec.GridSpec(
            3,
            1,
            wspace=specs["gs_wspace"],
            hspace=specs["gs_hspace"],
            left=specs["gs_left"],
            right=specs["gs_right"],
            bottom=specs["gs_bottom"],
            top=specs["gs_top"],
        )  # nrows, ncols

        first_date = self.data[self.dtfield].iloc[0]
        last_date = self.data[self.dtfield].iloc[-1]
        ls_dates = [first_date, last_date]

        if mode is None:
            mode = "canopy"

        if mode == "river":
            # ------------ P plot ------------
            n_pmax = self.data["P"].max() / n_dt
            ax = fig.add_subplot(gs[0, 0])
            plt.plot(self.data[self.dtfield], self.data["P"] / n_dt, color=specs["color_P"],
                     zorder=2, label=self.vars["P"]["TeX"])
            s_title1 = "$P$ ({} mm)".format(round(self.data["P"].sum(), 1))
            plt.title(s_title1, loc="left")
            # normalize X axis
            plt.xlim(ls_dates)
            # normalize Y axis
            ymax_p = specs["ymax_P"]
            if ymax_p is None:
                ymax_p = 1.2 * n_pmax
            plt.ylim(0, ymax_p)
            plt.ylabel("mm/{}".format(self.params["dt"]["units"].lower()))

            # ------------ E plot ------------
            ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
            e_sum = round(self.data["E"].sum(), 1)
            s_symbol4 = self.vars["E_pot"]["TeX"]
            ax2.plot(self.data[self.dtfield], self.data["E"] / n_dt, c="tab:red", zorder=1, label=self.vars["E"]["TeX"])
            ax2.plot(self.data[self.dtfield], self.data["E_pot"] / n_dt, c="tab:gray", alpha=0.5, zorder=2,
                     linestyle="--",
                     label=s_symbol4)
            e_max = self.data["E_pot"].max()
            if e_max == 0:
                ax2.set_ylim(-1, 1)
            else:
                ax2.set_ylim(0, 1.2 * e_max / n_dt)
            ax2.set_title("{} ({} mm)".format(self.vars["E"]["TeX"], e_sum), loc="right")
            ax2.set_ylabel("mm/d")

            # ------------ Q plot ------------
            n_qmax = np.max([self.data["Q_hf"].max() / n_dt, self.data_obs["Q_obs"].max()])
            # n_qmax = self.data["Q"].max() / n_dt
            q_sum = round(self.data["Q"].sum(), 1)
            ax = fig.add_subplot(gs[1, 0])
            s_symbol0 = self.vars["Q"]["TeX"]
            s_symbol1 = self.vars["Q_bf"]["TeX"]
            s_symbol2 = self.vars["Q_hf"]["TeX"]
            # titles
            plt.title("$Q$ ({} mm) ".format(q_sum), loc="left")
            plt.title("$Q_{obs}$" + " ({} mm)".format(round(self.data_obs["Q_obs"].sum(), 1)), loc="right")
            # plots
            plt.plot(self.data[self.dtfield], self.data["Q"] / n_dt, color=specs["color_Q"], zorder=2, label=s_symbol0)
            plt.plot(self.data[self.dtfield], self.data["Q_bf"] / n_dt, color=specs["color_Q_bf"], zorder=1, label=s_symbol1)
            plt.plot(self.data[self.dtfield], self.data["Q_hf"] / n_dt, color="silver", zorder=0, label=s_symbol2)
            plt.plot(
                self.data_obs[self.dtfield],
                self.data_obs["Q_obs"],
                ".",
                color=specs["color_Q_obs"],
            )
            plt.legend(loc="upper left")
            # tic labels
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.set_xticks(ls_dates)
            # normalize X axis
            plt.xlim(ls_dates)
            # normalize Y axis
            ymax_q = specs["ymax_Q_hf"]
            if ymax_q is None:
                ymax_q = 1.1 * n_qmax
            plt.ylim(0, 1.1 * ymax_q)
            plt.ylabel("mm/{}".format(self.params["dt"]["units"].lower()))

            # ------------ G plot ------------
            n_smax = self.data["G"].max()
            ax = fig.add_subplot(gs[2, 0])
            plt.title("$G$ ($\mu$ = {} mm)".format(round(self.data["G"].mean(), 1)), loc="left")
            plt.plot(self.data[self.dtfield], self.data["G"], color=specs["color_G"], label=self.vars["G"]["TeX"])
            plt.hlines(
                y=self.params["G_cap"]["value"],
                xmin=self.data[self.dtfield].min(),
                xmax=self.data[self.dtfield].max(),
                colors="orange",
                linestyles="--",
                label=self.params["G_cap"]["TeX"]
            )
            plt.hlines(
                y=self.params["G_cap"]["value"] - self.params["D_et_a"]["value"],
                xmin=self.data[self.dtfield].min(),
                xmax=self.data[self.dtfield].max(),
                colors="green",
                linestyles="--",
                label=self.params["D_et_a"]["TeX"]
            )
            plt.legend(loc="upper left", ncols=1)

            # ticks
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.set_xticks(ls_dates)
            # normalize X axis
            plt.xlim(ls_dates)
            plt.ylabel("mm")
            # normalize Y axis
            ymax_s = specs["ymax_G"]
            if ymax_s is None:
                ymax_s = 1.1 * self.params["G_cap"]["value"]
            plt.ylim(0, ymax_s)

        if mode == "canopy":
            # ------------ P plot ------------
            n_pmax = self.data["P"].max() / n_dt
            ax = fig.add_subplot(gs[0, 0])
            plt.plot(self.data[self.dtfield], self.data["P"] / n_dt, color=specs["color_P"],
                     zorder=2, label=self.vars["P"]["TeX"])
            s_title1 = "$P$ ({} mm)".format(round(self.data["P"].sum(), 1))
            plt.title(s_title1, loc="left")
            # normalize X axis
            plt.xlim(ls_dates)
            # normalize Y axis
            ymax_p = specs["ymax_P"]
            if ymax_p is None:
                ymax_p = 1.2 * n_pmax
            plt.ylim(0, ymax_p)
            plt.ylabel("mm/{}".format(self.params["dt"]["units"].lower()))

            # ------------ E plot ------------
            e_sum = round(self.data["E_c"].sum(), 1)
            ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
            ax2.plot(self.data[self.dtfield], self.data["E_c"] / n_dt, c="tab:red", zorder=1, label=self.vars["E_c"]["TeX"])
            e_max = self.data["E_c"].max()
            if e_max == 0:
                ax2.set_ylim(-1, 1)
            else:
                ax2.set_ylim(0, 1.2 * e_max / n_dt)
            ax2.set_title("{} ({} mm)".format(self.vars["E_c"]["TeX"], e_sum), loc="right")
            ax2.set_ylabel("mm/d")

            # ------------ Ps plot ------------
            n_emax = self.data["P_s"].max() / n_dt
            ps_sum = round(self.data["P_s"].sum(), 1)
            psf_sum = round(self.data["P_sf"].sum(), 1)
            ax = fig.add_subplot(gs[1, 0], sharex=ax)
            # titles
            s_symbol1 = self.vars["P_s"]["TeX"]
            s_symbol2 = self.vars["P_sf"]["TeX"]
            plt.title(f"{s_symbol1} ({ps_sum} mm) and {s_symbol2} ({psf_sum} mm)", loc="left")
            # plots
            plt.plot(self.data[self.dtfield], self.data["P_s"] / n_dt, color=specs["color_P_s"], label=s_symbol1)
            plt.plot(self.data[self.dtfield], self.data["P_sf"] / n_dt, color=specs["color_P_sf"], label=s_symbol2)
            # normalize X axis
            plt.xlim(ls_dates)
            # normalize Y axis
            ymax_p = specs["ymax_P"]
            if ymax_p is None:
                ymax_p = 1.2 * n_pmax
            plt.ylim(0, ymax_p)
            plt.ylabel("mm/{}".format(self.params["dt"]["units"].lower()))

            # ------------ C plot ------------
            n_smax = self.data["C"].max()
            ax = fig.add_subplot(gs[2, 0], sharex=ax)
            s_symbol1 = self.vars["C"]["TeX"]
            plt.title("{} ($\mu$ = {} mm)".format(s_symbol1, round(self.data["C"].mean(), 1)), loc="left")
            plt.plot(self.data[self.dtfield], self.data["C"], color=specs["color_C"], label=s_symbol1)
            # normalize X axis
            plt.xlim(ls_dates)
            plt.ylabel("mm")
            # normalize Y axis
            ymax_s = specs["ymax_C"]
            if ymax_s is None:
                ymax_s = 1.2 * n_smax
            if ymax_s <= 0:
                plt.ylim(-1, 1)
            else:
                plt.ylim(0, ymax_s)

        if mode == "surface":
            # ------------ P plot ------------
            n_pmax = self.data["P"].max() / n_dt
            ax = fig.add_subplot(gs[0, 0])
            plt.plot(self.data[self.dtfield], self.data["P"] / n_dt, color=specs["color_P"],
                     zorder=2, label=self.vars["P"]["TeX"])
            s_title1 = "$P$ ({} mm)".format(round(self.data["P"].sum(), 1))
            plt.title(s_title1, loc="left")

            # normalize X axis
            plt.xlim(ls_dates)
            # normalize Y axis
            ymax_p = specs["ymax_P"]
            if ymax_p is None:
                ymax_p = 1.2 * n_pmax
            plt.ylim(0, ymax_p)
            plt.ylabel("mm/{}".format(self.params["dt"]["units"].lower()))

            # ------------ Ps plot ------------
            n_emax = self.data["P_s"].max() / n_dt
            ps_sum = round(self.data["P_s"].sum(), 1)
            # titles
            s_symbol1 = self.vars["P_s"]["TeX"]
            s_title2 = "{} ({} mm)".format(s_symbol1, round(self.data["P_s"].sum(), 1))
            plt.title(f"{s_title1} and {s_title2}", loc="left")
            # plots
            plt.plot(self.data[self.dtfield], self.data["P_s"] / n_dt, color=specs["color_P_s"], label=s_symbol1)
            plt.legend(loc="upper left")

            # ------------ E plot ------------
            s_symbol1 = self.vars["E_s"]["TeX"]
            s_symbol2 = self.vars["E_c"]["TeX"]
            es_sum = round(self.data["E_s"].sum(), 1)
            ec_sum = round(self.data["E_c"].sum(), 1)
            ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
            ax2.plot(self.data[self.dtfield], self.data["E_c"] / n_dt, c="tab:orange", zorder=0, label=s_symbol1)
            ax2.plot(self.data[self.dtfield], self.data["E_s"] / n_dt, c="tab:red", zorder=1, label=s_symbol2)
            es_max = np.max([self.data["E_s"].max(), self.data["E_c"].max()])
            if es_max <= 0.001:
                ax2.set_ylim(-1, 1)
            else:
                ax2.set_ylim(0, 1.2 * es_max / n_dt)
            s_title1 = "{} ({} mm)".format(s_symbol1, ec_sum)
            s_title2 = "{} ({} mm)".format(s_symbol2, es_sum)
            ax2.set_title(f"{s_title1} and {s_title2}", loc="right")
            ax2.set_ylabel("mm/d")
            plt.legend(loc="upper right")

            # ------------ Qof, Quf, Qif plot ------------
            ax = fig.add_subplot(gs[1, 0], sharex=ax)
            s_symbol1 = self.vars["Q_of"]["TeX"]
            s_symbol2 = self.vars["Q_uf"]["TeX"]
            s_symbol3 = self.vars["Q_if"]["TeX"]
            s_symbol4 = self.vars["Q_hf"]["TeX"]
            # titles
            s_title1 = "{} ({} mm)".format(s_symbol1, round(self.data["Q_of"].sum(), 1))
            s_title2 = "{} ({} mm)".format(s_symbol2, round(self.data["Q_uf"].sum(), 1))
            s_title3 = "{} ({} mm)".format(s_symbol3, round(self.data["Q_if"].sum(), 1))
            plt.title(f"{s_title1}, {s_title2} and {s_title3}", loc="left")
            plt.plot(self.data[self.dtfield], self.data["Q_hf"] / n_dt, color="navy", label=s_symbol4)
            plt.plot(self.data[self.dtfield], self.data["Q_of"] / n_dt, color="magenta", label=s_symbol1)
            plt.plot(self.data[self.dtfield], self.data["Q_uf"] / n_dt, color="red", label=s_symbol2)
            plt.plot(self.data[self.dtfield], self.data["Q_if"] / n_dt, color="green", label=s_symbol3)
            plt.ylim(0, ymax_p)
            ax.set_ylabel("mm/d")
            plt.legend(loc="upper left")

            # ------------ Runoff coef plot ------------
            ax2 = ax.twinx()
            s_symbol1 = self.vars["Q_of_f"]["TeX"]
            plt.plot(self.data[self.dtfield], self.data["Q_of_f"], color="gray", label=s_symbol1)
            plt.ylim(-1, 2)
            ax2.set_yticks([0, 1])
            ax2.set_ylabel("mm/mm")
            f_mean = round(self.data["Q_of_f"].mean(), 2)
            f_max = round(self.data["Q_of_f"].max(), 2)
            ax2.set_title(f"{s_symbol1} ($\mu$ = {f_mean}; max = {f_max})", loc="right")
            plt.legend(loc="upper right")

            # ------------ S plot ------------
            n_smax = self.data["S"].max()
            ax = fig.add_subplot(gs[2, 0], sharex=ax)
            s_symbol1 = self.vars["S"]["TeX"]
            plt.title("{} ($\mu$ = {} mm)".format(s_symbol1, round(self.data["S"].mean(), 1)), loc="left")
            plt.plot(self.data[self.dtfield], self.data["S"], color=specs["color_S"], label=s_symbol1)
            plt.hlines(
                y=self.params["S_uf_a"]["value"],
                xmin=self.data[self.dtfield].min(),
                xmax=self.data[self.dtfield].max(),
                colors="orange",
                label=self.params["S_uf_a"]["TeX"]
            )
            plt.hlines(
                y=self.params["S_uf_cap"]["value"],
                xmin=self.data[self.dtfield].min(),
                xmax=self.data[self.dtfield].max(),
                colors="purple",
                label=self.params["S_uf_cap"]["TeX"]
            )
            plt.hlines(
                y=self.params["S_of_a"]["value"],
                xmin=self.data[self.dtfield].min(),
                xmax=self.data[self.dtfield].max(),
                colors="red",
                label=self.params["S_of_a"]["TeX"]
            )
            plt.legend(loc="upper left")
            # ticks

            # normalize X axis
            plt.xlim(ls_dates)
            plt.ylabel("mm")
            # normalize Y axis
            ymax_s = specs["ymax_S"]
            if ymax_s is None:
                ymax_s = 1.2 * n_smax
            if ymax_s <= 0:
                plt.ylim(-1, 1)
            else:
                plt.ylim(0, ymax_s)

        if mode == "soil":
            # ------------ Qif plot ------------
            ax = fig.add_subplot(gs[0, 0])
            n_emax = self.data["Q_if"].max() / n_dt
            ps_sum = round(self.data["Q_if"].sum(), 1)
            # titles
            s_symbol0 = self.vars["P_s"]["TeX"]
            s_symbol1 = self.vars["Q_if"]["TeX"]
            s_symbol2 = self.vars["Q_vf"]["TeX"]
            s_title0 = "{} ({} mm)".format(s_symbol0, round(self.data["P_s"].sum(), 1))
            s_title1 = "{} ({} mm)".format(s_symbol1, round(self.data["Q_if"].sum(), 1))
            s_title2 = "{} ({} mm)".format(s_symbol2, round(self.data["Q_vf"].sum(), 1))
            plt.title(f"{s_title0}, {s_title1} and {s_title2}", loc="left")
            # plots
            plt.plot(self.data[self.dtfield], self.data["Q_if"] / n_dt, color=specs["color_Q_if"], label=s_symbol1)
            plt.plot(self.data[self.dtfield], self.data["Q_vf"] / n_dt, color="steelblue", label=s_symbol2)
            plt.legend(loc="upper left", ncols=3)
            # normalize X axis
            plt.xlim(ls_dates)
            plt.ylabel("mm/d")
            # normalize Y axis
            ymax_qif = specs["ymax_Q_if"]
            if ymax_qif is None:
                ymax_qif = 1.2 * n_emax
            plt.ylim(0, ymax_qif)

            # ------------ E plot ------------
            s_symbol1 = self.vars["E_s"]["TeX"]
            s_symbol2 = self.vars["E_c"]["TeX"]
            s_symbol3 = self.vars["E_t"]["TeX"]
            s_symbol4 = self.vars["E_pot"]["TeX"]
            es_sum = round(self.data["E_s"].sum(), 1)
            ec_sum = round(self.data["E_c"].sum(), 1)
            et_sum = round(self.data["E_t"].sum(), 1)
            ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
            ax2.plot(self.data[self.dtfield], self.data["E_pot"] / n_dt, c="tab:gray", alpha=0.5, zorder=2, linestyle="--",
                     label=s_symbol4)
            ax2.plot(self.data[self.dtfield], self.data["E_c"] / n_dt, c="tab:orange", zorder=0, label=s_symbol1)
            ax2.plot(self.data[self.dtfield], self.data["E_s"] / n_dt, c="tab:red", zorder=1, label=s_symbol2)
            ax2.plot(self.data[self.dtfield], self.data["E_t"] / n_dt, c="tab:green", zorder=1, label=s_symbol3)

            es_max = np.max([self.data["E_s"].max(), self.data["E_c"].max(), self.data["E_t"].max()])
            if es_max <= 0.001:
                ax2.set_ylim(-1, 1)
            else:
                ax2.set_ylim(0, 1.2 * es_max / n_dt)
            s_title1 = "{} ({} mm)".format(s_symbol1, ec_sum)
            s_title2 = "{} ({} mm)".format(s_symbol2, es_sum)
            s_title3 = "{} ({} mm)".format(s_symbol3, et_sum)
            ax2.set_title(f"{s_title1}, {s_title2} and {s_title3}", loc="right")
            ax2.set_ylabel("mm/d")
            plt.legend(loc="upper right", ncols=4)

            # ------------ Qbf  ------------
            ax = fig.add_subplot(gs[1, 0], sharex=ax)
            s_symbol1 = self.vars["Q_bf"]["TeX"]
            # titles
            s_title1 = "{} ({} mm)".format(s_symbol1, round(self.data["Q_bf"].sum(), 1))
            plt.title(f"{s_title1}", loc="left")
            plt.plot(self.data[self.dtfield], self.data["Q_bf"] / n_dt, color="navy", label=s_symbol1)
            qb_max = self.data["Q_bf"].max() / n_dt
            if qb_max == 0:
                plt.ylim(-1, 1)
            else:
                plt.ylim(0, 1.2 * qb_max)
            ax.set_ylabel("mm/d")
            plt.legend(loc="upper left", ncols=1)

            # ------------ G plot ------------
            n_smax = self.data["G"].max()
            ax = fig.add_subplot(gs[2, 0], sharex=ax)
            s_symbol1 = self.vars["V"]["TeX"]
            s_symbol2 = self.vars["G"]["TeX"]
            s_symbol3 = self.vars["D"]["TeX"]
            s_symbol4 = self.vars["D_v"]["TeX"]
            s_title1 = "{} ($\mu$ = {} mm)".format(s_symbol1, round(self.data["V"].mean(), 1))
            s_title2 = "{} ($\mu$ = {} mm)".format(s_symbol2, round(self.data["G"].mean(), 1))
            s_title3 = "{} ($\mu$ = {} mm)".format(s_symbol3, round(self.data["D"].mean(), 1))
            plt.title(f"{s_title1}, {s_title2} and {s_title3}", loc="left")
            plt.plot(self.data[self.dtfield], self.data["V"], color=specs["color_V"], label=s_symbol1)
            plt.plot(self.data[self.dtfield], self.data["G"], color=specs["color_G"], label=s_symbol2)
            plt.plot(self.data[self.dtfield], self.data["D"], color="black", label=s_symbol3)
            plt.plot(self.data[self.dtfield], self.data["D_v"], color="black", linestyle="--", label=s_symbol4)
            plt.hlines(
                y=self.params["G_cap"]["value"],
                xmin=self.data[self.dtfield].min(),
                xmax=self.data[self.dtfield].max(),
                colors="orange",
                linestyles="--",
                label=self.params["G_cap"]["TeX"]
            )
            plt.hlines(
                y=self.params["G_cap"]["value"] - self.params["D_et_a"]["value"],
                xmin=self.data[self.dtfield].min(),
                xmax=self.data[self.dtfield].max(),
                colors="green",
                linestyles="--",
                label=self.params["D_et_a"]["TeX"]
            )
            plt.legend(loc="upper left", ncols=6)
            # normalize X axis
            plt.xlim(ls_dates)
            plt.ylabel("mm")
            # normalize Y axis
            ymax_s = specs["ymax_G"]
            if ymax_s is None:
                ymax_s = 1.2 * n_smax
            plt.ylim(0, ymax_s)

        ax.tick_params(axis='x', labelsize=10, labelcolor='black')

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
                return None

    @staticmethod
    def propagate_inflow(inflow, unit_hydrograph):
        """Flow routing model based on provided unit hydrograph.
        Inflow and Unit Hydrograh arrays are expected to be the same size.
        # todo [feature] improve so it can handle a smaller Unit Hydrograph (using slicing, etc)

        :param inflow: 1d numpy array of inflow
        :type inflow: :class:`numpy.ndarray`
        :param unit_hydrograph: 1d numpy array of Unit Hydrograph (sum=1)
        :type unit_hydrograph: :class:`numpy.ndarray`
        :return: outflow array
        :rtype: :class:`numpy.ndarray`
        """
        size = len(inflow)
        uh = unit_hydrograph[:]
        # loop over time
        for t in range(0, size):
            if t == 0:
                # create outflow vector
                outflow = inflow[t] * uh
            else:
                # convolution over time
                outflow[t:] = outflow[t:] + (inflow[t] * uh[:size - t])
        return outflow

class Local(Global):
    """This is a local Rainfall-Runoff model. Simulates the the catchment globally and locally
    by applying downscaling methods. Expected inputs:

    - `clim.csv` for Precipitation (P) and Potential Evapotranspiration (E_pot).
    - `path_areas.csv` for deriving the geomorphic unit hydrograph.
    - # todo [complete]

    # todo [major docstring] examples

    """

    def __init__(self, name="MyLocal", alias="Loc001"):
        # todo [docstring]
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "Local"

        # local input file
        # todo [RESUME HERE]
        print("Hello World!")
        # lulc table
        # soil table
        # maps
        # matrix
        # todo [reentry note]
        '''
        Instructions: make a model that handles both G2G and HRU approach without.
        The trick seems to have a area matrix that is the same...        
        '''




if __name__ == "__main__":
    print("Hello World!")
