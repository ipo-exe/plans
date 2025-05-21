"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all hydrology functions of PLANS.

Copyright (C) 2022 Iporã Brito Possantti
"""
import os.path
from pathlib import Path
import matplotlib.pyplot as plt
from plans.root import DataSet
from plans.datasets import TimeSeries
from plans.analyst import Bivar
import numpy as np
import pandas as pd

class Model(DataSet):
    """The core ``Model`` base/demo object.
    Expected to hold one :class:`pandas.DataFrame` as simulation data and a dictionary as parameters
    This is a Base and Dummy object that simulates a Linear Storage. Expected to be implemented downstream for
    custom applications.

    """

    def __init__(self, name="MyHydroModel", alias="HM01"):
        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)

        # overwriters
        self.object_alias = "HM"

        # file model
        self.file_model = None

        # parameters
        self.file_params = None # global parameters
        self._set_model_params()

        # simulation data
        self.data = None

        # observed data
        self.file_data_qobs = None
        self.data_qobs = None


        # defaults
        self.dtfield = "DateTime"

        # expected structure for folder data:
        '''
        self.workplace = {
            "inputs": {
                "params": {
                    "filepath": "/inputs/params.csv",
                    "description": "Global model parameters"
                }
            },

        }
        '''
        # run file
        self.file_model = None

    def _set_fields(self):
        """Set fields names.
        Expected to increment superior methods.

        """
        # ------------ call super ----------- #
        super()._set_fields()
        # Attribute fields
        self.file_params_fields = "File Parameters"
        self.folder_data_field = "Folder Data"

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
                "kind": "conceptual"
            },
            "S0": {
                "value": None,
                "units": "mm", # default is mm
                "dtype": np.float64,
                "description": "Storage initial condition",
                "kind": "conceptual"
            },
            "dt": {
                "value": None,
                "units": None,
                "dtype": np.float64,
                "description": "Time Step in k units",
                "kind": "procedural"
            },
            "dt_freq": {
                "value": None,
                "units": "unitless",
                "dtype": str,
                "description": "Time Step frequency flag",
                "kind": "procedural"
            },
            "t0": {
                "value": None,
                "units": "timestamp",
                "dtype": str,
                "description": "Simulation start",
                "kind": "procedural"
            },
            "tN": {
                "value": None,
                "units": "timestamp",
                "dtype": str,
                "description": "Simulation end",
                "kind": "procedural"
            },


        }
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
        del dict_meta[self.filedata_field]

        # customize local metadata:
        dict_meta_local = {
            self.file_params_fields: self.file_params,
            self.folder_data_field: self.folder_data,
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
        self.file_params = Path(dict_setter[self.file_params_fields])
        self.folder_data = Path(dict_setter[self.folder_data_field])

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
        if s_dt_unit_tag == '1'+ self.params["k"]["units"]:
            pass
        else:
            # compute time step in k units
            ft_aux = Model.get_timestep_factor(
                from_unit=s_dt_unit_tag,
                to_unit='1'+ self.params["k"]["units"],
            )
            # update
            self.params["dt"]["value"] = ft_aux
            self.params["dt"]["units"] = self.params["k"]["units"][:]
            self.params["dt_freq"]["value"] = s_dt_unit_tag

        return None

    def load_params(self):
        """Load parameter data

        :return: None
        :rtype: None
        """
        # -------------- load parameter data -------------- #

        # >>> develop logic

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

    def load_model(self):
        self.load_params()
        self.load_data()

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
            time_unit=self.params["dt_freq"]["value"]
        )
        vc_t = np.linspace(start=0, stop=len(vc_ts)*self.params["dt"]["value"],num=len(vc_ts), dtype=np.float64)

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
        """Solve the model for input and initial conditions by numerical methods.

        .. warning::

            This method overwrites model data.

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
    def get_timestep_series(start_time: str, end_time: str, time_unit: str) -> pd.DatetimeIndex:
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


class LinearStorage(Model):

    def __init__(self, name="MyLinearStorage", alias="LS01"):
        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)

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
                "kind": "conceptual"
            },
            "S0": {
                "value": None,
                "units": "mm", # default is mm
                "dtype": np.float64,
                "description": "Storage initial condition",
                "kind": "conceptual"
            },
            "dt": {
                "value": None,
                "units": None,
                "dtype": np.float64,
                "description": "Time Step in k units",
                "kind": "procedural"
            },
            "dt_freq": {
                "value": None,
                "units": "unitless",
                "dtype": str,
                "description": "Time Step frequency flag",
                "kind": "procedural"
            },
            "t0": {
                "value": None,
                "units": "timestamp",
                "dtype": str,
                "description": "Simulation start",
                "kind": "procedural"
            },
            "tN": {
                "value": None,
                "units": "timestamp",
                "dtype": str,
                "description": "Simulation end",
                "kind": "procedural"
            },


        }
        return None

    def load_params(self):
        """Load parameter data

        :return: None
        :rtype: None
        """
        # load dataframe
        df_ = pd.read_csv(self.file_params, sep=self.file_csv_sep, encoding="utf-8", dtype=str)
        ls_input_params = set(df_["Parameter"])

        # parse to dict
        for k in self.params:
            if k in set(ls_input_params):
                self.params[k]["value"] = self.params[k]["dtype"](df_.loc[df_['Parameter'] == k, 'Value'].values[0])
                self.params[k]["units"] = df_.loc[df_['Parameter'] == k, 'Units'].values[0]

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
        self.data_obs = pd.read_csv(file_qobs, sep=self.file_csv_sep, encoding=self.file_encoding, parse_dates=[self.dtfield])

        # -------------- load observation data -------------- #
        # >>> develop logic in downstream objects

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
        self.data["S"] = 0.0
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

        # analytical solution:
        df["S_a"].values[:] = self.params["S0"]["value"] * np.exp(-df["t"].values/k)

        # numerical solution:
        # loop over (Euler Method)
        for i in range(n_steps - 1):
            # Qt = dt * St / k
            df["Q"].values[i] = df["S"].values[i] * dt / k
            df["S"].values[i + 1] = df["S"].values[i] - df["Q"].values[i]

        # reset data
        df = pd.merge(left=df, right=self.data_obs, on=self.dtfield, how="left")
        self.data = df.copy()

        return None

    def evaluate(self):
        df = self.data.copy().dropna()
        bv = Bivar(df_data=df, x_name="S", y_name="S_obs")
        bv.view(show=True)

    # todo make a definitive view() method
    def view(self, show=True):
        plt.figure(figsize=(8, 4))
        plt.plot(self.data["DateTime"], self.data["S"], color="b", label="Simulated")
        plt.plot(self.data["DateTime"], self.data["S_obs"], ".", color="k", label="Observerd")
        plt.xlim(self.data["DateTime"].min(), self.data["DateTime"].max())
        plt.ylim(0, 600)
        plt.ylabel("S (mm)")
        plt.tight_layout()
        plt.legend()
        if show:
            plt.show()

    def view2(self, show=True):
        specs = self.view_specs.copy()
        ts = TimeSeries(name=self.name, alias=self.alias)
        ts.set_data(
            input_df=self.data,
            input_dtfield=self.dtfield,
            input_varfield="S"
        )
        fig = ts.view(show=False, return_fig=True)
        # obs series plot
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

if __name__ == "__main__":
    print("Hi")
    import pprint
    from analyst import Bivar
    m = Model()
    m.load_data(folder_data="C:/plans/testing/linear")
    pprint.pprint(m.params)
    m.run()
    pprint.pprint(m.data)
    rmse = Bivar.rmse(pred=m.data["S"].values, obs=m.data["S_a"])
    print(rmse)

    m.params["dt"]["value"] = 2
    m.params["dt"]["units"] = "h"

    m.run()
    pprint.pprint(m.data)
    rmse = Bivar.rmse(pred=m.data["S"].values, obs=m.data["S_a"])
    print(rmse)
