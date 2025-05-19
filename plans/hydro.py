"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all hydrology functions of PLANS.

Copyright (C) 2022 Iporã Brito Possantti
"""
import os.path
from pathlib import Path

import matplotlib.pyplot as plt

from root import DataSet
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

        # parameters
        self.filepath_params = None
        self._set_model_params()

        # expected structure for folder data:
        self.workplace = {
            "inputs": {
                "params": {
                    "filepath": "/inputs/params.csv",
                    "description": "Global model parameters"
                }
            },

        }

    def _set_fields(self):
        """Set fields names.
        Expected to increment superior methods.

        """
        # ------------ call super ----------- #
        super()._set_fields()
        # Attribute fields
        self.workplace_field = "Workplace"

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
            self.workplace_field: self.folder_data,
        }

        # update
        dict_meta.update(dict_meta_local)
        return dict_meta

    def update(self):
        """Refresh all mutable attributes based on data (includins paths).
        Base method. This is overwrinting the superior method.

        :return: None
        :rtype: None
        """
        # refresh all mutable attributes

        # set fields
        self._set_fields()

        if self.data is not None:
            # data size (rows)
            self.size = len(self.data)

        # view specs at the end
        self._set_view_specs()

        # update dt
        self.update_dt()

        # ... continues in downstream objects ... #
        return None

    def load_params(self):
        """Load parameter data from expected file.

        :return: None
        :rtype: None
        """
        # load parameters expected file
        self.filepath_params = Path(self.folder_data + self.workplace["inputs"]["params"]["filepath"])

        # load dataframe
        df_ = pd.read_csv(self.filepath_params, sep=";", encoding="utf-8", dtype=str)
        ls_input_params = set(df_["Param"])
        # parse to dict
        for k in self.params:
            if k in set(ls_input_params):
                self.params[k]["value"] = self.params[k]["dtype"](df_.loc[df_['Param'] == k, 'Value'].values[0])
                self.params[k]["units"] = df_.loc[df_['Param'] == k, 'Units'].values[0]

        # handle input dt
        self.update_dt()

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

    def load_data(self, folder_data):
        """Load simulation data from folder. Expected to overwrite superior methods.

        :param folder_data: file path to data.
        :type folder_data: str
        :return: None
        :rtype: None
        """

        # -------------- overwrite relative path input -------------- #
        self.folder_data = os.path.abspath(folder_data)

        # load parameters from expected file
        self.load_params()

        # load observed data from expected file


        # -------------- update other mutables -------------- #
        self.update()

        # ... continues in downstream objects ... #

        return None

    def setup(self):
        """Set simulation data

        .. warning::

            This method overwrites model data.


        :return: None
        :rtype: None
        """

        # ensure to update mutables
        self.update()

        # get timestep series
        vc_ts = Model.get_timestep_series(
            start_time=self.params["t0"]["value"],
            end_time=self.params["tN"]["value"],
            time_unit=self.params["dt_freq"]["value"]
        )
        vc_t = np.linspace(start=0, stop=len(vc_ts)*self.params["dt"]["value"],num=len(vc_ts), dtype=np.float64)

        # set dataframe
        self.data = pd.DataFrame(
            {
                "t": vc_t,
                "DateTime": vc_ts,
            }
        )

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
        self.data = df.copy()

        return None


    def run(self):
        """Simulate model.

        :return: None
        :rtype: None
        """
        self.setup()
        self.solve()
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
