"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all dataset objects of plans.

Copyright (C) 2022 Iporã Brito Possantti

************ GNU GENERAL PUBLIC LICENSE ************

https://www.gnu.org/licenses/gpl-3.0.en.html

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

__version__ = "0.1.0"


class DailySeries:
    """
    The basic daily time series object

    Example of using this object:

    """

    def __init__(self, metadata, varfield, datefield="Date"):
        '''
        Deploy the daily time series object

         Keys in metadata must include:
        * Name -- name of dataset :class:`str`
        * Variable -- name of variable :class:`str`
        * Latitude -- latitude of dataset :class:`float`
        * Longitude -- latitude of dataset :class:`float`
        * CRS -- standard name of Coordinate Reference System :class:`str`

        :param data: DataFrame of time series. Must include the variable and date fields.
        :type data: :class:`pandas.DataFrame`
        :param metadata: Metadata of time series.
        :type metadata: dict
        :param varfield: name of variable field
        :type varfield: str
        :param datefield: name of date field (default: Date)
        :type datefield: str
        '''
        # -------------------------------------
        # set basic attributes
        self.data = None  # start with no data
        self.metadata = metadata
        self.varfield = varfield
        self.datefield = datefield


    def __str__(self):
        s_aux = "\n{}\n{}\n".format(
            self.metadata,
            self.data
        )
        return s_aux


    def set_data(self, dataframe):
        '''
        Set the data from incoming pandas DataFrame.
        Note: must include varfield and datefield
        :param dataframe: incoming pandas DataFrame
        :type dataframe: :class:`pandas.DataFrame`
        '''
        # slice only interest fields
        self.data = dataframe[[self.datefield, self.varfield]]
        # ensure datetime format
        self.data[self.datefield] = pd.to_datetime(self.data[self.datefield])


    def load_data(self, file):
        '''
        Load data from CSV file.

        CSV file must:
        * be a txt file
        * use ; as field separator
        * include datefields and varfields

        :param file: path to CSV file
        :type file: str
        '''
        self.file = file
        # -------------------------------------
        # import data
        df_aux = pd.read_csv(
            self.file,
            sep=";",
            parse_dates=[self.datefield]
        )
        # slice only varfield and datefield from dataframe
        self.data = df_aux[[self.datefield, self.varfield]]


    def export_data(self, folder):
        '''
        Export dataset to CSV file
        :param folder: path to output directory
        :type folder: str
        '''
        self.data.to_csv(
            "{}/{}_{}.txt".format(folder, self.metadata["Variable"], self.metadata["Name"]),
            sep=";",
            index=False
        )


    def resample_sum(self, period="MS"):
        '''
        Resampler method for daily time series using the .sum() function
        :param period: pandas standard period code

        * `W-MON` -- weekly starting on mondays
        * `MS` --  monthly on start of month
        * `QS` -- quarterly on start of quarter
        * `YS` -- yearly on start of year

        :type period: str
        :return: resampled time series
        :rtype: :class:`pandas.DataFrame`
        '''
        df_aux = self.data.set_index(self.datefield)
        df_aux = df_aux.resample(period).sum()[self.varfield]
        df_aux = df_aux.reset_index()
        return df_aux


    def resample_mean(self, period="MS"):
        '''
        Resampler method for daily time series using the .mean() function
        :param period: pandas standard period code

        * `W-MON` -- weekly starting on mondays
        * `MS` --  monthly on start of month
        * `QS` -- quarterly on start of quarter
        * `YS` -- yearly on start of year

        :type period: str
        :return: resampled time series
        :rtype: :class:`pandas.DataFrame`
        '''
        df_aux = self.data.set_index(self.datefield)
        df_aux = df_aux.resample(period).mean()[self.varfield]
        df_aux = df_aux.reset_index()
        return df_aux


class PrecipitationSeries(DailySeries):
    """
    The precipitation daily time series object

    Example of using this object:

    """

    def __init__(self, name, file, varfield, datefield, location):
        # ---------------------------------------------------
        # use superior initialization
        super().__init__(name, file, "P", varfield, datefield, location)
        # override varfield
        self.data = self.data.rename(columns={varfield: "P"})


class MyClass:
    """A new Object"""
    
    def __init__(self, s_name="MyName"):
        """
        Initiation of the MyClass object.
        
        :param s_name: Name of object.
        :type s_name: str
        """
        self.name = s_name
        print(self.name)
    
    def do_stuff(self, s_str1, n_value):
        """
        A demo method.
        
        :param s_str1: string to print.
        :type s_str1: str
        :param n_value: value to print.
        :type n_value: float
        :return: a concatenated string
        :rtype: str
        
        """
        s_aux = s_str1 + str(n_value)
        return s_aux


def my_function(kind=None):
    """
    Return a list of random ingredients as strings.
    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    return ["shells", "gorgonzola", "parsley"]


if __name__ == '__main__':

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    meta = {
        "Name": "MyTS",
        "Variable": "Random",
        "Latitude": -30,
        "Longitude": -51,
        "CRS": "SIRGAS 2000"
    }

    ts = DailySeries(
        metadata=meta,
        varfield="R"
    )

    df = pd.DataFrame(
        {
            "Date": pd.date_range(start="2000-01-01", end="2020-01-01", freq="D"),
            "R": 0
        }
    )
    df["R"] = np.random.normal(loc=100, scale=5, size=len(df))

    ts.set_data(dataframe=df)
    print(ts)

