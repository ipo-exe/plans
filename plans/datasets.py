"""

This module provides a set of dataset objects for input, output and processing.

.. code-block:: python
    # Import Datasets
    import datasets
    # Call its only function
    datasets.my_function(kind=["cheeses"])

This file is part of the PLANS project.
Licensed under the GNU GENERAL PUBLIC LICENSE (GPL 3).
For more information, see https://github.com/ipo-exe/plans

"""

__version__ = "0.1.0"


class DailySeries:
    """
    The basic daily time series object

    Example of using this object:

    """

    def __init__(self, name, file, varfield, datefield, location):
        '''
        Deploy the basic Time Series object

        :param name: Time Series name
        :type name: str
        :param file: dataset path to `.txt` CSV file (separator = ;)
        :type file: str
        :param varfield: name of the variable field in file
        :type varfield: str
        :param datefield: name of date field in file
        :type datefield: str
        :param location: dictionary of `lat`, `long` and `CRS` of measuring station
        :type location: dict
        '''
        # -------------------------------------
        # set basic attributes
        self.name = name
        self.file = file
        self.varfield = varfield
        self.datefield = datefield
        self.location = location

        # -------------------------------------
        # import data
        df_aux = pd.read_csv(
            self.file,
            sep=";",
            parse_dates=[self.datefield]
        )
        # slice only varfield and datefield
        self.data = df_aux[[self.datefield, self.varfield]]


    def __str__(self):
        s_aux = "\n{}\n{}\n{}\n".format(
            self.name,
            self.location,
            self.data
        )
        return s_aux


    def _dummy(self):
        return self.data[self.varfield].mean()


    def resample_sum(self, period):
        '''
        Resampler method for daily time series
        :param period: pandas standard period code

        `W-MON` -- weekly starting on mondays
        `MS` --  monthly on start of month
        `QS` -- quarterly on start of quarter
        `YS` -- yearly on start of year

        :type period: str
        :return: resampled pandas dataframe
        :rtype: :class:`pandas.DataFrame`
        '''
        df_aux = self.df_daily.set_index(self.datefield)
        df_aux = df_aux.resample(period).sum()[self.varfield]
        df_aux = df_aux.reset_index()
        return df_aux

    def resample_mean(self, period):
        print(period)


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

    ts = DailySeries(
        name="MyTS",
        file="C:/bin/calib_series.txt",
        varfield="Prec",
        datefield="Date",
        location={
            "lat": -30.0,
            "long": -51.0,
            "CRS": "SIRGAS 2000"
        }
    )

    print(ts)
    n = ts._dummy()
    print(n)

    ts_p = PrecipitationSeries(
        name="MyTS",
        file="C:/bin/calib_series.txt",
        varfield="Prec",
        datefield="Date",
        location={
            "lat": -30.0,
            "long": -51.0,
            "CRS": "SIRGAS 2000"
        }
    )
    print(ts_p.data)
    print(ts_p._dummy())
    print(ts_p.new_method())
