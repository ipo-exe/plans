"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all dataset objects of PLANS.

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


    def plot_basic_view(self, show=False, folder="C:/data", filename="histogram", specs=None, dpi=96):
        '''
        Plot series basic view

        :type show: bool
        :param folder: output folder
        :type folder: str
        :param filename: image file name
        :type filename: str
        :param specs: specification dictionary
        :type specs: dict
        :param dpi: image resolution (default = 96)
        :type dpi: int
        '''

        from analyst import Univar

        plt.style.use('seaborn-v0_8')

        # get univar object
        uni = Univar(data=self.data[self.varfield].values)

        # get specs
        default_specs = {
            "color": "tab:grey",
            "suptitle": "Series Overview",
            "a_title": "Series",
            "b_title": "Histogram",
            "c_title": "CFC",
            "width": 5 * 1.618,
            "height": 5,
            "ylabel": "units",
            "ylim": (0, 1.2 * self.data[self.varfield].max()),
            "a_xlabel": "Date",
            "b_xlabel": "Frequency",
            "c_xlabel": "Probability",
            "a_data_label": "Data Series",
            "a_mavg_label": "Moving Average",
            "mavg period": 10,
            "mavg color": "tab:blue",
            "nbins": uni.nbins_fd()
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs


        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        gs = mpl.gridspec.GridSpec(4, 5, wspace=0.5, hspace=0.9, left=0.05, bottom=0.1, top=0.9, right=0.95)
        fig.suptitle(specs["suptitle"])

        # Series
        plt.subplot(gs[0:3, :3])
        plt.title("a. {}".format(specs["a_title"]), loc='left')
        plt.plot(
            self.data[self.datefield],
            self.data[self.varfield],
            marker='o',
            label="Data Series",
            color=specs["color"]
        )
        plt.plot(
            self.data[self.datefield],
            self.data[self.varfield].rolling(specs["mavg period"], min_periods=2).mean(),
            label=specs["a_mavg_label"],
            color=specs["mavg color"]
        )
        plt.ylim(specs["ylim"])
        plt.xlim(self.data[self.datefield].values[0], self.data[self.datefield].values[-1])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["a_xlabel"])
        plt.legend(
            frameon=True,
            loc=(0.0, -0.3),
            ncol=1
        )
        # Hist
        plt.subplot(gs[0:3, 3:4])
        plt.title("b. {}".format(specs["b_title"]), loc='left')
        plt.hist(
            x=self.data[self.varfield],
            bins=specs["nbins"],
            orientation='horizontal',
            color=specs["color"]
        )
        plt.ylim(specs["ylim"])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["b_xlabel"])

        # CFC
        df_freq = uni.assessment_frequency()
        plt.subplot(gs[0:3, 4:5])
        plt.title("c. {}".format(specs["c_title"]), loc='left')
        plt.plot(df_freq["Empirical Probability"], df_freq["Values"])
        plt.ylim(specs["ylim"])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["c_xlabel"])

        # show or save
        if show:
            plt.show()
        else:
            plt.savefig(
                '{}/{}.png'.format(folder, filename),
                dpi=96
            )


class PrecipSeries(DailySeries):
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

    ts.plot_basic_view(show=True)


