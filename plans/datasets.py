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
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

warnings.filterwarnings("ignore")


# -----------------------------------------
# Utility functions
def dataframe_prepro(dataframe):
    """Utility function for dataframe pre-processing.

    :param dataframe: incoming dataframe
    :type dataframe: :class:`pandas.DataFrame`
    :return: prepared dataframe
    :rtype: :class:`pandas.DataFrame`
    """
    # fix headings
    dataframe.columns = dataframe.columns.str.strip()
    # strip string fields
    for i in range(len(dataframe.columns)):
        # if data type is string
        if str(dataframe.dtypes.iloc[i]) == "base_object":
            # strip all data
            dataframe[dataframe.columns[i]] = dataframe[
                dataframe.columns[i]
            ].str.strip()
    return dataframe


def get_random_colors(size=10, cmap="tab20"):
    """Utility function to get a list of random colors

    :param size: Size of list of colors
    :type size: int
    :param cmap: Name of matplotlib color map (cmap)
    :type cmap: str
    :return: list of random colors
    :rtype: list
    """
    import matplotlib.colors as mcolors

    # Choose a colormap from matplotlib
    _cmap = plt.get_cmap(cmap)
    # Generate a list of random numbers between 0 and 1
    _lst_rand_vals = np.random.rand(size)
    # Use the colormap to convert the random numbers to colors
    _lst_colors = [mcolors.to_hex(_cmap(x)) for x in _lst_rand_vals]
    return _lst_colors


# -----------------------------------------
# Series data structures


class Collection:
    """
    This is the primitive objects collection
    """

    def __init__(self, base_object, name="myCatalog"):
        dct_meta = base_object.get_metadata()
        self.catalog = pd.DataFrame(columns=dct_meta.keys())
        self.collection = dict()
        self.name = name

    def update(self, details=False):
        """
        Update the collection catalog
        :param details: option to update catalog details
        :type details: bool
        :return: None
        :rtype: none
        """
        # update details
        if details:
            # create new catalog
            df_new_catalog = pd.DataFrame(columns=self.catalog.columns)
            for name in self.collection:
                dct_meta = self.collection[name].get_metadata()
                lst_keys = dct_meta.keys()
                _dct = dict()
                for k in lst_keys:
                    _dct[k] = [dct_meta[k]]
                # set new information
                df_aux = pd.DataFrame(_dct)
                # append
                df_new_catalog = pd.concat([df_new_catalog, df_aux], ignore_index=True)
            self.catalog = df_new_catalog.copy()
            del df_new_catalog
        # basic updates
        self.catalog = self.catalog.drop_duplicates(subset="Name", keep="last")
        self.catalog = self.catalog.sort_values(by="Name").reset_index(drop=True)
        return None

    def append(self, new_object):
        """
        Append new object to collection.
        Object is expected to have a .get_metadata() method that returns a dict
        :param new_object: object to append
        :type new_object: object
        :return: None
        :rtype: None
        """
        # append to collection
        self.collection[new_object.name] = new_object
        # set
        dct_meta = new_object.get_metadata()
        dct_meta_df = dict()
        for k in dct_meta:
            dct_meta_df[k] = [dct_meta[k]]
        df_aux = pd.DataFrame(dct_meta_df)
        self.catalog = pd.concat([self.catalog, df_aux], ignore_index=True)
        self.update()
        return None

    def remove(self, name):
        """Remove base_object from collection.

        :param name: object name attribute to remove
        :type name: str
        """
        # delete raster base_object
        del self.collection[name]
        # delete from catalog
        self.catalog = self.catalog.drop(
            self.catalog[self.catalog["Name"] == name].index
        ).reset_index(drop=True)
        return None


class DailySeries:
    """
    The basic daily time series base_object

    """

    def __init__(
        self,
        name,
        varname,
    ):
        """Deploy daily series base dataset

        :param name: name of series
        :type name: str
        :param varname: name of variable
        :type varname: str
        """
        # -------------------------------------
        # set basic attributes
        self.data = None  # start with no data
        self.name = name
        self.varname = varname
        self.datefield = "Date"
        self.file = None

    def set_data(self, dataframe, varfield, datefield="Date"):
        """Set the data from incoming class:`pandas.DataFrame`.

        :param dataframe: incoming :class:`pandas.DataFrame` base_object
        :type dataframe: :class:`pandas.DataFrame`
        :param varfield: name of variable field in the incoming :class:`pandas.DataFrame`
        :type varfield: str
        :param datefield: name of date field in the incoming :class:`pandas.DataFrame`
        :type datefield: str
        """
        # slice only interest fields
        df_aux = dataframe[[datefield, varfield]].copy()
        self.data = df_aux.rename(
            columns={datefield: self.datefield, varfield: self.varname}
        )
        # ensure datetime fig_format
        self.data[self.datefield] = pd.to_datetime(self.data[self.datefield])
        self.data = self.data.sort_values(by=self.datefield).reset_index(drop=True)

    def load_data(self, file, varfield, datefield="Date"):
        """Load data from ``csv`` file.

        :param file: path_main to ``csv`` file
        :type file:
        :param varfield:
        :type varfield:
        :param datefield:
        :type datefield:
        :return:
        :rtype:
        """

        self.file = file
        # -------------------------------------
        # import data
        df_aux = pd.read_csv(self.file, sep=";", parse_dates=[self.datefield])
        # set data
        self.set_data(dataframe=df_aux, varfield=varfield, datefield=datefield)

    def export_data(self, folder):
        """
        Export dataset to ``csv`` file
        :param folder: path_main to output directory
        :type folder: str
        :return: file path_main
        :rtype: str
        """
        s_filepath = "{}/{}_{}.txt".format(folder, self.varname, self.name)
        self.data.to_csv(s_filepath, sep=";", index=False)
        return s_filepath

    def resample_sum(self, period="MS"):
        """
        Resampler method for daily time series using the .sum() function
        :param period: pandas standard period code

        * `W-MON` -- weekly starting on mondays
        * `MS` --  monthly on start of month
        * `QS` -- quarterly on start of quarter
        * `YS` -- yearly on start of year

        :type period: str
        :return: resampled time series
        :rtype: :class:`pandas.DataFrame`
        """
        df_aux = self.data.set_index(self.datefield)
        df_aux = df_aux.resample(period).sum()[self.varname]
        df_aux = df_aux.reset_index()
        return df_aux

    def resample_mean(self, period="MS"):
        """
        Resampler method for daily time series using the .mean() function
        :param period: pandas standard period code

        * `W-MON` -- weekly starting on mondays
        * `MS` --  monthly on start of month
        * `QS` -- quarterly on start of quarter
        * `YS` -- yearly on start of year

        :type period: str
        :return: resampled time series
        :rtype: :class:`pandas.DataFrame`
        """
        df_aux = self.data.set_index(self.datefield)
        df_aux = df_aux.resample(period).mean()[self.varname]
        df_aux = df_aux.reset_index()
        return df_aux

    def view(
        self,
        show=True,
        folder="C:/data",
        filename=None,
        specs=None,
        dpi=150,
        fig_format="jpg",
    ):
        """
        Plot series basic view
        :type show: option to display plot. Default False
        :type show: bool
        :param folder: output folder
        :type folder: str
        :param filename: image file name
        :type filename: str
        :param specs: specification dictionary
        :type specs: dict
        :param dpi: image resolution (default = 96)
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        """
        import matplotlib.ticker as mtick
        from plans.analyst import Univar

        plt.style.use("seaborn-v0_8")

        # get univar base_object
        uni = Univar(data=self.data[self.varname].values)

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
            "ylim": (np.min(uni.data), np.max(uni.data)),
            "a_xlabel": "Date",
            "b_xlabel": "Frequency",
            "c_xlabel": "Probability",
            "a_data_label": "Data Series",
            "skip mavg": False,
            "a_mavg_label": "Moving Average",
            "mavg period": 10,
            "mavg color": "tab:blue",
            "nbins": uni.nbins_fd(),
            "series marker": "o",
            "series linestyle": "-",
            "series alpha": 1.0,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # Deploy figure
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        gs = mpl.gridspec.GridSpec(
            4, 5, wspace=0.5, hspace=0.9, left=0.075, bottom=0.1, top=0.9, right=0.95
        )
        fig.suptitle(specs["suptitle"])

        # plot Series
        plt.subplot(gs[0:3, :3])
        plt.title("a. {}".format(specs["a_title"]), loc="left")
        plt.plot(
            self.data[self.datefield],
            self.data[self.varname],
            linestyle=specs["series linestyle"],
            marker=specs["series marker"],
            label="Data Series",
            color=specs["color"],
            alpha=specs["series alpha"],
        )
        if specs["skip mavg"]:
            pass
        else:
            plt.plot(
                self.data[self.datefield],
                self.data[self.varname]
                .rolling(specs["mavg period"], min_periods=2)
                .mean(),
                label=specs["a_mavg_label"],
                color=specs["mavg color"],
            )
        plt.ylim(specs["ylim"])
        plt.xlim(self.data[self.datefield].min(), self.data[self.datefield].max())
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["a_xlabel"])
        plt.legend(frameon=True, loc=(0.0, -0.35), ncol=1)

        # plot Hist
        plt.subplot(gs[0:3, 3:4])
        plt.title("b. {}".format(specs["b_title"]), loc="left")
        plt.hist(
            x=self.data[self.varname],
            bins=specs["nbins"],
            orientation="horizontal",
            color=specs["color"],
            weights=np.ones(len(self.data)) / len(self.data),
        )
        plt.ylim(specs["ylim"])
        # plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["b_xlabel"])

        # Set the x-axis formatter as percentages
        xticks = mtick.PercentFormatter(xmax=1, decimals=1, symbol="%", is_latex=False)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(xticks)

        # plot CFC
        df_freq = uni.assess_frequency()
        plt.subplot(gs[0:3, 4:5])
        plt.title("c. {}".format(specs["c_title"]), loc="left")
        plt.plot(df_freq["Exceedance"] / 100, df_freq["Values"])
        plt.ylim(specs["ylim"])
        # plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["c_xlabel"])
        plt.xlim(0, 1)
        # Set the x-axis formatter as percentages
        xticks = mtick.PercentFormatter(xmax=1, decimals=0, symbol="%", is_latex=False)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(xticks)

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = self.name
            plt.savefig("{}/{}.{}".format(folder, filename, fig_format), dpi=dpi)
        plt.close(fig)
        return None


class PrecipSeries(DailySeries):
    """
    The precipitation daily time series base_object

    Example of using this base_object:

    """

    def __init__(self, name, file, varfield, datefield, location):
        # ---------------------------------------------------
        # use superior initialization
        super().__init__(name, file, "P", varfield, datefield, location)
        # override varfield
        self.data = self.data.rename(columns={varfield: "P"})


class RatingCurve:
    """
    This is the Rating Curve base_object
    """

    def __init__(self, name="MyRatingCurve"):
        """
        Initiate Rating Curve
        :param name: name of rating curve
        :type name: str
        """
        self.name = name
        self.date_start = None
        self.date_end = None
        self.n = None
        self.hmax = None
        self.hmin = None
        self.field_hobs = "Hobs"
        self.field_qobs = "Qobs"
        self.field_h = "H"
        self.field_q = "Q"
        self.name_h0 = "h0"
        self.name_a = "a"
        self.name_b = "b"
        self.field_ht = "Hobs - h0"
        self.field_htt = "ln(Hobs - h0)"
        self.field_qt = "ln(Qobs)"
        self.field_date = "Date"
        self.units_h = "m"
        self.units_q = "m3/s"
        self.source_data = None
        self.description = None

        # data attribute
        self.data = None
        self.a = 1
        self.b = 1
        self.h0 = 0
        self.rmse = None
        self.e_mean = None
        self.e_sd = None
        self.et_mean = None
        self.et_sd = None

    def __str__(self):
        dct_meta = self.get_metadata()
        lst_ = list()
        lst_.append("\n")
        lst_.append("Object: {}".format(type(self)))
        lst_.append("Metadata:")
        for k in dct_meta:
            lst_.append("\t{}: {}".format(k, dct_meta[k]))
        return "\n".join(lst_)

    def run(self, h):
        """
        Run the model Q = a * (H - h0)^b
        :param h: vector of H
        :type h: :class:`numpy.ndarray` or float
        :return: computed Q
        :rtype: :class:`numpy.ndarray` or float
        """
        return self.a * (np.power((h - self.h0), self.b))

    def extrapolate(self, hmin=None, hmax=None, n_samples=100):
        """
        Extrapolate Rating Curve model. Data is expected to be loaded.
        :param hmin: lower bound
        :type hmin: float
        :param hmax: upper bound
        :type hmax: float
        :param n_samples: number of evenly spaced samples between bounds
        :type n_samples: int
        :return: dataframe of extrapolated data
        :rtype: :class:`pandas.DataFrame`
        """
        # handle bounds
        if hmin is None:
            if self.hmin is None:
                hmin = 0
            else:
                hmin = self.hmin
        if hmax is None:
            if self.hmax is None:
                hmax = 100
            else:
                hmax = self.hmax
        # set h vector
        vct_h = np.linspace(hmin, hmax, n_samples)
        # run
        vct_q = self.run(h=vct_h)
        return pd.DataFrame({self.field_h: vct_h, self.field_q: vct_q})

    def update(self, h0=None, a=None, b=None):
        """
        Update rating curve model
        :param h0: h0 parameter
        :type h0: float or None
        :param a: a parameter
        :type a: float or None
        :param b: b parameter
        :type b: float or None
        :return: None
        :rtype: None
        """

        # set up parameters
        if h0 is not None:
            self.h0 = h0
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        # data model setup
        if self.data is None:
            pass
        else:
            from plans.analyst import Bivar

            # sort values by H
            self.data = self.data.sort_values(by=self.field_hobs).reset_index(drop=True)

            # get model values (reverse transform)
            self.data[self.field_qobs + "_Mean"] = self.run(
                h=self.data[self.field_hobs].values
            )
            # compute the model error
            self.data["e"] = (
                self.data[self.field_qobs] - self.data[self.field_qobs + "_Mean"]
            )

            # get first transform on H
            self.data[self.field_ht] = self.data[self.field_hobs] - self.h0
            # get second transform on H
            self.data[self.field_htt] = np.log(self.data[self.field_ht])
            # get transform on Q
            self.data[self.field_qt] = np.log(self.data[self.field_qobs])

            # get transformed Linear params
            c0t = np.log(self.a)
            c1t = self.b

            # now compute the tranformed model
            s_qt_model = self.field_qt + "_Mean"
            self.data[s_qt_model] = c0t + (c1t * self.data[self.field_htt])
            # compute the transformed error
            self.data["eT"] = self.data[self.field_qt] - self.data[s_qt_model]

            # update attributes
            self.rmse = np.sqrt(np.mean(np.square(self.data["e"])))
            self.e_mean = np.mean(self.data["e"])
            self.e_sd = np.std(self.data["e"])
            # get transformed attributes
            self.et_mean = np.mean(self.data["eT"])
            self.et_sd = np.std(self.data["eT"])

        return None

    def get_metadata(self):
        """
        Get all metadata from base_object
        :return: metadata
        :rtype: dict
        """
        return {
            "Name": self.name,
            "Date_Start": self.date_start,
            "Date_End": self.date_end,
            "N": self.n,
            "h0": self.h0,
            "a": self.a,
            "b": self.b,
            "RMSE": self.rmse,
            "Error_Mean": self.e_mean,
            "Error_SD": self.e_sd,
            "ErrorT_Mean": self.et_mean,
            "ErrorT_SD": self.et_sd,
            "H_max": self.hmax,
            "H_min": self.hmin,
            "H_units": self.units_h,
            "Q_units": self.units_q,
            "Source": self.source_data,
            "Description": self.description,
        }

    def load(
        self,
        table_file,
        hobs_field,
        qobs_field,
        date_field="Date",
        units_q="m3/s",
        units_h="m",
    ):
        """
        Load data from CSV file
        :param table_file: path_main to CSV file
        :type table_file: str
        :param hobs_field: name of observed Stage field
        :type hobs_field: str
        :param qobs_field: name of observed Discharge field
        :type qobs_field: str
        :param date_field: name of Date field
        :type date_field: str
        :param units_q: units of streamflow
        :type units_q: str
        :param units_h: units of stage
        :type units_h: str
        :return: None
        :rtype: None
        """
        _df = pd.read_csv(table_file, sep=";", parse_dates=[date_field])
        _df = dataframe_prepro(dataframe=_df)
        # select fields
        _df = _df[[date_field, hobs_field, qobs_field]].copy()
        # rename columns
        dct_rename = {
            date_field: self.field_date,
            hobs_field: self.field_hobs,
            qobs_field: self.field_qobs,
        }
        _df = _df.rename(columns=dct_rename)
        # set data
        self.data = _df.sort_values(by=self.field_date).reset_index(drop=True)
        # set attributes
        self.n = len(self.data)
        self.units_h = units_h
        self.units_q = units_q
        self.hmax = self.data[self.field_hobs].max()
        self.hmin = self.data[self.field_hobs].min()
        self.date_start = self.data[self.field_date].min()
        self.date_end = self.data[self.field_date].max()
        return None

    def fit(self, n_grid=20):
        """
        Fit Rating Curve method. Q = a * (H - h0)^b
        :param n_grid: number of intervals for h0 iteration
        :type n_grid: int
        :return: None
        :rtype: None
        """
        from plans.analyst import Bivar

        # estimate h0
        _h0_max = self.data[self.field_hobs].min()
        # get range of h0
        _h0_values = np.linspace(0, 0.99 * _h0_max, n_grid)
        # set fit dataframe
        _df_fits = pd.DataFrame(
            {
                "h0": _h0_values,
                "b": np.zeros(n_grid),
                "a": np.zeros(n_grid),
                "RMSE": np.zeros(n_grid),
            }
        )
        _df_fits.insert(0, "Model", "")
        # search loop
        for i in range(len(_df_fits)):
            # get h0
            n_h0 = _df_fits["h0"].values[i]
            # get transformed variables
            self.update(h0=n_h0)

            # set Bivar base_object for tranformed linear model
            biv = Bivar(df_data=self.data, x_name=self.field_htt, y_name=self.field_qt)
            # fit linear model
            biv.fit(model_type="Linear")
            ###biv.view()
            # retrieve re-transformed values
            _df_fits["a"].values[i] = np.exp(
                biv.models["Linear"]["Setup"]["Mean"].values[0]
            )
            _df_fits["b"].values[i] = biv.models["Linear"]["Setup"]["Mean"].values[1]
            _df_fits["RMSE"].values[i] = biv.models["Linear"]["RMSE"]

        # sort by metric
        _df_fits = _df_fits.sort_values(by="RMSE").reset_index(drop=True)

        self.h0 = _df_fits["h0"].values[0]
        self.a = _df_fits["a"].values[0]
        self.b = _df_fits["b"].values[0]
        self.update()
        return None

    def get_bands(self, extrap_f=2, n_samples=100, runsize=100, seed=None, talk=False):
        """
        Get uncertainty bands from Rating Curve model using Monte Carlo sampling on the transformed error
        :param extrap_f: extrapolation factor over upper bound
        :type extrap_f: float
        :param n_samples: number of extrapolation samples
        :type n_samples: int
        :param runsize: number of monte carlo simulations
        :type runsize: int
        :param seed: reproducibility seed
        :type seed: int or None
        :param talk: option for printing messages
        :type talk: bool
        :return: dictionary with output dataframes
        :rtype: dict
        """
        from plans.analyst import Univar

        # random state setup
        if seed is None:
            from datetime import datetime

            np.random.seed(int(datetime.now().timestamp()))
        else:
            np.random.seed(seed)

        # ensure model is up-to-date
        self.update()

        # resample error

        # get the transform error datasets:
        grd_et = np.random.normal(
            loc=0, scale=self.et_sd, size=(runsize, len(self.data))
        )
        # re-calc qobs_t for all error realizations
        grd_qt = grd_et + np.array([self.data["{}_Mean".format(self.field_qt)].values])
        # re-calc qobs
        grd_qobs = np.exp(grd_qt)

        # setup of montecarlo dataframe
        mc_models_df = pd.DataFrame(
            {
                "Id": [
                    "MC{}".format(str(i + 1).zfill(int(np.log10(runsize)) + 1))
                    for i in range(runsize)
                ],
                self.name_h0: np.zeros(runsize),
                self.name_a: np.zeros(runsize),
                self.name_b: np.zeros(runsize),
            }
        )
        # set up simulation data
        grd_qsim = np.zeros(shape=(runsize, n_samples))

        # for each error realization, fit model and extrapolate
        if talk:
            print("Processing models...")
        for i in range(runsize):
            # set new qobs
            self.data[self.field_qobs] = grd_qobs[i]
            # update
            self.update()
            # fit
            self.fit(n_grid=10)
            # extrapolate
            hmax = self.hmax * extrap_f
            _df_ex = self.extrapolate(hmin=0, hmax=hmax, n_samples=n_samples)
            # store results
            grd_qsim[i] = _df_ex[self.field_q].values
            mc_models_df[self.name_h0].values[i] = self.h0
            mc_models_df[self.name_a].values[i] = self.a
            mc_models_df[self.name_b].values[i] = self.b

        # extract h values
        vct_h = _df_ex[self.field_h].values
        # transpose data
        grd_qsim_t = np.transpose(grd_qsim)

        # set simulation dataframe
        mc_sim_df = pd.DataFrame(
            data=grd_qsim_t,
            columns=[
                "Q_{}".format(mc_models_df["Id"].values[i]) for i in range(runsize)
            ],
        )
        mc_sim_df.insert(0, value=vct_h, column=self.field_h)
        mc_sim_df = mc_sim_df.dropna(how="any").reset_index(drop=True)

        # clear up memory
        del grd_qsim
        del grd_qsim_t

        # set up stats data
        df_sts_dumm = Univar(data=np.ones(10)).assess_basic_stats()
        grd_stats = np.zeros(shape=(len(mc_sim_df), len(df_sts_dumm)))

        # retrieve stats from simulation
        if talk:
            print("Processing bands...")
        for i in range(len(mc_sim_df)):
            vct_data = mc_sim_df.values[i][1:]
            uni = Univar(data=vct_data)
            _df_stats = uni.assess_basic_stats()
            grd_stats[i] = _df_stats["Value"].values

        # set up stats dataframe
        mc_stats_df = pd.DataFrame(
            columns=[
                "Q_{}".format(df_sts_dumm["Statistic"].values[i])
                for i in range(len(df_sts_dumm))
            ],
            data=grd_stats,
        )
        mc_stats_df.insert(0, column=self.field_h, value=mc_sim_df[self.field_h])
        del grd_stats

        # return objects
        return {
            "Models": mc_models_df,
            "Simulation": mc_sim_df,
            "Statistics": mc_stats_df,
        }

    def view(
        self, show=True, folder="C:/data", filename=None, dpi=150, fig_format="jpg"
    ):
        """View Rating Curve

        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """
        from plans.analyst import Bivar

        biv = Bivar(
            df_data=self.data,
            x_name=self.field_hobs,
            y_name=self.field_qobs,
            name="{} Rating Curve".format(self.name),
        )
        specs = {
            "xlabel": "{} ({})".format(self.field_hobs, self.units_h),
            "ylabel": "{} ({})".format(self.field_qobs, self.units_q),
            "xlim": (0, 1.1 * self.data[self.field_hobs].max()),
            "ylim": (0, 1.1 * self.data[self.field_qobs].max()),
        }
        biv.view(
            show=show,
            folder=folder,
            filename=filename,
            specs=specs,
            dpi=dpi,
            fig_format=fig_format,
        )
        del biv
        return None

    def view_model(
        self,
        transform=False,
        show=True,
        folder="C:/data",
        filename=None,
        dpi=150,
        fig_format="jpg",
    ):
        """
        View model Rating Curve
        :param transform: option for plotting transformed variables
        :type transform: bool
        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """
        from plans.analyst import Bivar

        self.update()

        s_xfield = self.field_hobs
        s_yfield = self.field_qobs
        model_type = "Power"
        if transform:
            s_xfield = self.field_htt
            s_yfield = self.field_qt
            model_type = "Linear"

        # create Bivar base_object
        biv = Bivar(
            df_data=self.data,
            x_name=s_xfield,
            y_name=s_yfield,
            name="{} Rating Curve".format(self.name),
        )

        specs = {
            "xlabel": "{} ({})".format(s_xfield, self.units_h),
            "ylabel": "{} ({})".format(s_yfield, self.units_q),
            "xlim": (0, 1.1 * self.data[s_xfield].max()),
            "ylim": (0, 1.1 * self.data[s_yfield].max()),
        }

        # set Power model parameters
        params_model = [-self.h0, self.b, self.a]
        if transform:
            # get transformed Linear params
            c0t = np.log(self.a)
            c1t = self.b
            params_model = [c0t, c1t]
        biv.update_model(params_mean=params_model, model_type=model_type)

        biv.view_model(
            model_type=model_type,
            show=show,
            folder=folder,
            filename=filename,
            specs=specs,
            dpi=dpi,
            fig_format=fig_format,
        )
        del biv
        return None


class RatingCurveCollection(Collection):
    def __init__(self, name="MyRatingCurveCollection"):
        obj_aux = RatingCurve()
        super().__init__(base_object=obj_aux, name=name)
        # set up date fields and special attributes
        self.catalog["Date_Start"] = pd.to_datetime(self.catalog["Date_Start"])
        self.catalog["Date_End"] = pd.to_datetime(self.catalog["Date_End"])

    def load(
        self,
        name,
        table_file,
        hobs_field,
        qobs_field,
        date_field="Date",
        units_q="m3/s",
        units_h="m",
    ):
        """
        Load rating curve to colletion from CSV file
        :param name: Rating Curve name
        :type name: str
        :param table_file: path to CSV file
        :type table_file: str
        :param hobs_field: name of observed Stage field
        :type hobs_field: str
        :param qobs_field: name of observed Discharge field
        :type qobs_field: str
        :param date_field: name of Date field
        :type date_field: str
        :param units_q: units of streamflow
        :type units_q: str
        :param units_h: units of stage
        :type units_h: str
        :return: None
        :rtype: None
        """
        rc_aux = RatingCurve(name=name)
        rc_aux.load(
            table_file=table_file,
            hobs_field=hobs_field,
            qobs_field=qobs_field,
            date_field=date_field,
            units_q=units_q,
            units_h=units_h
        )
        self.append(new_object=rc_aux)
        # delete aux
        del rc_aux
        return None

    def view(
        self,
        show=True,
        folder="./output",
        filename=None,
        specs=None,
        dpi=150,
        fig_format="jpg",
    ):
        plt.style.use("seaborn-v0_8")
        lst_colors = get_random_colors(size=len(self.catalog))

        # get specs
        default_specs = {
            "suptitle": "Rating Curves Collection | {}".format(self.name),
            "width": 5 * 1.618,
            "height": 5,
            "xmin": 0,
            "xmax": 1.5 * self.catalog["H_max"].max(),
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # Deploy figure
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        fig.suptitle(specs["suptitle"])

        self.update(details=True)
        for i in range(len(self.catalog)):
            s_name = self.catalog["Name"].values[i]
            _df = self.collection[s_name].data
            _hfield = self.collection[s_name].field_hobs
            _qfield = self.collection[s_name].field_qobs
            plt.scatter(_df[_hfield], _df[_qfield], marker=".", color=lst_colors[i])

        plt.xlim(specs["xmin"], specs["xmax"])

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "{}_{}".format(self.varalias, self.name)
            plt.savefig("{}/{}.{}".format(folder, filename, fig_format), dpi=dpi)
        plt.close(fig)
        return None


class Streamflow:
    """
    The Streamflow (Discharge) base_object
    """

    def __init__(self, name, code):
        # -------------------------------------
        # set basic attributes
        self.name = name
        self.code = code
        self.source_data = None
        self.latitude = None
        self.longitude = None
        self.stage_series = None
        self.rating_curves = None

    # todo implement StreamFlow


# -----------------------------------------
# Base raster data structures


class Raster:
    """
    The basic raster map dataset.
    """

    def __init__(self, name="myRasterMap", dtype="float32"):
        """Deploy basic raster map base_object.

        :param name: map name
        :type name: str
        :param dtype: data type of raster cells - options: byte, uint8, int16, int32, float32, etc, defaults to float32
        :type dtype: str
        """
        # -------------------------------------
        # set basic attributes
        self.grid = None  # main grid
        self.backup_grid = None
        self.isaoi = False
        self.asc_metadata = {
            "ncols": None,
            "nrows": None,
            "xllcorner": None,
            "yllcorner": None,
            "cellsize": None,
            "NODATA_value": None,
        }
        self.nodatavalue = self.asc_metadata["NODATA_value"]
        self.cellsize = self.asc_metadata["cellsize"]
        self.name = name
        self.dtype = dtype
        self.cmap = "jet"
        self.varname = "Unknown variable"
        self.varalias = "Var"
        self.description = None
        self.units = "units"
        self.date = None  # "2020-01-01"
        self.source_data = None
        self.prj = None
        self.path_ascfile = None
        self.path_prjfile = None

    def __str__(self):
        dct_meta = self.get_metadata()
        lst_ = list()
        lst_.append("\n")
        lst_.append("Object: {}".format(type(self)))
        lst_.append("Metadata:")
        for k in dct_meta:
            lst_.append("\t{}: {}".format(k, dct_meta[k]))
        return "\n".join(lst_)

    def set_grid(self, grid):
        """Set data from incoming objects.

        :param grid: data grid
        :type grid: :class:`numpy.ndarray`
        """
        # overwrite incoming dtype
        self.grid = grid.astype(self.dtype)
        # mask nodata values
        self.mask_nodata()
        return None

    def set_asc_metadata(self, metadata):
        """Set metadata from incoming objects

        Example of metadata for ``.asc`` file raster:

        .. code-block:: python

            meta = {
                'ncols': 366,
                'nrows': 434,
                'xllcorner': 559493.08,
                'yllcorner': 6704832.2,
                'cellsize': 30,
                'NODATA_value': -1
            }.

        :param metadata: metadata dictionary
        :type metadata: dict
        """
        for k in self.asc_metadata:
            if k in metadata:
                self.asc_metadata[k] = metadata[k]
        # update nodata value and cellsize
        self.nodatavalue = self.asc_metadata["NODATA_value"]
        self.cellsize = self.asc_metadata["cellsize"]
        return None

    def load(self, asc_file, prj_file):
        """
        Load data from files to raster
        :param asc_file: path_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path_main to ``.prj`` projection file
        :type prj_file: str
        :return: None
        :rtype: None
        """
        self.load_asc_raster(file=asc_file)
        self.load_prj_file(file=prj_file)
        return None

    def load_asc_raster(self, file, nan=False):
        """A function to load data and metadata from ``.asc`` raster files.

        :param file: string of file path_main with the ``.asc`` extension
        :type file: str
        :param nan: boolean to convert nan values to np.nan, defaults to False
        :type nan: bool
        """
        # get file
        self.path_ascfile = file
        f_file = open(file)
        lst_file = f_file.readlines()
        f_file.close()
        #
        # get metadata constructor loop
        tpl_meta_labels = (
            "ncols",
            "nrows",
            "xllcorner",
            "yllcorner",
            "cellsize",
            "NODATA_value",
        )
        tpl_meta_format = ("int", "int", "float", "float", "float", "float")
        dct_meta = dict()
        for i in range(6):
            lcl_lst = lst_file[i].split(" ")
            lcl_meta_str = lcl_lst[len(lcl_lst) - 1].split("\n")[0]
            if tpl_meta_format[i] == "int":
                dct_meta[tpl_meta_labels[i]] = int(lcl_meta_str)
            else:
                dct_meta[tpl_meta_labels[i]] = float(lcl_meta_str)
        #
        # array constructor loop:
        lst_grid = list()
        for i in range(6, len(lst_file)):
            lcl_lst = lst_file[i].split(" ")[1:]
            lcl_lst[len(lcl_lst) - 1] = lcl_lst[len(lcl_lst) - 1].split("\n")[0]
            lst_grid.append(lcl_lst)
        # create grid file
        grd_data = np.array(lst_grid, dtype=self.dtype)
        #
        # replace NoData value by np.nan
        if nan:
            ndv = float(dct_meta["NODATA_value"])
            for i in range(len(grd_data)):
                lcl_row_sum = np.sum((grd_data[i] == ndv) * 1)
                if lcl_row_sum > 0:
                    for j in range(len(grd_data[i])):
                        if grd_data[i][j] == ndv:
                            grd_data[i][j] = np.nan

        self.set_asc_metadata(metadata=dct_meta)
        self.set_grid(grid=grd_data)
        return None

    def load_asc_metadata(self, file):
        """A function to load only metadata from ``.asc`` raster files.

        :param file: string of file path_main with the ``.asc`` extension
        :type file: str
        """

        with open(file) as f:
            def_lst = []
            for i, line in enumerate(f):
                if i >= 6:
                    break
                def_lst.append(line.strip())  # append each line to the list
        #
        # get metadata constructor loop
        meta_lbls = (
            "ncols",
            "nrows",
            "xllcorner",
            "yllcorner",
            "cellsize",
            "NODATA_value",
        )
        meta_format = ("int", "int", "float", "float", "float", "float")
        meta_dct = dict()
        for i in range(6):
            lcl_lst = def_lst[i].split(" ")
            lcl_meta_str = lcl_lst[len(lcl_lst) - 1].split("\n")[0]
            if meta_format[i] == "int":
                meta_dct[meta_lbls[i]] = int(lcl_meta_str)
            else:
                meta_dct[meta_lbls[i]] = float(lcl_meta_str)
        # set attribute
        self.set_asc_metadata(metadata=meta_dct)
        return None

    def load_prj_file(self, file):
        """Function for loading ``.prj`` aux file to the prj attribute

        :param file: string of file path_main with the ``.prj`` extension
        :type file: str
        :return: None
        :rtype: None
        """
        self.path_prjfile = file
        with open(file) as f:
            self.prj = f.readline().strip("\n")
        return None

    def export(self, folder, filename=None):
        """
        Export raster data to folder
        param folder: string of directory path_main,
        :type folder: str
        :param filename: string of file without extension, defaults to None
        :type filename: str
        :return: None
        :rtype: None
        """
        if filename is None:
            filename = self.name
        self.export_asc_raster(folder=folder, filename=filename)
        self.export_prj_file(folder=folder, filename=filename)
        return None

    def export_asc_raster(self, folder, filename=None):
        """Function for exporting an ``.asc`` raster file.

        :param folder: string of directory path_main,
        :type folder: str
        :param filename: string of file without extension, defaults to None
        :type filename: str
        :return: full file name (path_main and extension) string
        :rtype: str
        """
        if self.grid is None or self.asc_metadata is None:
            pass
        else:
            meta_lbls = (
                "ncols",
                "nrows",
                "xllcorner",
                "yllcorner",
                "cellsize",
                "NODATA_value",
            )
            ndv = float(self.asc_metadata["NODATA_value"])
            exp_lst = list()
            for i in range(len(meta_lbls)):
                line = "{}    {}\n".format(
                    meta_lbls[i], self.asc_metadata[meta_lbls[i]]
                )
                exp_lst.append(line)

            # ----------------------------------
            # data constructor loop:
            self.insert_nodata()  # insert nodatavalue

            def_array = np.array(self.grid, dtype=self.dtype)
            for i in range(len(def_array)):
                # replace np.nan to no data values
                lcl_row_sum = np.sum((np.isnan(def_array[i])) * 1)
                if lcl_row_sum > 0:
                    # print('Yeas')
                    for j in range(len(def_array[i])):
                        if np.isnan(def_array[i][j]):
                            def_array[i][j] = int(ndv)
                str_join = " " + " ".join(np.array(def_array[i], dtype="str")) + "\n"
                exp_lst.append(str_join)

            if filename is None:
                filename = self.name
            flenm = folder + "/" + filename + ".asc"
            fle = open(flenm, "w+")
            fle.writelines(exp_lst)
            fle.close()

            # mask again
            self.mask_nodata()

            return flenm

    def export_prj_file(self, folder, filename=None):
        """Function for exporting an ``.prj`` file.

        :param folder: string of directory path_main,
        :type folder: str
        :param filename: string of file without extension, defaults to None
        :type filename: str
        :return: full file name (path_main and extension) string
        :rtype: str or None
        """
        if self.prj is None:
            return None
        else:
            if filename is None:
                filename = self.name

            flenm = folder + "/" + filename + ".prj"
            fle = open(flenm, "w+")
            fle.writelines([self.prj])
            fle.close()
            return flenm

    def mask_nodata(self):
        """Mask grid cells as NaN where data is NODATA."""
        if self.nodatavalue is None:
            pass
        else:
            if self.grid.dtype.kind in ["i", "u"]:
                # for integer grid
                self.grid = np.ma.masked_where(self.grid == self.nodatavalue, self.grid)
            else:
                # for floating point grid:
                self.grid[self.grid == self.nodatavalue] = np.nan
        return None

    def insert_nodata(self):
        """Insert grid cells as NODATA where data is NaN."""
        if self.nodatavalue is None:
            pass
        else:
            if self.grid.dtype.kind in ["i", "u"]:
                # for integer grid
                self.grid = np.ma.filled(self.grid, fill_value=self.nodatavalue)
            else:
                # for floating point grid:
                self.grid = np.nan_to_num(self.grid, nan=self.nodatavalue)
        return None

    def rebase_grid(self, base_raster, inplace=False, method="linear_model"):
        """
        Rebase grid of raster. This function creates a new grid based on a provided raster. Both rasters
        are expected to be in the same coordinate system and having overlapping bounding boxes.
        :param base_raster: reference raster for rebase
        :type base_raster: :class:`datasets.Raster`
        :param inplace: option for rebase the own grid if True, defaults to False
        :type inplace: bool
        :param method: interpolation method - linear_model, nearest and cubic
        :type method: str
        :return: rebased grid
        :rtype: :class:`numpy.ndarray` or None
        """
        from scipy.interpolate import griddata

        # get data points
        _df = self.get_grid_datapoints(drop_nan=True)
        # get base grid data points
        _dfi = base_raster.get_grid_datapoints(drop_nan=False)
        # set data points
        grd_points = np.array([_df["x"].values, _df["y"].values]).transpose()
        grd_new_points = np.array([_dfi["x"].values, _dfi["y"].values]).transpose()
        _dfi["zi"] = griddata(
            points=grd_points, values=_df["z"].values, xi=grd_new_points, method=method
        )
        grd_zi = np.reshape(_dfi["zi"].values, newshape=base_raster.grid.shape)
        if inplace:
            # set
            self.set_grid(grid=grd_zi)
            self.set_asc_metadata(metadata=base_raster.asc_metadata)
            self.prj = base_raster.prj
            return None
        else:
            return grd_zi

    def apply_aoi_mask(self, grid_aoi, inplace=False):
        """Apply AOI (area of interest) mask to raster map.

        :param grid_aoi: map of AOI (masked array or pseudo-boolean)
        Expected to have the same grid shape.
        :type grid_aoi: :class:`numpy.ndarray`
        :param inplace: overwrite the main grid if True, defaults to False
        :type inplace: bool
        :return: None
        :rtype: None
        """
        if self.nodatavalue is None or self.grid is None:
            pass
        else:
            # ensure fill on masked values
            grid_aoi = np.ma.filled(grid_aoi, fill_value=0)
            # replace
            grd_mask = np.where(grid_aoi == 0, self.nodatavalue, self.grid)

            if inplace:
                pass
            else:
                # pass a copy to backup grid
                self.backup_grid = self.grid.copy()
            # set main grid
            self.set_grid(grid=grd_mask)
            self.isaoi = True
        return None

    def release_aoi_mask(self):
        """
        Release AOI mask from main grid. Backup grid is restored.
        :return: None
        :rtype: None
        """
        if self.isaoi:
            self.set_grid(grid=self.backup_grid)
            self.backup_grid = None
            self.isaoi = False
        return None

    def cut_edges(self, upper, lower, inplace=False):
        """Cutoff upper and lower values of grid.

        :param upper: upper value
        :type upper: float or int
        :param lower: lower value
        :type lower: float or int
        :param inplace: option to set the raster grid, defaults to False
        :type inplace: bool
        :return: the processed grid if inplace=False
        :rtype: :class:`numpy.ndarray` or None
        """
        if self.grid is None:
            return None
        else:
            new_grid = self.grid
            new_grid[new_grid < lower] = lower
            new_grid[new_grid > upper] = upper

            if inplace:
                self.set_grid(grid=new_grid)
                return None
            else:
                return new_grid

    def get_metadata(self):
        """
        Get all metadata from base_object
        :return: metadata
        :rtype: dict
        """
        return {
            "Name": self.name,
            "Variable": self.varname,
            "VarAlias": self.varalias,
            "Units": self.units,
            "Date": self.date,
            "Source": self.source_data,
            "Date": self.date,
            "Description": self.description,
            "cellsize": self.cellsize,
            "ncols": self.asc_metadata["ncols"],
            "rows": self.asc_metadata["nrows"],
            "xllcorner": self.asc_metadata["xllcorner"],
            "yllcorner": self.asc_metadata["yllcorner"],
            "NODATA_value": self.nodatavalue,
            "Prj": self.prj,
        }

    def get_bbox(self):
        """Get the Bounding Box of map
        :return: dictionary of xmin, xmax ymin and ymax
        :rtype: dict
        """
        return {
            "xmin": self.asc_metadata["xllcorner"],
            "xmax": self.asc_metadata["xllcorner"]
            + (self.asc_metadata["ncols"] * self.cellsize),
            "ymin": self.asc_metadata["yllcorner"],
            "ymax": self.asc_metadata["yllcorner"]
            + (self.asc_metadata["nrows"] * self.cellsize),
        }

    def get_grid_datapoints(self, drop_nan=False):
        """Get flat and cleared grid data points (x, y and z)

        :return: dataframe of x, y and z fields
        :rtype: :class:`pandas.DataFrame` or None
        """
        if self.grid is None:
            return None
        else:
            # get coordinates
            vct_i = np.zeros(self.grid.shape[0] * self.grid.shape[1])
            vct_j = vct_i.copy()
            vct_z = vct_i.copy()
            _c = 0
            for i in range(len(self.grid)):
                for j in range(len(self.grid[i])):
                    vct_i[_c] = i
                    vct_j[_c] = j
                    vct_z[_c] = self.grid[i][j]
                    _c = _c + 1

            # transform
            n_height = self.grid.shape[0] * self.cellsize
            vct_y = (
                self.asc_metadata["yllcorner"]
                + (n_height - (vct_i * self.cellsize))
                - (self.cellsize / 2)
            )
            vct_x = (
                self.asc_metadata["xllcorner"]
                + (vct_j * self.cellsize)
                + (self.cellsize / 2)
            )

            # drop nan or masked values:
            if drop_nan:
                vct_j = vct_j[~np.isnan(vct_z)]
                vct_i = vct_i[~np.isnan(vct_z)]
                vct_x = vct_x[~np.isnan(vct_z)]
                vct_y = vct_y[~np.isnan(vct_z)]
                vct_z = vct_z[~np.isnan(vct_z)]
            # built dataframe
            _df = pd.DataFrame(
                {
                    "x": vct_x,
                    "y": vct_y,
                    "z": vct_z,
                    "i": vct_i,
                    "j": vct_j,
                }
            )
            return _df

    def get_grid_data(self):
        """Get flat and cleared grid data.

        :return: 1d vector of cleared data
        :rtype: :class:`numpy.ndarray` or None
        """
        if self.grid is None:
            return None
        else:
            if self.grid.dtype.kind in ["i", "u"]:
                # for integer grid
                _grid = self.grid[~self.grid.mask]
                return _grid
            else:
                # for floating point grid:
                _grid = self.grid.ravel()[~np.isnan(self.grid.ravel())]
                return _grid

    def get_grid_stats(self):
        """Get basic statistics from flat and clear data.

        :return: dataframe of basic statistics
        :rtype: :class:`pandas.DataFrame` or None
        """
        if self.grid is None:
            return None
        else:
            from plans.analyst import Univar

            return Univar(data=self.get_grid_data()).assess_basic_stats()

    def view(
        self,
        show=True,
        folder="./output",
        filename=None,
        specs=None,
        dpi=150,
        fig_format="jpg",
    ):
        """Plot a basic pannel of raster map.

        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        """
        import matplotlib.ticker as mtick
        from plans.analyst import Univar

        plt.style.use("seaborn-v0_8")

        # get univar base_object
        uni = Univar(data=self.get_grid_data())
        if len(np.unique(uni.data)) <= 1:
            nbins = 1
        else:
            nbins = uni.nbins_fd()

        # get specs
        default_specs = {
            "color": "tab:grey",
            "cmap": self.cmap,
            "suptitle": "{} | {}".format(self.varname, self.name),
            "a_title": "{} ({})".format(self.varalias, self.units),
            "b_title": "Histogram",
            "c_title": "Metadata",
            "d_title": "Statistics",
            "width": 5 * 1.618,
            "height": 5,
            "b_ylabel": "percentage",
            "b_xlabel": self.units,
            "nbins": nbins,
            "vmin": None,
            "vmax": None,
            "hist_vmax": None,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        if specs["vmin"] is None:
            specs["vmin"] = np.min(uni.data)
        if specs["vmax"] is None:
            specs["vmax"] = np.max(uni.data)

        # Deploy figure
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        gs = mpl.gridspec.GridSpec(
            4, 5, wspace=0.8, hspace=0.1, left=0.05, bottom=0.1, top=0.85, right=0.95
        )
        fig.suptitle(specs["suptitle"])

        # plot map
        plt.subplot(gs[:3, :3])
        plt.title("a. {}".format(specs["a_title"]), loc="left")
        im = plt.imshow(
            self.grid, cmap=specs["cmap"], vmin=specs["vmin"], vmax=specs["vmax"]
        )
        fig.colorbar(im, shrink=0.5)
        plt.axis("off")

        # plot Hist
        plt.subplot(gs[:2, 3:])
        plt.title("b. {}".format(specs["b_title"]), loc="left")
        vct_result = plt.hist(
            x=uni.data,
            bins=specs["nbins"],
            color=specs["color"],
            weights=np.ones(len(uni.data)) / len(uni.data)
            # orientation="horizontal"
        )
        # get upper limit if none
        if specs["hist_vmax"] is None:
            specs["hist_vmax"] = 1.2 * np.max(vct_result[0])
        # plot mean line
        n_mean = np.mean(uni.data)
        plt.vlines(
            x=n_mean,
            ymin=0,
            ymax=specs["hist_vmax"],
            colors="tab:orange",
            linestyles="--",
            # label="mean ({:.2f})".fig_format(n_mean)
        )
        plt.text(
            x=n_mean + 2 * (specs["vmax"] - specs["vmin"]) / 100,
            y=0.9 * specs["hist_vmax"],
            s="{:.2f} (mean)".format(n_mean),
        )

        plt.ylim(0, specs["hist_vmax"])
        plt.xlim(specs["vmin"], specs["vmax"])
        # plt.ylabel(specs["b_ylabel"])
        plt.xlabel(specs["b_xlabel"])

        # Set the y-axis formatter as percentages
        yticks = mtick.PercentFormatter(xmax=1, decimals=1, symbol="%", is_latex=False)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(yticks)

        # ------------------------------------------------------------------
        # plot metadata

        # get datasets
        df_stats = self.get_grid_stats()
        lst_meta = []
        lst_value = []
        for k in self.asc_metadata:
            lst_value.append(self.asc_metadata[k])
            lst_meta.append(k)
        df_meta = pd.DataFrame({"Raster": lst_meta, "Value": lst_value})
        # metadata
        n_y = 0.25
        n_x = 0.08
        plt.text(
            x=n_x,
            y=n_y,
            s="c. {}".format(specs["c_title"]),
            fontsize=12,
            transform=fig.transFigure,
        )
        n_y = n_y - 0.01
        n_step = 0.025
        for i in range(len(df_meta)):
            s_head = df_meta["Raster"].values[i]
            if s_head == "cellsize":
                s_value = self.cellsize
                s_line = "{:>15}: {:<10.5f}".format(s_head, s_value)
            else:
                s_value = df_meta["Value"].values[i]
                s_line = "{:>15}: {:<10.2f}".format(s_head, s_value)
            n_y = n_y - n_step
            plt.text(
                x=n_x,
                y=n_y,
                s=s_line,
                fontsize=9,
                fontdict={"family": "monospace"},
                transform=fig.transFigure,
            )

        # stats
        n_y_base = 0.25
        n_x = 0.62
        plt.text(
            x=n_x,
            y=n_y_base,
            s="d. {}".format(specs["d_title"]),
            fontsize=12,
            transform=fig.transFigure,
        )
        n_y = n_y_base - 0.01
        n_step = 0.025
        for i in range(7):
            s_head = df_stats["Statistic"].values[i]
            s_value = df_stats["Value"].values[i]
            s_line = "{:>10}: {:<10.2f}".format(s_head, s_value)
            n_y = n_y - n_step
            plt.text(
                x=n_x,
                y=n_y,
                s=s_line,
                fontsize=9,
                fontdict={"family": "monospace"},
                transform=fig.transFigure,
            )
        n_y = n_y_base - 0.01
        for i in range(7, len(df_stats)):
            s_head = df_stats["Statistic"].values[i]
            s_value = df_stats["Value"].values[i]
            s_line = "{:>10}: {:<10.2f}".format(s_head, s_value)
            n_y = n_y - n_step
            plt.text(
                x=n_x + 0.15    ,
                y=n_y,
                s=s_line,
                fontsize=9,
                fontdict={"family": "monospace"},
                transform=fig.transFigure,
            )
        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "{}_{}".format(self.varalias, self.name)
            plt.savefig("{}/{}.{}".format(folder, filename, fig_format), dpi=dpi)
        plt.close(fig)
        return None


# -----------------------------------------
# Derived Raster data structures


class Elevation(Raster):
    """
    Elevation (DEM) raster map dataset.
    """

    def __init__(self, name):
        """Deploy dataset.

        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="float32")
        self.cmap = "BrBG_r"
        self.varname = "Elevation"
        self.varalias = "ELV"
        self.description = "Height above sea level"
        self.units = "m"

    def get_tpi(self, cell_radius):
        print("ah shit")

    def get_tpi_landforms(self, radius_micro, radius_macro):
        print("ah shit")


class Slope(Raster):
    """
    Slope raster map dataset.
    """

    def __init__(self, name):
        """Deploy dataset.

        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="float32")
        self.cmap = "OrRd"
        self.varname = "Slope"
        self.varalias = "SLP"
        self.description = "Slope of terrain"
        self.units = "deg."


class TWI(Raster):
    """
    TWI raster map dataset.
    """

    def __init__(self, name):
        """Deploy dataset.

        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="float32")
        self.cmap = "YlGnBu"
        self.varname = "TWI"
        self.varalias = "TWI"
        self.description = "Topographical Wetness Index"
        self.units = "index units"

class HAND(Raster):
    """
    HAND raster map dataset.
    """

    def __init__(self, name):
        """Deploy dataset.

        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="float32")
        self.cmap = "YlGnBu_r"
        self.varname = "HAND"
        self.varalias = "HAND"
        self.description = "Height Above the Nearest Drainage"
        self.units = "m"


class NDVI(Raster):
    """
    NDVI raster map dataset.
    """

    def __init__(self, name, date):
        """Deploy dataset.

        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="float32")
        self.cmap = "RdYlGn"
        self.varname = "NDVI"
        self.varalias = "NDVI"
        self.description = "Normalized difference vegetation index"
        self.units = "index units"
        self.date = date

    def set_grid(self, grid):
        super(NDVI, self).set_grid(grid)
        self.cut_edges(upper=1, lower=-1)
        return None

    def view(
        self,
        show=True,
        folder="./output",
        filename=None,
        specs=None,
        dpi=150,
        fig_format="jpg",
    ):
        """
        View NDVI raster
        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """
        # set specs
        default_specs = {"vmin": -1, "vmax": 1}
        if specs is None:
            specs = default_specs
        else:
            for k in default_specs:
                specs[k] = default_specs[k]
        # call super
        super().view(show, folder, filename, specs, dpi, fig_format=fig_format)
        return None


class ET24h(Raster):
    """
    ET 24h raster map dataset.
    """

    def __init__(self, name, date):
        """Deploy dataset.

        :param name: name of map
        :type name: str
        """
        import matplotlib as mpl
        from matplotlib.colors import ListedColormap

        super().__init__(name=name, dtype="float32")
        self.varname = "Daily Evapotranspiration"
        self.varalias = "ET24h"
        self.description = "Daily Evapotranspiration"
        self.units = "mm"
        # set custom cmap
        jet_big = mpl.colormaps["jet_r"]
        self.cmap = ListedColormap(jet_big(np.linspace(0.3, 0.75, 256)))
        self.date = date

    def set_grid(self, grid):
        super().set_grid(grid)
        self.cut_edges(upper=100, lower=0)
        return None

    def view(
        self,
        show=True,
        folder="./output",
        filename=None,
        specs=None,
        dpi=150,
        fig_format="jpg",
    ):
        """
        View ET raster
        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """
        default_specs = {"vmin": 0, "vmax": 10}
        if specs is None:
            specs = default_specs
        else:
            for k in default_specs:
                specs[k] = default_specs[k]
        super().view(show, folder, filename, specs, dpi, fig_format=fig_format)
        return None



class HabQuality(Raster):
    """
    Habitat Quality raster map dataset.
    """

    def __init__(self, name, date):
        """Deploy dataset.

        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="float32")
        self.varname = "Habitat Quality"
        self.varalias = "HQ"
        self.description = "Habitat Quality from the InVEST model"
        self.units = "index units"
        self.cmap = "RdYlGn"
        self.date = date

    def view(
        self,
        show=True,
        folder="./output",
        filename=None,
        specs=None,
        dpi=150,
        fig_format="jpg",
    ):
        """
        View HQ raster
        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """
        # set specs
        default_specs = {"vmin": 0, "vmax": 1}
        if specs is None:
            specs = default_specs
        else:
            for k in default_specs:
                specs[k] = default_specs[k]
        # call super
        super().view(show, folder, filename, specs, dpi, fig_format=fig_format)
        return None


class HabDegradation(Raster):
    """
    Habitat Degradation raster map dataset.
    """

    def __init__(self, name, date):
        """Deploy dataset.

        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="float32")
        self.varname = "Habitat Degradation"
        self.varalias = "HDeg"
        self.description = "Habitat Degradation from the InVEST model"
        self.units = "index units"
        self.cmap = "YlOrRd"
        self.date = date

    def view(
            self,
            show=True,
            folder="./output",
            filename=None,
            specs=None,
            dpi=150,
            fig_format="jpg",
    ):
        """
        View HDeg raster
        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """
        # set specs
        default_specs = {"vmin": 0, "vmax": 0.3}
        if specs is None:
            specs = default_specs
        else:
            for k in default_specs:
                specs[k] = default_specs[k]
        # call super
        super().view(show, folder, filename, specs, dpi, fig_format=fig_format)
        return None


# -----------------------------------------
# Quali Raster data structures


class QualiRaster(Raster):
    """
    Basic qualitative raster map dataset.

    Attributes dataframe must at least have:
    * :class:`Id` field
    * :class:`Name` field
    * :class:`Alias` field

    """

    def __init__(self, name="QualiMap", dtype="uint8"):
        """
        Deploy dataset
        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype=dtype)
        self.cmap = "tab20"
        self.varname = "Unknown variable"
        self.varalias = "Var"
        self.description = "Unknown"
        self.units = "category ID"
        self.table = None
        self.idfield = "Id"
        self.namefield = "Name"
        self.aliasfield = "Alias"
        self.colorfield = "Color"
        self.areafield = "Area"
        self.path_tablefile = None
        self._overwrite_nodata()

    def _overwrite_nodata(self):
        self.nodatavalue = 0
        self.asc_metadata["NODATA_value"] = self.nodatavalue
        return None

    def set_asc_metadata(self, metadata):
        super().set_asc_metadata(metadata)
        self._overwrite_nodata()
        return None

    def rebase_grid(self, base_raster, inplace=False):
        out = super().rebase_grid(base_raster, inplace, method="nearest")
        return out

    def reclassify(self, dct_reclass, method="fast"):
        # todo here
        print("ah shit")

    def load(self, asc_file, prj_file, table_file):
        """
        Load data from files to raster
        :param asc_file: path_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path_main to ``.prj`` projection file
        :type prj_file: str
        :param table_file: path_main to ``.txt`` table file
        :type table_file: str
        :return: None
        :rtype: None
        """
        super().load(asc_file=asc_file, prj_file=prj_file)
        self.load_table(file=table_file)
        return None

    def load_table(self, file):
        """Load attributes dataframe from ``csv`` ``.txt`` file (separator must be ;).

        :param file: path_main to file
        :type file: str
        """
        self.path_tablefile = file
        # read raw file
        df_aux = pd.read_csv(file, sep=";")
        # set to self
        self.set_table(dataframe=df_aux)
        return None

    def export(self, folder, filename=None):
        """
        Export raster data
        param folder: string of directory path_main,
        :type folder: str
        :param filename: string of file without extension, defaults to None
        :type filename: str
        :return: None
        :rtype: None
        """
        super().export(folder=folder, filename=filename)
        self.export_table(folder=folder, filename=filename)
        return None

    def export_table(self, folder, filename=None):
        """Export a CSV ``.txt``  file.

        :param folder: string of directory path_main
        :type folder: str
        :param filename: string of file without extension
        :type filename: str
        :return: full file name (path_main and extension) string
        :rtype: str
        """
        if filename is None:
            filename = self.name
        flenm = folder + "/" + filename + ".txt"
        self.table.to_csv(flenm, sep=";", index=False)
        return flenm

    def set_table(self, dataframe):
        """Set attributes dataframe from incoming :class:`pandas.DataFrame`.

        :param dataframe: incoming pandas dataframe
        :type dataframe: :class:`pandas.DataFrame`
        """
        self.table = dataframe_prepro(dataframe=dataframe.copy())
        self.table = self.table.sort_values(by=self.idfield).reset_index(drop=True)

    def set_random_colors(self):
        """Set random colors to attribute table."""
        if self.table is None:
            pass
        else:
            self.table[self.colorfield] = get_random_colors(
                size=len(self.table), cmap=self.cmap
            )
        return None

    def get_areas(self, merge=False):
        """Get areas in map of each category in table.

        :param merge: option to merge data with raster table
        :type merge: bool, defaults to False
        :return: areas dataframe
        :rtype: :class:`pandas.DataFrame`
        """
        if self.table is None or self.grid is None or self.prj is None:
            return None
        else:
            # get unit area in meters
            _cell_size = self.cellsize
            if self.prj[:6] == "GEOGCS":
                _cell_size = self.cellsize * 111111  # convert degrees to meters
            _n_unit_area = np.square(_cell_size)
            # get aux dataframe
            df_aux = self.table[["Id", "Name", "Alias"]].copy()
            _lst_count = []
            # iterate categories
            for i in range(len(df_aux)):
                _n_id = df_aux[self.idfield].values[i]
                _n_count = np.sum(1 * (self.grid == _n_id))
                _lst_count.append(_n_count)
            # set area fields
            lst_area_fields = []
            # Count
            s_count_field = "Cell_count"
            df_aux[s_count_field] = _lst_count
            lst_area_fields.append(s_count_field)

            # m2
            s_field = "{}_m2".format(self.areafield)
            lst_area_fields.append(s_field)
            df_aux[s_field] = df_aux[s_count_field].values * _n_unit_area

            # ha
            s_field = "{}_ha".format(self.areafield)
            lst_area_fields.append(s_field)
            df_aux[s_field] = df_aux[s_count_field].values * _n_unit_area / (100 * 100)

            # km2
            s_field = "{}_km2".format(self.areafield)
            lst_area_fields.append(s_field)
            df_aux[s_field] = (
                df_aux[s_count_field].values * _n_unit_area / (1000 * 1000)
            )

            # fraction
            s_field = "{}_f".format(self.areafield)
            lst_area_fields.append(s_field)
            df_aux[s_field] = df_aux[s_count_field] / df_aux[s_count_field].sum()
            # %
            s_field = "{}_%".format(self.areafield)
            lst_area_fields.append(s_field)
            df_aux[s_field] = 100 * df_aux[s_count_field] / df_aux[s_count_field].sum()
            df_aux[s_field] = df_aux[s_field].round(2)

            # handle merge
            if merge:
                for k in lst_area_fields:
                    self.table[k] = df_aux[k].values

            return df_aux

    def get_zonal_stats(self, raster_sample, merge=False, skip_count=False):
        """Get zonal stats from other raster map to sample.

        :param raster_sample: raster map to sample
        :type raster_sample: :class:`datasets.Raster`
        :param merge: option to merge data with raster table, defaults to False
        :type merge: bool
        :param skip_count: set True to skip count, defaults to False
        :type skip_count: bool
        :return: dataframe of zonal stats
        :rtype: :class:`pandas.DataFrame`
        """
        from plans.analyst import Univar

        # deploy dataframe
        df_aux = self.table[["Id", "Name", "Alias"]].copy()
        # store copy of raster
        grid_raster = raster_sample.grid
        varname = raster_sample.varname
        # collect statistics
        lst_stats = []
        for i in range(len(df_aux)):
            n_id = df_aux["Id"].values[i]
            # apply mask
            grid_aoi = 1 * (self.grid == n_id)
            raster_sample.apply_aoi_mask(grid_aoi=grid_aoi, inplace=True)
            # get basic stats
            raster_uni = Univar(data=raster_sample.get_grid_data(), name=varname)
            df_stats = raster_uni.assess_basic_stats()
            lst_stats.append(df_stats.copy())
            # restore
            raster_sample.grid = grid_raster

        # create empty fields
        lst_stats_field = []
        for k in df_stats["Statistic"]:
            s_field = "{}_{}".format(varname, k)
            lst_stats_field.append(s_field)
            df_aux[s_field] = 0.0

        # fill values
        for i in range(len(df_aux)):
            df_aux.loc[i, lst_stats_field[0] : lst_stats_field[-1]] = lst_stats[i][
                "Value"
            ].values

        # handle count
        if skip_count:
            df_aux = df_aux.drop(columns=["{}_count".format(varname)])
            lst_stats_field.remove("{}_count".format(varname))

        # handle merge
        if merge:
            for k in lst_stats_field:
                self.table[k] = df_aux[k].values

        return df_aux

    def view(
        self,
        show=True,
        folder="./output",
        filename=None,
        specs=None,
        dpi=150,
        fig_format="jpg",
        filter=False,
    ):
        """Plot a basic pannel of qualitative raster map.

        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :param filter: option for cutting off zero-area classes
        :type filter: bool
        :return: None
        :rtype: None
        """
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        plt.style.use("seaborn-v0_8")

        if self.colorfield in self.table.columns:
            pass
        else:
            self.set_random_colors()

        # hack for non-continuous ids
        _all_ids = np.arange(0, self.table[self.idfield].max() + 1)
        _lst_colors = []
        for i in range(0, len(_all_ids)):
            _df = self.table.query("{} >= {}".format(self.idfield, i)).copy()
            _color = _df[self.colorfield].values[0]
            _lst_colors.append(_color)

        # get specs
        default_specs = {
            "color": "tab:grey",
            "cmap": ListedColormap(_lst_colors),
            "suptitle": "{} ({}) | {}".format(self.varname, self.varalias, self.name),
            "a_title": "{} Map ({})".format(self.varalias, self.units),
            "b_title": "{} Prevalence".format(self.varalias),
            "c_title": "Metadata",
            "width": 8,
            "height": 5,
            "b_area": "km2",
            "b_xlabel": "Area",
            "b_xmax": None,
            "bars_alias": True,
            "vmin": 0,
            "vmax": self.table[self.idfield].max(),
            "gs_rows": 7,
            "gs_cols": 5,
            "gs_b_rowlim": 4,
            "legend_x": 0.4,
            "legend_y": 0.3,
            "legend_ncol": 1,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # -----------------------------------------------
        # ensure areas are computed
        df_aux = pd.merge(
            self.table[["Id", "Color"]], self.get_areas(), how="left", on="Id"
        )
        df_aux = df_aux.sort_values(by="{}_m2".format(self.areafield), ascending=True)
        if filter:
            df_aux = df_aux.query("{}_m2 > 0".format(self.areafield))

        # -----------------------------------------------
        # Deploy figure
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        gs = mpl.gridspec.GridSpec(
            specs["gs_rows"],
            specs["gs_cols"],
            wspace=0.8,
            hspace=0.05,
            left=0.05,
            bottom=0.1,
            top=0.85,
            right=0.95,
        )
        fig.suptitle(specs["suptitle"])

        # plot map
        plt.subplot(gs[:5, :3])
        plt.title("a. {}".format(specs["a_title"]), loc="left")
        im = plt.imshow(
            self.grid, cmap=specs["cmap"], vmin=specs["vmin"], vmax=specs["vmax"]
        )
        plt.axis("off")

        # place legend
        legend_elements = []
        for i in range(len(df_aux)):
            _color = df_aux[self.colorfield].values[i]
            _label = "{} ({})".format(
                df_aux[self.namefield].values[i],
                df_aux[self.aliasfield].values[i],
            )
            legend_elements.append(
                Patch(
                    facecolor=_color,
                    label=_label,
                )
            )
        plt.legend(
            frameon=True,
            fontsize=9,
            markerscale=0.8,
            handles=legend_elements,
            bbox_to_anchor=(specs["legend_x"], specs["legend_y"]),
            bbox_transform=fig.transFigure,
            ncol=specs["legend_ncol"],
        )

        # -----------------------------------------------
        # plot horizontal bar of areas

        plt.subplot(gs[: default_specs["gs_b_rowlim"], 3:])
        plt.title("b. {}".format(specs["b_title"]), loc="left")
        if specs["bars_alias"]:
            s_bar_labels = self.aliasfield
        else:
            s_bar_labels = self.namefield
        plt.barh(
            df_aux[s_bar_labels],
            df_aux["{}_{}".format(self.areafield, specs["b_area"])],
            color=df_aux[self.colorfield],
        )

        # Add labels for each bar
        if specs["b_xmax"] is None:
            specs["b_xmax"] = df_aux[
                "{}_{}".format(self.areafield, specs["b_area"])
            ].max()
        for i in range(len(df_aux)):
            v = df_aux["{}_{}".format(self.areafield, specs["b_area"])].values[i]
            p = df_aux["{}_%".format(self.areafield)].values[i]
            plt.text(
                v + specs["b_xmax"] / 50,
                i - 0.3,
                "{:.1f} ({:.1f}%)".format(v, p),
                fontsize=9,
            )
        plt.xlim(0, 1.5 * specs["b_xmax"])
        plt.xlabel("{} (km$^2$)".format(specs["b_xlabel"]))
        plt.grid(axis="y")

        # -----------------------------------------------
        # plot metadata
        lst_meta = []
        lst_value = []
        for k in self.asc_metadata:
            lst_value.append(self.asc_metadata[k])
            lst_meta.append(k)
        df_meta = pd.DataFrame({"Raster": lst_meta, "Value": lst_value})
        # metadata
        n_y = 0.25
        n_x = 0.62
        plt.text(
            x=n_x,
            y=n_y,
            s="c. {}".format(specs["c_title"]),
            fontsize=12,
            transform=fig.transFigure,
        )
        n_y = n_y - 0.01
        n_step = 0.025
        for i in range(len(df_meta)):
            s_head = df_meta["Raster"].values[i]
            if s_head == "cellsize":
                s_value = self.cellsize
                s_line = "{:>15}: {:<10.5f}".format(s_head, s_value)
            else:
                s_value = df_meta["Value"].values[i]
                s_line = "{:>15}: {:<10.2f}".format(s_head, s_value)
            n_y = n_y - n_step
            plt.text(
                x=n_x,
                y=n_y,
                s=s_line,
                fontsize=9,
                fontdict={"family": "monospace"},
                transform=fig.transFigure,
            )

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "{}_{}".format(self.varalias, self.name)
            plt.savefig("{}/{}.{}".format(folder, filename, fig_format), dpi=dpi)
        plt.close(fig)
        return None


class LULC(QualiRaster):
    """
    Land Use and Land Cover map dataset
    """

    def __init__(self, name, date):
        """Initialize :class:`LULC` map

        :param name: name of map
        :type name: str
        :param date: date of map in ``yyyy-mm-dd``
        :type date: str
        """
        super().__init__(name, dtype="uint8")
        self.cmap = "tab20b"
        self.varname = "Land Use and Land Cover"
        self.varalias = "LULC"
        self.description = "Classes of Land Use and Land Cover"
        self.units = "classes ID"
        self.date = date


class LULCChange(QualiRaster):
    """
    Land Use and Land Cover Change map dataset
    """

    def __init__(self, name, date_start, date_end, name_lulc):
        """Initialize :class:`LULCChange` map

        :param name: name of map
        :type name: str
        :param date_start: date of map in ``yyyy-mm-dd``
        :type date: str
        """
        super().__init__(name, dtype="uint8")
        self.cmap = "tab20b"
        self.varname = "LULC Change"
        self.varalias = "LULCC"
        self.description = "Change of Land Use and Land Cover"
        self.units = "Change ID"
        self.date_start = date_start
        self.date_end = date_end
        self.date = date_end
        self.table = pd.DataFrame(
            {
                self.idfield: [
                    1,
                    2,
                    3,
                ],
                self.namefield: ["Retraction", "Stable", "Expansion"],
                self.aliasfield: ["Rtr", "Stb", "Exp"],
                self.colorfield: ["tab:purple", "tab:orange", "tab:red"],
            }
        )


class Litology(QualiRaster):
    """
    Litology map dataset
    """

    def __init__(self, name="LitoMap"):
        super().__init__(name, dtype="uint8")
        self.cmap = "tab20c"
        self.varname = "Litological Domains"
        self.varalias = "Lito"
        self.description = "Litological outgcrop domains"
        self.units = "types ID"


class Soils(QualiRaster):
    """
    Soils map dataset
    """

    def __init__(self, name="SoilsMap"):
        super().__init__(name, dtype="uint8")
        self.cmap = "tab20c"
        self.varname = "Soil Types"
        self.varalias = "Soils"
        self.description = "Types of Soils and Substrate"
        self.units = "types ID"

    def get_hydro_soils(self, map_lito, map_hand, hand_threshold=2):
        # get table copy from lito
        new_table = map_lito.table[["Id", "Alias", "Name", "Color"]].copy()
        # process grid
        grd_soils = map_lito.grid.copy()
        grd_soils = grd_soils * (map_hand.grid > hand_threshold)
        n_all_id = new_table["Id"].max() + 1
        grd_alluvial = n_all_id * (map_hand.grid <= hand_threshold)
        grd_soils = grd_soils + grd_alluvial
        self.set_grid(grid=grd_soils)

        # edit table
        new_table["Name"] = "Residual " + new_table["Name"]
        new_table["Alias"] = "R" + new_table["Alias"]
        # new soil table
        df_aux = pd.DataFrame(
            {"Id": [n_all_id], "Alias": ["Alv"], "Name": ["Alluvial"], "Color": ["tan"]}
        )
        # append
        new_table = pd.concat([new_table, df_aux], ignore_index=True)
        # set table
        self.set_table(dataframe=new_table)
        # set more attributes
        self.prj = map_lito.prj
        self.set_asc_metadata(metadata=map_lito.asc_metadata)


class AOI(QualiRaster):
    """
    AOI map dataset
    """

    def __init__(self, name="AOIMap"):
        super().__init__(name, dtype="uint8")
        self.varname = "Area Of Interest"
        self.varalias = "AOI"
        self.description = "Boolean map an Area of Interest"
        self.units = "classes ID"
        self.table = pd.DataFrame(
            {"Id": [1], "Name": [self.name], "Alias": ["AOI"], "Color": "silver"}
        )

    def load(self, asc_file, prj_file):
        """
        Load data from files to raster
        :param asc_file: path_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path_main to ``.prj`` projection file
        :type prj_file: str
        :return: None
        :rtype: None
        """
        self.load_asc_raster(file=asc_file)
        self.load_prj_file(file=prj_file)
        return None

    def view(
        self,
        show=True,
        folder="./output",
        filename=None,
        specs=None,
        dpi=150,
        fig_format="jpg",
    ):
        """Plot a basic pannel of raster map.

        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        """
        map_aoi_aux = QualiRaster(name=self.name)
        df_aoi = pd.DataFrame(
            {
                "Id": [1, 2],
                "Alias": ["AOI", "EZ"],
                "Name": ["Area of Interest", "Exclusion Zone"],
                "Color": ["magenta", "silver"],
            }
        )
        # set up
        map_aoi_aux.varname = self.varname
        map_aoi_aux.varalias = self.varalias
        map_aoi_aux.units = self.units
        map_aoi_aux.set_table(dataframe=df_aoi)
        map_aoi_aux.set_asc_metadata(metadata=self.asc_metadata)
        map_aoi_aux.prj = self.prj
        # process grid
        self.insert_nodata()
        grd_new = 2 * np.ones(shape=self.grid.shape, dtype="byte")
        grd_new = grd_new - (1 * (self.grid == 1))
        self.mask_nodata()
        map_aoi_aux.set_grid(grid=grd_new)
        map_aoi_aux.view(
            show=show,
            folder=folder,
            filename=filename,
            specs=specs,
            dpi=dpi,
            fig_format=fig_format,
        )
        del map_aoi_aux
        return None


class Zones(QualiRaster):
    """
    Zones map dataset
    """

    def __init__(self, name="ZonesMap"):
        super().__init__(name, dtype="uint32")
        self.varname = "Zone"
        self.varalias = "ZN"
        self.description = "Ids map of zones"
        self.units = "zones ID"
        self.table = None

    def set_table(self):
        if self.grid is None:
            self.table = None
        else:
            self.insert_nodata()
            # get unique values
            vct_unique = np.unique(self.grid)
            # reapply mask
            self.mask_nodata()
            # set table
            self.table = pd.DataFrame(
                {
                    "Id": vct_unique,
                    "Alias": ["{}{}".format(self.varalias, vct_unique[i]) for i in range(len(vct_unique))],
                    "Name": ["{} {}".format(self.varname, vct_unique[i]) for i in range(len(vct_unique))]
                }
            )
            self.table = self.table.drop(self.table[self.table["Id"] == self.asc_metadata["NODATA_value"]].index)
            self.table["Id"] = self.table["Id"].astype(int)
            self.table = self.table.sort_values(by="Id")
            self.table = self.table.reset_index(drop=True)
            self.set_random_colors()
            del vct_unique
            return None

    def set_grid(self, grid):
        super().set_grid(grid)
        self.set_table()
        return None

    def load(self, asc_file, prj_file):
        """
        Load data from files to raster
        :param asc_file: path_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path_main to ``.prj`` projection file
        :type prj_file: str
        :return: None
        :rtype: None
        """
        self.load_asc_raster(file=asc_file)
        self.load_prj_file(file=prj_file)
        return None

    def get_aoi(self, zone_id):
        """
        Get the AOI map from a zone id
        :param zone_id: number of zone ID
        :type zone_id: int
        :return: AOI map
        :rtype: :class:`AOI` object
        """
        map_aoi = AOI(name="{} {}".format(self.varname, zone_id))
        map_aoi.set_asc_metadata(metadata=self.asc_metadata)
        map_aoi.prj = self.prj
        self.insert_nodata()
        map_aoi.set_grid(grid=1 * (self.grid == zone_id))
        self.mask_nodata()
        return map_aoi

    def view(
        self,
        show=True,
        folder="./output",
        filename=None,
        specs=None,
        dpi=150,
        fig_format="jpg",
    ):
        """Plot a basic pannel of raster map.

        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        """
        map_zones_aux = Raster(name=self.name)
        # set up
        map_zones_aux.varname = self.varname
        map_zones_aux.varalias = self.varalias
        map_zones_aux.units = self.units
        map_zones_aux.set_asc_metadata(metadata=self.asc_metadata)
        map_zones_aux.prj = self.prj
        map_zones_aux.cmap = "tab20"
        self.insert_nodata()
        map_zones_aux.set_grid(grid=self.grid)
        self.mask_nodata()
        map_zones_aux.view(
            show=show,
            folder=folder,
            filename=filename,
            specs=specs,
            dpi=dpi,
            fig_format=fig_format,
        )
        del map_zones_aux
        return None

# -----------------------------------------
# Raster Collection data structures

class RasterCollection(Collection):
    """
    The raster collection base dataset.
    This data strucute is designed for holding and comparing :class:`Raster` objects.
    """

    def __init__(self, name="myRasterCollection"):
        """Deploy the raster collection data structure.

        :param name: name of raster collection
        :type name: str
        :param dtype: data type of raster cells, defaults to float32
        :type dtype: str
        """
        obj_aux = Raster()
        super().__init__(base_object=obj_aux, name=name)
        # set up date fields and special attributes
        self.catalog["Date"] = pd.to_datetime(self.catalog["Date"])

    def load(
        self,
        name,
        asc_file,
        prj_file=None,
        varname=None,
        varalias=None,
        units=None,
        date=None,
        dtype="float32",
    ):
        """Load a :class:`Raster` base_object from a ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param asc_file: path_main to ``.asc`` raster file
        :type asc_file: str
        :param varname: :class:`Raster.varname` variable name attribute, defaults to None
        :type varname: str
        :param varalias: :class:`Raster.varalias` variable alias attribute, defaults to None
        :type varalias: str
        :param units: :class:`Raster.units` units attribute, defaults to None
        :type units: str
        :param date: :class:`Raster.date` date attribute, defaults to None
        :type date: str
        """
        # create raster
        rst_aux = Raster(name=name, dtype=dtype)
        # set attributes
        rst_aux.varname = varname
        rst_aux.varalias = varalias
        rst_aux.units = units
        rst_aux.date = date
        # load prj file
        if prj_file is None:
            pass
        else:
            rst_aux.load_prj_file(file=prj_file)
        # read asc file
        rst_aux.load_asc_raster(file=asc_file)
        # append to collection
        self.append(new_object=rst_aux)
        # delete aux
        del rst_aux
        return None

    def get_collection_stats(self):
        """Get basic statistics from collection.

        :return: statistics data
        :rtype: :class:`pandas.DataFrame`
        """
        # deploy dataframe
        df_aux = self.catalog[["Name"]].copy()
        lst_stats = []
        for i in range(len(self.catalog)):
            s_name = self.catalog["Name"].values[i]
            df_stats = self.collection[s_name].get_grid_stats()
            lst_stats.append(df_stats.copy())
        # deploy fields
        for k in df_stats["Statistic"]:
            df_aux[k] = 0.0

        # fill values
        for i in range(len(df_aux)):
            df_aux.loc[i, "Count":"Max"] = lst_stats[i]["Value"].values
        df_aux["Count"] = df_aux["Count"].astype(dtype="uint16")
        return df_aux

    def get_views(
        self, show=True, folder="./output", specs=None, dpi=150, fig_format="jpg"
    ):
        """Plot all basic pannel of raster maps in collection.

        :param show: boolean to show plot instead of saving,
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """

        # plot loop
        for k in self.collection:
            rst_lcl = self.collection[k]
            s_name = rst_lcl.name
            rst_lcl.view(
                show=show,
                specs=specs,
                folder=folder,
                filename=s_name,
                dpi=dpi,
                fig_format=fig_format,
            )
        return None

    def view_bboxes(
        self,
        colors=None,
        datapoints=False,
        show=True,
        folder="./output",
        filename=None,
        dpi=150,
        fig_format="jpg",
    ):
        """View Bounding Boxes of Raster collection

        :param colors: list of colors for plotting. expected to be the same runsize of catalog
        :type colors: list
        :param datapoints: option to plot datapoints as well, defaults to False
        :type datapoints: bool
        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: none
        """
        plt.style.use("seaborn-v0_8")
        fig = plt.figure(figsize=(5, 5))
        # get colors
        lst_colors = colors
        if colors is None:
            lst_colors = get_random_colors(size=len(self.catalog))
        # colect names and bboxes
        lst_x_values = list()
        lst_y_values = list()
        dct_bboxes = dict()
        dct_colors = dict()
        _c = 0
        for name in self.collection:
            dct_colors[name] = lst_colors[_c]
            lcl_bbox = self.collection[name].get_bbox()
            dct_bboxes[name] = lcl_bbox
            # append coordinates
            lst_x_values.append(lcl_bbox["xmin"])
            lst_x_values.append(lcl_bbox["xmax"])
            lst_y_values.append(lcl_bbox["ymin"])
            lst_y_values.append(lcl_bbox["ymax"])
            _c = _c + 1
        # get min and max
        n_xmin = np.min(lst_x_values)
        n_xmax = np.max(lst_x_values)
        n_ymin = np.min(lst_y_values)
        n_ymax = np.max(lst_y_values)
        # get ranges
        n_x_range = np.abs(n_xmax - n_xmin)
        n_y_range = np.abs(n_ymax - n_ymin)

        # plot loop
        for name in dct_bboxes:
            plt.scatter(
                dct_bboxes[name]["xmin"],
                dct_bboxes[name]["ymin"],
                marker="^",
                color=dct_colors[name],
            )
            if datapoints:
                df_dpoints = self.collection[name].get_grid_datapoints(drop_nan=False)
                plt.scatter(
                    df_dpoints["x"], df_dpoints["y"], color=dct_colors[name], marker="."
                )
            _w = dct_bboxes[name]["xmax"] - dct_bboxes[name]["xmin"]
            _h = dct_bboxes[name]["ymax"] - dct_bboxes[name]["ymin"]
            rect = plt.Rectangle(
                xy=(dct_bboxes[name]["xmin"], dct_bboxes[name]["ymin"]),
                width=_w,
                height=_h,
                alpha=0.5,
                label=name,
                color=dct_colors[name],
            )
            plt.gca().add_patch(rect)
        plt.ylim(n_ymin - (n_y_range / 3), n_ymax + (n_y_range / 3))
        plt.xlim(n_xmin - (n_x_range / 3), n_xmax + (n_x_range / 3))
        plt.gca().set_aspect("equal")
        plt.legend()

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "bboxes"
            plt.savefig("{}/{}.{}".format(folder, filename, fig_format), dpi=dpi)
        plt.close(fig)
        return None


class QualiRasterCollection(RasterCollection):
    """
    The raster collection base dataset.

    This data strucute is designed for holding and comparing :class:`QualiRaster` objects.
    """

    def __init__(self, name, varname, varalias, dtype="uint8"):
        """Deploy Qualitative Raster Series

        :param name: :class:`RasterSeries.name` name attribute
        :type name: str
        :param varname: :class:`Raster.varname` variable name attribute, defaults to None
        :type varname: str
        :param varalias: :class:`Raster.varalias` variable alias attribute, defaults to None
        :type varalias: str
        """
        super().__init__(
            name=name, varname=varname, varalias=varalias, dtype=dtype, units="ID"
        )

    def load(self, name, asc_file, prj_file=None, table_file=None):
        """Load a :class:`QualiRaster` base_object from ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param asc_file: path_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path_main to ``.prj`` projection file
        :type prj_file: str
        :param table_file: path_main to ``.txt`` table file
        :type table_file: str
        """
        # create raster
        rst_aux = QualiRaster(name=name)
        # read file
        rst_aux.load_asc_raster(file=asc_file)
        # load prj
        if prj_file is None:
            pass
        else:
            rst_aux.load_prj_file(file=prj_file)
        # set table
        if table_file is None:
            pass
        else:
            rst_aux.load_table(file=table_file)
        # append to collection
        self.append(new_object=rst_aux)
        # delete aux
        del rst_aux
        return None


class RasterSeries(RasterCollection):
    """A :class:`RasterCollection` where date matters and all maps in collections are
    expected to be the same variable, same projection and same grid.
    """

    def __init__(self, name, varname, varalias, units, dtype="float32"):
        """Deploy RasterSeries

        :param name: :class:`RasterSeries.name` name attribute
        :type name: str
        :param varname: :class:`Raster.varname` variable name attribute, defaults to None
        :type varname: str
        :param varalias: :class:`Raster.varalias` variable alias attribute, defaults to None
        :type varalias: str
        :param units: :class:`Raster.units` units attribute, defaults to None
        :type units: str
        """
        super().__init__(name=name)
        self.varname = varname
        self.varalias = varalias
        self.units = units
        self.dtype = dtype

    def load(self, name, date, asc_file, prj_file=None):
        """Load a :class:`Raster` base_object from a ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param date: :class:`Raster.date` date attribute, defaults to None
        :type date: str
        :param asc_file: path_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path_main to ``.prj`` projection file
        :type prj_file: str
        :return: None
        :rtype: None
        """
        # create raster
        rst_aux = Raster(name=name, dtype=self.dtype)
        # set attributes
        rst_aux.varname = self.varname
        rst_aux.varalias = self.varalias
        rst_aux.units = self.units
        rst_aux.date = date
        # read file
        rst_aux.load_asc_raster(file=asc_file)
        # append to collection
        self.append(new_object=rst_aux)
        # load prj file
        if prj_file is None:
            pass
        else:
            rst_aux.load_prj_file(file=prj_file)
        # delete aux
        del rst_aux
        return None

    def load_folder(self, folder, name_pattern="map_*", talk=False):
        """
        Load all rasters from a folder by following a name pattern.
        Date is expected to be at the end of name before file extension.
        :param folder: path_main to folder
        :type folder: str
        :param name_pattern: name pattern. example map_*
        :type name_pattern: str
        :param talk: option for printing messages
        :type talk: bool
        :return: None
        :rtype: None
        """
        #
        lst_maps = glob.glob("{}/{}.asc".format(folder, name_pattern))
        lst_prjs = glob.glob("{}/{}.prj".format(folder, name_pattern))
        if talk:
            print("loading folder...")
        for i in range(len(lst_maps)):
            asc_file = lst_maps[i]
            prj_file = lst_prjs[i]
            # get name
            s_name = os.path.basename(asc_file).split(".")[0]
            # get dates
            s_date_map = asc_file.split("_")[-1].split(".")[0]
            s_date_prj = prj_file.split("_")[-1].split(".")[0]
            # load
            self.load(
                name=s_name,
                date=s_date_map,
                asc_file=asc_file,
                prj_file=prj_file,
            )
        self.update(details=True)
        return None

    def apply_aoi_masks(self, grid_aoi):
        """Batch method to apply AOI mask over all maps in collection

        :param grid_aoi: aoi grid
        :type grid_aoi: :class:`numpy.ndarray`
        :return: None
        :rtype: None
        """
        for name in self.collection:
            self.collection[name].apply_aoi_mask(grid_aoi=grid_aoi, inplace=True)
        return None

    def rebase_grids(self, base_raster, talk=False):
        """Batch method for rebase all maps in collection

        :param base_raster: base raster for rebasing
        :type base_raster: :class:`datasets.Raster`
        :param talk: option for print messages
        :type talk: bool
        :return: None
        :rtype: None
        """
        if talk:
            print("rebase grids...")
        for name in self.collection:
            self.collection[name].rebase_grid(base_raster=base_raster, inplace=True)
        self.update(details=True)
        return None

    def get_series_stats(self):
        """Get the raster series statistics
        :return: dataframe of raster series statistics
        :rtype: :class:`pandas.DataFrame`
        """
        df_stats = self.get_collection_stats()
        df_series = pd.merge(
            self.catalog[["Name", "Date"]], df_stats, how="left", on="Name"
        )
        return df_series

    def get_views(
        self, show=True, folder="./output", specs=None, dpi=150, fig_format="jpg"
    ):
        """Plot all basic pannel of raster maps in collection.

        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """

        # get stats
        df_stas = self.get_collection_stats()
        n_vmin = df_stas["min"].max()
        n_max = df_stas["max"].max()

        # handle specs
        default_specs = {"vmin": n_vmin, "vmax": n_max, "hist_vmax": 0.05}
        if specs is None:
            specs = default_specs
        else:
            # overwrite incoming specs
            for k in default_specs:
                specs[k] = default_specs[k]

        # plot loop
        for k in self.collection:
            rst_lcl = self.collection[k]
            s_name = rst_lcl.name
            rst_lcl.view(
                show=show,
                specs=specs,
                folder=folder,
                filename=s_name,
                dpi=dpi,
                fig_format=fig_format,
            )
        return None

    def view_series_stats(
        self,
        statistic="mean",
        folder="./output",
        filename=None,
        specs=None,
        show=True,
        dpi=150,
        fig_format="jpg",
    ):
        """
        View raster series statistics

        :param statistic: statistc to view. Default mean
        :type statistic: str
        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """
        df_series = self.get_series_stats()
        ts = DailySeries(
            name=self.name, varname="{}_{}".format(self.varname, statistic)
        )
        ts.set_data(dataframe=df_series, varfield=statistic, datefield="Date")
        default_specs = {
            "suptitle": "{} | {} {} series".format(self.name, self.varname, statistic),
            "ylabel": self.units,
        }
        if specs is None:
            specs = default_specs
        else:
            # overwrite incoming specs
            for k in default_specs:
                specs[k] = default_specs[k]
        ts.view(
            show=show,
            folder=folder,
            filename=filename,
            specs=specs,
            dpi=dpi,
            fig_format=fig_format,
        )
        return None


class NDVISeries(RasterSeries):
    def __init__(self, name):
        # instantiate raster sample
        rst_aux = NDVI(name="dummy", date=None)
        super().__init__(
            name=name,
            varname=rst_aux.varname,
            varalias=rst_aux.varalias,
            units=rst_aux.units,
            dtype=rst_aux.dtype,
        )
        # remove
        del rst_aux

    def load(self, name, date, asc_file, prj_file):
        """Load a :class:`NDVI` base_object from a ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param date: :class:`Raster.date` date attribute, defaults to None
        :type date: str
        :param asc_file: path_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path_main to ``.prj`` projection file
        :type prj_file: str
        :return: None
        :rtype: None
        """
        # create raster
        rst_aux = NDVI(name=name, date=date)
        # read file
        rst_aux.load_asc_raster(file=asc_file)
        # append to collection
        self.append(new_object=rst_aux)
        # load prj file
        rst_aux.load_prj_file(file=prj_file)
        # delete aux
        del rst_aux
        return None


class ETSeries(RasterSeries):
    def __init__(self, name):
        # instantiate raster sample
        rst_aux = ET24h(name="dummy", date=None)
        super().__init__(
            name=name,
            varname=rst_aux.varname,
            varalias=rst_aux.varalias,
            units=rst_aux.units,
            dtype=rst_aux.dtype,
        )
        # remove
        del rst_aux

    def load(self, name, date, asc_file, prj_file):
        """Load a :class:`ET24h` base_object from a ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param date: :class:`Raster.date` date attribute, defaults to None
        :type date: str
        :param asc_file: path_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path_main to ``.prj`` projection file
        :type prj_file: str
        :return: None
        :rtype: None
        """
        # create raster
        rst_aux = ET24h(name=name, date=date)
        # read file
        rst_aux.load_asc_raster(file=asc_file)
        # append to collection
        self.append(new_object=rst_aux)
        # load prj file
        rst_aux.load_prj_file(file=prj_file)
        # delete aux
        del rst_aux
        return None


class QualiRasterSeries(RasterSeries):
    """A :class:`RasterSeries` where date matters and all maps in collections are
    expected to be :class:`QualiRaster` with the same variable, same projection and same grid.
    """

    def __init__(self, name, varname, varalias, dtype="uint8"):
        """Deploy Qualitative Raster Series

        :param name: :class:`RasterSeries.name` name attribute
        :type name: str
        :param varname: :class:`Raster.varname` variable name attribute, defaults to None
        :type varname: str
        :param varalias: :class:`Raster.varalias` variable alias attribute, defaults to None
        :type varalias: str
        """
        super().__init__(
            name=name, varname=varname, varalias=varalias, dtype=dtype, units="ID"
        )
        self.table = None

    def update_table(self):
        """
        Update series table (attributes)
        :return: None
        :rtype: None
        """
        if len(self.catalog) == 0:
            pass
        else:
            for i in range(len(self.catalog)):
                _name = self.catalog["Name"].values[i]
                if i == 0:
                    self.table = self.collection[_name].table.copy()
                else:
                    self.table = pd.concat(
                        [self.table, self.collection[_name].table.copy()]
                    )
        self.table = self.table.drop_duplicates(subset="Id", keep="last")
        return None

    def append(self, raster):
        """Append a :class:`Raster` base_object to collection.
        Pre-existing objects with the same :class:`Raster.name` attribute are replaced

        :param raster: incoming :class:`Raster` to append
        :type raster: :class:`Raster`
        """
        super().append(new_object=raster)
        self.update_table()
        return None

    def load(self, name, date, asc_file, prj_file=None, table_file=None):
        """Load a :class:`QualiRaster` base_object from ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param date: :class:`Raster.date` date attribute
        :type date: str
        :param asc_file: path_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path_main to ``.prj`` projection file
        :type prj_file: str
        :param table_file: path_main to ``.txt`` table file
        :type table_file: str
        """
        # create raster
        rst_aux = QualiRaster(name=name)
        # set attributes
        rst_aux.date = date
        # read file
        rst_aux.load_asc_raster(file=asc_file)
        # load prj
        if prj_file is None:
            pass
        else:
            rst_aux.load_prj_file(file=prj_file)
        # set table
        if table_file is None:
            pass
        else:
            rst_aux.load_table(file=table_file)
        # append to collection
        self.append(new_object=rst_aux)
        # delete aux
        del rst_aux

    def load_folder(self, folder, table_file, name_pattern="map_*", talk=False):
        """
        Load all rasters from a folder by following a name pattern.
        Date is expected to be at the end of name before file extension.
        :param folder: path_main to folder
        :type folder: str
        :param table_file: path_main to table file
        :type table_file: str
        :param name_pattern: name pattern. example map_*
        :type name_pattern: str
        :param talk: option for printing messages
        :type talk: bool
        :return: None
        :rtype: None
        """
        #
        lst_maps = glob.glob("{}/{}.asc".format(folder, name_pattern))
        lst_prjs = glob.glob("{}/{}.prj".format(folder, name_pattern))
        if talk:
            print("loading folder...")
        for i in range(len(lst_maps)):
            asc_file = lst_maps[i]
            prj_file = lst_prjs[i]
            # get name
            s_name = os.path.basename(asc_file).split(".")[0]
            # get dates
            s_date_map = asc_file.split("_")[-1].split(".")[0]
            s_date_prj = prj_file.split("_")[-1].split(".")[0]
            # load
            self.load(
                name=s_name,
                date=s_date_map,
                asc_file=asc_file,
                prj_file=prj_file,
                table_file=table_file,
            )
        return None

    def get_series_areas(self):
        """
        Get areas prevalance for all series
        :return: dataframe of series areas
        :rtype: :class:`pandas.DataFrame`
        """
        # compute areas for each raster
        for i in range(len(self.catalog)):
            s_raster_name = self.catalog["Name"].values[i]
            s_raster_date = self.catalog["Date"].values[i]
            # compute
            df_areas = self.collection[s_raster_name].get_areas()
            # insert name and date fields
            df_areas.insert(loc=0, column="Name_raster", value=s_raster_name)
            df_areas.insert(loc=1, column="Date", value=s_raster_date)
            # concat dataframes
            if i == 0:
                df_areas_full = df_areas.copy()
            else:
                df_areas_full = pd.concat([df_areas_full, df_areas])
        df_areas_full["Name"] = df_areas_full["Name"].astype("category")
        df_areas_full["Date"] = pd.to_datetime(df_areas_full["Date"])
        return df_areas_full

    def view_series_areas(
        self,
        specs=None,
        show=True,
        folder="./output",
        filename=None,
        dpi=150,
        fig_format="jpg",
    ):
        """
        View series areas
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """
        plt.style.use("seaborn-v0_8")
        # get specs
        default_specs = {
            "suptitle": "{} | Area Series".format(self.name),
            "width": 5 * 1.618,
            "height": 5,
            "ylabel": "Area prevalence (%)",
            "ylim": (0, 100),
            "legend_x": 0.85,
            "legend_y": 0.33,
            "legend_ncol": 3,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # compute areas
        df_areas = self.get_series_areas()

        # Deploy figure
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        gs = mpl.gridspec.GridSpec(
            3, 1, wspace=0.5, hspace=0.9, left=0.1, bottom=0.1, top=0.9, right=0.95
        )
        fig.suptitle(specs["suptitle"])

        # start plotting
        plt.subplot(gs[0:2, 0])
        for i in range(len(self.table)):
            # get attributes
            _id = self.table["Id"].values[i]
            _name = self.table["Name"].values[i]
            _alias = self.table["Alias"].values[i]
            _color = self.table["Color"].values[i]
            # filter series
            _df = df_areas.query("Id == {}".format(_id)).copy()
            plt.plot(_df["Date"], _df["Area_%"], color=_color, label=_name)
        plt.legend(
            frameon=True,
            fontsize=9,
            markerscale=0.8,
            bbox_to_anchor=(specs["legend_x"], specs["legend_y"]),
            bbox_transform=fig.transFigure,
            ncol=specs["legend_ncol"],
        )
        plt.xlim(df_areas["Date"].min(), df_areas["Date"].max())
        plt.ylabel(specs["ylabel"])
        plt.ylim(specs["ylim"])

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "{}_{}".format(self.varalias, self.name)
            plt.savefig("{}/{}.{}".format(folder, filename, fig_format), dpi=dpi)
        plt.close(fig)
        return None

    def get_views(
        self,
        show=True,
        folder="./output",
        specs=None,
        dpi=150,
        fig_format="jpg",
        filter=False,
        talk=False,
    ):
        """Plot all basic pannel of qualiraster maps in collection.

        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path_main to output folder, defaults to ``./output``
        :type folder: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :param filter: option for cutting off zero-area classes
        :type filter: bool
        :return: None
        :rtype: None
        """
        # plot loop
        for k in self.collection:
            rst_lcl = self.collection[k]
            s_name = rst_lcl.name
            if talk:
                print("exporting {} view...".format(s_name))
            rst_lcl.view(
                show=show,
                specs=specs,
                folder=folder,
                filename=self.name + "_" + s_name,
                dpi=dpi,
                fig_format=fig_format,
                filter=filter,
            )
        return None


class LULCSeries(QualiRasterSeries):
    """
    A :class:`QualiRasterSeries` for holding Land Use and Land Cover maps
    """

    def __init__(self, name):
        # instantiate raster sample
        rst_aux = LULC(name="dummy", date=None)
        super().__init__(
            name=name,
            varname=rst_aux.varname,
            varalias=rst_aux.varalias,
            dtype=rst_aux.dtype,
        )
        # remove
        del rst_aux

    def load(self, name, date, asc_file, prj_file=None, table_file=None):
        """Load a :class:`LULCRaster` base_object from ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param date: :class:`Raster.date` date attribute
        :type date: str
        :param asc_file: path_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path_main to ``.prj`` projection file
        :type prj_file: str
        :param table_file: path_main to ``.txt`` table file
        :type table_file: str
        :return: None
        :rtype: None
        """
        # create raster
        rst_aux = LULC(name=name, date=date)
        # read file
        rst_aux.load_asc_raster(file=asc_file)
        # load prj
        if prj_file is None:
            pass
        else:
            rst_aux.load_prj_file(file=prj_file)
        # set table
        if table_file is None:
            pass
        else:
            rst_aux.load_table(file=table_file)
        # append to collection
        self.append(raster=rst_aux)
        # delete aux
        del rst_aux
        return None

    def get_lulcc(self, date_start, date_end, by_lulc_id):
        """
        Get the :class:`LULCChange` of a given time interval and LULC class Id
        :param date_start: start date of time interval
        :type date_start: str
        :param date_end: end date of time interval
        :type date_end: str
        :param by_lulc_id: LULC class Id
        :type by_lulc_id: int
        :return: map of LULC Change
        :rtype: :class:`LULCChange`
        """
        # set up
        s_name_start = self.catalog.loc[self.catalog["Date"] == date_start][
            "Name"
        ].values[
            0
        ]  #
        s_name_end = self.catalog.loc[self.catalog["Date"] == date_end]["Name"].values[
            0
        ]

        # compute lulc change grid
        grd_lulcc = (1 * (self.collection[s_name_end].grid == by_lulc_id)) - (
            1 * (self.collection[s_name_start].grid == by_lulc_id)
        )
        grd_all = (1 * (self.collection[s_name_end].grid == by_lulc_id)) + (
            1 * (self.collection[s_name_start].grid == by_lulc_id)
        )
        grd_all = 1 * (grd_all > 0)
        grd_lulcc = (grd_lulcc + 2) * grd_all

        # get names
        s_name = self.name
        s_name_lulc = self.table.loc[self.table["Id"] == by_lulc_id]["Name"].values[0]
        # instantiate
        map_lulc_change = LULCChange(
            name="{} of {} from {} to {}".format(
                s_name, s_name_lulc, date_start, date_end
            ),
            name_lulc=s_name_lulc,
            date_start=date_start,
            date_end=date_end,
        )
        map_lulc_change.set_grid(grid=grd_lulcc)
        map_lulc_change.set_asc_metadata(
            metadata=self.collection[s_name_start].asc_metadata
        )
        map_lulc_change.prj = self.collection[s_name_start].prj

        return map_lulc_change

    def get_lulcc_series(self, by_lulc_id):
        """
        Get the :class:`QualiRasterSeries` of LULC Change for the entire LULC series for a given LULC Id
        :param by_lulc_id: LULC class Id
        :type by_lulc_id: int
        :return: Series of LULC Change
        :rtype: :class:`QualiRasterSeries`
        """
        series_lulcc = QualiRasterSeries(
            name="{} - Change Series".format(self.name),
            varname="Land Use and Land Cover Change",
            varalias="LULCC",
        )
        # loop in catalog
        for i in range(1, len(self.catalog)):
            raster = self.get_lulcc(
                date_start=self.catalog["Date"].values[i - 1],
                date_end=self.catalog["Date"].values[i],
                by_lulc_id=by_lulc_id,
            )
            series_lulcc.append(raster=raster)
        return series_lulcc

    def get_conversion_matrix(self, date_start, date_end, talk=False):
        """
        Compute the conversion matrix, expansion matrix and retraction matrix for a given interval
        :param date_start: start date of time interval
        :type date_start: str
        :param date_end: end date of time interval
        :type date_end: str
        :param talk: option for printing messages
        :type talk: bool
        :return: dict of outputs
        :rtype: dict
        """
        # get dates
        s_date_start = date_start
        s_date_end = date_end
        # get raster names
        s_name_start = self.catalog.loc[self.catalog["Date"] == date_start][
            "Name"
        ].values[
            0
        ]  #
        s_name_end = self.catalog.loc[self.catalog["Date"] == date_end]["Name"].values[
            0
        ]

        # compute areas
        df_areas_start = self.collection[s_name_start].get_areas()
        df_areas_end = self.collection[s_name_end].get_areas()
        # deploy variables
        df_conv = self.table.copy()
        df_conv["Date_start"] = s_date_start
        df_conv["Date_end"] = s_date_end
        df_conv["Area_f_start"] = df_areas_start["Area_f"].values
        df_conv["Area_f_end"] = df_areas_end["Area_f"].values
        df_conv["Area_km2_start"] = df_areas_start["Area_km2"].values
        df_conv["Area_km2_end"] = df_areas_end["Area_km2"].values

        lst_cols = list()
        for i in range(len(df_conv)):
            _alias = df_conv["Alias"].values[i]
            s_field = "to_{}_f".format(_alias)
            df_conv[s_field] = 0.0
            lst_cols.append(s_field)

        if talk:
            print("processing...")

        grd_conv = np.zeros(shape=(len(df_conv), len(df_conv)))
        for i in range(len(df_conv)):
            _id = df_conv["Id"].values[i]
            #
            # instantiate new LULC map
            map_lulc = LULC(name="Conversion", date=s_date_end)
            map_lulc.set_grid(grid=self.collection[s_name_end].grid)
            map_lulc.set_asc_metadata(
                metadata=self.collection[s_name_start].asc_metadata
            )
            map_lulc.set_table(dataframe=self.collection[s_name_start].table)
            map_lulc.prj = self.collection[s_name_start].prj
            #
            # apply aoi
            grd_aoi = 1 * (self.collection[s_name_start].grid == _id)
            map_lulc.apply_aoi_mask(grid_aoi=grd_aoi, inplace=True)
            #
            # bypass all-masked aois
            if np.sum(map_lulc.grid) is np.ma.masked:
                grd_conv[i] = np.zeros(len(df_conv))
            else:
                df_areas = map_lulc.get_areas()
                grd_conv[i] = df_areas["{}_f".format(map_lulc.areafield)].values

        # append to dataframe
        grd_conv = grd_conv.transpose()
        for i in range(len(df_conv)):
            df_conv[lst_cols[i]] = grd_conv[i]

        # get expansion matrix
        grd_exp = np.zeros(shape=grd_conv.shape)
        for i in range(len(grd_exp)):
            grd_exp[i] = df_conv["Area_f_start"].values * df_conv[lst_cols[i]].values
        np.fill_diagonal(grd_exp, 0)

        # get retraction matrix
        grd_rec = np.zeros(shape=grd_conv.shape)
        for i in range(len(grd_rec)):
            grd_rec[i] = df_conv["Area_f_start"].values[i] * grd_conv.transpose()[i]
        np.fill_diagonal(grd_rec, 0)

        return {
            "Dataframe": df_conv,
            "Conversion_Matrix": grd_conv,
            "Conversion_index": np.prod(np.diagonal(grd_conv)),
            "Expansion_Matrix": grd_exp,
            "Retraction_Matrix": grd_rec,
            "Date_start": date_start,
            "Date_end": date_end,
        }


if __name__ == "__main__":
    print("Hi!")
