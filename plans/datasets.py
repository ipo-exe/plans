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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# -----------------------------------------
# Series data structures


class DailySeries:
    """
    The basic daily time series object

    """

    def __init__(self, metadata, varfield, datefield="Date"):
        """
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
        """
        # -------------------------------------
        # set basic attributes
        self.data = None  # start with no data
        self.metadata = metadata
        self.varfield = varfield
        self.datefield = datefield
        self.name = self.metadata["Name"]

    def __str__(self):
        s_aux = "\n{}\n{}\n".format(self.metadata, self.data)
        return s_aux

    def set_data(self, dataframe):
        """
        Set the data from incoming pandas DataFrame.
        Note: must include varfield and datefield
        :param dataframe: incoming pandas DataFrame
        :type dataframe: :class:`pandas.DataFrame`
        """
        # slice only interest fields
        self.data = dataframe[[self.datefield, self.varfield]]
        # ensure datetime format
        self.data[self.datefield] = pd.to_datetime(self.data[self.datefield])

    def load_data(self, file):
        """
        Load data from CSV file.

        CSV file must:
        * be a txt file
        * use ; as field separator
        * include datefields and varfields

        :param file: path to CSV file
        :type file: str
        """
        self.file = file
        # -------------------------------------
        # import data
        df_aux = pd.read_csv(self.file, sep=";", parse_dates=[self.datefield])
        # slice only varfield and datefield from dataframe
        self.data = df_aux[[self.datefield, self.varfield]]

    def export_data(self, folder):
        """
        Export dataset to CSV file
        :param folder: path to output directory
        :type folder: str
        :return: file path
        :rtype: str
        """
        s_filepath = "{}/{}_{}.txt".format(
            folder, self.metadata["Variable"], self.metadata["Name"]
        )
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
        df_aux = df_aux.resample(period).sum()[self.varfield]
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
        df_aux = df_aux.resample(period).mean()[self.varfield]
        df_aux = df_aux.reset_index()
        return df_aux

    def plot_basic_view(
        self, show=False, folder="C:/data", filename=None, specs=None, dpi=96
    ):
        """
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
        """

        from analyst import Univar

        plt.style.use("seaborn-v0_8")

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
            4, 5, wspace=0.5, hspace=0.9, left=0.05, bottom=0.1, top=0.9, right=0.95
        )
        fig.suptitle(specs["suptitle"])

        # plot Series
        plt.subplot(gs[0:3, :3])
        plt.title("a. {}".format(specs["a_title"]), loc="left")
        plt.plot(
            self.data[self.datefield],
            self.data[self.varfield],
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
                self.data[self.varfield]
                .rolling(specs["mavg period"], min_periods=2)
                .mean(),
                label=specs["a_mavg_label"],
                color=specs["mavg color"],
            )
        plt.ylim(specs["ylim"])
        plt.xlim(
            self.data[self.datefield].values[0], self.data[self.datefield].values[-1]
        )
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["a_xlabel"])
        plt.legend(frameon=True, loc=(0.0, -0.3), ncol=1)

        # plot Hist
        plt.subplot(gs[0:3, 3:4])
        plt.title("b. {}".format(specs["b_title"]), loc="left")
        plt.hist(
            x=self.data[self.varfield],
            bins=specs["nbins"],
            orientation="horizontal",
            color=specs["color"],
        )
        plt.ylim(specs["ylim"])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["b_xlabel"])

        # plot CFC
        df_freq = uni.assessment_frequency()
        plt.subplot(gs[0:3, 4:5])
        plt.title("c. {}".format(specs["c_title"]), loc="left")
        plt.plot(df_freq["Exceedance"], df_freq["Values"])
        plt.ylim(specs["ylim"])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["c_xlabel"])
        plt.xlim(0, 100)

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = self.name
            plt.savefig("{}/{}.png".format(folder, filename), dpi=96)


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


class StreamflowSeries(DailySeries):
    pass

# -----------------------------------------
# Raster data structures
# todo __str__() method

class RasterMap:
    """
    The basic raster map dataset.
    """

    def __init__(self, name="myRasterMap", dtype="float32"):
        """
        Deploy basic raster map object
        :param name: map name
        :type name: str
        :param dtype: data type of raster cells - options: byte, uint8, int16, int32, float32, etc
        :type dtype: str
        """
        # -------------------------------------
        # set basic attributes
        self.grid = None  # start with no data
        self.asc_metadata = None
        self.name = name
        self.dtype = dtype
        self.cmap = "jet"
        self.varname = "Unknown variable"
        self.varalias = "Var"
        self.description = "Unknown"
        self.units = "units"

    def set_grid(self, grid):
        """
        Set data from incoming objects
        :param grid: data grid
        :type grid: :class:`numpy.ndarray`
        """
        self.grid = grid.astype(self.dtype)

    def set_metadata(self, metadata):
        """
        Set metadata from incoming objects

        Example of metadata for ASC raster:
        meta = {
            'ncols': 366,
            'nrows': 434,
            'xllcorner': 559493.08,
            'yllcorner': 6704832.2,
            'cellsize': 30,
            'NODATA_value': -1
        }

        :param metadata: metadata dictionary
        :type metadata: dict
        """

        self.asc_metadata = metadata

    def load_asc_raster(self, file, nan=False):
        """
        A function to load data and metadata from .ASC raster files
        :param file: string of file path with the '.asc' extension
        :type file: str
        :param nan: boolean to convert nan values to np.nan
        :type nan: bool
        """
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

        self.asc_metadata = dct_meta
        self.grid = grd_data

    def load_asc_metadata(self, file):
        """
        A function to load only metadata from .ASC raster files
        :param file: string of file path with the '.asc' extension
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
        self.asc_metadata = meta_dct

    def export_asc_raster(self, folder, filename=None):
        """
        Function for exporting an .ASC raster file.
        :param folder: string of directory path
        :type folder: str
        :param filename: string of file without extension
        :type filename: str
        :return: full file name (path and extension) string
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
            #
            # data constructor loop:
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
            return flenm

    def plot_basic_view(
        self, show=False, folder="C:/data", filename=None, specs=None, dpi=96
    ):
        from analyst import Univar

        plt.style.use("seaborn-v0_8")

        # get univar object
        uni = Univar(data=self.grid.flatten())

        # get specs
        default_specs = {
            "color": "tab:grey",
            "cmap": self.cmap,
            "suptitle": "{} | {}".format(self.varname, self.name),
            "a_title": "{} ({})".format(self.varname, self.units),
            "b_title": "Histogram",
            "c_title": "Metadata",
            "width": 5 * 1.618,
            "height": 5,
            "b_ylabel": "frequency",
            "b_xlabel": self.units,
            "nbins": uni.nbins_fd(),
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
            specs["vmin"] = np.min(self.grid)
        if specs["vmax"] is None:
            specs["vmax"] = np.max(self.grid)

        # Deploy figure
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        gs = mpl.gridspec.GridSpec(
            4, 5, wspace=0.8, hspace=0.1, left=0.01, bottom=0.1, top=0.85, right=0.95
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
            x=self.grid.flatten(),
            bins=specs["nbins"],
            color=specs["color"],
            # orientation="horizontal"
        )
        n_mean = np.mean(self.grid.flatten())
        if specs["hist_vmax"] is None:
            specs["hist_vmax"] = 1.2 * np.max(vct_result[0])
        plt.vlines(
            x=n_mean,
            ymin=0,
            ymax=specs["hist_vmax"],
            colors="tab:red",
            linestyles="--",
            # label="mean ({:.2f})".format(n_mean)
        )

        plt.ylim(0, specs["hist_vmax"])
        plt.xlim(specs["vmin"], specs["vmax"])
        plt.ylabel(specs["b_ylabel"])
        plt.xlabel(specs["b_xlabel"])
        plt.text(
            x=n_mean + 2 * (specs["vmax"] - specs["vmin"]) / 100,
            y=0.9 * specs["hist_vmax"],
            s="{:.2f} (mean)".format(n_mean),
        )

        # plot metadata
        n_y = 0.25
        plt.text(
            x=0.63,
            y=n_y,
            s="c. {}".format(specs["c_title"]),
            fontsize=12,
            transform=fig.transFigure,
        )
        n_y = n_y - 0.01
        n_step = 0.03
        for k in self.asc_metadata:
            s_head = k
            s_value = self.asc_metadata[k]
            s_line = s_head + ": " + str(s_value)
            n_y = n_y - n_step
            plt.text(x=0.65, y=n_y, s=s_line, fontsize=10, transform=fig.transFigure)

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "{}_{}".format(self.varalias, self.name)
            plt.savefig("{}/{}.png".format(folder, filename), dpi=96)


# -----------------------------------------
# Quanti raster data structures


class ElevationMap(RasterMap):
    """
    Elevation (DEM) raster map dataset.
    """

    def __init__(self, name="DemMap"):
        """
        Deploy dataset
        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="uint16")
        self.cmap = "gist_earth"
        self.varname = "Elevation"
        self.varalias = "ELV"
        self.description = "Height above sea level"
        self.units = "m"


class SlopeMap(RasterMap):
    """
    Slope raster map dataset.
    """

    def __init__(self, name="SlopeMap"):
        """
        Deploy dataset
        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="float32")
        self.cmap = "OrRd"
        self.varname = "Slope"
        self.varalias = "SLP"
        self.description = "Slope of terrain"
        self.units = "deg."


# -----------------------------------------
# Quali Raster data structures

class QualiRasterMap(RasterMap):
    """
    Basic qualitative raster map dataset.

    Attributes dataframe must at least have:
    * :class:`Id` field
    * :class:`Name` field
    * :class:`Alias` field

    """

    def __init__(self, name="QualiMap"):
        """
        Deploy dataset
        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="uint8")
        self.cmap = "tab20b"
        self.varname = "Unknown variable"
        self.varalias = "Var"
        self.description = "Unknown"
        self.units = "category ID"
        self.attributes = None
        self.nodatavalue = 0

    def load_asc_raster(self, file, nan=False):
        super().load_asc_raster(file, nan)
        # overwrite nodata
        self.asc_metadata["NODATA_value"] = self.nodatavalue

    def _dataframe_prepro(self, dataframe):
        """
        Utility function for dataframe preprossing
        :param dataframe: incoming dataframe
        :type dataframe: :class:`pandas.DataFrame`
        :return: prepared dataframe
        :rtype: :class:`pandas.DataFrame`
        """
        # fix headings
        dataframe.columns = dataframe.columns.str.strip()
        # strip string fields
        for i in range(len(dataframe.columns)):
            if str(dataframe.dtypes.iloc[i]) == "object":
                dataframe[dataframe.columns[i]] = dataframe[dataframe.columns[i]].str.strip()
        return dataframe

    def load_attributes(self, file):
        """
        Load attributes dataframe from CSV txt file (separator must be ;)
        :param file: path to file
        :type file: str
        """
        # read raw file
        df_aux = pd.read_csv(file, sep=";")
        # set to self
        self.attributes = self._dataframe_prepro(dataframe=df_aux)

    def set_attributes(self, dataframe):
        """
        Set attributes dataframe from incoming pandas dataframe
        :param dataframe: incoming pandas dataframe
        :type dataframe: :class:`pandas.DataFrame`
        """
        self.attributes = self._dataframe_prepro(dataframe=dataframe.copy())



if __name__ == "__main__":

    sfile = "C:/data/lulc.asc"
    rst_lulc = QualiRasterMap(name="Andreas")
    rst_lulc.varname = "LULC"
    rst_lulc.load_asc_raster(file=sfile)
    rst_lulc.plot_basic_view(show=True)
    rst_lulc.load_attributes(file="C:/data/lulc.txt")
    print(rst_lulc.attributes)
    rst_lulc.set_attributes(dataframe=rst_lulc.attributes[["Id", "Name", "Alias"]])
    print(rst_lulc.attributes)
    print(rst_lulc.asc_metadata)


    '''
    sfile = "C:/data/slope.asc"
    rst_slp = SlopeMap(name="Andreas")
    rst_slp.load_asc_raster(file=sfile)
    rst_slp.plot_basic_view(show=True)

    sfile = "C:/data/dem.asc"
    rst_dem = ElevationMap(name="Andreas")
    rst_dem.load_asc_raster(file=sfile)
    specs = {"vmin": None, "vmax": None, "hist_vmax": None}
    rst_dem.plot_basic_view(show=True, specs=specs)
    '''



