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
        # todo fix histogram as %
        import matplotlib.ticker as mtick
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
            "ylim": (0, 0.25),
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
            weights=np.ones(len(self.data)) / len(self.data),
        )
        plt.ylim(specs["ylim"])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["b_xlabel"])

        # Set the x-axis formatter as percentages
        xticks = mtick.PercentFormatter(xmax=1, decimals=1, symbol="%", is_latex=False)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(xticks)

        # plot CFC
        df_freq = uni.assess_frequency()
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
# Base raster data structures
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
        self.asc_metadata = {
            "ncols": None,
            "nrows": None,
            "xllcorner": 0.0,
            "yllcorner": 0.0,
            "cellsize": 30.0,
            "NODATA_value": -1,
        }
        self.nodatavalue = self.asc_metadata["NODATA_value"]
        self.cellsize = self.guess_cellsize()
        self.name = name
        self.dtype = dtype
        self.cmap = "jet"
        self.varname = "Unknown variable"
        self.varalias = "Var"
        self.description = "Unknown"
        self.units = "units"
        self.date = "2020-01-01"

    def set_grid(self, grid):
        """
        Set data from incoming objects
        :param grid: data grid
        :type grid: :class:`numpy.ndarray`
        """
        # overwrite incoming dtype
        self.grid = grid.astype(self.dtype)
        # mask nodata values
        self.mask_nodata()

    def set_asc_metadata(self, metadata):
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
        for k in self.asc_metadata:
            if k in metadata:
                self.asc_metadata[k] = metadata[k]
        # update nodata value
        self.nodatavalue = self.asc_metadata["NODATA_value"]
        self.guess_cellsize()

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

        self.set_asc_metadata(metadata=dct_meta)
        self.set_grid(grid=grd_data)

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
        self.set_asc_metadata(metadata=meta_dct)

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

    def mask_nodata(self):
        """
        Mask grid cells as NaN where data is NODATA
        """
        if self.nodatavalue is None:
            pass
        else:
            if self.grid.dtype.kind in ["i", "u"]:
                # for integer grid
                self.grid = np.ma.masked_where(self.grid == self.nodatavalue, self.grid)
            else:
                # for floating point grid:
                self.grid[self.grid == self.nodatavalue] = np.nan

    def insert_nodata(self):
        """
        Insert grid cells as NODATA where data is NaN
        """
        if self.nodatavalue is None:
            pass
        else:
            if self.grid.dtype.kind in ["i", "u"]:
                # for integer grid
                self.grid = np.ma.filled(self.grid, fill_value=self.nodatavalue)
            else:
                # for floating point grid:
                self.grid = np.nan_to_num(self.grid, nan=self.nodatavalue)

    def apply_aoi_mask(self, grid_aoi, inplace=False):
        """
        Apply AOI (area of interest) mask to raster map
        :param grid_aoi:
        map of AOI (masked array or pseudo-boolean).
        Must have the same size of grid
        :type grid_aoi: :class:`numpy.ndarray`
        :param inplace: set on the own grid if True
        :type inplace: bool
        :return:
        :rtype:
        """
        if self.nodatavalue is None:
            pass
        # ensure fill
        grid_aoi = np.ma.filled(grid_aoi, fill_value=0)
        # replace
        grd_mask = np.where(grid_aoi == 0, self.nodatavalue, self.grid)

        if inplace:
            self.grid = grd_mask
            # mask
            self.mask_nodata()
            return None
        else:
            return grd_mask

    def get_data(self):
        """
        Get flat and cleared data
        :return: 1d vector of data
        :rtype: :class:`numpy.ndarray`
        """
        return self.grid.ravel()[~np.isnan(self.grid.ravel())]

    def get_basic_stats(self):
        """
        Get basic statistics from flat and clear data
        :return: dataframe of basic statistics
        :rtype: :class:`pandas.DataFrame`
        """
        from analyst import Univar

        return Univar(data=self.get_data()).assess_basic_stats()

    def guess_cellsize(self):
        """
        Guess the cellsize in meters if is degrees
        :return:
        :rtype:
        """
        if self.asc_metadata["cellsize"] < 1:
            self.cellsize = self.asc_metadata["cellsize"] * 111 * 1000
        else:
            self.cellsize = self.asc_metadata["cellsize"]
        return self.cellsize

    def plot_basic_view(
        self, show=False, folder="C:/data", filename=None, specs=None, dpi=96
    ):
        """
        Plot basic view of raster map
        :param show: boolean to show plot instead of saving
        :type show: bool
        :param folder: path to output folder
        :type folder: str
        :param filename: name of file
        :type filename: str
        :param specs: specifications dictionary
        :type specs: dict
        :param dpi: image resolution
        :type dpi: int
        """
        import matplotlib.ticker as mtick
        from analyst import Univar

        plt.style.use("seaborn-v0_8")

        # get univar object
        uni = Univar(data=self.get_data())

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
            "b_ylabel": "percentage",
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
            specs["vmin"] = np.min(uni.data)
        if specs["vmax"] is None:
            specs["vmax"] = np.max(uni.data)

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
            colors="tab:red",
            linestyles="--",
            # label="mean ({:.2f})".format(n_mean)
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
# Quali Raster data structures


class QualiRasterMap(RasterMap):
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
        self.aliasfield = "Name"
        self.colorfield = "Color"
        self.areafield = "Area"
        self._overwrite_nodata()

    def _overwrite_nodata(self):
        self.nodatavalue = 0
        self.asc_metadata["NODATA_value"] = self.nodatavalue

    def set_asc_metadata(self, metadata):
        super().set_asc_metadata(metadata)
        self._overwrite_nodata()

    def mask_nodata(self):
        """
        Mask grid cells where data is NODATA
        """
        if self.nodatavalue is None:
            pass
        else:
            self.grid = np.ma.masked_where(self.grid == self.nodatavalue, self.grid)

    def insert_nodata(self):
        """
        Insert grid cells as NODATA where data is maked
        """
        if self.nodatavalue is None:
            pass
        else:
            self.grid = np.ma.filled(self.grid, fill_value=self.nodatavalue)

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
                dataframe[dataframe.columns[i]] = dataframe[
                    dataframe.columns[i]
                ].str.strip()
        return dataframe

    def load_table(self, file):
        """
        Load attributes dataframe from CSV txt file (separator must be ;)
        :param file: path to file
        :type file: str
        """
        # read raw file
        df_aux = pd.read_csv(file, sep=";")
        # set to self
        self.table = self._dataframe_prepro(dataframe=df_aux)

    def export_table(self, folder="C:/data", filename=None):
        """
        Export an CSV .txt  file.
        :param folder: string of directory path
        :type folder: str
        :param filename: string of file without extension
        :type filename: str
        :return: full file name (path and extension) string
        :rtype: str
        """
        if filename is None:
            filename = self.name
        flenm = folder + "/" + filename + ".txt"
        self.table.to_csv(flenm, sep=";", index=False)
        return flenm

    def set_table(self, dataframe):
        """
        Set attributes dataframe from incoming pandas dataframe
        :param dataframe: incoming pandas dataframe
        :type dataframe: :class:`pandas.DataFrame`
        """
        self.table = self._dataframe_prepro(dataframe=dataframe.copy())

    def set_random_colors(self):
        """
        Set random colors to attribute table
        """
        if self.table is None:
            pass
        else:
            import matplotlib.colors as mcolors

            # Choose a colormap from matplotlib
            _cmap = plt.get_cmap(self.cmap)
            # Generate a list of random numbers between 0 and 1
            _lst_rand_vals = np.random.rand(len(self.table))
            # Use the colormap to convert the random numbers to colors
            self.table[self.colorfield] = [
                mcolors.to_hex(_cmap(x)) for x in _lst_rand_vals
            ]

    def get_areas(self, merge=False):
        """
        Get areas in map of each category in table
        """
        if self.table is None or self.grid is None:
            pass
        else:
            # get unit area
            _n_unit_area = np.square(self.cellsize)
            df_aux = self.table[["Id", "Name", "Alias"]].copy()
            _lst_areas = []
            # iterate categories
            for i in range(len(df_aux)):
                _n_id = df_aux[self.idfield].values[i]
                _n_value = np.sum(1 * (self.grid == _n_id)) * _n_unit_area
                _lst_areas.append(_n_value)
            # set area fields
            s_field = "{}_m2".format(self.areafield)
            df_aux[s_field] = _lst_areas
            df_aux["{}_ha".format(self.areafield)] = df_aux[s_field] / (100 * 100)
            df_aux["{}_km2".format(self.areafield)] = df_aux[s_field] / (1000 * 1000)
            df_aux["{}_f".format(self.areafield)] = (
                df_aux[s_field] / df_aux[s_field].sum()
            )
            df_aux["{}_%".format(self.areafield)] = (
                100 * df_aux[s_field] / df_aux[s_field].sum()
            )
            df_aux["{}_%".format(self.areafield)] = df_aux[
                "{}_%".format(self.areafield)
            ].round(2)

            # handle merge
            if merge:
                for k in lst_area_fields:
                    self.table[k] = df_aux[k].values

            return df_aux

    def get_zonal_stats(self, raster_sample, merge=False, skip_count=False):
        """
        get zonal stats from other raster map to sample
        :param raster_sample: raster map to sample
        :type raster_sample: :class:`datasets.RasterMap`
        :param merge: set True to merge into the map table
        :type merge: bool
        :param skip_count: set True to skip count
        :type skip_count: bool
        :return: dataframe of zonal stats
        :rtype: :class:`pandas.DataFrame`
        """
        from analyst import Univar

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
            raster_sample.apply_aoi_mask(grid_aoi=grid_aoi)
            # get basic stats
            raster_uni = Univar(data=raster_sample.get_data(), name=varname)
            df_stats = raster_uni.assess_basic_stats()
            lst_stats.append(df_stats.copy())
            # restore
            raster_sample.grid = grid_raster

        # create empty fields
        lst_stats_field = []
        for k in df_stats["Stats"]:
            s_field = "{}_{}".format(varname, k)
            lst_stats_field.append(s_field)
            df_aux[s_field] = 0.0

        # fill values
        for i in range(len(df_aux)):
            df_aux.loc[i, lst_stats_field[0] : lst_stats_field[-1]] = lst_stats[i][
                "Data"
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

    def plot_basic_view(
        self, show=False, folder="C:/data", filename=None, specs=None, dpi=96
    ):
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        plt.style.use("seaborn-v0_8")

        if self.colorfield in self.table.columns:
            pass
        else:
            self.set_random_colors()

        # get specs
        default_specs = {
            "color": "tab:grey",
            "cmap": ListedColormap(self.table[self.colorfield]),
            "suptitle": "{} ({}) | {}".format(self.varname, self.varalias, self.name),
            "a_title": "{} Map ({})".format(self.varalias, self.units),
            "b_title": "{} Prevalence".format(self.varalias),
            "c_title": "Metadata",
            "width": 5 * 1.618,
            "height": 5,
            "b_area": "km2",
            "b_xlabel": "Area",
            "vmin": self.table[self.idfield].min(),
            "vmax": self.table[self.idfield].max(),
            "hist_vmax": None,
            "gs_rows": 7,
            "gs_cols": 5,
            "gs_b_rowlim": 4,
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
            specs["gs_rows"],
            specs["gs_cols"],
            wspace=0.8,
            hspace=0.05,
            left=0.01,
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
        for i in range(len(self.table)):
            legend_elements.append(
                Patch(
                    facecolor=self.table[self.colorfield].values[i],
                    label=self.table[self.namefield].values[i],
                )
            )
        plt.legend(
            frameon=True,
            fontsize=9,
            markerscale=0.8,
            handles=legend_elements,
            bbox_to_anchor=(0.5, 0.3),
            bbox_transform=fig.transFigure,
            ncol=3,
        )

        # plot hbar
        # ensure areas are computed
        df_aux = pd.merge(
            self.table[["Id", "Color"]], self.get_areas(), how="left", on="Id"
        )
        df_aux = df_aux.sort_values(by="{}_m2".format(self.areafield), ascending=True)
        plt.subplot(gs[: default_specs["gs_b_rowlim"], 3:])
        plt.title("b. {}".format(specs["b_title"]), loc="left")
        plt.barh(
            df_aux[self.namefield],
            df_aux["{}_{}".format(self.areafield, specs["b_area"])],
            color=df_aux[self.colorfield],
        )

        # Add labels for each bar
        _n_max = df_aux["{}_{}".format(self.areafield, specs["b_area"])].max()
        for i in range(len(df_aux)):
            v = df_aux["{}_{}".format(self.areafield, specs["b_area"])].values[i]
            p = df_aux["{}_%".format(self.areafield)].values[i]
            plt.text(
                v + _n_max / 50, i - 0.3, "{:.1f} ({:.1f}%)".format(v, p), fontsize=9
            )
        plt.xlim(0, 1.5 * _n_max)
        plt.xlabel("{} (km$^2$)".format(specs["b_xlabel"]))
        plt.grid(axis="y")

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
# Derived data structures


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
        super().__init__(name=name, dtype="float32")
        self.cmap = "BrBG_r"
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


class TWIMap(RasterMap):
    """
    TWI raster map dataset.
    """

    def __init__(self, name="TWIMap"):
        """
        Deploy dataset
        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="float32")
        self.cmap = "YlGnBu"
        self.varname = "TWI"
        self.varalias = "TWI"
        self.description = "Topographical Wetness Index"
        self.units = "index units"


class HANDMap(RasterMap):
    """
    HAND raster map dataset.
    """

    def __init__(self, name="HANDMap"):
        """
        Deploy dataset
        :param name: name of map
        :type name: str
        """
        super().__init__(name=name, dtype="float32")
        self.cmap = "YlGnBu_r"
        self.varname = "HAND"
        self.varalias = "HAND"
        self.description = "Height Above the Nearest Drainage"
        self.units = "m"


class LULCMap(QualiRasterMap):
    """
    Land Use and Land Cover map dataset
    """

    def __init__(self, name="LULCMap"):
        super().__init__(name, dtype="uint8")
        self.cmap = "tab20b"
        self.varname = "Land Use and Land Cover"
        self.varalias = "LULC"
        self.description = "Classes of Land Use and Land Cover"
        self.units = "classes ID"


class SoilsMap(QualiRasterMap):
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


class AOIMap(QualiRasterMap):
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
            {"Id": [1], "Name": [self.name], "Alias": ["-"], "Color": "tab:grey"}
        )


if __name__ == "__main__":
    b_aoi = False
    b_lulc = False
    b_dem = True
    b_slope = True

    output_dir = "C:/data"
    input_dir = "C:/data/gravatai/plans"

    s_name = "Gravatai"
    s_aux = "gravatai"
    if b_aoi:
        # -------------------------------------------------------------
        # [0] AOI map
        s_filename = "{}_basin_flu87398800".format(s_aux)
        # instantiate map
        rst_aoi = AOIMap(name="Andreas")
        # load file
        rst_aoi.load_asc_raster(file="{}/{}.asc".format(input_dir, s_filename))
        # view
        print(rst_aoi.table.to_string())
        rst_aoi.plot_basic_view(show=True)

    if b_lulc:
        # -------------------------------------------------------------
        # [1] LULC map
        s_filename = "{}_lulc_2020".format(s_aux)
        # instantiate map
        rst_lulc = LULCMap(name=s_name)
        # load files
        rst_lulc.load_asc_raster(file="{}/{}.asc".format(input_dir, s_filename))
        rst_lulc.load_table(file="{}/lulc.txt".format(input_dir))

        # -------------------------------------------------------------
        # [0] AOI map
        s_filename = "{}_basin_flu87398800".format(s_aux)
        # instantiate map
        rst_aoi = AOIMap(name=s_name)
        # load file
        rst_aoi.load_asc_raster(file="{}/{}.asc".format(input_dir, s_filename))

        # apply aoi inplace
        rst_lulc.apply_aoi_mask(grid_aoi=rst_aoi.grid, inplace=True)

        # rst_lulc.get_areas()
        # plot
        rst_lulc.plot_basic_view(show=True)
        # print(rst_lulc.table.to_string())

    if b_slope:
        # -------------------------------------------------------------
        # [2] Slope map
        s_filename = "{}_hand".format(s_aux)
        # instantiate map
        rst_slp = HANDMap(name=s_name)
        # load files
        rst_slp.load_asc_raster(file="{}/{}.asc".format(input_dir, s_filename))
        # apply aoi
        # rst_slp.apply_aoi_mask(grid_aoi=rst_aoi.grid)
        # plot
        rst_slp.plot_basic_view(show=True, specs={"vmin": 0, "vmax": 5})

    if b_dem:
        # -------------------------------------------------------------
        # [3] Elevation map
        s_filename = "{}_demh".format(s_aux)
        # instantiate map
        rst_dem = ElevationMap(name=s_name)
        # load files
        rst_dem.load_asc_raster(file="{}/{}.asc".format(input_dir, s_filename))
        # plot
        rst_dem.plot_basic_view(
            show=True,
        )
