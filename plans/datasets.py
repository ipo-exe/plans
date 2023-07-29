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
        if str(dataframe.dtypes.iloc[i]) == "object":
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


class DailySeries:
    """
    The basic daily time series object

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

        :param dataframe: incoming :class:`pandas.DataFrame` object
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
        # ensure datetime format
        self.data[self.datefield] = pd.to_datetime(self.data[self.datefield])
        self.data = self.data.sort_values(by=self.datefield).reset_index(drop=True)

    def load_data(self, file, varfield, datefield="Date"):
        """Load data from ``csv`` file.

        :param file: path to ``csv`` file
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
        :param folder: path to output directory
        :type folder: str
        :return: file path
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
        show=False,
        folder="C:/data",
        filename=None,
        specs=None,
        dpi=300,
        format="jpg",
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
        :param format: image format (ex: png or jpg). Default jpg
        :type format: str
        """
        import matplotlib.ticker as mtick
        from analyst import Univar

        plt.style.use("seaborn-v0_8")

        # get univar object
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
            plt.savefig("{}/{}.{}".format(folder, filename, format), dpi=dpi)
        plt.close(fig)
        return None


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


class Raster:
    """
    The basic raster map dataset.
    """

    def __init__(self, name="myRasterMap", dtype="float32"):
        """Deploy basic raster map object.

        :param name: map name, defaults to myRasterMap
        :type name: str
        :param dtype: data type of raster cells - options: byte, uint8, int16, int32, float32, etc, defaults to float32
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
        self.cellsize = self.asc_metadata["cellsize"]
        self.name = name
        self.dtype = dtype
        self.cmap = "jet"
        self.varname = "Unknown variable"
        self.varalias = "Var"
        self.description = None
        self.units = "units"
        self.date = None  # "2020-01-01"
        self.prj = None

    def __str__(self):
        return "Hello world!"

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

    def load_asc_raster(self, file, nan=False):
        """A function to load data and metadata from ``.asc`` raster files.

        :param file: string of file path with the ``.asc`` extension
        :type file: str
        :param nan: boolean to convert nan values to np.nan, defaults to False
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
        return None

    def load_asc_metadata(self, file):
        """A function to load only metadata from ``.asc`` raster files.

        :param file: string of file path with the ``.asc`` extension
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

        :param file: string of file path with the ``.prj`` extension
        :type file: str
        :return: None
        :rtype: None
        """
        with open(file) as f:
            self.prj = f.readline().strip("\n")
        return None

    def export_asc_raster(self, folder="./output", filename=None):
        """Function for exporting an ``.asc`` raster file.

        :param folder: string of directory path, , defaults to ``./output``
        :type folder: str
        :param filename: string of file without extension, defaults to None
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

    def export_prj_file(self, folder="./output", filename=None):
        """Function for exporting an ``.prj`` file.

        :param folder: string of directory path, , defaults to ``./output``
        :type folder: str
        :param filename: string of file without extension, defaults to None
        :type filename: str
        :return: full file name (path and extension) string
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

    def rebase_grid(self, base_raster, inplace=False, method="linear"):
        """
        Rebase grid of raster. This function creates a new grid based on a provided raster. Both rasters
        are expected to be in the same coordinate system and having overlapping bounding boxes.
        :param base_raster: reference raster for rebase
        :type base_raster: :class:`datasets.Raster`
        :param inplace: option for rebase the own grid if True, defaults to False
        :type inplace: bool
        :param method: interpolation method - linear, nearest and cubic
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
        Expected to have the same grid.
        :type grid_aoi: :class:`numpy.ndarray`
        :param inplace: set on the own grid if True, defaults to False
        :type inplace: bool
        :return: the processed grid if inplace=False
        :rtype: :class:`numpy.ndarray` or None
        """
        if self.nodatavalue is None or self.grid is None:
            return None
        else:
            # ensure fill
            grid_aoi = np.ma.filled(grid_aoi, fill_value=0)
            # replace
            grd_mask = np.where(grid_aoi == 0, self.nodatavalue, self.grid)

            if inplace:
                self.set_grid(grid=grd_mask)
                return None
            else:
                return grd_mask

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
            from analyst import Univar

            return Univar(data=self.get_grid_data()).assess_basic_stats()

    def view(
        self,
        show=False,
        folder="./output",
        filename=None,
        specs=None,
        dpi=300,
        format="jpg",
    ):
        """Plot a basic pannel of raster map.

        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param format: image format (ex: png or jpg). Default jpg
        :type format: str
        """
        import matplotlib.ticker as mtick
        from analyst import Univar

        plt.style.use("seaborn-v0_8")

        # get univar object
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
        n_y = 0.25
        n_x = 0.62
        plt.text(
            x=n_x,
            y=n_y,
            s="d. {}".format(specs["d_title"]),
            fontsize=12,
            transform=fig.transFigure,
        )
        n_y = n_y - 0.01
        n_step = 0.025
        for i in range(len(df_stats)):
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

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "{}_{}".format(self.varalias, self.name)
            plt.savefig("{}/{}.{}".format(folder, filename, format), dpi=dpi)
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
        show=False,
        folder="./output",
        filename=None,
        specs=None,
        dpi=300,
        format="jpg",
    ):
        """
        View NDVI raster
        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param format: image format (ex: png or jpg). Default jpg
        :type format: str
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
        super().view(show, folder, filename, specs, dpi, format=format)
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
        show=False,
        folder="./output",
        filename=None,
        specs=None,
        dpi=300,
        format="jpg",
    ):
        """
        View ET raster
        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param format: image format (ex: png or jpg). Default jpg
        :type format: str
        :return: None
        :rtype: None
        """
        default_specs = {"vmin": 0, "vmax": 10}
        if specs is None:
            specs = default_specs
        else:
            for k in default_specs:
                specs[k] = default_specs[k]
        super().view(show, folder, filename, specs, dpi, format=format)
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

    def load_table(self, file):
        """Load attributes dataframe from ``csv`` ``.txt`` file (separator must be ;).

        :param file: path to file
        :type file: str
        """
        # read raw file
        df_aux = pd.read_csv(file, sep=";")
        # set to self
        self.set_table(dataframe=df_aux)
        return None

    def export_table(self, folder="./output", filename=None):
        """Export an ``csv`` ``.txt``  file.

        :param folder: string of directory path, defaults to ``./output``
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
        show=False,
        folder="./output",
        filename=None,
        specs=None,
        dpi=300,
        format="jpg",
    ):
        """Plot a basic pannel of qualitative raster map.

        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param format: image format (ex: png or jpg). Default jpg
        :type format: str
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
            "height": 7,
            "b_area": "km2",
            "b_xlabel": "Area",
            "b_xmax": None,
            "bars_alias": True,
            "vmin": 0,
            "vmax": self.table[self.idfield].max(),
            "gs_rows": 7,
            "gs_cols": 5,
            "gs_b_rowlim": 4,
            "legend_x": 0.6,
            "legend_y": 0.3,
            "legend_ncol": 2,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

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
        for i in range(len(self.table)):
            _color = self.table[self.colorfield].values[i]
            _label = "{} ({})".format(
                self.table[self.namefield].values[i],
                self.table[self.aliasfield].values[i],
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

        # ensure areas are computed
        df_aux = pd.merge(
            self.table[["Id", "Color"]], self.get_areas(), how="left", on="Id"
        )
        df_aux = df_aux.sort_values(by="{}_m2".format(self.areafield), ascending=True)
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
            plt.savefig("{}/{}.{}".format(folder, filename, format), dpi=dpi)
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
        self.varname = "Land Use and Land Cover Change"
        self.varalias = "LULCC"
        self.description = "Change of Land Use and Land Cover"
        self.units = "Change ID"
        self.date_start = date_start
        self.date_end = date_end
        self.date = date_end
        self.table = pd.DataFrame(
            {
                self.idfield: [1, 2, 3,],
                self.namefield: ["Retraction", "Stable", "Expansion"],
                self.aliasfield: ["Rtr", "Stb", "Exp"],
                self.colorfield: ["tab:purple", "tab:orange", "tab:red"]
            }
        )



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
            {"Id": [1], "Name": [self.name], "Alias": ["AOI"], "Color": "tab:grey"}
        )


# -----------------------------------------
# Raster Collection data structures


class RasterCollection:
    """
    The raster collection base dataset.

    This data strucute is designed for holding and comparing :class:`Raster` objects.

    """

    def __init__(self, name="myRasterCollection", dtype="float32"):
        """Deploy the raster collection data structure.

        :param name: name of raster collection
        :type name: str
        :param dtype: data type of raster cells, defaults to float32
        :type dtype: str
        """
        self.catalog = pd.DataFrame(
            columns=[
                "Name",
                "Variable",
                "VarAlias",
                "Units",
                "Date",
                "cellsize",
                "ncols",
                "rows",
                "xllcorner",
                "yllcorner",
                "NODATA_value",
                "Prj",
            ]
        )
        self.catalog["Date"] = pd.to_datetime(self.catalog["Date"])
        self.collection = dict()
        self.dtype = dtype
        self.name = name

    def append_raster(self, raster):
        """Append a :class:`Raster` object to collection.
        Pre-existing objects with the same :class:`Raster.name` attribute are replaced

        :param raster: incoming :class:`Raster` to append
        :type raster: :class:`Raster`
        """
        # append to collection
        self.collection[raster.name] = raster
        # set
        df_aux = pd.DataFrame(
            {
                "Name": [raster.name],
                "Variable": [raster.varname],
                "VarAlias": [raster.varalias],
                "Units": [raster.units],
                "Date": [raster.date],
                "cellsize": [raster.cellsize],
                "ncols": [raster.asc_metadata["ncols"]],
                "rows": [raster.asc_metadata["nrows"]],
                "xllcorner": [raster.asc_metadata["xllcorner"]],
                "yllcorner": [raster.asc_metadata["yllcorner"]],
                "NODATA_value": [raster.nodatavalue],
                "Prj": [raster.prj],
            }
        )
        self.catalog = pd.concat([self.catalog, df_aux], ignore_index=True)
        self.update_catalog()
        return None

    def remove_raster(self, name):
        """Remove a :class:`Raster` object from collection.

        :param name: :class:`Raster.name` name attribute to remove
        :type name: str
        """
        # delete raster object
        del self.collection[name]
        # delete from catalog
        self.catalog = self.catalog.drop(
            self.catalog[self.catalog["Name"] == name].index
        ).reset_index(drop=True)
        return None

    def update_catalog(self, details=False):
        """
        Update the collection catalog
        :param details: option to update catalog details (looking into rasters)
        :type details: bool
        :return: None
        :rtype: none
        """
        # update details
        if details:
            # create new catalog
            df_new_catalog = pd.DataFrame(columns=self.catalog.columns)
            df_new_catalog["Date"] = pd.to_datetime(df_new_catalog["Date"])
            for name in self.collection:
                # set new information
                df_aux = pd.DataFrame(
                    {
                        "Name": [self.collection[name].name],
                        "Variable": [self.collection[name].varname],
                        "VarAlias": [self.collection[name].varalias],
                        "Units": [self.collection[name].units],
                        "Date": [self.collection[name].date],
                        "cellsize": [self.collection[name].cellsize],
                        "ncols": [self.collection[name].asc_metadata["ncols"]],
                        "rows": [self.collection[name].asc_metadata["nrows"]],
                        "xllcorner": [self.collection[name].asc_metadata["xllcorner"]],
                        "yllcorner": [self.collection[name].asc_metadata["yllcorner"]],
                        "NODATA_value": [self.collection[name].nodatavalue],
                        "Prj": [self.collection[name].prj],
                    }
                )
                df_new_catalog = pd.concat([df_new_catalog, df_aux], ignore_index=True)
            self.catalog = df_new_catalog.copy()
            del df_new_catalog
        # basic updates
        self.catalog = self.catalog.drop_duplicates(subset="Name", keep="last")
        self.catalog = self.catalog.sort_values(by="Name").reset_index(drop=True)
        return None

    def load_raster(
        self,
        name,
        asc_file,
        prj_file=None,
        varname=None,
        varalias=None,
        units=None,
        date=None,
    ):
        """Load a :class:`Raster` object from a ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param asc_file: path to ``.asc`` raster file
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
        rst_aux = Raster(name=name, dtype=self.dtype)
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
        self.append_raster(raster=rst_aux)
        # delete aux
        del rst_aux

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
            df_aux.loc[i, "count":"max"] = lst_stats[i]["Value"].values
        df_aux["count"] = df_aux["count"].astype(dtype="uint16")
        return df_aux

    def export_views(
        self, show=False, folder="./output", specs=None, dpi=300, format="jpg"
    ):
        """Plot all basic pannel of raster maps in collection.

        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path to output folder, defaults to ``./output``
        :type folder: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param format: image format (ex: png or jpg). Default jpg
        :type format: str
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
                format=format,
            )
        return None

    def view_bboxes(
        self,
        colors=None,
        datapoints=False,
        show=False,
        folder="./output",
        filename=None,
        dpi=300,
        format="jpg",
    ):
        """View Bounding Boxes of Raster collection

        :param colors: list of colors for plotting. expected to be the same size of catalog
        :type colors: list
        :param datapoints: option to plot datapoints as well, defaults to False
        :type datapoints: bool
        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param format: image format (ex: png or jpg). Default jpg
        :type format: str
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
            plt.savefig("{}/{}.{}".format(folder, filename, format), dpi=dpi)
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

    def load_raster(self, name, asc_file, prj_file=None, table_file=None):
        """Load a :class:`QualiRaster` object from ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param asc_file: path to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path to ``.prj`` projection file
        :type prj_file: str
        :param table_file: path to ``.txt`` table file
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
        self.append_raster(raster=rst_aux)
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
        super().__init__(name=name, dtype=dtype)
        self.varname = varname
        self.varalias = varalias
        self.units = units

    def load_folder(self, folder, name_pattern="map_*", talk=False):
        """
        Load all rasters from a folder by following a name pattern.
        Date is expected to be at the end of name before file extension.
        :param folder: path to folder
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
            self.load_raster(
                name=s_name,
                date=s_date_map,
                asc_file=asc_file,
                prj_file=prj_file,
            )
        return None

    def load_raster(self, name, date, asc_file, prj_file=None):
        """Load a :class:`Raster` object from a ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param date: :class:`Raster.date` date attribute, defaults to None
        :type date: str
        :param asc_file: path to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path to ``.prj`` projection file
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
        self.append_raster(raster=rst_aux)
        # load prj file
        if prj_file is None:
            pass
        else:
            rst_aux.load_prj_file(file=prj_file)
        # delete aux
        del rst_aux
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
        self.update_catalog(details=True)
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

    def view_series_stats(
        self,
        statistic="mean",
        folder="./output",
        filename=None,
        specs=None,
        show=False,
        dpi=300,
        format="jpg",
    ):
        """
        View raster series statistics

        :param statistic: statistc to view. Default mean
        :type statistic: str
        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param format: image format (ex: png or jpg). Default jpg
        :type format: str
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
            format=format,
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

    def load_raster(self, name, date, asc_file, prj_file):
        """Load a :class:`NDVI` object from a ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param date: :class:`Raster.date` date attribute, defaults to None
        :type date: str
        :param asc_file: path to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path to ``.prj`` projection file
        :type prj_file: str
        :return: None
        :rtype: None
        """
        # create raster
        rst_aux = NDVI(name=name, date=date)
        # read file
        rst_aux.load_asc_raster(file=asc_file)
        # append to collection
        self.append_raster(raster=rst_aux)
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

    def load_raster(self, name, date, asc_file, prj_file):
        """Load a :class:`ET24h` object from a ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param date: :class:`Raster.date` date attribute, defaults to None
        :type date: str
        :param asc_file: path to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path to ``.prj`` projection file
        :type prj_file: str
        :return: None
        :rtype: None
        """
        # create raster
        rst_aux = ET24h(name=name, date=date)
        # read file
        rst_aux.load_asc_raster(file=asc_file)
        # append to collection
        self.append_raster(raster=rst_aux)
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

    def append_raster(self, raster):
        """Append a :class:`Raster` object to collection.
        Pre-existing objects with the same :class:`Raster.name` attribute are replaced

        :param raster: incoming :class:`Raster` to append
        :type raster: :class:`Raster`
        """
        super().append_raster(raster=raster)
        self.update_table()
        return None

    def load_raster(self, name, date, asc_file, prj_file=None, table_file=None):
        """Load a :class:`QualiRaster` object from ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param date: :class:`Raster.date` date attribute
        :type date: str
        :param asc_file: path to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path to ``.prj`` projection file
        :type prj_file: str
        :param table_file: path to ``.txt`` table file
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
        self.append_raster(raster=rst_aux)
        # delete aux
        del rst_aux

    def load_folder(self, folder, table_file, name_pattern="map_*", talk=False):
        """
        Load all rasters from a folder by following a name pattern.
        Date is expected to be at the end of name before file extension.
        :param folder: path to folder
        :type folder: str
        :param table_file: path to table file
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
            self.load_raster(
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
        show=False,
        folder="./output",
        filename=None,
        dpi=300,
        format="jpg",
    ):
        """
        View series areas
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param format: image format (ex: png or jpg). Default jpg
        :type format: str
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
            plt.savefig("{}/{}.{}".format(folder, filename, format), dpi=dpi)
        plt.close(fig)
        return None

    def export_views(
        self, show=False, folder="./output", specs=None, dpi=300, format="jpg"
    ):
        """Plot all basic pannel of qualiraster maps in collection.

        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: path to output folder, defaults to ``./output``
        :type folder: str
        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param format: image format (ex: png or jpg). Default jpg
        :type format: str
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
                format=format,
            )
        return None


class LULCSeries(QualiRasterSeries):
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

    def load_raster(self, name, date, asc_file, prj_file=None, table_file=None):
        """Load a :class:`LULCRaster` object from ``.asc`` raster file.

        :param name: :class:`Raster.name` name attribute
        :type name: str
        :param date: :class:`Raster.date` date attribute
        :type date: str
        :param asc_file: path to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: path to ``.prj`` projection file
        :type prj_file: str
        :param table_file: path to ``.txt`` table file
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
        self.append_raster(raster=rst_aux)
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
        s_name_start = self.catalog.loc[self.catalog['Date'] == date_start]["Name"].values[0] #
        s_name_end = self.catalog.loc[self.catalog['Date'] == date_end]["Name"].values[0]

        # compute lulc change grid
        grd_lulcc = (1 * (self.collection[s_name_end].grid == by_lulc_id)) - (1 * (self.collection[s_name_start].grid == by_lulc_id))
        grd_all = (1 * (self.collection[s_name_end].grid == by_lulc_id)) + (1 * (self.collection[s_name_start].grid == by_lulc_id))
        grd_all = 1 * (grd_all > 0)
        grd_lulcc = (grd_lulcc + 2) * grd_all

        # get names
        s_name = self.name
        s_name_lulc = self.table.loc[self.table["Id"] == by_lulc_id]["Name"].values[0]
        # instantiate
        map_lulc_change = LULCChange(
            name="{}_{}_{}".format(s_name, s_name_lulc, date_end),
            name_lulc=s_name_lulc,
            date_start=date_start,
            date_end=date_end
        )
        map_lulc_change.set_grid(grid=grd_lulcc)
        map_lulc_change.set_asc_metadata(metadata=self.collection[s_name_start].asc_metadata)
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
            varalias="LULCC"
        )
        # loop in catalog
        for i in range(1, len(self.catalog)):
            raster = self.get_lulcc(
                date_start=self.catalog["Date"].values[i - 1],
                date_end=self.catalog["Date"].values[i],
                by_lulc_id=by_lulc_id
            )
            series_lulcc.append_raster(raster=raster)
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
        s_name_start = self.catalog.loc[self.catalog['Date'] == date_start]["Name"].values[0] #
        s_name_end = self.catalog.loc[self.catalog['Date'] == date_end]["Name"].values[0]

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
            map_lulc.set_asc_metadata(metadata=self.collection[s_name_start].asc_metadata)
            map_lulc.set_table(dataframe=self.collection[s_name_start].table)
            map_lulc.prj = self.collection[s_name_start].prj
            #
            # apply aoi
            grd_aoi = 1 * (self.collection[s_name_start].grid == _id)
            map_lulc.apply_aoi_mask(grid_aoi=grd_aoi, inplace=True)
            #
            # bypass all-masked aois
            if np.sum(map_lulc.grid ) is np.ma.masked:
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
            "Date_end": date_end
        }


if __name__ == "__main__":
    print("Hi!")
