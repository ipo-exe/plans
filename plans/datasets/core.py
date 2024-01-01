"""
The core module for datasets.

Description:
    The ``datasets.core`` module provides primitive (abstract) objects to handle ``plans`` data.

License:
    This software is released under the GNU General Public License v3.0 (GPL-3.0).
    For details, see: https://www.gnu.org/licenses/gpl-3.0.html

Author:
    Iporã Possantti

Contact:
    possantti@gmail.com


Overview
--------

todo
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis.
Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Pellentesque habitant morbi tristique senectus
et netus et malesuada fames ac turpis egestas.

>>> from plans.datasets.core import *

Class aptent taciti sociosqu ad litora torquent per
conubia nostra, per inceptos himenaeos. Nulla facilisi. Mauris eget nisl
eu eros euismod sodales. Cras pulvinar tincidunt enim nec semper.

Example
-------

todo
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis. Mauris gravida ex quam,
in porttitor lacus lobortis vitae. In a lacinia nisl.

.. code-block:: python

    import numpy as np
    from plans import analyst

    # get data to a vector
    data_vector = np.random.rand(1000)

    # instantiate the Univar object
    uni = analyst.Univar(data=data_vector, name="my_data")

    # view data
    uni.view()

Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl.

"""
import os, glob, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from plans.root import Collection, DataSet


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


# ------------- CHRONOLOGICAL OBJECTS -------------  #

class TimeSeries(DataSet):

    def __init__(self, name="MyTimeSeries", alias="TS0"):
        """Initialize the ``TimeSeries`` object.
        Expected to increment superior methods.

        :param name: unique object name
        :type name: str

        :param alias: unique object alias.
            If None, it takes the first and last characters from name
        :type alias: str

        """
        # ------------ set defaults ----------- #

        # upstream setups
        self.dtfield = "DateTime"
        self.varfield = "V"
        self.varalias = "Var"
        self.varname = "Variable"
        self.units = "units"
        self.agg = "mean"
        self.cmap = "Dark2"
        self.object_alias = "TS"
        self.rawcolor = "gray"  # alternativa color

        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)

        # overwriters
        self.color = "orange"  # overwrite from super

        # descriptors:
        self.code = None
        self.x = None
        self.y = None
        self.datarange_min = None
        self.datarange_max = None

        # --- auto update info
        # todo refactor names here later on
        self.dtfreq = None
        self.dtres = None
        self.start = None
        self.end = None
        self.var_min = None
        self.var_max = None
        self.isstandard = False

        self.gapsize = 6
        self.epochs_stats = None
        self.epochs_n = None
        self.smallgaps_n = None

        # incoming data
        self.file_input = None  # todo rename
        self.file_data_dtfield = self.dtfield
        self.file_data_varfield = self.varfield
        # ... continues in downstream objects ... #

    def _set_fields(self):
        """
        Set catalog fields names. Expected to increment superior methods.

        """
        # ------------ call super ----------- #
        super()._set_fields()

        # TimeSeries fields
        self.code_field = "Code"
        self.x_field = "X"
        self.y_field = "Y"

        self.varfield_field = "VarField"
        self.varname_field = "VarName"
        self.varalias_field = "VarAlias"
        self.datarange_min_field = "VarRange_min"
        self.datarange_max_field = "VarRange_max"
        self.var_min_field = "Var_min"
        self.var_max_field = "Var_max"

        self.dtfield_field = "DtField"
        self.dtfreq_field = "DtFreq"
        self.dtres_field = "DtRes"
        self.start_field = "Start"
        self.end_field = "End"

        self.isstandard_field = "IsStandard"

        # Epochs
        self.gapsize_field = "GapSize"
        self.epochs_n_field = "Epochs_n"
        self.smallgaps_n_field = "SmallGaps_n"
        self.epochs_id_field = "Epoch_Id"


        # file fields
        self.file_data_dtfield_field = "File_Data_DtField"
        self.file_data_varfield_field = "File_Data_VarField"


        # ... continues in downstream objects ... #

    def _set_view_specs(self):
        """Set view specifications.
        Expected to overwrite superior methods.

        :return: None
        :rtype: None
        """
        self.view_specs = {
            "folder": self.folder_src,
            "filename": self.name,
            "fig_format": "jpg",
            "dpi": 300,
            "title": "Time Series | {} | {} ({})".format(self.name, self.varname, self.varalias),
            "width": 5 * 1.618,
            "height": 3,
            "xvar": self.dtfield,
            "yvar": self.varfield,
            "xlabel": self.dtfield,
            "ylabel": self.units,
            "color": self.rawcolor,
            "xmin": None,
            "xmax": None,
            "ymin": 0,
            "ymax": None,
        }
        return None

    def _set_frequency(self):
        """Guess the datetime resolution of a time series based on the consistency of
        timestamp components (e.g., seconds, minutes).

        :return: None
        :rtype: None

        **Notes:**

        - This method infers the datetime frequency of the time series data
        based on the consistency of timestamp components.

        **Examples:**

        >>> ts._set_frequency()

        """
        # Handle void data
        if self.data is None:
            pass
        else:
            # Copy the data
            df = self.data.copy()

            # Extract components of the datetime
            df["year"] = df[self.dtfield].dt.year
            df["month"] = df[self.dtfield].dt.month
            df["day"] = df[self.dtfield].dt.day
            df["hour"] = df[self.dtfield].dt.hour
            df["minute"] = df[self.dtfield].dt.minute
            df["second"] = df[self.dtfield].dt.second

            # Check consistency within each unit of time
            if df["second"].nunique() > 1:
                self.dtfreq = "1min"
                self.dtres = "second"
            elif df["minute"].nunique() > 1:
                self.dtfreq = "20min"  # force to 20min
                self.dtres = "minute"
            elif df["hour"].nunique() > 1:
                self.dtfreq = "H"
                self.dtres = "hour"
            elif df["day"].nunique() > 1:
                self.dtfreq = "D"
                self.dtres = "day"
                # force gapsize to 1 when daily+ time scale
                self.gapsize = 1
            elif df["month"].nunique() > 1:
                self.dtfreq = "MS"
                self.dtres = "month"
                # force gapsize to 1 when daily+ time scale
                self.gapsize = 1
            else:
                self.dtfreq = "YS"
                self.dtres = "year"
                # force gapsize to 1 when daily+ time scale
                self.gapsize = 1

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

        # customize local metadata:
        dict_meta_local = {
            # TS info
            self.code_field: self.code,
            self.x_field : self.x,
            self.y_field : self.y,

            # Variable info
            self.varfield_field : self.varfield,
            self.varname_field : self.varname,
            self.varalias_field : self.varalias,
            self.datarange_min_field : self.datarange_min,
            self.datarange_max_field : self.datarange_max,
            self.var_min_field : self.var_min,
            self.var_max_field : self.var_max,

            # DateTime info
            self.dtfield_field : self.dtfield,
            self.dtfreq_field : self.dtfreq,
            self.dtres_field : self.dtres,
            self.start_field : self.start,
            self.end_field : self.end,

            # Standardization
            self.isstandard_field : self.isstandard,

            # Epochs info
            self.gapsize_field : self.gapsize,
            self.epochs_n_field : self.epochs_n,
            self.smallgaps_n_field : self.smallgaps_n,

            # file data info
            self.file_data_dtfield_field: self.file_data_dtfield,
            self.file_data_varfield_field: self.file_data_varfield,
        }

        # update
        dict_meta.update(dict_meta_local)
        # mode file_data to the end
        dict_meta.pop(self.filedata_field)
        dict_meta[self.filedata_field] = self.file_data
        return dict_meta

    def update(self):
        """Update internal attributes based on the current data.

        :return: None
        :rtype: None

        **Notes:**

        - Calls the ``set_frequency`` method to update the datetime frequency attribute.
        - Updates the ``start`` attribute with the minimum datetime value in the data.
        - Updates the ``end`` attribute with the maximum datetime value in the data.
        - Updates the ``var_min`` attribute with the minimum value of the variable field in the data.
        - Updates the ``var_max`` attribute with the maximum value of the variable field in the data.
        - Updates the ``data_size`` attribute with the length of the data.

        **Examples:**

        >>> ts.update()
        """
        self._set_frequency()
        self.start = self.data[self.dtfield].min()
        self.end = self.data[self.dtfield].max()
        self.var_min = self.data[self.varfield].min()
        self.var_max = self.data[self.varfield].max()
        self.size = len(self.data)
        self._set_view_specs()
        return None

    def set(self, dict_setter, load_data=True):
        """Set selected attributes based on an incoming dictionary.
        Expected to increment superior methods.

        :param dict_setter: incoming dictionary with attribute values
        :type dict_setter: dict

        :param load_data: option for loading data from incoming file. Default is True.
        :type load_data: bool

        """
        # ---------- call super --------- #
        super().set(dict_setter=dict_setter, load_data=False)

        # ---------- set basic attributes --------- #

        # -------------- set data logic here -------------- #
        if load_data:
            # handle missing keys
            if self.dtfield_field not in list(dict_setter.keys()):
                input_dtfield = None # assume default
            else:
                input_dtfield = dict_setter[self.dtfield_field]
            if self.varfield_field not in list(dict_setter.keys()):
                input_varfield = None # assume default
            else:
                input_varfield = dict_setter[self.varfield_field]
            if "Sep" not in list(dict_setter.keys()):
                in_sep = ";" # assume default
            # call load
            self.load_data(
                file_data=self.file_data,
                input_dtfield=input_dtfield,
                input_varfield=input_varfield,
                in_sep=in_sep,
            )

        # ... continues in downstream objects ... #

    def load_data(self, file_data, input_dtfield=None, input_varfield=None, in_sep=";"):
        """Load data from file. Expected to overwrite superior methods.

        :param file_data: str
            Absolute Path to the ``csv`` input file.
        :type file_data: str

        :param input_varfield: str
            Name of the incoming varfield.
        :type input_varfield: str

        :param input_dtfield: str, optional
            Name of the incoming datetime field. Default is "DateTime".
        :type input_dtfield: str

        :param sep: str, optional
            String separator. Default is ``;`.
        :type sep: str

        :return: None
        :rtype: None

        **Notes:**

        - Assumes the input file is in ``csv`` format.
        - Expects a datetime column in the format ``YYYY-mm-DD HH:MM:SS``.

        **Examples:**

        >>> ts.load_data("path/to/data.csv", input_dtfield="Date", input_varfield="TempDB", sep=",")

        .. important::

            The ``DateTime`` field in the incoming file must be a full timestamp ``YYYY-mm-DD HH:MM:SS``.
            Even if the data is a daily time series, make sure to include a constant, default timestamp
            like the following example:

            .. code-block:: text

                           DateTime; Temperature
                2020-02-07 12:00:00;        24.5
                2020-02-08 12:00:00;        25.1
                2020-02-09 12:00:00;        28.7
                2020-02-10 12:00:00;        26.5

        """

        # -------------- overwrite relative path input -------------- #
        file_data = os.path.abspath(file_data)

        # -------------- implement loading logic -------------- #

        # assume defaults
        if input_dtfield is None:
            input_dtfield = self.file_data_dtfield
        else:
            self.file_data_dtfield = input_dtfield

        if input_varfield is None:
            input_varfield = self.file_data_varfield
        else:
            self.file_data_varfield = input_varfield

        # -------------- call loading function -------------- #
        self.data = pd.read_csv(
            file_data,
            sep=in_sep,
            dtype={input_dtfield: str, input_varfield: float},
            usecols=[input_dtfield, input_varfield],
            parse_dates=[self.dtfield]
        )

        # -------------- post-loading logic -------------- #
        self.data = self.data.rename(
            columns={
                input_dtfield: self.dtfield,
                input_varfield: self.varfield
            }
        )

        # ------------ update related attributes ------------ #
        self.file_data = file_data[:]
        self.folder_src = os.path.dirname(self.file_data)

        self.update()

        return None

    # docs: ok
    def cut_edges(self, inplace=False):
        """Cut off initial and final NaN records in a given time series.

        :param inplace: bool, optional
            If True, the operation will be performed in-place, and the original data will be modified.
            If False, a new DataFrame with cut edges will be returned, and the original data will remain unchanged.
            Default is False.
        :type inplace: bool

        :return: :class:`pandas.DataFrame``` or None
            If inplace is False, a new DataFrame with cut edges.
            If inplace is True, returns None, and the original data is modified in-place.
        :rtype: :class:`pandas.DataFrame``` or None

        **Notes:**

        - This function removes leading and trailing rows with NaN values in the specified variable field.
        - The operation is performed on a copy of the original data, and the original data remains unchanged.

        **Examples:**

        >>> ts.cut_edges(inplace=True)

        >>> trimmed_ts = ts.cut_edges(inplace=False)
        """

        # get dataframe
        in_df = self.data.copy()
        def_len = len(in_df)

        # drop first nan lines
        drop_ids = list()

        # loop to collect indexes in the start of series
        for def_i in range(def_len): # go forward
            aux = in_df[self.varfield].isnull().iloc[def_i]
            if aux:
                drop_ids.append(def_i)
            else:
                break

        # loop to collect indexes in the end of series
        for def_i in range(def_len - 1, -1, -1): # go backward
            aux = in_df[self.varfield].isnull().iloc[def_i]
            if aux:
                drop_ids.append(def_i)
            else:
                break

        # loop to drop rows:
        for def_i in range(len(drop_ids)):
            in_df.drop(drop_ids[def_i], inplace=True)

        # output
        if inplace:
            self.data = in_df.copy()
            return None
        else:
            return in_df
    # docs: ok
    def standardize(self, start=None, end=None):
        """Standardize the data based on regular datetime steps and the time resolution.

        :param start: :class:`pandas.Timestamp`, optional
            The starting datetime for the standardization.
            Defaults to the first datetime in the data.
        :type start: :class:`pandas.Timestamp`

        :param end: :class:`pandas.Timestamp`, optional
            The ending datetime for the standardization.
            Defaults to the last datetime in the data.
        :type end: :class:`pandas.Timestamp`

        :return: None
        :rtype: None

        **Notes:**

        - Handles missing start and end values by using the first and last datetimes in the data.
        - Standardizes the incoming start and end datetimes to midnight of their respective days.
        - Creates a full date range with the specified frequency for the standardization period.
        - Groups the data by epochs (based on the frequency and datetime field), applies the specified aggregation function, and fills in missing values with left merges.
        - Cuts off any edges with missing data.
        - Updates internal attributes, including ``self.isstandard`` to indicate that the data has been standardized.

        **Examples:**

        >>> ts.standardize()

        .. warning::

            The ``standardize`` method modifies the internal data representation. Ensure to review the data after standardization.

        """

        def _insert_epochs(_df, _dtfield, _freq):
            _df["StEpoch"] = (
                pd.cut(
                    _df[_dtfield],
                    bins=pd.date_range(
                        start=_df[_dtfield].min(),
                        end=_df[_dtfield].max(),
                        freq=_freq,
                    ),
                    labels=False,
                )
                + 1
            )
            _df["StEpoch"].values[0] = 0
            return _df

        # Handle None values
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        # Standardize incoming start and end
        start = start.date()
        end = end + pd.Timedelta(days=1)
        end = end.date()

        # Get a standard date range for all periods
        dt_index = pd.date_range(start=start, end=end, freq=self.dtfreq)
        # Set dataframe
        df = pd.DataFrame({self.dtfield: dt_index})
        # Insert Epochs in standard data
        df = _insert_epochs(_df=df, _dtfield=self.dtfield, _freq=self.dtfreq)

        # Get non-standard data
        df2 = self.data.copy()

        # Insert Epochs in non-standard data
        df2 = _insert_epochs(_df=df2, _dtfield=self.dtfield, _freq=self.dtfreq)

        # Group by 'Epochs' and calculate agg function
        result_df = df2.groupby("StEpoch")[self.varfield].agg([self.agg]).reset_index()
        # Rename
        result_df = result_df.rename(columns={self.agg: self.varfield})

        # Merge
        df = pd.merge(
            left=df, right=result_df, left_on="StEpoch", right_on="StEpoch", how="left"
        )

        # Clear extra column
        self.data = df.drop(columns="StEpoch").copy()

        # Cut off edges
        self.cut_edges(inplace=True)

        # Set extra attributes
        self.isstandard = True

        # Update all attributes
        self.update()

        return None

    def clear_outliers(self, inplace=False):
        # todo docstring
        # copy local data
        vct = self.data[self.varfield].values

        # Clear lower values
        if self.datarange_min is not None:
            vct = np.where(vct < self.datarange_min, np.nan, vct)

        # Clear upper values
        if self.datarange_max is not None:
            vct = np.where(vct > self.datarange_max, np.nan, vct)

        # return case
        if inplace:
            # overwrite local data
            self.data[self.varfield] = vct
            return None
        else:
            return df

        # docs: ok
    def get_epochs(self, inplace=False):
        """Get Epochs (periods) for continuous time series (0 = gap epoch).

        :param inplace: bool, optional
            Option to set Epochs inplace. Default is False.
        :type inplace: bool
        :return: :class:`pandas.DataFrame``` or None
            A DataFrame if inplace is False or None.
        :rtype: :class:`pandas.DataFrame``, None

        **Notes:**

        This function labels continuous chunks of data as Epochs, with Epoch 0 representing gaps in the time series.

        **Examples:**

        >>> df_epochs = ts.get_epochs()
        """
        df = self.data.copy()
        # Create a helper column to label continuous chunks of data
        df["CumSum"] = (
            df[self.varfield]
            .isna()
            .astype(int)
            .groupby(df[self.varfield].notna().cumsum())
            .cumsum()
        )

        # get skip hint
        skip_v = np.zeros(len(df))
        for i in range(len(df) - 1):
            n_curr = df["CumSum"].values[i]
            n_next = df["CumSum"].values[i + 1]
            if n_next < n_curr:
                if n_curr >= self.gapsize:
                    n_start = i - n_curr
                    if i <= n_curr:
                        n_start = 0
                    skip_v[n_start + 1: i + 1] = 1
        df["Skip"] = skip_v

        # Set Epoch Field
        df[self.epochs_id_field] = 0
        # counter epochs
        counter = 1
        for i in range(len(df) - 1):
            if df["Skip"].values[i] == 0:
                df[self.epochs_id_field].values[i] = counter
            else:
                if df["Skip"].values[i + 1] == 0:
                    counter = counter + 1
        if df["Skip"].values[i + 1] == 0:
            df[self.epochs_id_field].values[i] = counter

        df = df.drop(columns=["CumSum", "Skip"])

        if inplace:
            self.data = df.copy()
            return None
        else:
            return df

    def update_epochs_stats(self):
        """Update all epochs statistics.

        :return: None
        :rtype: None

        **Notes:**

        - This function updates statistics for all epochs in the time series.
        - Ensures that the data is standardized by calling the ``standardize`` method if it's not already standardized.
        - Removes epoch 0 from the statistics since it typically represents non-standardized or invalid data.
        - Groups the data by ``Epoch_Id`` and calculates statistics such as count, start, and end timestamps for each epoch.
        - Generates random colors for each epoch using the ``get_random_colors`` function with a specified colormap (`cmap`` attribute).
        - Includes the time series name in the statistics for identification.
        - Organizes the statistics DataFrame to include relevant columns: ``Name``, ``Epoch_Id``, ``Count``, ``Start``, ``End``, and ``Color``.
        - Updates the attribute ``epochs_n`` with the number of epochs in the statistics.

        **Examples:**

        >>> ts.update_epochs_stats()

        """

        # Ensure data is standardized
        if not self.isstandard:
            self.standardize()

        # Get epochs
        df = self.get_epochs(inplace=False)

        # Remove epoch = 0
        df.drop(df[df[self.epochs_id_field] == 0].index, inplace=True)
        df[self.smallgaps_n_field] = 1 * df[self.varfield].isna().values

        # group by
        aggregation_dict = {
            self.epochs_n_field: (self.epochs_id_field, 'count'),
            self.start_field: (self.dtfield, 'min'),
            self.end_field: (self.dtfield, 'max'),
            self.smallgaps_n_field: (self.smallgaps_n_field, 'sum')
        }

        self.epochs_stats = (
            df.groupby(self.epochs_id_field)
            .agg(**aggregation_dict)
            .reset_index()
        )

        # Get colors
        self.epochs_stats[self.color_field] = get_random_colors(
            size=len(self.epochs_stats), cmap=self.cmap
        )

        # Include name
        self.epochs_stats[self.name_field] = self.name

        # Organize fields
        self.epochs_stats = self.epochs_stats[
            [
                self.name_field,
                self.epochs_id_field,
                self.epochs_n_field,
                self.smallgaps_n_field,
                self.start_field,
                self.end_field,
                self.color_field
            ]
        ]

        # Update attributes
        self.epochs_n = len(self.epochs_stats)
        self.smallgaps_n = self.epochs_stats[self.smallgaps_n_field].sum()

        return None

    # docs: ok
    def interpolate_gaps(self, method="linear", constant=0, inplace=False):
        """Fill gaps in a time series by interpolating missing values. If the time series is not in standard form, it will be standardized before interpolation.

        :param method: str, optional
            Specifies the interpolation method. Default is ``linear`.
            Other supported methods include ``constant``, ``nearest``, ``zero``, ``slinear``, ``quadratic``, ``cubic``, etc.
            The ``constant`` method applies a constant value to gaps.
            Refer to the documentation of scipy.interpolate.interp1d for more options.
        :type method: str

        :param constant: float, optional
            Value used if the case of ``constant`` method. Default is 0.
        :type constant: float

        :param inplace: bool, optional
            If True, the interpolation will be performed in-place, and the original data will be modified.
            If False, a new DataFrame with interpolated values will be returned, and the original data will remain unchanged.
            Default is False.
        :type inplace: bool

        :return: :class:`pandas.DataFrame``` or None
            If inplace is False, a new DataFrame with interpolated values.
            If inplace is True, returns None, and the original data is modified in-place.
        :rtype: :class:`pandas.DataFrame``` or None

        **Notes:**

        - The interpolation is performed for each unique epoch in the time series.
        - The ``method`` parameter determines the interpolation technique. Common options include ``constant``, ``linear``, ``nearest``, ``zero``, ``slinear``, ``quadratic``, and ``cubic`. See the documentation of scipy.interpolate.interp1d for additional methods and details.
        - If ``linear`` is chosen, the interpolation is a linear interpolation. For ``nearest``, it uses the value of the nearest data point. ``zero`` uses zero-order interpolation (nearest-neighbor). ``slinear`` and ``quadratic`` are spline interpolations of first and second order, respectively. ``cubic`` is a cubic spline interpolation.
        - If the method is ``linear``, the fill_value parameter is set to ``extrapolate`` to allow extrapolation beyond the data range.

        **Examples:**

        >>> ts.interpolate_gaps(method="linear", inplace=True)

        >>> interpolated_ts = ts.interpolate_gaps(method="linear", inplace=False)

        >>> interpolated_ts = ts.interpolate_gaps(method="constant", constant=1, inplace=False)
        """
        from scipy.interpolate import interp1d

        # Ensure data is standardized
        if not self.isstandard:
            self.standardize()

        # Get epochs for interpolation
        df = self.get_epochs(inplace=False)
        epochs = df["Epoch_Id"].unique()
        list_dfs = list()

        for epoch in epochs:
            df_aux1 = df.query("Epoch_Id == {}".format(epoch)).copy()
            if epoch == 0:
                df_aux1["{}_interp".format(self.varfield)] = np.nan
            else:
                if method == "constant":
                    df_aux1["{}_interp".format(self.varfield)] = np.where(
                        df_aux1[self.varfield].isna(),
                        constant,
                        df_aux1[self.varfield].values,
                    )
                else:  # use scipy methods
                    # clear from nan values
                    df_aux2 = df_aux1.dropna().copy()
                    # Create an interpolation function
                    interpolation_func = interp1d(
                        df_aux2[self.dtfield].astype(np.int64).values,
                        df_aux2[self.varfield].values,
                        kind=method,
                        fill_value="extrapolate",
                    )
                    # Interpolate full values
                    df_aux1["{}_interp".format(self.varfield)] = interpolation_func(
                        df_aux1[self.dtfield].astype(np.int64)
                    )
            # Append
            list_dfs.append(df_aux1)
        df_new = pd.concat(list_dfs, ignore_index=True)
        df_new = df_new.sort_values(by=self.dtfield).reset_index(drop=True)

        if inplace:
            self.data[self.varfield] = df_new["{}_interp".format(self.varfield)].values
            self.update()
            return None
        else:
            return df_new

    # docs: ok
    def aggregate(self, freq, bad_max, agg_funcs=None):
        """ "Aggregate the time series data based on a specified frequency using various aggregation functions.

        :param freq: str
            Pandas-like alias frequency at which to aggregate the time series data. Common options include:
            - ``H`` for hourly frequency
            - ``D`` for daily frequency
            - ``W`` for weekly frequency
            - ``MS`` for monthly/start frequency
            - ``QS`` for quarterly/start frequency
            - ``YS`` for yearly/start frequency
            More options and details can be found in the Pandas documentation:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
        :type freq: str

        :param bad_max: int
            The maximum number of ``Bad`` records allowed in a time window for aggregation. Records with more ``Bad`` entries
            will be excluded from the aggregated result.
            Default is 7.
        :type bad_max: int

        :param agg_funcs: dict, optional
            A dictionary specifying customized aggregation functions for each variable.
            Default is None, which uses standard aggregation functions (sum, mean, median, min, max, std, var, percentiles).
        :type agg_funcs: dict

        :return: :class:`pandas.DataFrame`
            A new :class:`pandas.DataFrame` with aggregated values based on the specified frequency.
        :rtype: :class:`pandas.DataFrame`

        **Notes:**

        - Resamples the time series data to the specified frequency using Pandas-like alias strings.
        - Aggregates the values using the specified aggregation functions.
        - Counts the number of ``Bad`` records in each time window and excludes time windows with more ``Bad`` entries
          than the specified threshold.

        **Examples:**

        >>> agg_result = ts.aggregate(freq='D', agg_funcs={'sum': 'sum', 'mean': 'mean'}, bad_max=5)

        """

        if agg_funcs is None:
            # Create a dictionary of standard and custom aggregation functions
            agg_funcs = {
                "sum": "sum",
                "mean": "mean",
                "median": "median",
                "min": "min",
                "max": "max",
                "std": "std",
                "var": "var",
            }
            '''
            # Add custom percentiles to the dictionary
            percentiles_to_compute = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles_to_compute:
                agg_funcs[f"p{p}"] = lambda x, p=p: np.percentile(x, p)
            '''

        # set list of tuples
        agg_funcs_list = [
            ("{}_{}".format(self.varfield, f), agg_funcs[f]) for f in agg_funcs
        ]

        # get data
        df = self.data.copy()
        df["Bad"] = df[self.varfield].isna().astype(int)
        # Set the 'datetime' column as the index
        df.set_index(self.dtfield, inplace=True)

        # Resample the time series to a frequency using aggregation functions
        agg_df1 = df.resample(freq)[self.varfield].agg(agg_funcs_list)
        agg_df2 = df.resample(freq)["Bad"].agg([("Bad_count", "sum")])

        # Reset the index to get 'DateTime' as a regular column
        agg_df1.reset_index(inplace=True)
        agg_df2.reset_index(inplace=True)

        # merge with bad dates
        agg_df = pd.merge(left=agg_df1, right=agg_df2, how="left", on=self.dtfield)

        # set new df
        agg_df_new = pd.DataFrame(
            {
                self.dtfield: pd.date_range(
                    start=agg_df[self.dtfield].values[0],
                    end=agg_df[self.dtfield].values[-1],
                    freq=freq,
                )
            }
        )
        # remove bad records
        agg_df.drop(agg_df[agg_df["Bad_count"] > bad_max].index, inplace=True)
        # remove bad column
        agg_df.drop(columns=["Bad_count"], inplace=True)

        # left join
        agg_df_new = pd.merge(
            left=agg_df_new, right=agg_df, on=self.dtfield, how="left"
        )

        return agg_df_new

    def upscale(self, freq, bad_max, inplace=True):
        # todo docstring
        df_upscale = self.aggregate(freq=freq, bad_max=bad_max, agg_funcs={self.agg: self.agg})
        df_upscale = df_upscale.rename(columns={"{}_{}".format(self.varfield, self.agg): self.varfield})

        #return case
        if inplace:
            # overwrite local data
            self.data = df_upscale
            #
            self._set_frequency()

            return None
        else:
            return df_upscale


    # todo method
    def downscale(self, freq):
        # todo dosctring
        # 0) update
        self.update()
        # 1) get new index
        dt_index = pd.date_range(start=self.start, end=self.end, freq=freq)
        print(dt_index[:5])
        # 2) factor
        factor_downscale = len(dt_index) / self.data_size
        print(factor_downscale)


    def view_epochs(self, show=True):
        """Get a basic visualization.
        Expected to overwrite superior methods.

        :param show: option for showing instead of saving.
        :type show: bool

        :return: None or file path to figure
        :rtype: None or str

        **Notes:**

        - Uses values in the ``view_specs()`` attribute for plotting

        **Examples:**

        Simple visualization:

        >>> ds.view(show=True)

        Customize view specs:

        >>> ds.view_specs["title"] = "My Custom Title"
        >>> ds.view_specs["xlabel"] = "The X variable"
        >>> ds.view(show=True)

        Save the figure:

        >>> ds.view_specs["folder"] = "path/to/folder"
        >>> ds.view_specs["filename"] = "my_visual"
        >>> ds.view_specs["fig_format"] = "png"
        >>> ds.view(show=False)

        """
        # get specs
        specs = self.view_specs.copy()
        # --------------------- pre-processing --------------------- #
        # preprocessing
        if self.epochs_stats is None:
            self.update_epochs_stats()

        # --------------------- figure setup --------------------- #
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height

        # handle min max
        if specs["xmin"] is None:
            specs["xmin"] = self.data[specs["xvar"]].min()
        if specs["ymin"] is None:
            specs["ymin"] = self.data[specs["yvar"]].min()
        if specs["xmax"] is None:
            specs["xmax"] = self.data[specs["xvar"]].max()
        if specs["ymax"] is None:
            specs["ymax"] = self.data[specs["yvar"]].max()


        # --------------------- plotting --------------------- #
        gaps_c = "tab:red"
        gaps_a = 0.6
        # plot loop
        for i in range(len(self.epochs_stats)):
            start = self.epochs_stats["Start"].values[i]
            end = self.epochs_stats["End"].values[i]
            df_aux = self.data.query(
                "{} >= '{}' and {} < '{}'".format(
                    self.dtfield, start, self.dtfield, end
                )
            )
            epoch_c = self.epochs_stats["Color"].values[i]
            epoch_id = self.epochs_stats["Epoch_Id"].values[i]
            plt.plot(
                df_aux[self.dtfield],
                df_aux[self.varfield],
                ".",
                color=epoch_c,
                label=f"Epoch_{epoch_id}",
            )
            # Fill the space where there are missing values
            plt.fill_between(
                df_aux[self.dtfield],
                y1=0,
                y2=1.2 * specs["ymax"],
                where=df_aux[self.varfield].isna(),
                color=gaps_c,
                alpha=gaps_a,
            )
        # Create a dummy plot for missing data symbol
        plt.plot([], [], color=gaps_c, alpha=gaps_a, label='Gaps')
        if self.epochs_n <= 12:
            plt.legend(frameon=True, ncol=3)

        # --------------------- post-plotting --------------------- #
        # set basic plotting stuff
        plt.title(specs["title"])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["xlabel"])

        plt.xlim(specs["xmin"], specs["xmax"])
        plt.ylim(specs["ymin"], 1.2 * specs["ymax"])

        # Adjust layout to prevent cutoff
        plt.tight_layout()

        # --------------------- end --------------------- #
        # show or save
        if show:
            plt.show()
            return None
        else:
            file_path = "{}/{}.{}".format(
                specs["folder"], specs["filename"], specs["fig_format"]
            )
            plt.savefig(file_path, dpi=specs["dpi"])
            plt.close(fig)
            return file_path

    def view(self, show=True):
        """Get a basic visualization.
        Expected to overwrite superior methods.

        :param show: option for showing instead of saving.
        :type show: bool

        :return: None or file path to figure
        :rtype: None or str

        **Notes:**

        - Uses values in the ``view_specs()`` attribute for plotting

        **Examples:**

        Simple visualization:

        >>> ds.view(show=True)

        Customize view specs:

        >>> ds.view_specs["title"] = "My Custom Title"
        >>> ds.view_specs["xlabel"] = "The X variable"
        >>> ds.view(show=True)

        Save the figure:

        >>> ds.view_specs["folder"] = "path/to/folder"
        >>> ds.view_specs["filename"] = "my_visual"
        >>> ds.view_specs["fig_format"] = "png"
        >>> ds.view(show=False)

        """
        # get specs
        specs = self.view_specs.copy()

        # --------------------- figure setup --------------------- #
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height

        # handle min max
        if specs["xmin"] is None:
            specs["xmin"] = self.data[specs["xvar"]].min()
        if specs["ymin"] is None:
            specs["ymin"] = self.data[specs["yvar"]].min()
        if specs["xmax"] is None:
            specs["xmax"] = self.data[specs["xvar"]].max()
        if specs["ymax"] is None:
            specs["ymax"] = self.data[specs["yvar"]].max()

        # --------------------- plotting --------------------- #
        plt.plot(
            self.data[specs["xvar"]],
            self.data[specs["yvar"]],
            ".",
            color=specs["color"]
        )

        # --------------------- post-plotting --------------------- #
        # set basic plotting stuff
        plt.title(specs["title"])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["xlabel"])
        plt.xlim(specs["xmin"], specs["xmax"])
        plt.ylim(specs["ymin"], 1.2 * specs["ymax"])

        # Adjust layout to prevent cutoff
        plt.tight_layout()

        # --------------------- end --------------------- #
        # show or save
        if show:
            plt.show()
            return None
        else:
            file_path = "{}/{}.{}".format(
                specs["folder"], specs["filename"], specs["fig_format"]
            )
            plt.savefig(file_path, dpi=specs["dpi"])
            plt.close(fig)
            return file_path



# DEPRECATED
class _TimeSeries:
    """
    The primitive time series object

    """

    def __init__(
        self, name="MyTS", alias=None, varname="variable", varfield="V", units="units"
    ):
        """Deploy time series object

        :param name: str, optional
            Name for the object.
            Default is "MyTS".
        :type name: str

        :param alias: str, optional
            Alias for the object.
            Default is None, and it will be set to the first three characters of the name.
        :type alias: str

        :param varname: str, optional
            Variable name.
            Default is "Variable".
        :type varname: str

        :param varfield: str, optional
            Variable field alias.
            Default is "V".
        :type varfield: str

        :param units: str, optional
            Units of the variable.
            Default is "units".
        :type units: str

        **Notes:**

        - The ``alias`` parameter defaults to the first three characters of the ``name`` parameter if not provided.
        - Various attributes related to optional and auto-update information are initialized to ``None`` or default values.
        - The ``dtfield`` attribute is set to "DateTime" as the default datetime field.
        - The ``cmap`` attribute is set to "tab20b" as a default colormap.
        - The method ``_set_view_specs`` is called to set additional specifications.

        **Examples:**

        Creating a time series object with default parameters:

        >>> ts_default = TimeSeries()

        Creating a time series object with custom parameters:

        >>> ts_custom = TimeSeries(name="CustomTS", alias="Cust", varname="Temperature", varfield="Temp", units="Celsius")
        """
        self.name_object = "Time Series"
        # Set initial attributes based on provided parameters
        self.name = name
        if alias is None:
            alias = name[:3]
        self.alias = alias
        self.varname = varname
        self.varalias = varfield
        self.varfield = varfield
        self.units = units
        self.dtfield = "DateTime"

        # --- optional info ---
        self.source_data = None
        self.description = None
        self.color = "tab:blue"
        self.rawcolor = "tab:blue"
        self.code = None
        self.x = None
        self.y = None
        self.datarange_min = None
        self.datarange_max = None

        # --- auto update info
        self.data = None
        self.data_size = None
        self.start = None
        self.end = None
        self.dtfreq = "20min"
        self.dtres = "minutes"
        self.var_min = None
        self.var_max = None
        self.isstandard = False
        self.agg = "mean"
        self.epochs_stats = None
        self.epochs_n = None
        self.smallgaps_n = None
        self.gapsize = 2
        self.file_input = None

        # hack attributes
        self.cmap = "Dark2"

        # set specs
        self._set_view_specs()

    def get_metadata(self):
        """Get metadata information from the base object."""
        # Prepare and return metadata as a dictionary
        return {
            "Name": self.name,
            "Alias": self.alias,
            "Variable": self.varname,
            "VarField": self.varfield,
            "DtField": self.dtfield,
            "Units": self.units,
            "Size": self.data_size,
            "Epochs": self.epochs_n,
            "Start": self.start,
            "End": self.end,
            "Min": self.var_min,
            "Max": self.var_max,
            "DT_freq": self.dtfreq,
            "DT_res": self.dtres,
            "IsStandard": self.isstandard,
            "GapSize": self.gapsize,
            "Color": self.color,
            "Source": self.source_data,
            "Description": self.description,
            "Code": self.code,
            "X": self.x,
            "Y": self.y,
            "File": self.file_input,
        }

    # docs: ok
    def load_data(
        self,
        input_file,
        input_varfield,
        input_dtfield="DateTime",
        sep=";",
        filter_dates=None,
    ):
        """Load data from file.

        :param input_file: str
            Path to the ``csv`` input file.
        :type input_file: str

        :param input_varfield: str
            Name of the incoming varfield.
        :type input_varfield: str

        :param input_dtfield: str, optional
            Name of the incoming datetime field. Default is "DateTime".
        :type input_dtfield: str

        :param sep: str, optional
            String separator. Default is ``;`.
        :type sep: str

        :return: None
        :rtype: None

        **Notes:**

        - Assumes the input file is in CSV format.
        - Expects a datetime column in the format "YYYY-mm-DD HH:MM:SS".

        **Examples:**

        >>> ts.load_data("data.csv", "Temperature", input_dtfield="Date", sep=",")

        .. important::

            The DateTime field in the incoming file must be a full timestamp ``YYYY-mm-DD HH:MM:SS``.
            Even if the data is a daily time series, make sure to include a constant, default timestamp
            like the following example:

            .. code-block:: text

                           DateTime; Temperature
                2020-02-07 12:00:00;        24.5
                2020-02-08 12:00:00;        25.1
                2020-02-09 12:00:00;        28.7
                2020-02-10 12:00:00;        26.5
        """
        # Load data from csv
        df = pd.read_csv(input_file, sep=sep, usecols=[input_dtfield, input_varfield])

        # Set data
        self.set_data(
            input_df=df,
            input_varfield=input_varfield,
            input_dtfield=input_dtfield,
            filter_dates=filter_dates,
        )

        # Set attributes
        self.file_input = input_file

        return None

    # docs: OK
    def set_data(
        self, input_df, input_dtfield, input_varfield, filter_dates=None, dropnan=True
    ):
        """Set time series data from an input DataFrame.

        :param input_df: pandas DataFrame
            Input DataFrame containing time series data.
        :type input_df: :class:`pandas.DataFrame`

        :param input_dtfield: str
            Name of the datetime field in the input DataFrame.
        :type input_dtfield: str

        :param input_varfield: str
            Name of the variable field in the input DataFrame.
        :type input_varfield: str

        :param filter_dates: list, optional
            List of [Start, End] used for filtering date range.
        :type dropnan: list

        :param dropnan: bool, optional
            If True, drop NaN values from the DataFrame. Default is True.
        :type dropnan: bool

        :return: None
        :rtype: None

        **Notes:**

        - Assumes the input DataFrame has a datetime column in the format "YYYY-mm-DD HH:MM:SS".
        - Renames columns to standard format (datetime: ``self.dtfield``, variable: ``self.varfield``).
        - Converts the datetime column to standard format.

        **Examples:**

        >>> input_data = pd.DataFrame({
        ...     'Date': ['2022-01-01 12:00:00', '2022-01-02 12:00:00', '2022-01-03 12:00:00'],
        ...     'Temperature': [25.1, 26.5, 24.8]
        ... })
        >>> ts.set_data(input_data, input_dtfield='Date', input_varfield='Temperature')

        """
        # Get a copy of the input DataFrame
        df = input_df.copy()

        # Drop NaN values if specified
        if dropnan:
            df = df.dropna()

        # Rename columns to standard format
        df = df.rename(
            columns={input_dtfield: self.dtfield, input_varfield: self.varfield}
        )

        # Convert datetime column to standard format
        df[self.dtfield] = pd.to_datetime(df[self.dtfield], format="%Y-%m-%d %H:%M:%S")

        if filter_dates is None:
            pass
        else:
            if filter_dates[0] is not None:
                df = df.query("{} >= '{}'".format(self.dtfield, filter_dates[0]))
            if filter_dates[1] is not None:
                df = df.query("{} < '{}'".format(self.dtfield, filter_dates[1]))

        # Set the data attribute
        self.data = df.copy()
        # update all
        self.update()

        return None

    # docs ok
    def export(self, folder):
        """Export data (time series and epoch stats) to csv files

        :param folder: str
            Path to output folder
        :type folder: str

        :return: None
        :rtype: None
        """
        filename = "{}_{}".format(self.varname, self.alias)
        if self.data is not None:
            self.data.to_csv("{}/{}.csv".format(folder, filename), index=False, sep=";")
        if self.epochs_stats is not None:
            self.epochs_stats.to_csv(
                "{}/{}_epochs.csv".format(folder, filename), index=False, sep=";"
            )
        return None

    # docs: ok
    def standardize(self, start=None, end=None):
        """Standardize the data based on regular datetime steps and the time resolution.

        :param start: :class:`pandas.Timestamp`, optional
            The starting datetime for the standardization.
            Defaults to the first datetime in the data.
        :type start: :class:`pandas.Timestamp`

        :param end: :class:`pandas.Timestamp`, optional
            The ending datetime for the standardization.
            Defaults to the last datetime in the data.
        :type end: :class:`pandas.Timestamp`

        :return: None
        :rtype: None

        **Notes:**

        - Handles missing start and end values by using the first and last datetimes in the data.
        - Standardizes the incoming start and end datetimes to midnight of their respective days.
        - Creates a full date range with the specified frequency for the standardization period.
        - Groups the data by epochs (based on the frequency and datetime field), applies the specified aggregation function, and fills in missing values with left merges.
        - Cuts off any edges with missing data.
        - Updates internal attributes, including ``self.isstandard`` to indicate that the data has been standardized.

        **Examples:**

        >>> ts.standardize()

        .. warning::

            The ``standardize`` method modifies the internal data representation. Ensure to review the data after standardization.

        """

        def _insert_epochs(_df, _dtfield, _freq):
            _df["StEpoch"] = (
                pd.cut(
                    _df[_dtfield],
                    bins=pd.date_range(
                        start=_df[_dtfield].min(),
                        end=_df[_dtfield].max(),
                        freq=_freq,
                    ),
                    labels=False,
                )
                + 1
            )
            _df["StEpoch"].values[0] = 0
            return _df

        # Handle None values
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        # Standardize incoming start and end
        start = start.date()
        end = end + pd.Timedelta(days=1)
        end = end.date()

        # Get a standard date range for all periods
        dt_index = pd.date_range(start=start, end=end, freq=self.dtfreq)
        # Set dataframe
        df = pd.DataFrame({self.dtfield: dt_index})
        # Insert Epochs in standard data
        df = _insert_epochs(_df=df, _dtfield=self.dtfield, _freq=self.dtfreq)

        # Get non-standard data
        df2 = self.data.copy()

        # Insert Epochs in non-standard data
        df2 = _insert_epochs(_df=df2, _dtfield=self.dtfield, _freq=self.dtfreq)

        # Group by 'Epochs' and calculate agg function
        result_df = df2.groupby("StEpoch")[self.varfield].agg([self.agg]).reset_index()
        # Rename
        result_df = result_df.rename(columns={self.agg: self.varfield})

        # Merge
        df = pd.merge(
            left=df, right=result_df, left_on="StEpoch", right_on="StEpoch", how="left"
        )

        # Clear extra column
        self.data = df.drop(columns="StEpoch").copy()

        # Cut off edges
        self.cut_edges(inplace=True)

        # Set extra attributes
        self.isstandard = True

        # Update all attributes
        self.update()

        return None

    # docs: ok
    def get_epochs(self, inplace=False):
        """Get Epochs (periods) for continuous time series (0 = gap epoch).

        :param inplace: bool, optional
            Option to set Epochs inplace. Default is False.
        :type inplace: bool
        :return: :class:`pandas.DataFrame``` or None
            A DataFrame if inplace is False or None.
        :rtype: :class:`pandas.DataFrame``, None

        **Notes:**

        This function labels continuous chunks of data as Epochs, with Epoch 0 representing gaps in the time series.

        **Examples:**

        >>> df_epochs = ts.get_epochs()
        """
        df = self.data.copy()
        # Create a helper column to label continuous chunks of data
        df["CumSum"] = (
            df[self.varfield]
            .isna()
            .astype(int)
            .groupby(df[self.varfield].notna().cumsum())
            .cumsum()
        )

        # get skip hint
        skip_v = np.zeros(len(df))
        for i in range(len(df) - 1):
            n_curr = df["CumSum"].values[i]
            n_next = df["CumSum"].values[i + 1]
            if n_next < n_curr:
                if n_curr >= self.gapsize:
                    n_start = i - n_curr
                    if i <= n_curr:
                        n_start = 0
                    skip_v[n_start + 1 : i + 1] = 1
        df["Skip"] = skip_v

        # Set Epoch Field
        df["Epoch_Id"] = 0
        # counter epochs
        counter = 1
        for i in range(len(df) - 1):
            if df["Skip"].values[i] == 0:
                df["Epoch_Id"].values[i] = counter
            else:
                if df["Skip"].values[i + 1] == 0:
                    counter = counter + 1
        if df["Skip"].values[i + 1] == 0:
            df["Epoch_Id"].values[i] = counter

        df = df.drop(columns=["CumSum", "Skip"])

        if inplace:
            self.data = df.copy()
            return None
        else:
            return df

    # docs: ok
    def update(self):
        """Update internal attributes based on the current data.

        :return: None
        :rtype: None

        **Notes:**

        - Calls the ``set_frequency`` method to update the datetime frequency attribute.
        - Updates the ``start`` attribute with the minimum datetime value in the data.
        - Updates the ``end`` attribute with the maximum datetime value in the data.
        - Updates the ``var_min`` attribute with the minimum value of the variable field in the data.
        - Updates the ``var_max`` attribute with the maximum value of the variable field in the data.
        - Updates the ``data_size`` attribute with the length of the data.

        **Examples:**

        >>> ts.update()
        """
        self._set_frequency()
        self.start = self.data[self.dtfield].min()
        self.end = self.data[self.dtfield].max()
        self.var_min = self.data[self.varfield].min()
        self.var_max = self.data[self.varfield].max()
        self.data_size = len(self.data)
        self._set_view_specs()
        return None

    # docs: ok
    def update_epochs_stats(self):
        """Update all epochs statistics.

        :return: None
        :rtype: None

        **Notes:**

        - This function updates statistics for all epochs in the time series.
        - Ensures that the data is standardized by calling the ``standardize`` method if it's not already standardized.
        - Removes epoch 0 from the statistics since it typically represents non-standardized or invalid data.
        - Groups the data by ``Epoch_Id`` and calculates statistics such as count, start, and end timestamps for each epoch.
        - Generates random colors for each epoch using the ``get_random_colors`` function with a specified colormap (`cmap`` attribute).
        - Includes the time series name in the statistics for identification.
        - Organizes the statistics DataFrame to include relevant columns: ``Name``, ``Epoch_Id``, ``Count``, ``Start``, ``End``, and ``Color``.
        - Updates the attribute ``epochs_n`` with the number of epochs in the statistics.

        **Examples:**

        >>> ts.update_epochs_stats()

        """

        # Ensure data is standardized
        if not self.isstandard:
            self.standardize()

        # Get epochs
        df = self.get_epochs(inplace=False)

        # Remove epoch = 0
        df.drop(df[df["Epoch_Id"] == 0].index, inplace=True)
        df["Gap"] = 1 * df[self.varfield].isna().values

        # Group by
        self.epochs_stats = (
            df.groupby("Epoch_Id")
            .agg(
                Count=("Epoch_Id", "count"),
                Start=(self.dtfield, "min"),
                End=(self.dtfield, "max"),
                Gaps=("Gap", "sum")
            )
            .reset_index()
        )

        # Get colors
        self.epochs_stats["Color"] = get_random_colors(
            size=len(self.epochs_stats), cmap=self.cmap
        )

        # Include name
        self.epochs_stats["Name"] = self.name

        # Organize
        self.epochs_stats = self.epochs_stats[
            ["Name", "Epoch_Id", "Count", "Gaps", "Start", "End", "Color"]
        ]

        # Update attributes
        self.epochs_n = len(self.epochs_stats)
        self.smallgaps_n = self.epochs_stats["Gaps"].sum()

        return None

    # docs: ok
    def interpolate_gaps(self, method="linear", constant=0, inplace=False):
        """Fill gaps in a time series by interpolating missing values. If the time series is not in standard form, it will be standardized before interpolation.

        :param method: str, optional
            Specifies the interpolation method. Default is ``linear`.
            Other supported methods include ``constant``, ``nearest``, ``zero``, ``slinear``, ``quadratic``, ``cubic``, etc.
            The ``constant`` method applies a constant value to gaps.
            Refer to the documentation of scipy.interpolate.interp1d for more options.
        :type method: str

        :param constant: float, optional
            Value used if the case of ``constant`` method. Default is 0.
        :type constant: float

        :param inplace: bool, optional
            If True, the interpolation will be performed in-place, and the original data will be modified.
            If False, a new DataFrame with interpolated values will be returned, and the original data will remain unchanged.
            Default is False.
        :type inplace: bool

        :return: :class:`pandas.DataFrame``` or None
            If inplace is False, a new DataFrame with interpolated values.
            If inplace is True, returns None, and the original data is modified in-place.
        :rtype: :class:`pandas.DataFrame``` or None

        **Notes:**

        - The interpolation is performed for each unique epoch in the time series.
        - The ``method`` parameter determines the interpolation technique. Common options include ``constant``, ``linear``, ``nearest``, ``zero``, ``slinear``, ``quadratic``, and ``cubic`. See the documentation of scipy.interpolate.interp1d for additional methods and details.
        - If ``linear`` is chosen, the interpolation is a linear interpolation. For ``nearest``, it uses the value of the nearest data point. ``zero`` uses zero-order interpolation (nearest-neighbor). ``slinear`` and ``quadratic`` are spline interpolations of first and second order, respectively. ``cubic`` is a cubic spline interpolation.
        - If the method is ``linear``, the fill_value parameter is set to ``extrapolate`` to allow extrapolation beyond the data range.

        **Examples:**

        >>> ts.interpolate_gaps(method="linear", inplace=True)

        >>> interpolated_ts = ts.interpolate_gaps(method="linear", inplace=False)

        >>> interpolated_ts = ts.interpolate_gaps(method="constant", constant=1, inplace=False)
        """
        from scipy.interpolate import interp1d

        # Ensure data is standardized
        if not self.isstandard:
            self.standardize()

        # Get epochs for interpolation
        df = self.get_epochs(inplace=False)
        epochs = df["Epoch_Id"].unique()
        list_dfs = list()

        for epoch in epochs:
            df_aux1 = df.query("Epoch_Id == {}".format(epoch)).copy()
            if epoch == 0:
                df_aux1["{}_interp".format(self.varfield)] = np.nan
            else:
                if method == "constant":
                    df_aux1["{}_interp".format(self.varfield)] = np.where(
                        df_aux1[self.varfield].isna(),
                        constant,
                        df_aux1[self.varfield].values,
                    )
                else:  # use scipy methods
                    # clear from nan values
                    df_aux2 = df_aux1.dropna().copy()
                    # Create an interpolation function
                    interpolation_func = interp1d(
                        df_aux2[self.dtfield].astype(np.int64).values,
                        df_aux2[self.varfield].values,
                        kind=method,
                        fill_value="extrapolate",
                    )
                    # Interpolate full values
                    df_aux1["{}_interp".format(self.varfield)] = interpolation_func(
                        df_aux1[self.dtfield].astype(np.int64)
                    )
            # Append
            list_dfs.append(df_aux1)
        df_new = pd.concat(list_dfs, ignore_index=True)
        df_new = df_new.sort_values(by=self.dtfield).reset_index(drop=True)

        if inplace:
            self.data[self.varfield] = df_new["{}_interp".format(self.varfield)].values
            self.update()
            return None
        else:
            return df_new

    # docs: ok
    def cut_edges(self, inplace=False):
        """Cut off initial and final NaN records in a given time series.

        :param inplace: bool, optional
            If True, the operation will be performed in-place, and the original data will be modified.
            If False, a new DataFrame with cut edges will be returned, and the original data will remain unchanged.
            Default is False.
        :type inplace: bool

        :return: :class:`pandas.DataFrame``` or None
            If inplace is False, a new DataFrame with cut edges.
            If inplace is True, returns None, and the original data is modified in-place.
        :rtype: :class:`pandas.DataFrame``` or None

        **Notes:**

        - This function removes leading and trailing rows with NaN values in the specified variable field.
        - The operation is performed on a copy of the original data, and the original data remains unchanged.

        **Examples:**

        >>> ts.cut_edges(inplace=True)

        >>> trimmed_ts = ts.cut_edges(inplace=False)
        """
        # get dataframe
        in_df = self.data.copy()
        def_len = len(in_df)
        # drop first nan lines
        drop_ids = list()
        # loop to collect indexes in the start of series
        for def_i in range(def_len):
            aux = in_df[self.varfield].isnull().iloc[def_i]
            if aux:
                drop_ids.append(def_i)
            else:
                break
        # loop to collect indexes in the end of series
        for def_i in range(def_len - 1, -1, -1):
            aux = in_df[self.varfield].isnull().iloc[def_i]
            if aux:
                drop_ids.append(def_i)
            else:
                break
        # loop to drop rows:
        for def_i in range(len(drop_ids)):
            in_df.drop(drop_ids[def_i], inplace=True)
        # output
        if inplace:
            self.data = in_df.copy()
            return None
        else:
            return in_df


    def clear_outliers(self, inplace=False):
        # todo docstring
        # copy local data
        vct = self.data[self.varfield].values
        # Clear lower values
        if self.datarange_min is not None:
            vct = np.where(vct < self.datarange_min, np.nan, vct)
        # Clear upper values
        if self.datarange_max is not None:
            vct = np.where(vct > self.datarange_max, np.nan, vct)
        # return case
        if inplace:
            # overwrite local data
            self.data[self.varfield] = vct
            return None
        else:
            return df

    # docs: ok
    def aggregate(self, freq, bad_max, agg_funcs=None):
        """ "Aggregate the time series data based on a specified frequency using various aggregation functions.

        :param freq: str
            Pandas-like alias frequency at which to aggregate the time series data. Common options include:
            - ``H`` for hourly frequency
            - ``D`` for daily frequency
            - ``W`` for weekly frequency
            - ``MS`` for monthly/start frequency
            - ``QS`` for quarterly/start frequency
            - ``YS`` for yearly/start frequency
            More options and details can be found in the Pandas documentation:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
        :type freq: str

        :param bad_max: int
            The maximum number of ``Bad`` records allowed in a time window for aggregation. Records with more ``Bad`` entries
            will be excluded from the aggregated result.
            Default is 7.
        :type bad_max: int

        :param agg_funcs: dict, optional
            A dictionary specifying customized aggregation functions for each variable.
            Default is None, which uses standard aggregation functions (sum, mean, median, min, max, std, var, percentiles).
        :type agg_funcs: dict

        :return: :class:`pandas.DataFrame`
            A new :class:`pandas.DataFrame` with aggregated values based on the specified frequency.
        :rtype: :class:`pandas.DataFrame`

        **Notes:**

        - Resamples the time series data to the specified frequency using Pandas-like alias strings.
        - Aggregates the values using the specified aggregation functions.
        - Counts the number of ``Bad`` records in each time window and excludes time windows with more ``Bad`` entries
          than the specified threshold.

        **Examples:**

        >>> agg_result = ts.aggregate(freq='D', agg_funcs={'sum': 'sum', 'mean': 'mean'}, bad_max=5)

        """

        if agg_funcs is None:
            # Create a dictionary of standard and custom aggregation functions
            agg_funcs = {
                "sum": "sum",
                "mean": "mean",
                "median": "median",
                "min": "min",
                "max": "max",
                "std": "std",
                "var": "var",
            }
            '''
            # Add custom percentiles to the dictionary
            percentiles_to_compute = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles_to_compute:
                agg_funcs[f"p{p}"] = lambda x, p=p: np.percentile(x, p)
            '''

        # set list of tuples
        agg_funcs_list = [
            ("{}_{}".format(self.varfield, f), agg_funcs[f]) for f in agg_funcs
        ]

        # get data
        df = self.data.copy()
        df["Bad"] = df[self.varfield].isna().astype(int)
        # Set the 'datetime' column as the index
        df.set_index(self.dtfield, inplace=True)

        # Resample the time series to a frequency using aggregation functions
        agg_df1 = df.resample(freq)[self.varfield].agg(agg_funcs_list)
        agg_df2 = df.resample(freq)["Bad"].agg([("Bad_count", "sum")])

        # Reset the index to get 'DateTime' as a regular column
        agg_df1.reset_index(inplace=True)
        agg_df2.reset_index(inplace=True)

        # merge with bad dates
        agg_df = pd.merge(left=agg_df1, right=agg_df2, how="left", on=self.dtfield)

        # set new df
        agg_df_new = pd.DataFrame(
            {
                self.dtfield: pd.date_range(
                    start=agg_df[self.dtfield].values[0],
                    end=agg_df[self.dtfield].values[-1],
                    freq=freq,
                )
            }
        )
        # remove bad records
        agg_df.drop(agg_df[agg_df["Bad_count"] > bad_max].index, inplace=True)
        # remove bad column
        agg_df.drop(columns=["Bad_count"], inplace=True)

        # left join
        agg_df_new = pd.merge(
            left=agg_df_new, right=agg_df, on=self.dtfield, how="left"
        )

        return agg_df_new

    def upscale(self, freq, bad_max):
        # todo docstring
        df_upscale = self.aggregate(freq=freq, bad_max=bad_max, agg_funcs={self.agg: self.agg})
        df_upscale = df_upscale.rename(columns={"{}_{}".format(self.varfield, self.agg): self.varfield})
        return df_upscale

    # todo method
    def downscale(self, freq):
        # todo dosctring
        # 0) update
        self.update()
        # 1) get new index
        dt_index = pd.date_range(start=self.start, end=self.end, freq=freq)
        print(dt_index[:5])
        # 2) factor
        factor_downscale = len(dt_index) / self.data_size
        print(factor_downscale)


    def _set_frequency(self):
        """Guess the datetime resolution of a time series based on the consistency of
        timestamp components (e.g., seconds, minutes).

        :return: None
        :rtype: None

        **Notes:**

        - This method infers the datetime frequency of the time series data based on the consistency of timestamp components.

        **Examples:**

        >>> ts._set_frequency()

        """
        # Handle void data
        if self.data is None:
            pass
        else:
            # Copy the data
            df = self.data.copy()

            # Extract components of the datetime
            df["year"] = df[self.dtfield].dt.year
            df["month"] = df[self.dtfield].dt.month
            df["day"] = df[self.dtfield].dt.day
            df["hour"] = df[self.dtfield].dt.hour
            df["minute"] = df[self.dtfield].dt.minute
            df["second"] = df[self.dtfield].dt.second

            # Check consistency within each unit of time
            if df["second"].nunique() > 1:
                self.dtfreq = "1min"
                self.dtres = "second"
            elif df["minute"].nunique() > 1:
                self.dtfreq = "20min"
                self.dtres = "minute"
            elif df["hour"].nunique() > 1:
                self.dtfreq = "H"
                self.dtres = "hour"
            elif df["day"].nunique() > 1:
                self.dtfreq = "D"
                self.dtres = "day"
                # force gapsize to 1 when daily+ time scale
                self.gapsize = 1
            elif df["month"].nunique() > 1:
                self.dtfreq = "MS"
                self.dtres = "month"
                # force gapsize to 1 when daily+ time scale
                self.gapsize = 1
            else:
                self.dtfreq = "YS"
                self.dtres = "year"
                # force gapsize to 1 when daily+ time scale
                self.gapsize = 1

        return None

    # docs: todo
    def _set_view_specs(self):
        self.view_specs = {
            "title": "{} | {} ({})".format(self.name, self.varname, self.varfield),
            "width": 5 * 1.618,
            "height": 3,
            "xlabel": "Date",
            "ylabel": self.units,
            "xmin": self.start,
            "xmax": self.end,
            "ymin": 0,
            "ymax": self.var_max,
        }
        return None


    def view_epochs(self,
        show=True,
        folder="./output",
        filename=None,
        dpi=300,
        fig_format="jpg",
        suff="",
        ):
        # todo docstring
        specs = self.view_specs.copy()
        # Deploy figure
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height

        # preprocessing
        if self.epochs_stats is None:
            self.update_epochs_stats()

        gaps_c = "tab:red"
        gaps_a = 0.6
        # plot loop
        for i in range(len(self.epochs_stats)):
            start = self.epochs_stats["Start"].values[i]
            end = self.epochs_stats["End"].values[i]
            df_aux = self.data.query(
                "{} >= '{}' and {} < '{}'".format(
                    self.dtfield, start, self.dtfield, end
                )
            )
            epoch_c = self.epochs_stats["Color"].values[i]
            epoch_id = self.epochs_stats["Epoch_Id"].values[i]
            plt.plot(
                df_aux[self.dtfield],
                df_aux[self.varfield],
                ".",
                color=epoch_c,
                label=f"Epoch_{epoch_id}",
            )
            # Fill the space where there are missing values
            plt.fill_between(
                df_aux[self.dtfield],
                y1=0,
                y2=1.2 * specs["ymax"],
                where=df_aux[self.varfield].isna(),
                color=gaps_c,
                alpha=gaps_a,
            )
        # Create a dummy plot for missing data symbol
        plt.plot([], [], color=gaps_c, alpha=gaps_a, label='Gaps')
        if self.epochs_n <= 12:
            plt.legend(frameon=True, ncol=3)

        # basic plot stuff
        plt.title(specs["title"] + " | Gap Size: {} | Epochs: {}".format(self.gapsize, self.epochs_n))
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["xlabel"])
        plt.xlim(specs["xmin"], specs["xmax"])
        plt.ylim(specs["ymin"], 1.2 * specs["ymax"])

        # Adjust layout to prevent cutoff
        plt.tight_layout()

        # show or save
        if show:
            plt.show()
            return None
        else:
            if filename is None:
                filename = "{}_{}{}_epochs".format(self.varname.lower(), self.alias, suff)
            file_path = "{}/{}.{}".format(folder, filename, fig_format)
            plt.savefig(file_path, dpi=dpi)
            plt.close(fig)
            return file_path

    # docs: OK
    def view(
        self,
        show=True,
        folder="./output",
        filename=None,
        dpi=300,
        fig_format="jpg",
        suff="",
    ):
        """Visualize the time series data using a scatter plot with colored epochs.

        :param show: bool, optional
            If True, the plot will be displayed interactively.
            If False, the plot will be saved to a file.
            Default is True.
        :type show: bool

        :param folder: str, optional
            The folder where the plot file will be saved. Used only if show is False.
            Default is "./output".
        :type folder: str

        :param filename: str, optional
            The base name of the plot file. Used only if show is False. If None, a default filename is generated.
            Default is None.
        :type filename: str or None

        :param dpi: int, optional
            The dots per inch (resolution) of the plot file. Used only if show is False.
            Default is 300.
        :type dpi: int

        :param fig_format: str, optional
            The format of the plot file. Used only if show is False.
            Default is "jpg".
        :type fig_format: str

        :param raw: bool, optional
            Option for considering a raw data series. No epochs analysis.
            Default is False.
        :type raw: str

        :return: None
            If show is True, the plot is displayed interactively.
            If show is False, the plot is saved to a file.
        :rtype: None

        **Notes:**

        This function generates a scatter plot with colored epochs based on the epochs' start and end times.
        The plot includes data points within each epoch, and each epoch is labeled with its corresponding ID.

        **Examples:**

        >>> ts.view(show=True)

        >>> ts.view(show=False, folder="./output", filename="time_series_plot", dpi=300, fig_format="png")
        """
        specs = self.view_specs.copy()
        # Deploy figure
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height

        plt.plot(
            self.data[self.dtfield],
            self.data[self.varfield],
            ".",
            color=self.rawcolor,
        )

        # basic plot stuff
        plt.title(specs["title"])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["xlabel"])
        plt.xlim(specs["xmin"], specs["xmax"])
        plt.ylim(specs["ymin"], 1.2 * specs["ymax"])

        # Adjust layout to prevent cutoff
        plt.tight_layout()

        # show or save
        if show:
            plt.show()
            return None
        else:
            if filename is None:
                filename = "{}_{}{}".format(self.varname.lower(), self.alias, suff)
            file_path = "{}/{}.{}".format(folder, filename, fig_format)
            plt.savefig(file_path, dpi=dpi)
            plt.close(fig)
            return file_path

class TimeSeriesCollection(Collection):
    """A collection of time series objects with associated metadata.

    The ``TimeSeriesCollection`` or simply TSC class extends the ``Collection`` class and is designed to handle time series data.

    """

    # docs: ok
    def __init__(self, name="myTSCollection", base_object=None):
        """Deploy the time series collection data structure.

        :param name: str, optional
            Name of the time series collection.
            Default is "myTSCollection".
        :type name: str

        :param base_object: TimeSeries, optional
            Base object for the time series collection.
            If None, a default TimeSeries object is created.
            Default is None.
        :type base_object: TimeSeries or None

        **Notes:**

        - If ``base_object`` is not provided, a default ``TimeSeries`` object is created.

        **Examples:**

        >>> ts_collection = TimeSeriesCollection(name="MyTSCollection")

        """

        # If base_object is not provided, create a default TimeSeries object
        if base_object is None:
            base_object = TimeSeries

        # Call the constructor of the parent class (TimeSeries)
        super().__init__(base_object=base_object, name=name)

        # Set up date fields and special attributes in the catalog
        self.catalog["Start"] = pd.to_datetime(
            self.catalog["Start"], format="%Y-%m-%d %H:%M:%S"
        )
        self.catalog["End"] = pd.to_datetime(
            self.catalog["End"], format="%Y-%m-%d %H:%M:%S"
        )

        # Set auto attributes
        self.start = None
        self.end = None
        self.var_min = None
        self.var_max = None
        self.isstandard = False
        self.datarange_min = None
        self.datarange_max = None

        # load view specs
        self._set_view_specs()

        # Set specific attributes
        self.name_object = "Time Series Collection"
        self.dtfield = "DateTime"
        self.overfield = "Overlapping"

    def __str__(self):
        return self.catalog.to_string(index=False)

    # docs: ok
    def update(self, details=False):
        """Update the time series collection.

        :param details: bool, optional
            If True, update additional details.
            Default is False.
        :type details: bool

        **Examples:**

        >>> ts_collection.update(details=True)
        """
        # Call the update method of the parent class (Collection)
        super().update(details=details)
        # Update start, end, min, and max attributes based on catalog information
        self.start = self.catalog["Start"].min()
        self.end = self.catalog["End"].max()
        self.var_min = self.catalog["Min"].min()
        self.var_max = self.catalog["Max"].max()

    # docs: ok
    def load_data(self, table_file, filter_dates=None):
        """Load data from table file (information table) into the time series collection.

        :param table_file: str
            Path to file. Expected to be a ``csv`` table.

            todo place this in IO files docs
            Required columns:

            - ``Id``: int, required. Unique number id.
            - ``Name``: str, required. Simple name.
            - ``Alias``: str, required. Short nickname.
            - ``X``: float, required. Longitude in WGS 84 Datum (EPSG4326).
            - ``Y``: float, required. Latitude in WGS 84 Datum (EPSG4326).
            - ``Code``: str, required.
            - ``Source``: str, required.
            - ``Description``: str, required.
            - ``Color``: str, required.
            - ``Units`` or ``<Varname>_Units``: str, required. Units of data.
            - ``VarField``or ``<Varname>_VarField``: str, required. Variable column in data file.
            - ``DtField``or ``<Varname>_DtField``: str, required. Date-time column in data file
            - ``File``or ``<Varname>_File``: str, required. Name or path to data time series ``csv`` file.

        :type table_file: str

        **Examples:**

        >>> ts_collection.load_data(table_file="data.csv")
        """
        input_df = pd.read_csv(table_file, sep=";")
        input_dir = os.path.dirname(table_file)
        self.set_data(df_info=input_df, src_dir=input_dir, filter_dates=filter_dates)
        return None

    # docs: ok
    def set_data(self, df_info, src_dir=None, filter_dates=None):
        """Set data for the time series collection from a info DataFrame.

        :param df_info: class:`pandas.DataFrame`
            DataFrame containing metadata information for the time series collection.
            This DataFrame is expected to have matching fields to the metadata keys.

            Required fields:

            - ``Id``: int, required. Unique number id.
            - ``Name``: str, required. Simple name.
            - ``Alias``: str, required. Short nickname.
            - ``X``: float, required. Longitude in WGS 84 Datum (EPSG4326).
            - ``Y``: float, required. Latitude in WGS 84 Datum (EPSG4326).
            - ``Code``: str, required
            - ``Source``: str, required
            - ``Description``: str, required
            - ``Color``: str, optional
            - ``Units`` or ``<Varname>_Units``: str, required. Units of data.
            - ``VarField``or ``<Varname>_VarField``: str, required. Variable column in data file.
            - ``DtField``or ``<Varname>_DtField``: str, required. Date-time column in data file
            - ``File``or ``<Varname>_File``: str, required. Name or path to data time series ``csv`` file.


        :type df_info: class:`pandas.DataFrame`

        :param src_dir: str, optional
            Path for input directory in the case for only file names in ``File`` column.
        :type src_dir: str

        :param filter_dates: list, optional
            List of Start and End dates for filter data
        :type filter_dates: str

        **Notes:**

        - The ``set_data`` method populates the time series collection with data based on the provided DataFrame.
        - It creates time series objects, loads data, and performs additional processing steps.
        - Adjust ``skip_process`` according to your data processing needs.

        **Examples:**

        >>> ts_collection.set_data(df, "path/to/data", filter_dates=["2020-01-01 00:00:00", "2020-03-12 00:00:00"])

        """

        # handle mutable fields
        str_varname = self.baseobject().varname
        list_mutables = ["Units", "DtField", "VarField", "File"]
        dict_m_fields = {}
        for m in list_mutables:
            if m in df_info.columns:
                # assume <Varname>_ prefix:
                dict_m_fields[m] = m
            else:
                dict_m_fields[m] = "{}_{}".format(str_varname, m)

        for i in range(len(df_info)):
            # create object
            ts = self.baseobject(name=df_info["Name"].values[i])
            ts.alias = df_info["Alias"].values[i]
            ts.units = df_info[dict_m_fields["Units"]].values[i]
            # handle input file
            input_file = df_info[dict_m_fields["File"]].values[i]
            # if is a path:
            if os.path.exists(input_file):
                pass
            else:
                # assume file name only and concat
                if src_dir is not None:
                    input_file = "{}/{}".format(src_dir, input_file)
            # load data
            ts.load_data(
                table_file=input_file,
                input_varfield=df_info[dict_m_fields["VarField"]].values[i],
                input_dtfield=df_info[dict_m_fields["DtField"]].values[i],
                filter_dates=filter_dates,
            )
            # extra attrs
            ts.x = df_info["X"].values[i]
            ts.y = df_info["Y"].values[i]
            ts.code = df_info["Code"].values[i]
            ts.source_data = df_info["Source"].values[i]
            ts.description = df_info["Description"].values[i]
            ts.color = df_info["Color"].values[i]
            # append
            self.append(new_object=ts)
        self.update(details=True)
        return None

    # docs: ok
    def clear_outliers(self):
        """Clear outliers in the collection based on the ``datarange_min`` and ``datarange_max`` attributes.

        :return: None
        :rtype: None
        """
        for name in self.collection:
            self.collection[name].datarange_min = self.datarange_min
            self.collection[name].datarange_max = self.datarange_max
            self.collection[name].clear_outliers(inplace=True)
        return None

    # docs: ok
    def merge_data(self):
        """Merge data from multiple sources into a single DataFrame.

        :return: DataFrame
            A merged DataFrame with datetime and variable fields from different sources.
        :rtype: pandas.DataFrame

        **Notes:**
        - Updates the catalog details.
        - Merges data from different sources based on the specified datetime field and variable field.
        - The merged DataFrame includes a date range covering the entire period.

        **Examples:**

        >>> merged_df = ts.merge_data()

        """
        # Ensure catalog is updated with details
        self.update(details=True)

        # Get start, end, and frequency from the catalog
        start = self.start
        end = self.end

        # todo handle this issue on the first freq
        # consider the first freq
        freq = self.catalog["DT_freq"].values[0]

        # Create a date range for the entire period
        dt_index = pd.date_range(start=start, end=end, freq=freq)
        df = pd.DataFrame({self.dtfield: dt_index})

        # Merging loop for each catalog entry
        for i in range(len(self.catalog)):
            # Get attributes from the catalog
            name = self.catalog["Name"].values[i]
            alias = self.catalog["Alias"].values[i]
            varfield = self.catalog["VarField"].values[i]
            dtfield = self.catalog["DtField"].values[i]

            # Handle datetime field and set up right DataFrame
            b_drop = False
            if self.dtfield == dtfield:
                suffs = ("", "")
                dt_right_after = dtfield
            else:
                suffs = ("", "_right")
                b_drop = True

            df_right = self.collection[name].data[[dtfield, varfield]].copy()
            df_right = df_right.rename(
                columns={varfield: "{}_{}".format(varfield, alias)}
            )

            # Left join the DataFrames
            df = pd.merge(
                how="left",
                left=df,
                left_on=self.dtfield,
                right=df_right,
                right_on=dtfield,
                suffixes=suffs,
            )

            # Clear the right datetime column if needed
            if b_drop:
                df = df.drop(columns=[dt_right_after])
        return df

    # docs: ok
    def standardize(self):
        """Standardize the time series data.

        This method standardizes all time series objects in the collection.

        :return: None
        :rtype: None

        **Notes:**

        - The method iterates through each time series in the collection and standardizes it.
        - After standardizing individual time series, the data is merged.
        - The merged data is then reset for each individual time series in the collection.
        - Epoch statistics are updated for each time series after the reset.
        - Finally, the collection catalog is updated with details.

        **Examples:**

        >>> ts_collection = TimeSeriesCollection()
        >>> ts_collection.standardize()

        """
        # Standardize and fill gaps across all series [overwrite]
        for name in self.collection:
            self.collection[name].standardize()
            self.collection[name].interpolate_gaps(inplace=True)

        # Merge data
        df = self.merge_data()

        # helper dict -- alias is the key
        dict_names = dict()
        for i in range(len(self.catalog)):
            a = self.catalog["Alias"].values[i]
            dict_names[a] = self.catalog["Name"].values[i]

        # Reset data for each time series
        for c in df.columns[1:]:
            # filter merged dataframe
            df_aux = df[[self.dtfield, c]].copy()

            alias = c.split("_")[1]
            name = dict_names[alias]
            # set data
            self.collection[name].set_data(
                df_info=df_aux,
                input_dtfield=self.dtfield,
                input_varfield=c,
                dropnan=False,
            )

        # Update epochs statistics for each time series
        for name in self.collection:
            self.collection[name].update_epochs_stats()

        # Update the collection catalog with details
        self.update(details=True)

        # Set standard flag
        self.isstandard = True

        return None

    # docs: ok
    def merge_local_epochs(self):
        """Merge local epochs statistics from individual time series within the collection.

        :return: Merged :class:`pandas.DataFrame``` containing epochs statistics.
        :rtype: :class:`pandas.DataFrame`

        **Notes:**

        - This method creates an empty list to store individual epochs statistics dataframes.
        - It iterates through each time series in the collection.
        - For each time series, it updates the local epochs statistics using the :meth:``update_epochs_stats`` method.
        - The local epochs statistics dataframes are then appended to the list.
        - Finally, the list of dataframes is concatenated into a single dataframe.

        **Examples:**

        >>> ts_collection = TimeSeriesCollection()
        >>> merged_epochs = ts_collection.merge_local_epochs()

        """
        # Create an empty list to store individual epochs statistics dataframes
        lst_df = list()

        # Iterate through each time series in the collection
        for name in self.collection:
            # Update epochs statistics for the current time series
            self.collection[name].update_epochs_stats()

            # Append the epochs statistics dataframe to the list
            lst_df.append(self.collection[name].epochs_stats.copy())

        # Concatenate the list of dataframes into a single dataframe
        df = pd.concat(lst_df).reset_index(drop=True)

        return df

    # docs: ok
    def get_epochs(self):
        """Calculate epochs for the time series data.

        :return: DataFrame with epochs information.
        :rtype: :class:`pandas.DataFrame`

        **Notes:**

        - This method merges the time series data to create a working DataFrame.
        - It creates a copy of the DataFrame for NaN-value calculation.
        - Converts non-NaN values to 1 and NaN values to 0.
        - Calculates the sum of non-NaN values for each row and updates the DataFrame with the result.
        - Extracts relevant columns for epoch calculation.
        - Sets 0 values in the specified ``overfield`` column to NaN.
        - Creates a new :class:`TimeSeries`` instance for epoch calculation using the ``overfield`` values.
        - Calculates epochs using the :meth:``get_epochs`` method of the new :class:`TimeSeries`` instance.
        - Updates the original DataFrame with the calculated epochs.

        **Examples:**

        >>> epochs_data = ts_instance.get_epochs()

        """
        # Merge data to create a working DataFrame
        df = self.merge_data()

        # Create a copy for nan-value calculation
        df_nan = df.copy()

        # Convert non-NaN values to 1, and NaN values to 0
        for v in df_nan.columns[1:]:
            df_nan[v] = df_nan[v].notna().astype(int)

        # Calculate the sum of non-NaN values for each row
        df_nan = df_nan.drop(columns=[self.dtfield])
        df[self.overfield] = df_nan.sum(axis=1)
        df[self.overfield] = df[self.overfield] / len(df_nan.columns)

        # Extract relevant columns for epoch calculation
        df_aux = df[[self.dtfield, self.overfield]].copy()

        # Set 0 values in the overfield column to NaN
        df_aux[self.overfield].replace(0, np.nan, inplace=True)

        # Create a new TimeSeries instance for epoch calculation
        ts_aux = TimeSeries(varfield=self.overfield)
        ts_aux.set_data(
            input_df=df_aux,
            input_varfield=self.overfield,
            input_dtfield=self.dtfield,
            dropnan=False,
        )

        # Calculate epochs and update the original DataFrame
        ts_aux.get_epochs(inplace=True)
        df["Epoch_Id"] = ts_aux.data["Epoch_Id"].values

        return df

    def _set_view_specs(self):
        self.view_specs = {
            "title": "{} | {}".format(self.name_object, self.name),
            "width": 8,
            "width_spacing": 1.5,
            "left": 0.1,
            "height": 5,
            "xlabel": "Date",
            "ylabel": "%",
            "gantt": "Gantt chart (epochs)",
            "prev": "Series prevalence",
            "over": "Overlapping data",
            "vmin": 0,
            "vmax": None,
            "ymin": 0,
            "ymax": None,
            "xmin": None,
            "xmax": None,
        }
        return None

    # docs: ok
    def view(
        self,
        show=True,
        folder="./output",
        filename=None,
        dpi=300,
        fig_format="jpg",
        suff="",
        usealias=False,
    ):
        """Visualize the time series collection.

        :param show: bool, optional
            If True, the plot will be displayed interactively.
            If False, the plot will be saved to a file.
            Default is True.
        :type show: bool

        :param folder: str, optional
            The folder where the plot file will be saved. Used only if show is False.
            Default is "./output".
        :type folder: str

        :param filename: str, optional
            The base name of the plot file. Used only if show is False. If None, a default filename is generated.
            Default is None.
        :type filename: str or None

        :param dpi: int, optional
            The dots per inch (resolution) of the plot file. Used only if show is False.
            Default is 300.
        :type dpi: int

        :param fig_format: str, optional
            The format of the plot file. Used only if show is False.
            Default is "jpg".
        :type fig_format: str

        :param usealias: bool, optional
            Option for using the Alias instead of Name in the plot.
            Default is False.
        :type usealias: bool

        :return: None
            If show is True, the plot is displayed interactively.
            If show is False, the plot is saved to a file.
        :rtype: None

        **Notes:**

        This function generates a scatter plot with colored epochs based on the epochs' start and end times.
        The plot includes data points within each epoch, and each epoch is labeled with its corresponding ID.

        **Examples:**

        >>> ts.view(show=True)

        >>> ts.view(show=False, folder="./output", filename="time_series_plot", dpi=300, fig_format="png")
        """
        if usealias:
            _names = self.catalog["Name"].values
            _alias = self.catalog["Alias"].values
            dict_alias = dict()
            for i in range(len(self.catalog)):
                dict_alias[_names[i]] = _alias[i]

        # pre-processing
        local_epochs_df = self.merge_local_epochs()

        # Aggregate sum based on the 'Name' field
        agg_df = local_epochs_df.groupby("Name")["Count"].sum().reset_index()
        agg_df = agg_df.sort_values(by="Count", ascending=True).reset_index(drop=True)
        names = agg_df["Name"].values
        if usealias:
            alias = [dict_alias[name] for name in names]
        preval = 100 * agg_df["Count"].values / agg_df["Count"].sum()
        heights = np.linspace(0, 1, len(names) + 1)
        heights = heights[: len(names)] + ((heights[1] - heights[0]) / 2)

        # get epochs
        epochs_df = self.get_epochs()

        # Assuming date_values is a list of datetime objects
        date_values = epochs_df[self.dtfield].values
        # Calculate the number of ticks you want (e.g., 5)
        num_ticks = 5
        # Calculate the step size to evenly space the ticks
        step_size = len(date_values) // (num_ticks - 1)
        # Generate indices for the ticks
        tick_indices = np.arange(0, len(date_values), step_size)
        # Select the corresponding dates for the ticks
        ticks = [date_values[i] for i in tick_indices]

        # plot
        specs = self.view_specs.copy()

        # Deploy figure
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        gs = mpl.gridspec.GridSpec(
            2,
            6,
            wspace=specs["width_spacing"],
            hspace=0.5,
            left=specs["left"],
            bottom=0.1,
            top=0.85,
            right=0.95,
        )
        fig.suptitle(specs["title"])

        # Gantt chart
        plt.subplot(gs[:1, :4])
        plt.title("a. {}".format(specs["gantt"]), loc="left")
        for i in range(len(names)):
            name = names[i]
            df_aux = local_epochs_df.query(f"Name == '{name}'")
            for j in range(len(df_aux)):
                x_time = [df_aux["Start"].values[j], df_aux["End"].values[j]]
                y_height = [heights[i], heights[i]]
                plt.plot(
                    x_time,
                    y_height,
                    color=df_aux["Color"].values[j],
                    linewidth=6,
                    solid_capstyle="butt",
                )
        plt.ylim(0, 1)
        plt.xlim((self.start, self.end))
        labels = names
        if usealias:
            labels = alias
        plt.yticks(heights, labels)
        plt.xlabel(specs["xlabel"])
        plt.xticks(ticks)
        plt.grid(axis="y")

        # overlapping chart
        plt.subplot(gs[1:, :4])
        plt.title("c. {}".format(specs["over"]), loc="left")
        plt.plot(
            epochs_df[self.dtfield], 100 * epochs_df[self.overfield], color="tab:blue"
        )
        plt.fill_between(
            epochs_df[self.dtfield],
            100 * epochs_df[self.overfield],
            color="tab:blue",
            alpha=0.4,
        )
        plt.ylim(-5, 105)
        plt.xlim((self.start, self.end))
        plt.xlabel(specs["xlabel"])
        plt.xticks(ticks)
        plt.ylabel("%")

        # Prevalence chart
        plt.subplot(gs[:1, 4:])
        plt.title("b. {}".format(specs["prev"]), loc="left")
        plt.barh(
            labels,
            preval,
            color="tab:gray",
        )
        # Add data values as text labels on the bars
        for index, value in enumerate(preval):
            plt.text(
                value,
                index,
                " {:.1f}%".format(value),
                ha="left",
                va="center",
                fontsize=10,
            )
        plt.xlim(0, 100)
        plt.grid(axis="x")
        plt.xlabel("%")

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = "{}{}".format(self.name, suff)
            plt.savefig("{}/{}.{}".format(folder, filename, fig_format), dpi=dpi)
            plt.close(fig)
        return None

    # docs: ok
    def export_views(
        self, folder, dpi=300, fig_format="jpg", suff="", skip_main=False, raw=False
    ):
        """Export views of time series data and individual time series within the collection.

        :param folder: str
            The folder path where the views will be exported.
        :type folder: str

        :param dpi: int, optional
            Dots per inch (resolution) for the exported images, default is 300.
        :type dpi: int

        :param fig_format: str, optional
            Format for the exported figures, default is "jpg".
        :type fig_format: str

        :param suff: str, optional
            Suffix to be appended to the exported file names, default is an empty string.
        :type suff: str

        :param skip_main: bool, optional
            Option for skipping the main plot (pannel)
        :type skip_main: bool

        :param raw: bool, optional
            Option for considering a raw data series. No epochs analysis.
            Default is False.
        :type raw: str

        :return: None
        :rtype: None

        **Notes:**

        - Updates the collection details and epoch statistics.
        - Calls the ``view`` method for the entire collection and individual time series with specified parameters.
        - Sets view specifications for individual time series, such as y-axis limits and time range.

        **Examples:**

        >>> tscoll.export_views(folder="/path/to/export", dpi=300, fig_format="jpg", suff="_views")

        """
        self.update(details=True)

        for name in self.collection:
            self.collection[name].update_epochs_stats()

        if skip_main:
            pass
        else:
            self.view(
                show=False, folder=folder, dpi=dpi, fig_format=fig_format, suff=suff
            )

        for name in self.collection:
            # specs
            self.collection[name].view_specs["ymax"] = self.var_max
            self.collection[name].view_specs["ymin"] = 0
            self.collection[name].view_specs["xmax"] = self.end
            self.collection[name].view_specs["xmin"] = self.start
            # view
            self.collection[name].view(
                show=False,
                folder=folder,
                dpi=dpi,
                fig_format=fig_format,
                suff=suff,
                raw=raw,
            )
        return None

    # docs todo
    def export_data(self, folder, filename=None, merged=True):
        if filename is None:
            filename = self.name
        if merged:
            df = self.get_epochs()
            df.to_csv("{}/{}.csv".format(folder, filename), sep=";", index=False)
            df = self.merge_local_epochs()
            df.to_csv("{}/{}_epochs.csv".format(folder, filename), sep=";", index=False)
        else:
            for name in self.collection:
                df = self.collection[name].export(folder=folder)

class TimeSeriesCluster(TimeSeriesCollection):
    # todo docstring
    """
    The ``TimeSeriesCluster`` instance is desgined for holding a collection
    of same variable time series. That is, no miscellaneus data is allowed.

    """

    def __init__(self, name="myTimeSeriesCluster", base_object=None):
        # todo docstring
        # initialize parent
        super().__init__(name=name, base_object=base_object)

        # Set extra specific attributes
        self.varname = base_object().varname
        self.varalias = base_object().varalias

        # Overwrite parent attributes
        self.name_object = "Time Series Cluster"

    '''
    def append(self, new_object):
        # todo docstring
        # only append if new object is the same of the base
        print(type(new_object))
        print(type(self.baseobject))
        if type(new_object) != type(self.baseobject):
            raise ValueError("new object is not instances of the same class.")
        super().append(new_object=new_object)
        return None
    '''

class StageSeriesCollection(TimeSeriesCluster):
    # todo docstring
    def __init__(self, name="MySSColection"):
        # todo docstring
        super().__init__(name=name, base_object=StageSeries)
        # overwrite parent attributes
        self.name_object = "Stage Series Collection"
        self._set_view_specs()

    def set_data(self, df_info, src_dir=None, filter_dates=None):
        # todo docstring
        """Set data for the time series collection from a info DataFrame.

        :param df_info: class:`pandas.DataFrame`
            DataFrame containing metadata information for the time series collection.
            This DataFrame is expected to have matching fields to the metadata keys.

            Required fields:

            - ``Id``: int, required. Unique number id.
            - ``Name``: str, required. Simple name.
            - ``Alias``: str, required. Short nickname.
            - ``Units``: str, required. Units of data.
            - ``VarField``: str, required. Variable column in data file.
            - ``DtField``: str, required. Date-time column in data file
            - ``File``: str, required. Name or path to data time series ``csv`` file.
            - ``X``: float, required. Longitude in WGS 84 Datum (EPSG4326).
            - ``Y``: float, required. Latitude in WGS 84 Datum (EPSG4326).
            - ``Code``: str, required
            - ``Source``: str, required
            - ``Description``: str, required
            - ``Color``: str, required
            - ``UpstreamArea``: float, required


        :type df_info: class:`pandas.DataFrame`

        :param src_dir: str, optional
            Path for source directory in the case for only file names in ``File`` column.
        :type src_dir: str

        :param filter_dates: list, optional
            List of Start and End dates for filter data
        :type filter_dates: str

        **Notes:**

        - The ``set_data`` method populates the time series collection with data based on the provided DataFrame.
        - It creates time series objects, loads data, and performs additional processing steps.
        - Adjust ``skip_process`` according to your data processing needs.

        **Examples:**

        >>> ts_collection.set_data(df, "path/to/data", filter_dates=["2020-01-01 00:00:00", "2020-03-12 00:00:00"])

        """
        # generic part
        super().set_data(df_info=df_info, filter_dates=filter_dates)
        # custom part
        for i in range(len(df_info)):
            name = df_info["Name"].values[i]
            self.collection[name].upstream_area = df_info["UpstreamArea"].values[i]
        self.update(details=True)
        return None

class TimeSeriesSamples(TimeSeriesCluster):
    # todo docstring
    """
    The ``TimeSeriesSamples`` instance is desgined for holding a collection
    of same variable time series arising from the same underlying process.
    This means that all elements in the collection are statistical samples.

    This instance allows for the ``reducer()`` method.

    """

    def __init__(self, name="myTimeSeriesSamples", base_object=None):
        # todo docstring
        # initialize parent
        super().__init__(name=name, base_object=base_object)
        # Overwrite parent attributes
        self.name_object = "Time Series Samples"

    # todo reducer
    def reducer(self, reducer_funcs=None, stepwise=False):
        # todo docstring

        df_merged = self.merge_data()

        # set up output dict
        dict_out = {self.dtfield: df_merged[self.dtfield].values}

        # get grid
        grd_data = df_merged[df_merged.columns[1:]].values

        for funcname in reducer_funcs:
            # set array
            vct_red = np.zeros(len(df_merged))
            # row loop options
            args = reducer_funcs[funcname]["Args"]
            if args is None:
                if stepwise:
                    for i in range(len(vct_red)):
                        vct_red[i] = reducer_funcs[funcname]["Func"](grd_data[i])
                else:
                    vct_red = reducer_funcs[funcname]["Func"](grd_data, axis=1)
            else:
                if stepwise:
                    for i in range(len(vct_red)):
                        vct_red[i] = reducer_funcs[funcname]["Func"](grd_data[i], args)
                else:
                    vct_red = reducer_funcs[funcname]["Func"](grd_data, args, axis=1)
            # append to dict
            dict_out["{}_{}".format(self.varalias, funcname)] = vct_red[:]

        # set dataframe
        df = pd.DataFrame(dict_out)

        return df

    def mean(self):
        # todo docstring
        df = self.reducer(reducer_funcs={"mean": {"Func": np.mean, "Args": None}})
        return df

    def rng(self):
        # todo docstring
        df = self.reducer(reducer_funcs={"rng": {"Func": np.r, "Args": None}})
        return df

    def std(self):
        # todo docstring
        df = self.reducer(reducer_funcs={"std": {"Func": np.std, "Args": None}})
        return df

    def min(self):
        # todo docstring
        df = self.reducer(reducer_funcs={"min": {"Func": np.min, "Args": None}})
        return df

    def max(self):
        # todo docstring
        df = self.reducer(reducer_funcs={"max": {"Func": np.max, "Args": None}})
        return df

    def percentile(self, p=90):
        # todo docstring
        df = self.reducer(reducer_funcs={"max": {"Func": np.percentile, "Args": p}})
        return df

    def percentiles(self, values=None):
        # todo docstring
        if values is None:
            values = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        dict_funcs = {}
        for v in values:
            if v < 10:
                dict_funcs["p0{}".format(v)] = {"Func": np.percentile, "Args": v}
            else:
                dict_funcs["p0{}".format(v)] = {"Func": np.percentile, "Args": v}

        df = self.reducer(reducer_funcs=dict_funcs)
        return df

    def stats(self, basic=False):
        # todo docstring
        df_stats = self.reducer(
            reducer_funcs={
                "sum": {"Func": np.sum, "Args": None},
                "mean": {"Func": np.mean, "Args": None},
                "std": {"Func": np.max, "Args": None},
                "min": {"Func": np.min, "Args": None},
                "max": {"Func": np.max, "Args": None},
            }
        )
        if basic:
            values = [25, 50, 75]
        # get percentiles
        df_per = self.percentiles(values=None)
        # join dataframes
        df = pd.merge(left=df_stats, right=df_per, on=self.dtfield)

        return df

class TimeSeriesSpatialSamples(TimeSeriesSamples):
    # todo docstring
    """
    The ``TimeSeriesSpatialSamples`` instance is desgined for holding a collection
    of same variable time series arising from the same underlying process in space.
    This means that all elements in the collection are statistical samples in space.

    This instance allows for the ``regionalization()`` method.

    """

    def __init__(self, name="myTimeSeriesSpatialSample", base_object=None):
        # todo docstring
        # initialize parent
        super().__init__(name=name, base_object=base_object)
        # Overwrite parent attributes
        self.name_object = "Time Series Spatial"

    def get_weights_by_name(self, name, method="average"):
        # todo docstring

        # remaining names
        df_remain = self.catalog.query("Name != '{}'".format(name))
        list_names = list(df_remain["Name"].values)

        # w array
        w = np.ones(len(list_names))

        # handle methods
        if method.lower() == "average":
            return w
        elif method.lower() == "idw":
            # Get x,y
            x = self.collection[name].x
            y = self.collection[name].y
            # Get distances
            x_i = np.array([self.collection[_].x for _ in list_names])
            y_i = np.array([self.collection[_].y for _ in list_names])
            d_i = np.sqrt(np.square(x - x_i) + np.square(y - y_i))
            # Compute IDW
            idw_i = 1 / (d_i + (np.mean(d_i) / 10000))  # handle zero distance
            # Normaliza to [0-100] range
            w = 100 * idw_i / np.sum(idw_i)
            return w
        else:
            return w

    # docs ok
    def regionalize(self, method="average"):
        """Regionalize the time series data using a specified method.

        :param method: str, optional
            Method for regionalization, default is "average".
        :type method: str

        :return: None
        :rtype: None

        **Notes:**

        - This method handles standardization. If the time series data is not standardized, it applies standardization.
        - Computes epochs for the time series data.
        - Iterates through each time series in the collection and performs regionalization.
        - For each time series, sets up source and destination vectors, computes weights, and calculates regionalized values.
        - Updates the destination column in-place with the regionalized values and updates epochs statistics.
        - Updates the collection catalog with details.

        **Examples:**

        >>> ts_instance = YourTimeSeriesClass()
        >>> ts_instance.regionalize(method="average")

        """
        # Handle stardardization
        if self.isstandard:
            pass
        else:
            self.standardize()
        # Compute Epochs
        df = self.get_epochs()

        for i in range(len(self.catalog)):
            varfield = self.catalog["VarField"].values[i]
            name = self.catalog["Name"].values[i]
            df_src_catalog = self.catalog.query("Name != '{}'".format(name))

            # Get destination vector
            dst_vct = self.collection[name].data[varfield].values
            dst_mask = 1 * np.isnan(dst_vct)

            # Set up source
            src_vars = df_src_catalog["VarField"].values
            src_names = df_src_catalog["Name"].values
            src_alias = df_src_catalog["Alias"].values
            src_columns = [
                "{}_{}".format(src_vars[i], src_alias[i])
                for i in range(len(df_src_catalog))
            ]
            df_src = df[src_columns].copy()  # ensure name order

            # Set up weights
            vct_w = self.get_weights_by_name(name=name, method=method)

            # get product w * values
            w_v = vct_w * ~np.isnan(df_src.values)
            # horizontal weights sum
            w_sum = np.sum(w_v, axis=1)
            w_sum = np.where(w_sum == 0, np.nan, w_sum)
            # output sum
            o_sum = np.nansum(df_src.values * w_v, axis=1)
            # output average
            o = o_sum / w_sum

            # set destination column inplace
            self.collection[name].data[varfield] = np.where(dst_mask == 1, o, dst_vct)
            self.collection[name].update()
            self.collection[name].update_epochs_stats()

        # Update the collection catalog with details
        self.update(details=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8")

    ts = TimeSeries()
    ts.varfield = "H"
    ts.varname = "Stage"
    ts.varalias = "Stg"
    ts.units = "cm"

    f = "C:/plans/docs/datasets/basins/stage_Tur.csv"

    setter_d = {
        "Name": "OOOps",
        "Alias": "ok",
        "File_Data": f,
        "Color": "red",
        "DtField": "DateTime",
        "VarField": "H_Tur_cm"
    }
    ts.gapsize = 60

    ts.set(dict_setter=setter_d, load_data=True)
    print(ts)
    ts.update_epochs_stats()
    print(ts.epochs_stats.to_string())
    #ts.view_epochs()
    plt.plot(ts.data[ts.dtfield], ts.data[ts.varfield], ".", color="magenta", zorder=2)
    ts.interpolate_gaps(method="cubic", inplace=True)
    plt.plot(ts.data[ts.dtfield], ts.data[ts.varfield], ".", color="blue", zorder=1)
    plt.show()
    #ts.view_epochs()
