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

def get_colors(size=10, cmap="tab20", randomize=True):
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
    if randomize:
        _lst_vals = np.random.rand(size)
    else:
        _lst_vals = np.linspace(0, 1, num=size)
    # Use the colormap to convert the random numbers to colors
    _lst_colors = [mcolors.to_hex(_cmap(x)) for x in _lst_vals]
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
        self.folder_src = None

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
            "linestyle": "."
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
        if self.data is not None:
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

    def load_data(self, file_data, input_dtfield=None, input_varfield=None, in_sep=";", filter_dates=None):
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
        df = pd.read_csv(
            file_data,
            sep=in_sep,
            dtype={input_dtfield: str, input_varfield: float},
            usecols=[input_dtfield, input_varfield],
            parse_dates=[self.file_data_dtfield]
        )

        # -------------- post-loading logic -------------- #
        df = df.rename(
            columns={
                input_dtfield: self.dtfield,
                input_varfield: self.varfield
            }
        )

        if filter_dates is None:
            pass
        else:
            if filter_dates[0] is not None:
                df = df.query("{} >= '{}'".format(self.dtfield, filter_dates[0]))
            if filter_dates[1] is not None:
                df = df.query("{} < '{}'".format(self.dtfield, filter_dates[1]))

        # Set the data attribute
        self.data = df.copy()

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
        self.epochs_stats[self.color_field] = get_colors(
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
            linestyle=specs["linestyle"],
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
        self.epochs_stats["Color"] = get_colors(
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
        self.folder_src = None

        # Set specific attributes
        self.name_object = "Time Series Collection"
        self.dtfield = "DateTime"
        self.overfield = "Overlapping"

        # load view specs
        self._set_view_specs()

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
        self.var_min = self.catalog["Var_min"].min()
        self.var_max = self.catalog["Var_max"].max()

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
                file_data=input_file,
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
        freq = self.catalog["DtFreq"].values[0]

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
                input_df=df_aux,
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
        ts_aux = TimeSeries()
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
        agg_df = local_epochs_df.groupby("Name")["Epochs_n"].sum().reset_index()
        agg_df = agg_df.sort_values(by="Epochs_n", ascending=True).reset_index(drop=True)
        names = agg_df["Name"].values
        if usealias:
            alias = [dict_alias[name] for name in names]
        preval = 100 * agg_df["Epochs_n"].values / agg_df["Epochs_n"].sum()
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
            self.collection[name].view_specs["folder"] = folder
            self.collection[name].view(show=False)
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


# ------------- SPATIAL OBJECTS -------------  #
# todo refactor to MbaE paradigm

class Raster:
    """
    The basic raster map dataset.
    """

    def __init__(self, name="myRasterMap", dtype="float32"):
        """Deploy a basic raster map object.

        :param name: Map name, defaults to "myRasterMap"
        :type name: str
        :param dtype: Data type of raster cells. Options: byte, uint8, int16, int32, float32, etc., defaults to "float32"
        :type dtype: str

        **Attributes:**

        - ``grid`` (None): Main grid of the raster.
        - ``backup_grid`` (None): Backup grid for AOI operations.
        - ``isaoi`` (False): Flag indicating whether an AOI mask is applied.
        - ``asc_metadata`` (dict): Metadata dictionary with keys: ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value.
        - ``nodatavalue`` (None): NODATA value from asc_metadata.
        - ``cellsize`` (None): Cell size from asc_metadata.
        - ``name`` (str): Name of the raster map.
        - ``dtype`` (str): Data type of raster cells.
        - ``cmap`` ("jet"): Default color map for visualization.
        - ``varname`` ("Unknown variable"): Variable name associated with the raster.
        - ``varalias`` ("Var"): Variable alias.
        - ``description`` (None): Description of the raster map.
        - ``units`` ("units"): Measurement units of the raster values.
        - ``date`` (None): Date associated with the raster map.
        - ``source_data`` (None): Source data information.
        - ``prj`` (None): Projection information.
        - ``path_ascfile`` (None): Path to the .asc raster file.
        - ``path_prjfile`` (None): Path to the .prj projection file.
        - ``view_specs`` (None): View specifications for visualization.

        **Examples:**

        >>> # Create a raster map with default settings
        >>> raster = Raster()

        >>> # Create a raster map with custom name and data type
        >>> custom_raster = Raster(name="CustomRaster", dtype="int16")
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
        # get view specs
        self.view_specs = None
        self._set_view_specs()

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
        """Set the data grid for the raster object.

        This function allows setting the data grid for the raster object. The incoming grid should be a NumPy array.

        :param grid: :class:`numpy.ndarray`
            The data grid to be set for the raster.
        :type grid: :class:`numpy.ndarray`

        **Notes:**

        - The function overwrites the existing data grid in the raster object with the incoming grid, ensuring that the data type matches the raster's dtype.
        - Nodata values are masked after setting the grid.

        **Examples:**

        >>> # Example of setting a new grid
        >>> new_grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> raster.set_grid(new_grid)
        """
        # overwrite incoming dtype
        self.grid = grid.astype(self.dtype)
        # mask nodata values
        self.mask_nodata()
        return None

    def set_asc_metadata(self, metadata):
        """Set metadata for the raster object based on incoming metadata.

        This function allows setting metadata for the raster object from an incoming metadata dictionary. The metadata should include information such as the number of columns, number of rows, corner coordinates, cell size, and nodata value.

        :param metadata: dict
            A dictionary containing metadata for the raster. Example metadata for a ``.asc`` file raster:

            .. code-block:: python

                meta = {
                    'ncols': 366,
                    'nrows': 434,
                    'xllcorner': 559493.08,
                    'yllcorner': 6704832.2,
                    'cellsize': 30,
                    'NODATA_value': -1
                }

        :type metadata: dict

        **Notes:**

        - The function updates the raster object's metadata based on the provided dictionary, ensuring that existing metadata keys are preserved.
        - It specifically updates nodata value and cell size attributes in the raster object.

        **Examples:**

        >>> # Example of setting metadata
        >>> metadata_dict = {'ncols': 200, 'nrows': 300, 'xllcorner': 500000.0, 'yllcorner': 6000000.0, 'cellsize': 25, 'NODATA_value': -9999}
        >>> raster.set_asc_metadata(metadata_dict)
        """
        for k in self.asc_metadata:
            if k in metadata:
                self.asc_metadata[k] = metadata[k]
        # update nodata value and cellsize
        self.nodatavalue = self.asc_metadata["NODATA_value"]
        self.cellsize = self.asc_metadata["cellsize"]
        return None

    def load(self, asc_file, prj_file=None):
        """Load data from files to the raster object.

        This function loads data from ``.asc`` raster and '.prj' projection files into the raster object.

        :param asc_file: str
            The path to the ``.asc`` raster file.
        :type asc_file: str

        :param prj_file: str, optional
            The path to the '.prj' projection file. If not provided, an attempt is made to use the same path and name as the ``.asc`` file with the '.prj' extension.
        :type prj_file: str

        :return: None
        :rtype: None

        **Notes:**

        - The function first loads the raster data from the ``.asc`` file using the ``load_asc_raster`` method.
        - If a '.prj' file is not explicitly provided, the function attempts to use a '.prj' file with the same path and name as the ``.asc`` file.
        - The function then loads the projection information from the '.prj' file using the ``load_prj_file`` method.

        **Examples:**

        >>> # Example of loading data
        >>> raster.boot(asc_file="path/to/raster.asc")

        >>> # Example of loading data with a specified projection file
        >>> raster.boot(asc_file="path/to/raster.asc", prj_file="path/to/raster.prj")
        """
        self.load_asc_raster(file=asc_file)
        if prj_file is None:
            # try to use the same path and name
            prj_file = asc_file.split(".")[0] + ".prj"
            if os.path.isfile(prj_file):
                self.load_prj_file(file=prj_file)
        else:
            self.load_prj_file(file=prj_file)
        return None

    def load_tif_raster(self, file):
        """Load data from '.tif' raster files.

        This function loads data from '.tif' raster files into the raster object. Note that metadata may be provided from other sources.

        :param file: str
            The file path of the '.tif' raster file.
        :type file: str

        :return: None
        :rtype: None

        **Notes:**

        - The function uses the Pillow (PIL) library to open the '.tif' file and converts it to a NumPy array.
        - Metadata may need to be provided separately, as this function focuses on loading raster data.
        - The loaded data grid is set using the ``set_grid`` method of the raster object.

        **Examples:**

        >>> # Example of loading data from a '.tif' file
        >>> raster.load_tif_raster(file="path/to/raster.tif")
        """
        from PIL import Image

        # Open the TIF file
        img_data = Image.open(file)
        # Convert the PIL image to a NumPy array
        grd_data = np.array(img_data)
        # set grid
        self.set_grid(grid=grd_data)
        return None

    def load_asc_raster(self, file):
        """Load data and metadata from ``.asc`` raster files.

        This function loads both data and metadata from ``.asc`` raster files into the raster object.

        :param file: str
            The file path to the ``.asc`` raster file.
        :type file: str

        :return: None
        :rtype: None

        **Notes:**

        - The function reads the content of the ``.asc`` file, extracts metadata, and constructs the data grid.
        - The metadata includes information such as the number of columns, number of rows, corner coordinates, cell size, and nodata value.
        - The data grid is constructed from the array information provided in the ``.asc`` file.
        - The function depends on the existence of a properly formatted ``.asc`` file.
        - No additional dependencies beyond standard Python libraries are required.

        **Examples:**

        >>> # Example of loading data and metadata from a ``.asc`` file
        >>> raster.load_asc_raster(file="path/to/raster.asc")
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
        self.set_asc_metadata(metadata=dct_meta)
        self.set_grid(grid=grd_data)
        return None

    def load_asc_metadata(self, file):
        """Load only metadata from ``.asc`` raster files.

        This function extracts metadata from ``.asc`` raster files and sets it as attributes in the raster object.

        :param file: str
            The file path to the ``.asc`` raster file.
        :type file: str

        :return: None
        :rtype: None

        **Notes:**

        - The function reads the first six lines of the ``.asc`` file to extract metadata.
        - Metadata includes information such as the number of columns, number of rows, corner coordinates, cell size, and nodata value.
        - The function sets the metadata as attributes in the raster object using the ``set_asc_metadata`` method.
        - This function is useful when only metadata needs to be loaded without the entire data grid.

        **Examples:**

        >>> # Example of loading metadata from a ``.asc`` file
        >>> raster.load_asc_metadata(file="path/to/raster.asc")
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
        """Load '.prj' auxiliary file to the 'prj' attribute.

        This function loads the content of a '.prj' auxiliary file and sets it as the 'prj' attribute in the raster object.

        :param file: str
            The file path to the '.prj' auxiliary file.
        :type file: str

        :return: None
        :rtype: None

        **Notes:**

        - The function reads the content of the '.prj' file and assigns it to the 'prj' attribute.
        - The 'prj' attribute typically contains coordinate system information in Well-Known Text (WKT) format.
        - This function is useful for associating coordinate system information with raster data.

        **Examples:**

        >>> # Example of loading coordinate system information from a '.prj' file
        >>> raster.load_prj_file(file="path/to/raster.prj")
        """
        self.path_prjfile = file
        with open(file) as f:
            self.prj = f.readline().strip("\n")
        return None

    def copy_structure(self, raster_ref, n_nodatavalue=None):
        """Copy structure (asc_metadata and prj file) from another raster object.

        This function copies the structure, including asc_metadata and prj, from another raster object to the current raster object.

        :param raster_ref: :class:`datasets.Raster`
            The reference incoming raster object from which to copy asc_metadata and prj.
        :type raster_ref: :class:`datasets.Raster`

        :param n_nodatavalue: float, optional
            The new nodata value for different raster objects. If None, the nodata value remains unchanged.
        :type n_nodatavalue: float

        :return: None
        :rtype: None

        **Notes:**

        - The function copies the asc_metadata and prj attributes from the reference raster object to the current raster object.
        - If a new nodata value is provided, it updates the 'NODATA_value' in the copied asc_metadata.
        - This function is useful for ensuring consistency in metadata and coordinate system information between raster objects.

        **Examples:**

        >>> # Example of copying structure from a reference raster object
        >>> new_raster.copy_structure(raster_ref=reference_raster, n_nodatavalue=-9999.0)
        """
        dict_meta = raster_ref.asc_metadata.copy()
        # handle new nodatavalue
        if n_nodatavalue is None:
            pass
        else:
            dict_meta["NODATA_value"] = n_nodatavalue
        self.set_asc_metadata(metadata=dict_meta)
        self.prj = raster_ref.prj[:]
        return None

    def export(self, folder, filename=None):
        """Export raster data to a folder.

        This function exports raster data, including the ``.asc`` raster file and '.prj' projection file, to the specified folder.

        :param folder: str
            The directory path to export the raster data.
        :type folder: str

        :param filename: str, optional
            The name of the exported files without extension. If None, the name of the raster object is used.
        :type filename: str

        :return: None
        :rtype: None

        **Notes:**

        - The function exports the raster data to the specified folder, creating ``.asc`` and '.prj' files.
        - If a filename is not provided, the function uses the name of the raster object.
        - The exported files will have the same filename with different extensions (``.asc`` and '.prj').
        - This function is useful for saving raster data to a specified directory.

        **Examples:**

        >>> # Example of exporting raster data to a folder
        >>> raster.export(folder="path/to/export_folder", filename="exported_raster")
        """
        if filename is None:
            filename = self.name
        self.export_asc_raster(folder=folder, filename=filename)
        self.export_prj_file(folder=folder, filename=filename)
        return None

    def export_asc_raster(self, folder, filename=None):
        """Export an ``.asc`` raster file.

        This function exports the raster data as an ``.asc`` file to the specified folder.

        :param folder: str
            The directory path to export the ``.asc`` raster file.
        :type folder: str

        :param filename: str, optional
            The name of the exported file without extension. If None, the name of the raster object is used.
        :type filename: str

        :return: str
            The full file name (path and extension) of the exported ``.asc`` raster file.
        :rtype: str

        **Notes:**

        - The function exports the raster data to an ``.asc`` file in the specified folder.
        - If a filename is not provided, the function uses the name of the raster object.
        - The exported ``.asc`` file contains metadata and data information.
        - This function is useful for saving raster data in ASCII format.

        **Examples:**

        >>> # Example of exporting an ``.asc`` raster file to a folder
        >>> raster.export_asc_raster(folder="path/to/export_folder", filename="exported_raster")
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
        """Export a '.prj' file.

        This function exports the coordinate system information to a '.prj' file in the specified folder.

        :param folder: str
            The directory path to export the '.prj' file.
        :type folder: str

        :param filename: str, optional
            The name of the exported file without extension. If None, the name of the raster object is used.
        :type filename: str

        :return: str or None
            The full file name (path and extension) of the exported '.prj' file, or None if no coordinate system information is available.
        :rtype: str or None

        **Notes:**

        - The function exports the coordinate system information to a '.prj' file in the specified folder.
        - If a filename is not provided, the function uses the name of the raster object.
        - The exported '.prj' file contains coordinate system information in Well-Known Text (WKT) format.
        - This function is useful for saving coordinate system information associated with raster data.

        **Examples:**

        >>> # Example of exporting a '.prj' file to a folder
        >>> raster.export_prj_file(folder="path/to/export_folder", filename="exported_prj")
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
        """Mask grid cells as NaN where data is NODATA.

        :return: None
        :rtype: None

        **Notes:**

        - The function masks grid cells as NaN where the data is equal to the specified NODATA value.
        - If NODATA value is not set, no masking is performed.
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
        return None

    def insert_nodata(self):
        """Insert grid cells as NODATA where data is NaN.

        :return: None
        :rtype: None

        **Notes:**

        - The function inserts NODATA values into grid cells where the data is NaN.
        - If NODATA value is not set, no insertion is performed.
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
        return None

    def rebase_grid(self, base_raster, inplace=False, method="linear_model"):
        """Rebase the grid of a raster.

        This function creates a new grid based on a provided reference raster. Both rasters are expected to be in the same coordinate system and have overlapping bounding boxes.

        :param base_raster: :class:`datasets.Raster`
            The reference raster used for rebase. It should be in the same coordinate system and have overlapping bounding boxes.
        :type base_raster: :class:`datasets.Raster`

        :param inplace: bool, optional
            If True, the rebase operation will be performed in-place, and the original raster's grid will be modified. If False, a new rebased grid will be returned, and the original data will remain unchanged. Default is False.
        :type inplace: bool

        :param method: str, optional
            Interpolation method for rebasing the grid. Options include "linear_model," "nearest," and "cubic." Default is "linear_model."
        :type method: str

        :return: :class:`numpy.ndarray`` or None
            If inplace is False, a new rebased grid as a NumPy array.
            If inplace is True, returns None, and the original raster's grid is modified in-place.
        :rtype: :class:`numpy.ndarray`` or None

        **Notes:**

        - The rebase operation involves interpolating the values of the original grid to align with the reference raster's grid.
        - The method parameter specifies the interpolation method and can be "linear_model," "nearest," or "cubic."
        - The rebase assumes that both rasters are in the same coordinate system and have overlapping bounding boxes.

        **Examples:**

        >>> # Example with inplace=True
        >>> raster.rebase_grid(base_raster=reference_raster, inplace=True)

        >>> # Example with inplace=False
        >>> rebased_grid = raster.rebase_grid(base_raster=reference_raster, inplace=False)
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
        """Apply AOI (area of interest) mask to the raster map.

        This function applies an AOI (area of interest) mask to the raster map, replacing values outside the AOI with the NODATA value.

        :param grid_aoi: :class:`numpy.ndarray`
            Map of AOI (masked array or pseudo-boolean). Expected to have the same grid shape as the raster.
        :type grid_aoi: :class:`numpy.ndarray`

        :param inplace: bool, optional
            If True, overwrite the main grid with the masked values. If False, create a backup and modify a copy of the grid.
            Default is False.
        :type inplace: bool

        :return: None
        :rtype: None

        **Notes:**

        - The function replaces values outside the AOI (where grid_aoi is 0) with the NODATA value.
        - If NODATA value is not set, no replacement is performed.
        - If inplace is True, the main grid is modified. If False, a backup of the grid is created before modification.
        - This function is useful for focusing analysis or visualization on a specific area within the raster map.

        **Examples:**

        >>> # Example of applying an AOI mask to the raster map
        >>> raster.apply_aoi_mask(grid_aoi=aoi_mask, inplace=True)
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
        """Release AOI mask from the main grid. Backup grid is restored.

        This function releases the AOI (area of interest) mask from the main grid, restoring the original values from the backup grid.

        :return: None
        :rtype: None

        **Notes:**

        - If an AOI mask has been applied, this function restores the original values to the main grid from the backup grid.
        - If no AOI mask has been applied, the function has no effect.
        - After releasing the AOI mask, the backup grid is set to None, and the raster object is no longer considered to have an AOI mask.

        **Examples:**

        >>> # Example of releasing the AOI mask from the main grid
        >>> raster.release_aoi_mask()
        """
        if self.isaoi:
            self.set_grid(grid=self.backup_grid)
            self.backup_grid = None
            self.isaoi = False
        return None

    def cut_edges(self, upper, lower, inplace=False):
        """Cutoff upper and lower values of the raster grid.

        :param upper: float or int
            The upper value for the cutoff.
        :type upper: float or int

        :param lower: float or int
            The lower value for the cutoff.
        :type lower: float or int

        :param inplace: bool, optional
            If True, modify the main grid in-place. If False, create a processed copy of the grid.
            Default is False.
        :type inplace: bool

        :return: :class:`numpy.ndarray`` or None
            The processed grid if inplace is False. If inplace is True, returns None.
        :rtype: Union[None, np.ndarray]

        **Notes:**

        - Values in the raster grid below the lower value are set to the lower value.
        - Values in the raster grid above the upper value are set to the upper value.
        - If inplace is False, a processed copy of the grid is returned, leaving the original grid unchanged.
        - This function is useful for clipping extreme values in the raster grid.

        **Examples:**

        >>> # Example of cutting off upper and lower values in the raster grid
        >>> processed_grid = raster.cut_edges(upper=100, lower=0, inplace=False)
        >>> # Alternatively, modify the main grid in-place
        >>> raster.cut_edges(upper=100, lower=0, inplace=True)
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
        """Get all metadata from the base object.

        :return: Metadata dictionary.

            - "Name" (str): Name of the raster.
            - "Variable" (str): Variable name.
            - "VarAlias" (str): Variable alias.
            - "Units" (str): Measurement units.
            - "Date" (str): Date information.
            - "Source" (str): Data source.
            - "Description" (str): Description of the raster.
            - "cellsize" (float): Cell size of the raster.
            - "ncols" (int): Number of columns in the raster grid.
            - "nrows" (int): Number of rows in the raster grid.
            - "xllcorner" (float): X-coordinate of the lower-left corner.
            - "yllcorner" (float): Y-coordinate of the lower-left corner.
            - "NODATA_value" (Union[float, None]): NODATA value in the raster.
            - "Prj" (str): Projection information.
            - "Path_ASC" (str): File path to the ASC raster file.
            - "Path_PRJ" (str): File path to the PRJ projection file.
        :rtype: dict
        """
        return {
            "Name": self.name,
            "Variable": self.varname,
            "VarAlias": self.varalias,
            "Units": self.units,
            "Date": self.date,
            "Source": self.source_data,
            "Description": self.description,
            "cellsize": self.cellsize,
            "ncols": self.asc_metadata["ncols"],
            "nrows": self.asc_metadata["nrows"],
            "xllcorner": self.asc_metadata["xllcorner"],
            "yllcorner": self.asc_metadata["yllcorner"],
            "NODATA_value": self.nodatavalue,
            "Prj": self.prj,
            "Path_ASC": self.path_ascfile,
            "Path_PRJ": self.path_prjfile,
        }

    def get_bbox(self):
        """Get the Bounding Box of the map.

        :return: Dictionary of xmin, xmax, ymin, and ymax.

            - "xmin" (float): Minimum x-coordinate.
            - "xmax" (float): Maximum x-coordinate.
            - "ymin" (float): Minimum y-coordinate.
            - "ymax" (float): Maximum y-coordinate.
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
        """Get flat and cleared grid data points (x, y, and z).

        :param drop_nan: Option to ignore nan values.
        :type drop_nan: bool

        :return: DataFrame of x, y, and z fields.
        :rtype: :class:`pandas.DataFrame``` or None
            If the grid is None, returns None.

        **Notes:**

        - This function extracts coordinates (x, y, and z) from the raster grid.
        - The x and y coordinates are determined based on the grid cell center positions.
        - If drop_nan is True, nan values are ignored in the resulting DataFrame.
        - The resulting DataFrame includes columns for x, y, z, i, and j coordinates.

        **Examples:**

        >>> # Get grid data points with nan values included
        >>> datapoints_df = raster.get_grid_datapoints(drop_nan=False)
        >>> # Get grid data points with nan values ignored
        >>> clean_datapoints_df = raster.get_grid_datapoints(drop_nan=True)
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

        :return: 1D vector of cleared data.
        :rtype: :class:`numpy.ndarray`` or None
            If the grid is None, returns None.

        **Notes:**

        - This function extracts and flattens the grid data, removing any masked or NaN values.
        - For integer grids, the masked values are ignored.
        - For floating-point grids, both masked and NaN values are ignored.

        **Examples:**

        >>> # Get flattened and cleared grid data
        >>> data_vector = raster.get_grid_data()
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
        """Get basic statistics from flat and cleared data.

        :return: DataFrame of basic statistics.
        :rtype: :class:`pandas.DataFrame``` or None
            If the grid is None, returns None.

        **Notes:**

        - This function computes basic statistics from the flattened and cleared grid data.
        - Basic statistics include measures such as mean, median, standard deviation, minimum, and maximum.
        - Requires the 'plans.analyst' module for statistical analysis.

        **Examples:**

        >>> # Get basic statistics from the raster grid
        >>> stats_dataframe = raster.get_grid_stats()
        """
        if self.grid is None:
            return None
        else:
            from plans.analyst import Univar

            return Univar(data=self.get_grid_data()).assess_basic_stats()

    def get_aoi(self, by_value_lo, by_value_hi):
        """Get the AOI map from an interval of values (values are expected to exist in the raster).

        :param by_value_lo: Number for the lower bound (inclusive).
        :type by_value_lo: float
        :param by_value_hi: Number for the upper bound (inclusive).
        :type by_value_hi: float

        :return: AOI map.
        :rtype: :class:`AOI`` object

        **Notes:**

        - This function creates an AOI (Area of Interest) map based on a specified value range.
        - The AOI map is constructed as a binary grid where values within the specified range are set to 1, and others to 0.

        **Examples:**

        >>> # Get AOI map for values between 10 and 20
        >>> aoi_map = raster.get_aoi(by_value_lo=10, by_value_hi=20)
        """
        map_aoi = AOI(name="{} {}-{}".format(self.varname, by_value_lo, by_value_hi))
        map_aoi.set_asc_metadata(metadata=self.asc_metadata)
        map_aoi.prj = self.prj
        # set grid
        self.insert_nodata()
        map_aoi.set_grid(
            grid=1 * (self.grid >= by_value_lo) * (self.grid <= by_value_hi)
        )
        self.mask_nodata()
        return map_aoi

    def _set_view_specs(self):
        """Set default view specs.

        :return: None
        :rtype: None

        **Notes:**

        - This private method sets default view specifications for visualization.
        - The view specs include color, colormap, titles, dimensions, and other parameters for visualization.
        - These default values can be adjusted based on specific requirements.

        **Examples:**

        >>> # Set default view specifications
        >>> obj._set_view_specs()
        """
        self.view_specs = {
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
            "nbins": 100,
            "vmin": None,
            "vmax": None,
            "hist_vmax": None,
            "project_name": None,
            "zoom_window": None
        }
        return None

    def view(
        self,
        accum=True,
        show=True,
        stats=True,
        folder="./output",
        filename=None,
        dpi=300,
        fig_format="jpg",
    ):
        """Plot a basic panel of the raster map.

        :param accum: boolean to include an accumulated probability plot, defaults to True
        :type accum: bool
        :param show: boolean to show the plot instead of saving, defaults to True
        :type show: bool
        :param stats: boolean to include stats, defaults to True
        :type stats: bool
        :param folder: path to the output folder, defaults to "./output"
        :type folder: str
        :param filename: name of the file, defaults to None
        :type filename: str
        :param dpi: image resolution, defaults to 300
        :type dpi: int
        :param fig_format: image format (e.g., jpg or png), defaults to "jpg"
        :type fig_format: str

        **Notes:**

        - This function generates a basic panel for visualizing the raster map, including the map itself, a histogram,
          metadata, and basic statistics.
        - The panel includes various customization options such as color, titles, dimensions, and more.
        - The resulting plot can be displayed or saved based on the specified parameters.

        **Examples:**

        >>> # Show the plot without saving
        >>> raster.view()

        >>> # Save the plot to a file
        >>> raster.view(show=False, folder="./output", filename="raster_plot", dpi=300, fig_format="png")
        """
        import matplotlib.ticker as mtick
        from plans.analyst import Univar

        # get univar base_object
        uni = Univar(data=self.get_grid_data())

        specs = self.view_specs

        if specs["vmin"] is None:
            specs["vmin"] = np.min(self.grid)
        if specs["vmax"] is None:
            specs["vmax"] = np.max(self.grid)

        if specs["project_name"] is None:
            suff = ""
        else:
            suff = "_{}".format(specs["project_name"])

        # Deploy figure
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        gs = mpl.gridspec.GridSpec(
            4, 5, wspace=0.8, hspace=0.1, left=0.05, bottom=0.1, top=0.85, right=0.95
        )
        fig.suptitle(specs["suptitle"] + suff)

        # plot map
        plt.subplot(gs[:3, :3])
        # handle zoom window
        i_min = 0
        i_max = len(self.grid)
        j_min = 0
        j_max = len(self.grid[0])
        if specs["zoom_window"] is not None:
            i_min = specs["zoom_window"]["i_min"]
            i_max = specs["zoom_window"]["i_max"]
            j_min = specs["zoom_window"]["j_min"]
            j_max = specs["zoom_window"]["j_max"]
        # plot image
        im = plt.imshow(
            self.grid[i_min: i_max, j_min: j_max], cmap=specs["cmap"], vmin=specs["vmin"], vmax=specs["vmax"]
        )
        plt.title("a. {}".format(specs["a_title"]), loc="left")
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
            x=n_mean - 20 * (specs["vmax"] - specs["vmin"]) / 100,
            y=0.9 * specs["hist_vmax"],
            s="$\mu$ = {:.2f}".format(n_mean),
            color="red"
        )

        plt.ylim(0, specs["hist_vmax"])
        plt.xlim(specs["vmin"], specs["vmax"])

        # plt.ylabel(specs["b_ylabel"])
        plt.xlabel(specs["b_xlabel"])

        # Set the y-axis formatter as percentages
        yticks = mtick.PercentFormatter(xmax=1, decimals=1, symbol="%", is_latex=False)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(yticks)

        # --------
        # plot accumulated probability
        if accum:
            ax2 = ax.twinx()
            vct_cump = np.cumsum(a=vct_result[0]) / np.sum(vct_result[0])
            plt.plot(vct_result[1][1:], vct_cump, color="darkred")
            ax2.grid(False)

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
                if s_value is None:
                    s_value = '-'
                    s_line = "{:>15}: {:<10}".format(s_head, s_value)
                else:
                    s_line = "{:>15}: {:<10.5f}".format(s_head, s_value)
            else:
                s_value = df_meta["Value"].values[i]
                if s_value is None:
                    s_value = '-'
                    s_line = "{:>15}: {:<10}".format(s_head, s_value)
                else:
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

        # plot stats
        if stats:
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
                    x=n_x + 0.15,
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
                filename = "{}_{}{}".format(self.varalias, self.name, suff)
            # save figure
            plt.savefig(
                "{}/{}{}.{}".format(folder, filename, suff, fig_format), dpi=dpi
            )
            plt.close(fig)
            # save stats:
            if stats:
                df_stats.to_csv("{}/{}{}.csv".format(folder, filename, suff), sep=";", index=False)
        return None

class QualiRaster(Raster):
    """
    Basic qualitative raster map dataset.

    Attributes dataframe must at least have:
    * :class:`Id`` field
    * :class:`Name`` field
    * :class:`Alias`` field

    """

    def __init__(self, name="QualiMap", dtype="uint8"):
        """Initialize dataset.

        :param name: name of map
        :type name: str
        :param dtype: data type of raster cells, defaults to uint8
        :type dtype: str
        """
        # prior setup
        self.path_csvfile = None
        self.table = None
        self.idfield = "Id"
        self.namefield = "Name"
        self.aliasfield = "Alias"
        self.colorfield = "Color"
        self.areafield = "Area"
        # call superior
        super().__init__(name=name, dtype=dtype)
        # overwrite
        self.cmap = "tab20"
        self.varname = "Unknown variable"
        self.varalias = "Var"
        self.description = "Unknown"
        self.units = "category ID"
        self.path_csvfile = None
        self._overwrite_nodata()
        # NOTE: view specs is set by setting table

    def _overwrite_nodata(self):
        """No data in QualiRaster is set by default to 0"""
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

    def reclassify(self, dict_ids, df_new_table, talk=False):
        """Reclassify QualiRaster Ids in grid and table

        :param dict_ids: dictionary to map from "Old_Id" to "New_id"
        :type dict_ids: dict
        :param df_new_table: new table for QualiRaster
        :type df_new_table: :class:`pandas.DataFrame`
        :param talk: option for printing messages
        :type talk: bool
        :return: None
        :rtype: None
        """
        grid_new = self.grid.copy()
        for i in range(len(dict_ids["Old_Id"])):
            n_old_id = dict_ids["Old_Id"][i]
            n_new_id = dict_ids["New_Id"][i]
            if talk:
                print(">> reclassify Ids from {} to {}".format(n_old_id, n_new_id))
            grid_new = (grid_new * (grid_new != n_old_id)) + (
                n_new_id * (grid_new == n_old_id)
            )
        # set new grid
        self.set_grid(grid=grid_new)
        # reset table
        self.set_table(dataframe=df_new_table)
        return None

    def load(self, asc_file, prj_file, table_file):
        """
        Load data from files to raster
        :param asc_file: folder_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: folder_main to ``.prj`` projection file
        :type prj_file: str
        :param table_file: folder_main to ``.txt`` table file
        :type table_file: str
        :return: None
        :rtype: None
        """
        super().load(asc_file=asc_file, prj_file=prj_file)
        self.load_table(file=table_file)
        return None

    def load_table(self, file):
        """Load attributes dataframe from ``csv`` ``.txt`` file (separator must be ;).

        :param file: folder_main to file
        :type file: str
        """
        self.path_csvfile = file
        # read raw file
        df_aux = pd.read_csv(file, sep=";")
        # set to self
        self.set_table(dataframe=df_aux)
        return None

    def export(self, folder, filename=None):
        """
        Export raster data
        param folder: string of directory folder_main,
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

        :param folder: string of directory folder_main
        :type folder: str
        :param filename: string of file without extension
        :type filename: str
        :return: full file name (folder_main and extension) string
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
        # set view specs
        self._set_view_specs()
        return None

    def clear_table(self):
        """Clear the unfound values in the map from the table."""
        if self.grid is None:
            pass
        else:
            # found values:
            lst_ids = np.unique(self.grid)
            # filter dataframe
            filtered_df = self.table[self.table[self.idfield].isin(lst_ids)]
            # reset table
            self.set_table(dataframe=filtered_df.reset_index(drop=True))
        return None

    def set_random_colors(self):
        """Set random colors to attribute table."""
        if self.table is None:
            pass
        else:
            self.table[self.colorfield] = get_colors(
                size=len(self.table), cmap=self.cmap
            )
            # reaload table for reset viewspecs
            self.set_table(dataframe=self.table)
        return None

    def get_areas(self, merge=False):
        """Get export_areas in map of each category in table.

        :param merge: option to merge data with raster table
        :type merge: bool, defaults to False
        :return: export_areas dataframe
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
        df_aux1 = self.table.copy()
        self.clear_table()  # clean
        df_aux = self.table.copy()  # get copy
        df_aux = df_aux[["Id", "Name", "Alias"]].copy()  # filter
        self.set_table(dataframe=df_aux1)  # restore uncleaned table
        ##### df_aux = self.table[["Id", "Name", "Alias"]].copy()

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

    def get_aoi(self, by_value_id):
        """
        Get the AOI map from a specific value id (value is expected to exist in the raster)
        :param by_value_id: category id value
        :type by_value_id: int
        :return: AOI map
        :rtype: :class:`AOI`` object
        """
        map_aoi = AOI(name="{} {}".format(self.varname, by_value_id))
        map_aoi.set_asc_metadata(metadata=self.asc_metadata)
        map_aoi.prj = self.prj
        # set grid
        self.insert_nodata()
        map_aoi.set_grid(grid=1 * (self.grid == by_value_id))
        self.mask_nodata()
        return map_aoi

    def get_metadata(self):
        """Get all metadata from base_object

        :return: metadata
        :rtype: dict
        """
        _dict = super().get_metadata()
        _dict["Path_CSV"] = self.path_csvfile
        return _dict

    def _set_view_specs(self):
        """
        Get default view specs
        :return: None
        :rtype: None
        """
        if self.table is None:
            pass
        else:
            from matplotlib.colors import ListedColormap

            # handle no color field in table:
            if self.colorfield in self.table.columns:
                pass
            else:
                self.set_random_colors()

            # hack for non-continuous ids:
            _all_ids = np.arange(0, self.table[self.idfield].max() + 1)
            _lst_colors = []
            for i in range(0, len(_all_ids)):
                _df = self.table.query("{} >= {}".format(self.idfield, i)).copy()
                _color = _df[self.colorfield].values[0]
                _lst_colors.append(_color)
            # setup
            self.view_specs = {
                "color": "tab:grey",
                "cmap": ListedColormap(_lst_colors),
                "suptitle": "{} ({}) | {}".format(
                    self.varname, self.varalias, self.name
                ),
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
                "legend_x": 0.55,
                "legend_y": 0.3,
                "legend_ncol": 2,
                "project_name": None,
                "zoom_window": None,
                "plot_metadata": True
            }
        return None

    def view(
        self,
        show=True,
        export_areas=True,
        folder="./output",
        filename=None,
        dpi=300,
        fig_format="jpg",
        filter=False,
        n_filter=6,
    ):
        """Plot a basic pannel of qualitative raster map.

        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param export_areas: option to export areas table, defaults to True
        :type export_areas: bool
        :param folder: folder_main to output folder, defaults to ``./output``
        :type folder: str
        :param filename: name of file, defaults to None
        :type filename: str
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :param filter: option for collapsing to n classes max (create "other" class)
        :type filter: bool
        :param n_filter: number of total classes + others
        :type n_filter: int
        :type zoom:
        :return: None
        :rtype: None
        """
        from matplotlib.patches import Patch

        # pass specs
        specs = self.view_specs

        if specs["project_name"] is None:
            suff = ""
        else:
            suff = "_{}".format(specs["project_name"])

        # -----------------------------------------------
        # ensure export_areas are computed
        df_areas = pd.merge(
            self.table[["Id", "Color"]], self.get_areas(), how="left", on="Id"
        )
        # new aux
        df_aux = df_areas.sort_values(by="{}_m2".format(self.areafield), ascending=True)
        if filter:
            if len(df_aux) > n_filter:
                n_limit = df_aux["{}_m2".format(self.areafield)].values[-n_filter]
                df_aux2 = df_aux.query("{}_m2 < {}".format(self.areafield, n_limit))
                df_aux2 = pd.DataFrame(
                    {
                        "Id": [0],
                        "Color": ["tab:grey"],
                        "Name": ["Others"],
                        "Alias": ["etc"],
                        "Cell_count": [df_aux2["Cell_count"].sum()],
                        "{}_m2".format(self.areafield): [
                            df_aux2["{}_m2".format(self.areafield)].sum()
                        ],
                        "{}_ha".format(self.areafield): [
                            df_aux2["{}_ha".format(self.areafield)].sum()
                        ],
                        "{}_km2".format(self.areafield): [
                            df_aux2["{}_km2".format(self.areafield)].sum()
                        ],
                        "{}_f".format(self.areafield): [
                            df_aux2["{}_f".format(self.areafield)].sum()
                        ],
                        "{}_%".format(self.areafield): [
                            df_aux2["{}_%".format(self.areafield)].sum()
                        ],
                    }
                )
                df_aux = df_aux.query("{}_m2 >= {}".format(self.areafield, n_limit))
                df_aux = pd.concat([df_aux, df_aux2])
                df_aux = df_aux.drop_duplicates(subset="Id")
                df_aux = df_aux.sort_values(by="{}_m2".format(self.areafield))
                df_aux = df_aux.reset_index(drop=True)

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
        fig.suptitle(specs["suptitle"] + suff)

        # plot map
        plt.subplot(gs[:5, :3])
        # handle zoom window
        i_min = 0
        i_max = len(self.grid)
        j_min = 0
        j_max = len(self.grid[0])
        if specs["zoom_window"] is not None:
            i_min = specs["zoom_window"]["i_min"]
            i_max = specs["zoom_window"]["i_max"]
            j_min = specs["zoom_window"]["j_min"]
            j_max = specs["zoom_window"]["j_max"]
        # plot image
        im = plt.imshow(
            self.grid[i_min: i_max, j_min: j_max], cmap=specs["cmap"], vmin=specs["vmin"], vmax=specs["vmax"]
        )
        plt.title("a. {}".format(specs["a_title"]), loc="left")
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
            fontsize=8,
            markerscale=0.4,# 0.8
            handles=legend_elements,
            bbox_to_anchor=(specs["legend_x"], specs["legend_y"]),
            bbox_transform=fig.transFigure,
            ncol=specs["legend_ncol"],
        )

        # -----------------------------------------------
        # plot horizontal bar of export_areas
        plt.subplot(gs[: specs["gs_b_rowlim"], 3:])
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
        if specs["plot_metadata"]:
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
                filename = "{}_{}{}".format(self.varalias, self.name, suff)
            # save fig
            plt.savefig(
                "{}/{}{}.{}".format(folder, filename, suff, fig_format), dpi=dpi
            )
            plt.close(fig)
            # save areas
            if export_areas:
                df_areas.to_csv("{}/{}{}.csv".format(folder, filename, suff), sep=";", index=False)
        return None

class QualiHard(QualiRaster):
    """
    A Quali-Hard is a hard-coded qualitative map (that is, the table is pre-set)
    """

    def __init__(self, name="qualihard"):
        super().__init__(name, dtype="uint8")
        self.varname = "QualiRasterHard"
        self.varalias = "QRH"
        self.description = "Preset Classes"
        self.units = "classes ID"
        self.set_table(dataframe=self.get_table())

    def get_table(self):
        df_aux = pd.DataFrame(
            {
                "Id": [1, 2, 3],
                "Alias": ["A", "B", "C"],
                "Name": ["Class A", "Class B", "Class C"],
                "Color": ["red", "green", "blue"],
            }
        )
        return df_aux

    def load(self, asc_file, prj_file=None):
        """Load data from files to raster

        :param asc_file: folder_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: folder_main to ``.prj`` projection file
        :type prj_file: str
        :return: None
        :rtype: None
        """
        self.load_asc_raster(file=asc_file)
        if prj_file is None:
            # try to use the same path and name
            prj_file = asc_file.split(".")[0] + ".prj"
            if os.path.isfile(prj_file):
                self.load_prj_file(file=prj_file)
        else:
            self.load_prj_file(file=prj_file)
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
                    "Alias": [
                        "{}{}".format(self.varalias, vct_unique[i])
                        for i in range(len(vct_unique))
                    ],
                    "Name": [
                        "{} {}".format(self.varname, vct_unique[i])
                        for i in range(len(vct_unique))
                    ],
                }
            )
            self.table = self.table.drop(
                self.table[self.table["Id"] == self.asc_metadata["NODATA_value"]].index
            )
            self.table["Id"] = self.table["Id"].astype(int)
            self.table = self.table.sort_values(by="Id")
            self.table = self.table.reset_index(drop=True)
            self.set_random_colors()
            # set view specs
            self._set_view_specs()
            # fix some view_specs:
            self.view_specs["b_xlabel"] = "zones ID"
            del vct_unique
            return None

    def set_grid(self, grid):
        super().set_grid(grid)
        self.set_table()
        return None

    def load(self, asc_file, prj_file):
        """
        Load data from files to raster
        :param asc_file: folder_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: folder_main to ``.prj`` projection file
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
        :rtype: :class:`AOI`` object
        """
        map_aoi = AOI(name="{} {}".format(self.varname, zone_id))
        map_aoi.set_asc_metadata(metadata=self.asc_metadata)
        map_aoi.prj = self.prj
        # set grid
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
        :param folder: folder_main to output folder, defaults to ``./output``
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
        # set Raster map for plotting
        map_zones_aux = Raster(name=self.name)
        # set up
        map_zones_aux.varname = self.varname
        map_zones_aux.varalias = self.varalias
        map_zones_aux.units = self.units
        map_zones_aux.set_asc_metadata(metadata=self.asc_metadata)
        map_zones_aux.prj = self.prj
        map_zones_aux.cmap = "tab20"

        # grid setup
        self.insert_nodata()
        map_zones_aux.set_grid(grid=self.grid)
        self.mask_nodata()
        map_zones_aux._set_view_specs()
        map_zones_aux.view_specs["vmin"] = self.table["Id"].min()
        map_zones_aux.view_specs["vmax"] = self.table["Id"].max()
        # update extra view specs:
        for k in self.view_specs:
            map_zones_aux.view_specs[k] = self.view_specs[k]
        # call view
        map_zones_aux.view(
            accum=False,
            show=show,
            folder=folder,
            filename=filename,
            dpi=dpi,
            fig_format=fig_format,
        )
        del map_zones_aux
        return None




# -----------------------------------------
# Raster Collection data structures

class RasterCollection(Collection):
    """
    The raster test_collection base dataset.
    This data strucute is designed for holding and comparing :class:`Raster`` objects.
    """

    def __init__(self, name="myRasterCollection"):
        """Deploy the raster test_collection data structure.

        :param name: name of raster test_collection
        :type name: str
        """
        obj_aux = Raster
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
        skip_grid=False,
    ):
        """Load a :class:`Raster`` base_object from a ``.asc`` raster file.

        :param name: :class:`Raster.name`` name attribute
        :type name: str
        :param asc_file: folder_main to ``.asc`` raster file
        :type asc_file: str
        :param varname: :class:`Raster.varname`` variable name attribute, defaults to None
        :type varname: str
        :param varalias: :class:`Raster.varalias`` variable alias attribute, defaults to None
        :type varalias: str
        :param units: :class:`Raster.units`` units attribute, defaults to None
        :type units: str
        :param date: :class:`Raster.date`` date attribute, defaults to None
        :type date: str
        :param skip_grid: option for loading only the metadata
        :type skip_grid: bool
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
        if skip_grid:
            rst_aux.load_asc_metadata(file=asc_file)
        else:
            rst_aux.load_asc_raster(file=asc_file)
        # append to test_collection
        self.append(new_object=rst_aux)
        # delete aux
        del rst_aux
        return None

    def issamegrid(self):
        u_cols = len(self.catalog["ncols"].unique())
        u_rows = len(self.catalog["nrows"].unique())
        if u_cols == 1 and u_rows == 1:
            return True
        else:
            return False

    def reducer(
        self,
        reducer_func,
        reduction_name,
        extra_arg=None,
        skip_nan=False,
        talk=False,
    ):
        """This method reduces the test_collection by applying a numpy broadcasting function (example: np.mean)

        :param reducer_func: reducer numpy function (example: np.mean)
        :type reducer_func: numpy function
        :param reduction_name: name for the output raster
        :type reduction_name: str
        :param extra_arg: extra argument for function (example: np.percentiles) - Default: None
        :type extra_arg: any
        :param skip_nan: Option for skipping NaN values in map
        :type skip_nan: bool
        :param talk: option for printing messages
        :type talk: bool
        :return: raster object based on the first object found in the test_collection
        :rtype: :class:`Raster`
        """
        import copy

        # return None if there is different grids
        if self.issamegrid():
            # get shape parameters
            n = len(self.catalog)
            _first = self.catalog["Name"].values[0]
            n_flat = (
                self.collection[_first].grid.shape[0]
                * self.collection[_first].grid.shape[1]
            )

            # create the merged grid
            grd_merged = np.zeros(shape=(n, n_flat))
            # insert the flat arrays
            for i in range(n):
                _name = self.catalog["Name"].values[i]
                _vct_flat = self.collection[_name].grid.flatten()
                grd_merged[i] = _vct_flat
            # transpose
            grd_merged_T = grd_merged.transpose()

            # setup stats vector
            vct_stats = np.zeros(n_flat)
            # fill vector
            for i in range(n_flat):
                _vct = grd_merged_T[i]
                # remove NaN
                if skip_nan:
                    _vct = _vct[~np.isnan(_vct)]
                    # handle void vector
                    if len(_vct) == 0:
                        _vct = np.nan
                if extra_arg is None:
                    vct_stats[i] = reducer_func(_vct)
                else:
                    vct_stats[i] = reducer_func(_vct, extra_arg)

            # reshape
            grd_stats = np.reshape(
                a=vct_stats, newshape=self.collection[_first].grid.shape
            )
            # return set up
            output_raster = copy.deepcopy(self.collection[_first])
            output_raster.set_grid(grd_stats)
            output_raster.name = reduction_name
            return output_raster
        else:
            if talk:
                print("Warning: different grids found")
            return None

    def mean(self, skip_nan=False, talk=False):
        """Reduce Collection to the Mean raster

        :param skip_nan: Option for skipping NaN values in map
        :type skip_nan: bool
        :param talk: option for printing messages
        :type talk: bool
        :return: raster object based on the first object found in the test_collection
        :rtype: :class:`Raster`
        """
        output_raster = self.reducer(
            reducer_func=np.mean,
            reduction_name="{} Mean".format(self.name),
            skip_nan=skip_nan,
            talk=talk,
        )
        return output_raster

    def std(self, skip_nan=False, talk=False):
        """Reduce Collection to the Standard Deviation raster

        :param skip_nan: Option for skipping NaN values in map
        :type skip_nan: bool
        :param talk: option for printing messages
        :type talk: bool
        :return: raster object based on the first object found in the test_collection
        :rtype: :class:`Raster`
        """
        output_raster = self.reducer(
            reducer_func=np.std,
            reduction_name="{} SD".format(self.name),
            skip_nan=skip_nan,
            talk=talk,
        )
        return output_raster

    def min(self, skip_nan=False, talk=False):
        """Reduce Collection to the Min raster

        :param skip_nan: Option for skipping NaN values in map
        :type skip_nan: bool
        :param talk: option for printing messages
        :type talk: bool
        :return: raster object based on the first object found in the test_collection
        :rtype: :class:`Raster`
        """
        output_raster = self.reducer(
            reducer_func=np.min,
            reduction_name="{} Min".format(self.name),
            skip_nan=skip_nan,
            talk=talk,
        )
        return output_raster

    def max(self, skip_nan=False, talk=False):
        """Reduce Collection to the Max raster

        :param skip_nan: Option for skipping NaN values in map
        :type skip_nan: bool
        :param talk: option for printing messages
        :type talk: bool
        :return: raster object based on the first object found in the test_collection
        :rtype: :class:`Raster`
        """
        output_raster = self.reducer(
            reducer_func=np.max,
            reduction_name="{} Max".format(self.name),
            skip_nan=skip_nan,
            talk=talk,
        )
        return output_raster

    def sum(self, skip_nan=False, talk=False):
        """Reduce Collection to the Sum raster

        :param skip_nan: Option for skipping NaN values in map
        :type skip_nan: bool
        :param talk: option for printing messages
        :type talk: bool
        :return: raster object based on the first object found in the test_collection
        :rtype: :class:`Raster`
        """
        output_raster = self.reducer(
            reducer_func=np.sum,
            reduction_name="{} Sum".format(self.name),
            skip_nan=skip_nan,
            talk=talk,
        )
        return output_raster

    def percentile(self, percentile, skip_nan=False, talk=False):
        """Reduce Collection to the Nth Percentile raster

        :param percentile: Nth percentile (from 0 to 100)
        :type percentile: float
        :param skip_nan: Option for skipping NaN values in map
        :type skip_nan: bool
        :param talk: option for printing messages
        :type talk: bool
        :return: raster object based on the first object found in the test_collection
        :rtype: :class:`Raster`
        """
        output_raster = self.reducer(
            reducer_func=np.percentile,
            reduction_name="{} {}th percentile".format(self.name, str(percentile)),
            skip_nan=skip_nan,
            talk=talk,
            extra_arg=percentile,
        )
        return output_raster

    def median(self, skip_nan=False, talk=False):
        """Reduce Collection to the Median raster

        :param skip_nan: Option for skipping NaN values in map
        :type skip_nan: bool
        :param talk: option for printing messages
        :type talk: bool
        :return: raster object based on the first object found in the test_collection
        :rtype: :class:`Raster`
        """
        output_raster = self.reducer(
            reducer_func=np.median,
            reduction_name="{} Median".format(self.name),
            skip_nan=skip_nan,
            talk=talk,
        )
        return output_raster

    def get_collection_stats(self):
        """Get basic statistics from test_collection.

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
        # convert to integer
        df_aux["Count"] = df_aux["Count"].astype(dtype="uint32")
        return df_aux

    def get_views(self, show=True, folder="./output", dpi=300, fig_format="jpg"):
        """Plot all basic pannel of raster maps in test_collection.

        :param show: boolean to show plot instead of saving,
        :type show: bool
        :param folder: folder_main to output folder, defaults to ``./output``
        :type folder: str
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
        """View Bounding Boxes of Raster test_collection

        :param colors: list of colors for plotting. expected to be the same runsize of catalog
        :type colors: list
        :param datapoints: option to plot datapoints as well, defaults to False
        :type datapoints: bool
        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: folder_main to output folder, defaults to ``./output``
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
            lst_colors = get_colors(size=len(self.catalog))
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
    The raster test_collection base dataset.

    This data strucute is designed for holding and comparing :class:`QualiRaster`` objects.
    """

    def __init__(self, name):
        """Deploy Qualitative Raster Series

        :param name: :class:`RasterSeries.name`` name attribute
        :type name: str
        :param varname: :class:`Raster.varname`` variable name attribute, defaults to None
        :type varname: str
        :param varalias: :class:`Raster.varalias`` variable alias attribute, defaults to None
        :type varalias: str
        """
        super().__init__(name=name)

    def load(self, name, asc_file, prj_file=None, table_file=None):
        """Load a :class:`QualiRaster`` base_object from ``.asc`` raster file.

        :param name: :class:`Raster.name`` name attribute
        :type name: str
        :param asc_file: folder_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: folder_main to ``.prj`` projection file
        :type prj_file: str
        :param table_file: folder_main to ``.txt`` table file
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
        # append to test_collection
        self.append(new_object=rst_aux)
        # delete aux
        del rst_aux
        return None

class RasterSeries(RasterCollection):
    """A :class:`RasterCollection`` where date matters and all maps in collections are
    expected to be the same variable, same projection and same grid.
    """

    def __init__(self, name, varname, varalias, units, dtype="float32"):
        """Deploy RasterSeries

        :param name: :class:`RasterSeries.name`` name attribute
        :type name: str
        :param varname: :class:`Raster.varname`` variable name attribute, defaults to None
        :type varname: str
        :param varalias: :class:`Raster.varalias`` variable alias attribute, defaults to None
        :type varalias: str
        :param units: :class:`Raster.units`` units attribute, defaults to None
        :type units: str
        """
        super().__init__(name=name)
        self.varname = varname
        self.varalias = varalias
        self.units = units
        self.dtype = dtype

    def load(self, name, date, asc_file, prj_file=None):
        """Load a :class:`Raster`` base_object from a ``.asc`` raster file.

        :param name: :class:`Raster.name`` name attribute
        :type name: str
        :param date: :class:`Raster.date`` date attribute, defaults to None
        :type date: str
        :param asc_file: folder_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: folder_main to ``.prj`` projection file
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
        # append to test_collection
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
        """Load all rasters from a folder by following a name pattern. Date is expected to be at the end of name before file extension.

        :param folder: folder_main to folder
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

    def apply_aoi_masks(self, grid_aoi, inplace=False):
        """Batch method to apply AOI mask over all maps in test_collection

        :param grid_aoi: aoi grid
        :type grid_aoi: :class:`numpy.ndarray`
        :param inplace: overwrite the main grid if True, defaults to False
        :type inplace: bool
        :return: None
        :rtype: None
        """
        for name in self.collection:
            self.collection[name].apply_aoi_mask(grid_aoi=grid_aoi, inplace=inplace)
        return None

    def release_aoi_masks(self):
        """Batch method to release the AOI mask over all maps in test_collection

        :return: None
        :rtype: None
        """
        for name in self.collection:
            self.collection[name].release_aoi_mask()
        return None

    def rebase_grids(self, base_raster, talk=False):
        """Batch method for rebase all maps in test_collection

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
        self,
        show=True,
        folder="./output",
        view_specs=None,
        dpi=300,
        fig_format="jpg",
        talk=False,
    ):
        """Plot all basic pannel of raster maps in test_collection.

        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: folder_main to output folder, defaults to ``./output``
        :type folder: str
        :param view_specs: specifications dictionary, defaults to None
        :type view_specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :param talk: option for print messages
        :type talk: bool
        :return: None
        :rtype: None
        """

        # get stats
        df_stats = self.get_collection_stats()
        n_vmin = df_stats["Min"].max()
        n_vmax = df_stats["Max"].max()

        # plot loop
        for k in self.collection:
            rst_lcl = self.collection[k]
            s_name = rst_lcl.name
            if talk:
                print("plotting view of {}...".format(s_name))

            # handle specs
            rst_lcl.view_specs["vmin"] = n_vmin
            rst_lcl.view_specs["vmax"] = n_vmax
            rst_lcl.view_specs["hist_vmax"] = 0.05
            if view_specs is None:
                pass
            else:
                # overwrite incoming specs
                for k in view_specs:
                    rst_lcl.view_specs[k] = view_specs[k]
            # plot
            rst_lcl.view(
                show=show,
                folder=folder,
                filename=s_name,
                dpi=dpi,
                fig_format=fig_format,
            )
        return None

    def view_series_stats(
        self,
        statistic="Mean",
        folder="./output",
        filename=None,
        specs=None,
        show=True,
        dpi=150,
        fig_format="jpg",
    ):
        """View raster series statistics

        :param statistic: statistc to view. Default mean
        :type statistic: str
        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: folder_main to output folder, defaults to ``./output``
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

class QualiRasterSeries(RasterSeries):
    """A :class:`RasterSeries`` where date matters and all maps in collections are
    expected to be :class:`QualiRaster`` with the same variable, same projection and same grid.
    """

    def __init__(self, name, varname, varalias, dtype="uint8"):
        """Deploy Qualitative Raster Series

        :param name: :class:`RasterSeries.name`` name attribute
        :type name: str
        :param varname: :class:`Raster.varname`` variable name attribute, defaults to None
        :type varname: str
        :param varalias: :class:`Raster.varalias`` variable alias attribute, defaults to None
        :type varalias: str
        """
        super().__init__(
            name=name, varname=varname, varalias=varalias, dtype=dtype, units="ID"
        )
        self.table = None

    def update_table(self, clear=True):
        """Update series table (attributes)
        :param clear: option for clear table from unfound values. default: True
        :type clear: bool
        :return: None
        :rtype: None
        """
        if len(self.catalog) == 0:
            pass
        else:
            for i in range(len(self.catalog)):
                _name = self.catalog["Name"].values[i]
                # clear table from unfound values
                if clear:
                    self.collection[_name].clear_table()
                # concat all tables
                if i == 0:
                    self.table = self.collection[_name].table.copy()
                else:
                    self.table = pd.concat(
                        [self.table, self.collection[_name].table.copy()]
                    )
        # clear from duplicates
        self.table = self.table.drop_duplicates(subset="Id", keep="last")
        self.table = self.table.reset_index(drop=True)
        return None

    def append(self, raster):
        """Append a :class:`Raster`` base_object to test_collection. Pre-existing objects with the same :class:`Raster.name`` attribute are replaced

        :param raster: incoming :class:`Raster`` to append
        :type raster: :class:`Raster`
        """
        super().append(new_object=raster)
        self.update_table()
        return None

    def load(self, name, date, asc_file, prj_file=None, table_file=None):
        """Load a :class:`QualiRaster`` base_object from ``.asc`` raster file.

        :param name: :class:`Raster.name`` name attribute
        :type name: str
        :param date: :class:`Raster.date`` date attribute
        :type date: str
        :param asc_file: folder_main to ``.asc`` raster file
        :type asc_file: str
        :param prj_file: folder_main to ``.prj`` projection file
        :type prj_file: str
        :param table_file: folder_main to ``.txt`` table file
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
        # append to test_collection
        self.append(new_object=rst_aux)
        # delete aux
        del rst_aux

    def w_load_file(self, file_info):
        """Worker function to load a single file."""
        s_name, s_date_map, asc_file, prj_file, table_file = file_info
        self.load(
            name=s_name,
            date=s_date_map,
            asc_file=asc_file,
            prj_file=prj_file,
            table_file=table_file,
        )

    def load_folder(self, folder, table_file, name_pattern="map_*", talk=False, use_parallel=False, num_threads=None):
        """
        Load all rasters from a folder by following a name pattern. Date is expected to be at the end of name before file extension.
        Supports both serial and parallel processing using threads.

        :param folder: folder_main to folder
        :type folder: str
        :param table_file: folder_main to table file
        :type table_file: str
        :param name_pattern: name pattern. example map_*
        :type name_pattern: str
        :param talk: option for printing messages
        :type talk: bool
        :param use_parallel: flag to use parallel processing
        :type use_parallel: bool
        :param num_threads: number of threads to use
        :type num_threads: int, optional
        :return: None
        :rtype: None
        """
        lst_maps = glob.glob(f"{folder}/{name_pattern}.asc")
        lst_prjs = glob.glob(f"{folder}/{name_pattern}.prj")

        if talk:
            print("loading folder...")

        if use_parallel:
            import threading
            # Prepare data for parallel processing
            file_info_list = [(os.path.basename(asc_file).split(".")[0],
                               asc_file.split("_")[-1].split(".")[0],
                               asc_file,
                               prj_file,
                               table_file)
                              for asc_file, prj_file in zip(lst_maps, lst_prjs)]

            threads = []
            for file_info in file_info_list:
                #print(file_info)
                thread = threading.Thread(target=self.w_load_file, args=(file_info,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        else:
            # serial processing
            for i in range(len(lst_maps)):
                asc_file = lst_maps[i]
                # print(asc_file)
                prj_file = lst_prjs[i]
                # get name
                s_name = os.path.basename(asc_file).split(".")[0]
                if talk:
                    print(s_name)
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

    def _load_folder(self, folder, table_file, name_pattern="map_*", talk=False):
        """Load all rasters from a folder by following a name pattern. Date is expected to be at the end of name before file extension.

        :param folder: folder_main to folder
        :type folder: str
        :param table_file: folder_main to table file
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
            #print(asc_file)
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
        """Get export_areas prevalance for all series

        :return: dataframe of series export_areas
        :rtype: :class:`pandas.DataFrame`
        """
        # compute export_areas for each raster
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
        export_areas=True,
        folder="./output",
        filename=None,
        dpi=300,
        fig_format="jpg",
    ):
        """View series areas

        :param specs: specifications dictionary, defaults to None
        :type specs: dict
        :param show: option to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: folder_main to output folder, defaults to ``./output``
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
            "filter_by_id": None,  # list of ids
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # compute export_areas
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
            if specs["filter_by_id"] == None:
                # filter series
                _df = df_areas.query("Id == {}".format(_id)).copy()
                plt.plot(_df["Date"], _df["Area_%"], color=_color, label=_name)
            else:
                if _id in specs["filter_by_id"]:
                    # filter series
                    _df = df_areas.query("Id == {}".format(_id)).copy()
                    plt.plot(_df["Date"], _df["Area_%"], color=_color, label=_name)
                else:
                    pass
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
            # save fig
            plt.savefig("{}/{}.{}".format(folder, filename, fig_format), dpi=dpi)
            # save areas
            if export_areas:
                df_areas.to_csv("{}/{}.csv".format(folder, filename), sep=";", index=False)
        plt.close(fig)
        return None

    def get_views(
        self,
        show=True,
        export_areas=True,
        filter=False,
        n_filter=6,
        folder="./output",
        view_specs=None,
        dpi=300,
        fig_format="jpg",
        talk=False,
    ):
        """Plot all basic pannel of raster maps in test_collection.

        :param show: boolean to show plot instead of saving, defaults to False
        :type show: bool
        :param folder: folder_main to output folder, defaults to ``./output``
        :type folder: str
        :param view_specs: specifications dictionary, defaults to None
        :type view_specs: dict
        :param dpi: image resolution, defaults to 96
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :param talk: option for print messages
        :type talk: bool
        :return: None
        :rtype: None
        """

        # plot loop
        for k in self.collection:
            rst_lcl = self.collection[k]
            s_name = rst_lcl.name
            if talk:
                print("plotting view of {}...".format(s_name))
            if view_specs is None:
                pass
            else:
                # overwrite incoming specs
                for k in view_specs:
                    rst_lcl.view_specs[k] = view_specs[k]
            # plot
            rst_lcl.view(
                show=show,
                export_areas=export_areas,
                folder=folder,
                filename=s_name,
                dpi=dpi,
                fig_format=fig_format,
                filter=filter,
                n_filter=n_filter,
            )
        return None


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8")

    r = Raster()
    r.load(
        asc_file="C:/plans/docs/datasets/topo/slope.asc",
        prj_file="C:/plans/docs/datasets/topo/slope.prj"
    )
    r.view()
