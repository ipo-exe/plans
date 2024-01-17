"""
Handle chronological datasets.

Description:
    The ``datasets.chrono`` module provides objects to handle chronological (time series) data.

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
import numpy as np
import pandas as pd
from plans.datasets.core import *


# ------------- TIME SERIES OBJECTS -------------  #

class RainSeries(TimeSeries):
    """A class for representing and working with rainfall time series data.

    The ``RainSeries`` class extends the ``TimeSeries`` class and focuses on handling rainfall data.

    **Examples:**

    >>> rainfall_data = RainSeries(name="Monsoon2022", alias="MS2022")

    """

    def __init__(self, name="MyRainfallSeries", alias=None):
        """Initialize a RainSeries object.

        :param name: str, optional
            Name of the rainfall series. Default is "MyRainfallSeries".
        :type name: str

        :param alias: str, optional
            Alias for the rainfall series. Default is None.
        :type alias: str

        """
        # Use the superior initialization from the parent class (TimeSeries)
        super().__init__(name, alias=alias)
        self.varname="Rain"
        self.varfield="P"
        self.units="mm"
        # Overwrite attributes specific to RainSeries
        self.name_object = "Rainfall Time Series"
        self.agg = "sum"  # Aggregation method, set to "sum" by default
        self.gapsize = 7 * 72  # Maximum gap size of 1 week assuming measure device turns off when is not raining
        self.outlier_min = 0
        self.rawcolor = "darkgray"

    def interpolate_gaps(self, inplace=False, method=None,):
        # overwrite interpolation method with constant=0
        super().interpolate_gaps(method="constant", constant=0, inplace=inplace)

    def _set_frequency(self):
        super()._set_frequency()
        # overwrite gapsize to 1 week
        dict_gaps = {
            "second": int(7 * 24 * 60 * 60),
            "minute": int(7 * 24 * 3),
            "hour": int(7 * 24),
            "day": int(7),
            "month": 1,
            "year": 1,
        }
        self.gapsize = dict_gaps[self.dtres]
        return None

class StageSeries(TimeSeries):
    """A class for representing and working with river stage time series data.

    The ``StageSeries`` class extends the ``TimeSeries`` class and focuses on handling river stage data.

    **Attributes:**

    - ``name`` (str): Name of the river stage series.
    - ``alias`` (str): Alias for the river stage series.
    - ``varname`` (str): Variable name, set to "Stage".
    - ``varfield`` (str): Variable field, set to "H".
    - ``units`` (str): Measurement units, set to "cm".
    - ``code`` (str or None): Custom code associated with the river stage series. Default is None.
    - ``agg`` (str): Aggregation method, set to "mean" by default.
    - ``gapsize`` (int): Maximum gap size allowed for data interpolation, set to 10.
    - ``x`: X-coordinate information associated with the river stage series. Default is None.
    - ``y`: Y-coordinate information associated with the river stage series. Default is None.

    **Examples:**

    >>> river_stage_data = StageSeries(name="River2022", alias="R2022")

    """

    def __init__(self, name="MyStageSeries", alias=None):
        """Initialize a StageSeries object.

        :param name: str, optional
            Name of the river stage series. Default is "MyStageSeries".
        :type name: str

        :param alias: str, optional
            Alias for the river stage series. Default is None.
        :type alias: str

        """

        # Use the superior initialization from the parent class (TimeSeries)
        super().__init__(name, alias=alias)
        self.varname = "Stage"
        self.varfield = "H"
        self.units = "cm"
        # Overwrite attributes specific to StageSeries
        self.name_object = "Stage Time Series"
        self.agg = "mean"  # Aggregation method, set to "mean" by default
        self.gapsize = 10  # Maximum gap size allowed for data interpolation
        # Extra attributes specific to StageSeries
        self.upstream_area = None

    def get_metadata(self):
        """Get all metadata from the base object.

        :return: metadata
        :rtype: dict

        **Notes:**

        - Metadata includes information from the base class (TimeSeries) and additional TempSeries-specific attributes.
        - The returned dictionary contains key-value pairs with metadata information.

        **Examples:**

        >>> metadata = stage_data.get_metadata()
        >>> print(metadata)

        """
        # Get metadata from the base class (TimeSeries)
        base_metadata = super().get_metadata()
        # Additional TimeSeries-specific metadata
        extra_metadata = {
            "UpstreamArea": self.upstream_area,
        }
        # Combine both base and specific metadata
        base_metadata.update(extra_metadata)
        return base_metadata

class TempSeries(TimeSeries):
    """A class for representing and working with temperature time series data.

    The ``TemperatureSeries`` class extends the ``TimeSeries`` class and focuses on handling temperature data.

    **Examples:**

    >>> temperature_data = TempSeries(name="Temperature2022", alias="Temp2022")

    """
    def __init__(self, name="MyTemperatureSeries", alias=None):
        """Initialize a TempSeries object.

        :param name: str, optional
            Name of the temperature series. Default is "MyTemperatureSeries".
        :type name: str

        :param alias: str, optional
            Alias for the temperature series. Default is None.
        :type alias: str

        """
        # Use the superior initialization from the parent class (TimeSeries)
        super().__init__(name, alias=alias)
        self.varname = "Temperature"
        self.varfield = "Temp"
        self.units = "Celcius"
        # Overwrite attributes specific
        self.name_object = "Temp Time Series"
        self.agg = "mean"  # Aggregation method, set to "sum" by default
        self.gapsize = 6  # Maximum gap size of 6 hours assuming hourly Temperature
        self.datarange_max = 50
        self.datarange_min = -20
        self.rawcolor = "orange"


class FlowSeries(TimeSeries):
    """A class for representing and working with streamflow time series data."""
    def __init__(self, name="MyFlowSeries", alias=None):
        """Initialize a FlowSeries object.

        :param name: str, optional
            Name of the streamflow series. Default is "MyFlowSeries".
        :type name: str

        :param alias: str, optional
            Alias for the flow

        """
        # Use the superior initialization from the parent class (TimeSeries)
        super().__init__(name, alias=alias)
        self.varname = "Flow"
        self.varfield = "Q"
        self.units = "m3/s"
        # Overwrite attributes specific
        self.name_object = "Flow Time Series"
        self.agg = "mean"  # Aggregation method, set to "mean" by default
        self.gapsize = 6  # Maximum gap size of 6 hours assuming hourly Flow
        self.datarange_max = 300000  # Amazon discharge
        self.datarange_min = 0
        self.rawcolor = "navy"
        # Specific attributes
        self.upstream_area = None # in sq km

    @staticmethod
    def frequency(dataframe, var_field, zero=True, step=1):
        """This fuction performs a frequency analysis on a given time series.

        :param dataframe: pandas DataFrame object with time series
        :param var_field: string of variable field
        :param zero: boolean control to consider values of zero. Default: True
        :return: pandas DataFrame object with the following columns:
             - 'Pecentiles' - percentiles in % of array values (from 0 to 100 by steps of 1%)
             - 'Exceedance' - exeedance probability in % (reverse of percentiles)
             - 'Frequency' - count of values on the histogram bin defined by the percentiles
             - 'Probability'- local bin empirical probability defined by frequency/count
             - 'Values' - values percentiles of bins

        """
        # get dataframe right
        in_df = dataframe[[var_field]].copy()
        in_df = in_df.dropna()
        if zero:
            pass
        else:
            mask = in_df[var_field] != 0
            in_df = in_df[mask]
        def_v = in_df[var_field].values
        ptles = np.arange(0, 100 + step, step)
        cfc = np.percentile(def_v, ptles)
        exeed = 100 - ptles
        freq = np.histogram(def_v, bins=len(ptles))[0]
        prob = freq / np.sum(freq)
        out_dct = {
            'Percentiles': ptles,
            'Exceedance': exeed,
            'Frequency': freq,
            'Probability': prob,
            'Values': cfc
        }
        out_df = pd.DataFrame(out_dct)
        return out_df

    @staticmethod
    def view_cfcs(freqs, specs=None, show=True, colors=None, labels=None):
        import matplotlib.pyplot as plt
        default_specs = {
            "folder": "C:/data",
            "filename": "cfcs",
            "fig_format": "jpg",
            "dpi": 300,
            "width": 4,
            "height": 6,
            "xmin": 0,
            "xmax": 100,
            "ymin": 0,
            "ymin_log": 1,
            "ymax": None,
            "log": True,
            "title": "CFCs",
            "ylabel": "m3/s",
            "xlabel": "Exeed. Prob. (%)"

        }
        # get specs
        if specs is not None:
            default_specs.update(specs)
        specs = default_specs.copy()

        # --------------------- figure setup --------------------- #
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height

        # handle min max
        if specs["ymax"] is None:
            lst_max = [freq["Values"].max() for freq in freqs]
            specs["ymax"] = max(lst_max)

        if colors is None:
            colors = ["navy" for freq in freqs]
        if labels is None:
            labels = [None for freq in freqs]

        # --------------------- plotting --------------------- #
        for i in range(len(freqs)):
            plt.plot(
                freqs[i]["Exceedance"],
                freqs[i]["Values"],
                color=colors[i],
                label=labels[i]
            )
        if labels is not None:
            plt.legend()

        # --------------------- post-plotting --------------------- #
        # set basic plotting stuff
        plt.title(specs["title"])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["xlabel"])
        plt.xlim(specs["xmin"], specs["xmax"])
        if specs["log"]:
            plt.ylim(specs["ymin_log"], 1.2 * specs["ymax"])
            plt.yscale("log")
        else:
            plt.ylim(specs["ymin"], 1.2 * specs["ymax"])

        # Get current axes
        ax = plt.gca()
        # Set the y-ticks more densely
        ax.set_yticks([1, 2.5, 5, 10, 25, 50, 100, 250, 500])

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






# ------------- TIME SERIES COLLECTIONS -------------  #

class RainSeriesSamples(TimeSeriesSpatialSamples):
    # todo docstring
    def __init__(self, name="MyRSColection"): # todo docstring

        super().__init__(name=name, base_object=RainSeries)
        # overwrite parent attributes
        self.name_object = "Rainfall Series Samples"
        self._set_view_specs()

class TempSeriesSamples(TimeSeriesSpatialSamples):  # todo docstring
    def __init__(self, name="MyTempSColection"): # todo docstring
        super().__init__(name=name, base_object=TempSeries)
        # overwrite parent attributes
        self.name_object = "Temperature Series Sample"
        self._set_view_specs()

def main():
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8")

    f = "C:/data/series.csv"

    qts = FlowSeries()
    qts.load_data(file_data=f, input_dtfield="Date", input_varfield="Fobs")
    print(qts.dtfreq)
    print(qts.data.head(10))
    qts.standardize()
    print(qts.data.head(10))
    freq = FlowSeries.frequency(dataframe=qts.data, var_field="Q")
    print(freq.head())
    freq2 = freq.copy()
    freq2["Values"] = freq2["Values"] + 10
    specs = {
        "ylabel": "mm",
    }
    FlowSeries.view_cfcs(freqs=[freq, freq2], specs=specs, colors=["blue", "red"], show=False)



if __name__ == "__main__":
    main()

