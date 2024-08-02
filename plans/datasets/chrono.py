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
import matplotlib.pyplot as plt
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
        self.varalias = "H"
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

    def get_annual_maxima(self):
        from scipy import stats

        # set agg to max
        self.agg = "max"
        df_an = self.upscale(freq="YS", bad_max=self.gapsize, inplace=False)
        df_an = df_an.dropna()

        # fix dates to match actual maxima
        df_an["tag1"] = df_an[self.dtfield].dt.year.astype(str) + " - " + df_an[self.varfield].astype(str)
        df_d = self.data.copy()
        df_d["tag2"] = df_d[self.dtfield].dt.year.astype(str) + " - " + df_d[self.varfield].astype(str)
        # join
        df_an = pd.merge(left=df_an, left_on="tag1", right=df_d, right_on="tag2")
        df_an = df_an.drop_duplicates(subset="tag1").reset_index(drop=True)

        # remake df
        df_an = pd.DataFrame(
            {
                self.dtfield : df_an[f"{self.dtfield}_y"],
                f"{self.varfield}_amax": df_an[f"{self.varfield}_y"],
            }
        )

        # get extra values
        df_an = df_an.sort_values(by=f"{self.varfield}_amax", ascending=False).reset_index(drop=True)

        df_an["Rank"] = df_an.index + 1

        df_an["P(X)_Empirical"] = StageSeries.px_empirical(ranks=df_an["Rank"].values)
        df_an["P(X)_Weibull"] = StageSeries.px_weibull(ranks=df_an["Rank"].values)
        df_an["P(X)_Gringorten"] = StageSeries.px_gringorten(ranks=df_an["Rank"].values)

        # model Gumbel using the method of moments

        # using scipy
        # Fit a Gumbel distribution to the data
        params = stats.gumbel_r.fit(df_an[f"{self.varfield}_amax"].values, method="MM") # MLM or MM
        gumbel_a = params[0]
        gumbel_b = params[1]

        # Goodness of fit tests
        ks_stat, ks_p_value = stats.kstest(
            df_an[f"{self.varfield}_amax"].values,
            cdf='gumbel_r',
            args=params
        )
        # accept Null Hypothesis
        is_gumbel = True
        if ks_p_value <= 0.05:
            is_gumbel = False

        # QQ plots
        qq, qq_params = stats.probplot(
            x=df_an[f"{self.varfield}_amax"].values,
            dist="gumbel_r",
            sparams=params
        )#, plot=ax)

        df_qq = pd.DataFrame(
            {
                f"{self.varfield}_amax": qq[1],
                "T-Q": qq[0]
            }
        )

        # compute gumbel values
        df_an["P(X)_Gumbel"] = 1 - StageSeries.gumbel_fx(
            x=df_an[f"{self.varfield}_amax"].values,
            a=gumbel_a,
            b=gumbel_b
        )
        # return period
        df_an["T(X)_Gumbel"] = 1 / df_an["P(X)_Gumbel"]

        # compute uncertainty 90% bands for T(X) in the Annual Series
        t = stats.t.ppf((1 + 0.9) / 2, df=len(df_an) - 1)
        gumbel_se = StageSeries.gumbel_se(
            std_sample=np.std(a=df_an[f"{self.varfield}_amax"].values),
            n_sample=len(df_an),
            tr=df_an["T(X)_Gumbel"]
        )
        # Standar error
        df_an[f"{self.varfield}_amax_SE90"] = t * gumbel_se

        h_max = df_an[f"{self.varfield}_amax"] + df_an[f"{self.varfield}_amax_SE90"]
        h_min = df_an[f"{self.varfield}_amax"] - df_an[f"{self.varfield}_amax_SE90"]
        tr_max = 1 / (1 - StageSeries.gumbel_fx(x=h_max, a=gumbel_a, b=gumbel_b))
        tr_min = 1 / (1 - StageSeries.gumbel_fx(x=h_min, a=gumbel_a, b=gumbel_b))

        df_an["T(X)_Gumbel_P05"] = tr_min
        df_an["T(X)_Gumbel_P95"] = tr_max

        # compute uncertainty 90% bands for a full range of TRs
        trs = np.arange(1.5, 100, step=0.5)
        k = StageSeries.gumbel_freqfactor(trs)
        stages = np.mean(a=df_an[f"{self.varfield}_amax"]) + k * np.std(a=df_an[f"{self.varfield}_amax"])

        gumbel_se = StageSeries.gumbel_se(
            std_sample=np.std(a=df_an[f"{self.varfield}_amax"]),
            n_sample=len(df_an),
            tr=trs
        )
        h_range = t * gumbel_se
        h_max = stages + h_range
        h_min = stages - h_range

        df_trs = pd.DataFrame(
            {
                "T(X)_Gumbel": trs,
                f"{self.varfield}_amax": stages,
                f"{self.varfield}_amax_P05": h_min,
                f"{self.varfield}_amax_P95": h_max
            }
        )

        # reset agg
        self.agg = "mean"

        return {
            "Data": df_an,
            "T_Data": df_trs,
            "Gumbel_a": gumbel_a,
            "Gumbel_b": gumbel_b,
            "KS_Stat": ks_stat,
            "KS_p_value": ks_p_value,
            "IsGumbel": is_gumbel,
            "QQ_Data":df_qq,
            "QQ_c1": qq_params[1],
            "QQ_c0": qq_params[0],
            "QQ_r2": qq_params[2],
        }

    # todo remove this
    def get_mol(self):
        """Get the MEAN ORDINARY LEVEL according to many methods

        :return:
        :rtype:
        """
        # 1) get annual max values
        df_ys = self.upscale(freq="YS", bad_max=10, inplace=False)

        # get basic stats
        mol_sample_mean = df_ys["H"].mean()  # sample mean
        mol_sample_std = df_ys["H"].std()

        # 2yr Return Period using Gumbel
        k = StageSeries.gumbel_freqfactor(t=2)
        mol_2yr = mol_sample_mean + k * mol_sample_std

        # Apply SPU method

        # get the size of series
        size = len(df_ys)
        # Sort values in descending order
        df_ys_sort = df_ys.sort_values(by="H", ascending=False).reset_index(drop=True)
        # compute rank and TR (empirical)
        df_ys_sort["Rank"] = df_ys_sort.index + 1
        df_ys_sort["Prob_exceed"] = df_ys_sort["Rank"] / size
        df_ys_sort["TR"] = size / df_ys_sort["Rank"]
        # filter 3 and 20
        df_ys_spu = df_ys_sort.query("TR >= 3 and TR <= 20")
        mol_spu = df_ys_spu["H"].mean()

        d_out = {
            "MOL": mol_sample_mean,
            "MOL 2yr (Gumbel)": mol_2yr,
            "MOL (SPU)": mol_spu,
            "Annual": df_ys,
            "SPU": df_ys_spu
        }
        return d_out


    @staticmethod
    def gumbel_fx(x, a, b):
        aux_1 = (x - a) / b
        aux_2 = np.exp(- aux_1)
        gumbel_fx = np.exp(- aux_2)
        return gumbel_fx

    @staticmethod
    def gumbel_freqfactor(t=2):
        aux1 = t / (t - 1)
        aux2 = np.log(aux1)
        aux3 = np.log(aux2)
        aux4 = 0.5772 + aux3
        aux5 = - np.sqrt(6) * aux4 / np.pi
        return aux5

    @staticmethod
    def gumbel_se(std_sample, n_sample, tr):
        aux1 = std_sample / np.sqrt(n_sample)
        k = StageSeries.gumbel_freqfactor(t=tr)
        aux2 = 1 + (1.14 * k) + (1.1 * np.square(k))
        aux3 = np.sqrt(aux2)
        return aux1 * aux3



    @staticmethod
    def px_empirical(ranks):
        """Get the empirical exceedance probability P(X)

        :param ranks: vector of ranks
        :type ranks: class:`numpy.array`
        :return: empirical exceedance probability P(X)
        :rtype: class:`numpy.array`
        """
        return ranks / len(ranks)

    @staticmethod
    def px_weibull(ranks):
        """Get the Weibull exceedance probability P(X)

        :param ranks: vector of ranks
        :type ranks: class:`numpy.array`
        :return: Weibull exceedance probability P(X)
        :rtype: class:`numpy.array`
        """
        return ranks / (len(ranks) + 1)
    @staticmethod
    def px_gringorten(ranks):
        """Get the Gringorten exceedance probability P(X)

        :param ranks: vector of ranks
        :type ranks: class:`numpy.array`
        :return: Gringorten exceedance probability P(X)
        :rtype: class:`numpy.array`
        """
        return (ranks - 0.44) / (len(ranks) + 0.12)



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

