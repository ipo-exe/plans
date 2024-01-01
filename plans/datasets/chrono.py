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

>>> from plans.datasets.chrono import *

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
        super().__init__(name, alias=alias, varname="Rain", varfield="P", units="mm")
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
        super().__init__(name, alias=alias, varname="Stage", varfield="H", units="cm")
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
        super().__init__(
            name, alias=alias, varname="Temperature", varfield="Temp", units="Celsius"
        )
        # Overwrite attributes specific
        self.name_object = "Temp Time Series"
        self.agg = "mean"  # Aggregation method, set to "sum" by default
        self.gapsize = 6  # Maximum gap size of 6 hours assuming hourly Temperature
        self.datarange_max = 50
        self.datarange_min = -20
        self.rawcolor = "orange"


# ------------- TIME SERIES COLLECTIONS -------------  #

class RainSeriesSamples(TimeSeriesSpatialSamples):
    # todo docstring
    def __init__(self, name="MyRSColection"):
        # todo docstring
        super().__init__(name=name, base_object=RainSeries)
        # overwrite parent attributes
        self.name_object = "Rainfall Series Samples"
        self._set_view_specs()

class TempSeriesSamples(TimeSeriesSpatialSamples):
    # todo docstring
    def __init__(self, name="MyTempSColection"):
        # todo docstring
        super().__init__(name=name, base_object=TempSeries)
        # overwrite parent attributes
        self.name_object = "Temperature Series Sample"
        self._set_view_specs()


if __name__ == "__main__":
    t = TimeSeries()
