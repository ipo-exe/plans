``map_aoi.asc``
------------------------------------------------------------
[`raster map`_] Map of the Area Of Interest (AOI). Expected to be a boolean map (1 and 0 values only). I can be a copy of the main basin map..

 - Data type: 8-bit unsigned integer
 - No-data value: 0


``map_basin_<station alias>.asc``
------------------------------------------------------------
[`raster map`_] Map of the basin related to a given flow station. Alias (unique nickname) of station is expected ate the end of file name. Example: ``map_basin_Reach1.asc``.

 - Data type: 8-bit unsigned integer
 - No-data value: 0


``map_et_<date>.asc``
------------------------------------------------------------
[`raster map`_] Description text.

 - Data type: 32-bit floating point
 - No-data value: -1


``map_hand.asc``
------------------------------------------------------------
[`raster map`_] Description text.

 - Data type: 32-bit floating point
 - No-data value: -1


``map_lst_<date>.asc``
------------------------------------------------------------
[`raster map`_] Description text.

 - Data type: 32-bit floating point
 - No-data value: 0


``map_lulc_<date>.asc``
------------------------------------------------------------
[`raster map`_] Description text.

 - Data type: 8-bit unsigned integer
 - No-data value: 0


``map_ndvi_<date>.asc``
------------------------------------------------------------
[`raster map`_] Description text.

 - Data type: 32-bit floating point
 - No-data value: -9999


``map_slope.asc``
------------------------------------------------------------
[`raster map`_] Description text.

 - Data type: 32-bit floating point
 - No-data value: -1


``map_soils.asc``
------------------------------------------------------------
[`raster map`_] Description text.

 - Data type: 8-bit unsigned integer
 - No-data value: 0


``map_twi.asc``
------------------------------------------------------------
[`raster map`_] Description text.

 - Data type: 32-bit floating point
 - No-data value: -1


``series_prec_<station alias>.txt``
------------------------------------------------------------
[`daily time series`_] Description text.



``series_stage_<station alias>.txt``
------------------------------------------------------------
[`daily time series`_] Description text.



``series_temp_<station alias>.txt``
------------------------------------------------------------
[`daily time series`_] Description text.



``series_curve_<station alias>.txt``
------------------------------------------------------------
[`time series`_] Description text.



``table_lulc.txt``
------------------------------------------------------------
[`table`_] Description text.



``table_parameters.txt``
------------------------------------------------------------
[`table`_] Description text.



``table_soils.txt``
------------------------------------------------------------
[`table`_] Description text.



``table_stations.txt``
------------------------------------------------------------
[`table`_] Description text.



.. _raster map: https://plans-docs.readthedocs.io/en/latest/iofiles.html#raster-maps
.. _table: https://plans-docs.readthedocs.io/en/latest/iofiles.html#tables
.. _time series: https://plans-docs.readthedocs.io/en/latest/iofiles.html#time-series
.. _daily time series: https://plans-docs.readthedocs.io/en/latest/iofiles.html#daily-time-series
