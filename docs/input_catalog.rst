``acc``
------------------------------------------------------------
[`io-raster`_] Map of accumulated drainage area

 - Map Units: sq. meters
 - Expected in folder(s): ``datasets/topo``.


``basins``
------------------------------------------------------------
[`io-qualiraster`_] Map of modelled basins

 - Map Units: Id code
 - Expected in folder(s): ``datasets/basins``.


.. note::

	This map can include ungauged basins



``dem``
------------------------------------------------------------
[`io-raster`_] Map of elevation (digital elevation model)

 - Map Units: meters
 - Expected in folder(s): ``datasets/topo``.


``et_*``
------------------------------------------------------------
[`io-raster`_] Map of evapotranspiration estimated by remote-sensing

 - Map Units: mm
 - Suffix: Date in the format YYYY-MM-DD
 - Expected in folder(s): ``datasets/rs/et``.


``hand``
------------------------------------------------------------
[`io-raster`_] Map of Height Above the Nearest Drainage

 - Map Units: meters
 - Expected in folder(s): ``datasets/topo``.


``ldd``
------------------------------------------------------------
[`io-raster`_] Map of local drain direction (PC raster convention)

 - Map Units: Id code
 - Expected in folder(s): ``datasets/topo``.


``litho``
------------------------------------------------------------
[`io-qualiraster`_] Map of lithological classes

 - Map Units: Id code
 - Expected in folder(s): ``datasets/soil``.


``lulc_*``
------------------------------------------------------------
[`io-qualiraster`_] Map of Land Use and Land Cover classes

 - Map Units: Id code
 - Suffix: Date in the format YYYY-MM-DD
 - Expected in folder(s): ``datasets/lulc/bas``; ``datasets/lulc/bau``; ``datasets/lulc/nbs``; ``datasets/lulc/obs``.


``ndvi_*``
------------------------------------------------------------
[`io-raster`_] Map of the NDVI vegetation index

 - Map Units: index
 - Suffix: Date in the format YYYY-MM-DD
 - Expected in folder(s): ``datasets/rs/ndvi``.


``outlets``
------------------------------------------------------------
[`io-qualiraster`_] Map of basin outlets

 - Map Units: Id code
 - Expected in folder(s): ``datasets/basins``.


.. note::

	The outlet is only one pixel per basin



.. warning::

	This map must be consistend with the ``acc`` map



``rain_zones``
------------------------------------------------------------
[`io-qualiraster`_] Map of rain gauge zones

 - Map Units: Id code
 - Expected in folder(s): ``datasets/rain/bau``; ``datasets/rain/obs``.


``slope``
------------------------------------------------------------
[`io-raster`_] Map of slope

 - Map Units: degrees
 - Expected in folder(s): ``datasets/topo``.


``soils``
------------------------------------------------------------
[`io-qualiraster`_] Map of soil types

 - Map Units: Id code
 - Expected in folder(s): ``datasets/soil``.


``twi``
------------------------------------------------------------
[`io-raster`_] Map of Topographical Wetness Index

 - Map Units: Index
 - Expected in folder(s): ``datasets/topo``.


``basins_info``
------------------------------------------------------------
[`io-attribute`_] Relational table for basins

 - Basic Fields:
	 - ``Id``: Unique Id number (integer)
	 - ``Name``: Unique name
	 - ``Alias``: Unique short name
	 - ``Color``: Unique color code

 - Extra Fields:
	 - ``X``: longitude coorditate (m)
	 - ``Y``: latitude coordinate (m)
	 - ``Downstream_Id``: code Id of downstream basin
	 - ``UpstreamArea``: basin drainage area (sq. m)
	 - ``Code``: field code of basin
	 - ``Source``: source of stream gauge data
	 - ``Description``: basin description

 - Expected in folder(s): ``datasets/basins``.


``clim_*``
------------------------------------------------------------
[`io-timeseries`_] Time series of climatic variables

 - Basic Fields:
	 - ``Datetime``: Timestamp in the format YYYY-MM-DD HH

 - Extra Fields:
	 - ``T``: temperature (Celcius)
	 - ``Ws``: wind speed (m/s)

 - Expected in folder(s): ``datasets/clim/bau``; ``datasets/clim/obs``.


``clim_info``
------------------------------------------------------------
[`io-attribute`_] Relational table for climatic stations

 - Basic Fields:
	 - ``Id``: Unique Id number (integer)
	 - ``Name``: Unique name
	 - ``Alias``: Unique short name
	 - ``Color``: Unique color code

 - Extra Fields:
	 - ``X``: longitude coorditate (m)
	 - ``Y``: latitude coordinate (m)
	 - ``Code``: field code of climatic station
	 - ``Source``: source of climatic station data
	 - ``Description``: climatic station description

 - Expected in folder(s): ``datasets/clim/bau``; ``datasets/clim/obs``.


``rain_*``
------------------------------------------------------------
[`io-timeseries`_] Time series of rainfall

 - Basic Fields:
	 - ``Datetime``: Timestamp in the format YYYY-MM-DD HH

 - Extra Fields:
	 - ``P``: rainfall (mm)

 - Suffix: Alias of rain gauge
 - Expected in folder(s): ``datasets/rain/bau``; ``datasets/rain/obs``.


``rain_info``
------------------------------------------------------------
[`io-attribute`_] Relational table for rain gauges

 - Basic Fields:
	 - ``Id``: Unique Id number (integer)
	 - ``Name``: Unique name
	 - ``Alias``: Unique short name
	 - ``Color``: Unique color code

 - Extra Fields:
	 - ``X``: longitude coorditate (m)
	 - ``Y``: latitude coordinate (m)
	 - ``Code``: field code of rain gauge
	 - ``Source``: source of rain gauge data
	 - ``Description``: rain gauge description

 - Expected in folder(s): ``datasets/rain/bau``; ``datasets/rain/obs``.


``stage_*``
------------------------------------------------------------
[`io-timeseries`_] Time series of river stage

 - Basic Fields:
	 - ``Datetime``: Timestamp in the format YYYY-MM-DD HH

 - Extra Fields:
	 - ``H``: river stage (cm)
	 - ``Q``: rive flow (m3/s)

 - Suffix: Alias of stream gauge
 - Expected in folder(s): ``datasets/basins``.


