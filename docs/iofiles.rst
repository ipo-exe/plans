.. image:: https://raw.githubusercontent.com/ipo-exe/plans/main/docs/figs/logo.png
    :width: 150 px
    :align: center
    :alt: Logo


############################################
I/O Reference
############################################

This is Input and Output reference documentation of ``plans`` tool.

.. toctree::
   iofiles


********************************************
Data structures
********************************************

Files used in ``plans`` are related to the following two data structures:

- Table
- Raster map

A **Table** can store a frame of data in rows and columns in a single file.
A **Raster map** can store a map grid in a matrix of numbers in a single file.
Extra files may be required for complete information about the map.

Input files must be formatted in by standard way, otherwise the tool is not going to work.
The standards are meant to be simple and user-friendly for any human and Operating System.
All kinds of files can be opened and edited by hand in Notepad-like applications. They are described below.

.. _table:

Tables
============================================

A **table** in ``plans`` is a frame of data defined by rows and columns. Each column represents a **field** that must be *homogeneous*.
This means that each field stores the same **data type**, like text, datetime, integer numbers or real numbers.
The first row stores the names of the fields. The subsequent rows stores the data itself.

A table must follow this general rules:

- the file must be a plain file with ``.csv`` or ``.txt`` extension
- semi-column ``;`` must be the separator of columns
- period ``.`` must be the separator of decimal places for real numbers

For example, in the following table ``Id`` is an integer-number field, ``NDVI_mean`` is a real-number field and the remaining are text fields.

.. code-block::

   Id;     Name; Alias; NDVI_mean;   Color
    1;    Water;   Wtr;      -0.9;    blue
    2;   Forest;   Fst;      0.87;   green
    3;    Crops;   Crp;      0.42; magenta
    4;  Pasture;   Pst;      0.76;  orange
    5;    Urban;   Urb;      0.24;    grey


.. note:: No need for column alignment

   ``plans`` is *not* sensitive to *spaces* in table ``.txt`` files. Hence, columns in table files can be either beautifully aligned like the above example or compacted like the following:

   .. code-block::

        Id;Name;Alias;NDVI_mean;Color
        1;Water;Wtr;-0.9;blue
        2;Forest;Fst;0.87;green
        3;Crops;Crp;0.42;magenta
        4;Pasture;Pst;0.76;orange
        5;Urban;Urb;0.24;grey


.. _timeseries:

Time series
--------------------------------------------

A **time series** in ``plans`` is a special kind of :ref:`table` file which must have a ``Datetime`` field (generally in the first column).
The ``Datetime`` field is a text field that stores dates in the format ``yyyy-mm-dd HH:MM:SS.SSS`` (year, month, day, hour, minute and seconds).
The other fields generally are real number fields that stores the state of *variables* like precipitation ``Plu`` and temperature ``T``.

Time series files tends to have a large number of rows. The first 10 rows of a time series file looks like this:

.. code-block::

                  Datetime;  Plu;    T
   2020-01-01 00:00:00.000;  0.0; 20.1
   2020-01-02 00:00:00.000;  5.1; 24.3
   2020-01-03 00:00:00.000;  0.0; 25.8
   2020-01-04 00:00:00.000; 12.9; 21.4
   2020-01-05 00:00:00.000;  0.0; 21.5
   2020-01-06 00:00:00.000;  0.0; 23.6
   2020-01-07 00:00:00.000;  8.6; 20.6
   2020-01-08 00:00:00.000;  4.7; 28.3
   2020-01-09 00:00:00.000;  0.0; 27.1

.. note:: Automatic fill of time information

   ``plans`` will automatically fill a constant time information (hours, minute and seconds) if only the date is passed, like in the above example.


.. warning:: Beware of small gaps and voids in time series

    ``plans`` will automatically try to fill or interpolate small gaps and voids in a given time series.
    However, be aware that this may cause a unnoticed impact on the model outputs.
    A best option is to interpolate and fill voids prior to the processing so you can understand what is going on.

    For instance, consider the following time series that has a date gap (missing Jan/3 and Jan/4 dates) and a data void for ``Plu`` in Jan/8:

    .. code-block::
        :emphasize-lines: 3,4,7

                       Datetime;  Plu;    T
        2020-01-01 00:00:00.000;  0.0; 20.1
        2020-01-02 00:00:00.000;  5.1; 24.3
        2020-01-05 00:00:00.000;  0.0; 21.5
        2020-01-06 00:00:00.000;  0.0; 23.6
        2020-01-07 00:00:00.000;  8.6; 20.6
        2020-01-08 00:00:00.000;     ; 28.3
        2020-01-09 00:00:00.000;  0.0; 27.1

    In this case, ``plans`` would interpolate temperature ``T`` and fill with 0 the precipitation ``Plu``:

    .. code-block::
        :emphasize-lines: 3,4,7

                       Datetime;  Plu;    T
        2020-01-01 00:00:00.000;  0.0; 20.1
        2020-01-02 00:00:00.000;  5.1; 24.3
        2020-01-03 00:00:00.000;  0.0; 23.3
        2020-01-04 00:00:00.000;  0.0; 22.4
        2020-01-05 00:00:00.000;  0.0; 21.5
        2020-01-06 00:00:00.000;  0.0; 23.6
        2020-01-07 00:00:00.000;  8.6; 20.6
        2020-01-08 00:00:00.000;  0.0; 28.3
        2020-01-09 00:00:00.000;  0.0; 27.1


.. _attribute:

Attribute table
--------------------------------------------

An **attribute table** is a special kind of :ref:`table` file which must have at least the following fields:

- ``Id``: [integer number] Unique numeric code;
- ``Name``: [text] Unique name;
- ``Alias``: [text] Unique short nickname;
- ``Color``: [text] Color HEX code or name available in ``matplotlib``;

**Extra fields** are also required, depending on each dataset.

.. warning::

    Attention to white spaces and special characters because ``plans`` is **not** designed for handling special
    characters and white spaces in names. If needed, use underline ``_`` instead of white space.

.. warning::

    ``plans`` **is case-sensitive** so be consistent with your own naming conventions (i.e.: ``Name`` is different than ``name``).


.. _raster:

Raster map
============================================

A **raster** map in ``plans`` is a frame of data defined by a matrix of cells storing numbers (integer or real values)
and encoded in way that it can be georreferenced in a given Coordinate Reference System (CRS).
The raster map data structure is composed at least by two files:

- [mandatory] the main **grid file** (``.asc`` extension);
- [optional] the auxiliary **projection file** (``.prj`` extension);

Both files may be readily obtained using GIS desktop applications. The projection file is not mandatory but is quite
useful to open the map in the right place and to check consistency of multiple maps.


The grid file
--------------------------------------------

The grid file must follow this general rules:

- the file must be a plain file with ``.asc`` extension
- the first 6 lines must encode a **heading**, specifying the following metadata:
    - ``ncols``: [integer number] columns of the matrix
    - ``nrows``: [integer number]  rows of the matrix
    - ``xllcorner``: [real number] X (longitude) of the lower left corner in the CRS units (meters or degrees)
    - ``yllcorner``: [real number] Y (longitude) of the lower left corner in the CRS units (meters or degrees)
    - ``cellsize``: [positive real number] cell resolution in the CRS units (meters or degrees)
    - ``NODATA_value``: [real number] encoding cells with no data
- after the first 6 lines, the matrix cells must be arranged using blank spaces for value separation.
- period ``.`` must be the separator of decimal places for real numbers

Raster maps tends to have a large number of rows and columns.
The first 10 rows and columns of a ``.asc`` raster file looks like this:

.. code-block::

    ncols        311
    nrows        375
    xllcorner    347528.8
    yllcorner    6714069.8
    cellsize     30.0
    NODATA_value -1
     297 305 317 331 347 360 370 382 403 414 ...
     298 307 321 336 353 368 381 398 411 422 ...
     298  -1 321 338 356 372 385 400 415 427 ...
     297  -1 319 334 353 370 381 395 410 423 ...
     296 305 316 334 351 366 376 386 398 416 ...
     294 303 316 333 347 358 368 379 394 409 ...
     290 299 312 328 342 351 361 375 392 407 ...
     288 297 308 320 333 344 358 375 394 410 ...
     287 297 308 319 329 343 362 382 401 415 ...
     288 297 309 324 336 351 369 391 408 422 ...
     290 297 310 328 343 359 379 399 417 427 ...
     ...

.. note:: Convert files from ``.tif`` to ``.asc`` using GIS and python

    Most GIS desktop applications have tools for converting the commonly distributed ``.tif`` raster files
    to the ``.asc`` format used in ``plans``.

    Hence, you actually only have to worry about setting up the *data type* (integer or real) and
    the *no-data value* in the moment of exporting your ``.tif`` raster files to ``.asc`` format.

    In ``QGIS 3``, you may adapt the following python code for automating the conversion from ``.tif``
    raster files to the ``.asc`` format (the ``.prj`` file is also created):

    .. code-block:: python

        # This code is for QGIS python console
        import processing

        # Set file names
        input_file = 'path/to/input.tif'
        output_file = 'path/to/output.asc'

        '''
        In gdal data types are encoded in the following way:
        1: 8-bit unsigned integer (byte)
        2: 16-bit signed integer
        3: 16-bit unsigned integer
        4: 32-bit signed integer
        5: 32-bit unsigned integer
        6: 32-bit floating-point (real value)
        '''

        # Call gdal:translate
        processing.run("gdal:translate", {
           'INPUT':input_file,  # set input tif raster
           'TARGET_CRS':QgsCoordinateReferenceSystem('EPSG:4326'),  # set CRS EPSG
           'NODATA':-1,  # set no-data value
           'DATA_TYPE':6,  # 32-bit floating-point
           'FORMAT':"AAIGrid",
           'OUTPUT':output_file,  # set input tif raster
        })

    Alternatively, you may use ``rasterio`` python library in other environments, such as in ``colab`` cloud notebooks:

    .. code-block:: python

        # This code assumes rasterio is already installed via pip install
        import rasterio

        # Set file names
        input_file = 'path/to/input.tif'
        output_file = 'path/to/output.asc'

        # Read the input TIF file using rasterio
        with rasterio.open(input_file) as src:
            meta = src.meta.copy()  # Get metadata
            '''
            Rasterio encoded data types as in numpy (some examples):
            uint8: 8-bit unsigned integer (byte)
            int32: 32-bit signed integer
            float32: 32-bit floating-point (real value)
            '''
            # Update the metadata to change the format to ASC
            data_type = 'float32'
            meta.update({'driver': 'AAIGrid', 'dtype': data_type})
            # Open the output ASC file using rasterio
            with rasterio.open(output_file, 'w', **meta) as dst:
                # Copy the input data to the output file
                data = src.read(1) # read only the first band
                dst.write(data.astype(data_type)) # ensure data type


.. _qualiraster:

Qualitative raster map
--------------------------------------------

A **quali-raster** in ``plans`` is a special kind of :ref:`raster` file in which an auxiliary :ref:`attribute`
must be provided alongside the grid and projection files.

.. note:: one attribute table can feed many maps

    The same attribute table file can supply the information required of many raster maps. For instance, consider
    a set of 10 land use and land cover maps, for different years. They all can use the same attribute table file.


********************************************
Conventions
********************************************

Text.

Hydrological variables
============================================




********************************************
Input files reference
********************************************

Text.

Input files summary
============================================

Text.

Input files catalog
============================================

.. include:: input_catalog.rst



********************************************
Output files reference
********************************************

Text.

Output files summary
============================================

Text.

Output files catalog
============================================

Text.

.. DANGER::
   Beware killer rabbits!

.. This table is not displayed live in github
.. csv-table:: Doctable
    :file: /docs/table.csv
    :widths: 33, 33, 33
    :header-rows: 0
