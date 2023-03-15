I/O Reference
############################################

This is Input and Output reference documentation of ``plans`` tool.

.. toctree::
   iofiles

------------

File formats
********************************************

Files used in ``plans`` are of three kinds:

- Tables
- Time series
- Raster maps

Input files must be formatted in by standard way, otherwise the tool is not going to work. The standards are meant to be simple and user-friendly for any human and Operating System. All kinds of files can be opened and edited by hand in Notepad applications. They are described below.

Tables
============================================

A **table** is a frame of data defined by rows and columns. It must follow this general rules:

- the file must be a plain text file with ``.txt`` extension
- semi-column ``;`` must be the separator of columns
- period ``.`` must be the separator of decimal places for real numbers

Each column represents a **field** that must be *homogeneous*. This means that each field stores the same **data type**, like text, integer numbers or real numbers. The first row stores the names of the fields. The subsequent rows stores the data itself.

For example, in the following table ``Id`` is an integer-number field, ``NDVI_mean`` is a real-number field and the remaining are text fields.

.. code-block::

   Id;     Name; Alias; NDVI_mean;   Color
    1;    Water;   Wtr;      -0.9;    blue
    2;   Forest;   Fst;      0.87;   green
    3;    Crops;   Crp;      0.42; magenta
    4;  Pasture;   Pst;      0.76;  orange
    5;    Urban;   Urb;      0.24;    grey


.. note::

   ``plans`` is *not* sensitive to *spaces* in table files. Hence, files can be either beautifully aligned like the above example or compacted like the following:

   .. code-block::

        Id;Name;Alias;NDVI_mean;Color
        1;Water;Wtr;-0.9;blue
        2;Forest;Fst;0.87;green
        3;Crops;Crp;0.42;magenta
        4;Pasture;Pst;0.76;orange
        5;Urban;Urb;0.24;grey


Time series
============================================

A **time series** is a special kind of table file which must have a ``Date`` field. The ``Date`` field is a text field that stores dates in the format ``yyyy-mm-dd`` (year, month, day). The other fields generally are real number fields that stores the state of *variables* like precipitation ``P`` and temperature ``T``.

Time series files tends to have a large number of rows. The first 10 rows of a time series file looks like this:

.. code-block::

         Date;    P;    T
   2020-01-01;  0.0; 20.1
   2020-01-02;  5.1; 24.3
   2020-01-03;  0.0; 25.8
   2020-01-04; 12.9; 21.4
   2020-01-05;  0.0; 21.5
   2020-01-06;  0.0; 23.6
   2020-01-07;  8.6; 20.6
   2020-01-08;  4.7; 28.3
   2020-01-09;  0.0; 27.1

Daily time series
--------------------------------------------

A **daily time series** is a special time series file that must meet some extra requirements:

- no *date gaps* are allowed. All days from the start to end must be in the sequence of rows
- no *data voids* are allowed. All information must be filled in the variable field.

.. warning::

    In daily time series gaps and voids must be *filled* in the pre-processing phase of dataset preparation with interpolation os statistical techniques.

    For instance, the following daily time series is *not* suited for ``plans`` because it has a date gap (missing Jan/3 and Jan/4 dates) and a data void for ``P`` in Jan/8:

    .. code-block::
        :emphasize-lines: 3,4,7

             Date;    P;    T
        2020-01-01;  0.0; 20.1
        2020-01-02;  5.1; 24.3
        2020-01-05;  0.0; 21.5
        2020-01-06;  0.0; 23.6
        2020-01-07;  8.6; 20.6
        2020-01-08;     ; 28.3
        2020-01-09;  0.0; 27.1

Raster maps
============================================

A **raster map** is a matrix of cells storing numbers (integer or real values) and encoded in way that it can be georreferenced in a given Coordinate Reference System (CRS). It must follow this general rules:

- the file must be a plain text file with ``.asc`` extension
- the first 6 lines must encode a **heading**, specifying the following metadata:
    - ``ncols``: integer number of columns of the matrix
    - ``nrows``: integer number  of rows of the matrix
    - ``xllcorner``: real number of X (longitude) of the lower left corner in the CRS units (meters or degrees)
    - ``yllcorner``: real number of Y (longitude) of the lower left corner in the CRS units (meters or degrees)
    - ``cellsize``: positive real number of the cell resolution in the CRS units (meters or degrees)
    - ``NODATA_value``: real number encoding cells with no data
- after the first 6 lines, the matrix cells must be arranged using blank spaces for value separation.
- period ``.`` must be the separator of decimal places for real numbers

Raster maps tends to have a large number of rows and columns. The first 10 rows and columns of a ``.asc`` raster file looks like this:

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

.. note::

    Most GIS desktop applications have special tools for converting ``.tif`` raster files to the ``.asc`` format used in ``plans``. Hence, you only  have to worry about setting up the data type (integer or real) and the nodata value in the moment of exporting your ``.tif`` raster files.

    While in QGIS 3, you may adapt the following python code for converting ``.tif`` raster files to the ``.asc`` format:

   .. code-block:: python
        # this code is for QGIS python console
        import processing

        # call gdal:translate tool
        processing.run("gdal:translate", {
            'INPUT':"./path/to/input_file.tif", # set input tif raster
            'TARGET_CRS':QgsCoordinateReferenceSystem('EPSG:4326'), # set CRS EPSG
            'NODATA':-1, # set no-data value
            'COPY_SUBDATASETS':False,
            'OPTIONS':'',
            'EXTRA':'',
            'DATA_TYPE':6, # set data type (1: byte, 2:
            'OUTPUT':"./path/to/output_file.asc", # set input tif raster
        })



Glossary
********************************************

Conventions
============================================
Text.


Input files
============================================
Text.


Output files
============================================
Text.


.. DANGER::
   Beware killer rabbits!

.. This table is not displayed live in github
.. csv-table:: Doctable
    :file: /docs/table.csv
    :widths: 33, 33, 33
    :header-rows: 0
