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

Input files must be formatted in by standard way, otherwise the tool is not going to work. The standards are meant to be simple and user-friendly for any human and Operating System. This file kinds are described below.

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

Gaps and voids must be filled in the pre-processing phase of dataset preparation with interpolation os statistical techniques. For instance, the following daily time series is not suited for running in ``plans`` because it has a date gap (missing Jan/3 and Jan/4 dates) and a data void for P in Jan/8:

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

Text.


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
