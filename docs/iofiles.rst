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

A **table** is a frame of data defined by rows and columns. General rules:

- the file must be a plain text file with ``.txt`` extension
- semi-column ``;`` is the separator of columns
- period ``.`` is the separator of decimal places for real numbers

Each column represents a **field** that must be *homogeneous*. This means that each field stores the same **data type**, like text, integer numbers or real numbers. The first row stores the names of the fields. The subsequent rows stores the data itself.

For example, in the following table ``Id`` is an integer field, ``NDVI_mean`` is a real number field and the remaining are text fields.

.. code-block::

   Id;     Name; Alias; NDVI_mean;   Color
    1;    Water;     W;      -0.9;    blue
    2;   Forest;     W;      0.87;   green
    3;    Crops;     W;      0.42; magenta
    4;  Pasture;     W;      0.76;  orange
    5;    Urban;     W;      0.24;    grey


Time series
============================================

Text.


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
