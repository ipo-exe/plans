.. image:: https://raw.githubusercontent.com/ipo-exe/plans/main/docs/figs/logo.png
    :width: 150 px
    :align: center
    :alt: Logo

--------------------------------------------

Usage
############################################


Quick Overview
********************************************

.. _installation:

Installation
============================================

To run the ``plans`` tool in a virtual or local machine you must clone the latest version of this repository in your system. Also, you must install Python_ and a few and well-known dependencies:

- numpy.
- scipy.
- matplotlib.
- pandas.

The source code of the tool lives in the ``./plans-version/plans`` directory, so it actually works as a Python package. If you are in the ``./plans-version`` level, you can use Python to import the tool and start writing scripts with the modules:

.. code-block:: python

    import plans

This also allows you to easily integrate the tool with other useful Python libraries for pre and post processing needs.


Input datasets
============================================

Inputs (and also outputs) datasets must be simple plain text files such as ``.txt`` for *csv* tables and ``.asc`` for *raster* maps. Therefore, you may use some third-party applications like `Notepad ++`_ and QGIS_ for pre-processing your data to fit your data to the standards of ``plans``.


Typical workflow
============================================

Typically, to run plans you will go into the following steps (with some iteration):

1. Gather observed and scenario datasets for your Area Of Interest (``AOI``).
2. Pre-process the data so it fits into the standards of the tool.
3. Use the tool to create a Project for your AOI
4. Load the datasets to the Project.
5. Use the tool to assess the datasets.
6. Use the tool to calibrate the simulation models.
7. Use the tool to simulate scenarios.
8. Integrate with extra tools and application for more visualization and data analysis.

.. reference definitions

.. _Notepad ++ : https://notepad-plus-plus.org/

.. _QGIS: https://www.qgis.org/en/site/

.. _Python: https://www.python.org/