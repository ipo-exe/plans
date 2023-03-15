.. badges

|license| |toplang| |docs|

.. |license| image:: https://img.shields.io/github/license/ipo-exe/plans
    :alt: License

.. |toplang| image:: https://img.shields.io/github/languages/top/ipo-exe/plans
    :alt: Top Language

.. |docs| image:: https://readthedocs.org/projects/plans-docs/badge/?version=latest
    :target: https://plans-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://raw.githubusercontent.com/ipo-exe/plans/main/docs/figs/logo.png
    :width: 200 px
    :align: center
    :alt: Logo


``plans`` - Planning Nature-based Solutions
###########################################

The ``plans`` model is a computer tool designed to help watershed restoration projects. It is uses simulation models for mapping *hydrological processes* such as `surface runoff`_ and infiltration_ for a given area of interest. Those simulation maps then can be used by water resource managers for effectively planning the expansion of `Nature-based Solutions for Water`_ (*e.g.*, reforestation and soil-conservation agriculture).

Example of model outputs:

.. image:: https://raw.githubusercontent.com/ipo-exe/plans/main/docs/figs/cat.gif
    :width: 200 px
    :align: center
    :alt: Example of use

For scientific and technical information about the ``plans`` project as a whole, see the following literature:

*Possantti, I., Marques G. A modelling framework for nature-based solutions expansion planning considering the benefits to downstream urban water users. Environmental Modelling & Software. Volume 152, 105381, ISSN 1364-8152.* https://doi.org/10.1016/j.envsoft.2022.105381.

*Possantti, I., Barbedo, R., Kronbauer, M., Collischonn, W., Marques, G. A comprehensive strategy for modeling watershed restoration priority areas under epistemic uncertainty: A case study in the Atlantic Forest, Brazil. Journal of Hydrology. Volume 617, Part B, 2023, 129003, ISSN 0022-1694.* https://doi.org/10.1016/j.jhydrol.2022.129003.


Quick overview
**************

Installation
-------------

To run the ``plans`` tool in a virtual or local machine you must load the latest version of this repository in your system. Also, you must install Python_ and a few and well-known dependencies:

- numpy.
- scipy.
- matplotlib.
- pandas.

The source code of the tool lives in the ``./plans-version/plans`` directory, so it actually works as a Python package. If you are in the ``./plans-version`` level, you can use Python to import the tool and start writing scripts with the modules:

.. code-block:: python

    import plans


Input datasets
--------------

Inputs (and also outputs) datasets must be simple plain text files such as ``.txt`` for *csv* tables and ``.asc`` for *raster* maps. Therefore, you may use some third-party applications like Notepad++_ and QGIS_ for pre-processing your data to fit your data to the standards of ``plans``.

However, ``plans`` allows you to easily integrate the tool with other useful Python libraries for pre and post processing needs.




Typical workflow
----------------




Documentation website
*********************

If you want more information on how to use the ``plans`` tool, please now move to the `Documentation Website`_ on Read the Docs. Do not navigate documentation pages on Github since some features (like tables and warnings) may not render properly.



.. reference definitions

.. _Documentation Website: https://plans-docs.readthedocs.io/en/latest/?badge=latest

.. _surface runoff: https://en.wikipedia.org/wiki/Surface_runoff

.. _infiltration: https://en.wikipedia.org/wiki/Infiltration_(hydrology)

.. _Nature-based solutions for Water: https://www.undp.org/publications/nature-based-solutions-water

.. _Notepad++ : https://notepad-plus-plus.org/

.. _QGIS: https://www.qgis.org/en/site/

.. _Python: https://www.python.org/

.. image definitions

