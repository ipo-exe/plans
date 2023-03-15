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

For scientific and technical information of the ``plans`` project as a whole, see the following literature:

*Possantti, I., Marques G. A modelling framework for nature-based solutions expansion planning considering the benefits to downstream urban water users. Environmental Modelling & Software. Volume 152, 105381, ISSN 1364-8152.* https://doi.org/10.1016/j.envsoft.2022.105381.

*Possantti, I., Barbedo, R., Kronbauer, M., Collischonn, W., Marques, G. A comprehensive strategy for modeling watershed restoration priority areas under epistemic uncertainty: A case study in the Atlantic Forest, Brazil. Journal of Hydrology. Volume 617, Part B, 2023, 129003, ISSN 0022-1694.* https://doi.org/10.1016/j.jhydrol.2022.129003.


Tool overview
*************

The ``plans`` tool is 100% Python and uses few and well-known dependencies:

- numpy.
- scipy.
- matplotlib.
- pandas.

Inputs and outputs datasets are simple plain text files such as ``.txt`` for csv tables and ``.asc`` for raster maps. Therefore, you may use some third-party applications like ``QGIS`` for pre-processing and post-processing.

Documentation website
*********************

If you want more information on how to use the ``plans`` tool, please now move to the `Documentation Website`_ on Read the Docs. Do not navigate documentation pages on Github since some features (like tables and warnings) may not render properly.


.. list-table:: Table Title
   :widths: 25 25 50
   :header-rows: 1

   * - Heading row 1, column 1
     - Heading row 1, column 2
     - Heading row 1, column 3
   * - Row 1, column 1
     -
     - Row 1, column 3
   * - Row 2, column 1
     - Row 2, column 2
     - Row 2, column 3

Example Project usage
---------------------

This project has a standard Sphinx layout which is built by Read the Docs almost the same way that you would build it locally (on your own laptop!).

You can build and view this documentation project locally - we recommend that you activate `a local Python virtual environment first <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment>`_:

.. code-block:: console

    # Install required Python dependencies (Sphinx etc.)
    pip install -r docs/requirements.txt

    # Enter the Sphinx project
    cd docs/
    
    # Run the raw sphinx-build command
    sphinx-build -M html . _build/


You can also build the documentation locally with ``make``:

.. code-block:: console

    # Enter the Sphinx project
    cd docs/
    
    # Build with make
    make html
    
    # Open with your preferred browser, pointing it to the documentation index page
    firefox _build/html/index.html


.. reference definitions

.. _Documentation Website: https://plans-docs.readthedocs.io/en/latest/?badge=latest

.. _surface runoff: https://en.wikipedia.org/wiki/Surface_runoff

.. _infiltration: https://en.wikipedia.org/wiki/Infiltration_(hydrology)

.. _Nature-based solutions for Water: https://www.undp.org/publications/nature-based-solutions-water

.. image definitions

