.. badges

|license| |toplang| |docs| |style|

.. |license| image:: https://img.shields.io/github/license/ipo-exe/plans
    :alt: License

.. |toplang| image:: https://img.shields.io/github/languages/top/ipo-exe/plans
    :alt: Top Language

.. |docs| image:: https://readthedocs.org/projects/plans-docs/badge/?version=latest
    :target: https://plans-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code Style

.. image:: https://raw.githubusercontent.com/ipo-exe/plans/main/docs/figs/logo.png
    :width: 150 px
    :align: center
    :alt: Logo

------------

``plans`` - Planning Nature-based Solutions
##################################################

The ``plans`` tool is a software designed to help watershed conservation projects.

It is uses a simulation model for estimating *hydrological processes* such as `surface runoff`_ and infiltration_ for a given area of interest. Model outputs include simulated hydrological processes in time (tables) and space (maps). Those results then can be used by water resource managers for effectively planning the expansion of `Nature-based Solutions for Water`_ (*e.g.*, reforestation and soil-conservation agriculture).

Example of model outputs
*****************************************************************

Mapping infiltration potential:

.. image:: https://raw.githubusercontent.com/ipo-exe/plans/main/docs/figs/htwi_pif_animation.gif
    :width: 600 px
    :align: center
    :alt: Example of use

Mapping riparian wetlands dynamics:

.. image:: https://raw.githubusercontent.com/ipo-exe/plans/main/docs/figs/htwi_rw_animation.gif
    :width: 600 px
    :align: center
    :alt: Example of use 2


Resources needed
*****************************************************************

In order to use the ``plans`` tool properly, you will need access to a digital computer with ``python`` and some basic libraries installed, like ``numpy``, ``pandas``, etc.
Also, you will need to prepare some input data, like climatic time series tables and land use maps.


Documentation website
*****************************************************************

If you want more information on how to use the ``plans`` tool, please now move to the `Documentation Website`_ on Read the Docs. Do not navigate documentation pages on Github since some features (like tables and warnings) may not render properly.

Scientific literature
*****************************************************************

For scientific and technical information about the ``plans`` project as a whole, see the following literature:

    *Possantti, I., Barbedo, R., Kronbauer, M., Collischonn, W., Marques, G. A comprehensive strategy for modeling watershed restoration priority areas under epistemic uncertainty: A case study in the Atlantic Forest, Brazil. Journal of Hydrology. Volume 617, Part B, 2023, 129003, ISSN 0022-1694.* https://doi.org/10.1016/j.jhydrol.2022.129003.

    *Possantti, I., Marques G. A modelling framework for nature-based solutions expansion planning considering the benefits to downstream urban water users. Environmental Modelling & Software. Volume 152, 105381, ISSN 1364-8152.* https://doi.org/10.1016/j.envsoft.2022.105381.


.. reference definitions

.. _Documentation Website: https://plans-docs.readthedocs.io/en/latest/?badge=latest

.. _surface runoff: https://en.wikipedia.org/wiki/Surface_runoff

.. _infiltration: https://en.wikipedia.org/wiki/Infiltration_(hydrology)

.. _Nature-based solutions for Water: https://www.undp.org/publications/nature-based-solutions-water





