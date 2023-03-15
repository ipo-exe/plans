.. badges

|license| |toplang| |docs|

.. |license| image:: https://img.shields.io/github/license/ipo-exe/plans
    :alt: License

.. |toplang| image:: https://img.shields.io/github/languages/top/ipo-exe/plans
    :alt: Top Language


.. |docs| image:: https://readthedocs.org/projects/plans-docs/badge/?version=latest
    :target: https://plans-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. Logo

.. image:: https://raw.githubusercontent.com/ipo-exe/plans/main/docs/figs/logo.png
    :width: 200 px
    :align: center
    :alt: an image


PLANS - Planning Nature-based Solutions
===============================================

.. DANGER::
   Beware killer rabbits!
   
.. This README.rst should work on Github and is also included in the Sphinx documentation project in docs/ -
therefore, README.rst uses absolute links for most things so it renders properly on GitHub

This example shows a basic Sphinx project with Read the Docs. You're encouraged to view it to get inspiration and copy & paste from the files in the source code. If you are using Read the Docs for the first time, have a look at the official `Read the Docs Tutorial <https://docs.readthedocs.io/en/stable/tutorial/index.html>`__.

.. This table is not displayed live in github
.. csv-table:: Doctable
    :file: /docs/table.csv
    :widths: 33, 33, 33
    :header-rows: 0
    
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


Using the example in your own project
-------------------------------------

If you are new to Read the Docs, you may want to refer to the `Read the Docs User documentation <https://docs.readthedocs.io/>`_.

If you are copying this code in order to get started with your documentation, you need to:

#. place your ``docs/`` folder alongside your Python project. If you are starting a new project, you can adapt the `pyproject.toml` example configuration.
#. use your existing project repository or create a new repository on Github, GitLab, Bitbucket or another host supported by Read the Docs
#. copy ``.readthedocs.yaml`` and the ``docs/`` folder into your project.
#. customize all the files, replacing example contents.
#. add your own Python project, replacing the ``pyproject.toml`` configuration and ``lumache.py`` module.
#. rebuild the documenation locally to see that it works.
#. *finally*, register your project on Read the Docs, see `Importing Your Documentation <https://docs.readthedocs.io/en/stable/intro/import-guide.html>`_.


Read the Docs tutorial
----------------------

To get started with Read the Docs, you may also refer to the `Read the Docs tutorial <https://docs.readthedocs.io/en/stable/tutorial/>`__.
It provides a full walk-through of building an example project similar to the one in this repository.

Cross-Refs
----------------------

.. Place taget to paragraph

.. _my target:
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer eu egestas ipsum. Curabitur aliquam, nulla eget ornare commodo, nisl lectus auctor felis, quis facilisis libero justo ac nisl. Maecenas efficitur arcu sem, vitae vehicula purus posuere vitae. Donec at justo justo. Phasellus eros nisl, malesuada quis convallis eu, gravida vel magna. Mauris varius nunc vel dui fringilla pellentesque. Phasellus eget laoreet ligula. Mauris sed aliquam dui, ac lacinia nisi. 
