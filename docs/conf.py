# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.folder_main here. If the directory is relative to the
# documentation root, use os.folder_main.abspath to make it absolute, like shown here.
# mask here
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = "plans"
copyright = "2023, Iporã Possantti"
author = "Iporã Possantti"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # mask here
    'sphinx_copybutton'
]


# In your conf.py file
autodoc_mock_imports = [
    'numpy',
    'pandas',
    'scipy',
    'matplotlib',
    'warnings',
    'PIL',
    'processing',
    'qgis',
    'osgeo',
    'geopandas',
    'PyPDF2',
    'rasterio',
]

autodoc_member_order = 'bysource'

# Exclude the __dict__, __weakref__, and __module__ attributes from being documented
exclude_members = ['__dict__', '__weakref__', '__module__', '__str__']

# Configure autodoc options
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': True,
    'special-members': True,
    'show-inheritance': True,
    'exclude-members': ','.join(exclude_members)
}

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Enable numref
numfig = True
