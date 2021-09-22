# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime
import sphinx_rtd_theme  # noqa: F401
sys.path.insert(0, os.path.abspath('..'))
from disstans import __version__ as disstans_version  # noqa: E402


# -- Project information -----------------------------------------------------

project = 'DISSTANS'
author = 'Tobias Köhne'
copyright = f'{datetime.datetime.now().year}, {author}'
version = disstans_version

# The full version, including alpha/beta/rc tags
release = disstans_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest',
              'sphinx.ext.napoleon', 'sphinx.ext.viewcode',
              'sphinx_rtd_theme', 'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax', 'sphinx.ext.autosectionlabel',
              'sphinxcontrib.video']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# complain about broken links, check every once in a while
# nitpicky = True
# nitpick_ignore = [('py:class', 'optional'), ('py:class', 'function'),
#                   ('py:class', 'iterable')]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Other settings ------------------------------------------------

# autodoc settings
# add_module_names = False
autodoc_default_options = {'undoc-members': True,
                           'exclude-members': '__init__, __module__, __dict__, '
                                              '__weakref__, __hash__',
                           'member-order': 'groupwise'}

# intersphinx settings
intersphinx_mapping = {'python': ('https://docs.python.org/3.9/', None),
                       'numpy': ('https://numpy.org/doc/1.20/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy-1.6.3/reference/', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/version/1.2.4/', None),
                       'matplotlib': ('https://matplotlib.org/3.4.2/', None),
                       'scikit-learn': ('https://scikit-learn.org/0.24/', None)}

# ReadTheDocs theme settings
html_theme_options = {'collapse_navigation': False}

# allow the reusing of 'Classes' and 'Functions' section labels by prefixing the document name
autosectionlabel_prefix_document = True

# add copybutton.js (https://github.com/readthedocs/sphinx_rtd_theme/issues/167)
def setup(app):  # noqa: E302
    app.add_js_file('copybutton.js')
