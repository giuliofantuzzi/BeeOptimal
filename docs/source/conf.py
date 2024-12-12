# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../')) 


# def setup(app):
#     app.add_css_file('style.css')
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

def setup(app):
    app.add_js_file("https://cdn.plot.ly/plotly-latest.min.js")

project = 'BeeOptimal documentation'
copyright = '2024, Giulio Fantuzzi'
author = 'Giulio Fantuzzi'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google/NumPy style docstrings
    'sphinx.ext.viewcode',  # Links to source code
    'nbsphinx',             # For Jupyter notebooks
    'nbsphinx_link',        # For Jupyter notebooks
    'sphinx_rtd_dark_mode',  # Dark mode
    'sphinx.ext.mathjax',
]


templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme        = 'sphinx_rtd_theme'
default_dark_mode = False
html_logo         = '_static/logo_BeeOptimal.png'
html_static_path  = ['_static']
html_css_files = [
    'style.css',
]
