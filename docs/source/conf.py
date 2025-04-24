# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SpacGPA'
copyright = '2025, Yupu Xu'
author = 'Yupu Xu'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.napoleon",
        "sphinx.ext.autosummary",
        "sphinx.ext.autosectionlabel",
        "myst_parser",
        "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = []

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
autosummary_generate = True
autodoc_typehints = "description"
autodoc_mock_imports = ["torch", "scanpy", "anndata"]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_require_title = False
