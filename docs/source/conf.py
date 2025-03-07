# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../sample_code/"))
sys.path.insert(0, os.path.abspath("../../sample_code/subfolder"))
sys.path.insert(0, os.path.abspath("../../sample_code/subfolder/subsubfolder"))

sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms"))
sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms/common"))
sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms/conf"))
sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms/conf/algo"))
sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms/lfd"))
sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms/lfd/algos"))
sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms/lfd/algos/BC"))
sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms/lfd/dataset"))
sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms/lfd/logging"))
sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms/lfd/models"))
sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms/lfd/scripts"))
sys.path.insert(0, os.path.abspath("../../vr_learning_algorithms/lfd/utils"))

project = "MJP_VR"
copyright = "2025, VinRobotics"
author = "VinRobotics"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "myst_parser",
]

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True,
}

autodoc_mock_imports = ["jax", "optax", "wandb", "vr_learning_algorithms"]


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "amsmath",  # Enables math support
    "dollarmath",  # Enables inline math with $...$
    "html_admonition",
    "html_image",
]

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
# html_theme = "default"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
