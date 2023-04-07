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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'PyGNOME'
copyright = 'Public Domain'
author = 'NOAA Emergency Response Division'

# reading version from the gnome.__init__ without importing
with open('../../gnome/__init__.py') as init_file:
    for line in init_file:
        parts = line.strip().split()
        try:
            if parts[1] == '=' and parts[0] == "__version__":
                release = parts[2].strip("'")
                break
        except:
            pass
    else:
        raise ValueError("Could not extract version from the gnome.__init__")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [#'nbsphinx',
#              'sphinx.ext.autodoc',
              #'sphinx.ext.napoleon',
              'sphinx.ext.todo',
              #'sphinx.ext.coverage',
              #'sphinx.ext.mathjax',
              # 'sphinx.ext.viewcode',
              ]

# to make autodoc include __init__:
# autoclass_content = 'both'
# autodoc_member_order = 'bysource'

# autoapi options -- see docs here:
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#event-autoapi-skip-member

extensions.append('autoapi.extension')

autoapi_type = 'python'
autoapi_dirs = ['../../gnome/']
autoapi_python_class_content = 'both'
autoapi_keep_files = True


def skip_schema_classes(app, what, name, obj, skip, options):
    """
    Have auto-api skip all the Schema classes
    """
    if what == "class" and "Schema" in name:
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_schema_classes)

# This still is generating errors in the gnome.persist modules
# Though it doesn't write the html ?!?
# def skip_schema_docs(app, what, name, obj, skip, options):
#     # this was reaching in a documenting a bunch of colander stuff
#     if (what in {"module", "package"}) and "gnome.persist" in name:
#         skip = True
#     return skip

# def setup(sphinx):
#     sphinx.connect("autoapi-skip-member", skip_schema_docs)


# this suppresses warnings about files that can't be processed
suppress_warnings = ["autoapi.python_import_resolution",
                     "autoapi.not_readable"]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/GNOME_logo_225px-wide.png"
html_favicon = "_static/GNOME_favicon_32x32.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False


