Documentation on PyGNOME

The Sphinx system is used for the Python-oriented docs.

To build the Sphinx docs , you need the Sphinx system (a python package):

http://sphinx.pocoo.org/

And also a couple other dependencies::

    sphinx
    sphinx_rtd_theme
    nbsphinx
    sphinx-autoapi


Once installed, you can build the HTML docs with::

  make html

Run in this directory.

The built docs will be in the build/html dir.

To edit the docs, edit the files in the source dir. They pretty much match the structure of the rendered docs.

The API documentation itself is automatically build by the "apidoc" system, by extracting all  the docstrings from the gnome package.




