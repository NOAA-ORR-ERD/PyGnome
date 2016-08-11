Documentation on GNOME and webGNOME

The Sphinx system is used for the Python-oriented docs, and the C++ API docs.

To build the Sphinx docs , you need the Sphinx system (a python package):

http://sphinx.pocoo.org/

Once installed, you can build the HTML docs with:

sphinx-build -b html ./  HTML_DOCS 

or:

make html

Run in the sphinx_docs directory. The first option will put HTML docs in a HTML_DOCS subfolder within sphinx_docs.
The second option will put them in _build/html


