ABOUT

WebGnome is a web frontend to the py_gnome library. It allows the user to
create a Gnome model, adjust its parameters and generate a sequence of images
containing particle movements, all through an HTML/CSS/JavaScript interface.


INSTALLING

For best results, create a virtual environment for the entire Gnome project.
Activate the environment and install all of WebGnome's dependencies with:

    pip install -r requirements.txt

The project requires SciPy, NumPy and netCDF4, so there are a fair number of
non-Python dependencies like a Fortran compiler and netcdf, which you can obtain
with the package manager of your choice.


RUNNING

Run the development server with the following command:

    pserve development.ini --reload


WEB SERVICES

All of the functionality in WebGnome is powered by a set of web services that
you can call directly while running the application. These are documented --
lightly right now. See DOCUMENTATION for details.


DOCUMENTATION

Project documentation for Python files and the project overview is built with
Sphinx, while documentation for the JavaScript application is built with Docco.

A Fabric command will generate all of the documentation for you:

    fab docs

After generating the docs, open doc/_build/html/index.html for the project
overview. This includes some general descriptions of WebGnome components and
links to auto-generated documentation for Python modules and JavaScript files.
