ABOUT

WebGnome is a web frontend to the py_gnome library. It allows the user to
create a Gnome model, adjust its parameters and generate a sequence of images
containing particle movements, all through an HTML/CSS/JavaScript interface.


INSTALLING

For best results, it is suggested that you create and activate a virtual environment
for the entire Gnome project.
[How does one do this?  Link to how-to?]  Then...


First, make your current working directory ".\GNOME\py_gnome", and run:

    python setup.py develop

Second,  make your current working directory ".\GNOME\web\gnome\webgnome"
and install all of WebGNOME's package dependencies with:

    pip install -r requirements.txt

If "pip install -r requirements.txt" results in an error on Windows, use the
command-line error text to identify which package is causing the error. 
Comment-out (with a #) the reference to that package in
".\GNOME\web\gnome\webgnome\requirements.txt" and instead
download/run the installer for that package from  
http://www.lfd.uci.edu/~gohlke/pythonlibs/.  Repeat the process of (1) running
the "pip install -r requirements.txt" command and (2) commenting-out any
packages that cause and error then instead installing that package from
http://www.lfd.uci.edu/~gohlke/pythonlibs/ until no errors result (or until
you get an error that doesn't relate to one of the packages).

Third, make your current working directory ".\GNOME\web\gnome\webgnome"
and run:

    python setup.py develop





The project requires SciPy, NumPy and NetCDF4, so there are a fair number of
non-Python dependencies like a Fortran compiler and NetCDF, which you can obtain
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

The docs include general descriptions of WebGnome components and links to
auto-generated documentation for Python modules and JavaScript files.

Pre-built documentation is included in the repository at the following path:

    doc/_build/html/index.html

You can also rebuild the documentation using an included Fabric command. Since
Docco is a node.js library, you will need node.js, which includes the NPM package
manager. Use the following command to install Docco from the project root
(GNOME/web/gnome/webgnome):

    npm install docco

After you have Sphinx and Docco, you can run the Fabric command:

    fab docs


TESTS

Make sure you have the "nose" package installed (see
https://nose.readthedocs.org/en/latest/ or the nose Windows installer at
http://www.lfd.uci.edu/~gohlke/pythonlibs/).  Open a command window and make
your current working directory ".\GNOME\web\gnome\webgnome\webgnome".
Then run the tests with the following command:

    nosetests