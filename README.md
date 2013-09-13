# GNOME2 #

<img src="http://gnome.orr.noaa.gov/py_gnome_testdata/GnomeIcon128.png" alt="Gnome Logo" title="Gnome" align="right">

The General NOAA Operational Modeling Environment.

GNOME ( General NOAA Operational Modeling Environment ) is a modeling tool
developed by used by the National Oceanic and Atmospheric Administration (NOAA),
Office of Response and Restoration (ORR), Emergency Response Division.


It is designed to support oil and other hazardous material spills in the coastal environment.

This repository contains the source code to the currently under-development version of the model.

**This is code that is under active development -- it should not be considered endorsed or supported by NOAA for any use.**

For a supported version, please see our main web site:  
http://response.restoration.noaa.gov/gnome

## Installation in Development Mode ##

Since this is development work, it might be good to create and run this in a virtual environment.
[Virtual env](http://www.virtualenv.org/en/latest/) and 
[Virtual env wrapper](http://virtualenvwrapper.readthedocs.org/en/latest/) eases management of virtual envs.  

A virtualenv is *not required* to run PyGnome or WebGnome.
Depending on your access level/permissions, you may require a virtualenv if you cannot 
install python packages in the global site-packages. 

### Linux (Tested in 32-bit, raring 13.04) ###

- PyGnome uses NetCDF and also currently uses PIL. 
- WebGnome requires libxml and npm.

For Linux use appropriate package manager (apt-get on ubuntu) to download/install binary dependencies.

#### Binary Dependencies ####

1. Python Imaging Library (PIL) requires:  
    `$ sudo apt-get install libjpeg-dev libfreetype6-dev zlib1g-dev`
   
   Use apt-get to build dependencies for PIL  
    `$ sudo apt-get install build-dep python-imaging`

   This did not build symlinks to libraries for me in /usr/lib, so had to
   manually create them:  
```
    $ sudo ln -s /usr/lib/`uname -i`-linux-gnu/libfreetype.so /usr/lib/  
    $ sudo ln -s /usr/lib/`uname -i`-linux-gnu/libjpeg.so /usr/lib/  
    $ sudo ln -s /usr/lib/`uname -i`-linux-gnu/libz.so /usr/lib/  
```
    
2. netCDF4 python module requires NetCDF libraries: libhdf5-serial-dev, libnetcdf-dev

3. libxml used by webgnome requires: libxml2-dev, libxslt1-dev

4. npm is a javascript package manager used by webgnome

```
    $ sudo apt-get install libhdf5-serial-dev libnetcdf-dev libxml2-dev libxslt1-dev npm
```

5. The following python packages, documented in PyGnome's requirements.txt,
   may need to be manually installed. Binaries for 
   [Numpy](http://packages.ubuntu.com/raring/python/python-numpy) and 
   [Cython](http://packages.ubuntu.com/raring/python/cython) can be installed using apt-get. 
   Current binaries for these are sufficiently new: (Numpy >=1.7.0) and (Cython >= 0.17.1).  

   If you use virtualenv and apt-get to install these system site packages.
   Remember to use the [--system-site-packages](https://pypi.python.org/pypi/virtualenv)
   option when creating a new virtualenv so the system packages are available in the virtualenv.

   Alternatively, pip install should also work. 
   The following builds the latest packages in your virtualenv once the above dependencies are met.
   ```
        $ pip install numpy
        $ pip install cython
        $ pip install netCDF4
        $ pip install PIL
   ```

The remaining dependencies are python packages and can be installed using pip install -r requirements.txt  
See [Build PyGnome/WebGnome](https://github.com/NOAA-ORR-ERD/GNOME2#build-pygnomewebgnome) section below.

### Windows 7 (32-bit, using VS-2008) ###

- PyGnome requires PIL. 
  It statically links against netcdf-3, included with PyGnome source code

Binary dependencies for PyGnome are obtained from
[Chris Gohlke's website](http://www.lfd.uci.edu/~gohlke/pythonlibs/).

#### Binary Dependencies ####

Download and install [Python 2.7.5](http://www.python.org/download/)  
Make sure all binaries are consistent for your Pyton install.
For instance, binaries ending in *win32-py2.7.exe are for Python 2.7.* (32-bit) 
From [Chris Gohlke's website](http://www.lfd.uci.edu/~gohlke/pythonlibs/) download/install 
following packages also listed in GNOME2/py_gnome/requirements.txt. The requirements.txt
also documents the minimum version number:  

1. [setuptools](http://www.lfd.uci.edu/~gohlke/pythonlibs/#setuptools)
2. [pip](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pip)
   - test pip installed correctly - on command line invoke pip command.  
     This should give usage information for 'pip'.  
     pip list will show installed packages.  
     ```
         $ pip  
         Usage:  
           pip &lt;command&gt; [options]  
            
         Commands:  
           install                     Install packages.  
           ...  
           
         $ pip list  
         pip (1.4.1)  
         setuptools (1.1.4)  
     ```
3. [numpy-MKL](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
4. [Cython](http://www.lfd.uci.edu/~gohlke/pythonlibs/#cython)
5. Install [PIL](http://www.pythonware.com/products/pil/) instead of Pillow
6. Desired version of [netCDF4](http://www.lfd.uci.edu/~gohlke/pythonlibs/#netcdf4) 

The remaining dependencies are python packages and can be installed using pip install -r requirements.txt  
See [Build PyGnome/WebGnome](https://github.com/NOAA-ORR-ERD/GNOME2#build-pygnomewebgnome) section below.

### Build PyGnome/WebGnome ###

1. Clone GNOME2 (following is for cloning over https):  
```
    $ git clone https://github.com/NOAA-ORR-ERD/GNOME2.git  
    $ cd GNOME2
```


2. pip install dependencies and build in develop mode first as install mode may still need some testing/work. If this successfully completes, then run unit_tests  
```
    $ cd py_gnome  
    $ pip install -r requirements.txt  
    $ python setup.py develop  
    $ py.test --runslow tests/unit_tests/  
```

3. Once everything passes, install webgnome requirements and build webgnome.  Run tests if build succeeds.  
```
    $ cd ../web/gnome/webgnome  
    $ pip install -r requirements  
    $ python setup.py develop  
    $ nosetests webgnome/tests/*  
```

4. Run development server  
    `$ pserve development.ini --reload`

webgnome will be served up on [http://localhost:6543/](http://localhost:6543/)
