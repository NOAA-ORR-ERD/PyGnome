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

### Linux (Tested in 32-bit, raring 13.04) ###

- PyGnome uses NetCDF and also currently uses PIL. 
- WebGnome requires libxml and npm.

For Linux use appropriate package manager (apt-get on ubuntu) to download/install binary dependencies.

#### Binary Dependencies ####

1. Python Imaging Library (PIL) requires:  
    libjpg-dev, libfrreetype6-dev, zlib1g-dev
   
   Use apt-get to build dependencies for PIL  
    `$ sudo apt-get build-dep python-imaging`

   This did not build symlinks to libraries for me in /usr/lib, so had to
   manually create them:  
```
    $ sudo ln -s /usr/lib/`uname -i`-linux-gnu/libfreetype.so /usr/lib/  
    $ sudo ln -s /usr/lib/`uname -i`-linux-gnu/libjpeg.so /usr/lib/  
    $ sudo ln -s /usr/lib/`uname -i`-linux-gnu/libz.so /usr/lib/  
```
    
2. netCDF4 python module requires NetCDF libraries:  
    libhdf5-serial-dev, libnetcdf

3. libxml used by webgnome requires:  
    libxml2-def, libxslt1-dev

4. npm is a javascript package manager used by webgnome  

### Build PyGnome/WebGnome ###

Since this is development work, it might be a good to create and run this in a virtual environment.

[Virtual env](http://www.virtualenv.org/en/latest/)  
[Virtual env wrapper](http://virtualenvwrapper.readthedocs.org/en/latest/) eases management of virtual envs  
A virtualenv is *not required* to run PyGnome or WebGnome.

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
