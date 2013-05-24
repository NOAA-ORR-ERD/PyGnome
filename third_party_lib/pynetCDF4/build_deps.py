#!/usr/bin/env python

"""
script to download, build and install the libs needed for
the netcdf4 Python package.

look in the "if __name__" clause at the bottom:
  you can turn on and off different libs, and different parts of the process


NOTE: HDF (at least) doesn't seem to like building i386 and x86_64 at the same time.
       If we want that, we may have to build separately and lipo them together

"""

import os
import shutil

# where you want all the libs installed
# use this if you only want it for the python package
prefix = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'static_libs' )

# use this if you want the full install of hdf and netcdf available on your system
# prefix = "/usr/local"

print "Installing libs to:", prefix

if not os.path.exists(prefix):
    os.mkdir(prefix)

cwd = os.getcwd()


def download(name):
    name = name+'.tar.gz'
    if name.startswith('netcdf-4'):
        cftp = 'curl --output {0} ftp://ftp.unidata.ucar.edu/pub/netcdf/{0} '.format(name)
    elif name.startswith('netCDF4'):
        cftp = "curl --output {0} http://netcdf4-python.googlecode.com/files/{0}".format(name)
    else:
        cftp = 'curl --output {0} ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/{0} '.format(name)
    print "downloading:", name
    print cftp

    os.system(cftp)

def unpack(lib):
    lib = lib +'.tar.gz'
    cmd = "tar -xvf "+lib
    print "Unpacking:", cmd
    os.system( cmd )

def configure(lib):
    if lib.startswith("netcdf-4"):
        # ./configure CFLAGS="-arch i386 -I/Users/chris.barker/HAZMAT/GNOME-dev/GNOME-GIT/third_party_lib/pynetCDF4/static_libs/include -L/Users/chris.barker/HAZMAT/GNOME-dev/GNOME-GIT/third_party_lib/pynetCDF4/static_libs/lib"  --prefix=/Users/chris.barker/HAZMAT/GNOME-dev/GNOME-GIT/third_party_lib/pynetCDF4/static_libs
        conf = './configure CFLAGS="-arch i386 -I{0}/include -L{0}/lib"  --prefix={0}'.format(prefix)
    elif lib.startswith("hdf5"):
        conf = './configure  CFLAGS="-arch i386" --prefix='+prefix

    os.chdir(lib)
    print conf
    if os.system(conf):
        os.chdir(cwd)
        raise Exception("Configuration of %s failed!"%lib)

    os.chdir(cwd)


def build(lib):
    os.chdir(lib)
    print ("building: "+lib)
    
    if os.system('make'):
        raise Exception("building of %s failed"%lib)

    os.chdir(cwd)

def check(lib):
    print ("checking: "+lib)
    

    os.chdir(lib)
    ## is this needed ????
    # elif lib.startswith("hdf5"):
    #     print "hdf requires h5dump to be built to run check..."
    
    #     os.chdir(os.path.join(lib,'tools/h5dump'))
    #     if os.system('make'):
    #         os.chdir(cwd)
    #         raise Exception("building h5dump failed")
    #     os.chdir(cwd)
        
    if os.system('make check'):
        os.chdir(cwd)
        raise Exception("checking of %s failed"%lib)
    os.chdir(cwd)


def install(lib):
    os.chdir(lib)
    print ("installing: "+lib)
    
    if os.system('make install'):
        os.chdir(cwd)
        raise Exception("installing of %s failed"%lib)

    os.chdir(cwd)


def build_setup_dot_py(lib):
    template = open(setup_static.template).read()
    template.replace('%%%put_location_of_static_libs_here%%%', 'r"%s"'%prefix)
    os.chdir(lib)

def build_py_netcdf(lib):
    template = open("setup_static.template").read()
    template = template.replace('%%%put_location_of_static_libs_here%%%', 'r"%s"'%prefix)
    os.chdir(lib)
    open("setup_static.py",'w').write(template)


    print ("building: "+lib)
    if os.system("python setup_static.py build"):
        os.chdir(cwd)
        raise Exception("Building py_netCDF4 failed")
    os.chdir(cwd)

def install_py_netcdf(lib):
    os.chdir(lib)

    print ("installing: "+lib)
    if os.system("python setup_static.py install"):
        os.chdir(cwd)
        raise Exception("Installing py_netCDF4 failed")
    os.chdir(cwd)

def test_py_netcdf(lib):
    os.chdir( os.path.join(lib, "test") )

    print ("testing: "+lib)
    if os.system("python run_all.py"):
        os.chdir(cwd)
        raise Exception("Testing py_netCDF4 failed")
    os.chdir(cwd)


if __name__ == "__main__":

    ## this libs to download and bulid
    libs = [ 
            "hdf5-1.8.10-patch1",
            "netcdf-4.2.1",
            ]
    
    for lib in libs:
        download(lib)
        unpack(lib)
        configure(lib)
        build(lib)
        check(lib)
        install(lib)
        pass

    #download and build python package
    lib = "netCDF4-1.0.4"

    download(lib)
    unpack(lib)
    build_py_netcdf(lib)
    install_py_netcdf(lib)
    test_py_netcdf(lib)




