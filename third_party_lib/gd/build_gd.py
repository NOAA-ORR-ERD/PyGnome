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

    ## note: bitbucket does not seem t o support direct download links liek this
    ##       is there a way?
    ##       go here to download by hand: https://bitbucket.org/libgd/gd-libgd/downloads
    cmd = 'curl --output {0} https://bitbucket.org/libgd/gd-libgd/downloads/{0} '.format(name)

    if name.startswith('libpng'):
        print "you need to download this by hand from sourceforge: http://sourceforge.net/projects/libpng/files/libpng16/1.6.2/libpng-1.6.2.tar.gz/download"
        #cmd = 'curl --output {0} http://sourceforge.net/projects/libpng/files/libpng16/1.6.2/libpng-1.6.2.tar.gz/download'.format(name)
        return 1
    print "downloading:", name
    print cmd

    os.system(cmd)

def unpack(lib):
    lib = lib +'.tar.gz'
    cmd = "tar -xvf "+lib
    print "Unpacking:", cmd
    os.system( cmd )

def configure(lib):
    ## need this with the not-yet fixed configure:
    #conf = './configure  --disable-shared --with-png={0} CFLAGS="-arch i386" LDFLAGS="-L{0}/lib" --prefix={0}'.format(prefix)
    ## this should work with the new one
    conf = './configure  --disable-shared --with-png={0} CFLAGS="-arch i386" --prefix={0}'.format(prefix)


    os.chdir(lib)
    print conf
    os.system("make clean")
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


if __name__ == "__main__":

    ## this libs to download and bulid
    libs = [ 
            #"libpng-1.6.2",
            "libgd-2.1.0-rc2",
            ]
    
    for lib in libs:
        #download(lib)
        #unpack(lib)
        configure(lib)
        build(lib)
        #check(lib)
        install(lib)
        pass





