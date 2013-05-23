#!/usr/bin/env python

"""
script to download, build and install the libs

look in the "if __name__" clause at the bottom:
  you can turn on and off different libs, and differenbt parts of the process


NOTE: HDF (at least) doesn't seem to like building i386 and x86_64 at the same time.
       If we want that, we may have to build separately and lipo them together

"""

import os
import shutil

# where you want all the libs installed
# use this if you only want it for teh python package
prefix = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'static_libs' )

# use this if you want the full install available on your system
#prefix = "/usr/local"

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

def config(lib_loc, conf):
    os.chdir(lib_loc)
    print conf
    os.system(conf)
    os.chdir(cwd)

def install(lib_loc, check=True):
    os.chdir(lib_loc)
    print ("installing: "+lib_loc)
    if check:
        os.system('make check install')
    else:
        os.system('make install')
    os.chdir(cwd)


def build(lib):
     os.chdir(lib)
     print ("building "+lib)
     os.system('make')
     os.chdir(cwd)

def inst_zlib(lib, prefix=prefix):

    conf = './configure --prefix=%s'%(prefix)
    config(lib, conf) 

    build(lib)

    install(lib)
    
    #print ("\n\nRemoving {0}\n\n".format(lib))
    #shutil.rmtree(lib)

def build(lib):
    os.chdir(lib)
    print ("building: "+lib)
    
    if os.system('make'):
        raise Exception("building of %s failed"%lib)

    os.chdir(cwd)

def check(lib):
    os.chdir(lib)
    print ("checking: "+lib)
    
    ## is this needed ????
    # elif lib.startswith("hdf5"):
    #     print "hdf requires h5dump to be built to run check..."
    
    #     os.chdir(os.path.join(lib,'tools/h5dump'))
    #     if os.system('make'):
    #         os.chdir(cwd)
    #         raise Exception("building h5dump failed")
    #     os.chdir(cwd)

        
    if os.system('make check'):
        raise Exception("checking of %s failed"%lib)

    os.chdir(cwd)


def install(lib):
    os.chdir(lib)
    print ("installing: "+lib)
    
    if os.system('make install'):
        raise Exception("installing of %s failed"%lib)

    os.chdir(cwd)


def inst_hdf5(lib):

    # stuff I tried:
    #./configure CFLAGS="-arch i386"
    #./configure CFLAGS="-arch i386" --disable-shared

    # first version: conf = './configure --with-zlib={0} --prefix={0}'.format(prefix)
    #conf = './configure  CFLAGS="-arch i386 -arch x86_64" --prefix={0}'.format(prefix)
    conf = './configure  CFLAGS="-arch x86_64" --prefix={0}'.format(prefix)

    config(lib, conf)

    # hdf5 also requires we build h5dump because otherwise some tests fail
    
    print ("make tools/h5dump so we dont get test failures")
    os.chdir(os.path.join(lib,'tools/h5dump'))
    os.system('make')
    os.chdir(cwd)

    install(lib, check=False)
    
    #print ("\n\nRemoving {0}\n\n".format(lib))
    #shutil.rmtree(lib)


def inst_netcdf(lib):

    ## tried:
    # ./configure CFLAGS="-arch i386 -I/Users/chris.barker/local/include" LDFLAGS="-L/Users/chris.barker/local/lib" --disable-shared --prefix=/Users/chris.barker/local

    ## conf = './configure --prefix='+prefix
    conf = './configure CFLAGS="-arch i386 -arch x86_64" --prefix='+prefix

    os.environ['CPPFLAGS']='-I{0}'.format(os.path.join(prefix,'include'))
    os.environ['LDFLAGS']='-L{0}'.format(os.path.join(prefix,'lib'))
    #print os.environ['CPPFLAGS']
    #print os.environ['LDFLAGS']

    config(lib, conf) 
    
    install(lib)

    #print ("\n\nRemoving {0}\n\n".format(lib))
    #shutil.rmtree(lib)

# def download_pynetcdf(name):
#     filename = name+".tar.gz"
#     cmd = "curl --output {0} http://netcdf4-python.googlecode.com/files/{0}".format(filename)
#     print "running: ", cmd
#     os.system(cmd)

#     unpack(name)

def configure(lib):
    if lib.startswith("netcdf-4"):
        conf = './configure CFLAGS="-arch i386" --prefix='+prefix
    elif lib.startswith("hdf5"):
        conf = './configure  CFLAGS="-arch i386" --prefix='+prefix

    #os.environ['CPPFLAGS']='-I{0}'.format(os.path.join(prefix,'include'))
    #os.environ['LDFLAGS']='-L{0}'.format(os.path.join(prefix,'lib'))
    #print os.environ['CPPFLAGS']
    #print os.environ['LDFLAGS']

    os.chdir(lib)
    print conf
    if os.system(conf):
        raise Exception("Configuration of %s failed!"%lib)

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
        raise Exception("Bulding py_netCDF4 failed")
    os.chdir(cwd)


if __name__ == "__main__":

    ## this libs to download and bulid
    libs = [ "hdf5-1.8.10-patch1"]
#    libs = [ "netcdf-4.2.1"]
#    libs = [ "netcdf-4.2.1", "hdf5-1.8.10-patch1"]
    
    for lib in libs:
        #download(lib)
        #unpack(lib)
        #configure(lib)
        #build(lib)
        #check(lib)
        #install(lib)
        pass

    #download and build python package
    lib = "netCDF4-1.0.4"

    #download(lib)
    #unpack(lib)
    build_py_netcdf(lib)



    # ## hdf5 ...
    # hdf5 = "hdf5-1.8.10-patch1"
    # #download(hdf5)

    # inst_hdf5(hdf5)

    # ## netcdf4 
    # netcdf = "netcdf-4.2.1"
    # #download(netcdf, extra=False)
    # #inst_netcdf(netcdf)

    # #pynetcdf = "netCDF4-1.0.4"
    # #download_pynetcdf(pynetcdf)
    # #build_pyNetCDF(pynetcdf)


