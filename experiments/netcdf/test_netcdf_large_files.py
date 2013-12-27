#!/usr/bin/env python

"""
Some code to test writing large files with the NetCDF4 package

Used to help debug a problem with GNOME2 and writing large output files.

"""
import sys, time
import numpy as np
import netCDF4 as nc

print "imports successful"


# length of array to test
N = 1024 * 1024 * 16
# "blocks" is the size of data written out in each run through the loop
num_blocks = 16
block_size = N/num_blocks
# create a test file:

print "creating file"
nc_file = nc.Dataset('big_test.nc', 'w', format='NETCDF4')

# create a dimension
big_dim = nc_file.createDimension('big_dim', None) # unlimited
#big_dim = nc_file.createDimension('big_dim', N) # pre-set

print "creating variables"
# create a variable
# default chunking
# big_var = nc_file.createVariable('big_var', np.float64, ('big_dim',))
# setting a chunksize
big_var = nc_file.createVariable('big_var', np.float64, ('big_dim',), chunksizes=(1024, ) )



start = time.time()
for i in range(num_blocks):
    print "creating data:", i
    data = np.linspace(0, 2*block_size-2, block_size)

    print "writing data to file"
    big_var[i*block_size:(i+1)*block_size] = data
    print "written chunk:", i
    sys.stdout.flush()

print "closing file"
nc_file.close()

print "It took %s seconds to run"%(time.time() - start)

