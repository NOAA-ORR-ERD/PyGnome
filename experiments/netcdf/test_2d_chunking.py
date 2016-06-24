#!/usr/bin/env python

"""
Testing setting chnuking for 2-d arrays

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
nc_file = nc.Dataset('2d_test.nc', 'w', format='NETCDF4')

# create a dimension
big_dim = nc_file.createDimension('big_dim', None) # unlimited
three = nc_file.createDimension('three', 3)


print "creating variables"
# create a variable
# setting a chunksize1
big_var = nc_file.createVariable('big_var', np.float64, ('big_dim', 'three'), chunksizes=(1024*1024, 3) )

start = time.time()
for i in range(num_blocks):
    print "creating data:", i
    data = np.linspace(0, 2*block_size-2, block_size).reshape((-1,1)) * (1,2,3)

    print "writing data to file"
    big_var[i*block_size:(i+1)*block_size,:] = data
    print "written chunk:", i
    sys.stdout.flush()

print "closing file"
nc_file.close()

print "It took %s seconds to run"%(time.time() - start)

