#!/usr/bin/env python

"""
Some code to test writing large files with the NetCDF4 package

Used to help debug a problem with GNOME2 and writing large output files.

"""
import sys, time
import os
import numpy as np
import netCDF4 as nc

print "imports successful"



def do_test(N,
            K, #number of arrays
            block_size=1024, # "blocks" is the size of data written out in each run through the loop
            chunksizes=(1024, ) # chunksize is the size of chunks in teh netcdf file
            ):

    num_blocks = N / block_size

    print "creating file"
    filename = 'big_test.nc'
    nc_file = nc.Dataset(filename, 'w', format='NETCDF4')

    # create a dimension
    big_dim = nc_file.createDimension('big_dim', None) # unlimited
    #big_dim = nc_file.createDimension('big_dim', N) # pre-set

    print "creating variables"
    # create a variable
    # default chunking
    # big_var = nc_file.createVariable('big_var', np.float64, ('big_dim',))
    # setting a chunksize

    vars = [ nc_file.createVariable('big_var_%i'%i, np.float64, ('big_dim',), chunksizes=chunksizes ) for i in range(K)]

    start = time.time()
    for i in range(num_blocks):
        #print "creating data:", i
        data = np.linspace(0, 2*block_size-2, block_size)

        #print "writing data to file"
        for var in vars:
            var[i*block_size:(i+1)*block_size] = data
            #print "written chunk:", i
        sys.stdout.flush()

    #print "closing file"
    nc_file.close()

    run_time = time.time() - start
    print "It took %s seconds to run"%(run_time)

    file_size = os.path.getsize(filename)

    return run_time, file_size


block_size = 1024
num_arrays = 4
chunk_size = 1024

num_runs = 7

with open('chunking_test_results.csv', 'w') as results_file:
    results_file.write("block_size: %s\n"%block_size)
    results_file.write("number of arrays: %s\n"%num_arrays)
    results_file.write("chunk_size: %s\n"%chunk_size)
    results_file.write("\n")
    results_file.write("     array_length, run time(seconds),   file_size(bytes) \n")
      
    for i in range(num_runs):
        print "writing file for N = 10^%i"%i
        run_time, file_size  = do_test(N = 1024 * 10**i,
                                       K = num_arrays, #number of arrays
                                       block_size = block_size, # "blocks" is the size of data written out in each run through the loop
                                       chunksizes = (chunk_size, ),# chunksize is the size of chunks in teh netcdf file
                                       )
        print "run_time: %s, file_size: %s"%(run_time, file_size)
        results_file.write("%17i, %17f, %17i \n"%(1024 * 10**i, run_time, file_size) )
        results_file.flush() 





