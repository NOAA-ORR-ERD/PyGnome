#!/usr/bin/env python

"""
some testing of reading BNA data

notes: 
it took 10.840414 seconds to read the BNA file the old way
it took 5.563194 seconds to read the BNA file using fromfile for each point
it took 4.507970 seconds to read the BNA file using FileScanN for each point
it took 1.159140 seconds to read the BNA file using FileScanN for each polygon

"""

import numpy as np
import time

from FileScanner import FileScan, FileScanN

bnafilename = "12659polys_350709pts.bna"

def ReadBNA1(filename):
    file = open(filename,'rU')
    Output = []
    line = file.readline()
    while line:
        num_points = int(line.split(',')[2])
        polygon_type = line.split(',')[1].replace('"','')
        
        polygon = np.zeros((num_points,2),np.float)
        for i in range(num_points):
            polygon[i,:] = map(float, file.readline().split(','))
        Output.append((polygon_type, polygon))
        line = file.readline().strip()
    file.close()
    return Output

def ReadBNA2(filename):
    file = open(filename,'rU')
    Output = []
    line = file.readline()
    while line:
        num_points = int(line.split(',')[2])
        polygon_type = line.split(',')[1].replace('"','')
        
        polygon = np.zeros((num_points,2),np.float)
        for i in range(num_points):
            polygon[i,:] = np.fromfile(file, dtype=np.float, count=2, sep=',')
        Output.append((polygon_type, polygon))
        line = file.readline().strip()
    file.close()
    return Output

def ReadBNA3(filename):
    file = open(filename,'rU')
    Output = []
    line = file.readline()
    while line:
        num_points = int(line.split(',')[2])
        polygon_type = line.split(',')[1].replace('"','')
        
        polygon = np.zeros((num_points,2),np.float)
        for i in range(num_points):
            polygon[i,:] = FileScanN(file, 2)
        Output.append((polygon_type, polygon))
        line = file.readline().strip()
    file.close()
    return Output

def ReadBNA4(filename):
    file = open(filename,'rU')
    Output = []
    line = file.readline()
    while line:
        num_points = int(line.split(',')[2])
        polygon_type = line.split(',')[1].replace('"','')
        polygon = FileScanN(file, num_points*2).reshape(num_points, 2)
        Output.append((polygon_type, polygon))
        line = file.readline().strip()
    file.close()
    return Output

def test1():
    start = time.time()
    ReadBNA1(bnafilename)
    print "it took %f seconds to read the BNA file the old way"%(time.time() - start)

def test2():
    start = time.time()
    ReadBNA2(bnafilename)
    print "it took %f seconds to read the BNA file using fromfile for each point"%(time.time() - start)

def test3():
    start = time.time()
    ReadBNA3(bnafilename)
    print "it took %f seconds to read the BNA file using FileScanN for each point"%(time.time() - start)

def test4():
    start = time.time()
    ReadBNA4(bnafilename)
    print "it took %f seconds to read the BNA file using FileScanN for each polygon"%(time.time() - start)

def main():
    test1()
    test2()
    test3()
    test4()
    test1()
    test2()
    test3()
    test4()

if __name__ == "__main__":
    main()
