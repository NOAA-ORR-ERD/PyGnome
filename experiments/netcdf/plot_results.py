#!/usr/bin/env python

"""
plots the results of teh chunking tests
"""

import numpy as np
import  matplotlib.pyplot as plt

infile = open("chunking_test_results_1.csv", 'r')

header = [infile.readline() for i in range(3)]

#skip to data:
[infile.readline() for i in range(2)]

data = np.loadtxt(infile, delimiter=",")

plt.subplot(2,1,1)
plt.plot(data[:,0]/ 1024, data[:,1])
plt.xlabel('length of arrays (k elements)')
plt.ylabel('run time (s)')

plt.subplot(2,1,2)
plt.plot(data[:,0]/ 1024, data[:,2])
plt.xlabel('length of arrays (k elements)')
plt.ylabel('file size (bytes)')


plt.show()
print data