#!/usr/bin/env python

"""
computing fraction of oil in a given droplet size from rosin-rammler distribution

CDF for Weibull distribution:

1 - exp( -(k/lambda)**k)

"""

from math import exp, log

def fraction_below_d(d, alpha, lambda_):
    alpha = float(alpha)
    lambda_ - float(lambda_)
    return 1 - exp( -(d/lambda_)**alpha)

if __name__ == "__main__":

    alpha = 1.8
    lambda_=.00456 # API large droplet case

#    alpha = 1.5
#    lambda_= 1 # API large droplet case
    
    median = lambda_ * log(2)**(1/alpha)
    print "median:", median
    print "median fraction:", fraction_below_d(median, alpha, lambda_)


    for d in [0.0001, 0.001, 0.005, 0.01 ]:

        print "d:", d
        print "fraction:", fraction_below_d(d, alpha, lambda_)









