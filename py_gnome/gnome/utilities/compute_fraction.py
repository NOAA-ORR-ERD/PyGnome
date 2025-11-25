#!/usr/bin/env python

"""
computing fraction of oil in a given droplet size from rosin-rammler
distribution

CDF for Weibull distribution:

1 - exp( -(k/lambda)**k)

"""
from math import exp


def fraction_below_d(d, alpha, lambda_):
    alpha = float(alpha)
    lambda_ - float(lambda_)
    return 1 - exp(-(d/lambda_)**alpha)
