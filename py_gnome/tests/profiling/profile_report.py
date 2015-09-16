#!/usr/bin/env python

"""
simple profile report script
"""

import sys
import pstats


p = pstats.Stats(sys.argv[1])

p.strip_dirs()
p.sort_stats('tottime')
p.print_stats(20)

