#!/usr/bin/env python

"""
script that loads a save file and outputs the oil budget table
"""

import sys

import gnome.scripting as gs
from gnome.outputters.weathering import WeatheringOutput

try:
    savefile = sys.argv[1]
except IndexError:
    print "You must provide a savefile to load"
    sys.exit(1)


print "Loading:", savefile

model = gs.Model.load_savefile(savefile)
model.outputters = OilBudgetOutput(filename="TestRun1.csv")





# note: this outputs a bunch of JSON -- one for each timestep.
#       maybe a MassBalanceOutputter that ouputs a CSV file would be in order?

