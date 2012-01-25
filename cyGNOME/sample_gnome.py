#!/usr/bin/env python

import cyGNOME
import model


# start up a model:

M = model.Model()
M.startTime = a_datetime_object
M.run_length =  #2 days
M.timestep = 
....

M.setMap(cyGNOME.SimpleMap(bounds = (....) ) )

M.addMover(cyGNOME.Random(bUseDepthDependent = True,
                          fOptimize.isOptimizedForStep = 0
                          fOptimize.isFirstStep = 0
                          fDiffusionCoefficient = 100000
                          fUncertaintyFactor = 2))
# need a Spill class
M.addSpill(A_spill) 

M.runFullSpill()

list_of_pngs = M.drawSpill()

