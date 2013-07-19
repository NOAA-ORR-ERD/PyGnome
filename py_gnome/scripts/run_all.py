#!/usr/bin/env python

'''
Created on Apr 4, 2013

runs all scripts in here
'''

import os
import glob

import script_runner

# list of scripts you dont want to run (it's an "in" test, so you don't need the whole file name.)
excludes = [#"boston",
            #"chesapeake_bay",
            #"long_island",
            #"guam",
            #"mariana",
           ]

#===============================================================================
# scripts = ['script_boston/script_boston.py',
#           'script_guam/script_guam.py',
#           'script_chesapeake_bay/script_chesapeake_bay.py',
#           'script_long_island/script_long_island.py',
#           'script_mississippi_river/script_lower_mississippi.py'
#           ]
#===============================================================================
scripts = glob.glob(os.path.join(os.path.dirname(__file__),'script_*/script_*.py'))

for exclude in excludes:
    scripts = [name for name in scripts if exclude not in name]

for script in scripts:
    # clean directories first
    #script_runner.clean(os.path.dirname(script))
    #print "\n cleaned script directory: {0}\n".format(os.path.dirname(script))    
    # run script and do post_run if it exists
    image_dir = os.path.join( os.path.dirname(script),'images')
    
    model,imp_script = script_runner.load_model(script, image_dir)
    script_runner.run(model)

    print "\n completed model run for: {0}\n".format(script)
    
    if hasattr(imp_script, 'post_run'):
        imp_script.post_run(model)
        print "\n completed post model run for: {0}\n".format(script)
    
    # save model state
    saveloc = os.path.join( os.path.dirname(script),'save_model')
    script_runner.save(model, saveloc)
    print "\n completed saving model state for: {0}\n".format(script) 
    
    try:
        script_runner.run_from_save(saveloc)
        print "\n completed loading and running saved model state from: {0}\n".format(saveloc)
    except Exception as ex:
        print ex
