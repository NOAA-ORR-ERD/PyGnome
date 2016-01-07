#!/usr/bin/env python
'''
Created on Apr 4, 2013

runs all scripts in here
'''

import os, sys, glob
import subprocess

import script_runner


def run_all_with_script_runner():
    scripts = glob.glob(os.path.join(os.path.dirname(__file__),
                        'script_*/script_*.py'))

    for script in scripts:
        print 'Begin processing script: {0}'.format(script)

        # clean directories first
        # script_runner.clean(os.path.dirname(script))
        # print "\n cleaned script directory: {0}\n".format(os.path.dirname(script))
        # run script and do post_run if it exists

        image_dir = os.path.join(os.path.dirname(script), 'images')

        (model, imp_script) = script_runner.load_model(script, image_dir)
        script_runner.run(model)

        print 'completed model run for: {0}'.format(script)

        if hasattr(imp_script, 'post_run'):
            imp_script.post_run(model)

            print 'completed post model run for: {0}'.format(script)

        # save model _state

        saveloc = os.path.join(os.path.dirname(script), 'save_model')

        # todo: maybe separate this since it is currently not working
        ## make sure it is rewound so we're not doing a MIDRUN save
        ## which does not currently work
        #model.rewind()

        #script_runner.save(model, saveloc)

        #print 'completed saving model _state for: {0}'.format(script)

        #try:
        #    script_runner.run_from_save(os.path.join(saveloc, 'Model.json'))
        #    print ('completed loading and running saved model _state '
        #           'from: {0}'.format(saveloc))
        #except Exception, ex:
        #    print ('Exception in script_runner.run_from_save(saveloc)'
        #           '\n\t{0}'.format(ex))

def run_all_alone():
    scripts = glob.glob(os.path.join(os.path.dirname(__file__),
                        'script_*/script_*.py'))

    print scripts

    ## this could (and probably should) be made much smarter
    ##   should it use subprocess??
    for script in scripts:
        print "**************************"
        print "*"
        print "*  Running:   %s"%script
        print "*"
        print "**************************"
        subprocess.check_call(["python", script], shell=False)



if __name__ == "__main__":
    print sys.argv
    if len(sys.argv) > 1:
        if sys.argv[1] == 'alone':
            run_all_alone()
    else:
        run_all_with_script_runner()


