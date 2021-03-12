#!/usr/bin/env python
'''
Created on Apr 4, 2013

runs all scripts in here
'''

import sys
import datetime
from pathlib import Path
import subprocess

# import script_runner

# Note: these really should be updated or removed
KNOWN_FAILURES = ['script_tamoc/script_tamoc_old.py',
                  'script_TAP/script_old_TAP.py',
                  'script_TAP/script_new_TAP.py',
                  'script_tamoc/script_arctic_tamoc.py',
                  'script_tamoc/script_gulf_tamoc.py',
                  'script_ice/script_ice.py',
                  ]


# def run_all_with_script_runner(to_skip=[]):
#     scripts = glob.glob(os.path.join(os.path.dirname(__file__),
#                         'script_*/script_*.py'))
#     print(scripts)

#     default_skip = KNOWN_FAILURES

#     for script in to_skip:
#         default_skip = [s for s in default_skip if script not in s]
#     to_skip.extend(default_skip)
#     for script in to_skip:
#         scripts = [s for s in scripts if script not in s]
#     print(scripts)

#     for script in scripts:
#         print('Begin processing script: {0}'.format(script))

#         # clean directories first
#         # script_runner.clean(os.path.dirname(script))
#         # print "\n cleaned script directory: {0}\n".format(os.path.dirname(script))
#         # run script and do post_run if it exists

#         image_dir = os.path.join(os.path.dirname(script), 'images')

#         (model, imp_script) = script_runner.load_model(script, image_dir)
#         script_runner.run(model)

#         print('completed model run for: {0}'.format(script))

#         if hasattr(imp_script, 'post_run'):
#             imp_script.post_run(model)

#             print('completed post model run for: {0}'.format(script))

#         # save model _state

#         saveloc = os.path.join(os.path.dirname(script), 'save_model')

#         # todo: maybe separate this since it is currently not working
#         ## make sure it is rewound so we're not doing a MIDRUN save
#         ## which does not currently work
#         #model.rewind()

#         #script_runner.save(model, saveloc)

#         #print 'completed saving model _state for: {0}'.format(script)

#         #try:
#         #    script_runner.run_from_save(os.path.join(saveloc, 'Model.json'))
#         #    print ('completed loading and running saved model _state '
#         #           'from: {0}'.format(saveloc))
#         #except Exception, ex:
#         #    print ('Exception in script_runner.run_from_save(saveloc)'
#         #           '\n\t{0}'.format(ex))


def run_all(to_skip=[]):
    """
    Runs all the scripts, each in a subprocess

    Then reports success and failures to stdout, as well as to:

    script_results.txt

    :param to_skip: list of scripts to skip -- useful for known failures

    script returns number of failures as error code, so it can be used
    in a CI test run. (e.g. 0 error code means zero failed scripts)
    """

    scripts = set(Path(__file__).parent.glob('script_*/script_*.py'))

    print("Skipping:")
    for p in to_skip:
        print(f"{p}")

    scripts = scripts.difference(to_skip)

    # fixme: it would be good to keep track of the errors
    successes = []
    failures = []
    # for script in list(scripts)[:1]:
    for script in scripts:
        print("**************************")
        print("*")
        print("*  Running:   %s"%script)
        print("*")
        print("**************************")
        try:
            subprocess.check_call(["python", script], shell=False)
            successes.append(script)
        except subprocess.CalledProcessError:
            failures.append(script)

    with open("script_results.txt", 'w', encoding='utf-8') as outfile:
        outfile.write("PyGNOME Script Runner Report\n")
        outfile.write(f"Produced: {datetime.datetime.now()}\n")
        outfile.write("\nScripts that ran without Errors:\n")
        outfile.write("--------------------------------\n\n")
        outfile.writelines(f"{script}\n" for script in successes)

        outfile.write("\nScripts that Errored out:\n")
        outfile.write("-------------------------\n\n")
        outfile.writelines(f"{script}\n" for script in failures)


    return successes, failures


if __name__ == "__main__":
    print(sys.argv)
    try:
        sys.argv.remove("no_skip")
        no_skip = True
    except ValueError:
        no_skip = False

    to_skip = sys.argv[1:]

    if not (to_skip or no_skip):
        to_skip = KNOWN_FAILURES
    # run_all_with_script_runner(to_skip)
    successes, failures = run_all(to_skip)
    print("Successful scripts:")
    for s in successes:
        print(s)
    print("Scripts with Errors:")
    for s in failures:
        print(s)
    sys.exit(len(failures))



