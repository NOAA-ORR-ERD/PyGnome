#!/usr/bin/env python

'''
Script to run all the example scripts.

Mostly here so that they can be tested in the CI

python run_all.py
    run all the scripts.

NOTE: This is hard-coded with the script names
      It should be updated when  new ones are added, etc.
'''

import sys
import datetime
from pathlib import Path
import subprocess

# scripts to run -- path relative to this dir
scripts_to_run = [
    "gridded_script.py",
    "ice_example.py",
    "simple_script.py",
    "weathering_script.py",
    "polygon_release_script.py",
    "current_uncertainty/current_uncertainty_example.py",
]

HERE = Path(__file__).parent

def run_all(scripts_to_run):
    """
    Runs all the scripts, each in a subprocess

    Then reports success and failures to stdout, as well as to:

    script_results.txt

    script returns number of failures as error code, so it can be used
    in a CI test run. (e.g. 0 error code means zero failed scripts)
    """
    scripts = {p.relative_to(Path.cwd())
               for p in (HERE / script for script in scripts_to_run)}

    print(scripts)

    print(f"{len(scripts)} total scripts")

    for p in scripts:
        print(f"Running: {repr(p)}")

    successes = []
    failures = []

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
    successes, failures = run_all(scripts_to_run)
    print("Successful scripts:")

    for s in successes:
        print(s)

    print("Scripts with Errors:")

    for s in failures:
        print(s)

    sys.exit(len(failures))
