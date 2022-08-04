#!/usr/bin/env python
'''
Script to run all the testing scripts

python run_all.py
    run all the scripts, except the ones with known issues.

python run_all.py no_skip
    run all the scripts, skipping none

python run_all.py script_one script_two
    run the specified scripts.
'''

import sys
import datetime
from pathlib import Path
import subprocess


def load_known_failures():
    known_fails = []
    with open('known_broken_scripts.txt') as infile:
        for line in infile:
            line = line.strip().split('#')[0]
            if not line:
                continue
            known_fails.append(Path(line))
    return known_fails


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
    to_skip = set(to_skip)

    for p in to_skip:
        print(f"Skipping: {repr(p)}")

    print(f"{len(scripts)} total scripts")
    scripts = scripts.difference(to_skip)
    print(f"{len(scripts)} total scripts")


    for p in scripts:
        print(f"Running: {repr(p)}")

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

    to_skip = [Path(p) for p in sys.argv[1:]]

    if not (to_skip or no_skip):
        to_skip = load_known_failures()
    successes, failures = run_all(to_skip)
    print("Successful scripts:")
    for s in successes:
        print(s)
    print("Scripts with Errors:")
    for s in failures:
        print(s)
    sys.exit(len(failures))



