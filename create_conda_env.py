#!/usr/bin/env python

import sys
import subprocess
from pathlib import Path

USAGE = """
create_conda_env.py environment_name [run|test|all]

create a conda environment for the gnome project

saves typing in all the requirements files

default is "all" -- full dev environment

"run" will only give you wnat you need to run the code

"build" will add what's needed to build PYGNOME

"test" will add what you need to run the tests

"docs" will add what's need to build the docs

Example:

./create_conda_env gnome build test

Will create and environment called "gnome" with everything needed to
build, run and test PyGNOME

NOTE: currently hard-coded for Python 3.9
"""

PYTHON="3.9"
if __name__ == "__main__":
    try:
        env_name = sys.argv[1]
    except IndexError:
        print(USAGE)
        sys.exit(1)
    argv = sys.argv[2:]
    if not argv or "all" in argv:
        here = Path(__file__).parent
        reqs = here.glob("conda_requirements*.txt")
        reqs = [str(r) for r in reqs]
    else:
        reqs = ["conda_requirements.txt"]
        if "build" in argv:
            reqs.append("conda_requirements_build.txt")
        if "test" in argv:
            reqs.append("conda_requirements_test.txt")
        if "docs" in argv:
            reqs.append("conda_requirements_docs.txt")

    cmd = ["conda", "create", "-n", env_name, f"python={PYTHON}"]
    for req in reqs:
        cmd.extend(["--file", req])
    print("running\n", cmd)
    subprocess.run(cmd)




