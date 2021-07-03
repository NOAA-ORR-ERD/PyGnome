#!/usr/bin/env python

"""
script to remove future stuff I wish had never been put in there

can be run on a dir full of python files
"""






import sys
from pathlib import Path

lines_to_remove = """
from builtins import range
from builtins import super
from future import standard_library
standard_library.install_aliases()
from builtins import *
from builtins import object
""".split("\n")

lines_to_remove = [line.strip() for line in lines_to_remove if line.strip()]

line_starts = """
from builtins
from future
standard_library.
""".split("\n")

line_starts = [line.strip() for line in line_starts if line.strip()]

print("removing lines that start with:", line_starts)


def to_remove(line):

    for start in line_starts:
        if line.strip().startswith(start):
            return True
    return False


def strip_file(filename):
    with open(str(filename)) as pyfile:
        lines = [line for line in pyfile if not to_remove(line)]

    # print("new file:")
    # for line in lines:
    #     print(line, end='')
    # write it back out
    with open(str(filename), 'w') as pyfile:
        pyfile.writelines(lines)


if __name__ == "__main__":
    dir_to_process = Path(sys.argv[1])

    print("processing dir:", dir_to_process)

    for pyfile in dir_to_process.glob('**/*.py'):
        print("processing file:", pyfile)
        strip_file(pyfile)
