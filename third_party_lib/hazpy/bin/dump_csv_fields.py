#!/usr/bin/env python
"""Dump the first record in a CSV file (presumably the field names).

Opens the file in universal-newline mode regardless of what the Python docs
say.  We have lots of CSV files that must be converted from Macintosh format,
whereas we don't have many files with embedded newlines in fields.
"""
import csv, optparse

DESCRIPTION = __doc__.splitlines()[0]
USAGE = "%prog [options] FILENAME.csv"

def do_file(filename, binary=False):
    mode = binary and "rb" or "rU"
    f = open(filename, mode)
    reader = csv.reader(f)
    try:
        for row in reader.next():
            print row
    except StopIteration:
        print "error: source file is empty"

def main():
    parser = optparse.OptionParser()
    pao = parser.add_option
    pao("--binary", action="store_true", 
        help="open file in binary mode. "
             "(Default is text with universal newline conversion.)")
    opts, args = parser.parse_args()
    if len(args) != 1:
        parser.error("wrong number of command-line arguments")
    do_file(args[0], opts.binary)
    
if __name__ == "__main__":  main()
