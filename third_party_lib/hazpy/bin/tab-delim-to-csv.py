#!/usr/bin/env python
"""Convert a tab-delimited file to CSV format."""
import csv, optparse

def convert_file(src, dst):
    in_ = open(src, "rU")
    out = open(dst, "wb")
    writer = csv.writer(out)
    for lin in in_:
        # .writerow requires a sequence, not an iterator.
        row = [x.strip() for x in lin.split("\t")]
        writer.writerow(row)
    in_.close()
    out.close()

def main():
    parser = optparse.OptionParser(
        usage="%prog [options]  SRC_FILE_TAB_DELIM.txt  DEST_FILE.csv",
        description="Convert a tab-delimited file to CSV format.")
    opts, args = parser.parse_args()
    if len(args) != 2:
        parser.error("wrong number of command-line arguments")
    convert_file(args[0], args[1])

if __name__ == "__main__":  main()
