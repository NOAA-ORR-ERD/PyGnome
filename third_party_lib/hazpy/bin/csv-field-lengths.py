#!/usr/bin/env python
"""Display the maximum/minimum length of each column in the specified CSV files.
"""

import argparse
import csv

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    paa = parser.add_argument
    paa("csv_files", action="store", nargs="*",
        help="The CSV files to process")
    paa("--header", action="store_true", 
        help="Read the column names from the first row.")
    paa("--nl", action="store_true", dest="universal_newlines", 
        help="Convert newlines (for files created on another platform).")
    return parser

def get_field_names(reader, f, is_header):
    row = next(reader)
    if is_header:
        header = row
    else:
        header = range(1, len(header)+1)
        header = map(str, header)
        f.seek(0)
    return header

def do_file(filename, mode, is_header):
    print "{}:".format(filename)
    f = open(filename, mode)
    reader = csv.reader(f)
    fieldnames = get_field_names(reader, f, is_header)
    len_columns = len(fieldnames)
    index_range = range(len_columns)
    max_header_length = max(len(x) for x in fieldnames)
    max_lengths = [0] * len_columns
    min_lengths = [999999999999] * len_columns
    for row in reader:
        for i in index_range:
            max_lengths[i] = max(max_lengths[i], len(row[i]))
            min_lengths[i] = min(min_lengths[i], len(row[i]))
    max_header_len = max(len(x) for x in fieldnames)
    fmt = "    {:<20} :  {:>12,} max length (min {})"
    for i in index_range:
        print fmt.format(fieldnames[i], max_lengths[i], min_lengths[i])
    print

def main():
    parser = get_parser()
    args = parser.parse_args()
    if not args.csv_files:
        parser.error("no files specified")
    if args.universal_newlines:
        mode = "rU"
    else:
        mode = "rb"
    print "Showing maximum column lengths in the specified CSV files."
    for filename in args.csv_files:
        do_file(filename, mode, args.header)


if __name__ == "__main__":
    main()
