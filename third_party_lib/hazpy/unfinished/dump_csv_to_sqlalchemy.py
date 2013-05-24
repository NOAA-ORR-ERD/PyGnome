#!/usr/bin/env python
"""Dump CSV files to literal SQL tables.

Unfinished.
"""

import argparse
import logging
import os

import sqlalchemy as sa

DEFAULT_ENCODING = "utf-8"

class FileInfo(object):
    def __init__(self, path):
        self.path = path
        name = os.path.basename(path)
        dot_pos = name.find(".")
        if dot_pos != -1:
            name = name[:dot_pos]
        self.tablename = name
        self.table = None

def get_parser():
    description = __doc__.splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    paa = parser.add_argument
    paa("dburl", action="store", help="SQLAlchemy database URL (destination)")
    paa("files", action="store", nargs="+", help="Source files (CSV).")
    paa("--encoding", action="store", 
        help="Encoding (default '{}')".format(DEFAULT_ENCODING))
    paa("-f", "--filemaker", "--fm", action="store_true", dest="filemaker",
        help="Do FileMaker/Macintosh conversions.")
    paa("--sql", action="store_true", 
        help="Log SQL statements (except inserts).")
    parser.set_defaults(encoding="utf-8")
    return parser

def get_table_names(files, error)
    names_seen = set()
    duplicate_names = []
    ret = []
    for path in files:
        name = os.path.basename(path)

def main():
    parser = get_parser()
    args = parser.parse_args()
    engine = sa.create_engine(args.dburl)
    conn = engine.connect()
    ensure_unique_files(args.files, parser.error)


if __name__ == "__main__":  main()
