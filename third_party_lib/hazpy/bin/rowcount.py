#!/usr/bin/env python
"""Print the record count of all tables in a SQL database.
"""
import argparse
import logging

import sqlalchemy as sa

logging.basicConfig(level=logging.WARN)

def get_parser():
    description = __doc__[:__doc__.find("\n")]
    parser = argparse.ArgumentParser(description=description)
    paa = parser.add_argument
    paa("dburl", help="SQLAlchemy database URL")
    paa("--sql", action="store_true", help="LOG SQL statements")
    return parser

'''
max_table_len = max(len(x) for x in tables if len(x) < 60)

for table in tables:
    c.execute("SELECT COUNT(*) FROM %s" % table)
    count = c.fetchone()[0]
    print "%-*s: %6d records" % (max_table_len, table, count)
'''

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.sql:
        logging.getLogger("sqlalchemy.engine").setLevel(log.INFO)
    engine = sa.create_engine(args.dburl)
    conn = engine.connect()
    md = sa.MetaData(conn)
    md.reflect()
    tablenames = md.tables.keys()
    tablenames.sort()
    max_name_len = max(map(len, tablenames))
    for tn in tablenames:
        tbl = md.tables[tn]
        sql = sa.select([sa.func.count()]).select_from(tbl)
        count = conn.execute(sql).scalar()
        print "{:{}} {:<}".format(tn, max_name_len, count)

if __name__ == "__main__":  main()
