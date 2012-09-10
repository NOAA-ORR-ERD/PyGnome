#!/usr/bin/env python
"""Update the monthly totals, and optionally delete old records.
"""
import datetime
import optparse

try:
    import dateutil
except ImportError:
    raise ImportError("please install the 'python-dateutil' package")
from dateutil.relativedelta import relativedelta

import sqlalchemy as sa
import sqlalchemy.orm as orm

import hazpy.sitestats.common as common
import hazpy.sitestats.model as model

USAGE = "%prog [options]   SQLALCHEMY_URL"

DESCRIPTION = """\
Update the monthly site statistics and optionally delete old records.  
"""

#### Global variables
conn = None   # SQLAlchemy connection or engine, set in main().
today = datetime.date.today()
parser = optparse.OptionParser(usage=USAGE, description=DESCRIPTION)

#### Command-line options
parser.add_option("--create-all", action="store_true",
    help="Create all tables (overwrites existing tables!)")
parser.add_option("--create-tables", action="store", type="string",
    metavar="TABLE1,TABLE2,...",
    help="Create certain tables (overwrites existing ones!)")
parser.add_option("--list-tables", action="store_true",
    help="List the table names that can be created")
parser.add_option("--monthly", action="store_true",
    help="Calculate monthly totals")
parser.add_option("--year", action="store", type="int",
    metavar="N",
    help="Year to calculate (requires --monthly and --month)")
parser.add_option("--month", action="store", type="int",
    metavar="N",
    help="Month to calculate (requires --monthly and --year)")
parser.add_option("--delete-access", action="store", type="int",
    metavar="N",
    help="Delete access records older than N months")
parser.add_option("--delete-referer", action="store", type="int",
    metavar="N",
    help="Delete referer records older than N months")
common.add_common_options(parser)

#### Command functions
def list_tables(all_tablenames):
    for tn in all_tablenames:
        print tn

def create_all(all_tablenames):
    print "Creating all tables:", " ".join(all_tablenames)
    model.Base.metadata.drop_all(bind=conn, checkfirst=True)
    model.Base.metadata.create_all(bind=conn)

def create_tables(tablenames_option, all_tables):
    tablenames = tablenames_option.split(",")
    for tn in tablenames:
        if tn not in all_tables:
            parser.error("no such table '%s'" % tn)
    print "Creating tables:", " ".join(tablenames)
    for tn in tablenames:
        table = all_tables[tn]
        table.drop(bind=conn, checkfirst=True)
        table.create(bind=conn)

def monthly(year, month):
    if year and month:
        summarize_month(year, month)
        return
    elif year or month:
        parser.error("must specify both --year and --month, or neither")
    # Set ``month`` to first day of current month.
    current_month = datetime.date.today().replace(day=1)
    previous_month = current_month - relativedelta(months=1)
    summarize_month(current_month.year, current_month.month)
    summarize_month(previous_month.year, previous_month.month)

#### Utility functions
def delete_old_records(table, months_to_keep, option_name):
    if months_to_keep < 2:
        parser.error("%s value must be greater than 2" % option_name)
    this_month = datetime.date.today().replace(day=1)
    cutoff = this_month - relativedelta(months=months_to_keep-1)
    print "Deleting %s records older than %s" % (table.name, cutoff)
    delete = table.delete(table.c.timestamp < cutoff)
    conn.execute(delete)

def summarize_month(year, month):
    print "Summarizing year", year, "month", month
    sess = orm.create_session(bind=conn)
    sess.begin()   # Dunno why this is necessary.
    cols = model.Access.__table__.columns
    IN_YEAR = sa.func.year(cols.timestamp) == year
    IN_MONTH = sa.func.month(cols.timestamp) == month
    WHERE = sa.and_(IN_YEAR, IN_MONTH)
    # Get hit count
    sql = sa.select([sa.func.count()], WHERE)
    hits = conn.execute(sql).fetchone()[0]
    # Get IP count
    sql = sa.select([sa.func.count(cols.remote_addr.distinct())], WHERE)
    ips = conn.execute(sql).fetchone()[0]
    # Get session count
    sql = sa.select([sa.func.count(cols.session.distinct())], WHERE)
    sessions = conn.execute(sql).fetchone()[0]
    # Get page views
    static_prefixes = [
        "/favico",
        "/robots",
        "/images",
        "/javasc",
        "/styles",
        "/static",
        ]
    not_static = ~ sa.func.left(cols.url, 7).in_(static_prefixes)
    where = sa.and_(IN_YEAR, IN_MONTH, not_static)
    sql = sa.select([sa.func.count()], where)
    page_views = conn.execute(sql).fetchone()[0]
    # Modify existing monthly record or create new record.
    q = sess.query(model.Monthly).filter_by(year=year, month=month)
    monthly = q.first()
    if not monthly:
        monthly = model.Monthly()
        monthly.year = year
        monthly.month = month
        sess.add(monthly)
    monthly.hits = hits
    monthly.page_views = page_views
    monthly.ips = ips
    monthly.sessions = sessions
    sess.commit()

#### Main routine
def main():
    global conn
    all_tables = dict((x.name, x)
        for x in model.Base.metadata.sorted_tables)
    all_tablenames = sorted(all_tables.iterkeys())
    opts, args = parser.parse_args()
    common.init_logging(opts.sql)
    if opts.list_tables:
        list_tables(all_tablenames)
        return
    if len(args) == 0:
        parser.error("which database?")
    if len(args) > 1:
        parser.error("too many command-line arguments")
    dburl = args[0]
    engine = sa.create_engine(dburl)
    conn = engine.connect()
    if opts.create_all:
        create_all(all_tables)
    if opts.create_tables:
        create_tables(opts.create_tables, all_tables)
    if opts.monthly:
        monthly(opts.year, opts.month)
    if (opts.year or opts.month) and not opts.monthly:
        parser.error(
            "must pass --monthly when using --year and --month")
    if opts.delete_access:
        delete_old_records(model.Access.__table__, opts.delete_access, 
            "--delete-access")
    if opts.delete_referer:
        delete_old_records(model.Referer.__table__, opts.delete_referer,
            "--delete-referer")
        

if __name__ == "__main__":
    main()
