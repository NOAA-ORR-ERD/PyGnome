#!/usr/bin/env python
"""Update the monthly totals, and optionally delete old records.
"""
import datetime
import optparse
import os
import socket

try:
    import dateutil
except ImportError:
    raise ImportError("please install the 'python-dateutil' package")
from dateutil.relativedelta import relativedelta

import sqlalchemy as sa
import sqlalchemy.orm as orm

import hazpy.sitestats.common as common
import hazpy.sitestats.constants as constants
import hazpy.sitestats.model as model

USAGE = "%prog [options]   SITE   SQLALCHEMY_URL"

DESCRIPTION = """\
Update the monthly site statistics.  Reads Access table and writes to Monthly
table.  
"""

static_prefixes = [   # URL prefixes that are not page views.
    "/" + x[:6] for x in constants.STATIC_SECTIONS]

#### Global variables
conn = None   # SQLAlchemy connection or engine, set in main().
today = datetime.date.today()
parser = optparse.OptionParser(usage=USAGE, description=DESCRIPTION)

#### Command-line options
parser.add_option("--year", action="store", type="int",
    metavar="N",
    help="Year to calculate (requires --month)")
parser.add_option("--month", action="store", type="int",
    metavar="N",
    help="Month to calculate (requires --year)")
common.add_common_options(parser)

#### Utility functions
def process_records(records, site, date, uscg_only, sess):
    hits = 0
    page_views = 0
    ips = set()
    sessions = set()
    # Iterate access records and summarize totals.
    for r in records:
        hits += 1
        if r.url[:7] not in static_prefixes:
            page_views += 1
        ips.add(r.remote_addr)
        if r.session:
            sessions.add(r.session)
    # Find or create Monthly record.
    q = sess.query(model.Monthly)
    q = q.filter_by(
        site=site, 
        year=date.year, 
        month=date.month,
        uscg_only=uscg_only)
    monthly = q.first()
    if not monthly:
        monthly = model.Monthly(site, date.year, 
            date.month, uscg_only)
        sess.add(monthly)
    monthly.hits = hits
    # Update Monthly record.
    monthly.page_views = page_views
    monthly.ips = len(ips)
    monthly.sessions = len(sessions)


def summarize_month(site, date):
    want_uscg = site in constants.USCG_SITES
    sess = orm.create_session(bind=conn)
    sess.begin()   # Dunno why this is necessary.
    # Select Access or LegacyAccess records.
    if date >= constants.ACCESS_CUTOFF:
        access = model.Access.__table__
    else:
        access = model.LegacyAccess.__table__
    a = access.columns
    # SQL statement pieces.
    cols = [a.id, a.remote_addr, a.session, a.url]
    in_year = sa.func.year(a.timestamp) == date.year
    in_month = sa.func.month(a.timestamp) == date.month
    in_site = a.site == site
    in_uscg = a.remote_addr.like(constants.USCG_REMOTE_ADDR_LIKE)
    # SQL statement factory.
    def make_select(*where_conditions):
        conditions = [in_year, in_month, a.site == site]
        conditions.extend(where_conditions)
        where = sa.and_(*conditions)
        return sa.select(cols, where)
    # Select all monthly records.
    sql = make_select()
    rslt = conn.execute(sql)
    process_records(rslt, site, date, False, sess)
    # For USCG sites, select only USCG users.
    if site in constants.USCG_SITES:
        sql = make_select(in_uscg)
        rslt = conn.execute(sql)
        process_records(rslt, site, date, True, sess)
    sess.commit()

#### Main routine
def main():
    global conn
    opts, args = parser.parse_args()
    common.init_logging(opts.log_sql)
    if len(args) != 2:
        parser.error("wrong number of command-line arguments")
    site = args[0]
    dburl = args[1]
    engine = sa.create_engine(dburl)
    conn = engine.connect()
    if opts.year and opts.month:
        date = datetime.date(opts.year, opts.month, 1)
        summarize_month(site, date)
    elif opts.year or opts.month:
        parser.error("must specify both --year and --month or neither")
    else:
        current = today.replace(day=1)
        previous = current - relativedelta(months=1)
        summarize_month(site, current)
        summarize_month(site, previous)
        

if __name__ == "__main__":
    main()
