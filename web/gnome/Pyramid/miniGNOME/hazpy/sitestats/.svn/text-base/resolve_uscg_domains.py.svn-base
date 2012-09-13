#!/usr/bin/env python
"""Resolve USCG domains.  (Run nightly)
"""
import datetime
import logging
import optparse
import os
import socket

import sqlalchemy as sa
import sqlalchemy.orm as orm
from webhelpers.containers import unique

from hazpy.sqlalchemy_util import count
import hazpy.sitestats.common as common
import hazpy.sitestats.constants as constants
import hazpy.sitestats.model as model

USAGE = "%prog [options]   SQLALCHEMY_URL"

DESCRIPTION = """\
Update the monthly site statistics.  Reads Access table and writes to Monthly
table.  
"""

static_prefixes = [   # URL prefixes that are not page views.
    "/" + x[:6] for x in constants.STATIC_SECTIONS]

#### Global variables
conn = None   # SQLAlchemy connection or engine, set in main().
now = datetime.datetime.now()
parser = optparse.OptionParser(usage=USAGE, description=DESCRIPTION)
log = logging.getLogger()

#### Command-line options
parser.add_option("--debug", action="store_true",
    help="Enable debug logging")
parser.add_option("--legacy", action="store_true",
    help="Read LegacyAccess table too.")
common.add_common_options(parser)

#### Utility functions
def check_table(access):
    domain = model.Domain.__table__
    a = access.columns
    d = domain.columns
    insert = domain.insert()
    new_records = []
    insert_batch_size = 100
    inserted_count = 0
    # Using outerjoin, but exists has the same performance.
    where = sa.and_(
        a.site.in_(constants.USCG_SITES), 
        d.remote_addr == None)
    sql = sa.select([a.remote_addr], where, 
        from_obj=[access.outerjoin(domain)],
        distinct=True)
    # Assemble set of unknown addresses.
    ips = set(x[0] for x in conn.execute(sql) if x)
    total = len(ips)
    n = 0
    for ip in ips:
        n += 1
        log.debug("checking IP %d of %d (%s)", n, total, ip)
        domain = None
        is_uscg = NotSet
        if "," in ip:
            # Multiple IPs in REMOTE_ADDR.  Not sure how to handle this,
            # so set ``is_uscg`` to NULL.  That will be equivalent to false
            # in the reports, but still distinguishable in case we want to
            # retry them later.
            is_uscg = None
        else:
            try:
                domain = socket.gethostbyaddr(ip)[0]
            except socket.herror, e:
                if e[0] == 1:  # Unknown host, assume non USCG.
                    is_uscg = False
                elif e[0] == 2:  # Could not reach nameserver.
                    is_uscg = None  
                else:
                    log_exception(e)
                    continue
            except socket.error, e:  # Base class of socket.herror
                log_exception(e)
                continue
            else:
                is_uscg = domain == "uscg.mil" or domain.endswith(".uscg.mil")
        assert is_uscg is not NotSet
        r = {
            "remote_addr": ip,
            "is_uscg": is_uscg,
            "timestamp": now,
            "domain": domain,
            }
        new_records.append(r)
        if len(new_records) >= insert_batch_size:
            log.debug("inserting %d Domain records", len(new_records))
            conn.execute(insert, new_records)
            inserted_count += len(new_records)
            del new_records[:]
    if new_records:
        log.debug("inserting %d remaining Domain records", len(new_records))
        conn.execute(insert, new_records)
        inserted_count += len(new_records)
    log.debug("inserted %d total Domain records", inserted_count)

            
def log_exception(e):
    log.debug("caught %s: %s", e.__class__.__name__, e)

class NotSet(object):
    pass

#### Main routine
def main():
    global conn
    opts, args = parser.parse_args()
    common.init_logging(opts.log_sql)
    if opts.debug:
        log.setLevel(logging.DEBUG)
    if len(args) != 1:
        parser.error("wrong number of command-line arguments")
    dburl = args[0]
    engine = sa.create_engine(dburl)
    log.debug("Starting")
    conn = engine.connect()
    check_table(model.Access.__table__)
    if opts.legacy:
        log.debug("Checking LegacyAccess table")
        check_table(model.LegacyAccess.__table__)

if __name__ == "__main__":
    main()
