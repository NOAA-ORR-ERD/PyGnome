#!/usr/bin/env python
"""Output the raw referers on standard output.  (Run nightly)
"""
import cgi
import logging
import optparse
import os
import sys
import urlparse

import sqlalchemy as sa
from unipath import Path

import hazpy.sitestats.common as common
import hazpy.sitestats.constants as constants
import hazpy.sitestats.model as model
from hazpy.sqlalchemy_util import count as count_records

USAGE = "%prog [options]   SITE   SQLALCHEMY_URL  OUTPUT_DIR"

DESCRIPTION = """\
Output a website's the raw referers.
GOOGLE_OUT and OTHER_OUT are two text files to receive the results.
"""

#### Global variables
conn = None   # SQLAlchemy engine or connection, set in main().
parser = optparse.OptionParser(usage=USAGE, description=DESCRIPTION)
log = logging.getLogger()

meta = sa.MetaData()
Referer = sa.Table("Referer", meta,
    sa.Column("id", sa.types.Integer, primary_key=True),
    sa.Column("site", sa.types.String(10)),
    sa.Column("timestamp", sa.types.DateTime),
    sa.Column("scheme", sa.types.Binary(6)),
    sa.Column("domain", sa.types.Binary(64)),
    sa.Column("path", sa.types.Binary(255)),
    sa.Column("query", sa.types.Binary(255)),
    )

#### Command-line options
parser.add_option("--debug", action="store_true",
    help="Enable debug logging")
common.add_common_options(parser)

#### Commands
def output_raw_referers(site, conn, output_dir, is_google):
    ref = Referer.columns
    revdomain = sa.func.reverse(ref.domain).label("revdomain")
    fields = [ref.scheme, ref.domain, ref.path, ref.query]
    order = [revdomain, ref.path, ref.query]
    google_cond = ref.domain.like("%google%")
    where = sa.and_(
        ref.site == site,
        ~ ref.path.like("%translate%"),
        ref.domain.op("regexp")(R".[a-z][a-z]*$"),
        google_cond if is_google else ~google_cond,
        )
    sql = sa.select(fields, where, order_by=order, distinct=True)
    rslt = conn.execute(sql)
    filename = "%s_raw_referers_%s.txt" % (site, 
        "google" if is_google else "other") 
    path = Path(output_dir, filename)
    log.debug("Writing %s", path)
    f = open(path, "w")
    for r in rslt:
        query = "?%s" % r.query if r.query else ""
        lin = "%s://%s%s%s\n" % (r.scheme, r.domain, r.path, query)
        f.write(lin)
    f.close()

def output_search_terms(site, conn, output_dir):
    ref = Referer.columns
    fields = [ref.query]
    in_domains = sa.or_(
        ref.domain.like("%google%"),
        ref.domain.like("%yahoo%"),
        ref.domain.like("%msn%"),
        )
    where = sa.and_(
        ref.site == site,
        ~ ref.path.like("%translate%"),
        in_domains,
        )
    sql = sa.select(fields, where, distinct=True)
    rslt = conn.execute(sql)
    terms = set()
    for r in rslt:
        query = str(r.query)
        params = cgi.parse_qs(query)
        found_terms = params.get("q")   # Google and MSN search term.
        if not found_terms:
            found_terms = params.get("p")   # Yahoo search term.
        if not found_terms:
            continue
        term = found_terms[0]
        tup = (term.lower(), term)
        terms.add(tup)
    terms = sorted(terms)
    path = Path(output_dir, "%s_search_terms.txt" % site)
    log.debug("Writing %s", path)
    f = open(path, "w")
    for term_lower, term in terms:
        lin = "%s\n" % term
        f.write(lin)
    f.close()
    


#### Main routine
def main():
    global conn
    opts, args = parser.parse_args()
    common.init_logging(opts.log_sql)
    if opts.debug:
        log.setLevel(logging.DEBUG)
    if len(args) != 3:
        parser.error("wrong number of command-line arguments")
    site, dburl, output_dir = args
    engine = sa.create_engine(dburl)
    log.debug("Starting")
    conn = engine.connect()
    output_dir = Path(output_dir)
    output_dir.mkdir()
    output_raw_referers(site, conn, output_dir, True)
    output_raw_referers(site, conn, output_dir, False)
    output_search_terms(site, conn, output_dir)

if __name__ == "__main__":
    main()
