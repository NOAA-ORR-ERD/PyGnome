"""Daily maintenance for the site stats database.
"""
import argparse
import datetime
import logging

from dateutil.relativedelta import relativedelta as rdelta
import sqlalchemy as sa
import sqlalchemy.orm as orm

import hazpy.sitestats.constants as const
from hazpy.sitestats.model import Access, Monthly, Referer

DELETE_CUTOFF_MONTHS = 3
INEWS_FEED_URLS = [u"/incidents.atom", u"/incidents.rss"]

logging.basicConfig()
log = logging.getLogger()

def get_parser():
    description = __doc__.splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    paa = parser.add_argument
    paa("dburl", help="SQLAlchemy database URL.")
    paa("--debug", action="store_true", help="Enable debug logging.")
    paa("--sql", action="store_true", help="Log SQL commands.")
    paa("--delete", action="store_true", 
        help="Enable periodic deletion of old records.")
    return parser

def init_logging(is_debug, is_sql):
    if is_debug:
        log.setLevel(logging.DEBUG)
    if is_sql:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

def update_month(site, date, sess, feed_callback):
    log.debug("Updating site '{}' year {} month {}".format(
        site, date.year, date.month))
    adding = False
    # Make a site & month filter for Access queries.  (``smf``)
    start = date.replace(day=1)
    end = date + rdelta(months=1, day=1)
    smf = sa.and_(Access.site == site, Access.ts >= start, Access.ts < end)
    # Fetch the Monthly record. (``mth``)
    q_mth = sess.query(Monthly)
    q_mth = q_mth.filter_by(site=site, year=date.year, month=date.month)
    mth = q_mth.first()
    # Create the Monthly record if it doesn't exist.
    if not mth:
        mth = Monthly(site=site, year=date.year, month=date.month)
        mth.feeds = 0
        mth.feedlinks = 0
        adding = True
        # Don't add it to the session yet because not all fields are
        # initialized.
    # Set page view count.
    static_prefixes = [u"/{}".format(x[:6]) for x in const.STATIC_SECTIONS]
    non_static_filter = ~ sa.func.substr(Access.url, 1, 7).in_(static_prefixes)
    q = sess.query(sa.func.count()).select_from(Access).filter(smf)
    q = q.filter(non_static_filter)
    q = q.filter(~ Access.url.in_(INEWS_FEED_URLS))
    mth.page_views = q.scalar()
    # Set unique remote IP count.
    q = sess.query(sa.func.count(Access.remote_addr.distinct())).filter(smf)
    mth.ips = q.scalar()
    # Set unique session ID count.
    q = sess.query(sa.func.count(Access.session.distinct())).filter(smf)
    mth.sessions = q.scalar()
    # Set feed count and feed link count.
    if feed_callback:
        feeds, feedlinks = feed_callback(site, smf, sess)
        mth.feeds = feeds
        mth.feedlinks = feedlinks
    mth.updated = sa.func.current_timestamp()
    if adding:
        sess.add(mth)   
    # Not committing or returning the month object.

def get_inews_feeds(site, smf, sess):
    feed_urls = [u"/incidents.atom", u"/incidents.rss"]  # Without query
    # Get "feeds" total.
    q = sess.query(sa.func.count(Access.query.distinct())).filter(smf)
    q = q.filter(Access.url.in_(INEWS_FEED_URLS))
    feeds = q.scalar()
    # Get "feedlinks" total.
    q = sess.query(sa.func.count()).select_from(Access).filter(smf)
    q = q.filter(Access.url.like(u"/incident/%"))
    q = q.filter(Access.query.like(u"f=%"))
    feedlinks = q.scalar()
    return feeds, feedlinks

def purge_old_records(sess):
    # We're doing bulk deletes in unsynchronized mode, so first flush all
    # pending changes and expire all in-memory ORM instances, but don't
    # commit yet. Expiring prevents inconsistencies because the session
    # doesn't know which records are deleted. It doesn't actually matter in
    # this program because the changes are to a different table than the
    # deletions, but it's a good principle for safety.
    sess.flush()
    sess.expire_all()
    cutoff = datetime.date.today() - rdelta(months=DELETE_CUTOFF_MONTHS, day=1)
    sess.query(Access).filter(Access.ts < cutoff).delete(False)
    sess.query(Referer).filter(Referer.ts < cutoff).delete(False)
    # Not committing.

def main():
    parser = get_parser()
    args = parser.parse_args()
    init_logging(args.debug, args.sql)
    engine = sa.create_engine(args.dburl)
    conn = engine.connect()
    sess = orm.Session(bind=conn)
    this_month = datetime.date.today().replace(day=1)
    last_month = this_month - rdelta(months=1)
    update_month(u"cameo", this_month, sess, None)
    update_month(u"cameo", last_month, sess, None)
    update_month(u"inews", this_month, sess, get_inews_feeds)
    update_month(u"inews", last_month, sess, get_inews_feeds)
    update_month(u"rlink", this_month, sess, None)
    update_month(u"rlink", last_month, sess, None)
    if args.delete:
        purge_old_records(sess)
    sess.commit()

if __name__ == "__main__":  main()
