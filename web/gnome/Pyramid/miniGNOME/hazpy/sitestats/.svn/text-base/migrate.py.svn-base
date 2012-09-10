"""Migrate the site statistics data from MySQL to PostgreSQL and Hazpy 1.2.

Site-specific impacts:

 * CRT: Do not migrate any CRT records.
 * All other sites: Migrate 'access' and 'referer' within last 90 days. Migrate
   all 'monthly' records except USCG-only ones.
 * CAMEO: Archive search and event results into 'archive' table as version
   '2.0.1'. Do not migrate raw search and event records.
 * Inews: Migrate all event records. Do not migrate search records (too old).
 * Rlink: Migrate all search and 'event' records.
"""

from __future__ import print_function
import argparse
import collections
import datetime
import logging
import time
import urlparse as urlparse
import warnings

import sqlalchemy as sa
import sqlalchemy.exc as sa_exc
import sqlalchemy.orm as orm
from webhelpers.containers import Counter, except_keys
from webhelpers.text import rchop

import hazpy.sitestats.constants as constants
import hazpy.sitestats.model as model
import hazpy.sqlalchemy_util as sa_util
import hazpy.timecounter as timecounter

# Ignore legacy warnings about implicit Unicode conversions in SQLAlchemy.
warnings.filterwarnings("ignore",
    "Unicode type received non-unicode bind param value",
    sa_exc.SAWarning)  # Inserted in front of filters list

def make_cutoff(days_ago):
    cutoff = datetime.date.today() - datetime.timedelta(days=days_ago)
    cutoff = datetime.datetime(cutoff.year, cutoff.month, 1, 0, 0, 0)
    return cutoff

CUTOFF_ACCESS = make_cutoff(90)
THROTTLE = 1000

MONTHS = [None,
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"]

# All parts of the program use the startup time as 'now'.
now = datetime.datetime.now()

# Global metadata and session; set by init_databases().
src_md = sa.MetaData()
dst_Session = orm.sessionmaker()

# Global time counter.
stopwatch = timecounter.TimeCounter(print)

# Choices for the --truncate option: destination tablenames. 
TABLES = ["access", "referer", "search", "asearch", "event", "monthly",
    "archive"]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
sql_log = sa_util.SALogger()


def get_parser():
    description = __doc__.splitlines()[0]
    epilog = """\
Use --create if the destination tables don't exist or you want to rebuild them
from scratch. Use --truncate to rebuild specific tables while preserving
others.  CAUTION: it will not check whether the specified tables correspond to
the specified actions, or whether it will insert duplicate records or raise
duplicate-key errors, or whether the specified actions will completely build a
truncated table. Most actions build the same-name table, but to completely
build the 'archive' table you'll need -archive-search, --archive-advanced, and
--archive-events. To completely build the 'monthly' table you'll need
--monthly and --feed."""
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    # Non-action arguments
    paa = parser.add_argument
    paa("src_dburl", help="Source SQLAlchemy database URL")
    paa("dst_dburl", help="Destination SQLAlchemy database URL")
    paa("--sql", action="store_true", 
        help="Log SQL statements (except inserts).")
    paa("--truncate", "-t", action="append", choices=TABLES, 
        help="Truncate the specified destination table. (May be repeated.)")
    paa("--create", "-c", action="store_true",
        help="Drop and recreate all destination tables.")
    # Action arguments
    description = ("If no actions are specified, all will be performed.")
    actions = parser.add_argument_group("Actions", description)
    gaa = actions.add_argument
    gaa("--access", action="store_true", help="Copy recent Access records.")
    gaa("--referer", action="store_true", help="Copy recent Referer records.")
    gaa("--search", action="store_true",
        help="Copy Search records (rlink only).")
    # --asearch: not used
    gaa("--event", action="store_true",
        help="Copy Event records (inews/rlink only).")
    gaa("--monthly", action="store_true",
        help="Copy Monthly records. (See also --feeds and --feedlinks.)")
    gaa("--feeds", "--feed", action="store_true",
        help="Calculate newsfeed usage and update the Monthly table.")
    gaa("--archive-search", action="store_true",
        help="Archive the Cameo name/unna/cas searches.")
    gaa("--archive-advanced", action="store_true",
        help="Archive the Cameo advanced searches.")
    gaa("--archive-event", action="store_true",
        help="Archive the Cameo chemical/mychemicals/unna events.")
    return parser

def init_databases(src_dburl, dst_dburl, is_create, truncate_tables):
    """Initialize the database connections.

    Modifies global variables ``src_md`` and ``model.Base.metadata``.
    """
    # Connect to the source and destination databases.
    src_engine = sa.create_engine(src_dburl, logging_name="src",
        convert_unicode=True)
    dst_engine = sa.create_engine(dst_dburl, logging_name="dst",
        server_side_cursors=True)
    src_conn = src_engine.connect()
    dst_conn = dst_engine.connect()
    # Bind the global metadata for each database.
    src_md.bind = src_conn
    model.Base.metadata.bind = dst_conn
    dst_Session.configure(bind=dst_conn)
    # Reflect the source tables.
    src_md.reflect()
    # Create or truncate tables as specified.
    if is_create:
        print("Dropping all destination tables.")
        model.Base.metadata.drop_all()
        print("Recreating them.")
        model.Base.metadata.create_all()
        stopwatch.checkpoint("Created tables")
    elif truncate_tables:
        for tblname in truncate_tables:
            print("Truncating table '{}'".format(tblname))
            sql = sa.text("TRUNCATE {} RESTART IDENTITY".format(tblname))
            dst_conn.execute(sql)
            stopwatch.checkpoint("Truncated tables")
    return src_conn, dst_conn


def migrate(migrator):
    logname = migrator.get_logname()
    print("{}: begin.".format(logname))
    migrator()
    what = "{}: finish".format(logname)
    stopwatch.checkpoint(what)
    print()


class _Migrator(object):
    """Base class of all migrators.
    
    Subclasses should override the .__call__ method."""

    def __call__(self):
        raise NotImplementedError("subclass responsibility")

    def get_logname(self):
        ret = rchop(self.__class__.__name__, "Migrator")
        ret = ret.lower().replace("_", "-")
        return ret

    ## Convenience methods for subclasses. (Not called by the base class.)
    def get_src_sql(self):
        """Return a SQL SELECT on the source table.

        Return a SELECT on the table named in the 'src_tablename' class
        attribute. If the table contains a 'timestamp' column, it will be
        lablelled 'ts' in the output. The results are sorted by the 'id' field.
        The implementation calls 'self.customize_src_sql' to add
        subclass-specific clauses (usually WHERE clauses).
        """
        tbl = src_md.tables[self.src_tablename]
        cc = tbl.columns
        fields = []
        for col in tbl.columns:
            if col.name in self.skip_src_columns:
                continue
            elif col.name == "timestamp":
                col = col.label("ts")
            fields.append(col)
        sql = sa.select(fields, order_by=tbl.columns.id)
        sql = sql.where(cc.site != u"crt")
        if self.cutoff:
            sql = sql.where(tbl.columns.timestamp >= self.cutoff)
        sql = self.customize_src_sql(sql, tbl.columns)
        return sql

    def customize_src_sql(self, sql, cc):
        """Add clauses to the source SELECT if necessary and return it.

        * sql: a SQLAlchemy Select on all source columns, ordered by ID.
        * cc: the table's columns collection.

        The base implementation returns ``sql`` unchanged.
        """
        return sql


class _InsertMigrator(_Migrator):
    """Base class of INSERT migrators."""

    # Subclasses must define: 
    # src_orm_class, dst_orm_class

    cutoff = None
    skip_src_columns = set()

    def __call__(self):
        src_conn = src_md.bind
        dst_conn = model.Base.metadata.bind
        throttle_msg = "    ... {:,} records"
        unicode_error_msg = (
            "Caught UnicodeDecodeError on id={}, skipping record")
        insert = self.dst_orm_class.__table__.insert()
        trans = dst_conn.begin()
        count = 0
        pending = []
        def insert_pending_records():
            """Pop records from the pending list and insert them."""
            print(throttle_msg.format(count))
            dst_conn.execute(insert, pending)
            del pending[:]
        sql = self.get_src_sql()
        rslt = src_conn.execute(sql)
        with sql_log.disabling():
            for r in rslt:
                try:
                    r = dict(r)
                    # Delete the 'id' column if present to force a new
                    # autoincrement ID.
                    r.pop("id", None)
                    r = self.convert_record(r)
                except UnicodeDecodeError:
                    print(unicode_error_msg.format(r["id"]))
                    continue
                if r is None:
                    continue
                pending.append(r)
                count += 1
                if count % THROTTLE == 0:
                    insert_pending_records()
            if pending:
                insert_pending_records()
        trans.commit()

    def convert_record(self, r):
        """Return a dict record to insert, or None to skip this record.
        
        The base method returns the argument unchanged.
        """
        return r


#### CONCRETE MIGRATOR CLASSES ####
class MonthlyMigrator(_InsertMigrator):
    """Copy all Monthly records.

    Set the new 'feeds' and 'feedlinks' columns to 0. FeedMigrator will update
    them.
    """
    src_tablename = "Monthly"
    dst_orm_class = model.Monthly

    def customize_src_sql(self, sql, cc):
        sql = sql.where(~ cc.uscg_only)
        return sql

    def convert_record(self, r):
        r["feeds"] = 0
        r["feedlinks"] = 0
        r["updated"] = now
        return r


class SearchMigrator(_InsertMigrator):
    """Migrate the search records that won't be archived.

    The 'value' column is the search term, lowercased.
    
    Copy the 'rlink' records back to 2009-07-05 only. Earlier rlink records
    don't tell whether they were successful. Cameo searches will be archived
    in the Archive table. Inews data is too old; its last record is 2009.
    """
    src_tablename = "Search"
    dst_orm_class = model.Search

    def get_src_sql(self):
        """Select all 'rlink' records, counting up the terms."""
        tbl = src_md.tables[self.src_tablename]
        cc = tbl.columns
        success = (cc.count > 0).label("success")
        year = sa.extract("YEAR", cc.timestamp).label("year")
        earliest = sa.func.min(cc.timestamp).label("earliest")
        latest = sa.func.max(cc.timestamp).label("latest")
        value = sa.func.lower(cc.term).label("value")
        count = sa.func.count().label("count")
        group = [cc.site, cc.type, success, year, value]
        fields = group + [count, earliest, latest]
        sql = sa.select(fields, group_by=group, order_by=group)
        sql = sql.where(cc.site == "rlink")
        sql = sql.where(cc.count != None)
        return sql

    def convert_record(self, r):
        success = r.pop("success")
        r["type"] = u"name" if success else u"name-fail"
        return r


class Archive_SearchMigrator(_Migrator):
    """Archive the Cameo 2.0.1 searches and events.
    
    Top 100 name searches; top 10 cas searches; top 10 unna searches.

    The 'value' column is the search term, lowercased.
    """
    def __call__(self):
        dst_conn = model.Base.metadata.bind
        trans = dst_conn.begin()
        self.migrate(u"name", 100)
        self.migrate(u"cas", 25)
        self.migrate(u"unna", 25)
        trans.commit()

    def migrate(self, type_, number):
        msg_fmt = "archive-search: fetching top {} CAMEO {} {} search terms"
        cc = src_md.tables["Search"].columns
        count = sa.func.count().label("count")
        value = sa.func.lower(cc.term).label("value")
        earliest = sa.func.min(cc.timestamp).label("earliest")
        latest = sa.func.max(cc.timestamp).label("latest")
        fields = [count, value, earliest, latest]
        base_sql = sa.select(fields, group_by=[value], 
            order_by=[count.desc(), value], limit=number)
        base_sql = base_sql.where(cc.site == u"cameo")
        base_sql = base_sql.where(cc.type == type_)
        # Successful
        print(msg_fmt.format(number, type_, "successful"))
        new_type = "search-{}".format(type_)
        sql = base_sql.where(cc.count > 0)
        self.select_and_insert(new_type, sql)
        # Unsuccessful
        print(msg_fmt.format(number, type_, "unsuccessful"))
        new_type = "search-{}-fail".format(type_)
        sql = base_sql.where(cc.count == 0)
        self.select_and_insert(new_type, sql)

    def select_and_insert(self, new_type, sql):
        src_conn = src_md.bind
        dst_conn = model.Base.metadata.bind
        insert = model.Archive.__table__.insert()
        records = []
        for r in src_conn.execute(sql):
            dic = {
                "site": u"cameo",
                "version": u"2.0.1",
                "type": new_type,
                "count": r["count"],
                "value": r["value"],
                "earliest": r["earliest"],
                "latest": r["latest"],
                }
            records.append(dic)
        with sql_log.disabling():
            dst_conn.execute(insert, records)


class Archive_AdvancedMigrator(_Migrator):
    """Archive the Cameo 2.0.1 advanced search.
    
    Top 10 advanced search fields.

    The 'value' column is a search field name.
    """
    def __call__(self):
        src_conn = src_md.bind
        dst_conn = model.Base.metadata.bind
        insert = model.Archive.__table__.insert()
        trans = dst_conn.begin()
        cc = src_md.tables["Search"].columns
        # Can't use SQL COUNT(*) because value may contain multiple fields.
        fields = [cc.term.label("value")]
        sql = sa.select([cc.term])
        sql = sql.where(cc.site == "cameo")
        sql = sql.where(cc.type == "advanced")
        # Including both successful and unsuccessful searches.
        msg_fmt = "archive-advanced: fetching CAMEO advanced search fields"
        counter = Counter()
        for r in src_conn.execute(sql):
            for term in r[0].split():
                counter(term)
        popular_fields = counter.get_popular(10)
        records = []
        for count, value in popular_fields:
            dic = {
                "site": "cameo", 
                "version": "2.0.1", 
                "type": "advanced",
                "count": count,
                "value": value,
                "earliest": None,
                "latest": None,
                }
            records.append(dic)
        if records:
            with sql_log.disabling():
                dst_conn.execute(insert, records)
        trans.commit()


class RefererMigrator(_InsertMigrator):
    src_tablename = "Referer"
    dst_orm_class = model.Referer
    cutoff = CUTOFF_ACCESS


class EventMigrator(_InsertMigrator):
    """Migrate the event records that won't be archived.

    Copy the non-cameo records (inews, rlink). The values for these
    sites all normalized.

    The 'value' column is the record ID or section name of the event.
    """
    src_tablename = "Event"
    dst_orm_class = model.Event

    def get_src_sql(self):
        """Select all non-cameo 'event' records, counting the occurrences."""
        tbl = src_md.tables[self.src_tablename]
        cc = tbl.columns
        year = sa.extract("YEAR", cc.timestamp).label("year")
        earliest = sa.func.min(cc.timestamp).label("earliest")
        latest = sa.func.max(cc.timestamp).label("latest")
        value = sa.func.lower(cc.value).label("value")
        count = sa.func.count().label("count")
        group = [cc.site, cc.type, year, value]
        fields = group + [count, earliest, latest]
        sql = sa.select(fields, group_by=group, order_by=group)
        sql = sql.where(cc.site.in_([u"inews", u"rlink"]))
        return sql


class Archive_EventMigrator(_Migrator):
    """Archive the Cameo 2.0.1 events.
    
    Top 25 chemical datasheet views, MyChemicals additions, UN/NA records, 
    and react records. (Don't archive help usage.) Canonicalize the 
    material IDs in the 'value' field.

    The 'value' column is the record ID or section name of the event.
    """
    def __call__(self):
        src_conn = src_md.bind
        dst_conn = model.Base.metadata.bind
        insert = model.Archive.__table__.insert()
        type_param = sa.bindparam("type", sa.types.Unicode)
        trans = dst_conn.begin()
        cc = src_md.tables["Event"].columns
        # Uppercase value because valid values are material keys (with an
        # uppercase prefix) or numbers (which are the same in either case).
        value = sa.func.upper(cc.value).label("value")
        count = sa.func.count().label("count")
        earliest = sa.func.min(cc.timestamp).label("earliest")
        latest = sa.func.max(cc.timestamp).label("latest")
        sql = sa.select([count, value, earliest, latest], group_by=[value], 
            order_by=[count.desc(), value])
        sql = sql.where(cc.site == "cameo")
        sql = sql.where(cc.type == type_param)
        begin_msg = "archive-event: fetching top {} CAMEO {} events"
        end_msg = "archive-event: read {} {} source records"
        # It doesn't seem possible to make a bind parameter for LIMIT.
        def select_and_insert(type_, new_type, number, prefixes):
            """Select the top N events of the type. Convert values to material
            keys if possible, adding 'prefix' if it's a bare number.
            """
            print(begin_msg.format(number, type_))
            sql2 = sql.limit(number)
            rslt = src_conn.execute(sql2, type=type_)
            records = []
            src_count = 0
            for r in rslt:
                src_count += 1
                v = r["value"]
                if v[:2] in prefixes and len(v) > 2:
                    pass
                elif v.isdigit():
                    if prefixes[0] == u"UN":
                        v = "UN{:04}" + v
                    else:
                        v = prefixes[0] + v
                else:
                    continue   # Don't insert this record; it's invalid.
                new_row = {
                    "site": u"cameo",
                    "version": u"2.0.1",
                    "type": new_type,
                    "success": True,
                    "count": r["count"],
                    "value": v,
                    "earliest": r["earliest"],
                    "latest": r["latest"],
                    }
                records.append(new_row)
            with sql_log.disabling():
                dst_conn.execute(insert, records)
            stopwatch.checkpoint(end_msg.format(src_count, type_))
        select_and_insert(u"chemical", u"view-chemical", 25, [u"CH"])
        select_and_insert(u"mychemicals", u"mychemicals", 25, [u"CH", u"RG"])
        select_and_insert(u"unna", u"view-unna", 25, [u"UN"])
        select_and_insert(u"react", u"view-react", 25, [u"RG"])
        trans.commit()
        

class AccessMigrator(_InsertMigrator):
    src_tablename = "Access"
    dst_orm_class = model.Access
    cutoff = CUTOFF_ACCESS

    def convert_record(self, r):
        u = urlparse.urlsplit(r["url"])
        r["url"] = u.path
        r["query"] = u.query
        r["username"] = r.pop("user")
        return r


class FeedMigrator(_Migrator):   # UNUSED: superceded by class below.
    """Calculate newsfeed subscriptions and update the Monthly table.

    Count all distinct subscription IDs in the month. All feed requests 
    without a subscription ID are treated as a single ID.

    Skip months that have no existing Monthly record.
    """
    src_tablename = "Access"
    dst_orm_object = model.Monthly

    def __call__(self):
        print("feeds: fetching newsfeed access records")
        sess = dst_Session()
        # by_month variables: ``{(year, month)`` : int}
        views_by_month, feeds_by_month, links_by_month = self.get_data()
        #print("views_by_month =>", views_by_month)
        #print("feeds_by_month =>", feeds_by_month)
        #print("links_by_month =>", links_by_month)
        q = sess.query(model.Monthly).filter_by(site="inews")
        for mth in q:
            key = mth.year, mth.month
            if key in views_by_month:
                mth.page_views = views_by_month[key]
            if key in feeds_by_month:
                mth.feeds = feeds_by_month[key]
            if key in links_by_month:
                mth.feedlinks = links_by_month[key]
        sess.commit()

    def get_data(self):
        """Calculate inews monthly page views and newsfeed usage.

        Return three dicts, all keyed by ``(year, month)``. The first is the
        number of page views (excluding newsfeeds but including feed
        linkbacks). The second is the number of feed subscriptions (unique feed
        IDs occurring more than once in the month). The third is the number of
        feed linkbacks.
        """
        src_conn = src_md.bind
        cc = src_md.tables["Access"].columns
        year = sa.extract("YEAR", cc.timestamp).label("year")
        month = sa.extract("MONTH", cc.timestamp).label("month")
        count = sa.func.count().label("count")
        count_distinct_urls = sa.func.count(cc.url.distinct()).label("count")
        group = [year, month]
        def make_sql(count_col):
            sql = sa.select([year, month, count_col], cc.site == "inews", 
                group_by=[year, month])
            return sql
        # 1. Page views by month
        sql = make_sql(count)
        sql = sql.where(~ cc.url.like(u"/incidents.%"))
        views_by_month = self.make_month_dict(src_conn.execute(sql))
        stopwatch.checkpoint("feeds: selected page views by month")
        # 2. Feeds by month.
        sql = make_sql(count_distinct_urls)
        sql = sql.where(cc.url.like(u"/incidents.%"))
        feeds_by_month = self.make_month_dict(src_conn.execute(sql))
        stopwatch.checkpoint("feeds: selected feeds by month")
        # 3. Feed links by month.
        sql = make_sql(count)
        sql = sql.where(cc.url.like(u"/incident/%?f=%"))
        links_by_month = self.make_month_dict(src_conn.execute(sql))
        stopwatch.checkpoint("feeds: selected linkbacks by month")
        return views_by_month, feeds_by_month, links_by_month

    def make_month_dict(self, rows):
        """Execute a three-column query result into a dict.

        ``rows`` is an iterable of sequences, normally a SQLAlchemy result.

        The first two columns are assumed to be the year and month, and will
        form the dict key: ``(year, month)``. The third column will be the dict
        value.
        """
        return {(x[0], x[1]) : x[2] for x in rows}
        

class FeedMigrator(_Migrator):
    """Calculate newsfeed subscriptions and update the Monthly table.

    This implementation queries each month separately due to a disk-space
    limitation on the server. (The /tmp partition is not big enough for a huge
    query.)

    Count all distinct subscription IDs in the month. All feed requests 
    without a subscription ID are treated as a single ID.

    Skip months that have no existing Monthly record.
    """
    def __call__(self):
        print("feeds: fetching newsfeed access records")
        print("feeds: selecting 'monthly' records")
        sess = dst_Session()
        cc = src_md.tables["Access"].columns
        q = sess.query(model.Monthly).filter_by(site="inews")
        q = q.order_by(model.Monthly.year, model.Monthly.month)
        for mth in q:
            month_str = "{} {}".format(MONTHS[mth.month], mth.year)
            if mth.year < 2010 or (mth.year == 2010 and mth.month < 4):
                print("feeds: skipping", month_str, "because the original",
                    "Access records are purged")
                continue
            print("feeds: selecting inews totals for", month_str)
            date_range = self.get_date_range(cc.timestamp, mth.year, mth.month)
            stopwatch.checkpoint("feeds: ... counted page views")
            mth.page_views = self.get_views_for_month(date_range)
            stopwatch.checkpoint("feeds: ... counted feeds")
            mth.feeds = self.get_feeds_for_month(date_range)
            stopwatch.checkpoint("feeds: ... counted feedlinks")
            mth.feedlinks = self.get_feedlinks_for_month(date_range)
        sess.commit()

    def get_date_range(self, col, year, month):
        """Return a SQL WHERE on the datetime column for the specified month.
        """
        if month < 12:
            end_year = year
            end_month = month + 1
        else:
            end_year = year + 1
            end_month = 1
        start = datetime.datetime(year, month, 1, 0, 0, 0)
        end = datetime.datetime(end_year, end_month, 1, 0, 0, 0)
        sql = sa.and_(col >= start, col < end)
        return sql

    def get_views_for_month(self, date_range):
        src_conn = src_md.bind
        cc = src_md.tables["Access"].columns
        sql = sa.select([sa.func.count()])
        sql = sql.where(cc.site == u"inews").where(date_range)
        sql = sql.where(~ cc.url.like(u"/incidents.%"))
        return src_conn.execute(sql).scalar()

    def get_feeds_for_month(self, date_range):
        src_conn = src_md.bind
        cc = src_md.tables["Access"].columns
        sql = sa.select([sa.func.count(cc.url.distinct())])
        sql = sql.where(cc.site == u"inews").where(date_range)
        sql = sql.where(cc.url.like(u"/incidents.%"))
        return src_conn.execute(sql).scalar()

    def get_feedlinks_for_month(self, date_range):
        src_conn = src_md.bind
        cc = src_md.tables["Access"].columns
        sql = sa.select([sa.func.count()])
        sql = sql.where(cc.site == u"inews").where(date_range)
        sql = sql.where(cc.url.like(u"/incident/%?f=%"))
        return src_conn.execute(sql).scalar()


#### MAIN ROUTINE ####
def main():
    print("Starting.")
    parser = get_parser()
    args = parser.parse_args()
    if args.create and args.truncate:
        parser.error("can't specify both --create and --truncate")
    sql_log.initialize(args.sql)
    init_databases(args.src_dburl, args.dst_dburl, args.create, args.truncate)
    print()
    all_actions = not any([args.access, args.referer, args.search, args.event,
        args.monthly, args.feeds, 
        args.archive_search, args.archive_advanced, args.archive_event])
    ##
    ## Perform the actions from smallest table to largest table, while
    ## respecting dependency order.
    ##
    # Small actions (less than 2 million records each)
    if args.monthly or all_actions:
        migrate(MonthlyMigrator())
    if args.search or all_actions:
        migrate(SearchMigrator())
    if args.archive_search or all_actions:
        migrate(Archive_SearchMigrator())
    if args.archive_advanced or all_actions:
        migrate(Archive_AdvancedMigrator())
    # Referer actions (12 million records)
    if args.referer or all_actions:
        migrate(RefererMigrator())
    # Event actions (13 million records)
    if args.event or all_actions:
        migrate(EventMigrator())
    if args.archive_event or all_actions:
        migrate(Archive_EventMigrator())
    # Access actions (277 million records)
    if args.access or all_actions:
        migrate(AccessMigrator())
    if args.feeds or all_actions:   # Depends on 'monthly' action.
        migrate(FeedMigrator())
    stopwatch.finish()
    print()
    
if __name__ == "__main__":  main()
