"""Sitestats model.

Also a command-line script to create the tables.
"""

import sqlalchemy as sa
import sqlalchemy.orm as orm
import sqlalchemy.ext.declarative as declarative

Base = declarative.declarative_base()
Session = orm.sessionmaker()   # Basic session class, not bound to any db.

#### TRAFFIC-TRACKING LOGS (access and referer) ####
class Access(Base):
    """Request access log.

    FIELDS
    ------
    * id: autoincrement.
    * site: site key.
    * ts: timestamp.
    * remote_addr: client IP.
    * status:  HTTP status (int).
    * url: URL requested.
    * query: Query string.
    * username: username if logged in; NULL if not logged in or anonymous site.
    * session: session ID; NULL if unknown.

    The XXX maintenance utility extracts aggregate summaries into the Monthly
    table for the current and previous month, and purges Access records after
    61-90 days.
    
    For the Monthly table, the current and past month are recal culated daily
    based on the 'Access.ts' field, which tells which month the record belongs
    to.
    """
    __tablename__ = "access"

    id = sa.Column(sa.types.Integer, primary_key=True)
    site = sa.Column(sa.types.Unicode, nullable=False, index=True)
    ts = sa.Column(sa.types.DateTime, nullable=False, index=True)
    remote_addr = sa.Column(sa.types.Unicode, nullable=False)
    status = sa.Column(sa.types.Integer, nullable=False)
    user = sa.Column(sa.types.Unicode, nullable=True)
    url = sa.Column(sa.types.Unicode, nullable=False)
    query = sa.Column(sa.types.Unicode, nullable=False)
    session = sa.Column(sa.types.Unicode, nullable=True, index=True)


class Referer(Base):
    """

    The XXX maintenance utility purges records after 61-90 days.
    """
    __tablename__ = "referer"

    id = sa.Column(sa.types.Integer, primary_key=True)
    site = sa.Column(sa.types.Unicode, nullable=False, index=True)
    ts = sa.Column(sa.types.DateTime, nullable=False, index=True)
    scheme = sa.Column(sa.types.Unicode, nullable=False)

    domain = sa.Column(sa.types.Unicode, nullable=False)
    path = sa.Column(sa.types.Unicode, nullable=False)
    query = sa.Column(sa.types.Unicode, nullable=False)


#### EVENT-TRACKING LOGS (searches and section views) ####
class Search(Base):
    """Searches that record the search term.

    SITE    TYPE         MEANING                CONTEXT
    ----    ----         -------                -------
    cameo   name         Name search.           simple or advanced
    cameo   cas          CAS search.            simple
    cameo   unna         UNNA search.           simple
    cameo   description  Description search.    advanced
    inews   name         Incident name.         simple or advanced

    'value' is the search term. Convert it to lower case before inserting it.

    When a site rolls out a new version, the XXX utility (not written yet) can
    be used to summarize the existing results in the Archive table and delete
    the original records.

    The columns are the same as the Event table. This table can be considered a
    polymorphic peer with Event.
    """
    __tablename__ = "search"

    id = sa.Column(sa.types.Integer, primary_key=True)
    site = sa.Column(sa.types.Unicode, nullable=False, index=True)
    type = sa.Column(sa.types.Unicode, nullable=False, index=True)
    year = sa.Column(sa.types.Integer, nullable=False, index=True)
    count = sa.Column(sa.types.Integer, nullable=False)
    value = sa.Column(sa.types.Unicode, nullable=False, index=True)
    earliest = sa.Column(sa.types.DateTime, nullable=False)
    latest = sa.Column(sa.types.DateTime, nullable=False)


#### MISCELLANEOUS EVENTS
class Event(Base):
    """Count the visits to site-specific sections or datasheets.

    Caution: the list below is out of date. XXX

    SITE    TYPE         VALUE    MEANING
    ----    ----         -----    -------
    cameo   chemical     material key  View chemical datasheet.
    cameo   unna         material key  View UN/NA datasheet.
    cameo   react        material key  View react group datasheet.
    cameo   mychemicals  material key  Add material to MyChemicals.
    rlink   incident     orr_id        View incident page.
    inews   incident     orr_id        View incident page.

    'value' is the record ID or section name of the event. Leave it in its
    canonical case. (Uppercase, lower, or mixed.)

    When a site rolls out a new version, the XXX utility (not written yet) can
    be used to summarize the existing results in the Archive table and delete
    the original records.

    Columns:

    * **id**: autoincrement.
    * **site**: site key.
    * **type**: site-specific value; the type of list this is.
    * **year**: the year this record pertains to.
    * **count**: the number of occurrences of 'value'. An integer 1 or higher.
    * **value**: the value being counted (a record number, section key,
      search term, etc).  Must be a string.
    * **earliest**: when the first instance of this value occurred. (Usually
      the record creation time.)
    * **latest**: when the most recent instance of this value occurred.
      (Usually the last-modified time.)
    """
    __tablename__ = "event"

    id = sa.Column(sa.types.Integer, nullable=False, primary_key=True)
    site = sa.Column(sa.types.Unicode, index=True)
    type = sa.Column(sa.types.Unicode, nullable=False, index=True)
    year = sa.Column(sa.types.Integer, nullable=False, index=True)
    count = sa.Column(sa.types.Integer, nullable=False)
    value = sa.Column(sa.types.Unicode, nullable=False, index=True)
    earliest = sa.Column(sa.types.DateTime, nullable=False)
    latest = sa.Column(sa.types.DateTime, nullable=False)


#### SUMMARY LOGS (extracted from above tables) ####
class Monthly(Base):
    """Archived monthly usage totals.
    
    Fields:
    -------
    * id: autoincrement.
    * site: site key.
    * year: year.
    * month: month.
    * page_views: number of page views in month. (Not including newsfeed
      subscriptions, but including feed linkbacks.)
    * ips: number of unique remote IPs in month.
    * sessions: number of unique session IDs in month.
    * feeds: number of distinct newsfeed users per month.
    * feedlinks: number of datasheet views coming from links in newsfeeds.
    * updated: when this record was last updated. (This is not called 'ts' 
      to avoid confusion between the semantic date (year/month) and the
      update date.)
    """

    __tablename__ = "monthly"

    id = sa.Column(sa.types.Integer, primary_key=True)
    site = sa.Column(sa.types.Unicode, nullable=False, index=True)
    year = sa.Column(sa.types.Integer, index=True)
    month = sa.Column(sa.types.Integer, index=True)
    page_views = sa.Column(sa.types.Integer, nullable=False)
    ips = sa.Column(sa.types.Integer, nullable=False)
    sessions = sa.Column(sa.types.Integer, nullable=False)
    feeds = sa.Column(sa.types.Integer, nullable=False)
    feedlinks = sa.Column(sa.types.Integer, nullable=False)
    updated = sa.Column(sa.types.DateTime, nullable=False)


class Archive(Base):
    """Archived results for previous site versions.

    The columns are the same as for Event except:

    * **version**: the site version.
    * **type**: same as above, except to distinguish between overlapping
      search types and event types, searches have "search-" prefixed, and
      ambiguous datasheet views have "view-" prefixed.
    * **earliest** and **latest**: nulls are allowed because we're not sure
      if we'll always be able to calculate this. 'earliest' means the first
      occurrence of the value in this version; 'latest' means the most recent
      occurrence.
    """

    __tablename__ = "archive"

    id = sa.Column(sa.types.Integer, primary_key=True)
    site = sa.Column(sa.types.Unicode, nullable=False, index=True)
    version = sa.Column(sa.types.Unicode, nullable=False, index=True)
    type = sa.Column(sa.types.Unicode, nullable=False, index=True)
    count = sa.Column(sa.types.Integer, nullable=False)
    value = sa.Column(sa.types.Unicode, nullable=False)
    earliest = sa.Column(sa.types.DateTime, nullable=True)
    latest = sa.Column(sa.types.DateTime, nullable=True)


def main():
    """Command-line script to create tables."""
    import argparse
    import sqlalchemy as sa
    parser = argparse.ArgumentParser(description="Create tables.")
    parser.add_argument("dburl", help="SQLAlchemy database URL.")
    parser.add_argument("--sql", action="store_true", help="Log SQL.")
    args = parser.parse_args()
    engine = sa.create_engine(args.dburl, echo=args.sql)
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":  main()
