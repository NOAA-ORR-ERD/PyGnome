"""A website logger with a SQLAlchemy backend.

The ``Logger`` class should be instantiated separately for each request
because it logs the request start time.  The constructor takes two arguments:

    ``site``: a string site ID such as "cameo", or perhaps "cameo1"
for a beta deployment.  

    ``engine``: a SQLAlchemy engine with read/write permission to a database
    with tables as specified in ``hazpy.sitestats.model``.  If ``None``,
    disable logging.

The public methods write a log record to one or more tables.  They all do 
nothing if the engine has been set to ``None``.
"""

import datetime
import urlparse

import hazpy.sitestats.constants as constants
import hazpy.sitestats.model as model

class Logger(object):
    insert_access = model.Access.__table__.insert()
    insert_referer = model.Referer.__table__.insert()
    insert_search = model.Search.__table__.insert()
    insert_event = model.Event.__table__.insert()

    def __init__(self, site, engine):
        self.site = self._to_unicode(site)
        self.engine = engine or None
        self.now = datetime.datetime.now()

    def log_access(self, remote_addr, status, url, session_id=None, 
        username=None):
        if not self.engine:
            return
        u = urlparse.urlsplit(url)
        r = {
            "site": self.site,
            "ts": self.now,
            "remote_addr": self._to_unicode(remote_addr),
            "status": status,
            "url": self._to_unicode(u.path),
            "query": self._to_unicode(u.query),
            "session": self._to_unicode(session_id) or None,
            "user": self._to_unicode(username) or None,
            }
        rslt = conn = self.engine.execute(self.insert_access, r)
        rslt.close()  # Explicitly close to avoid lingering connections.

    def log_access_pylons(self, request, response, session_id=None, 
        username=None):
        if not self.engine:
            return
        remote_addr = request.environ.get("HTTP_X_FORWARDED_FOR",
            request.remote_addr)
        if request.query_string:
            url = "%s?%s" % (request.path_info, request.query_string)
        else:
            url = request.path_info
        self.log_access(remote_addr=remote_addr, status=response.status_int, 
            url=url, session_id=session_id, username=username)

    def log_referer(self, referer):
        if not self.engine or not referer:
            return
        u = urlparse.urlsplit(referer)
        if not u.hostname or u.hostname in constants.IGNORE_REFERER_DOMAINS:
            return
        r = {
        "site": self.site,
            "ts": self.now,
            "scheme": self._to_unicode(u.scheme) or u"",
            "domain": self._to_unicode(u.hostname) or u"",
            "path": self._to_unicode(u.path) or u"",
            "query": self._to_unicode(u.query) or u"",
            }
        rslt = self.engine.execute(self.insert_referer, r)
        rslt.close()  # Explicitly close to avoid lingering connections.

    def log_search(self, type_, term, success):
        """'success' is anything that evaluates to boolean: True, False, 
        the result count, or a list of results."""
        if not self.engine or not term:
            return
        if not success:
            type_ += "-fail"
        self._increment(model.Search, type_, term.lower())

    def log_event(self, type_, value):
        if not self.engine or not value:
            return
        self._increment(model.Event, type_, value)

    #### Private methods
    def _to_unicode(self, s, from_http_header=False):
        """Convert 's' to a Unicode string, handling any errors.
        
        By default assume the source encoding is 'utf-8'. If 'from_http_header'
        is true, use 'latin-1' instead for wider compatibility with HTTP
        headers from unknown sources.

        If 's' is None, return it unchanged.
        """
        if s is None:
            return None
        elif isinstance(s, unicode):
            return s.strip()
        elif not isinstance(s, str):
            return unicode(s).strip()
        if from_http_header:
            encoding = "latin-1"
        else:
            encoding = "utf-8"
        return unicode(s, encoding, "replace").strip()

    def _increment(self, orm_class, type_, value):
        """Increment a search or event record count, or insert the record if
        it doesn't exist.

        * orm_class: an ORM class containing the following fields: site,
          type, year, count, and the field named by 'value_field'.
        * type_: the record type.
        * value: the value being recorded. It must be a string.
        """
        type_ = self._to_unicode(type_)
        value = self._to_unicode(value)
        year = self.now.year
        conn = self.engine.connect()
        sess = model.Session(bind=conn)
        q = sess.query(orm_class)
        q = q.filter_by(site=self.site, type=type_, year=year, value=value)
        e = q.first()    # 'e' is the event or search record.
        if e:
            e.count += 1
        else:   
            e = orm_class()
            sess.add(e)
            e.site = self.site
            e.type = type_
            e.year = year
            e.count = 1
            e.value = value
            e.earliest = self.now
        e.latest = self.now
        sess.commit()
        conn.close()  # Explicitly close to avoid lingering connections.
