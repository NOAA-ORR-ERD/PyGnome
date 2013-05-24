"""SQLAlchemy utility functions."""

import contextlib
import logging

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.orm.attributes import InstrumentedAttribute

log = logging.getLogger(__name__)

def create_engine(dburl):
    """OR&R standard database engine. (For MySQL, not needed for PG or SQLite.)
    
    Standard configuration:
    ``pool_recycle=3600`` because MySQL automatically disconnects idle
    connections without informing the application.  

    ``convert_unicode=True`` because applications want Unicode strings.
    All databases should be set to UTF-8 character set.  If a column
    must store non-character data, it should be BLOB / BINARY type.

    This function may have to be revised if later Postgres uses need
    Postgres-specific options.
    """
    return sa.create_engine(dburl, 
        pool_recycle=3600, convert_unicode=True)

def has_table(bind, table_name):
    """Does a table named ``table_name`` exist?"""

    return bind.dialect.has_table(bind, table_name)

def set_log_level_for_engine(engine, level):
    """Set the log level for a SQLAlchemy engine.

    ``engine`` is a SQLAlchemy engine.
    ``level`` is a log level (see Python's ``logging`` package).

    This is a finer-grained counterpart to the engine's ``echo`` property.
    """
    engine_id = hex(id(engine))
    logger_name = "sqlalchemy.engine.base.Engine.0x..." + engine_id[-4:]
    log.debug("setting logger [%s] to level %s", logger_name, level)
    logger = logging.getLogger(logger_name).setLevel(level)

def disable_sql_logging_for_engine(engine):
    """Disable SQL logging for a particular SQLAlchmey engine.

    ``engine.echo = False`` does not properly disable logging if the parent
    logger (sqlalchemy.engine) is set to INFO or DEBUG.  This function
    definitively set's the engine's log level to WARN to suppress the logging.
    """
    set_log_level_for_engine(engine, logging.WARN)

class SALogger(object):
    """A logger proxy for SQLAlchemy that can be switched on and off.

    Usage::

        sql_log = SALogger()
        sql_log.initialize(is_user_want_sql_logging)
        sql_log.disable()
        engine.execute(TABLE.insert(), LOTS_OF_DATA)
        sql_log.enable()

        with sql_log.disabling:
            engine.execute(TABLE.insert(), LOTS_OF_DATA)

    The instance proxies a Python logger, which is its ``.logger`` attribute.
    Calling the instance will be passed through to the logger, as will calling
    a method not defined in this class (like ``.info()``) or reading an
    attribute not defined in this class (like ``.level``). Setting an attribute
    will not be passed through because you're supposed to use the logger
    methods instead.

    The logger has two enabling levels beyond what Python's logger provides.
    Both must be true to enable logging. The first level is a program-wide
    boolean, which usually comes from a command-line argument. You can pass it
    to the constructor, or to a special ``.initialize()`` method if the value
    is not known until you parse the command line. Until it's initialized it's
    assumed to be false, which suppresses logging.

    The second level is the ``.enable()`` and ``.disable()`` methods
    and the ``disabling`` context manager. These are used to temporary disable
    logging
    while executing too-verbose queries such as large INSERT statements. 
    In fact, the default disabling message assumes you're disabling
    it for insert, but you can override it with the ``reason`` argument.
    
    Remember, the logging most be both initialized to true *and* enabled in
    order for log messages to appear.
    """
    def __init__(self, want=None, logname="sqlalchemy.engine"):
        self.logger = logger = logging.getLogger(logname)
        logger.setLevel(logging.WARN)
        self.want = False
        if want is not None:
            self.initalize(want)

    def initialize(self, want=True):
        self.want = want
        if want:
            self.enable()

    def enable(self):
        if not self.want:
            return
        self.logger.setLevel(logging.INFO)
        self.logger.info("enabling SQL log")

    def disable(self, reason="for insert statements"):
        if not self.want:
            return
        message = "disabling SQL log"
        if reason:
            message = message + " " + reason
        self.logger.info(message)
        self.logger.setLevel(logging.WARN)

    @contextlib.contextmanager
    def disabling(self, reason="for insert statements"):
        self.disable(reason)
        try:
            yield
        finally:
            self.enable()


def scalar(select_obj, conn, default=None):
    """Return the first field of the first row as a scalar, or ``default``
    if no rows are selected.

    Note: this function is not necessary:
        value = engine.execute(sql).scalar()
    But the default will always be None in that case.
    """
    r = conn.execute(select_obj).fetchone()
    if not r:
        return default
    return r[0]

def column(select_obj, conn):
    """Return the first column of the result as a list.
    """
    return [x[0] for x in conn.execute(select_obj)]

def column_set(select_obj, conn):
    """Return the first column of the result as a set.
    """
    return set(x[0] for x in conn.execute(select_obj))


def has_value(it):
    """Does a record exist?

    ``it`` is any iterable, including a SQLAlchemy query or result object.

    I fetch the next value from the iterator and discard it.  I return
    ``True`` if there was a value, or ``False`` if not (i.e., if
    ``StopIteration`` was encountered).
    """
    try:
        next(iter(it))
    except StopIteration:
        return False
    return True
    

def count(select_obj, conn):
    """Count the results in a SQLAlchemy select.  

    ``select_obj``: a SQLAlchemy select object.
    ``conn``: a SQLAlchemy connection, engine, or bound session.

    This performs a redundant query, which takes almost as long as the
    original query.  Depending on your application, this may be more or
    less efficient than fetching all records and then looking at the 
    length of the list.
    """
    subselect = select_obj.alias("subselect")
    sql = sa.select([sa.func.count()], from_obj=[subselect])
    return scalar(sql, conn)

def make_deferred_properties(columns, defer_group, except_column_names):
    """Make a deferred group covering all columns except those specified.

       SQLAlchemy has a 'deferred' feature that allows you to avoid loading
       large infrequently-used columns until you explicitly access them.
       Typically the deferred columns appear only on detail pages, while the
       underferred columns also appear in indexes and simple searches.
       SQLAlchemy normally requires you to list deferred columns explicitly.
       This convenience function builds the list for you, deferring all
       columns not listed as undeferred.

       'columns': pass the .columns attribute from a SQLAlchemy Table.
       'defer_group' is the name of the defer group to create.
       'except_column_names' is a list of column names not to defer.

       Usage:

           _properties = make_deferred_properties(t_mytable.columns, "details",
               ["id", "title", "author"])
           sqlalchemy.orm.mapper(MyClass, t_mytable, properties=_properties)

           # Example query
           q = Session.query(MyClass)
           if details:
                q = q.option(sqlalchemy.orm.undefer_group("details"))
           records = q.all()
    """
    ret = {}
    for col in columns:
        if col.name not in except_column_names:
            ret[col.name] = orm.deferred(col, group=defer_group)
    return ret

def iter_instrumented_attrs(ormclass):
    """Iterate the attributes of the ORM class which are "instrumented";
    i.e., linked to table columns or SQL expressions.
    """
    for attr, value in vars(ormclass).iteritems():
        if isinstance(value, InstrumentedAttribute):
            yield attr


class MultiText(sa.types.TypeDecorator):
    """Store a tuple of string values as a single delimited string.
    
    Legal values are a tuple of strings, or ``None`` for NULL.
    Lists are not allowed because SQLAlchemy can't recognize in-place
    modifications.

    Note that during SQL queries (e.g., column LIKE "%ABC%"), the
    comparision is against the delimited string.  This may cause unexpected
    results if the control value contains the delimeter as a substring.
    
    By default it shadows a ``UnicodeText`` column. To change this, override
    the ``impl`` class attribute in a subclass.
    """

    impl = sa.types.UnicodeText

    def __init__(self, delimiter, *args, **kw):
        """Constructor.

        The first positional arg is the delimiter, and is required.
        
        All other positional and keyword args are passed to the underlying
        column type.
        """
        if not isinstance(delimiter, basestring):
            msg = "arg ``delimiter`` must be string, not %r"
            raise TypeError(msg % delimiter)
        self.delimiter = delimiter
        sa.types.TypeDecorator.__init__(self, *args, **kw)

    def process_bind_param(self, value, dialect):
        """Convert a tuple of strings to a single delimited string.

        Exceptions:
            ``TypeError`` if the value is neither a tuple nor ``None``.
            ``TypeError`` if any element is not a string.
            ``ValueError`` if any element contains the delimeter as a substring.
        """
        if value is None:
            return None
        if not isinstance(value, tuple):
            msg = "%s value must be a tuple, not %r"
            tup = self.__class__.__name__, value
            raise TypeError(msg % tup)
        for i, element in enumerate(value):
            if self.delimiter in element:
                msg = "delimiter %r found in index %d of %s: %r"
                tup = (self.delimiter, i, self.__class__.__name, value)
                raise ValueError(msg % tup)
        return self.delimiter.join(value)

    def process_result_value(self, value, dialect):
        """Convert a delimited string to a tuple of strings."""
        if value is None:
            return None
        elif value == "":
            return ()
        elements = value.split(self.delimiter)
        return tuple(elements)

    def copy(self):
        return self.__class__(self.delimiter, self.impl.length)


#### Column functions
def year(date_col):
    """Return SQL expression for extracting a year from a date/time."""
    year = sa.func.date_part("year", date_col)
    return sa.cast(year, sa.types.Integer)
