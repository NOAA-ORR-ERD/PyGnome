#!/usr/bin/env python
import logging
from logging import warn, info, debug
import optparse
import sys
import time
import unicodedata

import sqlalchemy as sa

from hazpy.misc import Accumulator, Counter, singular_plural, truncate_string

USAGE = "%prog [options] mysql://USER:PASSWORD@localhost/rlink ACTION"

DESCRIPTION = """\
Analyze or convert invalid UTF-8 characters in ResponseLINK's Incident and
Entry tables.  The first arg is a SQLAlchemy database URI.  The second is the
desired action, which may be one of:
"""

PREFERRED_ENCODING = "utf-8"
OTHER_ENCODINGS = ["latin-1", "windows-1252", "macroman"]

INCIDENT_COLUMNS = [
    "name",
    "location",
    "description",
    "responsible",
    "notified_by",
    "lead_ssc",
    "commodity",
    "other_cause",
    "noaa_divisions_involved",
    "public_affairs_html",
    "ptl_entered_min",
    "ptl_entered_max",
    "ptl_unit",
    "actl_entered_min",
    "actl_entered_max",
    "actl_unit",
    ]
ENTRY_COLUMNS = ["title", "content"]
ENCODINGS = ["windows-1252", "macroman"]
PRETTY_ENCODING_NAMES = {"windows-1252": "WINDOWS", "macroman": "MACINTOSH"}

conn = None         # Database connection; set by init_database().
t_incidents = None  # Incident table; set by init_database().
t_entry = None      # Entry table; set byinit_database().

class NoUpdate(object):
    pass

#### Base action classes
class DatabaseVisitor(object):
    """Go through all values in INCIDENT_COLUMNS and ENTRY_COLUMNS in
       column-wise order.
    """
    description = "Abstract base class"

    def run(self):
        print self.description
        print
        start_time = time.time()
        self.before()
        self.iter_incidents()
        self.iter_entries()
        self.after()
        secs = time.time() - start_time
        print
        print "Finished in %d seconds, exiting." % secs

    def iter_incidents(self):
        ic = t_incident.columns
        for colname in INCIDENT_COLUMNS:
            col = getattr(ic, colname)
            fields = [
                ic.orr_id.label("id"), 
                col.label("value"), 
                ic.create_date.label("ts"),
                ]
            order_by = [ic.orr_id]
            self.iter_column(t_incident, col, fields, order_by)

    def iter_entries(self):
        ec = t_entry.columns
        for colname in ENTRY_COLUMNS:
            col = getattr(ec, colname)
            fields = [
                ec.orr_id,
                ec.entry_id,
                col.label("value"),
                ec.entry_date.label("ts"),
                ]
            order_by = [ec.entry_id]
            self.iter_column(t_entry, col, fields, order_by)

    def iter_column(self, table, column, fields, order_by):
        """'table' is a SQLAlchemy Table. 
           'column' is a SQLALchemy Column.
           'fields' is a list of Columns to select.
               The variable column must be labeled 'value'.
               The timestamp column must be labeled 'ts'.
           'order_by' is a SQLAlchemy order_by list.
        """
        print "*** Table '%s', column '%s'" % (table.name, column.name)
        self.record_count = 0
        self.before_column(table, column)
        sql = sa.select(fields).order_by(*order_by)
        for result in conn.execute(sql):
            self.record_count += 1
            self.action(table, column, result)
        self.after_column(table, column)
        print

    def action(self, table, column, result):
        raise NotImplementedError("subclass responsibility")

    def before_column(self, table, column, id_column):
        pass

    def after_column(self, table, column, id_column):
        pass

    def before(self):
        pass

    def after(self):
        pass


class ModifyingDatabaseVisitor(DatabaseVisitor):
    def before_column(self, table, column, id_column):
        self.modify_count = 0
        self.update = table.update(
            id_column==sa.bindparam("id"),
            values={column.name: sa.bindparam("value")})

    def after_column(self, table, column, id_column):
        print "%d records modified" % self.modify_count

    def action(self, table, column, r):
        value = self.convert(table, column, r)
        if value is not NoUpdate:
            self.update(id=r.id, value=value)
            self.modify_count += 1

    def convert(self, table, column, r):
        """Return the new value for the field, or NoUpdate to leave it alone.
        """
        raise NotImplementedError("subclass responsibility")


    
#### Action classes
class SimpleTest(DatabaseVisitor):
    description = "Check if we can read all the database values"

    def action(self, table, column, r):
        pass

    def after_column(self, table, column, id_column):
        print "   %d records found" % self.record_count

class Summary(DatabaseVisitor):
    description = "Summarize bad foreign chars in each column."

    def before_column(self, table, column, id_column):
        self.bad_chars = Counter()

    def action(self, table, column, r):
        if decode(r.value, "utf-8") is not None:
            return
        for c in r.value:
            if ord(c) > 127:
                self.bad_chars.register(c)

    def after_column(self, table, column):
        print "%d records total" % self.record_count
        for c in sorted(self.bad_chars.keys()):
            count = self.bad_chars[c]
            time_s = singular_plural(count, "time", "times")
            charnames = get_pretty_names_string(c, ENCODINGS, "[UNKNOWN]")
            print "%r: %d %s %s" % (c, count, time_s, charnames)


class QuickFix(ModifyingDatabaseVisitor):
    description = "Fix invalid values in the database in a simplistic way"

    def convert(self, table, column, r):
        if r.value is None:
            return NoUpdate
        if decode(r.value, "utf-8") is not None:
            return NoUpdate
        found_bad_char = False
        chars = list(r.value)
        for c in chars:
            if ord(c) >= 128:
                found_bad_char = True
                c = pretty_char_names_str(c, ENCODINGS, repr(c))
        if not found_bad_char:
            return NoUpdate
        return "".join(chars)

action_map = {
    "test":  SimpleTest,
    "summary":  Summary,
    "quick-fix":  QuickFix,
    }

#### Main routine

def decode(s, encoding):
    """Convert a string to Unicode.  Return None if unsuccessful."""
    try:
        return s.decode(encoding)
    except UnicodeDecodeError:
        return None

def name_that_character(c, encoding):
    """Return the Unicode name for a 'str' character based on the specified
       encoding.  Return None if unsuccessful.
    """
    try:
        uchr = decode(c, encoding)
    except UnicodeDecodeError:
        return None
    return unicodedata.name(uchr, None)
        
def get_pretty_names_string(c, encodings, default):
    """Return a string like "[WINDOWS LATIN BLA BLA; MACINTOSH LATIN BLA BLA]"
    """
    names = []
    for encoding in encodings:
        uname = name_that_character(c, encoding)
        if uname is None:
            continue
        pretty_encoding = PRETTY_ENCODING_NAMES.get(encoding, encoding)
        uname = "%s %s" % (pretty_encoding, uname)
        names.append(uname)
    if not names:
        return default
    names_str = "; ".join(names)
    return "[%s]" % names_str

def init_database(dburi):
    global conn, t_incident, t_entry
    if "convert_unicode" in dburi:
        parser.error("database URI must not have 'convert_unicode' option set")
    engine = sa.create_engine(dburi)
    conn = engine.connect()
    meta = sa.MetaData()
    t_incident = sa.Table("Incident", meta, autoload=True, autoload_with=conn)
    t_entry = sa.Table("Entry", meta, autoload=True, autoload_with=conn)
        
class MyOptionParser(optparse.OptionParser):
    def get_description(self):
        # It would be nicer to indent the action descriptions like
        # the option descriptions.
        chunks = [self.description, "\n"]
        for name in sorted(action_map.keys()):
            tup = name, action_map[name].description
            lin = "  %-18s  %s\n" % tup
            chunks.append(lin)
        return "".join(chunks)

    def format_description(self, formatter):
        return self.get_description()

    def error(self, msg):
        self.print_usage(sys.stderr)
        sys.stderr.write("Use --help for a list of actions and options\n")
        self.exit(2, "error: %s\n" % msg)

def main():
    parser = MyOptionParser(usage=USAGE, description=DESCRIPTION)
    pao = parser.add_option
    pao("--log-sql", action="store_true",
        help="Log SQL statements executed")
    pao("--log-sql-results", action="store_true",
        help="Log SQL result info")
    opts, args = parser.parse_args()
    if   opts.log_sql_results:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.DEBUG)
    elif opts.log_sql:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    try:
        dburi, action = args
    except ValueError:
        parser.error("wrong number of command-line options")
    init_database(dburi)
    try:
        class_ = action_map[action]
    except KeyError:
        parser.error("unrecognized action")
    class_().run()

if __name__ == "__main__":  main()


'''
def quick_fix():
    raise NotImplementedError()
    encodings = [PREFERRED_ENCODING] + OTHER_ENCODINGS
    for c in incident_fields:
        print "Analyzing table 'Incident' column '%s':" % c.name
    select_columns = [
        col.label("value"),
        primary_key_col.label("id"),
        timestamp_col.label("ts"),
        ]
    sql = sa.select(select_columns)
    for r in engine.execute(sql):
        u, encoding = decode(r.value, encodings)
        if encoding not in [PREFERRED_ENCODING, None]:
            accum.register(encoding, r)
    for encoding in sorted(accum.keys()):
        print "  %s:" % encoding
        for r in accum[encoding][:20]:
            tup = r.id, r.ts.strftime("%Y-%m-%d"), truncate_string(r.value, 50)
            print "    #%s: %s: %s" % tup
    print

def analyze_column(rslt):
    charsets = PREFERRED_CHARSET + OTHER_CHARSETS
    for r in sa.select(table).order_by(order_by):
        print "%s #%s" % (table.name, logging_record_id(r, order_by))
        for c in columns:
            value = r[c]
            u = decode(value, PREFERRED_CHARSET)
            if u is not None:
                continue   # Value is OK.
            for charset in OTHER_CHARSETS:
                u = decode(value, charset)
                if u is not None:
                    yield value
                    

def logging_record_id(record, key_columns):
    keys = [str(record[k]) for k in key_columns]
    return "/".join(keys)

def decode(value, encodings):
    for encoding in encodings:
        try:
            u = value.decode(encoding)
        except UnicodeDecodeError:
            pass
        else:
            return u, encoding
    return None, None

def analyze_table(table, columns, order_by):
    encodings = [PREFERRED_ENCODING] + OTHER_ENCODINGS
    sql = table.select().order_by(*order_by)
    for r in engine.execute(sql):
        for c in columns:
            value = r[c]
            u, encoding = decode(value, encodings)
            if encoding != PREFERRED_ENCODING:
                yield r, c, u, encoding

def count_encodings():
    for c in incident_fields:
        _count_encodings(t_incident, c)
    for c in entry_fields:
        _count_encodings(t_entry, c)
        
def _count_encodings(table, column):
    print "Analyzing table '%s' column '%s':" % (table.name, column.name)
    encodings = ["ascii", PREFERRED_ENCODING] + OTHER_ENCODINGS
    ctr = Counter()
    sql = sa.select([column])
    for r in engine.execute(sql):
        value = r[0]
        u, encoding = decode(value, encodings)
        ctr.register(encoding)
    for key in sorted(ctr.keys()):
        print "    %s: %s" % (key, ctr[key])
    print

def list_bad_encodings():
    for c in incident_fields:
        _list_bad_encodings(t_incident, c, ic.orr_id, ic.create_date)
    for c in entry_fields:
        _list_bad_encodings(t_entry, c, ec.entry_id, ec.entry_date)
    
def _list_bad_encodings(table, col, primary_key_col, timestamp_col):
    print "Analyzing table '%s' column '%s':" % (table.name, col.name)
    encodings = ["ascii", PREFERRED_ENCODING] + OTHER_ENCODINGS
    accum = Accumulator()
    select_columns = [
        col.label("value"),
        primary_key_col.label("id"),
        timestamp_col.label("ts"),
        ]
    order_by = [timestamp_col.desc()]
    sql = sa.select(select_columns).order_by(*order_by)
    for r in engine.execute(sql):
        u, encoding = decode(r.value, encodings)
        if encoding != "ascii" and encoding != PREFERRED_ENCODING:
            accum.register(encoding, r)
    for encoding in sorted(accum.keys()):
        print "  %s:" % encoding
        for r in accum[encoding][:20]:
            tup = r.id, r.ts.strftime("%Y-%m-%d"), truncate_string(r.value, 50)
            print "    #%s: %s: %s" % tup
    print

def modify():
    for c in incident_fields:
        _modify(t_incident, c, ic.orr_id, ic.create_date)
    for c in entry_fields:
        _modify(t_entry, c, ec.entry_id, ec.entry_date)
'''
