#!/usr/bin/env python
"""Miscellaneous functions.
These depend only on the Python standard library, not on third-pary modules.
"""
import bz2, datetime, gzip, itertools, os, re, subprocess, sys, tempfile
import textwrap

debug = bool(os.environ.get("DEBUG"))

TRANSLATION = None   # Set in convert_word_chars().

class DumbObject(object):
    """A container for arbitrary attributes."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

def setdefault(dic, key, factory):
    """If 'key' exists in 'dic', return its value.  Otherwise call 'factory()'
       to produce a default value, and both set the key to it and return it.

       This is more efficient than dict.setdefault if the default value is
       a new container (list, dict, set, class instance) or expensive to
       compute, because it calls the factory only when a default is needed.
       With dict.setdefault, the default value is created every time even if
       it's not used.

       Typical usage:
       >>> shopping_list = {}
       >>> setdefault(shopping_list, "Wal-Mart", list).append("paper towels")
       >>> setdefault(shopping_list, "Wal-Mart", list).append("DVD player")
       >>> setdefault(shopping_list, "Safeway", list).append("lettuce")
       >>> print shopping_list
       {"Wal-Mart": ["paper towels", "DVD player"], "Safeway: ["lettuce"]}

       Python 2.5 has a collections.defaultdict that does this in a more
       elegant way.
    """
    if key in dic:
        return dic[key]
    value = factory()
    dic[key] = value
    return value

def percent_of(part, whole):
    """What percent of 'whole' is 'part'?
       Example:  percent_of(5, 100)  =>  5.0
    """
    return part * 100 / whole

def average(r):
    return sum(r) / len(r)

mean = average

def median(r):
    s = list(r)
    s.sort()
    return s[len(s) // 2]

def stddev(r):
    """Standard deviation, from the Python Cookbook
       http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/442412
    """
    avg = average(r)
    sdsq = sum([(i - avg) ** 2 for i in r])
    return (sdsq / (len(r) - 1 or 1)) ** 0.5

def split_path_ext(p):
    dir, filename = os.path.split(p)
    base, ext = os.path.splitext(filename)
    return dir, base, ext
    
def any2datetime_date(d):
    """Convert a foreign date object (e.g., mxDateTime) to datetime.date ."""
    return datetime.date(d.year, d.month, d.day)

def singular_plural(n, singular, plural):
    if n == 1:
        return singular
    else:
        return plural

class FancyList(list):
    def column(self, col):
        return [x[col] for x in self]

def chop_at(s, sub, inclusive=False):
    """Truncate string 's' at the first occurence of 'sub'.
       If 'inclusive' is true, truncate just after 'sub' rather than at it.
    """
    pos = s.find(sub)
    if pos == -1:
        return s
    if inclusive:
        return s[:pos+len(sub)]
    return s[:pos]

def lchop(s, sub):
    if s.startswith(sub):
        s = s[len(sub):]
    return s
    
def rchop(s, sub):
    if s.endswith(sub):
        s = s[:-len(sub)]
    return s

def strip_leading_whitespace(s):
    ret = [x.lstrip() for x in s.splitlines(True)]
    return "".join(ret)

sentence_end_rx = re.compile(
    R"\.\s+[A-Z][a-z]", re.IGNORECASE)

def truncate_string(s, length):
    indicator = " ..."
    length = max(length-len(indicator), 0)
    m = sentence_end_rx.search(s)
    if m is not None:
        s = s[:m.start()+1]
    if len(s) > length:
        s = s[:length]
        last_space = s.rfind(" ")
        if last_space != -1:
            s = s[:last_space]
        s += indicator
    return s

# Characters we prefer in ResponseLINK.
# Used as arg to unicode.translate.  
# NOT FOR 'str' STRINGS!!!  Convert 'str' values to unicode first using:
# "Windows string".decode("windows-1252", "replace")  -or-
# "Macintosh string".decode("mac_roman", "replace")
win_mac_char_conversions = {
    0x00ba: U"\xb0",  # Masc ordinal -> degree.
    0x2018: U"'",
    0x2019: U"'",
    0x201c: U'"',
    0x201d: U'"',
    0x2013: U"-",
    0x2014: U"--",
    0x2026: U"...",
    }

def only_some_keys(dic, *keys):
    """Return a copy of the dict with only certain keys present.  The source
       may be any mapping; the result is always a Python dictionary.
    """
    ret = {}
    for key in keys:
        ret[key] = dic[key]   # Raises KeyError.
    return ret

def except_keys(dic, *keys):
    """Return a copy of the dict without the specified keys.
    """
    ret = dic.copy()
    for key in keys:
        try:
            del ret[key]
        except KeyError:
            pass
    return ret

def extract_keys(dic, *keys):
    """Return two copies of the dict.  The first has only the keys
       specified.  The second has all the *other* keys from the original dict.
    """
    for k in keys:
        if k not in dic:
            raise KeyError("key %r is not in original mapping" % k)
    r1 = {}
    r2 = {}
    for k, v in dic.items():
        if k in keys:
            r1[k] = v
        else:
            r2[k] = v
    return r1, r2

def ordered_items(dic, key_order, other_keys_too=True):
    """Like dict.iteritems() but yield the keys in 'key_order' if they exist.
       If 'other_keys_too' is true, then yield the other items in an 
       arbitrary order.
    """
    d = dic.copy()
    for key in key_order:
        if key in d:
            yield key, d.pop(key)
    if other_keys_too:
        for key, value in d.iteritems():
            yield key, value

def nulls_to_empty(dic, *keys):
    """Convert None values to ''.
       @param dic
       @param keys Only these keys are affected.  If none, operate on all keys.
       @return None (Modifies 'dic' in place.)
       Useful for SQL queries where NULL values are not allowed.
    """
    if not keys:
        keys = dic.keys()
    for key in keys:
        if dic[key] is None:
            dic[key] = ''
    return None

def convert_nulls(dic, null_value):
    """Replace all None values in a dict with an arbitrary value.
       @param dic
       @param null_value The value to be substituted.
       @return None (Modifies 'dic' in place.)
       Unlike nulls_to_empty(), you cannot limit this function to certain keys.
   """
    for key in dic.iterkeys():
        if dic[key] is None:
            dic[key] = null_value

def del_quiet(dic, *keys):
    """Delete keys from dict, with no error if they don't exist."""
    for key in keys:
        try:
            del dic[key]
        except KeyError:
            pass

def columnize(lis, columns, horizontal=False, fill=None):
    if columns < 1:
        raise ValueError("arg 'columns' must be >= 1")
    if horizontal:
        ret = []
        for i in range(0, len(lis), columns):
            row = lis[i:i+columns]
            row_len = len(row)
            if row_len < columns:
                extra = [fill] * (columns - row_len)
                row.extend(extra)
            ret.append(row)
        return ret
    lisLen = len(lis)
    columnLen, remainder = divmod(lisLen, columns)
    if remainder:
        columnLen += 1
    ret = [None] * columns
    for i in range(columns):
        start = i * columnLen
        end = min(start + columnLen, lisLen)
        #print "i=%d, start=%d, end=%d, element=%r" % (i, start, end, lis[start:end])
        ret[i] = lis[start:end]
    return ret

def columnize_as_rows(lis, columns, horizontal=False, fill=None):
    """Like 'zip' but fill any missing elements."""
    data = columnize(lis, columns, horizontal, fill)
    rowcount = len(data)
    length = max(len(x) for x in data)
    for c, lis in enumerate(data):
        n = length - len(lis)
        if n > 0:
            extension = [fill] * n
            lis.extend(extension)
    return zip(*data)

def izip_fill(*iterables, **kw):
    """Like itertools.izip but use a default value for the missing elements
       in short lists rather than stopping at the end of the shortest list.

       @param *iterables the iterables to zip.
       @param **kw 'default' specifies the default value (default None).
    """
    iterables = map(iter, iterables)
    default = kw.pop('default', None)
    if kw:
        raise TypeError("unrecognized keyword arguments")
    columns = len(iterables)
    columns_range = range(columns)
    while True:
        found_data = False
        row = [None] * columns
        for i in columns_range:
            try:
                row[i] = iterables[i].next()
                found_data = True
            except StopIteration:
                row[i] = default
        if not found_data:
            break
        yield tuple(row)

def any(seq, pred=None):
    """From recipe in itertools docs."""
    for elem in itertools.ifilter(pred, seq):
        return True
    return False

def all(seq, pred=None):
    """From recipe in itertools docs."""
    for elem in itertoos.ifilterfalse(pred, seq):
        return False
    return True

def no(seq, pred=None):
    """From recipe in itertools docs."""
    for elem in ifilter(pred, seq):
        return False
    return True

def count_true(seq, pred=lambda x: x):
    """Couldn't get itertools 'quantify' recipe to work."""
    ret = 0
    for x in seq:
        if pred(x):
            ret += 1
    return ret

def convert_or_none(value, type_):
    """Return the value converted to the type, or None if error.
       'type_' may be a Python type or any function.
    """
    try:
        return type_(value)
    except Exception:
        return None

class Counter(dict):
    def __init__(self):
        self.total = 0  # Number of times .register has been called.

    def __call__(self, key):
        if key in self:
            self[key] += 1
        else:
            self[key] = 1
        self.total += 1

    add = register = __call__  # Backward compatibility.

    def get_popular(self, max_items=None):
        """Return the counter keys as a list of (count, key) pairs in reverse
           order of count (so the key with the highest count is first).
           If 'max_items' is provided, return no more than that many items.
        """
        data = [(x[1], x[0]) for x in self.iteritems()]
        data.sort(key=lambda x: (sys.maxint - x[0], x[1]))
        if max_items:
            return data[:max_items]
        else:
            return data

    def get_sorted_values(self):
        data = self.items()
        data.sort()
        return data

class Accumulator(dict):
    def __call__(self, key, value):
        self.setdefault(key, []).append(value)

    add = register = __call__  # Backward compatibility

class UniqueAccumulator(dict):
    def __call__(self, key, value):
        self.setdefault(key, set()).add(value)

    add = register = __call__   # Backward compatibility.

class SimpleStats(object):
    def __init__(self, keep_values):
        self.keep_values = keep_values
        self.list = []
        self.set = set()
        self.count = 0
        self.min = None
        self.max = None

    def __call__(self, value):
        if self.count == 0:
            self.min = self.max = value
        else:
            self.min = min(self.min, value)
            self.max = max(self.max, value)
        self.count += 1
        if self.keep_values:
            self.list.append(value)
            self.set.add(value)

    def __nonzero__(self):
        return bool(self.count)
                
    

def iter_sparse_dict(full_range, dic, default=None, keyfunc=None):
    for key in full_range:
        if keyfunc:
            key2 = keyfunc(key)
        else:
            key2 = key
        yield key, dic.get(key2, default)
    
    

def wrap_paragraphs(text, wrapper=None, width=72, **kw):
    if not wrapper:
        wrapper = textwrap.TextWrapper(width=width, **kw)
    result = []
    lines = text.splitlines(True)
    lines_len = len(lines)
    start = 0
    end = None
    while start < lines_len:
        # Leave short lines as-is.
        if len(lines[start]) <= width:
            result.append(lines[start])
            start += 1
            continue
        # Found a long line, peek forward to end of paragraph.
        end = start + 1
        while end < lines_len and not lines[end].isspace():
            end += 1
        # 'end' is one higher than last long lone.
        paragraph = ''.join(lines[start:end])
        paragraph = wrapper.fill(paragraph) + "\n"
        result.append(paragraph)
        start = end
        end = None
    #if "C4H6" in text:
    #    import pprint
    #    pprint.pprint(lines)
    #    print "*****************"
    #    pprint.pprint(result)
    return "".join(result)

def batch(iterable, max_size, action, prepare_element=None):
    """Split 'iterable' into batches of 'max_size' and run 'action' on
       each batch.  Returns a list of whatever 'action' returns, so you 
       may get a list of None.  'prepare_element' can be a function that
       modifies each element in 'iterable' before the action.
    """
    batch_range = range(max_size)
    iterable = iter(iterable)
    ret = []
    last = False
    while not last:
        batch = []
        for i in batch_range:
            try:
                elm = iterable.next()
            except StopIteration:
                last = True
                break
            if prepare_element is not None:
                elm = prepare_element(elm)
            batch.append(elm)
        if batch:
            elm2 = action(batch)
            ret.append(elm2)
    return ret

def od(s):    
    fd, path = tempfile.mkstemp()
    os.write(fd, s)
    os.close(fd)
    cmd = ["od", "-ah", path]
    return subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE
        ).stdout.read()

def unique(lis):
    """Return a list without duplicates.  Preserves the order of first
       occurrence.
       @param lis iterable The elements to uniqify.
       @return list A new list.
    """
    seen = set()
    ret = []
    for elm in lis:
        if elm not in seen:
            ret.append(elm)
            seen.add(elm)
    return ret

def smart_open(filename, mode):
    """Unified front end for opening plain files and compressed files."""
    if   filename.endswith(".bz2"):
        opener = bz2.BZ2File
    elif filename.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open
    return opener(filename, mode)
    
def american_number_commas(n):
    """Add commas between every third digit per American usage."""
    n = str(n)
    pos = n.find('.')
    if pos != -1:
        digits = list(n[:pos])
        fract = n[pos:]
    else:
        digits = list(n)
        fract = ""
    for i in range( len(digits)-3, 0, -3):
        digits.insert(i, ',')
    return "".join(digits) + fract

