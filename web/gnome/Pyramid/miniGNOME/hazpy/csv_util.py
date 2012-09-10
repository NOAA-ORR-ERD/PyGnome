"""CSV file helpers."""

import csv
import datetime
import re

rx_linebreak = re.compile(u"\r\n|\r|\n")

class Formatter(object):
    """Convert a Python type to a valid CSV value."""

    null = ""
    true = "1"
    false = "0"
    encoding = "utf-8"
    encoding_errors = "xmlcharrefreplace"
    linebreak_char = "\v"
    date_format = "%Y-%m-%d"
    time_format = "%H:%M:%S"
    datetime_format = date_format + " " + time_format

    def __init__(self):
        self.converters = {}

    def __call__(self, v):
        # Convert Python primitive types
        if v is None:
            return self.null
        if v is True:
            return self.true
        if v is False:
            return self.false
        if isinstance(v, datetime.datetime):
            return v.strftime(self.datetime_format)
        if isinstance(v, datetime.date):
            return v.strftime(self.date_format)
        if isinstance(v, datetime.time):
            return v.strftime(self.time_format)
        # Convert custom types (will also do string conversion afterward)
        for type_ in self.converters:
            if isinstance(v, type_):
                v = self.converters[type_](v)
                break
        # Convert to string, encode Unicode, and replace line breaks
        if isinstance(v, str):
            pass
        elif isinstance(v, unicode):
            v = v.encode(self.encoding, self.encoding_errors)
        else:
            v = unicode(v).encode(self.encoding, self.encoding_errors)
        v = rx_linebreak.sub(self.linebreak_char, v)
        return v


class FormattingDictWriter(csv.DictWriter):
    """Like csv.DictWriter but format every value.

    The default behavior for missing or extra keys is the opposite of
    ``DictWriter``, and the arguments to customize them are different.
    Missing keys will raise KeyError, on the asumption that the key is
    misspelled. Extra keys are ignored, on the assumption that the dict is used
    for more than just the CSV file.  To specify a default value for missing
    keys, use the ``default`` arg. To raise an error if extra keys are found,
    pass ``ignore_extra_keys=False``. The superclass arguments 'restval' and
    'extrasaction' are not allowed.
    """

    def __init__(self, f, fieldnames, formatter=None, default=None,
        ignore_extra_keys=True, *args, **kw):
        if not formatter:
            formatter = Formatter()
        self.formatter = formatter
        if "restval" in kw:
            msg = "subclass uses 'default' arg instead of 'restval'"
            raise TypeError(msg)
        if "extrasection" in kw:
            msg = "subclass uses 'ignore_extra' arg instead of 'extrasection'"
            raise TypeError(msg)
        extrasaction = "ignore" if ignore_extra_keys else "raise"
        csv.DictWriter.__init__(self, f, fieldnames, restval=default,
            extrasaction=extrasaction, *args, **kw)

    def writeheader(self, fieldnames=None):
        """Write a header row containing the field names.

        If ``fieldnames`` is specified, it overrides the internal names.
        its length must be the same as the original fieldnames.
        """
        if fieldnames is None:
            fieldnames = self.fieldnames
        self.writerow(fieldnames)

    def writerow(self, rowdict):
        """Write a row based on a dict of values.

        The fields will be written in the order specified by the constructor.

        If ``rowdict`` is a list or tuple, it must be the same length as
        ``self.fieldnames``. The values will be formatted and written without
        changing their order.
        """
        if isinstance(rowdict, (list, tuple)):
            row = rowdict
            if len(row) != len(self.fieldnames):
                msg = "expected {0} sequence items, found {0}"
                msg = msg.format(len(self.fieldnames), len(row))
                raise ValueError(msg)
        else:
            row = self._dict_to_list(rowdict)
        row = map(self.formatter, row)
        self.writer.writerow(row)

    def writerows(self, rowdicts):
        """Write several CSV rows, formatting all values.
        
        Unlike `writerow``, the ``rowdicts`` must be mappings, not sequences.
        """
        rows = []
        for rowdict in rowdicts:
            row = self._dict_to_list(rowdict)
            row = map(self.formatter, row)
            rows.append(row)
        self.writer.writerows(rows)

    #### Private methods
    def _dict_to_list(self, rowdict):
        """Override superclass implementation. If the ``restval`` constructor
        arg was ``None``, raise an error if any row is missing field keys.
        """
        if self.extrasaction == "raise":
            wrong_fields = [k for k in rowdict if k not in self.fieldnames]
            if wrong_fields:
                raise ValueError("dict contains fields not in fieldnames: " +
                                 ", ".join(wrong_fields))
        if self.restval is None:
            return [rowdict[key] for key in self.fieldnames]
        else:
            return [rowdict.get(key, self.restval) for key in self.fieldnames]
