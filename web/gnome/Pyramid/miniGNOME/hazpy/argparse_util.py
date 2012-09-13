"""Utilities for ``argparse``

Argparse is included in Python 2.7 and 3.2. For older versions, download
"argparse" from PyPI or install the Linux package "python-argparse".

Argparse is a replacement for ``optparse``, an older command-line parser
in the standard library.
"""
import argparse  # "argparse" package on PyPI, "python-argparse" in Ubuntu

class Split(argparse.Action):
    """Split a comma-delimited argument into a list of values.

    Usage:

        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--colors", action=Split)
        Split(option_strings=['--colors'], dest='colors', nargs=None, const=None, default=None, type=None, choices=None, help=None, metavar=None)
        >>> opts = parser.parse_args(["--colors=red,yellow,blue"])
        >>> opts.colors
        ['red', 'yellow', 'blue']

    This is similar to the "append" action, but sometimes it's more convenient
    to have a comma-delimited argument rather than multiple identical
    arguments. (The equivalent with append would be: "--color=red
    --color=yellow --color=blue".)

    You can change the delimiter by subclassing and overriding the
    ``.separator`` class attribute. (We can't set an instance attribute in
    the constructor because argparse requires an Action class, not an Action
    instance.)
    """

    separator = ","

    def __call__(self, parser, namespace, values, option_string=None):
        values = values.split(self.separator)
        setattr(namespace, self.dest, values)

