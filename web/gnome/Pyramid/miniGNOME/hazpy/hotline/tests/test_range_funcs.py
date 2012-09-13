"""Nose tests.  To use:
   easy_install 'Nose>=0.10'   
   Or if not available yet: easy_install 'Nose==dev'.
   Then run "nosetests hazpy.hotline.tests.test_range_funcs
"""
from nose.tools import eq_ as X

from hazpy.hotline.range_funcs import numeric_release_for_entered as N
from hazpy.hotline.range_funcs import entered_release_for_numeric as E
from hazpy.hotline.range_funcs import format_range as F

def test_entered():
    X( E(None, None, False),        ("", "", "") )
    X( E(None, None, True),         ("", "", "") )
    X( E(100.223, None, False),     ("100", "", "gallons") )
    X( E(100.223, None, True),      ("100", "", "pounds") )
    X( E(100.223, 101.9, False),    ("100", "101", "gallons") )
    X( E(0.223, 0.9, False),        ("0.22", "0.90", "gallons") )
    X( E(0.228, 0.9, False),        ("0.23", "0.90", "gallons") )
    X( E(0.223, 1234567890, False), ("0.22", "1,234,567,890", "gallons") )
    X( E(1234567890, None, False),  ("1,234,567,890", "", "gallons") )


def test_numeric():
    X( N("", "", ""),                        (None, None, False) )
    X( N("100", "", "gallons"),              (100.0, None, False) )
    X( N("100", "", "pounds"),               (100.0, None, True) )
    X( N("100", "101", "gallons"),           (100.0, 101, False) )
    X( N("0.22", "0.90", "gallons"),         (0.22, 0.90, False) )
    X( N("0.23", "0.90", "gallons"),         (0.23, 0.90, False) )
    X( N("0.22", "1,234,567,890", "gallons"),(0.22, 1234567890.0, False) )
    X( N("1,234,567,890", "", "gallons"),    (1234567890.0, None, False) )
    X( N("1,234,567,890.12", "", "gallons"), (1234567890.12, None, False) )

def test_display_range():
    X( F("", "", "", None, None, False),                 "")
    X( F("100", "", "gallons", 100.0, None, False),      "100 gallons")
    X( F("100", "", "pounds", 100.0, None, True),        "100 pounds")
    X( F("100", "101", "gallons", 100.0, 101.0, False),  "100 - 101 gallons")
    X( F("0.22", "0.9", "gallons", 0.22, 0.90, False),   "0.22 - 0.9 gallons")
    X( F("0.23", "0.901", "gallons", 0.23, 0.901, False), 
        "0.23 - 0.901 gallons")
    X( F("0.22", "1,234,567,890", "gallons", 0.22, 1234567890.0, False), 
        "0.22 - 1,234,567,890 gallons")
    X( F("0.22", "1,234,567,890.12", "gallons", 0.22, 1234567890.12, False),
        "0.22 - 1,234,567,890.12 gallons")
