"""
tests of various schema nodes on their own
"""

from datetime import datetime, timedelta
from colander import DateTime, SequenceSchema, SchemaNode

from cftime import DatetimeGregorian as cfdatetime

import pytest

# class TimeSequenceSchema(base_schema.ObjTypeSchema):
#     data = SequenceSchema(
#         SchemaNode(
#             DateTime(default_tzinfo=None)
#         )
#     )

tss = SequenceSchema(SchemaNode(DateTime(default_tzinfo=None)))


def test_datetimes():
    dt = [datetime(2020, 10, 10 + i, 12, 30) for i in range(10)]

    serial = tss.serialize(dt)
    dt2 = tss.deserialize(serial)

    assert dt == dt2


# @pytest.mark.skip("these are known not to work")

@pytest.mark.xfail(reason="cfdatetime is not supported")
def test_cfdatetimes():
    """
    test of serializing cfdatetimes

    It does not work, but keeping this, in case we decide to support it one day
    """
    dt = [cfdatetime(2020, 10, 10 + i, 12, 30) for i in range(10)]

    serial = tss.serialize(dt)

    print(serial)

    dt2 = tss.deserialize(serial)

    print(dt2)

    assert dt == dt2

