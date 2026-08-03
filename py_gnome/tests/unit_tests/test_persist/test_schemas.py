"""
tests of various schema nodes on their own
"""

from datetime import datetime, timedelta
from colander import DateTime, SequenceSchema, SchemaNode
import numpy as np

from cftime import DatetimeGregorian as cfdatetime

from gnome.gnomeobject import GnomeId
from gnome.persist.base_schema import StringListSchema, Invalid, ObjTypeSchema, ObjType, GeneralGnomeObjectSchema
from gnome.persist.extend_colander import DataSchemaNode

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

def test_StringListSchema():
    schema = StringListSchema()

    list_of_str = ['fred', 'bob', 'jim']

    ser = schema.serialize(list_of_str)

    assert ser == list_of_str  # in this case, it's the same :-)

    deser = schema.deserialize(ser)

    assert deser == list_of_str  # it should round trip

    print(ser)

    # it converts to str for you
    ser = schema.serialize(['string', 5])
    assert ser == ['string', '5']

    with pytest.raises(Invalid):  # should fail if not a list of strings
        deser = schema.deserialize(['string', 5])

        print(deser)


from gnome.utilities.serializable_demo_objects import GnomeID_OBJ1, GnomeID_OBJ2
class Test_GnomeID_Serialization_and_Save_Features(object):
    
    def setup_obj1_Nones(self):
        test1_obj = GnomeID_OBJ1(raw_numpy=None, raw_masked=None, string_array=None, raw_numeric_list=None, raw_string_list=None, sub_obj=None)
        return test1_obj
    
    def setup_both_obj_Nones(self):
        test2_obj = GnomeID_OBJ2(raw_numpy=None, raw_scalar=None)
        test1_obj = GnomeID_OBJ1(raw_numpy=None, raw_masked=None, string_array=None, raw_numeric_list=None, raw_string_list=None, sub_obj=test2_obj)
        return test1_obj, test2_obj
    
    def setup_both_obj_data(self):
        test2_obj = GnomeID_OBJ2(raw_numpy=np.array([1, 2, 3]), raw_scalar=5)
        test1_obj = GnomeID_OBJ1(raw_numpy=np.array([4, 5, 6]),
                                           raw_masked=np.ma.MaskedArray(data=np.array([7, 8, 9]),mask=np.array([False, True, False])),
                                           string_array=np.array(['a', 'b', 'c']),
                                           raw_numeric_list=[1, 2, 3],
                                           raw_string_list=['x', 'y', 'z'],
                                           sub_obj=test2_obj)
        return test1_obj, test2_obj
    
    def test_obj1_Nones(self):
        test1_obj = self.setup_obj1_Nones()
        ser = test1_obj.serialize()
        deser = GnomeID_OBJ1.deserialize(ser)
        assert test1_obj == deser
        
        _json_, _zipfile, _refs = test1_obj.save(None)
        loaded_obj = GnomeID_OBJ1.load(_zipfile)
        assert test1_obj == loaded_obj
        
    def test_both_obj_Nones(self):
        test1_obj, test2_obj = self.setup_both_obj_Nones()
        ser = test1_obj.serialize()
        assert ser['sub_obj'] is not None
        
        deser = GnomeID_OBJ1.deserialize(ser)
        assert test1_obj == deser
        
        _json_, _zipfile, _refs = test1_obj.save(None)
        loaded_obj = GnomeID_OBJ1.load(_zipfile)
        assert test1_obj == loaded_obj
        
    def test_both_obj_data(self):
        test1_obj, test2_obj = self.setup_both_obj_data()
        ser = test1_obj.serialize()
        assert '??' in ser['raw_numpy']
        assert '??' in ser['raw_masked']
        assert '??' in ser['string_array']
        assert ser['raw_numeric_list'] == [1, 2, 3]
        assert ser['raw_string_list'] == ['x', 'y', 'z']
        assert ser['sub_obj'] is not None
        assert '??' in ser['sub_obj']['raw_numpy']
        assert ser['sub_obj']['raw_scalar'] == 5
        
        with pytest.raises(NotImplementedError):
            deser = GnomeID_OBJ1.deserialize(ser)
        
        _json_, _zipfile, _refs = test1_obj.save(None)
        loaded_obj = GnomeID_OBJ1.load(_zipfile)
        assert test1_obj == loaded_obj