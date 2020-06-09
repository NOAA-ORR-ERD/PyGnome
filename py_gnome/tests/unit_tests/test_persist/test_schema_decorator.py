"""
test code for the schema decorator

only started -- needs to be fleshed out and tested.

key missing feature: having it build a full schema from subclasses

"""

from gnome.persist.schema_decorator import serializable

from colander import SchemaNode, String, Float, Integer, Boolean


def test_simple():
    @serializable
    class Example():
        x = 5 # not a schema object
        y = Float()
        z = Integer(strict=True)
        s = String()
        def __init__(self, x, y, z, s):
            self.x = x
            self.y = y
            self.z = z
            self.s = s



    ex = Example(1, 2.0, 3, "some text")

    assert ex.x == 1

    assert ex._schema.__name__ == "ExampleSchema"





