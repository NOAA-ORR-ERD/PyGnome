"""
Test if this is how we want id property of 
object that inherits from GnomeObject to behave
"""

import pytest

from uuid import uuid1, UUID
from gnome.gnomeobject import GnomeObject

def test_exceptions():
    with pytest.raises(ValueError):
        go = GnomeObject()
        print "\n id exists: {0}".format(go.id)   # calling getter, assigns an id
        go.id = uuid1()
        
    with pytest.raises(ValueError):
        go = GnomeObject()
        go.id = '1234'        

@pytest.mark.parametrize('value',[str(uuid1()), uuid1()])
def test_assign_id(value):
    """
    Assign a new id from a UUID or a string
    """
    go = GnomeObject()
    go.id = value
    assert go.id == str(value)
    