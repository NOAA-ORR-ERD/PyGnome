'''
test functionality of the save_load module used to persist save files
'''
from gnome.persist import References
from gnome.movers import constant_wind_mover


def test_reference_object():
    '''
    get a reference to an object,then retrieve the object by reference
    '''
    a = 1
    refs = References()
    r1 = refs.reference(a)
    obj = refs.retrieve(r1)
    assert obj is a

    r2 = refs.reference(a)
    assert r2 == r1


def test_gnome_obj_reference():
    '''
    create two equal but different objects and make sure a new reference is
    created for each
    '''
    l_ = [constant_wind_mover(0, 0) for i in range(2)]
    assert l_[0] == l_[1]
    assert l_[0] is not l_[1]

    refs = References()
    r_l = [refs.reference(item) for item in l_]
    assert len(r_l) == len(l_)
    assert r_l[0] != r_l[1]

    for ix, ref in enumerate(r_l):
        assert refs.retrieve(ref) is l_[ix]

    unknown = constant_wind_mover(0, 0)
    assert refs.retrieve(unknown) is None
