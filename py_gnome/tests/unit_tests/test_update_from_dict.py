'''
update_from_dict has been giving problems on spills so just add a couple of
simple tests for spills, and movers ordered collection
'''

import copy
from datetime import datetime

import pytest
import numpy as np

from gnome.model import Model
from gnome.spills.spill import point_line_spill
from gnome.movers import SimpleMover
from gnome.movers import RandomMover

from gnome.gnomeobject import GnomeId
from gnome.utilities.orderedcollection import OrderedCollection



l_spills = [point_line_spill(10, (0, 0, 0),
                                     datetime.now().replace(microsecond=0),
                                     name='sp1'),
            point_line_spill(15, (0, 0, 0),
                                     datetime.now().replace(microsecond=0),
                                     name='sp2'),
            point_line_spill(20, (0, 0, 0),
                                     datetime.now().replace(microsecond=0),
                                     name='sp3'),
            point_line_spill(5, (0, 0, 0),
                                     datetime.now().replace(microsecond=0),
                                     name='sp4')]
l_mv = [SimpleMover(velocity=(1, 2, 3)), RandomMover()]


# Some unit tests of the utilities
# _attr_changed tests
@pytest.mark.parametrize('val1, val2', [(3, 3),
                                        ('a string', 'a string'),
                                        ([1, 2, 3, 4], [1, 2, 3, 4]),
                                        (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])),
                                        # OrderedCollection required the same objects
                                        # ints are interned, so this works.
                                        (OrderedCollection([1, 2, 3, 4]),
                                         OrderedCollection([1, 2, 3, 4])),
                                        ])
def test_attr_changed_not(val1, val2):
    assert not GnomeId._attr_changed(val1, val2)


@pytest.mark.parametrize('val1, val2', [(3, 4),
                                        ('a string', 'a different string'),
                                        ([1, 2, 3, 4], [1, 2, 4, 4]),
                                        (np.array([1, 2, 3, 4]), np.array([1, 2, 4, 4])),
                                        (np.array([1, 2, 3, 4]), np.array([1, 2, 3])),
                                        (np.array([1, 2, 3, 4]), np.array([])),
                                        # two different GnomeId object should not be "same"
                                        (GnomeId(), GnomeId()),
                                        # lists are unique, so this should fail
                                        (OrderedCollection([[1.0], [2.0], [3.0], [4.0]]),
                                         OrderedCollection([[1.0], [2.0], [3.0], [4.0]])),
                                        (OrderedCollection([1.0, 2.0, 3.0, 4.0]),
                                         OrderedCollection([1.0, 2.0, 4.0, 4.0])),
                                        # order matters (d'uh!)
                                        (OrderedCollection([1.0, 2.0, 4.0, 3.0]),
                                         OrderedCollection([1.0, 2.0, 3.0, 4.0])),
                                        # length matters (d'uh!)
                                        (OrderedCollection([1.0, 2.0, 3.0]),
                                         OrderedCollection([1.0, 2.0, 3.0, 4.0])),
                                        ])
def test_attr_changed_yes(val1, val2):
    assert GnomeId._attr_changed(val1, val2) is True





def define_mdl(test=0):
    '''
    WebAPI will update/replace nested objects so do that for the test as well
    Setup some test cases:

    0 - empty model w/ no changes
    1 - model with l_mv and l_spills but no changes
    2 - empty model but add l_mv and l_spills to json_ so it changed
    3 - add l_mv and l_spills to model, then delete some elements and update
    via json
    '''
    def get_json(mdl):
        json_ = mdl.serialize()
        return json_

    mdl = Model()

    if test == 0:
        return (mdl, get_json(mdl), False)

    elif test == 1:
        'add a mover and spill but no change'
        mdl.movers += l_mv
        mdl.spills += l_spills
        json_ = get_json(mdl)
        json_['movers'] = [mv.serialize() for mv in l_mv]
        json_['spills'] = [s.serialize() for s in l_spills]
        return (mdl, json_, False)

    elif test == 2:
        'add l_mv, l_spills to empty model'
        json_ = get_json(mdl)
        json_['movers'] = [mv.serialize() for mv in l_mv]
        json_['spills'] = [s.serialize() for s in l_spills]
        return (mdl, json_, True)

    elif test == 3:
        'add l_mv, l_spills to model, then delete some items'
        mdl.movers += l_mv
        mdl.spills += l_spills
        copy_l_mv = copy.deepcopy(l_mv)
        copy_l_spills = copy.deepcopy(l_spills)
        del copy_l_mv[-1]
        del copy_l_spills[-2]
        del copy_l_spills[0]
        json_ = get_json(mdl)
        json_['movers'] = [mv.serialize() for mv in copy_l_mv]
        json_['spills'] = [s.serialize() for s in copy_l_spills]
        return (mdl, json_, True)


@pytest.mark.parametrize("model_num", [0, 1, 2, 3])
def test_spills_update_from_dict(model_num):
    model, json_, exp_updated = define_mdl(model_num)
    updated = model.update_from_dict(json_)
    assert updated is exp_updated
