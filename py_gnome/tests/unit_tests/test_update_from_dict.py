'''
update_from_dict has been giving problems on spills so just add a couple of
simple tests for spills, and movers orderedcollection
'''
import copy
from datetime import datetime

import pytest

from gnome.model import Model
from gnome.spill import point_line_release_spill
from gnome.movers import SimpleMover
from gnome.movers import RandomMover


l_spills = [point_line_release_spill(10, (0, 0, 0),
                                     datetime.now().replace(microsecond=0),
                                     name='sp1'),
            point_line_release_spill(15, (0, 0, 0),
                                     datetime.now().replace(microsecond=0),
                                     name='sp2'),
            point_line_release_spill(20, (0, 0, 0),
                                     datetime.now().replace(microsecond=0),
                                     name='sp3'),
            point_line_release_spill(5, (0, 0, 0),
                                     datetime.now().replace(microsecond=0),
                                     name='sp4')]
l_mv = [SimpleMover(velocity=(1, 2, 3)), RandomMover()]


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
        json_ = Model.deserialize(mdl.serialize('webapi'))
        json_['map'] = mdl.map
        return json_

    mdl = Model()

    if test == 0:
        return (mdl, get_json(mdl), False)

    elif test == 1:
        'add a mover and spill but no change'
        mdl.movers += l_mv
        mdl.spills += l_spills
        json_ = get_json(mdl)
        json_['movers'] = l_mv
        json_['spills'] = l_spills
        return (mdl, json_, False)

    elif test == 2:
        'add l_mv, l_spills to empty model'
        json_ = get_json(mdl)
        json_['movers'] = l_mv
        json_['spills'] = l_spills
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
        json_['movers'] = copy_l_mv
        json_['spills'] = copy_l_spills
        return (mdl, json_, True)


@pytest.mark.parametrize(("model", "json_", "exp_updated"),
                         [define_mdl(0),
                          define_mdl(1),
                          define_mdl(2),
                          define_mdl(3)])
def test_spills_update_from_dict(model, json_, exp_updated):
    updated = model.update_from_dict(json_)
    assert updated is exp_updated
