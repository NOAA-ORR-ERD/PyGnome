
from gnome.spills.spill import Spill
from gnome.utilities.appearance import SpillAppearance, Colormap


def test_spill_appearance_serialization():
    sp = Spill()
    cm = Colormap(k1='v1')
    sa_attribs = {'foo': 'bar', 'baz': 'bin', 'colormap': cm}
    sp._appearance = SpillAppearance(**sa_attribs)
    assert sp._appearance.foo == 'bar'
    assert sp._appearance.colormap.k1 == 'v1'
    ser = sp.serialize()
    assert ser['_appearance']['foo'] == 'bar'
    sp2 = Spill.deserialize(ser)
    assert sp == sp2
    assert sp._appearance == sp2._appearance
    assert sp2._appearance.foo == 'bar'
    assert sp._appearance.colormap == sp2._appearance.colormap
    assert sp2._appearance.colormap.k1 == 'v1'


def test_spill_appearance_save_load():
    sp = Spill()
    cm = Colormap(k1='v1')
    sa_attribs = {'foo': 'bar', 'baz': 'bin', 'colormap': cm}
    sp._appearance = SpillAppearance(**sa_attribs)

    json_, saveloc, refs = sp._appearance.save(None)
    assert json_['foo'] == 'bar'
    assert '.json' in json_['colormap']

