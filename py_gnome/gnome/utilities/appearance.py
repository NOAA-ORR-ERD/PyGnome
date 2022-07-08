'''
Models to hold web client appearance state information.
'''

from gnome.persist.base_schema import ObjTypeSchema, ObjType
from gnome.gnomeobject import GnomeId
from colander import drop


class AppearanceSchema(ObjTypeSchema):
    def __init__(self, unknown='preserve', *args, **kwargs):
        super(AppearanceSchema, self).__init__(*args, **kwargs)
        self.typ = ObjType(unknown)


class ColormapSchema(AppearanceSchema):
    pass

class SpillAppearanceSchema(AppearanceSchema):
    colormap = ColormapSchema(test_equal=False, missing=drop)
'''
class Appearance(object):
    __metaclass__ = GnomeObjMeta
    _schema = AppearanceSchema

    def __init__(self, **kwargs):
        self.id = str(uuid1())
        if 'name' in kwargs:
            self._name = kwargs.pop('name')
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.appearance_keys = kwargs.keys()

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            self._name = '{}_{}'.format(self.__class__.__name__.split('.')[-1],
                                        str(self.__class__._instance_count))

            return self._name

    @name.setter
    def name(self, v):
        self._name = v

    @property
    def obj_type(self):
        try:
            obj_type = ('{0.__module__}.{0.__class__.__name__}'.format(self))
        except AttributeError:
            obj_type = '{0.__class__.__name__}'.format(self)

        return obj_type

    @classmethod
    def new_from_dict(cls, dict_):
        read_only_attrs = cls._schema().get_nodes_by_attr('read_only')

        [dict_.pop(n, None) for n in read_only_attrs]
        new_obj = cls(**dict_)
        return new_obj

    def to_dict(self, json_=None):
        data = {}
        for k in self.appearance_keys:
            data[k] = getattr(self, k)
        data['id'] = self.id
        data['obj_type'] = self.obj_type
        data['name'] = self.name
        return data

    def update_from_dict(self, dict_, refs=None):
        read_only_attrs = self._schema().get_nodes_by_attr('read_only')

        [dict_.pop(n, None) for n in read_only_attrs]
        self.appearance_keys = dict_.keys()
        for k, v in dict_.items():
            setattr(self, k, v)
        return True

    def serialize(self, *args, **kwargs):
        return self.to_dict()

    @classmethod
    def deserialize(cls, json_, refs=None):
        return cls.new_from_dict(json_)
'''
class Appearance(GnomeId):
    _schema = AppearanceSchema
    def __init__(self, **kwargs):
        keys = Appearance._schema().get_nodes_by_attr('all')
        k2 = dict([(key, kwargs.get(key)) for key in keys])
        read_only_attrs = Appearance._schema().get_nodes_by_attr('read_only')
        for n in read_only_attrs:
            k2.pop(n, None)
        super(Appearance, self).__init__(**k2)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.appearance_keys = list(kwargs.keys())

    def update_from_dict(self, dict_, refs=None):
        super(Appearance, self).update_from_dict(dict_, refs=refs)
        updatable = self._schema().get_nodes_by_attr('update')
        read_only_attrs = self._schema().get_nodes_by_attr('read_only')
        for name in updatable + read_only_attrs:
            dict_.pop(name)
        for k, v in dict_.items():
            setattr(self, k, v)

    def to_dict(self, json_=None):
        data = super(Appearance, self).to_dict(json_=json_)
        for k in self.appearance_keys:
            data[k] = getattr(self, k)
        return data



class Colormap(Appearance):
    _schema = ColormapSchema

class SpillAppearance(Appearance):
    _schema = SpillAppearanceSchema
    def __init__(self, colormap=None, **kwargs):
        self.colormap = colormap
        super(SpillAppearance, self).__init__(**kwargs)

class MapAppearance(Appearance):
    _schema = AppearanceSchema

class VectorAppearance(Appearance):
    _schema = AppearanceSchema

class GridAppearance(Appearance):
    _schema = AppearanceSchema

class MoverAppearance(Appearance):
    _schema = AppearanceSchema

class PolygonReleaseSchema(Appearance):
    _schema = AppearanceSchema