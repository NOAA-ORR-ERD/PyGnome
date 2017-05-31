import gridded
from gnome.environment import Environment
import copy
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime
from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.persist import base_schema


class TimeSchema(base_schema.ObjType):
    filename = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())], missing=drop)
    varname = SchemaNode(String(), missing=drop)
    data = SchemaNode(typ=Sequence(), children=[SchemaNode(DateTime(None))], missing=drop)


class GridSchema(base_schema.ObjType):
    filename = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())])


class VariableSchemaBase(base_schema.ObjType):
    name = SchemaNode(String(), missing=drop)
    units = SchemaNode(String(), missing=drop)
    time = TimeSchema(missing=drop)  # SequenceSchema(SchemaNode(DateTime(default_tzinfo=None), missing=drop), missing=drop)


class VariableSchema(VariableSchemaBase):
    varname = SchemaNode(String())
    grid = GridSchema(missing=drop)
    data_file = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())])
    grid_file = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())])


class VectorVariableSchema(VariableSchemaBase):
    varnames = SequenceSchema(SchemaNode(String()))
    grid = GridSchema(missing=drop)
    data_file = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())])
    grid_file = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())])


class Time(gridded.time.Time, serializable.Serializable):

    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = TimeSchema

    _state.add_field([serializable.Field('filename', save=True, update=True, isdatafile=True),
                      serializable.Field('varname', save=True, update=True),
                      serializable.Field('data', save=True, update=True)])


class Grid(gridded.grids.Grid, serializable.Serializable):

    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = GridSchema
    _state.add_field([serializable.Field('filename', save=True, update=True, isdatafile=True)])

    def __new__(cls, *args, **kwargs):
        '''
        If you construct a Grid object directly, you will always
        get one of the child types based on your input
        '''
        if cls is not Grid_U and cls is not Grid_S:
            if 'faces' in kwargs:
                cls = Grid_U
            else:
                cls = Grid_S
        return super(type(cls), cls).__new__(cls)


class Grid_U(gridded.grids.Grid_U):
    def draw_to_plot(self, ax, features=None, style=None):
        import matplotlib
        def_style = {'color': 'blue',
                     'linestyle': 'solid'}
        s = def_style.copy()
        if style is not None:
            s.update(style)
        lines = self.get_lines()
        lines = matplotlib.collections.LineCollection(lines, **s)
        ax.add_collection(lines)


class Grid_S(gridded.grids.Grid_S):
    def draw_to_plot(self, ax, features=None, style=None):
        def_style = {'node': {'color': 'green',
                              'linestyle': 'dashed',
                              'marker': 'o'},
                     'center': {'color': 'blue',
                                'linestyle': 'solid'},
                     'edge1': {'color': 'purple'},
                     'edge2': {'color': 'olive'}}
        if features is None:
            features = ['node']
        st = def_style.copy()
        if style is not None:
            for k in style.keys():
                st[k].update(style[k])
        for f in features:
            s = st[f]
            lon, lat = self._get_grid_vars(f)
            ax.plot(lon, lat, **s)
            ax.plot(lon.T, lat.T, **s)



class Depth(gridded.depth.Depth):
    pass


class Variable(gridded.Variable, serializable.Serializable):
    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = VariableSchema
    _state.add_field([serializable.Field('units', save=True, update=True),
                      serializable.Field('time', save=True, update=True, save_reference=True),
                      serializable.Field('grid', save=True, update=True, save_reference=True),
                      serializable.Field('varname', save=True, update=True),
                      serializable.Field('data_file', save=True, update=True, isdatafile=True),
                      serializable.Field('grid_file', save=True, update=True, isdatafile=True)])

    default_names = []
    cf_names = []

    _default_component_types = copy.deepcopy(gridded.Variable._default_component_types)
    _default_component_types.update({'time': Time,
                                     'grid': Grid,
                                     'depth': Depth})

    @classmethod
    def new_from_dict(cls, dict_):
        if 'data' not in dict_:
            return cls.from_netCDF(**dict_)
        return super(Variable, cls).new_from_dict(dict_)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        self._time = t


class VectorVariable(gridded.VectorVariable, serializable.Serializable):

    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = VectorVariableSchema
    _state.add_field([serializable.Field('units', save=True, update=True),
                      serializable.Field('time', save=True, update=True, save_reference=True),
                      serializable.Field('grid', save=True, update=True, save_reference=True),
                      serializable.Field('variables', save=True, update=True, read=True, iscollection=True),
                      serializable.Field('varnames', save=True, update=True),
                      serializable.Field('data_file', save=True, update=True, isdatafile=True),
                      serializable.Field('grid_file', save=True, update=True, isdatafile=True)])


    _default_component_types = copy.deepcopy(gridded.VectorVariable._default_component_types)
    _default_component_types.update({'time': Time,
                                     'grid': Grid,
                                     'depth': Depth,
                                     'variable': Variable})

    @classmethod
    def new_from_dict(cls, dict_):
        if 'variables' not in dict_:
            if 'varnames' in dict_:
                vn = dict_.get('varnames')
                if 'constant' in vn[-1]:
                    dict_['varnames'] = dict_['varnames'][0:2]
            return cls.from_netCDF(**dict_)
        return super(VectorVariable, cls).new_from_dict(dict_)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        self._time = t
