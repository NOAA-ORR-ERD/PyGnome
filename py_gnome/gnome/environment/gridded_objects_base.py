import gridded
import datetime
import StringIO
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

    @classmethod
    def from_file(cls, filename=None, **kwargs):
        if isinstance(filename, list):
            filename = filename[0]
        fn = open(filename, 'r')
        t = []
        for l in fn:
            l = l.rstrip()
            if l is not None:
                t.append(datetime.datetime.strptime(l, '%c'))
        fn.close()
        return Time(t)

    def save(self, saveloc, references=None, name=None):
        '''
        Write Wind timeseries to file or to zip,
        then call save method using super
        '''
#         if self.filename is None:
#             self.filename = self.id + '_time.txt'
#             if zipfile.is_zipfile(saveloc):
#                 self._write_time_to_zip(saveloc, self.filename)
#             else:
#                 datafile = os.path.join(saveloc, self.filename)
#                 self._write_time_to_file(datafile)
#             rv = super(Time, self).save(saveloc, references, name)
#             self.filename = None
#         else:
#             rv = super(Time, self).save(saveloc, references, name)
#         return rv
        super(Time, self).save(saveloc, references, name)

    def _write_time_to_zip(self, saveloc, ts_name):
        '''
        use a StringIO type of file descriptor and write directly to zipfile
        '''
        fd = StringIO.StringIO()
        self._write_time_to_fd(fd)
        self._write_to_zip(saveloc, ts_name, fd.getvalue())

    def _write_time_to_file(self, datafile):
        '''write timeseries data to file '''
        with open(datafile, 'w') as fd:
            self._write_time_to_fd(fd)

    def _write_time_to_fd(self, fd):
        for t in self.time:
            fd.write(t.strftime('%c') + '\n')

    @classmethod
    def new_from_dict(cls, dict_):
        if 'varname' not in dict_:
            dict_['time'] = dict_['data']
#             if 'filename' not in dict_:
#                 raise ValueError
            return cls(**dict_)
        else:
            return cls.from_netCDF(**dict_)


class Grid_U(gridded.grids.Grid_U, serializable.Serializable):

    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = GridSchema
    _state.add_field([serializable.Field('filename', save=True, update=True, isdatafile=True)])

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

    @classmethod
    def new_from_dict(cls, dict_):
        dict_.pop('json_')
        filename = dict_['filename']
        rv = cls.from_netCDF(filename)
        rv.__class__._restore_attr_from_save(rv, dict_)
        rv._id = dict_.pop('id') if 'id' in dict_ else rv.id
        rv.__class__._def_count -= 1
        return rv

        def get_cells(self):
            return self.nodes[self.faces]

class Grid_S(gridded.grids.Grid_S, serializable.Serializable):

    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = GridSchema
    _state.add_field([serializable.Field('filename', save=True, update=True, isdatafile=True)])

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

    @classmethod
    def new_from_dict(cls, dict_):
        dict_.pop('json_')
        filename = dict_['filename']
        rv = cls.from_netCDF(filename)
        rv.__class__._restore_attr_from_save(rv, dict_)
        rv._id = dict_.pop('id') if 'id' in dict_ else rv.id
        rv.__class__._def_count -= 1
        return rv

    def get_cells(self):
        if not hasattr(self, '_cell_trees'):
            self.build_celltree()
        n = self._cell_trees['node'][1]
        f = self._cell_trees['node'][2]
        return n[f]


class Grid(gridded.grids.Grid):

    @staticmethod
    def from_netCDF(*args, **kwargs):
        kwargs['_default_types'] = (('ugrid', Grid_U), ('sgrid', Grid_S))
        return gridded.grids.Grid.from_netCDF(*args, **kwargs)



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
