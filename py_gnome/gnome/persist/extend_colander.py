'''
Extend colander's basic types for serialization/deserialization
of gnome specific types
'''

import datetime
import os
import ujson
import zipfile

import numpy as np

from colander import (Float, DateTime, Sequence, Tuple, List, SchemaType,
                      TupleSchema, SequenceSchema, null, SchemaNode, String, Invalid, MappingSchema, Mapping)

import gnome.basic_types
from gnome.utilities import inf_datetime, round_sf_array

import pdb

import xarray
import netCDF4

from collections import namedtuple

LoadSpec = namedtuple('LoadSpec', ['typ', 'pth', 'varname'])

class DataSchemaNode(SchemaNode):
    schema_type = Mapping
    """
    A SchemaType that can serialize and deserialize various data types including:
    - numpy.ndarray
    - numpy.ma.MaskedArray
    - xarray.DataArray
    - netCDF4.Variable
    - list, tuple
    - int, float
    """
    def serialize(self, appstruct):
        typ = None
        pth = None
        varname = self.name
        if isinstance(appstruct, np.ma.MaskedArray):
            typ = 'maskedarray'
            pth = '??'
        elif isinstance(appstruct, np.ndarray):
            typ = 'ndarray'
            pth = '??'
        elif isinstance(appstruct, xarray.DataArray):
            if 'source' in appstruct.encoding:
                #note the filename is the full path. This needs to be trimmed before use in a save file.
                typ = 'xarray'
                pth = appstruct.encoding['source']
                varname = appstruct.name
            else:
                #in-memory DataArray
                raise NotImplementedError("In-memory xarray.DataArray serialization is not implemented. Please save to a file first.")
        elif isinstance(appstruct, netCDF4.Variable):
            if appstruct._grp.filepath() is None:
                #in-memory netCDF4.Variable
                raise NotImplementedError("In-memory netCDF4.Variable serialization is not implemented. Please save to a file first.")
            if isinstance(appstruct._grp, netCDF4.MFDataset):
                #note the filename is the full path. This needs to be trimmed before use in a save file.
                typ = 'netCDF4.MFDataset'
                pth = appstruct._grp.filepath()
            else:
                typ = 'netCDF4.Dataset'
                pth = appstruct._grp.filepath()
            varname = appstruct.name
        else:
            a = np.array(appstruct)
            if a.dtype == object:
                raise ValueError("objects or object arrays are not supported for serialization by DataSchemaNode")
            ujson.dumps(a)
            return appstruct
        return LoadSpec(typ, pth, varname)

    def _save(self, appstruct, zipfile_, refs):
        """
        For files that already exist on disk, we just need to add them to the zipfile and return the path.
        It is a little more complicated for in-memory data.
        """
        typ, fn, vname = self.serialize(appstruct)
        if fn == '??' and typ not in ['ndarray', 'maskedarray']:
            raise NotImplementedError("In-memory netCDF/Xarray saving is not implemented. Please save to a file first.")
        if typ in ['ndarray', 'maskedarray']:
            # put the data back in because this needs to be added to the zipfile at a higher level (Gnome Obj)
            p_fn = appstruct
        elif typ in ['xarray', 'netCDF4.Dataset', 'netCDF4.MFDataset']:
            p_fn = self._process_supporting_file(fn, zipfile_)
            fn = p_fn
        else:
            raise ValueError(f"Unsupported type for saving: {typ}")
        return LoadSpec(typ, p_fn, vname)

    def deserialize(self, node, cstruct):
        pdb.set_trace()
        if isinstance(cstruct, (int, float, list, tuple)):
            return cstruct
        typ, fn, vname = cstruct
        lookup = {
            'ndarray': lambda: np.load(fn, allow_pickle=False)[vname],
            'maskedarray': lambda: np.ma.MaskedArray(data=np.load(fn, allow_pickle=False)[vname], mask=np.load(fn, allow_pickle=False)[vname + '_mask']),
            'xarray': lambda: xarray.open_dataset(fn)[vname],
            'netCDF4.Dataset': lambda: netCDF4.Dataset(fn)[vname],
            'netCDF4.MFDataset': lambda: netCDF4.MFDataset(fn)[vname],
        }
        return lookup[typ]()
    
    def load(self, cstruct, saveloc, refs):
        #saveloc will be the directory where the zipfile was extracted to, or the zipfile itself if it is still open.
        #This function should return the actual data object we want to rehydrate. (eg, a numpy array, xarray.DataArray, etc.)
        if isinstance(cstruct, list) and len(cstruct) == 3 \
            and isinstance(cstruct[0], str) \
            and cstruct[0] in ['ndarray', 'maskedarray', 'xarray', 'netCDF4.Dataset', 'netCDF4.MFDataset']:
            typ, fn, vname = cstruct
            
            temp_fn = self._load_supporting_file(fn, saveloc)
            ds_obj = v_obj = None
            if typ+':'+temp_fn in refs:
                #Dataset already loaded so don't do it again.
                #append typ because theoretically the same file could be referenced
                #by different types (nc.Dataset, xr.Dataset) in different places.
                ds_obj = refs[typ+':'+temp_fn]
                print ('found ref for {0} in refs'.format(typ+':'+temp_fn))
            elif typ == 'ndarray':
                ds_obj = np.load(temp_fn, allow_pickle=False)
            elif typ == 'maskedarray':
                ds_obj = np.load(temp_fn, allow_pickle=False)
            elif typ == 'xarray':
                ds_obj = xarray.open_dataset(temp_fn)
            elif typ == 'netCDF4.Dataset':
                ds_obj = netCDF4.Dataset(temp_fn)
            elif typ == 'netCDF4.MFDataset':
                ds_obj = netCDF4.MFDataset(temp_fn)
            else:
                raise ValueError(f"Unsupported type for loading: {typ}")
            
            if typ == 'maskedarray':
                v_obj = np.ma.MaskedArray(data=ds_obj[vname], mask=ds_obj[vname + '_mask'])
            else:
                v_obj = ds_obj[vname]

            if typ+':'+temp_fn not in refs:
                refs[typ+':'+temp_fn] = ds_obj

            return v_obj
        else:
            return cstruct

    def _process_supporting_file(self, raw_path, zipfile_):
        '''
        raw_path is the filename stored on the object
        and path to the file on disk.
        zipfile_ is an open zipfile.ZipFile in append mode
        returns the name of the file in the archive
        '''
        d_fname = os.path.split(raw_path)[1]
        # add datafile to zip archive

        if d_fname not in zipfile_.namelist():
            zipfile_.write(raw_path, d_fname)

        return d_fname

    def _load_supporting_file(self, filename, saveloc):
        '''
        filename is the name of the file in the zip
        saveloc can be a folder or open zipfile.ZipFile object
        if saveloc is a folder and the filename exists inside,
        this does not return an altered name, nor does it extract to the
        temporary directory.  An altered filename is returned if it cannot
        find the filename directly or if saveloc is an open zipfile in a
        temporary directory
        '''
        if filename is None:
            return
        if isinstance(saveloc, zipfile.ZipFile):
            dirname = os.path.dirname(saveloc.fp.name)

            # Keep an eye on this. It may cause previously existing files
            # to be recognized incorrectly as what's in the zip since it's a
            # simple existence check
            if not os.path.exists(os.path.join(dirname, filename)):
                saveloc.extract(filename, dirname)
                return os.path.join(dirname, filename)
            else:
                return os.path.join(dirname, filename)
        elif os.path.exists(os.path.join(saveloc, filename)):
            return os.path.join(saveloc, filename)
        elif os.path.exists(filename):
            return filename
        else:
            return filename

class UnknownMappingSchema(MappingSchema):
    # identical to MappingSchema except it preserves unknown entries
    # This is useful for serializing *simple* dicts (meaning no custom types)
    def schema_type(self, **kw):
        return Mapping(unknown='preserve')

class LocalDateTime(DateTime):
    def __init__(self, *args, **kwargs):
        kwargs['default_tzinfo'] = kwargs.get('default_tzinfo', None)
        super(LocalDateTime, self).__init__(*args, **kwargs)

    def strip_timezone(self, _datetime):
        if (_datetime and isinstance(_datetime, (datetime.datetime,
                                                 datetime.date))):
            _datetime = _datetime.replace(tzinfo=None)

        return _datetime

    def serialize(self, node, appstruct):
        """
        Serialize a DateTime object

        returns an iso formatted string
        """
        if isinstance(appstruct, datetime.datetime):
            appstruct = self.strip_timezone(appstruct)

            return super(LocalDateTime, self).serialize(node, appstruct)
        elif isinstance(appstruct, (inf_datetime.InfTime,
                                    inf_datetime.MinusInfTime)):
            return appstruct.isoformat()

    def deserialize(self, node, cstruct):
        if cstruct in ('inf', '-inf'):
            return inf_datetime.InfDateTime(cstruct)
        else:
            dt = super(LocalDateTime, self).deserialize(node, cstruct)

            return self.strip_timezone(dt)


class DefaultTuple(Tuple):
    """
    A Tuple subclass that provides defaults from child nodes.

    Required because Tuple returns `colander.null` by default
    when ``appstruct`` is not provided, instead of creating a Tuple of
    default values.
    """
    def serialize(self, node, appstruct):
        items = super(DefaultTuple, self).serialize(node, appstruct)

        if items is null and node.children:
            items = tuple([field.default for field in node.children])

        return items


class NumpyFixedLen(Tuple):
    """
    A subclass of :class:`colander.Tuple` that converts itself to a Tuple and
    back to a numpy array. This is used to define schemas for Numpy arrays that
    have a fixed size like WorldPoint, 3D velocity of SimpleMover.
    """
    def serialize(self, node, appstruct):
        """
        Serialize a fixed length numpy array
        """
        if appstruct is null:  # colander.null
            return null

        return super(NumpyFixedLen, self).serialize(node, appstruct.tolist())

    def deserialize(self, node, cstruct):
        if cstruct is null:
            return null

        return np.array(cstruct, dtype=np.float64)


class NumpyArray(List):
    """
    A subclass of :class:`colander.List` that converts itself to a more general
    numpy array of greater than length 1.
    """
    def serialize(self, node, appstruct):
        """
        Serialize a numpy array
        """
        if appstruct is null:  # colander.null
            return null

        return super(NumpyArray, self).serialize(node, np.array(appstruct).tolist())

    def deserialize(self, node, cstruct):
        if cstruct is null:
            return null

        return np.array(cstruct, dtype=np.float64)


class NumpyFixedLenSchema(TupleSchema):
    schema_type = NumpyFixedLen


class WorldPointNumpy(NumpyFixedLenSchema):
    '''
    Define same schema as WorldPoint; however, the base class
    NumpyFixedLenSchema serializes/deserializes it from/to a numpy array
    '''
    long = SchemaNode(Float())
    lat = SchemaNode(Float())
    z = SchemaNode(Float())

class DatetimeValue2dArray(Sequence):
    """
    A subclass of :class:`colander.Sequence` that converts itself to a numpy
    array using :class:`gnome.basic_types.datetime_value_2d` as the data type.

    todo: serialize/deserialize must happen for each element - not very
        efficient.
    """
    def serialize(self, node, appstruct):
        """
        Serialize a 2D Datetime value array
        """
        if appstruct is null:  # colander.null
            return null

        # getting serialized by PyGnome so data should be correct
        # is the list() call required? Can we pass a iterable
        # into serialize?
        series = list(zip(appstruct['time'].astype(object),
                     appstruct['value'].tolist()))

        return super(DatetimeValue2dArray, self).serialize(node, series)

    def deserialize(self, node, cstruct):
        if cstruct is null:
            return null

        items = (super(DatetimeValue2dArray, self)
                 .deserialize(node, cstruct, accept_scalar=False))
        timeseries = np.array(items, dtype=gnome.basic_types.datetime_value_2d)

        return timeseries  # validator requires numpy array


class DatetimeValue1dArray(Sequence):
    """
    A subclass of :class:`colander.Sequence` that converts itself to a numpy
    array using :class:`gnome.basic_types.datetime_value_2d` as the data type.
    """
    def serialize(self, node, appstruct):
        if appstruct is null:  # colander.null
            return null

        appstruct = list(zip(appstruct['time'].astype(object), appstruct['value']))

        return super(DatetimeValue1dArray, self).serialize(node, appstruct)

    def deserialize(self, node, cstruct):
        if cstruct is null:
            return null

        items = (super(DatetimeValue1dArray, self)
                 .deserialize(node, cstruct, accept_scalar=False))

        timeseries = np.array(items, dtype=gnome.basic_types.datetime_value_1d)

        return timeseries  # validator requires numpy array

class NullableString(String):
    def serialize(self, node, appstruct):
        if appstruct is None:
            return str(None)
        return super(NullableString, self).serialize(node, appstruct)

    def deserialize(self, node, cstruct):
        if cstruct == str(None):
            return None
        return super(NullableString, self).deserialize(node, cstruct)


class TimeDelta(Float):
    """
    Add a type to serialize/deserialize timedelta objects
    """
    def serialize(self, node, appstruct):
        if appstruct is not null:
            return super(TimeDelta, self).serialize(node,
                                                    appstruct.total_seconds())
        else:
            return super(TimeDelta, self).serialize(node, null)

    def deserialize(self, *args, **kwargs):
        sec = super(TimeDelta, self).deserialize(*args, **kwargs)

        if sec is not null:
            return datetime.timedelta(seconds=sec)
        else:
            return sec


class OrderedCollectionType(Sequence):
    # identical to SequenceSchema except it can tolerate a 'get'
    def _validate(self, node, value, accept_scalar):
        if (hasattr(value, '__iter__') and
            not isinstance(value, str)):
            return list(value)
        if accept_scalar:
            return [value]
        else:
            raise Invalid(node, '{0} is not iterable'.format(value))


"""
Following define new schemas for above custom types. This is so
serialize/deserialize is called correctly.

Specifically a new DefaultTypeSchema and a DatetimeValue2dArraySchema
"""

class FilenameSchema(SequenceSchema):
    def __init__(self, *args, **kwargs):
        kwargs['typ'] = Sequence(accept_scalar=True)
        super(FilenameSchema, self).__init__(SchemaNode(String()), *args, **kwargs)

    def serialize(self, appstruct, options=None):
        rv = super(FilenameSchema, self).serialize(appstruct)
        if rv and options is not None:
            if not options.get('raw_paths', True):
                for i, filename in enumerate(rv):
                    rv[i] = os.path.split(filename)[1]
        if rv and len(rv) == 1:
            return rv[0]
        return rv

    def deserialize(self, cstruct):
        """
        Deserialize a file name
        """
        if cstruct is None or cstruct is null:
            return None
        rv = super(FilenameSchema, self).deserialize(cstruct)
        if len(rv) == 1:
            return rv[0]
        else:
            return rv

'''
np_array = NumpyArraySchema(
    Float(), save=True
)
'''

class NumpyArraySchema(SchemaNode):
    '''
    This schema cannot nest any further schemas inside since it does not follow
    Colander convention for serializing and deserializing.

    It will serialize a numpy array to nested lists of lists of numbers, using
    array.tolist(). It will attempt to convert the array to the type specified
    with the precision specified before doing so.

    It deserializes lists of lists of numbers into a numpy array of dtype
    specified with dtype specified, if at all.

    :param dtype: numpy data type to (de-)serialize to/from
    '''

    def __init__(self, *args, **kwargs):
        # fixme: where/how is this used?? -- why not a class attribute?
        #        and shouldn't it be ndarray, or ???
        self.typ = np.float64
        self.dtype = kwargs.pop('dtype', np.float64)
        self.precision = kwargs.pop('precision', 8)

    def serialize(self, appstruct):
        """
        Serialize a numpy array

        returns data as a list
        """
        if not isinstance(appstruct, (np.ndarray, list, tuple)):
            raise ValueError('Cannot serialize: {0} is not a numpy array, list, or tuple'.format(appstruct))

        return round_sf_array(appstruct, self.precision).astype(self.dtype, copy=False).tolist()

    def deserialize(self, cstruct):
        """
        Deserialize a numpy array

        returns a numpy array from a list
        """
        return np.array(cstruct, dtype=self.dtype)


class OrderedCollectionSchema(SequenceSchema):
    schema_type = OrderedCollectionType


class DefaultTupleSchema(TupleSchema):
    schema_type = DefaultTuple


class DatetimeValue2dArraySchema(SequenceSchema):
    schema_type = DatetimeValue2dArray


class DatetimeValue1dArraySchema(SequenceSchema):
    schema_type = DatetimeValue1dArray
