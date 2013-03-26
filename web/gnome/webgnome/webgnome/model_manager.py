"""
model_manager.py: Manage a pool of running models.
"""
import copy
import datetime
import logging
import os
import uuid
import numpy

from hazpy.file_tools import haz_files
from webgnome import util

# XXX: This except block should not be necessary.
try:
    import gnome
except ImportError:
    print 'Import error! Could not find gnome library.'
    # If we failed to find the model package,
    # it could be that we are running webgnome
    # from an in-place virtualenv.  Let's add
    # a relative path to our py_gnome sources
    # and see if we can import the Model.
    # If we fail in this attempt...oh well.
    import sys
    import gnome
    sys.path.append('../../../py_gnome')

import gnome.utilities.map_canvas
from gnome import basic_types
from gnome.model import Model
from gnome.movers import WindMover, RandomMover
from gnome.spill import SurfaceReleaseSpill
from gnome.environment import Wind
from gnome.map import MapFromBNA, GnomeMap


logger = logging.getLogger(__name__)


class BaseWebObject(object):
    """
    All WebGnome subclasses of Gnome objects have a name.
    """
    default_name = 'Item'

    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', self.default_name)
        super(BaseWebObject, self).__init__(*args, **kwargs)

    @property
    def name(self):
        if self._name:
            return self._name
        return super(BaseWebObject, self).__repr__()

    @name.setter
    def name(self, name):
        self._name = name


class WebWind(BaseWebObject, Wind):
    default_name = 'Wind'
    source_types = (
        ('undefined', 'Undefined'),
        ('manual', 'Manual Data'),
        ('nws', 'NWS Wind Data'),
        ('buoy', 'Buoy Station ID'),
    )
    state = copy.deepcopy(Wind.state)
    state.add(create=['id'])

    def __init__(self, *args, **kwargs):
        self.description = kwargs.pop('description', None)
        self.source_type = kwargs.pop('source_type', None)
        self.source_id = kwargs.pop('source_id', None)
        self.longitude = kwargs.pop('longitude', None)
        self.latitude = kwargs.pop('latitude', None)
        self.updated_at = kwargs.pop('updated_at', None)

        super(WebWind, self).__init__(*args, **kwargs)

    @property
    def timeseries(self):
        return self.get_timeseries(units=self.units)

    @timeseries.setter
    def timeseries(self, value):
        self.set_timeseries(value, units=self.units)


class WebWindMover(BaseWebObject, WindMover):
    """
    A subclass of :class:`gnome.movers.WindMover` that provides
    webgnome-specific functionality.
    """
    default_name = 'Wind Mover'
    state = copy.deepcopy(WindMover.state)
    state.add(create=['uncertain_angle_scale_units', 'name'],
              update=['uncertain_angle_scale_units', 'name'])

    def __init__(self, *args, **kwargs):
        self.is_constant = kwargs.pop('is_constant', True)
        self.on = kwargs.pop('on', True)

        # TODO: What to do with this value? Conversion?
        self.uncertain_angle_scale_units = kwargs.pop(
            'uncertain_angle_scale_units', None)

        super(WebWindMover, self).__init__(*args, **kwargs)


class WebRandomMover(BaseWebObject, RandomMover):
    """
    A subclass of :class:`gnome.movers.RandomMover` that provides
    webgnome-specific functionality.
    """
    default_name = 'Random Mover'
    state = copy.deepcopy(RandomMover.state)
    state.add(create=['name'], update=['name'])

    def __init__(self, *args, **kwargs):
        self.on = kwargs.pop('on', True)
        super(WebRandomMover, self).__init__(*args, **kwargs)


class WebSurfaceReleaseSpill(BaseWebObject, SurfaceReleaseSpill):
    """
    A subclass of :class:`gnome.movers.WindMover` that provides
    webgnome-specific functionality.
    """
    default_name = 'Spill'
    state = copy.deepcopy(SurfaceReleaseSpill.state)
    state.add(create=['name'], update=['name'])

    def __init__(self, *args, **kwargs):
        self.is_active = kwargs.pop('is_active', True)
        super(WebSurfaceReleaseSpill, self).__init__(*args, **kwargs)

    def _reshape(self, lst):
        return numpy.asarray(
            lst, dtype=basic_types.world_point_type).reshape((len(lst),))

    def start_position_from_dict(self, start_position):
        self.start_position = self._reshape(start_position)

    def end_position_from_dict(self, end_position):
        self.end_position = self._reshape(end_position)

    def start_position_to_dict(self):
        return self.start_position.tolist()

    def end_position_to_dict(self):
        return self.end_position.tolist()


class WebMapFromBNA(BaseWebObject, MapFromBNA):
    """
    A subclass of :class:`gnome.map.MapFromBNA` that provides
    webgnome-specific functionality.
    """
    default_name = 'Map'
    state = copy.deepcopy(MapFromBNA.state)
    state.add(create=['name', 'relative_path'],
              update=['name', 'relative_path'])

    # TODO: Better way to remove from all lists?
    state.update.remove('filename')
    state.create.remove('filename')

    def __init__(self, *args, **kwargs):
        self.relative_path = kwargs.pop('relative_path', None)
        super(WebMapFromBNA, self).__init__(*args, **kwargs)

    def map_bounds_to_dict(self):
        """
        Map bounds may be a tuple, if it's the default value provided by
        :class:`webgnome.schema.MapSchema`, or it may be a NumPy list,
        in which case we should call the tolist() method to get a list.
        """
        if self.map_bounds is not None and hasattr(self.map_bounds, 'tolist'):
            return self.map_bounds.tolist()
        return self.map_bounds


class WebGnomeMap(BaseWebObject, GnomeMap):
    default_name = 'Map'
    state = copy.deepcopy(GnomeMap.state)
    state.add(create=['name'], update=['name'])


class WebModel(BaseWebObject, Model):
    """
    A subclass of :class:`gnome.model.Model` that provides webgnome-specific
    functionality.
    """
    mover_keys = {
        WebWindMover: 'wind_movers',
        WebRandomMover: 'random_movers',
        # gnome.movers.CatsMover: 'cats_movers'
    }

    spill_keys = {
        WebSurfaceReleaseSpill: 'surface_release_spills'
    }

    def __init__(self, *args, **kwargs):
        data_dir = kwargs.pop('data_dir')
        _map = kwargs.pop('map', None)
        self.package_root = kwargs.pop('package_root')

        # Set the model's id
        super(WebModel, self).__init__()

        if _map is None:
            self.map = WebGnomeMap()

        self.base_dir = os.path.join(self.package_root, data_dir, str(self.id))
        self.base_dir_relative = os.path.join(data_dir, str(self.id))
        self.static_data_dir = os.path.join(self.base_dir, 'data')

        # Create the base directory for all of the model's data.
        util.mkdir_p(self.base_dir)
        util.mkdir_p(self.static_data_dir)

        # Patch the object with an empty ``time_steps`` array for the time being.
        # TODO: Add output caching in the model?
        self.time_steps = []
        self.runtime = None
        self.background_image = None

    @property
    def data_dir(self):
        """
        Return the expected path to the files for the current run of the model.
        """
        if not self.runtime:
            return

        return os.path.join(self.base_dir, self.runtime)

    @property
    def background_image_path(self):
        if not self.background_image:
            return

        return os.path.join(self.base_dir, self.background_image)

    @property
    def duration_hours(self):
        if self.duration.seconds:
            return self.duration.seconds / 60 / 60

    def add_bna_map(self, filename, map_data):
        """
        Adds a BNA map that exists at ``filename``, a path relative to the
        webgnome package directory.

        This might be a map file in a location file or in a running model's
        data directory.

        Creates the land-water map and the canvas, and saves the background
        image for the map.
        """
        map_file = os.path.join(self.package_root, filename)

        # Create the land-water map
        self.map = WebMapFromBNA(map_file, relative_path=filename, **map_data)

        # TODO: Should size be user-configurable?
        canvas = gnome.utilities.map_canvas.MapCanvas((800, 600))
        polygons = haz_files.ReadBNA(map_file, "PolygonSet")
        canvas.set_land(polygons)
        self.output_map = canvas

        # Delete an existing background image file.
        if self.background_image and os.path.exists(self.background_image_path):
            try:
                os.remove(self.background_image_path)
            except OSError as e:
                logger.error('Could not delete file: %s. Error was: %s' % (
                    self.background_image, e))

        # Save the background image.
        self.background_image = 'background_image_%s.png' % util.get_runtime()
        self.output_map.draw_background()
        self.output_map.save_background(self.background_image_path)

    def remove_map(self):
        self.map = None
        self.background_image = None
        self.output_map = None
        self.rewind()

    def build_subtree(self, data, objs, keys):
        """
        Build a subtree of dicts for spills and movers on the model.

        Traverses ``objs``, a list of objects, and calls ``to_dict`` on each
        one if the method is available. The dict representation of the object
        is added to the ``data`` dict using the key found in ``keys`` for the
        object's class.
        """
        for key in keys.values():
            if key not in data:
                data[key] = []

        for obj in objs:
            key = keys.get(obj.__class__, None)

            if not key or not hasattr(obj, 'to_dict'):
                continue

            if key not in data:
                data[key] = []

            obj_data = obj.to_dict(do='create')

            # XXX: Temporary hack to fix new Serializable code
            if hasattr(obj, 'wind'):
                obj_data['wind'] = obj.wind.to_dict(do='create')

            data[key].append(obj_data)

        return data
    
    def to_dict(self, include_spills=True, include_movers=True):
        """
        Return a dictionary representation of this model. Includes subtrees
        (lists of dictionaries) for any movers and spills configured.
        """
        data = {
            'uncertain': self.uncertain,
            'time_step': (self.time_step / 60.0) / 60.0,
            'start_time': self.start_time,
            'duration_days': 0,
            'duration_hours': 0,
            'id': self.id,
        }

        if self.map and hasattr(self.map, 'to_dict'):
            data['map'] = self.map.to_dict('create')

        if self.duration.days:
            data['duration_days'] = self.duration.days
    
        if self.duration_hours:
            data['duration_hours'] = self.duration_hours

        if include_movers:
            data = self.build_subtree(data, self.movers, self.mover_keys)

        if include_spills:
            data = self.build_subtree(data, self.spills, self.spill_keys)

        return data

    def from_dict(self, data):
        """
        Set fields on this model from the dict ``data``.

        Note: does not set movers or spills.
        """
        def add_mover(data, cls):
            _id = data.pop('id', None)
            data = cls(**data)
            if _id:
                data._id = uuid.UUID(_id)
            self.movers.add(data)

        def add_spill(data, cls):
            _id = data.pop('id', None)
            spill = cls(**data)
            # TODO: Why doesn't this work anymore?
            # if _id:
            #     spill._id = uuid.UUID(_id)
            self.spills.add(spill)

        self.uncertain = data['uncertain']
        self.start_time = data['start_time']
        self.time_step = data['time_step'] * 60 * 60
        self.duration = datetime.timedelta(
            days=data['duration_days'],
            seconds=data['duration_hours'] * 60 * 60)

        map_data = data.get('map', None)
        wind_movers = data.get('wind_movers', None)
        random_movers = data.get('random_movers', None)
        surface_spills = data.get('surface_release_spills', None)

        if map_data:
            relative_path = map_data.pop('relative_path')
            # Ignore map bounds and filename - will be set from the source file.
            map_data.pop('map_bounds', None)
            map_data.pop('filename', None)
            self.add_bna_map(relative_path, map_data)

        if wind_movers:
            for mover_data in wind_movers:
                mover_data['wind'] = WebWind(**mover_data.pop('wind'))
                add_mover(mover_data, WebWindMover)

        if random_movers:
            for mover_data in random_movers:
                add_mover(mover_data, WebRandomMover)

        if surface_spills:
            for spill_data in surface_spills:
                add_spill(spill_data, WebSurfaceReleaseSpill)

        return self


class ModelManager(object):
    """
    An object that manages a pool of in-memory :class:`gnome.model.Model`
    instances in a dictionary.
    """
    class DoesNotExist(Exception):
        pass

    def __init__(self, data_dir, package_root):
        self.data_dir = data_dir
        self.package_root = package_root
        self.running_models = {}

    def create(self, **kwargs):
        """
        Create a new :class:`WebModel`, adds it to the `running_models` dict
        and returns the new object.
        """
        if 'package_root' not in kwargs:
            kwargs['package_root'] = self.package_root
        if 'data_dir' not in kwargs:
            kwargs['data_dir'] = self.data_dir

        model = WebModel(**kwargs)
        self.running_models[str(model.id)] = model
        return model

    def get_or_create(self, model_id, **kwargs):
        """
        Get a running :class:`WebModel` instance if one exists with the ID
        ``model_id``. Otherwise, create a new model and return it.

        Return a tuple of the signature (model, created) where ``model`` is
        the model object and ``created`` is a boolean signifying whether the
        object was created or not.
        """
        created = False
        model = None

        if model_id:
            model = self.running_models.get(str(model_id), None)

        if model is None:
            model = self.create(**kwargs)
            created = True

        return model, created

    def get(self, model_id):
        """
        Return a model if one exists in `running_models` with the ID
        ``model_id``, else raises :class:`ModelManager.DoesNotExist`.
        """
        model_id = str(model_id)
        if not model_id in self.running_models:
            raise self.DoesNotExist
        return self.running_models.get(model_id)

    def delete(self, model_id):
        """
        Delete the model whose ID matches ``model_id``.

        Using a ``model_id`` that does not exist in `running_models` is a no-op.
        """
        self.running_models.pop(model_id, None)

    def exists(self, model_id):
        """
        Return True of a model exists in `running_models with the ID
        ``model_id``, False if not.
        """
        return model_id in self.running_models
