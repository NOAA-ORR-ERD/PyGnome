"""
model_manager.py: Manage a pool of running models.
"""
import copy
import datetime
import logging
import os
import threading
from gnome.outputters import Renderer

import numpy
np = numpy

from webgnome import util

from gnome.basic_types import world_point_type
from gnome.utilities import projections

from gnome.model import Model
from gnome.map import MapFromBNA, GnomeMap
from gnome.environment import Wind
from gnome.spill import PointLineRelease
from gnome.movers import WindMover, RandomMover, CatsMover, GridCurrentMover


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
    source_types = (('undefined', 'Undefined'),
                    ('manual', 'Manual Data'),
                    ('nws', 'NWS Wind Data'),
                    ('buoy', 'Buoy Station ID'),
                    )

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
    state = copy.deepcopy(WindMover._state)
    state.add(save=['uncertain_angle_scale_units', 'name'],
              update=['uncertain_angle_scale_units', 'name'])

    def __init__(self, *args, **kwargs):
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
    state = copy.deepcopy(RandomMover._state)
    state.add(save=['name'], update=['name'])

    def __init__(self, *args, **kwargs):
        self.on = kwargs.pop('on', True)
        super(WebRandomMover, self).__init__(*args, **kwargs)


class WebCatsMover(BaseWebObject, CatsMover):
    """
    A subclass of :class:`gnome.movers.CatsMover` that provides
    webgnome-specific functionality.
    """
    default_name = 'Cats Mover'
    state = copy.deepcopy(CatsMover._state)
    state.add(save=['name'], update=['name'])

    def __init__(self, base_dir, filename, *args, **kwargs):
        filename = os.path.join(base_dir, filename)
        super(WebCatsMover, self).__init__(filename, *args, **kwargs)


class WebGridCurrentMover(BaseWebObject, GridCurrentMover):
    """
    A subclass of :class:`gnome.movers.GridCurrentMover` that provides
    webgnome-specific functionality.
    """
    default_name = 'Grid Current Mover'
    state = copy.deepcopy(GridCurrentMover._state)
    state.add(save=['name'], update=['name'])

    def __init__(self, base_dir, filename, topology_file, *args, **kwargs):
        filename = os.path.join(base_dir, filename)
        topology_file = os.path.join(base_dir, topology_file)
        super(WebGridCurrentMover, self).__init__(filename, topology_file,
                                                  *args, **kwargs)


class WebPointSourceRelease(BaseWebObject, PointLineRelease):
    """
    A subclass of :class:`gnome.movers.WindMover` that provides
    webgnome-specific functionality.
    """
    default_name = 'Spill'
    state = copy.deepcopy(PointLineRelease._state)
    state.add(save=['name'], update=['name'])

    def __init__(self, *args, **kwargs):
        self.is_active = kwargs.pop('is_active', True)
        super(WebPointSourceRelease, self).__init__(*args, **kwargs)

    def _reshape(self, lst):
        if lst is None:
            return
        return np.asarray(lst, dtype=world_point_type).reshape((len(lst),))

    def start_position_from_dict(self, start_position):
        self.start_position = self._reshape(start_position)

    def end_position_from_dict(self, end_position):
        self.end_position = self._reshape(end_position)

    def start_position_to_dict(self):
        return self.start_position.tolist()

    def end_position_to_dict(self):
        if self.end_position is None:
            return
        return self.end_position.tolist()


class WebMapFromBNA(BaseWebObject, MapFromBNA):
    """
    A subclass of :class:`gnome.map.MapFromBNA` that provides
    webgnome-specific functionality.
    """
    default_name = 'Map'
    state = copy.deepcopy(MapFromBNA._state)
    state.add(save=['name'], update=['name'])

    def __init__(self, base_dir, filename, *args, **kwargs):
        self.base_dir = base_dir
        self.filename = filename
        super(WebMapFromBNA, self).__init__(filename, *args, **kwargs)

    def map_bounds_to_dict(self):
        """
        Map bounds may be a tuple, if it's the default value provided by
        :class:`webgnome.schema.MapSchema`, or it may be a NumPy list,
        in which case we should call the tolist() method to get a list.

        TODO: Should the Schema object handle this? Does it already?
        """
        if self.map_bounds is not None and hasattr(self.map_bounds, 'tolist'):
            return self.map_bounds.tolist()
        return self.map_bounds

    def filename_to_dict(self):
        """
        Create a relative path to the filename by removing its base directory.
        """
        to_remove = len(self.base_dir)

        if not self.base_dir.endswith(os.path.sep):
            to_remove += 1

        return self.filename[to_remove:]


class WebRenderer(BaseWebObject, Renderer):
    """
    A :class:`gnome.renderer.Renderer` subclass with WebGnome-specific helpers.
    """
    @property
    def background_image_path(self):
        return os.path.join(self.images_dir, self.background_map_name)

    def clear_output_dir(self):
        """
        Override parent method to skip clearing the output directory, in case
        the request immediately before this was to run the model, in which
        case the first step is generated.
        """
        pass


class WebGnomeMap(BaseWebObject, GnomeMap):
    default_name = 'Map'
    state = copy.deepcopy(GnomeMap._state)
    state.add(save=['name'], update=['name'])


class WebModel(BaseWebObject, Model):
    """
    A subclass of :class:`gnome.model.Model` that provides webgnome-specific
    functionality.
    """
    mover_keys = {
        WebWindMover: 'wind_movers',
        WebRandomMover: 'random_movers',
        WebCatsMover: 'cats_movers',
        WebGridCurrentMover: 'grid_current_movers'
    }

    spill_keys = {
        WebPointSourceRelease: 'surface_release_spills'
    }

    environment_keys = {
        WebWind: 'winds'
    }

    def __init__(self, *args, **kwargs):
        self.lock = threading.RLock()
        kwargs['cache_enabled'] = kwargs.get('cache_enabled', True)

        data_dir = kwargs.pop('data_dir')
        self.package_root = kwargs.pop('package_root')
        self.renderer = None

        # Model.__init__ ends up calling rewind(), which looks for base_dir to
        # set up the data directory. Set `base_dir` to None to skip those steps
        # during superclass initialization.
        self.base_dir = None

        # Set the model's ID, which we need to set the base_dir.
        super(WebModel, self).__init__(*args, **kwargs)

        # Remove the default map object.
        if self.map:
            self.remove_map()

        self.base_dir = os.path.join(self.package_root, data_dir, str(self.id))
        self.base_dir_relative = os.path.join(data_dir, str(self.id))

        # The static data dir is for things like file uploads that are
        # not bound to a datetime for caching purposes.
        self.static_data_dir = os.path.join(self.base_dir, 'data')

        # Create the base directory for all of the model's data.
        util.mkdir_p(self.base_dir)
        util.mkdir_p(self.static_data_dir)

        # Patch the object with an empty ``time_steps`` array for
        # the time being.
        # TODO: Add output caching in the model?
        self.time_steps = []
        self.changed_at = datetime.datetime.now()

    @property
    def data_dir(self):
        """
        Return the expected path to the files for the current run of the model.

        This path is bound to the current value of ``self.changed_at`` so that
        we can invalidate a browser-based image cache when a model changes.
        """
        if not self.base_dir:
            return

        return os.path.join(self.base_dir,
                            util.get_filename_safe_time(self.changed_at))

    @property
    def duration_hours(self):
        if self.duration.seconds:
            return self.duration.seconds / 60 / 60

    @property
    def changed_at(self):
        """
        The datetime the model was last changed.
        """
        return self._changed_at

    @changed_at.setter
    def changed_at(self, change_time):
        """
        Set the datetime the model was last changed.

        This will change the data directory for images created for the model,
        effectively busting any cached versions of those images.

        This method is called, among other places, in __init__,
        before the model has a base_dir or an ID, so we defend against
        that possibility.

        Note that previous data directories are kept around.

        XXX: Seems bad that updating the `changed_at` field also creates a
        directory, changes a renderer's image directory and writes a background
        image...
        """
        self._changed_at = change_time

        if self.data_dir and not os.path.exists(self.data_dir):
            util.mkdir_p(self.data_dir)

        if self.renderer:
            self.renderer.images_dir = self.data_dir
            # Save a new background image.
            self.renderer.prepare_for_model_run(self._cache)

    def mark_changed(self):
        """
        Update the `self.changed_at` field to the current time.
        """
        self.changed_at = datetime.datetime.now()

    def add_renderer(self, renderer):
        """
        Add ``renderer`` to the collection of outputters and keep a reference
        to it in `self.renderer`.
        """
        self.renderer = renderer
        self.outputters += self.renderer

    def remove_renderer(self):
        """
        Remove the model's current renderer -- removes the reference in `self`
        and removes the outputter from the model's `outputters` collection.
        """
        if self.renderer:
            self.outputters.remove(self.renderer.id)
            self.renderer = None

    def add_bna_map(self, filename, map_data):
        """
        Adds a BNA map that exists at ``filename``, a path relative to the
        webgnome package directory.

        This might be a map file in a location file or in a running model's
        data directory.

        Creates a renderer using the map file, with a reference stored as
        `self.renderer`.
        """
        map_file = os.path.join(self.package_root, filename)
        # Create the land-water map
        self.map = WebMapFromBNA(self.package_root, map_file, **map_data)
        self.setup_renderer()

    def setup_renderer(self):
        """
        Add a renderer for the model.

        If one already exists, it will be removed.

        This method updates the model's `changed_at` value and writes the
        background image for the current map.
        """
        if not self.map:
            raise ValueError('Cannot setup a renderer '
                             'if the model lacks a map')

        if self.renderer:
            self.remove_renderer()

        # TODO: Should size be configurable?
        self.add_renderer(
            WebRenderer(self.map.filename, self.data_dir,
                        image_size=(800, 600),
                        projection_class=projections.GeoProjection,
                        cache=self._cache))

        # XXX: mark_changed() updates the data_dir and the renderer's
        # images_dir, and renders a new background image, so it should happen
        # after we create the renderer.
        self.mark_changed()

    def remove_map(self):
        self.map = None
        self.remove_renderer()
        self.output_map = None
        self.rewind()

    def rewind(self):
        """
        Rewind the model.

        Reset WebGnome-specific caches, set the background image name with a
        cache-buster value and call Model.rewind().
        """
        self.mark_changed()
        self.timestamps = self.get_timestamps()
        self.time_steps = []

        return super(WebModel, self).rewind()

    def get_timestamps(self):
        """
        Get the expected timestamps for a model run.
        """
        timestamps = []

        # XXX: Why is _num_time_steps a float? Is this ok?
        for step_num in range(int(self._num_time_steps)):
            if step_num == 0:
                dt = self.start_time
            else:
                delta = datetime.timedelta(seconds=step_num * self.time_step)
                dt = self.start_time + delta
            timestamps.append(dt)

        return timestamps

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

            obj_data = obj.to_dict()

            data[key].append(obj_data)

        return data

    def to_dict(self,
                include_spills=True,
                include_movers=True,
                include_wind=True):
        """
        Return a dictionary representation of this model. Includes subtrees
        (lists of dictionaries) for any movers and spills configured.
        """
        data = {'uncertain': self.uncertain,
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

        if include_wind:
            data = self.build_subtree(data, self.environment,
                                      self.environment_keys)

        if include_movers:
            data = self.build_subtree(data, self.movers, self.mover_keys)

        if include_spills:
            data = self.build_subtree(data, self.spills, self.spill_keys)

        return data

    def from_dict(self, data):
        """
        Set fields on this model from the dict ``data``.
        """
        self.uncertain = data['uncertain']
        self.start_time = data['start_time']
        self.time_step = data['time_step'] * 60 * 60
        self.duration = datetime.timedelta(
            days=data['duration_days'],
            seconds=data['duration_hours'] * 60 * 60)

        map_data = data.get('map', None)
        winds = data.get('winds', None)
        wind_movers = data.get('wind_movers', None)
        cats_movers = data.get('cats_movers', None)
        grid_current_movers = data.get('grid_current_movers', None)
        random_movers = data.get('random_movers', None)
        surface_spills = data.get('surface_release_spills', None)

        if map_data:
            # Ignore map bounds - will be set from the source file.
            map_data.pop('map_bounds', None)
            # Ignore obj_type - only used for serialization. TODO: Better way?
            map_data.pop('obj_type', None)
            # Make the filename, which is stored as a relative path, absolute
            filename = os.path.join(self.package_root,
                                    map_data.pop('filename'))
            self.add_bna_map(filename, map_data)

        def add_to_collection(collection, data, cls):
            obj = cls(**data)
            collection.add(obj)

        if winds:
            for wind in winds:
                add_to_collection(self.environment, wind, WebWind)

        if wind_movers:
            for mover_data in wind_movers:
                mover_data['wind'] = self.environment.get(mover_data['wind_id'])
                add_to_collection(self.movers, mover_data, WebWindMover)

        if cats_movers:
            for mover_data in cats_movers:
                mover_data['base_dir'] = self.package_root
                add_to_collection(self.movers, mover_data, WebCatsMover)

        if grid_current_movers:
            for mover_data in grid_current_movers:
                mover_data['base_dir'] = self.package_root
                add_to_collection(self.movers, mover_data, WebGridCurrentMover)

        if random_movers:
            for mover_data in random_movers:
                add_to_collection(self.movers, mover_data, WebRandomMover)

        if surface_spills:
            for spill_data in surface_spills:
                add_to_collection(self.spills, spill_data,
                                  WebPointSourceRelease)

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

        Using a model_id that does not exist in running_models is a no-op.
        """
        self.running_models.pop(model_id, None)

    def exists(self, model_id):
        """
        Return True of a model exists in `running_models with the ID
        ``model_id``, False if not.
        """
        return model_id in self.running_models
