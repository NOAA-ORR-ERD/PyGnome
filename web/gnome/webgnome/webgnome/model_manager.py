"""
model_manager.py: Manage a pool of running models.
"""
import datetime

import os
from hazpy.file_tools import haz_files
import numpy
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
from gnome.weather import Wind
from gnome.map import MapFromBNA


class Serializable(object):
    serializable_fields = []

    def to_dict(self):
        """
        Return a dictionary containing the serialized representation of this
        object, using `self.serializable_fields` to look up fields on the object
        that the dictionary should contain.

        For every field, if there is a method defined on the object such that
        the method name is `{field_name}_to_dict`, use the return value of that
        method as the field value.

        Note: any field in `self.serializable_fields` that does not exist on the
        object and does not have a to_dict method will raise an AttributeError.
        """
        data = {}

        for key in self.serializable_fields:
            to_dict_fn_name = '%s_to_dict' % key

            if hasattr(self, to_dict_fn_name):
                value = getattr(self, to_dict_fn_name)()
            else:
                value = getattr(self, key)

            if hasattr(value, 'to_dict'):
                value = value.to_dict()

            data[key] = value

        return data

    def from_dict(self, data):
        """
        Set the state of this object using the dictionary ``data`` by looking up
        the value of each key in ``data`` that is also in
        `self.serializable_fields`.

        For every field, the choice of how to set the field is as follows:

        If there is a method defined on the object such that the method name is
        `{field_name}_from_dict`, call that method with the field's data.

        If the field on the object has a ``from_dict`` method, then use that
        method instead.

        If neither method exists, then set the field with the value from
        ``data`` directly on the object.
        """
        for key in self.serializable_fields:
            if not key in data:
                continue

            from_dict_fn_name = '%s_from_dict' % key
            field = getattr(self, key)
            value = data[key]

            if hasattr(self, from_dict_fn_name):
                getattr(self, from_dict_fn_name)(value)
            elif hasattr(field, 'from_dict'):
                field.from_dict(value)
            else:
                setattr(self, key, value)

        return self


class BaseWebObject(Serializable):
    @property
    def name(self):
        if self._name:
            return self._name
        return super(BaseWebObject, self).__repr__()

    @name.setter
    def name(self, name):
        self._name = name


class WebWind(Wind, BaseWebObject):
    default_name = 'Wind'
    source_types = (
        ('manual', 'Manual Data'),
        ('nws', 'NWS Wind Data'),
        ('buoy', 'Buoy Station ID'),
    )
    serializable_fields = [
        'id',
        'units', # set the units before timeseries
        'timeseries',
        'latitude',
        'longitude',
        'description',
        'source',
        'source_type',
        'updated_at'
    ]

    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name', 'Wind')
        self.description = kwargs.pop('description', None)
        self.source_type = kwargs.pop('source_type', None)
        self.source = kwargs.pop('source', None)
        self.longitude = kwargs.pop('longitude', None)
        self.latitude = kwargs.pop('latitude', None)
        self.updated_at = kwargs.pop('updated_at', None)

        super(WebWind, self).__init__(*args, **kwargs)

    @property
    def units(self):
        return self._user_units

    @units.setter
    def units(self, value):
        self._user_units = value

    @property
    def timeseries(self):
        return self.get_timeseries(units=self.user_units)

    @timeseries.setter
    def timeseries(self, value):
        self.set_timeseries(value, units=self.user_units)

    def timeseries_to_dict(self):
        series = []

        for wind_value in self.timeseries:
            dt = wind_value[0].astype(object)
            series.append(
                dict(datetime=dt, speed=wind_value[1][0],
                          direction=wind_value[1][1])
            )

        return series


class WebWindMover(WindMover, BaseWebObject):
    """
    A subclass of :class:`gnome.movers.WindMover` that provides
    webgnome-specific functionality.
    """
    default_name = 'Wind Mover'
    serializable_fields = [
        'id',
        'wind',
        'on',
        'name',
        'active_start',
        'active_stop',
        'uncertain_duration',
        'uncertain_speed_scale',
        'uncertain_angle_scale',
        'uncertain_angle_scale_units',
        'uncertain_time_delay'
    ]

    def __init__(self, *args, **kwargs):
        self.is_constant = kwargs.pop('is_constant', True)
        self.on = kwargs.pop('on', True)
        self.name = kwargs.pop('name', 'Wind Mover')

        # TODO: What to do with this value? Conversion?
        self.uncertain_angle_scale_units = kwargs.pop(
            'uncertain_angle_scale_units', None)

        super(WebWindMover, self).__init__(*args, **kwargs)


class WebRandomMover(RandomMover, BaseWebObject):
    """
    A subclass of :class:`gnome.movers.RandomMover` that provides
    webgnome-specific functionality.
    """
    default_name = 'Random Mover'
    serializable_fields = [
        'id',
        'on',
        'name',
        'active_start',
        'active_stop',
        'diffusion_coef'
    ]

    def __init__(self, *args, **kwargs):
        self.on = kwargs.pop('on', True)
        self.name = kwargs.pop('name', 'Random Mover')
        super(WebRandomMover, self).__init__(*args, **kwargs)


class WebSurfaceReleaseSpill(SurfaceReleaseSpill, BaseWebObject):
    """
    A subclass of :class:`gnome.movers.WindMover` that provides
    webgnome-specific functionality.
    """
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', None)
        self.is_active = kwargs.pop('is_active', True)
        super(WebSurfaceReleaseSpill, self).__init__(*args, **kwargs)

    serializable_fields = [
        'id',
        'release_time',
        'start_position',
        'windage_range',
        'windage_persist',
        'name',
        'num_elements',
        'is_active'
    ]

    @property
    def hour(self):
        return self.release_time.hour

    @property
    def minute(self):
        return self.release_time.minute

    @property
    def start_position_x(self):
        return self.start_position[0]

    @property
    def start_position_y(self):
        return self.start_position[1]

    @property
    def start_position_z(self):
        return self.start_position[2]

    @property
    def windage_min(self):
        return self.windage_range[0]

    @property
    def windage_max(self):
        return self.windage_range[1]

    def start_position_from_dict(self, start_position):
        self.start_position = numpy.asarray(
            start_position,
            dtype=basic_types.world_point_type).reshape((3,))

    def start_position_to_dict(self):
        return self.start_position.tolist()


class WebMapFromBNA(MapFromBNA, BaseWebObject):
    """
    A subclass of :class:`gnome.map.MapFromBNA` that provides
    webgnome-specific functionality.
    """
    serializable_fields = [
        'id',
        'filename',
        'name',
        'map_bounds',
        'refloat_halflife'
    ]

    def map_bounds_to_dict(self):
        return self.map_bounds.tolist()

    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name', None)
        self.filename = args[0]
        super(WebMapFromBNA, self).__init__(*args, **kwargs)


class WebModel(Model, BaseWebObject):
    """
    A subclass of :class:`gnome.model.Model` that provides webgnome-specific
    functionality.

    TODO: Use Serializable mixin's to_dict and from_dict mechanism.
    """
    mover_keys = {
        WebWindMover: 'wind_movers',
        WebRandomMover: 'random_movers'
    }

    spill_keys = {
        WebSurfaceReleaseSpill: 'surface_release_spills'
    }

    def __init__(self, *args, **kwargs):
        # Create the base directory for all of the model's data.
        self.base_dir = os.path.join(kwargs.pop('model_images_dir'),
                                     str(self.id))
        util.mkdir_p(self.base_dir)

        super(WebModel, self).__init__()

        # Patch the object with an empty ``time_steps`` array for the time being.
        # TODO: Add output caching in the model?
        self.time_steps = []
        self.runtime = None

    @property
    def data_dir(self):
        """
        Return the expected path to the files for the current run of the model.
        """
        if not self.runtime:
            return

        return os.path.join(self.base_dir, self.runtime)

    @property
    def background_image(self):
        """
        Return the path to the file containing the background image for the
        current map.
        """
        if not self.output_map:
            return

        return os.path.join(self.base_dir, 'background_map.png')

    @property
    def duration_hours(self):
        if self.duration.seconds:
            return self.duration.seconds / 60 / 60

    def add_bna_map(self, filename, map_data):
        """
        Add a BNA map that exists at ``filename``, a path relative to the base
        directory for the model.

        Creates the land-water map and the canvas, and saves the background
        image for the map.
        """
        map_file = os.path.join(self.base_dir, filename)

        # Create the land-water map
        self.map = WebMapFromBNA(map_file, **map_data)

        # TODO: Should size be user-configurable?
        canvas = gnome.utilities.map_canvas.MapCanvas((800, 600))
        polygons = haz_files.ReadBNA(filename, "PolygonSet")
        canvas.set_land(polygons)
        self.output_map = canvas

        # Save the background image.
        self.output_map.draw_background()
        self.output_map.save_background(self.background_image)

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

            data[key].append(obj.to_dict())

        return data
    
    def to_dict(self, include_spills=True, include_movers=True):
        """
        Return a dictionary representation of this model. Includes subtrees
        (lists of dictionaries) for any movers and spills configured.
        """
        if self.map and hasattr(self.map, 'to_dict'):
            _map = self.map.to_dict()
        else:
            _map = None

        data = {
            'is_uncertain': self.is_uncertain,
            'time_step': (self.time_step / 60.0) / 60.0,
            'start_time': self.start_time,
            'duration_days': 0,
            'duration_hours': 0,
            'id': self.id,
            'map': _map
        }

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
        self.is_uncertain = data['is_uncertain']
        self.start_time = data['start_time']
        self.time_step = data['time_step'] * 60 * 60
        self.duration = datetime.timedelta(
            days=data['duration_days'],
            seconds=data['duration_hours'] * 60 * 60)


class ModelManager(object):
    """
    An object that manages a pool of in-memory :class:`gnome.model.Model`
    instances in a dictionary.
    """
    class DoesNotExist(Exception):
        pass

    def __init__(self):
        self.running_models = {}

    def create(self, **kwargs):
        """
        Create a new :class:`WebModel`, adds it to the `running_models` dict
        and returns the new object.
        """
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
