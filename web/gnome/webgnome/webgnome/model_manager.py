"""
model_manager.py: Manage a pool of running models.
"""
import datetime

# XXX: This except block should not be necessary.
import os
from hazpy.file_tools import haz_files
from webgnome import util

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
    sys.path.append('../../../py_gnome')

import gnome.utilities.map_canvas
from gnome.model import Model
from gnome.movers import WindMover
from gnome.spill import PointReleaseSpill
from gnome.map import MapFromBNA


class WebWindMover(WindMover):
    """
    A subclass of :class:`gnome.movers.WindMover` that provides
    webgnome-specific functionality.
    """
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', 'Wind Mover')
        self.is_constant = kwargs.pop('is_constant', True)

        # TODO: What to do with this value? Conversion?
        self.uncertain_angle_scale_units = kwargs.pop(
            'uncertain_angle_scale_units', None)

        super(WebWindMover, self).__init__(*args, **kwargs)

    def from_dict(self, data):
        self.wind = data['wind']
        self._name = data['name']
        self.is_active = data['is_active']
        self.uncertain_duration = data['uncertain_duration']
        self.uncertain_speed_scale = data['uncertain_speed_scale']
        self.uncertain_angle_scale = data['uncertain_angle_scale']
        self.uncertain_time_delay = data['uncertain_time_delay']

        # TODO: What to do here?
        self.uncertain_angle_scale_units = 'deg'

        return self

    def to_dict(self):
        series = []

        for timeseries in self.wind.get_timeseries(units=self.wind.user_units):
            dt = timeseries[0].astype(object)
            series.append(
                dict(datetime=dt, speed=timeseries[1][0],
                          direction=timeseries[1][1])
            )

        return {
            'wind': {
                'timeseries': series,
                'units': self.wind.user_units
            },

            'id': self.id,
            'name': self.name,
            'is_active': self.is_active,
            'uncertain_duration': self.uncertain_duration,
            'uncertain_time_delay': self.uncertain_time_delay,
            'uncertain_speed_scale': self.uncertain_speed_scale,
            'uncertain_angle_scale': self.uncertain_angle_scale,

            # XXX: Does WindMover always return the angle scale in degrees?
            'uncertain_angle_scale_units': self.uncertain_angle_scale_units
        }

    @property
    def name(self):
        if self._name:
            return self._name
        return super(WebWindMover, self).__repr__()


class WebPointReleaseSpill(PointReleaseSpill):
    """
    A subclass of :class:`gnome.movers.WindMover` that provides
    webgnome-specific functionality.
    """
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', None)
        self.is_active = kwargs.pop('is_active', True)
        super(WebPointReleaseSpill, self).__init__(*args, **kwargs)

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

    @property
    def name(self):
        if self._name:
            return self._name
        return super(WebPointReleaseSpill, self).__repr__()

    def from_dict(self, data):
        self.release_time = data['release_time']
        self.start_position = data['start_position']
        self.windage_range = data['windage']
        self._name = data['name']
        self.is_uncertain = data['uncertain']
        self.num_LEs = data['num_LEs']

        return self

    def to_dict(self):
        return {
            'id': self.id,
            'release_time': self.release_time,
            'start_position': self.start_position,
            'windage': self.windage_range,
            'name': self._name,
            'uncertain': self.is_uncertain,
            'num_LEs': self.num_LEs
        }


class WebMapFromBNA(MapFromBNA):
    """
    A subclass of :class:`gnome.map.MapFromBNA` that provides
    webgnome-specific functionality.
    """
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', None)
        self.filename = args[0]
        super(WebMapFromBNA, self).__init__(*args, **kwargs)

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return id(self)

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'name': self.name,
            'refloat_halflife': self.refloat_halflife
        }


class WebModel(Model):
    """
    A subclass of :class:`gnome.model.Model` that provides webgnome-specific
    functionality.
    """
    mover_keys = {
        WebWindMover: 'wind_movers'
    }

    spill_keys = {
        WebPointReleaseSpill: 'point_release_spills'
    }

    def __init__(self, *args, **kwargs):
        self.base_dir = self._make_base_dir(kwargs.pop('model_images_dir'))

        super(WebModel, self).__init__()

        # Patch the object with an empty ``time_steps`` array for the time being.
        # TODO: Add output caching in the model.
        self.time_steps = []
        self.runtime = None

    def _make_base_dir(self, dir):
        base_dir = os.path.join(dir, str(self.id))

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        return base_dir

    @property
    def data_dir(self):
        if not self.runtime:
            self.runtime = util.get_runtime()

        data_dir = os.path.join(self.base_dir, self.runtime)

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        return data_dir

    @property
    def background_image(self):
        if not self.output_map:
            return

        image_path = os.path.join(self.data_dir, 'background_map.png')

        if not os.path.exists(image_path):
            self.output_map.draw_background()
            self.output_map.save_background(image_path)

        return image_path

    @property
    def duration_hours(self):
        if self.duration.seconds:
            return self.duration.seconds / 60 / 60

    def has_mover_with_id(self, mover_id):
        """
        Return True if the model has a mover with the ID ``mover_id``.

        TODO: The manager patches :class:`gnome.model.Model` with this method,
        but the method should belong to that class.
        """
        return int(mover_id) in self._movers

    def has_spill_with_id(self, spill_id):
        """
        Return True if the model has a spill with the ID ``spill_id``.

        TODO: The manager patches :class:`gnome.model.Model` with this method,
        but the method should belong to that class.
        """
        return int(spill_id) in self.spills

    def add_bna_map(self, filename, map_data):
        map_file = os.path.join(self.data_dir, filename)

        # Create the land-water map
        self.map = WebMapFromBNA(map_file, **map_data)

        # TODO: Should size be user-configurable?
        canvas = gnome.utilities.map_canvas.MapCanvas((800, 600))
        polygons = haz_files.ReadBNA(filename, "PolygonSet")
        canvas.set_land(polygons)
        self.output_map = canvas

    def build_subtree(self, data, objs, keys):
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
        data = {
            'uncertain': self.uncertain,
            'time_step': self.time_step,
            'start_time': self.start_time,
            'duration_days': 0,
            'duration_hours': 0,
            'map': self.map.to_dict() if self.map else None
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
        self.uncertain = data['uncertain']
        self.start_time = data['start_time']
        self.time_step = data['time_step']
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
        model = WebModel(**kwargs)
        self.running_models[model.id] = model
        return model

    def get_or_create(self, model_id, **kwargs):
        """
        Return a running :class:`WebModel` instance if the user has a
        valid ``model_id`` key in his or her session. Otherwise, create a new
        model and return it.
        """
        created = False
        model = None

        if model_id:
            model = self.running_models.get(model_id, None)

        if model is None:
            model = self.create(**kwargs)
            created = True

        return model, created

    def get(self, model_id):
        if not model_id in self.running_models:
            raise self.DoesNotExist
        return self.running_models.get(model_id)

    def add(self, model_id, model):
        self.running_models[model_id] = model

    def delete(self, model_id):
        self.running_models.pop(model_id, None)

    def exists(self, model_id):
        return model_id in self.running_models
