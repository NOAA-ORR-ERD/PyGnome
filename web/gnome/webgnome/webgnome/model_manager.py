"""
model_manager.py: Manage a pool of running models.
"""
import util

# XXX: This except block should not be necessary.
try:
    from gnome.model import Model
    from gnome.movers import WindMover
    from gnome.spill import PointReleaseSpill
except ImportError:
    print 'Import error!'
    # If we failed to find the model package,
    # it could be that we are running webgnome
    # from an in-place virtualenv.  Let's add
    # a relative path to our py_gnome sources
    # and see if we can import the Model.
    # If we fail in this attempt...oh well.
    import sys
    sys.path.append('../../../py_gnome')
    from gnome.model import Model
    from gnome.movers import WindMover


class Wind(object):
    """
    An object that represents a single wind value in a wind time series, using
    object fields so that it can be used to instantiate a form field.
    """
    def __init__(self, date, speed, speed_type, direction):
        self.date = date
        self.hour = date.hour
        self.minute = date.minute
        self.speed = speed
        self.speed_type = speed_type
        self.direction = direction


class WebWindMover(WindMover):
    """
    A subclass of :class:`gnome.movers.WindMover` that provides
    webgnome-specific functionality.
    """
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', 'Wind Mover')
        self.is_constant = kwargs.pop('is_constant', True)
        super(WebWindMover, self).__init__(*args, **kwargs)

    @property
    def timeseries(self):
        series = []

        for timeseries in super(WebWindMover, self).timeseries:
            dt = timeseries[0].astype(object)
            series.append(
                Wind(date=dt, speed=timeseries[1][0], speed_type='meters',
                     direction=timeseries[1][1])
            )

        return series

    @timeseries.setter
    def timeseries(self, value):
        return WindMover.timeseries.__set__(self, value)

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
        self._name = kwargs.pop('name', 'Spill')
        super(WebPointReleaseSpill, self).__init__(*args, **kwargs)

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



class WebModel(Model):
    """
    A subclass of :class:`gnome.model.Model` that provides webgnome-specific
    functionality.
    """
    def __init__(self, *args, **kwargs):
        super(WebModel, self).__init__()

        # Patch the object with an empty ``time_steps`` array for the time being.
        # TODO: Add output caching in the model.
        self.time_steps = []

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
        return int(spill_id) in self._spills


class ModelManager(object):
    """
    An object that manages a pool of in-memory :class:`gnome.model.Model`
    instances in a dictionary.
    """
    class DoesNotExist(Exception):
        pass

    def __init__(self):
        self.running_models = {}

    def create(self):
        model = WebModel()
        self.running_models[model.id] = model
        return model

    def get_or_create(self, model_id):
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
            model = self.create()
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
