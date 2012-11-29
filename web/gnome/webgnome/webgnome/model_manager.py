"""
model_manager.py: Manage a pool of running models.
"""
import util

# XXX: This except block should not be necessary.
try:
    from gnome.model import Model
    from gnome.movers import WindMover
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
    def __init__(self, date, speed, speed_type, direction_degrees):
        self.date = date
        self.hour = date.hour
        self.minute = date.minute
        self.speed = speed
        self.speed_type = speed_type
        # XXX: Hard-coded value. Can't use a constant here from WindForm due
        # to a circular import: form.movers -> model_manager -> forms.movers
        self.direction = 'Degrees true'
        self.direction_degrees = direction_degrees


class WindMoverProxy(util.Proxy):
    """
    A proxy for :class:`gnome.movers.WindMover` that provides webgnome-specific
    functionality.
    """
    def __init__(self, *args, **kwargs):
        self.name = None

        super(WindMoverProxy, self).__init__(*args, **kwargs)

    @property
    def timeseries(self):
        series = []

        for timeseries in self._target.timeseries:
            dt = timeseries[0].astype(object)
            series.append(
                Wind(date=dt, speed=timeseries[1][0], speed_type='meters',
                    direction_degrees=timeseries[1][1])
            )

        return series

    @timeseries.setter
    def timeseries(self, datetime_value_2d):
        """
        Wrap the ``timeseries`` property setter, as :class:`util.Proxy` doesn't
        forward property setters.

        XXX: The name of this variable seems out of date.
        """
        self._target.timeseries = datetime_value_2d

    def __repr__(self):
        """
        This method doesn't forward because :class:`WindMoverProxy` inherits
        a new ``__repr__`` method from :class:`util.Proxy` due to it deriving
        from ``object``.
        """
        if self.name:
            return self.name
        return self._target.__repr__()


class ModelProxy(util.Proxy):
    """
    A proxy for :class:`gnome.model.Model` that provides webgnome-specific
    functionality.
    """
    def __init__(self, *args, **kwargs):
        # Patch the object with an empty ``time_steps`` array for the time being.
        # TODO: Add output caching in the model.
        self.time_steps = []
        super(ModelProxy, self).__init__(*args, **kwargs)

    def has_mover_with_id(self, mover_id):
        """
        Return True if the model has a mover with the ID ``mover_id``.

        TODO: The manager patches :class:`gnome.model.Model` with this method,
        but the method should belong to that class.
        """
        return int(mover_id) in self._target._movers

    def has_spill_with_id(self, spill_id):
        """
        Return True if the model has a spill with the ID ``spill_id``.

        TODO: The manager patches :class:`gnome.model.Model` with this method,
        but the method should belong to that class.
        """
        return int(spill_id) in self._target._spills


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
        proxy = ModelProxy(Model())
        self.running_models[proxy.id] = proxy
        return proxy

    def get_or_create(self, model_id):
        """
        Return a running :class:`ModelProxy` instance if the user has a
        valid ``model_id`` key in his or her session. Otherwise, create a new
        model and return it.
        """
        proxy = None
        created = False

        if model_id:
            proxy = self.running_models.get(model_id, None)

        if proxy is None:
            proxy = self.create()
            created = True

        return proxy, created

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
