"""
model_manager.py: Manage a pool of running models.
"""
try:
    from gnome.model import Model
except:
    # If we failed to find the model package,
    # it could be that we are running webgnome
    # from an in-place virtualenv.  Let's add
    # a relative path to our py_gnome sources
    # and see if we can import the Model.
    # If we fail in this attempt...oh well.
    import sys
    sys.path.append('../../../py_gnome')
    from gnome.model import Model


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
        model = Model()

        # Patch the object with an empty ``time_steps`` array for the time being.
        # TODO: Add output caching in the model.
        model.time_steps = []

        def has_mover_with_id(model, mover_id):
            """
            Return True if the model has a mover with the ID ``mover_id``.

            TODO: The manager patches :class:`gnome.model.Model` with this method,
            but the method should belong to that class.
            """
            return int(mover_id) in model._movers

        def get_mover(model, mover_id):
            return model._movers.get(int(mover_id))

        def has_spill_with_id(model, spill_id):
            """
            Return True if the model has a spill with the ID ``spill_id``.

            TODO: The manager patches :class:`gnome.model.Model` with this method,
            but the method should belong to that class.
            """
            return int(spill_id) in model._spills

        def get_spill(model, spill_id):
            return model._spills.get(int(spill_id))

        setattr(model.__class__, 'has_mover_with_id', has_mover_with_id)
        setattr(model.__class__, 'has_spill_with_id', has_spill_with_id)
        setattr(model.__class__, 'get_mover', get_mover)
        setattr(model.__class__, 'get_spill', get_spill)

        self.running_models[model.id] = model
        return model

    def get_or_create(self, model_id):
        """
        Return a running :class:`gnome.model.Model` instance if the user has a
        valid ``model_id`` key in his or her session. Otherwise, create a new
        model and return it.
        """
        model = None
        created = False

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
