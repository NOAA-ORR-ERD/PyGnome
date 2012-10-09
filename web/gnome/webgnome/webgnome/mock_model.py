import glob
import datetime
import os
import uuid


class ModelManager(object):
    """
    An object that manages a pool of in-memory `py_gnome.model.Model` instances
    in a dictionary.
    """
    def __init__(self):
        self.running_models = {}

    def get_or_create(self, model_id):
        """
        Return a running `py_gnome.model.Model` instance if the user has a valid
        `model_id` key in his or her session. Otherwise, create a new model and
        return it.
        """
        model = None
        created = False

        if model_id:
            model = self.running_models.get(model_id, None)

        if model is None:
            model = MockModel()
            self.running_models[model.id] = model
            created = True

        return model, created

    def get(self, model_id):
        model = self.running_models.get(model_id, None)
        return model

    def add(self, model_id, model):
        self.running_models[model_id] = model

    def remove(self, model_id):
        if model_id in self.running_models:
            self.running_models.pop(model_id)

    def exists(self, model_id):
        exists = False
        if model_id:
            exists = model_id in self.running_models
        return exists


class MockModel(object):
    """
    A mock stand-in for `py_gnome.model.Model`.
    """
    def __init__(self):
        self.id = uuid.uuid4()

    def get_movers(self):
        return []

    def get_settings(self):
        return [
            {'name': 'ID', 'value': self.id}
        ]

    def get_map(self):
        return {'name': 'My map'}

    def get_spills(self):
        return []

    def run(self):
        frames_glob = os.path.join(
            os.path.dirname(__file__), 'static', 'img', 'test_frames', '*.jpg')
        images = glob.glob(frames_glob)

        # Mock out some timestamps until we accept this input from the user.
        two_weeks_ago = datetime.datetime.now() - datetime.timedelta(weeks=4)

        timestamps = [two_weeks_ago + datetime.timedelta(days=day_num)
                      for day_num in range(len(images))]

        return [
            dict(url=image.split('webgnome')[-1], timestamp=timestamps[i])
            for i, image in enumerate(images)]
