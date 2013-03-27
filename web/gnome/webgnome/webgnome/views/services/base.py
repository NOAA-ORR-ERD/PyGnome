from cornice import Service


class BaseResource(object):
    optional_fields = []

    def __init__(self, request):
        self.request = request

    @property
    def id(self):
        _id = self.request.matchdict.get('id', None)
        if _id:
            return _id

    @property
    def settings(self):
        return self.request.registry.settings

    def prepare(self, data):
        """
        Prepare ``data``, a dict of data for the resource, by deleting any keys
        that match `self.optional_fields` which are blank.
        """
        for key in self.optional_fields:
            value = data.get(key, None)

            if value is None:
                del data[key]

        return data

