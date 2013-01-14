
class BaseResource(object):
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