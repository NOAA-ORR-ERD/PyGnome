from webgnome import util


class BaseResource(object):
    def __init__(self, request):
        self.request = request
        self.model = util.get_model_from_request(request)

    @property
    def id(self):
        _id = self.request.matchdict.get('id', None)
        if _id:
            return int(_id)

    @property
    def settings(self):
        return self.request.registry.settings