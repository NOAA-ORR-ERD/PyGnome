from lxml import etree
from pyramid.httpexceptions import HTTPServerError
from cornice.resource import resource, view
import requests
from webgnome import util
from webgnome.views.services.base import BaseResource


@resource(path='/nws/wind', renderer='gnome_json',
          description='National Weather Service wind data.')
class Wind(BaseResource):
    def error(self, status, message):
        """
        Create an error response with HTTP status code ``status``. The response
        will be a JSON object with an 'error' key set to ``message``.
        """
        self.request.response.status = status
        return {'error': message}

    @view(validators=util.valid_coordinate_pair)
    def get(self):
        url = self.settings['nws.wind_url']
        coordinates = self.request.validated['coordinates']
        url += '?lat=%s&lon=%s&FcstType=digitalDWML' % (
            coordinates['lat'], coordinates['long'])
        r = requests.get(url)

        if r.status_code != 200:
            return self.error(500, 'Could not contact NWS wind data service.')
        elif 'forecast is unavailable for the requested location' in r.content:
            return self.error(400, 'No forecast found for that location.')

        try:
            doc = etree.fromstring(r.content)
            times = [n.text for n in doc.xpath('data/time-layout/start-valid-time')]
            speeds = [n.text for n in doc.xpath("data/parameters/wind-speed[@type='sustained']/value")]
            directions = [n.text for n in doc.xpath("data/parameters/direction[@type='wind']/value")]
            description = [n.text for n in doc.xpath('data/location/description')]
            description += [n.text for n in doc.xpath('data/location/area-description')]
            results = {'description': ' '.join(description),
                       'results': zip(times, speeds, directions)}
        except etree.XMLSyntaxError:
            return self.error(
                500, 'XML syntax error in NWS wind data response.')

        return results