import colander
import logging
import requests

from lxml import etree
from cornice.resource import resource, view
from webgnome import util, schema
from webgnome.views.services.base import BaseResource


logger = logging.getLogger(__name__)


@resource(path='/nws/wind', renderer='gnome_json',
          description='National Weather Service wind data.')
class NwsWind(BaseResource):
    def error(self, status, message):
        """
        Create an error response with HTTP status code ``status``. The response
        will be a JSON object with an 'error' key set to ``message``.
        """
        self.request.response.status = status
        return {'error': message}

    @view(validators=util.valid_coordinate_pair)
    def get(self):
        wind_schema = schema.WindSchema().bind()
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

            # create timeseries records and prune the
            # ones that have None values
            ts = [(t, (s, d)) for t, s, d in zip(times, speeds, directions)
                  if s is not None and d is not None]

            description = [n.text for n in doc.xpath('data/location/description')]
            description += [n.text for n in doc.xpath('data/location/area-description')]
        except etree.XMLSyntaxError:
            message = 'XML syntax error in NWS wind data response.'
            logger.exception(message)
            return self.error(500, message)


        wind_data = {
            'json_': 'webapi',
            'latitude': coordinates['lat'],
            'longitude': coordinates['long'],
            'source_type': 'nws',
            'description': ' '.join(description),
            'units': 'knots',
            'timeseries': ts
        }

        try:
            wind = wind_schema.deserialize(wind_data)
            wind = wind_schema.serialize(wind)
        except colander.Invalid as e:
            message = 'Schema error in NWS wind data response.'
            logger.exception(message)
            return self.error(500, message)

        return wind
