from lxml import etree
from pyramid.httpexceptions import HTTPServerError
from cornice.resource import resource, view
import requests
from webgnome import util
from webgnome.views.services.base import BaseResource


@resource(path='/nws/wind', renderer='gnome_json',
          description='National Weather Service Wind Data.')
class Wind(BaseResource):
    @view(validators=util.valid_coordinate_pair)
    def get(self):
        url = self.settings['nws.wind_url']
        coordinates = self.request.validated['coordinates']
        url += '?lat=%s&lon=%s&FcstType=digitalDWML' % (
            coordinates['lat'], coordinates['lon'])
        r = requests.get(url)

        if r.status_code != 200:
            raise HTTPServerError('Could not contact NWS wind data service.')

        print r.content

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
            raise HTTPServerError('XML syntax error in NWS wind data response.')

        return results