import argparse
import json
import sys
import os

from pyramid.paster import bootstrap
from webgnome import util


def main():
    parser = argparse.ArgumentParser(description='Create a location file')
    parser.add_argument('--lat', action='store', dest='lat', type=float, required=True)
    parser.add_argument('--lon', action='store', dest='lon', type=float, required=True)
    parser.add_argument('--name', action='store', dest='name', type=str, required=True)
    parser.add_argument('--filename', action='store', dest='filename', type=str, required=True)

    args = parser.parse_args()

    env = bootstrap('../development.ini')
    location_file_dir = env['registry'].settings.location_file_dir
    path = os.path.join(location_file_dir, args.filename)

    if os.path.exists(path):
        print >> sys.stderr, 'A location file with the name "%s" already ' \
                  'exists.' % args.filename
        exit(1)

    util.mkdir_p(path)

    json_config = json.dumps({
        'name': args.name,
        'latitude': args.lat,
        'longitude': args.lon,
    })

    with open(os.path.join(path, 'config.json'), 'wb') as f:
        f.write(json_config)

    print 'Create location file: %s' % path
    print 'Contents of config.json: \n%s' % json_config

    env['closer']()

if __name__ == '__main__':
    main()
