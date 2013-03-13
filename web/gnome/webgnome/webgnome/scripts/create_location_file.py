import argparse
import sys
import os

from pyramid.paster import bootstrap
from webgnome import util


def main():
    parser = argparse.ArgumentParser(description='Create a location file')
    parser.add_argument('--lat', action='store', dest='latitude', type=float, required=True)
    parser.add_argument('--lon', action='store', dest='longitude', type=float, required=True)
    parser.add_argument('--name', action='store', dest='name', type=str, required=True)
    parser.add_argument('--filename', action='store', dest='filename', type=str, required=True)

    args = parser.parse_args()

    env = bootstrap('../development.ini')
    location_file_dir = env['registry'].settings.location_file_dir
    path = os.path.join(location_file_dir, args.filename)

    # Save all arguments except 'filename' in the new config.json file.
    data = args.__dict__
    filename = data.pop('filename')

    try:
        json_config = util.create_location_file(path, **data)
    except util.LocationFileExists:
        print >> sys.stderr, 'A location file with the name "%s" already ' \
                             'exists.' % filename
        exit(1)

    else:
        print 'Create location file: %s' % path
        print 'Contents of config.json: \n%s' % json_config

    env['closer']()

if __name__ == '__main__':
    main()
