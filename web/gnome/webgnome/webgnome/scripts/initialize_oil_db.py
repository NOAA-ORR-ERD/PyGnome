import sys
import os

from pyramid.paster import (setup_logging,
                            get_appsettings,
                            )

from gnome.db.oil_library.initializedb import (initialize_sql,
                                               load_database,
                                               )


def usage(argv):
    cmd = os.path.basename(argv[0])
    print('usage: %s <config_uri>\n'
          '(example: "%s development.ini")' % (cmd, cmd))
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        usage(sys.argv)
    config_uri = sys.argv[1]

    setup_logging(config_uri)
    settings = get_appsettings(config_uri)

    initialize_sql(settings)
    load_database(settings)
