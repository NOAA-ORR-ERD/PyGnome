import os
import sys
from pyramid.paster import get_app

sys.stdout = sys.stderr
base_dir = os.path.abspath(os.path.dirname(__file__))
settings_file = os.environ.get('WEBGNOME_SETTINGS', 'development.ini')
application = get_app(os.path.join(base_dir, settings_file), 'main')
