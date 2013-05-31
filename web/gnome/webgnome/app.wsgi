import os
from pyramid.paster import get_app


settings_file = os.environ.get('WEBGNOME_SETTINGS', 'development.ini')
base_dir = os.path.join(os.path.dirname(__file__))


application = get_app(os.path.join(base_dir, settings_file), 'main')
