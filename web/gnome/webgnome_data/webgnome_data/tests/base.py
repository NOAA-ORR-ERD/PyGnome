"""
base.py: Base classes for different types of tests.
"""
import os
import shutil
from unittest import TestCase

from pyramid import testing
from paste.deploy.loadwsgi import appconfig
from webtest import TestApp

from webgnome_data import main


class GnomeTestCase(TestCase):
    def setUp(self):
        here = os.path.dirname(__file__)
        self.project_root = os.path.abspath(os.path.dirname(here))

    def get_settings(self, config_file='../../test.ini'):
        here = os.path.dirname(__file__)
        return appconfig('config:%s' % config_file, relative_to=here)


class FunctionalTestBase(GnomeTestCase):
    def setUp(self):
        super(FunctionalTestBase, self).setUp()

        self.settings = self.get_settings()
        app = main(None, **self.settings)
        self.testapp = TestApp(app)

    def tearDown(self):
        'Clean up any images the model generated after running tests.'
        test_images_dir = os.path.join(self.project_root, 'static', 'img',
                                       self.settings['model_data_dir'])
        shutil.rmtree(test_images_dir, ignore_errors=True)


class UnitTestBase(GnomeTestCase):
    def setUp(self):
        super(UnitTestBase, self).setUp()

        self.config = testing.setUp()
        self.settings = self.get_settings()

    def tearDown(self):
        testing.tearDown()

    def get_request(self, *args, **kwargs):
        return testing.DummyRequest(*args, **kwargs)

    def get_resource(self, *args, **kwargs):
        return testing.DummyResource(*args, **kwargs)
