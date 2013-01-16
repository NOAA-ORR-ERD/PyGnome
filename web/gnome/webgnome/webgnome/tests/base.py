"""
base.py: Base classes for different types of tests.
"""
import os
import shutil
import unittest

from pyramid import testing
from paste.deploy.loadwsgi import appconfig
from webtest import TestApp

from webgnome import main


class GnomeTestCase(unittest.TestCase):
    def get_settings(self, config_file='../../test.ini'):
        dirname = os.path.dirname(__file__)
        return appconfig('config:%s' % config_file, relative_to=dirname)

class FunctionalTestBase(GnomeTestCase):
    def setUp(self):
        self.settings = self.get_settings()
        app = main(None, **self.settings)
        self.testapp = TestApp(app)

    def tearDown(self):
        """
        Clean up any images the model generated after running tests.
        """
        project_root = os.path.abspath(
            os.path.dirname(os.path.dirname(__file__)))
        test_images_dir = os.path.join(
            project_root, 'static', 'img', self.settings['model_images_dir'])
        shutil.rmtree(test_images_dir, ignore_errors=True)


class UnitTestBase(GnomeTestCase):
    def setUp(self):
        self.config = testing.setUp()
        self.settings = self.get_settings()

    def tearDown(self):
        testing.tearDown()

    def get_request(self, *args, **kwargs):
        return testing.DummyRequest(*args, **kwargs)

    def get_resource(self, *args, **kwargs):
        return testing.DummyResource(*args, **kwargs)
