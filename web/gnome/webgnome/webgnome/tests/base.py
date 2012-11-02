"""
base.py: Base classes for different types of tests.
"""
import os
import unittest

from pyramid import testing
from paste.deploy.loadwsgi import appconfig


class FunctionalTestBase(unittest.TestCase):
   def setUp(self):
        from webgnome import main
        app = main({})
        from webtest import TestApp
        self.testapp = TestApp(app)


class UnitTestBase(unittest.TestCase):
    def setUp(self):
        self.config = testing.setUp()

        # Application settings. TODO: Use ``test.ini`` file?
        cwd = os.getcwd()
        self.settings = appconfig(
            'config:../../development.ini', relative_to=cwd)

    def tearDown(self):
        testing.tearDown()

    def get_request(self, *args, **kwargs):
        return testing.DummyRequest(*args, **kwargs)

    def get_resource(self, *args, **kwargs):
        return testing.DummyResource(*args, **kwargs)


