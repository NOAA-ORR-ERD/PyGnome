from webgnome.tests.service_tests import ModelHelperMixin
from base import FunctionalTestBase


class ModelViewTests(FunctionalTestBase, ModelHelperMixin):
    def test_get_index(self):
        # Need some tests here, mostly to verify that JSON necessary to
        # bootstrap the JS app is present on the page.
        pass
