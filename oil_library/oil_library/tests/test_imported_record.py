import sys
import os

import unittest
import pytest
from pytest import raises

sqlalchemy = pytest.importorskip('sqlalchemy')
zope = pytest.importorskip('zope')
zope.sqlalchemy = pytest.importorskip('zope.sqlalchemy')
transaction = pytest.importorskip('transaction')

from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

from oil_library.models import (DBSession, Base,
                                Oil, ImportedRecord,
                                Synonym,
                                Density,
                                KVis,
                                DVis,
                                Cut,
                                Toxicity)

here = os.path.dirname(__file__)
db_file = os.path.join(here, r'OilLibrary.db')


sqlalchemy_url = 'sqlite:///%s' % db_file


class BaseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(sqlalchemy_url)
        Base.metadata.create_all(cls.engine)
        print Base

    def setUp(self):
        self.session = DBSession()
        if not self.session.bind:
            self.session.bind = self.engine
        transaction.begin()

    def tearDown(self):

        # for unit testing, we throw away any modifications we may have made.

        transaction.abort()
        self.session.close()

    def add_objs_and_assert_ids(self, objs):
        if type(objs) is list:
            for o in objs:
                self.session.add(o)
            self.session.flush()
            for o in objs:
                assert o.id is not None
        else:
            self.session.add(objs)
            self.session.flush()
            assert objs.id is not None


class OilTestCase(BaseTestCase):

    @classmethod
    def get_mock_oil_file_record(cls):
        return {
            u'oil_name': u'Test Oil',
            u'adios_oil_id': u'AD99999',
            u'location': u'Sand Point',
            u'field_name': u'Sand Point',
            u'reference': u'Test Oil Reference',
            u'api': u'2.68e1',
            u'pour_point_min_k': u'2.6715e2',
            u'pour_point_max_k': u'2.6715e2',
            u'product_type': u'Crude',
            u'comments': u'Test Oil Comments',
            u'asphaltenes': u'2e-2',
            u'wax_content': u'7e-2',
            u'aromatics': u'4e-2',
            u'water_content_emulsion': u'9e-1',
            u'emuls_constant_min': u'0e0',
            u'emuls_constant_max': u'0e0',
            u'flash_point_min_k': u'2.8e2',
            u'flash_point_max_k': u'2.8e2',
            u'oil_water_interfacial_tension_n_m': u'2.61e-2',
            u'oil_water_interfacial_tension_ref_temp_k': u'2.7e2',
            u'oil_seawater_interfacial_tension_n_m': u'2.38e-2',
            u'oil_seawater_interfacial_tension_ref_temp_k': u'2.7e2',
            u'cut_units': u'volume',
            u'oil_class': u'Group 3',
            u'adhesion': u'2.8e-1',
            u'benzene': u'6e-2',
            u'naphthenes': u'7e-2',
            u'paraffins': u'8e-2',
            u'polars': u'9e-2',
            u'resins': u'1.2e-1',
            u'saturates': u'1.1e-1',
            u'sulphur': u'1.3e-1',
            u'reid_vapor_pressure': u'1.5e-1',
            u'viscosity_multiplier': u'16',
            u'nickel': u'14.7',
            u'vanadium': u'33.9',
            u'conrandson_residuum': u'1.7e-1',
            u'conrandson_crude': u'1.8e-1',
            u'dispersability_temp_k': u'2.8e2',
            u'preferred_oils': True,
            u'k0y': u'2.024e-6',
            }

    def assert_mock_oil_object(self, oil):
        assert oil.oil_name == u'Test Oil'
        assert oil.adios_oil_id == u'AD99999'
        assert oil.location == u'Sand Point'
        assert oil.field_name == u'Sand Point'
        assert oil.reference == u'Test Oil Reference'
        assert oil.api == u'2.68e1'
        assert oil.pour_point_min_k == u'2.6715e2'
        assert oil.pour_point_max_k == u'2.6715e2'
        assert oil.product_type == u'Crude'
        assert oil.comments == u'Test Oil Comments'
        assert oil.asphaltenes == u'2e-2'
        assert oil.wax_content == u'7e-2'
        assert oil.aromatics == u'4e-2'
        assert oil.water_content_emulsion == u'9e-1'
        assert oil.emuls_constant_min == u'0e0'
        assert oil.emuls_constant_max == u'0e0'
        assert oil.flash_point_min_k == u'2.8e2'
        assert oil.flash_point_max_k == u'2.8e2'
        assert oil.oil_water_interfacial_tension_n_m == u'2.61e-2'
        assert oil.oil_water_interfacial_tension_ref_temp_k == u'2.7e2'
        assert oil.oil_seawater_interfacial_tension_n_m == u'2.38e-2'
        assert oil.oil_seawater_interfacial_tension_ref_temp_k == u'2.7e2'
        assert oil.cut_units == u'volume'
        assert oil.oil_class == u'Group 3'
        assert oil.adhesion == u'2.8e-1'
        assert oil.benzene == u'6e-2'
        assert oil.naphthenes == u'7e-2'
        assert oil.paraffins == u'8e-2'
        assert oil.polars == u'9e-2'
        assert oil.resins == u'1.2e-1'
        assert oil.saturates == u'1.1e-1'
        assert oil.sulphur == u'1.3e-1'
        assert oil.reid_vapor_pressure == u'1.5e-1'
        assert oil.viscosity_multiplier == u'16'
        assert oil.nickel == u'14.7'
        assert oil.vanadium == u'33.9'
        assert oil.conrandson_residuum == u'1.7e-1'
        assert oil.conrandson_crude == u'1.8e-1'
        assert oil.dispersability_temp_k == u'2.8e2'
        assert oil.preferred_oils is True
        assert oil.k0y == u'2.024e-6'

    def test_init_no_args(self):
        oil_obj = ImportedRecord()
        assert oil_obj is not None
        assert oil_obj.id is None
        with raises(IntegrityError):
            self.session.add(oil_obj)
            self.session.flush()

    def test_init_with_args(self):
        oil_obj = ImportedRecord(**self.get_mock_oil_file_record())
        self.assert_mock_oil_object(oil_obj)
        self.add_objs_and_assert_ids(oil_obj)


class SynonymTestCase(BaseTestCase):

    '''
        Synonyms are pretty basic objects.  The complexity
        comes when integrated in many-to-many relationships
        with the Oil object
    '''

    def test_init_no_args(self):
        with raises(TypeError):
            synonym_obj = Synonym()
            assert synonym_obj is not None

    def test_init_with_args(self):
        synonym_obj = Synonym('synonym')
        assert synonym_obj is not None
        assert synonym_obj.name == 'synonym'
        self.add_objs_and_assert_ids(synonym_obj)


class DensityTestCase(BaseTestCase):

    @classmethod
    def get_mock_density_file_record(cls):
        return {u'kg_m_3': u'9.037e2', u'ref_temp_k': u'2.7315e2',
                u'weathering': u'0e0'}

    def assert_mock_density_object(self, density):
        assert density.kg_m_3 == u'9.037e2'
        assert density.ref_temp_k == u'2.7315e2'
        assert density.weathering == u'0e0'

    def test_init_no_args(self):
        density_obj = Density()
        self.add_objs_and_assert_ids(density_obj)

    def test_init_with_args(self):
        density_obj = Density(**self.get_mock_density_file_record())
        self.assert_mock_density_object(density_obj)
        self.add_objs_and_assert_ids(density_obj)


class KVisTestCase(BaseTestCase):

    @classmethod
    def get_mock_kvis_file_record(cls):
        return {u'm_2_s': u'5.59e-5', u'ref_temp_k': u'2.7315e2',
                u'weathering': u'0e0'}

    def assert_mock_kvis_object(self, kvis):
        assert kvis.m_2_s == u'5.59e-5'
        assert kvis.ref_temp_k == u'2.7315e2'
        assert kvis.weathering == u'0e0'

    def test_init_no_args(self):
        kvis_obj = KVis()
        self.add_objs_and_assert_ids(kvis_obj)

    def test_init_with_args(self):
        kvis_obj = KVis(**self.get_mock_kvis_file_record())
        self.assert_mock_kvis_object(kvis_obj)
        self.add_objs_and_assert_ids(kvis_obj)


class DVisTestCase(BaseTestCase):

    @classmethod
    def get_mock_dvis_file_record(cls):
        return {u'kg_ms': u'4.73e-2', u'ref_temp_k': u'2.7315e2',
                u'weathering': u'0e0'}

    def assert_mock_dvis_object(self, dvis):
        assert dvis.kg_ms == u'4.73e-2'
        assert dvis.ref_temp_k == u'2.7315e2'
        assert dvis.weathering == u'0e0'

    def test_init_no_args(self):
        dvis_obj = DVis()
        self.add_objs_and_assert_ids(dvis_obj)

    def test_init_with_args(self):
        dvis_obj = DVis(**self.get_mock_dvis_file_record())
        self.assert_mock_dvis_object(dvis_obj)
        self.add_objs_and_assert_ids(dvis_obj)


class CutTestCase(BaseTestCase):

    @classmethod
    def get_mock_cut_file_record(cls):
        return {u'vapor_temp_k': u'3.1015e2',
                u'liquid_temp_k': u'3.8815e2', u'fraction': u'1e-2'}

    def assert_mock_cut_object(self, cut):
        assert cut.vapor_temp_k == u'3.1015e2'
        assert cut.liquid_temp_k == u'3.8815e2'
        assert cut.fraction == u'1e-2'

    def test_init_no_args(self):
        cut_obj = Cut()
        self.add_objs_and_assert_ids(cut_obj)

    def test_init_with_args(self):
        cut_obj = Cut(**self.get_mock_cut_file_record())
        self.assert_mock_cut_object(cut_obj)
        self.add_objs_and_assert_ids(cut_obj)


class ToxicityTestCase(BaseTestCase):

    @classmethod
    def get_mock_toxicity_file_record(cls):
        return {
            u'species': u'Daphnia Magna',
            u'tox_type': u'EC',
            u'after_24h': None,
            u'after_48h': u'0.61',
            u'after_96h': None,
            }

    def assert_mock_toxicity_object(self, toxicity):
        assert toxicity.tox_type == u'EC'
        assert toxicity.species == u'Daphnia Magna'
        assert toxicity.after_24h is None
        assert toxicity.after_48h == u'0.61'
        assert toxicity.after_96h is None

    def test_init_no_args(self):
        toxicity_obj = Toxicity()
        assert toxicity_obj is not None
        with raises(IntegrityError):
            self.session.add(toxicity_obj)
            self.session.flush()

    def test_init_with_args(self):
        toxicity_obj = Toxicity(**self.get_mock_toxicity_file_record())
        self.assert_mock_toxicity_object(toxicity_obj)
        self.add_objs_and_assert_ids(toxicity_obj)

    def test_init_with_invalid_type(self):
        toxicity_args = self.get_mock_toxicity_file_record()
        toxicity_args[u'tox_type'] = u'invalid'
        toxicity_obj = Toxicity(**toxicity_args)  # IGNORE:W0142
        with raises(IntegrityError):
            self.session.add(toxicity_obj)
            self.session.flush()


class IntegrationTestCase(BaseTestCase):

    def test_add_synonym_to_oil(self):
        oil_obj = ImportedRecord(**OilTestCase.get_mock_oil_file_record())
        synonym_obj = Synonym('test oil')

        oil_obj.synonyms.append(synonym_obj)

        self.add_objs_and_assert_ids([oil_obj, synonym_obj])
        assert oil_obj.synonyms == [synonym_obj]
        assert synonym_obj.imported == [oil_obj]

    def test_oils_that_share_a_synonym(self):
        oil_args = OilTestCase.get_mock_oil_file_record()
        oil_obj1 = ImportedRecord(**oil_args)
        oil_args.update({u'oil_name': u'Test Oil 2',
                        u'adios_oil_id': u'AD99998'})
        oil_obj2 = ImportedRecord(**oil_args)
        synonym_obj = Synonym('test oil')

        oil_obj1.synonyms.append(synonym_obj)
        oil_obj2.synonyms.append(synonym_obj)

        self.add_objs_and_assert_ids([oil_obj1, oil_obj2, synonym_obj])
        assert oil_obj1.synonyms == [synonym_obj]
        assert oil_obj2.synonyms == [synonym_obj]
        assert synonym_obj.imported == [oil_obj1, oil_obj2]

    def test_add_density_to_oil(self):
        oil_obj = ImportedRecord(**OilTestCase.get_mock_oil_file_record())
        density_obj = \
            Density(**DensityTestCase.get_mock_density_file_record())

        oil_obj.densities.append(density_obj)

        self.add_objs_and_assert_ids([oil_obj, density_obj])
        assert oil_obj.densities == [density_obj]
        assert density_obj.imported == oil_obj

    def test_add_kvis_to_oil(self):
        oil_obj = ImportedRecord(**OilTestCase.get_mock_oil_file_record())
        kvis_obj = KVis(**KVisTestCase.get_mock_kvis_file_record())

        oil_obj.kvis.append(kvis_obj)

        self.add_objs_and_assert_ids([oil_obj, kvis_obj])
        assert oil_obj.kvis == [kvis_obj]
        assert kvis_obj.imported == oil_obj

    def test_add_dvis_to_oil(self):
        oil_obj = ImportedRecord(**OilTestCase.get_mock_oil_file_record())
        dvis_obj = DVis(**DVisTestCase.get_mock_dvis_file_record())

        oil_obj.dvis.append(dvis_obj)

        self.add_objs_and_assert_ids([oil_obj, dvis_obj])
        assert oil_obj.dvis == [dvis_obj]
        assert dvis_obj.imported == oil_obj

    def test_add_cut_to_oil(self):
        oil_obj = ImportedRecord(**OilTestCase.get_mock_oil_file_record())
        cut_obj = Cut(**CutTestCase.get_mock_cut_file_record())

        oil_obj.cuts.append(cut_obj)

        self.add_objs_and_assert_ids([oil_obj, cut_obj])
        assert oil_obj.cuts == [cut_obj]
        assert cut_obj.imported == oil_obj

    def test_add_toxicity_to_oil(self):
        oil_obj = ImportedRecord(**OilTestCase.get_mock_oil_file_record())
        toxicity_obj = \
            Toxicity(**ToxicityTestCase.get_mock_toxicity_file_record())

        oil_obj.toxicities.append(toxicity_obj)

        self.add_objs_and_assert_ids([oil_obj, toxicity_obj])
        assert oil_obj.toxicities == [toxicity_obj]
        assert toxicity_obj.imported == oil_obj
