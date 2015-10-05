'''
test the Oil.from_json function
'''
from oil_library import get_oil
from oil_library.models import Oil


def test_get_oil_from_json():
    '''
        Ok, here we will test our ability to construct an oil object from
        a json payload.
    '''
    oil_from_db = get_oil('NORTHWEST CHARGE STOCK, CHEVRON')

    oil_from_json = Oil.from_json(oil_from_db.tojson())

    # Do we want to fill in these properties???
    assert oil_from_json.imported is None
    assert oil_from_json.estimated is None

    for attr in ('adhesion_kg_m_2',
                 'api',
                 'bullwinkle_fraction',
                 'bullwinkle_time',
                 'emulsion_water_fraction_max',
                 'flash_point_max_k',
                 'flash_point_min_k',
                 'id',
                 'name',
                 'oil_seawater_interfacial_tension_n_m',
                 'oil_seawater_interfacial_tension_ref_temp_k',
                 'oil_water_interfacial_tension_n_m',
                 'oil_water_interfacial_tension_ref_temp_k',
                 'pour_point_max_k',
                 'pour_point_min_k',
                 'soluability',
                 'sulphur_fraction',
                 'k0y',):
        assert getattr(oil_from_db, attr) == getattr(oil_from_json, attr)

    for db_obj, json_obj in zip(oil_from_db.cuts, oil_from_json.cuts):
        for attr in ('liquid_temp_k', 'vapor_temp_k', 'fraction'):
            assert getattr(db_obj, attr) == getattr(json_obj, attr)

    for db_obj, json_obj in zip(oil_from_db.densities,
                                oil_from_json.densities):
        for attr in ('kg_m_3', 'ref_temp_k', 'weathering'):
            assert getattr(db_obj, attr) == getattr(json_obj, attr)

    for db_obj, json_obj in zip(oil_from_db.kvis,
                                oil_from_json.kvis):
        for attr in ('m_2_s', 'ref_temp_k', 'weathering'):
            assert getattr(db_obj, attr) == getattr(json_obj, attr)

    for db_obj, json_obj in zip(oil_from_db.molecular_weights,
                                oil_from_json.molecular_weights):
        for attr in ('g_mol', 'ref_temp_k', 'sara_type'):
            assert getattr(db_obj, attr) == getattr(json_obj, attr)

    for db_obj, json_obj in zip(oil_from_db.sara_densities,
                                oil_from_json.sara_densities):
        for attr in ('density', 'ref_temp_k', 'sara_type'):
            assert getattr(db_obj, attr) == getattr(json_obj, attr)

    for db_obj, json_obj in zip(oil_from_db.sara_fractions,
                                oil_from_json.sara_fractions):
        for attr in ('fraction', 'ref_temp_k', 'sara_type'):
            assert getattr(db_obj, attr) == getattr(json_obj, attr)

    for db_obj, json_obj in zip(oil_from_db.categories,
                                oil_from_json.categories):
        assert getattr(db_obj, 'name') == getattr(json_obj, 'name')
        assert (getattr(db_obj.parent, 'name') ==
                getattr(json_obj.parent, 'name'))
