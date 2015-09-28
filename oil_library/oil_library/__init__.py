import os
from itertools import chain
from collections import namedtuple
from pkg_resources import get_distribution

from sqlalchemy import create_engine
from sqlalchemy.exc import UnboundExecutionError
from sqlalchemy.orm.exc import NoResultFound

import unit_conversion as uc

from oil_library.models import Oil, Density, DBSession
from oil_library.mock_oil import sample_oil_to_mock_oil
from oil_library.oil_props import OilProps

try:
    __version__ = get_distribution('OilLibrary').version
except:
    __version__ = 'not_found'

# Some standard oils - scope is module level, non-public
# create mock_oil objects instead of dict - we always want the same instance
# of mock_oil object for say 'oil_conservative' otherwise equality fails
_sample_oils = {
    'oil_gas':
        sample_oil_to_mock_oil(max_cuts=2,
                               **{'name': 'oil_gas',
                                  'api': uc.convert('Density',
                                                    'gram per cubic centimeter',
                                                    'API degree', 0.75),
                                  'kvis': [{'m_2_s': 1.32e-6,
                                            'ref_temp_k': 273.15},
                                           {'m_2_s': 9.98e-7,
                                            'ref_temp_k': 288.15},
                                           {'m_2_s': 8.6e-7,
                                            'ref_temp_k': 311.0}],
                                  }
                               ),
    'oil_jetfuels':
        sample_oil_to_mock_oil(max_cuts=2,
                               **{'name': 'oil_jetfuels',
                                  'api': uc.convert('Density',
                                                    'gram per cubic centimeter',
                                                    'API degree', 0.81),
                                  'kvis': [{'m_2_s': 6.9e-6,
                                            'ref_temp_k': 255.0},
                                           {'m_2_s': 2.06e-6,
                                            'ref_temp_k': 273.0},
                                           {'m_2_s': 2.08e-6,
                                            'ref_temp_k': 288.0},
                                           {'m_2_s': 1.3e-6,
                                            'ref_temp_k': 313.0}],
                                  }
                               ),
    'oil_diesel':
        sample_oil_to_mock_oil(max_cuts=2,
                               **{'name': 'oil_diesel',
                                  'api': uc.convert('Density',
                                                    'gram per cubic centimeter',
                                                    'API degree', 0.87),
                                  'kvis': [{'m_2_s': 6.5e-6,
                                            'ref_temp_k': 273.0},
                                           {'m_2_s': 3.9e-6,
                                            'ref_temp_k': 288.0},
                                           {'m_2_s': 2.27e-6,
                                            'ref_temp_k': 311.0}],
                                  }
                               ),
    'oil_4':
        sample_oil_to_mock_oil(max_cuts=2,
                               **{'name': 'oil_4',
                                  'api': uc.convert('Density',
                                                    'gram per cubic centimeter',
                                                    'API degree', 0.90),
                                  'kvis': [{'m_2_s': 0.06,
                                            'ref_temp_k': 273.0},
                                           {'m_2_s': 0.03,
                                            'ref_temp_k': 278.0},
                                           {'m_2_s': 0.0175,
                                            'ref_temp_k': 283.0},
                                           {'m_2_s': 0.0057,
                                            'ref_temp_k': 288.0}],
                                  }
                               ),
    'oil_crude':
        sample_oil_to_mock_oil(max_cuts=2,
                               **{'name': 'oil_crude',
                                  'api': uc.convert('Density',
                                                    'gram per cubic centimeter',
                                                    'API degree', 0.90),
                                  'kvis': [{'m_2_s': 0.0005,
                                            'ref_temp_k': 273.0},
                                           {'m_2_s': 0.0006,
                                            'ref_temp_k': 288.0},
                                           {'m_2_s': 8.3e-5,
                                            'ref_temp_k': 293.0},
                                           {'m_2_s': 8.53e-5,
                                            'ref_temp_k': 311.0}],
                                  }
                               ),
    'oil_6':
        sample_oil_to_mock_oil(max_cuts=2,
                               **{'name': 'oil_6',
                                  'api': uc.convert('Density',
                                                    'gram per cubic centimeter',
                                                    'API degree', 0.99),
                                  'kvis': [{'m_2_s': 0.25,
                                            'ref_temp_k': 273.0},
                                           {'m_2_s': 0.038,
                                            'ref_temp_k': 278.0},
                                           {'m_2_s': 0.019,
                                            'ref_temp_k': 283.0},
                                           {'m_2_s': 0.017,
                                            'ref_temp_k': 288.0},
                                           {'m_2_s': 0.000826,
                                            'ref_temp_k': 323.0}],
                                  }
                               ),
    # 'oil_conservative':
    #    sample_oil_to_mock_oil(max_cuts=2,
    #                           **{'name': 'oil_conservative',
    #                              'api': uc.convert('Density',
    #                                                'gram per cubic centimeter',
    #                                                'API degree', 1)}),
    # 'chemical':
    #    sample_oil_to_mock_oil(max_cuts=2,
    #                           **{'name': 'chemical',
    #                              'api': uc.convert('Density',
    #                                                'gram per cubic centimeter',
    #                                                'API degree', 1)}),
    }

'''
currently, the DB is created and located when package is installed
'''
_oillib_path = os.path.dirname(__file__)
_db_file = os.path.join(_oillib_path, 'OilLib.db')


def _get_db_session():
    'we can call this from scripts to access valid DBSession'
    # not sure we want to do it this way - but let's use for now
    session = DBSession()

    try:
        session.get_bind()
    except UnboundExecutionError:
        session.bind = create_engine('sqlite:///' + _db_file)

    return session


def get_oil(oil_, max_cuts=None):
    """
    function returns the Oil object given the name of the oil as a string.

    :param oil_: The oil that spilled.
                 - If it is a dictionary of items, then we will assume it is
                   a JSON payload sufficient for creating an Oil object.
                 - If it is one of the names stored in _sample_oil dict,
                   then an Oil object with specified API is returned.
                 - Otherwise, query the database for the oil_name and return
                   the associated Oil object.
    :type oil_: str or dict

    Optional arg:

    :param max_cuts: This is ** only ** used for _sample_oils which dont have
        distillation cut information. For testing, this allows us to model the
        oil with variable number of cuts, with equally divided mass. For a
        real oil pulled from the database, this is ignored.
    :type max_cuts: int

    NOTE I:
    -------
    One issue is that the kwargs in Oil contain spaces, like 'oil_'. This
    can be handled if the user defines a dict as follows:
        kw = {'oil_': 'new oil', 'Field Name': 'field name'}
        get_oil(**kw)
    however, the following will not work:
        get_oil('oil_'='new oil', 'Field Name'='field name')

    This is another reason, we need an interface between the SQL object and the
    end user.
    """
    if isinstance(oil_, dict):
        prune_db_ids(oil_)
        return Oil.from_json(oil_)

    if oil_ in _sample_oils.keys():
        return _sample_oils[oil_]
    else:
        '''
        db_file should exist - if it doesn't then create if first
        should we raise error here?
        '''
        session = _get_db_session()

        try:
            oil = session.query(Oil).filter(Oil.name == oil_).one()
            oil.densities
            oil.kvis
            oil.cuts
            oil.sara_fractions
            oil.sara_densities
            oil.molecular_weights
            return oil
        except NoResultFound, ex:
            ex.message = ("oil with name '{0}', not found in database.  "
                          "{1}".format(oil_, ex.message))
            ex.args = (ex.message, )
            raise ex


def prune_db_ids(oil_):
    '''
        If we are instantiating an oil using a JSON payload, we do not
        need any id to be passed.  It is not necessary, and is in fact
        misleading.
        We probably only need to do it here in this module.
    '''
    for attr in ('id', 'oil_id', 'imported_record_id', 'estimated_id'):
        if attr in oil_:
            del oil_[attr]

    for list_attr in ('cuts', 'densities', 'kvis', 'molecular_weights',
                      'sara_fractions', 'sara_densities'):
        for item in oil_[list_attr]:
            for attr in ('id', 'oil_id', 'imported_record_id', 'estimated_id'):
                if attr in item:
                    del item[attr]


def get_oil_props(oil_info, max_cuts=None):
    '''
    returns the OilProps object
    max_cuts is only used for 'fake' sample_oils. It's a way to allow testing.
    When pulling record from database, this is ignored.
    '''
    oil_ = get_oil(oil_info, max_cuts)
    return OilProps(oil_)


def build_oil_props(props, max_cuts=2):
    '''
    Builds an OilProps object from a dict of properties and max_cuts
    '''
    return OilProps(sample_oil_to_mock_oil(max_cuts, **props))
