import os
from itertools import chain
from collections import namedtuple

import sqlalchemy

from hazpy import unit_conversion
uc = unit_conversion

from oil_library.models import Oil, DBSession
from oil_library.oil_props import OilProps


# Some standard oils - scope is module level, non-public
_sample_oils = {
    'oil_gas': {'Oil Name': 'oil_gas',
                'API': uc.convert('Density', 'gram per cubic centimeter',
                                  'API degree', 0.75)},
    'oil_jetfuels': {'Oil Name': 'oil_jetfuels',
                     'API': uc.convert('Density', 'gram per cubic centimeter',
                                       'API degree',
                                       0.81)},
    'oil_diesel': {'Oil Name': 'oil_diesel',
                   'API': uc.convert('Density', 'gram per cubic centimeter',
                                     'API degree', 0.87)},
    'oil_4': {'Oil Name': 'oil_4',
              'API': uc.convert('Density', 'gram per cubic centimeter',
                                'API degree', 0.90)},
    'oil_crude': {'Oil Name': 'oil_crude',
                  'API': uc.convert('Density', 'gram per cubic centimeter',
                                    'API degree', 0.90)},
    'oil_6': {'Oil Name': 'oil_6',
              'API': uc.convert('Density', 'gram per cubic centimeter',
                                'API degree', 0.99)},
    'oil_conservative': {'Oil Name': 'oil_conservative',
                         'API': uc.convert('Density',
                                           'gram per cubic centimeter',
                                           'API degree', 1)},
    'chemical': {'Oil Name': 'chemical',
                 'API': uc.convert('Density', 'gram per cubic centimeter',
                                   'API degree', 1)},
    }

'''
currently, the DB is created and located when package is installed
'''
_oillib_path = os.path.dirname(__file__)
_db_file = os.path.join(_oillib_path, 'OilLib.db')


def get_oil(oil_name):
    """
    function returns the Oil object given the name of the oil as a string.

    :param oil_: name of the oil that spilled. If it is one of the names
            stored in _sample_oil dict, then an Oil object with specified
            API is returned.
            Otherwise, query the database for the oil_name and return the
            associated Oil object.
    :type oil_: str

    It should be updated to take **kwargs so if user wants to define a new
    oil with specific properties, they can do so by defining properties
    in kwargs.

    NOTE I:
    -------
    One issue is that the kwargs in Oil contain spaces, like 'Oil Name'. This
    can be handled if the user defines a dict as follows:
        kw = {'Oil Name': 'new oil', 'Field Name': 'field name'}
        get_oil(**kw)
    however, the following will not work:
        get_oil('Oil Name'='new oil', 'Field Name'='field name')

    This is another reason, we need an interface (business logic) between the
    SQL object and the end user.

    NOTE II:
    --------
    currently, the _sample_oils contained in dict in this module are not part
    of the database. May want to add them to the final persistent database to
    make a consistent interface which always accesses DB for any 'oil_name'
    """

    if oil_name in _sample_oils.keys():
        return OilProps(Oil(**_sample_oils[oil_name]))

    else:
        '''
        db_file should exist - if it doesn't then create if first
        should we raise error here?
        '''

        # not sure we want to do it this way - but let's use for now
        engine = sqlalchemy.create_engine('sqlite:///' + _db_file)

        # let's use global DBSession defined in oillibrary
        # alternatively, we could define a new scoped_session
        # Not sure what's the proper way yet but this needs
        # to be revisited at some point.
        # session_factory = sessionmaker(bind=engine)
        # DBSession = scoped_session(session_factory)
        DBSession.bind = engine

        try:
            oil_= DBSession.query(Oil).filter(Oil.name == oil_name).one()
            return OilProps(oil_)
        except sqlalchemy.orm.exc.NoResultFound, ex:
            # or sqlalchemy.orm.exc.MultipleResultsFound as ex:
            ex.message = ("oil with name '{0}' not found in database. "
                          "{1}".format(oil_name, ex.message))
            ex.args = (ex.message, )
            raise ex


def oil_from_density(density, name='user_oil', units='kg/m^3'):
    """
    :param name: name of oil
    :param density: density of oil
    :param units='API': units of density

    """
    valid_density_units = list(chain.from_iterable([item[1] for item in
                               uc.ConvertDataUnits['Density'].values()]))
    valid_density_units.extend(uc.GetUnitNames('Density'))

    if density is None:
        raise ValueError("Density value required")

    if units not in valid_density_units:
        raise uc.InvalidUnitError('Desired density units must be from '
                                  'following list to be valid: '
                                  '{0}'.format(self.valid_density_units))

    if units != 'API':
        api = uc.convert('Density', units, 'API Degree', density)
    else:
        api = density

    return Oil(**{'Oil Name': name, 'API': api})
