'''
OilProps class which serves as a wrapper around
gnome.db.oil_library.models.Oil class

It also contains a dict containing a small number of sample_oils if user does
not wish to query database to use an oil.

It contains a function that takes a string for 'oil_name' and returns and Oil
object used to initialize and OilProps object

Not sure at present if this needs to be serializable?
'''

import os
from itertools import chain

from hazpy import unit_conversion
import sqlalchemy

from hazpy import unit_conversion
from itertools import chain
from gnome.db.oil_library.models import Oil, DBSession
from gnome.db.oil_library.initializedb import initialize_sql, \
    load_database
from gnome.utilities.remote_data import get_datafile


# Some standard oils - scope is module level, non-public
_sample_oils = {
    'oil_gas': {'Oil Name': 'oil_gas',
                'API': unit_conversion.convert('Density',
                'gram per cubic centimeter', 'API degree', 0.75)},
    'oil_jetfuels': {'Oil Name': 'oil_jetfuels',
                     'API': unit_conversion.convert('Density',
                     'gram per cubic centimeter', 'API degree',
                     0.81)},
    'oil_diesel': {'Oil Name': 'oil_diesel',
                   'API': unit_conversion.convert('Density',
                   'gram per cubic centimeter', 'API degree',
                   0.87)},
    'oil_4': {'Oil Name': 'oil_4',
              'API': unit_conversion.convert('Density',
              'gram per cubic centimeter', 'API degree', 0.90)},
    'oil_crude': {'Oil Name': 'oil_crude',
                  'API': unit_conversion.convert('Density',
                  'gram per cubic centimeter', 'API degree',
                  0.90)},
    'oil_6': {'Oil Name': 'oil_6',
              'API': unit_conversion.convert('Density',
              'gram per cubic centimeter', 'API degree', 0.99)},
    'oil_conservative': {'Oil Name': 'oil_conservative',
                         'API': unit_conversion.convert('Density',
                         'gram per cubic centimeter', 'API degree',
                         1)},
    'chemical': {'Oil Name': 'chemical',
                 'API': unit_conversion.convert('Density',
                 'gram per cubic centimeter', 'API degree', 1)},
    }

""" 
currently, the DB is stored locally - use this for now till we have
a persistent DB that we can query 
"""
_oillib_path = os.path.join(os.path.split(os.path.realpath(__file__))[0],
                           '../../../../web/gnome/webgnome/webgnome/data')
_db_file = os.path.join(_oillib_path, 'OilLibrary.db')

# No need to create DB, we'll just download the DB file from remote server:
# (http://gnome.orr.noaa.gov/py_gnome_testdata/)
# At some point, DB will be persisted on server and we just need to
# query it. At no point should this be creating the DB.
#==============================================================================
# def _db_from_flatfile():
#     """ 
#     creates the sqllite database from the OilLib flatfile 
#     """
#     oillib_file = os.path.join(_oillib_path, 'OilLib')
#     sqlalchemy_url = 'sqlite:///{0}'.format(_db_file)
#     settings = {'sqlalchemy.url': sqlalchemy_url,
#             'oillib.file': oillib_file}
#     initialize_sql(settings)
#     load_database(settings)
#==============================================================================
    
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
        return Oil(**_sample_oils[oil_name])

    else:
        if not os.path.exists(_db_file):            
            #_db_from_flatfile()
            get_datafile(_db_file)
        
        # not sure we want to do it this way - but let's use for now
        engine = sqlalchemy.create_engine('sqlite:///'+ _db_file)
        
        # let's use global DBSession defined in oillibrary
        # alternatively, we could define a new scoped_session
        # Not sure what's the proper way yet but this needs
        # to be revisited at some point.
        # session_factory = sessionmaker(bind=engine)
        # DBSession = scoped_session(session_factory)
        DBSession.bind = engine

        try:
            return DBSession.query(Oil).filter(Oil.name == oil_name).one()
        except sqlalchemy.orm.exc.NoResultFound, ex:
            # or sqlalchemy.orm.exc.MultipleResultsFound as ex:
            ex.message = \
                "oil with name '{0}' not found in database. {1}".\
                    format(oil_name,ex.message)
            ex.args = (ex.message, )
            raise ex


class OilProps(object):
    """
    Class gets the oil properties, specifically, it 
    has a property called density that returns a scalar as opposed to a
    list of Densities. Default for the property is to return value in SI units
    (kg/m^3)
     
    It can also return Density in user specified units
    """
    valid_density_units = list(chain.from_iterable([item[1] for item in
                               unit_conversion.ConvertDataUnits['Density'
                               ].values()]))
    valid_density_units.extend(unit_conversion.GetUnitNames('Density'))

    def __init__(self, oil_):
        """
        If oil_ is amongst self._sample_oils dict, then use the properties
        defined here. If not, then query the Oil database to check if oil_
        exists and get the properties from DB. 
        
        :param oil_: name of the oil 
        :type oil_: str
        """
        self.oil = get_oil(oil_)

    name = property(lambda self: self.oil.name)
    density = property(lambda self: self.get_density())

    def get_density(self, units='kg/m^3'):
        """
        :param units=kg/m^3: optional input if output units should be
            something other than kg/m^3
        """

        if self.oil.api is None:
            raise ValueError("Oil with name '{0}' does not contain 'api'"\
                " property.".format(self.oil.name))

        if units not in self.valid_density_units:
            raise unit_conversion.InvalidUnitError("Desired density units"\
                " must be from following list to be valid: {0}".\
                format(self.valid_density_units))

        # since Oil object can have various densities depending on temperature, 
        # lets return API in correct units
        return unit_conversion.convert('Density', 'API degree', units,
                self.oil.api)