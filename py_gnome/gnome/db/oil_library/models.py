from pyramid.security import Allow, Authenticated
from sqlalchemy import (
    Table,
    Column,
    Integer,
    Text,
    String,
    Float,
    Boolean,
    Enum,
    ForeignKey,
    )

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import (
    scoped_session,
    sessionmaker,
    relationship,
    )

from zope.sqlalchemy import ZopeTransactionExtension #@UnresolvedImport IGNORE:E0611

DBSession = scoped_session(sessionmaker(extension=ZopeTransactionExtension()))
Base = declarative_base()

class RootFactory(object):
    __acl__ = [
            (Allow, Authenticated, 'view'),
            (Allow, 'group:editors', 'edit'),
            ]
    def __init__(self, request):
        pass

# UNMAPPED association table (Oil <--many-to-many--> Synonym)
oil_to_synonym = Table('oil_to_synonym', Base.metadata,
        Column('oil_id', Integer, ForeignKey('oils.id')),
        Column('synonym_id', Integer, ForeignKey('synonyms.id')),
        )

class Oil(Base):
    __tablename__ = 'oils'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True)
    adios_oil_id = Column(String(16), unique=True)

    # demographic fields
    custom = Column(Boolean, default=False)
    location = Column(String(64))
    field_name = Column(String(64))
    reference = Column(Text)
    api = Column(Float(53))
    pour_point_min = Column(Float(53))
    pour_point_min_indicator = Column(String(2))
    pour_point_max = Column(Float(53))
    product_type = Column(String(16))
    comments = Column(Text)
    asphaltene_content = Column(Float(53))
    wax_content = Column(Float(53))
    aromatics = Column(Float(53))
    water_content_emulsion = Column(Float(53))
    emuls_constant_min = Column(Float(53))
    emuls_constant_max = Column(Float(53))
    flash_point_min = Column(Float(53))
    flash_point_min_indicator = Column(String(2))
    flash_point_max = Column(Float(53))
    oil_water_interfacial_tension = Column(Float(53))
    oil_water_interfacial_tension_ref_temp = Column(Float(53))
    oil_seawater_interfacial_tension = Column(Float(53))
    oil_seawater_interfacial_tension_ref_temp = Column(Float(53))
    cut_units = Column(String(16))
    oil_class = Column(String(16))
    adhesion = Column(Float(53))
    benezene = Column(Float(53))
    naphthenes = Column(Float(53))
    paraffins = Column(Float(53))
    polars = Column(Float(53))
    resins = Column(Float(53))
    saturates = Column(Float(53))
    sulphur = Column(Float(53))
    reid_vapor_pressure = Column(Float(53))
    viscosity_multiplier = Column(Float(53))
    nickel = Column(Float(53))
    vanadium = Column(Float(53))
    conrandson_residuum = Column(Float(53))
    conrandson_crude = Column(Float(53))
    dispersability_temp = Column(Float(53))
    preferred_oils = Column(Boolean, default=False)
    koy = Column(Float(53))

    # relationship fields
    synonyms = relationship('Synonym', secondary=oil_to_synonym, backref='oils')
    densities = relationship('Density', backref='oil', cascade="all, delete, delete-orphan")
    kvis = relationship('KVis', backref='oil', cascade="all, delete, delete-orphan")
    dvis = relationship('DVis', backref='oil', cascade="all, delete, delete-orphan")
    cuts = relationship('Cut', backref='oil', cascade="all, delete, delete-orphan")
    toxicities = relationship('Toxicity', backref='oil', cascade="all, delete, delete-orphan")

    def __init__(self, **kwargs):
        self.name = kwargs.get('Oil Name')
        self.adios_oil_id = kwargs.get('ADIOS Oil ID')
        self.location = kwargs.get('Location')
        self.field_name = kwargs.get('Field Name')
        # DONE - populate synonyms
        self.reference = kwargs.get('Reference')
        self.api = kwargs.get('API')

        # kind of weird behavior...
        # pour_point_min can have the following values
        #     '<' which means "less than" the max value
        #     '>' which means "greater than" the max value
        #     ''  which means no value.  Max should also have no value in this case
        # So it is not possible for a column to be a float and a string too.
        if kwargs.get('Pour Point Min (K)') in ('<','>'):
            self.pour_point_min_indicator = kwargs.get('Pour Point Min (K)')
            self.pour_point_min = None
        else:
            self.pour_point_min = kwargs.get('Pour Point Min (K)')
        self.pour_point_max = kwargs.get('Pour Point Max (K)')

        self.product_type = kwargs.get('Product Type')
        self.comments = kwargs.get('Comments')
        self.asphaltene_content = kwargs.get('Asphaltene Content')
        self.wax_content = kwargs.get('Wax Content')
        self.aromatics = kwargs.get('Aromatics')
        self.water_content_emulsion = kwargs.get('Water Content Emulsion')
        self.emuls_constant_min = kwargs.get('Emuls Constant Min')
        self.emuls_constant_max = kwargs.get('Emuls Constant Max')

        # same kind of weird behavior as pour point...
        if kwargs.get('Flash Point Min (K)') in ('<','>'):
            self.flash_point_min_indicator = kwargs.get('Flash Point Min (K)')
            self.flash_point_min = None
        else:
            self.flash_point_min = kwargs.get('Flash Point Min (K)')
        self.flash_point_max = kwargs.get('Flash Point Max (K)')

        self.oil_water_interfacial_tension = kwargs.get('Oil/Water Interfacial Tension (N/m)')
        self.oil_water_interfacial_tension_ref_temp = kwargs.get('Oil/Water Interfacial Tension Ref Temp (K)')
        self.oil_seawater_interfacial_tension = kwargs.get('Oil/Seawater Interfacial Tension (N/m)')
        self.oil_seawater_interfacial_tension_ref_temp = kwargs.get('Oil/Seawater Interfacial Tension Ref Temp (K)')
        # DONE - populate densities
        # DONE - populate kvis
        # DONE - populate dvis
        # DONE - populate cuts
        self.cut_units = kwargs.get('Cut Units')
        self.oil_class = kwargs.get('Oil Class')
        # DONE - populate toxicity
        self.adhesion = kwargs.get('Adhesion')
        self.benezene = kwargs.get('Benezene')
        self.naphthenes = kwargs.get('Naphthenes')
        self.paraffins = kwargs.get('Paraffins')
        self.polars = kwargs.get('Polars')
        self.resins = kwargs.get('Resins')
        self.saturates = kwargs.get('Saturates')
        self.sulphur = kwargs.get('Sulphur')
        self.reid_vapor_pressure = kwargs.get('Reid Vapor Pressure')
        self.viscosity_multiplier = kwargs.get('Viscosity Multiplier')
        self.nickel = kwargs.get('Nickel')
        self.vanadium = kwargs.get('Vanadium')
        self.conrandson_residuum = kwargs.get('Conrandson Residuum')
        self.conrandson_crude = kwargs.get('Conrandson Crude')
        self.dispersability_temp = kwargs.get('Dispersability Temp (K)')
        self.preferred_oils = True if kwargs.get('Preferred Oils') == 'X' else False
        self.koy = kwargs.get('K0Y')

    @property
    def viscosities(self):
        '''
            get a list of all kinematic viscosities associated with this
            oil object.  The list is compiled from the registered
            kinematic and dynamic viscosities.
            the viscosity fields contain:
              - kinematic viscosity in m^2/sec
              - reference temperature in degrees kelvin
              - weathering ???
            Viscosity entries are ordered by (weathering, temperature)
            If we are using dynamic viscosities, we calculate the
            kinematic viscosity from the density that is closest
            to the respective reference temperature
        '''
        # first we get the kinematic viscosities if they exist
        ret = []
        if self.kvis:
            ret = [(k.meters_squared_per_sec, k.ref_temp,
                    0.0 if k.weathering==None else k.weathering)
                    for k in self.kvis]
        if self.dvis:
            # If we have any DVis records, we need to get the
            # dynamic viscosities, convert to kinematic, and
            # add them if possible.
            # We have dvis at a certain (temperature, weathering).
            # We need to get density at the same weathering and
            # the closest temperature in order to calculate the kinematic.
            # There are lots of oil entries where the dvis do not have
            # matching densities for (temp, weathering)
            densities = [(d.kg_per_m_cubed, d.ref_temp,
                         0.0 if d.weathering==None else d.weathering)
                         for d in self.densities]
            for v, t, w in [(d.kg_per_msec, d.ref_temp, d.weathering)
                            for d in self.dvis]:
                if w == None:
                    w = 0.0
                # if we already have a KVis at the same
                # (temperature, weathering), we do not need
                # another one
                if len([vv for vv in ret if vv[1]==t and vv[2]==w]) > 0:
                    continue

                # grab the densities with matching weathering
                dlist = [(d[0], abs(t - d[1]))
                                for d in densities
                                if d[2] == w]
                if len(dlist) == 0:
                    continue
                # grab the density with the closest temperature
                density = sorted(dlist, key=lambda x:x[1])[0][0]
                # kvis = dvis/density
                ret.append(((v/density),t,w))
        ret.sort(key=lambda x: (x[2], x[1]))
        kwargs = ['(m^2/s)','Ref Temp (K)','Weathering']
        # caution: although we will have a list of real
        #          KVis objects, they are not persisted
        #          in the database.
        ret = [(KVis(**dict(zip(kwargs, v)))) for v in ret]
        return ret

    def __repr__(self):
        return "<Oil('%s')>" % (self.name)



class Synonym(Base):
    __tablename__ = 'synonyms'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "<Synonym('%s')>" % (self.name)

class Density(Base):
    __tablename__ = 'densities'
    id = Column(Integer, primary_key=True)
    oil_id  = Column(Integer, ForeignKey('oils.id'))

    # demographics
    kg_per_m_cubed = Column(Float(53))
    ref_temp = Column(Float(53))
    weathering = Column(Float(53))

    def __init__(self, **kwargs):
        self.kg_per_m_cubed = kwargs.get('(kg/m^3)')
        self.ref_temp = kwargs.get('Ref Temp (K)')
        self.weathering = kwargs.get('Weathering')

    def __repr__(self):
        return "<Density('%s')>" % (self.id)

class KVis(Base):
    __tablename__ = 'kvis'
    id = Column(Integer, primary_key=True)
    oil_id  = Column(Integer, ForeignKey('oils.id'))

    # demographics
    meters_squared_per_sec = Column(Float(53))
    ref_temp = Column(Float(53))
    weathering = Column(Float(53))

    def __init__(self, **kwargs):
        self.meters_squared_per_sec = kwargs.get('(m^2/s)')
        self.ref_temp = kwargs.get('Ref Temp (K)')
        self.weathering = kwargs.get('Weathering')

    def __repr__(self):
        return "<KVIs('%s')>" % (self.id)

class DVis(Base):
    __tablename__ = 'dvis'
    id = Column(Integer, primary_key=True)
    oil_id  = Column(Integer, ForeignKey('oils.id'))

    # demographics
    kg_per_msec = Column(Float(53))
    ref_temp = Column(Float(53))
    weathering = Column(Float(53))

    def __init__(self, **kwargs):
        self.kg_per_msec = kwargs.get('(kg/ms)')
        self.ref_temp = kwargs.get('Ref Temp (K)')
        self.weathering = kwargs.get('Weathering')

    def __repr__(self):
        return "<DVIs('%s')>" % (self.id)

class Cut(Base):
    __tablename__ = 'cuts'
    id = Column(Integer, primary_key=True)
    oil_id  = Column(Integer, ForeignKey('oils.id'))

    # demographics
    vapor_temp = Column(Float(53))
    liquid_temp = Column(Float(53))
    fraction = Column(Float(53))

    def __init__(self, **kwargs):
        self.vapor_temp = kwargs.get('Vapor Temp (K)')
        self.liquid_temp = kwargs.get('Liquid Temp (K)')
        self.fraction = kwargs.get('Fraction')

    def __repr__(self):
        return "<Cut('%s')>" % (self.id)

class Toxicity(Base):
    __tablename__ = 'toxicities'
    id = Column(Integer, primary_key=True)
    oil_id  = Column(Integer, ForeignKey('oils.id'))

    # demographics
    tox_type = Column(Enum('EC','LC'))
    species = Column(String(16))
    after_24_hours = Column(Float(53))
    after_48_hours = Column(Float(53))
    after_96_hours = Column(Float(53))

    def __init__(self, **kwargs):
        self.tox_type = kwargs.get('Toxicity Type')
        self.species = kwargs.get('Species')
        self.after_24_hours = kwargs.get('24h')
        self.after_48_hours = kwargs.get('48h')
        self.after_96_hours = kwargs.get('96h')

    def __repr__(self):
        return "<Cut('%s')>" % (self.id)




