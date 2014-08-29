
from sqlalchemy import (Table,
                        Column,
                        Integer,
                        Text,
                        String,
                        Float,
                        Boolean,
                        Enum,
                        ForeignKey)

from sqlalchemy.ext.declarative import declarative_base as real_declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.orm.relationships import (RelationshipProperty,
                                          ONETOMANY, MANYTOONE, MANYTOMANY)
from sqlalchemy.orm import (scoped_session,
                            sessionmaker,
                            relationship,
                            backref)

from zope.sqlalchemy import ZopeTransactionExtension

DBSession = scoped_session(sessionmaker(extension=ZopeTransactionExtension()))

#Base = declarative_base()

# Let's make this a class decorator
declarative_base = lambda cls: real_declarative_base(cls=cls)


@declarative_base
class Base(object):
    """
    Add some default properties and methods to the SQLAlchemy declarative base.
    """

    def relationships(self, direction):
        return sorted([p.key for p in self.__mapper__.iterate_properties
                       if (isinstance(p, RelationshipProperty)
                           and p.direction == direction)])

    @property
    def one_to_many_relationships(self):
        return self.relationships(ONETOMANY)

    @property
    def many_to_many_relationships(self):
        return self.relationships(MANYTOMANY)

    @property
    def many_to_one_relationships(self):
        return self.relationships(MANYTOONE)

    @property
    def columns(self):
        return [c.name for c in self.__table__.columns]

    def columnitems(self, recurse=True):
        ret = dict((c, getattr(self, c)) for c in self.columns)
        if recurse:
            # Note: Right now our schema has a maximum of one level of
            #       indirection between objects, so short-circuiting the
            #       recursion in all cases is just fine.
            #       If we were to design a deeper schema, this would need
            #       to change.
            for r in self.one_to_many_relationships:
                ret[r] = [a.tojson(recurse=False) for a in getattr(self, r)]
            for r in self.many_to_many_relationships:
                ret[r] = [a.tojson(recurse=False) for a in getattr(self, r)]
            for r in self.many_to_one_relationships:
                ret[r] = getattr(self, r).tojson(recurse=False)
        return ret

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.columnitems)

    def tojson(self, recurse=True):
        return self.columnitems(recurse)


# UNMAPPED association table (Oil <--many-to-many--> Synonym)
oil_to_synonym = Table('oil_to_synonym', Base.metadata,
                       Column('oil_id', Integer, ForeignKey('oils.id')),
                       Column('synonym_id', Integer,
                              ForeignKey('synonyms.id')),
                       )


# UNMAPPED association table (Oil <--many-to-many--> Category)
oil_to_category = Table('oil_to_category', Base.metadata,
                       Column('oil_id', Integer, ForeignKey('oils.id')),
                       Column('category_id', Integer,
                              ForeignKey('categories.id')),
                       )


class Oil(Base):
    __tablename__ = 'oils'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    adios_oil_id = Column(String(16), unique=True, nullable=False)

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
    synonyms = relationship('Synonym', secondary=oil_to_synonym,
                            backref='oils')
    categories = relationship('Category', secondary=oil_to_category,
                              backref='oils')
    densities = relationship('Density', backref='oil',
                             cascade="all, delete, delete-orphan")
    kvis = relationship('KVis', backref='oil',
                        cascade="all, delete, delete-orphan")
    dvis = relationship('DVis', backref='oil',
                        cascade="all, delete, delete-orphan")
    cuts = relationship('Cut', backref='oil',
                        cascade="all, delete, delete-orphan")
    toxicities = relationship('Toxicity', backref='oil',
                              cascade="all, delete, delete-orphan")

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
        #     ''  which means no value.  Max should also have no value
        #         in this case
        # So it is not possible for a column to be a float and a string too.
        if kwargs.get('Pour Point Min (K)') in ('<', '>'):
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
        if kwargs.get('Flash Point Min (K)') in ('<', '>'):
            self.flash_point_min_indicator = kwargs.get('Flash Point Min (K)')
            self.flash_point_min = None
        else:
            self.flash_point_min = kwargs.get('Flash Point Min (K)')
        self.flash_point_max = kwargs.get('Flash Point Max (K)')

        self.oil_water_interfacial_tension = kwargs.get('Oil/Water Interfacial Tension (N/m)')
        self.oil_water_interfacial_tension_ref_temp = kwargs.get('Oil/Water Interfacial Tension Ref Temp (K)')
        self.oil_seawater_interfacial_tension = kwargs.get('Oil/Seawater Interfacial Tension (N/m)')
        self.oil_seawater_interfacial_tension_ref_temp = kwargs.get('Oil/Seawater Interfacial Tension Ref Temp (K)')
        self.cut_units = kwargs.get('Cut Units')
        self.oil_class = kwargs.get('Oil Class')
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
        self.preferred_oils = (True if kwargs.get('Preferred Oils') == 'X'
                               else False)
        self.koy = kwargs.get('K0Y')

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
    oil_id = Column(Integer, ForeignKey('oils.id'))

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
    oil_id = Column(Integer, ForeignKey('oils.id'))

    # demographics
    meters_squared_per_sec = Column(Float(53))
    ref_temp = Column(Float(53))
    weathering = Column(Float(53))

    def __init__(self, **kwargs):
        self.meters_squared_per_sec = kwargs.get('(m^2/s)')
        self.ref_temp = kwargs.get('Ref Temp (K)')
        self.weathering = kwargs.get('Weathering')

    def __repr__(self):
        return ('<KVis('
                'meters_squared_per_sec={0.meters_squared_per_sec}, '
                'ref_temp={0.ref_temp}, '
                'weathering={0.weathering}'
                ')>'.format(self))


class DVis(Base):
    __tablename__ = 'dvis'
    id = Column(Integer, primary_key=True)
    oil_id = Column(Integer, ForeignKey('oils.id'))

    # demographics
    kg_per_msec = Column(Float(53))
    ref_temp = Column(Float(53))
    weathering = Column(Float(53))

    def __init__(self, **kwargs):
        self.kg_per_msec = kwargs.get('(kg/ms)')
        self.ref_temp = kwargs.get('Ref Temp (K)')
        self.weathering = kwargs.get('Weathering')

    def __repr__(self):
        return ('<DVis('
                'kg_per_msec={0.kg_per_msec}, '
                'ref_temp={0.ref_temp}, '
                'weathering={0.weathering}'
                ')>'.format(self))


class Cut(Base):
    __tablename__ = 'cuts'
    id = Column(Integer, primary_key=True)
    oil_id = Column(Integer, ForeignKey('oils.id'))

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
    oil_id = Column(Integer, ForeignKey('oils.id'))

    # demographics
    tox_type = Column(Enum('EC', 'LC'), nullable=False)
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
        return "<Toxicity('%s')>" % (self.id)


class Category(Base):
    '''
        This is a self referential object suitable for building a
        hierarchy of nodes.  The relationship will be one-to-many
        child nodes.
        So Categories will be a tree of terms that the user can use to
        narrow down the list of oils he/she is interested in.
        We will support the notion that an oil can have many categories,
        and a category can contain many oils.
        Thus, Oil objects will be linked to categories in a many-to-many
        relationship.
    '''
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey(id))
    name = Column(String(50), nullable=False)

    children = relationship('Category',
                            # cascade deletions
                            cascade="all, delete-orphan",

                            # many to one + adjacency list
                            # - remote_side is required to reference the
                            #   'remote' column in the join condition.
                            backref=backref("parent", remote_side=id),
                            )

    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent

    def append(self, nodename):
        self.children.append(Category(nodename, parent=self))

    def __repr__(self):
        return ('Category(name={0}, id={1}, parent_id={2})'
                .format(self.name, self.id, self.parent_id))
