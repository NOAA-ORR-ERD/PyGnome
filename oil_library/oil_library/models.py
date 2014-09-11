
from slugify import slugify_filename

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
    oil_name = Column(String(100), unique=True, nullable=False)
    adios_oil_id = Column(String(16), unique=True, nullable=False)

    custom = Column(Boolean, default=False)
    location = Column(String(64))
    field_name = Column(String(64))
    reference = Column(Text)
    api = Column(Float(53))
    pour_point_min_k = Column(Float(53))
    pour_point_min_indicator = Column(String(2))
    pour_point_max_k = Column(Float(53))
    product_type = Column(String(16))
    comments = Column(Text)
    asphaltene_content = Column(Float(53))
    wax_content = Column(Float(53))
    aromatics = Column(Float(53))
    water_content_emulsion = Column(Float(53))
    emuls_constant_min = Column(Float(53))
    emuls_constant_max = Column(Float(53))
    flash_point_min_k = Column(Float(53))
    flash_point_min_indicator = Column(String(2))
    flash_point_max_k = Column(Float(53))
    oil_water_interfacial_tension_n_m = Column(Float(53))
    oil_water_interfacial_tension_ref_temp_k = Column(Float(53))
    oil_seawater_interfacial_tension_n_m = Column(Float(53))
    oil_seawater_interfacial_tension_ref_temp_k = Column(Float(53))
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
    dispersability_temp_k = Column(Float(53))
    preferred_oils = Column(Boolean, default=False)
    k0y = Column(Float(53))

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
        for a, v in kwargs.iteritems():
            a = slugify_filename(a).lower()
            if (a in self.columns
                and a not in ('pour_point_min_k',
                              'flash_point_min_k'
                              'preferred_oils')
                ):
                setattr(self, a, v)

        # kind of weird behavior...
        # pour_point_min can have the following values
        #     '<' which means "less than" the max value
        #     '>' which means "greater than" the max value
        #     ''  which means no value.  Max should also have no value
        #         in this case
        # So it is not possible for a column to be a float and a string too.
        if kwargs.get('Pour Point Min (K)') in ('<', '>'):
            self.pour_point_min_indicator = kwargs.get('Pour Point Min (K)')
            self.pour_point_min_k = None
        else:
            self.pour_point_min_k = kwargs.get('Pour Point Min (K)')

        # same kind of weird behavior as pour point...
        if kwargs.get('Flash Point Min (K)') in ('<', '>'):
            self.flash_point_min_indicator = kwargs.get('Flash Point Min (K)')
            self.flash_point_min_k = None
        else:
            self.flash_point_min_k = kwargs.get('Flash Point Min (K)')

        self.preferred_oils = (True if kwargs.get('Preferred Oils') == 'X'
                               else False)

    def __repr__(self):
        return "<Oil('%s')>" % (self.oil_name)


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

    kg_m_3 = Column(Float(53))
    ref_temp_k = Column(Float(53))
    weathering = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            a = slugify_filename(a).lower()
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return ("<Density({0.kg_m_3} kg/m^3 at {0.ref_temp}K)>"
                .format(self))


class KVis(Base):
    __tablename__ = 'kvis'
    id = Column(Integer, primary_key=True)
    oil_id = Column(Integer, ForeignKey('oils.id'))

    m_2_s = Column(Float(53))
    ref_temp_k = Column(Float(53))
    weathering = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            a = slugify_filename(a).lower()
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return ('<KVis({0.m_2_s} m^2/s at {0.ref_temp}K)>'
                .format(self))


class DVis(Base):
    __tablename__ = 'dvis'
    id = Column(Integer, primary_key=True)
    oil_id = Column(Integer, ForeignKey('oils.id'))

    kg_ms = Column(Float(53))
    ref_temp_k = Column(Float(53))
    weathering = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            a = slugify_filename(a).lower()
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return ('<DVis({0.kg_ms} kg/ms at {0.ref_temp}K)>'
                .format(self))


class Cut(Base):
    __tablename__ = 'cuts'
    id = Column(Integer, primary_key=True)
    oil_id = Column(Integer, ForeignKey('oils.id'))

    vapor_temp_k = Column(Float(53))
    liquid_temp_k = Column(Float(53))
    fraction = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            a = slugify_filename(a).lower()
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        lt = '{0}K'.format(self.liquid_temp_k) if self.liquid_temp_k else None
        vt = '{0}K'.format(self.vapor_temp_k) if self.vapor_temp_k else None
        return ('<Cut([{0}, {1}], {2})>'
                .format(lt, vt, self.fraction))


class Toxicity(Base):
    __tablename__ = 'toxicities'
    id = Column(Integer, primary_key=True)
    oil_id = Column(Integer, ForeignKey('oils.id'))

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
