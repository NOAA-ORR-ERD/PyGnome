
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
from sqlalchemy.orm.collections import InstrumentedList
from sqlalchemy.orm.relationships import (RelationshipProperty,
                                          ONETOMANY, MANYTOONE, MANYTOMANY)
from sqlalchemy.orm import (scoped_session,
                            sessionmaker,
                            relationship,
                            backref)

from zope.sqlalchemy import ZopeTransactionExtension

DBSession = scoped_session(sessionmaker(extension=ZopeTransactionExtension()))

# Let's make declarative_base a class decorator
declarative_base = lambda cls: real_declarative_base(cls=cls)


@declarative_base
class Base(object):
    """
    Add some default properties and methods to the SQLAlchemy declarative base.
    """

    def relationships(self, direction):
        return sorted([p.key for p in self.__mapper__.iterate_properties
                       if (isinstance(p, RelationshipProperty) and
                           p.direction == direction)])

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

    def columnitems(self, recurse=2):
        ret = dict((c, getattr(self, c)) for c in self.columns)
        if recurse > 0:
            recurse = recurse - 1

            for r in self.one_to_many_relationships:
                if isinstance(getattr(self, r), InstrumentedList):
                    ret[r] = [a._tojson(recurse=recurse)
                              for a in getattr(self, r)]
                elif getattr(self, r) is not None:
                    ret[r] = getattr(self, r)._tojson(recurse=recurse)

            for r in self.many_to_many_relationships:
                ret[r] = [a._tojson(recurse=recurse) for a in getattr(self, r)]

            for r in self.many_to_one_relationships:
                if getattr(self, r) is not None:
                    ret[r] = getattr(self, r)._tojson(recurse=recurse)
        return ret

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.columnitems)

    def _tojson(self, recurse=2):
        'recursive called by columnitems'
        return self.columnitems(recurse)

    def tojson(self):
        '''
            The _tojson() routine recursively includes a bunch of redundant
            content that we don't want to return.  So we will prune it.
        '''
        oil_json = self._tojson()
        for c in oil_json['categories']:
            if 'oils' in c:
                del c['oils']

        for list_attr in ('cuts', 'densities', 'kvis', 'molecular_weights',
                          'sara_fractions', 'sara_densities'):
            for attr in oil_json[list_attr]:
                if 'imported' in attr:
                    del attr['imported']
                if 'oil' in attr:
                    del attr['oil']

        return oil_json


# UNMAPPED many-to-many association table
imported_to_synonym = Table('imported_to_synonym', Base.metadata,
                            Column('imported_record_id', Integer,
                                   ForeignKey('imported_records.id')),
                            Column('synonym_id', Integer,
                                   ForeignKey('synonyms.id')),
                            )


# UNMAPPED many-to-many association table
oil_to_category = Table('oil_to_category', Base.metadata,
                        Column('oil_id', Integer,
                               ForeignKey('oils.id')),
                        Column('category_id', Integer,
                               ForeignKey('categories.id')),
                        )


class ImportedRecord(Base):
    '''
        This object, and its related objects, is created from a
        single record inside the OilLib flat file.  The OilLib flat file
        is itself created from a filemaker export process, and is in two
        dimensional tabular format.
    '''
    __tablename__ = 'imported_records'
    id = Column(Integer, primary_key=True)

    oil_name = Column(String(100), unique=True, nullable=False)
    adios_oil_id = Column(String(16), unique=True, nullable=False)

    custom = Column(Boolean, default=False)
    location = Column(String(64))
    field_name = Column(String(64))
    reference = Column(Text)
    api = Column(Float(53))
    pour_point_min_k = Column(Float(53))
    pour_point_max_k = Column(Float(53))
    product_type = Column(String(16))
    comments = Column(Text)
    asphaltenes = Column(Float(53))
    wax_content = Column(Float(53))
    aromatics = Column(Float(53))
    water_content_emulsion = Column(Float(53))
    emuls_constant_min = Column(Float(53))
    emuls_constant_max = Column(Float(53))
    flash_point_min_k = Column(Float(53))
    flash_point_max_k = Column(Float(53))
    oil_water_interfacial_tension_n_m = Column(Float(53))
    oil_water_interfacial_tension_ref_temp_k = Column(Float(53))
    oil_seawater_interfacial_tension_n_m = Column(Float(53))
    oil_seawater_interfacial_tension_ref_temp_k = Column(Float(53))
    cut_units = Column(String(16))
    oil_class = Column(String(16))
    adhesion = Column(Float(53))
    benzene = Column(Float(53))
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
    synonyms = relationship('Synonym', secondary=imported_to_synonym,
                            backref='imported')
    densities = relationship('Density', backref='imported',
                             cascade="all, delete, delete-orphan")
    kvis = relationship('KVis', backref='imported',
                        cascade="all, delete, delete-orphan")
    dvis = relationship('DVis', backref='imported',
                        cascade="all, delete, delete-orphan")
    cuts = relationship('Cut', backref='imported',
                        cascade="all, delete, delete-orphan")
    toxicities = relationship('Toxicity', backref='imported',
                              cascade="all, delete, delete-orphan")
    oil = relationship('Oil', backref='imported', uselist=False)

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return "<ImportedRecord('%s')>" % (self.oil_name)


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
    imported_record_id = Column(Integer, ForeignKey('imported_records.id'))
    oil_id = Column(Integer, ForeignKey('oils.id'))

    kg_m_3 = Column(Float(53))
    ref_temp_k = Column(Float(53))
    weathering = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return ("<Density({0.kg_m_3} kg/m^3 at {0.ref_temp_k}K)>"
                .format(self))


class KVis(Base):
    __tablename__ = 'kvis'
    id = Column(Integer, primary_key=True)
    imported_record_id = Column(Integer, ForeignKey('imported_records.id'))
    oil_id = Column(Integer, ForeignKey('oils.id'))

    m_2_s = Column(Float(53))
    ref_temp_k = Column(Float(53))
    weathering = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return ('<KVis({0.m_2_s} m^2/s at {0.ref_temp_k}K)>'
                .format(self))


class DVis(Base):
    __tablename__ = 'dvis'
    id = Column(Integer, primary_key=True)
    imported_record_id = Column(Integer, ForeignKey('imported_records.id'))

    kg_ms = Column(Float(53))
    ref_temp_k = Column(Float(53))
    weathering = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return ('<DVis({0.kg_ms} kg/ms at {0.ref_temp_k}K)>'
                .format(self))


class Cut(Base):
    __tablename__ = 'cuts'
    id = Column(Integer, primary_key=True)
    imported_record_id = Column(Integer, ForeignKey('imported_records.id'))
    oil_id = Column(Integer, ForeignKey('oils.id'))

    vapor_temp_k = Column(Float(53))
    liquid_temp_k = Column(Float(53))
    fraction = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        lt = '{0}K'.format(self.liquid_temp_k) if self.liquid_temp_k else None
        vt = '{0}K'.format(self.vapor_temp_k) if self.vapor_temp_k else None
        return ('<Cut(liquid_temp={0}, vapor_temp={1}, fraction={2})>'
                .format(lt, vt, self.fraction))


class Toxicity(Base):
    __tablename__ = 'toxicities'
    id = Column(Integer, primary_key=True)
    imported_record_id = Column(Integer, ForeignKey('imported_records.id'))

    tox_type = Column(Enum('EC', 'LC'), nullable=False)
    species = Column(String(16))
    after_24h = Column(Float(53))
    after_48h = Column(Float(53))
    after_96h = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return ('<Toxicity({0.species}, {0.tox_type}, '
                '[{0.after_24h}, {0.after_48h}, {0.after_96h}])>'
                .format(self))


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

    def __init__(self, name, parent=None, **kwargs):
        self.name = name
        self.parent = parent

    def append(self, nodename):
        self.children.append(Category(nodename, parent=self))

    def __repr__(self):
        return ('<Category(name={0}, id={1}, parent_id={2})>'
                .format(self.name, self.id, self.parent_id))


class Estimated(Base):
    '''
        This object, is created as a single place where we can track
        which fields in the Oil object were estimated, and which ones
        were simply copied from the imported record.
    '''
    __tablename__ = 'estimated'
    id = Column(Integer, primary_key=True)

    # flag fields
    name = Column(Boolean, default=False)
    api = Column(Boolean, default=False)
    densities = Column(Boolean, default=False)
    viscosities = Column(Boolean, default=False)
    oil_water_interfacial_tension_n_m = Column(Boolean, default=False)
    oil_water_interfacial_tension_ref_temp_k = Column(Boolean, default=False)
    pour_point_min_k = Column(Boolean, default=False)
    pour_point_max_k = Column(Boolean, default=False)
    flash_point_min_k = Column(Boolean, default=False)
    flash_point_max_k = Column(Boolean, default=False)
    emulsion_water_fraction_max = Column(Boolean, default=False)
    sara_fractions = Column(Boolean, default=True)
    bullwinkle_fraction = Column(Boolean, default=False)
    adhesion_kg_m_2 = Column(Boolean, default=False)
    sulphur_fraction = Column(Boolean, default=False)
    soluability = Column(Boolean, default=False)
    cuts = Column(Boolean, default=False)
    molecular_weights = Column(Boolean, default=True)

    # relationship fields
    oil = relationship('Oil', backref='estimated', uselist=False)

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return "<Estimated('%s')>" % (self.id)


class Oil(Base):
    '''
        This is where we will put our estimated oil properties.
    '''
    __tablename__ = 'oils'
    id = Column(Integer, primary_key=True)
    imported_record_id = Column(Integer, ForeignKey('imported_records.id'))
    estimated_id = Column(Integer, ForeignKey('estimated.id'))

    name = Column(String(100), unique=True, nullable=False)
    adios_oil_id = Column(String(16))
    api = Column(Float(53))
    oil_water_interfacial_tension_n_m = Column(Float(53))
    oil_water_interfacial_tension_ref_temp_k = Column(Float(53))
    oil_seawater_interfacial_tension_n_m = Column(Float(53))
    oil_seawater_interfacial_tension_ref_temp_k = Column(Float(53))
    pour_point_min_k = Column(Float(53))
    pour_point_max_k = Column(Float(53))
    flash_point_min_k = Column(Float(53))
    flash_point_max_k = Column(Float(53))
    emulsion_water_fraction_max = Column(Float(53))
    bullwinkle_time = Column(Float(53))
    bullwinkle_fraction = Column(Float(53))
    adhesion_kg_m_2 = Column(Float(53))
    soluability = Column(Float(53))
    k0y = Column(Float(53))

    nickel_ppm = Column(Float(53))
    vanadium_ppm = Column(Float(53))

    saturates_fraction = Column(Float(53))
    aromatics_fraction = Column(Float(53))
    resins_fraction = Column(Float(53))
    asphaltenes_fraction = Column(Float(53))

    polars_fraction = Column(Float(53))
    benzene_fraction = Column(Float(53))
    sulphur_fraction = Column(Float(53))
    paraffins_fraction = Column(Float(53))
    wax_content_fraction = Column(Float(53))
    naphthenes_fraction = Column(Float(53))

    categories = relationship('Category', secondary=oil_to_category,
                              backref='oils')

    densities = relationship('Density', backref='oil',
                             cascade="all, delete, delete-orphan")
    kvis = relationship('KVis', backref='oil',
                        cascade="all, delete, delete-orphan")
    cuts = relationship('Cut', backref='oil',
                        cascade="all, delete, delete-orphan")
    sara_fractions = relationship('SARAFraction', backref='oil',
                                  cascade="all, delete, delete-orphan")
    sara_densities = relationship('SARADensity', backref='oil',
                                  cascade="all, delete, delete-orphan")
    molecular_weights = relationship('MolecularWeight', backref='oil',
                                     cascade="all, delete, delete-orphan")

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            if (a in self.columns):
                setattr(self, a, v)

    @classmethod
    def from_json(cls, oil_json):
        oil_obj = cls(**oil_json)
        oil_obj._add_one_to_many_relationships_from_json(oil_json)
        oil_obj._add_categories_from_json(oil_json)
        return oil_obj

    def _add_one_to_many_relationships_from_json(self, oil_json):
        for r in self.one_to_many_relationships:
            if r in oil_json:
                current_attr = getattr(self, r)
                py_class = self._get_class_from_relationship_property(r)

                if py_class is not None:
                    for kwargs in oil_json[r]:
                        obj = py_class(**kwargs)
                        current_attr.append(obj)

    def _add_categories_from_json(self, oil_json):
        '''
            categories is a many-to-many relationship, but has properties
            that are not generalizable.  This handles the specific
            behavior of our Categories class.
        '''
        attr = 'categories'
        if (attr in oil_json and
                len(oil_json[attr]) > 0):
            if attr in self.many_to_many_relationships:
                current_attr = getattr(self, attr)
                py_class = self._get_class_from_relationship_property(attr)

                for i in oil_json[attr]:
                    current_attr.append(self.create_linked_obj(py_class, i))
            else:
                raise AttributeError('Oil object has no attribute '
                                     '"categories"')

    def create_linked_obj(self, py_class, json_):
        '''
            Create an object plus all objects that have a parent relationship.
        '''
        if 'parent' in json_:
            json_['parent'] = self.create_linked_obj(py_class,
                                                     json_['parent'])

        return py_class(**json_)

    def _get_class_from_relationship_property(self, attr_name):
        for p in self.__mapper__.iterate_properties:
            if (isinstance(p, RelationshipProperty)) and p.key == attr_name:
                return p.mapper.class_manager.class_

        return None

    def __repr__(self):
        return '<Oil("{0.name}")>'.format(self)


class SARAFraction(Base):
    __tablename__ = 'sara_fractions'
    id = Column(Integer, primary_key=True)
    oil_id = Column(Integer, ForeignKey('oils.id'))

    sara_type = Column(Enum('Saturates', 'Aromatics', 'Resins', 'Asphaltenes'),
                       nullable=False)
    fraction = Column(Float(53))
    ref_temp_k = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return ('<SARAFraction({0.sara_type}={0.fraction} at {0.ref_temp_k}K)>'
                .format(self))


class SARADensity(Base):
    __tablename__ = 'sara_densities'
    id = Column(Integer, primary_key=True)
    oil_id = Column(Integer, ForeignKey('oils.id'))

    sara_type = Column(Enum('Saturates', 'Aromatics', 'Resins', 'Asphaltenes'),
                       nullable=False)
    density = Column(Float(53))
    ref_temp_k = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return ('<SARADensity({0.sara_type}={0.density} at {0.ref_temp_k}K)>'
                .format(self))


class MolecularWeight(Base):
    __tablename__ = 'molecular_weights'
    id = Column(Integer, primary_key=True)
    oil_id = Column(Integer, ForeignKey('oils.id'))

    sara_type = Column(Enum('Saturates', 'Aromatics', 'Resins', 'Asphaltenes'),
                       nullable=False)
    g_mol = Column(Float(53))
    ref_temp_k = Column(Float(53))

    def __init__(self, **kwargs):
        for a, v in kwargs.iteritems():
            if (a in self.columns):
                setattr(self, a, v)

    def __repr__(self):
        return ('<MolecularWeight('
                '{0.sara_type}={0.g_mol}gm/mol at {0.ref_temp_k}K)>'
                .format(self))
