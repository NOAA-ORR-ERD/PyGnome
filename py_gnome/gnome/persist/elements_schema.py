'''
Schema classes for gnome.elements.* module
'''

from colander import (SchemaNode, SequenceSchema, TupleSchema, MappingSchema,
                      Int, Float, Range,
                      OneOf)

from gnome.persist.base_schema import Id


class Windage(TupleSchema):

    min_windage = SchemaNode(Float(), validator=Range(0, 1.0),
                             default=0.01)
    max_windage = SchemaNode(Float(), validator=Range(0, 1.0),
                             default=0.04)
    name = 'windage_range'


class InitWindages(Id):
    """
    windages initializer values
    """
    windage_range = Windage()
    windage_persist = SchemaNode(Int(), default=900,
        description='windage persistence in minutes')
    name = 'windages'


class InitMassComponentsFromOilProps(Id):
    name = 'mass_components'
    description = 'only need to persist the obj_type during serialization'


class InitHalfLivesFromOilProps(Id):
    name = 'half_lives'
    description = 'only need to persist the obj_type during serialization'


class InitMassFromTotalMass(Id):
    name = 'mass'
    description = 'only need to persist the obj_type during serialization'


class InitMassFromVolume(Id):
    name = 'mass'
    description = 'only need to persist the obj_type during serialization'


class InitMassFromPlume(Id):
    name = 'mass'
    description = 'only need to persist the obj_type during serialization'


class WeibullDistribution(MappingSchema):
    alpha = SchemaNode(Float())
    lambda_ = SchemaNode(Float(), default=1.0)
    min_ = SchemaNode(Float())
    max_ = SchemaNode(Float())


class UniformDistribution(MappingSchema):
    low = SchemaNode(Float(), default=0.0)
    high = SchemaNode(Float(), default=0.1)


class InitRiseVelFromDist(Id):
    name = 'mass'
    distribution = OneOf([WeibullDistribution(),
                          UniformDistribution()])


class InitRiseVelFromDropletSizeFromDist(Id):
    name = 'mass'
    distribution = OneOf([WeibullDistribution(),
                          UniformDistribution()])


class Initializers(MappingSchema):
    """
    Initializers used by the ElementType object
    add initializer classes dynamically based on the ElementType instance
    """
    name = 'initializers'


class ElementType(Id):
    """
    Serialize gnome.elements.ElementType
    The initializers property will be added dynamically since it depends
    on the data for a particle instance of the ElementType object

    todo: also need to add substance property. Need to make it serializable
    first. Some work to be done here
    """
    name = 'ElementType'
    description = 'schema for type of substance released (ElementType)'
