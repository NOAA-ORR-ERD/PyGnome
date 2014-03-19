'''
Schema classes for gnome.elements.* module
'''

from colander import (SchemaNode, SequenceSchema, TupleSchema, MappingSchema,
                      Int, Float, Range, drop, OneOf)

from gnome.persist.base_schema import Id


''' Distributions schemas - these are used by Init functions that require
pulling from a distribution '''


class UniformDistribution(Id):
    low = SchemaNode(Float(), name='low', default=0.0,
        description='lower bound for uniform distribution')
    high = SchemaNode(Float(), name='high', default=0.1,
        description='lower bound for uniform distribution')
    name = 'uniform'


class NormalDistribution(Id):
    mean = SchemaNode(Float(), name='mean',
        description='mean for normal distribution')
    sigma = SchemaNode(Float(), name='sigma',
        description='standard deviation for normal distribution')
    name = 'normal'


class LogNormalDistribution(Id):
    '''same parameters as Normal - keep in its own class to since serialize/
    deserialize automatically looks for this class name. Helps keep things
    consistent.
    '''
    NormalDistribution(name='lognormal')


class WeibullDistribution(Id):
    alpha = SchemaNode(Float(), name='alpha',
        description='shape parameter for weibull distribution')
    lambda_ = SchemaNode(Float(), name='lambda_', default=1.0,
        description='scale parameter for weibull distribution')
    min_ = SchemaNode(Float(), name='min_',
        description='lower bound? for weibull distribution',
        missing=drop)
    max_ = SchemaNode(Float(), name='max_',
        description='upper bound? for weibull distribution',
        missing=drop)
    name = 'weibull'


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


class InitRiseVelFromDist(Id):
    name = 'rise_vel'
    description = 'rise velocity initializer - dynamically add distribution'


class InitRiseVelFromDropletSizeFromDist(Id):
    name = 'rise_vel'
    description = 'droplet size initializer - dynamically add distribution'


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
