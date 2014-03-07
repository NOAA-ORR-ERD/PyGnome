'''
Schema representing the Graph class object
'''

from colander import (SchemaNode, SequenceSchema,
                      Float, String,
                      drop)

from gnome.persist.base_schema import Id


class PointSeries(SequenceSchema):
    point = SchemaNode(Float())


class Points(SequenceSchema):
    point_series = PointSeries()


class Labels(SequenceSchema):
    label = SchemaNode(String(), missing='')


class Formats(SequenceSchema):
    format = SchemaNode(String(), missing='')


class GraphSchema(Id):
    title = SchemaNode(String(), missing=drop)
    points = Points()
    labels = Labels()
    formats = Formats()
