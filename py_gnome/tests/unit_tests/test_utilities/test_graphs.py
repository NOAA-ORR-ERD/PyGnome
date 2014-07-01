'''
Tests serialize/deserialize to create and update objects

It just tests the interface works, doesn't actually change values
'''

from gnome.utilities.weathering.graphs import Graph
#from gnome.persist.graph_schema import GraphSchema
from gnome.utilities.weathering.graphs import GraphSchema


class TestGraphSchema(object):
    def test_create(self, sample_graph):
        serial_dict = GraphSchema().serialize(sample_graph.to_serialize('save'))
        deserial_dict = GraphSchema().deserialize(serial_dict)
        new_graph = Graph.new_from_dict(deserial_dict)

        # TODO: not sure if we want to duplicate the id here or not.
        # I think it might depend on whether we are creating a new
        # object or updating it.
        #assert new_graph.id == sample_graph.id

        assert new_graph.points == [list(x) for x in sample_graph.points]
        assert new_graph.labels == list(sample_graph.labels)
        assert new_graph.formats == list(sample_graph.formats)

        assert new_graph.title == sample_graph.title
        pass

    def test_update(self, sample_graph):
        '''
           Just tests methods don't fail and the schema is properly defined.
           It doesn't update any properties.
        '''
        serial_dict = GraphSchema().serialize(sample_graph.to_serialize())
        deserial_dict = GraphSchema().deserialize(serial_dict)
        sample_graph.update_from_dict(deserial_dict)
        pass
