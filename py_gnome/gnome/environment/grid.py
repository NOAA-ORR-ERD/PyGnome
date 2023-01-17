"""
grid for wind or current data
"""

from colander import (SchemaNode, drop, Float)

from gnome.utilities.time_utils import date_to_sec
from gnome.persist import base_schema
from gnome.cy_gnome.cy_grid_curv import CyTimeGridWindCurv
from gnome.cy_gnome.cy_grid_rect import CyTimeGridWindRect

from .environment import Environment


class GridSchema(base_schema.ObjTypeSchema):
    grid_type = SchemaNode(Float(), missing=drop)


class Grid(Environment):
    '''
    Defines a grid for a current or wind
    '''

    _schema = GridSchema

    def __init__(self, filename, topology_file=None, grid_type=1,
                 extrapolate=False, time_offset=0,
                 **kwargs):
        """
            Initializes a grid object from a file and a grid type.

            Maybe allow a grid to be passed in eventually, otherwise
            filename required

            All other keywords are optional. Optional parameters (kwargs):

            :param grid_type: default is 1 - regular grid
                              (eventually figure this out from file)
        """
        self._grid_type = grid_type

        self.filename = filename
        self.topology_file = topology_file

        if self._grid_type == 1:
            self.grid = CyTimeGridWindRect(filename)
        elif self._grid_type == 2:
            self.grid = CyTimeGridWindCurv(filename, topology_file)
        else:
            raise Exception('grid_type not implemented ')

        self.grid.load_data(filename, topology_file)

        super(Grid, self).__init__(**kwargs)

    def __repr__(self):
        self_ts = None
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'timeseries={1})'
                .format(self, self_ts))

    def __str__(self):
        return ("Grid ( "
                "grid_type='curvilinear')")

    @property
    def grid_type(self):
        return self._grid_type

    @grid_type.setter
    def grid_type(self, value):
        """
        probably will figure out from the file
        """
        # may want a check on value
        self._grid_type = value

    extrapolate = property(lambda self: self.grid.extrapolate,
                           lambda self, val: setattr(self.grid,
                                                     'extrapolate',
                                                     val))

    time_offset = property(lambda self: self.grid.time_offset / 3600.,
                           lambda self, val: setattr(self.grid,
                                                     'time_offset',
                                                     val * 3600.))

    def prepare_for_model_run(self, model_time):
        """
        Not sure we need to do anything here
        """
        pass

    def prepare_for_model_step(self, model_time):
        """
        Make sure we have the right data loaded
        """
        model_time = date_to_sec(model_time)
        self.grid.set_interval(model_time)

    def get_value(self, time, location):
        '''
        Return the value at specified time and location.
        '''
        data = self.grid.get_value(time, location)

        return data

    def get_values(self, model_time, positions, velocities):
        '''
        Return the values for the given positions

        '''
        data = self.grid.get_values(model_time, positions, velocities)

        return data

    def serialize(self, json_='webapi'):

        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()

        serial = schema.serialize(toserial)

        return serial

    @classmethod
    def deserialize(cls, json_):
        return cls._schema().deserialize(json_)
