'''
Ship drift mover
'''

import os

import numpy as np

from colander import (SchemaNode, String, Float, drop)

from gnome.basic_types import (velocity_rec,
                               world_point,
                               world_point_type,
                               status_code_type,
                               oil_status)
from gnome.array_types import gat

from gnome.utilities import projections
from gnome.utilities import rand

from gnome.environment import Grid
from gnome.movers import Mover, ProcessSchema


class ShipDriftMoverSchema(ProcessSchema):
    wind_file = SchemaNode(String(), save=True, missing=drop, isdatafile=True,
                           test_equal=False)
    topology_file = SchemaNode(String(), missing=drop, save=True,
                               isdatafile=True, test_equal=False)
    wind_scale = SchemaNode(Float(), missing=drop, save=True, update=True)
    grid_type = SchemaNode(Float(), missing=drop, save=True, update=True)
    drift_angle = SchemaNode(Float(), missing=drop, save=True, update=True)


class ShipDriftMover(Mover):
    """
    mover to model ship drift
    """

    _schema = ShipDriftMoverSchema

    def __init__(self,
                 wind_file=None,
                 topology_file=None,
                 grid_type=1,
                 drift_angle=0,
                 time_offset=0,
                 **kwargs):
        """
        :param wind_file: file containing wind data on a grid
        :param topology_file: Default is None. When exporting topology, it
                              is stored in this file
        :param wind_scale: Value to scale wind data
        :param extrapolate: Allow current data to be extrapolated before and
                            after file data
        :param time_offset: Time zone shift if data is in GMT

        Pass optional arguments to base class
        uses super: ``super(ShipDriftMover,self).__init__(**kwargs)``
        """
        if not os.path.exists(wind_file):
            raise ValueError('Path for wind file does not exist: {0}'
                             .format(wind_file))

        if topology_file is not None:
            if not os.path.exists(topology_file):
                raise ValueError('Path for Topology file does not exist: {0}'
                                 .format(topology_file))

        # is wind_file and topology_file is stored with cy_gridwind_mover?
        self.wind_file = wind_file
        self.topology_file = topology_file

        self.name = os.path.split(wind_file)[1]
        self.drift_angle = drift_angle
        self._wind_scale = kwargs.pop('wind_scale', 1)

        self.grid_type = grid_type
        self.grid = Grid(wind_file, topology_file, grid_type)

        self.mover = Mover()

        super(ShipDriftMover, self).__init__(**kwargs)

        # have to override any uncertainty
        # self.grid.load_data(wind_file, topology_file)

        self.model_time = 0

        self.positions = np.zeros((0, 3), dtype=world_point_type)
        self.delta = np.zeros((0, 3), dtype=world_point_type)
        self.status_codes = np.zeros((0, 1), dtype=status_code_type)

        self.array_types.update({'windages':gat('windages'),
                                 'windage_range':gat('windage_range'),
                                 'windage_persist':gat('windage_persist')})

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """
        return ('ShipDriftMover('
                'active_range={0.active_range}, '
                'on={0.on})'.format(self, self.mover))

    def __str__(self):
        return ('ShipDriftMover - current _state.\n'
                '  active_range time={1.active_range}\n'
                '  current on/off status={1.on}'
                .format(self, self.mover))

    wind_scale = property(lambda self: self._wind_scale,
                          lambda self, val: setattr(self, 'wind_scale', val))

    extrapolate = property(lambda self: self.grid.extrapolate,
                           lambda self, val: setattr(self.grid,
                                                     'extrapolate', val))

    time_offset = property(lambda self: self.grid.time_offset / 3600.,
                           lambda self, val: setattr(self.grid,
                                                     'time_offset',
                                                     val * 3600.))

    def export_topology(self, topology_file):
        """
        :param topology_file=None: absolute or relative path where topology
                                   file will be written.
        """
        if topology_file is None:
            raise ValueError('Topology file path required: {0}'.
                             format(topology_file))

        self.grid.export_topology(topology_file)

    def prepare_for_model_run(self):
        """
        Override this method if a derived mover class needs to perform any
        actions prior to a model run
        """
        # May not need this function
        pass

    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
        """
        Call base class method using super
        Also updates windage for this timestep

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of model as a date time object
        """
        # not sure if we need to redefine this or what we want to do here
        super(ShipDriftMover, self).prepare_for_model_step(sc, time_step,
                                                           model_time_datetime)

        # if no particles released, then no need for windage
        # TODO: revisit this since sc.num_released shouldn't be None
        if sc.num_released is None or sc.num_released == 0:
            return

        self.grid.prepare_for_model_step(model_time_datetime)
        # here we might put in drift angle stuff ?

        if self.active:
            rand.random_with_persistance(sc['windage_range'][:, 0],
                                         sc['windage_range'][:, 1],
                                         sc['windages'],
                                         sc['windage_persist'],
                                         time_step)

    def prepare_data_for_get_move(self, sc, model_time_datetime):
        """
        organizes the spill object into inputs for calling with Cython
        wrapper's get_move(...)

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param model_time_datetime: current model time as datetime object
        """
        self.model_time = self.datetime_to_seconds(model_time_datetime)

        # Get the data:
        try:
            self.positions = sc['positions']
            self.status_codes = sc['status_codes']
        except KeyError as err:
            raise ValueError('The spill container does not have the required'
                             'data arrays\n' + str(err))

        self.positions = (self.positions.view(dtype=world_point)
                          .reshape((len(self.positions),)))

        self.delta = np.zeros(len(self.positions), dtype=world_point)

    def get_move(self, sc, time_step, model_time_datetime):
        """
        Override base class functionality because mover has a different
        get_move signature

        :param sc: an instance of the gnome.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time
                                    object
        """
        self.prepare_data_for_get_move(sc, model_time_datetime)

        # will need to override get_move using grid's get_values

        vels = np.zeros(len(self.positions), dtype=velocity_rec)
        in_water_mask = self.status_codes == oil_status.in_water

        if self.active and len(self.positions) > 0:
            self.grid.get_values(self.model_time, self.positions, vels)

            self.delta['lat'][in_water_mask] = vels['v'] * time_step
            self.delta['long'][in_water_mask] = vels['u'] * time_step

            self.delta['lat'][in_water_mask] *= sc['windages']
            self.delta['long'][in_water_mask] *= sc['windages']

            self.delta = (projections.FlatEarthProjection
                          .meters_to_lonlat(self.delta
                                            .view(dtype=np.float64)
                                            .reshape(-1, 3),
                                            self.positions
                                            .view(dtype=np.float64)
                                            .reshape(-1, 3)))

        return (self.delta.view(dtype=world_point_type)
                .reshape((-1, len(world_point))))

    def model_step_is_done(self, sc=None):
        """
        This method gets called by the model after everything else is done
        in a time step, and is intended to perform any necessary clean-up
        operations. Subclassed movers can override this method.
        """
        # Probably don't need this function
        pass
