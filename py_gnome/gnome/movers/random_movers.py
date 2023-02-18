'''
Movers using diffusion as the forcing function
'''

import numpy as np

from colander import (SchemaNode, Float, Boolean, drop)

from gnome.basic_types import oil_status
from gnome.cy_gnome.cy_random_mover import CyRandomMover
from gnome.cy_gnome.cy_random_mover_3d import CyRandomMover3D

from gnome.environment import IceConcentration
from gnome.environment.gridded_objects_base import PyGrid
from gnome.environment.gridded_objects_base import VariableSchema

from gnome.movers import CyMover, ProcessSchema
from gnome.persist.validators import convertible_to_seconds
from gnome.persist.extend_colander import LocalDateTime
from gnome.utilities.inf_datetime import InfTime, MinusInfTime


class RandomMoverSchema(ProcessSchema):
    diffusion_coef = SchemaNode(Float(), save=True, update=True, missing=drop)
    uncertain_factor = SchemaNode(Float(), save=True, update=True,
                                  missing=drop)
    data_start = SchemaNode(LocalDateTime(), validator=convertible_to_seconds,
                            read_only=True)
    data_stop = SchemaNode(LocalDateTime(), validator=convertible_to_seconds,
                           read_only=True)


class RandomMover(CyMover):
    # This mover class inherits from CyMover and contains CyRandomMover

    # The real work is done by CyRandomMover.
    # CyMover sets everything up that is common to all movers.
    """
    "Random Walk" diffusion mover

    Moves the elements each time step in a random direction, according to the
    specified diffusion coefficient.
    """
    _schema = RandomMoverSchema

    def __init__(self,
                 diffusion_coef=100000.0,
                 uncertain_factor=2.0,
                 **kwargs):
        # Uses super to invoke CyMover__init__ method.
        """
        :param diffusion_coef: Diffusion coefficient for random diffusion.
            Default is 100,000 cm2/sec
        :type diffusion_coef: float or integer in units of cm^2/s

        :param uncertain_factor: Uncertainty factor. Default is 2.0

        Remaining kwargs are passed onto :class:`gnome.movers.Mover` __init__

        See Mover documentation for remaining valid kwargs.
        """
        # diffusion_coeff = kwargs.pop('diffusion_coef', 100000)
        # uncertain_factor = kwargs.pop('uncertain_factor', 2)

        self.mover = CyRandomMover(diffusion_coef=diffusion_coef,
                                   uncertain_factor=uncertain_factor)

        super(RandomMover, self).__init__(**kwargs)

    @property
    def data_start(self):
        return MinusInfTime()

    @property
    def data_stop(self):
        return InfTime()

    @property
    def diffusion_coef(self):
        return self.mover.diffusion_coef

    @diffusion_coef.setter
    def diffusion_coef(self, value):
        self.mover.diffusion_coef = value

    @property
    def uncertain_factor(self):
        return self.mover.uncertain_factor

    @uncertain_factor.setter
    def uncertain_factor(self, value):
        self.mover.uncertain_factor = value

    def __repr__(self):
        return ('RandomMover(diffusion_coef={0}, uncertain_factor={1}, '
                'active_range={2}, on={3})'
                .format(self.diffusion_coef, self.uncertain_factor,
                        self.active_range, self.on))


class IceAwareRandomMoverSchema(RandomMoverSchema):
    ice_concentration = VariableSchema(
        missing=drop, save=True, update=True, save_reference=True
    )


class IceAwareRandomMover(RandomMover):

    _schema = IceAwareRandomMoverSchema

    _req_refs = {'ice_concentration': IceConcentration}

    def __init__(self, ice_concentration=None, **kwargs):
        self.ice_concentration = ice_concentration
        super(IceAwareRandomMover, self).__init__(**kwargs)

    @classmethod
    def from_netCDF(cls, filename=None,
                    dataset=None,
                    grid_topology=None,
                    units=None,
                    time=None,
                    ice_concentration=None,
                    grid=None,
                    grid_file=None,
                    data_file=None,
                    **kwargs):
        if filename is not None:
            data_file = filename
            grid_file = filename

        if grid is None:
            grid = PyGrid.from_netCDF(grid_file,
                                    grid_topology=grid_topology)

        if ice_concentration is None:
            ice_concentration = (IceConcentration
                                 .from_netCDF(filename=filename,
                                              dataset=dataset,
                                              data_file=data_file,
                                              grid_file=grid_file,
                                              time=time, grid=grid,
                                              **kwargs))

        return cls(ice_concentration=ice_concentration, **kwargs)

    def get_move(self, sc, time_step, model_time_datetime):
        status = sc['status_codes'] != oil_status.in_water
        positions = sc['positions']
        deltas = np.zeros_like(positions)

        cctn = (self.ice_concentration.at(positions, model_time_datetime,
                                            extrapolate=True)
                  .copy())

        if np.any(cctn >= 0.2):
            ice_mask = cctn >= 0.8
            water_mask = cctn < 0.2
            interp_mask = np.logical_and(cctn >= 0.2, cctn < 0.8)

            ice_vel_factor = cctn.copy()
            ice_vel_factor[ice_mask] = 0
            ice_vel_factor[water_mask] = 1
            ice_vel_factor[interp_mask] = 1 - ((ice_vel_factor[interp_mask] - 0.2) * 10) / 6

            deltas = (super(IceAwareRandomMover, self)
                      .get_move(sc, time_step, model_time_datetime))

            #deltas *= ice_vel_factor[:,None]
            deltas *= ice_vel_factor
            deltas[status] = (0,0,0)

            return deltas
        else:
            return (super(IceAwareRandomMover, self)
                      .get_move(sc, time_step, model_time_datetime))


class RandomMover3DSchema(ProcessSchema):
    vertical_diffusion_coef_above_ml = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    vertical_diffusion_coef_below_ml = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    mixed_layer_depth = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    horizontal_diffusion_coef_above_ml = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    horizontal_diffusion_coef_below_ml = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    surface_is_allowed = SchemaNode(Boolean())


class RandomMover3D(CyMover):
    """
    This mover class inherits from CyMover and contains CyRandomMover3D

    The real work is done by CyRandomMover3D.
    CyMover sets everything up that is common to all movers.
    """
    _schema = RandomMover3DSchema

    def __init__(self,
                 vertical_diffusion_coef_above_ml=5,
                 vertical_diffusion_coef_below_ml=0.11,
                 horizontal_diffusion_coef_above_ml=100_000,
                 horizontal_diffusion_coef_below_ml=126,
                 mixed_layer_depth=10.0,
                 surface_is_allowed=False,
                 **kwargs):

        # Uses super to invoke base class __init__ method.

        """
        :param vertical_diffusion_coef_above_ml: Vertical diffusion coefficient for random diffusion
                                                 above the mixed layer. Default is 5 cm2/s
        :param vertical_diffusion_coef_below_ml: Vertical diffusion coefficient for random diffusion
                                                 below the mixed layer. Default is .11 cm2/s
        :param mixed_layer_depth: Mixed layer depth. Default is 10 meters
        :param horizontal_diffusion_coef_above_ml: Horizontal diffusion coefficient for random diffusion
                                                   above the mixed layer. Default is 100000 cm2/s
        :param horizontal_diffusion_coef_below_ml: Horizontal diffusion coefficient for random diffusion
                                                   below the mixed layer. Default is 126 cm2/s
        :param surface_is_allowed: Vertical diffusion will ignore surface particles if this is True. Default
                                   is False.

        Remaining kwargs are passed onto Mover's __init__ using super.

        See Mover documentation for remaining valid kwargs.
        """
        self.mover = CyRandomMover3D(vertical_diffusion_coef_above_ml=vertical_diffusion_coef_above_ml,
                                     vertical_diffusion_coef_below_ml=vertical_diffusion_coef_below_ml,
                                     horizontal_diffusion_coef_above_ml=horizontal_diffusion_coef_above_ml,
                                     horizontal_diffusion_coef_below_ml=horizontal_diffusion_coef_below_ml,
                                     mixed_layer_depth=mixed_layer_depth,
                                     surface_is_allowed=surface_is_allowed,
                                     )
        super().__init__(**kwargs)

    @property
    def horizontal_diffusion_coef_above_ml(self):
        return self.mover.horizontal_diffusion_coef_above_ml

    @property
    def horizontal_diffusion_coef_below_ml(self):
        return self.mover.horizontal_diffusion_coef_below_ml

    @property
    def vertical_diffusion_coef_above_ml(self):
        return self.mover.vertical_diffusion_coef_above_ml

    @property
    def vertical_diffusion_coef_below_ml(self):
        return self.mover.vertical_diffusion_coef_below_ml

    @property
    def mixed_layer_depth(self):
        return self.mover.mixed_layer_depth

    @property
    def surface_is_allowed(self):
        return self.mover.surface_is_allowed

    @horizontal_diffusion_coef_above_ml.setter
    def horizontal_diffusion_coef_above_ml(self, value):
        self.mover.horizontal_diffusion_coef_above_ml = value

    @horizontal_diffusion_coef_below_ml.setter
    def horizontal_diffusion_coef_below_ml(self, value):
        self.mover.horizontal_diffusion_coef_below_ml = value

    @vertical_diffusion_coef_above_ml.setter
    def vertical_diffusion_coef_above_ml(self, value):
        self.mover.vertical_diffusion_coef_above_ml = value

    @vertical_diffusion_coef_below_ml.setter
    def vertical_diffusion_coef_below_ml(self, value):
        self.mover.vertical_diffusion_coef_below_ml = value

    @mixed_layer_depth.setter
    def mixed_layer_depth(self, value):
        self.mover.mixed_layer_depth = value

    @surface_is_allowed.setter
    def surface_is_allowed(self, value):
        self.mover.surface_is_allowed = value

    def __repr__(self):
        return ('RandomMover3D(vertical_diffusion_coef_above_ml={0}, '
                'vertical_diffusion_coef_below_ml={1}, mixed_layer_depth={2}, '
                'horizontal_diffusion_coef_above_ml={3}, '
                'horizontal_diffusion_coef_below_ml={4}, '
                'surface_is_allowed={5}, '
                'active_range={6}, on={7})'
                .format(self.vertical_diffusion_coef_above_ml,
                        self.vertical_diffusion_coef_below_ml,
                        self.mixed_layer_depth,
                        self.horizontal_diffusion_coef_above_ml,
                        self.horizontal_diffusion_coef_below_ml,
                        self.surface_is_allowed,
                        self.active_range, self.on))
