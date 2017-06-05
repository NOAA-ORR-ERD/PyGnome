'''
Movers using diffusion as the forcing function
'''

import copy
import numpy as np

from colander import (SchemaNode, Float, drop)

from gnome.basic_types import (oil_status)
from gnome.cy_gnome.cy_random_mover import CyRandomMover
from gnome.cy_gnome.cy_random_vertical_mover import CyRandomVerticalMover

from gnome.utilities.serializable import Serializable, Field

from gnome.environment import IceConcentration
from gnome.environment.gridded_objects_base import PyGrid
from gnome.environment.gridded_objects_base import VariableSchema

from gnome.movers import CyMover, ProcessSchema
from gnome.persist.base_schema import ObjType


class RandomMoverSchema(ObjType, ProcessSchema):
    diffusion_coef = SchemaNode(Float(), missing=drop)
    uncertain_factor = SchemaNode(Float(), missing=drop)


class RandomMover(CyMover, Serializable):
    """
    This mover class inherits from CyMover and contains CyRandomMover

    The real work is done by CyRandomMover.
    CyMover sets everything up that is common to all movers.
    """
    _state = copy.deepcopy(CyMover._state)
    _state.add(update=['diffusion_coef', 'uncertain_factor'],
               save=['diffusion_coef', 'uncertain_factor'])
    _schema = RandomMoverSchema

    def __init__(self, **kwargs):
        """
        Uses super to invoke base class __init__ method.

        Optional parameters (kwargs)

        :param diffusion_coef: Diffusion coefficient for random diffusion.
            Default is 100,000 cm2/sec
        :param uncertain_factor: Uncertainty factor. Default is 2

        Remaining kwargs are passed onto :class:`gnome.movers.Mover` __init__
        using super.  See Mover documentation for remaining valid kwargs.
        """
        diffusion_coeff = kwargs.pop('diffusion_coef', 100000)
        uncertain_factor = kwargs.pop('uncertain_factor', 2)

        self.mover = CyRandomMover(diffusion_coef=diffusion_coeff,
                                   uncertain_factor=uncertain_factor)

        super(RandomMover, self).__init__(**kwargs)

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
                'active_start={2}, active_stop={3}, on={4})'
                .format(self.diffusion_coef, self.uncertain_factor,
                        self.active_start, self.active_stop, self.on))


class IceAwareRandomMoverSchema(RandomMoverSchema):
    ice_concentration = VariableSchema(missing=drop)


class IceAwareRandomMover(RandomMover):

    _state = copy.deepcopy(RandomMover._state)
    _state.add_field([Field('ice_concentration',
                            save=True, read=True, save_reference=True)])
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
            grid = Grid.from_netCDF(grid_file,
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

        interp = self.ice_concentration.at(positions, model_time_datetime,
                                           extrapolate=True).copy()
        interp_mask = np.logical_and(interp >= 0.2, interp < 0.8)

        if len(np.where(interp_mask)[0]) != 0:
            ice_mask = interp >= 0.8

            deltas = (super(IceAwareRandomMover, self)
                      .get_move(sc, time_step, model_time_datetime))

            interp -= 0.2
            interp *= 1.25
            interp *= 1.3333333333

            deltas[:, 0:2][ice_mask] = 0
            # scale winds from 100-0% depending on ice coverage
            deltas[:, 0:2][interp_mask] *= (1 - interp[interp_mask][:, np.newaxis])
            deltas[status] = (0, 0, 0)

            return deltas
        else:
            return (super(IceAwareRandomMover, self)
                    .get_move(sc, time_step, model_time_datetime))


class RandomVerticalMoverSchema(ObjType, ProcessSchema):
    vertical_diffusion_coef_above_ml = SchemaNode(Float(), missing=drop)
    vertical_diffusion_coef_below_ml = SchemaNode(Float(), missing=drop)

    mixed_layer_depth = SchemaNode(Float(), missing=drop)

    horizontal_diffusion_coef_above_ml = SchemaNode(Float(), missing=drop)
    horizontal_diffusion_coef_below_ml = SchemaNode(Float(), missing=drop)


class RandomVerticalMover(CyMover, Serializable):
    """
    This mover class inherits from CyMover and contains CyRandomVerticalMover

    The real work is done by CyRandomVerticalMoraneomver.
    CyMover sets everything up that is common to all movers.
    """
    _state = copy.deepcopy(CyMover._state)
    _state.add(update=['vertical_diffusion_coef_above_ml',
                       'vertical_diffusion_coef_below_ml',
                       'horizontal_diffusion_coef_above_ml',
                       'horizontal_diffusion_coef_below_ml',
                       'mixed_layer_depth'],
               save=['vertical_diffusion_coef_above_ml',
                     'vertical_diffusion_coef_below_ml',
                     'horizontal_diffusion_coef_above_ml',
                     'horizontal_diffusion_coef_below_ml',
                     'mixed_layer_depth'])
    _schema = RandomVerticalMoverSchema

    def __init__(self, **kwargs):
        """
        Uses super to invoke base class __init__ method.

        Optional parameters (kwargs)
        :param vertical_diffusion_coef_above_ml: Vertical diffusion coefficient
            for random diffusion above the mixed layer. Default is 5 cm2/s
        :param vertical_diffusion_coef_below_ml: Vertical diffusion coefficient
            for random diffusion below the mixed layer. Default is .11 cm2/s
        :param mixed_layer_depth: Mixed layer depth. Default is 10 meters.
        :param horizontal_diffusion_coef_above_ml: Horizontal diffusion
                                                   coefficient for random
                                                   diffusion above the mixed
                                                   layer. Default is
                                                   100000 cm2/s.
        :param horizontal_diffusion_coef_below_ml: Horizontal diffusion
                                                   coefficient for random
                                                   diffusion below the mixed
                                                   layer. Default is 126 cm2/s.

        Remaining kwargs are passed onto Mover's __init__ using super.
        See Mover documentation for remaining valid kwargs.
        """
        self.mover = CyRandomVerticalMover(vertical_diffusion_coef_above_ml=kwargs.pop('vertical_diffusion_coef_above_ml', 5),
                                           vertical_diffusion_coef_below_ml=kwargs.pop('vertical_diffusion_coef_below_ml', .11),
                                           horizontal_diffusion_coef_above_ml=kwargs.pop('horizontal_diffusion_coef_above_ml', 100000),
                                           horizontal_diffusion_coef_below_ml=kwargs.pop('horizontal_diffusion_coef_below_ml', 126),
                                           mixed_layer_depth=kwargs.pop('mixed_layer_depth', 10.))
        super(RandomVerticalMover, self).__init__(**kwargs)

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

    def __repr__(self):
        return ('RandomVerticalMover(vertical_diffusion_coef_above_ml={0}, '
                'vertical_diffusion_coef_below_ml={1}, mixed_layer_depth={2}, '
                'horizontal_diffusion_coef_above_ml={3}, '
                'horizontal_diffusion_coef_below_ml={4}, '
                'active_start={5}, active_stop={6}, on={6})'
                .format(self.vertical_diffusion_coef_above_ml,
                        self.vertical_diffusion_coef_below_ml,
                        self.mixed_layer_depth,
                        self.horizontal_diffusion_coef_above_ml,
                        self.horizontal_diffusion_coef_below_ml,
                        self.active_start, self.active_stop, self.on))
