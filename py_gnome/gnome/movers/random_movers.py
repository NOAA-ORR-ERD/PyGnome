'''
Movers using diffusion as the forcing function
'''

import copy

from gnome.utilities.serializable import Serializable
from gnome.movers import CyMover
from gnome.cy_gnome.cy_random_mover import CyRandomMover
from gnome.cy_gnome.cy_random_vertical_mover import CyRandomVerticalMover


class RandomMover(CyMover, Serializable):
    """
    This mover class inherits from CyMover and contains CyRandomMover

    The real work is done by CyRandomMover.
    CyMover sets everything up that is common to all movers.
    """
    _state = copy.deepcopy(CyMover._state)
    _state.add(update=['diffusion_coef'], create=['diffusion_coef'])

    def __init__(self, **kwargs):
        """
        Uses super to invoke base class __init__ method.

        Optional parameters (kwargs)
        :param diffusion_coef: Diffusion coefficient for random diffusion.
                               Default is 100,000 cm2/sec

        Remaining kwargs are passed onto :class:`gnome.movers.Mover` __init__
        using super.  See Mover documentation for remaining valid kwargs.
        """
        self.mover = \
            CyRandomMover(diffusion_coef=kwargs.pop('diffusion_coef',
                          100000))
        super(RandomMover, self).__init__(**kwargs)

    @property
    def diffusion_coef(self):
        return self.mover.diffusion_coef

    @diffusion_coef.setter
    def diffusion_coef(self, value):
        self.mover.diffusion_coef = value

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """

        return ('RandomMover(diffusion_coef={0}, '
                'active_start={1}, active_stop={2}, '
                'on={3})'.format(self.diffusion_coef,
                                 self.active_start, self.active_stop, self.on))


class RandomVerticalMover(CyMover, Serializable):
    """
    This mover class inherits from CyMover and contains CyRandomVerticalMover

    The real work is done by CyRandomVerticalMoraneomver.
    CyMover sets everything up that is common to all movers.
    """
    _state = copy.deepcopy(CyMover._state)
    _state.add(update=['vertical_diffusion_coef_above_ml',
                      'vertical_diffusion_coef_below_ml',
                      'mixed_layer_depth'],
              create=['vertical_diffusion_coef_above_ml',
                      'vertical_diffusion_coef_below_ml',
                      'mixed_layer_depth'])

    def __init__(self, **kwargs):
        """
        Uses super to invoke base class __init__ method.

        Optional parameters (kwargs)
        :param vertical_diffusion_coef_above_ml: Vertical diffusion coefficient
                                                 for random diffusion above the
                                                 mixed layer.
                                                 Default is 5 cm2/s
        :param vertical_diffusion_coef_below_ml: Vertical diffusion coefficient
                                                 for random diffusion below the
                                                 mixed layer.
                                                 Default is .11 cm2/s
        :param mixed_layer_depth: Mixed layer depth. Default is 10 meters.

        Remaining kwargs are passed onto Mover's __init__ using super.
        See Mover documentation for remaining valid kwargs.
        """
        self.mover = CyRandomVerticalMover(vertical_diffusion_coef_above_ml=kwargs.pop('vertical_diffusion_coef_above_ml', 5),
                                           vertical_diffusion_coef_below_ml=kwargs.pop('vertical_diffusion_coef_below_ml', .11),
                                           mixed_layer_depth=kwargs.pop('mixed_layer_depth', 10.))
        super(RandomVerticalMover, self).__init__(**kwargs)

    @property
    def vertical_diffusion_coef_above_ml(self):
        return self.mover.vertical_diffusion_coef_above_ml

    @property
    def vertical_diffusion_coef_below_ml(self):
        return self.mover.vertical_diffusion_coef_below_ml

    @property
    def mixed_layer_depth(self):
        return self.mover.mixed_layer_depth

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
        '''
        .. todo:: We probably want to include more information.
        '''
        return ('RandomVerticalMover(vertical_diffusion_coef_above_ml={0}, '
                'vertical_diffusion_coef_below_ml={1}, '
                'mixed_layer_depth={2}, active_start={3}, active_stop={4}, '
                'on={5})'.format(self.vertical_diffusion_coef_above_ml,
                                 self.vertical_diffusion_coef_below_ml,
                                 self.mixed_layer_depth,
                                 self.active_start, self.active_stop, self.on))
