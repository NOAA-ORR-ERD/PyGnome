'''
Movers using diffusion as the forcing function
'''
import copy

from gnome.utilities import serializable
from gnome.movers import CyMover
from gnome.cy_gnome.cy_random_mover import CyRandomMover
from gnome.cy_gnome.cy_random_vertical_mover import CyRandomVerticalMover

class RandomMover(CyMover, serializable.Serializable):
    """
    This mover class inherits from CyMover and contains CyRandomMover

    The real work is done by CyRandomMover.
    CyMover sets everything up that is common to all movers.
    """
    state = copy.deepcopy(CyMover.state)
    state.add(update=['diffusion_coef'], create=['diffusion_coef'])
    
    def __init__(self, **kwargs):
        """
        Uses super to invoke base class __init__ method. 
        
        Optional parameters (kwargs)
        :param diffusion_coef: Diffusion coefficient for random diffusion. Default is 100,000 cm2/sec
        
        Remaining kwargs are passed onto :class:`gnome.movers.Mover` __init__ using super. 
        See Mover documentation for remaining valid kwargs.
        """
        self.mover = CyRandomMover(diffusion_coef=kwargs.pop('diffusion_coef',100000))
        super(RandomMover,self).__init__(**kwargs)

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
        return "RandomMover(diffusion_coef=%s,active_start=%s, active_stop=%s, on=%s)" % (self.diffusion_coef,self.active_start, self.active_stop, self.on)


class RandomVerticalMover(CyMover, serializable.Serializable):
    """
    This mover class inherits from CyMover and contains CyRandomVerticalMover

    The real work is done by CyRandomVerticalMoraneomver.
    CyMover sets everything up that is common to all movers.
    """
    state = copy.deepcopy(CyMover.state)
    state.add(update=['vertical_diffusion_coef'], create=['vertical_diffusion_coef'])
    
    def __init__(self, **kwargs):
        """
        Uses super to invoke base class __init__ method. 
        
        Optional parameters (kwargs)
        :param vertical_diffusion_coef: Vertical diffusion coefficient for random diffusion. Default is 5 cm2/s
        
        Remaining kwargs are passed onto Mover's __init__ using super. 
        See Mover documentation for remaining valid kwargs.
        """
        self.mover = CyRandomVerticalMover(vertical_diffusion_coef=kwargs.pop('vertical_diffusion_coef',5))
        super(RandomVerticalMover,self).__init__(**kwargs)

    @property
    def vertical_diffusion_coef(self):
        return self.mover.vertical_diffusion_coef
    @vertical_diffusion_coef.setter
    def vertical_diffusion_coef(self, value):
        self.mover.vertical_diffusion_coef = value

    def __repr__(self):
        """
        .. todo:: 
            We probably want to include more information.
        """
        return "RandomVerticalMover(vertical_diffusion_coef=%s,active_start=%s, active_stop=%s, on=%s)" % (self.vertical_diffusion_coef,self.active_start, self.active_stop, self.on)
