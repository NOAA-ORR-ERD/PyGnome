'''
    weathering package

    This is where we keep a reasonably organized assortment of algorithms
    for calculating behavior due to weathering.
'''
from adios2 import Adios2

from .lee_huibers import LeeHuibers
from .riazi import Riazi
from .stokes import Stokes
from .pierson_moskowitz import PiersonMoskowitz
from .delvigne_sweeney import DelvigneSweeney
from .ding_farmer import DingFarmer
