
from gnome.spills.initializers import InitRiseVelFromDist
from gnome.utilities.distributions import UniformDistribution

import pytest


def test_init_InitRiseVelFromDist():
    ud = UniformDistribution(1,1)
    rise_vel = InitRiseVelFromDist(ud)
    assert isinstance(rise_vel,InitRiseVelFromDist)

def test_badDist_InitRiseVelFromDist():
    ud = 'This should be a distribution'
    with pytest.raises(TypeError):
        rise_vel = InitRiseVelFromDist(ud)
    



