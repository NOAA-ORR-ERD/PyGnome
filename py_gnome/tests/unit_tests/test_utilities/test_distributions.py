"""
tests for the distributions used for droplet size

NOTE: not the least bit complete
"""

import pytest

from gnome.utilities.distributions import get_distribution_by_name


@pytest.mark.parametrize('name', ['UniformDistribution',
                                  'NormalDistribution',
                                  'LogNormalDistribution',
                                  'WeibullDistribution',
                                  ])
def test_get_dist_by_name(name):
    dist = get_distribution_by_name(name)

    assert hasattr(dist, 'set_values')

