#!/usr/bin/env python
'''
Classes that generate various types of probability distributions
'''

import numpy as np
from gnome.gnomeobject import GnomeId
from colander import Float, SchemaNode, drop

from gnome.persist.base_schema import ObjTypeSchema
from gnome.utilities.compute_fraction import fraction_below_d


def get_distribution_by_name(dist_name):
    """
    Return a distribution object corresponding to its name

    note: this isn't really helpful until / unless we
    standardize the initialization interface.
    """
    try:
        return ALL_DISTS[dist_name]
    except KeyError:
        raise ValueError(f"Distributiuon: {dist_name} doesn't exist.\n"
                         f"options are: {list(ALL_DISTS.keys())}")

class DistributionBase(GnomeId):
    """
    Base class for distributions, just to make it easier to know
    what is a distribution class

    At them moment, all they need is a set_values method

    NOTE: if possible, it would be good to have the same
    interface to all distributions. For example, see: the
    distributions in numpy.random. for example:

    `random.normal(loc=0.0, scale=1.0, size=None)`

    all distribution have a location and a scale,
    which have different meanings depending on the distribution.
    """

    def set_values(self, np_array):
        raise NotImplementedError


class UniformDistributionSchema(ObjTypeSchema):
    low = SchemaNode(
        Float(), default=0.0,
        description='lower bound for uniform distribution',
        save=True, update=True
    )
    high = SchemaNode(
        Float(), name='high', default=0.1,
        description='lower bound for uniform distribution',
        save=True, update=True
    )


class NormalDistributionSchema(ObjTypeSchema):
    mean = SchemaNode(
        Float(),
        description='mean for normal distribution',
        save=True, update=True
    )
    sigma = SchemaNode(
        Float(),
        description='standard deviation for normal distribution',
        save=True, update=True
    )


class LogNormalDistributionSchema(NormalDistributionSchema):
    '''
        Same parameters as Normal

        keep in its own class since serialize/deserialize automatically
        looks for this class name. Helps keep things consistent.
    '''
    pass


class WeibullDistributionSchema(ObjTypeSchema):
    alpha = SchemaNode(
        Float(),
        description='shape parameter for weibull distribution',
        save=True, update=True
    )
    lambda_ = SchemaNode(
        Float(), default=1.0,
        description='scale parameter for weibull distribution',
        save=True, update=True
    )
    min_ = SchemaNode(
        Float(),
        description='lower bound? for weibull distribution',
        missing=drop, save=True, update=True
    )
    max_ = SchemaNode(
        Float(),
        description='upper bound? for weibull distribution',
        missing=drop, save=True, update=True
    )


class UniformDistribution(DistributionBase):
    _schema = UniformDistributionSchema

    'Uniform Probability Distribution'

    def __init__(self, low=0., high=0.1, **kwargs):
        '''
        :param low: For the Uniform distribution, it is lower bound.
        :param high: For the Uniform distribution, it is upper bound.
        '''
        super(UniformDistribution, self).__init__(**kwargs)

        self.low = low
        self.high = high
        self._check_uniform_args()

    def _check_uniform_args(self):
        if None in (self.low, self.high):
            raise TypeError('Uniform probability distribution requires '
                            'low and high')

    def _uniform(self, np_array):
        np_array[:] = np.random.uniform(self.low, self.high, len(np_array))

    def set_values(self, np_array):
        self._uniform(np_array)


class NormalDistribution(DistributionBase):
    _schema = NormalDistributionSchema

    'Normal Probability Distribution'
    def __init__(self, mean=0., sigma=0.1, **kwargs):
        '''
        :param mean: The mean of the normal distribution
        :param sigma: The standard deviation of normal distribution
        '''
        super(NormalDistribution, self).__init__(**kwargs)
        self.mean = mean
        self.sigma = sigma
        self._check_normal_args()

    def _check_normal_args(self):
        if None in (self.mean, self.sigma):
            raise TypeError('Normal probability distribution requires '
                            'mean and sigma')

    def _normal(self, np_array):
        np_array[:] = np.random.normal(self.mean, self.sigma, len(np_array))

    def set_values(self, np_array):
        self._normal(np_array)


class LogNormalDistribution(DistributionBase):
    _schema = LogNormalDistributionSchema

    'Log Normal Probability Distribution'
    def __init__(self, mean=0., sigma=0.1, **kwargs):
        '''
        :param mean: The mean of the normal distribution
        :param sigma: The standard deviation of normal distribution
        '''
        super(LogNormalDistribution, self).__init__(**kwargs)
        self.mean = mean
        self.sigma = sigma
        self._check_lognormal_args()

    def _check_lognormal_args(self):
        if None in (self.mean, self.sigma):
            raise TypeError('Log Normal probability distribution requires '
                            'mean and sigma')

    def _lognormal(self, np_array):
        np_array[:] = np.random.lognormal(self.mean, self.sigma, len(np_array))

    def set_values(self, np_array):
        self._lognormal(np_array)


class WeibullDistribution(DistributionBase):
    _schema = WeibullDistributionSchema

    'Log Normal Probability Distribution'
    def __init__(self, alpha=None, lambda_=1.0, min_=None, max_=None, **kwargs):
        '''
        :param alpha: The shape parameter 'alpha' - labeled as 'a' in
                      numpy.random.weibull distribution
        :param lambda_: The scale parameter for the distribution - required for
                        2-parameter weibull distribution (Rosin-Rammler).
        '''
        super(WeibullDistribution, self).__init__(**kwargs)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.min_ = min_
        self.max_ = max_
        self._check_weibull_args()

    def _check_weibull_args(self):
        if self.alpha is None:
            raise TypeError('Weibull distribution requires alpha')

        if self.min_ is not None:
            if self.min_ < 0:
                raise ValueError('Weibull distribution requires minimum >= 0')

            if fraction_below_d(self.min_, self.alpha, self.lambda_) > 0.999:
                raise ValueError('Weibull distribution requires '
                                 'minimum < 99.9% of total distribution')

        if self.max_ is not None:
            if self.max_ <= 0:
                raise ValueError('Weibull distribution requires maximum > 0')

            if fraction_below_d(self.max_, self.alpha, self.lambda_) < 0.001:
                raise ValueError('Weibull distribution requires '
                                 'maximum > 0.1% of total distribution')

            if self.min_ is not None and self.max_ < self.min_:
                raise ValueError('Weibull distribution requires '
                                 'maximum > minimum')

            if self.max_ < 0.00005:
                raise ValueError('Weibull distribution requires '
                                 'maximum > .000025 (25 microns)')

    def _weibull(self, np_array):
        np_array[:] = self.lambda_ * np.random.weibull(self.alpha,
                                                       len(np_array))

        if self.min_ is not None and self.max_ is not None:
            for x in range(len(np_array)):
                while np_array[x] < self.min_ or np_array[x] > self.max_:
                    np_array[x] = self.lambda_ * np.random.weibull(self.alpha)
        elif self.min_ is not None:
            for x in range(len(np_array)):
                while np_array[x] < self.min_:
                    np_array[x] = self.lambda_ * np.random.weibull(self.alpha)
        elif self.max_ is not None:
            for x in range(len(np_array)):
                while np_array[x] > self.max_:
                    np_array[x] = self.lambda_ * np.random.weibull(self.alpha)

    def set_values(self, np_array):
        self._weibull(np_array)


class RayleighDistribution():
    @classmethod
    def sigma_from_wind(cls, avg_speed):
        return np.sqrt(2.0 / np.pi) * avg_speed

    @classmethod
    def pdf(cls, x, sigma):
        return ((x / sigma ** 2.0) *
                np.exp((-1.0 / 2.0) * (x ** 2.0 / sigma ** 2.0)))

    @classmethod
    def cdf(cls, x, sigma):
        return 1.0 - np.exp((-1.0 / 2.0) * (x ** 2.0 / sigma ** 2.0))

    @classmethod
    def quantile(cls, f, sigma):
        return (sigma * np.sqrt((-1.0 * np.log((1.0 - f) ** 2.0)) + 0j)).real


ALL_DISTS = {name: obj for name, obj in vars().items()
             if isinstance(obj, type) and issubclass(obj, DistributionBase)}


if __name__ == '__main__':
    # generates TypeError
    #DistributionBase()

    UniformDistribution(low=0, high=0.1)

    NormalDistribution(mean=0, sigma=0.1)

    LogNormalDistribution(mean=0, sigma=0.1)

    WeibullDistribution(alpha=1.8, lambda_=0.000248, min_=None, max_=None)
