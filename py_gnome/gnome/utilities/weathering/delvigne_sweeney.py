import numpy as np


class DelvigneSweeney(object):
    '''
        Delvigne and Sweeney (1988) computes the fraction of
        breaking waves f_bw.  DS assumes no breaking waves for winds
        less than 10 knots.
    '''

    @staticmethod
    def breaking_waves_frac(wind_speed, peak_wave_period):
        '''
            Field observations of Holthuysen and Herbers (1986)
            and Toba et al. (1971) lead to a simple empirical relation
            for spilling breakers in deep water.
        '''
        F_wc = 0.032 * (wind_speed - 5.0) / peak_wave_period

        return np.clip(F_wc, 0.0, 1.0)
