
class DelvigneSweeney(object):
    '''
        Delvigne and Sweeney (1988) computes the fraction of
        breaking waves f_bw.  DS assumes no breaking waves for winds
        less than 10 knots.
    '''
    @classmethod
    def breaking_waves_frac(cls, wind_speed, peak_wave_period):
        return 0.032 * (wind_speed - 5.0) / peak_wave_period
