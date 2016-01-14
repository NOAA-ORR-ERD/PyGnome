
from gnome import constants

g = constants.gravity


class PiersonMoskowitz(object):
    '''
        Pierson-Moskowitz spectrum for fully developed, wind induced,
        surface waves.
        This relates significant wave height to U_10 (m/s), which is the
        wind speed at 10m elevation.
    '''
    @classmethod
    def significant_wave_height(cls, wind_speed):
        '''
            significant wave height H_13 (m)
        '''
        return (wind_speed ** 2.0) * 0.22 / g

    @classmethod
    def peak_wave_period(cls, wind_speed):
        '''
            peak wave period T_w (s)
        '''
        return wind_speed * 3.0 / 4.0
