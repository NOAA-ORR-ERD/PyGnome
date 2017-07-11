
from gnome.constants import gravity as g


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

    @classmethod
    def peak_wave_speed(cls, wind_speed):
        '''
            peak wave speed 
        '''
        return wind_speed * 1.17

    @classmethod
    def peak_angular_frequency(cls, wind_speed):
        '''
            peak angular frequency (1/s)
        '''
        if wind_speed > 0:
            return .86 / (g * wind_speed)
        else:
            return .86 / g	# set minimum wind U=1 ?
