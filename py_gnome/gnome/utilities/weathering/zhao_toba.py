
from gnome.utilities.weathering import PiersonMoskowitz
from gnome.constants import gravity as g


class ZhaoToba(object):
    '''
        Zhao and Toba (2001) percent whitecap coverage formula
        They use a Reynolds-like dimensionless number rather than an
        integer power of the wind speed fits the data better
    '''
    @classmethod
    def percent_whitecap_coverage(cls, wind_speed):
        '''
            percent whitecap coverage
            drag coefficient reduces linearly with wind speed
            for winds less than 2.4 m/s
        '''


        if wind_speed is 0:
            return 0
            
        if wind_speed > 2.4:
            C_D = .0008 + wind_speed * 10**(-5)
        else:
            C_D = (.0008 + 2.4 * 10**(-5)) * wind_speed / 2.4

        visc_air = 1.5 * 10**(-5) # m2/s
        peak_ang_freq = PiersonMoskowitz.peak_angular_frequency(wind_speed)
        R_Bw = C_D * wind_speed**2 / (visc_air * peak_ang_freq)
        Wc = 3.88 * 10**(-5) * R_Bw**(1.09)

        return Wc

