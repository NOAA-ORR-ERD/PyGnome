
class DingFarmer(object):
    @classmethod
    def calm_between_wave_breaks(cls, breaking_waves_frac,
                                 peak_wave_period):
        '''
            The time available (calm period) for the droplets to re-float

            D&F note that the duration of the breaking event is about half
            the wave period, presumably 0.5 * T_w, although this is unclear
            from their text.
        '''
        return (1.0 / breaking_waves_frac - 0.5) * peak_wave_period

    @classmethod
    def refloat_time(cls, significant_wave_height,
                     water_phase_xfer_velocity):
        '''
            Assuming that the 'average' droplet is inserted to a depth of
            0.75 * H_13, calculate the average re-float time T_rfl for the
            droplet.
        '''
        return 0.75 * significant_wave_height / water_phase_xfer_velocity

    @classmethod
    def water_column_time_fraction(cls,
                                   breaking_waves_frac,
                                   peak_wave_period,
                                   significant_wave_height,
                                   water_phase_xfer_velocity):
        '''
            The time fraction that the droplet spends in the water column
            f_wc
        '''
        return (cls.refloat_time(significant_wave_height,
                                 water_phase_xfer_velocity) /
                cls.calm_between_wave_breaks(breaking_waves_frac,
                                             peak_wave_period))
