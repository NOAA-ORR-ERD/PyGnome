include "type_defs.pxi"
cdef extern from "WindMover_c.h":
    cdef cppclass WindMover_c:
        double fSpeedScale
        double fAngleScale
        double fMaxSpeed
        double fMaxAngle
        double fSigma2
        double fSigmaTheta
        unsigned long fUncertainStartTime
        unsigned long fDuration
        Boolean bUncertaintyPointOpen
        Boolean bSubsurfaceActive
        double fGamma
        Boolean fIsConstantWind
        VelocityRec fConstantValue
        LEWindUncertainRec **fWindUncertaintyList
        long **fLESetSizes
        double breaking_wave_height
        double mix_layer_depth
        OSErr get_move(int n, long model_time, long step_len, char *wp_ra, char *wind_ra, char *dispersion_ra, double breaking_wave, double mix_layer, char *uncertain_ra, char* time_vals, int num_times)
        OSErr get_move(int n, long model_time, long step_len, char *wp_ra, char *wind_ra, char *dispersion_ra, double breaking_wave, double mix_layer, char* time_vals, int num_times)
        OSErr PrepareForModelStep( Seconds&,  Seconds&, Seconds&, bool)