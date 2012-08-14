include "type_defs.pxi"
include "mover.pxi"

cdef extern from "OSSMTimeValue_c.h":
    cdef cppclass OSSMTimeValue_c:
        OSSMTimeValue_c()
        OSSMTimeValue_c(Mover_c *, TimeValuePairH, short)
        TimeValuePairH timeValues
        short fUserUnits # JLM
        double fScaleFactor # user input for scaling height derivatives or hydrology files
        WorldPoint fStationPosition
        Boolean bOSSMStyle
        double fTransport
        double fVelAtRefPt
