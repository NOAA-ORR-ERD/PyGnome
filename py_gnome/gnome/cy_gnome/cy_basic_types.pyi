"""
"interface" file -- used to add docstrings that autoapi can read

Definitions in cy_basic_types.pyx

"""


class oil_status(IntEnum):
    """
    (Maps to C enum)

    Enum for the element status. Values are::

        not_released = OILSTAT_NOTRELEASED = 0
        in_water = OILSTAT_INWATER = 2
        on_land = OILSTAT_ONLAND = 3
        off_maps = OILSTAT_OFFMAPS = 7
        evaporated = OILSTAT_EVAPORATED = 10
        to_be_removed = OILSTAT_TO_BE_REMOVED = 12
        on_tideflat = OILSTAT_ON_TIDEFLAT = 32
    """


class numerical_methods(IntEnum):
    """
    Enum for the integration options:

    euler = EULER = 0 (Euler method)

    rk4 = RK4 = 1 (4th order Runge-Kutta)
    """

class spill_type(IntEnum):
    """
    ::
        forecast = FORECAST_LE = 1

        uncertainty = UNCERTAINTY_LE = 2
    """

class ts_format(IntEnum):
    """
    Contains enum type for the timeseries (ts) either given directly or
    read from datafile, by OSSMTimeValue.
    For instance, a standard wind file would contain magnitude and direction info:

      ts_format.magnitude_direction = 5,

      'r-theta' is another alias for this so,

      `ts_format.r_theta` = 5


    It could also contain uv info. Tides would contain uv with v == 0
    Hydrology file would also contain uv format
    from TypeDefs.h::

        *   M19REALREAL = 1,
            M19HILITEDEFAULT = 2
            M19MAGNITUDEDEGREES = 3
            M19DEGREESMAGNITUDE = 4
        *   M19MAGNITUDEDIRECTION = 5
            M19DIRECTIONMAGNITUDE = 6
            M19CANCEL = 7
            M19LABEL = 8
    """
