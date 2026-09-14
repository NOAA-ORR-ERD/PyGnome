
"""
Scripting package for GNOME with assorted utilities that make it easier to
write scripts.

The ultimate goal is to be able to run py_gnome for the "common" use cases
with only functionality available in this module

Classes and helper functions are imported from various py_gnome modules
(spill, environment, movers etc).

we recommend that this module be used like so::

  import gnome.scripting as gs

Then you will have easy access to most of the stuff you need to write
py_gnome scripts with, e.g.::

    model = gs.Model(start_time="2018-04-12T12:30",
                     duration=gs.days(2),
                     time_step=gs.minutes(15))

    model.map = gs.MapFromBNA('coast.bna', refloat_halflife=0.0)

    model.spills += gs.point_line_spill(num_elements=1000,
                                                start_position=(-163.75,
                                                                69.75,
                                                                0.0),
                                                release_time="2018-04-12T12:30")
"""
from gnome import initialize_console_log
from gnome.basic_types import oil_status_map
from gnome.environment import (
    FileGridCurrent,
    GridCurrent,
    GridWind,
    IceAwareCurrent,
    IceAwareWind,
    SteadyUniformCurrent,
    Tide,
    Water,
    Waves,
)
from gnome.environment.environment_objects import IceConcentration, IceVelocity
from gnome.environment.wind import PointWind, Wind, constant_wind, wind_from_values
from gnome.maps.map import GnomeMap, MapFromBNA
from gnome.model import Model
from gnome.movers import (
    CatsMover,
    ComponentMover,
    CurrentMover,
    IceAwareRandomMover,
    PointWindMover,
    RandomMover,
    RandomMover3D,
    RiseVelocityMover,
    WindMover,
)
from gnome.movers.c_wind_movers import (
    constant_point_wind_mover,
    point_wind_mover_from_file,
)
from gnome.outputters import (
    KMZOutput,
    NetCDFOutput,
    OilBudgetOutput,
    Renderer,
    ShapeOutput,
    WeatheringOutput,
)
from gnome.spills.gnome_oil import GnomeOil
from gnome.spills.release import PointLineRelease, PolygonRelease
from gnome.spills.spill import (
    Spill,
    grid_spill,
    point_line_spill,
    polygon_release_spill,
    spatial_release_spill,
    subsurface_spill,
)
from gnome.spills.substance import NonWeatheringSubstance
from gnome.utilities import convert_longitude
from gnome.utilities.inf_datetime import InfTime, MinusInfTime
from gnome.utilities.remote_data import get_datafile
from gnome.utilities.time_utils import asdatetime

from .time_utils import (
    days,
    hours,
    minutes,
    now,
    seconds,
    weeks,
)
from .utilities import (
    PrintFinder,
    make_images_dir,
    remove_netcdf,
    set_verbose,
)


def load_model(filename):
    """
    load a model from an existing savefile

    :param filename: filename (path) of the save file to load.

    :returns: A configured Model object.

    :note: This is simply a handy wrapper around the Model.load_savefile
           classmethod
    """
    return Model.load_savefile(filename)


def constant_point_current_mover(speed, direction, units='m/s'):
    """
    utility function to create a point current mover with a constant uniform current:

    a single velocity at all time and space.

    :param speed: current speed
    :param direction: direction of the current (direction to, not from), in degrees CCW from north
    :param units='m/s': the units that the input wind speed is in.
                    options: 'm/s', 'knot', 'mph', others...

    :return: returns a gnome.movers.PyCurrentMover object all set up.
    """

    curr = SteadyUniformCurrent(speed, direction, units)

    return CurrentMover(curr)
