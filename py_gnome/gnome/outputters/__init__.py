
from .binary import BinaryOutput
from .erma_data_package import ERMADataPackageOutput
from .geo_json import IceGeoJsonOutput, TrajectoryGeoJsonOutput
from .image import IceImageOutput
from .json import CurrentJsonOutput, IceJsonOutput, SpillJsonOutput
from .kmz import KMZOutput
from .netcdf import NetCDFOutput, NetCDFOutputSchema
from .oil_budget import OilBudgetOutput
from .outputter import BaseOutputterSchema, Outputter
from .renderer import Renderer, RendererSchema
from .shape import ShapeOutput
from .weathering import WeatheringOutput

# NOTE: no need for __all__ if you want export everything!
outputters = [Outputter,
              NetCDFOutput,
              Renderer,
              WeatheringOutput,
              BinaryOutput,
              TrajectoryGeoJsonOutput,
              IceGeoJsonOutput,
              IceJsonOutput,
              CurrentJsonOutput,
              SpillJsonOutput,
              KMZOutput,
              IceImageOutput,
              ShapeOutput,
              ERMADataPackageOutput]

# any reason for this to be a list rather than a set?
schemas = {cls._schema for cls in outputters if hasattr(cls, '_schema')}

