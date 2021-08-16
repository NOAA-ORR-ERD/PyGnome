'''
Default behavior:
Apply colander monkey patch by default

Put all the common Schema nodes in one namespace
'''

from . import monkey_patch_colander
# from . import base_schema, extend_colander, validators
from .save_load import (Savable, References, load, is_savezip_valid)

monkey_patch_colander.apply()

del monkey_patch_colander

# import everything, so it can all be in one place
from colander import (Int, Float, DateTime, Sequence, Tuple, List, TupleSchema,
                      SequenceSchema, null, SchemaNode, String, Invalid,
                      Boolean, drop)

from .extend_colander import (TupleSchema, SequenceSchema, SchemaNode,
                              NumpyFixedLenSchema, FilenameSchema,
                              NumpyArraySchema, OrderedCollectionSchema,
                              DefaultTupleSchema, DatetimeValue2dArraySchema,
                              DatetimeValue1dArraySchema, TimeDelta,
                              LocalDateTime)

from .validators import (convertible_to_seconds, ascending_datetime,
                         no_duplicate_datetime, positive)

from .base_schema import (ObjTypeSchema, GeneralGnomeObjectSchema, LongLatBounds, PolygonSetSchema,
                         WorldPoint, ImageSize, now)

