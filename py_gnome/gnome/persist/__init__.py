'''
Default behavior:
Apply colander monkey patch by default
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from gnome.persist import monkey_patch_colander
from gnome.persist import base_schema, extend_colander, validators
from gnome.persist.save_load import (Savable,
                                     References,
                                     load,
                                     is_savezip_valid)

monkey_patch_colander.apply()

__all__ = [base_schema,
           extend_colander,
           validators,
           Savable,
           References,
           load,
           is_savezip_valid]
