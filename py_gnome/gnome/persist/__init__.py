'''
Default behavior:
Apply colander monkey patch by default
'''
from gnome.persist import monkey_patch_colander
from gnome.persist import base_schema, extend_colander, validators

monkey_patch_colander.apply()

__all__ = [base_schema, extend_colander, validators]
