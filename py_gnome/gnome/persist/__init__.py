'''
Default behavior:
Apply colander monkey patch by default
'''
from gnome.persist import monkey_patch_colander

monkey_patch_colander.apply()

