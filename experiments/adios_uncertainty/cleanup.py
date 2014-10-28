
import os
import shutil

for f in ('CLISShio.txt',
          'LI_tidesWAC.CUR',
          'LongIslandSoundMap.BNA',
          'script_long_island.nc',
          'script_long_island_uncertain.nc'):
    try:
        os.remove(f)
    except:
        pass

for d in ('images', 'images_2'):
    try:
        shutil.rmtree(d)
    except:
        pass
