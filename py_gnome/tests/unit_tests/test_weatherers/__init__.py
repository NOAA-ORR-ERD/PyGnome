import os

# create a dump folder that tests can use to output files
dump = os.path.join(os.path.dirname(__file__), 'dump')
try:
    os.removedirs(dump)
    os.makedirs(dump)
except:
    pass