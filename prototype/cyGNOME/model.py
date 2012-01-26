# model beginnings

import cyGNOME
import numpy
import random                # really basic, for now.

_WORLDPOINT = numpy.dtype([('pLong', numpy.int), ('pLat', numpy.int)])
_LEREC = numpy.dtype([('leUnits', numpy.int), ('leKey', numpy.int), ('leCustomData', numpy.int), \
            ('p', _WORLDPOINT), ('z', numpy.double), ('releaseTime', numpy.uint), \
            ('ageInHrsWhenReleased', numpy.double), ('clockRef', numpy.uint), \
            ('pollutantType', numpy.short), ('mass', numpy.double), ('density', numpy.double), \
            ('windage', numpy.double), ('dropletSize', numpy.int), ('dispersionStatus', numpy.short), \
            ('riseVelocity', numpy.double), ('statusCode', numpy.short), ('lastWaterPt', _WORLDPOINT), ('beachTime', numpy.uint)], align=True)

class LEList:
    LEs = numpy.ndarray((1000), _LEREC)     # numpy array of Python objects (pointers).

class Map:
    
    map_base = 0

    def __init__(self, x1, x2, y1, y2):
        map_base = cyGNOME.Map()
        map_base.create()
        map_base.setMapBounds(x1, y1, x2, y2)

    def doesContain(self, x, y):
        return map_base.doesContain(x, y)
    
    def movementCheck(self, x1, y1, z1, x2, y2, z2, isDispersed):
        x, y, z = map_base.movementCheck(x1, y1, z1, x2, y2, z2, isDispersed)
        return x, y, z

    def allowableSpillPoint(self, x, y):
        return map_base.allowableSpillPoint(x, y)

class Model:

    LEListCollection = LEList()

    timeStep = 0
    numSteps = 0

    randomMover = -1
    windMover = -1
    map = -1

    def initLEs(self):
        for y in range(0, 1000):
            self.LEListCollection.LEs[y]['p']['pLong'] = random.random()*90
            self.LEListCollection.LEs[y]['p']['pLat'] = random.random()*90
            self.LEListCollection.LEs[y]['z'] = 0    
            self.LEListCollection.LEs[y]['leKey'] = y

    def __init__(self):
        self.initLEs()
        self.randomMover = cyGNOME.RandomMover()
        self.windMover = cyGNOME.WindMover()
        self.randomMover.create()
        self.windMover.create()
        self.randomMover.set()

    def SetRMover(self):
        self.randomMover.set()

    def DelRMover(self, Mover):
        randomMover.delete()
        del randomMover        

    def DelWMover(self, Mover):
        windMover.delete()
        del windMover

    def RandomStep(self, timeStep):
        print 'showing model long: '
        print self.LEListCollection.LEs[0]['p']['pLong']
        print ' done.\n'
        cyGNOME._getRandomMove(self.randomMover, timeStep, self.LEListCollection.LEs)    
        print 'showing model long: '
        print self.LEListCollection.LEs[0]['p']['pLong']
        print ' done.\n'

    def SetWind(self):
        pass

    def SetSpill(self):
        pass


