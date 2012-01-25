import cython
import random
import math

from libcpp.vector cimport vector
from cython.operator import preincrement as preinc
include "cyGNOMEDefs.pxi"

cimport numpy as np
import numpy as np

cdef class RandomMover:

	cdef Random_c *mover
	def create(self):
		self.mover = new Random_c()
	def set(self):
		_setRandomMover(self)
	def delete(self):
		del self.mover
	def reset(self):
		self.delete()
		self.create()

cdef class WindMover:

	cdef WindMover_c *mover
	def create(self):
		self.mover = new WindMover_c()
	def delete(self):
		del self.mover
	def reset(self):
		self.delete()
		self.create()

cdef class Map:
	
	cdef Map_c *map
	def create(self, x1, y1, x2, y2):
		self.map = new Map_c()
	def delete(self):
		del self.mover
	def reset(self):
		self.delete()
		self.create()
	def setMapBounds(self, x1, y1, x2, y2):
		cdef WorldRect newBounds
		if(x1 <= x2):
			newBounds.loLong = x1
			newBounds.hiLong = x2
		else:
			newBounds.loLong = x2
			newBounds.hiLong = x1
		if(y1 <= y2):
			newBounds.loLat = y1
			newBounds.hiLat = y2
		else:
			newBounds.loLat = y2
			newBounds.hiLat = y1
		self.map.SetMapBounds(newBounds)

	def doesContain(self, x, y):
		cdef WorldPoint p
		p.pLat = y
		p.pLong = x
		return self.map.InMap(p)

	def movementCheck(self, x1, y1, z1, x2, y2, z2, isDispersed):
		cdef WorldPoint3D fromWPt, toWPt
		cdef WorldPoint3D result
		fromWPt.p.pLat = y1
		fromWPt.p.pLong = x1
		fromWPt.z = z1
		toWPt.p.pLat = y2
		toWPt.p.pLong = x2
		toWPt.z = z2
		result = self.map.MovementCheck(fromWPt, toWPt, isDispersed)
		return result.p.pLong, result.p.pLat, result.z

	def allowableSpillPoint(self, x, y):
		cdef WorldPoint p
		p.pLat = y
		p.pLong = x
		return self.map.IsAllowableSpillPoint(p)


def _setRandomMover(RandomMover obj, *args):

	cdef Random_c *Mover = obj.mover
	
	if(len(args) != 5):
		Mover.bUseDepthDependent = 0				
		Mover.fOptimize.isOptimizedForStep = 0
		Mover.fOptimize.isFirstStep = 0
		Mover.fDiffusionCoefficient = 100000
		Mover.fUncertaintyFactor = 2
	else:
		Mover.bUseDepthDependent = args[0]
		Mover.fDiffusionCoefficient = args[1]
		Mover.fUncertaintyFactor = args[2]
		

def _setWindMover(WindMover obj, *args):

	cdef WindMover_c *Mover = obj.mover
	
	if(len(args) != 14):
		Mover.fUncertainStartTime = 0
		Mover.fDuration = 3*3600
						
		Mover.fSpeedScale = 2
		Mover.fAngleScale = .4
		Mover.fMaxSpeed = 30 #mps
		Mover.fMaxAngle = 60 #degrees
		Mover.fSigma2 = 0
		Mover.fSigmaTheta = 0 
		Mover.bUncertaintyPointOpen = 0
		Mover.bSubsurfaceActive = 0
		Mover.fGamma = 1.
		
		Mover.fIsConstantWind = 1
		Mover.fConstantValue.u = 0.0
		Mover.fConstantValue.v = 0.0
		
	else:
		pass # for now

def _getRandomMove(RandomMover obj, int t, np.ndarray[LERec, ndim=1] LEs):

	cdef Random_c *Mover
	cdef WorldPoint3D wp3
	cdef LERec tRec
	cdef int i
	
	Mover = obj.mover
	
	for i in range(0, 1000):
		tRec.p.pLong = LEs[i].p.pLong
		tRec.p.pLat = LEs[i].p.pLat
		tRec.z = LEs[i].z
		wp3 = Mover.GetMove(t, 0, 0, &tRec, 0)	
		LEs[i].p.pLong = np.abs(math.fmod(LEs[i].p.pLong + wp3.p.pLong, 90))
		LEs[i].p.pLat = np.abs(math.fmod(LEs[i].p.pLat + wp3.p.pLat, 90))
		LEs[i].z = np.abs(LEs[i].z + wp3.z)			
		if i == 0:
			print 'showing wp3 long: '
			print wp3.p.pLong
			print ' done.\n'

	return (1)	

def _getWindMove(WindMover obj, t, np.ndarray[LERec, ndim=1] LEs):
	cdef WindMover_c *Mover
	cdef WorldPoint3D wp3
	cdef LERec tRec
	
	Mover = obj.mover
	
	for i in range(0, 1000):
		tRec.p.pLong = LEs[i].p.pLong
		tRec.p.pLat = LEs[i].p.pLat
		tRec.z = LEs[i].z
		tRec.dispersionStatus = LEs[i].disp
		tRec.windage = LEs[i].wind
		wp3 = Mover.GetMove(t, 0, 0, &tRec, 0)
		LEs[i].p.pLong = math.fmod(wp3.p.pLong, 90)
		LEs[i].p.pLat = math.fmod(wp3.p.pLat, 90)
		LEs[i].z = wp3.z		

	return (1)
		


# this was the very first one:

def sim(x, y, z, RandomMover obj):
	
	cdef Random_c *Mover	
	cdef WorldPoint3D wp3
	cdef LERec theLE[20]
	theLE[0].p.pLong = x
	theLE[0].p.pLat = y
	theLE[0].z = z
	
	Mover = obj.mover

	Mover.bUseDepthDependent = 0
	Mover.fOptimize.isOptimizedForStep = 0
	Mover.fOptimize.isFirstStep = 0
	Mover.fDiffusionCoefficient = 100000
	Mover.fUncertaintyFactor = 2

	for i in range(0, 20):
		theLE[i].p.pLong = random.random()*90
		theLE[i].p.pLat = random.random()*90
		theLE[i].z = 0				
		theLE[i].leKey = i
			
	timeSteps = []
	for i in range (1, 10, 2):
		timeSteps += [i]
		
	print "initial positions: \n"
	for x in range(0, 20):
		print theLE[x]	

	for t in timeSteps:
		print "step {0}:".format(t)
		for x in range(0, 5):
			wp3 = Mover.GetMove(t, 0, 0, &theLE[x], 0)	
			theLE[x].p.pLong = math.fmod(wp3.p.pLong, 90)
			theLE[x].p.pLat = math.fmod(wp3.p.pLat, 90)
			theLE[x].z = wp3.z			
			print theLE[x]

	print 'done.'
