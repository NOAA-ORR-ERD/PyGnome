/*
 *  TriGridVel3D_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TriGridVel3D_c__
#define __TriGridVel3D_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "TriGridVel_c.h"

typedef struct {
	double avOilConcOverSelectedTri;
	double maxOilConcOverSelectedTri;
	Seconds time;
} outputData,*outputDataP,**outputDataHdl;


typedef struct {
	long maxLayer;
	long maxTri;
	Seconds time;
} maxLayerData,*maxLayerDataP,**maxLayerDataHdl;

class TriGridVel3D_c : virtual public TriGridVel_c {

protected:
	FLOATH fDepthsH;
	Boolean	**fTriSelected;
	Boolean	**fPtsSelected;
	// maybe define a new class to handle all the output...
	outputDataHdl fOilConcHdl; 
	maxLayerDataHdl fMaxLayerDataHdl;
	double **fTriAreaHdl;
	DOUBLEH fDosageHdl;
public:
	Boolean	bShowSelectedTriangles;
	float	fPercentileForMaxConcentration;
	Boolean bCalculateDosage;
	Boolean bShowDosage;
	float fDosageThreshold;	// this may need to be an array
	long fMaxTri;	// or selected tri
	Boolean bShowMaxTri;
	
						TriGridVel3D_c();
	//virtual ClassID 	GetClassID 	() { return TYPE_TRIGRIDVEL3D; }
	void SetDepths(FLOATH depthsH){fDepthsH=depthsH;}
	FLOATH  GetDepths(){return fDepthsH;}
	void 	ScaleDepths(double scaleFactor);
	virtual double GetDepthAtPoint(WorldPoint p);
	long 	GetNumDepths(void);
	//long	GetNumTriangles(void);
	long GetNumPoints(void);
	long 	GetNumOutputDataValues(void);
	void 	GetTriangleVerticesWP(long i, WorldPoint *w);
	OSErr 	GetTriangleVerticesWP3D(long i, WorldPoint3D *w);
	OSErr 	GetTriangleVertices3D(long i, long *x, long *y, long *z);
	outputDataHdl GetOilConcHdl(){return fOilConcHdl;}
	double **GetTriAreaHdl(){return fTriAreaHdl;}
	double **GetDosageHdl(Boolean initHdl);	
	OSErr 	GetTriangleVertices(long i, long *x, long *y);
	OSErr 	GetTriangleDepths(long i, float *z);
	OSErr 	GetMaxDepthForTriangle(long triNum, double *maxDepth);
	OSErr 	GetTriangleCentroidWC(long trinum, WorldPoint *p);
	double	GetTriArea(long triNum);

	OSErr 	CalculateDepthSliceVolume(double *triVol, long triNum,float upperDepth, float lowerDepth);
	double	GetMaxAtPreviousTimeStep(Seconds time);
	void	AddToOutputHdl(double avConcOverSelectedTriangles,double maxConcOverSelectedTriangles,Seconds time);
	void 	AddToTriAreaHdl(double *triAreaArray, long numValues);
	void	ClearOutputHandles();
	Boolean FloatPointInTriangle(double x1, double y1,double x2, double y2,
								 double x3, double y3,double xp, double yp);	
	Boolean **GetTriSelection(Boolean initHdl);
	virtual void 		Dispose ();

	
};


#endif