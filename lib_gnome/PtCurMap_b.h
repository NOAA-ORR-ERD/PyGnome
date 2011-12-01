/*
 *  PtCurMap_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __PtCurMap_b__
#define __PtCurMap_b__

#include "Earl.h"
#include "TypeDefs.h"

class PtCurMap_b : virtual public Map_b {
	
public:
	LONGH			fBoundarySegmentsH;
	LONGH			fBoundaryTypeH;		// 1 land, 2 water
	LONGH			fBoundaryPointsH;	// for curvilinear grids
	LONGH			fSegSelectedH;
	LONGH			fSelectedBeachHdl;	//not sure if both are needed
	LONGH			fSelectedBeachFlagHdl;	//not sure if both are needed
	#ifdef IBM
	HDIB			fWaterBitmap;
	HDIB			fLandBitmap;
	#else
	BitMap			fWaterBitmap; 
	BitMap			fLandBitmap; 
	#endif
	Boolean			bDrawLandBitMap;
	Boolean			bDrawWaterBitMap;
	Boolean			bShowSurfaceLEs;

	public:
	float			fContourDepth1;
	float			fContourDepth2;
	float			fContourDepth1AtStartOfRun;
	float			fContourDepth2AtStartOfRun;
	float			fBottomRange;
	DOUBLEH			fContourLevelsH;
	Rect			fLegendRect;
	Boolean			bShowLegend;
	short			fDiagnosticStrType;		// 0 no diagnostic string, 1 tri area, 2 num LEs, 3 conc levels, 4 depths, 5 subsurface particles
	Boolean			bDrawContours;

	long			fWaterDensity;
	double			fMixedLayerDepth;
	double			fBreakingWaveHeight;

	double			*fTriAreaArray;
	//long			*fDepthSliceArray;	// number of LEs in each layer (1m) of depth slice
	float			*fDepthSliceArray;	//changed to ppm in each layer (1m) of depth slice 7/21/03

	Boolean			bUseSmoothing;
	Boolean			bUseLineCrossAlgorithm;
	float			fMinDistOffshore;	// set how far LEs reflect off shoreline, so they don't get stuck
	//Boolean			bShowElapsedTime;	// should be a model field
	short			fWaveHtInput;	// 0 from wind speed, 1 breaking wave height measure, 2 significant wave height measure
	DropletInfoRecH	fDropletSizesH;

	Boolean			bTrackAllLayers;

};

#endif