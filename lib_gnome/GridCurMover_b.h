/*
 *  GridCurMover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridCurMover_b__
#define __GridCurMover_b__

#include "Basics.h"
#include "TypeDefs.h"
#include "PtCurMover.h"

class GridCurMover_b { 

public:
	long fNumRows;
	long fNumCols;
	PtCurTimeDataHdl fTimeDataHdl;
	LoadedData fStartData; 
	LoadedData fEndData;
	short fUserUnits;
	char fPathName[kMaxNameLen];
	char fFileName[kMaxNameLen];
	Boolean fOverLap;
	Seconds fOverLapStartTime;
	PtCurFileInfoH	fInputFilesHdl;
	
	
};


enum {
	I_GRIDCURNAME = 0 ,
	I_GRIDCURACTIVE, 
	I_GRIDCURGRID, 
	I_GRIDCURARROWS,
	//I_GRIDCURSCALE,
	I_GRIDCURUNCERTAINTY,
	I_GRIDCURSTARTTIME,
	I_GRIDCURDURATION, 
	I_GRIDCURALONGCUR,
	I_GRIDCURCROSSCUR,
	//I_GRIDCURMINCURRENT
};



#endif
