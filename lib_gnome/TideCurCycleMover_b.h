/*
 *  TideCurCycleMover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TideCurCycleMover_b__
#define __TideCurCycleMover_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "CATSMover_b.h"
#include "PtCurMover.h"

class TideCurCycleMover_b : virtual public CATSMover_b {

public:
	//long fNumRows;
	//long fNumCols;
	//PtCurTimeDataHdl fTimeDataHdl;
	Seconds **fTimeHdl;
	LoadedData fStartData; 
	LoadedData fEndData;
	float fFillValue;
	float fDryValue;
	short fUserUnits;
	char fPathName[kMaxNameLen];
	char fFileName[kMaxNameLen];
	LONGH fVerdatToNetCDFH;		// these two fields will be in curvilinear if we extend there
	WORLDPOINTFH fVertexPtsH;	// may not need this if set pts in dagtree	
	long fNumNodes;
	short fPatternStartPoint;	// maxflood, maxebb, etc
	float fTimeAlpha;
	char fTopFilePath[kMaxNameLen];
	
};

#endif
