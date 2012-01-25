/*
 *  OSSMTimeValue_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __OSSMTimeValue_b__
#define __OSSMTimeValue_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "TimeValue/TimeValue_b.h"

class OSSMTimeValue_b : virtual public TimeValue_b {
	
public:
	TimeValuePairH			timeValues;
	char 					fileName [kMaxNameLen];
	short					fFileType; //JLM
	short					fUserUnits; //JLM
	double					fScaleFactor; // user input for scaling height derivatives or hydrology files
	char 					fStationName [kMaxNameLen];
	WorldPoint				fStationPosition;
	Boolean					bOSSMStyle;
	double					fTransport;
	double					fVelAtRefPt;
	
};

#endif