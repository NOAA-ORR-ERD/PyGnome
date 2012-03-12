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

#include "Basics.h"
#include "TypeDefs.h"
#include "TimeValue_b.h"

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
	
	virtual void 			GetTimeFileName (char *theName) { strcpy (theName, fileName); }
	virtual short			GetFileType	() { if (fFileType == PROGRESSIVETIDEFILE) return SHIOHEIGHTSFILE; else return fFileType; }
	virtual void			SetFileType	(short fileType) { fFileType = fileType; }

};

#endif
