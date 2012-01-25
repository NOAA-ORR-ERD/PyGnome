/*
 *  TOSSMTimeValue.cpp
 *  gnome
 *
 *  Created by Alex Hadjilambris on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "TOSSMTimeValue.h"

TOSSMTimeValue::TOSSMTimeValue(TMover *theOwner,TimeValuePairH tvals,short userUnits) : TTimeValue(theOwner) 
{ 
	fileName[0]=0;
	timeValues = tvals;
	fUserUnits = userUnits;
	fFileType = OSSMTIMEFILE;
	fScaleFactor = 0.;
	fStationName[0] = 0;
	fStationPosition.pLat = 0;
	fStationPosition.pLong = 0;
	bOSSMStyle = true;
	fTransport = 0;
	fVelAtRefPt = 0;
}


TOSSMTimeValue::TOSSMTimeValue(TMover *theOwner) : TTimeValue(theOwner) 
{ 
	fileName[0]=0;
	timeValues = 0;
	fUserUnits = kUndefined; 
	fFileType = OSSMTIMEFILE;
	fScaleFactor = 0.;
	fStationName[0] = 0;
	fStationPosition.pLat = 0;
	fStationPosition.pLong = 0;
	bOSSMStyle = true;
	fTransport = 0;
	fVelAtRefPt = 0;
}

void TOSSMTimeValue::Dispose()
{
	if (timeValues)
	{
		DisposeHandle((Handle)timeValues);
		timeValues = 0;
	}
	
	TTimeValue::Dispose();
}