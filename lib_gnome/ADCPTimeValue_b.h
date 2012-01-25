/*
 *  ADCPTimeValue_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADCPTimeValue_b__
#define __ADCPTimeValue_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "TimeValue/TimeValue_b.h"

class ADCPTimeValue_b : virtual public TimeValue_b {

public:
	TimeValuePairH3D			timeValues;
	char 					fileName [kMaxNameLen];
	short					fFileType; //JLM
	short					fUserUnits; //JLM
	double					fScaleFactor; // user input for scaling height derivatives or hydrology files
	char 					fStationName [kMaxNameLen];
	WorldPoint				fStationPosition;	// will need a handle for a set of positions and a set of depths for each
	double					fStationDepth;
	long					fNumBins;
	double					fBinSize;
	long					fGMTOffset;
	short					fSensorOrientation;
	//Boolean					bOSSMStyle;
	Boolean					bStationPositionOpen;
	Boolean					bStationDataOpen;
	DOUBLEH					fBinDepthsH;

	virtual void 			GetTimeFileName (char *theName) { strcpy (theName, fileName); }
	virtual void 			SetTimeFileName (char *theName) { strcpy (fileName, theName); }
	virtual void 			GetStationName (char *theName) { strcpy (theName, fStationName); }
	virtual void 			SetStationName (char *theName) { strcpy (fStationName, theName); }
	virtual short			GetUserUnits(){return fUserUnits;}
	virtual void			SetUserUnits(short userUnits){fUserUnits=userUnits;}
	virtual short			GetFileType	() { return fFileType; }
	virtual void			SetFileType	(short fileType) { fFileType = fileType; }
	long					GetNumBins(){return fNumBins; }
	double					GetBinSize(){return fBinSize; }
	long					GetSensorOrientation(){return fSensorOrientation; }
	double					GetStationDepth(){return fStationDepth; }
	WorldPoint				GetStationPosition(){return fStationPosition; }
	

};

#endif
