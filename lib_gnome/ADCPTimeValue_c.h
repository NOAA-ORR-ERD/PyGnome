/*
 *  ADCPTimeValue_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADCPTimeValue_c__
#define __ADCPTimeValue_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "TimeValue_c.h"

class ADCPTimeValue_c : virtual public TimeValue_c {

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
	virtual OSErr			GetTimeValue(const Seconds& start_time, const Seconds& end_time, const Seconds& current_time, VelocityRec *value);
	virtual OSErr			GetTimeValueAtDepth (long depthIndex, Seconds time, VelocityRec *value);
	double					GetBinDepth(long depthIndex);
	virtual OSErr			CheckStartTime (Seconds time);
	virtual void			RescaleTimeValues (double oldScaleFactor, double newScaleFactor);
	OSErr					GetDepthIndices(float depthAtPoint, float totalDepth, long *depthIndex1, long *depthIndex2);
	virtual long			GetNumValues ();
	virtual TimeValuePairH3D	GetTimeValueHandle () { return timeValues; }
	virtual void			SetTimeValueHandle (TimeValuePairH3D t) ;	
	virtual double			GetMaxValue();

protected:
	OSErr					GetInterpolatedComponent (Seconds forTime, double *value, short index);
	OSErr					GetInterpolatedComponentAtDepth (long depthIndex, Seconds forTime, double *value, short index);
	OSErr					GetTimeChange (long a, long b, Seconds *dt);
	

};


#endif
