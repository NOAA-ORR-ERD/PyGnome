/*
 *  ComponentMover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ComponentMover_b__
#define __ComponentMover_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "CurrentMover/CurrentMover_b.h"


class TCATSMover;
class TOSSMTimeValue;
class TComponentMover;

class ComponentMover_b : virtual public CurrentMover_b {
	
public:
	TCATSMover			*pattern1;
	TCATSMover			*pattern2;
	Boolean				bPat1Open;
	Boolean				bPat2Open;
	TOSSMTimeValue		*timeFile;
	
	WorldPoint			refP;
	Boolean 			bRefPointOpen;
	
	double				pat1Angle;
	double				pat2Angle;
	
	double				pat1Speed;
	double				pat2Speed;
	
	long				pat1SpeedUnits;
	long				pat2SpeedUnits;
	
	double				pat1ScaleToValue;
	double				pat2ScaleToValue;
	
	long				scaleBy;
	
	Boolean			bUseAveragedWinds;
	Boolean			bExtrapolateWinds;
	Boolean			bUseMainDialogScaleFactor;
	double			fScaleFactorAveragedWinds;
	double			fPowerFactorAveragedWinds;
	long				fPastHoursToAverage;
	TimeValuePairH	fAveragedWindsHdl;
	
	//							optimize fields don't need to be saved
	TC_OPTIMZE			fOptimize;
	
	long				timeMoverCode;
	char 				windMoverName [64]; 	// file to match at refP
	
};



#endif
