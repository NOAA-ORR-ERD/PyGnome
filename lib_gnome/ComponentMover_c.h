/*
 *  ComponentMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ComponentMover_c__
#define __ComponentMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "CurrentMover_c.h"

#ifdef pyGNOME
#define TCATSMover CATSMover_c
#define TOSSMTimeValue OSSMTimeValue_c
#define TComponentMover ComponentMover_c
#endif

class TCATSMover;
class TOSSMTimeValue;
class TComponentMover;


class ComponentMover_c : virtual public CurrentMover_c {

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
	
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, const Seconds&, bool); // AH 04/16/12
	virtual void 		ModelStepIsDone();
	OSErr				SetOptimizeVariables (char *errmsg);
	OSErr				CalculateAveragedWindsHdl(char *errmsg);
	OSErr				GetAveragedWindValue(Seconds time, VelocityRec *avValue);
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep);

	virtual WorldPoint3D 	GetMove (Seconds model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual	Boolean 		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);	
	
};

#undef TCATSMover
#undef TOSSMTimeValue
#undef TComponentMover
#endif
