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


//class ComponentMover_c : virtual public CurrentMover_c {
class DLL_API ComponentMover_c : virtual public CurrentMover_c {

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
#ifndef pyGNOME
	TimeValuePairH	fAveragedWindsHdl;
#else
	VelocityRec		fAveragedWindVelocity;
#endif
	
	//							optimize fields don't need to be saved
	TC_OPTIMZE			fOptimize;
	
	long				timeMoverCode;
	char 				windMoverName [64]; 	// file to match at refP
	
#ifndef pyGNOME
	ComponentMover_c (TMap *owner, char *name);
#endif
	ComponentMover_c ();
	virtual			   ~ComponentMover_c () { Dispose (); }
	virtual void		Dispose ();
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	virtual void 		ModelStepIsDone();
	OSErr				SetOptimizeVariables (char *errmsg, const Seconds& model_time, const Seconds& time_step);
#ifndef pyGNOME
	OSErr				CalculateAveragedWindsHdl(char *errmsg);
	OSErr				GetAveragedWindValue(Seconds time, const Seconds& time_step, VelocityRec *avValue);
#else
	OSErr 				CalculateAveragedWindsVelocity(const Seconds& model_time, char *errmsg);
#endif
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep);

	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual	Boolean 		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);	

	void				SetRefPosition (WorldPoint p) { refP = p;}
	void				GetRefPosition (WorldPoint *p) { (*p) = refP;}

	void				SetTimeFile (TOSSMTimeValue *newTimeFile);
	TOSSMTimeValue		*GetTimeFile () { return (timeFile); }
	
	//virtual	OSErr TextRead(vector<string> &linesInFile);
#ifdef pyGNOME
	virtual	OSErr TextRead(char* catsPath1, char* catsPath2);
#endif
	OSErr get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID);
};

#undef TCATSMover
#undef TOSSMTimeValue
#undef TComponentMover
#endif
