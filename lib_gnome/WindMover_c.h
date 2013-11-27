/*
 *  WindMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __WindMover_c__
#define __WindMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "Mover_c.h"
#include "ExportSymbols.h"

#ifdef pyGNOME
#define TOSSMTimeValue OSSMTimeValue_c
//#define TMap Map_c
#endif

class TOSSMTimeValue;
class TMap;

class DLL_API WindMover_c : virtual public Mover_c {
	
protected:
	LONGH				fLESetSizes;		// cumulative total num le's in each set
	LEWindUncertainRecH	fWindUncertaintyList;
	void				Init();	// initializes local variables to defaults - called by constructor
	
public:
	double fSpeedScale;
	double fAngleScale;
	double fMaxSpeed;
	double fMaxAngle;
	double fSigma2;				// time dependent std for speed
	double fSigmaTheta; 		// time dependent std for angle
	Boolean bIsFirstStep;
	Seconds fModelStartTime;
	double fUncertaintyDiffusion;
	
	Boolean fIsConstantWind;
	VelocityRec fConstantValue;
	
	Boolean bTimeFileOpen;
	Boolean bUncertaintyPointOpen;
	Boolean bSubsurfaceActive;
	double	fGamma;	// fudge factor for subsurface windage
	TOSSMTimeValue *timeDep;
	
	Rect fWindBarbRect;
	Boolean bShowWindBarb;
	
	VelocityRec	current_time_value;		// AH 07/16/2012
	
#ifndef pyGNOME
	WindMover_c (TMap *owner, char* name);
#endif
	WindMover_c ();
	virtual			   ~WindMover_c ();	// move to cpp file for debugging
	virtual void		Dispose ();

	//virtual ClassID 	GetClassID () { return TYPE_WINDMOVER; }
	//virtual Boolean		IAm(ClassID id) { if(id==TYPE_WINDMOVER) return TRUE; return Mover_c::IAm(id); }
	
	virtual OSErr		AllocateUncertainty (int numLESets, int* LESetsSizesList);
	virtual OSErr		ReallocateUncertainty(int numLEs, short* statusCodes);	
	virtual void		DisposeUncertainty ();
	virtual OSErr		AddUncertainty(long setIndex,long leIndex,VelocityRec *v);
	virtual void 		UpdateUncertaintyValues(Seconds elapsedTime);
	virtual OSErr		UpdateUncertainty(const Seconds& elapsedTime, int numLESets, int* LESetsSizesList);

	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	virtual void		ModelStepIsDone();
	virtual WorldPoint3D GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	void				SetTimeDep (TOSSMTimeValue *newTimeDep); 
	TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
	void				DeleteTimeDep ();
	void				ClearWindValues (); 
	void				SetIsConstantWind (Boolean isConstantWind) { fIsConstantWind = isConstantWind; }
	OSErr				GetTimeValue(const Seconds& current_time, VelocityRec *value);
	OSErr				CheckStartTime(Seconds time);
	OSErr				get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, double* windage, short* LE_status, LEType spillType, long spillID);
};

#undef TOSSMTimeValue
//#undef TMap
#endif
