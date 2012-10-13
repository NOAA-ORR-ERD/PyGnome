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
#define TMap Map_c
#endif

class TOSSMTimeValue;
class TMap;

class GNOMEDLL_API WindMover_c : virtual public Mover_c {
	
protected:
	LONGH				fLESetSizes;		// cumulative total num le's in each set
	LEWindUncertainRecH	fWindUncertaintyList;
	
public:
	double fSpeedScale;
	double fAngleScale;
	double fMaxSpeed;
	double fMaxAngle;
	double fSigma2;				// time dependent std for speed
	double fSigmaTheta; 		// time dependent std for angle
	
	Boolean fIsConstantWind;
	VelocityRec fConstantValue;
	
	Boolean bTimeFileOpen;
	Boolean bUncertaintyPointOpen;
	Boolean bSubsurfaceActive;
	double	fGamma;	// fudge factor for subsurface windage
	TOSSMTimeValue *timeDep;
	
	Rect fWindBarbRect;
	Boolean bShowWindBarb;
	
	Seconds tap_offset;		// AH 06/20/2012
	VelocityRec	current_time_value;		// AH 07/16/2012
	
	WindMover_c (TMap *owner, char* name);
	WindMover_c ();
	virtual ClassID 	GetClassID () { return TYPE_WINDMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_WINDMOVER) return TRUE; return Mover_c::IAm(id); }
	
#ifndef pyGNOME
	virtual OSErr		AllocateUncertainty ();
#endif
	
	virtual void		DisposeUncertainty ();
	virtual OSErr		AddUncertainty(long setIndex,long leIndex,VelocityRec *v);
	virtual void 		UpdateUncertaintyValues(Seconds elapsedTime);
	virtual OSErr		UpdateUncertainty(void);

	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool); // AH 07/10/2012
	virtual WorldPoint3D GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	void				SetTimeDep (TOSSMTimeValue *newTimeDep); 
	TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
	void				DeleteTimeDep ();
	void				ClearWindValues (); 
	void				SetIsConstantWind (Boolean isConstantWind) { fIsConstantWind = isConstantWind; }
	OSErr				GetTimeValue(const Seconds& current_time, VelocityRec *value);
	OSErr				CheckStartTime(Seconds time);
	OSErr				get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, double* LE_windage, short* LE_status, short LE_type);
};

#undef TOSSMTimeValue
#undef TMap
#endif
