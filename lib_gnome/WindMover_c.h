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

#ifdef pyGNOME
#define TOSSMTimeValue OSSMTimeValue_c
#define TMap Map_c
#endif

class TOSSMTimeValue;
class TMap;
class WindMover_c : virtual public Mover_c {

	
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
	double	breaking_wave_height;	// AH 06/20/2012
	double	mixed_layer_depth;
	
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

	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, const Seconds&, bool); // AH 04/16/12
	virtual WorldPoint3D 	GetMove (Seconds model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	void				SetTimeDep (TOSSMTimeValue *newTimeDep) { timeDep = newTimeDep; }
	TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
	void				DeleteTimeDep ();
	void				ClearWindValues (); 
	void				SetIsConstantWind (Boolean isConstantWind) { fIsConstantWind = isConstantWind; }
	OSErr				GetTimeValue(Seconds time, VelocityRec *value);
	OSErr				CheckStartTime(Seconds time);
	OSErr				get_move(int, long, long, char *, char *, char *, char *, double, double, double, double, char *, char*, int);
	OSErr				get_move(int, long, long, char *, char *, char *, char *, double, double, char*, int);

};

#undef TOSSMTimeValue
#undef TMap
#endif
