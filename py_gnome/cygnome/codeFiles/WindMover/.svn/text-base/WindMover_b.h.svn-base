/*
 *  WindMover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __WindMover_b__
#define __WindMover_b__

#include "Mover_b.h"

class TOSSMTimeValue;

class WindMover_b : virtual public Mover_b {

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
	
};

#endif