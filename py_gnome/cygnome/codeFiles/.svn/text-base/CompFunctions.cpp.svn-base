/*
 *  TimeFunctions.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#include "CROSS.H"

double UorV(VelocityRec vector, short index)
{
	return (index == 0) ? vector.u : vector.v;
}

double UorV(VelocityRec3D vector, short index)
{
	return (index == 0) ? vector.u : vector.v;
}

double Hermite(double v1, double s1, double t1,
			   double v2, double s2, double t2,
			   double time)
{
	double h, dt, x;
	
	// hermite fit function
	// v1 and v2 are the two values
	// s1 and s2 are the two slopes
	// t1 and t2 are the two times
	// time is the time for the interpolation
	// return value is interpolated value at time
	
	dt = t2 - t1;
	if (dt < 0.0) dt = -dt;
	
	if (dt < TIMEVALUE_TOLERANCE) return v2;
	
	x = (time - t1) / (t2 - t1);
	s1 = s1* (t2 - t1);
	s2 = s2* (t2 - t1);
	
	h = (   2.0 * v1 - 2.0 * v2 +       s1 + s2) * x * x * x
	+ ( - 3.0 * v1 + 3.0 * v2 - 2.0 * s1 - s2) * x * x
	+ (                       +       s1     ) * x
	+ (         v1                           ) * 1.0;
	
	return h;
}

// ************************************************************
void Hermite(double v1,     // value at t1
             double s1,     // slope at t1
			 double t1,     // time t1
			 double v2,     // value at t2
			 double s2,     // slope at t2
			 double t2,     // time t2
			 double theTime,   // time for interpolation
			 double *vTime) // returns value at time
{	
	// Hermite interpolation function taken from OSSM
	double x,dt;
	/**************************************************************/
	//
	// check to make sure t1 and t2 are not the same
	//
	dt = t2 - t1;
	if( dt<0.000001){
		*vTime = v2;
		return;
	}
	//
	dt = t2-t1;
	x = (theTime-t1) / dt;
	s1 = s1 * dt;
	s2 = s2 * dt;
	
	// OK here is the 4 by 4
	
	*vTime = ( 2.0 * v1 - 2.0*v2 +    s1 + s2) * x*x*x
	+(-3.0 * v1 + 3.0*v2 -2.0*s1 - s2) * x*x
	+(                        s1     ) * x
	+(       v1                      );
	return;
	
	// note in the OSSM code Gary sets back the values of s1 and s2
	// we don't do that in C because those suckers are passed as values
}