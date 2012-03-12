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

//++ The following have been moved from GenUtil,


float max4(float f1, float f2, float f3, float f4)
{
	float x = _max(f1, f2), y = _max(f3, f4);
	
	return _max(x, y);
}

float min4(float f1, float f2, float f3, float f4)
{
	float x = _min(f1, f2), y = _min(f3, f4);
	
	return _min(x, y);
}

double logB(double b, double x)
{
	return log10(x) / log10(b);
}

float hypotenuse(float a, float b)
{
	return sqrt(a*a + b*b);
}

double myfabs(double x)
{
	return (x < 0) ? -x : x;
}


void SetSign(FLOATPTR n, short code)
{
	if (*n == -0.0)
		*n = 0.0;
	else {
		*n = abs(*n);
		if (code == 2) *n = -*n;
	}
}

short ScaleToShort(long n)
{
	if (n < -32000) return -32000;
	if (n > 32000) return 32000;
	return n;
}

long GetRandom(long low, long high)
{
	float scale, n;
	
	scale = (float)(high - low) / (float)RAND_MAX;
	
	n = low + rand() * scale;
	
	return (long)n;
}

float GetRandomFloat(float low, float high)
{
	float scale, n;
	
	scale = (float)(high - low) / (float)RAND_MAX;
	
	n = low + rand() * scale;
	
	return n;
}

void GetRandomVectorInUnitCircle(float *u,float *v)
{	// JLM 9/11/98
	do
	{
		*u = GetRandomFloat(-1.0,1.0);
		*v = GetRandomFloat(-1.0,1.0);
	} while ( (*u)*(*u) +  (*v)*(*v) > 1.0);
}


char *SwapN(char *s, short n)
{
	char c;
	short i;
	
	for (i = 0 ; i < n / 2 ; i++) {
		c = s[i];
		s[i] = s[n - (i + 1)];
		s[n - (i + 1)] = c;
	}
	
	return s;
}

// Given an integer key, a list of pairs of integers, and the
// number of pairs in the list, return the integer paired with the key,
// or 0 if the key is not in the list.
long Assoc(long key, LONGPTR table, short n)
{
	short i;
	
	for (i = 0 ; i < n ; i++)
		if (table[i * 2] == key)
			return table[i * 2 + 1];
	
	return 0;
}

void SwitchShorts(SHORTPTR a, SHORTPTR b)
{
	short temp = *a;
	
	*a = *b;
	*b = temp;
}

void SwitchLongs(LONGPTR a, LONGPTR b)
{
	long temp = *a;
	
	*a = *b;
	*b = temp;
}

void SwitchStrings(CHARPTR a, CHARPTR b)
{
	char temp[256];
	
	strcpy(temp, a);
	strcpy(a, b);
	strcpy(b, temp);
}

short NumDecimals(CHARPTR str)
{
	char		str2[256];
	short		i, c = 0;
	
	strcpy(str2, str);
	i = strlen(str2) - 1;
	while (str2[i] == '0')
		str2[i--] = 0;
	
	while (i >= 0) {
		if (str2[i--] == '.')
			return c;
		c++;
	}
	
	return 0;
}

Boolean EarlierThan(Seconds time1, Seconds time2)
{
	return ((unsigned long)time1) < ((unsigned long)time2);
}

Boolean LaterThan(Seconds time1, Seconds time2)
{
	return ((unsigned long)time1) > ((unsigned long)time2);
}