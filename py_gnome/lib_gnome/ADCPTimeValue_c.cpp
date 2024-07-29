/*
 *  ADCPTimeValue_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "ADCPTimeValue_c.h"
#include "CROSS.H"

double UorV(VelocityRec vector, short index);
double UorV(VelocityRec3D vector, short index);

long ADCPTimeValue_c::GetNumValues()
{
	return timeValues == 0 ? 0 : _GetHandleSize((Handle)timeValues)/sizeof(TimeValuePair3D);
}


double ADCPTimeValue_c::GetMaxValue()
{
	long i,numValues = GetNumValues();
	TimeValuePair3D tv;
	double maxval = -1,val;
	for(i=0;i<numValues;i++)
	{
		tv=(*timeValues)[i];
		val = sqrt(tv.value.v * tv.value.v + tv.value.u * tv.value.u);
		if(val > maxval)maxval = val;
	}
	return maxval; // JLM
}

double ADCPTimeValue_c::GetBinDepth(long depthIndex)
{
	double binDepth = 0.;
	if (depthIndex < 0 || depthIndex > fNumBins - 1) return 0.;
	if (fBinDepthsH)
		binDepth = INDEXH(fBinDepthsH,depthIndex);	
	return binDepth;
}

OSErr ADCPTimeValue_c::GetDepthIndices(float depthAtPoint, float totalDepth, long *depthIndex1, long *depthIndex2)
{
	long i;
	OSErr err = 0;
	
	if (!fBinDepthsH) 
	{
		*depthIndex1 = UNASSIGNEDINDEX;
		*depthIndex2 = UNASSIGNEDINDEX;
		return -1;
	}
	// should also look at stationDepth -  don't try to use bins that are below stationDepth
	if (fSensorOrientation == 2)	// downward looking, bins go from top to bottom
	{
		if (depthAtPoint == 0 || totalDepth == 0 || depthAtPoint <= INDEXH(fBinDepthsH,0)) 
		{	// use top value
			*depthIndex1 = 0;
			*depthIndex2 = UNASSIGNEDINDEX;
			return err;
		}
		for (i=0;i<fNumBins-1;i++)
		{
			if (depthAtPoint > INDEXH(fBinDepthsH,i) && depthAtPoint < INDEXH(fBinDepthsH,i+1))
			{
				*depthIndex1 = i;
				*depthIndex2 = i+1;
				return err;
			}
		}
		if (depthAtPoint>=INDEXH(fBinDepthsH,fNumBins-1))
		{	// use bottom value
			*depthIndex1 = fNumBins-1;
			*depthIndex2 = UNASSIGNEDINDEX;
			return err;
		}
	}
	else if (fSensorOrientation == 1)	// upward looking, bins go from bottom to top
	{
		if (depthAtPoint == 0 || totalDepth == 0 || depthAtPoint <= INDEXH(fBinDepthsH,fNumBins-1)) 
		{	// use top value
			*depthIndex1 = fNumBins-1;
			*depthIndex2 = UNASSIGNEDINDEX;
			return err;
		}
		for (i=fNumBins-1;i>0;i--)
		{
			if (depthAtPoint > INDEXH(fBinDepthsH,i) && depthAtPoint < INDEXH(fBinDepthsH,i-1))
			{
				*depthIndex1 = i;
				*depthIndex2 = i-1;
				return err;
			}
		}
		if (depthAtPoint>=INDEXH(fBinDepthsH,0))
		{	// use bottom value
			*depthIndex1 = 0;
			*depthIndex2 = UNASSIGNEDINDEX;
			return err;
		}
	}	
	else 
		return -1;
	return 0;
	
}

OSErr ADCPTimeValue_c::GetTimeChange(long a, long b, Seconds *dt)
{
	// NOTE: Must be called with a < b, else bogus value may be returned.
	
	(*dt) = INDEXH(timeValues, b).time - INDEXH(timeValues, a).time;
	
	if (*dt == 0)
	{	// better error message, JLM 4/11/01 
		// printError("Duplicate times in time/value table."); return -1; 
		char msg[256];
		char timeS[128];
		DateTimeRec time;
		char* p;
		SecondsToDate (INDEXH(timeValues, a).time, &time);
		Date2String(&time, timeS);
		if (p = strrchr(timeS, ':')) p[0] = 0; // remove seconds
		sprintf(msg,"Duplicate times in time/value table.%s%s%s",NEWLINESTRING,timeS,NEWLINESTRING);
		SecondsToDate (INDEXH(timeValues, b).time, &time);
		Date2String(&time, timeS);
		if (p = strrchr(timeS, ':')) p[0] = 0; // remove seconds
		strcat(msg,timeS);
		printError(msg); return -1; 
	}
	
	return 0;
}



OSErr ADCPTimeValue_c::GetInterpolatedComponent(Seconds forTime, double *value, short index)
{
	Boolean linear = FALSE;
	long a, b, i, n = GetNumValues();	// divide by numBins..., also may store start/end time
	double dv, slope, slope1, slope2, intercept;
	Seconds dt;
	Boolean useExtrapolationCode = false;
	long startIndex,midIndex,endIndex;
	OSErr err = 0;
	
	// interpolate value from timeValues array
	n = n / fNumBins;
	// only one element => values are constant
	if (n == 1) { *value = UorV(INDEXH(timeValues, 0).value, index); return 0; }
	
	// only two elements => use linear interopolation
	if (n == 2) { a = 0; b = 1; linear = TRUE; }
	
	if (forTime < INDEXH(timeValues, 0).time) 
	{	// before first element
		if(useExtrapolationCode)
		{ 	// old method
			a = 0; b = 1; linear = TRUE;  //  => use slope to extrapolate 
		}
		else
		{
			// new method  => use first value,  JLM 9/16/98
			*value = UorV(INDEXH(timeValues, 0).value, index); return 0; 
		}
	}
	
	if (forTime > INDEXH(timeValues, n - 1).time) 
	{	// after last element
		if(useExtrapolationCode)
		{ 	// old method
			a = n - 2; b = n - 1; linear = TRUE; //  => use slope to extrapolate 
		}
		else
		{	// new method => use last value,  JLM 9/16/98
			*value = UorV(INDEXH(timeValues, n-1).value, index); return 0; 
		}
	}
	
	if (linear) {
		if (err = GetTimeChange(a, b, &dt)) return err;
		
		dv = UorV(INDEXH(timeValues, b).value, index) - UorV(INDEXH(timeValues, a).value, index);
		slope = dv / dt;
		intercept = UorV(INDEXH(timeValues, a).value, index) - slope * INDEXH(timeValues, a).time;
		(*value) = slope * forTime + intercept;
		
		return 0;
	}
	
	// find before and after elements
	
	/////////////////////////////////////////////////
	// JLM 7/21/00, we need to speed this up for when we have a lot of values
	// code goes here, (should we use a static to remember a guess of where to start) before we do the binary search ?
	// use a binary method 
	startIndex = 0;
	endIndex = n-1;
	while(endIndex - startIndex > 3)
	{
		midIndex = (startIndex+endIndex)/2;
		if (forTime <= INDEXH(timeValues, midIndex).time)
			endIndex = midIndex;
		else
			startIndex = midIndex;
	}
	/////////////////////////////////////////
	
	
	for (i = startIndex; i < n; i++) {
		if (forTime <= INDEXH(timeValues, i).time) {
			dt = INDEXH(timeValues, i).time - forTime;
			if (dt <= TIMEVALUE_TOLERANCE)
			{ (*value) = UorV(INDEXH(timeValues, i).value, index); return 0; } // found match
			
			a = i - 1;
			b = i;
			break;
		}
	}
	
	dv = UorV(INDEXH(timeValues, b).value, index) - UorV(INDEXH(timeValues, a).value, index);
	if (fabs(dv) < TIMEVALUE_TOLERANCE) // check for constant value
	{ (*value) = UorV(INDEXH(timeValues, b).value, index); return 0; }
	
	if (err = GetTimeChange(a, b, &dt)) return err;
	
	// interpolated value is between positions a and b
	
	// compute slopes before using Hermite()
	
	if (b == 1) { // special case: between first two elements
		slope1 = dv / dt;
		dv = UorV(INDEXH(timeValues, 2).value, index) - UorV(INDEXH(timeValues, 1).value, index);
		if (err = GetTimeChange(1, 2, &dt)) return err;
		slope2 = dv / dt;
		slope2 = 0.5 * (slope1 + slope2);
	}
	
	else if (b ==  n - 1) { // special case: between last two elements
		slope2 = dv / dt;
		dv = UorV(INDEXH(timeValues, n - 2).value, index) - UorV(INDEXH(timeValues, n - 3).value, index);
		if (err = GetTimeChange(n - 3, n - 2, &dt)) return err;
		slope1 = dv / dt;
		slope1 = 0.5 * (slope1 + slope2);
	}
	
	else { // general case
		slope = dv / dt;
		dv = UorV(INDEXH(timeValues, b + 1).value, index) - UorV(INDEXH(timeValues, b).value, index);
		if (err = GetTimeChange(b, b + 1, &dt)) return err;
		slope2 = dv / dt;
		dv = UorV(INDEXH(timeValues, a).value, index) - UorV(INDEXH(timeValues, a - 1).value, index);
		if (err = GetTimeChange(a, a - 1, &dt)) return err;
		slope1 = dv / dt;
		slope1 = 0.5 * (slope1 + slope);
		slope2 = 0.5 * (slope2 + slope);
	}
	
	// if (v1 == v2) newValue = v1;
	
	(*value) = Hermite(UorV(INDEXH(timeValues, a).value, index), slope1, INDEXH(timeValues, a).time,
					   UorV(INDEXH(timeValues, b).value, index), slope2, INDEXH(timeValues, b).time, forTime);
	
	return 0;
}

OSErr ADCPTimeValue_c::GetInterpolatedComponentAtDepth(long depthIndex, Seconds forTime, double *value, short index)
{
	Boolean linear = FALSE;
	long a, b, i, n = GetNumValues();	// divide by numBins..., also may store start/end time
	double dv, slope, slope1, slope2, intercept;
	Seconds dt;
	Boolean useExtrapolationCode = false;
	long startIndex,midIndex,endIndex, valuesToSkip = 0;
	OSErr err = 0;
	
	// interpolate value from timeValues array
	n = n / fNumBins;
	valuesToSkip = depthIndex*n;
	// only one element => values are constant
	if (n == 1) { *value = UorV(INDEXH(timeValues, valuesToSkip).value, index); return 0; }
	
	// only two elements => use linear interopolation
	if (n == 2) { a = 0; b = 1; linear = TRUE; }
	
	if (forTime < INDEXH(timeValues, valuesToSkip).time) 
	{	// before first element
		if(useExtrapolationCode)
		{ 	// old method
			a = 0; b = 1; linear = TRUE;  //  => use slope to extrapolate 
		}
		else
		{
			// new method  => use first value,  JLM 9/16/98
			*value = UorV(INDEXH(timeValues, valuesToSkip).value, index); return 0; 
		}
	}
	
	if (forTime > INDEXH(timeValues, valuesToSkip + n - 1).time) 
	{	// after last element
		if(useExtrapolationCode)
		{ 	// old method
			a = valuesToSkip+ n - 2; b = valuesToSkip + n - 1; linear = TRUE; //  => use slope to extrapolate 
		}
		else
		{	// new method => use last value,  JLM 9/16/98
			*value = UorV(INDEXH(timeValues, valuesToSkip + n-1).value, index); return 0; 
		}
	}
	
	if (linear) {
		if (err = GetTimeChange(valuesToSkip+a, valuesToSkip+b, &dt)) return err;
		
		dv = UorV(INDEXH(timeValues, valuesToSkip+b).value, index) - UorV(INDEXH(timeValues, valuesToSkip+a).value, index);
		slope = dv / dt;
		intercept = UorV(INDEXH(timeValues,valuesToSkip+ a).value, index) - slope * INDEXH(timeValues, valuesToSkip+a).time;
		(*value) = slope * forTime + intercept;
		
		return 0;
	}
	
	// find before and after elements
	
	/////////////////////////////////////////////////
	// JLM 7/21/00, we need to speed this up for when we have a lot of values
	// code goes here, (should we use a static to remember a guess of where to start) before we do the binary search ?
	// use a binary method 
	startIndex = 0+valuesToSkip;
	endIndex = n-1+valuesToSkip;
	while(endIndex - startIndex > 3)
	{
		midIndex = (startIndex+endIndex)/2;
		if (forTime <= INDEXH(timeValues, midIndex).time)
			endIndex = midIndex;
		else
			startIndex = midIndex;
	}
	/////////////////////////////////////////
	
	
	for (i = startIndex; i < n+valuesToSkip; i++) {
		if (forTime <= INDEXH(timeValues, i).time) {
			dt = INDEXH(timeValues, i).time - forTime;
			if (dt <= TIMEVALUE_TOLERANCE)
			{ (*value) = UorV(INDEXH(timeValues, i).value, index); return 0; } // found match
			
			a = i - 1;
			b = i;
			break;
		}
	}
	
	dv = UorV(INDEXH(timeValues, b).value, index) - UorV(INDEXH(timeValues, a).value, index);
	if (fabs(dv) < TIMEVALUE_TOLERANCE) // check for constant value
	{ (*value) = UorV(INDEXH(timeValues, b).value, index); return 0; }
	
	if (err = GetTimeChange(a, b, &dt)) return err;
	
	// interpolated value is between positions a and b
	
	// compute slopes before using Hermite()
	
	if (b == (valuesToSkip+1)) { // special case: between first two elements
		slope1 = dv / dt;
		dv = UorV(INDEXH(timeValues, valuesToSkip+2).value, index) - UorV(INDEXH(timeValues, valuesToSkip+1).value, index);
		if (err = GetTimeChange(valuesToSkip+1, valuesToSkip+2, &dt)) return err;
		slope2 = dv / dt;
		slope2 = 0.5 * (slope1 + slope2);
	}
	
	else if (b ==  valuesToSkip + n - 1) { // special case: between last two elements
		slope2 = dv / dt;
		dv = UorV(INDEXH(timeValues, valuesToSkip + n - 2).value, index) - UorV(INDEXH(timeValues, valuesToSkip + n - 3).value, index);
		if (err = GetTimeChange(valuesToSkip+ n - 3, valuesToSkip + n - 2, &dt)) return err;
		slope1 = dv / dt;
		slope1 = 0.5 * (slope1 + slope2);
	}
	
	else { // general case
		slope = dv / dt;
		dv = UorV(INDEXH(timeValues, b + 1).value, index) - UorV(INDEXH(timeValues, b).value, index);
		if (err = GetTimeChange(b, b + 1, &dt)) return err;
		slope2 = dv / dt;
		dv = UorV(INDEXH(timeValues, a).value, index) - UorV(INDEXH(timeValues, a - 1).value, index);
		if (err = GetTimeChange(a, a - 1, &dt)) return err;
		slope1 = dv / dt;
		slope1 = 0.5 * (slope1 + slope);
		slope2 = 0.5 * (slope2 + slope);
	}
	
	// if (v1 == v2) newValue = v1;
	
	(*value) = Hermite(UorV(INDEXH(timeValues, a).value, index), slope1, INDEXH(timeValues, a).time,
					   UorV(INDEXH(timeValues, b).value, index), slope2, INDEXH(timeValues, b).time, forTime);
	
	return 0;
}

void ADCPTimeValue_c::SetTimeValueHandle(TimeValuePairH3D t)
{
	if(timeValues && t != timeValues)DisposeHandle((Handle)timeValues);
	timeValues=t;
}

OSErr ADCPTimeValue_c::CheckStartTime(Seconds forTime)
{
	OSErr err = 0;
	long a, b, i, n = GetNumValues();
	if (!timeValues || _GetHandleSize((Handle)timeValues) == 0)
	{
		//TechError("ADCPTimeValue::GetTimeValue()", "timeValues", 0); 
		// no value to return
		//	value->u = 0;
		//	value->v = 0;
		return -1; 
	}
	
	// only one element => values are constant
	if (n == 1) return -2;/*{ *value = UorV(INDEXH(timeValues, 0).value, index); return 0; }*/
	
	// only two elements => use linear interpolation
	//	if (n == 2) { a = 0; b = 1; linear = TRUE; }	// may want warning here
	
	if (forTime < INDEXH(timeValues, 0).time) 
	{	// before first element
		// new method  => use first value,  JLM 9/16/98
		//	*value = UorV(INDEXH(timeValues, 0).value, index); return 0; 
		return -1;
	}
	
	if (forTime > INDEXH(timeValues, n - 1).time) 
	{	// after last element
		//	*value = UorV(INDEXH(timeValues, n-1).value, index); return 0;
		return -1;
	}
	
	//	if (err = GetInterpolatedComponent(forTime, &value -> u, kUCode)) return err;
	//	if (err = GetInterpolatedComponent(forTime, &value -> v, kVCode)) return err;
	
	return 0;
}

OSErr ADCPTimeValue_c::GetTimeValue(const Seconds& current_time, VelocityRec *value)
{	// need to have depth indices too
	Boolean linear = FALSE;
	long a, b, i, n = GetNumValues();
	//double dv, slope, slope1, slope2, intercept;
	Seconds dt;
	//Boolean useExtrapolationCode = false;
	long startIndex,midIndex,endIndex;
	OSErr err = 0;
	
	if (!timeValues || _GetHandleSize((Handle)timeValues) == 0)
	{
		//TechError("ADCPTimeValue::GetTimeValue()", "timeValues", 0); 
		// no value to return
		value->u = 0;
		value->v = 0;
		return -1; 
	}
	
	if (err = GetInterpolatedComponent(current_time, &value -> u, kUCode)) return err;
	if (err = GetInterpolatedComponent(current_time, &value -> v, kVCode)) return err;
	/*if (forTime < INDEXH(timeValues, 0).time) 
	 {	// before first element
	 (*value).u = INDEXH(timeValues, 0).value.u;
	 (*value).v = INDEXH(timeValues, 0).value.v; 
	 
	 return 0; 
	 }
	 
	 if (forTime > INDEXH(timeValues, n - 1).time) 
	 {	// after last element
	 (*value).u = INDEXH(timeValues, n-1).value.u; 
	 (*value).v = INDEXH(timeValues, n-1).value.v; 
	 return 0;
	 }
	 /////////////////////////////////////////////////
	 // JLM 7/21/00, we need to speed this up for when we have a lot of values
	 // code goes here, (should we use a static to remember a guess of where to start) before we do the binary search ?
	 // use a binary method 
	 startIndex = 0;
	 endIndex = n-1;
	 while(endIndex - startIndex > 3)
	 {
	 midIndex = (startIndex+endIndex)/2;
	 if (forTime <= INDEXH(timeValues, midIndex).time)
	 endIndex = midIndex;
	 else
	 startIndex = midIndex;
	 }
	 /////////////////////////////////////////
	 
	 
	 for (i = startIndex; i < n; i++) {
	 if (forTime <= INDEXH(timeValues, i).time) {
	 dt = INDEXH(timeValues, i).time - forTime;
	 if (dt <= TIMEVALUE_TOLERANCE)
	 { 
	 //(*value) = UorV(INDEXH(timeValues, i).value, index); return 0; 
	 (*value).u = INDEXH(timeValues, i).value.u; 
	 (*value).v = INDEXH(timeValues, i).value.v; 
	 return 0; 
	 } // found match
	 
	 a = i - 1;
	 b = i;
	 break;
	 }
	 }*/
	
	
	return 0;
}

OSErr ADCPTimeValue_c::GetTimeValueAtDepth(long depthIndex, Seconds forTime, VelocityRec *value)
{	// need to have depth indices too
	Boolean linear = FALSE;
	long a, b, i, n = GetNumValues();
	//double dv, slope, slope1, slope2, intercept;
	Seconds dt;
	//Boolean useExtrapolationCode = false;
	long startIndex,midIndex,endIndex;
	OSErr err = 0;
	
	if (!timeValues || _GetHandleSize((Handle)timeValues) == 0)
	{
		//TechError("ADCPTimeValue::GetTimeValue()", "timeValues", 0); 
		// no value to return
		value->u = 0;
		value->v = 0;
		return -1; 
	}
	
	if (err = GetInterpolatedComponentAtDepth(depthIndex, forTime, &value -> u, kUCode)) return err;
	if (err = GetInterpolatedComponentAtDepth(depthIndex, forTime, &value -> v, kVCode)) return err;
	/*if (forTime < INDEXH(timeValues, 0).time) 
	 {	// before first element
	 (*value).u = INDEXH(timeValues, 0).value.u;
	 (*value).v = INDEXH(timeValues, 0).value.v; 
	 
	 return 0; 
	 }
	 
	 if (forTime > INDEXH(timeValues, n - 1).time) 
	 {	// after last element
	 (*value).u = INDEXH(timeValues, n-1).value.u; 
	 (*value).v = INDEXH(timeValues, n-1).value.v; 
	 return 0;
	 }
	 /////////////////////////////////////////////////
	 // JLM 7/21/00, we need to speed this up for when we have a lot of values
	 // code goes here, (should we use a static to remember a guess of where to start) before we do the binary search ?
	 // use a binary method 
	 startIndex = 0;
	 endIndex = n-1;
	 while(endIndex - startIndex > 3)
	 {
	 midIndex = (startIndex+endIndex)/2;
	 if (forTime <= INDEXH(timeValues, midIndex).time)
	 endIndex = midIndex;
	 else
	 startIndex = midIndex;
	 }
	 /////////////////////////////////////////
	 
	 
	 for (i = startIndex; i < n; i++) {
	 if (forTime <= INDEXH(timeValues, i).time) {
	 dt = INDEXH(timeValues, i).time - forTime;
	 if (dt <= TIMEVALUE_TOLERANCE)
	 { 
	 //(*value) = UorV(INDEXH(timeValues, i).value, index); return 0; 
	 (*value).u = INDEXH(timeValues, i).value.u; 
	 (*value).v = INDEXH(timeValues, i).value.v; 
	 return 0; 
	 } // found match
	 
	 a = i - 1;
	 b = i;
	 break;
	 }
	 }*/
	
	
	return 0;
}

void ADCPTimeValue_c::RescaleTimeValues (double oldScaleFactor, double newScaleFactor)
{
	long i,numValues = GetNumValues();
	TimeValuePair3D tv;
	
	for (i=0;i<numValues;i++)
	{
		tv = INDEXH(timeValues,i);
		tv.value.u /= oldScaleFactor;	// get rid of old scale factor
		tv.value.v /= oldScaleFactor;	// get rid of old scale factor
		tv.value.u *= newScaleFactor;
		tv.value.v *= newScaleFactor;
		INDEXH(timeValues,i) = tv;
	}
	return;
}