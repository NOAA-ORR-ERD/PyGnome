/************************************************************/
/*    computational code for computing tide heights         */
/*    based on NOS tide data.                               */
/************************************************************/


#include "Basics.h"
#include "TypeDefs.h"
#include "Shio.h"
#include <iostream>

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "CompFunctions.h"
#include "Replacements.h"
#endif

using namespace std;

/***************************************************************************************/

short GetTideCurrent(DateTimeRec *BeginDate,DateTimeRec *EndDate,
						short numOfConstituents, 
						CONSTITUENT *constituent,	// Amplitude-phase array structs
						CURRENTOFFSET *offset,		// Current offset data
						COMPCURRENTS *answers,		// Current-time struc with answers
						double* XODE, double* VPU, 
						Boolean DaylightSavings,		// Daylight Savings Flag
						char *staname)					// Station name
						
// Here we compute several arrays containing
// the reference curve and the highs and lows
// the reference curve is computed at user specified dt
// the highs and low will be computed to the minute like NTP4
//
{

	char	str1[256]="";
	double	beginHour=0.0,endHour=0.0,timestep=10.0,deltaTime=0.0,dt=0.0;
	double	oldBeginHour=0.0,oldEndHour=0.0;
	double	biggestOffset=0.0,absBiggestOffset=0.0,theHour=0.0;
	short	errorFlag=0,numOfYearCorrections=0;
	short	beginDay=0,beginMonth=0,beginYear=0,endDay=0,endMonth=0,endYear=0;
	short	L2Flag=0,HFlag=0,rotFlag=0;


	/* Variables to overlay reference curve */
	double		tval=0.0,ehour=0.0,julianoffset=0.0;
	long		i=0,size1=0,size2=0,numpnts=0;

	/* Check to see that offset data is valid */
	errorFlag=CheckCurrentOffsets(offset);
	if(errorFlag){ goto Error; }
#ifndef pyGNOME
	SetWatchCursor();
#endif	
	/* Find beginning hour and ending hour */
	beginDay	= BeginDate->day;
	beginMonth	= BeginDate->month;
	beginYear	= BeginDate->year;
	timestep	= 10.0;
	dt			= (timestep / 60.0);	/* convert to hours */

	errorFlag = GetJulianDayHr(beginDay,beginMonth,beginYear,&beginHour);
	if(errorFlag!=0){ goto Error; }

	endDay		= EndDate->day;
	endMonth	= EndDate->month;
	endYear		= EndDate->year;	

	errorFlag	= GetJulianDayHr(endDay,endMonth,endYear,&endHour);
	if(errorFlag!=0){ goto Error; }

	// check to see if we span year
	if( endYear > beginYear ){
		errorFlag = GetJulianDayHr(31,12,beginYear,&theHour);
		endHour=(theHour+endHour+24.0);
	}else{
		// end hour set to end of day
		endHour=(endHour+24.0);
	}


	// OK now adjust beginning and ending hour so
	// that when we apply offsets, we still span the
	// requested days

	biggestOffset = offset->MinBefFloodTime.val;
	absBiggestOffset = fabs(biggestOffset);
			
	if( absBiggestOffset < fabs(offset->MinBefEbbTime.val) ){
		biggestOffset = offset->MinBefEbbTime.val;
		absBiggestOffset = fabs(biggestOffset);
	}

	if( absBiggestOffset < fabs(offset->EbbTime.val) ){
		biggestOffset = offset->EbbTime.val;
		absBiggestOffset = fabs(biggestOffset);
	}
	
	if( absBiggestOffset < fabs(offset->FloodTime.val) ){
		biggestOffset = offset->FloodTime.val;
		absBiggestOffset = fabs(biggestOffset);
	}

	// Now check if daylight savings time is in effect
	// If it is, set beginning hour back one.

	if( DaylightSavings ){
		beginHour		= beginHour		-1.0;
		endHour			= endHour		-1.0;
	}

	// Now save original begin and end hour and modify
	// begin and end hour to take into account under and overshoot
	// due to offsets

	oldBeginHour = beginHour;
	oldEndHour = endHour;
	
	// Lookit, we just offset both ends by the largest
	// offset.  This way, we don't have to worry about
	// what the offset is for, where it falls on the
	// curve and if it positive or negative.
	
	endHour = endHour + absBiggestOffset;
	beginHour = beginHour - absBiggestOffset;
	
	// OK here we set the begin hour back one time step so
	// we don't miss a max or min that is within 1 timestep
	// We will also go on step pass the end hour
	
	beginHour -= dt;
 	endHour += dt;

	if( beginHour >= endHour ){
		errorFlag=1;
		goto Error;
	}
	
	deltaTime = endHour - beginHour;
	if( deltaTime <= dt ){
		errorFlag=2;
		goto Error;
	}


	// OK go get year data
	// Begin by declaring space for the two arrays we need
	// Note the number we get should match the number of
	// constituents Andy is sending me.

	//if(constituent->kPrime) numOfConstituents = (short)( (long)GetHandleSize((Handle)constituent->kPrime) / (long)sizeof(float) );
	if(numOfConstituents < 5){
		errorFlag=6;
		goto Error;
	}
	

	// Now get control flag to see if we gotta change L2 stuff
	if( constituent)
	{
		errorFlag=GetControlFlags(constituent,&L2Flag,&HFlag,&rotFlag);     
	}
	//else
	//{
	//	errorFlag=GetControlFlagsAlternate(cntrlvars,&L2Flag,&HFlag,&rotFlag);     
	//}
	if(errorFlag!=0){ goto Error; }

	// OK we got all the pieces we need, now go an compute reference curve
	errorFlag = GetRefCurrent(	constituent,		// Amplitude-phase array structs
								XODE,			// Year correction to amplitude
								VPU,				// Year correction to phase
								numOfConstituents,	// Number of constituents usually 37
								beginHour,			// Begin hour from start of year
								endHour,			// End hour from start of year
								timestep,			// Comp Time step in minutes
								answers);		// Current-time struc with answers
#ifndef pyGNOME
	SetWatchCursor();
#endif
	if(errorFlag!=0){ goto Error; }

	/* Do the offset correction for the station */
	errorFlag = DoOffSetCurrents(answers,offset,oldBeginHour,oldEndHour,timestep,rotFlag);
	if(errorFlag!=0){ goto Error; }

	// cleanup points before start or past end times
	FixCEnds(answers,oldBeginHour, oldEndHour,rotFlag);

	// fix zero crossings for rotary stations

	if( (rotFlag==1) || (rotFlag==2) ){
		//
		// with new algorithm to determine flood/ebb
		// we shouldn't need this check.
		//
		//	errorFlag = CheckCurrDir(constituent,answers);
		//	if(errorFlag!=0){
		//		goto Error;
		//	}

		// Check to see if rotary smoothing is called for
		/*if( answers->speedKey != 0 )*/{ FixCurve(answers); }	// GK suggested commenting out to fix Norfolk problem w unusable tide data, so far it hasn't helped 6.22.06
		
		// Now fill in array for major-minor axis
		errorFlag = RotToMajorMinor(constituent,offset,answers);
		if(errorFlag!=0){ goto Error; }
	}

	// Reset time arrays to begin at start time
	ResetCTime(answers,oldBeginHour);


	// OK now check if we are doing Akutan pass
	// If we are, we gotta handle it as special case
	// davew - What a suprise! It appears that the name has changed
	// in the new data set from NOS!
	
	//if( strcmp(staname,"AKUTAN PASS; ALEUTIAN ISLANDS") == 0 )
	if( strcmp(staname,"AKUTAN PASS") == 0 )
	{
		AkutanFix(answers, rotFlag);
	}

	//HUnlock((Handle)answers->timeHdl); HUnlock((Handle)answers->speedHdl);


Error:



	CompCErrors(errorFlag,str1);
	
	if(errorFlag!=0){ 
#ifndef pyGNOME
		settings.doNotPrintError = false;//JLM 6/19/98, allows dialogs to come up more than once
#endif
		printError(str1); }

	return( errorFlag );
}

// **********************************************
void AkutanFix (COMPCURRENTS *answers, short rotFlag)
// Special fix for Akutan Pass currents
// that are over 7.5 knots
{
	double	*CHdl,*uVelHdl,*vVelHdl;
	double	*MaxMinHdl;

	long	numOfPoints,numOfMaxMins;	
	long	i;
	
	double	v,ratio,dv,sign,vfix;
	
	// **************************************************
	
	MaxMinHdl = answers->EbbFloodSpeeds;
	CHdl = answers->speed;
	uVelHdl = answers->u;
	vVelHdl= answers->v;
	
	numOfPoints = answers->nPts;
	numOfMaxMins = answers->numEbbFloods;
	
	//OK do the fix on amplitude array
	
	for (i=0;i<numOfPoints;i++){
		sign = 1.0;
		v = CHdl[i];
		if(v < 0.0){
			sign = -1.0;
			v = -v;
		}
		
		if( v > 7.5) {
			dv = v - 7.5;
			vfix = .635 * dv;
			answers->speed[i] = sign * (v - vfix);
			ratio = (v-vfix)/v;

			// davew  12/22/3: added if statement
			// pointers for u and v velocity components are 
			// only valid if tidal currents are rotary
			if ( (rotFlag==1) || (rotFlag==2) )
			{
				(answers->u)[i] = (answers->u)[i] * ratio;
				(answers->v)[i] = (answers->v)[i] * ratio;
	 		}
		}
	
	}
	
	// Now do max-min array
	
	for (i=0;i<numOfMaxMins; i++){
		sign = 1.0;
		v = MaxMinHdl[i];
		if(v<0.0){
			sign = -1.0;
			v = -v;
		}
		
		if(v>7.5){
			dv = v - 7.5;
			vfix = .635 * dv;
			answers->EbbFloodSpeeds[i] = sign * (v - vfix);
		}	
	
	}
	
	return;
}

// **********************************************
short CheckCurrDir(CONSTITUENT *constituent,COMPCURRENTS *answers)
// Not being used at this point...
// This checks to see if the rotary currents have the correct
// flood and ebb directions
// We go to the first max velocity and check it's direction
// If the direction is wrong, we negate all the velocity magnitudes
// and change the labels on the max and mins.
{	
	
	double		*theCurrentHdl,*maxMinHdl;
	double		*uHdl,*vHdl;
	short		*keyHdl;
	short numberOfPoints,i,nFloodEbbs,flag;
	double u,v;
	double v0,v1,v2,s1,s2;
	
	numberOfPoints	= answers->nPts;
	theCurrentHdl	= answers->speed;
	nFloodEbbs		= answers->numEbbFloods;
	maxMinHdl		= answers->EbbFloodSpeeds;
	keyHdl			= answers->EbbFlood;
	uHdl			= answers->u;
	vHdl			= answers->v;
	
	// go get the first max velocity
	// we need to pull it out of the speed array and not the
	// max min arrays because the speed arrays have the u and v components
	//
	
	flag = 0;
	for(i=1;i< (numberOfPoints-1);i++){
	
		v0 = theCurrentHdl[i-1];
		v1 = theCurrentHdl[i];
		v2 = theCurrentHdl[i+1];
		if(v0<0.0)v0=-v0;
		if(v1<0.0)v1=-v1;
		if(v2<0.0)v2=-v2;
		
		s1 = v1-v0;
		s2 = v2 - v1;
		if( (s1>0.0) && (s2<0.0) ){
			// OK we have found first max at i
			v0 = theCurrentHdl[i];
			u = uHdl[i];
			v = vHdl[i];
			flag = GetVelDir(constituent,u,v);
			if(flag==33)return flag;
			break;
		}
	}
	
	if(flag==0)return 0;
	if( (flag>0) && (v0>0.0) ) return 0;
	if( (flag<0) && (v0<0.0) ) return 0;
	
	// if we get here, we gotta flip the world!!!

	for(i=0;i< numberOfPoints;i++){
		theCurrentHdl[i] = -theCurrentHdl[i];
	}
	
	for (i=0;i<nFloodEbbs;i++){
	
		maxMinHdl[i] = -maxMinHdl[i];
		if( keyHdl[i] == MaxEbb ){
			keyHdl[i]=MaxFlood;
		}
		else if( keyHdl[i] == MaxFlood ){
			keyHdl[i]=MaxEbb;
		}
		else if( keyHdl[i] == MinBeforeFlood ){
			keyHdl[i]=MinBeforeEbb;
		}
		else if( keyHdl[i] == MinBeforeEbb ){
			keyHdl[i]=MinBeforeFlood;
		}
	}
	
	return 0;
}

//***********************************************
void CleanUpCompCurrents (COMPCURRENTS *CPtr)
{
	double		*hHdl,*HLHHdl;
	EXTFLAG	*tHdl,*HLTHdl;
	short		*HLHdl;
	
	tHdl	= CPtr->time;
	hHdl	= CPtr->speed;
	HLHHdl	= CPtr->EbbFloodSpeeds;
	HLTHdl	= CPtr->EbbFloodTimes;
	HLHdl	= CPtr->EbbFlood;
	
	/*if(tHdl) DisposeHandle((Handle)tHdl);
	if(hHdl) DisposeHandle((Handle)hHdl);
	if(HLHHdl) DisposeHandle((Handle)HLHHdl);
	if(HLTHdl) DisposeHandle((Handle)HLTHdl);
	if(HLHdl) DisposeHandle((Handle)HLHdl);*/
	
	if(tHdl) {delete [] tHdl; tHdl = 0;}
	if(hHdl) {delete [] hHdl; hHdl = 0;}
	if(HLHHdl) {delete [] HLHHdl; HLHHdl = 0;}
	if(HLTHdl) {delete [] HLTHdl; HLTHdl = 0;}
	if(HLHdl) {delete [] HLHdl; HLHdl = 0;}

	CPtr->time			= 0;
	CPtr->speed			= 0;
	CPtr->EbbFloodSpeeds	= 0;
	CPtr->EbbFloodTimes	= 0;
	CPtr->EbbFlood		= 0;

	return;
}


void CompCErrors(short errorNum,char *errStr)
{
	switch (errorNum){
	
		case 0:
			break;
		case 1:
			strcpy(errStr,"start hour later than end hour!");
			break;
		case 2:
			strcpy(errStr,"time step less than total time interval");
			break;
		case 3:
			strcpy(errStr,"month number is invalid");
			break;
		case 4:
			strcpy(errStr,"day value invalid for given month");
			break;
		case 5:
			strcpy(errStr,"year value invalid");
			break;
		case 6:
			strcpy(errStr,"too few constituents");
			break;
		case 7:
			strcpy(errStr,"not enough memory for VPU array");
			break;
		case 8:
			strcpy(errStr,"not enough memory for XODE array");
			break;
		case 9:
			strcpy(errStr,"not enough memory for AMPA array");
			break;
		case 10:
			strcpy(errStr,"not enough memory for epoch array");
			break;
		case 11:
			strcpy(errStr,"not enough memory for CHDl handle");
			break;
		case 12:
			strcpy(errStr,"not enough memory for THdl handle");
			break;
		case 13:
			strcpy(errStr,"not enough memory for MaxMinHHdl handle");
			break;
		case 14:
			strcpy(errStr,"not enough memory for MaxMinTHdl handle");
			break;
		case 15:
			strcpy(errStr,"not enough memory for tHdl handle");
			break;
		case 16:
			strcpy(errStr,"consecutive heights with same value");
			break;
		case 17:
			strcpy(errStr,"resource for year data invalid");
			break;
		case 18:
			strcpy(errStr,"could not find valid high or low value");
			break;
		case 19:
			strcpy(errStr,"not enough max and min values");
			break;
		case 20:
			strcpy(errStr,"not enough memory for CArray array");
			break;
		case 21:
			strcpy(errStr,"not enough memory for TArray array");
			break;
		case 22:
			strcpy(errStr,"min B flood offset data unavailable");
			break;
		case 23:
			strcpy(errStr,"flood offset data unavailable");
			break;
		case 24:
			strcpy(errStr,"flood speed ration unavailable");
			break;
		case 25:
			strcpy(errStr,"bad MaxMin code");
			break;
		case 26:
			strcpy(errStr,"min B Ebb offset data unavailable");
			break;
		case 27:
			strcpy(errStr,"ebb offset data unavailable");
			break;
		case 28:
			strcpy(errStr,"Offsets for currents unusable for selected time span.");
			break;
		case 29:
			strcpy(errStr,"Time offsets unusable for selected time period.");
			break;
		case 30:
			strcpy(errStr,"control parameters bad");
			break;
		case 31:
			strcpy(errStr,"not enough memory for u and v components");
			break;
		case 32:
			strcpy(errStr,"Time offset data is insufficient to compute currents");
			break;
		case 33:
			strcpy(errStr,"Flood and Ebb directions unavailable");
			break;
		case 34:
			strcpy(errStr,"No valid velocities to rotate");
			break;
		case 35:
			strcpy(errStr,"not enough memory for rotated u and v");
			break;
		case 36:
			strcpy(errStr,"error in finding offset ratio for current");
			break;
		case 37:
			strcpy(errStr,"not enough memory for temp max min array");
			break;
		case 38:
			strcpy(errStr,"interpolated time out of range");
			break;
		case 39:
			strcpy(errStr,"no max or min currents in record");
			break;
		case 40:
			strcpy(errStr,"time not in allowable max-min range");
			break;
		case 41:
			strcpy(errStr,"MaxMinCount >= maxMinHdlNumElements in GetRefCurrent");
			break;
		/////case kInterrupt:
		/////	strcpy(errStr,"User Interrupt.");
		/////	break;
		default:
			strcpy(errStr,"unknown error code");
			break;
	}
	return;
}

// **********************************************************

short CurveFitOffsetCurve(COMPCURRENTS *answers,
							double startTime,
							double endTime,
							double timestep)   
{
// Here we generate the offset curve by curve fitting the high low
// stuff ...

	short errorFlag,flag;
	long NoOfMaxMins,numOfPoints,i,j;
	double t;
	short EbbToMinBFlood = 0, MinBFloodToFlood = 1;
	short FloodToMinBEbb=2, MinBEbbToEbb=3;
	short FloodToEbb=4, EbbToFlood=5, FloodToFlood=6, EbbToEbb=7;
	short maxFlood = 1, minBFlood = 0, maxEbb = 3, minBEbb = 2;
	double previousTime,nextTime,previousValue,nextValue;
	double previousMinusOneTime,previousMinusOneValue;
	double nextPlusOneTime,nextPlusOneValue;
	double zeroSlope,SpeedValue;
	
	double *	CArrayPtr;
	EXTFLAG		*TArrayPtr;
	double		*MaxMinHdl,*ValHdl;
	EXTFLAG		*MaxMinTimeHdl,*TimeHdl;
	short		*MaxMinFlagHdl;
	short		*OldMaxMinFlagPtr;
	

	CArrayPtr=NULL;
	TArrayPtr=NULL;
	errorFlag=0;
	
	/* Get handles just to make code more readable */
	MaxMinHdl		= answers->EbbFloodSpeeds;
	MaxMinTimeHdl	= answers->EbbFloodTimes;
	MaxMinFlagHdl	= answers->EbbFlood;
	ValHdl			= answers->speed;
	TimeHdl			= answers->time;
	
	// Begin by declaring temp space on heap 
	// for a copy of the maxs and mins

	NoOfMaxMins = answers->numEbbFloods;
	if(NoOfMaxMins<1){
		errorFlag=19;
		goto Error;
	}

	/*CArrayPtr=(double *)NewPtrClear(sizeof(double)*(NoOfMaxMins+2));
	if(CArrayPtr==nil){
		errorFlag=20;
		goto Error;
	}*/
	//CArrayPtr = (double *)calloc(NoOfMaxMins+2,sizeof(double));
	CArrayPtr = new double[NoOfMaxMins+2];
	if (CArrayPtr==NULL) {errorFlag=20; goto Error;}

	// Note the two extra high/low times used for interpolation
	// are stored at the end of the Time array
	
	/*TArrayPtr=(EXTFLAGPTR)NewPtrClear(sizeof(EXTFLAG)*(NoOfMaxMins+2) );
	if(TArrayPtr==nil){
		errorFlag=21;
		goto Error;
	}*/
	//TArrayPtr = (EXTFLAG *)calloc(NoOfMaxMins+2,sizeof(EXTFLAG));
	TArrayPtr = new EXTFLAG[NoOfMaxMins+2];
	if (TArrayPtr==NULL) {errorFlag=21; goto Error;}
	
	/*OldMaxMinFlagPtr=(short *)NewPtrClear(sizeof(short)*(NoOfMaxMins + 2) );
	if(OldMaxMinFlagPtr==nil){
		errorFlag=37;
		goto Error;
	}*/
	//OldMaxMinFlagPtr = (short *)calloc(NoOfMaxMins+2,sizeof(short));
	OldMaxMinFlagPtr = new short[NoOfMaxMins+2];
	if (OldMaxMinFlagPtr==NULL) {errorFlag=37; goto Error;}
	
	//OK now make copy of reference station data 
	// use copy instead of original data
    // 04/16/2013 - G.K. - but do not copy out of sequence points
	for(i=0, j=0; i<(NoOfMaxMins+2); i++)
        if(MaxMinTimeHdl[i].flag == 0)
        {
		    CArrayPtr[j] = MaxMinHdl[i];
		    TArrayPtr[j].val = MaxMinTimeHdl[i].val;
            TArrayPtr[j].flag = MaxMinTimeHdl[i].flag;
		    OldMaxMinFlagPtr[j] = MaxMinFlagHdl[i];
            j++;
	    }	
    // 04/16/2013 - G.K.:
    NoOfMaxMins = j-2;
	
	/* for (i=0; i< (NoOfMaxMins+2); i++){
		CArrayPtr[i] = MaxMinHdl[i];
		TArrayPtr[i].val = MaxMinTimeHdl[i].val;
		OldMaxMinFlagPtr[i] = MaxMinFlagHdl[i];
	}*/	


// ********	
		
	numOfPoints = answers->nPts;
		
	zeroSlope = 0.0;
			
 	for (i=0; i<numOfPoints; i++){
	
		t= startTime + i * (timestep/60.0);
		if(t > (endTime) ){
			t = endTime;
		}
		
		errorFlag = GetFloodEbbKey(t,
					  TArrayPtr,
					  OldMaxMinFlagPtr,
					  CArrayPtr,
					  NoOfMaxMins,
					  &flag);
		
		if(errorFlag == 0){
				 		
			errorFlag = GetFloodEbbSpans(t,
					  	TArrayPtr,
					  	OldMaxMinFlagPtr,
					 	CArrayPtr,
					  	NoOfMaxMins,
					  	&previousTime,
					 	&nextTime,
					 	&previousValue,
					    &nextValue,
						&previousMinusOneTime,
						&nextPlusOneTime,
						&previousMinusOneValue,
						&nextPlusOneValue,
					 	0);
						
			if(flag==FloodToMinBEbb){
				flag = 1;
	 			DoHermite(zeroSlope,previousMinusOneTime,previousValue,previousTime,
	 		          nextValue,nextTime,nextPlusOneValue,nextPlusOneTime,t,
					  &SpeedValue,flag);
			}
			
			else if(flag==MinBFloodToFlood){
				flag = 2;
	 			DoHermite(previousMinusOneValue,previousMinusOneTime,previousValue,previousTime,nextValue,
	 		          nextTime,zeroSlope,nextPlusOneTime,t,
					  &SpeedValue,flag);
			}

			else if(flag==EbbToMinBFlood){
				flag = 1;
	 			DoHermite(zeroSlope,previousMinusOneTime,previousValue,previousTime,
	 		          nextValue,nextTime,nextPlusOneValue,nextPlusOneTime,t,
					  &SpeedValue,flag);
			}

			else if(flag==MinBEbbToEbb){
				flag = 2;
	 			DoHermite(previousMinusOneValue,previousMinusOneTime,previousValue,previousTime,nextValue,
	 		          nextTime,zeroSlope,nextPlusOneTime,t,
					  &SpeedValue,flag);
			}
			
			else if(flag==EbbToFlood){
				flag = 3;
	 			DoHermite(zeroSlope,previousMinusOneTime,previousValue,previousTime,nextValue,
	 		          nextTime,zeroSlope,nextPlusOneTime,t,
					  &SpeedValue,flag);
			}
			else if(flag==FloodToEbb){
				flag = 3;
	 			DoHermite(zeroSlope,previousMinusOneTime,previousValue,previousTime,nextValue,
	 		          nextTime,zeroSlope,nextPlusOneTime,t,
					  &SpeedValue,flag);
			}
            else if(flag==FloodToFlood){
                flag = 0;
	 			DoHermite(previousMinusOneValue,previousMinusOneTime,previousValue,previousTime,nextValue,
	 		          nextTime,nextPlusOneValue,nextPlusOneTime,t,
					  &SpeedValue,flag);
            }
            else if(flag==EbbToEbb){
                flag = 0;
	 			DoHermite(previousMinusOneValue,previousMinusOneTime,previousValue,previousTime,nextValue,
	 		          nextTime,nextPlusOneValue,nextPlusOneTime,t,
					  &SpeedValue,flag);
            }
			
			TimeHdl[i].val = t;
	 		ValHdl[i] = SpeedValue;
		}
	}
 
Error:
	//if(CArrayPtr) DisposePtr((Ptr)CArrayPtr);
	//if(TArrayPtr) DisposePtr((Ptr)TArrayPtr);
	//if(OldMaxMinFlagPtr) DisposePtr ((Ptr)OldMaxMinFlagPtr);
	//if(CArrayPtr) {free(CArrayPtr); CArrayPtr = NULL;}
	//if(TArrayPtr) {free(TArrayPtr); TArrayPtr = NULL;}
	//if(OldMaxMinFlagPtr) {free(OldMaxMinFlagPtr); OldMaxMinFlagPtr = NULL;}
	if (CArrayPtr) delete [] CArrayPtr;
	if (TArrayPtr) delete [] TArrayPtr;
	if (OldMaxMinFlagPtr) delete [] OldMaxMinFlagPtr;
	return(errorFlag);

}

// ************************************************************

// ************************************************************
void DoHermite(double v0,     // value at t0
			   double t0,     // time t0
			   double v1,     
			   double t1,     
			   double v2,     
			   double t2,     
			   double v3,    
			   double t3,    
			   double t,      // time for interpolation
			   double *vTime, // returns value at time
			   short    flag)   // if flag == 0 then compute slope
			                    // if flag == 1 then v0 contains slope at t1
								// if flag == 2 then v3 contains slope at t2
								// if flag == 3 then v0 and v3 contain slopes
{	
	double s1,s2,theValue;
	
	if(flag==0){
		s1 = ( (v1-v0)/(t1-t0) + (v2-v1)/(t2-t1) ) * .5;
		s2 = ( (v3-v2)/(t3-t2) + (v2-v1)/(t2-t1) ) * .5;
	
	}
	else if(flag==1){
		s1 = v0;
		s2 = ( (v3-v2)/(t3-t2) + (v2-v1)/(t2-t1) ) * .5;
	}
	else if(flag==2){
		s1 = ( (v1-v0)/(t1-t0) + (v2-v1)/(t2-t1) ) * .5;
		s2 = v3;
	}
	else if(flag==3){
		s1 = v0;
		s2 = v3;
	}
	
	Hermite(v1,s1,t1,v2,s2,t2,t,&theValue);
	*vTime = theValue;
	return;
	
}

//**************************************************************

short DoOffSetCurrents (COMPCURRENTS *answers,  // Hdl to reference station heights
					CURRENTOFFSET *offset,      // Hdl to offset data
					double OldBeginHour,
					double OldEndHour,
					double timestep,
					short rotFlag)

{
	short errorFlag;
	long NoOfMaxMins,i;
	double MaxFloodOffset,MaxEbbOffset,MinBFloodOffset,MinBEbbOffset;
	double FloodRatio,EbbRatio;
	short EbbToMinBFlood = 0, MinBFloodToFlood = 1;
	short FloodToMinBEbb=2, MinBEbbToEbb=3;
	short FloodToEbb=4, EbbToFlood=5;
	short maxFlood = 1, minBFlood = 0, maxEbb = 3, minBEbb = 2;
	
	
	short mainStationFlag;
	
	short Key, offSetKey = 0, curveFitKey = 1;
	double		*MaxMinHdl;
	EXTFLAG *MaxMinTimeHdl;
	short		*MaxMinFlagHdl;
	short tryFixNum;
	
	EXTFLAG debugMe[100] ;  //debug , code goes here
	EXTFLAG debugMeAfter[100] ;  //debug , code goes here
	short debugMeFlagHdl[100] ;  //debug , code goes here
	

	errorFlag = 0;
	Key = curveFitKey;
	mainStationFlag = 0;
	
	/* Get handles just to make code more readable */
	MaxMinHdl		= answers->EbbFloodSpeeds;
	MaxMinTimeHdl	= answers->EbbFloodTimes;
	MaxMinFlagHdl	= answers->EbbFlood;
	
	// Begin by declaring temp space on heap 
	// for a copy of the maxs and mins
	
	NoOfMaxMins = answers->numEbbFloods;
	if(NoOfMaxMins<1){
		errorFlag=19;
		goto Error;
	}
	
	// Check to make sure we got offset data before using it
	
	if( offset->MinBefFloodTime.dataAvailFlag!=1){
		errorFlag=22;
		goto Error;
	}
	MinBFloodOffset = offset->MinBefFloodTime.val;
	if(MinBFloodOffset!=0.0)mainStationFlag = 1;
	
	if( offset->FloodTime.dataAvailFlag!=1){
		errorFlag=23;
		goto Error;
	}
	MaxFloodOffset =offset->FloodTime.val;
	if(MaxFloodOffset!=0.0)mainStationFlag = 1;

	if( offset->MinBefEbbTime.dataAvailFlag!=1){
		errorFlag=26;
		goto Error;
	}
	MinBEbbOffset =offset->MinBefEbbTime.val;
	if(MinBEbbOffset!=0.0)mainStationFlag = 1;
	
	if( offset->EbbTime.dataAvailFlag!=1){
		errorFlag=27;
		goto Error;
	}
	MaxEbbOffset =offset->EbbTime.val;
	if(MaxEbbOffset!=0.0)mainStationFlag = 1;
	
	if( offset->FloodSpdRatio.dataAvailFlag!=1){
		errorFlag=24;
		goto Error;
	}
	FloodRatio = offset->FloodSpdRatio.val;
	if(FloodRatio!=1.0)mainStationFlag = 1;
	
	if( offset->EbbSpdRatio.dataAvailFlag!=1){
		errorFlag=25;
		goto Error;
	}
	EbbRatio = offset->EbbSpdRatio.val;
	if(EbbRatio!=1.0)mainStationFlag = 1;
	
	
	// Check to see if we have main station ... if we do,
	// forget about offsets, just fill the U V handles and return
	
	// davew 2/9/4: Cheasapeake Bay Entrance was returning nil for u,v handles (not what we wanted)
	// Bushy and I fixed it by REMOVING the if(rotFlag==3) check. We don't know what this check
	// was for or what 3 means. We do know that
	// rotFlag == 0: station not rotary
	// rotFlag == 1: North, East rotary
	// rotFlag == 2: Major, Minor rotary
	// rotFlag == 3: ?????
	// rotFlag < 0 || rotFlag > 3: bad data

	if( mainStationFlag==0){
		//if(rotFlag==3){
			errorFlag = OffsetUV (answers,offset);
		//}
        // 02/24/14 G.K. - commented out next to apply interpolation cases for main stations too
        // TODO: test it more for all key cases and rotary stations
        // return errorFlag;	
    }
	
	// OK we are OK with max and min corrections
	// We now need to fix the full array

	// Two options
	// We can curve fit the high low array
	// Or we can adjust the reference curve
	// point by point
	

	

	if(Key==offSetKey){
 		errorFlag = OffsetReferenceCurve(answers,offset);
	}
	
	
	/////////////////////////////////////////////////
	// debug , code goes here
	{
		for(i=0; i < 100 && i< (NoOfMaxMins+2); i++)
		{
			debugMe[i] = MaxMinTimeHdl[i];
		}
		debugMe[0] = debugMe[0];
	}
	/////////////////////////////////////////////////

	

//  *** OK now loop through and add corrections to the hummer
	for(i=0; i< (NoOfMaxMins+2); i++){
		
		if( MaxMinFlagHdl[i] == maxFlood){
			MaxMinTimeHdl[i].val = MaxMinTimeHdl[i].val + MaxFloodOffset;
			MaxMinHdl[i]= MaxMinHdl[i] * FloodRatio;
		}
		else if( MaxMinFlagHdl[i] == minBFlood){
			MaxMinTimeHdl[i].val = MaxMinTimeHdl[i].val + MinBFloodOffset;
			// OK if we are at offset station, make sure we zero out
			// the currents at min ... note this may mean misrepresentation
			// of rotary offset stations
			if(MinBFloodOffset!=0.0) MaxMinHdl[i] = 0.0;
		}
		else if( MaxMinFlagHdl[i] == minBEbb){
			MaxMinTimeHdl[i].val = MaxMinTimeHdl[i].val + MinBEbbOffset;
			if(MinBEbbOffset!=0.0) MaxMinHdl[i] = 0.0;
		}
		else if( MaxMinFlagHdl[i] == maxEbb){
			MaxMinTimeHdl[i].val = MaxMinTimeHdl[i].val + MaxEbbOffset;
			MaxMinHdl[i]= MaxMinHdl[i] * EbbRatio;
		}
		else {
			errorFlag=28;
			goto Error;
		}
	}

	/////////////////////////////////////////////////
	// debug , code goes here
	{
		for(i=0; i < 100 && i< (NoOfMaxMins+2); i++)
		{
			debugMeAfter[i] = MaxMinTimeHdl[i];
			debugMeFlagHdl[i] = MaxMinFlagHdl[i];
		}
		debugMeAfter[0] = debugMeAfter[0];
	}
	/////////////////////////////////////////////////


	// now check times to make sure they are still in order
	// if not, we mark the points that are out of
	// sequence and move on.  When we do the hermite
	// we will ignore the points out of sequence.
	// Then when we interpolate over the out of sequence
	// points, we will flag those points so Andy can
	// use a different line weight to plot with

    // 04/12/2013 - G.K. - actually old code marks out of sequence 
    // peak points but does nothing with them later - just sets a 
    // correspondent error flag. Now we will try to interpolate
    // ones later in CurveFitOffsetCurve(...) function (since curveFitKey 
    // branch is active here). So commented out setting the error flag.

	#define MAXNUMTRIES 5
	for(tryFixNum = 0;tryFixNum< MAXNUMTRIES;tryFixNum++)
	{
		Boolean changedValue = false;
		for(i=1;i<NoOfMaxMins; i++)
		{
			if( MaxMinTimeHdl[i].val < MaxMinTimeHdl[i-1].val )
			{
				Boolean tryFixingPts = (tryFixNum < (MAXNUMTRIES - 1) 
												&& (NoOfMaxMins > 2)
												//&& (!OptionKeyDown()) // for debugging
												);
				if(tryFixingPts)
				{	// new plan, try to ignore the points causing the error
					// if one of them is a minBFlood or minBEbb gets out of sequence
					// Just take the midpoint of the time values set the time difference to 1 minute
					if(MaxMinTimeHdl[i-1].flag == minBEbb 
						|| MaxMinTimeHdl[i-1].flag == minBFlood
						|| MaxMinTimeHdl[i].flag == minBEbb
						|| MaxMinTimeHdl[i].flag == minBFlood)
					{
						float avgTime = (MaxMinTimeHdl[i].val + MaxMinTimeHdl[i-1].val)/2;
						float oneHalfMinute = 0.5/60.0; // in units of hours
						MaxMinTimeHdl[i-1].val = avgTime - oneHalfMinute;
						MaxMinTimeHdl[i].val = avgTime + oneHalfMinute;
						changedValue = true;
						continue;
					}
				}
				MaxMinTimeHdl[i].flag = outOfSequenceFlag;
				MaxMinTimeHdl[i-1].flag = outOfSequenceFlag;
				//errorFlag=29;
				//goto Error;
			}
		}
		if(!changedValue) break; // all values are OK
	}

	if(Key==curveFitKey){
 		errorFlag = CurveFitOffsetCurve( answers,
										 OldBeginHour,
										 OldEndHour,
										 timestep); 
	}

	// OK now we adjust the u and v vectors to match the
	// offset station data
	
	errorFlag = OffsetUV ( answers,offset);

Error:

	return(errorFlag);
}






short FindExtraFE(double			*AMPA,				// amplitude corrected for year
					double			*epoch,				// epoch corrected for year
					short			numOfConstituents,	// number of frequencies
					COMPCURRENTS	*answers,			// Height-time struc with answers
					double			*zeroTime,			// first max min time
					double			*zeroValue,			// Value at *zeroTime
					short			*zeroFlag,			// key to what *zeroValue is
					double			*extraTime,			// last max min time
					double			*extraValue,		// Value at *extraTime
					short			*extraFlag,			// key to what *extraValue is
					double			refCur,				// reference current in knots
					short			CFlag)				// hydraulic station flag
						
// Function to compute extra max and min values
// We need one value before first high low and one after
// So we can interpolate

{
	long i,nPts,nFE;
	double c,lastValue;
	double t,firstTime,lastTime,dt;
	double	*CHdl;
	EXTFLAG *THdl;
	short		*tHdl;
	short EbbKey = 3, FloodKey = 1,MinBFloodKey = 0,MinBEbbKey = 2;
	// OK begin by looking for first one before our data
	
	// Check first if max value or min value
	
		tHdl = answers->EbbFlood;
		THdl = answers->time;
		CHdl = answers->speed;
		nPts = answers->nPts;
		nFE  = answers->numEbbFloods;
		
		firstTime = THdl[0].val;
		lastValue = CHdl[0];
		
		dt = 1.0/60.0;
		
		// go for max of 24 hours at 1 minute intervals
		// and look for max or min
		
		// if first value is min before flood
		// look for previous ebb
		if( tHdl[0]==MinBFloodKey){ 
			//look for max ebb
			for(i=1;i<1441;i++){
				t = firstTime - dt*i;
				c = RStatCurrent(t,AMPA,epoch,refCur,numOfConstituents,CFlag);
				if(c>=lastValue)break;
				lastValue = c;
			}
				*zeroFlag = EbbKey;
		}
		
		// if first value is max flood
		// look for min before flood
		else if(tHdl[0]==FloodKey){
			//look for zero
			for(i=1;i<1441;i++){
				t = firstTime - dt*i;
				c = RStatCurrent(t,AMPA,epoch,refCur,numOfConstituents,CFlag);
				if( (c>=0) && (lastValue<0) )break;
				// check to make sure we cross zero
				if(c>lastValue){
					*zeroFlag = EbbKey;
					break;
				}
				lastValue = c;
			}
		}
		
		// if first value is min before ebb
		// look for flood
		else if(tHdl[0]==MinBEbbKey){
			//look for max flood
			for(i=1;i<1441;i++){
				t = firstTime - dt*i;
				c = RStatCurrent(t,AMPA,epoch,refCur,numOfConstituents,CFlag);
				if( c<=lastValue )break;
				lastValue = c;
			}
				*zeroFlag = FloodKey;
		}
		
		// if first value is max ebb
		// look for min before ebb
		else if(tHdl[0]==EbbKey){
			//look for zero
			for(i=1;i<1441;i++){
				t = firstTime - dt*i;
				c = RStatCurrent(t,AMPA,epoch,refCur,numOfConstituents,CFlag);
				if( (c<=0) && (lastValue>0) )break;
				// check to make sure we cross zero
				if(c<lastValue){
						*zeroFlag = FloodKey;
						break;
				}
				
				lastValue = c;
			}
		}
		
		else {  // error
			return 25;
		}
		
		*zeroTime = t + dt/2.0;
		*zeroValue = RStatCurrent(*zeroTime,AMPA,epoch,refCur,numOfConstituents,CFlag);
		
		// Now find the time after the last time
		
		lastTime = THdl[nPts-1].val;
		lastValue = CHdl[nPts-1];
				
		// go for max of 24 hours at 1 minute intervals
		// and look for max or min
		
		
		// if last time is a min before flood
		// look for flood
		if( tHdl[nFE-1]==MinBFloodKey){ 
			
			for(i=1;i<1441;i++){
				t = lastTime + dt*i;
				c = RStatCurrent(t,AMPA,epoch,refCur,numOfConstituents,CFlag);
				if(c<=lastValue)break;
				lastValue = c;
			}
			*extraFlag = FloodKey;
		}
		
		// if last time is a max flood
		// look for min before ebb
		else if( tHdl[nFE-1]==FloodKey){
			
			for(i=1;i<1441;i++){
				t = lastTime + dt*i;
				c = RStatCurrent(t,AMPA,epoch,refCur,numOfConstituents,CFlag);
				if( (c<=0.0) && (lastValue>=0.0)){
					*extraFlag = MinBEbbKey;
					break;
				}
				
				//OK now check incase we don't slack but go
				// directly to max ebb ... currents don't turn for ebb
				if( c > lastValue){
					*extraFlag = EbbKey;
					break;
				}
				
				lastValue = c;
			}
		}
		
		// if last time is a min before ebb
		// look for ebb
		else if( tHdl[nFE-1]==MinBEbbKey){
			
			for(i=1;i<1441;i++){
				t = lastTime + dt*i;
				c = RStatCurrent(t,AMPA,epoch,refCur,numOfConstituents,CFlag);
				if( c>=lastValue)break;
				lastValue = c;
			}
			*extraFlag = EbbKey;
		}
		
		// if last time is a max ebb
		// look for min before flood
		else if( tHdl[nFE-1]==EbbKey){
			
			for(i=1;i<1441;i++){
				t = lastTime + dt*i;
				c = RStatCurrent(t,AMPA,epoch,refCur,numOfConstituents,CFlag);
				if( (c>=0.0) && (lastValue<=0.0)){
					*extraFlag = MinBFloodKey;
					break;
				}
				//OK now check incase we don't slack but go
				// directly to max flood ... currents don't turn for flood
				if( c < lastValue){
					*extraFlag = FloodKey;
					break;
				}
				lastValue = c;
			}
		}
		
		*extraTime = t - dt/2.0;
		*extraValue = RStatCurrent(*extraTime,AMPA,epoch,refCur,numOfConstituents,CFlag);
		
	return 0;
}

//

short FindExtraFERot(double				*AMPA,				// amplitude corrected for year
						double			*epoch,				// epoch corrected for year
						short			numOfConstituents,	// number of frequencies
						COMPCURRENTS	*answers,			// Height-time struc with answers
						double			*zeroTime,			// first max-min time
						double			*zeroValue,			// first max-min value
						short			*zeroFlag,			// first max-min flag
						double			*lastTime,			// last max-min value
						double			*lastValue,			// last max-min value
						short			*lastFlag,			// last max-min flag
						CONSTITUENT *constituent)		// Handle to constituent data               
						
// Function to compute extra high and low values
// if current data is rotary data
// We need one value before first high low and one after
// So we can interpolate

{
	long i,nPts,nFE;
	double dt,theCurrent,theTime,firstTime;
	double uVelocity,vVelocity;
	double		*CHdl;
	EXTFLAG 	*THdl;
	short		*KeyHdl;
	short direcKey,zeroKey,slopeKey,MaxMinFlag,errorFlag;
	double vMajor,vMinor;
	double t0,t1,t2;
	double val0,val1,val2;

	short minBFloodKey = 0;
	short floodKey = 1;
	short minBEbbKey = 2;
	short ebbKey = 3;
	
	
	// OK begin by looking for first one before our data begins
	
	// Check first if max value or min value
	
		KeyHdl = answers->EbbFlood;
		THdl = answers->time;
		CHdl = answers->speed;
		nPts = answers->nPts;
		nFE  = answers->numEbbFloods;
		
		t2 = THdl[1].val;
		t1 = THdl[0].val;
		val2 = CHdl[1];
		val1 = CHdl[0];
		firstTime = t1;
		
		dt = 10.0/60.0;
		
		// go for max of 24 hours at 1 minute intervals
		// and look for max or min.
		
		for (i=1;i<145;i++){
			t0 = firstTime - dt*i;
			val0 = RStatCurrentRot(t0,AMPA,epoch,numOfConstituents,constituent,999.0,999.0,
				&uVelocity,&vVelocity,&vMajor,&vMinor,&direcKey);
		
			// check for zero crossing
			zeroKey = zeroCross(val0,val1,val2);
			
			// check for slope change
			slopeKey = slopeChange(val0,val1,val2);
			
			// OK if we have a slope change or a sign change
			// we loop through at 1 minute intervals and
			// pick up the time and value we need to the minute
			
			if( (zeroKey!=0) || (slopeKey!=0) ){
			
				// OK now figure out what we are looking for ... to set MaxMinFlag
				
				// if first max - min was minBFloodKey we are looking
				// for ebb
				if( KeyHdl[0]==minBFloodKey){
					MaxMinFlag = ebbKey;
				}
				// if first is flood, look for min before flood
				else if(KeyHdl[0]==floodKey) {
					MaxMinFlag = minBFloodKey;
				}
				// if first is min before ebb, we look for flood
				else if(KeyHdl[0]==minBEbbKey){
					MaxMinFlag = floodKey;
				}
				// OK all's left if ebb for first and look for min before ebb
				else {
					MaxMinFlag = minBEbbKey;
				}
				
				// OK now go find the max or min to the minute
				
				errorFlag = FindFloodEbbRot(t0,t2,MaxMinFlag,AMPA,epoch,numOfConstituents,
					&theCurrent,&theTime,constituent,999,999);
					
				if(errorFlag!=0)return errorFlag;
				
				break;
			}
			
			// If we get here, we just continue
			else {
				t2 = t1;
				t1 = t0;
				val2 = val1;
				val1 = val0;
			}
		}
		
		//OK we have found max - min before first data point
		// Save info and go on to find max - min beyond last data
		
		*zeroTime = theTime;
		*zeroValue = theCurrent;
		*zeroFlag = MaxMinFlag;

// *************************************************

	// Now find the time after the last time
		
		t0 = THdl[nPts-2].val;
		t1 = THdl[nPts-1].val;
		val0 = CHdl[nPts-2];
		val1 = CHdl[nPts-1];
		
		dt = 10.0/60.0;
		
		firstTime = t1;
		
		// go for max of 24 hours at 10 minute intervals
		// and look for max or min.
		
		for (i=1;i<145;i++){
			t2 = firstTime + dt*i;
			val2 = RStatCurrentRot(t2,AMPA,epoch,numOfConstituents,constituent,val0,val1,
				&uVelocity,&vVelocity,&vMajor,&vMinor,&direcKey);
		
			// check for zero crossing
			zeroKey = zeroCross(val0,val1,val2);
			
			// check for slope change
			slopeKey = slopeChange(val0,val1,val2);
			
			// OK if we have a slope change or a sign change
			// we loop through at 1 minute intervals and
			// pick up the time and value we need to the minute
			// Can't check if in first step becuase we already picked
			// it up in the regular data.
			if( ( (zeroKey!=0) || (slopeKey!=0) ) && (i>1) ){
			
				// OK now figure out what we are looking for ... to set MaxMinFlag
				
				// if last max - min was minBFloodKey we are looking
				// for flood
				if( KeyHdl[nFE-1]==minBFloodKey){
					MaxMinFlag = floodKey;
				}
				// if last is flood, look for min before ebb
				else if(KeyHdl[nFE-1]==floodKey) {
					MaxMinFlag = minBEbbKey;
				}
				// if last is min before ebb, we look for ebb
				else if(KeyHdl[nFE-1]==minBEbbKey){
					MaxMinFlag = ebbKey;
				}
				// OK all's left if ebb for last and look for min before flood
				else {
					MaxMinFlag = minBFloodKey;
				}
				
				// OK now go find the max or min to the minute
				
				errorFlag = FindFloodEbbRot(t0,t2,MaxMinFlag,AMPA,epoch,numOfConstituents,
					&theCurrent,&theTime,constituent,val0,val1);
					
				if(errorFlag!=0)return errorFlag;
				
				break;
			}
			
			// If we get here, we just continue
			else {
				t0 = t1;
				t1 = t2;
				val0 = val1;
				val1 = val2;
			}
		}
		
		//OK we have found max - min before first data point
		// Save info and go on to find max - min beyond last data
		
		*lastTime = theTime;
		*lastValue = theCurrent;
		*lastFlag = MaxMinFlag;
		
	return 0;
}

/***************************************************************************************/

short FindFloodEbb(	double	startTime,			// start time in hrs from begining of year
					double	endTime,			// end time in hrs from begining of year
					short	MaxMinFlag,			// flag = 0 for low, flag = 1 for high
					double	*AMPA,				// corrected amplitude array
					double	*epoch,				// corrected epoch array
					short	numOfConstituents,	// number of frequencies
					double	*theCurrent,		// the high or low tide value
					double	*theTime,			// the high or low tide time
					double	refCur,				// reference current in knots
					short	CFlag)				// hydraulic station flag
					
// routine to take start time, end time and solve for
// current at 1 minute intervals.  It picks out the low or high
// value and returns it and it's time.
{
	double timeSpan,t,c;
	double t0,t1,c0,c1;
	short findFlag;
	long numOfSteps,i;
	short maxFlood=1,minBFlood=0;
	short minBEbb=2, maxEbb=3;
	 
	// compute number of time steps ... if start time and end time
	// less than or equal to 3 minutes apart, interpolate and return
	
	timeSpan = endTime - startTime;
	if(timeSpan<= .05) {  // less than 3 minutes
		t = (startTime + endTime)/2.0;
		c = RStatCurrent(t,AMPA,epoch,refCur,numOfConstituents,CFlag);
	    *theCurrent = c;
		*theTime = t;
		return 0;
	}
	
	// Now if we get here, we gotta step through time and search
	// for flood or ebb on 1 minute intervals
	// The plan is that if we are looking for a low, then
	// we march along and saving the last value and when
	// the slope changes to positive we grab and the last
	// value and cut out.
	// We do just the opposite for searching for high values.
	// We also gotta find zero crossings
	
	numOfSteps = timeSpan * 60.0 + 1.5;
	
	t0 = startTime;
	c0 = RStatCurrent(t0,AMPA,epoch,refCur,numOfConstituents,CFlag);
	findFlag=-1;
	
	for (i=1; i<numOfSteps; i++){
		t1 = t0 + 1.0/60.0;
		c1 = RStatCurrent(t1,AMPA,epoch,refCur,numOfConstituents,CFlag);
		if(MaxMinFlag==maxFlood){
			if( c1<=c0){
				t=t0;
				c=c0;
				findFlag=maxFlood;
				break;
			}
		}
		else if(MaxMinFlag==maxEbb){
			if( c1>=c0) {
				t=t0;
				c=c0;
				findFlag=maxEbb;
				break;
			}
		}
		else if(MaxMinFlag==minBEbb){
			if( (c0>0) && (c1<0)) {
				t=t0;
				c=c0;
				findFlag=minBEbb;
				break;
			}
		}
		else if(MaxMinFlag==minBFlood){
			if( (c0<0) && (c1>0)){
				t=t0;
				c=c0;
				findFlag=minBFlood;
				break;
			}
		}
		
		c0=c1;
		t0=t1;
	}
	//if(findFlag==-1)return 18; // error!
	if(findFlag==-1)
	{
		*theCurrent = 0;
		*theTime = (startTime + endTime)/2;
	}
	else
	{
		*theCurrent = c;
		*theTime = t;
	}
	return 0;
}


/***************************************************************************************/

short FindFloodEbbRot(double		startTime,			// start time in hrs from begining of year
					double			endTime,			// end time in hrs from begining of year
					short			MaxMinFlag,			// flag = 0 for low, flag = 1 for high
					double			*AMPA,				// corrected amplitude array
					double			*epoch,				// corrected epoch array
					short			numOfConstituents,	// number of frequencies
					double			*theCurrent,		// the high or low tide value
					double			*theTime,			// the high or low tide time
					CONSTITUENT		*constituent,
					double			oldTwoCurrentsAgo,
					double			oldLastCurrent)
					
// routine to take start time, end time and solve for
// current at 1 minute intervals.  It picks out the low or high
// value and returns it and its time.
{
	double timeSpan,t,c,uVelocity,vVelocity,vMajor,vMinor;
	double t0,t1,c0,c1;
	double twoCurrentsAgo,lastCurrent;
	short findFlag;
	long numOfSteps,i;
	short maxFlood=1,minBFlood=0;
	short minBEbb=2, maxEbb=3;
	short direcKey;
	 
	// compute number of time steps ... if start time and end time
	// less than or equal to 3 minutes apart, interpolate and return
	 c1 = 0.0;
	 
	timeSpan = endTime - startTime;
	if(timeSpan<= .05) {  // less than 3 minutes
		t = (startTime + endTime)/2.0;
		c = RStatCurrentRot(t,AMPA,epoch,numOfConstituents,constituent,oldTwoCurrentsAgo,
							oldLastCurrent,&uVelocity,&vVelocity,&vMajor,&vMinor,&direcKey);
	    *theCurrent = c;
		*theTime = t;
		return 0;
	}
	
	// Now if we get here, we gotta step through time and search
	// for flood or ebb on 1 minute intervals
	// The plan is that if we are looking for a low, then
	// we march along and saving the last value and when
	// the slope changes to positive we grab and the last
	// value and cut out.
	// We do just the opposite for searching for high values.
	// We also gotta find zero crossings
	
	numOfSteps = timeSpan * 60.0 + 1.5;
	
	t0 = startTime;
	c0 = oldTwoCurrentsAgo;
	findFlag=-1;
	
	for (i=1; i<numOfSteps; i++){
		t1 = t0 + 1.0/60.0;
		if(i==1){
			twoCurrentsAgo = 999.0;
			lastCurrent = c0;
		}
		else{
			twoCurrentsAgo = lastCurrent;
			lastCurrent = c1;
		}
		
		c1 = RStatCurrentRot(t1,AMPA,epoch,numOfConstituents,constituent,twoCurrentsAgo,
							lastCurrent,&uVelocity,&vVelocity,&vMajor,&vMinor,&direcKey);
		if(MaxMinFlag==maxFlood){
			if( (c1<=c0) && (c0!=999.0) ){
				t=t0;
				c=c0;
				findFlag=maxFlood;
				break;
			}
		}
		else if(MaxMinFlag==maxEbb){
			if( (c1>=c0) && (c0!=999.0) ){
				t=t0;
				c=c0;
				findFlag=maxEbb;
				break;
			}
		}
		else if(MaxMinFlag==minBEbb){
			if( ( (c0>0) && (c1<0)) && (c0!=999.0) ){
				t=t0;
				c=c0;
				findFlag=minBEbb;
				break;
			}
			// here we check if we go through a min without
			// crossing the zero axis
			else if( (fabs(c1)>fabs(c0)) && (c0!=999.0) ){
				t=t0;
				c=c0;
				findFlag=minBEbb;
				break;
			}
		}
		else if(MaxMinFlag==minBFlood){
			if(  ( (c0<0) && (c1>0)) && (c0!=999.0) ){
				t=t0;
				c=c0;
				findFlag=minBFlood;
				break;
			}
			else if( (fabs(c1)>fabs(c0))&& (c0!=999.0) ){
				t=t0;
				c=c0;
				findFlag=minBFlood;
				break;
			}
		}
		
		c0=c1;
		t0=t1;
	}
	//if(findFlag==-1)return 18; // error!
	if(findFlag==-1)
	{
		*theCurrent = 0;
		*theTime = (startTime + endTime)/2.;
	}
	else
	{
		*theCurrent = c;
		*theTime = t;
	}
	return 0;
}

short FixAngle(short angle)
{	
// input an angle from north ... direction is toward
/**************************************************************/
	// first quardrant
	if( (angle>=0) && (angle<=90) ){
		angle = 90 - angle;
		return angle;
	}
	if( (angle>90) && (angle<=180) ){
		angle = 450 - angle;
		return angle;
	}
	if( (angle>180) && (angle<=270) ){
		angle = 450 - angle;
		return angle;
	}
	if( (angle>270) && (angle<=360) ){
		angle = 450 - angle;
		return angle;
	}
	else

		return 0;

}

/***************************************************************************************/

void FixCEnds(COMPCURRENTS *answers,double beginTime, double endTime,short rotFlag)
// Fix the ends of the plotted points so we don't overshoot
// What we will do is set all the points before the beginTime to
//  the beginTime and all the points after endTime to endTime
//
{
	double		*theCurrentHdl,*theUHdl,*theVHdl;
	EXTFLAG *theTimeHdl,*EFTimesHdl;
	long numberOfPoints,i,startCross,endCross,numberOfEbbsFloods;
	double w1,w2,dt,dt1,ratio;
	double firstPointValue,lastPointValue;
	
	numberOfPoints	= answers->nPts;
	theTimeHdl		= answers->time;
	theCurrentHdl	= answers->speed;
	theUHdl			= answers->u;
	theVHdl			= answers->v;
	
	// OK check if offset station that uses rotary 
	// if it does, we turn off the rotFlag because
	// the u v handles are nilled out
	
	if( (theUHdl==0) || (theVHdl==0) ) rotFlag = 0;
	
	// the max min array stuff
	numberOfEbbsFloods = answers->numEbbFloods;
	EFTimesHdl = answers->EbbFloodTimes;
	
	// begin by finding where it crosses beginTime
	// and endTime
	
	startCross = -1;
	endCross = -1;
	for (i=1;i<numberOfPoints;i++){
		if(startCross==-1){
			if(theTimeHdl[i].val>beginTime){
				startCross = i-1;
			}
		}
		if(endCross==-1){
			if( endTime <= theTimeHdl[i].val ){
				endCross = i-1;
			}
		}
	}
	
	if(startCross==-1)startCross = 0;
	if(endCross==-1)endCross = numberOfPoints-2;
	
	// Now gotta get interpolated values
	
	dt = theTimeHdl[startCross+1].val - theTimeHdl[startCross].val;
	dt1 = beginTime - theTimeHdl[startCross].val;
	w2 = dt1/dt;
	w1 = 1.0 - w2;
	
	firstPointValue = w1 * theCurrentHdl[startCross] + 
	                  w2 * theCurrentHdl[startCross+1];
					  

	dt = theTimeHdl[endCross+1].val - theTimeHdl[endCross].val;
	dt1 = endTime - theTimeHdl[endCross].val;
	w2 = dt1/dt;
	w1 = 1.0 - w2;
	
	lastPointValue = w1 * theCurrentHdl[endCross] + 
	                 w2 * theCurrentHdl[endCross+1];
					  
	// Now reset one array value and time if they overshoot
	// If overshoot is more than one point, just change flag on time array
	
	for (i=0;i<(startCross+1);i++){
	
		if(i==startCross){
			theTimeHdl[i].val = beginTime;
		
			// OK now fix u and v velocity values if we have
			// rotary currents
		
			if( (rotFlag==1) || (rotFlag==2) ){
				ratio = firstPointValue/theCurrentHdl[i];
				theUHdl[i]=ratio*theUHdl[i];
				theVHdl[i]=ratio*theVHdl[i];
			}
		
			theCurrentHdl[i] = firstPointValue;
		}
		else {
			theTimeHdl[i].flag=1;
		}
	}

	for (i=(endCross+1);i<numberOfPoints;i++){
		if( i == (endCross+1) ){
			theTimeHdl[i].val = endTime;
		
			// OK now fix u and v velocity values if we have
			// rotary currents
		
			if(  (rotFlag==1) || (rotFlag==2) ){
				ratio = lastPointValue/theCurrentHdl[i];
				theUHdl[i]=ratio*theUHdl[i];
				theVHdl[i]=ratio*theVHdl[i];
			}
		
			theCurrentHdl[i] = lastPointValue;
		}
		else {
			theTimeHdl[i].flag = 1;
		}
	}
	
	// Now check flood and ebb array for overshoot
	
	for (i=0;i<numberOfEbbsFloods;i++){
		if( EFTimesHdl[i].val < beginTime) {
			EFTimesHdl[i].flag = 1;
		}
		else if( EFTimesHdl[i].val > endTime) {
			EFTimesHdl[i].flag = 1;
		}
	}
	
	return;
}

void FixCurve(COMPCURRENTS *answers)
                
//
// The idea here is to go through the velocity array for plotting and
// change the curve in the vicinity of the zero crossing.
// 
{
	double		*theCurrentHdl,*maxMinHdl;
	EXTFLAG 	*theTimeHdl,*maxMinTHdl;
	short 		*keyHdl;
	long numberOfPoints,i,nFloodEbbs,j,flag;
	double t,c,t0,t1,t2,t3;
	double v0,v1,v2,v3,s;
	
	numberOfPoints	= answers->nPts;
	theTimeHdl		= answers->time;
	theCurrentHdl	= answers->speed;
	nFloodEbbs		= answers->numEbbFloods;
	maxMinHdl		= answers->EbbFloodSpeeds;
	maxMinTHdl		= answers->EbbFloodTimes;
	keyHdl			= answers->EbbFlood;
	
	
	// Now loop through and do the Hermite fit
	
	for (i=1;i<nFloodEbbs;i++){
		t1 = maxMinTHdl[i-1].val;
		t2 = maxMinTHdl[i].val;
		v1 = maxMinHdl[i-1];
		v2 = maxMinHdl[i];
		if(i==1){
			v0 = theCurrentHdl[0];
			t0 = theTimeHdl[0].val;
			v3 = maxMinHdl[i+1];
			t3 = maxMinTHdl[i+1].val;
		}
		else if(i==(nFloodEbbs-1)){
			v3 = theCurrentHdl[numberOfPoints-1];
			t3 = theTimeHdl[numberOfPoints-1].val;
			v0 = maxMinHdl[i-2];
			t0 = maxMinTHdl[i-2].val;
		}
		else {
			v0 = maxMinHdl[i-2];
			t0 = maxMinTHdl[i-2].val;
			v3 = maxMinHdl[i+1];
			t3 = maxMinTHdl[i+1].val;
		}
		
		// OK now loop through all the points and if they fall
		// between t0 and t3, recompute using hermite
		
		flag = 0;
		for (j=0; j<numberOfPoints; j++){
			t = theTimeHdl[j].val;
			
				if( (t<=t2) && (t>=t1) ){
					DoHermite(v0,t0,v1,t1,v2,t2,v3,t3,t,&c,flag);
					theCurrentHdl[j] = c;
				}
		
		}
	}

		for (j=0; j<numberOfPoints; j++){
			t = theTimeHdl[j].val;
			
			if( (t<maxMinTHdl[0].val) && (t>theTimeHdl[0].val) ){
					v0 = theCurrentHdl[0];
					t0 = theTimeHdl[0].val;
					v2 = maxMinHdl[0];
					t2 = maxMinTHdl[0].val;
					v3 = maxMinHdl[1];
					t3 = maxMinTHdl[1].val;
					 s = (v2-v0)/(t2-t0);
					 v1 = v0;
					 t1 = t0;
					DoHermite(s,t0,v1,t1,v2,t2,v3,t3,t,&c,1);
					theCurrentHdl[j] = c;
			}
			else if( (t>maxMinTHdl[nFloodEbbs-1].val) && (t<theTimeHdl[numberOfPoints-1].val) ){
					v0 = maxMinHdl[nFloodEbbs-2];
					t0 = maxMinTHdl[nFloodEbbs-2].val;
					v1 = maxMinHdl[nFloodEbbs-1];
					t1 = maxMinTHdl[nFloodEbbs-1].val;
					v2 = theCurrentHdl[numberOfPoints-1];
					t2 = theTimeHdl[numberOfPoints-1].val;
					t3 = t2;
					s  = (v2-v1)/(t2-t1);
					
					DoHermite(v0,t0,v1,t1,v2,t2,s,t3,t,&c,2);
					theCurrentHdl[j] = c;
			}
		
		}
	return;
}

/***************************************************************************************/

short FixMajMinFlags(EXTFLAG *THdl,
					 double *CHdl,
					 short NumOfSteps,
					 double *MaxMinHdl,
					 EXTFLAG *MaxMinTHdl,
					 short MaxMinCount)
					 
// Function to check plot flag for major - minor current
// station.  Function makes sure that we have no plot
// flags in pairs.
{
	short i,j,k,k1,isign,jsign,jstop;
	double oldSpeed,newSpeed,dv,oldTime,newTime,dt;
	
	
		for (i= 1; i< (NumOfSteps-1); i++){
			if(  ( (THdl[i].flag==1) && (THdl[i+1].flag!=1) ) &&
				 ( (THdl[i].flag==1) && (THdl[i-1].flag!=1) ) ){
				
				// OK back up and look for transition
				isign = 1;
				if(CHdl[i] < 0.0)isign = -1;
				jstop = i-5;
				if(jstop<0)jstop = 0;
				k = 0;
				
				for (j= (i-1);j>jstop;j--){
					jsign = 1;
					if( CHdl[j] < 0.0) jsign = -1;
					if(jsign!=isign){
						k = j;
						break;
					}
				}
				
				if(k!=0){
					for (j=k;j<i;j++){
						THdl[j].flag=1;
					}
				}
				
				else {
					// OK if we get here we gotta search foward
					// to look for sign change
				
					jstop = i+6;
					
					if(jstop > NumOfSteps ) jstop = NumOfSteps;
					for (j= (i+1);j<jstop;j++){
						jsign = 1;
						if( CHdl[j] < 0.0) jsign = -1;
						if(jsign!=isign){
							k = j;
							break;
						}
					}
					if(k!=0){
						for(j=i;j<(k+1);j++){
							THdl[j].flag=1;
						}
					
						// OK here we modify the index i so we
						// can continue the search foward for 
						// more wierdness
					
						i = k + 1;
					}
				
					// here we have problems!
					else {
						// Let's just take out the offending code
						THdl[j].flag=0;
					}
					
				}
								
			}
		}

// Now check to make sure that we span the transition between
// flood and ebb with the no plot flag

		
		for (i= 1; i< (NumOfSteps-1); i++){
		
			if(  ( (THdl[i].flag==1) && (THdl[i+1].flag==1) ) ){
				// OK make sure we cross the 0 line
				k = 0;
				isign = 1;
				if(CHdl[i] < 0.0)isign = -1;
				jstop = i+6;
				if(jstop > NumOfSteps ) jstop = NumOfSteps;
				for (j= (i+1);j<jstop;j++){
					jsign = 1;
					if( CHdl[j] < 0.0) jsign = -1;
					if(jsign!=isign){
						k = j;
						break;
					}
				}
				
				// OK now extend the no plot flag if it doesn't span the
				// transition from flood to ebb
				
				for (j=i;j<(k+1);j++){
					THdl[j].flag=1;
				}
			}
			
		}
		
	// OK now we check max min arrays.  This check will look for
	// a flat curve where the algorythm picks up on extra mins and
	// maxs.  The tide book ignores these so we will too.  The
	// check (and I hope it works) will set the no plot flag if
	// the current differences are less than 0.005 knots in less
	// than 1 hour.  For an example, check Boston Harbor entrance
	// on March 17, 1994.
	
	for (i=0;i< (MaxMinCount-1);i++){
		 k=i;
		 k1 = i+1;
		 if(MaxMinTHdl[i].flag==1) {
		 	if(i>0)k=i-1;
		 }
		 if(MaxMinTHdl[i+1].flag==1) {
		 	k1=i+2;
		 }
		 
		oldSpeed = MaxMinHdl[k];
		newSpeed = MaxMinHdl[k1];
		 
		// check difference
		dv = fabs(newSpeed - oldSpeed);
		if( (dv<0.005)){
			// now check time difference
			oldTime = MaxMinTHdl[k].val;
			newTime = MaxMinTHdl[k1].val;
			dt = fabs(newTime - oldTime);
			if(dt<1.0){
				// OK ingnore max min ... curve too flat
				MaxMinTHdl[k1].flag = 1;
			}
		}
	}
	
	
	// OK here is a wierd one
	// We check to see if for rotary stations the transition from
	// flood to ebb doesn't take a quantum jump.  This could happen
	// if our check for the dot product doesn't produce any no plot flags
	
	for (i=0;i< (NumOfSteps-1);i++){
		if( ( THdl[i].flag==0) && (THdl[i+1].flag==0) ) {
			oldSpeed = CHdl[i];
			newSpeed = CHdl[i+1];
		 	if( ( (oldSpeed>0.0) && (newSpeed<0.0) ) || ( (oldSpeed < 0.0) &&
			      (newSpeed > 0.0) ) ){
				  dv = fabs( newSpeed - oldSpeed);
				  if(dv > 0.1 ){
				  	THdl[i].flag = 1;
				  	THdl[i+1].flag = 1;
				  }
			}
		}
	}
	return 0;
}

//*********************************************************************
short GetCDirec(CONSTITUENT *constituent, short *FDir, short *EDir)
{	
	short errorFlag,flood,ebb;
	
    // Assume space already allocated

	//flood = constituent->DatumControls.FDir;
	//ebb = constituent->DatumControls.EDir;
	
	flood = constituent[0].DatumControls.FDir;			// more explicit
	ebb = constituent[0].DatumControls.EDir;			// more explicit
	
	*FDir = FixAngle(flood);
	*EDir = FixAngle(ebb);
	
	// check value
	errorFlag = 0;
	if( (*FDir<0) || (*FDir>360) )errorFlag = 33;
	if( (*EDir<0) || (*EDir>360) )errorFlag = 33;
	if(*FDir==*EDir)errorFlag = 33;
	return errorFlag;
}

/*---------------------------------------------*/
short GetControlFlags(CONSTITUENT *constituent,	// Constituent data handle
                      short *L2Flag,					// L2 frequency flag
					  short *HFlag,						// Hydraulic station flag
                      short *RotFlag)					// Rotary station flag.........if any flag = 1 we got anomaly
{	
	short	err=0;
	
    // Assume space already allocated

	(*L2Flag)	= (constituent->DatumControls).L2Flag;
	(*HFlag)	= (constituent->DatumControls).HFlag;
	(*RotFlag)	= (constituent->DatumControls).RotFlag;
	
	// Check values to make sure it makes sense

	if( (*L2Flag < 0) || (*L2Flag > 1) ){ err = 1; }
	if( (*HFlag < 0) || (*HFlag > 1) ){ err = 1; }
	if( (*RotFlag < 0) || (*RotFlag > 3) ){ err = 1; }

	return(err);
}

/*---------------------------------------------*/
short GetControlFlagsAlternate(	CONTROLVAR *cntrlvars,	// Control variables structure
                     			short *L2Flag,			// L2 frequency flag
								short *HFlag,			// Hydraulic station flag
               					short *RotFlag)			// Rotary station flag.........if any flag = 1 we got anomaly
{	
	// code not used
	short	err=0;
	
    // Assume space already allocated

	(*L2Flag)	= cntrlvars->L2Flag;
	(*HFlag)	= cntrlvars->HFlag;
	(*RotFlag)	= cntrlvars->RotFlag;
	
	// Check values to make sure it makes sense

	if( (*L2Flag < 0) || (*L2Flag > 1) ){ err = 1; }
	if( (*HFlag < 0) || (*HFlag > 1) ){ err = 1; }
	if( (*RotFlag < 0) || (*RotFlag > 3) ){ err = 1; }

	return(err);
}

/*---------------------------------------------*/
double GetDatum(CONSTITUENT *constituent)
{	
	double datum;

	datum = constituent->DatumControls.datum;
	return datum;
}

/*---------------------------------------------*/
double GetDatum2(CONSTITUENT *constituent,short index)

// function to pull of reference current for rotary data
{	
	double datum;

	datum =constituent[index].DatumControls.datum;
	
	return datum;
}


/*---------------------------------------------*/
short GetFloodEbbKey(double t,
					  EXTFLAG *TArrayPtr,
					  short *MaxMinFlagPtr,
					  double *CArrayPtr,
					  short numOfMaxMins,
					  short *flag)
{
	short i;
	short EbbToMinBFlood = 0, MinBFloodToFlood = 1;
	short FloodToMinBEbb=2, MinBEbbToEbb=3;
	short FloodToEbb=4, EbbToFlood=5, FloodToFlood=6, EbbToEbb=7;
	short errorFlag,previousIndex,nextIndex;
	double previousValue,nextValue;
	short maxFlood = 1, minBFlood = 0, maxEbb = 3, minBEbb = 2;
	
	errorFlag = 0;
	nextIndex = -1;

//
//  Special check for first segment
//
	if(TArrayPtr[0].val>t){
		// back up to previous max min and make sure
		// we are in range
		if(TArrayPtr[numOfMaxMins].val>t){
			return 40;
		}
		
		previousValue = CArrayPtr[numOfMaxMins];
		nextValue = CArrayPtr[0];
		previousIndex = numOfMaxMins;
		nextIndex = 0;
	}
	
//
// Special check for last segment
//
	if(TArrayPtr[numOfMaxMins-1].val<t){
		if(TArrayPtr[numOfMaxMins+1].val<t){
			return 40;
		}
		
		previousValue = CArrayPtr[numOfMaxMins-1];
		nextValue = CArrayPtr[numOfMaxMins+1];
		previousIndex = numOfMaxMins-1;
		nextIndex = numOfMaxMins+1;
	}
	
	 if(nextIndex==-1){

		for (i=0;i<numOfMaxMins;i++){
	
			if(TArrayPtr[i].val<=t){
				previousValue = CArrayPtr[i];
				previousIndex = i;
			}
			if(nextIndex==-1){
				if(TArrayPtr[i].val>t){
					nextIndex=i;
					nextValue = CArrayPtr[i];
					break;
				}
			}
		}
	
	}
	
	// OK by the time we get here, we have a previous and a next value ...
	
	if(nextIndex==-1)return 40;
	
	// Now check values to come up with code
	
	// Normal stuff first where one of the values is 0 
	
	if(previousValue==0.0){
		if(nextValue<0.0){
			*flag = MinBEbbToEbb;
			return 0;
		}
		else if(nextValue>0.0){
			*flag = MinBFloodToFlood;
			return 0;
		}
		else {
			return 40;
		}
	}
	if(nextValue==0.0){
		if(previousValue<0.0){
			*flag = EbbToMinBFlood;
			return 0;
		}
		else if(previousValue>0.0){
			*flag = FloodToMinBEbb;
			return 0;
		}
		else {
			return 40;
		}
	}
	
	// Now look for weird ones with no zero velocities

	if(MaxMinFlagPtr[previousIndex]==minBFlood){
		if(MaxMinFlagPtr[nextIndex]==maxFlood){
			*flag = MinBFloodToFlood;
			return 0;
		}
		return 40;
	}

	if(MaxMinFlagPtr[previousIndex]==minBEbb){
		if(MaxMinFlagPtr[nextIndex]==maxEbb){
			*flag = MinBEbbToEbb;
			return 0;
		}
		return 40;
	}

	
	if(MaxMinFlagPtr[previousIndex]==maxFlood){
		if(MaxMinFlagPtr[nextIndex]==maxEbb){
			*flag = FloodToEbb;
			return 0;
		}
		if(MaxMinFlagPtr[nextIndex]==minBEbb){
			*flag = FloodToMinBEbb;
			return 0;
		}
        // 03/05/2014 G.K.: multiple flood case
		if(MaxMinFlagPtr[nextIndex] == maxFlood)
        {
			*flag = FloodToFlood;
			return 0;
		}
		return 40;
	}

	if(MaxMinFlagPtr[previousIndex]==maxEbb){
		if(MaxMinFlagPtr[nextIndex]==maxFlood){
			*flag = EbbToFlood;
			return 0;
		}
		if(MaxMinFlagPtr[nextIndex]==minBFlood){
			*flag = EbbToMinBFlood;
			return 0;
		}
        // 03/05/2014 G.K.: multiple ebb case
		if(MaxMinFlagPtr[nextIndex] == maxEbb)
        {
			*flag = EbbToEbb;
			return 0;
		}
		return 40;
	}

	// If we get to here, we got problem
	return 40;
}
