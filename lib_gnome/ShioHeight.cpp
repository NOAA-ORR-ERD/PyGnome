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

//#include "Global.h"
//#include "TCTypedefs.h"
//#include "CompHeight.h"
//#include "CompCurrent.h"
//#include "ShowProgress.h"
//#include "Shio.h"
//#include "MyUtils.h"

short GetTideHeight(	DateTimeRec *BeginDate,DateTimeRec *EndDate,
						double* XODE, double* VPU, 	// Year correction
						short numOfConstituents, 
						CONSTITUENT *constituent,	// Amplitude-phase array structs
						HEIGHTOFFSET *htOffset,		// Height offset data
						COMPHEIGHTS *answers,		// Height-time struc with answers

						//double **minmaxvalhdl,			// Custom entered reference curve vals
						//double **minmaxtimehdl,			// Custom entered reference curve vals
						//long nminmax,					// number of entries
						//CONTROLVAR *cntrlvars,			// custom control vars
						Boolean DaylightSavings)		// Daylight Savings Flag.
						
// Here we compute several arrays containing
// the reference curve and the highs and lows
// the reference curve is computed at user specified dt
// the highs and low will to computed to the minute like NTP4
//
{
	char		str1[256]="";
	double		beginHour=0.0,endHour=0.0,timestep=10.0,deltaTime=0.0,dt=0.0;
	double		oldBeginHour=0.0,oldEndHour=0.0;
	double		biggestOffset=0.0,absBiggestOffset=0.0,theHour=0.0;
	long		numberOfSteps=0;
	short		errorFlag=0;
	short		beginDay=0,beginMonth=0,beginYear=0,endDay=0,endMonth=0,endYear=0;


	/* Variables to overlay reference curve */
	//EXTFLAG		*reftimeHdl=0; 
	//double		*refheightHdl=0;
	double		tval=0.0,ehour=0.0,julianoffset=0.0,grafBeginHour=0.0,grafEndHour=0.0;
	long		i=0,size1=0,size2=0,numpnts=0;


	/* Determine if offset information is valid by checking dataAvailFlag flag */
	errorFlag=CheckHeightOffsets(htOffset);
	if(errorFlag)goto Error;
#ifndef pyGNOME
	/////SetCursor(*GetCursor(watchCursor));
	SetWatchCursor();
#endif

	/* Find beginning hour and ending hour */
	beginDay	= BeginDate->day;
	beginMonth	= BeginDate->month;
	beginYear	= BeginDate->year;
	timestep	= 10.0;	/* timestep always in 10 minute increments now */
	dt			= (timestep / 60.0);	/* convert to hours */

	errorFlag	= GetJulianDayHr(beginDay,beginMonth,beginYear,&beginHour);
	if(errorFlag!=0){ goto Error; }


	endDay		= EndDate->day;
	endMonth	= EndDate->month;
	endYear		= EndDate->year;	

	errorFlag	= GetJulianDayHr(endDay,endMonth,endYear,&endHour);
	if(errorFlag!=0){ goto Error; }

	// check to see if we span year
	if( endYear > beginYear ){
		errorFlag=GetJulianDayHr(31,12,beginYear,&theHour);
		if(errorFlag!=0){ goto Error; }

		endHour=(theHour+endHour+24.0);
	}else{
		// end hour set to end of day
		endHour=(endHour+24.0);
	}
	/* Set the begin and end hours for overlaying the reference curve */
	grafBeginHour=beginHour;
	grafEndHour=endHour;


	// OK now adjust beginning and ending hour so
	// that when we apply offsets, we still span the
	// requested days
	
	biggestOffset = htOffset->HighTime.val;
	absBiggestOffset = fabs(biggestOffset);
	
	if( absBiggestOffset < fabs( htOffset->LowTime.val) ){
		biggestOffset = htOffset->LowTime.val;
		absBiggestOffset = fabs(biggestOffset);
	}

	// OK now check if daylight savings time is in effect
	// if it is, we set beginHour to beginHour-1.
	// Remember, spring forward, so if it is 0000 hour in standard
	// time, it translates into 0100 in daylight.  This means
	// to get data between 0000 and 0100 in daylight, we need
	// to compute from one hour less

	if( DaylightSavings ){
		beginHour		= beginHour		-1.0;
		endHour			= endHour		-1.0;
		grafBeginHour	= grafBeginHour	-1.0;
		grafEndHour		= grafEndHour	-1.0;
	}

	oldBeginHour = beginHour;
	oldEndHour = endHour;

	// Lookit, we just offset both ends by the largest
	// offset.  This way, we don't have to worry about
	// what the offset is for, where it falls on the
	// curve and if it positive or negative.
	
	endHour = endHour + absBiggestOffset;
	beginHour = beginHour - absBiggestOffset;
	
	// Pick up on extra time step at either end
	// just incase the first high or low is within
	// the first time step ... can't compute
	// gradient in first time step
	
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

	numberOfSteps = (long)( ((endHour-beginHour)/dt) + 1.0 );
	
	// OK go get year data
	// Begin by declaring space for the two arrays we need
	// Note the number we get should match the number of
	// constituents Andy is sending me.
	
	//if(constituent->kPrime) numOfConstituents = (short)( (long)GetHandleSize((Handle)constituent->kPrime) / (long)sizeof(float) );
	if( numOfConstituents < 5 ){
		errorFlag=6;
		goto Error;
	}
	
	

	/////errorFlag = GetYearData(XODEPtr,VPUPtr,beginYear);
	/////if(errorFlag!=0){ goto Error; }


	// OK we got all the pieces we need, now go an compute reference curve
	errorFlag = GetReferenceCurve(
									constituent,		// Amplitude-phase array structs
									numOfConstituents,	// Number of constituents usually 37
									XODE,				// Year correction to amplitude
									VPU,				// Year correction to phase
									beginHour,			// Begin hour from start of year
									endHour,			// End hour from start of year
									timestep,			// Comp Time step in minutes
									answers);		// Height-time struc with answers

	/////SetCursor(*GetCursor(watchCursor));
#ifndef pyGNOME
	SetWatchCursor();
#endif
	/* If an error was encountered, go to the errorCode function */
	if(errorFlag!=0){ goto Error; }



	/*----------- Attempt to overlay the reference curve -----------*/
	// actually, the reference curve data is simply appended onto the offset station curve
	// would need to redo this to eliminate handles

	

	

	/***********************************************************

	if( gOverlayReferenceCurve ){
		ehour=(grafEndHour-grafBeginHour);
		
		// Make space for ref curve, then append it to offset curve data 
		numpnts = (long)( (long)GetHandleSize((Handle)(answers->timeHdl)) / (long)sizeof(EXTFLAG) );
		reftimeHdl=(EXTFLAG **)NewHandleClear( numpnts * sizeof(EXTFLAG) );
		errorFlag = (short)MemError(); if( errorFlag != 0 ){ goto Error; }

		for(i=0;i<numpnts;i++){
			(*reftimeHdl)[i]=(*answers->timeHdl)[i]; (*reftimeHdl)[i].flag=1;
			(*reftimeHdl)[i].val -= grafBeginHour; 
			tval=(*reftimeHdl)[i].val;
			
			if( (tval > 0.0) && (tval < ehour)){ (*reftimeHdl)[i].flag=2; }
		}
		(*reftimeHdl)[0].flag=1;


		refheightHdl=(double **)NewHandleClear( GetHandleSize((Handle)(answers->heightHdl)) );
		errorFlag = (short)MemError(); if( errorFlag != 0 ){ goto Error; }

		for(i=0;i<numpnts;i++){
			(*refheightHdl)[i]=(*answers->heightHdl)[i];
		}
	}
	***********************************************************************/
	/* Do the offset correction for the station */
	errorFlag = DoOffSetCorrection(answers,htOffset);
	if(errorFlag!=0){ goto Error; }

	/* Cleanup points before begin hour or past end hour */
	FixHEnds(answers,oldBeginHour,oldEndHour);

	/* Reset time arrays to start from midnight of first day */
	ResetTime(answers,oldBeginHour);

	

	/***************************************

	if( gOverlayReferenceCurve && reftimeHdl && refheightHdl ){
		// Append time data 
		size1=GetHandleSize((Handle)reftimeHdl);
		size2=GetHandleSize((Handle)answers->timeHdl);

		HUnlock((Handle)answers->timeHdl); HUnlock((Handle)reftimeHdl);

		SetHandleSize((Handle)answers->timeHdl,(size1+size2));
		errorFlag=(short)MemError(); if(errorFlag!=0){ goto Error; }

		HLock((Handle)answers->timeHdl); HLock((Handle)reftimeHdl);

		numpnts=(long)( size1 / sizeof(EXTFLAG) );
		for(i=0;i<numpnts;i++){
			(*answers->timeHdl)[numpnts+i]=(*reftimeHdl)[i];
		}

		//get rid of reftimeHdl 
		HUnlock((Handle)answers->timeHdl); HUnlock((Handle)reftimeHdl);
		DisposeHandle((Handle)reftimeHdl); reftimeHdl=nil;


		// Append height data 
		size1=GetHandleSize((Handle)refheightHdl);
		size2=GetHandleSize((Handle)answers->heightHdl);

		HUnlock((Handle)answers->heightHdl); HUnlock((Handle)refheightHdl);

		SetHandleSize((Handle)answers->heightHdl,(size1+size2));
		errorFlag=(short)MemError(); if(errorFlag!=0){ goto Error; }

		HLock((Handle)answers->heightHdl); HLock((Handle)refheightHdl);

		for(i=0;i<numpnts;i++){
			(*answers->heightHdl)[numpnts+i] = (*refheightHdl)[i];
		}
		HUnlock((Handle)answers->heightHdl); HUnlock((Handle)refheightHdl);
		DisposeHandle((Handle)refheightHdl); refheightHdl=nil;
	}// End reference curve overlay 
	*************************************************************/


	//if( reftimeHdl ){ DisposeHandle((Handle)reftimeHdl); reftimeHdl=nil; }
	//if( refheightHdl ){ DisposeHandle((Handle)refheightHdl); refheightHdl=nil; }

	//HUnlock((Handle)answers->timeHdl); HUnlock((Handle)answers->heightHdl);


Error:


	CompErrors(errorFlag,str1);
	if((errorFlag!=0)){ printError(str1); }

	return( errorFlag );
}

/***********************************************************/

short CheckHeightOffsets(HEIGHTOFFSET *htOffset)
{
	short			err=0;
	double minTimeDiff = -24.0;
	double maxTimeDiff = 24.0;
	double minAddFac = -26.0;
	double maxAddFac = 26.0;
	double minMultFac = 0.0;
	double maxMultFac = 6.0;
	

	if (!htOffset)
	{
		err=25;
		return(err);
	}

	/* The dataAvailFlag field is a short of value 0 or 1 (0==insufficient data) */

	if(!htOffset->HighTime.dataAvailFlag){			/* High water time offset		*/
		err=22;
	}
	if(!htOffset->LowTime.dataAvailFlag){			/* Low water time offset		*/
		err=23;
	}
	if(!htOffset->HighHeight_Mult.dataAvailFlag){	/* High water height multiplier	*/
		err=26;
	}
	if(!htOffset->HighHeight_Add.dataAvailFlag){	/* High water height additive	*/
		err=27;
	}
	if(!htOffset->LowHeight_Mult.dataAvailFlag){	/* Low water height multiplier	*/
		err=28;
	}
	if(!htOffset->LowHeight_Add.dataAvailFlag){		/* Low water height additive	*/
		err=29;
	}
	
	if(err!=0)  return err;

	// OK if we have value, check it to see if it is within range
	
	// check High water time correction
	if( htOffset->HighTime.val < minTimeDiff ) err = 30;
	if( htOffset->HighTime.val > maxTimeDiff ) err = 30;
	
	// check Low water time correction
	if( htOffset->LowTime.val < minTimeDiff ) err = 31;
	if( htOffset->LowTime.val > maxTimeDiff ) err = 31;
	
	// check High water additive correction
	if( htOffset->HighHeight_Add.val < minAddFac ) err = 32;
	if( htOffset->HighHeight_Add.val > maxAddFac ) err = 32;
	
	// check Low water additive correction
	if( htOffset->LowHeight_Add.val < minAddFac ) err = 33;
	if( htOffset->LowHeight_Add.val > maxAddFac ) err = 33;

	// check High water mult correction
	if( htOffset->HighHeight_Mult.val < minMultFac ) err = 34;
	if( htOffset->HighHeight_Mult.val > maxMultFac ) err = 34;
	
	// check Low water mult correction
	if( htOffset->LowHeight_Mult.val < minMultFac ) err = 35;
	if( htOffset->LowHeight_Mult.val > maxMultFac ) err = 35;
	
	return(err);
}

void CleanUpCompHeights (COMPHEIGHTS *cPtr)
{
	double		*hHdl,*HLHHdl;
	EXTFLAG	*tHdl,*HLTHdl;
	short		*HLHdl;
	
	tHdl	= cPtr->time;
	hHdl	= cPtr->height;
	HLHHdl	= cPtr->HighLowHeights;
	HLTHdl	= cPtr->HighLowTimes;
	HLHdl	= cPtr->HighLow;
	
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

	cPtr->time			= 0;
	cPtr->height			= 0;
	cPtr->HighLowHeights	= 0;
	cPtr->HighLowTimes	= 0;
	cPtr->HighLow			= 0;
	
	return;
}


void CompErrors(short errorNum,char *errStr)
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
			strcpy(errStr,"not enough memory for HHDl handle");
			break;
		case 12:
			strcpy(errStr,"not enough memory for THdl handle");
			break;
		case 13:
			strcpy(errStr,"not enough memory for HLHHdl handle");
			break;
		case 14:
			strcpy(errStr,"not enough memory for HLTHdl handle");
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
			strcpy(errStr,"high and low array handle invalid");
			break;
		case 20:
			strcpy(errStr,"not enough memory for HArray array");
			break;
		case 21:
			strcpy(errStr,"not enough memory for TArray array");
			break;
		case 22:
			strcpy(errStr,"Insufficient data to compute heights. The high tide time offset data is unavailable.");
			break;
		case 23:
			strcpy(errStr,"Insufficient data to compute heights. The low tide time offset data is unavailable.");
			break;
		case 24:
			strcpy(errStr,"Unable to compute heights. Time offset data unusable for selected time span.");
			break;
		case 25:
			strcpy(errStr,"Programmer error. Handle to offset data nil");
			break;
		case 26:
			strcpy(errStr,"Insufficient data to compute heights. The high tide multiplicative factor unavailable.");
			break;
		case 27:
			strcpy(errStr,"Insufficient data to compute heights. The high water additive factor unavailable.");
			break;
		case 28:
			strcpy(errStr,"Insufficient data to compute heights. The low water multiplicative factor unavailable");
			break;
		case 29:
			strcpy(errStr,"Insufficient data to compute heights. The low water additive factor unavailable");
			break;
		case 30:
			strcpy(errStr,"high tide time corrections out of range");
			break;
		case 31:
			strcpy(errStr,"low tide time corrections out of range");
			break;
		case 32:
			strcpy(errStr,"high tide add height difference out of range");
			break;
		case 33:
			strcpy(errStr,"low tide add height difference out of range");
			break;
		case 34:
			strcpy(errStr,"high tide mult factor out of range");
			break;
		case 35:
			strcpy(errStr,"low tide mult factor out of range");
			break;
		case 41:
			strcpy(errStr,"HighLowCount >= maxPeaks in GetReferenceCurve");
			break;
		//case kInterrupt:
		//	strcpy(errStr,"User Interrupt.");
		//	break;
		default:
			strcpy(errStr,"unknown error code");
			break;
	}
	return;

}

short DoOffSetCorrection (COMPHEIGHTS *answers,  // Hdl to reference station heights
					HEIGHTOFFSET *htOffset)  // Hdl to offset data

{
	short		NoOfHighsLows,index,errorFlag,flag;
	long		numOfPoints,i,j;
	double		highOffset,lowOffset,t;
	double		HighAdd,HighMult;
	double		LowAdd,LowMult;
	short		low = 0, high = 1, lowToHigh = 0, highToLow = 1;
	double		timeCorrection,HeightCorrectionMult,HeightCorrectionAdd;
	double		w1,w2;
	double		*HLHeightHdl = 0,*ValHdl = 0;
	EXTFLAG		*HLTimeHdl = 0,*TimeHdl = 0;
	short		*HLFlagHdl = 0;
	double		*HArrayPtr = 0;
	EXTFLAG		*TArrayPtr = 0;
	
	errorFlag=0;
	
	// Get handles just to make code more readable
	
	HLHeightHdl	= answers->HighLowHeights;
	HLTimeHdl	= answers->HighLowTimes;
	HLFlagHdl	= answers->HighLow;
	ValHdl		= answers->height;
	TimeHdl		= answers->time;
	
	// Begin by declaring temp space on heap 
	// for a copy of the highs and lows
	
	NoOfHighsLows = answers->numHighLows;
	if(NoOfHighsLows<1){
		errorFlag=19;
		goto Error;
	}
	try
	{
		HArrayPtr = new double[NoOfHighsLows];
		TArrayPtr = new EXTFLAG[NoOfHighsLows + 2];
		
		// davew: Andy used NewPtrClear & NewHandleClear and it looks like he relied
		// on it, in at least some cases, to initialize his arrays
	}
	catch (...)
	{
		errorFlag=20;
		goto Error;
	}
	
	/*HArrayPtr = (double *)NewPtrClear(sizeof(double)*NoOfHighsLows);
	if(HArrayPtr==0){
		errorFlag=20;
		goto Error;
	}*/
	//HArrayPtr = (double *)calloc(NoOfHighsLows,sizeof(double));
	//HArrayPtr = new double[NoOfHighsLows];
	//if (HArrayPtr==NULL) {errorFlag=20; goto Error;}
	// Note the two extra high/low times used for interpolation
	// are stored at the end of the Time array
	
	/*TArrayPtr = (EXTFLAGPTR)NewPtrClear(sizeof(EXTFLAG)*(NoOfHighsLows+2) );
	if(TArrayPtr==0){
		errorFlag=21;
		goto Error;
	}*/
	//TArrayPtr = (EXTFLAG *)calloc(NoOfHighsLows+2,sizeof(EXTFLAG));
	//TArrayPtr = new EXTFLAG[NoOfHighsLows + 2];
	//if (TArrayPtr==NULL) {errorFlag=21; goto Error;}
	
	//OK now make copy of reference station data before
	// doing correction to it
	
	for (i=0; i<NoOfHighsLows; i++){
		HArrayPtr[i] = HLHeightHdl[i];
		TArrayPtr[i].val = HLTimeHdl[i].val;
		// davew: init other fields
		TArrayPtr[i].flag = 0;
		TArrayPtr[i].xtra = 0;
	}
	
	// OK grab extra two values in time array
	
	TArrayPtr[NoOfHighsLows].val = HLTimeHdl[NoOfHighsLows].val;
	TArrayPtr[NoOfHighsLows+1].val = HLTimeHdl[NoOfHighsLows+1].val;
	
	
	// davew: init other fields
	TArrayPtr[NoOfHighsLows].flag = 0;
	TArrayPtr[NoOfHighsLows].xtra = 0;
	TArrayPtr[NoOfHighsLows+1].flag = 0;
	TArrayPtr[NoOfHighsLows+1].xtra = 0;
	
	//OK do the high and low arrays first
	
	// Begin by applying the High and Low time offsets
	
	// Check to make sure we got offset data before using it
	
	if( htOffset->HighTime.dataAvailFlag!=1){
		errorFlag=22;
		goto Error;
	}
	highOffset = htOffset->HighTime.val;
	
	if( htOffset->LowTime.dataAvailFlag!=1){
		errorFlag=23;
		goto Error;
	}
	lowOffset = htOffset->LowTime.val;
	
	if( htOffset->HighHeight_Mult.dataAvailFlag!=1){
		HighMult=1.0;
	}
	else {
		HighMult = htOffset->HighHeight_Mult.val;
	}

	if( htOffset->HighHeight_Add.dataAvailFlag!=1){
		HighAdd=0.0;
	}
	else {
		HighAdd = htOffset->HighHeight_Add.val;
	}

	if( htOffset->LowHeight_Mult.dataAvailFlag!=1){
		LowMult=1.0;
	}
	else {
		LowMult = htOffset->LowHeight_Mult.val;
	}

	if( htOffset->LowHeight_Add.dataAvailFlag!=1){
		LowAdd=0.0;
	}
	else {
		LowAdd = htOffset->LowHeight_Add.val;
	}
	
	for(i=0; i<NoOfHighsLows + 2; i++){
	//for(i=0; i<NoOfHighsLows; i++){
		
		if( HLFlagHdl[i] == low){
			HLTimeHdl[i].val = HLTimeHdl[i].val + lowOffset;
			HLHeightHdl[i]= LowMult * HLHeightHdl[i] + LowAdd;
		}
		else {
			HLTimeHdl[i].val = HLTimeHdl[i].val + highOffset;
			HLHeightHdl[i]= HighMult * HLHeightHdl[i] + HighAdd;
		}
	}
	
	// now check times to make sure they are still in order
	// if not, kick out and flag error
	
	for(i=1;i<NoOfHighsLows; i++){
		if( HLTimeHdl[i].val < HLTimeHdl[i-1].val ){
			errorFlag=24;
			goto Error;
		}
	}
	
	// OK we are OK with high and low corrections
	// We now need to fix the full array

	// OK here is the strategy we go from point to point,
	// figure out between which high and low, the point falls.
	// Then interpolate the height offset and the time offset
	// of the point and apply it.
	// The points that are at the ends and don't fall
	// between a high and low will have to be handled as special cases
	

	// 02/04/2013 - Modified to use hermite interpolation instead of
    // linear one for multiple and add offest factors
    // It fixed a "twin peaks bug", but problem peaks may look a little bit "fat" now.
    // Also Bushy's concern was - hermite may show peaks coming up little early sometimes.
    // Left hermite for awhile - it looks like it doesn't spoil good stations,
    // but fixes most bad ones - will see and test more

	numOfPoints = answers->nPts;
		
	for (i=0; i<numOfPoints; i++){
		t = TimeHdl[i].val;
		if(i==0)index=-99;
		
		errorFlag = GetWeights(t,TArrayPtr,&w1,&w2,&index,NoOfHighsLows);
		flag = lowToHigh;
		if(index==-99){
			if( HLFlagHdl[0]==low){
				flag = highToLow;
			}
		}
		else {
			if( HLFlagHdl[index]==high){
				flag = highToLow;
			}		
		}
		
		if(errorFlag == 0){
			if(flag==lowToHigh){
				timeCorrection = lowOffset*w1 + highOffset*w2;
				//HeightCorrectionMult = LowMult*w1 + HighMult*w2;
				//HeightCorrectionAdd = LowAdd*w1 + HighAdd*w2;
			}
			else {
				timeCorrection = lowOffset*w2 + highOffset*w1;
				//HeightCorrectionMult = LowMult*w2 + HighMult*w1;
				//HeightCorrectionAdd = LowAdd*w2 + HighAdd*w1;
			}
			TimeHdl[i].val = TimeHdl[i].val + timeCorrection;
			//ValHdl[i] = (ValHdl[i]) * HeightCorrectionMult + HeightCorrectionAdd;
		}
		
	}

    double *pSlopesMult, *pSlopesAdd;

	pSlopesMult = new double[NoOfHighsLows];    // allocate memeory for slope array
	pSlopesAdd = new double[NoOfHighsLows];    // allocate memeory for slope array

	pSlopesMult[0]               =0.;			// start point and end point - different case
	pSlopesAdd[0]                =0.;			// start point and end point - different case
	pSlopesMult[NoOfHighsLows-1] =0.;			// use slope = 0 for start end end points
	pSlopesAdd[NoOfHighsLows-1]  =0.;			// use slope = 0 for start end end points

	// calculate slopes
    for(i=1; i<NoOfHighsLows-1; i++)	    
    {
        if(HLFlagHdl[i] == low)
        {
            pSlopesMult[i] = ((LowMult - HighMult) / (HLTimeHdl[i].val - HLTimeHdl[i-1].val) +
                              (HighMult - LowMult) / (HLTimeHdl[i+1].val - HLTimeHdl[i].val)) * .5;
            pSlopesAdd[i] = ((LowAdd - HighAdd) / (HLTimeHdl[i].val - HLTimeHdl[i-1].val) +
                             (HighAdd - LowAdd) / (HLTimeHdl[i+1].val - HLTimeHdl[i].val)) * .5;
        }
        else
        {
            pSlopesMult[i] = ((HighMult - LowMult) / (HLTimeHdl[i].val - HLTimeHdl[i-1].val) +
                              (LowMult - HighMult) / (HLTimeHdl[i+1].val - HLTimeHdl[i].val)) * .5;
            pSlopesAdd[i] = ((HighAdd - LowAdd) / (HLTimeHdl[i].val - HLTimeHdl[i-1].val) +
                             (LowAdd - HighAdd) / (HLTimeHdl[i+1].val - HLTimeHdl[i].val)) * .5;
        }
    }

    // for height values do hermite interpolation
    double dT0, dT1, dSMult0, dSMult1, dSAdd0, dSAdd1;

    // interpolate by hermite
	for(i=0, j=0; i<numOfPoints && j<NoOfHighsLows; i++)
	{
        t = TimeHdl[i].val;
        
        if(t > HLTimeHdl[j].val)
            j++;

        if(j == 0)
        {
            dT0 = HLTimeHdl[NoOfHighsLows].val;
            dT1 = HLTimeHdl[0].val;
            dSMult0 = 0.;
            dSMult1 = pSlopesMult[0];
            dSAdd0 = 0.;
            dSAdd1 = pSlopesAdd[0];
        }
        else
            if(j == NoOfHighsLows)
            {
                dT0 = HLTimeHdl[NoOfHighsLows-1].val;
                dT1 = HLTimeHdl[NoOfHighsLows+1].val;
                dSMult0 = pSlopesMult[NoOfHighsLows-1];
                dSMult1 = 0.;
                dSAdd0 = pSlopesAdd[NoOfHighsLows-1];
                dSAdd1 = 0.;
            }
            else
            {
                dT0 = HLTimeHdl[j-1].val;
                dT1 = HLTimeHdl[j].val;
            }

        if(HLFlagHdl[j] == low)
        {
            HeightCorrectionMult = Hermite(HighMult, dSMult0, dT0,
                                           LowMult, dSMult1, dT1, t);
            HeightCorrectionAdd = Hermite(HighAdd, dSAdd0, dT0,
                                           LowAdd, dSAdd1, dT1, t);
        }
        else
        {
            HeightCorrectionMult = Hermite(LowMult, dSMult0, dT0,
                                           HighMult, dSMult1, dT1, t);
            HeightCorrectionAdd = Hermite(LowAdd, dSAdd0, dT0,
                                           HighAdd, dSAdd1, dT1, t);
        }
		ValHdl[i] = ValHdl[i] * HeightCorrectionMult + HeightCorrectionAdd;
    }
	
Error:
	//if(HArrayPtr) DisposePtr((Ptr)HArrayPtr);
	//if(TArrayPtr) DisposePtr((Ptr)TArrayPtr);
	//if(HArrayPtr) {free(HArrayPtr); HArrayPtr = NULL;}
	//if(TArrayPtr) {free(TArrayPtr); TArrayPtr = NULL;}
	if (HArrayPtr) {delete [] HArrayPtr; HArrayPtr=0;}
	if (TArrayPtr) {delete [] TArrayPtr; TArrayPtr=0;}
	if (pSlopesMult) {delete [] pSlopesMult; pSlopesMult=0;}
	if (pSlopesAdd) {delete [] pSlopesAdd; pSlopesAdd=0;}

	return(errorFlag);
}


short FindExtraHL(double *AMPA,					// amplitude corrected for year
					double *epoch,				// epoch corrected for year
					short numOfConstituents,	// number of frequencies
					COMPHEIGHTS *answers,		// Height-time struc with answers
					double *zeroTime,			// first high-low
					double *extraTime,			// last high-low
					double datum)
// Function to compute extra high and low values
// We need one value before first high low and one after
// So we can interpolate

{
	long		i,nPts;
	short		nHL;
	double		h,lastValue;
	double		t,firstTime,lastTime,dt;
	double		*HHdl=0;
	EXTFLAG		*THdl=0;
	short		*tHdl=0;
	
	// OK begin by looking for first one before our data
	
	// Check first high-low to see if it is a high or low
	
	tHdl = answers->HighLow;
	THdl = answers->time;
	HHdl = answers->height;
	nPts = answers->nPts;
	nHL  = answers->numHighLows;
	
	firstTime = THdl[0].val;
	lastValue = HHdl[0];
	
	dt = 1.0/60.0;
	
	// go for max of 24 hours at 1 minute intervals
	// and look for max or min
	
	if( tHdl[0]==0){ 
		//look for high
		for(i=1;i<1441;i++){
			t = firstTime - (dt*i);
			h = RStatHeight(t,AMPA,epoch,numOfConstituents,datum);
			if(h<=lastValue)break;
			lastValue = h;
		}
	}
	else {
		//look for low
		for(i=1;i<1441;i++){
			t = firstTime - (dt*i);
			h = RStatHeight(t,AMPA,epoch,numOfConstituents,datum);
			if(h>=lastValue)break;
			lastValue = h;
		}
	}
	*zeroTime = t + dt/2.0;

	
	// Now find the time after the last time
	
	lastTime = THdl[nPts-1].val;
	lastValue = HHdl[nPts-1];
			
	// go for max of 24 hours at 1 minute intervals
	// and look for max or min
	
	if( tHdl[nHL-1]==0){ 
		//look for high
		for(i=1;i<1441;i++){
			t = lastTime + (dt*i);
			h = RStatHeight(t,AMPA,epoch,numOfConstituents,datum);
			if(h<=lastValue)break;
			lastValue = h;
		}
	}
	else {
		//look for low
		for(i=1;i<1441;i++){
			t = lastTime + (dt*i);
			h = RStatHeight(t,AMPA,epoch,numOfConstituents,datum);
			if(h>=lastValue)break;
			lastValue = h;
		}
	}
	*extraTime = t - dt/2.0;
		
		
	return 0;
}

/***************************************************************************************/

short FindHighLow(double startTime,				// start time in hrs from begining of year
					double endTime,				// end time in hrs from begining of year
					short HighLowFlag,			// flag = 0 for low, flag = 1 for high
					double *AMPA,				// corrected amplitude array
					double *epoch,				// corrected epoch array
					short numOfConstituents,	// number of frequencies
					double *theHeight,			// the high or low tide value
					double *theTime,			// the high or low tide time
					double datum)				// height datum
					
// routine to take start time, end time and solve for
// height at 1 minute intervals.  It picks out the low or high
// value and returns it and its time.
{
	double timeSpan,t=0.0,h=0.0;
	double t0,t1,h0,h1;
	long low=0,findFlag,numOfSteps,i;
	 
	// compute number of time steps ... if start time and end time
	// less than or equal to 3 minutes apart, interpolate and return
	
	timeSpan = endTime - startTime;
	if(timeSpan<= .05) {  // less than 3 minutes
		t = (startTime + endTime)/2.0;
		h = RStatHeight(t,AMPA,epoch,numOfConstituents,datum);
	    *theHeight = h;
		*theTime = t;
		return 0;
	}
	
	// Now if we get here, we gotta step through time and search
	// for high or low on 1 minute intervals
	// The plan is that if we are looking for a low, then
	// we march along and saving the last value and when
	// the slope changes to positive we grab and the last
	// value and cut out.
	// We do just the opposite for searching for high values.

	numOfSteps = (long)(timeSpan * 60.0 + 1);
	
	t0 = startTime;
	h0 = RStatHeight(t0,AMPA,epoch,numOfConstituents,datum);
	findFlag=0;
	
	for (i=1; i<numOfSteps; i++){
		t1 = t0 + 1.0/60.0;
		h1 = RStatHeight(t1,AMPA,epoch,numOfConstituents,datum);
		if(HighLowFlag==low){
			if(h1>=h0){
				t=t0;
				h=h0;
				findFlag=1;
				break;
			}
		}
		else {
			if(h1<h0){
				t=t0;
				h=h0;
				findFlag=1;
				break;
			}
		}
		h0=h1;
		t0=t1;
	}
	//if(findFlag==0)return 18; // error!
	if(findFlag==0)
	{
		*theHeight = 0;
		*theTime = (startTime + endTime)/2;
	}
	else
	{
		*theHeight = h;
		*theTime = t;
	}
	return 0;
}

/***************************************************************************************/

void FixHEnds(COMPHEIGHTS *answers,double beginTime, double endTime)
// Fix the ends of the plotted points so we don't overshoot
// What we will do is set all the points before the beginTime to
//  the beginTime and all the points after endTime to endTime
//
// correction Feb. 7. 1994 ... fixed ends for high-low array
//
{
	double *theHeightHdl;
	EXTFLAG *theTimeHdl,*theHLTimeHdl;
	long numberOfPoints,i,startCross,endCross,numberOfHighLows;
	double w1,w2,dt,dt1;
	double firstPointValue,lastPointValue;
	
	numberOfPoints	= answers->nPts;
	theTimeHdl		= answers->time;
	theHeightHdl	= answers->height;
	
	numberOfHighLows = answers->numHighLows;
	theHLTimeHdl    = answers->HighLowTimes;
	
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
			if(endTime<theTimeHdl[i].val){
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
	
	firstPointValue = w1 * theHeightHdl[startCross] + 
	                  w2 * theHeightHdl[startCross+1];
					  

	dt = theTimeHdl[endCross+1].val - theTimeHdl[endCross].val;
	dt1 = endTime - theTimeHdl[endCross].val;
	w2 = dt1/dt;
	w1 = 1.0 - w2;
	
	lastPointValue = w1 * theHeightHdl[endCross] + 
	                 w2 * theHeightHdl[endCross+1];
					  
	// Now reset two array values and times if the overshoot
	// If more than one value overshoots on either end, we
	// flag those points a non plotters
	
	for (i=0;i<(startCross+1);i++){
		if(i<startCross){
			theTimeHdl[i].flag = 1;
		}
		else {
			theTimeHdl[i].val = beginTime;
			theHeightHdl[i] = firstPointValue;
		}
	}

	for (i=(endCross+1);i<numberOfPoints;i++){
		//if(i>endCross){
		if(i>endCross+1){	// end time was never being set
			theTimeHdl[i].flag=1;
		}
		else {
			theTimeHdl[i].val = endTime;
			theHeightHdl[i] = lastPointValue;
		}
	}

	// OK now check all the high-low values and mark the ones that
	// are outside of our plotting range
	
	for (i=0;i<numberOfHighLows;i++){
		if( theHLTimeHdl[i].val < beginTime) {
			theHLTimeHdl[i].flag = 1;
		}
		else if( theHLTimeHdl[i].val > endTime) {
			theHLTimeHdl[i].flag = 1;
		}
	
	}
	return;
}

/***************************************************************************************/

short GetJulianDayHr(short day,		// day of month (1 - 31)
					short month,	// month (1- 12)
					short year,		// year (1904 - 2020)
					double *hour)	// returns hours from beginning of year
{
	double	DaysInMonth[13] = {0.0,31.0,28.0,31.0,30.0,31.0,30.0,31.0,31.0,30.0,31.0,30.0,31.0};
	double	theHour=0.0;
	short	i=0,err=0;
	
// Check for Leap year. This is out of Kernighan and Ritchie, page 41.
	
	if( (year % 4 == 0 && year % 100 != 0) || year % 400 == 0) DaysInMonth[2]=29.0;
	
// error check
	if((month<1) || (month>12) ){ err=3; goto Error; } // Bad month
	
	if( (day<1) || (day>DaysInMonth[month] ) ){ err=4; goto Error; } // Bad day
	
	if( (year<1904) || (year>2020) ){ err=5; goto Error; } // Bad year
	
	// Compute the hour now
	for(i=1;i<month; i++){
		theHour = theHour + ( DaysInMonth[i] * 24.0 );
	}
	theHour = theHour + (24.0 * (double)day);
	
	// correct to beginning of day instead of end
	
	theHour -= 24.0;
	
	(*hour) = theHour;


Error:	

	return(err);	
}

/***************************************************************************************/

short GetReferenceCurve(CONSTITUENT *constituent,	// Amplitude-phase array structs
					    short numOfConstituents,		// number of frequencies
					    double *XODE,					// Year correction to amplitude
					    double *VPU,					// Year correction to phase
						double beginHour,				// beginning hr from start of year
						double endHour,					// ending hr from start of year
						double timestep,				// time step in minutes
					    COMPHEIGHTS *answers)			// Height-time struc with answers
// Function to compute reference curve at caller specified delta t and 
// also return highs and lows correct to the minute

// OK here is the algorithm, we step along computing heights at delta t intervals
// As we go along we keep track of the slope between the last three points.
// If the slope changes, we have a local min or max.  We back up and compute
// heights to the minute between the last 3 points.  If the slope change was
// from positive to negative, we look for the highest value in our minute
// interval and that gets stored as our high tide.  If the slope change was
// from negative to positive, we look for the lowest value in our minute
// array and that gets stored as our low tide.

{
	double		theHeight=0.0,oldHeight=0.0;
	double		theTime=0.0,slope1=0.0,slope2=0.0,t0=0.0,t1=0.0,t2=0.0;
	double		MaxMinTime=0.0,MaxMinHeight=0.0;
	double		theHigh=0.0,theLow=0.0,theHighTime=0.0,theLowTime=0.0;
	double		zeroTime=0.0,lastTime=0.0;
	double		referenceHeight=0.0;
	double		*AMPAPtr=0,*epochPtr=0;
	double		*HHdl=0,*HLHHdl=0;
	EXTFLAG		*THdl=0,*HLTHdl=0;
	long		i=0, maxPeaks=0, NumOfSteps=0,HighLowCount=0,pcnt=0;
	short		*tHdl=0,errorFlag=0;
	Boolean		stop=false;
	OSErr 		err = 0;


	// OK begin by figuring out how many time steps we gotta do
	NumOfSteps = (long)( ((endHour-beginHour)*60.0) / timestep );
	
	// Add two incase timestep not even into hour
	// and because we start at begin time
	NumOfSteps += 2; 
	
	// OK we will allow for 4 highs and lows per day plus one
	
	maxPeaks = (long)( ((endHour - beginHour)/24.0) * 4.0 ) + 4;
	
	// Allocate memory
	try
	{
		AMPAPtr = new double[numOfConstituents];
		epochPtr = new double[numOfConstituents];
		HHdl = new double[NumOfSteps];					// This one is the height array
		THdl = new EXTFLAG[NumOfSteps];				// This one is the time array
		HLHHdl = new double[maxPeaks];					// This one is the array for high and low heights
		HLTHdl = new EXTFLAG[maxPeaks + 2];			// This one is the array for high and low times
													// Note we store two extra values for the time before the
													// the first and the time after the last
		tHdl = new short[maxPeaks];					// This one is the array for high and low types
		
		// davew: Andy used NewPtrClear & NewHandleClear and it looks like he relied
		// on it, in at least some cases, to initialize his arrays
		
		// davew: AMPA & epoch get set below, lets init the others to be safe
		for (i = 0; i < NumOfSteps; i++)
		{
			HHdl[i] = 0.0;
			THdl[i].val = 0.0;
			THdl[i].flag = 0;
			THdl[i].xtra = 0;
		}
		for (i = 0; i < maxPeaks; i++)
		{
			HLHHdl[i] = 0.0;
			HLTHdl[i].val = 0.0;
			HLTHdl[i].flag = 0;
			HLTHdl[i].xtra = 0;
			tHdl[i] = 0;
		}
		HLTHdl[maxPeaks].val = 0.0;
		HLTHdl[maxPeaks].flag = 0;
		HLTHdl[maxPeaks].xtra = 0;
		HLTHdl[maxPeaks+1].val = 0.0;
		HLTHdl[maxPeaks+1].flag = 0;
		HLTHdl[maxPeaks+1].xtra = 0;
	}
	catch (...)
	{
		errorFlag=13;
		goto Error;
	}
	
	// compute amplitude and phase arrays corrected for year
	
	/*AMPAPtr = (double *)NewPtrClear( numOfConstituents*sizeof(double) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=9;
		goto Error;
	}*/
	//AMPAPtr = (double *)calloc(numOfConstituents,sizeof(double));
	//if (AMPAPtr==NULL) {errorFlag=9; goto Error;}

	/*epochPtr = (double *)NewPtrClear( numOfConstituents*sizeof(double) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=10;
		goto Error;
	}*/
	//epochPtr = (double *)calloc(numOfConstituents,sizeof(double));
	//if (epochPtr==NULL) {errorFlag=10; goto Error;}
	
	for (i=0; i<numOfConstituents; i++){
		AMPAPtr[i]	= (double)constituent[i].H * (double)XODE[i];
		epochPtr[i]	= (double)VPU[i] - (double)constituent[i].kPrime;
	}
	
	// OK now time step and compute heights for reference
	
	// Begin by allocating space for solution arrays
	
	// This one is the height array
	/*HHdl = (double **)NewHandleClear( NumOfSteps*sizeof(double) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=11;
		goto Error;
	}*/
	
	// This one is the time array
	/*THdl = (EXTFLAG **)NewHandleClear( NumOfSteps*sizeof(EXTFLAG) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=12;
		goto Error;
	}*/
	
	// This one is the array for high and low heights
	/*HLHHdl = (double **)NewHandleClear( maxPeaks*sizeof(double) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=13;
		goto Error;
	}*/
	// This one is the array for high and low times
	// Note we store two extra values for the time before the
	// the first and the time after the last
	/*HLTHdl = (EXTFLAGHDL)NewHandleClear( (maxPeaks + 2)*sizeof(EXTFLAG) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=14;
		goto Error;
	}*/
	
	// This one is the array for high and low types
	/*tHdl = (short **)NewHandleClear( maxPeaks*sizeof(short) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=15;
		goto Error;
	}*/
	
	theTime = beginHour;
	oldHeight = 0;
	t0 = theTime;
	t1 = theTime;
	t2 = theTime;
	slope1 = 1.0;
	slope2 = 1.0;
	HighLowCount = 0;
	
	// Get reference height
	referenceHeight = GetDatum(constituent);
	stop=false;
	for (i= 0; i<NumOfSteps; i++){

		/////pcnt = ((i+1)*100)/NumOfSteps;
		/////stop=ShowProgress(pcnt,"Computing tidal heightsÉ");
		/////if(stop==true){
		/////	errorFlag=kInterrupt;
		/////	goto Error;
		/////}

		theHeight = RStatHeight(theTime,AMPAPtr,epochPtr,numOfConstituents,referenceHeight);
		THdl[i].val = theTime;
		HHdl[i] = theHeight;

		// track the slope

		if(i>0){
			if(i==1){
				slope1 = theHeight - oldHeight;
				slope2 = slope1;
				t1 = theTime;
				t2 = theTime;
			}
			else if(i==2){
				slope2 = theHeight - oldHeight;
				t2 = theTime;
			}
			else {
				slope1 = slope2;
				slope2 = theHeight - oldHeight;
				t0 = t1;
				t1 = t2;
				t2 = theTime;
			}
		
			// OK let's see the deal with highs and lows
			
			// By the time we get here, we have solved for
			// at least 2 points.  The first time is in t0,
			// the second time is in t1, and the third time,
			// if we have one is in t2
			
			// special case ... never happen
			if(slope1==0){
				// We we got zero slope, split the difference
				// in time and compute height at that time
				MaxMinTime = (t0 + t1)/2.0;
				// OK now compute height at MaxMinTime and check
				// to see if it is a high or a low
				MaxMinHeight = RStatHeight(MaxMinTime,AMPAPtr,epochPtr,numOfConstituents,referenceHeight);
				if(HighLowCount >= maxPeaks) { err = 41; goto Error;}
				if(MaxMinHeight<oldHeight){
					// low tide
					tHdl[HighLowCount]  = 0;
				}
				if(MaxMinHeight>oldHeight){
					// high tide
					tHdl[HighLowCount] = 1;
				}
				else {
					errorFlag=16;
					goto Error;
				}
				HLHHdl[HighLowCount] = MaxMinHeight;
				HLTHdl[HighLowCount].val = MaxMinTime;
				HighLowCount = HighLowCount + 1;
			}
			
			// special case ... never happen
			else if(slope2==0){
				// here we don't do anything, we just move on
				// the computation will take place when we get to
				// next step and slope1 become slope2
			}
			
			// OK for these normal cases, here is what we do
			// We recompute with a 1 minute time step and 
			// take the max or min value .. 
			
			// low tide case
			else if( (slope1<0) && (slope2>0) ){
				errorFlag = FindHighLow(t0,theTime,0,AMPAPtr,epochPtr,numOfConstituents,
							&theLow,&theLowTime,referenceHeight);
				if(errorFlag!=0){ goto Error; }
				if(HighLowCount >= maxPeaks) { err = 41; goto Error;}
				HLHHdl[HighLowCount] = theLow;
				HLTHdl[HighLowCount].val = theLowTime;
				tHdl[HighLowCount] = 0;
				HighLowCount = HighLowCount + 1;
			}
			// high tide case
			else if( (slope1>0) && (slope2<0) ) {
				errorFlag = FindHighLow(t0,theTime,1,AMPAPtr,epochPtr,numOfConstituents,
							&theHigh,&theHighTime,referenceHeight);
				if(errorFlag!=0){ goto Error; }
				if(HighLowCount >= maxPeaks) { err = 41; goto Error;}
				HLHHdl[HighLowCount] = theHigh;
				HLTHdl[HighLowCount].val = theHighTime;
				tHdl[HighLowCount] = 1;
				HighLowCount = HighLowCount + 1;
			}

		}
		
		oldHeight = theHeight;
	//	theTime = ( theTime + (timestep/60.0) );
		theTime = beginHour + ( ( ((double)(i+1)) * timestep ) / 60.0 );
	
	}

	/////SetCursor(*GetCursor(watchCursor));
#ifndef pyGNOME
	SetWatchCursor();
#endif
	// ******* OK ALL DONE
	// set the answers into the solution structure and return
	
	answers->nPts				= NumOfSteps;
	answers->time				= THdl;
	answers->height				= HHdl;
	answers->numHighLows		= HighLowCount;
	answers->HighLowHeights		= HLHHdl;
	answers->HighLowTimes		= HLTHdl;
	answers->HighLow			= tHdl;

	// Get extra high and low times for interpolation
	
	errorFlag=FindExtraHL(AMPAPtr,epochPtr,numOfConstituents,answers,&zeroTime,&lastTime,referenceHeight);
	if(errorFlag!=0){
		goto Error;
	}
	
	// OK store the values into the last two slots of the time array for highs and lows
	
	HLTHdl[HighLowCount].val=zeroTime;
	HLTHdl[HighLowCount+1].val=lastTime;
	
Error:
	/////DisposeProgressBox();
	if(errorFlag){
		//if(HHdl){ DisposeHandle((Handle)HHdl); }
		//if(THdl){ DisposeHandle((Handle)THdl); }
		//if(HLHHdl){ DisposeHandle((Handle)HLHHdl); }
		//if(HLTHdl){ DisposeHandle((Handle)HLTHdl); }
		//if(tHdl){ DisposeHandle((Handle)tHdl); }
		if(HHdl) {delete [] HHdl; HHdl = 0;}
		if(THdl) {delete [] THdl; THdl = 0;}
		if(HLHHdl) {delete [] HLHHdl;}
		if(HLTHdl) {delete [] HLTHdl; HLTHdl = 0;}
		if(tHdl) {delete [] tHdl; tHdl = 0;}
	}
	//if(AMPAPtr){ DisposePtr((Ptr)AMPAPtr); }
	//if(epochPtr){ DisposePtr((Ptr)epochPtr); }
	//if(AMPAPtr) {free(AMPAPtr); AMPAPtr = NULL;}
	//if(epochPtr) {free(epochPtr); epochPtr = NULL;}
	if(AMPAPtr) {delete [] AMPAPtr; AMPAPtr = 0;}
	if(epochPtr) {delete [] epochPtr; epochPtr = 0;}
	return errorFlag;
}


short GetWeights(double t,
				EXTFLAG *TArray,
				double *w1,
				double *w2,
				short *index,
				short NoOfHighsLows)
				
	// Function to get interpolation weight functions for t
	// Note that it is assumed that TArray has two extra
	// values for first and last values
{
	double t0=0.0,t1=0.0;
	short start,i;
		
		start = *index;
		
		if(start<-1){  // first time through
			if(t<TArray[0].val){
				if(t<TArray[NoOfHighsLows].val){
					// error
					return -1;
				}
				t0 = TArray[NoOfHighsLows].val;
				t1 = TArray[0].val;
			}
			else {
				start = 0;
			}
		}
		
		if(start>=0){
			if(t>TArray[NoOfHighsLows-1].val){
				t0 = TArray[NoOfHighsLows-1].val;
				t1 = TArray[NoOfHighsLows+1].val;
				*index = NoOfHighsLows-1;
			}
			else if(t<TArray[0].val){
				t0 = TArray[NoOfHighsLows].val;
				t1 = TArray[0].val;
				*index = -99;
			}
			
			else {
				for(i=(NoOfHighsLows-1);i>=start;i--){
					if( t>=TArray[i].val){
						t0=TArray[i].val;
						t1=TArray[i+1].val;
						*index = i;
						break;
					}
				}
			}
			
		}
		
		// OK we got t0 < t < t1, get weight functions
		
		*w1 = (t1-t)/(t1-t0);
		*w2 = 1.0 - (t1-t)/(t1-t0);
		
	return 0;
}


/***************************************************************************************/
double RStatHeight(double	theTime,	// time in hrs from begin of year
                     double	*AMPA,		// amplitude corrected for year
					 double	*epoch,		// epoch corrected for year
					 short	ncoeff,		// number of coefficients to use
					 double	datum)		// datum in feet
					                       
{
/*  Compute cosine stuff and return value */


/*  initialize 114 frequencies ... if we pickup more, like */
/*  for Anchorage, we add them here */

	double f[] = {	28.9841042,   // M(2)        1   12.421 hrs. Principal lunar
					30.0000000,   // S(2)        2   12.000 hrs  Principal solar
					28.4397295,   // N(2)        3   12.685 hrs  Larger lunar elliptic
					15.0410686,   // K(1)        4   23.934 hrs  Luni-solar diurnal
					57.9682084,   // M(4)        5    6.210 hrs  
					13.9430356,   // O(1)        6   25.819 hrs  Principal lunar diurnal
					86.9523127,   // M(6)        7    4.14  hrs
					44.0251729,   // MK(3)       8    8.11  hrs
					60.0000000,   // S(4)        9    6.00  hrs
					57.4238337,   // MN(4)      10    6.27  hrs
					28.5125831,   // Nu(2)      11   12.626 hrs  Larger lunar evectional
					90.0000000,   // S(6)       12    4.000 hrs
					27.9682084,   // Mu(2)      13   12.872 hrs  Variational
					27.8953548,   // 2N(2)      14   12.905 hrs  Lunar ellipic second order
					16.1391017,   // OO(1)      15   22.306 hrs  
					29.4556253,   // Lambda(2)  16   12.222 hrs  Smaller lunar elliptic
					15.0000000,   // S(1)       17   24.000 hrs
					14.4966939,   // M(1)       18   24.833 hrs
					15.5854433,   // J(1)       19   23.098 hrs
					0.5443747,    // Mm         20   27.55  day  Lunar monthly
					0.0821373,    // Ssa        21  182.6   day
					0.0410686,    // Sa         22  365.2   day  Annual
					1.0158958,    // Msf        23   14.7   day
					1.0980331,    // Mf         24   13.66  day  Lunar fortnightly
					13.4715145,   // Rho(1)     25   26.72  hrs
					13.3986609,   // Q(1)       26   26.868 hrs  Larger lunar elliptic
					29.9589333,   // T(2)       27   12.016 hrs  Larger solar elliptic
					30.0410667,   // R(2)       28   11.98  hrs
					12.8542862,   // 2Q(1)      29   28.01  hrs
					14.9589314,   // P(1)       30   24.066 hrs  Principal solar diurnal
					31.0158958,   // 2SM(2)     31   11.61  hrs
					43.4761563,   // M(3)       32    8.28  hrs
					29.5284789,   // L(2)       33   12.192 hrs  Smaller lunar elliptic
					42.9271398,   // 2MK(3)     34    8.37  hrs
					30.0821373,   // K(2)       35   11.97  hrs
					115.9364169,  // M(8)       36    3.105 hrs
					58.9841042,   // MS(4)      37    6.103 hrs
					12.9271398,   // Sigma(1)   38   27.848 hrs
					14.0251729,   // MP(1)      39   25.668
					14.5695476,   // Chi(1)     40   24.709
					15.9748272,   // 2PO(1)     41   22.535
					16.0569644,   // SO(1)      42   22.420
					30.5443747,   // MSN(2)     43   11.786
					27.4238337,   // MNS(2)     44   13.127
					28.9019669,   // OP(2)      45   12.456
					29.0662415,   // MKS(2)     46   12.386
					26.8794590,   // 2NS(2)     47   13.393
					26.9523126,   // MLN2S(2)   48   13.357
					27.4966873,   // 2ML2S(2)   49   13.092
					31.0980331,   // SKM(2)     50   11.576
					27.8039338,   // 2MS2K(2)   51   12.948
					28.5947204,   // MKL2S(2)   52   12.590
					29.1483788,   // M2(KS)(2)  53   12.351
					29.3734880,   // 2SN(MK)(2) 54   12.256
					30.7086493,   // 2KM(SN)(2) 55   11.723
					43.9430356,   // SO(3)      56    8.192
					45.0410686,   // SK(3)      57    7.993
					42.3827651,   // NO(3)      58    8.494
					59.0662415,   // MK(4)      59    6.095
					58.4397295,   // SN(4)      60    6.160
					57.4966873,   // 2MLS(4)    61    6.261
					56.9523127,   // 3MS(4)     62    6.321
					58.5125831,   // ML(4)      63    6.153
					56.8794590,   // N(4)       64    6.329
					59.5284789,   // SL(4)      65    6.048
					71.3668693,   // MNO(5)     66    5.044
					71.9112440,   // 2MO(5)     67    5.006
					73.0092770,   // 2MK(5)     68    4.931
					74.0251728,   // MSK(5)     69    4.863
					74.1073100,   // 3KM(5)     70    4.858
					72.9271398,   // 2MP(5)     71    4.936
					71.9933813,   // 3MP(5)     72    5.000
					72.4649023,   // MNK(5)     73    4.968
					88.9841042,   // 2SM(6)     74    4.046
					86.4079380,   // 2MN(6)     75    4.166
					87.4238337,   // MSN(6)     76    4.118
					87.9682084,   // 2MS(6)     77    4.092
					85.3920421,   // 2NMLS(6)   78    4.216
					85.8635632,   // 2NM(6)     79    4.193
					88.5125831,   // MSL(6)     80    4.067
					87.4966873,   // 2ML(6)     81    4.114
					89.0662415,   // MSK(6)     82    4.042
					85.9364168,   // 2MLNS(6)   83    4.189
					86.4807916,   // 3MLS(6)    84    4.163
					88.0503457,   // 2MK(6)     85    4.089
					100.3509735,  // 2MNO(7)    86   42.738 days
					100.9046318,  // 2NMK(7)    87   16.581 days
					101.9112440,  // 2MSO(7)    88    7.848 days
					103.0092771,  // MSKO(7)    89    4.984 days
					116.4079380,  // 2MSN(8)    90   21.941
					116.9523127,  // 3MS(8)     91   21.236
					117.9682084,  // 2(MS)(8)   92   20.035
					114.8476674,  // 2(MN)(8)   93   24.246
					115.3920422,  // 2MN(8)     94   23.389
					117.4966873,  // 2MSL(8)    95   20.575
					115.4648958,  // 4MLS(8)    96   23.279
					116.4807916,  // 3ML(8)     97   21.844
					117.0344500,  // 3MK(8)     98   21.134
					118.0503457,  // 2MSK(8)    99   19.944
					129.8887360,  // 2M2NK(9)  100   12.045
					130.4331108,  // 3MNK(9)   101   11.829
					130.9774855,  // 4MK(9)    102   11.621
					131.9933813,  // 3MSK(9)   103   11.252
					144.3761464,  // 4MN(10)   104    8.112
					144.9205211,  // M(10)     105    8.014
					145.3920422,  // 3MNS(10)  106    7.931
					145.9364169,  // 4MS(10)   107    7.837
					146.4807916,  // 3MSL(10)  108    7.745
					146.9523127,  // 3M2S(10)  109    7.667
					160.9774855,  // 4MSK(11)  110    5.904
					174.3761464,  // 4MNS(12)  111    4.840
					174.9205211,  // 5MS(12)   112    4.805
					175.4648958,  // 4MSL(12)  113    4.770
					175.9364169   // 4M2S(12)  114    4.741
	};

	double DegreesToRadians = 0.01745329252;		
	double height,argu,degrees;
	long i;					   

/* OK do the computational loop */

	height = 0.0;
	if(ncoeff<37) ncoeff=37;
	if(ncoeff>114)ncoeff = 114;
	
	for (i=0; i<ncoeff; i++){
	  // Don't do math if we don't have to.
	  if(AMPA[i]!=0){
	  	 degrees = f[i] * theTime + epoch[i];
		 degrees = fmod(degrees,360.0);
		 argu = degrees * DegreesToRadians;
	   //  argu =  (f[i] * theTime + epoch[i]) * DegreesToRadians;
         height = height + AMPA[i]*cos(argu);
	  }
	}
	height = height + datum;
	return height;
}




void ResetTime(COMPHEIGHTS *answers,double beginHour)
// function to reset time arrays to start from midnight of first day
{
	EXTFLAG *theTimeHdl;
	long numberOfPoints,i;
	//double theTime;
	
	numberOfPoints = answers->nPts;
	theTimeHdl = answers->time;

	// Reset Time array for plotting points
	
	for(i=0;i<numberOfPoints;i++){
	  	theTimeHdl[i].val = theTimeHdl[i].val - beginHour;
	}

	// OK now reset time array for highs and lows
	
	numberOfPoints = answers->numHighLows;
	theTimeHdl = answers->HighLowTimes;
	
	for(i=0;i<numberOfPoints;i++){
		//theTime = theTimeHdl[i].val;
	  	theTimeHdl[i].val = theTimeHdl[i].val - beginHour;
	}
	return;
}



