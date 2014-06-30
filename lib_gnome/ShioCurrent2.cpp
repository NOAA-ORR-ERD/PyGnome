/************************************************************/
/*    computational code for computing tide heights         */
/*    based on NOS tide data.                               */
/************************************************************/


#include "Basics.h"
#include "TypeDefs.h"
#include "Shio.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif


#include <iostream>
using namespace std;


/*---------------------------------------------*/
short GetFloodEbbSpans(double t,
					  EXTFLAG *TArrayPtr,
					  short *MaxMinFlagPtr,
					  double *CArrayPtr,
					  short numOfMaxMins,
					  double *previousTime,
					  double *nextTime,
					  double *previousValue,
					  double *nextValue,
					  double *previousMinusOneTime,
					  double *nextPlusOneTime,
					  double *previousMinusOneValue,
					  double *nextPlusOneValue,
					  short whatFlag)

					      
//
// send in the time and max min array and 
// return the times and values from max min array
// that spans the time time t
// if whatFlag = 0 then function returns max and mins that span t
// if whatFlag = 1 then function returns only maxs that span t
// 
{
	short i;
	short FloodKey = 1, EbbKey = 3;
	short onlyMaxs = 1;
	short errorFlag;
	short previousIndex,nextIndex;
	short previousMinusOneIndex,nextPlusOneIndex;
	double dh,dt;
	
	errorFlag = 0;
	previousIndex = -1;
	nextIndex = -1;
	previousMinusOneIndex = -1;
	nextPlusOneIndex = -1;

// Here the code will do Flood and Ebbs only
//
  if(whatFlag==onlyMaxs){
  
	for (i=0;i<(numOfMaxMins);i++){
		if( (MaxMinFlagPtr[i]==FloodKey) || (MaxMinFlagPtr[i]==EbbKey)){
			if(TArrayPtr[i].val<=t){
				previousIndex=i;
			}
			if(nextIndex==-1){
				if(TArrayPtr[i].val>t){
					nextIndex=i;
					break;
				}
			}
		}
	
	}
	
	if(previousIndex!=-1){
		*previousTime = TArrayPtr[previousIndex].val;
		*previousValue = CArrayPtr[previousIndex];
		
		if( (previousIndex-1)>=0){
			previousMinusOneIndex = previousIndex-1;
			// Check for case where we don't go through a min
			if( (MaxMinFlagPtr[previousMinusOneIndex]==FloodKey) ||
			    (MaxMinFlagPtr[previousMinusOneIndex]==EbbKey)) {
			
				*previousMinusOneTime = TArrayPtr[previousMinusOneIndex].val;
				*previousMinusOneValue = CArrayPtr[previousMinusOneIndex];
			}
			
			// Now check for case where we go through a min
			
			if( (previousIndex-2)>=0){
				previousMinusOneIndex = previousIndex - 2;
				if(previousMinusOneIndex>=0){
					*previousMinusOneTime = TArrayPtr[previousMinusOneIndex].val;
					*previousMinusOneValue = CArrayPtr[previousMinusOneIndex];
				}
			}
		}
		// OK we are at endpoint so we need to extrapolate
		if(previousMinusOneIndex == -1){
			// OK we use the next one and get the slope to extrapolate
			
			// if nextIndex is -1 then we are dead ... no maxs at all
			if(nextIndex==-1)return 39;
			
			dt = TArrayPtr[nextIndex].val - *previousTime;
			dh = CArrayPtr[previousIndex] - *previousValue;
			
			*previousMinusOneTime = *previousTime - dt;
			*previousMinusOneValue = *previousValue - dt;
		}
		
	}
	else{
		if( (MaxMinFlagPtr[numOfMaxMins]==FloodKey) ||
		(MaxMinFlagPtr[numOfMaxMins]==EbbKey)){
			*previousTime = TArrayPtr[numOfMaxMins].val;
			*previousValue = CArrayPtr[numOfMaxMins];
			
			dt = TArrayPtr[0].val-TArrayPtr[numOfMaxMins].val;
			dh = fabs( CArrayPtr[0]-CArrayPtr[numOfMaxMins]);
			
			*previousMinusOneTime = *previousTime - 2.0*dt;
			*previousMinusOneValue = *previousValue - 2.0*dh;
		}
		else{
			dt = TArrayPtr[1].val-TArrayPtr[0].val;
			dh = fabs( CArrayPtr[1]-CArrayPtr[0]);
			if( (MaxMinFlagPtr[0]==FloodKey) ||
				(MaxMinFlagPtr[0]==EbbKey) ) {
					dt = dt;
					dh = dh;
			}
			*previousTime = TArrayPtr[numOfMaxMins].val - dt;
			*previousMinusOneTime = TArrayPtr[0].val - dt * 3.0;
			if(MaxMinFlagPtr[0]==FloodKey){
				dh = - dh;
			}
			*previousValue = CArrayPtr[numOfMaxMins] + dh;
			*previousMinusOneValue = CArrayPtr[0];
		}
	}
	
	if(nextIndex!=-1){
		*nextTime = TArrayPtr[nextIndex].val;
		*nextValue = CArrayPtr[nextIndex];
		
		if( (nextIndex+1) < numOfMaxMins ){
			nextPlusOneIndex = nextIndex+1;
			*nextPlusOneTime = TArrayPtr[nextPlusOneIndex].val;
			*nextPlusOneValue = CArrayPtr[nextPlusOneIndex];
		}
		
		else {
			nextPlusOneIndex = numOfMaxMins+1;
			*nextPlusOneTime = TArrayPtr[nextPlusOneIndex].val;
			*nextPlusOneValue = CArrayPtr[nextPlusOneIndex];
		}
		
	}
	else {
		*nextTime = TArrayPtr[numOfMaxMins+1].val;
		*nextValue = CArrayPtr[numOfMaxMins+1];

		dt = TArrayPtr[numOfMaxMins-1].val-TArrayPtr[numOfMaxMins+1].val;
		dh =  CArrayPtr[numOfMaxMins-1]-CArrayPtr[numOfMaxMins+1];
			
		*nextPlusOneTime = *nextTime + 2.0*dt;
		*nextPlusOneValue = *nextValue - 2.0*dh;
			
	}
	
  }
  // *** end of code to do only max floods and max ebbs
  
  // Now do all max and min values
  
  else {
	for (i=0;i<(numOfMaxMins);i++){
			if(TArrayPtr[i].val<=t){
				previousIndex=i;
			}
		if(nextIndex==-1){
				if(TArrayPtr[i].val>t){
					nextIndex=i;
					break;
				}
		}
	
	}
	if(previousIndex!=-1){
		*previousTime = TArrayPtr[previousIndex].val;
		*previousValue = CArrayPtr[previousIndex];
		
		if( (previousIndex-1)>=0) {
			previousMinusOneIndex = previousIndex-1;
			*previousMinusOneTime = TArrayPtr[previousMinusOneIndex].val;
			*previousMinusOneValue = CArrayPtr[previousMinusOneIndex];
		}
		else {
			previousMinusOneIndex = numOfMaxMins;
			*previousMinusOneTime = TArrayPtr[previousMinusOneIndex].val;
			*previousMinusOneValue = CArrayPtr[previousMinusOneIndex];
		}
	}
	else{
		*previousTime = TArrayPtr[numOfMaxMins].val;
		*previousValue = CArrayPtr[numOfMaxMins];
		
		dt = TArrayPtr[0].val - TArrayPtr[numOfMaxMins].val;
		dh = CArrayPtr[0] - CArrayPtr[numOfMaxMins];
		
		*previousMinusOneTime = *previousTime - dt;
		*previousMinusOneValue = *previousValue - dh;
	}

	if(nextIndex!=-1){
		*nextTime = TArrayPtr[nextIndex].val;
		*nextValue = CArrayPtr[nextIndex];
		
		if( (nextIndex+1)<numOfMaxMins) {
			nextPlusOneIndex = nextIndex+1;
			*nextPlusOneTime = TArrayPtr[nextPlusOneIndex].val;
			*nextPlusOneValue = CArrayPtr[nextPlusOneIndex];
		}
		else {
			nextPlusOneIndex = numOfMaxMins+1;
			*nextPlusOneTime = TArrayPtr[nextPlusOneIndex].val;
			*nextPlusOneValue = CArrayPtr[nextPlusOneIndex];
		}
	}
	else {
		*nextTime = TArrayPtr[numOfMaxMins+1].val;
		*nextValue = CArrayPtr[numOfMaxMins+1];
				
		dt = TArrayPtr[numOfMaxMins+1].val - TArrayPtr[numOfMaxMins-1].val;
		dh = CArrayPtr[numOfMaxMins+1] - CArrayPtr[numOfMaxMins-1];
		
		*nextPlusOneTime = *nextTime + dt;
		*nextPlusOneValue = *nextValue + dh;
	}
	
  }
  
  if(t<*previousTime)errorFlag = 38;
  if(t>*nextTime)errorFlag = 38;
  
  return errorFlag;

}

/*---------------------------------------------*/
void GetMajorMinorAxis(CONSTITUENT *constituent,
                                    short *majorAxis,
									short *minorAxis)

// function to estimate major and minor axis from flood, ebb direction data

{	

	short flood,ebb;
	
	//flood = constituent->DatumControls.FDir;
	//ebb = constituent->DatumControls.EDir;
	
	flood = constituent[0].DatumControls.FDir;			// more explicit
	ebb = constituent[0].DatumControls.EDir;			// more explicit

	// we will use flood direction as major axis
	// then rotate 90 degrees, counterclockwise for minor axis
	
	*majorAxis = flood;
	*minorAxis = flood+90;
	
	if(*minorAxis>=360) *minorAxis = *minorAxis-360;
	
	// note the direction is consistent with the data
	// to the north is 0 degrees and to the east is 90 degrees
	
	return;
}

short GetRefCurrent(CONSTITUENT *constituent,	// Amplitude-phase array structs
					double* XODE, double* VPU, 		// Year correction
					short numOfConstituents,		// number of frequencies
					double beginHour,				// beginning hr from start of year
					double endHour,					// ending hr from start of year
					double timestep,				// time step in minutes
					COMPCURRENTS *answers)			// Height-time struc with answers
// Function to compute reference curve at caller specified delta t and 
// also return highs and lows correct to the minute

// OK here is the algorythm, we step along computing heights at delta t intervals
// As we go along we keep track of the slope between the last three points.
// If the slope changes, we have a local min or max.  We back up and compute
// heights to the minute between the last 3 points.  If the slope change was
// from positive to negative, we look for the highest value in our minute
// interval and that gets stored as our high tide.  If the slope change was
// from negative to positive, we look for the lowest value in our minute
// array and that gets stored as our low tide.

{
	double	theCurrent=0.0,oldCurrent=0.0,uVelocity=0.0,vVelocity=0.0;
	double	theTime=0.0,slope1=0.0,slope2=0.0,t0=0.0,t1=0.0,t2=0.0;
	double	MaxMinTime=0.0,MaxMinCurrent=0.0;
	double	theFlood=0.0,theEbb=0.0,theMinBFTime=0.0,theMinBETime=0.0,theMin=0.0;
	double	theEbbTime=0.0,theFloodTime=0.0;
	double	zeroTime=0.0,lastTime=0.0;
	double	refCur=0.0,vMajor=0.0,vMinor=0.0;
	double	twoCurrentsAgo=0.0,lastCurrent=0.0;
	double	zeroValue=0.0,lastValue=0.0;
	double	*AMPAPtr=nil,*epochPtr=nil;
	double	*CHdl=0,*MaxMinHdl=0,*uVelHdl=0,*vVelHdl=0;
	EXTFLAG	*THdl=0,*MaxMinTHdl=0;
	long	maxPeaks=0,NumOfSteps=0,pcnt=0;
	short	*tHdl=0,findFlag=0,direcKey=0,zeroFlag=0,lastFlag=0;
	long	i=0,j=0,MaxMinCount=0,errorFlag=0,actualNoOfConst=0;
	short	CFlag=0,rotFlag=0,L2Flag=0;
	//Boolean	stop=false;
	
	long maxMinHdlNumElements; //JLM


	// OK begin by figuring out how many time steps we gotta do
	NumOfSteps = (long)( ((endHour-beginHour)*60.0) / timestep );
	
	// Add two incase timestep not even into hour
	// and because we start at begin time
	NumOfSteps += 2; 
	
	// compute amplitude and phase arrays corrected for year
	
	try
	{
		AMPAPtr = new double[numOfConstituents];
		epochPtr = new double[numOfConstituents];
		
		// davew: Andy used NewPtrClear & NewHandleClear and it looks like he relied
		// on it, in at least some cases, to initialize his arrays
		// in this case we don't have to init, they get set below
	}
	catch (...)
	{
		errorFlag=9;
		goto Error;
	}

	/*AMPAPtr = (double *)NewPtrClear( numOfConstituents*sizeof(double) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=9;
		goto Error;
	}
	AMPAPtr = (double *)calloc(numOfConstituents,sizeof(double));
	if (AMPAPtr==NULL) {errorFlag=9; goto Error;}
	
	epochPtr = (double *)NewPtrClear( numOfConstituents*sizeof(double) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=10;
		goto Error;
	}
	epochPtr = (double *)calloc(numOfConstituents,sizeof(double));
	if (epochPtr==NULL) {errorFlag=10; goto Error;}*/

	errorFlag=GetControlFlags(constituent,&L2Flag,&CFlag,&rotFlag);     
	
	actualNoOfConst = numOfConstituents;
	if( (rotFlag==1)||(rotFlag==2)){
		actualNoOfConst = numOfConstituents/2;
	}

	// OK now check if we gotta do L2 correction
	// as far as I know, only Carquinez Strait needs
	// the correction
 
	if(L2Flag==1){
		if(constituent[31].H !=0.0){
			XODE[32] = XODE[0] * XODE[2];
			VPU[32] = 2.0 * VPU[0] - VPU[2];
			//((*YHdl)[32]).XODE = ((*YHdl)[0]).XODE * ((*YHdl)[2]).XODE;
			//((*YHdl)[32]).VPU = 2.0 * ((*YHdl)[0]).VPU - ((*YHdl)[2]).VPU;
		//	((*YHdl)[31]).XODE = 0.0;
		}
	}
 


	for (i=0; i<numOfConstituents; i++){
		if(i<(actualNoOfConst) ){
			j=i;
		}
		else{
			j=i-actualNoOfConst;
		}
		
/*#ifdef pyGNOME
		AMPAPtr[i] = constituent[i].H ;// AH 03/21/2012 * XODE[j];
		epochPtr[i] = - constituent[i].kPrime;	// TODO: Until we get YearData
#else*/
	 	AMPAPtr[i] = constituent[i].H * XODE[j];
		epochPtr[i] = VPU[j] - constituent[i].kPrime;
//#endif
	}
	
	// OK now time step and compute currents for reference
	
	// Begin by allocating space for solution arrays
	
	// This one is the current array
	/*CHdl = (double **)NewHandleClear( NumOfSteps*sizeof(double) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=11;
		goto Error;
	}
	
	// This one is the time array
	THdl = (EXTFLAG **)NewHandleClear( NumOfSteps*sizeof(EXTFLAG) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=12;
		goto Error;
	}*/
		
	// OK we will allow for 4 highs and lows per day plus one extra set
	// JLM,  allow for 6 highs and lows per day plus one extra set
	
	maxPeaks = ( (endHour - beginHour)/24.0 )*12.0  + 12;
	
	// This one is the array for max and min current values
	maxMinHdlNumElements = maxPeaks+2;
	maxPeaks += 2;
	/*MaxMinHdl = (double **)NewHandleClear( maxMinHdlNumElements*sizeof(double) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=13;
		goto Error;
	}

	// This one is the array for max and min current times
	// Note we store two extra values for the time before the
	// the first and the time after the last
	MaxMinTHdl = (EXTFLAG **)NewHandleClear( maxMinHdlNumElements*sizeof(EXTFLAG) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=14;
		goto Error;
	}
	
	// This one is the array for max min labels
	tHdl = (short **)NewHandleClear( maxMinHdlNumElements*sizeof(short) ); errorFlag=(short)MemError();
	if( errorFlag != 0 ){
		errorFlag=15;
		goto Error;
	}*/


	try
	{
		// This one is the current array - MKH ERROR
		CHdl = new double[NumOfSteps];
		
		// This one is the time array
		THdl = new EXTFLAG[NumOfSteps];
		
		// This one is the array for max and min current values
		MaxMinHdl = new double[maxMinHdlNumElements];
		
		// This one is the array for max and min current times
		// Note we store two extra values for the time before the
		// the first and the time after the last
		MaxMinTHdl = new EXTFLAG[maxMinHdlNumElements];

		// This one is the array for max min labels
		tHdl = new short[maxMinHdlNumElements];

		// allocate space for u and v velocity components if
		// tidal currents are rotary
		
		// davew 2/9/4: we use these even if the station is not rotary,
		// so I'm not sure why there weren't always just allocated
		// I'm going to allocate them all the time
		
		//if( (rotFlag==1) || (rotFlag==2) )
		//{
			uVelHdl = new double[NumOfSteps];
			vVelHdl = new double[NumOfSteps];
			
		//	for (i = 0; i < NumOfSteps; i++)
		//	{
		//		uVelHdl[i] = 0.0;
		//		vVelHdl[i] = 0.0;
		//	}
		//}
		
		// davew: Andy used NewPtrClear & NewHandleClear and it looks like he relied
		// on it, in at least some cases, to initialize his arrays
		
		// in this case it looks complicated below, so I'm gonna init the arrays
		
		for (i = 0; i < NumOfSteps; i++)
		{
			CHdl[i] = 0.0;
			
			THdl[i].val = 0.0;
			THdl[i].flag = 0;
			THdl[i].xtra = 0;
			
			uVelHdl[i] = 0.0;
			vVelHdl[i] = 0.0;
		}
		
		for (i = 0; i < maxPeaks; i++)
		{
			MaxMinHdl[i] = 0.0;
			tHdl[i] = 0;
			
			MaxMinTHdl[i].val = 0.0;
			MaxMinTHdl[i].flag = 0;
			MaxMinTHdl[i].xtra = 0;
		}
	}
	catch (...)
	{
		errorFlag=11;
		goto Error;
	}


	theTime = beginHour;
	oldCurrent = 0;
	t0 = theTime;
	t1 = theTime;
	t2 = theTime;
	slope1 = 1.0;
	slope2 = 1.0;
	MaxMinCount = 0;
	theCurrent = 0.0;
	
	// OK here is the plan
	// We step through and compute the currents
	// As we go along, we look for sign changes in the value
	// which would indicate a min value and we look for sign
	// changes in the slope which would indicate a max value

	// Get reference cur
	refCur = GetDatum(constituent);
		
	findFlag = 0;
	//stop=false;
	for (i= 0; i<NumOfSteps; i++){

		/////pcnt = ((i+1)*100)/NumOfSteps;
		/////stop=ShowProgress(pcnt,"Computing tidal currents…");
		/////if(stop==true){
		/////	errorFlag=kInterrupt;
		/////	goto Error;
		/////}

 
		// OK we gotta check RotFlag to see if we gotta compute rotated
		// currents
		if( (rotFlag==1) || (rotFlag==2) ) {
			if(i==0){
				twoCurrentsAgo=999.0;
				lastCurrent=999.0;
			}
			else {
				twoCurrentsAgo = lastCurrent;
				lastCurrent = theCurrent;
			}
			
 			theCurrent = RStatCurrentRot(theTime,AMPAPtr,epochPtr,
 		            	 numOfConstituents,constituent,twoCurrentsAgo,lastCurrent,
 						 &uVelocity,&vVelocity,&vMajor,&vMinor,&direcKey);
		
			if(direcKey==33){
				errorFlag = direcKey;
				goto Error;
			}
			
 			if(direcKey==0) THdl[i].flag = 1;  // don't plot the sucker
		}
		else {
 			theCurrent = RStatCurrent(theTime,AMPAPtr,epochPtr,
 		            	 refCur,numOfConstituents,CFlag);
		}
  		THdl[i].val = theTime;
 		CHdl[i] = theCurrent;

 		if( (rotFlag==1) || (rotFlag==2) ){
 			uVelHdl[i]=uVelocity;
 			vVelHdl[i]=vVelocity;
 		}
		
		
		// track the slope
Retry:		
		
		if(i>0){

			if(i==1){
				slope1 = theCurrent - oldCurrent;
				slope2 = slope1;
				t1 = theTime;
				t2 = theTime;
			}
			else if(i==2){
				slope2 = theCurrent - oldCurrent;
				t2 = theTime;
			}
			else {
				slope1 = slope2;
				slope2 = theCurrent - oldCurrent;
				t0 = t1;
				t1 = t2;
				t2 = theTime;
			}
		
			// OK let's see the deal with max currents
			
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
				MaxMinCurrent = RStatCurrentRot(theTime,AMPAPtr,epochPtr,
		            	 		numOfConstituents,constituent,999.0,999.0,
								&uVelocity,&vVelocity,&vMajor,&vMinor,&direcKey);
								
				if(direcKey==33){
					errorFlag = direcKey;
					goto Error;
				}
				
				if(MaxMinCount >= maxMinHdlNumElements) { errorFlag = 41;goto Error;}
				if(MaxMinCurrent<oldCurrent){
					// max ebb
					tHdl[MaxMinCount]  = 3;
				}
				if(MaxMinCurrent>oldCurrent){
					// max flood
					tHdl[MaxMinCount] = 1;
				}
				else {
					errorFlag=16;
					goto Error;
				}
				MaxMinHdl[MaxMinCount] = MaxMinCurrent;
				MaxMinTHdl[MaxMinCount].val = MaxMinTime;
				//MaxMinCount = MaxMinCount + 1;
				if(MaxMinCount + 1 < maxPeaks) 
                    MaxMinCount++;		// othewise, keep writing over last value
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
			
			// max ebb case
			else if( (slope1<0) && (slope2>0) ){
				if( (rotFlag==1) || (rotFlag==2) ){
					errorFlag = FindFloodEbbRot(t0,theTime,3,AMPAPtr,epochPtr,numOfConstituents,
							&theEbb,&theEbbTime,constituent,twoCurrentsAgo,lastCurrent);
				}
				else {
					errorFlag = FindFloodEbb(t0,theTime,3,AMPAPtr,epochPtr,numOfConstituents,
							&theEbb,&theEbbTime,refCur,CFlag);
				}
				if(errorFlag!=0){
					if( (i==1) && (findFlag==0) ){
						CHdl[0] = -CHdl[0];
						lastCurrent = -lastCurrent;
						oldCurrent = -oldCurrent;
						findFlag = 1;
						errorFlag = 0;
						goto Retry;
					}
					goto Error;
				}
				if(MaxMinCount >= maxMinHdlNumElements) { errorFlag = 41;goto Error;}
				MaxMinHdl[MaxMinCount] = theEbb;
				MaxMinTHdl[MaxMinCount].val = theEbbTime;
                // 02/25/2014 G.K.: added multiple ebbs/floods case
                if(MaxMinCount > 0)
                    if(tHdl[MaxMinCount - 1] == MinBeforeEbb)
				        tHdl[MaxMinCount] = MaxEbb;
                    else
                        tHdl[MaxMinCount] = tHdl[MaxMinCount - 1];
                else
				    tHdl[MaxMinCount] = MaxEbb;

				if(MaxMinCount + 1 < maxPeaks) 
                    MaxMinCount++;		// othewise, keep writing over last value
				//tHdl[MaxMinCount] = 3;
				//MaxMinCount = MaxMinCount + 1;
			}
			// max flood case
			else if( (slope1>0) && (slope2<0) ) {
				if( (rotFlag==1) || (rotFlag==2) ){
					errorFlag = FindFloodEbbRot(t0,theTime,1,AMPAPtr,epochPtr,numOfConstituents,
							&theFlood,&theFloodTime,constituent,twoCurrentsAgo,lastCurrent);
				}
				else {
					errorFlag = FindFloodEbb(t0,theTime,1,AMPAPtr,epochPtr,numOfConstituents,
							&theFlood,&theFloodTime,refCur,CFlag);
				}
				if(errorFlag!=0){
					if( (i==1) && (findFlag==0) ){
						CHdl[0] = -CHdl[0];
						lastCurrent = -lastCurrent;
						oldCurrent = -oldCurrent;
						findFlag = 1;
						errorFlag = 0;
						goto Retry;
					}
					goto Error;
				}
				if(MaxMinCount >= maxMinHdlNumElements) { errorFlag = 41;goto Error;}
				MaxMinHdl[MaxMinCount] = theFlood;
				MaxMinTHdl[MaxMinCount].val = theFloodTime;

                // 02/25/2014 G.K.: added multiple ebbs/floods case
                if(MaxMinCount > 0)
                    if(tHdl[MaxMinCount - 1] == MinBeforeFlood)
				        tHdl[MaxMinCount] = MaxFlood;
                    else
                        tHdl[MaxMinCount] = tHdl[MaxMinCount - 1];
                else
				    tHdl[MaxMinCount] = MaxFlood;

                if(MaxMinCount + 1 < maxPeaks) 
                    MaxMinCount++;		// othewise, keep writing over last value
				//tHdl[MaxMinCount] = 1;
				//MaxMinCount = MaxMinCount + 1;
			}
			
			// OK now look for min values by checking for zero crossings
	
			// min before flood case
			if( (theCurrent>0) && (oldCurrent<0) ){
				if( (rotFlag==1) || (rotFlag==2) ){
					errorFlag = FindFloodEbbRot(t0,theTime,0,AMPAPtr,epochPtr,numOfConstituents,
							&theMin,&theMinBFTime,constituent,twoCurrentsAgo,lastCurrent);
				}
				else {
					errorFlag = FindFloodEbb(t0,theTime,0,AMPAPtr,epochPtr,numOfConstituents,
							&theMin,&theMinBFTime,refCur,CFlag);
				}
				if(errorFlag!=0){
					if( (i==1) && (findFlag==0) ){
						CHdl[0] = -CHdl[0];
						lastCurrent = -lastCurrent;
						oldCurrent = -oldCurrent;
						findFlag = 1;
						errorFlag = 0;
						goto Retry;
					}
					goto Error;
				}
				if(MaxMinCount >= maxMinHdlNumElements) { errorFlag = 41;goto Error;}
				MaxMinHdl[MaxMinCount] = 0.0;
				if( (rotFlag==1) || (rotFlag==2) ){
					MaxMinHdl[MaxMinCount] = theMin;
				}
				MaxMinTHdl[MaxMinCount].val = theMinBFTime;
				tHdl[MaxMinCount] = 0;
				if(MaxMinCount + 1 < maxPeaks) 
                    MaxMinCount++;		// otherwise, keep writing over last value
				//MaxMinCount = MaxMinCount + 1;
			}
	
			// min before ebb case
			else if( (theCurrent<0) && (oldCurrent>0) ){
				if( (rotFlag==1) || (rotFlag==2) ){
					errorFlag = FindFloodEbbRot(t0,theTime,2,AMPAPtr,epochPtr,numOfConstituents,
							&theMin,&theMinBETime,constituent,twoCurrentsAgo,lastCurrent);
				}
				else {
					errorFlag = FindFloodEbb(t0,theTime,2,AMPAPtr,epochPtr,numOfConstituents,
							&theMin,&theMinBETime,refCur,CFlag);
				}
				if(errorFlag!=0){
					if( (i==1) && (findFlag==0) ){
						CHdl[0] = -CHdl[0];
						lastCurrent = -lastCurrent;
						oldCurrent = -oldCurrent;
						findFlag = 1;
						errorFlag = 0;
						goto Retry;
					}
					goto Error;
				}
				if(MaxMinCount >= maxMinHdlNumElements) { errorFlag = 41;goto Error;}
				MaxMinHdl[MaxMinCount] = 0.0;
				if( (rotFlag==1) || (rotFlag==2) ){
					MaxMinHdl[MaxMinCount] = theMin;
				}
				MaxMinTHdl[MaxMinCount].val = theMinBETime;
				tHdl[MaxMinCount] = 2;
				if(MaxMinCount + 1 < maxPeaks) 
                    MaxMinCount++;		// otherwise, keep writing over last value
				//MaxMinCount = MaxMinCount + 1;
			}
	
			else if(oldCurrent==0){
				if(MaxMinCount >= maxMinHdlNumElements) { errorFlag = 41;goto Error;}
				MaxMinHdl[MaxMinCount] = 0.0;
				MaxMinTHdl[MaxMinCount].val = t1;
				tHdl[MaxMinCount] = 0;
				if(theCurrent<0)tHdl[MaxMinCount] = 1;
				if(MaxMinCount + 1 < maxPeaks) 
                    MaxMinCount++;		// otherwise, keep writing over last value
				//MaxMinCount = MaxMinCount + 1;
			}
		}
		oldCurrent = theCurrent;
//		theTime = ( theTime + (timestep/60.0) );
		theTime = beginHour + ( ( ((double)(i+1)) * timestep ) / 60.0 );
	}
#ifndef pyGNOME
	SetWatchCursor();
#endif
	// we are not going to check for a max ebb or flood that hits the 
	// zero axis exactly ... never happen right?
	
	
	// OK before we leave, let's check the plotflag array if the
	// station is rotary, and major-minor axis.  On these stations
	// we aren't given the major - minor axis direction so we don't always
	// pick up the transition between flood and ebb
	

	if( (rotFlag==2) || (rotFlag==1) ){  // rotary with major - minor axis
	
		// OK run through the flag array and make sure that
		// the no plot flag come in pairs.  If not we will
		// extend the no plot flag to include the transition from
		// ebb to flood or flood to ebb.
		
		   errorFlag = FixMajMinFlags(THdl,CHdl,NumOfSteps,MaxMinHdl,MaxMinTHdl,MaxMinCount);
	}
	
done:
	// ******* OK ALL DONE
	// set the answers into the solution structure and return
	
	answers->nPts				= NumOfSteps;
	answers->time				= THdl;
	answers->speed			= CHdl;
	answers->u				= uVelHdl;
	answers->v				= vVelHdl;

	// davew 4/1/3: there appers to be a crashing bug because the following arrays are not initializes 
	// and then get deleted in OffsetUV()
	answers->uMinor = 0;
	answers->vMajor = 0;
	answers->speedKey = 0;		// go ahead and init this field

	answers->numEbbFloods		= MaxMinCount;
	answers->EbbFloodSpeeds	= MaxMinHdl;
	answers->EbbFloodTimes	= MaxMinTHdl;
	answers->EbbFlood			= tHdl;

	// Get extra high and low times for interpolation
	
	if( (rotFlag==1) || (rotFlag==2) ) {
		errorFlag = FindExtraFERot(AMPAPtr,epochPtr,numOfConstituents,
					answers,&zeroTime,&zeroValue,&zeroFlag,
					&lastTime,&lastValue,&lastFlag,
					constituent);  
	}
	else {
		errorFlag = FindExtraFE(AMPAPtr,epochPtr,numOfConstituents,answers,
	 							&zeroTime,&zeroValue,&zeroFlag,
								&lastTime,&lastValue,&lastFlag,
								refCur,CFlag);
	}
	
	if(errorFlag!=0){
		goto Error;
	}
	
	
	// OK store the values into the last two slots of the time array for highs and lows
	
	if(MaxMinCount >= maxMinHdlNumElements-1) { errorFlag = 41;goto Error;}
	MaxMinTHdl[MaxMinCount].val=zeroTime;
	MaxMinHdl[MaxMinCount]=zeroValue;
	tHdl[MaxMinCount]=zeroFlag;
	
	if(MaxMinCount + 1 < maxPeaks) 
        MaxMinCount++;		// othewise, keep writing over last value
	
	MaxMinTHdl[MaxMinCount+1].val=lastTime;
	MaxMinHdl[MaxMinCount+1]=lastValue;
	tHdl[MaxMinCount+1]=lastFlag;

Error:
	/////DisposeProgressBox();
	if(errorFlag){
		if(CHdl) delete [] CHdl;
		if(THdl) delete [] THdl;
		if(MaxMinHdl) delete [] MaxMinHdl;
		if(MaxMinTHdl) delete [] MaxMinTHdl;
		if(uVelHdl) delete [] uVelHdl;
		if(vVelHdl) delete [] vVelHdl;
		if(tHdl) delete [] tHdl;
		
		answers->nPts			= 0;
		answers->time			= 0;
		answers->speed			= 0;
		answers->u				= 0;
		answers->v				= 0;
		answers->uMinor = 0;
		answers->vMajor = 0;
		answers->speedKey = 0;
		answers->numEbbFloods	= 0;
		answers->EbbFloodSpeeds	= 0;
		answers->EbbFloodTimes	= 0;
		answers->EbbFlood		= 0;

		/*if(CHdl)		DisposeHandle((Handle)CHdl);
		if(THdl)		DisposeHandle((Handle)THdl);
		if(MaxMinHdl)	DisposeHandle((Handle)MaxMinHdl);
		if(MaxMinTHdl)	DisposeHandle((Handle)MaxMinTHdl);
		if(uVelHdl)		DisposeHandle((Handle)uVelHdl);
		if(vVelHdl)		DisposeHandle((Handle)vVelHdl);
		if(tHdl)		DisposeHandle((Handle)tHdl);*/
	}
	//if(AMPAPtr)		DisposePtr((Ptr)AMPAPtr);
	//if(epochPtr)	DisposePtr((Ptr)epochPtr);
	//if(AMPAPtr) {free(AMPAPtr); AMPAPtr = NULL;}
	//if(epochPtr) {free(epochPtr); epochPtr = NULL;}
	if (AMPAPtr) delete [] AMPAPtr;
	if (epochPtr) delete [] epochPtr;
	
	return(errorFlag);
}

// ***********************************************************



/*****************************************************************

double geRotAngle(short xAngle)
{
//
// Given xAngle in degrees from north, rotating counterclockwise
// (this is what everyone uses), function returns an angle
// between the x-axis in a east-west, north-south coordinate
// system and the angle sent in. The returned angle is 0
// if we point east, plus 90 if point to north.
//
	double theAngle;
	short sign;
	
	// OK begin by checking for wierd stuff
	// make sure all angles are positive and
	// between 0 and 360
	
	sign = 1;
	if(xAngle<0){
		xAngle=-xAngle;
		sign = -1;
	}
	if(xAngle>360){
		xAngle = mod(xAngle,360);
	}
	if(sign==-1){
		xAngle = 360 - xAngle;
	}
	
	if(xAngle<=90){
		theAngle = 90 - xAngle;
	}
	else if(xAngle<=180){
		theAngle = 360 - (xAngle-90);
	}
	else if(xAngle<=270){
		theAngle = 270 - (xAngle-180);
	}
	else if(xAngle<=360){
		theAngle = 180 - (xAngle-270);
	}
	return theAngle;
}


********************************************************************/




// **************************************

short GetSlackRatio(double t,
					short flag,
				    EXTFLAG *EbbFloodTimesPtr,
					short *EbbFloodArrayPtr,
				    short NoOfMaxMins,
				    double FloodRatio,
				    double EbbRatio,
					double *newRatio)
				
	// Function to compute slack ratio 
	// The ratios are interpolated between the
	// last flood and ebb.  If point falls in
	// first segment or last segment, we extroplate
	// according to distance between first or last
	// flood and ebb.
{

//#ifdef usethiscode	// DaveW dropped this conditional

	short errorFlag;
	long i,j,k;
	double t0,t1,dt,w0,w1,v0,v1,slackTime;
	short EbbToMinBFlood = 0, MinBFloodToFlood = 1;
	short FloodToMinBEbb=2, MinBEbbToEbb=3;
	//short MaxFlood=1,MaxEbb=3,MinBFlood=0,MinBEbb=2;
	short maxIndex[151],keyIndex[151],firstMax,lastMax,numOfMax;
	short firstMaxKey;
	short minBefore,minAfter;
	
	// Begin by finding indecies to max flood and ebbs
	// we don't need the mins before
	
	errorFlag = 0;
	
	j=0;
	minBefore = -1;
	minAfter = -1;
	for(i=0;i<NoOfMaxMins;i++){
		
		if( EbbFloodArrayPtr[i]==MaxFlood){
			maxIndex[j]=i;
			keyIndex[j]=MaxFlood;
			j=j+1;
		}
		else if(EbbFloodArrayPtr[i]==MaxEbb){
			maxIndex[j]=i;
			keyIndex[j]=MaxEbb;
			j=j+1;
		}
		else {
			if(minBefore==-1){
				if(t>EbbFloodTimesPtr[i].val){
					minBefore = i;
				}
			}
			if(minAfter==-1){
				if(t<EbbFloodTimesPtr[i].val){
					minAfter = i;
				}
			}
		}
		if(j==150)break;
	}
	
	firstMax = maxIndex[0];
	numOfMax = j-1;
	lastMax = maxIndex[numOfMax];
	
	// OK now we gotta find out which min we are
	// getting ratio for and what it's time is
	// for.  We will use the flag for this
	
	if(flag==EbbToMinBFlood){
		// Then we need to get next min and it's time
		if(minAfter>-1){
			slackTime = EbbFloodTimesPtr[minAfter].val;
		}
		else {
			slackTime = EbbFloodTimesPtr[NoOfMaxMins+1].val;
		}
	}
	else if(flag==MinBFloodToFlood){
		// Then we need to get next min and it's time
		if(minBefore>-1){
			slackTime = EbbFloodTimesPtr[minBefore].val;
		}
		else {
			slackTime = EbbFloodTimesPtr[NoOfMaxMins].val;
		}
	}
	else if(flag==FloodToMinBEbb){
		// Then we need to get next min and it's time
		if(minAfter>-1){
			slackTime = EbbFloodTimesPtr[minAfter].val;
		}
		else {
			slackTime = EbbFloodTimesPtr[NoOfMaxMins+1].val;
		}
	}
	else if(flag==MinBEbbToEbb){
		// Then we need to get next min and it's time
		if(minBefore>-1){
			slackTime = EbbFloodTimesPtr[minBefore].val;
		}
		else {
			slackTime = EbbFloodTimesPtr[NoOfMaxMins].val;
		}
	}
	
	
	// OK check to see where we are in the
	// high low array
	
	// less than first max
	if(t<EbbFloodTimesPtr[firstMax].val){
		
		// OK we need a way to interpolate if we are in the first
		// segment.  First check if first max, min is a max or a min
		// if it is a min, then we got max already stored so use it
		
		if(EbbFloodArrayPtr[0]==MinBeforeFlood){
			// This means we already have computed previous
			// max ebb time so grab the sucker and run
			t0 = EbbFloodTimesPtr[NoOfMaxMins].val;
			firstMaxKey = MaxEbb;
		}
		else if(EbbFloodArrayPtr[0]==MinBeforeEbb){
			// This means we already have computed previous
			// max flood time so grab the sucker and run
			t0 = EbbFloodTimesPtr[NoOfMaxMins].val;
			firstMaxKey = MaxFlood;
		}
		
		// if we get here, we gotta guess at dt, the time
		// difference between a max ebb and max flood for the
		// first interval.  Too much hassle to go back and
		// actually compute a previous max so we go ahead and
		// use the first actual dt between the first max ebb
		// and flood.  
		else if(EbbFloodArrayPtr[0]==MaxFlood){
			j = maxIndex[1];
			dt =EbbFloodTimesPtr[j].val - EbbFloodTimesPtr[firstMax].val;
			t0 = EbbFloodTimesPtr[firstMax].val - dt;
			if(t0>slackTime){
				// here we got a problem
				// the time spacing between ebb and floods
				// vary a lot ... give up
				// pick half way between flood and ebb
				dt = EbbFloodTimesPtr[firstMax].val - slackTime;
				t0 = slackTime - dt;
			}
			t1 = EbbFloodTimesPtr[firstMax].val;
			firstMaxKey = MaxEbb;
		}
		else if(EbbFloodArrayPtr[0]==MaxEbb){
			j = maxIndex[1];
			dt =EbbFloodTimesPtr[j].val - EbbFloodTimesPtr[firstMax].val;
			t0 = EbbFloodTimesPtr[firstMax].val - dt;
			if(t0>slackTime){
				// here we got a problem
				// the time spacing between ebb and floods
				// vary a lot ... give up
				// pick half way between flood and ebb
				dt = EbbFloodTimesPtr[firstMax].val - slackTime;
				t0 = slackTime - dt;
			}
			t1 = EbbFloodTimesPtr[firstMax].val;
			firstMaxKey = MaxFlood;
		
		}
		
	}
	
	// greater than last max
	else if(slackTime>EbbFloodTimesPtr[lastMax].val){

		
		// OK we need a way to interpolate if we are in the last
		// segment.  First check if last max, min is a max or a min
		// if it is a min, then we got max already stored so use it
		
		if(EbbFloodArrayPtr[NoOfMaxMins-1]==MinBeforeFlood){
			// This means we already have computed next
			// max flood time so grab the sucker and run
			t1 = EbbFloodTimesPtr[NoOfMaxMins+1].val;
			firstMaxKey = MaxEbb;
		}
		else if(EbbFloodArrayPtr[NoOfMaxMins-1]==MinBeforeEbb){
			// This means we already have computed next
			// max ebb time so grab the sucker and run
			t1 = EbbFloodTimesPtr[NoOfMaxMins+1].val;
			firstMaxKey = MaxFlood;
		}
		
		// if we get here, we gotta guess at dt, the time
		// difference between a max ebb and max flood for the
		// last interval.  Too much hassle to go back and
		// actually compute a previous max so we go ahead and
		// use the last actual dt between the last max ebb
		// and flood.  
		else if(EbbFloodArrayPtr[NoOfMaxMins-1]==MaxFlood){
			j = maxIndex[numOfMax-1];
			dt =EbbFloodTimesPtr[lastMax].val - EbbFloodTimesPtr[j].val;
			t1 = EbbFloodTimesPtr[lastMax].val - dt;
			if(t1<slackTime){
				// here we got a problem
				// the time spacing between ebb and floods
				// vary a lot ... give up
				// pick half way between flood and ebb
				dt = slackTime - EbbFloodTimesPtr[lastMax].val;
				t1 = slackTime + dt;
			}
			t0 = EbbFloodTimesPtr[lastMax].val;
			firstMaxKey = MaxFlood;
		}
		else if(EbbFloodArrayPtr[NoOfMaxMins-1]==MaxEbb){
			j = maxIndex[numOfMax-1];
			dt =EbbFloodTimesPtr[lastMax].val - EbbFloodTimesPtr[j].val;
			t1 = EbbFloodTimesPtr[lastMax].val - dt;
			if(t1<slackTime){
				// here we got a problem
				// the time spacing between ebb and floods
				// vary a lot ... give up
				// pick half way between flood and ebb
				dt = slackTime - EbbFloodTimesPtr[lastMax].val;
				t1 = slackTime + dt;
			}
			t0 = EbbFloodTimesPtr[lastMax].val;
			firstMaxKey = MaxEbb;
		}
				
	}
	// find were we are
	else{
		t0 = -1.0;
		t1 = -1.0;
		for (i=0;i< lastMax;i++){
			j = maxIndex[i];
			k = maxIndex[i+1];
			if( (t>=EbbFloodTimesPtr[j].val) && (t<=EbbFloodTimesPtr[k].val) ){
				t0 = EbbFloodTimesPtr[j].val;
				t1 = EbbFloodTimesPtr[k].val;
				firstMaxKey = EbbFloodArrayPtr[j];
				break;
			}
		}
		// If we drop to here, we have an error
		// flag the sucker and leave
		if(t0==-1.0){
			errorFlag = 36;
			return errorFlag;
		}
	}
			
	// OK we got t0 < slackTime < t1, get weight functions
	
	w1 = (slackTime-t0)/(t1-t0);
	w0 = 1.0 - w1;
	
	// Now compute the interpolated speed ratio
	
	v0 = FloodRatio;
	v1 = EbbRatio;
	
	if(firstMaxKey==MaxEbb){
		v0 = EbbRatio;
		v1 = FloodRatio;
	}
	
//	*newRatio = v0 * w0 + v1 * w1;
	
//#endif

	*newRatio = EbbRatio;
	
	if(EbbRatio>FloodRatio) *newRatio = FloodRatio;
	
	
	return(0);
}


// ************************************************************
void getVector(short degrees, double *u, double *v)
// takes angle degrees and returns a unit vector
// beginning at (0,0) and ending at (u,v)
// note degrees is 0 along U axis and rotates counterclockwise
{
	double exDegrees;
	double radians = 0.017453293;
	
	// Check for default cases
	if(degrees==0){
		*u = 1.0;
		*v = 0.0;
	}
	else if(degrees==90){
		*u = 0.0;
		*v = 1.0;
	}
	else if(degrees==180){
		*u = -1.0;
		*v = 0.0;
	}
	else if(degrees==270){
		*u = 0.0;
		*v = -1.0;
	}
	else if(degrees==360){
		*u = 1.0;
		*v = 0.0;
	}
	// non default case 
	else {
		exDegrees = degrees;
		exDegrees = exDegrees * radians;
		*u = cos(exDegrees);
		*v = sin(exDegrees);
	}
	return;
}

// ********************************************
short GetVelDir(CONSTITUENT *constituent,double u,double v)
// if returns 1 it is a flood
// if returns -1 it is an ebb
// if returns 0, it is undetermined
//
// The algorythm does the dot products between the velocity vector (u,v)
// and the flood direction (fx,fy) and the ebb direction (ex,ey)
// if the sign of the dot product is positive with the flood and 
// negative with the ebb, then we got a flood
// if the sign of the dot product is negative with the flood and
// positive with the ebb, then we got an ebb
// If the sign is both positive or both negative, then it is
// undetermined.  This can happen because the flood and ebb 
// directions are not 180 degrees out of phase.

{
	short errorFlag,sFloodDir,sEbbDir;
	double FDir,EDir;
	double Fx,Fy,Ex,Ey;
	double dotF,dotE;
	
		errorFlag = GetCDirec(constituent,&sFloodDir,&sEbbDir);
		if(errorFlag!=0)return errorFlag;
		
		FDir=sFloodDir;
		EDir=sEbbDir;
		
	// OK now we need to get vectors for flood direction and ebb direction
	
		getVector(FDir,&Fx,&Fy);
		getVector(EDir,&Ex,&Ey);
		
	// OK do dot product
		
		dotF = u*Fx + v*Fy;
		dotE = u*Ex + v*Ey;
	
	// Now check sign
		if( (dotF>0.0) && (dotE<0.0) ) return 1;
		if( (dotF<0.0) && (dotE>0.0) ) return -1;
		return 0;
}

// **********************************************************

short OffsetReferenceCurve(COMPCURRENTS *answer,    //  to reference station heights
						   CURRENTOFFSET *offset)   //  to offset Data
{
	short errorFlag,flag;
	long NoOfMaxMins,numOfPoints,i;
	double MaxFloodOffset,MaxEbbOffset,MinBFloodOffset,MinBEbbOffset,t;
	double slackRatio;
	double FloodRatio,EbbRatio;
	short EbbToMinBFlood = 0, MinBFloodToFlood = 1;
	short FloodToMinBEbb=2, MinBEbbToEbb=3;
	short FloodToEbb=4, EbbToFlood=5;
	//short maxFlood = 1, minBFlood = 0, maxEbb = 3, minBEbb = 2;
	double timeCorrection,SpeedCorrectionMult;
	double w1,w2,temp;
	//double t0,t3,dt,EbbTime,FloodTime;
	short index,indexPlusOne;
	//double previousTime,nextTime;
	//double tOld,rOld,fOld,dtOld,dtNew,oldStartValue,oldEndValue;
	//double nextValue,previousValue;
	
	double		*CArrayPtr=NULL;
	EXTFLAG		*TArrayPtr=NULL;
	double		*MaxMinHdl=NULL,*ValHdl=NULL;
	EXTFLAG		*MaxMinTimeHdl=NULL,*TimeHdl=NULL;
	short		*MaxMinFlagHdl = NULL;
	short		*OldMaxMinFlagPtr = NULL;
	

	CArrayPtr=NULL;
	TArrayPtr=NULL;
	errorFlag=0;
	
	/* Get handles just to make code more readable */
	MaxMinHdl		= answer->EbbFloodSpeeds;
	MaxMinTimeHdl	= answer->EbbFloodTimes;
	MaxMinFlagHdl	= answer->EbbFlood;
	ValHdl			= answer->speed;
	TimeHdl			= answer->time;
	
	// Begin by declaring temp space on heap 
	// for a copy of the maxs and mins
	
	NoOfMaxMins = answer->numEbbFloods;
	if(NoOfMaxMins<1){
		errorFlag=19;
		goto Error;
	}

	/*CArrayPtr=(double *)NewPtrClear(sizeof(double)*(NoOfMaxMins+2));
	if(CArrayPtr==Nnil){
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
	
	//OK now make copy of reference station data before
	// doing correction to it
	
	for (i=0; i< (NoOfMaxMins+2); i++){
		CArrayPtr[i] = MaxMinHdl[i];
		TArrayPtr[i].val = MaxMinTimeHdl[i].val;
		OldMaxMinFlagPtr[i] = MaxMinFlagHdl[i];
	}	


// ********

	MinBFloodOffset = offset->MinBefFloodTime.val;
	
	MaxFloodOffset =offset->FloodTime.val;

	MinBEbbOffset =offset->MinBefEbbTime.val;
	
	MaxEbbOffset =offset->EbbTime.val;
	
	FloodRatio = offset->FloodSpdRatio.val;
	
	EbbRatio = offset->EbbSpdRatio.val;


// ********
	// OK here is the strategy we go from point to point,
	// figure out between which high and low, the point falls.
	// Then interpolate the height offset and the time offset
	// of the point and apply it.
	// The points that are at the ends and don't fall
	// between a high and low will have to be handled as special cases
	
	numOfPoints = answer->nPts;
		
    // 2/25/2014 G.K.: added varaible initialization
    t = TimeHdl[0].val;
    SpeedCorrectionMult = 1.;
    flag = 99;
   	//previousTime = t;
    //nextTime = t;
    //dtNew = 0.;
    indexPlusOne = 0;

 	for (i=0; i<numOfPoints; i++){
	
		//tOld = t;
		//rOld = SpeedCorrectionMult;
		//fOld = flag;
		//oldStartValue = previousTime;
		//oldEndValue = nextTime;
		
		t = TimeHdl[i].val;
		if(i==0)index=-99;
		
		//dtOld = dtNew;
		//dtNew = t-tOld;
		
		errorFlag = GetWeights(t,TArrayPtr,&w1,&w2,&index,NoOfMaxMins);
		
		// davew 3/20/3: don't know what to do with this: it doesn't generate an error
		// I'm going to comment it out
		//if( (w1<0.0) || (w1>1.0) )printError("ShioCurrent - BAD WEIGHT FACTORS");
		
		// in first segment
		if(index==-99){
			if( MaxMinFlagHdl[0]==MaxFlood){
				flag = MinBFloodToFlood;
				if( MaxMinFlagHdl[NoOfMaxMins]==MaxEbb){
					flag = EbbToFlood;
				}
			}
			else if( MaxMinFlagHdl[0]==MinBeforeFlood){
				flag = EbbToMinBFlood;
			}
			else if( MaxMinFlagHdl[0]==MaxEbb){
				flag = MinBEbbToEbb;
				if( MaxMinFlagHdl[NoOfMaxMins]==MaxFlood){
					flag = FloodToEbb;
				}
			}
			else {
				flag = FloodToMinBEbb;
			}
		}
		
		// Not in first segment
		else {
			if( MaxMinFlagHdl[index]==MaxFlood){
				flag = FloodToMinBEbb;
				indexPlusOne = index + 1;
				
				// Check if last segment
				if(indexPlusOne>NoOfMaxMins)indexPlusOne = NoOfMaxMins + 1;
				if( MaxMinFlagHdl[indexPlusOne]==MaxEbb){
					flag = FloodToEbb;
				}
			}
			else if( MaxMinFlagHdl[index]==MinBeforeFlood){
				flag = MinBFloodToFlood;
			}
			else if( MaxMinFlagHdl[index]==MaxEbb){
				flag = EbbToMinBFlood;
				
				// Check if last segment
				if(indexPlusOne>NoOfMaxMins)indexPlusOne = NoOfMaxMins + 1;
				if( MaxMinFlagHdl[indexPlusOne]==MaxFlood){
					flag = EbbToFlood;
				}
				
			}
			else {
				flag = MinBEbbToEbb;
			}
		}
		
		if(errorFlag == 0){
			
	 		if( (flag!=EbbToFlood) || (flag!=FloodToEbb) ) {
				errorFlag = GetSlackRatio(t,
						flag,
				    	TArrayPtr,
						OldMaxMinFlagPtr,
				    	NoOfMaxMins,
				    	FloodRatio,
				    	EbbRatio,
						&slackRatio);
				if(errorFlag!=0) return errorFlag;
			}
	/* 		
			errorFlag = GetFloodEbbSpans(t,
					  	TArrayPtr,
					  	OldMaxMinFlagPtr,
					 	CArrayPtr,
					  	NoOfMaxMins,
					  	&previousTime,
					 	 &nextTime,
					 	 &previousValue,
					 	 &nextValue,
					 	 1);
	*/
			if(flag==FloodToMinBEbb){
				timeCorrection = MaxFloodOffset*w1 + MinBEbbOffset*w2;
				//FloodTime = previousTime;
				//EbbTime = nextTime;
				//dt = EbbTime - FloodTime;
				//t0 = FloodTime - dt;
				//t3 = EbbTime + dt;
	//			DoHermite(0.0,t0,FloodRatio,FloodTime,EbbRatio,
	//		          EbbTime,0.0,t3,t,&SpeedCorrectionMult,3);
	 			SpeedCorrectionMult = FloodRatio*w1 + w2*slackRatio;
			}
			
			else if(flag==MinBFloodToFlood){
				timeCorrection = MinBFloodOffset*w1 + MaxFloodOffset*w2;
				//EbbTime = previousTime;
				//FloodTime = nextTime;
				//dt = FloodTime - EbbTime;
				//t0 = EbbTime - dt;
				//t3 = FloodTime + dt;
		//		DoHermite(0.0,t0,EbbRatio,EbbTime,FloodRatio,
		//		          FloodTime,0.0,t3,t,&SpeedCorrectionMult,3);
	 			SpeedCorrectionMult = w1*slackRatio + FloodRatio*w2;
			}

			else if(flag==EbbToMinBFlood){
				timeCorrection = MinBFloodOffset*w1 + MaxFloodOffset*w2;
				//EbbTime = previousTime;
				//FloodTime = nextTime;
				//dt = FloodTime - EbbTime;
				//t0 = EbbTime - dt;
				//t3 = FloodTime + dt;
		//		DoHermite(0.0,t0,EbbRatio,EbbTime,FloodRatio,
		//		          FloodTime,0.0,t3,t,&SpeedCorrectionMult,3);
	 			SpeedCorrectionMult = EbbRatio*w1 + w2*slackRatio;
			}

			else if(flag==MinBEbbToEbb){
				timeCorrection = MinBEbbOffset*w1 + MaxEbbOffset*w2;
				//FloodTime = previousTime;
				//EbbTime = nextTime;
				//dt = FloodTime - EbbTime;
				//t0 = FloodTime - dt;
				//t3 = EbbTime + dt;
		//		DoHermite(0.0,t0,FloodRatio,FloodTime,EbbRatio,
		//		          EbbTime,0.0,t3,t,&SpeedCorrectionMult,3);
	 			SpeedCorrectionMult = w1*slackRatio + EbbRatio*w2;
			}
			
			else if(flag==EbbToFlood){
				timeCorrection = MinBFloodOffset*w1 + MaxFloodOffset*w2;
				//EbbTime = previousTime;
				//FloodTime = nextTime;
				//dt = FloodTime - EbbTime;
				//t0 = EbbTime - dt;
				//t3 = FloodTime + dt;
		//		DoHermite(0.0,t0,EbbRatio,EbbTime,FloodRatio,
		//		          FloodTime,0.0,t3,t,&SpeedCorrectionMult,3);
	 			SpeedCorrectionMult = EbbRatio + FloodRatio*w2;
			}
			else if(flag==FloodToEbb){
				timeCorrection = MaxFloodOffset*w1 + MaxEbbOffset*w2;
				//EbbTime = nextTime;
				//FloodTime = previousTime;
				//dt = FloodTime - EbbTime;
				//t0 = FloodTime - dt;
				//t3 = EbbTime + dt;
		//		DoHermite(0.0,t0,FloodRatio,FloodTime,EbbRatio,
		//	          EbbTime,0.0,t3,t,&SpeedCorrectionMult,3);
	 			SpeedCorrectionMult = w1*FloodRatio + EbbRatio*w2;
			}
			
			TimeHdl[i].val=TimeHdl[i].val + timeCorrection;
	 		ValHdl[i] = (ValHdl[i]) * SpeedCorrectionMult;
	//		ValHdl[i] = 1.0 * SpeedCorrectionMult;
			temp = ValHdl[i];
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

short OffsetUV ( COMPCURRENTS *answers,		// Current-time struc with answers
			     CURRENTOFFSET *offset)		// Current offset data
// Function to set uHdl and vHdl data for offset flood and ebb directions
// They come in with nil if the station is non rotary or with
// the u and v from the reference station.  We need to rotate them
// If the station has data that says the average min current is non zero,
// then we will zero out the u and v handles and return.  We don't know
// enough.  If the stuff is zero or unavailable, then we will have the
// u and v in the flood and ebb directions with the magnitude varying.
{
	double minBFloodSpeed,minBEbbSpeed;
	double floodDir,ebbDir,uMag,argu,tenMinutes;
	double ebbU,ebbV,floodU,floodV,uVel,t;
	long 	numberOfPoints,i,j,k;
	short 	errorFlag,rotKey;
	short fAngle,eAngle;
	
	long numOfMaxMins;
	short keepFlag;
	
	double *uHdl=NULL,*vHdl=NULL,*uMinorHdl=NULL,*vMajorHdl=NULL,*speedHdl=NULL;
	EXTFLAG *theTimeHdl=NULL;
	EXTFLAG  *MaxMinTimesHdl=NULL;
	short		*MaxMinKeyHdl=NULL;

	short maxFlood=1;
	short maxEbb=3;
	
// initialize
	
	errorFlag = 0;
	
	uHdl = answers->u;
	vHdl = answers->v;
	speedHdl = answers->speed;
	theTimeHdl = answers->time;
	
	uMinorHdl = answers->uMinor;
	vMajorHdl = answers->vMajor;

	numberOfPoints	= answers->nPts;
	
	numOfMaxMins = answers->numEbbFloods;
	MaxMinTimesHdl = answers->EbbFloodTimes;
	MaxMinKeyHdl = answers->EbbFlood;
	
	minBFloodSpeed = 0.0;
	minBEbbSpeed = 0.0;
	floodDir = 0.0;
	ebbDir = 0.0;
	rotKey = 1;

	// Begin by checking the type of station
		
	if( ( offset->MinBFloodSpd.dataAvailFlag)==1) {
		if( ( offset->MinBFloodSpd.val)!=0.0) {
			minBFloodSpeed = offset->MinBFloodSpd.val;
			rotKey = 0;
		}
	}

	if( ( offset->MinBEbbSpd.dataAvailFlag)==1) {
		if( ( offset->MinBEbbSpd.val)!=0.0) {
			minBEbbSpeed = offset->MinBEbbSpd.val;
			rotKey = 0;
		}
	}

	if( ( offset->MinBFloodDir.dataAvailFlag)==1) {
		rotKey = 0;
	}

	if( ( offset->MinBEbbDir.dataAvailFlag)==1) {
		rotKey = 0;
	}
	
	// *********
	
	if( offset->MaxFloodDir.dataAvailFlag ==1 ) {
		fAngle = offset->MaxFloodDir.val;
		fAngle = FixAngle(fAngle);
		floodDir = fAngle;
	}
	else {
		rotKey = -1;
	}

	if( offset->MaxEbbDir.dataAvailFlag ==1 ) {
		eAngle = offset->MaxEbbDir.val;
		eAngle = FixAngle(eAngle);
		ebbDir = eAngle;
	}
	else {
		rotKey = -1;
	}


	// OK compare angles of flood and ebb, if the angles are more than 
	// 15 degrees off from being 180 out of phase, we kill the station.
	// The sucker is actually rotary and we have no clue as to how they
	// rotate.
	
	if(rotKey==1){	
		
		argu = fabs(floodDir - ebbDir);
		if( (argu < 165.0) || ( argu > 195) ){
			rotKey = 0;
		}
	
	}
	
//	If we get here and rotKey == 0 then we don't know what's going on
//	and we zero out the u and v handles and return
	
	if(rotKey==-1 ){
		if(uHdl!=0){
			//DisposeHandle( (Handle)answers->uHdl);
			delete [] answers->u;
			answers->u=0;
			uHdl=0;
		}
		if(vHdl!=0){
			//DisposeHandle( (Handle)answers->vHdl);
			delete [] answers->v;
			answers->v=0;
			vHdl=0;
		}
		if(uMinorHdl!=0){
			//DisposeHandle( (Handle)answers->uMinorHdl);
			delete [] answers->uMinor;
			answers->uMinor=0;
			uMinorHdl=0;
		}
		if(vMajorHdl!=0){
			//DisposeHandle( (Handle)answers->vMajorHdl);
			delete [] answers->vMajor;
			answers->vMajor=0;
			vMajorHdl=0;
		}

		// OK take off the no plot flags and return
	
		if(rotKey==-1){
			for(i=0;i<numberOfPoints;i++){
				theTimeHdl[i].flag = 0;
			}
		
			return errorFlag;		
	 	}
 		
	}
	
	

	// if we get here, we gotta rotate the sucker to flood and ebb
	
	// first lets get rid of the uMinor and vMajor handles.
	// we don't do this for the offset stations

	if(uMinorHdl!=0){
		//DisposeHandle( (Handle)answers->uMinorHdl);
		delete [] answers->uMinor;
		answers->uMinor=0;
		uMinorHdl=0;
	}
	if(vMajorHdl!=0){
		//DisposeHandle( (Handle) answers->vMajorHdl);
		delete [] answers->vMajor;
		answers->vMajor=0;
		vMajorHdl=0;
	}
	
	// OK first find out if we got u and v handles already
	// if the data came off a rotary reference, then we got
	// u and v handles, if not we don't and need to create
	// them.
	
	
	if(uHdl==0){
		//uHdl = (double **)NewHandleClear(sizeof(double)*numberOfPoints);
		uHdl = new double[numberOfPoints];
		if(uHdl==0){
			errorFlag=31;
			return errorFlag;
		}
	}
	
	if(vHdl==0){
		//vHdl = (double **)NewHandleClear(sizeof(double)*numberOfPoints);
		vHdl = new double[numberOfPoints];
		if(vHdl==0){
			errorFlag=31;
			return errorFlag;
		}
	}
	
	
	// OK we now have array space so we loop through and fill the sucker up
	
	// Get a flood unit vector and an ebb unit vector
	
	argu = floodDir * PI/180.0;
	
	floodU = cos(argu);
	floodV = sin(argu);
	
	argu = ebbDir * PI/180.0;
	
	ebbU = cos(argu);
	ebbV = sin(argu);
	
	// Now loop through and do the u and v's from the magnitude of the vector
	
	for (i=0;i<numberOfPoints;i++){
		uVel = speedHdl[i];
		uMag = fabs(uVel);
		
		if(uVel>0.0){
			uHdl[i] = uMag * floodU;
			vHdl[i] = uMag * floodV;
		}
		else {
			uHdl[i] = uMag * ebbU;
			vHdl[i] = uMag * ebbV;
		}
	}

	// OK take off the no plot flags and return
	// unless we have min velocities.  Then we check
	// what the magnitudes of the velocities and
	// flag them as no plot.
	
    long NoOfMaxMins;
    short flag;
	double  *CArrayPtr;
	EXTFLAG	*TArrayPtr;
	short	*OldMaxMinFlagPtr;

	CArrayPtr=NULL;
	TArrayPtr=NULL;
	
	// Begin by declaring temp space on heap 
	// for a copy of the maxs and mins
	
	NoOfMaxMins = answers->numEbbFloods;
	if(NoOfMaxMins < 1)
    {
		errorFlag = 19;
		goto Error;
	}

	CArrayPtr = new double[NoOfMaxMins+2];
	if(CArrayPtr==NULL){
		errorFlag = 20;
		goto Error;
	}
	
	TArrayPtr = new EXTFLAG[NoOfMaxMins+2];
	if(TArrayPtr==NULL){
		errorFlag = 21;
		goto Error;
	}
	
	OldMaxMinFlagPtr = new short[NoOfMaxMins+2];
	if(OldMaxMinFlagPtr==NULL){
		errorFlag = 37;
		goto Error;
	}
	
    for(i=0, j=0; i<(NoOfMaxMins+2); i++)
        if(answers->EbbFloodTimes[i].flag == 0)
        {
		    CArrayPtr[j] = answers->EbbFloodSpeeds[i];
            TArrayPtr[j].val = answers->EbbFloodTimes[i].val;
            TArrayPtr[j].flag = answers->EbbFloodTimes[i].flag;
		    OldMaxMinFlagPtr[j] = answers->EbbFloodTimes[i].flag;
            j++;
	    }	
    NoOfMaxMins = j-2;

	for(i=0;i<numberOfPoints;i++){
		theTimeHdl[i].flag = 0;
		
        if(GetFloodEbbKey(theTimeHdl[i].val,
					   TArrayPtr,
					   OldMaxMinFlagPtr,
					   CArrayPtr,
					   NoOfMaxMins,
					   &flag))
        {
				uHdl[i] = 0.0;
				vHdl[i] = 0.0;
                continue;
		}

		if(minBFloodSpeed>0.0 && (flag == 0 || flag == 1)){
			uVel = speedHdl[i];
			//if(uVel>0.0){
				uMag = uVel;
				if(uMag<=minBFloodSpeed){
					theTimeHdl[i].flag = 1;
				}
			//}
		}

		if(minBEbbSpeed>0.0 && (flag == 2 || flag == 3)){
			uVel = speedHdl[i];
			//if(uVel<0.0){
				uMag = fabs(uVel);
				if(uMag<=minBFloodSpeed){
					theTimeHdl[i].flag = 1;
				}
			//}
		}
		
		// OK if rotKey == 0 then we zero out the
		// u and v currents except for within 10 minutes
		// of max flood and ebb
		if(rotKey==0){
		
			keepFlag = 0;
			tenMinutes = 1./6.;
			for (j=0;j<numOfMaxMins;j++){
				k = MaxMinKeyHdl[j];
				if ( (k==maxFlood) || (k==maxEbb) ){
					t = MaxMinTimesHdl[j].val;
				}
                // G.K.: bug?? - 2/8/08 - what if k is not equal max flood or max ebb?
                // added just to initialize variable
                else
                    t = 0;  // ??? - is this right?
				if( fabs(t - theTimeHdl[i].val) <= tenMinutes){
					keepFlag = 1;
					break;
				}
			}
			
			if(keepFlag==0){
				uHdl[i] = 0.0;
				vHdl[i] = 0.0;
			}
			
		}
		
	}


	answers->u = uHdl;
	answers->v = vHdl;
	
Error:
	
	if (CArrayPtr) delete [] CArrayPtr;
	if (TArrayPtr) delete [] TArrayPtr;
	if (OldMaxMinFlagPtr) delete [] OldMaxMinFlagPtr;

	//return 0;
	return errorFlag;
}

// ************************************************************

void ResetCTime(COMPCURRENTS *answers,double beginHour)
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
	
	numberOfPoints = answers->numEbbFloods;
	theTimeHdl = answers->EbbFloodTimes;
	
	for(i=0;i<numberOfPoints;i++){
		//theTime = theTimeHdl[i].val;
	  	theTimeHdl[i].val = theTimeHdl[i].val - beginHour;
	}
	 
	return;
}

//************************************************************
void rotateVector(double degrees,
				  double u0,
				  double v0,
				  double *u1,
				  double *v1)
{				  
//
// formula for pure rotation of a vector
// degrees is angle difference between (u0,v0) 
// reference coordinate system and (u1,v1)
// coordinate system.  
//
	double theCosine,theSine,degreesToRadians;
	
	degreesToRadians = PI/180.;
	
	theCosine = cos (degrees * degreesToRadians);
	theSine = sin (degrees * degreesToRadians);
	
	*u1 = theCosine * u0 + theSine   * v0;
	*v1 =-theSine   * u0 + theCosine * v0;
	
	return;
}

//**************************************************************

short RotToMajorMinor(CONSTITUENT *constituent,
					  CURRENTOFFSET *offset,
                      COMPCURRENTS *answers)

// function to rotate velocity vector to major-minor axis
// The x component will be the minor axis component
// The y component will be the major axis component

{	
	double	velAngle,u,v,uMag,ang,ang0;
	short	majorDir,minorDir;
	long	nPts,i;
	double	*minorHdl=0,*majorHdl=0,*uHdl=0,*vHdl=0,*magHdl=0;
	long	handleSize=0;
	double	radiansToDegrees,degreesToRadians;
	double	uRotated,vRotated,piOver2;
	
	radiansToDegrees = 180.0/PI;
	degreesToRadians = PI/180.0;
	piOver2 = PI/2.0;
	

// ***********  Here we check to see if it is an offset station
//              if it is, we don't do the major minor axis stuff

	if( ( offset->MinBefFloodTime.dataAvailFlag)==1) {
		if( ( offset->MinBefFloodTime.val)!=0.0) {
			return 0;
		}
	}

	if( ( offset->FloodTime.dataAvailFlag)==1) {
		if( ( offset->FloodTime.val)!=0.0) {
			return 0;
		}
	}

	if( ( offset->MinBefEbbTime.dataAvailFlag)==1) {
		if( ( offset->MinBefEbbTime.val)!=0.0) {
			return 0;
		}
	}

	if( ( offset->EbbTime.dataAvailFlag)==1) {
		if( ( offset->EbbTime.val)!=0.0) {
			return 0;
		}
	}

	if( ( offset->FloodSpdRatio.dataAvailFlag)==1) {
		if( ( offset->FloodSpdRatio.val)!=1.0) {
			return 0;
		}
	}

	if( ( offset->EbbSpdRatio.dataAvailFlag)==1) {
		if( ( offset->EbbSpdRatio.val)!=1.0) {
			return 0;
		}
	}

// **********

	// Begin by getting the major and minor axis direction	
	
	GetMajorMinorAxis(constituent,&majorDir,&minorDir);

	// Put things into a normal coordinate reference frame
	majorDir = FixAngle(majorDir);
	minorDir = FixAngle(minorDir);
	
	minorHdl = answers->uMinor;
	majorHdl = answers->vMajor;
	uHdl = answers->u;
	vHdl = answers->v;
	magHdl = answers->speed;
	
	// OK if we already have a non zero handle for the rotated
	// velocities, then I will wipe it out and allocated
	// space for a new handle.  I do this because I want to
	// make sure that the size of the handle matches what we
	// need.
	
	/*if( (minorHdl!=nil)||(majorHdl!=nil) ){
		HUnlock((Handle)answers->uMinorHdl);
		HUnlock((Handle)answers->vMajorHdl);
		DisposeHandle((Handle)answers->uMinorHdl);
		DisposeHandle((Handle)answers->vMajorHdl);
		minorHdl = answers->uMinorHdl =  0;
		majorHdl = answers->vMajorHdl = 0;
	}*/
	
	nPts = answers->nPts;
	
	try
	{
		minorHdl = new double[nPts];
		//majorHdl = new double[nPts];
		
		// davew: Andy used NewPtrClear & NewHandleClear and it looks like he relied
		// on it, in at least some cases, to initialize his arrays
		// in this case we don't have to init, they get set below
	}
	catch (...)
	{
		return 35;
	}
	try
	{
		majorHdl = new double[nPts];
	}
	catch (...)
	{
		return 35;
	}

	// Now set handle size to match u and v handles
	//handleSize = (long)GetHandleSize((Handle)uHdl);
	//if(handleSize==0){ return 34; }
	
	// Now allocate the space
	
	//minorHdl = (double **)NewHandleClear(handleSize);
	//majorHdl = (double **)NewHandleClear(handleSize);
	
	// clean up and return if error
	//if( (minorHdl==nil) || (majorHdl==nil) ){
		//DisposeHandle((Handle)minorHdl);
		//DisposeHandle((Handle)majorHdl);
		//return 35;
	//}
	
	//nPts = answers->nPts;
	
	ang0 = majorDir;
	
	// OK now loop through all the velocities and
	// compute and save the rotated suckers
	
	for (i=0;i<nPts;i++){
		u = uHdl[i];
		v = vHdl[i];
		uMag = magHdl[i];
		
		// figure out angle between new axis and velocity vector
		if(u!=0.0){
			velAngle = atan(v/u);
		}
		else {
			velAngle = 0.0;
		}
		
		// fix angle if it is in second or third quadrant
		
		if( (u<0.0)&&(v>0.0) ){
			velAngle = PI + velAngle;
		}
		else if( (u<0.0)&&(v<0.0) ) {
			velAngle = PI + velAngle;
		}
		else if( (u>0.0)&&(v<0.0) ){
			velAngle = 2*PI+velAngle;
		}
		
		ang = velAngle - ang0 * degreesToRadians;
		vRotated = uMag*cos(ang);
		uRotated = uMag*sin(ang);
		// Check sign
		// if projected angle greater than pi/2 than
		// projection is negative
		if( (ang>piOver2) || (ang<-piOver2) ) {
			vRotated = - fabs(vRotated);
		}
		else {
			vRotated = fabs(vRotated);
		}
		if( ang > 0.0 ){
			if(ang < PI){
				uRotated =-fabs(uRotated);
			}
			else {
				uRotated = +fabs(uRotated);
			}
		}
		else if(ang < 0.0){
			if(ang < -PI ){
			  uRotated = +fabs(uRotated);
			}
			else {
			  uRotated =-fabs(uRotated);
			}
		}
		
		majorHdl[i] = vRotated;
		minorHdl[i] = uRotated;
	}
	
	// assign the handle and return
	answers->uMinor = minorHdl;
	answers->vMajor = majorHdl;
	
	return 0;
}

/***************************************************************************************/
double RStatCurrent(double	theTime,	// time in hrs from begin of year
                     double	*AMPA,		// amplitude corrected for year
					 double	*epoch,		// epoch corrected for year
					 double	refCur,		// reference current in knots
					 short	ncoeff,		// number of coefficients to use
					 short	CFlag)		// hydraulic station flag
					                       
{
/*  Compute cosine stuff and return value */


/*  initialize the basic 37 frequencies ... if we pickup more, like */
/*  for Anchorage, we add them here */

//                  frequency 1/hr     symbol  index   period      name

	double f[] = {	28.9841042,   // M2         1   12.421 hrs. Principal lunar
					30.0000000,   // S2         2   12.000 hrs  Principal solar
					28.4397295,   // N2         3   12.685 hrs  Larger lunar elliptic
					15.0410686,   // K1         4   23.934 hrs  Luni-solar diurnal
					57.9682084,   // 6.21 hrs   5    6.210 hrs  
					13.9430356,   // O1         6   25.819 hrs  Principal lunar diurnal
					86.9523127,   // 4.14 hrs   7    4.14  hrs
					44.0251729,   // 8.11 hrs   8    8.11  hrs
					60.0000000,   // 6.00 hrs   9    6.00  hrs
					57.4238337,   // 6.27 hrs  10    6.27  hrs
					28.5125831,   // v2        11   12.626 hrs  Larger lunar evectional
					90.0000000,   // 4.00 hrs  12    4.000 hrs
					27.9682084,   // m2        13   12.872 hrs  Variational
					27.8953548,   // 2N2       14   12.905 hrs  Lunar ellipic second order
					16.1391017,   // 22.31 hrs 15   22.306 hrs  
					29.4556253,   // l2        16   12.222 hrs  Smaller lunar elliptic
					15.0000000,   // 24.00 hrs 17   24.000 hrs
					14.4966939,   // 24.83 hrs 18   24.833 hrs
					15.5854433,   // 23.10 hrs 19   23.098 hrs
					0.5443747,    // 27.55 day 20   27.55  day  Lunar monthly
					0.0821373,    // 182.6 day 21  182.6   day
					0.0410686,    // 365.2 day 22  365.2   day  Annual
					1.0158958,    // 14.7  day 23   14.7   day
					1.0980331,    // 13.66 day 24   13.66  day  Lunar fortnightly
					13.4715145,   // 26.72 hrs 25   26.72  hrs
					13.3986609,   // Q1        26   26.868 hrs  Larger lunar elliptic
					29.9589333,   // T2        27   12.016 hrs  Larger solar elliptic
					30.0410667,   // 11.98 hrs 28   11.98  hrs
					12.8542862,   // 28.01 hrs 29   28.01  hrs
					14.9589314,   // P1        30   24.066 hrs  Principal solar diurnal
					31.0158958,   // 11.61 hrs 31   11.61  hrs
					43.4761563,   // 8.28  hrs 32    8.28  hrs
					29.5284789,   // L2        33   12.192 hrs  Smaller lunar elliptic
					42.9271398,   // 8.37  hrs 34    8.37  hrs
					30.0821373,   // 11.97 hrs 35   11.97  hrs
					115.9364169,  // 3.105 hrs 36    3.105 hrs
					58.9841042,    // 6.103 hrs 37    6.103 hrs
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
	double current,argu,degrees;
	short i;					    
					 
/* OK do the computational loop */

	current = 0.0;
	if(ncoeff<=0) ncoeff=37;
		
	for (i=0; i<ncoeff; i++){
	  // Don't do math if we don't have to.
	  if(AMPA[i]!=0){
	  	 degrees = (f[i] * theTime + epoch[i]);
		 degrees = fmod(degrees,360.0);
		 argu = degrees * DegreesToRadians;
 //   argu =  (f[i] * theTime + epoch[i]) * DegreesToRadians;
         current = current + AMPA[i]*cos(argu);
	  }
	}
		
	// Here check to see if we have hydraulic station
	if(CFlag==1){
		if(current>0.0){
			current = sqrt(current);
		}
		else if(current<0.0){
			current = -sqrt(-current);
		}
	}
	
	current = refCur + current;
	
	return current;
}


/***************************************************************************************/
double RStatCurrentRot(double		theTime,		// time in hrs from begin of year
                     double			*AMPA,			// amplitude corrected for year
					 double			*epoch,			// epoch corrected for year
					 short			ncoeff,			// number of coefficients to use
					 CONSTITUENT	*constituent,	// constituent handle
					 double			twoValuesAgo,	// previous, previous value
					 double			lastValue,		// previous value
					 double			*uVelocity,		// east-west component of velocity
					 double			*vVelocity,		// north-south component of velocity
					 double			*vmajor,		// major axis velocity if computed
					 double			*vminor,		// minor axis velocity if computed
					 short			*direcKey)		// direction key -1 ebb, 1 flood, 0 indeterminate
					                       
{
/*  Compute cosine stuff and return value */
/* for rotary currents */


/*  initialize the basic 37 frequencies ... if we pickup more, like */
/*  for Anchorage, we add them here */

//                  frequency 1/hr     symbol  index   period      name

	double f[] = {	28.9841042,   // M2         1   12.421 hrs. Principal lunar
					30.0000000,   // S2         2   12.000 hrs  Principal solar
					28.4397295,   // N2         3   12.685 hrs  Larger lunar elliptic
					15.0410686,   // K1         4   23.934 hrs  Luni-solar diurnal
					57.9682084,   // 6.21 hrs   5    6.210 hrs  
					13.9430356,   // O1         6   25.819 hrs  Principal lunar diurnal
					86.9523127,   // 4.14 hrs   7    4.14  hrs
					44.0251729,   // 8.11 hrs   8    8.11  hrs
					60.0000000,   // 6.00 hrs   9    6.00  hrs
					57.4238337,   // 6.27 hrs  10    6.27  hrs
					28.5125831,   // v2        11   12.626 hrs  Larger lunar evectional
					90.0000000,   // 4.00 hrs  12    4.000 hrs
					27.9682084,   // m2        13   12.872 hrs  Variational
					27.8953548,   // 2N2       14   12.905 hrs  Lunar ellipic second order
					16.1391017,   // 22.31 hrs 15   22.306 hrs  
					29.4556253,   // l2        16   12.222 hrs  Smaller lunar elliptic
					15.0000000,   // 24.00 hrs 17   24.000 hrs
					14.4966939,   // 24.83 hrs 18   24.833 hrs
					15.5854433,   // 23.10 hrs 19   23.098 hrs
					0.5443747,    // 27.55 day 20   27.55  day  Lunar monthly
					0.0821373,    // 182.6 day 21  182.6   day
					0.0410686,    // 365.2 day 22  365.2   day  Annual
					1.0158958,    // 14.7  day 23   14.7   day
					1.0980331,    // 13.66 day 24   13.66  day  Lunar fortnightly
					13.4715145,   // 26.72 hrs 25   26.72  hrs
					13.3986609,   // Q1        26   26.868 hrs  Larger lunar elliptic
					29.9589333,   // T2        27   12.016 hrs  Larger solar elliptic
					30.0410667,   // 11.98 hrs 28   11.98  hrs
					12.8542862,   // 28.01 hrs 29   28.01  hrs
					14.9589314,   // P1        30   24.066 hrs  Principal solar diurnal
					31.0158958,   // 11.61 hrs 31   11.61  hrs
					43.4761563,   // 8.28  hrs 32    8.28  hrs
					29.5284789,   // L2        33   12.192 hrs  Smaller lunar elliptic
					42.9271398,   // 8.37  hrs 34    8.37  hrs
					30.0821373,   // 11.97 hrs 35   11.97  hrs
					115.9364169,  // 3.105 hrs 36    3.105 hrs
					58.9841042,    // 6.103 hrs 37    6.103 hrs
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
	double current,argu,degrees,floodAngle,ebbAngle;
	double u,v,uRefCur,vRefCur,absoluteCurrentValue;
	double sign,lastSlope,absoluteLastValue;
	double absoluteTwoValuesAgo,FDir,EDir,newSlope;
	double xVel,yVel;
	short i,j,start,istop,hardWayFlag,errorFlag;
	short L2Flag,HFlag,RotFlag;
	short MajorMinorKey=2;
	short Undetermined = 0;
	short sFloodDir,sEbbDir;
	short directionFlag;

	 errorFlag = GetControlFlags(constituent,&L2Flag,&HFlag,&RotFlag);
	 
     current = 0.0;
	 *vmajor = 0.0;
	 *vminor = 0.0;
	 
  // initialize variable to track sign change
  // if lastValue or twoValuesAgo are zero,
  // then we gotta compute if flood or ebb the hard way
  
   hardWayFlag = 0;
   if( (twoValuesAgo==999.0) || (lastValue==999.0) )hardWayFlag = 1;
   
   if(hardWayFlag==0){
   
   		sign = -1.0;
   		if(lastValue>=0) sign = 1.0;
   
   		absoluteTwoValuesAgo = fabs(twoValuesAgo);
   
   		absoluteLastValue = fabs(lastValue);

   		lastSlope = absoluteLastValue - absoluteTwoValuesAgo;
   }
   
/* OK do the computational loop */
/* Note that north-south component is first in data*/

	v = 0.0;
	istop = ncoeff/2;;
	start = 0;
	
	vRefCur = GetDatum(constituent);
	
	for (i=start; i< istop; i++){
	  // Don't do math if we don't have to.
	  if(AMPA[i]!=0.0){
	  	 degrees = (f[i] * theTime + epoch[i]);
		 degrees = fmod(degrees,360.0);
		 argu = degrees * DegreesToRadians;
         v = v + AMPA[i]*cos(argu);
	  }
	}
	
	v = vRefCur + v;
	
	// now do the other axis
	
	start = istop;
	istop = ncoeff;

	u = 0.0;
	uRefCur = 0.0;
		 
	
 	uRefCur = GetDatum(constituent);
	
		for (i=start; i< istop; i++){
		  // Don't do math if we don't have to.
		  if(AMPA[i]!=0.0){
		  	 j=i-start;
	 	 	 degrees = (f[j] * theTime + epoch[i]);
			 degrees = fmod(degrees,360.0);
			 argu = degrees * DegreesToRadians;
       	  	 u = u + AMPA[i]*cos(argu);
	 	 }
		}		
	
 	u = uRefCur + u;
	
	// Now get the vector sum
	
	absoluteCurrentValue = sqrt(u*u + v*v);
	
	// Now get east-west and north-south velocity components
	
	if(RotFlag==MajorMinorKey){
		
		// Save major minor axis stuff
		*vmajor = v;
		*vminor = u;
		
		// take projection of velocity upon axis
		errorFlag = GetCDirec(constituent,&sFloodDir,&sEbbDir);
			if(errorFlag==33){
				*direcKey=errorFlag;
				return -99.0;
			}
			FDir=sFloodDir;
			EDir=sEbbDir;
			
			// OK now rotate vector so we have velocity
			// in east-west, north-south coordinate system
			
			argu = 90.0 - FDir;
			rotateVector( argu,u,v,&xVel,&yVel);
			
			*uVelocity = xVel;
			*vVelocity = yVel;
			
			u = xVel;
			v = yVel;
	}
	
	// OK if we already have east and north station, problem done
	else {
		*uVelocity = u;
		*vVelocity = v;
	}
	
	// now figure out the sign of value
	// if ebb, then sign will be negative
	// if flood, then positive
	// We first check dot product
	// if it is unambiguous then we go with flood or ebb assignment
	// if not sure we go through the hoops since we gotta assume
	// something ....
	// Note if ambiguous, we will flag as such so we don't plot the
	// hummer.
	
	directionFlag = GetVelDir(constituent,u,v);
	*direcKey = directionFlag;
	if(directionFlag==33){
		return -99.0;
	}
	
	// OK we are going to cross check
	// if we flag projection on flood and major axis
	// then we return flood
	// if we flag projection on ebb and negative on
	// major axis, we return ebb
	// otherwise, indeterminate
	
	if(directionFlag==1){
		if(*vmajor>=0.0){
			current = absoluteCurrentValue;
			return current;
		}
		else {
			directionFlag = Undetermined;
		}
	}
	else if(directionFlag==-1){
		if(*vmajor<=0.0){
			current = -absoluteCurrentValue;
			return current;
		}
		else {
			directionFlag = Undetermined;
		}
	}
	
	// OK now for data points where we don't know if it's flood or ebb
	// we gotta do something so we can guestimate where the min before
	// flood and min before ebbs are so offset stations can be used
	
	//
	
	if(directionFlag == Undetermined){
	
		// Let's check for going through a minimum value
		// If we do, then we have a slope change and we
		// will flip the sign
		
		if(hardWayFlag==0){
			current = absoluteCurrentValue * sign;
			if(lastSlope<=0.0){
				newSlope = absoluteCurrentValue - absoluteLastValue;
				if(newSlope>=0.0){
					current = -current;
					return current;
				}
			}
			return current;
		}
		
		// now what happens if we don't have last value and slope yet
		
		// OK if we have major-minor station, we key on
		// sign of major axis component
		if(RotFlag==MajorMinorKey){
			if(*vmajor >= 0.0){
				current = absoluteCurrentValue;
			}
			else {
				current = -absoluteCurrentValue;
			}
			return current;
		}
		
		// If not major - minor, we compute angle of velocity vector
		// and check if it is closer to the ebb direction or the flood
		// direction
		
		errorFlag = GetCDirec(constituent,&sFloodDir,&sEbbDir);
		
		FDir = sFloodDir;
		EDir = sEbbDir;
		
		FDir = FDir * PI/180.0;
		
		if(u!=0.0){
			argu = atan(v/u);
			if( u < 0.0) argu = argu + PI;
		}
		else {
			argu = PI/2.0;
			if(v<0.0) argu = PI + argu;
		}
		argu = argu * 180.0/PI;
		if(argu<0.0)argu=360.0 + argu;
		
		// OK now we determine flood or ebb by comparing the angles
		
		floodAngle = (double)sFloodDir-argu;
		floodAngle = fabs(floodAngle);
		
		ebbAngle = (double)sEbbDir - argu;
		ebbAngle = fabs(ebbAngle);
		
		current = absoluteCurrentValue;
		if(ebbAngle<=floodAngle)current = -current;
	}
	
	return current;
}

void short2Str(short sValue, char *theStr)
{	
	sprintf(theStr,"%hd",sValue);
	return;
}


void hr2TStr(double exHours, char *theStr)
{	
// input an double of hours and return a string of hour:minute
    long      nchr,ndec;
	short     sHour,sMin;
	char str1[256],str2[256];;
/**************************************************************/
   sHour = exHours;
   if( (sHour<0)||(sHour>24) ){
   		ndec = 2;

		sprintf(theStr,"%.2lf",exHours);
//		nchr=val2st(exHours,ndec,theStr);
		return;
   }
   
   sMin = (exHours-sHour)*60.0;
   
   short2Str(sHour,str1);
   short2Str(sMin,str2);
   
   strcat(str1,":");
   if(sMin<10){
   	strcat(str1,"0");
   }
   strcat(str1,str2);
   
   strcpy(theStr,str1);
   return;
}


short CheckCurrentOffsets(CURRENTOFFSET *offset)
{
	short			err=0;
	if(!offset){
		err=32;
		goto Error;
	}
	
	
	/* The dataAvailFlag field is a short of value 0 or 1 (0==insufficient data) */

	if(!offset->MinBefFloodTime.dataAvailFlag){	/* Min Before Flood time offset	*/
		err=32;
		goto Error;
	}

	if(!offset->FloodTime.dataAvailFlag){			/* Flood time offset			*/
		err=32;
		goto Error;
	}

	if(!offset->MinBefEbbTime.dataAvailFlag){		/* Min Before Ebb time offset	*/
		err=32;
		goto Error;
	}

	if(!offset->EbbTime.dataAvailFlag){			/* Ebb time offset				*/
		err=32;
		goto Error;
	}

	if(!offset->FloodSpdRatio.dataAvailFlag){		/* Flood Speed Ratio			*/
		err=32;
		goto Error;
	}

	if(!offset->EbbSpdRatio.dataAvailFlag){		/* Ebb Speed Ratio				*/
		err=32;
		goto Error;
	}

Error:
	return(err);
}

// ************************************************************************

short slopeChange(double lastValue,double nowValue,double nextValue)
{
	// function checks for change in slope and returns
	//  0 if no slope change
	//  1 if change from negative to positive
	// -1 if change from positive to negative
	
	// no times are used because it is assumed the three values are
	// given sequentially.  lastValue is happens before nowValue which
	// happens before nextValue ... 
	
	double dv0,dv1;
	short flag;
	
	flag = 0;
	
	dv0 = nowValue - lastValue;
	dv1 = nextValue - nowValue;
	
	if(dv0<0.0){
		if(dv1>=0.0){
			flag = 1;
		}
	}
	else if(dv0>0.0){
		if(dv1<=0.0){
			flag = -1;
		}
	}
	else{  // here we start with zero slope
		if(dv1>0.0)flag =1;
		if(dv1<0.0)flag = -1;
	}
	return flag;
}

// ************************************************************

short zeroCross(double lastValue,double nowValue,double nextValue)
{
	// function checks for zero crossing and returns
	//  0 if no zero crossing
	//  1 if cross is from negative to positive
	// -1 if cross is from positive to negative
	
	// no times are used because it is assumed the three values are
	// given sequentially.  lastValue is happens before nowValue which
	// happens before nextValue ... 
	
	short flag;
	
	flag = 0;

	// check for odd case
	if(lastValue==nextValue){
		if(lastValue<0.0){
			if(nowValue>0.0){
				flag = 1;
			}
		}
		if(lastValue>0.0){
			if(nowValue<0.0){
				flag = -1;
			}
		}
		else {
			if(nowValue>0.0) flag =1;
			if(nowValue<0.0)flag = -1;
		}
	}
	
	// Now normal case
	
	if(lastValue>0.0){
		if(nextValue<=0.0){
			flag = -1;
		}
	}
	if(lastValue<0.0){
		if(nextValue>=0.0){
			flag = 1;
		}
	}
	return flag;
}
