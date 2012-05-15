/*
 *  Model_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 1/6/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "Model_c.h"
#include "MemUtils.h"
#include "StringFunctions.h"
#include "CompFunctions.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

Boolean gNoaaVersion = FALSE;
Boolean gMearnsVersion = FALSE;
Boolean gDispersedOilVersion = FALSE;


Model_c::Model_c(Seconds start)
{
	stepsCount = 0;
	ncSnapshot = false;
	writeNC = false;
	fDrawMovement = 0;//JLM
	fWizard = nil;
	fSquirreledLastComputeTimeLEList = nil;
	LESetsList = nil;
	mapList = nil;
	fOverlayList = nil;
	uMap = nil;
	weatherList = nil;
	LEFramesList = nil;
	frameMapList = nil;
	movieFrameIndex = 0;
	modelMode = ADVANCEDMODE;
#ifndef pyGNOME	
//	fDialogVariables = DefaultTModelDialogVariables(start);
#endif
	strcpy (fSaveFileName, kDefSaveFileName);		// sohail
	fOutputFileName[0] = 0;
	fOutputTimeStep = 3600;
	fWantOutput = FALSE;
	
	modelTime = fDialogVariables.startTime;
	lastComputeTime = fDialogVariables.startTime;
	
	bSaveRunBarLEs = true;
	LEDumpInterval = 3600;	// dump interval for LE's used for run-bar
	
	ResetMainKey();
#ifndef pyGNOME
//	SetDirty(FALSE);
#endif	
	fSettingsOpen = TRUE;
	fSpillsOpen = TRUE;
	bMassBalanceTotalsOpen = false;
	mapsOpen = TRUE;
	fOverlaysOpen = TRUE;
	uMoverOpen = TRUE;
	weatheringOpen = TRUE;
	
	fMaxDuration = 3.*24;	// 3 days
	
	// JLM found this comment but no does not believe it, 11/15/99
	// IT MUST ALWAYS START OUT TRUE TO ENSURE 
	// THAT LE UNCERTAINTY ARRAYS GET INITIALIZED
	// bLEsDirty = true;  
	bLEsDirty = false; 
	
	fRunning = FALSE;
	bMakeMovie = FALSE;
	bHindcast = FALSE;
	
	bSaveSnapshots = false;
	fTimeOffsetForSnapshots = 0;	
	fSnapShotFileName[0] = 0;
}

void	Model_c::SetModelTime (Seconds newModelTime) 
{ 
	//if(newModelTime > this->GetEndTime())
	//	newModelTime = this->GetEndTime();
	//	
	//if(newModelTime < this->GetStartTime())
	//	newModelTime = this->GetStartTime();	
	//hmmm.. this would mean we had to be careful which order we set these in
	
	modelTime = newModelTime; 
}

void Model_c::ResetMainKey()
{
	nextKeyValue = 1;
}

long Model_c::GetNextLEKey()
{
	return nextKeyValue++;
}


WorldRect Model_c::GetMapBounds(void)
{ // the bound of all maps
	WorldRect boundingRect;
	TMap *map;
	long n;
	
	boundingRect = voidWorldRect;
	for (n = mapList->GetItemCount() - 1; n >= 0 ; n--) {
		mapList->GetListItem((Ptr)&map, n);
		boundingRect = UnionWRect(boundingRect,map->GetMapBounds());
	}
	return boundingRect;
}

TMap *Model_c::GetBestMap(WorldPoint p)
{
	long i, n;
	TMap *map;
	
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&map, i);
		if (map->InMap(p))
			return map;
	}
	
	return 0;
}


void Model_c::UpdateWindage(TLEList* theLEList)
{
	long i, numLEs;
	LERec theLE;
	double originalWindageRange, currentWindageRange, meanWindage, persistence;
	double windage, windageA, windageB, currentWindageA, currentWindageB;
	WindageRec windageRec = (*(dynamic_cast<TOLEList*>(theLEList))).fWindageData;
	Boolean timeAfterSpillStart;
	
	if (!bHindcast)
		//timeAfterModelStart = !(modelTime == model->GetStartTime());
		timeAfterSpillStart = (modelTime > (dynamic_cast<TOLEList*>(theLEList))->GetSpillStartTime());
	else
		//timeAfterModelStart = !(modelTime == model->GetEndTime());
		//timeAfterSpillStart = (modelTime < ((TOLEList*)theLEList)->GetSpillStartTime());
		timeAfterSpillStart = (modelTime < (dynamic_cast<TOLEList*>(theLEList))->GetSpillEndTime());
	
	
	// new algorithm to eliminate dependence of spread on time step 12/22/00
	numLEs = theLEList->GetLECount();
	windageA = windageRec.windageA;
	windageB = windageRec.windageB;
	persistence = windageRec.persistence;
	//if (persistence==-1 && !(modelTime == model->GetStartTime())) return;	// infinite persistence already set
	if (persistence==-1 && timeAfterSpillStart) return;	// infinite persistence already set
	if (persistence != -1)
	{
		originalWindageRange = windageB - windageA;
		currentWindageRange = originalWindageRange * sqrt(persistence * 3600 / (double)GetTimeStep());
		meanWindage = (windageA + windageB) / 2.;
		currentWindageA = meanWindage - currentWindageRange / 2.;
		currentWindageB = meanWindage + currentWindageRange / 2.;
	}
	for (i = 0; i<numLEs; i++) 
	{
		theLEList -> GetLE (i, &theLE);
		if (persistence == -1)	// infinite persistence
			theLE.windage = GetRandomFloat(windageA, windageB);
		else	// standard persistence
			theLE.windage = GetRandomFloat(currentWindageA, currentWindageB);
		theLEList -> SetLE (i, &theLE);
	}
}

OSErr GetAdiosIndices(AdiosInfoRecH adiosBudgetTable, Seconds time, long *startIndex, long *endIndex)
{
	long i,numBudgetTableItems;
	*startIndex = 0;
	*endIndex = 0;
	if (!adiosBudgetTable) return -1;
	numBudgetTableItems = _GetHandleSize((Handle)adiosBudgetTable)/sizeof(**adiosBudgetTable);
	if (time > INDEXH(adiosBudgetTable,numBudgetTableItems-1).timeAfterSpill)
		return -2;
	for (i=1;i<numBudgetTableItems;i++)
	{
		if (time<=INDEXH(adiosBudgetTable,i).timeAfterSpill)
		{
			*startIndex = i-1;
			*endIndex = i;
			return noErr;
		}
	}
	return noErr;	// time before first adios budget table value - should do something here?
}

void Model_c::DisperseOil(TLEList* theLEList, long index)
{
	long i, numLEs, startIndex=0, endIndex=0, startIndexEvap, endIndexEvap, startIndexNat, endIndexNat, startIndexRem, endIndexRem;
	long disperseStep, totalSteps, numBudgetTableItems, numDropletSizes = 0;
	float x,y,rand,rand2;
	LERec theLE,savedLE;
	DispersionRec dispInfo = (dynamic_cast<TOLEList*>(theLEList)) -> GetDispersionInfo();
	AdiosInfoRecH adiosBudgetTable = (dynamic_cast<TOLEList *>(theLEList)) -> GetAdiosInfo();
	Boolean chemicalDispersion = dispInfo.bDisperseOil, naturalDispersion = false;
	DropletInfoRecH dropSizeHdl = 0;
	
	//if we read in Adios Table will have to do evaporation too, status = oilstat_evaporated
	//and make sure Gnome doesn't - maybe set oil type to conservative??
	//or check if AdiosDataH exists
	if (adiosBudgetTable) naturalDispersion = true;
	// special case if both are set, need to account for large reduction in number of LEs after
	// chemical dispersion
	theLEList -> GetLE (index, &theLE);
	savedLE = theLE;
	if (naturalDispersion)
	{
		OSErr err = 0;
		long adiosStartIndex, adiosEndIndex;
		float totalAmountToDisperse,totalAmountToEvaporate,totalAmountToRemove;
		float startAmtDisp, endAmtDisp, startAmtEvap, endAmtEvap, startAmtRem, endAmtRem, frac=0;
		Seconds currentTime, previousTime=0, adiosIntervalStartTime, adiosIntervalEndTime, duration;
		numBudgetTableItems = (dynamic_cast<TOLEList*>(theLEList)) -> GetNumAdiosBudgetTableItems();
		if (numBudgetTableItems<1) 
		{
			printError("Problem accessing Adios Budget Table data"); 
			return;
		}
		totalAmountToDisperse = INDEXH(adiosBudgetTable,numBudgetTableItems-1).amountDispersed;
		totalAmountToEvaporate = INDEXH(adiosBudgetTable,numBudgetTableItems-1).amountEvaporated;
		totalAmountToRemove = INDEXH(adiosBudgetTable,numBudgetTableItems-1).amountRemoved;
		currentTime = GetModelTime() - (dynamic_cast<TOLEList*>(theLEList)) ->fSetSummary.startRelTime;
		//currentTime = GetModelTime() - GetStartTime();
		if (currentTime>0) previousTime = currentTime - GetTimeStep();
		err = GetAdiosIndices(adiosBudgetTable,currentTime,&adiosStartIndex,&adiosEndIndex);
		//if (err == -2)	//after table ends, may have chemical dispersion, let it go
		//return;// this should be after table ends, assume chemical would be before...
		if (err == -1)	// shouldn't happen
		{
			printError("Problem accessing Adios Budget Table data"); 
			return;
		}
		// maybe static the previous endIndex and use as next startIndex..., reset each time a new run is done
		startAmtDisp = INDEXH(adiosBudgetTable,adiosStartIndex).amountDispersed;
		endAmtDisp = INDEXH(adiosBudgetTable,adiosEndIndex).amountDispersed;
		startAmtEvap = INDEXH(adiosBudgetTable,adiosStartIndex).amountEvaporated;
		endAmtEvap = INDEXH(adiosBudgetTable,adiosEndIndex).amountEvaporated;
		startAmtRem = INDEXH(adiosBudgetTable,adiosStartIndex).amountRemoved;
		endAmtRem = INDEXH(adiosBudgetTable,adiosEndIndex).amountRemoved;
		adiosIntervalStartTime = INDEXH(adiosBudgetTable,adiosStartIndex).timeAfterSpill;
		adiosIntervalEndTime = INDEXH(adiosBudgetTable,adiosEndIndex).timeAfterSpill;
		// will need to notice if no change from previous time step
		if (adiosIntervalEndTime!=adiosIntervalStartTime)
			frac = (float)(currentTime - adiosIntervalStartTime)/(float)(adiosIntervalEndTime-adiosIntervalStartTime); // how far into the interval
		// also want % of total to determine how many steps
		duration = INDEXH(adiosBudgetTable,numBudgetTableItems-1).timeAfterSpill;
		disperseStep = (currentTime + GetTimeStep())/GetTimeStep(); //releaseTime
		//disperseStep = (currentTime + GetTimeStep())/GetTimeStep(); //releaseTime
		// for endIndex figure out how many LEs total should be dispersed at this time
		totalSteps = duration/GetTimeStep() + 1;
		//theLEList -> GetLE (index, &theLE);
		//savedLE = theLE;
		numLEs = theLEList->GetLECount();
		
		// also need to pay attention if time dependent release, then total available LEs changes
		//if ((TOLEList*)theLEList) ->fSetSummary.bWantEndRelTime && (TOLEList*)theLEList) ->numOfLEs > 1)
		//timeStep = (fSetSummary.endRelTime  - fSetSummary.startRelTime) / (double) (fSetSummary.numOfLEs - 1);
		//representativeLE.releaseTime = fSetSummary.startRelTime + i * timeStep;
		
		// may want to make this more random, rather than marking in order, could tie into dispersant step
		//startIndexNat = numLEs * (startAmtDisp + frac * (endAmtDisp - startAmtDisp)) / totalAmountToDisperse;
		//startIndexEvap = numLEs * (startAmtEvap + frac * (endAmtEvap - startAmtEvap)) / totalAmountToEvaporate;
		endIndexNat = numLEs * (startAmtDisp + frac * (endAmtDisp - startAmtDisp)) / totalAmountToDisperse;
		endIndexEvap = numLEs * (startAmtEvap + frac * (endAmtEvap - startAmtEvap)) / totalAmountToEvaporate;
		endIndexRem = numLEs * (startAmtRem + frac * (endAmtRem - startAmtRem)) / totalAmountToRemove;
		//if (disperseStep==1 || (((TOLEList*)theLEList) ->fSetSummary.startRelTime != ((TOLEList*)theLEList) ->fSetSummary.endRelTime && GetModelTime() - theLE.releaseTime) <= GetTimeStep())	// time dependent release
		//if (disperseStep==1 ||  (GetModelTime() - theLE.releaseTime) < GetTimeStep())	// time dependent release
		if (disperseStep==1 ||  ((GetModelTime() - theLE.releaseTime) < GetTimeStep() && theLE.releaseTime > (dynamic_cast<TOLEList*>(theLEList)) ->fSetSummary.startRelTime))	// time dependent release
		{
			// mark the LEs that will be dispersed
			// what if combine lasso with adios??
			//if (dispInfo.lassoSelectedLEsToDisperse && dispInfo.timeToDisperse == 0 && theLE.dispersionStatus)
			// need to calculate based on time lasso was applied, it would have superceded any evaporate or disperse setting
			// probably want to use originally set values, except if lassoed before first time step. THen what?
			// then don't mark any already marked LEs and proceed 
			//if (!dispInfo.lassoSelectedLEsToDisperse || (dispInfo.lassoSelectedLEsToDisperse && dispInfo.timeToDisperse == 0 && theLE.dispersionStatus != DISPERSE))
			//{
			//y = GetRandomFloat(0, 1.0);
			x = GetRandomFloat(0, 1.0);
			//if(x > 1-totalAmountToDisperse) // percent within the area or of total amount of oil?
			if(x <= totalAmountToDisperse) // percent within the area or of total amount of oil?
			{
				theLE.dispersionStatus = DISPERSE_NAT;
			}
			// total amount to evaporate should be scaled by 1/(1-totalAmtDispersed)
			// to make up for the dispersed LEs bringing down the total available
			//else if(y <= totalAmountToEvaporate / (1-totalAmountToDisperse))
			//else if(x < 1-totalAmountToDisperse && x > 1-(totalAmountToDisperse + totalAmountToEvaporate) )
			else if(x > totalAmountToDisperse && x <= totalAmountToDisperse + totalAmountToEvaporate )
			{
				theLE.dispersionStatus = EVAPORATE;
			}
			//else if(x < 1-(totalAmountToDisperse + totalAmountToEvaporate) && x > 1 - (totalAmountToDisperse + totalAmountToEvaporate + totalAmountToRemove) )
			else if(x > totalAmountToDisperse + totalAmountToEvaporate && x <= totalAmountToDisperse + totalAmountToEvaporate + totalAmountToRemove )
			{
				theLE.dispersionStatus = REMOVE;
			}
			//else
			//theLE.dispersionStatus = DONT_DISPERSE;
			//}
			if (chemicalDispersion && dispInfo.lassoSelectedLEsToDisperse && savedLE.dispersionStatus == DISPERSE)
				theLE.dispersionStatus = savedLE.dispersionStatus;
			
		} 
		if (!chemicalDispersion) goto weatherLE;
	}
	if (!chemicalDispersion) return;
	if (this->GetModelTime() - (dynamic_cast<TOLEList*>(theLEList)) ->fSetSummary.startRelTime < dispInfo.timeToDisperse) goto weatherLE;
	//if (this->GetModelTime() - this->GetStartTime() < dispInfo.timeToDisperse) goto weatherLE;
	//if (this->GetModelTime() - this->GetStartTime() < dispInfo.timeToDisperse && !naturalDispersion) return;
	// code goes here, check wind speed >= 7 knots, can assume only one wind mover? only at first step?
	
	//theLEList -> GetLE (index, &theLE);
	//if (dispInfo.lassoSelectedLEsToDisperse && thisLE.beachTime >= GetModelTime() && thisLE.beachTime <= GetModelTime() + GetTimeStep()) timeToDisperse = true;
	if (dispInfo.lassoSelectedLEsToDisperse && theLE.beachTime >= GetStartTime() )
	{
		if (theLE.beachTime >= GetModelTime() && theLE.beachTime <= GetModelTime() + GetTimeStep())
			disperseStep = (GetModelTime() - theLE.beachTime + GetTimeStep())/GetTimeStep();
		else return;
	}
	//disperseStep = (GetModelTime() - GetStartTime() - dispInfo.timeToDisperse + GetTimeStep())/GetTimeStep();
	else
		disperseStep = (GetModelTime() - (dynamic_cast<TOLEList*>(theLEList)) ->fSetSummary.startRelTime - dispInfo.timeToDisperse + GetTimeStep())/GetTimeStep();
	totalSteps = dispInfo.duration/GetTimeStep() + 1;
	numLEs = theLEList->GetLECount();
	startIndex = numLEs * (disperseStep - 1) / totalSteps;
	endIndex = numLEs * disperseStep / totalSteps;
	if (WPointInWRect(theLE.p.pLong,theLE.p.pLat,&dispInfo.areaToDisperse))
	{
		//if (disperseStep==1 && !dispInfo.lassoSelectedLEsToDisperse) // if lasso selected, already set
		if ((disperseStep==1 || (GetModelTime() - theLE.releaseTime) < GetTimeStep()) && !dispInfo.lassoSelectedLEsToDisperse) // if lasso selected, already set
		{
			// mark the LEs that will be dispersed
			x = GetRandomFloat(0, 1.0);
			if(x <= dispInfo.amountToDisperse) // percent within the area or of total amount of oil?
			{
				theLE.dispersionStatus = DISPERSE;
			}
			//else
			//theLE.dispersionStatus = DONT_DISPERSE;
		}
		if (disperseStep==1 && dispInfo.lassoSelectedLEsToDisperse && !naturalDispersion)
		{
			if (theLE.dispersionStatus != DISPERSE) theLE.dispersionStatus=DONT_DISPERSE;
		}
	}
	
weatherLE:
	if (theLE.dispersionStatus == EVAPORATE && (/*index>=startIndex &&*/ index<endIndexEvap))
		//if (theLE.dispersionStatus == EVAPORATE && (index%totalSteps == disperseStep))
	{	
		theLE.statusCode = OILSTAT_EVAPORATED;
		theLE.dispersionStatus = HAVE_EVAPORATED;
	}
	if (theLE.dispersionStatus == REMOVE && (/*index>=startIndex &&*/ index<endIndexRem))
		//if (theLE.dispersionStatus == EVAPORATE && (index%totalSteps == disperseStep))
	{	
		theLE.statusCode = OILSTAT_OFFMAPS;
		theLE.dispersionStatus = HAVE_REMOVED;
	}
	if (theLE.dispersionStatus == DISPERSE && (/*index>=startIndex &&*/ index<endIndex)
		|| (theLE.dispersionStatus == DISPERSE_NAT && index<endIndexNat))
	{
		// disperse percent of LEs each time, each time depth range is the same
		PtCurMap *map = GetPtCurMap();
		if (!map) {printError("Programmer error - TModel::DisperseOil()");return;}
		//double breakingWaveHeight = map->fBreakingWaveHeight;
		double breakingWaveHeight = map->GetBreakingWaveHeight();
		if (breakingWaveHeight == 0) printNote("Oil cannot be dispersed because there is no wind");
		double depthAtPoint = map->DepthAtPoint(theLE.p);	// or check rand instead
		if (depthAtPoint >= breakingWaveHeight * 1.5 || depthAtPoint <= 0)		
			rand = GetRandomFloat(1e-6, breakingWaveHeight*1.5);
		else
			rand = GetRandomFloat(1e-6, depthAtPoint);
		theLE.z = rand; 	// check if in vertical map
		rand2 = GetRandomFloat(0,1);
		/*if (rand2>=0 && rand2<.056)
		 theLE.dropletSize = 10;
		 else if (rand2>=.056 && rand2<.147)
		 theLE.dropletSize = 20;
		 else if (rand2>=.147 && rand2<.267)
		 theLE.dropletSize = 30;
		 else if (rand2>=.267 && rand2<.414)
		 theLE.dropletSize = 40;
		 else if (rand2>=.414 && rand2<.586)
		 theLE.dropletSize = 50;
		 else if (rand2>=.586 && rand2<.782)
		 theLE.dropletSize = 60;
		 else if (rand2>=.782 && rand2<=1.0)
		 theLE.dropletSize = 70;*/
		
		dropSizeHdl = map->GetDropletSizesH();
		if (dropSizeHdl) numDropletSizes = _GetHandleSize((Handle)dropSizeHdl)/sizeof(**dropSizeHdl);
		for (i=0;i<numDropletSizes;i++)
		{
			if (rand2 < (*dropSizeHdl)[i].probability) theLE.dropletSize = (*dropSizeHdl)[i].dropletSize;
			else
			{
				theLE.dropletSize = (*dropSizeHdl)[numDropletSizes-1].dropletSize;
			}
		}
		if (theLE.dispersionStatus == DISPERSE_NAT)
			theLE.dispersionStatus = HAVE_DISPERSED_NAT;
		else
			theLE.dispersionStatus = HAVE_DISPERSED;
		//theLE.dispersionStatus = HAVE_DISPERSED;
	}
	
	theLEList -> SetLE (index, &theLE);
	return;
}

void Model_c::ReDisperseOil(LERec* thisLE, double breakingWaveHeight)
{
	long i,numDropletSizes=0;
	float rand,rand2;
	LERec theLE = (*thisLE);
	DropletInfoRecH dropSizeHdl = 0;
	//DispersionRec dispInfo = ((TOLEList*)theLEList) -> GetDispersionInfo();
	
	//if (!dispInfo.bDisperseOil) return;	// for now using redisperse oil to handle chemicals at bottom that surface
	
	PtCurMap *map = GetPtCurMap();
	if (!map) {printError("Programmer error - TModel::ReDisperseOil()");return;}
	//double breakingWaveHeight = map->fBreakingWaveHeight;
	double depthAtPoint = map->DepthAtPoint(theLE.p);	// or check rand instead
	if (depthAtPoint <= 0)
	{
		OSErr err = 0;
	}
	if (breakingWaveHeight == 0) printNote("Oil cannot be redispersed because there is no wind");
	if (depthAtPoint >= breakingWaveHeight * 1.5 || depthAtPoint <= 0)
		rand = GetRandomFloat(1e-6, breakingWaveHeight*1.5);
	else
		rand = GetRandomFloat(1e-6, depthAtPoint);
	theLE.z = rand; 
	rand2 = GetRandomFloat(0,1);
	/*if (rand2>=0 && rand2<.056)
	 theLE.dropletSize = 10;
	 else if (rand2>=.056 && rand2<.147)
	 theLE.dropletSize = 20;
	 else if (rand2>=.147 && rand2<.267)
	 theLE.dropletSize = 30;
	 else if (rand2>=.267 && rand2<.414)
	 theLE.dropletSize = 40;
	 else if (rand2>=.414 && rand2<.586)
	 theLE.dropletSize = 50;
	 else if (rand2>=.586 && rand2<.782)
	 theLE.dropletSize = 60;
	 else if (rand2>=.782 && rand2<=1.0)
	 theLE.dropletSize = 70;*/
	
	dropSizeHdl = map->GetDropletSizesH();
	if (dropSizeHdl) numDropletSizes = _GetHandleSize((Handle)dropSizeHdl)/sizeof(**dropSizeHdl);
	for (i=0;i<numDropletSizes;i++)
	{
		if (rand2 < (*dropSizeHdl)[i].probability) theLE.dropletSize = (*dropSizeHdl)[i].dropletSize;
		else
		{
			theLE.dropletSize = (*dropSizeHdl)[numDropletSizes-1].dropletSize;
		}
	}
	// never changed the status, just immediately redispersed
	/*if (theLE.dispersionStatus == DISPERSE_NAT)
	 theLE.dispersionStatus = HAVE_DISPERSED_NAT;
	 else
	 theLE.dispersionStatus = HAVE_DISPERSED;*/
	
	//theLEList -> SetLE (index, &theLE);
	(*thisLE) = theLE;
	return;
}



TLEList* Model_c::GetMirroredLEList(TLEList* owner)
{
	TLEList* thisLEList;
	long i,n;
	if(!owner) return nil;
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		LESetsList->GetListItem((Ptr)&thisLEList, i);
		if(thisLEList)
		{
			if(owner->MatchesUniqueID(thisLEList->fOwnersUniqueID)) return thisLEList;
		}
	}
	return nil;
}

Boolean Model_c::IsWaterPoint(WorldPoint p)
{
	TMap *bestMap = GetBestMap (p);
	if (!bestMap) return false; // OFFMAPS;
	if (bestMap -> OnLand (p)) return false; //  ONLAND
	return true; // INWATER
}

Boolean Model_c::IsAllowableSpillPoint(WorldPoint p)
{
	TMap *bestMap = GetBestMap (p);
	if (!bestMap) return false; // OFFMAPS;
	return bestMap -> IsAllowableSpillPoint (p);
}

Boolean Model_c::HaveAllowableSpillLayer(WorldPoint p)
{
	TMap *bestMap = GetBestMap (p);
	if (!bestMap) return false; // OFFMAPS;
	return bestMap -> HaveAllowableSpillLayer ();
}

TMap* Model_c::GetMap(char* mapName)
{
	// loop through each mover in the universal map
	TMap *map;
	char thisName[kMaxNameLen];
	long i,n;
	
	// universal map
	if(!strcmpnocase("UMAP",mapName)) return this->uMap; // special case
	map =  this->uMap;
	map -> GetClassName (thisName);
	if(!strcmpnocase(thisName,mapName)) return map;
	
	// other maps
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&map, i);
		map -> GetClassName (thisName);
		if(!strcmpnocase(thisName,mapName)) return map;
	}
	
	return nil;
}

/////////////////////////////////////////////////
TMap* Model_c::GetMap(ClassID desiredClassID)
{
	long i,n;
	TMap *map;
	n = model -> mapList->GetItemCount() ;
	for (i=0; i<n; i++)
	{
		model -> mapList->GetListItem((Ptr)&map, i);
		if(map -> IAm(desiredClassID)) return map;
	}
	return nil;
}

TMover* Model_c::GetMover(char* moverName)
{
	// loop through each mover in the universal map
	TMover *thisMover = nil;
	TMap *map;
	char thisName[kMaxNameLen];
	long i,n,k,d;
	
	// universal movers
	for (k = 0, d = this->uMap->moverList->GetItemCount (); k < d; k++)
	{
		this->uMap->moverList -> GetListItem ((Ptr) &thisMover, k);
		thisMover -> GetClassName (thisName);
		if(!strcmpnocase(thisName,moverName)) return thisMover;
	}
	
	// movers that belong to a map
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&map, i);
		for (k = 0, d = map -> moverList -> GetItemCount (); k < d; k++)
		{
			map -> moverList -> GetListItem ((Ptr) &thisMover, k);
			thisMover -> GetClassName (thisName);
			if(!strcmpnocase(thisName,moverName)) return thisMover;
		}
	}
	
	return nil;
}

TMover* Model_c::GetMover(ClassID desiredClassID)
{
	// loop through each mover in the universal map
	TMover *thisMover = nil;
	TMap *map;
	long i,n,k,d;
	//ClassID classID;
	
	// universal movers
	for (k = 0, d = this->uMap->moverList->GetItemCount (); k < d; k++)
	{
		this->uMap->moverList -> GetListItem ((Ptr) &thisMover, k);
		//classID = thisMover -> GetClassID ();
		//if(classID == desiredClassID) return thisMover;
		if(thisMover -> IAm(desiredClassID)) return thisMover;
	}
	
	// movers that belong to a map
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&map, i);
		for (k = 0, d = map -> moverList -> GetItemCount (); k < d; k++)
		{
			map -> moverList -> GetListItem ((Ptr) &thisMover, k);
			//classID = thisMover -> GetClassID ();
			//if(classID == desiredClassID) return thisMover;
			if(thisMover -> IAm(desiredClassID)) return thisMover;
		}
	}
	
	return nil;
}

TCurrentMover* Model_c::GetPossible3DCurrentMover()
{// this is for CDOG and really only NetCDF files are allowed, at this point triangular grids
	TMover *thisMover = nil;
	TMap *map;
	long i,n,k,d;
	// movers that belong to a map - can't be a universal mover, always put on a map
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&map, i);
		for (k = 0, d = map -> moverList -> GetItemCount (); k < d; k++)
		{
			map -> moverList -> GetListItem ((Ptr) &thisMover, k);
			//classID = thisMover -> GetClassID ();
			//if(classID == desiredClassID) return thisMover;
			//if(thisMover -> IAm(desiredClassID)) return thisMover;
			// code goes here, check if mover is 3D
			if(thisMover -> IAm(TYPE_PTCURMOVER) || thisMover -> IAm(TYPE_TRICURMOVER) || thisMover -> IAm(TYPE_NETCDFMOVER)
			   || thisMover -> IAm(TYPE_NETCDFMOVERCURV) || thisMover -> IAm(TYPE_NETCDFMOVERTRI)) return dynamic_cast<TCurrentMover*>(thisMover);
		}
	}
	
	/*for (i = 0, d = this -> moverList -> GetItemCount (); i < d; i++)
	 {
	 this -> moverList -> GetListItem ((Ptr) &thisMover, i);
	 //classID = thisMover -> GetClassID ();
	 //if(classID == desiredClassID) return thisMover(;
	 //if (thisMover -> IAm(TYPE_CURRENTMOVER)) return ((TCurrentMover*)thisMover);	// show movement only handles currents, not wind and dispersion
	 // might want to be specific since this could allow CATSMovers...
	 if(thisMover -> IAm(TYPE_PTCURMOVER) || thisMover -> IAm(TYPE_TRICURMOVER) || thisMover -> IAm(TYPE_NETCDFMOVER)
	 || thisMover -> IAm(TYPE_NETCDFMOVERCURV) || thisMover -> IAm(TYPE_NETCDFMOVERTRI)) return (TCurrentMover*)thisMover;
	 }*/
	return nil;
}


TLEList* Model_c::GetLEListOwner(TLEList* mirroredLEList)
{
	TLEList* thisLEList;
	long i,n;
	if(!mirroredLEList) return nil;
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		LESetsList->GetListItem((Ptr)&thisLEList, i);
		if(thisLEList)
		{
			if(thisLEList->MatchesUniqueID(mirroredLEList->fOwnersUniqueID)) return thisLEList;
		}
	}
	return nil;
}

TWindMover* Model_c::GetWindMover(Boolean createIfDoesNotExist)
{	
	char moverName[kMaxNameLen];
	TWindMover *mover = nil;
	// for now don't get the spatially varying wind movers 2/27/03
	//mover = (TWindMover*)this->GetMover(TYPE_WINDMOVER);
	mover = dynamic_cast<TWindMover *>(this->GetMover("Variable Wind"));
	if (!mover) mover = dynamic_cast<TWindMover *>(this->GetMover("Constant Wind"));
	if(!mover && createIfDoesNotExist)
	{ // create one and add it to the universal movers
		TOSSMTimeValue *timeFile = nil;
		mover = new TWindMover(this->uMap, "");
		if (!mover) { TechError("GetWindMover()", "new TWindMover()", 0); return nil; }
		timeFile = new TOSSMTimeValue (mover);
		if (!timeFile) { TechError("GetWindMover()", "new TOSSMTimeValue()", 0); delete mover; return nil; }
		mover->SetTimeDep(timeFile);
		mover->SetActive(false);
		this->uMap->AddMover(mover, 0);
	}
	return mover;
}

TRandom* Model_c::GetDiffusionMover(Boolean createIfDoesNotExist)
{	
	char moverName[kMaxNameLen];
	TRandom *mover = nil;
	strcpy(moverName,"Diffusion"); // not shown to user right ?
	mover = dynamic_cast<TRandom *>(this->GetMover(moverName));
	if(!mover && createIfDoesNotExist)
	{ // create one and add it to the universal movers
		//mover = mover = new TRandom(this->uMap, moverName);
		mover = new TRandom(this->uMap, moverName);
		if (!mover) { TechError("GetDiffusionMover()", "new TRandom()", 0); return nil; }
		this->uMap->AddMover(mover, 0);
	}
	return mover;
}

TRandom3D* Model_c::Get3DDiffusionMover()
{	
	char moverName[kMaxNameLen];
	TRandom3D *mover = nil;
	strcpy(moverName,"3D Diffusion"); // not shown to user right ?
	mover = dynamic_cast<TRandom3D *>(this->GetMover(moverName));
	return mover;
}	


long Model_c::NumLEs(LETYPE  leType)
{	
	long i,n,numLEs = 0;
	TLEList *list;
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) {
		LESetsList->GetListItem((Ptr)&list, i);
		if (list -> GetLEType () == leType) 	
			numLEs += list->numOfLEs;
	}
	return numLEs;
}

long Model_c::NumOutputSteps(Seconds outputTimeStep)
{
	long numSteps =0;
	short oldIndex,nowIndex;
	Seconds newModelTime = GetStartTime ();
	Seconds oldTime;
	Seconds stopTime = GetEndTime();
	Seconds stepTime = GetTimeStep();
	
	
	// we will write at the LEs at time 0
	numSteps++;
	
	oldTime = newModelTime;
	while (newModelTime < stopTime) 
	{
		// Note Output is called AFTER being incremented 
		newModelTime += stepTime;
		
		nowIndex = (newModelTime - fDialogVariables.startTime) / outputTimeStep;
		oldIndex = (oldTime - fDialogVariables.startTime)   / outputTimeStep;
		if(nowIndex > oldIndex)
		{
			numSteps++;
		}
		oldTime = newModelTime;
	}
	
	return numSteps;
}

OSErr Model_c::TellMoversPrepareForStep()
{
	long i,j,k, d,n;
	TMap *map;
	TMover *thisMover;
	OSErr err = 0;
	// loop through all maps except universal map
	if (!mapList) return -3;	// special error for a hard exit
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&map, i);
		for (k = 0, d = map -> moverList -> GetItemCount (); k < d; k++)
		{
			map -> moverList -> GetListItem ((Ptr) &thisMover, k);
			if (err = thisMover->PrepareForModelStep()) return err;
		}
	}
	
	// loop through each mover in the universal map
	for (k = 0, d = uMap -> moverList -> GetItemCount (); k < d; k++)
	{
		uMap -> moverList -> GetListItem ((Ptr) &thisMover, k);
		if (err = thisMover->PrepareForModelStep()) return err;
	}
	return err;
}

void Model_c::TellMoversStepIsDone()
{
	long i,j,k, d,n;
	TMap *map;
	TMover *thisMover;
	// loop through all maps except universal map
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&map, i);
		for (k = 0, d = map -> moverList -> GetItemCount (); k < d; k++)
		{
			map -> moverList -> GetListItem ((Ptr) &thisMover, k);
			thisMover->ModelStepIsDone();
		}
	}
	
	// loop through each mover in the universal map
	for (k = 0, d = uMap -> moverList -> GetItemCount (); k < d; k++)
	{
		uMap -> moverList -> GetListItem ((Ptr) &thisMover, k);
		thisMover->ModelStepIsDone();
	}
}

Boolean Model_c::ThereIsAnEarlierSpill(Seconds timeOfInterest, TLEList *someLEListToIgnore)
{
	TLEList *thisLEList;
	long i,n;
	LETYPE leType;
	OSErr err = 0;
	for (i = 0, n = LESetsList->GetItemCount() ; i < n && !err; i++) 
	{
		LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE) continue;
		if(thisLEList == someLEListToIgnore) continue;
		if(thisLEList -> IAm(TYPE_OSSMLELIST))
		{
			TOLEList *thisOLEList = dynamic_cast<TOLEList*>(thisLEList); // typecast
			if(thisOLEList->fSetSummary.startRelTime < timeOfInterest)
				return true;
		}
	}
	return false;
}

Boolean Model_c::ThereIsALaterSpill(Seconds timeOfInterest, TLEList *someLEListToIgnore)
{	// for hindcast option, so looking for latest spill end release time
	TLEList *thisLEList;
	long i,n;
	LETYPE leType;
	OSErr err = 0;
	Seconds testTime;
	
	for (i = 0, n = LESetsList->GetItemCount() ; i < n && !err; i++) 
	{
		LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE) continue;
		if(thisLEList == someLEListToIgnore) continue;
		if(thisLEList -> IAm(TYPE_OSSMLELIST))
		{
			TOLEList *thisOLEList = dynamic_cast<TOLEList*>(thisLEList); // typecast
			if (thisOLEList->fSetSummary.bWantEndRelTime) testTime = thisOLEList->fSetSummary.endRelTime;
			else testTime = thisOLEList->fSetSummary.startRelTime;
			if(testTime > timeOfInterest)
				return true;
		}
	}
	return false;
}


void Model_c::PossiblyReFloatLE(TMap *theMap, TLEList *theLEList, long i, LETYPE leType)
{
	LERec theLE;
	Boolean refloat = true;
	
	theLEList -> GetLE (i, &theLE);
	
	if (theMap -> CanReFloat (fDialogVariables.computeTimeStep, &theLE))
	{
		switch(leType)
		{
			default:
			case FORECAST_LE:
			{
				// JLM 1/11/99
				// beach half life is set to 1 hour (OSSM's default)
				// code goes here to allow user to set this etc.
				// should probably be map dependent etc....
				
				double halfLifeInHrs,thisStepInHours;
				float probOfRefloatingThisTimeStep,x;
				
				halfLifeInHrs = theMap -> RefloatHalfLifeInHrs (theLE.p);
				if ( halfLifeInHrs <= 0.0)
					refloat = true;
				else
				{
					thisStepInHours = model->GetTimeStep()/3600.0;
					probOfRefloatingThisTimeStep = 1.0 - pow(0.5,thisStepInHours/halfLifeInHrs);
					x = GetRandomFloat(0, 1.0);
					if(x <= probOfRefloatingThisTimeStep) 
						refloat = true;
					else 
						refloat = false;
					break;
				}
			}
				
		}
		////////
		if(refloat)
			theLEList -> ReFloatLE (i);
	}
	
}



OSErr Model_c::GetTotalBudgetTableHdl(short desiredMassVolUnits, BudgetTableDataH *totalBudgetTable)
{
	long i, j, n;
	TLEList *thisLEList;
	double density;
	BudgetTableData budgetTable; 
	BudgetTableDataH thisBudgetTableH = 0, totalBudgetTableH = 0;
	long sizeOfBudgetHdl, sizeOfTotalBudgetHdl = 0;
	short massunits;
	Boolean convertMassUnits = false;
	OSErr err = 0;
	
	for (i = 0, n = model->LESetsList->GetItemCount() ; i < n ; i++) 
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		if(thisLEList -> GetLEType() == UNCERTAINTY_LE ) continue;
		
		massunits = (dynamic_cast<TOLEList*>(thisLEList)) -> GetMassUnits();
		density =  (dynamic_cast<TOLEList*>(thisLEList)) -> fSetSummary.density;	
		if (massunits!=desiredMassVolUnits) convertMassUnits = true;
		thisBudgetTableH = (dynamic_cast<TOLEList*>(thisLEList)) -> GetBudgetTable();
		if (thisBudgetTableH) sizeOfBudgetHdl = _GetHandleSize((Handle)thisBudgetTableH)/sizeof(BudgetTableData);
		if (totalBudgetTableH)
		{
			sizeOfTotalBudgetHdl = _GetHandleSize((Handle)totalBudgetTableH)/sizeof(BudgetTableData);
			if (sizeOfTotalBudgetHdl > sizeOfBudgetHdl)	{printError("Budget tables don't match"); err = -1; goto done;}
		}
		else
		{
			totalBudgetTableH = (BudgetTableData**)_NewHandleClear((sizeOfBudgetHdl)*sizeof(BudgetTableData));
			if(!totalBudgetTableH) {TechError("TModel::GetTotalBudgetTableHdl()", "_NewHandleClear()", 0); err = memFullErr; goto done;}
		}
		
		for(j = 0; j < sizeOfBudgetHdl; j++)
		{	// will need to consider possibility that spills are in different units
			budgetTable = INDEXH(thisBudgetTableH,j);
			if (convertMassUnits)
			{
				budgetTable.amountReleased = VolumeMassToVolumeMass(budgetTable.amountReleased,density,massunits,desiredMassVolUnits);
				budgetTable.amountFloating = VolumeMassToVolumeMass(budgetTable.amountFloating,density,massunits,desiredMassVolUnits);
				budgetTable.amountEvaporated = VolumeMassToVolumeMass(budgetTable.amountEvaporated,density,massunits,desiredMassVolUnits);
				budgetTable.amountDispersed = VolumeMassToVolumeMass(budgetTable.amountDispersed,density,massunits,desiredMassVolUnits);
				budgetTable.amountBeached = VolumeMassToVolumeMass(budgetTable.amountBeached,density,massunits,desiredMassVolUnits);
				budgetTable.amountOffMap = VolumeMassToVolumeMass(budgetTable.amountOffMap,density,massunits,desiredMassVolUnits);
				budgetTable.amountRemoved = VolumeMassToVolumeMass(budgetTable.amountRemoved,density,massunits,desiredMassVolUnits);
			}
			INDEXH(totalBudgetTableH,j).timeAfterSpill = budgetTable.timeAfterSpill;		// times are the same for all spills
			INDEXH(totalBudgetTableH,j).amountReleased += budgetTable.amountReleased;
			INDEXH(totalBudgetTableH,j).amountFloating += budgetTable.amountFloating;
			INDEXH(totalBudgetTableH,j).amountDispersed += budgetTable.amountDispersed;
			INDEXH(totalBudgetTableH,j).amountEvaporated += budgetTable.amountEvaporated;
			INDEXH(totalBudgetTableH,j).amountBeached += budgetTable.amountBeached;
			INDEXH(totalBudgetTableH,j).amountOffMap += budgetTable.amountOffMap;
			INDEXH(totalBudgetTableH,j).amountRemoved += budgetTable.amountRemoved;
		}
	}
	
	// if there is no data delete the handle
	if (sizeOfBudgetHdl==0)	{if(totalBudgetTableH) DisposeHandle((Handle)totalBudgetTableH); totalBudgetTableH=0;}
	*totalBudgetTable = totalBudgetTableH;
	
done:
	if (err) {if(totalBudgetTableH) DisposeHandle((Handle)totalBudgetTableH); totalBudgetTableH=0;}
	return err;
}
// JLE 1/6/99 Return  the total amounts in units of "units". 

OSErr Model_c::GetTotalAmountStatistics(short desiredMassvolUnits,double *amtTotal,double *amtReleased,double *amtEvap,double *amtDisp,double *amtBeached,double * amtOffmap, double *amtFloating, double *amtRemoved)
{
	long i,numLESets = LESetsList->GetItemCount();
	double amttotal,amtevap ,amtbeached,amtoffmap,amtfloating,amtreleased,amtdispersed,amtremoved;
	
	TLEList		*thisLEList;
	
	// Get statistics in cm^3
	*amtTotal=*amtReleased = *amtEvap = *amtBeached=*amtOffmap = *amtFloating = *amtDisp = *amtRemoved = 0;
	for ( i = 0; i < numLESets; i++)
	{
		LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if(thisLEList -> GetLEType() != UNCERTAINTY_LE )
		{
			thisLEList->GetLEAmountStatistics(desiredMassvolUnits,&amttotal,&amtreleased,&amtevap,&amtdispersed,&amtbeached,&amtoffmap,&amtfloating,&amtremoved);
			*amtTotal+=amttotal;
			*amtReleased += amtreleased;
			*amtEvap += amtevap;
			*amtBeached += amtbeached;
			*amtOffmap += amtoffmap;
			*amtFloating += amtfloating;
			*amtDisp += amtdispersed;
			*amtRemoved += amtremoved;
		}
	}
	
	return noErr;
}

OSErr Model_c::GetTotalAmountSpilled(short desiredMassVolUnits,double *amtTotal)
{
	long i,numLESets = LESetsList->GetItemCount();
	double amttotal, massFrac, density, totalMass;
	short massunits;
	
	TLEList		*thisLEList;
	
	// Get statistics in cm^3
	*amtTotal = 0;
	for ( i = 0; i < numLESets; i++)
	{
		LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if(thisLEList -> GetLEType() != UNCERTAINTY_LE )
		{
			totalMass = (dynamic_cast<TOLEList*>(thisLEList))->GetTotalMass();
			massFrac = totalMass / thisLEList->GetNumOfLEs();
			massunits = (dynamic_cast<TOLEList*>(thisLEList))->GetMassUnits();
			density =  (dynamic_cast<TOLEList*>(thisLEList))->fSetSummary.density;	
			amttotal = VolumeMassToVolumeMass(totalMass,density,massunits,desiredMassVolUnits);
			*amtTotal+=amttotal;
		}
	}
	
	return noErr;
}

WorldPoint3D Model_c::TurnLEAlongShoreLine(WorldPoint3D waterPoint, WorldPoint3D beachedPoint, TMap *bestMap)
{
	WorldPoint3D movedPoint = {0,0,0.}, testPoint = {0,0,0.};
	WorldPoint3D firstEndPoint = {0,0,0.}, secondEndPoint = {0,0,0.};
	double alpha, testAngle, firstAngle, secondAngle;
	double dist, latDiff, longDiff, shorelineLength;
	long i;
	
	// code goes here, may not need to check bestMap if currentbeachesLE changes to newBestMap
	// want map to do this since it will be easier for PtCurMaps...
	TMap *newBestMap = 0;
	
	if (!bestMap -> InMap (beachedPoint.p))
	{	// the LE has left the map it was on
		newBestMap = GetBestMap (beachedPoint.p);
		if (newBestMap) {
			// it has moved to a new map
			bestMap = newBestMap; 
		}
		else
		{	// it has moved off all maps, shouldn't get here
			return waterPoint;	// something went wrong don't do anything
		}
	}
	
	if (bestMap -> OnLand (beachedPoint.p))	
	{
		// find "shoreline" by searching an arc around the beached point
		// for the first water point on either side. Then move LE parallel 
		// to shoreline in the direction the beaching vector tends towards
		WorldPoint center;
		center.pLong = (waterPoint.p.pLong + beachedPoint.p.pLong) / 2;
		center.pLat = (waterPoint.p.pLat + beachedPoint.p.pLat) / 2;
		
		latDiff = LatToDistance(beachedPoint.p.pLat - waterPoint.p.pLat);
		longDiff = LongToDistance(beachedPoint.p.pLong - waterPoint.p.pLong, center);
		alpha = atan(latDiff/longDiff);
		dist = DistanceBetweenWorldPoints(beachedPoint.p, waterPoint.p); 
		for (i=0;i<180;i++)
		{
			testAngle = PI*(i+1)/180.;
			testPoint.p.pLat = waterPoint.p.pLat + DistanceToLat(dist*sin(alpha+testAngle));
			testPoint.p.pLong = waterPoint.p.pLong + DistanceToLong(dist*cos(alpha+testAngle),center);
			
			if (bestMap -> OnLand(testPoint.p) || !bestMap -> InMap(testPoint.p)) continue;
			else 
			{	// found a water point, call this the first shoreline edge
				firstEndPoint.p = testPoint.p;
				firstAngle = testAngle;
				break;
			}
		}
		if(i>=179) return waterPoint;	// didn't find shoreline so don't let current move LE
		for (i=0;i<180;i++)
		{
			testAngle = PI*(i+1)/180.;
			testPoint.p.pLat = waterPoint.p.pLat + DistanceToLat(dist*sin(alpha-testAngle));
			testPoint.p.pLong = waterPoint.p.pLong + DistanceToLong(dist*cos(alpha-testAngle),center);
			
			if (bestMap -> OnLand(testPoint.p) || !bestMap -> InMap(testPoint.p)) continue;
			else 
			{	// found a water point, call this the second shoreline edge
				secondEndPoint.p = testPoint.p;
				secondAngle = testAngle;
				break;
			}
		}
		if(i>=179) return waterPoint; // didn't find shoreline so don't let current move LE
		
		shorelineLength = DistanceBetweenWorldPoints(firstEndPoint.p,secondEndPoint.p);
		// turn direction determined by which is greater, firstAngle or secondAngle, towards smaller one, if same?
		if (firstAngle < secondAngle)
		{
			movedPoint.p.pLat = waterPoint.p.pLat + DistanceToLat(LatToDistance(firstEndPoint.p.pLat - secondEndPoint.p.pLat)*(dist/shorelineLength));
			movedPoint.p.pLong = waterPoint.p.pLong + DistanceToLong(LongToDistance(firstEndPoint.p.pLong - secondEndPoint.p.pLong,center)*(dist/shorelineLength),center);
		}
		else
		{
			movedPoint.p.pLat = waterPoint.p.pLat + DistanceToLat(LatToDistance(secondEndPoint.p.pLat - firstEndPoint.p.pLat)*(dist/shorelineLength));
			movedPoint.p.pLong = waterPoint.p.pLong + DistanceToLong(LongToDistance(secondEndPoint.p.pLong - firstEndPoint.p.pLong,center)*(dist/shorelineLength),center);
		}
		movedPoint.z = beachedPoint.z;
		// check that movedPoint is not onLand
		if (bestMap -> InMap(movedPoint.p) && !bestMap -> OnLand(movedPoint.p))
			return movedPoint;
		else return waterPoint;
	}
	return waterPoint;	// shouldn't get here
}
/////////////////////////////////////////////////

Boolean Model_c::CurrentBeachesLE(WorldPoint3D startPoint, WorldPoint3D *movedPoint, TMap *bestMap)
{
	Boolean useNewMovementCheckCode = true; 
	WorldPoint3D testPoint = {0,0,0.};
	//////////////////
	// check for transition off maps, beaching, etc
	//////////////////
	testPoint.p = (*movedPoint).p;
	if (useNewMovementCheckCode && fDialogVariables.preventLandJumping)
	{ 
		//////////////////////////////////////////
		// use the from and to points as a vector and check to see if the oil can move 
		// along that vector without hitting the shore
		//////////////////////////////////////////
		
		TMap *newBestMap = 0;
		
		testPoint = bestMap -> MovementCheck(startPoint,testPoint, false);	// not using this routine...
		if (!bestMap -> InMap (testPoint.p))
		{	// the LE has left the map it was on
			newBestMap = GetBestMap (testPoint.p);
			if (newBestMap) {
				// it has moved to a new map
				// so we need to do the movement check on the new map as well
				// code goes here, we should worry about it jumping across maps
				// i.e. we should verify the maps rects intersect
				bestMap = newBestMap; // set bestMap for the beaching check, careful - does this change bestMap on return?
				testPoint = bestMap -> MovementCheck(startPoint,testPoint, false);
			}
			else
			{	// it has moved off all maps
				return false;	// off all maps, let it go for now
			}
		}
		////////
		// check for beaching 
		////////
		if (bestMap -> OnLand (testPoint.p))
		{
			(*movedPoint).p = testPoint.p;	// return beached point
			return true;	// mover beaches LE
		}
		else
			return false;	  // mover does not beach LE
	}
	else
	{	// old code, check for transition off maps and beaching 
		
		// check for transition off our map and into another
		if (!bestMap -> InMap (testPoint.p))
		{
			TMap *newBestMap = GetBestMap (testPoint.p);
			if (newBestMap)
				bestMap = newBestMap;
			else
				return false;	// off all maps, let it go for now
		}
		// check for beaching in the best map
		if (bestMap -> OnLand (testPoint.p))
			return true;	// mover beaches LE
		else
			return false;	  // mover does not beach LE
	}
	return false;
}

// AH: I know that this doesn't look quite right.

Boolean Model_c::ThereIsA3DMover(float *arrowDepth)
{	// for now until we decide who is allowed to use 3D stuff
	//rework this 
	
	TMover *mover = GetMover(TYPE_PTCURMOVER);
	if (mover && gNoaaVersion)
	{	
		if (/*OK*/ (dynamic_cast<PtCurMover*>(mover)) -> fVar.gridType != TWO_D)
		{
			*arrowDepth = /*OK*/ (dynamic_cast<PtCurMover*>(mover)) -> fVar.arrowDepth;
			return true;
		}
	}
	else	
	{
		mover = this->GetMover(TYPE_CATSMOVER3D);
		if (mover)
		{
			// need to handle case where 3D regular grid netCDF is using CATS grid for the boundary
			mover = this->GetMover(TYPE_NETCDFMOVER);
			if (mover && gNoaaVersion)
			{
				if (/*OK*/ (dynamic_cast<NetCDFMover*>(mover)) -> fVar.gridType != TWO_D)
				{
					*arrowDepth = /*OK*/ (dynamic_cast<NetCDFMover*>(mover)) -> fVar.arrowDepth;
					return true;
				}
			}
			*arrowDepth = 0;	// CatsMovers only show surface velocities
			return true;
		}
		else
		{
			mover = this->GetMover(TYPE_TRICURMOVER);
			if (mover)
			{
				*arrowDepth = /*OK*/ (dynamic_cast<TriCurMover*>(mover)) -> fVar.arrowDepth;
				return true;
			}
			else
			{	// code goes here, should we also have a noaa.ver here?
				mover = this->GetMover(TYPE_NETCDFMOVER);
				if (mover && gNoaaVersion)
				{
					if (/*OK*/(dynamic_cast<NetCDFMover*>(mover)) -> fVar.gridType != TWO_D)
					{
						*arrowDepth = /*OK*/ (dynamic_cast<NetCDFMover*>(mover)) -> fVar.arrowDepth;
						return true;
					}
				}
				else
				{
					mover = this->GetMover(TYPE_COMPOUNDMOVER);
					if (mover && gNoaaVersion)
					{
						*arrowDepth = /*CHECK*/(dynamic_cast<TCompoundMover*>(mover)) -> GetArrowDepth();
						return /*CHECK*/(dynamic_cast<TCompoundMover*>(mover))->IAmA3DMover();
					}
				}
			}
		}
	}
	if (gNoaaVersion) return true;
	return false;
}

long Model_c::GetNumMovers(char* moverName)
{
	// loop through each mover in the universal map
	TMover *thisMover = nil;
	TMap *map;
	char thisName[kMaxNameLen];
	long i,n,k,d,numMovers=0;
	
	// universal movers
	for (k = 0, d = this->uMap->moverList->GetItemCount (); k < d; k++)
	{
		this->uMap->moverList -> GetListItem ((Ptr) &thisMover, k);
		thisMover -> GetClassName (thisName);
		if(!strcmpnocase(thisName,moverName)) numMovers++;
	}
	
	// movers that belong to a map
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&map, i);
		for (k = 0, d = map -> moverList -> GetItemCount (); k < d; k++)
		{
			map -> moverList -> GetListItem ((Ptr) &thisMover, k);
			thisMover -> GetClassName (thisName);
			if(!strcmpnocase(thisName,moverName)) numMovers++;
		}
	}
	
	return numMovers;
}

long Model_c::GetNumWindMovers()
{
	long numWindMovers=0;
	numWindMovers += GetNumMovers("Constant Wind");
	numWindMovers += GetNumMovers("Variable Wind");
	return numWindMovers;
}

// For keeping good habits,
#undef TModel 
#undef TMap 
#undef TMover 
#undef TWindMover 
#undef TRandom 
#undef TRandom3D
#undef TOSSMTimeValue
#undef TCurrentMover