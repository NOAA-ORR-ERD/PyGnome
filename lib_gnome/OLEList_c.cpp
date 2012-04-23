/*
 *  OLEList_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "OLEList_c.h"
#include "MemUtils.h"
#include "CompFunctions.h"
#include "StringFunctions.h"
#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

OLEList_c::OLEList_c()
{
	memset(&fSetSummary,0,sizeof(fSetSummary));
	bMassBalanceOpen = false;
	bReleasePositionOpen = false;
	initialLEs = nil;
	binitialLEsVisible = true;
	bShowDispersantArea = false;
	memset(&fDispersantData,0,sizeof(fDispersantData));
	//memset(&fWindageData,0,sizeof(fWindageData));
	fWindageData.windageA = .01;
	fWindageData.windageB = .04;
	fWindageData.persistence = .25;	// in hours		
	fAdiosDataH = nil;	
	fBudgetTableH = nil;	
	fOilTypeName[0] = 0;
#ifndef pyGNOME
	fColor = colors[BLACK];	// default to draw LEs in black
#endif
}

void OLEList_c::AddToBudgetTableHdl(BudgetTableData *budgetTable)
{	// may want to switch this to a model field, then can total all the spills information
	// rather than have a separate budget table for each spill
	long i,sizeOfHdl;
	OSErr err = 0;
	if(!fBudgetTableH)
	{
		fBudgetTableH = (BudgetTableData**)_NewHandle(0);
		if(!fBudgetTableH) {TechError("TOLEList::AddToBudgetTableHdl()", "_NewHandle()", 0); err = memFullErr; return;}
	}
	sizeOfHdl = _GetHandleSize((Handle)fBudgetTableH)/sizeof(BudgetTableData);
	_SetHandleSize((Handle) fBudgetTableH, (sizeOfHdl+1)*sizeof(BudgetTableData));
	if (_MemError()) { TechError("TOLEList::AddToBudgetTableHdl()", "_SetHandleSize()", 0); return; }
	(*fBudgetTableH)[sizeOfHdl].timeAfterSpill = (*budgetTable).timeAfterSpill;				
	(*fBudgetTableH)[sizeOfHdl].amountReleased = (*budgetTable).amountReleased;				
	(*fBudgetTableH)[sizeOfHdl].amountFloating = (*budgetTable).amountFloating;				
	(*fBudgetTableH)[sizeOfHdl].amountDispersed = (*budgetTable).amountDispersed;				
	(*fBudgetTableH)[sizeOfHdl].amountEvaporated = (*budgetTable).amountEvaporated;				
	(*fBudgetTableH)[sizeOfHdl].amountBeached = (*budgetTable).amountBeached;				
	(*fBudgetTableH)[sizeOfHdl].amountOffMap = (*budgetTable).amountOffMap;
	(*fBudgetTableH)[sizeOfHdl].amountRemoved = (*budgetTable).amountRemoved;
	return;
}

/**************************************************************************************************/
WindageRec OLEList_c::GetWindageInfo ()
{
	WindageRec	info;
	
	memset(&info,0,sizeof(info));
	info.windageA = this -> fWindageData.windageA;
	info.windageB = this -> fWindageData.windageB;
	info.persistence = this -> fWindageData.persistence;
	
	return info;
}

void OLEList_c::SetWindageInfo (WindageRec info)
{
	this -> fWindageData.windageA = info.windageA;
	this -> fWindageData.windageB = info.windageB;
	this -> fWindageData.persistence = info.persistence;
	
	return;
}

/**************************************************************************************************/
DispersionRec OLEList_c::GetDispersionInfo ()
{
	DispersionRec	info;
	
	memset(&info,0,sizeof(info));
	info.bDisperseOil = this -> fDispersantData.bDisperseOil;
	info.timeToDisperse	= this -> fDispersantData.timeToDisperse;
	info.duration = this -> fDispersantData.duration;
	info.api = this -> fDispersantData.api;		
	info.areaToDisperse.hiLat	= this -> fDispersantData.areaToDisperse.hiLat;
	info.areaToDisperse.loLat	= this -> fDispersantData.areaToDisperse.loLat;
	info.areaToDisperse.hiLong	= this -> fDispersantData.areaToDisperse.hiLong;
	info.areaToDisperse.loLong	= this -> fDispersantData.areaToDisperse.loLong;
	info.amountToDisperse		= this -> fDispersantData.amountToDisperse;	
	info.lassoSelectedLEsToDisperse = this -> fDispersantData.lassoSelectedLEsToDisperse;
	
	return info;
}

/**************************************************************************************************/
void OLEList_c::SetDispersionInfo (DispersionRec info)
{
	this -> fDispersantData.bDisperseOil = info.bDisperseOil;
	this -> fDispersantData.timeToDisperse = info.timeToDisperse;
	this -> fDispersantData.duration = info.duration;
	this -> fDispersantData.api = info.api;
	this -> fDispersantData.areaToDisperse.hiLat 	= info.areaToDisperse.hiLat;		
	this -> fDispersantData.areaToDisperse.loLat 	= info.areaToDisperse.loLat;			
	this -> fDispersantData.areaToDisperse.hiLong 	= info.areaToDisperse.hiLong;	
	this -> fDispersantData.areaToDisperse.loLong 	= info.areaToDisperse.loLong;	
	this -> fDispersantData.amountToDisperse 	= info.amountToDisperse;	
	this -> fDispersantData.lassoSelectedLEsToDisperse = info.lassoSelectedLEsToDisperse;
	
	return;
}


long OLEList_c::GetNumAdiosBudgetTableItems(void)
{
	long numInHdl = 0;
	if (fAdiosDataH) numInHdl = _GetHandleSize((Handle)fAdiosDataH)/sizeof(**fAdiosDataH);
	
	return numInHdl;
}


OSErr OLEList_c::CalculateAverageIntrusionDepth(double *avDepth, double *stdDev)
{	// this was just a diagnostic for Debra, not using anymore...
	long j, numSubsurfaceLEs = 0;
	double totalDepth = 0, totalDev = 0;
	LERec thisLE;
	OSErr err = 0;
	for (j = 0; j < this -> numOfLEs; j++)
	{
		this -> GetLE (j, &thisLE);
		if (thisLE.z > 0)
		{
			numSubsurfaceLEs ++;
			totalDepth += thisLE.z;			
		}
	}
	if (numSubsurfaceLEs>0) totalDepth /= numSubsurfaceLEs;
	for (j = 0; j < this -> numOfLEs; j++)
	{
		this -> GetLE (j, &thisLE);
		if (thisLE.z > 0)
		{
			totalDev += (thisLE.z - totalDepth)*(thisLE.z - totalDepth);
		}
	}
	*avDepth = totalDepth;
	if (numSubsurfaceLEs > 0) *stdDev = sqrt(totalDev/numSubsurfaceLEs); 
	return err;		
}


short OLEList_c::GetMassUnitType()
{
	short masstype;
	switch(GetMassUnits())
	{
		case GALLONS:
		case BARRELS:
		case CUBICMETERS:
			masstype= VOLUMETYPE;
			
		case KILOGRAMS: 
		case METRICTONS:
		case SHORTTONS: 
			masstype= MASSTYPE;
	}
	return masstype;
}

// Return amount statistics in desiredMassVolUnits 
void OLEList_c::GetLEAmountStatistics(short desiredMassVolUnits, double *amtTotal,double *amtReleased,double *amtEvaporated,double *amtDispersed,
									 double *amtBeached,double * amtOffmap, double *amtFloating, double *amtRemoved)
{	// this is only called in Standard mode
	long numReleased,numEvap,numBeached,numOffmap,numfloating,numDisp=0,numRemoved=0;
	double massFrac = GetTotalMass()/GetNumOfLEs();
	short massunits = this->GetMassUnits();
	double density =  GetPollutantDensity(this->GetOilType());	
	
	this->GetLEStatistics(&numReleased,&numEvap,&numBeached,&numOffmap,&numfloating);
	if (fDispersantData.bDisperseOil || fAdiosDataH || fSetSummary.z > 0) this->RecalculateLEStatistics(&numDisp,&numfloating,&numRemoved,&numOffmap);
	
	//*amtTotal = GetTotalMass();
	*amtTotal = VolumeMassToVolumeMass(GetTotalMass(),density,massunits,desiredMassVolUnits);
	*amtReleased = VolumeMassToVolumeMass(numReleased*massFrac,density,massunits,desiredMassVolUnits);
	*amtEvaporated = VolumeMassToVolumeMass(numEvap*massFrac,density,massunits,desiredMassVolUnits);
	*amtDispersed = VolumeMassToVolumeMass(numDisp*massFrac,density,massunits,desiredMassVolUnits);
	*amtBeached = VolumeMassToVolumeMass(numBeached*massFrac,density,massunits,desiredMassVolUnits);
	*amtOffmap = VolumeMassToVolumeMass(numOffmap*massFrac,density,massunits,desiredMassVolUnits);
	*amtFloating = VolumeMassToVolumeMass(numfloating*massFrac,density,massunits,desiredMassVolUnits);
	*amtRemoved = VolumeMassToVolumeMass(numRemoved*massFrac,density,massunits,desiredMassVolUnits);
}