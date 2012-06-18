/*
 *  LEList_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/12/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "LEList_c.h"
#include "GEOMETRY.H"
#include "CompFunctions.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

LEList_c::LEList_c()
{
	LEHandle = 0;
	this->numOfLEs = 0;
	//this->massUnits = 0;
	this-> fLeType = FORECAST_LE;
	memset(&fOwnersUniqueID,0,sizeof(fOwnersUniqueID));
		
	bOpen = FALSE;
}



WorldRect LEList_c::GetLEBounds()
{
	long i;
	WorldPoint p;
	WorldRect bounds = voidWorldRect;
	
	for (i = 0 ; i < this->numOfLEs ; i++) {
		p = INDEXH(LEHandle, i).p;
		AddWPointToWRect(p.pLat, p.pLong, &bounds);
	}
	
	return bounds;
}

//JLE 1/6/99

void LEList_c::GetLEStatistics(long* numReleased,long* numEvaporated,long* numBeached, long* numOffMap, long* numFloating)
{
	long i,numNotReleased = 0;
	*numEvaporated = *numBeached = *numOffMap = *numFloating = 0;
	if (LEHandle) {
		for (i = 0 ; i < this->numOfLEs ; i++) {
			switch( INDEXH(LEHandle, i).statusCode)
			{
				case OILSTAT_NOTRELEASED: numNotReleased++; break;
				case OILSTAT_OFFMAPS: (*numOffMap)++; break;
				case OILSTAT_ONLAND: (*numBeached)++; break;
				case OILSTAT_EVAPORATED: (*numEvaporated)++; break;
				case OILSTAT_INWATER:(*numFloating)++; break;
			}
		}
	}
	*numReleased = this->numOfLEs - numNotReleased;
}

void LEList_c::RecalculateLEStatistics(long* numDispersed,long* numFloating,long* numRemoved,long* numOffMaps)
{
	// code goes here, might want to calculate percent dissolved here
	long i;
	if (LEHandle) {
		for (i = 0 ; i < this->numOfLEs ; i++) {	// code goes here, Alan occasionally seeing budget table floating < 0, maybe not released has z>0 or dispersed Le is getting beached?
			if( ((INDEXH(LEHandle, i).dispersionStatus == HAVE_DISPERSED || INDEXH(LEHandle, i).dispersionStatus == HAVE_DISPERSED_NAT) || INDEXH(LEHandle,i).z > 0) && !(INDEXH(LEHandle, i).statusCode==OILSTAT_EVAPORATED) && !(INDEXH(LEHandle, i).statusCode==OILSTAT_OFFMAPS) && !(INDEXH(LEHandle, i).statusCode==OILSTAT_NOTRELEASED) && !(INDEXH(LEHandle, i).statusCode==OILSTAT_ONLAND))
				//if( ((INDEXH(LEHandle, i).dispersionStatus == HAVE_DISPERSED || INDEXH(LEHandle, i).dispersionStatus == HAVE_DISPERSED_NAT) || INDEXH(LEHandle,i).z > 0) && INDEXH(LEHandle, i).statusCode==OILSTAT_INWATER )
			{
				(*numDispersed)++; 
				(*numFloating)--; 
				if ((*numFloating) < 0) 
				{
					//printNote("Num floating < 0 in RecalculateLEStatistics");
				}
			}
			if( (INDEXH(LEHandle, i).dispersionStatus == HAVE_REMOVED) && (INDEXH(LEHandle, i).statusCode==OILSTAT_OFFMAPS) )
			{
				(*numRemoved)++; 
				(*numOffMaps)--; 
				if ((*numOffMaps) < 0) 
				{
					printNote("Num off maps < 0 in RecalculateLEStatistics");
				}
			}
		}
	}
}

void LEList_c::BeachLE(long i, WorldPoint beachPosition)
{
	if (GetLEStatus (i) == OILSTAT_INWATER)
	{
		INDEXH(LEHandle, i).lastWaterPt = INDEXH(LEHandle, i).p;
		SetLEPosition(i, beachPosition);
		INDEXH(LEHandle, i).beachTime = model->modelTime;
		SetLEStatus(i, OILSTAT_ONLAND);
	}
}

void LEList_c::ReFloatLE(long i)
{
	SetLEPosition(i, INDEXH(LEHandle, i).lastWaterPt);
	SetLEStatus(i, OILSTAT_INWATER);
}

OilStatus LEList_c::GetLEStatus(long i)
{
	return INDEXH(LEHandle, i).statusCode;
}

void LEList_c::SetLEStatus(long i, OilStatus status)
{
	INDEXH(LEHandle, i).statusCode = status;
}

WorldPoint LEList_c::GetLEPosition(long i)
{
	return INDEXH(LEHandle, i).p;
}

void LEList_c::SetLEPosition(long i, WorldPoint p)
{
	INDEXH(LEHandle, i).p = p;
}

Seconds LEList_c::GetLEReleaseTime(long i)
{
	return INDEXH(LEHandle, i).releaseTime;
}

void LEList_c::ReleaseLE(long i)
{
	INDEXH(LEHandle, i).statusCode = OILSTAT_INWATER;
}
