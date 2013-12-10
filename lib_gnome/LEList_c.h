/*
 *  LEList_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/12/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __LEList_c__
#define __LEList_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "ClassID_c.h"

class LEList_c : virtual public ClassID_c {

public:
	long			numOfLEs;
	//short			massUnits;
	LERecH			LEHandle; 	// handle to LE array
	LETYPE 			fLeType;
	//UNIQUEID		fOwnersUniqueID; // set if owned by another LE set, i.e this is a mirrored set

					LEList_c ();
	virtual OSErr	Reset (Boolean newKeys) { return noErr; }
	virtual LETYPE 	GetLEType () { return fLeType; }
	long			GetNumOfLEs(){return numOfLEs;}
	void			SetLEStatus (long leNum, OilStatus status);
	OilStatus		GetLEStatus (long leNum);
	void			ReFloatLE (long leNum);
	void			GetLE (long leNum, LERecP theLE);
	void			SetLE (long leNum, LERecP theLE);
	WorldPoint		GetLEPosition (long leNum);
	void			SetLEPosition (long leNum, WorldPoint p);
	Seconds			GetLEReleaseTime (long leNum);
	void			ReleaseLE (long leNum);
	void			AgeLE (long leNum);
//	virtual void	BeachLE (long leNum, WorldPoint beachPosition);		minus AH 06/20/2012
	WorldRect		GetLEBounds ();
	void 			GetLEStatistics(long* numReleased,long* numEvaporated,long* numBeached, long* numOffMap, long* numFloating);
	void 			RecalculateLEStatistics(long* numEvaporated,long* numFloating, long* numRemoved, long* numOffMaps); // 	to account for dispersion
	virtual void 	GetLEAmountStatistics(short desiredMassVolUnits, double *amtTotal,double *amtReleased,double *amtEvaporated,
										  double *amtDispersed,double *amtBeached,double *amtOffmap, double *amtFloating, double *amtRemoved){};
	virtual long	GetLECount () { return numOfLEs; }  
	
	virtual long	GetMassUnits () { return -1; } // they must override
	virtual short	GetMassUnitType () { return -1; } // they must override
	virtual double	GetTotalMass () { return -1; } ; // they must override
	
	//virtual ClassID GetClassID () { return TYPE_LELIST; }
	//virtual Boolean	IAm(ClassID id) { if(id==TYPE_LELIST) return TRUE; return ClassID_c::IAm(id); }
	

};

inline void LEList_c::GetLE(long i, LERecP theLE)
{
	*theLE = INDEXH(LEHandle, i);
}

inline void LEList_c::SetLE(long i, LERecP theLE)
{
	INDEXH(LEHandle, i) =  *theLE;
}

#endif