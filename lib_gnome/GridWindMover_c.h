/*
 *  GridWindMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridWindMover_c__
#define __GridWindMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "WindMover_c.h"
#include "ExportSymbols.h"


#ifndef pyGNOME
//#include "GridVel.h"
#include "TimeGridVel.h"
#else
#include "TimeGridVel_c.h"
#define TimeGridVel TimeGridVel_c
#endif


class DLL_API GridWindMover_c : virtual public WindMover_c {

public:
	
	Boolean 	bShowGrid;
	Boolean 	bShowArrows;
	
	char		fPathName[kMaxNameLen];
	char		fFileName[kPtCurUserNameLen]; // short file name
	//char		fFileName[kMaxNameLen]; // short file name - might want to allow longer names
	
	TimeGridVel	*timeGrid;	//VelocityH		grid; 
	
	short fUserUnits;
	float fWindScale;
	float fArrowScale;
	Boolean fIsOptimizedForStep;

#ifndef pyGNOME
	GridWindMover_c (TMap *owner, char* name);
#endif
	GridWindMover_c ();
	~GridWindMover_c () { Dispose (); }
	virtual void		Dispose ();
	
	//virtual ClassID 	GetClassID () { return TYPE_GRIDWINDMOVER; }
	//virtual Boolean		IAm(ClassID id) { if(id==TYPE_GRIDWINDMOVER) return TRUE; return WindMover_c::IAm(id); }

	virtual WorldRect GetGridBounds(){return timeGrid->GetGridBounds();}	
	void		SetTimeGrid(TimeGridVel *newTimeGrid) {timeGrid = newTimeGrid;}

	void	SetExtrapolationInTime(bool extrapolate) {timeGrid->SetExtrapolationInTime(extrapolate);}	
	bool	GetExtrapolationInTime() {return timeGrid->GetExtrapolationInTime();}	
	
	void	SetTimeShift(long timeShift) {timeGrid->SetTimeShift(timeShift);}	
	long	GetTimeShift() {return timeGrid->GetTimeShift();}	
	
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	virtual void 		ModelStepIsDone();
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	
	OSErr			TextRead(char *path,char *topFilePath);
	OSErr 			ExportTopology(char* path){return timeGrid->ExportTopology(path);}

	OSErr 			get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, double* windages, short* LE_status, LEType spillType, long spill_ID);
};

#endif
