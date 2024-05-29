/*
 *  GridCurrentMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridCurrentMover_c__
#define __GridCurrentMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "CurrentMover_c.h"
#include "ExportSymbols.h"

#ifndef pyGNOME
//#include "GridVel.h"
#include "TimeGridVel.h"
#else
#include "TimeGridVel_c.h"
#define TimeGridVel TimeGridVel_c
#endif

class DLL_API GridCurrentMover_c : virtual public CurrentMover_c {

public:
	UncertaintyParameters fUncertainParams;
	double fCurScale;
	char fPathName[kMaxNameLen];
	char fUserName[kPtCurUserNameLen];
	Boolean fIsOptimizedForStep;
	TimeGridVel *timeGrid;
	int num_method;
	Boolean fAllowVerticalExtrapolationOfCurrents;
	float	fMaxDepthForExtrapolation;
	
#ifndef pyGNOME
	GridCurrentMover_c (TMap *owner, char *name);
#endif
	GridCurrentMover_c (); 
	virtual ~GridCurrentMover_c () { Dispose (); }
	virtual void		Dispose ();
	
	//virtual ClassID 	GetClassID () { return TYPE_GRIDCURRENTMOVER; }
	//virtual Boolean	IAm(ClassID id) { if(id==TYPE_GRIDCURRENTMOVER) return TRUE; return CurrentMover_c::IAm(id); }

	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	
	virtual WorldRect	GetGridBounds(){return timeGrid->GetGridBounds();}	
	//virtual OSErr		InitMover (TimeGridVel *grid); 
	//virtual void		SetTimeGrid(TimeGridVel *newTimeGrid) {timeGrid = newTimeGrid;}
	void		SetTimeGrid(TimeGridVel *newTimeGrid) {timeGrid = newTimeGrid;}

	void	SetExtrapolationInTime(bool extrapolate) {timeGrid->SetExtrapolationInTime(extrapolate);}	
	bool	GetExtrapolationInTime() {return timeGrid->GetExtrapolationInTime();}	

	void	SetTimeShift(long timeShift) {timeGrid->SetTimeShift(timeShift);}	
	long	GetTimeShift() {return timeGrid->GetTimeShift();}	
	
	OSErr	GetDataStartTime(Seconds *startTime) {return timeGrid->GetDataStartTime(startTime);}	
	OSErr	GetDataEndTime(Seconds *endTime) {return timeGrid->GetDataEndTime(endTime);}	
	
	virtual OSErr 		PrepareForModelRun(); 
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	virtual void 		ModelStepIsDone();
	
			OSErr		TextRead(char *path,char *topFilePath);
			OSErr 		ExportTopology(char* path){return timeGrid->ExportTopology(path);}

			OSErr 		GetScaledVelocities(Seconds model_time, VelocityFRec *velocity);
			TopologyHdl GetTopologyHdl(void);
			GridCellInfoHdl GetCellDataHdl(void);
			LongPointHdl GetPointsHdl(void);
			WORLDPOINTH	GetTriangleCenters();
			long 		GetNumTriangles(void);
			long 		GetNumPoints(void);
			bool 		IsTriangleGrid(){return timeGrid->IsTriangleGrid();}
			bool 		IsDataOnCells(){return timeGrid->IsDataOnCells();}
			bool 		IsRegularGrid(){return timeGrid->IsRegularGrid();}

			OSErr		get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID);



private:
	WorldPoint3D scale_WP(WorldPoint3D, double);
	WorldPoint3D add_two_WP3D(const WorldPoint3D&, const WorldPoint3D&);
};

#endif
