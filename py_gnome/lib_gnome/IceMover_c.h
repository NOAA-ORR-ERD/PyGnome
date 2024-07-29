/*
 *  IceMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __IceMover_c__
#define __IceMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "CurrentMover_c.h"
#include "GridCurrentMover_c.h"
#include "ExportSymbols.h"

#ifndef pyGNOME
//#include "GridVel.h"
#include "TimeGridVel.h"
#else
#include "TimeGridVel_c.h"
#define TimeGridVel TimeGridVel_c
#endif

class DLL_API IceMover_c : virtual public GridCurrentMover_c {

public:
	//UncertaintyParameters fUncertainParams;
	//double fCurScale;
	//char fPathName[kMaxNameLen];
	//char fUserName[kPtCurUserNameLen];
	//Boolean fIsOptimizedForStep;
	//TimeGridVel *timeGrid;
	//TimeGridVel *iceTimeGrid;
	
	//Boolean fAllowVerticalExtrapolationOfCurrents;
	//float	fMaxDepthForExtrapolation;
	
#ifndef pyGNOME
	IceMover_c (TMap *owner, char *name);
#endif
	IceMover_c (); 
	virtual ~IceMover_c () { Dispose (); }
	virtual void		Dispose ();
	
	//virtual ClassID 	GetClassID () { return TYPE_ICEMOVER; }
	//virtual Boolean	IAm(ClassID id) { if(id==TYPE_ICEMOVER) return TRUE; return GridCurrentMover_c::IAm(id); }

	//virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	//VelocityRec			GetPatValue (WorldPoint p);
	//VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	
	//virtual WorldRect	GetGridBounds(){return timeGrid->GetGridBounds();}	
	//virtual OSErr		InitMover (TimeGridVel *grid); 
	//virtual void		SetTimeGrid(TimeGridVel *newTimeGrid) {timeGrid = newTimeGrid;}
	void		SetTimeGrid(TimeGridVel *newTimeGrid) {timeGrid = newTimeGrid;}

	//void	SetExtrapolationInTime(bool extrapolate) {timeGrid->SetExtrapolationInTime(extrapolate);}	
	//bool	GetExtrapolationInTime() {return timeGrid->GetExtrapolationInTime();}	

	//void	SetTimeShift(long timeShift) {timeGrid->SetTimeShift(timeShift);}	
	//long	GetTimeShift() {return timeGrid->GetTimeShift();}	
	
	virtual OSErr 		PrepareForModelRun(); 
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	virtual void 		ModelStepIsDone();
			// may need these functions eventually if add a separate ice grid
			//TopologyHdl GetTopologyHdl(void);
			//LongPointHdl GetPointsHdl(void);
			//WORLDPOINTH	GetTriangleCenters();
			//long 		GetNumTriangles(void);
			OSErr 		GetIceFields(Seconds model_time, double *ice_fraction, double *ice_thickness);
			OSErr 		GetIceVelocities(Seconds model_time, VelocityFRec *ice_velocity);
			OSErr 		GetMovementVelocities(Seconds model_time, VelocityFRec *ice_velocity);
			OSErr		TextRead(char *path,char *topFilePath);
			OSErr 		ExportTopology(char* path){return timeGrid->ExportTopology(path);}

			OSErr		get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID);

};

#endif
