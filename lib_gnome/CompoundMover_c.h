/*
 *  CompoundMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CompoundMover_c__
#define __CompoundMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "CurrentMover_c.h"
#include "CMYLIST.H"

#ifdef pyGNOME
#define TMap Map_c
#define TMover Mover_c
#define TCurrentMover CurrentMover_c
#define TTriGridVel TriGridVel_c
#define TTriGridVel3D TriGridVel3D_c
#endif

class TCurrentMover;
class TTriGridVel;
class TTriGridVel3D;
class TCompoundMap;
class TMover;

class CompoundMover_c : virtual public CurrentMover_c {

public:
	//TCATSMover			*pattern1;
	//TCATSMover			*pattern2;
	CMyList				*moverList; 			// list of the mover's component currents
	//Boolean				bPat1Open;
	//Boolean				bPat2Open;
	//TOSSMTimeValue		*timeFile;
	
	Boolean 			bMoversOpen;
	/*WorldPoint			refP;
	 Boolean 			bRefPointOpen;
	 
	 double				pat1Angle;
	 double				pat2Angle;
	 
	 double				pat1Speed;
	 double				pat2Speed;
	 
	 long				pat1SpeedUnits;
	 long				pat2SpeedUnits;
	 
	 double				pat1ScaleToValue;
	 double				pat2ScaleToValue;
	 
	 long				scaleBy;
	 
	 */
	//							optimize fields don't need to be saved
	//TC_OPTIMZE			fOptimize;
	
	//long				timeMoverCode;
	//char 				windMoverName [64]; 	// file to match at refP
	
	CompoundMover_c (TMap *owner, char *name);
	CompoundMover_c () {}
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep);
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, const Seconds&, const Seconds&, bool); // AH 07/10/2012
	virtual void 		ModelStepIsDone();
	
	virtual WorldPoint3D       GetMove(const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual	Boolean 		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual float		GetArrowDepth();

	virtual	LongPointHdl	GetPointsHdl();
	TTriGridVel*		GetGrid(Boolean wantRefinedGrid);
	TTriGridVel3D*		GetGrid3D(Boolean wantRefinedGrid);
	TCurrentMover*		Get3DCurrentMover();
	TTriGridVel3D*		GetGrid3DFromMoverIndex(long moverIndex);
	TCurrentMover*		Get3DCurrentMoverFromIndex(long moverIndex);
	virtual Boolean		IAmA3DMover();
	virtual OSErr		AddMover (TMover *theMover, short where);

};

#undef TMap
#undef TMover
#undef TCurrentMover
#undef TTriGridVel
#undef TTriGridVel3D
#endif
