/*
 *  CATSMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CATSMover_c__
#define __CATSMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "CurrentMover_c.h"

#ifndef pyGNOME
#include "TOSSMTimeValue.h"
#include "TMap.h"
#include "GridVel.h"
#else
#include "OSSMTimeValue_c.h"
#include "GridVel_c.h"
#include "Map_c.h"
#define TOSSMTimeValue OSSMTimeValue_c
#define TGridVel GridVel_c
#define TMap Map_c
#endif

class CATSMover_c : virtual public CurrentMover_c {

public:
	WorldPoint 		refP; 					// location of tide station or map-join pin
	TGridVel		*fGrid;					//VelocityH		grid; 
	long 			refZ; 					// meters, positive up
	short 			scaleType; 				// none, constant, or file
	double 			scaleValue; 			// constant value to match at refP
	char 			scaleOtherFile[32]; 	// file to match at refP
	double 			refScale; 				// multiply current-grid value at refP by refScale to match value
	Boolean 		bRefPointOpen;
	Boolean			bUncertaintyPointOpen;
	Boolean 		bTimeFileOpen;
	Boolean			bTimeFileActive;		// active / inactive flag
	Boolean 		bShowGrid;
	Boolean 		bShowArrows;
	double 			arrowScale;
	TOSSMTimeValue *timeDep;
	double			fEddyDiffusion;			// cm**2/s minimum eddy velocity for uncertainty
	double			fEddyV0;			//  in m/s, used for cutoff of minimum eddy for uncertainty
	TCM_OPTIMZE fOptimize; // this does not need to be saved to the save file
	
						CATSMover_c (TMap *owner, char *name);
						CATSMover_c () { timeDep = 0; fUncertaintyListH = 0; fLESetSizesH = 0; fUncertainStartTime = 0;}
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	void				SetRefPosition (WorldPoint p, long z) { refP = p; refZ = z; }
	void				GetRefPosition (WorldPoint *p, long *z) { (*p) = refP; (*z) = refZ; }
	virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	
	void				SetTimeDep (TOSSMTimeValue *newTimeDep) { timeDep = newTimeDep; }
	TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
	void				DeleteTimeDep ();
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	VelocityRec			GetSmoothVelocity (WorldPoint p);
	//OSErr				ComputeVelocityScale ();
	virtual OSErr		ComputeVelocityScale ();
	virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *velStr);
	virtual	OSErr		ReadTopology(char* path, TMap **newMap);

};

#undef TOSSMTimeValue
#undef TGridVel
#undef TMap
#endif
