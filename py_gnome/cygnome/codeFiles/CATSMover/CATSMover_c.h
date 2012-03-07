/*
 *  CATSMover_c.h
 *  c_gnome
 *
 *  Created by Generic Programmer on 2/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CATSMover_c__
#define __CATSMover_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "CurrentMover/CurrentMover_c.h"
#include "OSSMTimeValue/OSSMTimeValue_c.h"
#include "GridVel.h"

#ifdef pyGNOME
	#define TOSSMTimeValue OSSMTimeValue_c
	#define TMap Map_c
	#define TGridVel GridVel_c
	#define TTriGridVel TriGridVel_c
#endif

typedef struct {
	Boolean isOptimizedForStep;
	Boolean isFirstStep;
	double 	value;
} TCM_OPTIMZE;

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
	CATSMover_c () { 
		fGrid = 0; 
		timeDep = 0;
	}
	virtual ClassID 	GetClassID () { return TYPE_CATSMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_CATSMOVER) return TRUE; return CurrentMover_c::IAm(id); }	
	void				SetRefPosition (WorldPoint p, long z) { refP = p; refZ = z; }
	void				GetRefPosition (WorldPoint *p, long *z) { (*p) = refP; (*z) = refZ; }
	
	virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	
	
	void				SetTimeDep (TOSSMTimeValue *newTimeDep) { timeDep = newTimeDep; }
	TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
	void				DeleteTimeDep ();
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty, Seconds time);//JLM 5/12/99
	VelocityRec			GetSmoothVelocity (WorldPoint p);
	//OSErr				ComputeVelocityScale ();
	virtual OSErr		ComputeVelocityScale ();
	virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType, Seconds time);	
	virtual	OSErr 	ReadTopology(char* path, TMap **newMap);
	
	
};

#undef TGridVel
#undef TTriGridVel
#undef TOSSMTimeValue
#undef TMap
#endif