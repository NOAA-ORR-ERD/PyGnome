/*
 *  CATSMover3D_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CATSMover3D_c__
#define __CATSMover3D_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "CATSMover_c.h"

#ifdef pyGNOME
#include "TriGridVel3D_c.h"
#define TTriGridVel3D TriGridVel3D_c
#endif

class CATSMover3D_c : virtual public CATSMover_c {

public:
	TTriGridVel3D	*fRefinedGrid;			// store a second grid for contouring
	Boolean			bShowDepthContours;	// this should be a map item I think
	Boolean			bShowDepthContourLabels;
	
	/*WorldPoint 		refP; 					// location of tide station or map-join pin
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
	 public:
	 TCM_OPTIMZE fOptimize; // this does not need to be saved to the save file
	 */
	
	//virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	
	//virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
	
	//virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
	
	/*void				SetRefPosition (WorldPoint p, long z) { refP = p; refZ = z; }
	 void				GetRefPosition (WorldPoint *p, long *z) { (*p) = refP; (*z) = refZ; }
	 
	 void				SetTimeDep (TOSSMTimeValue *newTimeDep) { timeDep = newTimeDep; }
	 TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
	 void				DeleteTimeDep ();
	 VelocityRec			GetPatValue (WorldPoint p);
	 VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	 VelocityRec			GetSmoothVelocity (WorldPoint p);
	 OSErr       ComputeVelocityScale(const Seconds& model_time);
	 virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	 */		
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, const Seconds&, bool); // AH 04/16/12
	//virtual void 		ModelStepIsDone();
	virtual	Boolean 		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	
	//LongPointHdl 		GetPointsHdl(Boolean useRefinedGrid);
	virtual 			LongPointHdl 		GetPointsHdl();
	

};

#endif

