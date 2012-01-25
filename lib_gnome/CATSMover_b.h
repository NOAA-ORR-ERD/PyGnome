/*
 *  CATSMover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CATSMover_b__
#define __CATSMover_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "CurrentMover/CurrentMover_b.h"
#include "GridVel.h"
#include "OSSMTimeValue/TOSSMTimeValue.h"

class CATSMover_b : virtual public CurrentMover_b {

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
	
};

#endif
