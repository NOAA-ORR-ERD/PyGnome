/*
 *  CompoundMover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CompoundMover_b__
#define __CompoundMover_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "CurrentMover_b.h"
#include "Uncertainty.h"

class CMyList;

class CompoundMover_b : virtual public CurrentMover_b {

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

};

#endif
