/*
 *  CurrentMover_c.cpp
 *  c_gnome
 *
 *  Created by Generic Programmer on 2/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "CurrentMover_c.h"
#ifdef pyGNOME
#define TMap Map_c
#endif

CurrentMover_c::CurrentMover_c (TMap *owner, char *name) : Mover_c(owner, name)
{
	// set fields of our base class
	fDuration=48*3600; //48 hrs as seconds 
	fUncertainStartTime = 0;
	fTimeUncertaintyWasSet = 0;
	
	fDownCurUncertainty = -.3;  // 30%
	fUpCurUncertainty = .3; 	
	fRightCurUncertainty = .1;  // 10%
	fLeftCurUncertainty= -.1; 
	
	fLESetSizesH = 0;
	fUncertaintyListH = 0;
	
	bIAmPartOfACompoundMover = false;
	bIAmA3DMover = false;
}	