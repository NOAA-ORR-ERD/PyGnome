/*
 *  Mover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Mover_b__
#define __Mover_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "ClassID/ClassID_b.h"

class TMap;

class Mover_b : virtual public ClassID_b {

public:
	TMap				*moverMap;			// mover's owner
	Seconds				fUncertainStartTime;
	double				fDuration; 				// duration time for uncertainty;
	RGBColor			fColor;
	
protected:
	double				fTimeUncertaintyWasSet;	// time to measure next uncertainty update
	
};


#endif