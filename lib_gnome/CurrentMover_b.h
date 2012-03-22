/*
 *  CurrentMover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/22/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CurrentMover_b__
#define __CurrentMover_b__

#include "Mover_b.h"

class CurrentMover_b : virtual public Mover_b {

protected:
	LONGH			fLESetSizesH;			// cumulative total num le's in each set
	LEUncertainRecH	fUncertaintyListH;		// list of uncertain factors list elements of type LEUncertainRec

	public:
	double			fDownCurUncertainty;	
	double			fUpCurUncertainty;	
	double			fRightCurUncertainty;	
	double			fLeftCurUncertainty;	

	Boolean			bIAmPartOfACompoundMover;
	Boolean			bIAmA3DMover;

};


#endif
