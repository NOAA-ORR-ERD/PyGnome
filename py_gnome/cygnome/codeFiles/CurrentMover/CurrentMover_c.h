/*
 *  CurrentMover_c.h
 *  c_gnome
 *
 *  Created by Generic Programmer on 2/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CurrentMover_c__
#define __CurrentMover_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "Mover/Mover_c.h"

#ifdef pyGNOME
	#define TMap Map_c
#endif

class CurrentMover_c : virtual public Mover_c {
	
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

	CurrentMover_c (TMap *owner, char *name);
	CurrentMover_c () {}
	virtual ClassID 	GetClassID () { return TYPE_CURRENTMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_CURRENTMOVER) return TRUE; return Mover_c::IAm(id); }
#ifndef pyGNOME	
	virtual WorldRect GetGridBounds(){return theWorld;}	
#else
	virtual WorldRect GetGridBounds() { WorldRect t = {0.,0.,1.,1.}; return t;}
#endif
	virtual float		GetArrowDepth(){return 0.;}	
	virtual OSErr 		ReadTopology(char* path, TMap **newMap)	{return 2;}

	
};

#undef TMap
#endif