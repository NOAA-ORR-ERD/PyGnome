/*
 *  CurrentMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/22/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CurrentMover_c__
#define __CurrentMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "Mover_c.h"
#include "GEOMETRY.H"
#include "ExportSymbols.h"

class DLL_API CurrentMover_c : virtual public Mover_c {
	
public:
	LONGH			fLESetSizesH;			// cumulative total num le's in each set
	LEUncertainRecH	fUncertaintyListH;		// list of uncertain factors list elements of type LEUncertainRec
	Boolean bIsFirstStep;
	Seconds fModelStartTime;
	
public:
	double			fDownCurUncertainty;	
	double			fUpCurUncertainty;	
	double			fRightCurUncertainty;	
	double			fLeftCurUncertainty;	
	
	Boolean			bIAmPartOfACompoundMover;
	Boolean			bIAmA3DMover;
	
#ifndef pyGNOME
	CurrentMover_c (TMap *owner, char *name);
#endif
	CurrentMover_c ();
	virtual			   ~CurrentMover_c () { Dispose (); }
	virtual void		Dispose ();
	virtual void 		UpdateUncertaintyValues(Seconds elapsedTime);
	virtual OSErr		UpdateUncertainty(const Seconds& elapsedTime, int numLESets, int* LESetsSizesList);
	virtual OSErr		AllocateUncertainty (int numLESets, int* LESetsSizesList);
	virtual OSErr		ReallocateUncertainty(int numLEs, short* statusCodes);	
	virtual void		DisposeUncertainty ();
	
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	
	//temp fix
	virtual WorldRect GetGridBounds(){WorldRect theWorld = { -360000000, -90000000, 360000000, 90000000 }; return theWorld;}	
	//virtual WorldRect GetGridBounds(){return theWorld;}	
	virtual float		GetArrowDepth(){return 0.;}
	virtual Boolean		IAmA3DMover(){return false;}
	//virtual ClassID 	GetClassID () { return TYPE_CURRENTMOVER; }
	//virtual Boolean		IAm(ClassID id) { if(id==TYPE_CURRENTMOVER) return TRUE; return Mover_c::IAm(id); }
	
};

#endif
