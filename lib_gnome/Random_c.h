/*
 *  Random_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Random_c__
#define __Random_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "Mover_c.h"
#include "ExportSymbols.h"

class DLL_API Random_c : virtual public Mover_c {
	
public:
	double fDiffusionCoefficient; //cm**2/s
	TR_OPTIMZE fOptimize; // this does not need to be saved to the save file
	double fUncertaintyFactor;		// multiplicative factor applied when uncertainty is on
	Boolean bUseDepthDependent;
	
#ifndef pyGNOME
	Random_c (TMap *owner, char *name);
#endif
	Random_c();
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); // AH 07/10/2012
	virtual void 		ModelStepIsDone();
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	
	
	OSErr				get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID);

protected:
	void				Init();
};

#endif
