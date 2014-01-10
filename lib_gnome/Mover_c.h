/*
 *  Mover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Mover_c__
#define __Mover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "ClassID_c.h"
#include "RectUtils.h"
//#include "Map_c.h"
#include "ExportSymbols.h"

#ifdef pyGNOME
//#define TMap Map_c
#endif
#ifndef pyGNOME
#include "Map_c.h"
class TMap;
#endif

class DLL_API Mover_c : virtual public ClassID_c {

public:
#ifndef pyGNOME
	TMap				*moverMap;			// mover's owner
#endif
	Seconds				fUncertainStartTime;
	double				fDuration; 				// duration time for uncertainty;
	//RGBColor			fColor;
	
protected:
	double				fTimeUncertaintyWasSet;	// time to measure next uncertainty update

public:
#ifndef pyGNOME
	Mover_c (TMap *owner, char *name);
#endif
	Mover_c ();
	virtual				~Mover_c();
	virtual void		Dispose () {}

	virtual OSErr		AddUncertainty (long setIndex, long leIndex, VelocityRec *v) { return 0; }
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType); 
	
	virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *velStr) {return false;}
	virtual float		GetArrowDepth(){return 0.;}
	virtual LongPointHdl	GetPointsHdl(){return nil;}
	virtual OSErr 		PrepareForModelRun() { return noErr; } 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList) { return noErr; } // AH 07/10/2012

	virtual OSErr		UpdateUncertainty(void);
#ifndef pyGNOME
	void				GetMoverName (char *name) { GetClassName (name); }
	void				SetMoverName (char *name) { SetClassName (name); }
	TMap				*GetMoverMap () { return moverMap; }
	void				SetMoverMap (TMap *owner) { moverMap = owner; }
#endif
	virtual void 		ModelStepIsDone(){ return; }
	virtual OSErr 		ReallocateUncertainty(int numLEs, short* LE_Status){ return 0; }
	virtual Boolean		IAmA3DMover() {return false;}
	//virtual ClassID 	GetClassID () { return TYPE_MOVER; }
	//virtual Boolean		IAm(ClassID id) { if(id==TYPE_MOVER) return TRUE; return ClassID_c::IAm(id); }
	
};


//#undef TMap
#endif
