/*
 *  CATSMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CATSMover_c__
#define __CATSMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "CurrentMover_c.h"
#include "ExportSymbols.h"

#ifndef pyGNOME
#include "TOSSMTimeValue.h"
#include "TMap.h"
#include "GridVel.h"
#else
#include "OSSMTimeValue_c.h"
#include "GridVel_c.h"
//#include "Map_c.h"
#define TOSSMTimeValue OSSMTimeValue_c
#define TGridVel GridVel_c
//#define TMap Map_c
#endif

//class CATSMover_c : virtual public CurrentMover_c {
class DLL_API CATSMover_c : virtual public CurrentMover_c {

public:
	TGridVel		*fGrid;					//VelocityH		grid; 
	WorldPoint3D	refPt3D;
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
	float 			arrowDepth;
	Boolean			bApplyLogProfile;
	TOSSMTimeValue *timeDep;
	double			fEddyDiffusion;			// cm**2/s minimum eddy velocity for uncertainty
	double			fEddyV0;			//  in m/s, used for cutoff of minimum eddy for uncertainty
	TCM_OPTIMZE fOptimize; // this does not need to be saved to the save file	
	
#ifndef pyGNOME
						CATSMover_c (TMap *owner, char *name);
#endif
						CATSMover_c ();
	virtual			   ~CATSMover_c () { Dispose (); }
	virtual void		Dispose ();
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	void				SetRefPosition(WorldPoint3D pos) {this->refPt3D = pos; }
	void				GetRefPosition (WorldPoint3D *pos) { (*pos) = this->refPt3D; }
	WorldPoint3D		GetRefPosition () { return this->refPt3D; }	// overloaded for pyGnome
	virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	
	void				SetTimeDep (TOSSMTimeValue *newTimeDep);
	TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
	void				DeleteTimeDep ();
	VelocityRec			GetPatValue (WorldPoint3D p);
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint3D p,Boolean * useEddyUncertainty);//JLM 5/12/99
	VelocityRec			GetSmoothVelocity (WorldPoint p);
	virtual OSErr       ComputeVelocityScale(const Seconds& model_time);
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); // AH 07/10/2012
	virtual void 		ModelStepIsDone();
	virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *velStr);

	virtual	OSErr TextRead(vector<string> &linesInFile);
	virtual	OSErr TextRead(char* path);

	OSErr get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID);

};

#undef TOSSMTimeValue
#undef TGridVel
//#undef TMap
#endif
