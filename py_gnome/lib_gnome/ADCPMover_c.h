/*
 *  ADCPMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADCPMover_c__
#define __ADCPMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "CurrentMover_c.h"
#include "ADCPTimeValue.h"
#include "GridVel.h"
#include "CMYLIST.H"

/////////////////////////////////////////////////
// JLM 11/25/98
// structure to help reset stuff when the user cancels from the uncertainty dialog box
typedef struct
{
	Seconds			fUncertainStartTime;
	double			fDuration; 				// duration time for uncertainty;
	/////
	WorldPoint 		refP; 					// location of tide station or map-join pin
	long 				refZ; 					// meters, positive up
	short 			scaleType; 				// none, constant, or file
	double 			scaleValue; 			// constant value to match at refP
	char 				scaleOtherFile[32]; 	// file to match at refP
	double 			refScale; 				// multiply current-grid value at refP by refScale to match value
	//Boolean			bTimeFileActive;		// active / inactive flag
	Boolean 			bShowGrid;
	Boolean 			bShowArrows;
	double 			arrowScale;
	double			fEddyDiffusion;		
	double			fEddyV0;			
	double			fDownCurUncertainty;	
	double			fUpCurUncertainty;	
	double			fRightCurUncertainty;	
	double			fLeftCurUncertainty;	
} ADCPDialogNonPtrFields;

class ADCPMover_c :  virtual public CurrentMover_c {

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
	//ADCPTimeValue *timeDep;
	double			fEddyDiffusion;			// cm**2/s minimum eddy velocity for uncertainty
	double			fEddyV0;			//  in m/s, used for cutoff of minimum eddy for uncertainty
	Rect			fLegendRect;
	long			fBinToUse;
	TCM_OPTIMZE fOptimize; // this does not need to be saved to the save file
	CMyList		*timeDepList;
	
	OSErr				AddTimeDep(ADCPTimeValue *theTimeDep, short where);
	OSErr				DropTimeDep(ADCPTimeValue *theTimeDep);
	ADCPTimeValue *		AddADCP(OSErr *err);
	
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	void				SetRefPosition (WorldPoint p, long z) { refP = p; refZ = z; }
	void				GetRefPosition (WorldPoint *p, long *z) { (*p) = refP; (*z) = refZ; }
	
	virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	
	
	//void				SetTimeDep (ADCPTimeValue *newTimeDep) { timeDep = newTimeDep; }
	//ADCPTimeValue		*GetTimeDep () { return (timeDep); }
	//void				DeleteTimeDep ();
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);
	VelocityRec			GetVelocityAtPoint(WorldPoint3D p);
	OSErr       ComputeVelocityScale(const Seconds& model_time);
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); // AH 07/10/2012
	virtual void 		ModelStepIsDone();
	virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *velStr);


};

#endif
