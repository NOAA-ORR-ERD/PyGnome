/*
 *  Model_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 1/6/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Model_c__
#define __Model_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "GuiTypeDefs.h"
#include "CMYLIST.H"
#include "ClassID_c.h"
#include <string>
#include <map>

using std::map;
using std::string;

#ifdef pyGNOME
#define TModel Model_c
#define TMap Map_c
#define TMover Mover_c
#define TWindMover WindMover_c
#define TRandom Random_c
#define TRandom3D Random3D_c
#define TCurrentMover CurrentMover_c
#define TLEList LEList_c
#define TOLEList OLEList_c
#define LocaleWizard LocaleWizard_c
#endif

class TModel;
class TMap;
class TMover;
class TWindMover;
class TRandom;
class TRandom3D;
class TCurrentMover;
class LocaleWizard;
class TLEList;

class Model_c : virtual public ClassID_c {

protected:
	
	char		fSaveFileName [255];			
	char		fOutputFileName [255];
	Seconds 	fOutputTimeStep;
	Boolean 	fWantOutput;
	
public:
	TModelDialogVariables fDialogVariables;
	long		nextKeyValue; 	// next available key value for new LE's
	Seconds		modelTime; 		// current model time
	Seconds		lastComputeTime;// the latest time for which the model has computed output that is still valid
	Seconds		LEDumpInterval;	// interval at which LE's are saved to disk for time-bar support
	Boolean		bSaveRunBarLEs;// flag to indicate if model LEs are to be dumped for run-bar support 
	Boolean 	bLEsDirty;		// LE's have changed if true. Used to determine when to initialize uncertainty arrays
	// DO NOT SAVE THIS VARIABLE IN SAVE FILE. MUST BE TRUE WHEN TMODEL IS INSTANTIATED
	
	long		modelMode;		// advanced, novice, etc. modes
	Boolean 	fSettingsOpen;	// model settings being displayed?
	Boolean 	fSpillsOpen;	// spills being displayed?
	Boolean 	bMassBalanceTotalsOpen;	// mass balance totals being displayed?
	Boolean 	mapsOpen;		// individual maps being displayed?
	Boolean 	fOverlaysOpen;		// individual overlays being displayed?
	Boolean 	uMoverOpen;		// individual movers being displayed?
	Boolean 	weatheringOpen;	// weathering being displayed?
	Boolean 	fDrawMovement;
	float		fMaxDuration;	// maybe this should be a global ?
	
	Boolean 	fRunning; 		// we don't need to save this in the Save file
	Boolean		bMakeMovie;		// flag indicating if movies is being generated during the run
	short		movieFrameIndex;// index of next movie frame
	char		fMoviePicsPath[256];// path specification for folder containing the movie pict frames
	char		fMoviePath[256];	// movie file path
	
	
	Boolean 	bSaveSnapshots;	// flag indicating if pictures are being saved during the run
	Seconds 	fTimeOffsetForSnapshots;	// so user can decide when to start the output
	char 		fSnapShotFileName[256];
	
	Boolean bHindcast, writeNC, ncSnapshot;
	int ncID, ncID_C, /*stepsCount, */currentStep, outputStepsCount;
	map<string, int> ncVarIDs, ncDimIDs, ncVarIDs_C, ncDimIDs_C;
	char ncPath[256], ncPathConfidence[256];
	

	
	CMyList	*fSquirreledLastComputeTimeLEList;
	CMyList	*LESetsList;
	CMyList	*mapList;
	TMap	*uMap;
	CMyList	*weatherList;
	LocaleWizard *fWizard;
	CMyList	*fOverlayList; // JLM 6/4/10
	
	CMyList *LEFramesList;
	CMyList *frameMapList;		// copy of map list used to detect changes in actual map-list
	
	Model_c	(Seconds start);
	Model_c	() {}
	void				SetStartTime (Seconds newStartTime) { fDialogVariables.startTime = newStartTime; }
	Seconds				GetStartTime () { return fDialogVariables.startTime; }
	void				SetDuration (Seconds newDuration) { fDialogVariables.duration = newDuration; }
	Seconds				GetDuration () { return fDialogVariables.duration; }
	Seconds				GetEndTime () { return fDialogVariables.startTime + fDialogVariables.duration; }
	void				SetModelTime (Seconds newModelTime);
	Seconds				GetModelTime () { return modelTime; }
	void				SetTimeStep (Seconds newTimeStep) { fDialogVariables.computeTimeStep = newTimeStep; }
	Seconds				GetTimeStep () { return fDialogVariables.computeTimeStep; }
	Boolean				PreventLandJumping () { return fDialogVariables.preventLandJumping; }
	void				SetPreventLandJumping (Boolean bTrueFalse) { fDialogVariables.preventLandJumping = bTrueFalse; }
	Boolean				WantOutput () { return fWantOutput; }
	void				SetWantOutput (Boolean bTrueFalse) { fWantOutput = bTrueFalse; }
	void				SetOutputStep (Seconds newOutputStep) { fOutputTimeStep = newOutputStep; }
	Seconds				GetOutputStep () { return fOutputTimeStep; }
	Seconds 			GetRunDuration () { return modelTime - fDialogVariables.startTime; }		// zero-based current time
	 TMover*				GetMover(char* moverName);
	 TMover*				GetMover(ClassID desiredClassID);
	TMover*				GetMoverExact(ClassID desiredClassID);
	 TMap*				GetMap(char* mapName);
	 TMap*				GetMap(ClassID desiredClassID);
	 TLEList*			GetMirroredLEList(TLEList* owner);
	 TLEList*			GetLEListOwner(TLEList* mirroredLEList);
	 long				GetNumMovers(ClassID desiredClassID);
	 long				GetNumWindMovers();	
	 TWindMover*			GetWindMover(Boolean createIfDoesNotExist);
	 Boolean				ThereIsA3DMover(float *arrowDepth);
	Boolean				ThereIsASubsurfaceSpill();
	 TRandom*			GetDiffusionMover(Boolean createIfDoesNotExist);
	 TRandom3D*			Get3DDiffusionMover();
	 TCurrentMover*		GetPossible3DCurrentMover();
	 Boolean			IsUncertain () { return fDialogVariables.bUncertain; }

	 virtual	OSErr	GetTotalAmountStatistics(short desiredMassVolUnits,double *amtTotal,double *amtReleased,double *amtEvaporated,
										   double *amtDispersed,double *amtBeached,double * amtOffmap, double *amtFloating, double *amtRemoved);
	// map methods
	 TMap				*GetBestMap (WorldPoint p);
	 Boolean			IsWaterPoint(WorldPoint p);
	 Boolean			IsAllowableSpillPoint(WorldPoint p);
	 Boolean			HaveAllowableSpillLayer(WorldPoint p);
	 Boolean			CurrentBeachesLE(WorldPoint3D startPoint, WorldPoint3D *movedPoint, TMap *bestMap);
	 WorldPoint3D		TurnLEAlongShoreLine(WorldPoint3D waterPoint, WorldPoint3D beachedPoint, TMap *bestMap);
	 long				GetMapCount () { return mapList -> GetItemCount (); }
	 WorldRect			GetMapBounds (void);
	 long 				NumLEs(LETYPE  leType);
	 long 				NumOutputSteps(Seconds outputTimeStep);
	
	void				ResetMainKey ();
	long				GetNextLEKey ();
	void				ResetLEKeys ();
	
	void				ReDisperseOil(LERec* thisLE, double breakingWaveHeight);
	void				PossiblyReFloatLE (TMap *theMap, TLEList *theLEList, long i, LETYPE leType);
	void 				DisperseOil(TLEList* theLEList, long index);
	void 				UpdateWindage(TLEList* theLEList);
	OSErr 				TellMoversPrepareForRun();
	OSErr 				TellMoversPrepareForStep();
	void 				TellMoversStepIsDone();
	
	OSErr 				GetTotalBudgetTableHdl(short desiredMassVolUnits, BudgetTableDataH *totalBudgetTable);
	OSErr 				GetTotalAmountSpilled(short desiredMassvolUnits,double *amtTotal);
	Boolean				ThereIsAnEarlierSpill(Seconds timeOfInterest, TLEList *someLEListToIgnore);
	Boolean				ThereIsALaterSpill(Seconds timeOfInterest, TLEList *someLEListToIgnore);
	
	CMyList				*GetWeatherList () { return weatherList; }
	
	//++ Replacements:
	
	void				NewDirtNotification(void) { return; }
	void				NewDirtNotification (long flags) { return; }
	
};


#undef TModel 
#undef TMap 
#undef TMover 
#undef TWindMover 
#undef TRandom 
#undef TRandom3D
#undef TCurrentMover
#undef TLEList
#undef TOLEList
#undef LocaleWizard
#endif
