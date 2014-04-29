/*
 *  TModel.h
 *  gnome
 *
 *  Created by Generic Programmer on 1/6/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TModel__
#define __TModel__

#include "Model_c.h"
#include "TClassID.h"
#include <vector>

using std::vector;
using std::pair;

class TOverlay;
class TWeatherer;

class TModel : virtual public Model_c,  public TClassID
{
	
public:
	
	// bitmap version of map at current view and window size without LEs or movement grid
#ifdef MAC
	CGrafPtr mapImage;		
#else 		
	HDIB	mapImage;		
#endif
	
	TModel(Seconds start);
	virtual		   ~TModel () { Dispose (); }
	virtual OSErr	InitModel ();
	virtual void	Dispose ();
	virtual ClassID GetClassID () { return TYPE_MODEL; }
	virtual Boolean	IsDirty ();
	
	void			DisposeLEFrames ();
	void			DisposeModelLEs ();
	void			DisposeAllMoversOfType(ClassID desiredClassID);
	
	
	TWindMover* GetNthWindMover(long desiredNum0Relative);

	
	// drawing movement utils
	void DrawLEMovement(void);

	void DrawLegends(Rect r, WorldRect wRect); 
	
	// messages to the model
	void SuppressDirt (long suppressDirtFlags);
	void NewDirtNotification(void);
	void NewDirtNotification (long flags);
	
	OSErr BroadcastMessage(long messageCode, char* targetName, UNIQUEID targetUniqueID, char* dataStr, CHARH dataHdl);
	OSErr BroadcastMessage(long messageCode, char* targetName, char* dataStr, CHARH dataHdl);
	OSErr BroadcastMessage(long messageCode, UNIQUEID uid, char* dataStr, CHARH dataHdl);
	void BroadcastToSelectedItem(long messageCode, char* dataStr, CHARH dataHdl);
	
	virtual OSErr 	CheckAndPassOnMessage(TModelMessage *message);
	long 	GetNumForecastSpills();
	OSErr HandleRunSpillMessage(TModelMessage *message);
	// JLM 6/25/10 break up RUNSPILL in to making createspill and a RUN mesage
	OSErr HandleRunMessage(TModelMessage *message);
	OSErr HandleCreateSpillMessage(TModelMessage *message);
	
	
	OSErr   AddOverlay(TOverlay *theOverlay, short where);
	OSErr   DropOverlay(TOverlay *theOverlay);
	void	DrawOverlays(Rect r, WorldRect wRect);
	
	
	// model mode get and set methods
	void	SetModelMode (long theMode);
	long	GetModelMode () { return modelMode; }
	char*  	GetModelModeStr(char *modelModeStr);
	
	// LE methods
	CMyList	*GetLESetsList () { return (LESetsList); }
	OSErr	AddLEList (TLEList *theLEList, short where);
	OSErr	DropLEList (TLEList *theLEList, Boolean bDispose);
	

	OSErr	AddWeatherer (TWeatherer *theWeatherer, short where);
	OSErr	DropWeatherer (TWeatherer *theWeatherer);
	
	CMyList	*GetMapList () { return mapList; }
	OSErr	AddMap (TMap *theMap, short where);
	OSErr	DropMap (TMap *theMap);

	void	SetLastComputeTime(Seconds time);

	
	Boolean UserIsEditingSplots(void);
	long 	NumEditableSplotObjects(void);
	void 	SelectAnEditableSplotObject(void);
	Boolean 	EditableSplotObjectIsSelected();
	TClassID *ItemBeingEditedInMappingWindow(void);
	OSErr 	DropObject(TClassID  *object, Boolean bDispose);
	OSErr 	EditObjectInMapDrawingRect(TClassID *newObjectToEdit);
	void	CheckEditModeChange(void);
	

	
	// model run methods
	OSErr				Reset ();
	OSErr				Run (Seconds stopTime);
	OSErr 				FirstStepUserInputChecks(void);
	OSErr				Step ();
	OSErr				StepBackwards ();
	OSErr				StepBack ();
	OSErr 				RunTill(void);
	OSErr				RunTill(Seconds stopTime);
	void				Weather ();
	void				CleanUp ();
	
	// I/O methods
	virtual OSErr 		Read  (BFPB *bfpb); 		// read from the current position
	virtual OSErr 		Write (BFPB *bfpb); 		// write to  the current position
	virtual OSErr		ReadFromPath (char *path);	// creates a buffer, then calls Read  () 
	virtual OSErr		WriteToPath  (char *path);	// creates a buffer, then calls Write ()
	
	virtual void		Draw(Rect r, WorldRect view);
	virtual long		GetListLength();
	virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
	virtual OSErr 		AddItem(ListItem item);
	virtual OSErr 		SettingsItem(ListItem item);
	virtual OSErr 		DeleteItem(ListItem item);
	
	void				SetDialogVariables (TModelDialogVariables var);
	TModelDialogVariables GetDialogVariables () { return fDialogVariables; }
	void				SetUncertain (Boolean bNewUncertain); //JLM 9/1/98 
	Boolean 			DrawingDependsOnTime(void);
	void				SetSaveFileName (char *newName) { strcpy (fSaveFileName, newName); }
	void				GetSaveFileName (char *theName) { strcpy (theName, fSaveFileName); }
	void				SetOutputFileName (char *newName) { strcpy (fOutputFileName, newName); }
	void				GetOutputFileName (char *theName) { strcpy (theName, fOutputFileName); }
	OSErr				GetOutputFileName (short fileNumber, short typeCode, char *fileName);
	OSErr 				GetTempLEOutputFilePathName(short fileNumber, char* path,short *vRefNumPtr);//JLM
	OSErr 				GetTempLEOutputFilePathNameTake2(short fileNumber, char* path,short *vRefNumPtr);//JLM
	OSErr				SaveOSSMLEFile (Seconds fileTime, short n);
	OSErr				SaveKmlLEFile (Seconds fileTime, short n);
	OSErr				SaveKmlLEFileSeries (Seconds fileTime, short n);
	OSErr				FinishKmlFile ();
	OSErr				SaveMossLEFile (Seconds fileTime, short n);
	OSErr				SaveSimpleAsciiLEFile (Seconds fileTime, short fileNumber);
	Seconds				GetLastComputeTime () { return lastComputeTime; }
	short 				GetMassUnitsForTotals();
	void				MovementString(WorldPoint3D wp,char* str);

	OSErr 				ExportBudgetTableHdl(char* path);
	
	OSErr 				CheckMaxModelDuration(float durationHours,char * errStr);
	
	OSErr				OpenMovieFile ();
	OSErr				CloseMovieFile ();
	
	OSErr 				SquirrelAwayLastComputedTimeStuff (void);
	OSErr 				ReinstateLastComputedTimeStuff (void);
	OSErr 				DisposeLastComputedTimeStuff (void);
	OSErr 				TemporarilyShowFutureTime (Seconds futureTime);
	OSErr 				SetModelToPastTime (Seconds pastTime);
	double				GetVerticalMove(LERec *theLE);
	
private:
	long				GetLineIndex (long lineNum);
	void				ReleaseLEs ();

	
	Seconds 			ClosestSavedModelLEsTime(Seconds givenTime);
	Seconds 			NextSavedModelLEsTime(Seconds givenTime);
	Seconds 			PreviousSavedModelLEsTime(Seconds givenTime);
	OSErr				SaveModelLEs (Seconds forTime, short fileNumber);
	OSErr				SaveModelLEs (BFPB *bfpb);
	OSErr				LoadModelLEs (char *LEFileName);
	OSErr				LoadModelLEs (BFPB *bfpb);
	OSErr				LoadModelLEs (Seconds forTime, Seconds *actualTime);

	void				UpdateFrameMapList ();
	Boolean				HasFrameMapListChanged ();
	
	OSErr				SaveMovieFrame (WindowPtr movieWindow, Rect frameRect);
	OSErr 				SaveOutputSeriesFiles(Seconds oldTime,Boolean excludeRunBarFile);
	
	OSErr 				WriteRunSpillOutputFileHeader(BFPB *bfpb,Seconds outputStep,char* noteStr);
	OSErr 				AppendLEsToRunSpillOutputFile(BFPB *bfpb);
	OSErr				move_spills(vector<WorldPoint3D> **, vector<LERec *> **, vector< pair<bool, bool> > **, vector< pair<int, int> > **);
	OSErr				check_spills(vector<WorldPoint3D> *, vector <LERec *> *, vector< pair<bool, bool> > *, vector< pair<int, int> > *);

};

#endif
