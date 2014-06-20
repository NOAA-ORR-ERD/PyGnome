#include "CROSS.H"
#include "NetCDFStore.h"
#include "MYRANDOM.H"
#include "TimUtils.h"
#include "MakeMovie.h"
#include "TShioTimeValue.h"
#include "TideCurCycleMover.h"
//#include "shapefil.h"

#include <vector>


#ifdef MAC
#ifdef MPW
#pragma SEGMENT TMODEL
#endif
#endif

using std::vector;
using std::pair;

enum { I_MODELSETTINGS = 0, I_STARTTIME, I_ENDTIME, I_COMPUTESTEP, I_OUTPUTSTEP, I_UNCERTAIN,I_UNCERTAIN2, I_DRAWLEMOVEMENT, I_PREVENTLANDJUMPING, I_HINDCAST, I_DSTDISABLED=11};

void RefreshMapWindowContents();

char gRunSpillNoteStr[256];
BFPB gRunSpillForecastFile;


// global variables for drawing to map window
Boolean			sharedPrinting = FALSE;
short			aX, bX, aY, bY;
long			AX, BX, AY, BY, DX, DY;
WorldRect		sharedView;
Rect		 	gRect;
PixMapHandle 	screenImage = nil;
Rect		 	screenImageRect = {0,0,0,0};
Rect		 	mapImageRect = {0,0,0,0};
///////////////////////////////////////////////////////////////////////////

Seconds TimeDifference(Seconds a,Seconds b)
{ 	// because these are unsigned longs
	// I worry about using abs, JLM 1/7/99
	if(a >= b) return a-b;
	else return b-a;
}

void TModelOutOfMemoryAlert(char* routineName)
{
	char msg[512];
	char * name = "";
	if(routineName) name = routineName; // re-assign pointer
	sprintf(msg,"There is not enough memory allocated to the program.  Out of memory in TModel %s.",name
	);
	printError(msg);
}



TModelDialogVariables DefaultTModelDialogVariables(Seconds start)
{
	TModelDialogVariables var;
	memset(&var,0,sizeof(TModelDialogVariables));
	var.startTime = start;
	var.duration =  24 * 3600;
	var.computeTimeStep = 15 * 60;
	var.bUncertain = FALSE;
	var.preventLandJumping = true;
	

	return var;
}


Boolean EqualTModelDialogVariables(TModelDialogVariables* var1,TModelDialogVariables *var2)
{
	if(var1->startTime != var2->startTime) return false;
	if(var1->duration != var2->duration) return false;
	if(var1->computeTimeStep != var2->computeTimeStep) return false;
	if(var1->bUncertain != var2->bUncertain) return false;
	if(var1->preventLandJumping != var2->preventLandJumping) return false;
	return true;
}

TModel::TModel(Seconds start)
{
	//stepsCount = 0;
	outputStepsCount = 0;
	ncSnapshot = false;
	writeNC = false;
	fDrawMovement = 0;//JLM
	fWizard = nil;
	fSquirreledLastComputeTimeLEList = nil;
	LESetsList = nil;
	mapList = nil;
	fOverlayList = nil;
	uMap = nil;
	weatherList = nil;
	LEFramesList = nil;
	mapImage = nil;
	frameMapList = nil;
	movieFrameIndex = 0;
	modelMode = ADVANCEDMODE;

	fDialogVariables = DefaultTModelDialogVariables(start);
	strcpy (fSaveFileName, kDefSaveFileName);		// sohail
	fOutputFileName[0] = 0;
	fOutputTimeStep = 3600;
	fWantOutput = FALSE;
	
	modelTime = fDialogVariables.startTime;
	lastComputeTime = fDialogVariables.startTime;
	
	bSaveRunBarLEs = true;
	LEDumpInterval = 3600;	// dump interval for LE's used for run-bar
	
	ResetMainKey();
	SetDirty(FALSE);
	
	fSettingsOpen = TRUE;
	fSpillsOpen = TRUE;
	bMassBalanceTotalsOpen = false;
	mapsOpen = TRUE;
	fOverlaysOpen = TRUE;
	uMoverOpen = TRUE;
	weatheringOpen = TRUE;
	
	fMaxDuration = 3.*24;	// 3 days
						
	// JLM found this comment but no does not believe it, 11/15/99
	// IT MUST ALWAYS START OUT TRUE TO ENSURE 
	// THAT LE UNCERTAINTY ARRAYS GET INITIALIZED
	// bLEsDirty = true;  
	bLEsDirty = false; 
						
	fRunning = FALSE;
	bMakeMovie = FALSE;
	bHindcast = FALSE;

	bSaveSnapshots = false;
	fTimeOffsetForSnapshots = 0;	
	fSnapShotFileName[0] = 0;
}

OSErr TModel::InitModel()
{
	TWeatherer	*weatherer = nil;
	OSErr		err = noErr;
	
	if (!(fWizard = new LocaleWizard())) // JLM
		{ TechError("TModel::InitModel()", "new LocaleWizard()", 0); err = memFullErr; }
		
	if (!err) {
		if (!(fSquirreledLastComputeTimeLEList = new CMyList(sizeof(TLEList *))))
			{ TechError("TModel::InitModel()", "new CMyList()", 0); err = memFullErr; }
		else if (err = fSquirreledLastComputeTimeLEList->IList())
			{ TechError("TModel::InitModel()", "IList()", 0); }
	}
	
	if (!err) {
		if (!(LESetsList = new CMyList(sizeof(TLEList *))))
			{ TechError("TModel::InitModel()", "new CMyList()", 0); err = memFullErr; }
		else if (err = LESetsList->IList())
			{ TechError("TModel::InitModel()", "IList()", 0); }
	}
	
	if (!err) {
		if (!(mapList = new CMyList(sizeof(TMap *))))
			{ TechError("TModel::InitModel()", "new CMyList()", 0); err = memFullErr; }
		else if (err = mapList->IList())
			{ TechError("TModel::InitModel()", "IList()", 0); }
	}

	if (!err) {
		if (!(fOverlayList = new CMyList(sizeof(TOverlay *))))
			{ TechError("TModel::InitModel()", "new CMyList()", 0); err = memFullErr; }
		else if (err = fOverlayList->IList())
			{ TechError("TModel::InitModel()", "IList()", 0); }
	}

	
	
	if (!err) {
		if (!(frameMapList = new CMyList(sizeof(TMap *))))
			{ TechError("TModel::InitModel()", "new CMyList()", 0); err = memFullErr; }
		else if (err = frameMapList->IList())
			{ TechError("TModel::InitModel()", "IList()", 0); }
	}
	
	if (!err) {
		if (!(weatherList = new CMyList (sizeof (TWeatherer*))))
			{ TechError("TModel::InitModel()", "new CMyList()", 0); err = memFullErr; }
		else if (err = weatherList->IList())
			{ TechError("TModel::InitModel()", "IList()", 0); }
	}
	
	if (!err) {
		// create and add new weathering object
		weatherer = new TOSSMWeatherer ("Weathering");
		if (!weatherer)
			{ TechError("AddWeathererDialog()", "new TOSSMWeather()", 0); err = memFullErr; }

		if (err = weatherer->InitWeatherer())
		{
			delete weatherer;
			return err;
		}

		if (!err) {
			if (err = this -> AddWeatherer (weatherer, 0))
				{ TechError("AddWeathererDialog()", "AddWeatherer ()", 0); }
		}
	}
	
	if (!err)  // STH
	{
		if (!(LEFramesList = new CMyList (sizeof (LEFrameRec))))
			{ TechError("TModel::InitModel()", "new LEFramesList ()", 0); err = memFullErr; }
		else if (err = LEFramesList->IList())
			{ TechError("TModel::InitModel()", "IList()", 0); }
	}

	if (!err)
	{
		WorldRect	wRect;

		//wRect.loLong = -179999999;
		wRect.loLong = -359999999;
		//wRect.hiLong =  179999999;
		wRect.hiLong =  359999999;
		wRect.loLat  =  -89999999;
		wRect.hiLat  =   89999999;
		
		if (!(uMap = new TMap ("Universal Map", wRect)))
			{ TechError("TModel::InitModel()", "new TMap()", 0); err = memFullErr; }

		if (err = uMap->InitMap())
			{ TechError("TModel::InitModel()", "InitMap()", 0); err = memFullErr; }
	}
	
	if (err)
		Dispose ();

	return err;
}

void TModel::Dispose()
{
	long 	i, n;
	TMap	*thisMap;
	TMover	*thisMover;

	if (fWizard)
	{
		fWizard->Dispose(); //JLM
		delete fWizard;
		fWizard = 0;
	}
	
	this->DisposeLastComputedTimeStuff(); // JLM 1/7/99
	if (fSquirreledLastComputeTimeLEList != nil)
	{
		fSquirreledLastComputeTimeLEList->Dispose();
		delete fSquirreledLastComputeTimeLEList;
		fSquirreledLastComputeTimeLEList = nil;
	}

	DisposeModelLEs ();		// dispose of all LE's and the model's LE-sets-list, STH	
	if (LESetsList != nil)
	{
		LESetsList->Dispose();
		delete LESetsList;
		LESetsList = nil;
	}

	if (uMap)
	{
		uMap -> Dispose ();
		delete (uMap);
		uMap = nil;
	}
	
	if (mapList)
	{
		for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
			mapList->GetListItem((Ptr)&thisMap, i);
			thisMap->Dispose();
			delete thisMap;
			thisMap = nil;
		}
	
		mapList->Dispose();
		delete mapList;
		mapList = nil;
	}

	if (fOverlayList)
	{
		TOverlay *thisOverlay;
		for (i = 0, n = fOverlayList->GetItemCount() ; i < n ; i++) {
			fOverlayList->GetListItem((Ptr)&thisOverlay, i);
			thisOverlay->Dispose();
			delete thisOverlay;
			thisOverlay = nil;
		}
	
		fOverlayList->Dispose();
		delete fOverlayList;
		fOverlayList = nil;
	}
	

	if (frameMapList)
	{
		frameMapList->Dispose();
		delete frameMapList;
		frameMapList = nil;
	}

	if (weatherList)
	{
		weatherList -> Dispose ();
		delete weatherList;
		weatherList = nil;
	}
	
	if (LEFramesList)
	{
		DisposeLEFrames ();
		LEFramesList -> Dispose ();
		delete (LEFramesList);
		LEFramesList = nil;
	}
	
	if (mapImage)
	{
		#ifdef MAC
			KillGWorld (mapImage);
		#else 
			DestroyDIB(mapImage);
		#endif
		mapImage = nil;
	}
}
		


void TModel::SetUncertain (Boolean bNewUncertain) 
{	//JLM 9/1/98 
	if(bNewUncertain != fDialogVariables.bUncertain)
	{
		// we are changing the value
		Boolean needToReset = true;
		// code goes here
		// should this be based on user level ?
		if(!bNewUncertain)  {
			// JLM , 1/23/01, then we are turning off the uncertainty LEs.
			// Since this does not affect the certainty LEs, we don't need to reset
			needToReset = false;
		}
		if(needToReset)
			this->Reset();
		else 
			this->NewDirtNotification(DIRTY_MAPDRAWINGRECT); // because we need to redraw without the uncertainty LEs 
	}
	fDialogVariables.bUncertain = bNewUncertain;
}

void TModel::DisposeLEFrames ()
// disposes of data stored in the LEFramesList including LE files that were written for
// previous frames / time, STH
{
	OSErr	err = noErr;
	
	if (LEFramesList)
	{
		long		numOfFrames, i;
		LEFrameRec	thisFrame;
		
		numOfFrames = LEFramesList -> GetItemCount ();
		for (i = numOfFrames - 1; i >= 0; --i)
		{
			LEFramesList -> GetListItem ((Ptr) &thisFrame, i);
			hdelete (0, 0, thisFrame.frameLEFName);
			LEFramesList -> DeleteItem (i);
		}
	}

}

void TModel::SetDialogVariables (TModelDialogVariables var)
{ 
	Boolean mustReset = false;
	Boolean modelWasPreviouslyRun = this -> GetModelTime () > this -> GetStartTime();
	Boolean uncertaintyWasChanged = var.bUncertain != fDialogVariables.bUncertain;
	Boolean modelStartTimeHasChanged = var.startTime != fDialogVariables.startTime;
	
	fDialogVariables = var; 
	// JLM, make sure model time is between start time and end time and adjust if necessary
	if (this -> GetModelTime () < this -> GetStartTime ()) mustReset = true; 
	if (this -> GetModelTime () > this -> GetEndTime   ()) mustReset = true; 
	if (this -> GetLastComputeTime () < this -> GetStartTime ()) mustReset = true; 
	if (this -> GetLastComputeTime () > this -> GetEndTime   ()) mustReset = true;
	// make sure LEs and uncertainty set start together, 3/29/00
	if (uncertaintyWasChanged && (modelWasPreviouslyRun))  mustReset = true; 	
	// always reset if they change the start time, 4/6/00
	if (modelStartTimeHasChanged)  mustReset = true; 	

	if(mustReset) this -> Reset();
}



void TModel::Weather()
{
	return;
}

void TModel::CleanUp()
{
	return;
}


OSErr TModel::AddLEList(TLEList *theLEList, short where)
{
	OSErr err = 0;

	if (err = LESetsList->AppendItem((Ptr)&theLEList))
		{ TechError("TModel::AddLEList()", "AppendItem()", err); return err; }
	bLEsDirty = true;
	// code goes here
	// need to check LE release time against runbar to see if we need to reset the runbar
	
	this->NewDirtNotification();
	
	SelectListItemOfOwner(theLEList);
	
	return 0;
}

OSErr TModel::DropLEList(TLEList *theLEList, Boolean bDispose)
{
	long i;
	TLEList *uncertaintyLEs;
	OSErr err = 0;
	
	if(!theLEList) return -1;
	
	uncertaintyLEs = this->GetMirroredLEList(theLEList);
	
	if (!LESetsList->IsItemInList((Ptr)&theLEList, &i)) return -1;
	if (err = LESetsList->DeleteItem(i))
		{ TechError("TModel::DropLEList()", "DeleteItem()", err); return err; }
	
	if(uncertaintyLEs)
	{
		if (LESetsList->IsItemInList((Ptr)&uncertaintyLEs, &i))
			if (err = LESetsList->DeleteItem(i))
				{ TechError("TModel::DropLEList()", "DeleteItem()", err); return err; }
		uncertaintyLEs->Dispose(); // JLM , 9/15/98
		delete uncertaintyLEs; // JLM , 9/15/98
		uncertaintyLEs = nil; // JLM , 9/15/98
	}
	
	if (bDispose)
	{
		theLEList -> Dispose ();
		delete (theLEList);
	}
	
	bLEsDirty = true;
	this->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
	// because of past saved LE sets and uncertainty durations
	this->NewDirtNotification();
	
	return 0;
}

OSErr TModel::DropOverlay(TOverlay *theOverlay)
{
	long i;
	OSErr err = 0;
	
	if(!theOverlay) return -1;
	
	
	if (!fOverlayList->IsItemInList((Ptr)&theOverlay, &i)) return -1;
	if (err = fOverlayList->DeleteItem(i))
		{ TechError("TModel::DropOverlay()", "DeleteItem()", err); return err; }
	
	
	this->NewDirtNotification(DIRTY_LIST | DIRTY_MAPDRAWINGRECT);
	
	return 0;
}

void TModel::DrawOverlays(Rect r, WorldRect wRect) 
{
	TOverlay *thisOverlay;
	long i,n;

	// draw each of the overlays (in reverse order to show priority)		
	for (n = fOverlayList ->GetItemCount() - 1; n >= 0 ; n--) {
		fOverlayList ->GetListItem((Ptr)&thisOverlay, n);
		thisOverlay->Draw(r, wRect);
	}
}

OSErr TModel::AddOverlay(TOverlay *theOverlay, short where)
{
	OSErr err = 0;
	if (err = fOverlayList->AppendItem((Ptr)&theOverlay))
		{ TechError("TModel::AddOverlay()", "AppendItem()", err); return err; }
		
	InvalMapDrawingRect();
	SelectListItemOfOwner(theOverlay);

	return 0;
}




OSErr TModel::AddMap(TMap *theMap, short where)
{
	OSErr err = 0;
	if (err = mapList->AppendItem((Ptr)&theMap))
		{ TechError("TModel::AddMap()", "AppendItem()", err); return err; }
	
	ChangeCurrentView(AddWRectBorders(theMap->GetMapBounds(), 10), TRUE, TRUE);
	
	SelectListItemOfOwner(theMap);

	return 0;
}



OSErr TModel::AddWeatherer(TWeatherer *theWeatherer, short where)
{
	OSErr err = 0;
	if (err = weatherList->AppendItem((Ptr)&theWeatherer))
		{ TechError("TModel::AddWeathering()", "AppendItem()", err); return err; }
	
//	ChangeCurrentView(AddWRectBorders(theMap->bounds, 10), TRUE, TRUE);
	
	return 0;
}

OSErr TModel::DropWeatherer(TWeatherer *theWeatherer)
{
	long i;
	OSErr err = 0;

	if (weatherList->IsItemInList((Ptr)&theWeatherer, &i))
		if (err = weatherList->DeleteItem(i))
			{ TechError("TModel::DropWeatherer()", "DeleteItem()", err); return err; }
	
	return 0;
}

OSErr TModel::DropMap(TMap *theMap)
{
	long i;
	OSErr err = 0;
	
	if (mapList->IsItemInList((Ptr)&theMap, &i))
		if (err = mapList->DeleteItem(i))
			{ TechError("TModel::DropMap()", "DeleteItem()", err); return err; }

	return 0;
}


void TModel::DisposeAllMoversOfType(ClassID desiredClassID)
{
	// loop through each mover in the universal map
	TMover *thisMover = nil;
	TMap *map;
	long i,n,k,d;
	ClassID classID;
	OSErr err = 0;
	
	// universal movers
	d = this->uMap->moverList->GetItemCount ();
	for (k = d-1 ; k >= 0; k--)
	{
		this->uMap->moverList -> GetListItem ((Ptr) &thisMover, k);
		classID = thisMover -> GetClassID ();
		if(classID == desiredClassID) 
		{	// delete this mover
			if (err = this->uMap->moverList->DeleteItem(k))
				{ TechError("TModel::DisposeAllMoversOfType()", "DeleteItem()", err); }
			if(thisMover)
				{thisMover -> Dispose ();delete (thisMover); thisMover = 0;}
		}
	}
	
	// movers that belong to a map
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&map, i);
		d = map -> moverList -> GetItemCount ();
		for (k = d-1 ; k >= 0; k--)
		{
			map -> moverList -> GetListItem ((Ptr) &thisMover, k);
			classID = thisMover -> GetClassID ();
			if(classID == desiredClassID) 
			{
				if (err = map -> moverList ->DeleteItem(k))
					{ TechError("TModel::DisposeAllMoversOfType()", "DeleteItem()", err); }
				if(thisMover)
					{thisMover -> Dispose ();delete (thisMover); thisMover = 0;}
			}
		}
	}
}

OSErr TModel::GetTempLEOutputFilePathName(short fileNumber, char* path,short *vRefNumPtr)
{
	// Get the path and prefix of temporary LE files to be stored to support the Run-Bar, STH
	OSErr		err = noErr;
	char str[32];

#ifdef MAC
	short		vRefNum;
	long		parDirID;
	
	// get the dir-spec info for preferences folder
	err = FindFolder (kOnSystemDisk, kPreferencesFolderType, kCreateFolder, &vRefNum, &parDirID);
	
	if (!err)
	{
		char	dirName [255];
		long	prefsDirID;
		OSType folderType;
		FSSpec dirFSpec;
		
		strcpy (dirName, "GNOME Preferences");
		my_c2pstr (dirName);
		
		err = DirCreate (vRefNum, parDirID, (ConstStr255Param) dirName, &prefsDirID);
		if (!err || err == dupFNErr)
		{
			PathNameFromDirID (prefsDirID, vRefNum, dirName);
			my_p2cstr ((StringPtr) dirName);
			if (err == dupFNErr)
				strcat (dirName, "GNOME Preferences:");
			strcpy (path, dirName);
			sprintf(str, "LEFile.%03hd", fileNumber);
			strcat (path, str);
			*vRefNumPtr = vRefNum;
			err = noErr;
		}
	}
#else // IBM 
	long tempLesDirID; 
	long len, nBufferLen = 255;
	len = GetTempPath(nBufferLen,path);
	*vRefNumPtr = 0; // unused on the IBM
	//GetWindowsDirectory(path, 255);	// on XP Windows only allows administrator to write to the Windows directory
	if (path[strlen(path) - 1] != '\\')
		strcat(path, "\\"); // add backslash
	strcat(path, "GNOME Preferences");
	err = AddFolderIfMissing(0,0,path,&tempLesDirID);
	strcat(path, "\\"); // add backslash
	sprintf(str, "LEFile.%03hd", fileNumber);
	strcat (path, str);
#endif
	return err;

}
OSErr TModel::GetTempLEOutputFilePathNameTake2(short fileNumber, char* path,short *vRefNumPtr)
{
	char possiblePath[256];
	long tempLesDirID; 
	char str[32];
	OSErr err = 0;
	MyGetFolderName(TATvRefNum,TATdirID,TRUE,possiblePath);
	AddDelimiterAtEndIfNeeded(possiblePath);
	strcat(possiblePath, "GNOME Preferences");
	err = AddFolderIfMissing(0,0,possiblePath,&tempLesDirID);
	AddDelimiterAtEndIfNeeded(possiblePath);
	//strcat(path, "\\"); // add backslash
	sprintf(str, "LEFile.%03hd", fileNumber);
	strcat (possiblePath, str);
	strcpy(path,possiblePath);
	return err;
} 
/////////////////////////////////////////////////


OSErr TModel::GetOutputFileName (short fileNumber, short typeCode, char *outputFName)
{
	char	fileName[256], path[256];
	char shortFileName[256];
	char name[256];
	OSErr	err = noErr;
	
	outputFName[0] = 0;
	
	GetOutputFileName (path);
	SplitPathFile(path, fileName);
	
	if (strlen (fileName) == 0)
		strcpy (fileName, kDefLEFileName);
		
//////{
/// JLM debug, I need to get around this error
if(!(strlen (path) != 0 && FolderExists(0, 0, path)))
{
	// use/create a folder called tempLEs in the GNOME directory
	long tempLesDirID; 
	err = AddFolderIfMissing(TATvRefNum,TATdirID,"tempLEs",&tempLesDirID);
	if(err) return noErr; // could not find the folder
	MyGetFolderName(TATvRefNum,tempLesDirID,TRUE,path);
}
///////}
		

	if (strlen (path) != 0 && FolderExists(0, 0, path))
	{
		strcpy (outputFName, path);
		
		/////////
		{	// add delimiter if necessary
			long len = strlen(outputFName);
			if(outputFName[len-1] != DIRDELIMITER) 
			{
				outputFName[len] = DIRDELIMITER;
				outputFName[len+1] = 0;
			}
		}
		///////////////////
		
		
		
		strcpy(shortFileName,fileName);
		
		/////////////////////
		if (typeCode == UNCERTAINTY_LE)
			strcat(shortFileName,"UNCRTN");
		else if (typeCode == FORECAST_LE)
			strcat(shortFileName,"FORCST");
		//else if (typeCode == COMPLETE_LE)
			// nothing to add
			
			
		//////////////////
		if(fileNumber >= 0) // a negative fileNumber means use the user supplied name
		{
			char str[32];
			sprintf(str, ".%03hd",fileNumber);
			strcat(shortFileName,str);
		}
		/////////////////////
	#ifdef MAC
		// JLM 8/2/99 check the file name length on the MAC 
		if(strlen(shortFileName) > 31)
		{	// MAC file name length will be exceeded
			long addedLen = strlen(shortFileName) - strlen(fileName); 
			long maxLen = 31 - addedLen;
			char msg[256];
			sprintf(msg,"The name of the file is too long.  Please choose a file name no longer than %ld characters.",maxLen);
			printError(msg);
			return -1;
		}
	#endif
		/////////////////////
		
		strcat(outputFName,shortFileName);
		
	}
	else
		err = dirNFErr;
	
	return err;
}

OSErr TModel::WriteRunSpillOutputFileHeader(BFPB *bfpb,Seconds outputStep,char* noteStr)
{	// for now just write out the certain LE's -- code goes here
	OSErr err = 0;
	char buffer[1024] = "";
	char s[256];
	char text[256];
	long count;
	long numOutputSteps;
	

	strcpy(text, "[FILE INFORMATION]");
	strcat(text,IBMNEWLINESTRING);
	strcat(text, "  File type: TAPRUN");
	strcat(text,IBMNEWLINESTRING);
	//strcat(text, "  File version: 1");
	strcat(text, "  File version: 2");
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;

	strcpy(text, "[RUN PARAMETERS]");
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;
 
	Secs2DateString2 (fDialogVariables.startTime, s);
	sprintf(text, "  Model start time: %s", s);
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;
	
	/////////
	// JLM 2/27/01
	/*
	 if(gTapWindOffsetInSeconds != 0) {
	 Secs2DateString2 (fDialogVariables.startTime + gTapWindOffsetInSeconds, s);
	 sprintf(text, "  Wind start time: %s", s);
	 strcat(text,IBMNEWLINESTRING);
	 strcat(buffer,text); text[0] = 0;
	 
	}*/ // minus AH 06/20/2012
	
	////////////////

	strcpy(text, "  Run duration: ");
	StringWithoutTrailingZeros(s, fDialogVariables.duration / 3600.0 ,6); 
	strcat(text,s);
	strcat(text," hours");
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;

	sprintf(text, "  Output time step in seconds: %ld", fDialogVariables.computeTimeStep);
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;
	
	numOutputSteps = NumOutputSteps(outputStep);
	sprintf(text, "  Number of output steps: %ld", numOutputSteps);
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;
	
	sprintf(text, "  Number of LEs: %ld", this->NumLEs(FORECAST_LE));
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;
	
	sprintf(text, "  GNOME version: %s", GNOME_VERSION_STR);
	strcat(text,IBMNEWLINESTRING);
	#ifdef MAC
		strcat(text,"  GNOME platform: Macintosh");
	#else
		strcat(text,"  GNOME platform: Windows");
	#endif
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;
	
	strcpy(text, "  Note: ");
	strcat(buffer,text); text[0] = 0;
	if(noteStr && noteStr[0])
		strcat(buffer,noteStr); 
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;
	
	strcpy(text, "[BINARY FORMAT]");
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;

	#ifdef SWAP_BINARY
		strcpy(text, "  Endian: little");
	#else
		strcpy(text, "  Endian: big");
	#endif

	/*#ifdef MAC
		strcpy(text, "  Endian: big");
	#else
		strcpy(text, "  Endian: little");
	#endif*/
	strcat(text,IBMNEWLINESTRING);
	strcat(text, "  Longitude: LONG");
	strcat(text,IBMNEWLINESTRING);
	strcat(text, "  Latitude: LONG");
	strcat(text,IBMNEWLINESTRING);
	strcat(text, "  z: DOUBLE");
	strcat(text,IBMNEWLINESTRING);
	strcat(text, "  Bit flags: CHAR notReleased beached offMaps evaporated notOnSurface");
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;


	strcpy(text, "[BINARY DATA]");
	strcat(text,IBMNEWLINESTRING);
	strcat(buffer,text); text[0] = 0;
	
	count = strlen(buffer);
	if (err = FSWriteBuf(bfpb, &count, buffer))
		{ TechError("WriteRunSpillOutputFileHeader()", "FSWriteBuf()", err); return err; }
	
//done:
				
	return err;
}

OSErr TModel::AppendLEsToRunSpillOutputFile(BFPB *bfpb)
{	// for now just write out the certain LE's -- code goes here
	// append the LEs
	
	long pLong;
	long pLat;
	char code;
	OSErr err = 0;
	long sizeofTapLEInfo = (sizeof(long) + sizeof(long) + sizeof(char) + sizeof(double));
	long i,j,n,numLEs = 0,bufferSize,doubleSize = sizeof(double);
	long startIndex;
	char *buffer = 0;
	TLEList *list;
	TMap *bestMap;
	LEPropRec theLEPropRec;
	LERec theLE;
	Boolean notReleased,beached,offMaps,evaporated,notOnSurface;
	long bufferLEIndex = 0;
	
	numLEs  = this->NumLEs(FORECAST_LE);
	
	bufferSize = (numLEs * sizeofTapLEInfo);
	
	buffer  = (char*)_NewPtr(bufferSize);
	if(err = _MemError())  { TechError("AppendLEsToRunSpillOutputFile()", "_NewHandle()", 0); err = -1; goto done; }
	
	// fill in the buffer
	// code goes here
	// get each LE
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		LESetsList->GetListItem((Ptr)&list, i);
		
		if (list -> GetLEType () == FORECAST_LE) 
		{
			for (j = 0 ; j < list->numOfLEs ; j++) 
			{
				list -> GetLE (j, &theLE);
				
				notReleased = (theLE.statusCode == OILSTAT_NOTRELEASED);
				beached = (theLE.statusCode == OILSTAT_ONLAND);
				offMaps = (theLE.statusCode == OILSTAT_OFFMAPS);
				evaporated = (theLE.statusCode == OILSTAT_EVAPORATED);
				notOnSurface  = (theLE.z != 0.0);
				
				code = 0;
				if(notReleased) code+= 1;
				if(beached) code+= 2;
				if(offMaps) code+= 4;
				if(evaporated) code+= 8;
				if(notOnSurface) code+= 16;
				
				startIndex = bufferLEIndex*sizeofTapLEInfo;
				
				*(long*)(buffer+startIndex) = theLE.p.pLong;
				*(long*)(buffer+startIndex + 4) = theLE.p.pLat;
				//*(double*)(buffer+startIndex + 4) = theLE.z;
				*(double*)(buffer+startIndex + 8) = theLE.z;
				//buffer[startIndex + 8] = code;
				buffer[startIndex + 8 + doubleSize] = code;
				bufferLEIndex++;
			}
		}
	}
	
	if(bufferLEIndex != numLEs) 
		{ printError("bufferLEIndex != numLEs"); err = -1; goto done; }
	
	if (err = FSWriteBuf(bfpb, &bufferSize, buffer))
		{ TechError("AppendLEsToRunSpillOutputFile()", "FSWriteBuf()", err); goto done; }

done:

	if(buffer) {_DisposePtr((Ptr)buffer); buffer = 0;}

	return err;
}

/*OSErr TModel::SaveNetCDFLEFile (Seconds fileTime, short fileNumber)
{	// more like TAP file, put everything in one file, then let user pick time when loading in ?
	// pass in ncid to add to  existing file, but need to define variables first time only
	// need to convert fileTime into hours since Jan 1 2010 or some other base date
	OSErr err = 0;	
	int status, ncid, le_dim, time_dim, dimid[2];
	long numLEs;
	long n, i, j
	char title[] = "example netCDF dataset";
	TLEList *list;
	LERec theLE;
	if (firstTimeNeedToCreate)
	{
		ncid = nccreate(path,NC_CLOBBER);
		status = nc_def_dim(ncid, "particle", numLEs, &le_dim);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dim);
		if (status != NC_NOERR) {err = -1; goto done;}
		dimid[0] = lat_dim;
		status = nc_def_var (ncid, "lat", NC_DOUBLE, 2, dimid, &lat_id);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_put_att_text (ncid, lat_id, "long_name", strlen("latitude"), "latitude");
		status = nc_put_att_text (ncid, lat_id, "units", strlen("degrees_north"), "degrees_north");
		dimid[0] = time_dim;
		dimid[1] = le_dim;
		status = nc_def_var (ncid, "lon", NC_DOUBLE, 2, dimid, &lon_id);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_put_att_text (ncid, lon_id, "long_name", strlen("longitude"), "longitude");
		status = nc_put_att_text (ncid, lon_id, "units", strlen("degrees_east"), "degrees_east");
		status = nc_def_var (ncid, "time", NC_LONG, 1, dimid, &time_id);	// may need separate dimid[1] for time
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_put_att_text (ncid, time_id, "long_name", strlen("time"), "time");
		status = nc_put_att_text (ncid, time_id, "units", strlen("hours"), "hours");
		status = nc_put_att_text (ncid, NC_GLOBAL, "title", strlen(title), title);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_enddef(ncid);
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	else
	{
		// open the netcdf file
		status = nc_open(path, NC_NOWRITE, &ncid);
		//if (status != NC_NOERR) {err = -1; goto done;}
		if (status != NC_NOERR) 
		{
	#if TARGET_API_MAC_CARBON
			err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
				status = nc_open(outPath, NC_NOWRITE, &ncid);
	#endif
			if (status != NC_NOERR) {err = -1; goto done;}
	}

	}
	//write the data for this time - only forecast or uncertainty too?
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		LESetsList->GetListItem((Ptr)&list, i);
		
		if (!list->bActive) continue;	// only active?
		if (list -> GetLEType () == FORECAST_LE) 
		{
			for (j = 0 ; j < list->numOfLEs ; j++) 
			{
				list -> GetLE (j, &theLE);
			}
		}
	}
				
done:

}*/
OSErr TModel::SaveOSSMLEFile (Seconds fileTime, short fileNumber)
{
	long n, i, j, mapNum, count, createdDirID, uncertainLECount, forecastLECount;
	Seconds seconds, ossmRefTimeInSecsSince1904;
	long refYear;
	char refStr[32];
	TLEList *list;
	TMap *bestMap;
	DateTimeRec time;
	LEHeaderRec header;
	LEPropRec theLEPropRec;
	LERec theLE;
	BFPB forecastFile, uncertainFile;
	BFPB *targetFile; // JLM, use ptr not copy !!
	float currTimeInHrsAfterOssmRefTime;
	char *p, path[256], forecastLEFName[256], uncertainLEFName[256];
	OSErr err = 0;
	
	forecastFile.f = 0;
	uncertainFile.f = 0;
	
	err = GetOutputFileName (fileNumber, FORECAST_LE,    forecastLEFName);
	if(err) return err; // JLM 8/2/99
	err = GetOutputFileName (fileNumber, UNCERTAINTY_LE, uncertainLEFName);
	if(err) return err; // JLM 8/2/99

	hdelete(0, 0, forecastLEFName);
	if (err = hcreate(0, 0, forecastLEFName, '\?\?\?\?', 'BINA'))
		{ TechError("SaveOSSMLEFile()", "hcreate()", err); return err; }
	
	if (err = FSOpenBuf(0, 0, forecastLEFName, &forecastFile, 50000, FALSE))
		{ TechError("SaveOSSMLEFile()", "FSOpenBuf()", err); return err; }
	
	hdelete(0, 0, uncertainLEFName);// JLM 3/3/99 always delete sop we don't get confused
	if (IsUncertain ())  // open uncertainty LE file only if model is in uncertain mode
	{
		if (err = hcreate(0, 0, uncertainLEFName, '\?\?\?\?', 'BINA'))
			{ TechError("SaveOSSMLEFile()", "hcreate()", err); return err; }
		
		if (err = FSOpenBuf(0, 0, uncertainLEFName, &uncertainFile, 50000, FALSE))
			{ TechError("SaveOSSMLEFile()", "FSOpenBuf()", err); return err; }
	}
	
	// fill the header record
	//OSSM version 16.1 had off map = 7
	header.version = 16.1; //JLM 8/7/98

	strncpy(header.name, "L.E. FILE", 10);
	SecondsToDate (this->modelTime, &time);
	header.currDay = time.day;
	header.currMonth = time.month;
	header.currYear = (time.year % 100);// 2 digit year to be backwardly compatible
	header.currHour = time.hour;
	header.currMin = time.minute;
	
	////////////
	// Bushy says the reference for an ossm file is the previous year divisible by 4
	refYear = header.currYear - (header.currYear % 4);
	sprintf(refStr,"1/1/%02ld",refYear);
	ossmRefTimeInSecsSince1904 = DateString2Secs(refStr);
	///////////////
	
	seconds = this->modelTime - ossmRefTimeInSecsSince1904;
	header.currTimeInHrsAfterOssmRefTime = currTimeInHrsAfterOssmRefTime = seconds/3600.0;
	
	// calculate total number of LE's by adding up all the set-totals
	uncertainLECount = forecastLECount = 0;
	for (header.numRecords = 0, i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) {
		LESetsList->GetListItem((Ptr)&list, i);
		if (!list->bActive) continue;
		if (list -> GetLEType () == UNCERTAINTY_LE) {
			uncertainLECount += list->numOfLEs;
		}
		else {
			forecastLECount += list->numOfLEs;
		}
	}
	
	////////////
	// write the forecast-LE header to file
	///////////////
	header.numRecords = forecastLECount;	

//#ifdef IBM
#ifdef SWAP_BINARY
	header.currDay = (short)SwapShort((unsigned short)header.currDay);
	header.currMonth = (short)SwapShort((unsigned short)header.currMonth);
	header.currYear = (short)SwapShort((unsigned short)header.currYear);
	header.currHour = (short)SwapShort((unsigned short)header.currHour);
	header.currMin = (short)SwapShort((unsigned short)header.currMin);
	SwapFloat(&header.currTimeInHrsAfterOssmRefTime);
	SwapFloat(&header.version);
	header.numRecords = (long)SwapLong(*(unsigned long *)&header.numRecords);
#endif

	count = sizeof(LEHeaderRec);
	if (err = FSWriteBuf(&forecastFile, &count, (char *)&header))
		{ TechError("SaveOSSMLEFile()", "FSWriteBuf()", err); FSCloseBuf(&forecastFile); return -1; }

	////////////
	// write the uncertain-LE header to file if open
	///////////////
	if (uncertainFile.f)	
	{	// since forecast LE's used the same header, we need to reset  header.numRecords
		header.numRecords = uncertainLECount;	
//#ifdef IBM
#ifdef SWAP_BINARY
		header.numRecords = (long)SwapLong(*(unsigned long *)&header.numRecords);
#endif
		count = sizeof(LEHeaderRec);
		if (err = FSWriteBuf(&uncertainFile, &count, (char *)&header))
			{ TechError("SaveOSSMLEFile()", "FSWriteBuf()", err); FSCloseBuf(&uncertainFile); return -1; }
	}
	
	////////////
	// now write LEs
	///////////////
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) {
		LESetsList->GetListItem((Ptr)&list, i);
		
		if (!list->bActive) continue;
		// set the target file according to type of LE's in this list (STH)
		if (list -> GetLEType () == UNCERTAINTY_LE) {
			targetFile = &uncertainFile;		// model is in uncertain mode
		}
		else {
			targetFile = &forecastFile;
		}
		
		if (targetFile->f)			// if target file is open (STH)
		{	
			for (j = 0 ; j < list->numOfLEs ; j++) {
				list -> GetLE (j, &theLE);
				
				// fill theLEPropRec
				
				// reversed x-coordinate system for LE's
				theLEPropRec.pLong = -theLE.p.pLong / 1000000.0;
				theLEPropRec.pLat  =  theLE.p.pLat  / 1000000.0;
				
				theLEPropRec.pollutant = NewToOldPollutantCode(theLE.pollutantType);
				
				seconds = theLE.releaseTime; // seconds since 1904
				seconds -= ossmRefTimeInSecsSince1904; // convert to secs since ossm ref time 
				seconds /= 3600; // convert to hrs
				theLEPropRec.releaseTimeInHrsAfterRefTime = seconds;
				
				// GNOME has age is seconds since release
				// and we need to convert that to age at release
				theLEPropRec.ageWhenReleasedInHrsAfterReleaseTime = theLE.ageInHrsWhenReleased  - (currTimeInHrsAfterOssmRefTime - theLEPropRec.releaseTimeInHrsAfterRefTime);
				if(theLEPropRec.ageWhenReleasedInHrsAfterReleaseTime  < 0)
					theLEPropRec.ageWhenReleasedInHrsAfterReleaseTime = 0; // don't allow negative
				
				// JLM 2/26/99  //////////////////////////////
				// nMap == 7 was a code for off map in old ossm
				// Since we never use nMap, we can just set it to 1 or 7
				if (theLE.statusCode == OILSTAT_OFFMAPS)
				{ 
					theLEPropRec.nMap = OLD_OSSM_OFFMAPS;
				}
				else 
				{
					theLEPropRec.nMap = 1;
				}
				//////////////////////////////////////////////
				
				
				theLEPropRec.windKey = 0;
				theLEPropRec.beachHeight = 0;
				
				// old OSSM recognized beached by setting beachHeight = -50;
				// old OSSM recognized off map by setting nMap = 7;
				// old OSSM evaporated by adding 10 to the pollutant
				
				switch (theLE.statusCode) {
					case OILSTAT_INWATER: break;
					case OILSTAT_OFFMAPS: theLEPropRec.nMap = OLD_OSSM_OFFMAPS; 
					case OILSTAT_ONLAND: theLEPropRec.beachHeight = OLD_OSSM_BEACHED; break;
					case OILSTAT_EVAPORATED: theLEPropRec.pollutant += OLD_OSSM_EVAPORATED ; break; 
				}
				
//#ifdef IBM
#ifdef SWAP_BINARY
				SwapFloat(&theLEPropRec.pLat);
				SwapFloat(&theLEPropRec.pLong);
				SwapFloat(&theLEPropRec.releaseTimeInHrsAfterRefTime);
				SwapFloat(&theLEPropRec.ageWhenReleasedInHrsAfterReleaseTime);
				SwapFloat(&theLEPropRec.beachHeight);
				theLEPropRec.nMap = (long)SwapLong((unsigned long)theLEPropRec.nMap);
				theLEPropRec.pollutant = (long)SwapLong((unsigned long)theLEPropRec.pollutant);
				theLEPropRec.windKey = (long)SwapLong((unsigned long)theLEPropRec.windKey);
#endif
				count = sizeof(LEPropRec);
				
				if (err = FSWriteBuf(targetFile, &count, (char *)&theLEPropRec))
					{ TechError("SaveOSSMLEFile()", "FSWriteBuf()", err); FSCloseBuf(targetFile); return -1; }
			}
		}
	}
	
	if (forecastFile.f)  FSCloseBuf(&forecastFile);
	if (uncertainFile.f) FSCloseBuf(&uncertainFile);

	return 0;
}
/////////////////////////////////////////////////


short LogoResNumber(char* logoName)
{
	if(!strcmpnocase(logoName,"gnome.bmp")) return 130;
	if(!strcmpnocase(logoName,"noaa.bmp")) return 128;
	return 0;
}

void StrcpyLogoFileName(char* str)
{
	str[0] = 0;
	// look to see if the user has given us a logo.bmp file
	if(FileExists(TATvRefNum, TATdirID,"logo.bmp")) strcpy(str,"logo.bmp"); 
	else if(gNoaaVersion) strcpy(str,"noaa.bmp"); // we get it from our resource
	else strcpy(str,"gnome.bmp"); // we get it from our resource
}

Handle GetUserLogoHandle(Boolean printing)
{	// MAC returns a PicHandle
	// IBM returns a HBITMAP
	// look for the user specified logo file and use it if exists
	char logoFileName[256] = "logo.bmp";
	#ifdef MAC
		if(printing) strcpy(logoFileName,"logo.pict"); // use a pict on the mac when printing
		// but use the .bmp when writing the moss files
	#endif
	
	if(FileExists(TATvRefNum, TATdirID,logoFileName)) 
	{
		if(printing)
		{	// get the handle as a PicHandle or HDIB
			#ifdef MAC
				// on the MAC it is a PICT file
				PicHandle h = GetPICTFromFile(TATvRefNum, TATdirID,logoFileName);
				return (Handle)h;
			#else
				// on the IBM it is a BMP file
				char path[256];
				HDIB hDIB = 0;
				GetDirectoryFromID(TATdirID,path);
				strcat(path,logoFileName);
				hDIB = LoadDIB(path);
				return (Handle)hDIB; 
	
			#endif
		}
		else
		{ // get the contents of the file as a CHARH
			CHARH h = 0;
			OSErr err = ReadFileContents(NONTERMINATED,TATvRefNum, TATdirID, logoFileName, 0, 0, &h);
			if(err) {}  // I guess we might as well ignore this error
			return (Handle)h;
		}
	}
	return 0;
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////

OSErr WriteBMPResourceToNearbyFile(char* desiredFileName,short resNum,char* pathToSomeFileInDesiredDirectory,CHARH h)
{
	OSErr err = 0;
	char  bmpPath[256];
	long k,len;
	#ifdef MAC
		#define UNKNOWNCREATOR '????'
	#else
		#define UNKNOWNCREATOR 'FAKE'
	#endif
		
	// get desired path, use a copy just in case
	strcpy(bmpPath,pathToSomeFileInDesiredDirectory); 
	len = strlen(bmpPath);
	for(k = len-1; k >= 0; k--)
	{
		if(bmpPath[k] == DIRDELIMITER) {
			bmpPath[k+1] = 0;
			strcat(bmpPath,desiredFileName);
			StrToLower(bmpPath); // Jill has requested we force lower case to better support ArcView on unix machines
			break;
		}
	}
	
	if(h)
	{ // write this handle instead of the resource handle
		len = _GetHandleSize((Handle)h);
		err = WriteFileContents(0, 0, bmpPath, UNKNOWNCREATOR, 'BMP ', 0, len, h); 
		return err;
	}

	// check to see if we have resource of that name
	// if so, write that resource to a file of that name if we have one
	#ifdef MAC
	{
		Handle resHdl = GetResource('Logo',resNum);
		if(resHdl)
		{
			long lenOfHdl = _GetHandleSize(resHdl);
			_HLock(resHdl); // so it can't be purged !!!
			// NOTE: By using *resHdl rahter than resHdl, 
			// we avoid having WriteFileContents unlock resHdl
			err = WriteFileContents(0, 0, bmpPath, UNKNOWNCREATOR, 'BMP ', *resHdl, lenOfHdl, 0); 
			ReleaseResource(resHdl);// don't dispose of a resource handle !!!
			resHdl = 0;
		}
		else err = -1; // resource was not found
	}
	#else
	{	// IBM CODE
		// Hmm... in DIBUTL,  I see this comment
		//---------------------------
		// Calculating the size of the DIB is a bit tricky (if we want to
		// do it right).  The easiest way to do this is to call GlobalSize()
		// on our global handle, but since the size of our global memory may have
		// been padded a few bytes, we may end up writing out a few too
		// many bytes to the file (which may cause problems with some apps,
		// like HC 3.0).
		//---------------------------
		// which scares me. Would it be better to get it as some other type of resource ?
		// I don't see how that would help unless the problem is specific to DIB's
		// As a safety check, we will put the expected length in a text resource
		// and only write out the _min()

		long lenOfHdl;
		HGLOBAL r = 0;
		HRSRC hResInfo = 0; 
		char numStr[32];
		sprintf(numStr,"#%ld",resNum);
		hResInfo = FindResource(hInst,numStr,"LOGO");
		if(hResInfo) 
		{
			lenOfHdl = SizeofResource(hInst,hResInfo);
			r = LoadResource(hInst,hResInfo);
			if(r)
			{
				LPSTR ptr = (LPSTR)LockResource(r);
				if(ptr)
				{
					err = WriteFileContents(0, 0, bmpPath, UNKNOWNCREATOR, 'BMP ', ptr, lenOfHdl, 0); 
				}
				// documentation said you don't have to unlock a resource
				DeleteObject(r);
			}
		}
	}
	#endif
	
	return err;
}

void GetKmlTemplateDirectory(char* directoryPath)
{
	char applicationFolderPath[256];
#ifdef MAC
	//#include <sys/syslimits.h>
	CFURLRef appURL = CFBundleCopyBundleURL(CFBundleGetMainBundle());
	//CFURLGetFileSystemRepresentation(appURL, TRUE, (UInt8 *)directoryPath, PATH_MAX);
	CFURLGetFileSystemRepresentation(appURL, TRUE, (UInt8 *)directoryPath, kMaxNameLen);
	strcat(directoryPath, "/Contents/Resources/Data/kml_template/");
#else
	//dataDirectory = wxGetCwd();
	{
		char szPath[MAX_PATH], testTemplatePath[MAX_PATH];
		long size;
		OSErr err = 0;

		if(SUCCEEDED(SHGetFolderPath(NULL, 
							CSIDL_LOCAL_APPDATA|CSIDL_FLAG_CREATE, 
                             NULL, 
                             0, 							 
							 szPath)))
		{
			AddDelimiterAtEndIfNeeded(szPath);
			strcat(szPath,"Data\\kml_template\\");
			strcpy(testTemplatePath,szPath);
			strcat(testTemplatePath,"moss.kml");
			err = MyGetFileSize(0, 0, testTemplatePath, &size);
				//if (size != sizeof(settings))
					//hdelete(0, 0, szPath);
	
			//if (size>0) err = ReadFileContents(NONTERMINATED,0, 0,szPath, (char *)&settings, sizeof(Settings), 0);
		}
		if (err || size <= 0)
		{
			PathNameFromDirID(TATdirID,TATvRefNum,applicationFolderPath);
			my_p2cstr((StringPtr)applicationFolderPath);
			strcpy(directoryPath,applicationFolderPath);
			strcat(directoryPath,"Data\\kml_template\\");	// where to keep this on windows??
		}
		else strcpy(directoryPath,szPath);
	}
#endif
	return;
}

//static char *dotStr='\'\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00 \\x00\\x00\\x00 \\x08\\x06\\x00\\x00\\x00szz\\xf4\\x00\\x00\\x00\\x04sBIT\\x08\\x08\\x08\\x08|\\x08d\\x88\\x00\\x00\\x00\\tpHYs\\x00\\x00\\t:\\x00\\x00\\t:\\x01\\xf0d\\x92J\\x00\\x00\\x00\\x19tEXtSoftware\\x00www.inkscape.org\\x9b\\xee<\\x1a\\x00\\x00\\x01\\x00IDATX\\x85\\xed\\x96;\\x0e\\xc20\\x10D\\x1f\\x11==\\x12=u\\x94[p+N\\x91{p\\r\\x94*\\x05=\\ng\\xa0bh\\x1c)B^\\x93\\x8f\\x83]0R\\x1a\\\'\\x9eyr\\xbc^o$\\x91RE\\xd2\\xf4?@\\x0e\\x00\\xdb\\x19s\\n\\xa0\\x02J\\xe0\\xe8\\xc6n@\\x03\\\\\\x81\\xd7$7Ic\\x9f\\x9d\\xa4ZR\\\'[\\x9d\\xfbf7\\xd6wl\\xf8IR\\x1b\\x08\\xfeT\\xeb\\xe6D\\x018KzN\\x08\\xef\\xf5ts\\x17\\x01\\x9cf\\x86\\x0f!\\x82+\\xf1\\xed\\x9fOYvK\\xad\\x02{"\\x04PG\\x08\\xefU[9\\x1b\\xf9{A\\x01\\xdc\\x81\\xfd\\x8c2\\xf5\\xe9\\x01\\x1c\\xf0\\x94\\xa8u\\x10U\\x11\\xc3q^\\x95\\xef\\x85\\x05PF\\x0c\\x0fzZ\\x00Gc|\\x89\\xbc\\x9e\\xc9{\\x81\\x05p[!\\xcb\\xebi\\x014+\\x00x=\\xb3-\\xc3\\x17p\\x89\\x14\\x8e\\xf3\\xf2\\xb7\\xe9\\xc0I\\x98\\xfc(N\\xde\\x8c\\xb2h\\xc7Y\\\\H\\x86{"\\xfa\\x95\\xcc*\\xc3\\x90\\xa2^J\\xe7\\x00DU\\xb6\\xbd\\xe0\\x0f\\xf03\\xbd\\x01u{\\xe1\\t\\x9b\\xff\\x8b\\xe0\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82\'';
OSErr TModel::SaveKmlLEFile (Seconds fileTime, short fileNumber)
{
	// use gKMLPath
	// what about file series option
	OSErr err = 0;
	long n, i, j, totalNumLEs = 0, numPlacemarks=0;
	long numBeached=0, numInWater=0, numUncertainBeached=0, numUncertainInWater=0;
	TLEList *list;
	LEHeaderRec header;
	LERec theLE;
	BFPB fileMS3,fileMS4,fileMS5,fileMS6,fileMS7;
	char *p, path[256], fileName[256], nameStr[256], styleStr[256];
	char fileNumStr[64] = "";
	char trajectoryTime[64], trajectoryDate[64], preparedTime[32], preparedDate[32];
	Boolean bWriteUncertaintyLEs = this->IsUncertain();
	float lat,lon;
	WorldPoint *waterPts=0, *beachedPts=0, *uncertainWaterPts=0, *uncertainBeachedPts=0;
	short *placemarks=0;
	
	char hdrStr[256], ptHdrStr[256], ptHdrStr2[256], ptStr[256], mgHdrStr[256], mgHdrStr2[256], pmHdrStr[256], pmHdrStr2[256];
	
	char ch, source_file[256], target_file[256], template_folder[256], outPath[256], userPath[256], template_path[256];
	FILE *source, *target;
	
	GetKmlTemplateDirectory(template_folder);
#if TARGET_API_MAC_CARBON
	ConvertUnixPathToTraditionalPath((const char *) template_folder,template_path, kMaxNameLen);
#else
	strcpy(template_path,template_folder);
#endif
	sprintf(source_file,"%s%c%s",template_path,DIRDELIMITER,"x.png");
	
	sprintf(userPath,"%s%c",gKMLPath,DIRDELIMITER);
	if (!FolderExists(0, 0, userPath)) 
	{
		long dirID;
		err = dircreate(0, 0, userPath, &dirID);
		if(err) 
		{	// try to create folders 
			err = CreateFoldersInDirectoryPath(userPath);
			if (err)	
			{
				printError("Unable to create the directory for the output file.");
			}
		}
	}
	strcat(userPath,"x.png");
	err = MyCopyFile(0,0,source_file,0,0,userPath);

	sprintf(source_file,"%s%c%s",template_path,DIRDELIMITER,"dot.png");
	sprintf(userPath,"%s%c%s",gKMLPath,DIRDELIMITER,"dot.png");
	err = MyCopyFile(0,0,source_file,0,0,userPath);

#if TARGET_API_MAC_CARBON
	err = ConvertTraditionalPathToUnixPath((const char *) gKMLPath, outPath, kMaxNameLen) ;
#else
	strcpy(outPath,gKMLPath);
#endif
	strcpy(source_file,template_folder);
	strcat(source_file,"moss.kml");
	
	source = fopen(source_file, "r");
	
	if( source == NULL )
	{
		err=-1; return err;
	}
	
	sprintf(target_file,"%s%c%s",outPath,NEWDIRDELIMITER,"moss.kml");
	
	target = fopen(target_file, "w");
	
	if( target == NULL )
	{
		fclose(source);
		err = -1; return err;
	}
	
	/*while( ( ch = fgetc(source) ) != EOF )
		fputc(ch, target);
	
	fclose(source);*/
	
	// get total number of LEs and set up arrays for in water, beached, certain and uncertain
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		Boolean isUncertainLE;
		LESetsList->GetListItem((Ptr)&list, i);
		if (!list->bActive) continue;
		
		isUncertainLE = (list -> GetLEType () == UNCERTAINTY_LE);
		if (isUncertainLE) continue;
		totalNumLEs += list->numOfLEs;

	}
	if (totalNumLEs==0)
	{
		printNote("There are no LEs to output");
		err=-1;
		return err;
	}
	try
	{
		waterPts = new WorldPoint[totalNumLEs];	
		beachedPts = new WorldPoint[totalNumLEs];
	}
	catch (...)
	{
		TechError("TModel::SaveKmlLEFile()", "new WorldPoint()", 0); return -1;
	}
	if (bWriteUncertaintyLEs)
	{
		try
		{
			uncertainWaterPts = new WorldPoint[totalNumLEs];
			uncertainBeachedPts = new WorldPoint[totalNumLEs];
		}
		catch (...)
		{
			TechError("TModel::SaveKmlLEFile()", "new WorldPoint()", 0); return -1;
		}
	}
	
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		Boolean isUncertainLE;
		LESetsList->GetListItem((Ptr)&list, i);
		if (!list->bActive) continue;
		
		isUncertainLE = (list -> GetLEType () == UNCERTAINTY_LE);
		if(isUncertainLE && !bWriteUncertaintyLEs) continue; //don't write out uncertainty LEs unless uncertainty is on
		
		for (j = 0 ; j < list->numOfLEs ; j++) 
		{
			list -> GetLE (j, &theLE);
			
			// JLM, 3/1/99 it was decided to skip EVAPORATED and NOTRELEASED LE's
			switch (theLE.statusCode) {
				case OILSTAT_INWATER: 
				{
					if (isUncertainLE) {uncertainWaterPts[numUncertainInWater]=theLE.p; numUncertainInWater++;}
					else  {waterPts[numInWater]=theLE.p; numInWater++;}
					break;
				}
				case OILSTAT_OFFMAPS: 
				{
					if (isUncertainLE) {uncertainBeachedPts[numUncertainBeached]=theLE.p; numUncertainBeached++;}
					else  {beachedPts[numBeached]=theLE.p; numBeached++;}
					break; // what are we doing with these?
				}
				case OILSTAT_ONLAND: 
					if (isUncertainLE) {uncertainBeachedPts[numUncertainBeached]=theLE.p; numUncertainBeached++;}
					else  {beachedPts[numBeached]=theLE.p; numBeached++;}
					break; 
					
				default: continue; // skip other LE's	
			}
			////////////////////////////////////////////////
		}
	}
	try
	{
		placemarks = new short[4];	
	}
	catch (...)
	{
		TechError("TModel::SaveKmlLEFile()", "new short()", 0); return -1;
	}
	if (numInWater>0) {placemarks[numPlacemarks]= 0; numPlacemarks++;}
	if (numBeached>0) {placemarks[numPlacemarks]=1; numPlacemarks++;}
	if (numUncertainInWater>0) {placemarks[numPlacemarks]=2; numPlacemarks++;}
	if (numUncertainBeached>0) {placemarks[numPlacemarks]=3; numPlacemarks++;}
	
	GetDateTimeStrings(trajectoryTime, trajectoryDate, preparedTime, preparedDate);
	
	strcpy(hdrStr,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	strcpy(hdrStr,"<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n");
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	strcpy(hdrStr,"  <Document>\n");
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	strcpy(hdrStr,"    <name>moss</name>\n");
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	strcpy(hdrStr,"    <open>1</open>\n");
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	sprintf(hdrStr,"    <description><![CDATA[<b>Valid for:</b> %s, %s<br>\n",trajectoryTime,trajectoryDate);
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	sprintf(hdrStr,"<b>Issued:</b> %s, %s\n",preparedTime,preparedDate);
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	
	while( ( ch = fgetc(source) ) != EOF )
		fputc(ch, target);
	
	fclose(source);
	
	strcpy(ptHdrStr,"                 <Point>\n");
	strcpy(ptHdrStr2,"                 </Point>\n");
	strcpy(mgHdrStr,"      <MultiGeometry>\n");
	strcpy(mgHdrStr2,"      </MultiGeometry>\n");
	strcpy(pmHdrStr,"     <Placemark>\n");
	strcpy(pmHdrStr2,"     </Placemark>\n");
	for (i=0;i<numPlacemarks;i++)
	{
		//strcpy(hdrStr,"     <Placemark>\n");
		fwrite(pmHdrStr,sizeof(char),strlen(pmHdrStr),target);
		if (placemarks[i]== 0)
		{
			strcpy(nameStr,"      <name>Floating Splots (Best Guess)</name>\n");
			fwrite(nameStr,sizeof(char),strlen(nameStr),target);
			strcpy(styleStr,"      <styleUrl>#YellowDotIcon</styleUrl>\n");
			fwrite(styleStr,sizeof(char),strlen(styleStr),target);
			//strcpy(hdrStr,"      <MultiGeometry>\n");
			fwrite(mgHdrStr,sizeof(char),strlen(mgHdrStr),target);
			for (j=0;j<numInWater;j++)
			{
				lat = (double)waterPts[j].pLat / 1000000.0;
				lon = (double)waterPts[j].pLong / 1000000.0;
				//strcpy(ptHdrStr,"                 <Point>\n");
				//strcpy(ptHdrStr2,"                 </Point>\n");
				fwrite(ptHdrStr,sizeof(char),strlen(ptHdrStr),target);
				sprintf(ptStr,"                   <coordinates>%lf,%lf</coordinates>\n",lon,lat);	// total vertices
				fwrite(ptStr,sizeof(char),strlen(ptStr),target);
				fwrite(ptHdrStr2,sizeof(char),strlen(ptHdrStr2),target);
			}

		}
		else if (placemarks[i]==1)
		{
			strcpy(nameStr,"      <name>Beached Splots (Best Guess)</name>\n");
			fwrite(nameStr,sizeof(char),strlen(nameStr),target);
			strcpy(styleStr,"      <styleUrl>#YellowXIcon</styleUrl>\n");
			fwrite(styleStr,sizeof(char),strlen(styleStr),target);
			//strcpy(hdrStr,"      <MultiGeometry>\n");
			fwrite(mgHdrStr,sizeof(char),strlen(mgHdrStr),target);
			for (j=0;j<numBeached;j++)
			{
				lat = (double)beachedPts[j].pLat / 1000000.0;
				lon = (double)beachedPts[j].pLong / 1000000.0;
				//strcpy(ptHdrStr,"                 <Point>\n");
				//strcpy(ptHdrStr2,"                 </Point>\n");
				fwrite(ptHdrStr,sizeof(char),strlen(ptHdrStr),target);
				sprintf(ptStr,"                   <coordinates>%lf,%lf</coordinates>\n",lon,lat);	// total vertices
				fwrite(ptStr,sizeof(char),strlen(ptStr),target);
				fwrite(ptHdrStr2,sizeof(char),strlen(ptHdrStr2),target);
			}
		}
		else if (placemarks[i]==2)
		{
			strcpy(nameStr,"      <name>Uncertainty Floating Splots</name>\n");
			fwrite(nameStr,sizeof(char),strlen(nameStr),target);
			strcpy(styleStr,"      <styleUrl>#RedDotIcon</styleUrl>\n");
			fwrite(styleStr,sizeof(char),strlen(styleStr),target);
			//strcpy(hdrStr,"      <MultiGeometry>\n");
			fwrite(mgHdrStr,sizeof(char),strlen(mgHdrStr),target);
			for (j=0;j<numUncertainInWater;j++)
			{
				lat = (double)uncertainWaterPts[j].pLat / 1000000.0;
				lon = (double)uncertainWaterPts[j].pLong / 1000000.0;
				//strcpy(ptHdrStr,"                 <Point>\n");
				//strcpy(ptHdrStr2,"                 </Point>\n");
				fwrite(ptHdrStr,sizeof(char),strlen(ptHdrStr),target);
				sprintf(ptStr,"                   <coordinates>%lf,%lf</coordinates>\n",lon,lat);	// total vertices
				fwrite(ptStr,sizeof(char),strlen(ptStr),target);
				fwrite(ptHdrStr2,sizeof(char),strlen(ptHdrStr2),target);
			}
		}
		else if (placemarks[i]==3)
		{
			strcpy(nameStr,"      <name>Uncertainty Beached Splots</name>\n");
			fwrite(nameStr,sizeof(char),strlen(nameStr),target);
			strcpy(styleStr,"      <styleUrl>#RedXIcon</styleUrl>\n");
			fwrite(styleStr,sizeof(char),strlen(styleStr),target);
			//strcpy(hdrStr,"      <MultiGeometry>\n");
			fwrite(mgHdrStr,sizeof(char),strlen(mgHdrStr),target);
			for (j=0;j<numUncertainBeached;j++)
			{
				lat = (double)uncertainBeachedPts[j].pLat / 1000000.0;
				lon = (double)uncertainBeachedPts[j].pLong / 1000000.0;
				//strcpy(ptHdrStr,"                 <Point>\n");
				//strcpy(ptHdrStr2,"                 </Point>\n");
				fwrite(ptHdrStr,sizeof(char),strlen(ptHdrStr),target);
				sprintf(ptStr,"                   <coordinates>%lf,%lf</coordinates>\n",lon,lat);	// total vertices
				fwrite(ptStr,sizeof(char),strlen(ptStr),target);
				fwrite(ptHdrStr2,sizeof(char),strlen(ptHdrStr2),target);
			}
		}
		
		
		//strcpy(hdrStr,"      </MultiGeometry>\n");
		fwrite(mgHdrStr2,sizeof(char),strlen(mgHdrStr2),target);
		//strcpy(hdrStr,"     </Placemark>\n");
		fwrite(pmHdrStr2,sizeof(char),strlen(pmHdrStr2),target);
	}
	strcpy(hdrStr,"  </Document>\n");
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	strcpy(hdrStr,"</kml>\n");
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	

	fclose(target);
	
	if (waterPts)  {delete [] waterPts; waterPts = 0;}
	if (beachedPts)  {delete [] beachedPts; beachedPts = 0;}
	if (uncertainWaterPts)  {delete [] uncertainWaterPts; uncertainWaterPts = 0;}
	if (uncertainBeachedPts)  {delete [] uncertainBeachedPts; uncertainBeachedPts = 0;}
	
	if (placemarks)  {delete [] placemarks; placemarks = 0;}

	return err;
	
}

OSErr TModel::SaveKmlLEFileSeries (Seconds fileTime, short fileNumber)
{
	// use gKMLPath
	// what about file series option
	OSErr err = 0;
	long n, i, j, totalNumLEs = 0, numPlacemarks=0;
	long numBeached=0, numInWater=0, numUncertainBeached=0, numUncertainInWater=0;
	TLEList *list;
	LEHeaderRec header;
	LERec theLE;
	//BFPB fileMS3,fileMS4,fileMS5,fileMS6,fileMS7;
	DateTimeRec time;
	char *p, path[256], fileName[256], nameStr[256], styleStr[256], startTimeStr[256], endTimeStr[256], kmlTimeStr[32];
	char fileNumStr[64] = "";
	char trajectoryTime[64], trajectoryDate[64], preparedTime[32], preparedDate[32];
	Boolean bWriteUncertaintyLEs = this->IsUncertain();
	float lat,lon,z=1.;
	WorldPoint *waterPts=0, *beachedPts=0, *uncertainWaterPts=0, *uncertainBeachedPts=0;
	short *placemarks=0;
	
	char hdrStr[256], ptHdrStr[256], ptHdrStr2[256], ptStr[256], mgHdrStr[64], mgHdrStr2[64], pmHdrStr[64], pmHdrStr2[64];
	char timeSpanHdrStr[256], timeSpanHdrStr2[256];
	
	char ch, source_file[256], target_file[256], uncertain_file[256], template_folder[256], outPath[256], userPath[256], template_path[256];
	FILE *source, *target, *uncertainFile;
	
	GetKmlTemplateDirectory(template_folder);
	if(fileNumber==0)
	{
#if TARGET_API_MAC_CARBON
		ConvertUnixPathToTraditionalPath((const char *) template_folder,template_path, kMaxNameLen);
#else
		strcpy(template_path,template_folder);
#endif
		sprintf(source_file,"%s%c%s",template_path,DIRDELIMITER,"x.png");
		
		sprintf(userPath,"%s%c",gKMLPath,DIRDELIMITER);
		if (!FolderExists(0, 0, userPath)) 
		{
			long dirID;
			err = dircreate(0, 0, userPath, &dirID);
			if(err) 
			{	// try to create folders 
				err = CreateFoldersInDirectoryPath(userPath);
				if (err)	
				{
					printError("Unable to create the directory for the output file.");
				}
			}
		}
		strcat(userPath,"x.png");
		err = MyCopyFile(0,0,source_file,0,0,userPath);
		
		sprintf(source_file,"%s%c%s",template_path,DIRDELIMITER,"dot.png");
		sprintf(userPath,"%s%c%s",gKMLPath,DIRDELIMITER,"dot.png");
		err = MyCopyFile(0,0,source_file,0,0,userPath);
	}
#if TARGET_API_MAC_CARBON
	err = ConvertTraditionalPathToUnixPath((const char *) gKMLPath, outPath, kMaxNameLen) ;
#else
	strcpy(outPath,gKMLPath);
#endif
	
	if (fileNumber==0)
	{
		strcpy(source_file,template_folder);
		strcat(source_file,"moss.kml");
		
		source = fopen(source_file, "r");
		
		if( source == NULL )
		{
			err=-1; return err;
		}
	}
	sprintf(target_file,"%s%c%s",outPath,NEWDIRDELIMITER,"moss.kml");
	sprintf(uncertain_file,"%s%c%s",outPath,NEWDIRDELIMITER,"moss_uncertain.kml");
	
	//target = fopen(target_file, "w");
	if (fileNumber==0)
	{
		target = fopen(target_file, "w+");
		uncertainFile = fopen(uncertain_file, "w+");
	}
	else 
	{
		target = fopen(target_file, "a");
		uncertainFile = fopen(uncertain_file, "a");
	}

	
	if( target == NULL )
	{
		if (fileNumber==0) fclose(source);
		err = -1; return err;
	}
	
	if( uncertainFile == NULL )
	{
		fclose(target);
		if (fileNumber==0) fclose(source);
		err = -1; return err;
	}
	
	/*while( ( ch = fgetc(source) ) != EOF )
	 fputc(ch, target);
	 
	 fclose(source);*/
	
	// get total number of LEs and set up arrays for in water, beached, certain and uncertain
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		Boolean isUncertainLE;
		LESetsList->GetListItem((Ptr)&list, i);
		if (!list->bActive) continue;
		
		isUncertainLE = (list -> GetLEType () == UNCERTAINTY_LE);
		if (isUncertainLE) continue;
		totalNumLEs += list->numOfLEs;
		
	}
	if (totalNumLEs==0)
	{
		printNote("There are no LEs to output");
		err=-1;
		return err;
	}
	try
	{
		waterPts = new WorldPoint[totalNumLEs];	
		beachedPts = new WorldPoint[totalNumLEs];
	}
	catch (...)
	{
		TechError("TModel::SaveKmlLEFile()", "new WorldPoint()", 0); return -1;
	}
	if (bWriteUncertaintyLEs)
	{
		try
		{
			uncertainWaterPts = new WorldPoint[totalNumLEs];
			uncertainBeachedPts = new WorldPoint[totalNumLEs];
		}
		catch (...)
		{
			TechError("TModel::SaveKmlLEFile()", "new WorldPoint()", 0); return -1;
		}
	}
	
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		Boolean isUncertainLE;
		LESetsList->GetListItem((Ptr)&list, i);
		if (!list->bActive) continue;
		
		isUncertainLE = (list -> GetLEType () == UNCERTAINTY_LE);
		if(isUncertainLE && !bWriteUncertaintyLEs) continue; //don't write out uncertainty LEs unless uncertainty is on
		
		for (j = 0 ; j < list->numOfLEs ; j++) 
		{
			list -> GetLE (j, &theLE);
			
			// JLM, 3/1/99 it was decided to skip EVAPORATED and NOTRELEASED LE's
			switch (theLE.statusCode) {
				case OILSTAT_INWATER: 
				{
					if (isUncertainLE) {uncertainWaterPts[numUncertainInWater]=theLE.p; numUncertainInWater++;}
					else  {waterPts[numInWater]=theLE.p; numInWater++;}
					break;
				}
				/*case OILSTAT_OFFMAPS: 
				{
					if (isUncertainLE) {uncertainBeachedPts[numUncertainBeached]=theLE.p; numUncertainBeached++;}
					else  {beachedPts[numBeached]=theLE.p; numBeached++;}
					break; // what are we doing with these?
				}*/
				case OILSTAT_ONLAND: 
					if (isUncertainLE) {uncertainBeachedPts[numUncertainBeached]=theLE.p; numUncertainBeached++;}
					else  {beachedPts[numBeached]=theLE.p; numBeached++;}
					break; 
					
				default: continue; // skip other LE's	
			}
			////////////////////////////////////////////////
		}
	}
	try
	{
		placemarks = new short[4];	
	}
	catch (...)
	{
		TechError("TModel::SaveKmlLEFile()", "new short()", 0); return -1;
	}
	if (numInWater>0) {placemarks[numPlacemarks]= 0; numPlacemarks++;}
	if (numBeached>0) {placemarks[numPlacemarks]=1; numPlacemarks++;}
	if (numUncertainInWater>0) {placemarks[numPlacemarks]=2; numPlacemarks++;}
	if (numUncertainBeached>0) {placemarks[numPlacemarks]=3; numPlacemarks++;}
	if (numPlacemarks==0) return err;
	
	GetDateTimeStrings(trajectoryTime, trajectoryDate, preparedTime, preparedDate);	// will need to translate to kmz time
	SecondsToDate (model->GetModelTime(), &time);
	Date2KmlString(&time, kmlTimeStr);
	sprintf(startTimeStr,"          <begin>%s</begin>     <!-- kml:dateTime -->\n",kmlTimeStr);
	SecondsToDate (model->GetModelTime()+fOutputTimeStep, &time);
	Date2KmlString(&time, kmlTimeStr);
	sprintf(endTimeStr,"          <end>%s</end>         <!-- kml:dateTime -->\n",kmlTimeStr);	
	
	if(fileNumber==0)
	{
		strcpy(hdrStr,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
		strcpy(hdrStr,"<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n");
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
		strcpy(hdrStr,"  <Document>\n");
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
		strcpy(hdrStr,"    <name>moss</name>\n");
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
		strcpy(hdrStr,"    <open>1</open>\n");
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
		sprintf(hdrStr,"    <description><![CDATA[<b>Valid for:</b> %s, %s<br>\n",trajectoryTime,trajectoryDate);
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
		sprintf(hdrStr,"<b>Issued:</b> %s, %s\n",preparedTime,preparedDate);
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	
		while( ( ch = fgetc(source) ) != EOF )
			fputc(ch, target);
		
		fclose(source);
	}
	
	//strcpy(ptHdrStr,"                 <Point>\n");
	//strcpy(ptHdrStr2,"                 </Point>\n");
	strcpy(ptHdrStr,"             <Point>\n");
	strcpy(ptHdrStr2,"             </Point>\n");
	strcpy(mgHdrStr,"      <MultiGeometry>\n");
	strcpy(mgHdrStr2,"      </MultiGeometry>\n");
	strcpy(pmHdrStr,"     <Placemark>\n");
	strcpy(pmHdrStr2,"     </Placemark>\n");

	strcpy(timeSpanHdrStr,"      <TimeSpan id=\"ID\">\n");
	strcpy(timeSpanHdrStr2,"     </TimeSpan>\n");
	
	if (fileNumber==0)
	{
		strcpy(hdrStr," <Folder>\n");
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),uncertainFile);
		strcpy(nameStr," <name>Best Guess</name>\n");
		fwrite(nameStr,sizeof(char),strlen(nameStr),target);
		strcpy(nameStr," <name>Uncertainty</name>\n");
		fwrite(nameStr,sizeof(char),strlen(nameStr),uncertainFile);
	}
	
	for (i=0;i<numPlacemarks;i++)
	{
		//strcpy(hdrStr,"     <Placemark>\n");
		//fwrite(pmHdrStr,sizeof(char),strlen(pmHdrStr),target);
		//fwrite(pmHdrStr,sizeof(char),strlen(pmHdrStr),uncertainFile);
		if (placemarks[i]==0)
		{
			fwrite(pmHdrStr,sizeof(char),strlen(pmHdrStr),target);
			sprintf(nameStr,"      <name>%s, %s - Floating Splots (Best Guess)</name>\n",trajectoryTime,trajectoryDate);
			//strcpy(nameStr,"      <name>Floating Splots (Best Guess)</name>\n");	// add time here
			fwrite(nameStr,sizeof(char),strlen(nameStr),target);
			strcpy(styleStr,"      <styleUrl>#YellowDotIcon</styleUrl>\n");
			fwrite(styleStr,sizeof(char),strlen(styleStr),target);
			//strcpy(hdrStr,"      <MultiGeometry>\n");
			//<begin>2013-11-11T20:00:00Z</begin>     <!-- kml:dateTime -->
			//<end>2013-11-11T21:00:00Z</end>         <!-- kml:dateTime -->
			fwrite(timeSpanHdrStr,sizeof(char),strlen(timeSpanHdrStr),target);
			fwrite(startTimeStr,sizeof(char),strlen(startTimeStr),target);
			fwrite(endTimeStr,sizeof(char),strlen(endTimeStr),target);
			fwrite(timeSpanHdrStr2,sizeof(char),strlen(timeSpanHdrStr2),target);
			fwrite(mgHdrStr,sizeof(char),strlen(mgHdrStr),target);
			for (j=0;j<numInWater;j++)
			{
				lat = (double)waterPts[j].pLat / 1000000.0;
				lon = (double)waterPts[j].pLong / 1000000.0;
				//strcpy(ptHdrStr,"                 <Point>\n");
				//strcpy(ptHdrStr2,"                 </Point>\n");
				fwrite(ptHdrStr,sizeof(char),strlen(ptHdrStr),target);
				if (bWriteUncertaintyLEs) 
				{
					strcpy(hdrStr,"                     <altitudeMode>relativeToGround</altitudeMode>\n");
					fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
					sprintf(ptStr,"                     <coordinates>%lf,%lf,%lf</coordinates>\n",lon,lat,z);
					fwrite(ptStr,sizeof(char),strlen(ptStr),target);
				}
				else
				{
					sprintf(ptStr,"                   <coordinates>%lf,%lf</coordinates>\n",lon,lat);	// total vertices
					fwrite(ptStr,sizeof(char),strlen(ptStr),target);
				}

				//sprintf(ptStr,"                   <coordinates>%lf,%lf</coordinates>\n",lon,lat);	// total vertices
				//fwrite(ptStr,sizeof(char),strlen(ptStr),target);
				fwrite(ptHdrStr2,sizeof(char),strlen(ptHdrStr2),target);
			}
			fwrite(mgHdrStr2,sizeof(char),strlen(mgHdrStr2),target);
			fwrite(pmHdrStr2,sizeof(char),strlen(pmHdrStr2),target);
		}
		else if (placemarks[i]==1)
		{
			fwrite(pmHdrStr,sizeof(char),strlen(pmHdrStr),target);
			sprintf(nameStr,"      <name>%s, %s - Beached Splots (Best Guess)</name>\n",trajectoryTime,trajectoryDate);
			//strcpy(nameStr,"      <name>Beached Splots (Best Guess)</name>\n");
			fwrite(nameStr,sizeof(char),strlen(nameStr),target);
			strcpy(styleStr,"      <styleUrl>#YellowXIcon</styleUrl>\n");
			fwrite(styleStr,sizeof(char),strlen(styleStr),target);
			//strcpy(hdrStr,"      <MultiGeometry>\n");
			fwrite(timeSpanHdrStr,sizeof(char),strlen(timeSpanHdrStr),target);
			fwrite(startTimeStr,sizeof(char),strlen(startTimeStr),target);
			fwrite(endTimeStr,sizeof(char),strlen(endTimeStr),target);
			fwrite(timeSpanHdrStr2,sizeof(char),strlen(timeSpanHdrStr2),target);
			fwrite(mgHdrStr,sizeof(char),strlen(mgHdrStr),target);
			for (j=0;j<numBeached;j++)
			{
				lat = (double)beachedPts[j].pLat / 1000000.0;
				lon = (double)beachedPts[j].pLong / 1000000.0;
				//strcpy(ptHdrStr,"                 <Point>\n");
				//strcpy(ptHdrStr2,"                 </Point>\n");
				fwrite(ptHdrStr,sizeof(char),strlen(ptHdrStr),target);
				if (bWriteUncertaintyLEs) 
				{
					strcpy(hdrStr,"                     <altitudeMode>relativeToGround</altitudeMode>\n");
					fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
					sprintf(ptStr,"                     <coordinates>%lf,%lf,%lf</coordinates>\n",lon,lat,z);
					fwrite(ptStr,sizeof(char),strlen(ptStr),target);
				}
				else {
					sprintf(ptStr,"                   <coordinates>%lf,%lf</coordinates>\n",lon,lat);	// total vertices
					fwrite(ptStr,sizeof(char),strlen(ptStr),target);
				}

				//sprintf(ptStr,"                   <coordinates>%lf,%lf</coordinates>\n",lon,lat);	// total vertices
				//fwrite(ptStr,sizeof(char),strlen(ptStr),target);
				fwrite(ptHdrStr2,sizeof(char),strlen(ptHdrStr2),target);
			}
			fwrite(mgHdrStr2,sizeof(char),strlen(mgHdrStr2),target);
			fwrite(pmHdrStr2,sizeof(char),strlen(pmHdrStr2),target);
		}
		else if (placemarks[i]==2)
		{
			fwrite(pmHdrStr,sizeof(char),strlen(pmHdrStr),uncertainFile);
			sprintf(nameStr,"      <name>%s, %s - Uncertainty Floating Splots</name>\n",trajectoryTime,trajectoryDate);
			//strcpy(nameStr,"      <name>Uncertainty Floating Splots</name>\n");
			fwrite(nameStr,sizeof(char),strlen(nameStr),uncertainFile);
			strcpy(styleStr,"      <styleUrl>#RedDotIcon</styleUrl>\n");
			fwrite(styleStr,sizeof(char),strlen(styleStr),uncertainFile);
			//strcpy(hdrStr,"      <MultiGeometry>\n");
			fwrite(timeSpanHdrStr,sizeof(char),strlen(timeSpanHdrStr),uncertainFile);
			fwrite(startTimeStr,sizeof(char),strlen(startTimeStr),uncertainFile);
			fwrite(endTimeStr,sizeof(char),strlen(endTimeStr),uncertainFile);
			fwrite(timeSpanHdrStr2,sizeof(char),strlen(timeSpanHdrStr2),uncertainFile);
			fwrite(mgHdrStr,sizeof(char),strlen(mgHdrStr),uncertainFile);
			for (j=0;j<numUncertainInWater;j++)
			{
				lat = (double)uncertainWaterPts[j].pLat / 1000000.0;
				lon = (double)uncertainWaterPts[j].pLong / 1000000.0;
				//strcpy(ptHdrStr,"                 <Point>\n");
				//strcpy(ptHdrStr2,"                 </Point>\n");
				fwrite(ptHdrStr,sizeof(char),strlen(ptHdrStr),uncertainFile);
				sprintf(ptStr,"                   <coordinates>%lf,%lf</coordinates>\n",lon,lat);	// total vertices
				fwrite(ptStr,sizeof(char),strlen(ptStr),uncertainFile);
				fwrite(ptHdrStr2,sizeof(char),strlen(ptHdrStr2),uncertainFile);
			}
			fwrite(mgHdrStr2,sizeof(char),strlen(mgHdrStr2),uncertainFile);
			fwrite(pmHdrStr2,sizeof(char),strlen(pmHdrStr2),uncertainFile);
		}
		else if (placemarks[i]==3)
		{
			fwrite(pmHdrStr,sizeof(char),strlen(pmHdrStr),uncertainFile);
			sprintf(nameStr,"      <name>%s, %s - Uncertainty Beached Splots</name>\n",trajectoryTime,trajectoryDate);
			//strcpy(nameStr,"      <name>Uncertainty Beached Splots</name>\n");
			fwrite(nameStr,sizeof(char),strlen(nameStr),uncertainFile);
			strcpy(styleStr,"      <styleUrl>#RedXIcon</styleUrl>\n");
			fwrite(styleStr,sizeof(char),strlen(styleStr),uncertainFile);
			//strcpy(hdrStr,"      <MultiGeometry>\n");
			fwrite(timeSpanHdrStr,sizeof(char),strlen(timeSpanHdrStr),uncertainFile);
			fwrite(startTimeStr,sizeof(char),strlen(startTimeStr),uncertainFile);
			fwrite(endTimeStr,sizeof(char),strlen(endTimeStr),uncertainFile);
			fwrite(timeSpanHdrStr2,sizeof(char),strlen(timeSpanHdrStr2),uncertainFile);
			fwrite(mgHdrStr,sizeof(char),strlen(mgHdrStr),uncertainFile);
			for (j=0;j<numUncertainBeached;j++)
			{
				lat = (double)uncertainBeachedPts[j].pLat / 1000000.0;
				lon = (double)uncertainBeachedPts[j].pLong / 1000000.0;
				//strcpy(ptHdrStr,"                 <Point>\n");
				//strcpy(ptHdrStr2,"                 </Point>\n");
				fwrite(ptHdrStr,sizeof(char),strlen(ptHdrStr),uncertainFile);
				sprintf(ptStr,"                   <coordinates>%lf,%lf</coordinates>\n",lon,lat);	// total vertices
				fwrite(ptStr,sizeof(char),strlen(ptStr),uncertainFile);
				fwrite(ptHdrStr2,sizeof(char),strlen(ptHdrStr2),uncertainFile);
			}
			fwrite(mgHdrStr2,sizeof(char),strlen(mgHdrStr2),uncertainFile);
			fwrite(pmHdrStr2,sizeof(char),strlen(pmHdrStr2),uncertainFile);
		}
		else{
			printNote("err in placemarks");
		}
				
		//fwrite(mgHdrStr2,sizeof(char),strlen(mgHdrStr2),target);
		//fwrite(pmHdrStr2,sizeof(char),strlen(pmHdrStr2),target);
		//fwrite(mgHdrStr2,sizeof(char),strlen(mgHdrStr2),uncertainFile);
		//fwrite(pmHdrStr2,sizeof(char),strlen(pmHdrStr2),uncertainFile);
	}
	//strcpy(hdrStr,"  </Document>\n");
	//fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	//strcpy(hdrStr,"</kml>\n");
	//fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	
	
	fclose(target);
	fclose(uncertainFile);
	
	if (waterPts)  {delete [] waterPts; waterPts = 0;}
	if (beachedPts)  {delete [] beachedPts; beachedPts = 0;}
	if (uncertainWaterPts)  {delete [] uncertainWaterPts; uncertainWaterPts = 0;}
	if (uncertainBeachedPts)  {delete [] uncertainBeachedPts; uncertainBeachedPts = 0;}
	
	if (placemarks)  {delete [] placemarks; placemarks = 0;}
	
	return err;
	
}

OSErr TModel::FinishKmlFile ()
{
	OSErr err = 0;
	char *p, ch, outPath[256], target_file[256], target_path[256], kmz_path[256], uncertain_file[256], hdrStr[20], cmd[64];
	FILE *target, *uncertainFile;
	Boolean bWriteUncertaintyLEs = this->IsUncertain();
	int status;
	long i, numChops = 1;
	
#if TARGET_API_MAC_CARBON
	err = ConvertTraditionalPathToUnixPath((const char *) gKMLPath, outPath, kMaxNameLen) ;
#else
	strcpy(outPath,gKMLPath);
#endif
	
	sprintf(target_file,"%s%c%s",outPath,NEWDIRDELIMITER,"moss.kml");
	sprintf(uncertain_file,"%s%c%s",outPath,NEWDIRDELIMITER,"moss_uncertain.kml");
	
	//target = fopen(target_file, "w");
	target = fopen(target_file, "a");
	if( target == NULL )
	{
		err = -1; return err;
	}
	
	strcpy(hdrStr," </Folder>\n");
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);

	if (bWriteUncertaintyLEs)
	{
		uncertainFile = fopen(uncertain_file, "r");	
		if( uncertainFile == NULL )
		{
			fclose(target);
			err = -1; return err;
		}
		while( ( ch = fgetc(uncertainFile) ) != EOF )
			fputc(ch, target);
		strcpy(hdrStr," </Folder>\n");
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	}
	strcpy(hdrStr,"  </Document>\n");
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	strcpy(hdrStr,"</kml>\n");
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),target);
	
	fclose(target);
	if (bWriteUncertaintyLEs) fclose(uncertainFile);
	status = remove(uncertain_file);
	
	strcpy(target_path,target_file);
	for(i = 0; i < numChops; i++)
	{
		// chop the support files directory, i.e. go up one directory
		p = strrchr(target_path,NEWDIRDELIMITER);
		if(p) *p = 0;
	}
	strcpy(kmz_path, target_path);
	strcat(kmz_path,".kmz");
	
	//sprintf(cmd,"/usr/bin/zip -r kmz_path target_path\n");
	sprintf(cmd,"zip -r kmz_path target_path");
	status = system(cmd);
	return err;
}

OSErr TModel::SaveMossLEFile (Seconds fileTime, short fileNumber)
{
	// use gMossPath
	OSErr err = 0;
	long n, i, j, mapNum, count, createdDirID;
	Seconds seconds;
	TLEList *list;
	TMap *bestMap;
	LEHeaderRec header;
	LEPropRec theLEPropRec;
	LERec theLE;
	BFPB fileMS3,fileMS4,fileMS5,fileMS6,fileMS7;
	char *p, path[256], fileName[256] ;
	long fileItemNum,fileItemNum_4_5,fileItemNum_6_7;
	char fileNumStr[64] = "";
	char out[512];
	char trajectoryTime[64], trajectoryDate[64], preparedTime[32], preparedDate[32];
	Boolean bWriteUncertaintyLEs = this->IsUncertain();
	double halfLife;
	
	char logoFileName[256];
	StrcpyLogoFileName(logoFileName);
	
	
	/*if (!bWriteUncertaintyLEs) 
	{
		MULTICHOICEALERT2(1692,"",TRUE);	// ok is only option
		return -1;
	}*/
	fileMS3.f = fileMS4.f = fileMS5.f = fileMS6.f = fileMS7.f = 0;
	
	if(fileNumber >=0)
	{		
		sprintf(fileNumStr, "%03hd",fileNumber);
	}
	else 
	{ // a negative fileNumber means use the file name the user selected
		fileNumStr[0] = 0;
	}
	

	/////////////////////////////////////////////////
	// write file 3
	strcpy(fileName,gMossPath);
	strcat(fileName,fileNumStr);
	strcat(fileName,".ms3");
	StrToLower(fileName); // // Jill has requested we force lower case to better support ArcView on unix machines
	
	
	/////////////////////
	#ifdef MAC
	{
		// JLM 8/2/99 check the file name length on the MAC 
		char copyPath[256],shortFileName[256];
		strcpy(copyPath,fileName);
		SplitPathFile(copyPath,shortFileName);
		if(strlen(shortFileName) > 31)
		{	// MAC file name length will be exceeded
			long addedLen = 4; 
			long maxLen = 31 - addedLen;
			char msg[256];
			sprintf(msg,"The name of the file is too long.  Please choose a file name no longer than %ld characters.",maxLen);
			printError(msg);
			return -1;
		}
	}
	#endif
	/////////////////////

	hdelete(0, 0, fileName);
	if (err = hcreate(0, 0, fileName, 'ttxt', 'TEXT'))
		{ TechError("SaveMossLEFile()", "hcreate()", err); goto done; }
	
	if (err = FSOpenBuf(0, 0, fileName, &fileMS3, 2000, FALSE))
		{ TechError("SaveMossLEFile()", "FSOpenBuf()", err); goto done; }
	/////////////////////////////////////////////////
	
	sprintf(out, "0, SPILLID: %s%s", settings.headerSPILLID, UNIXNEWLINESTRING);
	count = strlen(out);
	if (err = FSWriteBuf(&fileMS3, &count, out))    goto FSWriteBufError;
	////

	sprintf(out, "0, FROM: %s%s", settings.headerFROM, UNIXNEWLINESTRING);
	count = strlen(out);
	if (err = FSWriteBuf(&fileMS3, &count, out))    goto FSWriteBufError;
	////

	sprintf(out, "0, CONTACT: %s%s", settings.headerCONTACT, UNIXNEWLINESTRING);
	count = strlen(out);
	if (err = FSWriteBuf(&fileMS3, &count, out))   goto FSWriteBufError; 
	////

	sprintf(out, "0, ISSUED: %s, %s%s", sharedPTime, sharedPDate, UNIXNEWLINESTRING);
	count = strlen(out);
	if (err = FSWriteBuf(&fileMS3, &count, out))   goto FSWriteBufError; 
	///////////////////////////////

	if(fileNumber >= 0)
	{	// the trajectoryTime is different for each file , so we must set it automatically
		GetDateTimeStrings(trajectoryTime, trajectoryDate, preparedTime, preparedDate);
	}
	else
	{ // the user saw the trajectoryTime, so use what they set it to be
		strcpy(trajectoryTime,sharedTTime);
		strcpy(trajectoryDate,sharedTDate);
	}
	
	sprintf(out, "0, VALIDFOR: %s, %s%s",trajectoryTime,trajectoryDate,UNIXNEWLINESTRING);
	count = strlen(out);
	if (err = FSWriteBuf(&fileMS3, &count, out))    goto FSWriteBufError;
	///////////////////////////////////

	sprintf(out, "0, ADDLEDATA: %s", UNIXNEWLINESTRING);
	count = strlen(out);
	if (err = FSWriteBuf(&fileMS3, &count, out))    goto FSWriteBufError;
	////

	// the caveat lines
	for(i = 0;i<5;i++)
	{
		sprintf(out, "0, CAVEAT: %s%s",settings.caveat[i], UNIXNEWLINESTRING);
		count = strlen(out);
		if (err = FSWriteBuf(&fileMS3, &count, out))    goto FSWriteBufError;
	}
	
	if(logoFileName[0])
	{
		sprintf(out, "0, FROMLOGO: %s%s",logoFileName, UNIXNEWLINESTRING);
		count = strlen(out);
		if (err = FSWriteBuf(&fileMS3, &count, out))    goto FSWriteBufError;
	}
	////
	
	if (fileMS3.f) FSCloseBuf(&fileMS3);
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	////////
	strcpy(fileName,gMossPath);
	strcat(fileName,fileNumStr);
	strcat(fileName,".ms4");
	StrToLower(fileName); // // Jill has requested we force lower case to better support ArcView on unix machines

	hdelete(0, 0, fileName);
	if (err = hcreate(0, 0, fileName, 'ttxt', 'TEXT'))
		{ TechError("SaveMossLEFile()", "hcreate()", err); goto done; }
	
	if (err = FSOpenBuf(0, 0, fileName, &fileMS4, 10000, FALSE))
		{ TechError("SaveMossLEFile()", "FSOpenBuf()", err); goto done; }
	/////////////////////////////////////////////////
	
	////////
	strcpy(fileName,gMossPath);
	strcat(fileName,fileNumStr);
	strcat(fileName,".ms5");
	StrToLower(fileName); // // Jill has requested we force lower case to better support ArcView on unix machines

	hdelete(0, 0, fileName);
	if (err = hcreate(0, 0, fileName, 'ttxt', 'TEXT'))
		{ TechError("SaveMossLEFile()", "hcreate()", err); goto done; }
	
	if (err = FSOpenBuf(0, 0, fileName, &fileMS5, 10000, FALSE))
		{ TechError("SaveMossLEFile()", "FSOpenBuf()", err); goto done; }
	
	/////////////////////////////////////////////////
	////////
	strcpy(fileName,gMossPath);
	strcat(fileName,fileNumStr);
	strcat(fileName,".ms6");
	StrToLower(fileName); // // Jill has requested we force lower case to better support ArcView on unix machines

	hdelete(0, 0, fileName); // always delete 6 & 7 , even if bWriteUncertaintyLEs is false !!!
	if(bWriteUncertaintyLEs)
	{
		if (err = hcreate(0, 0, fileName, 'ttxt', 'TEXT'))
			{ TechError("SaveMossLEFile()", "hcreate()", err); goto done; }
		
		if (err = FSOpenBuf(0, 0, fileName, &fileMS6, 10000, FALSE))
			{ TechError("SaveMossLEFile()", "FSOpenBuf()", err); goto done; }
	}
		
	/////////////////////////////////////////////////
	////////
	strcpy(fileName,gMossPath);
	strcat(fileName,fileNumStr);
	strcat(fileName,".ms7");
	StrToLower(fileName); // // Jill has requested we force lower case to better support ArcView on unix machines

	hdelete(0, 0, fileName); // always delete 6 & 7 , even if bWriteUncertaintyLEs is false !!!
	//
	if(bWriteUncertaintyLEs)
	{
		if (err = hcreate(0, 0, fileName, 'ttxt', 'TEXT'))
			{ TechError("SaveMossLEFile()", "hcreate()", err); goto done; }
		
		if (err = FSOpenBuf(0, 0, fileName, &fileMS7, 10000, FALSE))
			{ TechError("SaveMossLEFile()", "FSOpenBuf()", err); goto done; }
	}
	
	/////////////////////////////////////////////////

	// write files 4,5,6,7
	fileItemNum_4_5 = 0;
	fileItemNum_6_7 = 0;
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		Boolean isUncertainLE;
		LESetsList->GetListItem((Ptr)&list, i);
		if (!list->bActive) continue;
		
		isUncertainLE = (list -> GetLEType () == UNCERTAINTY_LE);
		if(isUncertainLE && !bWriteUncertaintyLEs) continue; //don't write out uncertainty LEs unless uncertainty is on

		halfLife = (*(dynamic_cast<TOLEList*>(list))).fSetSummary.halfLife;
		for (j = 0 ; j < list->numOfLEs ; j++) 
		{
			list -> GetLE (j, &theLE);
			
			// JLM, 3/1/99 it was decided to skip EVAPORATED and NOTRELEASED LE's
			switch (theLE.statusCode) {
				case OILSTAT_INWATER: 
				case OILSTAT_OFFMAPS: 
				case OILSTAT_ONLAND: 
					break; // these are the only types written out
					
				default: continue; // skip these LE's	
			}
			////////////////////////////////////////////////
			
			if(isUncertainLE) {
				fileItemNum_6_7++;
				fileItemNum = fileItemNum_6_7;
			}
			else {
				fileItemNum_4_5++;
				fileItemNum = fileItemNum_4_5;
			}

			////////////////////////////
			// write lines to file 4
			sprintf(out, "%5ld          %-30s     %5hd%s",
				 -fileItemNum, "LE POINT", 1, UNIXNEWLINESTRING);
			count = strlen(out);
			
			if(isUncertainLE) {
				if (fileMS6.f) err = FSWriteBuf(&fileMS6, &count, out);
			}
			else {// forecast LE
				if (fileMS4.f) err = FSWriteBuf(&fileMS4, &count, out);
			}

			if(err) goto FSWriteBufError;
			////
			sprintf(out, "%10.5f%10.5f%2d%s",
				 //(double)-theLE.p.pLong / 1000000.0,
				 // JLM 7/13/99, the MOSS files have negative as west, just like we do, fixes bug reported by user
				 (double)theLE.p.pLong / 1000000.0,
				 (double)theLE.p.pLat  / 1000000.0, 0, UNIXNEWLINESTRING);
			count = strlen(out);
			
			if(isUncertainLE){
				if (fileMS6.f) err = FSWriteBuf(&fileMS6, &count, out);
			}
			else {
				// forecast LE
				if (fileMS4.f) err = FSWriteBuf(&fileMS4, &count, out);
			}

			if(err) goto FSWriteBufError;
			/////////////////////////////////
				
			
			/// write lines to file 5
			///////////////
			{
				// item num
				// keyword  ABSOLUTEMASS or RELATIVEPROBABILITY
				// substance keyword GAS,JP4, etc
				// depth in meters
				// mass in kilograms
				// density in g/cu cm
				// age in hours since release
				// one of INWATER,ONBEACH,OFFMAP
				char* keyWord;
				char * pollutantKey;
				char* status;
				double ageInHrs, elapsedTimeInHrs, mass;
				
				if (bHindcast)	
					elapsedTimeInHrs = (double)(theLE.releaseTime - GetModelTime()) / 3600.;
				else
					elapsedTimeInHrs = (double)(GetModelTime() - theLE.releaseTime) / 3600.;
				
				
				if(isUncertainLE){
					keyWord = "RELATIVEPROBABILITY";
				}
				else {// forecast LE
					keyWord = "ABSOLUTEMASS";
				}
				
				switch (theLE.pollutantType) {
					case OIL_GAS: pollutantKey = "GAS"; break;
					case OIL_JETFUELS: pollutantKey = "JP4"; break;
					case OIL_DIESEL: pollutantKey = "DIESEL"; break;
					case OIL_4: pollutantKey = "IFO"; break;
					case OIL_CRUDE: pollutantKey = "MEDIUMCRUDE"; break;
					case OIL_6: pollutantKey = "BUNKER"; break;
					
					default: // fall thru
					case OIL_USER1:// fall thru 
					case OIL_USER2: // fall thru 
					case OIL_COMBINATION: // fall thru
					case OIL_CONSERVATIVE: pollutantKey = "CONSERVATIVE"; break;
					
					// other KEYS: JP5, LIGHTCRUDE, HEAVYCRUDE, LAPIO
				}


				switch (theLE.statusCode) {
					case OILSTAT_INWATER: status = "INWATER"; break;
					case OILSTAT_OFFMAPS: status = "OFFMAP"; break;
					case OILSTAT_ONLAND: status = "ONBEACH"; break;
					case OILSTAT_EVAPORATED: status = "EVAPORATED"; break;
					case OILSTAT_NOTRELEASED: status = "NOTRELEASED"; break;
					default: status = ""; break;
				}
				
				//ageInHrs = theLE.ageInHrsWhenReleased;
				ageInHrs = theLE.ageInHrsWhenReleased + elapsedTimeInHrs;
				mass = VolumeMassToKilograms(GetLEMass(theLE,halfLife), theLE.density, list->GetMassUnits());	// applies half life for chemicals and converts units
				sprintf(out, "%ld, %s, %s, %lf, %lf, %lf, %lf, %s%s",
					-fileItemNum, keyWord, pollutantKey,
					//theLE.z, theLE.mass, theLE.density, ageInHrs, status, UNIXNEWLINESTRING);
					theLE.z, mass, theLE.density, ageInHrs, status, UNIXNEWLINESTRING);
				count = strlen(out);
			
				if(isUncertainLE)
				{
					if (fileMS7.f) err = FSWriteBuf(&fileMS7, &count, out);
				}
				else // forecast LE
				{
					if (fileMS5.f) err = FSWriteBuf(&fileMS5, &count, out);
				}

				if(err) goto FSWriteBufError;

			}
		}
	}
	goto done;

FSWriteBufError:
	TechError("SaveMossLEFile()", "FSWriteBuf()", err);
	
done:	
	if (fileMS7.f) FSCloseBuf(&fileMS7);
	if (fileMS6.f) FSCloseBuf(&fileMS6);
	if (fileMS5.f) FSCloseBuf(&fileMS5);
	if (fileMS4.f) FSCloseBuf(&fileMS4);
	if (fileMS3.f) FSCloseBuf(&fileMS3);

	// for Jill write out the bmp files in the same directory
	if(!err)
	{
		char logoFileName[256];
		StrcpyLogoFileName(logoFileName);
		
		if(logoFileName[0])
		{
			// since this function is called for a series of moss files we only want to write the file out one time
			Boolean writeItThisTime = false;
			if(fileNumber >= 0 && fileNumber == 0)
			{	// it's a series, write it out with the first file and not with the others
				writeItThisTime = true;
			}
			else if(fileNumber < 0)
			{	// it is a single set of MOSS files
				writeItThisTime = true;
			}
			
			if(writeItThisTime)
			{
				short logoResNum = LogoResNumber(logoFileName);
				if(logoResNum > 0) (void)WriteBMPResourceToNearbyFile(logoFileName,logoResNum,gMossPath,0);
				else
				{ // user logo file
					Handle h = GetUserLogoHandle(FALSE);
					if(h)
					{
						(void)WriteBMPResourceToNearbyFile(logoFileName,-1,gMossPath,(CHARH)h);
						DisposeHandle(h); h = 0;
					}
				}
			}
		}
	}
	return err;
	
	
}

/*OSErr TModel::SaveShapeLEFile (Seconds fileTime, short fileNumber)
{	// Trying this out - would need to put call in Output, comment out include shapefil.h, and add function to header file
	OSErr err = 0;
	long n, i, j, mapNum, count, createdDirID;
	Seconds seconds;
	TLEList *list;
	TMap *bestMap;
	LEHeaderRec header;
	LEPropRec theLEPropRec;
	LERec theLE;
	char *p, path[256], fileName[256], fieldStr[32] ;
	long fileItemNum,fileItemNum_forecast=0,fileItemNum_uncertain=0;
	char out[512],forecastLEFName[256],uncertainLEFName[256];
	char trajectoryTime[64], trajectoryDate[64], preparedTime[32], preparedDate[32];
	Boolean bWriteUncertaintyLEs = this->IsUncertain();
	SHPHandle hSHP;
	int nShapeType=SHPT_POINT, nVertices, nVMax;
	double *padfX, *padfY, *padfZ = NULL;
	SHPObject *psObject;
    DBFHandle	hDBF;
	
	
	strcpy(fieldStr,"LEType");
	/////////////////////////////////////////////////
	err = GetOutputFileName (fileNumber, FORECAST_LE,    forecastLEFName);
	if(err) return err; // JLM 8/2/99
	err = GetOutputFileName (fileNumber, UNCERTAINTY_LE, uncertainLEFName);
	if(err) return err; // JLM 8/2/99}
	/////////////////////
	
	hSHP = SHPCreate(forecastLEFName,SHPT_POINT);
    if( hSHP == NULL )
    {
		printf( "SHPCreate(%s) failed.\n", forecastLEFName );
		err = -1; goto done;
    }
	hdelete(0, 0, forecastLEFName);
    hDBF = DBFCreate( forecastLEFName );
    if( hDBF == NULL )
    {
		printf( "DBFCreate(%s) failed.\n", forecastLEFName );
		err = -1; goto done;
    }
	if( DBFAddField( hDBF, fieldStr, FTString, 32, 0 ) == -1 )
	{
		printf( "DBFAddField(%s,FTString,%d,0) failed.\n",
			   fieldStr, 6 );
		err = -1; goto done;
	}
	/////////////////////////////////////////////////
	//if (IsUncertain ())  // open uncertainty LE file only if model is in uncertain mode
	//{
	 //}
	/////////////////////////////////////////////////
	
	//if(fileNumber >= 0)	// may want to do something with time
	 //{	// the trajectoryTime is different for each file , so we must set it automatically
	 //GetDateTimeStrings(trajectoryTime, trajectoryDate, preparedTime, preparedDate);
	// }
	// else
	 //{ // the user saw the trajectoryTime, so use what they set it to be
	// strcpy(trajectoryTime,sharedTTime);
	// strcpy(trajectoryDate,sharedTDate);
	 //}
	
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	
	/////////////////////////////////////////////////
	// add up all the LEs from all spills
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		Boolean isUncertainLE;
		LESetsList->GetListItem((Ptr)&list, i);
		if (!list->bActive) continue;
		
		isUncertainLE = (list -> GetLEType () == UNCERTAINTY_LE);
		if(isUncertainLE && !bWriteUncertaintyLEs) continue; //don't write out uncertainty LEs unless uncertainty is on
		
		//nVMax = list->numOfLEs;	// for one spill only
		nVMax = 1;	// for one spill only
		padfX = (double *) malloc(sizeof(double)*nVMax);
		padfY = (double *) malloc(sizeof(double)*nVMax);
		nVertices = nVMax;
		for (j = 0 ; j < list->numOfLEs ; j++) 
		{
			list -> GetLE (j, &theLE);
			//padfX[j] = theLE.p.pLong;
			//padfY[j] = theLE.p.pLat;
			padfX[0] = theLE.p.pLong / 1000000.;
			padfY[0] = theLE.p.pLat / 1000000.;
			// JLM, 3/1/99 it was decided to skip EVAPORATED and NOTRELEASED LE's
			switch (theLE.statusCode) {
				case OILSTAT_INWATER: 
				case OILSTAT_OFFMAPS: 
				case OILSTAT_ONLAND: 
					break; // these are the only types written out
					
				default: continue; // skip these LE's	
			}
			////////////////////////////////////////////////
			
			psObject = SHPCreateSimpleObject(nShapeType, nVertices, padfX, padfY, padfZ);
			SHPWriteObject(hSHP, -1, psObject);
			SHPDestroyObject(psObject);
			DBFWriteStringAttribute(hDBF, j, 0, "forecastLE" );
		}
		//psObject = SHPCreateObject(nShapeType, -1, nParts, panParts, NULL, nVertices, padfX, padfY, padfZ, padfM);
		//psObject = SHPCreateSimpleObject(nShapeType, nVertices, padfX, padfY, padfZ);
		//SHPWriteObject(hSHP, -1, psObject);
		//SHPDestroyObject(psObject);
		SHPClose(hSHP);
		DBFClose(hDBF);
		free(padfX);
		free(padfY);
		free(padfZ);
	}
	hSHP = SHPOpen(forecastLEFName,"rb");
	hDBF = DBFOpen(forecastLEFName,"rb");
	if (hSHP==NULL || hDBF == NULL)
	{
		goto done;
	}
	else 
	{
		int numEntries, numFields, numRecords;
		const char *attStr;
		SHPGetInfo(hSHP,&numEntries,&nShapeType,NULL,NULL); 
		numFields = DBFGetFieldCount(hDBF);
		numRecords = DBFGetRecordCount(hDBF);
		//padfX = (double *) malloc(sizeof(double)*numEntries);
		//padfY = (double *) malloc(sizeof(double)*numEntries);
		//padfZ = NULL;
		//nVertices = numEntries;
		//psObject = SHPCreateSimpleObject(nShapeType, nVertices, padfX, padfY, padfZ);
		for (j=0;j<numEntries;j++)
		{
			psObject = SHPReadObject(hSHP, j);
			attStr = DBFReadStringAttribute(hDBF, j, 0);
		}
		//psObject = SHPReadObject(hSHP, numEntries-1);
		SHPDestroyObject(psObject);
		//free(padfX);
		//free(padfY);
	}
	SHPClose(hSHP);
	
	goto done;
	
	
done:	
	
	return err;	
	
}*/

OSErr TModel::SaveSimpleAsciiLEFile (Seconds fileTime, short fileNumber)
{
	// use gMossPath
	OSErr err = 0;
	long n, i, j, mapNum, count, createdDirID;
	Seconds seconds;
	TLEList *list;
	TMap *bestMap;
	LEHeaderRec header;
	LEPropRec theLEPropRec;
	LERec theLE;
	BFPB forecastFile,uncertainFile;
	char *p, path[256], fileName[256] ;
	long fileItemNum,fileItemNum_forecast=0,fileItemNum_uncertain=0;
	char out[512],forecastLEFName[256],uncertainLEFName[256];
	char trajectoryTime[64], trajectoryDate[64], preparedTime[32], preparedDate[32];
	Boolean bWriteUncertaintyLEs = this->IsUncertain();
	double halfLife;
	
	
	forecastFile.f = uncertainFile.f = 0;
	
	/////////////////////////////////////////////////
	err = GetOutputFileName (fileNumber, FORECAST_LE,    forecastLEFName);
	if(err) return err; // JLM 8/2/99
	err = GetOutputFileName (fileNumber, UNCERTAINTY_LE, uncertainLEFName);
	if(err) return err; // JLM 8/2/99}
	/////////////////////

	hdelete(0, 0, fileName);
	if (err = hcreate(0, 0, forecastLEFName, 'ttxt', 'TEXT'))
		{ TechError("SaveAsciiLEFile()", "hcreate()", err); goto done; }
	
	if (err = FSOpenBuf(0, 0, forecastLEFName, &forecastFile, 50000, FALSE))
		{ TechError("SaveAsciiLEFile()", "FSOpenBuf()", err); goto done; }
	/////////////////////////////////////////////////
	if (IsUncertain ())  // open uncertainty LE file only if model is in uncertain mode
	{
		if (err = hcreate(0, 0, uncertainLEFName, '\?\?\?\?', 'BINA'))
			{ TechError("SaveOSSMLEFile()", "hcreate()", err); return err; }
		
		if (err = FSOpenBuf(0, 0, uncertainLEFName, &uncertainFile, 50000, FALSE))
			{ TechError("SaveAsciiLEFile()", "FSOpenBuf()", err); return err; }
	}
	/////////////////////////////////////////////////
	

	if(fileNumber >= 0)
	{	// the trajectoryTime is different for each file , so we must set it automatically
		GetDateTimeStrings(trajectoryTime, trajectoryDate, preparedTime, preparedDate);
	}
	else
	{ // the user saw the trajectoryTime, so use what they set it to be
		strcpy(trajectoryTime,sharedTTime);
		strcpy(trajectoryDate,sharedTDate);
	}
	
	sprintf(out, "0, VALIDFOR: %s, %s%s",trajectoryTime,trajectoryDate,UNIXNEWLINESTRING);
	count = strlen(out);
	if (err = FSWriteBuf(&forecastFile, &count, out))    goto FSWriteBufError;
	if (bWriteUncertaintyLEs) {if (err = FSWriteBuf(&uncertainFile, &count, out))    goto FSWriteBufError;}
	///////////////////////////////////

	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	
	/////////////////////////////////////////////////

	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		Boolean isUncertainLE;
		LESetsList->GetListItem((Ptr)&list, i);
		if (!list->bActive) continue;
		
		isUncertainLE = (list -> GetLEType () == UNCERTAINTY_LE);
		if(isUncertainLE && !bWriteUncertaintyLEs) continue; //don't write out uncertainty LEs unless uncertainty is on

		halfLife = (*(dynamic_cast<TOLEList*>(list))).fSetSummary.halfLife;
		for (j = 0 ; j < list->numOfLEs ; j++) 
		{
			list -> GetLE (j, &theLE);
			
			// JLM, 3/1/99 it was decided to skip EVAPORATED and NOTRELEASED LE's
			switch (theLE.statusCode) {
				case OILSTAT_INWATER: 
				case OILSTAT_OFFMAPS: 
				case OILSTAT_ONLAND: 
					break; // these are the only types written out
					
				default: continue; // skip these LE's	
			}
			////////////////////////////////////////////////
			
			if(isUncertainLE) {
				fileItemNum_uncertain++;
				fileItemNum = fileItemNum_uncertain;
			}
			else {
				fileItemNum_forecast++;
				fileItemNum = fileItemNum_forecast;
			}

			////////////////////////////
			// write lines to file 4
			sprintf(out, "%5ld          %-30s     %5hd%s",
				 -fileItemNum, "LE POINT", 1, UNIXNEWLINESTRING);
			count = strlen(out);
			
			if(isUncertainLE) {
				if (uncertainFile.f) err = FSWriteBuf(&uncertainFile, &count, out);
			}
			else {// forecast LE
				if (forecastFile.f) err = FSWriteBuf(&forecastFile, &count, out);
			}

			if(err) goto FSWriteBufError;
			////
			sprintf(out, "%10.5f%10.5f%2d%s",
				 //(double)-theLE.p.pLong / 1000000.0,
				 // JLM 7/13/99, the MOSS files have negative as west, just like we do, fixes bug reported by user
				 (double)theLE.p.pLong / 1000000.0,
				 (double)theLE.p.pLat  / 1000000.0, 0, UNIXNEWLINESTRING);
			count = strlen(out);
			
			if(isUncertainLE){
				if (uncertainFile.f) err = FSWriteBuf(&uncertainFile, &count, out);
			}
			else {
				// forecast LE
				if (forecastFile.f) err = FSWriteBuf(&forecastFile, &count, out);
			}

			if(err) goto FSWriteBufError;
			/////////////////////////////////
				
			
			/// write lines to file 5
			///////////////
			{
				// item num
				// keyword  ABSOLUTEMASS or RELATIVEPROBABILITY
				// substance keyword GAS,JP4, etc
				// depth in meters
				// mass in kilograms
				// density in g/cu cm
				// age in hours since release
				// one of INWATER,ONBEACH,OFFMAP
				char* keyWord;
				char * pollutantKey;
				char* status;
				float ageInHrs, elapsedTimeInHrs, mass;
				
				elapsedTimeInHrs = (GetModelTime() - theLE.releaseTime) / 3600.;
				
				if(isUncertainLE){
					keyWord = "RELATIVEPROBABILITY";
				}
				else {// forecast LE
					keyWord = "ABSOLUTEMASS";
				}
				
				switch (theLE.pollutantType) {
					case OIL_GAS: pollutantKey = "GAS"; break;
					case OIL_JETFUELS: pollutantKey = "JP4"; break;
					case OIL_DIESEL: pollutantKey = "DIESEL"; break;
					case OIL_4: pollutantKey = "IFO"; break;
					case OIL_CRUDE: pollutantKey = "MEDIUMCRUDE"; break;
					case OIL_6: pollutantKey = "BUNKER"; break;
					
					default: // fall thru
					case OIL_USER1:// fall thru 
					case OIL_USER2: // fall thru 
					case OIL_COMBINATION: // fall thru
					case OIL_CONSERVATIVE: pollutantKey = "CONSERVATIVE"; break;
					
					// other KEYS: JP5, LIGHTCRUDE, HEAVYCRUDE, LAPIO
				}


				switch (theLE.statusCode) {
					case OILSTAT_INWATER: status = "INWATER"; break;
					case OILSTAT_OFFMAPS: status = "OFFMAP"; break;
					case OILSTAT_ONLAND: status = "ONBEACH"; break;
					case OILSTAT_EVAPORATED: status = "EVAPORATED"; break;
					case OILSTAT_NOTRELEASED: status = "NOTRELEASED"; break;
					default: status = ""; break;
				}
				
				//ageInHrs = theLE.ageInHrsWhenReleased;
				ageInHrs = theLE.ageInHrsWhenReleased + elapsedTimeInHrs;
				mass = GetLEMass(theLE,halfLife);	// applies half life for chemicals
				sprintf(out, "%ld, %s, %s, %lf, %lf, %lf, %lf, %s%s",
					-fileItemNum, keyWord, pollutantKey,
					//theLE.z, theLE.mass, theLE.density, ageInHrs, status, UNIXNEWLINESTRING);
					theLE.z, mass, theLE.density, ageInHrs, status, UNIXNEWLINESTRING);
				count = strlen(out);
			
				if(isUncertainLE)
				{
					if (uncertainFile.f) err = FSWriteBuf(&uncertainFile, &count, out);
				}
				else // forecast LE
				{
					if (forecastFile.f) err = FSWriteBuf(&forecastFile, &count, out);
				}

				if(err) goto FSWriteBufError;

			}
		}
	}
	goto done;

FSWriteBufError:
	TechError("SaveAsciiLEFile()", "FSWriteBuf()", err);
	
done:	
	if (forecastFile.f) FSCloseBuf(&forecastFile);
	if (uncertainFile.f) FSCloseBuf(&uncertainFile);

	return err;	
	
}


OSErr TModel::FirstStepUserInputChecks(void)
{
	OSErr err = noErr;
	
	
	// check that the spills are in the modeled duration
	// code goes here, ?? should this be on the first step or every step ?
	// check if dispersing and if so, check if vertical diffusion has been set.
	// If not, alert the user, but let them continue.
	{	
		long i, j, n, m, numLEs;
		TLEList *thisLEList;
		LETYPE leType;
		ClassID thisClassID;
		Seconds startTime = this->GetStartTime();
		Seconds endTime = this->GetEndTime();
		Boolean bReleasedBeforeStartTime = FALSE;
		Boolean bReleasedAfterEndTime = FALSE;
		Boolean bOssmLeProblem = FALSE;
		Boolean bSprayedLeProblem = FALSE;
		Seconds spillStartTimeOfProblemSpill = 0;
		long numSpillsUserSees = 0;
		Boolean bSomeSpillIntersectsTheModeledInterval = FALSE;
		Boolean bSomeSpillIsDispersed = FALSE;
		Seconds earliestStartTime = this->GetEndTime(), latestStartTime = this->GetStartTime();
		char earliestSpillName[256],latestSpillName[256];
		Boolean bThereIsALineSourceSpill = false;
		

		// need to fix for hindcasting
		for (i = 0, n = LESetsList->GetItemCount() ; i < n && !err; i++) 
		{
			LESetsList->GetListItem((Ptr)&thisLEList, i);
			leType = thisLEList -> GetLEType();
			if(leType == UNCERTAINTY_LE && !this->IsUncertain()) continue;
			if (!thisLEList->bActive) continue;
			numSpillsUserSees++;

			thisClassID = thisLEList -> GetClassID();
			if(thisClassID == TYPE_OSSMLELIST || thisClassID == TYPE_SPRAYLELIST || thisClassID == TYPE_CDOGLELIST)
			{
				TOLEList *thisOLEList = (TOLEList*)thisLEList; // typecast
				if (thisOLEList->fDispersantData.bDisperseOil || thisOLEList->fAdiosDataH) bSomeSpillIsDispersed = true;
				Seconds thisStartTime = thisOLEList->fSetSummary.startRelTime;
				Seconds thisEndTime;
				if (thisOLEList->fSetSummary.bWantEndRelTime)
				{
					bThereIsALineSourceSpill = true;
					thisEndTime = thisOLEList->fSetSummary.endRelTime;
				}
				else
					thisEndTime = thisOLEList->fSetSummary.startRelTime;
				if((!bHindcast && thisStartTime < startTime) || (bHindcast && thisEndTime > endTime))
				{
					if (!bHindcast) 
					{
						spillStartTimeOfProblemSpill = thisStartTime;
						bReleasedBeforeStartTime = TRUE;
					}
					else 
					{
						spillStartTimeOfProblemSpill = thisEndTime;
						bReleasedAfterEndTime = TRUE;
					}
					switch(thisClassID)
					{
						case TYPE_CDOGLELIST:
						case TYPE_OSSMLELIST: bOssmLeProblem = TRUE; break;
						case TYPE_SPRAYLELIST: bSprayedLeProblem = TRUE; break;
					}
				}
				if((!bHindcast && thisStartTime > endTime) || (bHindcast && thisEndTime <= startTime))
				{
					if (!bHindcast) 
					{
						spillStartTimeOfProblemSpill = thisStartTime;
						bReleasedAfterEndTime = TRUE;
					}
					else 
					{
						spillStartTimeOfProblemSpill = thisEndTime;
						bReleasedBeforeStartTime = TRUE;
					}
					switch(thisClassID)
					{
						case TYPE_CDOGLELIST:
						case TYPE_OSSMLELIST: bOssmLeProblem = TRUE; break;
						case TYPE_SPRAYLELIST: bSprayedLeProblem = TRUE; break;
					}
				}
				if((!bHindcast && thisStartTime < earliestStartTime )|| (bHindcast && thisEndTime < earliestStartTime) ) 
				{
					if (!bHindcast) earliestStartTime = thisStartTime;
					else earliestStartTime = thisEndTime;
					strcpy(earliestSpillName,((TOLEList*)thisLEList)->fSetSummary.spillName);	// only do this if bReleasedBeforeStartTime ?
				}
				if(thisEndTime > latestStartTime)
				{
					latestStartTime = thisEndTime;
					strcpy(latestSpillName,((TOLEList*)thisLEList)->fSetSummary.spillName);	// only do this if bReleasedBeforeStartTime ?
				}
				if(!bHindcast && thisStartTime < endTime || (bHindcast && thisEndTime > startTime))	// need to fix for hindcast
					bSomeSpillIntersectsTheModeledInterval = TRUE;
			}
		}
		
		if(numSpillsUserSees == 1) 
		{	// there is exactly one spill, so we can be specific
			char msg[256] = "";
			char *what = 0;
			char *when = 0;
			if(bOssmLeProblem) 
			{
				if (!bHindcast || (bHindcast && !bThereIsALineSourceSpill)) what = "The Release Start Time";
				else what = "The Release End Time";
			}
			else if(bSprayedLeProblem)
				what = "The Overflight Time";
			
			if(bReleasedBeforeStartTime) 
				when = "before";
			else if(bReleasedAfterEndTime)
				when = "after";
				
			if(what && when)
			{
				sprintf(msg,"%s of the spill being modeled is %s the time interval being modeled.%s%sChange Model Start Time ?",what,when,NEWLINESTRING,NEWLINESTRING);
			
				if(msg[0]) {
					short buttonSelected  = MULTICHOICEALERT(1682,msg,FALSE);
					switch(buttonSelected){
						case 1:// change
							if (!bHindcast) model -> SetStartTime(earliestStartTime);
							//else model -> SetStartTime(latestStartTime - model->GetDuration());
							else 
							{
								if (bReleasedAfterEndTime)
									model -> SetStartTime(latestStartTime - model->GetDuration());
								else
									model -> SetStartTime(earliestStartTime - model->GetDuration());
							}
							model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
							return -1;// prevent the run
							break;  
						case 3: // cancel
							return -1;// prevent the run
							break;
					}
				}
			}
			//else if(earliestStartTime > startTime && !bHindcast || (latestStartTime < endTime && bHindcast))
			else if(earliestStartTime > startTime && !bHindcast || (latestStartTime < startTime && bHindcast))
			{
				sprintf(msg,"The spill being modeled begins after the Model Start Time.%s%sChange Model Start Time ?",NEWLINESTRING,NEWLINESTRING);

				if(msg[0]) {
					short buttonSelected  = MULTICHOICEALERT(1683,msg,FALSE);
					switch(buttonSelected){
						case 1:// change
							//model -> SetStartTime(earliestStartTime);
							if (!bHindcast) model -> SetStartTime(earliestStartTime);
							else model -> SetStartTime(latestStartTime - model->GetDuration());
							model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
							return -1;// prevent the run
							break;  
						case 3: // don't change
							break;// allow the run
					}
				}
			}

		}
		else if(numSpillsUserSees > 1) 
		{ 
			if(bReleasedBeforeStartTime)
			{
				//if (bSomeSpillIntersectsTheModeledInterval)	// for hindcast do this here, basically reverse the 2 checks
				if (bHindcast)
				{
					if (bSomeSpillIntersectsTheModeledInterval)
					{
							// ok, they probably already know
					}
					else
					{
						//
						printError("All of the spills are released before the time interval being modeled.  Please check your inputs.");
						return -1;// prevent the run
					}
				}
				else
				{
					char msg[256] = "";
					sprintf(msg,"The spill '%s' is released before the time interval being modeled.  Please check your inputs.",earliestSpillName);
					printError(msg);
					//printError("One of the spills is released before the time interval being modeled.  Please check your inputs.");
					return -1;// prevent the run
				}
			}
			else if(bReleasedAfterEndTime)
			{
				if (bHindcast)
				{
					char msg[256] = "";
					sprintf(msg,"The spill '%s' is released after the time interval being modeled.  Please check your inputs.",latestSpillName);
					printError(msg);
					//printError("One of the spills is released before the time interval being modeled.  Please check your inputs.");
					return -1;// prevent the run
				}
				else
				{
					if (bSomeSpillIntersectsTheModeledInterval)
					{
							// ok, they probably already know
					}
					else
					{
						//
						printError("All of the spills are released after the time interval being modeled.  Please check your inputs.");
						return -1;// prevent the run
					}
				}
				
			}
			if(earliestStartTime > startTime && !bHindcast || (latestStartTime < endTime && bHindcast))
			{
				char msg[256] = "";
				sprintf(msg,"All of the spills being modeled begin after the Model Start Time.%s%sChange Model Start Time ?",NEWLINESTRING,NEWLINESTRING);
				short buttonSelected  = MULTICHOICEALERT(1683,msg,FALSE);
				switch(buttonSelected){
					case 1:// change
						if (!bHindcast) model -> SetStartTime(earliestStartTime);
						else model -> SetStartTime(latestStartTime - model->GetDuration());
						model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
						return -1;// prevent the run
						break;  
					case 3: // don't change
						break;// allow the run
				}
			}
		}
		if(bSomeSpillIsDispersed)
		{
			TRandom3D *vertDiffMover = Get3DDiffusionMover();
			if (!vertDiffMover)
			{
				char msg[256] = "";
				sprintf(msg,"There is no vertical diffusion. The spill will not move vertically after it is dispersed.%s%sContinue the run ?",NEWLINESTRING,NEWLINESTRING);
				short buttonSelected  = MULTICHOICEALERT(1690,msg,FALSE);
				switch(buttonSelected){
					case 1:// continue
						break; // allow the run 
					case 3: // cancel
						return -1;// prevent the run
						break;
				}
			}
		}

	}
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	// code goes here, check the winds, etc
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////

	return err;
}

OSErr TModel::ExportBudgetTableHdl(char* path)
{
	long i, j, n;
	TLEList *thisLEList;
	char* suggestedFileName = "BudgetTable.dat";
	DateTimeRec dateTime;
	Seconds time;
	double totalMass, amttotal,amtevap,amtbeached,amtoffmap,amtfloating,amtreleased,amtdispersed,amtremoved=0;
	BudgetTableData budgetTable; 
	BudgetTableDataH totalBudgetTableH = 0;
	long numOutputValues;
	float timeInHours;
	char buffer[512],unitsStr[64],timeStr[64],massStr[64];
	char amtEvapStr[64],amtDispStr[64],amtBeachedStr[64],amtRelStr[64],amtFloatStr[64],amtOffStr[64],amtRemStr[64];
	short massUnits, totalBudgetMassUnits;
	BFPB bfpb;
	OSErr err = 0;

	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }


	// get the totals
	totalBudgetMassUnits = GetMassUnitsForTotals();	// use first LE set
	GetLeUnitsStr(unitsStr,totalBudgetMassUnits);
	GetTotalAmountSpilled(totalBudgetMassUnits,&amttotal);
	//GetTotalAmountStatistics(totalBudgetMassUnits,&amttotal,&amtreleased,&amtevap,&amtdispersed,&amtbeached,&amtoffmap,&amtfloating);
	StringWithoutTrailingZeros(massStr,amttotal,3);
	strcpy(buffer,"Amount Spilled - ");
	strcat(buffer,massStr);
	strcat(buffer," ");
	strcat(buffer,unitsStr);
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;

	GetTotalBudgetTableHdl(totalBudgetMassUnits,&totalBudgetTableH);

	//strcpy(buffer,"Hr\t\tRel\t   Float\t   Evap\t   Disp\t   Beach\t   OffMap");
	//strcpy(buffer,"Hr\tRel\tFloat\tEvap\tDisp\tBeach\tOffMap");
	strcpy(buffer,"Hr\tRel\tFloat\tEvap\tDisp\tBeach\tOffMap\tRemoved");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	//if (err = WriteMacValue(bfpb, buffer, strlen(buffer))) goto done;
	numOutputValues = _GetHandleSize((Handle)totalBudgetTableH)/sizeof(BudgetTableData);

	for(i = 0; i< numOutputValues;i++)
	{
		budgetTable = INDEXH(totalBudgetTableH,i);
		time = budgetTable.timeAfterSpill;
		amtreleased = budgetTable.amountReleased;
		amtfloating = budgetTable.amountFloating;
		amtdispersed = budgetTable.amountDispersed;
		amtevap = budgetTable.amountEvaporated;
		amtbeached = budgetTable.amountBeached;
		amtoffmap = budgetTable.amountOffMap;
		amtremoved = budgetTable.amountRemoved;

		timeInHours = (float)time/(float)model->LEDumpInterval;
		//StringWithoutTrailingZeros(timeStr,time/model->LEDumpInterval,3);
		StringWithoutTrailingZeros(timeStr,timeInHours,3);
		StringWithoutTrailingZeros(amtEvapStr,amtevap,3);
		StringWithoutTrailingZeros(amtDispStr,amtdispersed,3);
		StringWithoutTrailingZeros(amtFloatStr,amtfloating,3);
		StringWithoutTrailingZeros(amtRelStr,amtreleased,3);
		StringWithoutTrailingZeros(amtBeachedStr,amtbeached,3);
		StringWithoutTrailingZeros(amtOffStr,amtoffmap,3);
		StringWithoutTrailingZeros(amtRemStr,amtremoved,3);
		/////
		strcpy(buffer,timeStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtRelStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtFloatStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtEvapStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtDispStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtBeachedStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtOffStr);
		/*if (amtremoved>0)*/ {strcat(buffer,"\t");strcat(buffer,amtRemStr);}
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		//if (err = WriteMacValue(bfpb, buffer, strlen(buffer))) goto done;
	}
	
	// This does the same thing	a different way
	/*for (i = 0, n = model->LESetsList->GetItemCount() ; i < n ; i++) {
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		if(thisLEList -> GetLEType() == UNCERTAINTY_LE ) continue;
		((TOLEList*)thisLEList) -> ExportBudgetTableHdl(path,&bfpb);	// total all stats ??
	}*/

done:
	// 
	FSCloseBuf(&bfpb);
	if (totalBudgetTableH) {DisposeHandle((Handle)totalBudgetTableH); totalBudgetTableH = 0;}
	if(err) {	
		// the user has already been told there was a problem
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////


OSErr TModel::SquirrelAwayLastComputedTimeStuff (void)
{
	OSErr err =0;
	CMyList* saveLESetsList;
	// we can just clear the savedList and swap the CList classes
	
	// clear squirrel's  list
	this->DisposeLastComputedTimeStuff();
	
	// swap
	saveLESetsList = this->LESetsList;
	this->LESetsList = this->fSquirreledLastComputeTimeLEList;
	this->fSquirreledLastComputeTimeLEList = saveLESetsList;
	
	return err;
}
/////////////////////////////////////////////////


void TModel::SetLastComputeTime(Seconds time)
{
	if(time != this->lastComputeTime) this->DisposeLastComputedTimeStuff();
	this->lastComputeTime = time;
}


OSErr TModel::ReinstateLastComputedTimeStuff (void)
{	// i.e. restore the LEs for lastComputeTime
	OSErr err =0;
	CMyList* saveLESetsList;
	
	if(this->modelTime == this->lastComputeTime) return noErr;
	
	// we can just clear the modelList and swap the CList classes
	this->DisposeModelLEs ();
	// swap
	saveLESetsList = this->LESetsList;
	this->LESetsList = this->fSquirreledLastComputeTimeLEList;
	this->fSquirreledLastComputeTimeLEList = saveLESetsList;

	this->SetModelTime(this->lastComputeTime);
	
	return err;
}

/////////////////////////////////////////////////

void	CopyModelLEsToggles(CMyList *toLESetList,CMyList *fromLESetList)
{  // used when restoring past LEs
	long	i, nFrom,nTo;
	TLEList	*toLEList,*fromLEList;
	
	if(toLESetList && fromLESetList)
	{
		nFrom = fromLESetList -> GetItemCount ();
		nTo = toLESetList -> GetItemCount ();
		if(nFrom <= 0 || nFrom != nTo) return;
		for(i = 0; i < nFrom; i++)
		{
			fromLESetList -> GetListItem ((Ptr) &fromLEList, i);
			toLESetList -> GetListItem ((Ptr) &toLEList, i);
			if(toLEList && fromLEList)
			{	
				// check types
				TOLEList* toOLEList=0,*fromOLEList =0;
			
				if(toLEList->GetClassID() == TYPE_OSSMLELIST)
					toOLEList = (TOLEList*)toLEList;
				if(fromLEList->GetClassID() == TYPE_OSSMLELIST)
					fromOLEList = (TOLEList*)fromLEList;
				if(toOLEList && fromOLEList)
				{
					toOLEList->bOpen = fromOLEList->bOpen;
					toOLEList->bMassBalanceOpen = fromOLEList->bMassBalanceOpen;
					toOLEList->bReleasePositionOpen = fromOLEList->bReleasePositionOpen;
				}
			}
		}
	}
}


/////////////////////////////////////////////////

void TModel::ReleaseLEs()
{
	long i, j, n, m, numLEs;
	TLEList *thisLEList;
	LETYPE leType;
	
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !this->IsUncertain()) continue; //JLM 9/10/98
		if (!thisLEList->bActive) continue;
		numLEs = thisLEList->GetLECount();
		for (j = 0, m = numLEs ; j < m ; j++) {
			if (thisLEList->GetLEStatus(j) == OILSTAT_NOTRELEASED) 
			{	// check to see if it needs to be released
				if ((!bHindcast && (thisLEList->GetLEReleaseTime(j) <= modelTime)) || (bHindcast && (thisLEList->GetLEReleaseTime(j) >= modelTime)))
				{
					thisLEList->ReleaseLE(j);
					// JLM. 3/11/99, note: that ReleaseLE sets the age of this LE to 0.
					// Also note: if the LE is released mid-step , it is better to let the age be zero 
					// so that it weathers faster this first step.  In either case the amount of weathering is slightly less 
					// than it should be.
				}
			}
		}
	}
}


OSErr TModel::DisposeLastComputedTimeStuff (void)
{	// clear squirrel's  list
	OSErr err =0;
	long	i, n;
	TLEList	*thisLEList;
	
	if (fSquirreledLastComputeTimeLEList)
	{
		while (fSquirreledLastComputeTimeLEList -> GetItemCount() > 0)
		{
			// get the bottom-most LEList and drop & dispose of it
			fSquirreledLastComputeTimeLEList->GetListItem((Ptr)&thisLEList, 0);
			if (err = fSquirreledLastComputeTimeLEList->DeleteItem(0))
				{ TechError("TModel::DisposeLastComputedTimeStuff()", "DeleteItem()", err); return err; }
			if(thisLEList)
				{thisLEList -> Dispose ();delete (thisLEList); thisLEList = 0;}
		}
	}

	return err;
}

/////////////////////////////////////////////////
OSErr TModel::SetModelToPastTime (Seconds pastTime)
{
	OSErr err =0;
	Seconds actualTime,availablePastTime;
	Rect saveClip;
	GrafPtr savePort;
	
	// it is possible that the closest time to the past time
	// is the current ModelTime or LastComputeTime
	// so first find out what time we have stored
	availablePastTime = this->ClosestSavedModelLEsTime(pastTime);
	
	if(availablePastTime == this->GetModelTime()) goto draw; // nothing to do 
	
	// save the users current toggles
	CopyModelLEsToggles(this->fSquirreledLastComputeTimeLEList,this->LESetsList);
	
	if(availablePastTime == this->GetLastComputeTime()) 
	{
		err = ReinstateLastComputedTimeStuff(); 
		//DisplayCurrentTime(false);
		goto resetToggles;
	}
	
	// if we get here, we are going to venture into the past
	if(this->GetModelTime() == this->GetLastComputeTime())
	{	// this is the first time we are venturing into the past
		Seconds actualTime;
		SquirrelAwayLastComputedTimeStuff();
	}
	
	err = this -> LoadModelLEs (availablePastTime, &actualTime);

resetToggles:
	CopyModelLEsToggles(this->LESetsList,this->fSquirreledLastComputeTimeLEList);
	
draw:
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	saveClip = MyClipRect(MapDrawingRect());
	//DisplayCurrentTime (false);
	DrawMaps (mapWindow, MapDrawingRect(), settings.currentView, FALSE);
	(void)MyClipRect(saveClip);
	SetPortGrafPort(savePort);
	
	
	return err;
}



OSErr TModel::CheckMaxModelDuration(float durationHours,char * errStr)
{	// standard behavior will be to limit to 3 days unless in advanced mode
	//float maxDuration = (3*24) +0.00001; // 3 days
	float maxDuration = fMaxDuration +0.00001; // 3 days
	float minDuration = (1) - 0.00001; // 1 hour
	if(errStr) errStr[0] = 0;
	if( durationHours < minDuration)
	{
		if(errStr) strcpy(errStr,"The run duration must be greater than 1 hour.");
		return -1;
	}
	if( durationHours > maxDuration && this -> GetModelMode () < ADVANCEDMODE)
	{
		long days = fMaxDuration/24;
		if (errStr) 
		{
			if (days == 1) sprintf(errStr,"The run duration cannot exceed %ld day.",days);
			else sprintf(errStr,"The run duration cannot exceed %ld days.",days);
		}
		//if(errStr) strcpy(errStr,"The run duration cannot exceed 3 days.");
		return -2;
	}
	return noErr;
}


/////////////////////////////////////////////////
OSErr TModel::TemporarilyShowFutureTime (Seconds futureTime)
{
	// fake the display to show a time past the last compute time
	// without setting the model time past lastComputeTime
	// this will temporarily show the future time on the screen 
	// but not really change any of the model variables
	OSErr err =0;
	Rect saveClip;
	GrafPtr savePort;
	Seconds saveModelTime;
	static CMyList* sEmptyLESetsList = 0;
	CMyList* saveLESets;
	
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	saveClip = MyClipRect(MapDrawingRect());

	if(!sEmptyLESetsList)
	{
		if (!(sEmptyLESetsList = new CMyList(sizeof(TLEList *))))
				{ TechError("TModel::TemporarilyShowFutureTime()", "new CMyList()", 0); err = memFullErr;}
			else if (err = sEmptyLESetsList->IList())
				{ TechError("TModel::TemporarilyShowFutureTime()", "IList()", 0);}
		if(err) goto done;
	}
	
	// hide the LE's so they don't draw
	saveLESets = this->LESetsList;
	this->LESetsList = sEmptyLESetsList;
	//
	saveModelTime = this->GetModelTime();
	this->SetModelTime(futureTime);
	//DisplayCurrentTime (false);
	DrawMaps (mapWindow, MapDrawingRect(), settings.currentView, FALSE);
	this->SetModelTime(saveModelTime);
	//
	//restore the LE's
	this->LESetsList = saveLESets;

done:

	(void)MyClipRect(saveClip);
	SetPortGrafPort(savePort);

	return err;
}

/////////////////////////////////////////////////


OSErr TModel::SaveOutputSeriesFiles(Seconds oldTime,Boolean excludeRunBarFile)
{
	OSErr err = 0;
	short oldIndex,nowIndex;
	Boolean bTimeZero = 	(modelTime == fDialogVariables.startTime);
	Seconds timeSinceModelStart, previousTimeSinceModelStart;
	
	if (!bHindcast)
	{
		timeSinceModelStart = (modelTime - fDialogVariables.startTime);
		previousTimeSinceModelStart = (oldTime - fDialogVariables.startTime);
	}
	else
	{// if hindcasting track from endtime
		timeSinceModelStart = (GetEndTime() - modelTime);
		previousTimeSinceModelStart = (GetEndTime() - oldTime);
		bTimeZero = (modelTime == fDialogVariables.startTime +  fDialogVariables.duration);
	}
	
	if (this->bSaveRunBarLEs && !excludeRunBarFile)
	{
		//nowIndex = (modelTime - fDialogVariables.startTime) / LEDumpInterval;
		//oldIndex = (oldTime - fDialogVariables.startTime)   / LEDumpInterval;
		nowIndex = (timeSinceModelStart) / LEDumpInterval;
		oldIndex = (previousTimeSinceModelStart) / LEDumpInterval;
		if(nowIndex > oldIndex || bTimeZero)
		{
			err = SaveModelLEs (modelTime, nowIndex);
			if(err)
			{ 	// code goes here
				// what to do in case of an error ?
				// maybe we should keep stepping since we would just lose the ability to see past frames
				return err; 
			}
		}
	}
	/////////////////////////////////////////////////
	if (gRunSpillForecastFile.f)
	{
		{
			//nowIndex = (modelTime - fDialogVariables.startTime) / fOutputTimeStep;
			//oldIndex = (oldTime - fDialogVariables.startTime)   / fOutputTimeStep;
			nowIndex = (timeSinceModelStart) / fOutputTimeStep;
			oldIndex = (previousTimeSinceModelStart)   / fOutputTimeStep;
			if(bTimeZero) err = WriteRunSpillOutputFileHeader(&gRunSpillForecastFile,fOutputTimeStep,gRunSpillNoteStr);
			if(err) return err;
			if(nowIndex > oldIndex || bTimeZero)
			{
				err = AppendLEsToRunSpillOutputFile (&gRunSpillForecastFile);
				if(err)
				{ 	// since we are supposed to be saving files, this seems like an error worth stopping for
					return err; 
				}
			}
		}
	}
	/////////////////////////////////////////////////
	if (fWantOutput &&  this->GetModelMode () > NOVICEMODE)
	{
		//if (oldTime != fDialogVariables.startTime)
		{
			//nowIndex = (modelTime - fDialogVariables.startTime) / fOutputTimeStep;
			//oldIndex = (oldTime - fDialogVariables.startTime)   / fOutputTimeStep;
			nowIndex = (timeSinceModelStart) / fOutputTimeStep;
			oldIndex = (previousTimeSinceModelStart)   / fOutputTimeStep;
			if(nowIndex > oldIndex || bTimeZero)
			{
				err = SaveOSSMLEFile (modelTime, nowIndex);
				//err = SaveSimpleAsciiLEFile (modelTime, nowIndex);
				if(err)
				{ 	// since we are supposed to be saving files, this seems like an error worth stopping for
					return err; 
				}
			}
		}
	}
	/////////////////////////////////////////////////
	if (writeNC)
	{
		if(bTimeZero) {
			err = NetCDFStore::Create(this->ncPath, true, &this->ncID);
			if (err) return err;
			err = NetCDFStore::Define(this, false, &this->ncVarIDs, &this->ncDimIDs);
			if (err) return err;
			err = NetCDFStore::Capture(this, false, &this->ncVarIDs, &this->ncDimIDs);
			if (err) return err;
			if(this->IsUncertain()) {
				err = NetCDFStore::Create(this->ncPathConfidence, true, &this->ncID_C);
				if (err) return err;
				err = NetCDFStore::Define(this, true, &this->ncVarIDs_C, &this->ncDimIDs_C);
				if (err) return err;
				err = NetCDFStore::Capture(this, true, &this->ncVarIDs_C, &this->ncDimIDs_C);
				if (err) return err;
			}
		}
		
		
		//if (oldTime != fDialogVariables.startTime)
		else
		{
			//nowIndex = (modelTime - fDialogVariables.startTime) / fOutputTimeStep;
			//oldIndex = (oldTime - fDialogVariables.startTime)   / fOutputTimeStep;
			nowIndex = (timeSinceModelStart) / fOutputTimeStep;
			oldIndex = (previousTimeSinceModelStart)   / fOutputTimeStep;
			if(nowIndex > oldIndex/* || bTimeZero*/)
			{
				if(this->writeNC) {
					err = NetCDFStore::Capture(this, false, &this->ncVarIDs, &this->ncDimIDs);
					if (err) return err;
					if(this->IsUncertain())
					{
						err = NetCDFStore::Capture(this, true, &this->ncVarIDs, &this->ncDimIDs);
						if (err) return err;
					}
				}
				// we should return error
				/*if(err)
				{ 	// since we are supposed to be saving files, this seems like an error worth stopping for
					return err; 
				}*/
			}
		}
	}
	/////////////////////////////////////////////////
	if (bMakeMovie)
	{	// JLM, I think movies should use the same outputTimeStep as the saved LEs for TAT
		//if (oldTime != fDialogVariables.startTime)
		{
			//nowIndex = (modelTime - fDialogVariables.startTime) / fOutputTimeStep;
			//oldIndex = (oldTime - fDialogVariables.startTime)   / fOutputTimeStep;
			//nowIndex = (modelTime - fDialogVariables.startTime) / LEDumpInterval;	// use LEDumpInterval to match saved runbar LEs
			//oldIndex = (oldTime - fDialogVariables.startTime)   / LEDumpInterval;	// use LEDumpInterval to match saved runbar LEs
			nowIndex = (timeSinceModelStart) / LEDumpInterval;	// use LEDumpInterval to match saved runbar LEs
			oldIndex = (previousTimeSinceModelStart) / LEDumpInterval;	// use LEDumpInterval to match saved runbar LEs
			if(nowIndex > oldIndex || bTimeZero)
			{
				err = SaveMovieFrame (mapWindow, MapDrawingRect ());
				if(err)
				{ 	// since we are supposed to be saving pictures, this seems like an error worth stopping for
					return err;
				}
			}
		}
	}
	/////////////////////////////////////////////////
	if(gSaveMossFiles)
	{
		//if (oldTime != fDialogVariables.startTime)
		{
			//nowIndex = (modelTime - fDialogVariables.startTime) / fOutputTimeStep;
			//oldIndex = (oldTime - fDialogVariables.startTime)   / fOutputTimeStep;
			nowIndex = (timeSinceModelStart) / fOutputTimeStep;
			oldIndex = (previousTimeSinceModelStart)   / fOutputTimeStep;
			if(nowIndex > oldIndex || bTimeZero)
			{
				err = SaveMossLEFile (modelTime, nowIndex);
				//err = SaveKmlLEFileSeries (modelTime, nowIndex);
				if(err)
				{ 	// since we are supposed to be saving files, this seems like an error worth stopping for
					return err;
				}
			}
		}
	}

	/////////////////////////////////////////////////
	if(gSaveKMLFile)
	{
		//if (oldTime != fDialogVariables.startTime)
		{
			//nowIndex = (modelTime - fDialogVariables.startTime) / fOutputTimeStep;
			//oldIndex = (oldTime - fDialogVariables.startTime)   / fOutputTimeStep;
			nowIndex = (timeSinceModelStart) / fOutputTimeStep;
			oldIndex = (previousTimeSinceModelStart)   / fOutputTimeStep;
			if(nowIndex > oldIndex || bTimeZero)
			{
				//err = SaveMossLEFile (modelTime, nowIndex);
				err = SaveKmlLEFileSeries (modelTime, nowIndex);
				if(err)
				{ 	// since we are supposed to be saving files, this seems like an error worth stopping for
					return err;
				}
			}
		}
	}
	
	// option to output map + header,footer,... to file at requested times
	if  (bSaveSnapshots)
	{	// check if it is one of the set times
		char pictPath[256],timeStr[32];
		Rect r = MapDrawingRect();
		long timeNum, nameLen;

		//nowIndex = (modelTime - fDialogVariables.startTime) / 3600;	
		//oldIndex = (oldTime - fDialogVariables.startTime)   / 3600;
		nowIndex = (timeSinceModelStart) / 3600;	
		oldIndex = (previousTimeSinceModelStart)   / 3600;
		// may want to check if before or after oil has been dispersed and restart the time
		if(nowIndex > oldIndex || bTimeZero)
		{
			//timeNum = nowIndex;
			timeNum = nowIndex - fTimeOffsetForSnapshots / 3600;
			if (timeNum == 0 || timeNum == 3 || timeNum == 6 || timeNum == 12 || timeNum == 24 || timeNum == 48 || timeNum == 72 || timeNum == 96 || timeNum == 120)
			{
				// option to print instead of save? 
				//PrintMapToPrinter(); return noErr;
				strcpy(pictPath,fSnapShotFileName);	// need to strip the extension
				
			/*	nameLen=strlen(pictPath);	
				// Chop off extension
				if(nameLen >= 4) 
				{
					long i;
					for(i=1; i<=4; i++)
					{
						if(pictPath[nameLen-i] == '.')
						{	
							pictPath[nameLen-i]=0;
							break; // only take off last extension
						}
					}
				}*/
				// maybe add back in
				//sprintf(timeStr,"%d",timeNum);
				sprintf(timeStr,"%02ld",timeNum);
				strcat(pictPath,timeStr);
#ifdef MAC
#if MACB4CARBON
				strcat(pictPath,".pic");
#endif
#else
				strcat(pictPath,".bmp");
#endif

				if (timeNum == 0 && FileExists(0,0,pictPath))
				{
					char msg[256];
					strcpy(msg,"Delete existing file?");
					short buttonSelected  = MULTICHOICEALERT(1688,msg,FALSE);
					switch(buttonSelected){
						case 1:// ok
							break;// delete file and proceed
						case 3: // help
							model->bSaveSnapshots = false;
							return err;// don't save files after all, stop the run
							//break;
					}
				}
				hdelete(0, 0, pictPath);
				if (err = hcreate(0, 0, pictPath, 'ttxt', 'PICT'))
				{ TechError("PLOTDLG_CLICK()", "hcreate()", err); return -1; }	
				//err = SavePICTFile(pictPath,0,false);
				err = SavePlot(pictPath,0,r);
				if(err)
				{ 	// since we are supposed to be saving files, this seems like an error worth stopping for
					return err; 
				}
			}
		}
	}

	return err;
}

double TModel::GetVerticalMove(LERec *theLE)
{
	// now check ossm list
	// if ossm list use rise velocity
	// maybe should be mover or part of one
	double dz = 0,z = (*theLE).z, riseVelocity = (*theLE).riseVelocity;
	// if(spill->IAm(TYPE_OSSMLELIST)) {
	//if((*(TOLEList*)spill).fSetSummary.riseVelocity != 0)
	//if ((dynamic_cast<TOLEList *>(spill))->fSetSummary.riseVelocity != 0)
	if (riseVelocity > 0)
	 {
		 //PtCurMap *map = GetPtCurMap();
		 //TMap *map = Get3DMap();
		 //WorldPoint refPoint = (*theLE).p;	
		 //if (map && thisLE.statusCode == OILSTAT_INWATER && thisLE.z > 0) 
		 if ((*theLE).statusCode == OILSTAT_INWATER && z > 0) 
		 {	// check this
			 dz = -(riseVelocity/100.) * GetTimeStep();
			/* float depthAtPoint = INFINITE_DEPTH;
			 if (map) depthAtPoint = map->DepthAtPoint(refPoint);	
			 z -= (riseVelocity/100.) * GetTimeStep();
			 if (z < 0) 
			 z = 0;
			 if (z >= depthAtPoint) 
				 z = z + (riseVelocity/100.) * GetTimeStep();
			 if (z >= depthAtPoint) // shouldn't get here
				 z = depthAtPoint - (abs(riseVelocity)/100.) * GetTimeStep();*/
		 }
	 }
	//}
	return dz;
}

OSErr TModel::move_spills(vector<WorldPoint3D> **delta, vector<LERec *> **pmapping, vector< pair<bool, bool> > **dmapping, vector< pair<int, int> > **imapping) {
	
	int i, n, j, m, k, N, q, M, uncertaintyIndex = 0;
	int num_spills, num_maps;
	WorldPoint3D dp;
	TMap *t_map;
	TMover *mover;
	TLEList *list;
	LETYPE type;
	LERecP le_ptr;
	DispersionRec dispInfo;
	Seconds disperseTime;
	AdiosInfoRecH adiosBudgetTable;
	Boolean should_disperse;
	Boolean selected_disperse;
	TOSSMTimeValue *time_val_ptr;
	CMyList *time_val_list;
	TMap *map;	// in theory should be moverMap, unless universal...
	double z_move = 0;

	
	vector<LETYPE> *tmapping;
	
	num_maps = mapList->GetItemCount();
	
	try {
		*delta = new vector<WorldPoint3D>[num_maps]();
		*pmapping = new vector< LERec *>[num_maps]();
		*dmapping = new vector< pair< bool, bool> > [num_maps]();
		*imapping = new vector< pair<int, int> >[num_maps]();
		tmapping = new vector<LETYPE>[num_maps]();
	} catch(...) {
		printError("Cannot allocate required space in TModel::Step. Returning.\n");
		if(*delta)
			delete[] *delta;
		if(*pmapping)
			delete[] *pmapping;
		if(*dmapping)
			delete[] *dmapping;
		if(*imapping)
			delete[] *imapping;
		return 1;
	}

	for(i = 0, n = LESetsList->GetItemCount(); i < n; i++) {
		LESetsList->GetListItem((Ptr)&list, i);
		type = list->GetLEType();
		if(!list->IsActive()) continue;
		type = list->GetLEType();
		if(type == UNCERTAINTY_LE && !this->IsUncertain()) continue; //JLM 9/10/98
		UpdateWindage(list);
		dispInfo = ((TOLEList *)list) -> GetDispersionInfo();
		selected_disperse = dispInfo.lassoSelectedLEsToDisperse;
		//Seconds disperseTime = model->GetStartTime() + dispInfo.timeToDisperse;
		disperseTime = ((TOLEList*)list) ->fSetSummary.startRelTime + dispInfo.timeToDisperse;
		// for natural dispersion should start at spill start time, last until
		// the time of the final percent in the budget table
		// bDisperseOil will only be used for chemical (or turn into a short)
		adiosBudgetTable = ((TOLEList *)list) -> GetAdiosInfo();
		should_disperse = dispInfo.bDisperseOil && modelTime >= disperseTime && modelTime <= disperseTime + dispInfo.duration; 
		if (adiosBudgetTable) should_disperse = true;	// natural dispersion starts immediately, though should have an end
		//if (dispInfo.timeToDisperse < model->GetTimeStep() && modelTime == model->GetStartTime()+model->GetTimeStep()) timeToDisperse = true;	// make sure don't skip over dispersing if time step is large
		if (dispInfo.timeToDisperse < model->GetTimeStep() && modelTime == ((TOLEList*)list) ->fSetSummary.startRelTime+model->GetTimeStep()) should_disperse = true;	// make sure don't skip over dispersing if time step is large
		//}
		le_ptr = *list->LEHandle;
		for (j = 0, m = list->GetNumOfLEs(); j < m; j++, le_ptr++) {
			
			(*le_ptr).leCustomData = 0;
			if ((*le_ptr).statusCode == OILSTAT_NOTRELEASED) continue;
			if ((*le_ptr).statusCode == OILSTAT_OFFMAPS) continue;

			for(k = 0, N = mapList->GetItemCount(); k < N; k++) {
				mapList->GetListItem((Ptr)&t_map, k);
				if(t_map->InMap((*le_ptr).p)) {
					if ((*le_ptr).statusCode == OILSTAT_ONLAND)
						PossiblyReFloatLE(t_map, list, j, type);
					if((*le_ptr).statusCode == OILSTAT_INWATER) {
						dp.p.pLat = le_ptr->p.pLat;
						dp.p.pLong = le_ptr->p.pLong;
						dp.z = le_ptr->z;
						try	{
							(*pmapping)[k].push_back(le_ptr);
							(*dmapping)[k].push_back(pair<bool, bool>(selected_disperse, should_disperse));
							//(*imapping)[k].push_back(pair<int, int>(i, j));
							(*imapping)[k].push_back(pair<int, int>(uncertaintyIndex, j));
							(*delta)[k].push_back(dp);
							tmapping[k].push_back(type);
						} catch(...) {
							printError("Cannot allocate required space in TModel::Step. Returning.\n");
							delete[] tmapping;
							return 1;
						}
					}
					break;
				}
			}
		}
		if (type == UNCERTAINTY_LE) uncertaintyIndex++;
	}

	for (i = 0, n = uMap->moverList->GetItemCount();i <n; i++) {
		uMap->moverList->GetListItem((Ptr)&mover, i);
		if (!mover->IsActive ()) continue;
		switch(mover->GetClassID()) {			// AH 06/20/2012: maybe write a small function for this block, since we'll use it again.
				// set up the mover:
			case TYPE_WINDMOVER:
				break;
			case TYPE_RANDOMMOVER:
				// ..
				break;
			case TYPE_RANDOMMOVER3D:
				// ..
				break;
			case TYPE_CATSMOVER:
				time_val_ptr = ((TCATSMover*)mover)->timeDep;
				if(time_val_ptr) {
					if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
						dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
				}
				break;
			case TYPE_TIDECURCYCLEMOVER:
				time_val_ptr = ((TideCurCycleMover*)mover)->timeDep;
				if(time_val_ptr) {
					if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
						dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
				}
				// ..
				break;
			case TYPE_ADCPMOVER:
				time_val_list = ((ADCPMover*)mover)->timeDepList;
				if(time_val_list) {
					for(int i = 0; i < time_val_list->GetItemCount(); i++) {
						if(time_val_list->GetListItem((Ptr)&time_val_ptr, i)) break;
						if(time_val_ptr) {
							if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
								dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
						}
					}
				}
				// ..
				break;
			default:								
				break;			
		}
		for(j = 0, m = mapList->GetItemCount(); j < m; j++) {
			for(k = 0, N = (*pmapping)[j].size(); k < N; k++) {
				//dp = mover->GetMove(this->GetStartTime(), this->GetEndTime(), this->GetModelTime(), fDialogVariables.computeTimeStep, (*imapping)[j][k].first, (*imapping)[j][k].second, (*pmapping)[j][k], tmapping[j][k]);
				dp = mover->GetMove(this->GetModelTime(), fDialogVariables.computeTimeStep, (*imapping)[j][k].first, (*imapping)[j][k].second, (*pmapping)[j][k], tmapping[j][k]);
				(*delta)[j][k].p.pLat += dp.p.pLat;
				(*delta)[j][k].p.pLong += dp.p.pLong;
				(*delta)[j][k].z += dp.z;
			}
		}
	}
	
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&t_map, i);
		for (j = 0, m = t_map->moverList->GetItemCount (); j < m; j++) {
			t_map->moverList->GetListItem((Ptr)&mover, j);
			if (!mover->IsActive()) continue;
			switch(mover->GetClassID()) {			// AH 06/20/2012: maybe write a small function for this block, since we'll use it again.
					// set up the mover:
				case TYPE_WINDMOVER:
					break;
				case TYPE_RANDOMMOVER:
					// ..
					break;
				case TYPE_RANDOMMOVER3D:
					// ..
					break;
				case TYPE_CATSMOVER:
					time_val_ptr = ((TCATSMover*)mover)->timeDep;
					if(time_val_ptr) {
						if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
							dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
					}
					break;
				case TYPE_TIDECURCYCLEMOVER:
					time_val_ptr = ((TideCurCycleMover*)mover)->timeDep;
					if(time_val_ptr) {
						if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
							dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
					}
					// ..
					break;
				case TYPE_ADCPMOVER:
					time_val_list = ((ADCPMover*)mover)->timeDepList;
					if(time_val_list) {
						for(int i = 0; i < time_val_list->GetItemCount(); i++) {
							if(time_val_list->GetListItem((Ptr)&time_val_ptr, i)) break;
							if(time_val_ptr) {
								if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
									dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
							}
						}
					}
					// ..
					break;
				default:								
					break;			
			}
			for(k = 0, M = (*pmapping)[i].size(); k < M; k++) {
				//dp = mover->GetMove(this->GetStartTime(), this->GetEndTime(), this->GetModelTime(), fDialogVariables.computeTimeStep, (*imapping)[i][k].first, (*imapping)[i][k].second, (*pmapping)[i][k], tmapping[i][k]);
				dp = mover->GetMove(this->GetModelTime(), fDialogVariables.computeTimeStep, (*imapping)[i][k].first, (*imapping)[i][k].second, (*pmapping)[i][k], tmapping[i][k]);
				(*delta)[i][k].p.pLat += dp.p.pLat;
				(*delta)[i][k].p.pLong += dp.p.pLong;
				(*delta)[i][k].z += dp.z;
			}
		}
	}
	//for(j = 0, m = mapList->GetItemCount(); j < m; j++) {
	for(k = 0, N = (*pmapping)[0].size(); k < N; k++) {	// are all LEs in all mappings?
		//dp = mover->GetMove(this->GetStartTime(), this->GetEndTime(), this->GetModelTime(), fDialogVariables.computeTimeStep, (*imapping)[j][k].first, (*imapping)[j][k].second, (*pmapping)[j][k], tmapping[j][k]);
		z_move = this->GetVerticalMove((*pmapping)[0][k]);
		//(*delta)[j][k].p.pLat += dp.p.pLat;
		//(*delta)[j][k].p.pLong += dp.p.pLong;
		(*delta)[0][k].z += z_move;
	}
	//}
	
	delete[] tmapping;
	return noErr;
	
}

												 
OSErr TModel::check_spills(vector<WorldPoint3D> *delta, vector <LERec *> *pmapping, vector< pair<bool, bool> > *dmapping, vector< pair<int, int> > *imapping) {
	
	int i, j, k, q, n, m, N, M;
	int num_maps;
	 bool should_disperse;
	 bool prevent_land_jumps;
	 double distanceInKm;	
	 Boolean isDispersed;
	 Boolean use_new_move_check;
	 Boolean bBeachNearShore;
	 
	 #define BEACHINGDISTANCE 0.05 // in kilometers 

	 LERec *le_ptr;
	 TLEList *spill;
	 TMap *new_best_map, *mid_pt_best_map, *best_map;	// need to implement maps storage?
	 WorldPoint3D thisLE3D, midPt;
	 
	 use_new_move_check = true;
	 bBeachNearShore = true;	//JLM,10/20/98 always use this 
	 num_maps = mapList->GetItemCount();
	 prevent_land_jumps = fDialogVariables.preventLandJumping;

	 for(i = 0, n = num_maps; i < n; i++) {
		 m = pmapping[i].size();
		 for(j = 0; j < m; j++) {
			 le_ptr = pmapping[i][j];
			 LESetsList->GetListItem((Ptr)&spill, imapping[i][j].first);
			 if(le_ptr->leCustomData != -1) {
				 mapList->GetListItem((Ptr)&best_map, i);
				 // do move check.
				 // ..
				 if (use_new_move_check && prevent_land_jumps)
				 { 
					 //////////////////////////////////////////
					 // use the from and to points as a vector and check to see if the oil can move 
					 // along that vector without hitting the shore
					 //////////////////////////////////////////
					 
					 new_best_map = 0;
					 isDispersed = (le_ptr->dispersionStatus == HAVE_DISPERSED_NAT || le_ptr->dispersionStatus == HAVE_DISPERSED);
					 thisLE3D.p = le_ptr->p;
					 thisLE3D.z = le_ptr->z;
					 
					 //movedPoint = bestMap -> MovementCheck(thisLE.p,movedPoint);
					 delta[i][j] = best_map -> MovementCheck(thisLE3D,delta[i][j],isDispersed);
					 if (!best_map -> InMap (delta[i][j].p))
					 {	// the LE has left the map it was on
						 new_best_map = GetBestMap (delta[i][j].p);
						 if (new_best_map) {
							 // it has moved to a new map
							 // so we need to do the movement check on the new map as well
							 // code goes here, we should worry about it jumping across maps
							 // i.e. we should verify the maps rects intersect
							 best_map = new_best_map; // set bestMap for the loop
							 delta[i][j] = best_map -> MovementCheck(thisLE3D,delta[i][j],isDispersed);
						 }
						 else
						 {	// it has moved off all maps
							 le_ptr->p = delta[i][j].p;
							 le_ptr->z = delta[i][j].z;
							 le_ptr->statusCode = OILSTAT_OFFMAPS;
							 continue; 
						 }
					 }
					 ////////
					 // check for beaching, don't beach if below surface, checkmovement should handle reflection 
					 ////////
					 if (best_map -> OnLand (delta[i][j].p))
					 {
						 // we could be smarter about this since we know we have checked the
						 // the points on the land water bitmap and have beached it at the closest shore point, but
						 // for now we'll do the binary seach anyway
						 //////////////
						 // move the le onto beach and set flag to beached
						 // set the last water point to close off shore
						 // thisLE.p is the water point
						 // movedPoint is the land point
						 mid_pt_best_map = best_map;
						 distanceInKm = DistanceBetweenWorldPoints(le_ptr->p,delta[i][j].p);
						 while(distanceInKm > BEACHINGDISTANCE)
						 {
							 midPt.p.pLong = (le_ptr->p.pLong + delta[i][j].p.pLong)/2;
							 midPt.p.pLat = (le_ptr->p.pLat + delta[i][j].p.pLat)/2;
							 midPt.z = (le_ptr->z + delta[i][j].z)/2;
							 if (!mid_pt_best_map -> InMap (midPt.p))
							 {	// unusual case, it is on a different map
								 mid_pt_best_map = GetBestMap (midPt.p);
								 if(!mid_pt_best_map) 
								 {	// the midpt is off all maps
									 delta[i][j] = midPt;
									 le_ptr->statusCode = OILSTAT_OFFMAPS;
									 continue;
								 }
							 }
							 if (mid_pt_best_map -> OnLand (midPt.p)) delta[i][j] = midPt;
							 else 
							 {
								 le_ptr->p = midPt.p;// midpt is water
								 le_ptr->z = midPt.z;
							 }
							 distanceInKm = DistanceBetweenWorldPoints(le_ptr->p,delta[i][j].p);
						 }
						 ///////////////
						if ( delta[i][j].z > 0) 
						{
							// don't let subsurface LEs beach, refloat immediately
						}
						else
						{
							le_ptr->statusCode = OILSTAT_ONLAND;
							le_ptr->lastWaterPt = le_ptr->p;
							le_ptr->p = delta[i][j].p;
							le_ptr->z = delta[i][j].z;
						}
					 }
					 else
					 {
						 le_ptr->p = delta[i][j].p;	  // move the LE to new water position
						 le_ptr->z = delta[i][j].z;
					 }
				 }
				 else
				 {
					 //////////////////
					 // old code, check for transition off maps and then force beaching near shore
					 //////////////////
					 
					 // check for transition off our map and into another
					 if (!best_map -> InMap (delta[i][j].p))
					 {
						 TMap *new_best_map = GetBestMap (delta[i][j].p);
						 if (new_best_map)
							 best_map = new_best_map;
						 else
						 {
							 le_ptr->p = delta[i][j].p;
							 le_ptr->z = delta[i][j].z;
							 le_ptr->statusCode = OILSTAT_OFFMAPS;
							 continue;
						 }
					 }
					 
					 // check for beaching in the best map
					 if (best_map -> OnLand (delta[i][j].p))
					 {
						 // move the le onto beach and set flag to beached
						 le_ptr->statusCode = OILSTAT_ONLAND;
						 
						 /////{
						 /// JLM 9/18/98
						 if(bBeachNearShore)
						 {	// first beaching 
							 // code to make it beach near shore
							 //le_ptr->p is the water point
							 // delta[i][j] is the land point
							 mid_pt_best_map = best_map;
							 distanceInKm = DistanceBetweenWorldPoints(le_ptr->p,delta[i][j].p);
							 while(distanceInKm > BEACHINGDISTANCE)
							 {
								 midPt.p.pLong = (le_ptr->p.pLong + delta[i][j].p.pLong)/2;
								 midPt.p.pLat = (le_ptr->p.pLat + delta[i][j].p.pLat)/2;
								 midPt.z = (le_ptr->z + delta[i][j].z)/2;
								 if (!mid_pt_best_map -> InMap (midPt.p))
								 {	// unusual case, it is on a different map
									 mid_pt_best_map = GetBestMap (midPt.p);
									 if(!mid_pt_best_map) 
									 {	// the midpt is off all maps
										 delta[i][j] = midPt;
										 le_ptr->statusCode = OILSTAT_OFFMAPS;
										 continue;
									 }
								 }
								 if (mid_pt_best_map -> OnLand (midPt.p)) delta[i][j] = midPt;
								 else 
								 {
									 le_ptr->p = midPt.p;// midpt is water
									 le_ptr->z = midPt.z;
								 }
								 distanceInKm = DistanceBetweenWorldPoints(le_ptr->p,delta[i][j].p);
							 }
						 }
						 /////////}
						if ( delta[i][j].z > 0) 
						{
							le_ptr->statusCode = OILSTAT_INWATER;
							// don't let subsurface LEs beach, refloat immediately
						}
						else
						{
							le_ptr->lastWaterPt = le_ptr->p;
							le_ptr->p = delta[i][j].p;
							le_ptr->z = delta[i][j].z;
						}
					 }
					 else
					 {
						 le_ptr->p = delta[i][j].p;	  // move the LE to new water position
						 le_ptr->z = delta[i][j].z;
					 }
				 } // end old code
				 
				 
				 // end move check.
				 // check disperse.
				 
				 should_disperse = dmapping[i][j].first && le_ptr->beachTime >= GetModelTime() && le_ptr->beachTime < GetModelTime() + GetTimeStep();
				 should_disperse = should_disperse || dmapping[i][j].second;
				 if (should_disperse)
					// by dispersing here it's possible to miss some LEs that have already beached at the first step
					 if (le_ptr->statusCode == OILSTAT_INWATER)
						 DisperseOil (spill, imapping[i][j].second);
				 
				 // now check ossm list
				 // if ossm list use rise velocity
				 // maybe should be mover or part of one
				// this should be with the move_spills code
				/* if(spill->IAm(TYPE_OSSMLELIST)) {
					if((*(TOLEList*)spill).fSetSummary.riseVelocity != 0)
						{
						//PtCurMap *map = GetPtCurMap();
						TMap *map = Get3DMap();
						WorldPoint refPoint = le_ptr->p;	
						//if (map && thisLE.statusCode == OILSTAT_INWATER && thisLE.z > 0) 
						if (le_ptr->statusCode == OILSTAT_INWATER && le_ptr->z > 0) 
						{
							float depthAtPoint = INFINITE_DEPTH;
							if (map) depthAtPoint = map->DepthAtPoint(refPoint);	
							le_ptr->z -= (le_ptr->riseVelocity/100.) * GetTimeStep();
							if (le_ptr->z < 0) 
								le_ptr->z = 0;
							if (le_ptr->z >= depthAtPoint) 
								le_ptr->z = le_ptr->z + (le_ptr->riseVelocity/100.) * GetTimeStep();
							if (le_ptr->z >= depthAtPoint) // shouldn't get here
								le_ptr->z = depthAtPoint - (abs(le_ptr->riseVelocity)/100.) * GetTimeStep();
						}
					}
				}*/
			 } // end if(customData != -1)
			 
			 
			 if (!((TOLEList *)spill)->GetAdiosInfo())	{ // do not weather if using Adios Budget Table
				// make sure evaporation happens somewhere
				 short	wCount, wIndex;
				 Boolean	bWeathered;
				 
				 for (wIndex = 0, wCount = weatherList -> GetItemCount (); wIndex < wCount; wIndex++)
				 {
					 TWeatherer	*thisWeatherer;
					 
					 weatherList -> GetListItem ((Ptr) &thisWeatherer, wIndex);
					 if (thisWeatherer -> IsActive ())
						 thisWeatherer -> WeatherLE (le_ptr);
				 }
			 }
		 }
	 }
	 return noErr;
 }
												 
												 
/////////////////////////////////////////////////
/////////////////////////////////////////////////

OSErr TModel::Step ()
{
	long		i, j, k, c, d, n;
	Rect		saveClip;
	Seconds		oldTime = modelTime;
	GrafPtr		savePort;
	TLEList		*thisLEList;
	TMover		*thisMover;
	LERec		thisLE;
	TMap		*bestMap;
	WorldPoint3D	thisMove = {0,0,0.}, movedPoint = {0,0,0.};
	//WorldPoint3D	testPoint = {0,0,0.}, currentMovedPoint = {0,0,0.};
	LETYPE 		leType;
	OSErr		err = noErr;
	Boolean bBeachNearShore = true;//JLM,10/20/98 always use this 
	short oldIndex,nowIndex;
	Boolean useNewMovementCheckCode = true; // JLM 11/12/99 test this new method
	#define BEACHINGDISTANCE 0.05 // in kilometers 
	WorldPoint3D  midPt;
	TMap *midPtBestMap = 0;
	double distanceInKm;	
	short uncertaintyListIndex=0,listIndex=0;

	
	/*vector<WorldPoint3D> *delta;			// for storing moved points
	vector <LERec *> *pmapping;				// for associating les with maps (since the movers are)
	vector<pair<bool, bool> > *dmapping;	// for determining whether to disperse an individual le
	vector<pair<int, int> > *imapping;		// for keeping track of indices
	*/
#ifndef NO_GUI
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	saveClip = MyClipRect(MapDrawingRect());
	/*if (!gSuppressDrawing)*/ SetWatchCursor();
#endif
	
	if(this->modelTime >= this->GetEndTime()) return noErr; // JLM 2/8/99
	
	/////////////////////////////////////////////////
	// JLM 1/6/99
	if(this->modelTime < this->lastComputeTime )
	{
		// if the run time is less than the lastComputeTime,  
		// then the user has stepped back in time
		// Stepping should get the next file we have saved
		Seconds nextFileSeconds = this->NextSavedModelLEsTime(this->modelTime);
		Seconds actualTime;
		if(nextFileSeconds <= this->modelTime // there are no more saved LE times in files
			|| nextFileSeconds == this->lastComputeTime) // the next saved time is the lastCompute time
		{
			if(modelTime == fDialogVariables.startTime)
			{	// write out first time files
				err = this->SaveOutputSeriesFiles(modelTime,true);
				if(err) goto ResetPort; 
			}
			////////
			CopyModelLEsToggles(this->fSquirreledLastComputeTimeLEList,this->LESetsList);
			err = ReinstateLastComputedTimeStuff();
			CopyModelLEsToggles(this->LESetsList,this->fSquirreledLastComputeTimeLEList);
			this->NewDirtNotification(DIRTY_LIST);
			DrawMaps (mapWindow, MapDrawingRect(), settings.currentView, FALSE);
			//	
			err = this->SaveOutputSeriesFiles(oldTime,true);
			if(err) goto ResetPort; 
			///////
			goto ResetPort;
		}
		else if(nextFileSeconds > this->modelTime) 
		{	// there is a save file we can load
			if(modelTime == fDialogVariables.startTime)
			{	// write out first time files
				err = this->SaveOutputSeriesFiles(modelTime,true);
				if(err) goto ResetPort; 
			}
			////////
			this->SuppressDirt(DIRTY_EVERYTHING);
			CopyModelLEsToggles(this->fSquirreledLastComputeTimeLEList,this->LESetsList);
			err = this->LoadModelLEs (nextFileSeconds, &actualTime);
			CopyModelLEsToggles(this->LESetsList,this->fSquirreledLastComputeTimeLEList);
			this->SuppressDirt(0);
			this->NewDirtNotification(DIRTY_LIST);
			DrawMaps (mapWindow, MapDrawingRect(), settings.currentView, FALSE);
			//	
			err = this->SaveOutputSeriesFiles(oldTime,true);
			if(err) goto ResetPort; 
			////////

			goto ResetPort;
		}
		else
		{
			// there is no savedFile greater than 
			// modelTime.
			// This means we have to compute our way from the 
			// current model time
			this->SetLastComputeTime(this->modelTime);
		}
	}
	/////////////////////////////////////////////////

//if (!gSuppressDrawing)
#ifndef NO_GUI
	this->NewDirtNotification (DIRTY_LIST); //JLM, stepping can affect the list
#endif
	/////////////////////////////////////////////////
	// 1/25/99, JLM
	// We will set the seed for the rand() random number generator here
	// This allows us to set the seed in DrawLEMovement()
	// without affecting the nature of the random number generator here.
	// It has the added bonus of our getting the same answers for 
	// diffuson etc for a given step
	// NOTE: however that we only reset MyRandom() on the first step.
	// MyRandom() is currently only used for weathering.
	//
	//srand(this->modelTime); // for rand calls
	// JLM, Hmmmm... giving diffusion a plus shaped pattern, 
	//  so abandoned this idea 1/26/99
	/////////////////////////////
	
	if(this->modelTime == this->fDialogVariables.startTime)
	{	// FIRST STEP 
		//
		// on first step, reset the random seeds
		// so we always get the same answer,JLM 1/4/99
		ResetAllRandomSeeds(); 
		//
		
		/*if(this->writeNC) {
			NetCDFStore::Create(this->ncPath, true, &this->ncID);
			NetCDFStore::Define(this, false, &this->ncVarIDs, &this->ncDimIDs);
			NetCDFStore::Capture(this, false, &this->ncVarIDs, &this->ncDimIDs);
			if(this->IsUncertain()) {
				NetCDFStore::Create(this->ncPathConfidence, true, &this->ncID_C);
				NetCDFStore::Define(this, true, &this->ncVarIDs_C, &this->ncDimIDs_C);
				NetCDFStore::Capture(this, true, &this->ncVarIDs_C, &this->ncDimIDs_C);
			}
		}*/
		
		err = this->FirstStepUserInputChecks();
		if(err) goto ResetPort;
		if (err = this -> TellMoversPrepareForRun()) goto ResetPort;
	}
	/////////////////////////////////////////////////
	
	if (err = this -> TellMoversPrepareForStep()) goto ResetPort;
	
	ReleaseLEs(); // release new LE's if their time has come
	
	/////////////////////////////////////////////////
	if(modelTime == fDialogVariables.startTime)
	{	// write out first time files
		err = this->SaveOutputSeriesFiles(modelTime,false);
		if(err) goto ResetPort; 
	}
	
	/////////////////////////////////////////////////
	/// now step the LE's 
	/////////////////////////////////////////////////

	//	minus AH 06/19/2012:

	for (i = 0, n = LESetsList -> GetItemCount (); i < n; i++)
	{
		LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !this->IsUncertain()) continue; //JLM 9/10/98
		if(leType == UNCERTAINTY_LE)	// probably won't use uncertainty here anyway...
		{
			//listIndex = uncertaintyListIndex++;
			listIndex = uncertaintyListIndex;
			uncertaintyListIndex++;
		}
		else listIndex = 0;	// note this is not used for forecast LEs - maybe put in a flag to identify that

		SetChemicalHalfLife(((TOLEList *)thisLEList)->fSetSummary.halfLife);	// each spill can have a half life
		UpdateWindage(thisLEList);
		DispersionRec dispInfo = ((TOLEList *)thisLEList) -> GetDispersionInfo();
		//Seconds disperseTime = model->GetStartTime() + dispInfo.timeToDisperse;
		Seconds disperseTime = ((TOLEList*)thisLEList) ->fSetSummary.startRelTime + dispInfo.timeToDisperse;
		// for natural dispersion should start at spill start time, last until
		// the time of the final percent in the budget table
		// bDisperseOil will only be used for chemical (or turn into a short)
		AdiosInfoRecH adiosBudgetTable = ((TOLEList *)thisLEList) -> GetAdiosInfo();
		Boolean timeToDisperse = dispInfo.bDisperseOil && modelTime >= disperseTime 
							&& modelTime <= disperseTime + dispInfo.duration; 
		if (adiosBudgetTable) timeToDisperse = true;	// natural dispersion starts immediately, though should have an end
		//if (dispInfo.timeToDisperse < model->GetTimeStep() && modelTime == model->GetStartTime()+model->GetTimeStep()) timeToDisperse = true;	// make sure don't skip over dispersing if time step is large
		if (dispInfo.timeToDisperse < model->GetTimeStep() && modelTime == ((TOLEList*)thisLEList) ->fSetSummary.startRelTime+model->GetTimeStep()) timeToDisperse = true;	// make sure don't skip over dispersing if time step is large
		//}
		for (j = 0, c = thisLEList -> numOfLEs; j < c; j++)
		{
			thisLEList -> GetLE (j, &thisLE);
			thisLE.leCustomData = 0;
			
			if (thisLE.statusCode == OILSTAT_NOTRELEASED) continue; // we make no changes to the LE
			if (thisLE.statusCode == OILSTAT_OFFMAPS) continue;// we make no changes to the LE

			// LE has been released and is on at least one map
			bestMap = GetBestMap (thisLE.p);
			if (!bestMap)
			{  // it is already off the maps at the beginning of the step
				// (actually we probably found this out at the end of the previous step, but I guess we check again to be sure)
				thisLE.statusCode = OILSTAT_OFFMAPS;
				goto RecordStatus;	 // so that we save the new status 
			}

			// refloat all LE's at this time, LE's on land will be re-beached below
			if (thisLE.statusCode == OILSTAT_ONLAND)
			{
				PossiblyReFloatLE (bestMap, thisLEList, j,leType);
				thisLEList -> GetLE (j, &thisLE); // JLM 9/16/98 , we have to refresh the local variable thisLE
			}

			// move only the floating LEs
			if (thisLE.statusCode == OILSTAT_INWATER)
			{	// moved to the end of the step
				TOSSMTimeValue *time_val_ptr=0;
				//if (dispInfo.lassoSelectedLEsToDisperse && thisLE.beachTime >= GetModelTime() && thisLE.beachTime < GetModelTime() + GetTimeStep()) timeToDisperse = true;
				/*if (timeToDisperse)
				{
					DisperseOil (thisLEList, j);
					thisLEList -> GetLE (j, &thisLE); // JLM 9/16/98 , we have to refresh the local variable thisLE
					//thisLE.leCustomData = 1;	// for now use this so TRandom3D knows when to add z component
				}*/
	
				movedPoint.p = thisLE.p; //JLM 10/8/98, moved line here because we need to do this assignment after we re-float it
				movedPoint.z = thisLE.z; 
				//currentMovedPoint.p = thisLE.p; 
				//currentMovedPoint.z = thisLE.z; 
				
				/////////////////////
				/// find where the movers move the LE
				/////////////////////
				
				// loop through each mover in the universal map
				for (k = 0, d = uMap -> moverList -> GetItemCount (); k < d; k++)
				{
					uMap -> moverList -> GetListItem ((Ptr) &thisMover, k);
					if (!thisMover -> IsActive ()) continue; // to next mover
				
					switch(thisMover->GetClassID()) {			
						case TYPE_CATSMOVER:
							time_val_ptr = ((TCATSMover*)thisMover)->timeDep;
							if(time_val_ptr) {
								if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
									dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
							}
							break;
						case TYPE_TIDECURCYCLEMOVER:
							time_val_ptr = ((TideCurCycleMover*)thisMover)->timeDep;
							if(time_val_ptr) {
								if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
									dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
							}
							// ..
							break;
						case TYPE_CURRENTCYCLEMOVER:
							time_val_ptr = ((CurrentCycleMover*)thisMover)->timeDep;
							if(time_val_ptr) {
								if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
									dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
							}
							// ..
							break;
						default:
							break;
					}
					//thisMove = thisMover -> GetMove (model->GetStartTime(),fDialogVariables.computeTimeStep,i,j,&thisLE,leType);
					thisMove = thisMover -> GetMove (model->GetStartTime(),fDialogVariables.computeTimeStep,listIndex,j,&thisLE,leType);
					/*if(thisMover -> IAm(TYPE_CURRENTMOVER)) // maybe also for larvae (special LE type?)
					{	// check if current beaches LE, and if so don't add into overall move
						testPoint.p.pLat = currentMovedPoint.p.pLat + thisMove.p.pLat;
						testPoint.p.pLong = currentMovedPoint.p.pLong + thisMove.p.pLong;
						testPoint.z  = currentMovedPoint.z + thisMove.z;
						if (!CurrentBeachesLE(currentMovedPoint, &testPoint, bestMap))
						{
							currentMovedPoint.p.pLat += thisMove.p.pLat;
							currentMovedPoint.p.pLong += thisMove.p.pLong;
							currentMovedPoint.z += thisMove.z;
						}
						else
						{
							// code goes here, have current turn movement along the beach, for now don't move at all...
							// testPoint is now a beached point
							//currentMovedPoint = TurnLEAlongShoreLine(currentMovedPoint, testPoint, bestMap);
						}
					}
					else
					{*/	// non-current movers, add contribution to movedPoint
						//movedPoint.pLat  += thisMove.pLat;
						//movedPoint.pLong += thisMove.pLong;
						if (thisLE.leCustomData==-1)
						{	// for now this is the dry value flag
							goto WeatherLE;
						}
						else
						{
						movedPoint.p.pLat  += thisMove.p.pLat;
						movedPoint.p.pLong += thisMove.p.pLong;
						movedPoint.z += thisMove.z;
						}
					//}
				}

				// loop through each mover in the best map
				for (k = 0, d = bestMap -> moverList -> GetItemCount (); k < d; k++)
				{
					bestMap -> moverList -> GetListItem ((Ptr) &thisMover, k);
					if (!thisMover -> IsActive ()) continue; // to next mover
				
					switch(thisMover->GetClassID()) {			
						case TYPE_CATSMOVER:
							time_val_ptr = ((TCATSMover*)thisMover)->timeDep;
							if(time_val_ptr) {
								if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
									dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
							}
							break;
						case TYPE_TIDECURCYCLEMOVER:
							time_val_ptr = ((TideCurCycleMover*)thisMover)->timeDep;
							if(time_val_ptr) {
								if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
									dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
							}
							break;
							// ..
						case TYPE_CURRENTCYCLEMOVER:
							time_val_ptr = ((CurrentCycleMover*)thisMover)->timeDep;
							if(time_val_ptr) {
								if(time_val_ptr->GetClassID() == TYPE_SHIOTIMEVALUES)
									dynamic_cast<TShioTimeValue*>(time_val_ptr)->daylight_savings_off = settings.daylightSavingsTimeFlag;
							}
							// ..
							break;
						default:
							break;
					}
					//thisMove = thisMover -> GetMove (fDialogVariables.computeTimeStep,i,j,&thisLE,leType);
					thisMove = thisMover -> GetMove (GetModelTime(),fDialogVariables.computeTimeStep,listIndex,j,&thisLE,leType);
					/*if(thisMover -> IAm(TYPE_CURRENTMOVER)) 
					{	// check if current beaches LE, and if so don't add into overall move
						testPoint.p.pLat  = currentMovedPoint.p.pLat + thisMove.p.pLat;
						testPoint.p.pLong  = currentMovedPoint.p.pLong + thisMove.p.pLong;
						testPoint.z  = currentMovedPoint.z + thisMove.z;
						if (!CurrentBeachesLE(currentMovedPoint, &testPoint, bestMap))
						{
							currentMovedPoint.p.pLat += thisMove.p.pLat;
							currentMovedPoint.p.pLong += thisMove.p.pLong;
							currentMovedPoint.z += thisMove.z;
						}
						else
						{
							// code goes here, have current turn movement along the beach, for now don't move at all...
							//currentMovedPoint = TurnLEAlongShoreLine(currentMovedPoint, testPoint, bestMap);
						}
					}
					else
					{*/	// non-current movers, add contribution to movedPoint
						//movedPoint.pLat  += thisMove.pLat;
						//movedPoint.pLong += thisMove.pLong;
						if (thisLE.leCustomData==-1)
						{	// for now this is the dry value flag
							goto WeatherLE;
						}
						else
						{
						movedPoint.p.pLat  += thisMove.p.pLat;
						movedPoint.p.pLong += thisMove.p.pLong;
						movedPoint.z += thisMove.z;
						}
					//}
				}
				// Add contributions from all movers together
				//movedPoint.p.pLat  += currentMovedPoint.p.pLat - thisLE.p.pLat;	// original point counted twice
				//movedPoint.p.pLong += currentMovedPoint.p.pLong - thisLE.p.pLong; // original point counted twice
				//movedPoint.z += currentMovedPoint.z - thisLE.z; // original point counted twice
				//bestMap = GetBestMap (thisLE.p); // bestMap may have been changed
				if (thisLEList->IAm(TYPE_OSSMLELIST))
				{	// move this up before the map check
					if((*(TOLEList*)thisLEList).fSetSummary.riseVelocity != 0)
					{	// turn this into a mover
						double dz = GetVerticalMove(&thisLE);
						movedPoint.z += dz;
						//thisLE.z += dz;
					}
				}
				//////////////////
				// check for transition off maps, beaching, etc
				//////////////////
				// check if LE is in a dry triangle, in which case don't move
				if (useNewMovementCheckCode && fDialogVariables.preventLandJumping)
				{ 
					//////////////////////////////////////////
					// use the from and to points as a vector and check to see if the oil can move 
					// along that vector without hitting the shore
					//////////////////////////////////////////
					
					TMap *newBestMap = 0;
					Boolean isDispersed = (thisLE.dispersionStatus == HAVE_DISPERSED_NAT || thisLE.dispersionStatus == HAVE_DISPERSED);
					WorldPoint3D thisLE3D;
					thisLE3D.p = thisLE.p;
					thisLE3D.z = thisLE.z;
					
					//movedPoint = bestMap -> MovementCheck(thisLE.p,movedPoint);
					movedPoint = bestMap -> MovementCheck(thisLE3D,movedPoint,isDispersed);
					if (!bestMap -> InMap (movedPoint.p))
					{	// the LE has left the map it was on
						newBestMap = GetBestMap (movedPoint.p);
						if (newBestMap) {
							// it has moved to a new map
							// so we need to do the movement check on the new map as well
							// code goes here, we should worry about it jumping across maps
							// i.e. we should verify the maps rects intersect
							bestMap = newBestMap; // set bestMap for the loop
							movedPoint = bestMap -> MovementCheck(thisLE3D,movedPoint,isDispersed);
						}
						else
						{	// it has moved off all maps
							thisLE.p = movedPoint.p;
							thisLE.z = movedPoint.z;
							thisLE.statusCode = OILSTAT_OFFMAPS;
							goto RecordStatus; 
						}
					}
					////////
					// check for beaching, don't beach if below surface, checkmovement should handle reflection 
					////////
					if (bestMap -> OnLand (movedPoint.p))
					{
						// we could be smarter about this since we know we have checked the
						// the points on the land water bitmap and have beached it at the closest shore point, but
						// for now we'll do the binary seach anyway
						//////////////
						// move the le onto beach and set flag to beached
						// set the last water point to close off shore
						// thisLE.p is the water point
						// movedPoint is the land point
						midPtBestMap = bestMap;
						distanceInKm = DistanceBetweenWorldPoints(thisLE.p,movedPoint.p);
						while(distanceInKm > BEACHINGDISTANCE)
						{
							midPt.p.pLong = (thisLE.p.pLong + movedPoint.p.pLong)/2;
							midPt.p.pLat = (thisLE.p.pLat + movedPoint.p.pLat)/2;
							midPt.z = (thisLE.z + movedPoint.z)/2;
							if (!midPtBestMap -> InMap (midPt.p))
							{	// unusual case, it is on a different map
								midPtBestMap = GetBestMap (midPt.p);
								if(!midPtBestMap) 
								{	// the midpt is off all maps
									movedPoint = midPt;
									thisLE.statusCode = OILSTAT_OFFMAPS;
									goto RecordStatus;
								}
							}
							if (midPtBestMap -> OnLand (midPt.p)) movedPoint = midPt;
							else 
							{
								thisLE.p = midPt.p;// midpt is water
								thisLE.z = midPt.z;
							}
							distanceInKm = DistanceBetweenWorldPoints(thisLE.p,movedPoint.p);
						}
						///////////////
						thisLE.statusCode = OILSTAT_ONLAND;
						thisLE.lastWaterPt = thisLE.p;
						thisLE.p = movedPoint.p;
						thisLE.z = movedPoint.z;
					}
					else
					{
						thisLE.p = movedPoint.p;	  // move the LE to new water position
						thisLE.z = movedPoint.z;
					}
				}
				else
				{
					//////////////////
					// old code, check for transition off maps and then force beaching near shore
					//////////////////
				
					// check for transition off our map and into another
					if (!bestMap -> InMap (movedPoint.p))
					{
						TMap *newBestMap = GetBestMap (movedPoint.p);
						if (newBestMap)
							bestMap = newBestMap;
						else
						{
							thisLE.p = movedPoint.p;
							thisLE.z = movedPoint.z;
							thisLE.statusCode = OILSTAT_OFFMAPS;
							goto RecordStatus; 
						}
					}
	
					// check for beaching in the best map
					if (bestMap -> OnLand (movedPoint.p))
					{
						// move the le onto beach and set flag to beached
						thisLE.statusCode = OILSTAT_ONLAND;
	
						/////{
						/// JLM 9/18/98
						if(bBeachNearShore)
						{	// first beaching 
							// code to make it beach near shore
							//thisLE.p is the water point
							// movedPoint is the land point
							midPtBestMap = bestMap;
							distanceInKm = DistanceBetweenWorldPoints(thisLE.p,movedPoint.p);
							while(distanceInKm > BEACHINGDISTANCE)
							{
								midPt.p.pLong = (thisLE.p.pLong + movedPoint.p.pLong)/2;
								midPt.p.pLat = (thisLE.p.pLat + movedPoint.p.pLat)/2;
								midPt.z = (thisLE.z + movedPoint.z)/2;
								if (!midPtBestMap -> InMap (midPt.p))
								{	// unusual case, it is on a different map
									midPtBestMap = GetBestMap (midPt.p);
									if(!midPtBestMap) 
									{	// the midpt is off all maps
										movedPoint = midPt;
										thisLE.statusCode = OILSTAT_OFFMAPS;
										goto RecordStatus;
									}
								}
								if (midPtBestMap -> OnLand (midPt.p)) movedPoint = midPt;
								else 
								{
									thisLE.p = midPt.p;// midpt is water
									thisLE.z = midPt.z;
								}
								distanceInKm = DistanceBetweenWorldPoints(thisLE.p,movedPoint.p);
							}
						}
						/////////}
	
						thisLE.lastWaterPt = thisLE.p;
						thisLE.p = movedPoint.p;
						thisLE.z = movedPoint.z;
					}
					else
					{
						thisLE.p = movedPoint.p;	  // move the LE to new water position
						thisLE.z = movedPoint.z;
					}
				} // end old code
				if (dispInfo.lassoSelectedLEsToDisperse && thisLE.beachTime >= GetModelTime() && thisLE.beachTime < GetModelTime() + GetTimeStep()) timeToDisperse = true;
				if (timeToDisperse)
				{	// by dispersing here it's possible to miss some LEs that have already beached at the first step
					if (thisLE.statusCode == OILSTAT_INWATER)
					{
					thisLEList -> SetLE (j, &thisLE);
					DisperseOil (thisLEList, j);
					thisLEList -> GetLE (j, &thisLE); // JLM 9/16/98 , we have to refresh the local variable thisLE
					//thisLE.leCustomData = 1;	// for now use this so TRandom3D knows when to add z component
					}
				}
			}
			
			/////////////////////////////////////////////////
			/////////////////////////////////////////////////

			/*if (thisLEList->IAm(TYPE_OSSMLELIST))
			{	// move this up before the map check
				if((*(TOLEList*)thisLEList).fSetSummary.riseVelocity != 0)
				{	// turn this into a mover
					double dz = GetVerticalMove(&thisLE);
					thisLE.z += dz;*/
					/*PtCurMap *map = GetPtCurMap();
					WorldPoint refPoint = (thisLE).p;	
					//if (map && thisLE.statusCode == OILSTAT_INWATER && thisLE.z > 0) 
					if (thisLE.statusCode == OILSTAT_INWATER && thisLE.z > 0) 
					{
						float depthAtPoint = INFINITE_DEPTH;
						if (map) depthAtPoint = map->DepthAtPoint(refPoint);	
						thisLE.z -= (thisLE.riseVelocity/100.) * GetTimeStep();
						if (thisLE.z < 0) 
							thisLE.z = 0;
						if (thisLE.z >= depthAtPoint) 
							thisLE.z = thisLE.z + (thisLE.riseVelocity/100.) * GetTimeStep();
						if (thisLE.z >= depthAtPoint) // shouldn't get here
							thisLE.z = depthAtPoint - (abs(thisLE.riseVelocity)/100.) * GetTimeStep();
					}*/
				//}
			//}
		WeatherLE: //////////////////
			// now perform weathering for this LE
			// do not weather if using Adios Budget Table
			if (!adiosBudgetTable)	// make sure evaporation happens somewhere
			{
				short	wCount, wIndex;
				Boolean	bWeathered;
			
				for (wIndex = 0, wCount = weatherList -> GetItemCount (); wIndex < wCount; wIndex++)
				{
					TWeatherer	*thisWeatherer;

					weatherList -> GetListItem ((Ptr) &thisWeatherer, wIndex);
					if (thisWeatherer -> IsActive ())
						thisWeatherer -> WeatherLE (&thisLE);
				}
			}
			
			
		RecordStatus: //////////////////
			// put the modified le back into list
			thisLEList -> SetLE (j, &thisLE);
		}
	}
	
	 // end minus AH 06/19/2012.
	   // beware: I had to remove some comment delimiters in order to comment this last block.
	   // it's not a good idea to try to uncomment the block and use it as before, without double-checking
	   // that the comment sub blocks have been terminated properly.

	/*if(move_spills(&delta, &pmapping, &dmapping, &imapping));	// handle error?
	if(check_spills(delta, pmapping, dmapping, imapping));		// handle error?
	delete[] delta;
	delete[] pmapping;
	delete[] dmapping;
	delete[] imapping;*/
	
	{	// totals LEs from all spills
		PtCurMap *map = GetPtCurMap();
		//if (map && (dispInfo.bDisperseOil || adiosBudgetTable))
		if (map)	// allow tracking budget table if not dispersed
		{
			//if (map->bTrackAllLayers) map->TrackOutputDataInAllLayers();
			map->TrackOutputData();	
		}
	}

	this -> TellMoversStepIsDone();
		
	
	oldTime = modelTime;
	modelTime += fDialogVariables.computeTimeStep;
	SetLastComputeTime(modelTime); // JLM 1/7/99

	
	this->currentStep++;
	/*if(this->writeNC) {
		NetCDFStore::Capture(this, false, &this->ncVarIDs, &this->ncDimIDs);
		if(this->IsUncertain())
			NetCDFStore::Capture(this, true, &this->ncVarIDs, &this->ncDimIDs);
	}*/
	
	
	if(!gSuppressDrawing)
		DrawMaps (mapWindow, MapDrawingRect(), settings.currentView, FALSE);
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	// OUTPUT FILES -- if we have crossed an output time step boundary
	/////////////////////////////////////////////////
	err = this->SaveOutputSeriesFiles(oldTime,false);
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////

	
ResetPort:
	if (err==-3) return err;	// on IBM user exited midrun by clicking on the close window box
	//if(!gSuppressDrawing)
#ifndef NO_GUI	
	DisplayCurrentTime(true);
	MyClipRect(saveClip);
	SetPortGrafPort(savePort);
#endif
	return err;
}


OSErr TModel::StepBackwards ()
{
	long		i, j, k, c, d, n;
	Rect		saveClip;
	Seconds		oldTime = modelTime;
	GrafPtr		savePort;
	TLEList		*thisLEList;
	TMover		*thisMover;
	LERec		thisLE;
	TMap		*bestMap;
	WorldPoint3D	thisMove = {0,0,0.}, movedPoint = {0,0,0.};
	//WorldPoint3D	testPoint = {0,0,0.}, currentMovedPoint = {0,0,0.};
	LETYPE 		leType;
	OSErr		err = noErr;
	Boolean bBeachNearShore = true;//JLM,10/20/98 always use this 
	short oldIndex,nowIndex;
	Boolean useNewMovementCheckCode = true; // JLM 11/12/99 test this new method
	#define BEACHINGDISTANCE 0.05 // in kilometers 
	WorldPoint3D  midPt;
	TMap *midPtBestMap = 0;
	double distanceInKm;
	short uncertaintyListIndex=0,listIndex=0;
	
#ifndef NO_GUI
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	saveClip = MyClipRect(MapDrawingRect());
	/*if (!gSuppressDrawing)*/ SetWatchCursor();
#endif
	
	//if(this->modelTime >= this->GetEndTime()) return noErr; // JLM 2/8/99
	if(this->modelTime <= this->GetStartTime()) return noErr; // JLM 2/8/99
	
	/////////////////////////////////////////////////
	// JLM 1/6/99
	/*if(this->modelTime < this->lastComputeTime )
	{	// careful about stepping backwards...
		// if the run time is less than the lastComputeTime,  
		// then the user has stepped back in time
		// Stepping should get the next file we have saved
		Seconds nextFileSeconds = this->NextSavedModelLEsTime(this->modelTime);
		Seconds actualTime;
		if(nextFileSeconds <= this->modelTime // there are no more saved LE times in files
			|| nextFileSeconds == this->lastComputeTime) // the next saved time is the lastCompute time
		{
			if(modelTime == fDialogVariables.startTime)
			{	// write out first time files
				err = this->SaveOutputSeriesFiles(modelTime,true);
				if(err) goto ResetPort; 
			}
			////////
			CopyModelLEsToggles(this->fSquirreledLastComputeTimeLEList,this->LESetsList);
			err = ReinstateLastComputedTimeStuff();
			CopyModelLEsToggles(this->LESetsList,this->fSquirreledLastComputeTimeLEList);
			this->NewDirtNotification(DIRTY_LIST);
			DrawMaps (mapWindow, MapDrawingRect(), settings.currentView, FALSE);
			//	
			err = this->SaveOutputSeriesFiles(oldTime,true);
			if(err) goto ResetPort; 
			///////
			goto ResetPort;
		}
		else if(nextFileSeconds > this->modelTime) 
		{	// there is a save file we can load
			if(modelTime == fDialogVariables.startTime)
			{	// write out first time files
				err = this->SaveOutputSeriesFiles(modelTime,true);
				if(err) goto ResetPort; 
			}
			////////
			this->SuppressDirt(DIRTY_EVERYTHING);
			CopyModelLEsToggles(this->fSquirreledLastComputeTimeLEList,this->LESetsList);
			err = this->LoadModelLEs (nextFileSeconds, &actualTime);
			CopyModelLEsToggles(this->LESetsList,this->fSquirreledLastComputeTimeLEList);
			this->SuppressDirt(0);
			this->NewDirtNotification(DIRTY_LIST);
			DrawMaps (mapWindow, MapDrawingRect(), settings.currentView, FALSE);
			//	
			err = this->SaveOutputSeriesFiles(oldTime,true);
			if(err) goto ResetPort; 
			////////

			goto ResetPort;
		}
		else
		{
			// there is no savedFile greater than 
			// modelTime.
			// This means we have to compute our way from the 
			// current model time
			this->SetLastComputeTime(this->modelTime);
		}
	}*/
	/////////////////////////////////////////////////

//if (!gSuppressDrawing)
#ifndef NO_GUI
	this->NewDirtNotification (DIRTY_LIST); //JLM, stepping can affect the list
#endif
	/////////////////////////////////////////////////
	// 1/25/99, JLM
	// We will set the seed for the rand() random number generator here
	// This allows us to set the seed in DrawLEMovement()
	// without affecting the nature of the random number generator here.
	// It has the added bonus of our getting the same answers for 
	// diffuson etc for a given step
	// NOTE: however that we only reset MyRandom() on the first step.
	// MyRandom() is currently only used for weathering.
	//
	//srand(this->modelTime); // for rand calls
	// JLM, Hmmmm... giving diffusion a plus shaped pattern, 
	//  so abandoned this idea 1/26/99
	/////////////////////////////
	
	//if(this->modelTime == this->fDialogVariables.startTime)
	if(this->modelTime == this->fDialogVariables.startTime+this->fDialogVariables.duration)
	{	// FIRST STEP 
		//
		// on first step, reset the random seeds
		// so we always get the same answer,JLM 1/4/99
		ResetAllRandomSeeds(); 
		//
		err = this->FirstStepUserInputChecks();
		if(err) goto ResetPort;
		if (err = this -> TellMoversPrepareForRun()) goto ResetPort;
	}
	/////////////////////////////////////////////////
	
	if (err = this -> TellMoversPrepareForStep()) goto ResetPort;
	
	ReleaseLEs(); // release new LE's if their time has come
	
	/////////////////////////////////////////////////
	//if(modelTime == fDialogVariables.startTime)
	if(modelTime == fDialogVariables.startTime+this->fDialogVariables.duration)
	{	// write out first time files
		err = this->SaveOutputSeriesFiles(modelTime,false);
		if(err) goto ResetPort; 
	}
	
	/////////////////////////////////////////////////
	/// now step the LE's 
	/////////////////////////////////////////////////

		
	for (i = 0, n = LESetsList -> GetItemCount (); i < n; i++)
	{
		LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !this->IsUncertain()) continue; //JLM 9/10/98
		if(leType == UNCERTAINTY_LE)	// probably won't use uncertainty here anyway...
		{
			//listIndex = uncertaintyListIndex++;
			listIndex = uncertaintyListIndex;
			uncertaintyListIndex++;
		}
		else listIndex = 0;	// note this is not used for forecast LEs - maybe put in a flag to identify that
		if(!thisLEList->IsActive()) continue;

		UpdateWindage(thisLEList);
		//DispersionRec dispInfo = ((TOLEList *)thisLEList) -> GetDispersionInfo();
		//Seconds disperseTime = model->GetStartTime() + dispInfo.timeToDisperse;
		//Seconds disperseTime = ((TOLEList*)thisLEList) ->fSetSummary.startRelTime + dispInfo.timeToDisperse;
		// for natural dispersion should start at spill start time, last until
		// the time of the final percent in the budget table
		// bDisperseOil will only be used for chemical (or turn into a short)
		//AdiosInfoRecH adiosBudgetTable = ((TOLEList *)thisLEList) -> GetAdiosInfo();
		//Boolean timeToDisperse = dispInfo.bDisperseOil && modelTime >= disperseTime 
							//&& modelTime <= disperseTime + dispInfo.duration; 
		//if (adiosBudgetTable) timeToDisperse = true;	// natural dispersion starts immediately, though should have an end
		//if (dispInfo.timeToDisperse < model->GetTimeStep() && modelTime == model->GetStartTime()+model->GetTimeStep()) timeToDisperse = true;	// make sure don't skip over dispersing if time step is large
		//if (dispInfo.timeToDisperse < model->GetTimeStep() && modelTime == ((TOLEList*)thisLEList) ->fSetSummary.startRelTime+model->GetTimeStep()) timeToDisperse = true;	// make sure don't skip over dispersing if time step is large
		//}
		for (j = 0, c = thisLEList -> numOfLEs; j < c; j++)
		{
			thisLEList -> GetLE (j, &thisLE);
			thisLE.leCustomData = 0;
			
			if (thisLE.statusCode == OILSTAT_NOTRELEASED) continue; // we make no changes to the LE
			if (thisLE.statusCode == OILSTAT_OFFMAPS) continue;// we make no changes to the LE

			// LE has been released and is on at least one map
			bestMap = GetBestMap (thisLE.p);
			if (!bestMap)
			{  // it is already off the maps at the beginning of the step
				// (actually we probably found this out at the end of the previous step, but I guess we check again to be sure)
				thisLE.statusCode = OILSTAT_OFFMAPS;
				goto RecordStatus;	 // so that we save the new status 
			}

			// refloat all LE's at this time, LE's on land will be re-beached below
			if (thisLE.statusCode == OILSTAT_ONLAND)
			{
				PossiblyReFloatLE (bestMap, thisLEList, j,leType);
				thisLEList -> GetLE (j, &thisLE); // JLM 9/16/98 , we have to refresh the local variable thisLE
			}

			// move only the floating LEs
			if (thisLE.statusCode == OILSTAT_INWATER)
			{	// moved to the end of the step
				//if (dispInfo.lassoSelectedLEsToDisperse && thisLE.beachTime >= GetModelTime() && thisLE.beachTime < GetModelTime() + GetTimeStep()) timeToDisperse = true;
				/*if (timeToDisperse)
				{
					DisperseOil (thisLEList, j);
					thisLEList -> GetLE (j, &thisLE); // JLM 9/16/98 , we have to refresh the local variable thisLE
					//thisLE.leCustomData = 1;	// for now use this so TRandom3D knows when to add z component
				}*/
	
				movedPoint.p = thisLE.p; //JLM 10/8/98, moved line here because we need to do this assignment after we re-float it
				movedPoint.z = thisLE.z; 
				//currentMovedPoint.p = thisLE.p; 
				//currentMovedPoint.z = thisLE.z; 
				
				/////////////////////
				/// find where the movers move the LE
				/////////////////////
				
				// loop through each mover in the universal map
				for (k = 0, d = uMap -> moverList -> GetItemCount (); k < d; k++)
				{
					uMap -> moverList -> GetListItem ((Ptr) &thisMover, k);
					if (!thisMover -> IsActive ()) continue; // to next mover
				
					//thisMove = thisMover -> GetMove (this->GetStartTime(), this->GetEndTime(), this->GetModelTime(), fDialogVariables.computeTimeStep,i,j,&thisLE,leType);	// AH 07/10/2012
					//thisMove = thisMover -> GetMove (this->GetModelTime(), fDialogVariables.computeTimeStep,i,j,&thisLE,leType);	// AH 07/10/2012
					thisMove = thisMover -> GetMove (this->GetModelTime(), fDialogVariables.computeTimeStep,listIndex,j,&thisLE,leType);	
					/*if(thisMover -> IAm(TYPE_CURRENTMOVER)) // maybe also for larvae (special LE type?)
					{	// check if current beaches LE, and if so don't add into overall move
						testPoint.p.pLat = currentMovedPoint.p.pLat - thisMove.p.pLat;
						testPoint.p.pLong = currentMovedPoint.p.pLong - thisMove.p.pLong;
						testPoint.z  = currentMovedPoint.z - thisMove.z;
						if (!CurrentBeachesLE(currentMovedPoint, &testPoint, bestMap))
						{
							currentMovedPoint.p.pLat -= thisMove.p.pLat;
							currentMovedPoint.p.pLong -= thisMove.p.pLong;
							currentMovedPoint.z -= thisMove.z;
						}
						else
						{
							// code goes here, have current turn movement along the beach, for now don't move at all...
							// testPoint is now a beached point
							//currentMovedPoint = TurnLEAlongShoreLine(currentMovedPoint, testPoint, bestMap);
						}
					}
					else
					{*/	// non-current movers, add contribution to movedPoint
						//movedPoint.pLat  -= thisMove.pLat;
						//movedPoint.pLong -= thisMove.pLong;
						if (thisLE.leCustomData==-1)
						{	// for now this is the dry value flag
							goto WeatherLE;
						}
						else
						{
						movedPoint.p.pLat  -= thisMove.p.pLat;
						movedPoint.p.pLong -= thisMove.p.pLong;
						movedPoint.z -= thisMove.z;
						}
					//}
				}

				// loop through each mover in the best map
				for (k = 0, d = bestMap -> moverList -> GetItemCount (); k < d; k++)
				{
					bestMap -> moverList -> GetListItem ((Ptr) &thisMover, k);
					if (!thisMover -> IsActive ()) continue; // to next mover
				
					//thisMove = thisMover -> GetMove (this->GetStartTime(), this->GetEndTime(), this->GetModelTime(), fDialogVariables.computeTimeStep,i,j,&thisLE,leType);	// AH 07/10/2012
					//thisMove = thisMover -> GetMove (this->GetModelTime(), fDialogVariables.computeTimeStep,i,j,&thisLE,leType);	// AH 07/10/2012
					thisMove = thisMover -> GetMove (this->GetModelTime(), fDialogVariables.computeTimeStep,listIndex,j,&thisLE,leType);	
					/*if(thisMover -> IAm(TYPE_CURRENTMOVER)) 
					{	// check if current beaches LE, and if so don't add into overall move
						testPoint.p.pLat  = currentMovedPoint.p.pLat - thisMove.p.pLat;
						testPoint.p.pLong  = currentMovedPoint.p.pLong - thisMove.p.pLong;
						testPoint.z  = currentMovedPoint.z - thisMove.z;
						if (!CurrentBeachesLE(currentMovedPoint, &testPoint, bestMap))
						{
							currentMovedPoint.p.pLat -= thisMove.p.pLat;
							currentMovedPoint.p.pLong -= thisMove.p.pLong;
							currentMovedPoint.z -= thisMove.z;
						}
						else
						{
							// code goes here, have current turn movement along the beach, for now don't move at all...
							//currentMovedPoint = TurnLEAlongShoreLine(currentMovedPoint, testPoint, bestMap);
						}
					}
					else
					{*/	// non-current movers, add contribution to movedPoint
						//movedPoint.pLat  -= thisMove.pLat;
						//movedPoint.pLong -= thisMove.pLong;
						if (thisLE.leCustomData==-1)
						{	// for now this is the dry value flag
							goto WeatherLE;
						}
						else
						{
						movedPoint.p.pLat  -= thisMove.p.pLat;
						movedPoint.p.pLong -= thisMove.p.pLong;
						movedPoint.z -= thisMove.z;
						}
					//}
				}
				// Add contributions from all movers together
				//movedPoint.p.pLat  += currentMovedPoint.p.pLat - thisLE.p.pLat;	// original point counted twice
				//movedPoint.p.pLong += currentMovedPoint.p.pLong - thisLE.p.pLong; // original point counted twice
				//movedPoint.z += currentMovedPoint.z - thisLE.z; // original point counted twice
				//bestMap = GetBestMap (thisLE.p); // bestMap may have been changed
				//////////////////
				// check for transition off maps, beaching, etc
				//////////////////
				// check if LE is in a dry triangle, in which case don't move
				if (useNewMovementCheckCode && fDialogVariables.preventLandJumping)
				{ 
					//////////////////////////////////////////
					// use the from and to points as a vector and check to see if the oil can move 
					// along that vector without hitting the shore
					//////////////////////////////////////////
					
					TMap *newBestMap = 0;
					Boolean isDispersed = (thisLE.dispersionStatus == HAVE_DISPERSED_NAT || thisLE.dispersionStatus == HAVE_DISPERSED);
					WorldPoint3D thisLE3D;
					thisLE3D.p = thisLE.p;
					thisLE3D.z = thisLE.z;
					
					//movedPoint = bestMap -> MovementCheck(thisLE.p,movedPoint);
					movedPoint = bestMap -> MovementCheck(thisLE3D,movedPoint,isDispersed);
					if (!bestMap -> InMap (movedPoint.p))
					{	// the LE has left the map it was on
						newBestMap = GetBestMap (movedPoint.p);
						if (newBestMap) {
							// it has moved to a new map
							// so we need to do the movement check on the new map as well
							// code goes here, we should worry about it jumping across maps
							// i.e. we should verify the maps rects intersect
							bestMap = newBestMap; // set bestMap for the loop
							movedPoint = bestMap -> MovementCheck(thisLE3D,movedPoint,isDispersed);
						}
						else
						{	// it has moved off all maps
							thisLE.p = movedPoint.p;
							thisLE.z = movedPoint.z;
							thisLE.statusCode = OILSTAT_OFFMAPS;
							goto RecordStatus; 
						}
					}
					////////
					// check for beaching, don't beach if below surface, checkmovement should handle reflection 
					////////
					if (bestMap -> OnLand (movedPoint.p))
					{
						// we could be smarter about this since we know we have checked the
						// the points on the land water bitmap and have beached it at the closest shore point, but
						// for now we'll do the binary seach anyway
						//////////////
						// move the le onto beach and set flag to beached
						// set the last water point to close off shore
						// thisLE.p is the water point
						// movedPoint is the land point
						midPtBestMap = bestMap;
						distanceInKm = DistanceBetweenWorldPoints(thisLE.p,movedPoint.p);
						while(distanceInKm > BEACHINGDISTANCE)
						{
							midPt.p.pLong = (thisLE.p.pLong + movedPoint.p.pLong)/2;
							midPt.p.pLat = (thisLE.p.pLat + movedPoint.p.pLat)/2;
							midPt.z = (thisLE.z + movedPoint.z)/2;
							if (!midPtBestMap -> InMap (midPt.p))
							{	// unusual case, it is on a different map
								midPtBestMap = GetBestMap (midPt.p);
								if(!midPtBestMap) 
								{	// the midpt is off all maps
									movedPoint = midPt;
									thisLE.statusCode = OILSTAT_OFFMAPS;
									goto RecordStatus;
								}
							}
							if (midPtBestMap -> OnLand (midPt.p)) movedPoint = midPt;
							else 
							{
								thisLE.p = midPt.p;// midpt is water
								thisLE.z = midPt.z;
							}
							distanceInKm = DistanceBetweenWorldPoints(thisLE.p,movedPoint.p);
						}
						///////////////
						thisLE.statusCode = OILSTAT_ONLAND;
						thisLE.lastWaterPt = thisLE.p;
						thisLE.p = movedPoint.p;
						thisLE.z = movedPoint.z;
					}
					else
					{
						thisLE.p = movedPoint.p;	  // move the LE to new water position
						thisLE.z = movedPoint.z;
					}
				}
				else
				{
					//////////////////
					// old code, check for transition off maps and then force beaching near shore
					//////////////////
				
					// check for transition off our map and into another
					if (!bestMap -> InMap (movedPoint.p))
					{
						TMap *newBestMap = GetBestMap (movedPoint.p);
						if (newBestMap)
							bestMap = newBestMap;
						else
						{
							thisLE.p = movedPoint.p;
							thisLE.z = movedPoint.z;
							thisLE.statusCode = OILSTAT_OFFMAPS;
							goto RecordStatus; 
						}
					}
	
					// check for beaching in the best map
					if (bestMap -> OnLand (movedPoint.p))
					{
						// move the le onto beach and set flag to beached
						thisLE.statusCode = OILSTAT_ONLAND;
	
						/////{
						/// JLM 9/18/98
						if(bBeachNearShore)
						{	// first beaching 
							// code to make it beach near shore
							//thisLE.p is the water point
							// movedPoint is the land point
							midPtBestMap = bestMap;
							distanceInKm = DistanceBetweenWorldPoints(thisLE.p,movedPoint.p);
							while(distanceInKm > BEACHINGDISTANCE)
							{
								midPt.p.pLong = (thisLE.p.pLong + movedPoint.p.pLong)/2;
								midPt.p.pLat = (thisLE.p.pLat + movedPoint.p.pLat)/2;
								midPt.z = (thisLE.z + movedPoint.z)/2;
								if (!midPtBestMap -> InMap (midPt.p))
								{	// unusual case, it is on a different map
									midPtBestMap = GetBestMap (midPt.p);
									if(!midPtBestMap) 
									{	// the midpt is off all maps
										movedPoint = midPt;
										thisLE.statusCode = OILSTAT_OFFMAPS;
										goto RecordStatus;
									}
								}
								if (midPtBestMap -> OnLand (midPt.p)) movedPoint = midPt;
								else 
								{
									thisLE.p = midPt.p;// midpt is water
									thisLE.z = midPt.z;
								}
								distanceInKm = DistanceBetweenWorldPoints(thisLE.p,movedPoint.p);
							}
						}
						/////////}
	
						thisLE.lastWaterPt = thisLE.p;
						thisLE.p = movedPoint.p;
						thisLE.z = movedPoint.z;
					}
					else
					{
						thisLE.p = movedPoint.p;	  // move the LE to new water position
						thisLE.z = movedPoint.z;
					}
				} // end old code
				/*if (dispInfo.lassoSelectedLEsToDisperse && thisLE.beachTime >= GetModelTime() && thisLE.beachTime < GetModelTime() + GetTimeStep()) timeToDisperse = true;
				if (timeToDisperse)
				{	// by dispersing here it's possible to miss some LEs that have already beached at the first step
					if (thisLE.statusCode == OILSTAT_INWATER)
					{
					thisLEList -> SetLE (j, &thisLE);
					DisperseOil (thisLEList, j);
					thisLEList -> GetLE (j, &thisLE); // JLM 9/16/98 , we have to refresh the local variable thisLE
					//thisLE.leCustomData = 1;	// for now use this so TRandom3D knows when to add z component
					}
				}*/
			}
			
			/////////////////////////////////////////////////
			/////////////////////////////////////////////////

		WeatherLE: //////////////////
			// now perform weathering for this LE
			// do not weather if using Adios Budget Table
			//if (!adiosBudgetTable)	// make sure evaporation happens somewhere
			{
				short	wCount, wIndex;
				Boolean	bWeathered;
			
				for (wIndex = 0, wCount = weatherList -> GetItemCount (); wIndex < wCount; wIndex++)
				{
					TWeatherer	*thisWeatherer;

					weatherList -> GetListItem ((Ptr) &thisWeatherer, wIndex);
					if (thisWeatherer -> IsActive ())
						thisWeatherer -> WeatherLE (&thisLE);
				}
			}
			
			
		RecordStatus: //////////////////
			// put the modified le back into list
			thisLEList -> SetLE (j, &thisLE);
		}
	}

	this -> TellMoversStepIsDone();
		
	
	oldTime = modelTime;
	modelTime -= fDialogVariables.computeTimeStep;
	SetLastComputeTime(modelTime); // JLM 1/7/99

	if(!gSuppressDrawing)
		DrawMaps (mapWindow, MapDrawingRect(), settings.currentView, FALSE);
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	// OUTPUT FILES -- if we have crossed an output time step boundary
	/////////////////////////////////////////////////
	err = this->SaveOutputSeriesFiles(oldTime,false);
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////

	
ResetPort:
	if (err==-3) return err;	// on IBM user exited midrun by clicking on the close window box
	//if(!gSuppressDrawing)
#ifndef NO_GUI	
	DisplayCurrentTime(true);
	MyClipRect(saveClip);
	SetPortGrafPort(savePort);
#endif
	return err;
}

OSErr CreateFoldersInDirectoryPath(char *path)
{	// path contains folders only, no file name
	long len, i, numChops = 0, numFolders = 0;
	char savedFolderNames[256], delimStr[6];
	char *p, str[512];
	long dirID;
	OSErr err = 0;

	strcpy(str,path);
	// chop the delimiter if there is one
	len = strlen(str);
	if(len > 0 && str[len-1] == DIRDELIMITER)
		str[len-1] = 0;// chop this delimiter
	savedFolderNames[0] = 0;
	sprintf(delimStr,"%c",DIRDELIMITER);
	len = strlen(str);
	// count the number of directories to chop (# delimiters - 1)
	for(i = 1; i < len; i++)
	{
		if (str[i] == DIRDELIMITER)
		numChops++;
	}
	for (i=1;i<numChops;i++)	// don't try to create root folder
	{	// assume FolderExists has already been tried on the last folder and failed
		p = strrchr(str,DIRDELIMITER);
		// save the last folder name
		strcat(savedFolderNames,p);
		if(p) *(p+1) = 0; // chop off last folder name
		if (!FolderExists(0, 0, str)) 
		{
			err = dircreate(0, 0, str, &dirID);
			if(err) {	// keep going backwards
				*p = 0;	//chop off last delimiter
			}
			else 
			{
				numFolders = i;
				break;
			}
		}
		else
			break; //shouldn't happen
	}
	// try to create all the subfolders 
	for(i=0;i<numFolders;i++)
	{
		p =  strrchr(savedFolderNames,DIRDELIMITER);
		strcat(str,p+1);
		strcat(str,delimStr);
		err = dircreate(0,0,str,&dirID);
		if(err) {
			break;
		}
		else 
			if(p) *p = 0; // chop off last delimiter
	}
	if (numFolders == 0)	// not able to create any of the folders in the output path
	{
		err = -1;
	}
	return err;
} 


OSErr TModel::HandleRunMessage(TModelMessage *message)
{	// JLM, this is just like RunSpill but without creating any LEs
	OSErr err = 0;
	Boolean hadError = FALSE;
	char str[512], hindCastStr[256];
	char outputDirectory[256];
	char outputPath[256], ncOutputPath[256], moviePath[256], mossDirectory[256];
	long len;
	double runDurationInHrs;
	double timeStepInMinutes = GetTimeStep()/60,outputStepInMinutes = GetOutputStep()/60;
	Seconds startTime;
	Seconds saveOutputStep;
	Boolean saveBool,savebSaveRunBarLEs;
	long tempOutputDirID; 
	long dirID;
	char leFilePath[256] = "";
	WindageRec windageInfo;
	
	// write any errors to a "Errors.txt" file in the output directory
	
	// An example run would look like
	//MESSAGE run; TO model;  startTime DD,MM,YYYY,HH,mm; runDurationInHrs 120;timeStepInMinutes 15; outputStepInMinutes 60;outputFolder :TapOutput:;
	//
	///////////////
	
	message->GetParameterString("NETCDFPATH", ncOutputPath, 256);
	if(ncOutputPath[0]) {
		int tLen;
		char *p, classicPath[256];
		this->writeNC = true;
		if(strlen(ncOutputPath) == 0)
			strncpy(ncOutputPath, "UntitledOut", 12);

		if (ConvertIfUnixPath(ncOutputPath, classicPath)) strcpy(ncOutputPath,classicPath);
		err = ResolvePathFromCommandFile(ncOutputPath);
		if (err) ResolvePathFromApplication(ncOutputPath);
		strcpy(str,ncOutputPath);
		p =  strrchr(str,DIRDELIMITER);
		if(p) *(p+1) = 0; // chop off the file name
		// create the folder if it does not exist
		if (!FolderExists(0, 0, str)) 
		{
			long dirID;
			err = dircreate(0, 0, str, &dirID);
			if(err) 
			{	// try to create folders 
				err = CreateFoldersInDirectoryPath(str);
				if (err)	
				{
					printError("Unable to create the directory for the output file.");
				}
			}
		}
#ifdef MAC
		ConvertTraditionalPathToUnixPath(ncOutputPath, model->ncPath, 256);
		ConvertTraditionalPathToUnixPath(ncOutputPath, model->ncPathConfidence,256);
#else
		strncpy(model->ncPath, ncOutputPath, 256);
		strncpy(model->ncPathConfidence, ncOutputPath, 256);
#endif
		tLen = strlen(model->ncPath);
		if(!(tLen <= 256-11)) {
			strncpy(&model->ncPath[tLen-11], ".nc", 4);
			strncpy(&model->ncPathConfidence[tLen-11], "_uncert.nc", 11);
		} 
		else {
			if(strcmp(&model->ncPath[tLen-3], ".nc") != 0) {
				strncpy(&model->ncPath[tLen], ".nc", 4);
				strncpy(&model->ncPathConfidence[tLen], "_uncert.nc", 11);
			}
			else {
				strncpy(&model->ncPath[tLen-3], ".nc", 4);
				strncpy(&model->ncPathConfidence[tLen-3], "_uncert.nc", 11);
			}
		}
	}
	else
		this->writeNC = false;	
		
	/*message->GetParameterString("NETCDFPATH", ncOutputPath, 256);
		if(ncOutputPath[0]) {
		int tLen;
		char *p, classicPath[256];
		this->writeNC = true;
		if(strlen(ncOutputPath) == 0)
			strncpy(ncOutputPath, "UntitledOut", 12);

		if (ConvertIfUnixPath(ncOutputPath, classicPath)) strcpy(ncOutputPath,classicPath);
		err = ResolvePathFromCommandFile(ncOutputPath);
		if (err) ResolvePathFromApplication(ncOutputPath);
		strcpy(str,ncOutputPath);
		p =  strrchr(str,DIRDELIMITER);
		if(p) *(p+1) = 0; // chop off the file name
		// create the folder if it does not exist
		if (!FolderExists(0, 0, str)) 
		{
			long dirID;
			err = dircreate(0, 0, str, &dirID);
			if(err) 
			{	// try to create folders 
				err = CreateFoldersInDirectoryPath(str);
				if (err)	
				{
					printError("Unable to create the directory for the output file.");
				}
			}
		}
#ifdef MAC
		ConvertTraditionalPathToUnixPath(ncOutputPath, model->ncPath, 256);
		ConvertTraditionalPathToUnixPath(ncOutputPath, model->ncPathConfidence,256);
#else
		strncpy(model->ncPath, ncOutputPath, 256);
		strncpy(model->ncPathConfidence, ncOutputPath, 256);
#endif
		tLen = strlen(model->ncPath);
		if(!(tLen <= 256-11)) {
			strncpy(&model->ncPath[tLen-11], ".nc", 4);
			strncpy(&model->ncPathConfidence[tLen-11], "_uncert.nc", 11);
		} 
		else {
			if(strcmp(&model->ncPath[tLen-3], ".nc") != 0) {
				strncpy(&model->ncPath[tLen], ".nc", 4);
				strncpy(&model->ncPathConfidence[tLen], "_uncert.nc", 11);
			}
			else {
				strncpy(&model->ncPath[tLen-3], ".nc", 4);
				strncpy(&model->ncPathConfidence[tLen-3], "_uncert.nc", 11);
			}
		}
	}
	else
		this->writeNC = false;
*/
	gRunSpillNoteStr[0] = 0;
	message->GetParameterString("note",gRunSpillNoteStr,256);// this parameter is optional
	
	message->GetParameterString("moviePath",moviePath,256);
	if(moviePath[0]) {
		char classicPath[kMaxNameLen], * p;
		//ResolvePathFromApplication(moviePath);
		//StringSubstitute(moviePath, '/', DIRDELIMITER);
		if (ConvertIfUnixPath(moviePath, classicPath)) strcpy(moviePath,classicPath);
		err = ResolvePathFromCommandFile(moviePath);
		if (err) ResolvePathFromApplication(moviePath);
		strcpy(str,moviePath);
		p =  strrchr(str,DIRDELIMITER);
		if(p) *(p+1) = 0; // chop off the file name
		// create the folder if it does not exist
		if (!FolderExists(0, 0, str)) 
		{
			err = dircreate(0, 0, str, &dirID);
			if(err) 
			{	// try to create folders 
				err = CreateFoldersInDirectoryPath(str);
				if (err)	
				{
					printError("Unable to create the directory for the movie file.");
					hadError = TRUE;
				}
			}
		}
		if(!CanMakeMovie()) {
			printError("Unable to make a movie.  Check that Quicktime is properly installed.");
			hadError = TRUE;	
		}
		else 
		{
			if (!err) bMakeMovie=true;
		}
	}
	
	
	message->GetParameterString("outputPath",outputPath,256);
	if(outputPath[0]) {
		char classicPath[kMaxNameLen], * p;
		//ResolvePathFromApplication(outputPath);
		//StringSubstitute(outputPath, '/', DIRDELIMITER);
		if (ConvertIfUnixPath(outputPath, classicPath)) strcpy(outputPath,classicPath);
		err = ResolvePathFromCommandFile(outputPath);
		if (err) ResolvePathFromApplication(outputPath);
		strcpy(str,outputPath);
		p =  strrchr(str,DIRDELIMITER);
		if(p) *(p+1) = 0; // chop off the file name
		// create the folder if it does not exist
		/*if (!FolderExists(0, 0, str)) 
		{
			err = dircreate(0, 0, str, &dirID);
			if(err) {
				hadError = TRUE;
				printError("Unable to create the directory for the output file.");
			} 
		}*/
		if (!FolderExists(0, 0, str)) 
		{
			err = dircreate(0, 0, str, &dirID);
			if(err) 
			{	// try to create folders 
				err = CreateFoldersInDirectoryPath(str);
				if (err)	
				{
					printError("Unable to create the directory for the output file.");
					hadError = TRUE;
				}
			}
		}
	}


	///////////////////////////////////////////////////////////
	message->GetParameterString("outputFolder",outputDirectory,256);
	if(outputDirectory[0]) {
		char classicPath[kMaxNameLen];
		//ResolvePathFromApplication(outputDirectory);
		//StringSubstitute(outputDirectory, '/', DIRDELIMITER);
		if (ConvertIfUnixPath(outputDirectory, classicPath)) strcpy(outputDirectory,classicPath);
		err = ResolvePathFromCommandFile(outputDirectory);
		if (err) ResolvePathFromApplication(outputDirectory);
		len = strlen(outputDirectory);
		if(len > 0 && outputDirectory[len-1] != DIRDELIMITER)
			{ outputDirectory[len] = DIRDELIMITER; outputDirectory[len+1] = 0;} // make sure it ends with a delimiter
		
		// create the folder if it does not exist
		/*if (!FolderExists(0, 0, outputDirectory)) 
		{
			err = dircreate(0, 0, outputDirectory, &dirID);
			if(err) {
				hadError = TRUE;
				printError("Unable to create the output directory");
			} 
		}*/
		strcpy (str,outputDirectory);
		if (!FolderExists(0, 0, str)) 
		{
			err = dircreate(0, 0, str, &dirID);
			if(err) 
			{
				err = CreateFoldersInDirectoryPath(str);
				if (err)	
				{
					printError("Unable to create the directory for the output file.");
					hadError = TRUE;
				}
			}
		}
	}

	///////////////////////////////////////////////////////////
	message->GetParameterString("mossFolder",mossDirectory,256);
	if(mossDirectory[0]) {
		char classicPath[kMaxNameLen];
		if (ConvertIfUnixPath(mossDirectory, classicPath)) strcpy(mossDirectory,classicPath);
		err = ResolvePathFromCommandFile(mossDirectory);
		if (err) ResolvePathFromApplication(mossDirectory);
		len = strlen(mossDirectory);
		if(len > 0 && mossDirectory[len-1] != DIRDELIMITER)
		{ mossDirectory[len] = DIRDELIMITER; mossDirectory[len+1] = 0;} // make sure it ends with a delimiter
		
		// create the folder if it does not exist
		strcpy (str,mossDirectory);
		if (!FolderExists(0, 0, str)) 
		{
			err = dircreate(0, 0, str, &dirID);
			if(err) 
			{
				err = CreateFoldersInDirectoryPath(str);
				if (err)	
				{
					printError("Unable to create the directory for the moss output files.");
					hadError = TRUE;
				}
			}
		}
	}
	///////////////////////////////////////////////////////////
	/////////////////////////////////////////////////

	err = message->GetParameterAsSeconds("startTime",&startTime);
	if(err) {
		hadError = TRUE;
		printError("Bad startTime parameter");
	}

	
	{  // get the parameters from the message
		
		/////////////////////////////////////////////////
		// check to see if we have all the parameters 
		/////////////////////////////////////////////////

		///////////////////////////////////////////////////////////
		// check to see if they want to run backwards
		bHindcast = false;	// reset each time since this is optional parameter
		message->GetParameterString("runBackwards",hindCastStr,256);
		if(!strcmpnocase(hindCastStr,"yes") || !strcmpnocase(hindCastStr,"true")) { 	
			bHindcast = true;
		}
		else 
			bHindcast = false;
		/////////////////////////////////////////////////
		err = message->GetParameterAsDouble("runDurationInHrs",&runDurationInHrs);
		if(err || runDurationInHrs <= 0.0) {
			hadError = TRUE;
			printError("Bad runDurationInHrs parameter");
		} ///////////////////////////////////////////////////////////
		err = message->GetParameterAsDouble("timeStepInMinutes",&timeStepInMinutes);
		/*if(err || timeStepInMinutes <= 0.0) {
			hadError = TRUE;
			printError("Bad timeStepInMinutes parameter");
		} */ ///////////////////////////////////////////////////////////
		if(timeStepInMinutes <= 0.0) {
			hadError = TRUE;
			printError("Bad timeStepInMinutes parameter");
		} 
		if (err)	// allow a default - value was set in SAV file
		{
			timeStepInMinutes = GetTimeStep() / 60;
		}	///////////////////////////////////////////////////////////
	}	
	
	err = message->GetParameterAsDouble("outputStepInMinutes",&outputStepInMinutes);
	/*if(err || outputStepInMinutes <= 0.0) {
		hadError = TRUE;
		printError("Bad outputStepInMinutes parameter");
	}*/ ///////////////////////////////////////////////////////////
	if(outputStepInMinutes <= 0.0) {
		hadError = TRUE;
		printError("Bad outputStepInMinutes parameter");
	} 
	if (err) {// allow a default, output step matches time step
		outputStepInMinutes = timeStepInMinutes;
	} ///////////////////////////////////////////////////////////

	/////////////////////////////////////////////////

	/////////////////////////////////////////////////
	
	if(hadError)
	{
		printError("Run skipped due to bad data.");
		goto done; // bad data, don't do the run - should stop the entire command file??
	}
	
	/////////////////////////////////////////////////
	// setup the model 
	/////////////////////////////////////////////////


	this -> SetDuration((long)(runDurationInHrs*3600)); // convert to seconds
	if (!bHindcast)
		this -> SetStartTime(startTime);
	else
		this -> SetStartTime(startTime - model->GetDuration());
	
	this -> Reset();
	this -> SetTimeStep((long) (timeStepInMinutes*60));// convert to seconds
	
	saveOutputStep = this -> GetOutputStep();
	this -> SetOutputStep((long)(outputStepInMinutes*60)); // convert to seconds
	
	
	/////////////////////////////////////////////////
	// check that we have the expected data
	// if not make a note in the Error Log, but don't stop the run
	/////////////////////////////////////////////////

	// check that we have winds for the time period
	// check that we have a diffusion mover

	// create and open the output file
	memset(&gRunSpillForecastFile,0,sizeof(gRunSpillForecastFile));
	if(outputPath[0]) {
		hdelete(0, 0, outputPath);
		if (err = hcreate(0, 0, outputPath, '\?\?\?\?', 'BINA'))
			{ TechError("HandleRunMessage()", "hcreate()", err); goto done ; }
		
		if (err = FSOpenBuf(0, 0, outputPath, &gRunSpillForecastFile, 1000000, FALSE))
			{ TechError("HandleRunMessage()", "FSOpenBuf()", err); goto done ; }
	}

	
	if(moviePath[0] && bMakeMovie) {
		strcpy(fMoviePath,moviePath);
		this->OpenMovieFile();
	}
	
	/////////////////////////////////////////////////
	// do the run
	/////////////////////////////////////////////////
	// run the model outputting the LE's
	this -> Reset();
	saveBool = this->WantOutput(); // oh... just in case
	savebSaveRunBarLEs = this->bSaveRunBarLEs; 
	this->bSaveRunBarLEs = FALSE; // we want this to run as fast as possible, so don't write out these unnecessary files
	if(outputDirectory[0]) {
		this->SetWantOutput(true);
		strcpy(str,outputDirectory);
		strcat(outputDirectory,"LE_");
		this->SetOutputFileName (outputDirectory);
	}
	else {
		this->SetWantOutput(false);
	}
	if(mossDirectory[0]) {
		gSaveMossFiles=true;
		strcpy(gMossPath,mossDirectory);
	}
	else {
		gSaveMossFiles=false;
	}
	//err = this->Run(this->GetEndTime());
	if (model->bHindcast)
		model->Run(model->GetStartTime()); 
	else
		model->Run(model->GetEndTime());
	// reset the parameters we changed
	this->SetWantOutput(saveBool);
	gSaveMossFiles = false;
	this->bSaveRunBarLEs = savebSaveRunBarLEs; 
	
	// reset those model parameters that the user can not normally change
	this -> SetOutputStep(saveOutputStep);
	
	// close the output file
	if (gRunSpillForecastFile.f)  FSCloseBuf(&gRunSpillForecastFile);
	if (model->bMakeMovie ==true ) {this->CloseMovieFile(); model->bMakeMovie = false;}
	
done:	 // reset important globals, etc
	memset(&gRunSpillForecastFile,0,sizeof(gRunSpillForecastFile));
	gRunSpillNoteStr[0] = 0;
	/// gTapWindOffsetInSeconds = 0;	AH 06/20/2012
	return err;

}


OSErr TModel::HandleCreateSpillMessage(TModelMessage *message)
{	// JLM
	OSErr err = 0;
	Boolean hadError = FALSE;
	char str[512];
	char outputDirectory[256];
	char outputPath[256];
	long len;
	double runDurationInHrs;
	double timeStepInMinutes = GetTimeStep()/60,outputStepInMinutes = GetOutputStep()/60;
	long numLEs;
	WorldPoint startRelPos,endRelPos;
	Seconds startRelTime,endRelTime;
	Seconds saveOutputStep;
	Boolean saveBool,savebSaveRunBarLEs;
	long tempOutputDirID; 
	OilType pollutantType;
	short massUnits;
	double z,totalMass;
	long dirID;
	char leFilePath[256] = "";
	Boolean bUseLEsFromFile = false;
	WindageRec windageInfo;
	char spillName[kMaxNameLen]= "";
	
	// write any errors to a "Errors.txt" file in the output directory
	
	// An example standard TAP run would look like
	//MESSAGE runSpill;TO model;  startRelTime DD,MM,YYYY,HH,mm; numLEs 1000; startRelPos 123.7667 W 46.21667 N;
	//
	// An example would look like
	//MESSAGE createSpill;TO model;  LeFilePath :LeFileFORCST.048;
	///////////////
	

	///////////////////////////////////////////////////////////

	//	There are two methods supported at this time
	// (1) you specify the parameters for a spill
	// or (2) provide the path to a file containing the LE's at some time
	// When you provide a path to the LE's it will be assumed that the spill to be modeled
	// picks up where the file left off.
	/////////////////////////////////////////////////

	message->GetParameterString("Name",spillName,kMaxNameLen); // name is optional but necessary if the command file/wizard is going to send it messages
	

	// OK check to see if we have a file to use
	message->GetParameterString("LEFilePath",leFilePath,256);
	ResolvePath(leFilePath);
	if(leFilePath[0]) { 	
		bUseLEsFromFile = true;
		if(!FileExists(0,0,leFilePath)) {
			hadError = TRUE;
			printError("Bad LEFilePath parameter, file does not exist.");
		}
	}
	else 
		bUseLEsFromFile = false;
	/////////////////////////////////////////////////
	
	
	/////////////////////////////////////////////////
	if(!bUseLEsFromFile) 
	{
		Boolean allowLLWithoutDir = true;	
		// look for the rest of the parameters that specify the spill		
		err = message->GetParameterAsWorldPoint("startRelPos",&startRelPos,allowLLWithoutDir);
		if(err) {
			hadError = TRUE;
			printError("Bad startRelPos parameter");
		} ///////////////////////////////////////////////////////////
		err = message->GetParameterAsLong("numLEs",&numLEs);
		if(err || numLEs <= 0) {
			hadError = TRUE;
			printError("Bad numLEs parameter");
		} ///////////////////////////////////////////////////////////
	
		err = message->GetParameterAsSeconds("startRelTime",&startRelTime);
		if(err) {
			hadError = TRUE;
			printError("Bad startRelTime parameter");
		} ////////
		
		endRelTime = startRelTime;
		message->GetParameterString("endRelTime",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsSeconds("endRelTime",&endRelTime);
			if(err) {
				hadError = TRUE;
				printError("Bad endRelTime parameter");
			}
		} ////////
		
		endRelPos = startRelPos;
		message->GetParameterString("endRelPos",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsWorldPoint("endRelPos",&endRelPos,allowLLWithoutDir);
			if(err) {
				hadError = TRUE;
				printError("Bad endRelPos parameter");
			}
		} ////////
		
		pollutantType = OIL_CONSERVATIVE;
		message->GetParameterString("pollutantType",str,256);
		if(str[0]) {// this parameter is optional
			if(!strcmpnocase(str,"CONSERVATIVE") || !strcmpnocase(str,"Non-Weathering")) pollutantType = OIL_CONSERVATIVE;
			else if(!strcmpnocase(str,"BUNKER") || !strcmpnocase(str,"Fuel Oil #6")) pollutantType = OIL_6;
			else if(!strcmpnocase(str,"MEDIUMCRUDE") || !strcmpnocase(str,"Medium Crude")) pollutantType = OIL_CRUDE;
			else if(!strcmpnocase(str,"IFO") || !strcmpnocase(str,"Fuel Oil #4")) pollutantType = OIL_4;
			else if(!strcmpnocase(str,"DIESEL")) pollutantType = OIL_DIESEL;
			else if(!strcmpnocase(str,"JP4") || !strcmpnocase(str,"Kerosene / Jet Fuels")) pollutantType = OIL_JETFUELS;
			else if(!strcmpnocase(str,"GAS") || !strcmpnocase(str,"Gasoline")) pollutantType = OIL_GAS;
			else err = -1;
			if(err) {
				hadError = TRUE;
				printError("Bad pollutantType parameter");
			}
		} ////////
		
	
		massUnits = BARRELS;
		message->GetParameterString("massUnits",str,256);
		if(str[0]) {// this parameter is optional
			if(!strcmpnocase(str,"BARRELS")) massUnits = BARRELS;
			else if(!strcmpnocase(str,"GALLONS")) massUnits = GALLONS;
			else if(!strcmpnocase(str,"CUBICMETERS")) massUnits = CUBICMETERS;
			else if(!strcmpnocase(str,"KILOGRAMS")) massUnits = KILOGRAMS;
			else if(!strcmpnocase(str,"METRICTONS")) massUnits = METRICTONS;
			else if(!strcmpnocase(str,"SHORTTONS")) massUnits = SHORTTONS;
			else err = -1;
			if(err) {
				hadError = TRUE;
				printError("Bad massUnits parameter");
			}
		} ////////
		
		totalMass = numLEs; // default is the same as the number of LEs 
		message->GetParameterString("totalMass",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsDouble("totalMass",&totalMass);
			if(err || totalMass <= 0.0) {
				hadError = TRUE;
				printError("Bad totalMass parameter");
			}
		} ////////
	
	
/////////////////////////////////////////////////
	// new 1/7/03 windageInfo
		windageInfo.windageA = .01; // default 
		message->GetParameterString("windageA",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsDouble("windageA",&windageInfo.windageA);
			if(err || windageInfo.windageA < 0.0) {
				hadError = TRUE;
				printError("Bad windageA parameter");
			}
		} ////////
	
	
		windageInfo.windageB = .04; // default 
		message->GetParameterString("windageB",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsDouble("windageB",&windageInfo.windageB);
			if(err || windageInfo.windageB < 0.0 || windageInfo.windageB < windageInfo.windageA) {
				hadError = TRUE;
				printError("Bad windageB parameter");
			}
		} ////////
	
	
		windageInfo.persistence = .25; // default  
		message->GetParameterString("persistence",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsDouble("persistence",&windageInfo.persistence);
			if(err || (windageInfo.persistence < 0.0 && windageInfo.persistence != -1)) {	// persistence of -1 is infinite
				hadError = TRUE;
				printError("Bad persistence parameter");
			}
		} ////////
	
	
/////////////////////////////////////////////////
		z = 0.0;
		message->GetParameterString("z",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsDouble("z",&z);
			if(err || z < 0.0) {
				hadError = TRUE;
				printError("Bad z parameter");
			}
		} ////////
	}
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////

	
	/////////////////////////////////////////////////
	
	if(hadError)
	{
		printError("unable to create spill due to bad data.");
		goto done; // bad data, don't do the run - should stop the entire command file??
	}
	
	/////////////////////////////////////////////////
	// setup the model 
	/////////////////////////////////////////////////

	// turn drawing off
	//gSuppressDrawing = TRUE; 

	if(bUseLEsFromFile)  // JLM 2/13/01
	{	// load the file now and get the time from the file
		err = LoadOSSMFile(leFilePath,&startRelTime,spillName);
		if(err) {
			printError("LoadOSSMFile returned an error.");
			goto done;
		}
	}
	
	if (!bUseLEsFromFile)
	{	// do some error checking
		Boolean bWantEndRelPosition = (!EqualWPoints(startRelPos,endRelPos));
		if(!IsWaterPoint(startRelPos)) {
			printError("The release start position must be in the water."); 
			goto done;}
	
		if(!IsAllowableSpillPoint(startRelPos)){
			printError("This map has not been set up for spills in the area of your release start position.");
			goto done;}
	
		if(bWantEndRelPosition && !IsWaterPoint(endRelPos)) {
			printError("The release end position must be in the water."); 
		goto done;}
		
		if(bWantEndRelPosition && !IsAllowableSpillPoint(endRelPos)){				
			printError("This map has not been set up for spills in the area of your release end position."); 
			goto done;}

		if (z > 0)
		{
			TMap *map = Get3DMap();
			double botZ = INFINITE_DEPTH;
			if (map) botZ = map->DepthAtPoint(startRelPos) ;
			if (z > botZ)	
			{
				char errStr[64];
				sprintf(errStr,"The spill depth cannot be greater than total depth which is %g meters.",botZ);
				printError(errStr);
				goto done;
			}
		}
	}
	
	// set the spill
	if(!bUseLEsFromFile)  // JLM 2/13/01
	{	// original code, create the spill from scratch (not from an LE file)	
		LESetSummary  summary;
		TOLEList 	*forecastLEList=0;
		TOLEList 	*uncertaintyLEList=0;
	
		memset(&summary,0,sizeof(summary));

		strcpy(summary.spillName,spillName);
	
		summary.numOfLEs = numLEs;
		summary.pollutantType = pollutantType;
		summary.totalMass = totalMass; 
		summary.massUnits = massUnits;
		
		summary.startRelPos = startRelPos;
		summary.endRelPos = endRelPos;
		summary.bWantEndRelPosition = (!EqualWPoints(startRelPos,endRelPos));
		
		summary.startRelTime = startRelTime;
		summary.endRelTime = endRelTime;
		summary.bWantEndRelTime = (endRelTime != startRelTime);
		
		//summary.z = 0.0;
		summary.z = z;
		summary.density = GetPollutantDensity(summary.pollutantType);
		summary.ageInHrsWhenReleased = 0;
	
		// add the LE set
		forecastLEList = new TOLEList ();
		uncertaintyLEList = new TOLEList ();
		if (!forecastLEList || ! uncertaintyLEList) 
			err = memFullErr;
		
		if(!err)
		{
			forecastLEList -> SetClassName(spillName); // name is optional but necessary if the command file/wizard is going to send it messages
			forecastLEList -> SetWindageInfo(windageInfo);
			err = forecastLEList->Initialize(&summary,true);	
		}
	
		if(!err) {
			uncertaintyLEList->fLeType = UNCERTAINTY_LE;
			uncertaintyLEList->fOwnersUniqueID = forecastLEList->GetUniqueID();
			uncertaintyLEList -> SetWindageInfo(windageInfo);
			err = uncertaintyLEList->Initialize(&summary,true);
		}
	
		if(!err) 
			err = model->AddLEList(forecastLEList, 0);
		if(!err) 
			err = model->AddLEList(uncertaintyLEList, 0);
		if(err) 
		{ // probably ran out of memory, we can't do the run
			if(err == memFullErr)
				printError("Run skipped due to memFullErr setting the spill.");
			else
				printError("Run skipped due to error setting the spill.");
			goto done; // bad data, don't do the run
		}
		
	}
	

	
done:	
	return err;

}


OSErr TModel::HandleRunSpillMessage(TModelMessage *message)
{	// JLM
	OSErr err = 0;
	Boolean hadError = FALSE;
	char str[512], hindCastStr[256];
	char outputDirectory[256];
	char outputPath[256], ncOutputPath[256], moviePath[256], mossDirectory[256];
	long len;
	double runDurationInHrs;
	double timeStepInMinutes = GetTimeStep()/60,outputStepInMinutes = GetOutputStep()/60;
	long numLEs;
	WorldPoint startRelPos,endRelPos;
	Seconds startRelTime,endRelTime;
	Seconds saveOutputStep;
	Boolean saveBool,savebSaveRunBarLEs;
	long tempOutputDirID; 
	OilType pollutantType;
	short massUnits;
	double z,totalMass;
	long dirID;
	char leFilePath[256] = "";
	Boolean bUseLEsFromFile = false;
	Boolean bEverythingSetAsDesiredByHand = false;
	Boolean runBackwards = false;
	WindageRec windageInfo;
	
	// write any errors to a "Errors.txt" file in the output directory
	
	// An example standard TAP run would look like
	//MESSAGE runSpill;TO model;  startTime DD,MM,YYYY,HH,mm; runDurationInHrs 120;timeStepInMinutes 15; numLEs 1000;startRelPos 123.7667 W 46.21667 N;outputStepInMinutes 60;outputFolder :TapOutput:;
	//
	// An example extended outlook TAP run would look like
	//MESSAGE runSpill;TO model;  LeFilePath :LeFileFORCST.048; runDurationInHrs 6;timeStepInMinutes 30;outputStepInMinutes 60;outputPath :TapExtendedOutlookResults:testSingleFile3.txt;
	///////////////
	
	message->GetParameterString("NETCDFPATH", ncOutputPath, 256);
	if(ncOutputPath[0]) {
		int tLen;
		char *p, classicPath[256];
		this->writeNC = true;
		if(strlen(ncOutputPath) == 0)
			strncpy(ncOutputPath, "UntitledOut", 12);

		if (ConvertIfUnixPath(ncOutputPath, classicPath)) strcpy(ncOutputPath,classicPath);
		err = ResolvePathFromCommandFile(ncOutputPath);
		if (err) ResolvePathFromApplication(ncOutputPath);
		strcpy(str,ncOutputPath);
		p =  strrchr(str,DIRDELIMITER);
		if(p) *(p+1) = 0; // chop off the file name
		// create the folder if it does not exist
		if (!FolderExists(0, 0, str)) 
		{
			long dirID;
			err = dircreate(0, 0, str, &dirID);
			if(err) 
			{	// try to create folders 
				err = CreateFoldersInDirectoryPath(str);
				if (err)	
				{
					printError("Unable to create the directory for the output file.");
				}
			}
		}
#ifdef MAC
		ConvertTraditionalPathToUnixPath(ncOutputPath, model->ncPath, 256);
		ConvertTraditionalPathToUnixPath(ncOutputPath, model->ncPathConfidence,256);
#else
		strncpy(model->ncPath, ncOutputPath, 256);
		strncpy(model->ncPathConfidence, ncOutputPath, 256);
#endif
		tLen = strlen(model->ncPath);
		if(!(tLen <= 256-11)) {
			strncpy(&model->ncPath[tLen-11], ".nc", 4);
			strncpy(&model->ncPathConfidence[tLen-11], "_uncert.nc", 11);
		} 
		else {
			if(strcmp(&model->ncPath[tLen-3], ".nc") != 0) {
				strncpy(&model->ncPath[tLen], ".nc", 4);
				strncpy(&model->ncPathConfidence[tLen], "_uncert.nc", 11);
			}
			else {
				strncpy(&model->ncPath[tLen-3], ".nc", 4);
				strncpy(&model->ncPathConfidence[tLen-3], "_uncert.nc", 11);
			}
		}
	}
	else
		this->writeNC = false;
	
	gRunSpillNoteStr[0] = 0;
	message->GetParameterString("note",gRunSpillNoteStr,256);// this parameter is optional
	
	message->GetParameterString("moviePath",moviePath,256);
	if(moviePath[0]) {
		char classicPath[kMaxNameLen], * p;
		if (ConvertIfUnixPath(moviePath, classicPath)) strcpy(moviePath,classicPath);
		err = ResolvePathFromCommandFile(moviePath);
		if (err) ResolvePathFromApplication(moviePath);
		strcpy(str,moviePath);
		p =  strrchr(str,DIRDELIMITER);
		if(p) *(p+1) = 0; // chop off the file name
		// create the folder if it does not exist
		if (!FolderExists(0, 0, str)) 
		{
			err = dircreate(0, 0, str, &dirID);
			if(err) 
			{	// try to create folders 
				err = CreateFoldersInDirectoryPath(str);
				if (err)	
				{
					printError("Unable to create the directory for the movie file.");
					hadError = TRUE;
				}
			}
		}
		if(!CanMakeMovie()) {
			printError("Unable to make a movie.  Check that Quicktime is properly installed.");
			hadError = TRUE;	
		}
		else 
		{
			if (!err) bMakeMovie=true;
		}
	}
	
	
	message->GetParameterString("outputPath",outputPath,256);
	if(outputPath[0]) {
		char classicPath[kMaxNameLen], * p;
		//ResolvePathFromApplication(outputPath);
		//StringSubstitute(outputPath, '/', DIRDELIMITER);
		if (ConvertIfUnixPath(outputPath, classicPath)) strcpy(outputPath,classicPath);
		err = ResolvePathFromCommandFile(outputPath);
		if (err) ResolvePathFromApplication(outputPath);
		strcpy(str,outputPath);
		p =  strrchr(str,DIRDELIMITER);
		if(p) *(p+1) = 0; // chop off the file name
		// create the folder if it does not exist
		/*if (!FolderExists(0, 0, str)) 
		{
			err = dircreate(0, 0, str, &dirID);
			if(err) {
				hadError = TRUE;
				printError("Unable to create the directory for the output file.");
			} 
		}*/
		if (!FolderExists(0, 0, str)) 
		{
			err = dircreate(0, 0, str, &dirID);
			if(err) 
			{	// try to create folders 
				err = CreateFoldersInDirectoryPath(str);
				if (err)	
				{
					printError("Unable to create the directory for the output file.");
					hadError = TRUE;
				}
			}
		}
	}


	///////////////////////////////////////////////////////////
	message->GetParameterString("outputFolder",outputDirectory,256);
	if(outputDirectory[0]) {
		char classicPath[kMaxNameLen];
		//ResolvePathFromApplication(outputDirectory);
		//StringSubstitute(outputDirectory, '/', DIRDELIMITER);	
		if (ConvertIfUnixPath(outputDirectory, classicPath)) strcpy(outputDirectory,classicPath);
		err = ResolvePathFromCommandFile(outputDirectory);
		if (err) ResolvePathFromApplication(outputDirectory);
		len = strlen(outputDirectory);
		if(len > 0 && outputDirectory[len-1] != DIRDELIMITER)
			{ outputDirectory[len] = DIRDELIMITER; outputDirectory[len+1] = 0;} // make sure it ends with a delimiter
		
		// create the folder if it does not exist
		/*if (!FolderExists(0, 0, outputDirectory)) 
		{
			err = dircreate(0, 0, outputDirectory, &dirID);
			if(err) {
				hadError = TRUE;
				printError("Unable to create the output directory");
			} 
		}*/
		strcpy (str,outputDirectory);
		if (!FolderExists(0, 0, str)) 
		{
			err = dircreate(0, 0, str, &dirID);
			if(err) 
			{
				err = CreateFoldersInDirectoryPath(str);
				if (err)	
				{
					printError("Unable to create the directory for the output file.");
					hadError = TRUE;
				}
			}
		}
	}
	///////////////////////////////////////////////////////////
	message->GetParameterString("mossFolder",mossDirectory,256);
	if(mossDirectory[0]) {
		char classicPath[kMaxNameLen];
		if (ConvertIfUnixPath(mossDirectory, classicPath)) strcpy(mossDirectory,classicPath);
		err = ResolvePathFromCommandFile(mossDirectory);
		if (err) ResolvePathFromApplication(mossDirectory);
		len = strlen(mossDirectory);
		if(len > 0 && mossDirectory[len-1] != DIRDELIMITER)
		{ mossDirectory[len] = DIRDELIMITER; mossDirectory[len+1] = 0;} // make sure it ends with a delimiter
		
		// create the folder if it does not exist
		strcpy (str,mossDirectory);
		if (!FolderExists(0, 0, str)) 
		{
			err = dircreate(0, 0, str, &dirID);
			if(err) 
			{
				err = CreateFoldersInDirectoryPath(str);
				if (err)	
				{
					printError("Unable to create the directory for the moss output files.");
					hadError = TRUE;
				}
			}
		}
	}
	///////////////////////////////////////////////////////////
	
	//	There are three methods supported at this time
	// (1) you specify the parameters for a spill
	// or (2) provide the path to a file containing the LE's at some time
	// When you provide a path to the LE's it will be assumed that the spill to be modeled
	// picks up where the file left off.
	// or (3) you say that the model is already set up and to just run the output
	/////////////////////////////////////////////////
	
	
	// OK check to see if they have set everything by hand
	message->GetParameterString("EverythingSetAsDesiredByHand",str,256);
	if(!strcmpnocase(str,"yes") || !strcmpnocase(str,"true")) { 	
		bEverythingSetAsDesiredByHand = true;
	}
	else 
		bEverythingSetAsDesiredByHand = false;
	/////////////////////////////////////////////////
	
	if(!bEverythingSetAsDesiredByHand)
	{  // get the parameters from the message
		
		/////////////////////////////////////////////////
		// check to see if we have all the parameters 
		/////////////////////////////////////////////////
		// check to see if they want to run backwards
		bHindcast = false;	// reset each time since this is optional parameter
		message->GetParameterString("runBackwards",hindCastStr,256);
		if(!strcmpnocase(hindCastStr,"yes") || !strcmpnocase(hindCastStr,"true")) { 	
			bHindcast = true;
		}
		else 
			bHindcast = false;
	/////////////////////////////////////////////////
		err = message->GetParameterAsDouble("runDurationInHrs",&runDurationInHrs);
		if(err || runDurationInHrs <= 0.0) {
			hadError = TRUE;
			printError("Bad runDurationInHrs parameter");
		} ///////////////////////////////////////////////////////////
		err = message->GetParameterAsDouble("timeStepInMinutes",&timeStepInMinutes);
		/*if(err || timeStepInMinutes <= 0.0) {
			hadError = TRUE;
			printError("Bad timeStepInMinutes parameter");
		} */ ///////////////////////////////////////////////////////////
		if(timeStepInMinutes <= 0.0) {
			hadError = TRUE;
			printError("Bad timeStepInMinutes parameter");
		} 
		if (err)	// allow a default - value was set in SAV file
		{
			timeStepInMinutes = GetTimeStep() / 60;
		}	///////////////////////////////////////////////////////////
	}	
	
	err = message->GetParameterAsDouble("outputStepInMinutes",&outputStepInMinutes);
	/*if(err || outputStepInMinutes <= 0.0) {
		hadError = TRUE;
		printError("Bad outputStepInMinutes parameter");
	}*/ ///////////////////////////////////////////////////////////
	if(outputStepInMinutes <= 0.0) {
		hadError = TRUE;
		printError("Bad outputStepInMinutes parameter");
	} 
	if (err) {// allow a default, output step matches time step
		if (bEverythingSetAsDesiredByHand) outputStepInMinutes = GetTimeStep() / 60.;
		else outputStepInMinutes = timeStepInMinutes;
	} ///////////////////////////////////////////////////////////


	// OK check to see if we have a file to use
	message->GetParameterString("LEFilePath",leFilePath,256);
	ResolvePath(leFilePath);
	if(leFilePath[0]) { 	
		bUseLEsFromFile = true;
		if(!FileExists(0,0,leFilePath)) {
			hadError = TRUE;
			printError("Bad LEFilePath parameter, file does not exist.");
		}
	}
	else 
		bUseLEsFromFile = false;
	/////////////////////////////////////////////////
	
	
	/////////////////////////////////////////////////
	if(!bUseLEsFromFile && !bEverythingSetAsDesiredByHand) 
	{
		Boolean allowLLWithoutDir = true;	
		// look for the rest of the parameters that specify the spill		
		err = message->GetParameterAsWorldPoint("startRelPos",&startRelPos,allowLLWithoutDir);
		if(err) {
			hadError = TRUE;
			printError("Bad startRelPos parameter");
		} ///////////////////////////////////////////////////////////
		err = message->GetParameterAsLong("numLEs",&numLEs);
		if(err || numLEs <= 0) {
			hadError = TRUE;
			printError("Bad numLEs parameter");
		} ///////////////////////////////////////////////////////////
	
		err = message->GetParameterAsSeconds("startRelTime",&startRelTime);
		if(err) {
			hadError = TRUE;
			printError("Bad startRelTime parameter");
		} ////////
		
		endRelTime = startRelTime;
		message->GetParameterString("endRelTime",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsSeconds("endRelTime",&endRelTime);
			if(err) {
				hadError = TRUE;
				printError("Bad endRelTime parameter");
			}
		} ////////
		
		endRelPos = startRelPos;
		message->GetParameterString("endRelPos",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsWorldPoint("endRelPos",&endRelPos,allowLLWithoutDir);
			if(err) {
				hadError = TRUE;
				printError("Bad endRelPos parameter");
			}
		} ////////
		
		pollutantType = OIL_CONSERVATIVE;
		message->GetParameterString("pollutantType",str,256);
		if(str[0]) {// this parameter is optional
			if(!strcmpnocase(str,"CONSERVATIVE") || !strcmpnocase(str,"Non-Weathering")) pollutantType = OIL_CONSERVATIVE;
			else if(!strcmpnocase(str,"BUNKER") || !strcmpnocase(str,"Fuel Oil #6")) pollutantType = OIL_6;
			else if(!strcmpnocase(str,"MEDIUMCRUDE") || !strcmpnocase(str,"Medium Crude")) pollutantType = OIL_CRUDE;
			else if(!strcmpnocase(str,"IFO") || !strcmpnocase(str,"Fuel Oil #4")) pollutantType = OIL_4;
			else if(!strcmpnocase(str,"DIESEL")) pollutantType = OIL_DIESEL;
			else if(!strcmpnocase(str,"JP4") || !strcmpnocase(str,"Kerosene / Jet Fuels")) pollutantType = OIL_JETFUELS;
			else if(!strcmpnocase(str,"GAS") || !strcmpnocase(str,"Gasoline")) pollutantType = OIL_GAS;
			else err = -1;
			if(err) {
				hadError = TRUE;
				printError("Bad pollutantType parameter");
			}
		} ////////
		
	
		massUnits = BARRELS;
		message->GetParameterString("massUnits",str,256);
		if(str[0]) {// this parameter is optional
			if(!strcmpnocase(str,"BARRELS")) massUnits = BARRELS;
			else if(!strcmpnocase(str,"GALLONS")) massUnits = GALLONS;
			else if(!strcmpnocase(str,"CUBICMETERS")) massUnits = CUBICMETERS;
			else if(!strcmpnocase(str,"KILOGRAMS")) massUnits = KILOGRAMS;
			else if(!strcmpnocase(str,"METRICTONS")) massUnits = METRICTONS;
			else if(!strcmpnocase(str,"SHORTTONS")) massUnits = SHORTTONS;
			else err = -1;
			if(err) {
				hadError = TRUE;
				printError("Bad massUnits parameter");
			}
		} ////////
		
		totalMass = numLEs; // default is the same as the number of LEs 
		message->GetParameterString("totalMass",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsDouble("totalMass",&totalMass);
			if(err || totalMass <= 0.0) {
				hadError = TRUE;
				printError("Bad totalMass parameter");
			}
		} ////////
	
	
/////////////////////////////////////////////////
	// new 1/7/03 windageInfo
		windageInfo.windageA = .01; // default 
		message->GetParameterString("windageA",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsDouble("windageA",&windageInfo.windageA);
			if(err || windageInfo.windageA < 0.0) {
				hadError = TRUE;
				printError("Bad windageA parameter");
			}
		} ////////
	
	
		windageInfo.windageB = .04; // default 
		message->GetParameterString("windageB",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsDouble("windageB",&windageInfo.windageB);
			if(err || windageInfo.windageB < 0.0 || windageInfo.windageB < windageInfo.windageA) {
				hadError = TRUE;
				printError("Bad windageB parameter");
			}
		} ////////
	
	
		windageInfo.persistence = .25; // default  
		message->GetParameterString("persistence",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsDouble("persistence",&windageInfo.persistence);
			if(err || (windageInfo.persistence < 0.0 && windageInfo.persistence != -1)) {	// persistence of -1 is infinite
				hadError = TRUE;
				printError("Bad persistence parameter");
			}
		} ////////
	
	
/////////////////////////////////////////////////
		z = 0.0;
		message->GetParameterString("z",str,256);
		if(str[0]) {// this parameter is optional
			err = message->GetParameterAsDouble("z",&z);
			if(err || z < 0.0) {
				hadError = TRUE;
				printError("Bad z parameter");
			}
		} ////////
	}
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////

	/////////////////////////////////////////////////
	
	if(hadError)
	{
		printError("Run skipped due to bad data.");
		goto done; // bad data, don't do the run - should stop the entire command file??
	}
	
	/////////////////////////////////////////////////
	// setup the model 
	/////////////////////////////////////////////////

	// turn drawing off
	//gSuppressDrawing = TRUE; 

	if(bEverythingSetAsDesiredByHand) {
		// don't clear the spills, the user set them by hand	
	}
	else if(bUseLEsFromFile)  // JLM 2/13/01
	{	// load the file now and get the time from the file
		this->DisposeModelLEs();// clear any old spills
		err = LoadOSSMFile(leFilePath,&startRelTime);
		if(err) {
			printError("LoadOSSMFile returned an error.");
			goto done;
		}
	}
	else // the LEs will be created from the input parameters
	{	
		this->DisposeModelLEs();// clear any old spills
		// may want an option to set a bunch of spills 
		// we will create the spill below (because that is how the old code did it)
	}
	
	// set model start time, duration
	if(bEverythingSetAsDesiredByHand) {
		this -> Reset();
	}
	else {
		this -> SetDuration((long)(runDurationInHrs*3600)); // convert to seconds
		//if (!bHindcast)	// assume user knows what they are doing
			this -> SetStartTime(startRelTime);
		//else
			//this -> SetStartTime(startRelTime - model->GetDuration());
		this -> Reset();
		this -> SetTimeStep((long) (timeStepInMinutes*60));// convert to seconds
	}
	
	saveOutputStep = this -> GetOutputStep();
	this -> SetOutputStep((long)(outputStepInMinutes*60)); // convert to seconds
	

	if (!bUseLEsFromFile && !bEverythingSetAsDesiredByHand)
	{	// do some error checking
		Boolean bWantEndRelPosition = (!EqualWPoints(startRelPos,endRelPos));
		if(!IsWaterPoint(startRelPos)) {
			printError("The release start position must be in the water."); 
			goto done;}
		
		if(!IsAllowableSpillPoint(startRelPos)){
			printError("This map has not been set up for spills in the area of your release start position.");
			goto done;}
		
		if(bWantEndRelPosition && !IsWaterPoint(endRelPos)) {
			printError("The release end position must be in the water."); 
			goto done;}
		
		if(bWantEndRelPosition && !IsAllowableSpillPoint(endRelPos)){				
			printError("This map has not been set up for spills in the area of your release end position."); 
			goto done;}
			
		if (z > 0)
		{
			TMap *map = Get3DMap();
			double botZ = INFINITE_DEPTH;
			if (map) botZ = map->DepthAtPoint(startRelPos) ;
			if (z > botZ)	
			{
				char errStr[64];
				sprintf(errStr,"The spill depth cannot be greater than total depth which is %g meters.",botZ);
				printError(errStr);
				goto done;
			}
		}
	}
	
	
	// set the spill
	if(!bUseLEsFromFile && !bEverythingSetAsDesiredByHand)  // JLM 2/13/01
	{	// original code, create the spill from scratch (not from an LE file)	
		LESetSummary  summary;
		TOLEList 	*forecastLEList=0;
		TOLEList 	*uncertaintyLEList=0;
	
		memset(&summary,0,sizeof(summary));
	
		summary.numOfLEs = numLEs;
		summary.pollutantType = pollutantType;
		summary.totalMass = totalMass; 
		summary.massUnits = massUnits;
		
		summary.startRelPos = startRelPos;
		summary.endRelPos = endRelPos;
		summary.bWantEndRelPosition = (!EqualWPoints(startRelPos,endRelPos));
		
		summary.startRelTime = startRelTime;
		summary.endRelTime = endRelTime;
		summary.bWantEndRelTime = (endRelTime != startRelTime);
		
		//summary.z = 0.0;
		summary.z = z;
		summary.density = GetPollutantDensity(summary.pollutantType);
		summary.ageInHrsWhenReleased = 0;
	
		// add the LE set
		forecastLEList = new TOLEList ();
		uncertaintyLEList = new TOLEList ();
		if (!forecastLEList || ! uncertaintyLEList) 
			err = memFullErr;
		
		if(!err)
		{
			forecastLEList -> SetWindageInfo(windageInfo);
			err = forecastLEList->Initialize(&summary,true);	
		}
	
		if(!err) {
			uncertaintyLEList->fLeType = UNCERTAINTY_LE;
			uncertaintyLEList->fOwnersUniqueID = forecastLEList->GetUniqueID();
			uncertaintyLEList -> SetWindageInfo(windageInfo);
			err = uncertaintyLEList->Initialize(&summary,true);
		}
	
		if(!err) 
			err = model->AddLEList(forecastLEList, 0);
		if(!err) 
			err = model->AddLEList(uncertaintyLEList, 0);
		if(err) 
		{ // probably ran out of memory, we can't do the run
			if(err == memFullErr)
				printError("Run skipped due to memFullErr setting the spill.");
			else
				printError("Run skipped due to error setting the spill.");
			goto done; // bad data, don't do the run
		}
		
	}
	
	
	/////////////////////////////////////////////////
	// check that we have the expected data
	// if not make a note in the Error Log, but don't stop the run
	/////////////////////////////////////////////////

	// check that we have winds for the time period
	// check that we have a diffusion mover

	
	// create and open the output file
	memset(&gRunSpillForecastFile,0,sizeof(gRunSpillForecastFile));
	if(outputPath[0]) {
		hdelete(0, 0, outputPath);
		if (err = hcreate(0, 0, outputPath, '\?\?\?\?', 'BINA'))
			{ TechError("HandleRunSpillMessage()", "hcreate()", err); goto done ; }
		
		if (err = FSOpenBuf(0, 0, outputPath, &gRunSpillForecastFile, 1000000, FALSE))
			{ TechError("HandleRunSpillMessage()", "FSOpenBuf()", err); goto done ; }
	}

	if(moviePath[0] && bMakeMovie) {
		strcpy(fMoviePath,moviePath);
		this->OpenMovieFile();
	}
	
	
	/////////////////////////////////////////////////
	// do the run
	/////////////////////////////////////////////////
	// run the model outputting the LE's
	this -> Reset();
	saveBool = this->WantOutput(); // oh... just in case
	savebSaveRunBarLEs = this->bSaveRunBarLEs; 
	this->bSaveRunBarLEs = FALSE; // we want this to run as fast as possible, so don't write out these unnecessary files
	if(outputDirectory[0]) {
		this->SetWantOutput(true);
		strcpy(str,outputDirectory);
		strcat(outputDirectory,"LE_");
		this->SetOutputFileName (outputDirectory);
	}
	else {
		this->SetWantOutput(false);
	}
	if(mossDirectory[0]) {
		gSaveMossFiles=true;
		strcpy(gMossPath,mossDirectory);
	}
	else {
		gSaveMossFiles=false;
	}
	//err = this->Run(this->GetEndTime());
	if (model->bHindcast)
		model->Run(model->GetStartTime()); 
	else
		model->Run(model->GetEndTime());
	// reset the parameters we changed
	this->SetWantOutput(saveBool);
	gSaveMossFiles = false;
	this->bSaveRunBarLEs = savebSaveRunBarLEs; 
	
	// reset those model parameters that the user can not normally change
	this -> SetOutputStep(saveOutputStep);
	
	// close the output file
	if (gRunSpillForecastFile.f)  FSCloseBuf(&gRunSpillForecastFile);
	if (bMakeMovie==true) {this->CloseMovieFile();bMakeMovie=false;}
	
done:	 // reset important globals, etc
	memset(&gRunSpillForecastFile,0,sizeof(gRunSpillForecastFile));
	gRunSpillNoteStr[0] = 0;
	// gTapWindOffsetInSeconds = 0;		minus AH 06/20/2012
	return err;

}

OSErr TModel::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	char 	ourName[kMaxNameLen];
	char path[256] = "", classicPath[256], *p;
	char bSuppressDrawing[32] = "";
	OSErr err = 0;
	char msg[512], str[256];
	static unsigned long sStartTicks = 0;
	unsigned long nowTicks; 
	
	// see if the message is of concern to us
	strcpy(ourName,"MODEL");
	if(message->IsMessage(ourName))
	{
		long messageCode = message->GetMessageCode();
		switch(messageCode)
		{
			// new COMMANDFILE messages
			case M_RESET:
				this->Reset();
				break;
			case M_CLOSE:
				err = CloseSaveFile(FALSE,FALSE);
				err = model->fWizard->CloseMenuHit();
				//this->NewDirtNotification();// tell model about dirt
				break;
			case M_CLEARSPILLS:
				this->DisposeModelLEs();
				this->NewDirtNotification();// tell model about dirt
				break;
			case M_CLEARWINDS:
				this -> DisposeAllMoversOfType(TYPE_WINDMOVER);
				this -> NewDirtNotification();// tell model about dirt
				break;
			case M_OPEN:
				message->GetParameterString("PATH",path,256);
				if (strstr(path,"resnum"))
				{
					char *p, locFilePath[256], saveFilePath[256], firstPartOfFile[512], strLine[512];
					long line, lenToRead = 512;
					if (model->fWizard->HaveOpenWizardFile())
					{	// there is a location file
						model->fWizard->GetLocationFileFullPathName(locFilePath);
						p = strrchr(locFilePath,DIRDELIMITER);
						if(p) *(p+1) = 0;
						err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
		
						if (!err)
						{ 
							firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
							NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
							strcpy(saveFilePath,locFilePath);
							// add the relative part 
							strcat(saveFilePath,strLine);
							//strcpy(saveFilePath,strLine);
							strcpy(path,saveFilePath);	// may not want to overwrite path
						}
					}
				}
				else					
				{
					//code goes here, want Location File SAV files to be in same place as Location File
					char *p, locFilePath[256], saveFilePath[256], firstPartOfFile[512], strLine[512];
					long line, lenToRead = 512;
					if (model->fWizard->HaveOpenWizardFile())
					{	// there is a location file
						model->fWizard->GetLocationFileFullPathName(locFilePath);
						p = strrchr(locFilePath,DIRDELIMITER);
						//if(p) *(p+1) = 0;
						if(p) *(p) = 0;
						//err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
		
						//if (!err)
						{ 
							//firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
							//NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
							strcpy(saveFilePath,locFilePath);
							// add the relative part 
							//strcat(saveFilePath,strLine);
							strcat(saveFilePath,path);
							//strcpy(saveFilePath,strLine);
							strcpy(path,saveFilePath);	// may not want to overwrite path
						}
					}
				}
				ResolvePath(path);
				if(FileExists(0,0,path))
				{	// need code to handle resnum rather than full path
					(void)OpenSaveFileFromPath(path,true);
				}
				else {
					sprintf(msg,"Specified file for M_OPEN does not exist.%s%s",NEWLINESTRING,path);
					printError(msg);
					err = -7;	// so command file doesn't try to keep running
				}
				break;	
			case M_SAVE:
				message->GetParameterString("PATH",path,256);
				//ResolvePath(path);
				if (ConvertIfUnixPath(path, classicPath)) strcpy(path,classicPath);
				err = ResolvePathFromCommandFile(path);
				if (err) ResolvePathFromApplication(path);
				strcpy(str,path);
				p =  strrchr(str,DIRDELIMITER);
				if(p) *(p+1) = 0; // chop off the file name
				// create the folder if it does not exist
				if (!FolderExists(0, 0, str)) 
				{
					long dirID;
					err = dircreate(0, 0, str, &dirID);
					if(err) 
					{	// try to create folders 
						err = CreateFoldersInDirectoryPath(str);
						if (err)	
						{
							printError("Unable to create the directory for the output file.");
						}
					}
				}
				if (!err) err = SaveSaveFile(path);
				break;			
			case M_RUNSPILL:
				err = this->HandleRunSpillMessage(message);
				break;
			case M_CREATESPILL:
				err = this->HandleCreateSpillMessage(message);
				break;
			case M_RUN:
				err = this->HandleRunMessage(message);
				break;
			case M_STARTTIMER:
				sStartTicks = TickCount();
				break;
			case M_STOPTIMER:
				nowTicks = TickCount();
				if (!gSuppressDrawing)
				{
					sprintf(msg,"Time taken: %f seconds",(nowTicks - sStartTicks)/60.0);
					settings.doNotPrintError = false;// allows dialogs to come up more than once
					printNote(msg);
				}
				break;
			case M_QUIT:
				(void)CloseSaveFile(false,true);
				DoQuit();
				settings.quitting = true; // this will cause us to quit in the main event loop
				break;
			/////////////////////////////////////////////////
			/////////////////////////////////////////////////
			case M_SETFIELD:
			{					
				char typeName[64] ="";
				double theTimeStep = 0;
				long maxDuration=3*24;
				Boolean includeMinRegret=0, preventLandJumping=0;
				////////
				err = message->GetParameterAsDouble("TIMESTEP",&theTimeStep);
				if(!err)
				{	
					if(theTimeStep > 0)// do we have any other max or min limits ?
					{
						this->SetTimeStep ((long) round(theTimeStep*3600)); // convert from hrs to seconds
						theTimeStep = this -> GetTimeStep();
						this->NewDirtNotification();// tell model about dirt
					}
				}
				////////
				err = message->GetParameterAsBoolean("INCLUDEMINREGRET",&includeMinRegret);
				if(!err)
				{	
					//if(includeMinRegret > 0)// do we have any other max or min limits ?
					{
						this->SetUncertain (includeMinRegret); // convert from hrs to seconds
						this->NewDirtNotification();// tell model about dirt
					}
				}
				////////
				err = message->GetParameterAsBoolean("PREVENTLANDJUMPING",&preventLandJumping);
				if(!err)
				{	
					//if(preventLandJumping > 0)// do we have any other max or min limits ?
					{
						this->SetPreventLandJumping (preventLandJumping); // convert from hrs to seconds
						this->NewDirtNotification();// tell model about dirt
					}
				}
				////////
				err = message->GetParameterAsLong("MAXDURATION",&maxDuration);
				if(!err)
				{	// either need to re-set to 3 days if read in a new file (in InitModel) or put inside the timestep parse
					// so can tell whether or not a duration max has been set
					if(maxDuration > 0)
					{
						fMaxDuration = (float)maxDuration; // hours
					}
				}
				////////
				message->GetParameterString("TYPE",typeName,64);
				if(!strcmpnocase(typeName,"dateLimits")) 
				{	
					char yearStr[64]="";
					long startYear,startMonth,startDay,endYear,endMonth,endDay;
					DateTimeRec userTime, startTime, endTime;
					Seconds startTimeInSeconds, endTimeInSeconds;
					memset(&startTime,0,sizeof(startTime));
					memset(&endTime,0,sizeof(endTime));
					SecondsToDate(modelTime,&userTime);

					message->GetParameterString("StartYear",yearStr,64);
					// allow option to have month/day restriction for any year
					if(!strcmpnocase(yearStr,"any")) 
						startYear = userTime.year;		
					else
						err = message->GetParameterAsLong("StartYear",&startYear);
					if (!err) startTime.year = startYear;
					err = message->GetParameterAsLong("StartMonth",&startMonth);
					if (!err) startTime.month = startMonth;
					err = message->GetParameterAsLong("StartDay",&startDay);
					if (!err) startTime.day = startDay;

					message->GetParameterString("EndYear",yearStr,64);
					// allow option to have month/day restriction for any year
					if(!strcmpnocase(yearStr,"any")) 
						endYear = userTime.year;
					else
						err = message->GetParameterAsLong("EndYear",&endYear);
					if (!err) endTime.year = endYear;
					err = message->GetParameterAsLong("EndMonth",&endMonth);
					if (!err) endTime.month = endMonth;
					err = message->GetParameterAsLong("EndDay",&endDay);
					if (!err) endTime.day = endDay;
					
					if (!err)
					{
						// code goes here, should check whether month 2 < month 1 when there is no year restriction
						// e.g. if there is a restriction to winter season, would need to check over two years
						DateToSeconds(&startTime,&startTimeInSeconds);
						DateToSeconds(&endTime,&endTimeInSeconds);
						if (modelTime < startTimeInSeconds || modelTime > endTimeInSeconds)
						{
							printError("You have chosen a model time outside the limits of this location file");
						}
					}
					
				}
				////////
				message->GetParameterString("ERRORLOG",path,256);
				if(path[0]) {
					char classicPath[256], *p;
					//ResolvePath(path);
					if (ConvertIfUnixPath(path, classicPath)) strcpy(path,classicPath);
					err = ResolvePathFromCommandFile(path);
					if (err) ResolvePathFromApplication(path);
					strcpy(gCommandFileErrorLogPath,path);
					// create the folder for this file if need be
					if (p = strrchr(path, DIRDELIMITER)) *(p+1) = 0; // remove short file name
					if (!FolderExists(0, 0, path)) 
					{
						long dirID;
						err = dircreate(0, 0, path, &dirID);
						if(err) {
							printError("Unable to create the ErrorLog directory");
						} 
					}
					printNote("Error log file");
				}
				////////
				message->GetParameterString("SUPPRESSDRAWING",bSuppressDrawing,32);
				if(bSuppressDrawing[0]=='F'||bSuppressDrawing[0]=='f'||bSuppressDrawing[0]=='0') {
					gSuppressDrawing = false;
				}
				////////
				break;
			}
			
			case M_CREATEMAP:
			{
				char mapName[kMaxNameLen] = "";
				char typeName[64] ="";
				TMap *map = nil;
				Boolean unrecognizedType = false;
				message->GetParameterString("NAME",mapName,kMaxNameLen);
				message->GetParameterString("TYPE",typeName,64);
				message->GetParameterString("PATH",path,256);
				ResolvePath(path);
		
				if(!path[0]) goto noPathErr;
				if(!strcmpnocase(typeName,"Vector")) 
				{
					TVectorMap *vmap = new TVectorMap (mapName, emptyWorldRect);
					if(!vmap)  goto memErr;
					err = vmap->InitMap(path);
					if(err) goto initMapErr;
					map = (TMap *)vmap;
				}
				else if(!strcmpnocase(typeName,"OSSM")) 
				{
					TOSSMMap *omap = new TOSSMMap(mapName, emptyWorldRect);
					if(!omap)  goto memErr;
					err = omap->InitMap(path);
					if(err) goto initMapErr;
					map = (TMap *)omap;
				}
				else if(!strcmpnocase(typeName,"PTCUR")) 
				{					
					TMap *newMap = 0;
					TCurrentMover *newMover = 0;
					// check if path is a resnum with a path and if so convert to actual path
					char *p, locFilePath[256], netcdfFilePath[256], firstPartOfFile[512], strLine[512], moverName[kMaxNameLen];
					long line, lenToRead = 512;
					char topFilePath[256];
					message->GetParameterString("topFile",topFilePath,256);
					if (topFilePath[0]) ResolvePath(topFilePath);
					/*if(topFilePath[0])
					{
						strcpy(fTopFilePath,topFilePath);
					}*/
					if (model->fWizard->HaveOpenWizardFile())
					{	// there is a location file
						model->fWizard->GetLocationFileFullPathName(locFilePath);
						p = strrchr(locFilePath,DIRDELIMITER);
						if(p) *(p+1) = 0;
						err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
		
						if (!err)
						{ 
							firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
							NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
							strcpy(netcdfFilePath,locFilePath);
							// add the relative part 
							strcat(netcdfFilePath,strLine);
						}
					}
					else	// assume path is a file (for batch mode), not a file name in the resource
						strcpy(netcdfFilePath,path);
					if (!err)
					{
						if (!FileExists(0,0,netcdfFilePath)) 
						{
							if (model->fWizard->HaveOpenWizardFile())
								sprintf(msg,"The file %s was not found in %s",strLine,locFilePath); 
							else
								sprintf(msg,"The file %s was not found",netcdfFilePath); 
							// code goes here, give opportunity to find it 
							printError(msg);
						}
						else
						{
							//TMap *newMap = 0;
							strcpy(moverName,mapName);
							strcat(moverName,".cur");
							//newMover = CreateAndInitCurrentsMover (model->uMap,false,netcdfFilePath,moverName,&newMap);
							newMover = CreateAndInitLocationFileCurrentsMover (model->uMap,netcdfFilePath,moverName,&newMap,topFilePath);
							//if (newMap) {printError("An error occurred in TMap::CheckAndPassOnMessage()");}
						}
					}
					//TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,"ptcurfile",&newMap);	// already have path
					if (newMover)
					{
						if(newMover)
						{
							//Boolean timeFileChanged = false;
							if (!newMap) 
							{
								//err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
								err = model->uMap -> AddMover (newMover, 0);
							}
							else
							{
								newMap->SetClassName(mapName);	// name must match location/command file name to be able to add other movers
								err = model -> AddMap(newMap, 0);
								if (err) 
								{
									newMap->Dispose(); delete newMap; newMap = 0; 
									newMover->Dispose(); delete newMover; newMover = 0;
									return err; 
								}
								//err = AddMoverToMap(newMap, timeFileChanged, newMover);
								err = newMap -> AddMover (newMover, 0);
								if(err) 
								{
									newMap->Dispose(); delete newMap; newMap = 0; 
									newMover->Dispose(); delete newMover; newMover = 0;
									return err; 
								}
								newMover->SetMoverMap(newMap);
								//strcat(mapName,".cur");
								//newMover->SetMoverName(mapName);	
								this->NewDirtNotification();
								return err;
							}
						}
					}
				}
				else goto unrecogErr;
				
				if (map && !err) err = this->AddMap(map, 0);
				this->NewDirtNotification();
			}
		}
	}
	/////////////////////////////////////////////////
	// we have no sub-guys that that need us to pass this message 
	// because we use BroadcastMessage
	/////////////////////////////////////////////////

	/////////////////////////////////////////////////
	//  no need to pass on this message to our base class
	/////////////////////////////////////////////////
	return err;
initMapErr:
	printError("InitMap returned an error"); // Note: this message will not be shown to the user if InitMap put up an error.
	return err;
memErr:
	TModelOutOfMemoryAlert("CheckAndPassOnMessage");
	return err;
unrecogErr:
	printError("Unrecogonized map type in M_CREATEMAP message");
	return err;
noPathErr:
	printError("No PATH given in M_CREATEMAP message");
	return err;
}

/////////////////////////////////////////////////
OSErr TModel::BroadcastMessage(long messageCode, char* targetName, UNIQUEID targetUniqueID, char* dataStr, CHARH dataHdl)
{
	TModelMessage *message = 0;
	long i,n;
	UNIQUEID zeroID = ZeroID();
	OSErr err = 0;
	
	if(!EqualUniqueIDs(zeroID,targetUniqueID)) {
		// send the message by the UniqueID
		if(dataStr) message = new TModelMessage(messageCode,targetUniqueID,dataStr);
		else message = new TModelMessage(messageCode,targetUniqueID,dataHdl);
	}
	else { 
		// send the message by the name
		if(dataStr) message = new TModelMessage(messageCode,targetName,dataStr);
		else message = new TModelMessage(messageCode,targetName,dataHdl);
	}
	
	if(message)
	{
		// check to see if the message is for for us
		// for now only get error for TModel, where we can hit a cmdperiod
		// will stop all TAP runs on a cmdperiod, but no other error matters
		err = this->CheckAndPassOnMessage(message);
		if (err==-7) return err;
		
		// tell LE Sets ??
		for (i = 0, n = this->LESetsList->GetItemCount(); i < n; i++)
		{
			TLEList		*thisLEList = 0;
			this->LESetsList->GetListItem((Ptr) &thisLEList, i);
			if(thisLEList) err = thisLEList->CheckAndPassOnMessage(message);
		}
		
		// tell maps in mapList
		for (i = 0, n = this->mapList->GetItemCount() ; i < n ; i++) 
		{
			 TMap *thisMap = 0;
			 this->mapList->GetListItem((Ptr)&thisMap, i);
			if(thisMap) err = thisMap->CheckAndPassOnMessage(message);
		}

		// tell universalMap 
		err = this->uMap->CheckAndPassOnMessage(message);

		// tell weather objects
		for (i = 0, n = this->weatherList->GetItemCount (); i < n; i++)
		{
			TWeatherer *thisWeatherer = 0;
			this->weatherList->GetListItem((Ptr) &thisWeatherer, i);
			if(thisWeatherer) err = thisWeatherer->CheckAndPassOnMessage(message);
		}

		// tell the overlays
		for (i = 0, n = this->fOverlayList->GetItemCount() ; i < n ; i++) 
		{
			 TOverlay *thisOverlay = 0;
			 this->fOverlayList->GetListItem((Ptr)&thisOverlay, i);
			if(thisOverlay) err = thisOverlay->CheckAndPassOnMessage(message);
		}
			
		// Tell Wizard
		err = this->fWizard->CheckAndPassOnMessage(message);
		
		if(message) delete message; message = 0;
	}
	return err;
}

/////////////////////////////////////////////////
OSErr TModel::BroadcastMessage(long messageCode, char* targetName, char* dataStr, CHARH dataHdl)
{
	UNIQUEID uid = ZeroID();
	return this -> BroadcastMessage(messageCode,targetName,uid,dataStr,dataHdl);
}

/////////////////////////////////////////////////
OSErr TModel::BroadcastMessage(long messageCode, UNIQUEID uid, char* dataStr, CHARH dataHdl)
{
	return this -> BroadcastMessage(messageCode,nil,uid,dataStr,dataHdl);
}

/////////////////////////////////////////////////

/////////////////////////////////////////////////
void TModel::BroadcastToSelectedItem(long messageCode, char* dataStr, CHARH dataHdl)
{
	ListItem localItem;
	Boolean itemSelected = SelectedListItem(&localItem);
	UNIQUEID uid;
	OSErr err = 0;

	if (!itemSelected) 
		return;
	
	if (!localItem.owner) 
		return;
		
	uid = localItem.owner -> GetUniqueID(); 
	
	err = this -> BroadcastMessage(messageCode, uid, dataStr, dataHdl);
	
}

/////////////////////////////////////////////////


Boolean TModel::UserIsEditingSplots(void)
{	
	TClassID *object = this -> ItemBeingEditedInMappingWindow();
		
	if (!object)
		return FALSE;
	
	if(object -> IAm(TYPE_LELIST))
		return object -> UserIsEditingMeInMapDrawingRect();
	
	return FALSE;

}


long TModel::NumEditableSplotObjects(void)
{	
	long numEditableSplotSets = 0;
	long i,n;
	TLEList		*thisLEList;
	
	for (i = 0, n = LESetsList -> GetItemCount (); i < n; i++)
	{
		LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if (thisLEList -> IAmEditableInMapDrawingRect())
			numEditableSplotSets++;
	}
	
	return numEditableSplotSets;
}

Boolean TModel::EditableSplotObjectIsSelected()
{
	Boolean haveSelection = false;
	ListItem localItem;

	haveSelection = SelectedListItem(&localItem);
	if (haveSelection && localItem.owner)  {
		if(localItem.owner -> IAm(TYPE_LELIST))
			if (localItem.owner -> IAmEditableInMapDrawingRect())
				return true; // already selected
	}
	return false;
}

void TModel::SelectAnEditableSplotObject(void)
{	// used for example when the user selects  the eraser tool	
	long numEditableSplotSets = 0;
	long i,n;
	TLEList		*thisLEList;
	Boolean haveSelection;
	ListItem localItem;
	
	// first check the currently selected item
	haveSelection = SelectedListItem(&localItem);
	if (haveSelection && localItem.owner)  {
		if(localItem.owner -> IAm(TYPE_LELIST))
			if (localItem.owner -> IAmEditableInMapDrawingRect())
				return ; // already selected
	}
	///////////
	// check all the LE sets
	// find the last one in the list 
	// because the user probably wants to edit
	// the last one they created
	////////
	n = LESetsList -> GetItemCount ();
	for (i = n -1 ; i >= 0; i--)
	{
		LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if (thisLEList -> IAmEditableInMapDrawingRect()) {
			Boolean foundItem = SelectListItemOfOwner(thisLEList);
			if (foundItem) 
				break;
		}
	}
}

TClassID * TModel::ItemBeingEditedInMappingWindow(void)
{
	TClassID *prevItemOwner = 0;
	// the only items that can be edited are in the list
	//	so go through the list and ask them
	long i,n;
	short style = normal;
	char s[256];
	Boolean haveSelection;
	ListItem localItem;
	//long numItems;
	TLEList *thisLEList;
		
	ValidateListLength(); // make sure our list is up to date

	haveSelection = SelectedListItem(&localItem);
	if (haveSelection && localItem.owner)  {
		if (localItem.owner -> UserIsEditingMeInMapDrawingRect())
			return localItem.owner;
	}

	/////////////////////////////////////////////////
	// JLM 7/25/00, this gets to slow when there is a big list 
	// (i.e. one of the objects is displaying a lot of data)
	// So it is better to check the objects making up the list
	/////////////////////////////////////////////////
	//	// then go through the list until we find an item with the right owner
	//	numItems = this -> GetListLength();
	//	for(n= 0; n <numItems; n++)
	//	{
	//		localItem = this->GetNthListItem(n, 0, &style, s);
	//		if (localItem.owner && localItem.owner != prevItemOwner) {
	//			// ask this object
	//			prevItemOwner = localItem.owner;
	//			if (localItem.owner -> UserIsEditingMeInMapDrawingRect())
	//				return localItem.owner;
	//		}
	//	}
	/////////////////////////////////////////////////
	// right now, only SprayLE sets can be edited
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
	{
		LESetsList->GetListItem((Ptr)&thisLEList, i);
		if(thisLEList && thisLEList -> UserIsEditingMeInMapDrawingRect())
			return thisLEList;
	}
	
	/////////////////////////////////////////////////

	
	return nil; // no objects are being edited
	
}

OSErr TModel::DropObject(TClassID  *object, Boolean bDispose)
{
	if(!object) return 0;
	
	if(object -> IAm(TYPE_LELIST))
		return this -> DropLEList((TLEList *)object,bDispose);
	
	printError("Unknown class in DropObject");
	return -1;
}


OSErr TModel::EditObjectInMapDrawingRect(TClassID *newObjectToEdit)
{// returns an error if currently editable object refuses to leave edit mode
	TClassID *prevItemBeingEdited = 0;
	Boolean deleteMe = FALSE;
	Boolean refusesToLeaveEditMode = FALSE;
	OSErr err = 0;
	
beginning:
	prevItemBeingEdited = this -> ItemBeingEditedInMappingWindow();
	if (newObjectToEdit == prevItemBeingEdited) {
	
		// make sure they are still editable
		if (prevItemBeingEdited && ! prevItemBeingEdited -> IAmCurrentlyEditableInMapDrawingRect()) {
			refusesToLeaveEditMode = prevItemBeingEdited -> StopEditingInMapDrawingRect(&deleteMe);
			if(refusesToLeaveEditMode) goto refusal;
			if(deleteMe) 
				err = this -> DropObject(prevItemBeingEdited,TRUE);
		}
		return 0;
	}
		
	if (prevItemBeingEdited)
	{
		refusesToLeaveEditMode = prevItemBeingEdited -> StopEditingInMapDrawingRect(&deleteMe);
		if(refusesToLeaveEditMode) goto refusal;
		if(deleteMe) 
			err = this -> DropObject(prevItemBeingEdited,TRUE);
		goto beginning;// in case we have two objects that think they are being edited
	}
	
	if (newObjectToEdit) {
			if (newObjectToEdit -> IAmCurrentlyEditableInMapDrawingRect())
				newObjectToEdit -> StartEditingInMapDrawingRect();
	}

	return 0;

refusal:
	// 
		SelectListItemOfOwner(prevItemBeingEdited);
		return -1;
}


void TModel::CheckEditModeChange(void)
{	// the selected item gets a chance to be edited
	ListItem item;
	Boolean haveSelection;
	// an item will be edited if it is selected in the list
	// and the tools are right
	
	haveSelection = SelectedListItem(&item);
	if (haveSelection) 
		(void)this -> EditObjectInMapDrawingRect(item.owner);
	else
		(void)this -> EditObjectInMapDrawingRect(nil);
		
		
	// since EditObjectInMapDrawingRect can delete item.owner
	// reset our variables
	haveSelection = SelectedListItem(&item);
	
	// don't let then have an edit tool unless the object being edited is selected
	if(haveSelection && item.owner && item.owner -> UserIsEditingMeInMapDrawingRect()) {
		// they are editing, we don't have to enforce anything
	}
	else { 
		if (IsEditTool(settings.currentTool))
			SetTool(ARROWTOOL);
	}
}

/////////////////////////////////////////////////

Boolean GetPositionAfterNextStep(WorldPoint3D wp,Seconds timeStep, WorldPoint3D *theMovedPt)
{	// returns TRUE if some movers apply at this point
	WorldPoint3D	thisMove = {0,0,0.}, movedPoint = {0,0,0.};
	TMap		*bestMap;
	TMover		*thisMover;
	long		i, j,index,k,d;
	Boolean moversAffectPoint = false;
	LERec theLE;
	
	memset(&theLE,0,sizeof(theLE));
	theLE.p = wp.p; // GetMove now takes an LERec
	theLE.z = wp.z; // GetMove now takes an LERec
	//theLE.windage = GetRandomFloat(.01, .04); // need to set windage to get wind movement
	movedPoint = wp; // start at same place
	bestMap = model->GetBestMap(wp.p);
	if (!bestMap)
	{
		// off the maps
		// no movement, right ??
	}
	else
	{
		if (bestMap -> OnLand (wp.p))
		{
			// don't move land points
		}
		else
		{
			// loop through each mover in the universal map
			for (k = 0, d = model -> uMap -> moverList -> GetItemCount (); k < d; k++)
			{
				model -> uMap -> moverList -> GetListItem ((Ptr) &thisMover, k);
				if (!thisMover -> IsActive ()) continue;
				if (!thisMover -> IAm(TYPE_CURRENTMOVER)) continue;	// show movement only handles currents, not wind and dispersion
				moversAffectPoint = true; // we moved this LE
				//thisMove = thisMover -> GetMove (timeStep,0,0,wp,FORECAST_LE);
				//thisMove = thisMover -> GetMove (model->GetStartTime(), model->GetEndTime(), model->GetModelTime(), timeStep,0,0,&theLE,FORECAST_LE);	// AH 07/10/2012
				thisMove = thisMover -> GetMove (model->GetModelTime(), timeStep,0,0,&theLE,FORECAST_LE);	// no uncertainty included here
				//movedPoint.pLat  += thisMove.pLat;
				//movedPoint.pLong += thisMove.pLong;
				movedPoint.p.pLat  += thisMove.p.pLat;
				movedPoint.p.pLong += thisMove.p.pLong;
				movedPoint.z += thisMove.z;
			}

			// loop through each mover in the best map
			for (k = 0, d = bestMap -> moverList -> GetItemCount (); k < d; k++)
			{
				bestMap -> moverList -> GetListItem ((Ptr) &thisMover, k);
				if (!thisMover -> IsActive ()) continue;
				if (!thisMover -> IAm(TYPE_CURRENTMOVER)) continue;	// show movement only handles currents, not wind and dispersion
				moversAffectPoint = true; // we moved this LE
				//thisMove = thisMover -> GetMove (timeStep,0,0,wp,FORECAST_LE);
				//thisMove = thisMover -> GetMove (model->GetStartTime(), model->GetEndTime(), model->GetModelTime(), timeStep,0,0,&theLE,FORECAST_LE);	// AH 07/10/2012
				thisMove = thisMover -> GetMove (model->GetModelTime(), timeStep,0,0,&theLE,FORECAST_LE);	// no uncertainty included here
				movedPoint.p.pLat  += thisMove.p.pLat;
				movedPoint.p.pLong += thisMove.p.pLong;
				movedPoint.z += thisMove.z;
			}
		}
	}
	*theMovedPt = movedPoint; // return value through pointer
	return moversAffectPoint;
}

/////////////////////////////////////////////////

void TModel::MovementString(WorldPoint3D wp,char* str)
{
	str[0]=0;
	if(this->fDrawMovement)
	{
		// convert position to lat lng
		Seconds timeStep = this->GetTimeStep();
		WorldPoint3D movedPt;
		Boolean moversAffectPt = GetPositionAfterNextStep(wp,timeStep,&movedPt);
		if(moversAffectPt && timeStep != 0)
		{
			// for now offset
			double distanceInKm = DistanceBetweenWorldPoints(wp.p,movedPt.p);
			char magnitudeStr[32],uStr[32],vStr[32];
			char timeStr[32];
			double latDist = LatToDistance(movedPt.p.pLat - wp.p.pLat); // in kilometers
			double lngDist = LongToDistance(movedPt.p.pLong - wp.p.pLong,wp.p);// in kilometers
			StringWithoutTrailingZeros(magnitudeStr,distanceInKm*1000/timeStep,4);
			StringWithoutTrailingZeros(vStr,latDist*1000/timeStep,4);
			StringWithoutTrailingZeros(uStr,lngDist*1000/timeStep,4);
			sprintf(str," [mag = %s   u = %s   v = %s   <m/s>]",magnitudeStr,uStr,vStr);
		}
	}
}

/////////////////////////////////////////////////

static short gNumLEsHori = 30;
static short gNumLEsVert = 30;

typedef struct
{
	Boolean draw;
	Point	screenStartPt; 	// in local screen coordinates
	Point	screenEndPt; 		
} MovementLE,**MovementLEHdl;



void TModel::DrawLEMovement(void) 
{	// draws LE movement over 1 time step
	static WorldRect sPreviousView ={0,0,0,0};
	static Rect previousMapDrawingRect ={0,0,0,0};
	static Seconds sPreviousModelTime =-1;
	static Seconds sPreviousModelStep =-1;
	static MovementLEHdl sMovementLEHdl = 0;
	static short sNumLEsHori = -1;
	static short sNumLEsVert = -1;

	////////////
	/////////////////////////////////////////////////
	WorldPoint3D	 movedPoint = {0,0,0.}, wp = {0,0,0.};
	MovementLE wLE;
	short	displayWidth,displayHeight;
	long		i, j,index;
	//GrafPtr		savePort; NOTE: don't change port because of printing !!
	//Rect		saveClip;
	RGBColor saveColor;
	short numLEsHori = gNumLEsHori;
	short numLEsVert = gNumLEsVert;
	char* errorMessageStr = 0;
	float arrowDepth = 0.;
	
	long	numOfLEs = numLEsHori * numLEsVert;
	
	Boolean recalculate = false;
	Seconds currentModelTime = this->GetModelTime();
	WorldRect currentView =  settings.currentView;
	Seconds currentTimeStep = this->GetTimeStep();
	Rect currentMapDrawingRect = MapDrawingRect();
	Boolean userInAdvancedMode = (this->GetModelMode() == ADVANCEDMODE);
	//Boolean drawMovement = userInAdvancedMode && this->fDrawMovement;
	Boolean drawMovement = this->fDrawMovement;	// 3/1/03 Try allowing all users to look at currents
	
	/////////////////////////////////////////////////

	if (ThereIsA3DMover(&arrowDepth))	
		wp.z = arrowDepth;	// show movement grid at the depth level of interest
	
	if(!drawMovement)
	{
		if(sMovementLEHdl) DisposeHandle((Handle)sMovementLEHdl); sMovementLEHdl = nil;
		return;
	}
	
	//SetWatchCursor();
	
	if(EqualWRects(sPreviousView,currentView)) recalculate = true;
	if(sPreviousModelTime != currentModelTime) recalculate = true;
	if(sPreviousModelStep != currentTimeStep) recalculate = true;
	if(!sMovementLEHdl) recalculate = true; 
	if(sNumLEsHori != numLEsHori) recalculate = true;
	if(sNumLEsVert != numLEsVert) recalculate = true;
	
	if(recalculate)
	{
		if(sMovementLEHdl) DisposeHandle((Handle)sMovementLEHdl); sMovementLEHdl = nil;
		
		sMovementLEHdl = (MovementLEHdl)_NewHandleClear(numOfLEs*sizeof(MovementLE));
		if(!sMovementLEHdl)
		{
			errorMessageStr = "Out of memory";
		}
		else
		{
			/////////////////////////////////////////////////
			// fill in the lat lng positions
			/////////////////////////////////////////////////
			//
			
			///////////////////////////
			// Note: setting the seed here means we need to set the seed at of beginning of Step()
			//srand(this->modelTime); // JLM 1/25/99, set seed for rand calls
			// JLM, Hmmmm... giving diffusion a plus shaped pattern, 
			//  so abandoned this idea 1/26/99
			/////////////////////////////

			displayWidth = RectWidth(currentMapDrawingRect);
			displayHeight = RectHeight(currentMapDrawingRect);
			for (i = 0; i < numLEsHori; i++)
			{
				wLE.screenStartPt.h = currentMapDrawingRect.left+ round(displayWidth*(i+1)/(float)(numLEsHori+1));
				for (j = 0; j < numLEsVert; j++)
				{
					wLE.draw = false; 
					wLE.screenStartPt.v = currentMapDrawingRect.top + round(displayHeight*(j+1)/(float)(numLEsVert+1));
					index = (i*numLEsHori) + j;
					//wp = ScreenToWorldPoint(wLE.screenStartPt,currentMapDrawingRect,currentView);
					wp.p = ScreenToWorldPointRound(wLE.screenStartPt,currentMapDrawingRect,currentView); // temp fix for movement grid
					wLE.draw = GetPositionAfterNextStep(wp,currentTimeStep,&movedPoint);
					//wLE.screenEndPt  = WorldToScreenPoint(movedPoint,currentView,currentMapDrawingRect);
					wLE.screenEndPt  = WorldToScreenPointRound(movedPoint.p,currentView,currentMapDrawingRect); // temp fix for movement grid
					//wLE.screenStartPt  = WorldToScreenPoint(wp,currentView,currentMapDrawingRect); // temp fix for movement grid
					INDEXH(sMovementLEHdl,index) = wLE;
				}
			}
		}
	}

	
	/////////////////////////////////////////////////
	// draw the LE movement vectors
	/////////////////////////////////////////////////
	
	GetForeColor(&saveColor);
	
	RGBForeColor(&colors[DARKBLUE]);

	if(errorMessageStr)
	{ 	
		// 
		short w = stringwidth(errorMessageStr);
		short h = (currentMapDrawingRect.left + currentMapDrawingRect.right -w)/2;
		short v = (currentMapDrawingRect.top + currentMapDrawingRect.bottom)/2;
		MyMoveTo(h,v);
		drawstring(errorMessageStr);
	}
	else if(sMovementLEHdl)
	{
		for (i = 0; i < numLEsHori; i++)
		{
			for (j = 0; j < numLEsVert; j++)
			{
				index = (i*numLEsHori) + j;
				wLE = INDEXH(sMovementLEHdl,index);
				if(wLE.draw)
				{
					MyDrawArrow(wLE.screenStartPt.h,wLE.screenStartPt.v, 
									wLE.screenEndPt.h,wLE.screenEndPt.v);
				}
			}
		}
	}
	RGBForeColor(&saveColor);

}



TWindMover* TModel::GetNthWindMover(long desiredNum0Relative)
{
	long i,n,k,d;
	TMap *map;
	//TWindMover *mover = nil;
	TMover *thisMover = 0;
	char thisName[kMaxNameLen];
	long numWindMovers = 0;
	ClassID classID;
	//mover = (TWindMover*)this->GetMoverExact(TYPE_WINDMOVER);

	// universal movers
	for (k = 0, d = this->uMap->moverList->GetItemCount (); k < d; k++)
	{
		this->uMap->moverList -> GetListItem ((Ptr) &thisMover, k);
		classID = thisMover -> GetClassID ();
		if(classID == TYPE_WINDMOVER) /*return thisMover;*/
		//thisMover -> GetClassName (thisName);
		//if(!strcmpnocase(thisName,"Constant Wind")) 
		{
			if(desiredNum0Relative == numWindMovers)
				return dynamic_cast<TWindMover *>(thisMover);
			numWindMovers++;
		}
		/*if(!strcmpnocase(thisName,"Variable Wind")) 
		{
			if(desiredNum0Relative == numWindMovers)
				return dynamic_cast<TWindMover *>(thisMover);
			numWindMovers++;
		}*/
	}
	
	// movers that belong to a map
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&map, i);
		for (k = 0, d = map -> moverList -> GetItemCount (); k < d; k++)
		{
			map -> moverList -> GetListItem ((Ptr) &thisMover, k);
			classID = thisMover -> GetClassID ();
			if(classID == TYPE_WINDMOVER) /*return thisMover;*/
			//thisMover -> GetClassName (thisName);
			//if(!strcmpnocase(thisName,"Constant Wind")) 
			{
				if(desiredNum0Relative == numWindMovers)
					return dynamic_cast<TWindMover *>(thisMover);
				numWindMovers++;
			}
			/*if(!strcmpnocase(thisName,"Variable Wind")) 
			{
				if(desiredNum0Relative == numWindMovers)
					return dynamic_cast<TWindMover *>(thisMover);
				numWindMovers++;
			}*/
		}
	}
	return nil;
}

void TModel::DrawLegends(Rect r, WorldRect wRect) 
{
	long i,n, numWindMovers;
	TMap *map;
	TWindMover *windMover = 0;
	float arrowDepth;
	// if dispersed draw all spills together
	if (gDispersedOilVersion && ThereIsA3DMover(&arrowDepth))
	{	// draw contour scale here so it's on top of everything
		PtCurMap *map = GetPtCurMap();	// still could be 2D...
		#ifdef MAC
		Boolean showDepthContourLegend = ItemChecked(ANALYSISMENU,SHOWCONTOURLABELSITEM-ANALYSISMENU);
		#else
		Boolean showDepthContourLegend = ItemChecked2(GetSubMenu(GetMenu(mapWindow),4),SHOWCONTOURLABELSITEM-ANALYSISMENU);
		#endif
		if (map)
		{
			if (map->ThereIsADispersedSpill()) map->DrawContours(r,wRect);	// ??
			if (map->bShowLegend /*&& map->ThereIsADispersedSpill()*/) map->DrawContourScale(r,wRect);	
			if (showDepthContourLegend) map->DrawDepthContourScale(r,wRect);
		}
		// show legend for depth contours
	}
	/*for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&map, i);
		if (map->IAm(TYPE_VECTORMAP))
		{	// code goes here, get rid of ESI stuff?
			if (((TVectorMap*)map)->bShowLegend && ((TVectorMap*)map)->HaveESIMapLayer()) {map->DrawESILegend(r,wRect); break;}
		}
	}*/
	
	numWindMovers = GetNumWindMovers();
	if (numWindMovers>0)
	{
	//windMover = GetWindMover(false);	// code goes here, should check all winds, use GetMover() code explicitly, how many wind barbs would we want to draw?
		for (i=0;i<numWindMovers;i++)
		{
			windMover = GetNthWindMover(i);
		// code goes here, if ComponentMover, averaged winds option might want to draw averaged winds barb
			if (windMover && windMover->bActive && windMover->bShowWindBarb) windMover->DrawWindVector(r, wRect);
		}
	}
	//windMover = (TWindMover*)this->GetMoverExact(TYPE_WINDMOVER);	// want to draw more than one...
	//if (windMover && windMover->bActive && windMover->bShowWindBarb) windMover->DrawWindVector(r, wRect);
	return;
}

Boolean gHaveCollectedDirt  = false;
void TellPlayersAboutNewDirt(void)
{
	OSErr err = 0;
	if(gHaveCollectedDirt && model)
		err = model->BroadcastMessage(M_UPDATEVALUES, "*","*",0);	
	gHaveCollectedDirt = false;
}

void TellPlayersAboutNewListSelection(void)
{
	static long sPrevN = -2;
	static char sPrevS[256] = "";
	TClassID*  sPrevOwner = 0;
	OSErr err = 0;
	
	if(model) {
		long n = -3;
		short style = normal;
		char s[256] = "";
		Boolean itemSelected;
		ListItem localItem;
	
		memset(&localItem,0,sizeof(localItem));
		itemSelected = VLGetSelect(&n, &objects);
		if (itemSelected) 
			localItem = model->GetNthListItem(n, 0, &style, s);
	
		if(sPrevN == n 
			&& sPrevOwner == localItem.owner
			&& !strcmp(sPrevS,s))
			return; // same item and owner
			

			
		strcpy(sPrevS,s);
		sPrevOwner = localItem.owner;
		sPrevN = n;
		
		model -> CheckEditModeChange();
		err = model->BroadcastMessage(M_UPDATEVALUES, "*","*",0);
		// I think we should just send M_UPDATEVALUES
		// rather than have a separate message
	}
}


static long sSuppressDirtFlags = 0;
void TModel::SuppressDirt (long suppressDirtFlags)
{
	sSuppressDirtFlags = suppressDirtFlags;
}



void TModel::NewDirtNotification (void)
{ // calling this function informs the model that something is dirty
	long flags = DIRTY_EVERYTHING;
	
	if(this->GetModelMode() == ADVANCEDMODE) 
	{
		/// advanced users are responsible for reseting the runbar
		flags = flags & ~(DIRTY_RUNBAR); // note ~ is equivalent to BitNot on the mac
	}
		
	this->NewDirtNotification(flags);
}

/////////////////////////////////////////////////
void TModel::NewDirtNotification (long flags)
{ // calling this function informs the model that something is dirty
	
	if(sSuppressDirtFlags)
	{
		flags = flags & ~(sSuppressDirtFlags); // note ~ is equivalent to BitNot on the mac
	}
	
	// do we need to distinquish between things that affect the model run
	// and the resetting of the run bar ??
	if(flags & DIRTY_RUNBAR)
	{
		// reset the run bar
		this->Reset();
	}
	
	// cause an update in case
	//  it affects the screen
	if(flags & DIRTY_LIST)
		InvalListLength();
		
	if(flags & DIRTY_MAPDRAWINGRECT)
	{
		InvalMapDrawingRect();
		InvalidateMapImage();
	}
	
	if(flags & DIRTY_TOOLBAR)
	{
		InvalToolBarRect();
	}

	// actually, the tool bar can change based on a change in mode, 
	// so we need to redraw everything
	if(flags == DIRTY_EVERYTHING || flags & DIRTY_ENTIREWINDOW)
		InvalMapWindow();
	
	// set flag to tell players about now info
	gHaveCollectedDirt = true;
	
	// code goes here
	// set the dirty flag for the saveFile
	
}
/////////////////////////////////////////////////

Boolean TModel::IsDirty ()
{
	long		i, j, n;
	TMover		*thisMover;
	TMap		*thisMap;
	LERec		thisLE;
	Boolean		bIsDirty = false;
	
	if (bLEsDirty) bIsDirty = true;

	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++)
	{
		mapList->GetListItem((Ptr)&thisMap, i);
		if (thisMap->IsDirty())
		{
			bIsDirty = true;
			break;
		}
	}

	if (!bIsDirty)
	{			
		short	wCount, wIndex;
		Boolean	bWeathered;
	
		for (wIndex = 0, wCount = weatherList -> GetItemCount (); wIndex < wCount; wIndex++)
		{
			TWeatherer	*thisWeatherer;

			weatherList -> GetListItem ((Ptr) &thisWeatherer, wIndex);
			if (thisWeatherer -> IsDirty ())
			{
				bIsDirty = true;
				break;
			}
		}
	}
	
	return bIsDirty;
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////

static PopInfoRec ruPopTable[] = {
		{ RUNUNTIL, nil, RU_MONTH,   0, pMONTHS,       0, 1, FALSE, nil },
		{ RUNUNTIL, nil, RU_YEAR,    0,  pYEARS,       0, 1, FALSE, nil }
	};

OSErr RUInit(DialogPtr dialog, VOIDPTR data)
{
	DateTimeRec	time;
	Seconds guess;
	char 	timeS[256];
	char* p;
	
	if(UseExtendedYears())
		ruPopTable[1].menuID = pYEARS_EXTENDED;
	else
		ruPopTable[1].menuID = pYEARS;

	RegisterPopTable (ruPopTable, sizeof (ruPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (RUNUNTIL, dialog);

	SetDialogItemHandle(dialog, RU_HILITEDEFAULT, (Handle) FrameDefault);

	SecondsToDate (model -> GetModelTime (), &time);
	Date2String(&time, timeS);
	if (p = strrchr(timeS, ':')) p[0] = 0; // remove seconds
	mysetitext(dialog, RU_CURRENTMODELTIME, timeS);
	
	SecondsToDate (model -> GetEndTime (), &time);
	Date2String(&time, timeS);
	if (p = strrchr(timeS, ':')) p[0] = 0; // remove seconds
	mysetitext(dialog, RU_MODELENDTIME, timeS);
	
	guess = model -> GetModelTime () +3600; // one hr later
	if(guess > model -> GetEndTime ()) guess = model -> GetEndTime ();
	SecondsToDate (guess, &time); 
	SetPopSelection (dialog, RU_MONTH, time.month);
	SetPopSelection (dialog, RU_YEAR,  time.year - (FirstYearInPopup()  - 1));
	Long2EditText (dialog, RU_DAY, time.day);
	Long2EditText (dialog, RU_HOURS, time.hour);
	/////////////////////////////////////////////////
	// present the minutes with a leading zero if 9 or less, 1/11/99
	if(0 <= time.minute && time.minute <= 9)
	{
		strcpy(timeS,"00");
		timeS[1] = '0'+time.minute;
		mysetitext(dialog, RU_MINUTES, timeS);
	}
	else Long2EditText (dialog, RU_MINUTES, time.minute);
	/////////////////////////////////////////////////
	
	MySelectDialogItemText(dialog, RU_DAY, 0, 255);

	return 0;
}

short RUClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	Seconds	theTime;
	long	menuID_menuItem;
	OSErr err = 0;

	switch (itemNum) {
		
		case RU_OK:
			theTime = RetrievePopTime (dialog, RU_MONTH,&err);
			if(err) break;
			// JLM note: month, day, year  hr month must be sequential from RU_MONTH
			*(Seconds*)data = theTime; // return value through data
			return RU_OK;
		
		case RU_CANCEL: return RU_CANCEL;
		
		case RU_MONTH:
		case RU_YEAR:
			PopClick(dialog, itemNum, &menuID_menuItem);
		break;
		
		case RU_DAY:
		case RU_HOURS:
		case RU_MINUTES:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;
	}

	return 0;
}




/////////////////////////////////////////////////

OSErr TModel::RunTill(void)
{

	// ask the user to specify a time, and then run until that time
	Seconds stopTime;
	short item;
	Seconds untilTime = 0;

	item = MyModalDialog(RUNUNTIL, mapWindow, &stopTime, RUInit, RUClick);
	if(item != RU_OK) return 0;
	
	// draw the list and tool bar before we run
	// since the dialog covered it up
	UpdateMapWindow();
	RunTill (stopTime); 

	return 0;
}

OSErr TModel::RunTill (Seconds stopTime)
{	// code goes here, update for hindcasting
	Seconds	actualTime;

	if(stopTime > this->GetEndTime())
		stopTime = this->GetEndTime();
		
	if(stopTime < this->GetStartTime())
		stopTime = this->GetStartTime();	// STH
		
	return this->Run(stopTime);

}

OSErr TModel::Run (Seconds stopTime)
{
	Rect saveClip;
	GrafPtr savePort;
	OSErr err = 0;

	this->currentStep = 0;
	//this->stepsCount = ceil((float)fDialogVariables.duration / fDialogVariables.computeTimeStep);
	this->outputStepsCount = ceil((float)fDialogVariables.duration / this->GetOutputStep());
	//this->stepsCount++;
	this->outputStepsCount++;
	//if(bHindcast || this->modelTime > this->GetStartTime())
		//this->writeNC = false;
	

#ifndef NO_GUI
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	saveClip = MyClipRect(MapDrawingRect());
#endif
	fRunning = TRUE;
	//if (!gSuppressDrawing)
#ifndef NO_GUI
	DrawTools(PLAYERPICTBASE, PLAYBUTTON);// JLM 3/11/99 make it appear as a pause button right away
#endif
	//while (modelTime <= stopTime) {
	// JLM 2/8/99
	//while (modelTime < stopTime) {
	//while (modelTime > stopTime) {
	while ((!bHindcast && modelTime < stopTime) || (bHindcast && modelTime > stopTime)) {
		if (CmdPeriod()) 
		{ 
			err = -7; // special TAP error to allow to break out of all runs with a command period 4/1/03
			break; 
		}
		if (ClickedPause()) break;

#ifdef IBM
		MSG msg;
		//if (settings.inBackground && PeekMessage(&msg, NULL, WM_MOUSEFIRST, WM_MOUSELAST, PM_REMOVE))
		if (settings.inBackground && PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{	// allow user to get focus back in case they want to stop the run, 12/10/04 
			//TranslateMessage (&msg);	// these calls don't seem necessary
			//DispatchMessage (&msg);
			VLUpdate(&objects);
			SendMessage(toolWnd, WM_PAINT, 0, 0);
		}
		if (!settings.inBackground)
		{
			//if (PeekMessage(&msg, NULL, WM_MOUSEFIRST, WM_MOUSELAST, PM_REMOVE))
			if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
			{	// allow resizing during a run, 12/10/04 
				TranslateMessage (&msg);	
				DispatchMessage (&msg);
			}
		}
#endif

		if (!bHindcast)
			{if (err = Step()) break;}
		else
			{if (err = StepBackwards()) break;}
		//if (err) break;
	}
	fRunning = FALSE;
	
	if(this->writeNC) {
		err = NetCDFStore::fClose(this->ncID);
		if(this->IsUncertain())
			err = NetCDFStore::fClose(this->ncID_C);
	}
	
	this->writeNC = false;
	
	
	if (err==-3) return err;	// hard exit on IBM
#ifndef NO_GUI
	MyClipRect(saveClip);
	SetPortGrafPort(savePort);

	//if (!gSuppressDrawing)
//#ifndef NO_GUI
	DisplayCurrentTime(true);
#endif

	return err;
}

OSErr TModel::StepBack()
{
	Rect		saveClip;
	Seconds		oldTime = modelTime;
	GrafPtr		savePort;
	OSErr		err = noErr;
	
	//if(this->modelTime >= this->GetEndTime()) return noErr; // JLM 2/8/99
	if (this->modelTime==this->lastComputeTime) 
	{	
		SetModelToPastTime(modelTime-LEDumpInterval); 			
		this->NewDirtNotification(DIRTY_LIST);
		return err;
	}
	// JLM 1/6/99
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	saveClip = MyClipRect(MapDrawingRect());
	SetWatchCursor();
	
	if(this->modelTime - this->LEDumpInterval < this->lastComputeTime )
	{
		// if the run time is less than the lastComputeTime,  
		// then the user has stepped back in time
		// Stepping should get the next file we have saved
		Seconds nextFileSeconds = this->PreviousSavedModelLEsTime(this->modelTime);
		Seconds actualTime;
		// I don't think this is ever the case when going backwards
		/*if(nextFileSeconds >= this->modelTime // there are no more saved LE times in files
			|| nextFileSeconds == this->lastComputeTime) // the next saved time is the lastCompute time
		{
			if(modelTime == fDialogVariables.startTime)
			{	// write out first time files
				err = this->SaveOutputSeriesFiles(modelTime,true);
				if(err) goto ResetPort; 
			}
			////////
			CopyModelLEsToggles(this->fSquirreledLastComputeTimeLEList,this->LESetsList);
			err = ReinstateLastComputedTimeStuff();
			CopyModelLEsToggles(this->LESetsList,this->fSquirreledLastComputeTimeLEList);
			this->NewDirtNotification(DIRTY_LIST);
			DrawMaps (mapWindow, MapDrawingRect(), settings.currentView, FALSE);
			//	
			err = this->SaveOutputSeriesFiles(oldTime,true);
			if(err) goto ResetPort; 
			///////
			goto ResetPort;
		}*/
		//else if(nextFileSeconds < this->modelTime) 
		if(nextFileSeconds < this->modelTime) 
		{	// there is a save file we can load
			// I don't think this case will ever happen going backwards
			/*if(modelTime == fDialogVariables.startTime)
			{	// write out first time files
				err = this->SaveOutputSeriesFiles(modelTime,true);
				if(err) goto ResetPort; 
			}*/
			////////
			this->SuppressDirt(DIRTY_EVERYTHING);
			CopyModelLEsToggles(this->fSquirreledLastComputeTimeLEList,this->LESetsList);
			err = this->LoadModelLEs (nextFileSeconds, &actualTime);
			CopyModelLEsToggles(this->LESetsList,this->fSquirreledLastComputeTimeLEList);
			this->SuppressDirt(0);
			this->NewDirtNotification(DIRTY_LIST);
			DrawMaps (mapWindow, MapDrawingRect(), settings.currentView, FALSE);
			//	probably don't want to save anything during backwards step
			//err = this->SaveOutputSeriesFiles(oldTime,true);
			//if(err) goto ResetPort; 
			////////

			//goto ResetPort;
		}
		/*else
		{
			// there is no savedFile greater than 
			// modelTime.
			// This means we have to compute our way from the 
			// current model time
			this->SetLastComputeTime(this->modelTime);
		}*/
	}
	/////////////////////////////////////////////////
//ResetPort:
	DisplayCurrentTime(true);
	MyClipRect(saveClip);
	SetPortGrafPort(savePort);
	
	return err;
}

OSErr TModel::Reset()
{
	long i, n;
	TLEList *thisLEList;
	OSErr err = 0;
	
	if (OptionKeyDown()) return StepBack();	// allow to go backward through saved LE files - what about for hindcast??
	if (bHindcast)
		modelTime = fDialogVariables.startTime + fDialogVariables.duration;
	else
		modelTime = fDialogVariables.startTime;
	this->SetLastComputeTime(modelTime);
	
	for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) {
		LESetsList->GetListItem((Ptr)&thisLEList, i);
		if (err = thisLEList->Reset(FALSE)) return err;
	}
		
	DisposeLEFrames ();			// STH
	
	//if (!gSuppressDrawing)
#ifndef NO_GUI
	DisplayCurrentTime(true);
	InvalMapWindow();
#endif
	
	return 0;
}


#define TModelREADWRITEVERSION  5	// added bHindcast
//#define TModelREADWRITEVERSION  4	// updated as a flag for merged version and added gDispersedOilVersion
//#define TModelREADWRITEVERSION  3	// matched GNOME_beta from 9/10/03 to save view
//#define TModelREADWRITEVERSION  2

OSErr TModel::Write(BFPB *bfpb)
{
	char 		c;
	long 		i, n, version = TModelREADWRITEVERSION;
	ClassID 	id = GetClassID ();
	TLEList		*thisLEList;
	TMap		*thisMap;
	TWeatherer	*thisWeatherer;
	OSErr err = 0;
	
	StartReadWriteSequence("TModel::Write()");
	if (err = WriteMacValue(bfpb,id)) return err;
	if (err = WriteMacValue(bfpb,version)) return err;
	if (err = WriteMacValue(bfpb,nextKeyValue)) return err;
	
	// we don't need to write out weatherer

	// we have to write this individually to be cross platform
	if (err = WriteMacValue(bfpb,fDialogVariables.startTime)) return err;
	if (err = WriteMacValue(bfpb,fDialogVariables.duration)) return err;
	if (err = WriteMacValue(bfpb,fDialogVariables.computeTimeStep)) return err;
	if (err = WriteMacValue(bfpb,fOutputTimeStep)) return err;
	if (err = WriteMacValue(bfpb, fDialogVariables.bUncertain)) return err;
	if (err = WriteMacValue(bfpb, fWantOutput)) return err;
	if (err = WriteMacValue(bfpb,fDialogVariables.preventLandJumping)) return err;
	if (err = WriteMacValue(bfpb,modelTime)) return err;
	if (err = WriteMacValue(bfpb,lastComputeTime)) return err;

	// save output file name including path to save file
	if (err = WriteMacValue (bfpb, fOutputFileName, sizeof(fOutputFileName))) return err;

//	n = LESetsList->GetItemCount();		// LE Sets count
//	if (err = WriteMacValue(bfpb,n)) return err;	// now done in SaveModelLEs (), STH

	n = mapList->GetItemCount();		// map count
	if (err = WriteMacValue(bfpb,n)) return err;
	n = weatherList->GetItemCount();	// weathering count
	if (err = WriteMacValue(bfpb,n)) return err;
	if (err = WriteMacValue(bfpb, fSettingsOpen)) return err;
	if (err = WriteMacValue(bfpb, fSpillsOpen)) return err;
	if (err = WriteMacValue(bfpb, bMassBalanceTotalsOpen)) return err;
	if (err = WriteMacValue(bfpb,mapsOpen)) return err;
	if (err = WriteMacValue(bfpb, uMoverOpen)) return err;
	if (err = WriteMacValue(bfpb, weatheringOpen)) return err;
	if (err = WriteMacValue(bfpb, fDrawMovement)) return err;

	SaveModelLEs (bfpb);	// write all model LE sets to file, STH
	
	// write the universal map, JLM  7/31/98 
	id = uMap->GetClassID();
	if (err = WriteMacValue(bfpb,id)) return err;
	if (err = uMap->Write(bfpb)) return err;
			
	// write each of the maps

	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&thisMap, i);
		id = thisMap->GetClassID();
		if (err = WriteMacValue(bfpb,id)) return err;
		if (err = thisMap->Write(bfpb)) return err;
	}

	// write each of the weatherers

	for (i = 0, n = weatherList->GetItemCount() ; i < n ; i++) {
		weatherList->GetListItem((Ptr)&thisWeatherer, i);
		id = thisWeatherer->GetClassID();
		if (err = WriteMacValue(bfpb,id)) return err;
		if (err = thisWeatherer->Write(bfpb)) return err;
	}

if (err = WriteMacValue(bfpb,settings.currentView)) return err;
#ifdef MAC
	//if (err = WriteMacValue(bfpb,mapWindow->portRect)) return err;
	if (err = WriteMacValue(bfpb,GetWindowPortRect(mapWindow))) return err;
#else
{
	RECT r;
	Rect r2;
	GetWindowRect(mapWindow, &r);
	MakeMacRect(&r,&r2);
	if (err = WriteMacValue(bfpb,r2)) return err;
}
#endif	
	SetDirty(FALSE);

	if (err = WriteMacValue(bfpb,gDispersedOilVersion)) return err;	// should this be a setting?
	if (err = WriteMacValue(bfpb,bHindcast)) return err;	// should this be a setting?

	return 0;
}


OSErr TModel::Read(BFPB *bfpb)
{
	char 		c;
	long 		i, n, version, numLESets, numMaps, numWeathering;
	ClassID 	id;
	TLEList		*thisLEList;
	TMap		*thisMap;
	TWeatherer	*thisWeatherer;
	OSErr err = 0;
	Boolean bSaveFileInDispersedOilMode = false;
	
	StartReadWriteSequence("TModel::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TModel::Read()", "id == TYPE_MODEL", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > TModelREADWRITEVERSION || version < 1) { printSaveFileVersionError(); return -1; }
	//if (version != TModelREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	if (err = ReadMacValue(bfpb,&nextKeyValue)) return err;
	
	// we don't need to read weatherer

	// we have to read this individually to be cross platform
	//if (err = ReadMacValue(bfpb,&fDialogVariables, sizeof(fDialogVariables), TRUE)) return err;
	if (err = ReadMacValue(bfpb,&fDialogVariables.startTime)) return err;
	if (err = ReadMacValue(bfpb,&fDialogVariables.duration)) return err;
	if (err = ReadMacValue(bfpb,&fDialogVariables.computeTimeStep)) return err;
	if (err = ReadMacValue(bfpb,&fOutputTimeStep)) return err;
	if (err = ReadMacValue(bfpb, &fDialogVariables.bUncertain)) return err;
	if (err = ReadMacValue(bfpb, &fWantOutput)) return err;
	if (err = ReadMacValue(bfpb,&fDialogVariables.preventLandJumping)) return err;
	if (err = ReadMacValue(bfpb,&modelTime)) return err;
	if (err = ReadMacValue(bfpb,&lastComputeTime)) return err;

	// save output file name including path to save file
	if (err = ReadMacValue (bfpb, fOutputFileName, sizeof(fOutputFileName))) return err;
	/////////////////////////////////////////////////
	if(fOutputFileName[0])
	{ 	// if the path to the directory is no longer valid, blank it out
		// this also fixes a bug where the save files had bad paths
		// JLM 3/12/99
		char directoryPath[256];
		char shortFileName[256];
		strcpy(directoryPath,fOutputFileName);
		SplitPathFile(directoryPath,shortFileName);
		if(!directoryPath[0] || !FolderExists(0,0,directoryPath))
		{
			fOutputFileName[0] = 0;
		}
	}
	/////////////////////////////////////////////////


//	if (err = ReadMacValue(bfpb,&numLESets)) return err;	// now done in LoadModelLEs, STH
	if (err = ReadMacValue(bfpb,&numMaps)) return err;
	if (err = ReadMacValue(bfpb,&numWeathering)) return err;
	if (err = ReadMacValue(bfpb, &fSettingsOpen)) return err;
	if (err = ReadMacValue(bfpb, &fSpillsOpen)) return err;
	if (err = ReadMacValue(bfpb, &bMassBalanceTotalsOpen)) return err;
	if (err = ReadMacValue(bfpb, &mapsOpen)) return err;
	if (err = ReadMacValue(bfpb, &uMoverOpen)) return err;
	if (err = ReadMacValue(bfpb, &weatheringOpen)) return err;
	if (err = ReadMacValue(bfpb, &fDrawMovement)) return err;
	
	// allocate and read each of the LE sets
	if (err = LoadModelLEs (bfpb)) {printError("Error while reading LE's from file."); return -1; }
	
	// read the universal map, JLM 7/31/98 
	if (err = ReadMacValue(bfpb,&id)) return err;
	if(id !=  uMap->GetClassID()) {printError("Universal map type does not match."); return -1; }
	if (err = uMap->Read(bfpb)) return err;
		
	// allocate and read each of the maps

	for (i = 0 ; i < numMaps ; i++) {
		if (err = ReadMacValue(bfpb,&id)) return err;
		switch (id) {
			case TYPE_MAP: thisMap = new TMap("", voidWorldRect); break;
			case TYPE_OSSMMAP: thisMap = (TMap *)new TOSSMMap("", voidWorldRect); break;
			case TYPE_VECTORMAP: thisMap = (TMap *)new TVectorMap ("", voidWorldRect); break;
			case TYPE_PTCURMAP: thisMap = (TMap *)new PtCurMap ("", voidWorldRect); break;
			case TYPE_COMPOUNDMAP: thisMap = (TMap *)new TCompoundMap ("", voidWorldRect); break;
			case TYPE_MAP3D: thisMap = (TMap *)new Map3D ("", voidWorldRect); break;
			
			default: printError("Unrecognized map type in TModel::Read()."); return -1;
		}
		if (!thisMap) { TechError("TModel::Read()", "new TMap", 0); return -1; };
		if (err = thisMap->InitMap()) return err;
		if (err = thisMap->Read(bfpb)) return err;
		if (err = AddMap(thisMap, 0))
			{ delete thisMap; TechError("TModel::Read()", "AddMap()", err); return err; };
	}
	
	// allocate and read each of the weatherers
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	// JLM 9/9/99
	// we have to be careful that we don't wind up with two standard weathers
	// for now, it is probably OK to delete the standard one added in 
	// and just have the weathers that were saved in the file
	if (weatherList) {
		while (weatherList -> GetItemCount() > 0) {
			// get the bottom-most LEList and drop & dispose of it
			weatherList->GetListItem((Ptr)&thisWeatherer, 0);
			if (err = weatherList->DeleteItem(0)) { TechError("TModel::Read()", "DeleteItem()", err); return err; }
			if(thisWeatherer) {thisWeatherer -> Dispose ();delete (thisWeatherer); thisWeatherer = 0;}
		}
	}
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	

	for (i = 0 ; i < numWeathering ; i++) {
		if (err = ReadMacValue(bfpb,&id)) return err;
		switch (id) {
			case TYPE_WEATHERER: thisWeatherer = new TWeatherer(""); break;
			case TYPE_OSSMWEATHERER: thisWeatherer = (TOSSMWeatherer *)new TOSSMWeatherer(""); break;
			default: printError("Unrecognized weatherer type in TModel::Read()."); return -1;
		}
		if (!thisWeatherer) { TechError("TModel::Read()", "new TWeatherer", 0); return -1; };
		if (err = thisWeatherer->InitWeatherer()) return err;
		if (err = thisWeatherer->Read(bfpb)) return err;
		if (err = AddWeatherer(thisWeatherer, 0))
			{ delete thisWeatherer; TechError("TModel::Read()", "AddWeatherer()", err); return err; };
	}
	
	if (version > 2)
	{
		WorldRect viewRect;
		Rect savedWindowRect;
		Rect mdr = MapDrawingRect(), r2;
		// something is getting messed up with the view if transfer from bigger to smaller screen
		Rect screenRect = FullScreenMapWindowRect();
		{if (err = ReadMacValue(bfpb,&viewRect)) return err;}
		{if (err = ReadMacValue(bfpb,&savedWindowRect)) return err;}
		if (screenRect.left+screenRect.right < savedWindowRect.right - savedWindowRect.left) savedWindowRect.right = screenRect.left+screenRect.right+savedWindowRect.left;
		if (savedWindowRect.top > screenRect.top / 2 ) {savedWindowRect.top = 0; savedWindowRect.bottom = screenRect.bottom - screenRect.top;}
		// code goes here, some sort of check that saved window/view will fit on screen
#ifdef IBM
		{
			RECT r;
			MakeWindowsRect(&savedWindowRect,&r);
			SetWindowPos(mapWindow,HWND_TOP,r.left, r.top, r.right - r.left, r.bottom - r.top,0);
			ChangeCurrentView(viewRect,FALSE,FV_SAMESCALE);
			//MoveWindow(mapWindow, r.left, r.top, r.right - r.left, r.bottom - r.top, FALSE);
		}
#else
		{
			settings.currentView = viewRect;
			if (savedWindowRect.top != 0)	// what if one is zero ?
			{	// to fix files saved on windows brought over to Mac, seems to work ok the other way
				savedWindowRect.bottom = savedWindowRect.bottom - savedWindowRect.top;
				savedWindowRect.top = 0;
			}
			if (savedWindowRect.left != 0)
			{
				savedWindowRect.right = savedWindowRect.right - savedWindowRect.left;
				savedWindowRect.left = 0;
			}
#if TARGET_API_MAC_CARBON	
#ifndef NO_GUI
			SizeWindow(mapWindow,RectWidth(savedWindowRect),RectHeight(savedWindowRect) + TOOLBARHEIGHT, true);
#endif
#else
			mapWindow->portRect = savedWindowRect;	
#endif
#ifndef NO_GUI
			ResizeChildWindows();
			InvalMapWindowBorders();
			UpdateMapWindow();
#endif
		}
#endif
#ifndef NO_GUI
		MyShowWindow(mapWindow);
#endif

	}
	if (version > 3) 
	{
		if (err = ReadMacValue(bfpb,&bSaveFileInDispersedOilMode)) return err;	// need to up the version...
	}
		if (gDispersedOilVersion && !bSaveFileInDispersedOilMode && !gMearnsVersion) DisposeAnalysisMenu();
		if (bSaveFileInDispersedOilMode) InitAnalysisMenu();

	if (version > 4) 
	{
		if (err = ReadMacValue(bfpb,&bHindcast)) return err;	// need to up the version...
	}

	bLEsDirty = false;		// just read in

	return 0;
}

OSErr TModel::ReadFromPath(char *path)
{
	BFPB bfpb;
	OSErr err = 0;
	
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("ReadFromPath()", "FSOpenBuf()", err); return err; }
	err = Read(&bfpb);
	FSCloseBuf(&bfpb);
	
	return err;
}

OSErr TModel::WriteToPath(char *path)
{
	BFPB bfpb;
	OSErr err = 0;
	
	hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, APPL_SIG, 'SAVE'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	err = Write(&bfpb);
	FSCloseBuf(&bfpb);
	
	return err;
}

//#ifdef IBM //  for debra, hide movement grid label
//#define HIDEMOVEMENTOPTION  
//#endif
long NumOfModelSubitems(long theMode)
{
	long count = 0;
	count++;// model start time
	count++;// model duration
	count+=2;// uncertainty lines
	count++;// draw movement grid
	
	if (settings.daylightSavingsTimeFlag == DAYLIGHTSAVINGSOFF) count++;
	
	//if(theMode >= INTERMEDIATEMODE) 
	//{
		//count++;// output time step
	//}
	
	if(theMode >= ADVANCEDMODE) 
	{
		count++;// computational time step
		count++;// PreventLandJumping
		count++;// Hindcast

	//	#ifndef HIDEMOVEMENTOPTION
			//count++;// draw movement grid
		//#endif
	}
	return count;
}
 
char* TModel::GetModelModeStr(char *str)
{
	switch(GetModelMode())
	{
		case NOVICEMODE:
			strcpy(str,"Standard");
			break;
		case INTERMEDIATEMODE:
			//strcpy(str,"GIS Output");
			strcpy(str,"Standard");
			break;
		case ADVANCEDMODE:
			strcpy(str,"Diagnostic");
			break;
		default:
			strcpy(str,"Unknown Mode");
		}
		return str;
}

long TModel::GetNumForecastSpills()
{
	long i ,numLeSets,numForecastSpills=0;
	TLEList *thisLEList;
	
	for(i = 0,numLeSets = LESetsList->GetItemCount(); i < numLeSets ; i++)
	{
		LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if(thisLEList -> GetLEType() != UNCERTAINTY_LE )
			numForecastSpills++;
	}
	return numForecastSpills;
}

long TModel::GetListLength()
{
	long i, n, count = 0;	// minimum # of list items
	TLEList	*thisLEList;
	TMap	*thisMap;
	TMover	*thisMover;
	long	theMode;

	theMode = GetModelMode ();
	
	count += 1;				// add 1 for Model Settings heading
	if (fSettingsOpen)
	{
		count += NumOfModelSubitems(theMode);
	}
	
/////////////////////////////////////////////////
// JLM add wizard 
	if(fWizard) count += fWizard->GetListLength();
/////////////////////////////////////////////////


	count += 1;				// add 1 for Wind/Universal Movers heading
	if (uMoverOpen)
	{
		if (theMode < ADVANCEDMODE)
		{
			TWindMover *wind = GetWindMover(false); // don't create
			if(wind) count += wind->GetListLength();
		}
		else
		{	//ADVANCEDMODE
			for (i = 0, n = uMap->moverList->GetItemCount(); i < n ; i++) {
				uMap->moverList -> GetListItem((Ptr)&thisMover, i);
				count += thisMover->GetListLength();
			}
		}
	}
	
	/////////////////////////////////////////////////
	if (theMode == ADVANCEDMODE)
	{	
		count += 1;				// add 1 for Maps heading
		if (mapsOpen)
		{
			for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
				mapList->GetListItem((Ptr)&thisMap, i);
				count += thisMap->GetListLength();
			}
		}
	}
	
	/////////////////////////////////////////////////
	count += 1;				// add 1 for Spills heading
	if (fSpillsOpen)
	{
		/////////////////////////////////////////////////
		short numLeSets,numForecastSpills = this->GetNumForecastSpills();
		if(numForecastSpills > 1)
		{
			count++;// toggle for total mass balance
			if(bMassBalanceTotalsOpen)
			{
				count++;// released 
				count++;// floating
				count++;// beached
				count++;// evaporated
				count++;// offmap
			}
		}
		/////////////////////////////////////////////////

		
		/// add lines for every spill
		for (i = 0,n = LESetsList->GetItemCount(); i < n ; i++) {
			LESetsList->GetListItem((Ptr)&thisLEList, i);
			count += thisLEList->GetListLength();
		}
	}

	/*if (theMode == ADVANCEDMODE)
	{
		count += 1;				// add 1 for Weathering heading
		if (weatheringOpen)
		{
			TWeatherer	*thisWeatherer;
			
			for (i = 0, n = weatherList -> GetItemCount (); i < n; i++)
			{
				weatherList->GetListItem((Ptr)&thisWeatherer, i);
				count += thisWeatherer -> GetListLength ();
			}
		}
	}*/

	/////////////////////////////////////////////////
	//if (theMode == ADVANCEDMODE)	// only for noaa.ver ? or only if overlay has been loaded ?
	if (theMode == ADVANCEDMODE && (fOverlayList->GetItemCount() > 0 || gNoaaVersion))	// only for noaa.ver ? or only if overlay has been loaded ?
	{	
		count += 1;				// add 1 for Overlays heading
		if (fOverlaysOpen)
		{
			TOverlay *thisOverlay;
			for (i = 0, n = fOverlayList->GetItemCount() ; i < n ; i++) {
				fOverlayList->GetListItem((Ptr)&thisOverlay, i);
				count += thisOverlay->GetListLength();
			}
		}
	}

	
	return count;
}

long TModel::GetLineIndex (long lineNum)
{
	switch (this -> GetModelMode ())
	{
		case NOVICEMODE:
		case INTERMEDIATEMODE:
			if (lineNum == 1) return I_STARTTIME;
			if (lineNum == 2) return I_ENDTIME;
			if (lineNum == 3) return I_UNCERTAIN;
			if (lineNum == 4) return I_UNCERTAIN2;
			if (lineNum == 5) return I_DRAWLEMOVEMENT;
			if (lineNum == 6) return I_DSTDISABLED;
		break;
		
		/*case INTERMEDIATEMODE:
			if (lineNum == 1) return I_STARTTIME;
			if (lineNum == 2) return I_ENDTIME;
			//if (lineNum == 3) return I_OUTPUTSTEP;
			if (lineNum == 3) return I_UNCERTAIN;
			if (lineNum == 4) return I_UNCERTAIN2;
			if (lineNum == 5) return I_DRAWLEMOVEMENT;
			if (lineNum == 6) return I_DSTDISABLED;
		break;*/
		
		case ADVANCEDMODE:
		default:
			if (lineNum == 1) return I_STARTTIME;
			if (lineNum == 2) return I_ENDTIME;
			if (lineNum == 3) return I_COMPUTESTEP;
			//if (lineNum == 4) return I_OUTPUTSTEP;
			if (lineNum == 4) return I_UNCERTAIN;
			if (lineNum == 5) return I_UNCERTAIN2;
			if (lineNum == 6) return I_DRAWLEMOVEMENT;
			if (lineNum == 7) return I_PREVENTLANDJUMPING;
			if (lineNum == 8) return I_HINDCAST;
			if (lineNum == 9) return I_DSTDISABLED;
		break;
	}
	
	return -1;
}

ListItem TModel::GetNthListItem(long n, short indent, short *style, char *text)
{
	char *p, s[128];
	long i, m, count, theMode;
	TLEList *thisLEList;
	CMyList *uMoverList;
	TMap *thisMap;
	TMover *thisMover;
	DateTimeRec time;
	ListItem item = { this, 0, indent, 0 };
	
	theMode = GetModelMode ();
	
	if (n == 0) {
		item.index = I_MODELSETTINGS;
		item.bullet = fSettingsOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		strcpy(text, "Model Settings");
		*style = bold;
		
		return item;
	}
	
	n -= 1;
	
	if (fSettingsOpen) {
		long	durationHours = 0;
		short 	numModelSubItems = NumOfModelSubitems(theMode);
		if (n >= 0 && n < numModelSubItems) 
		{	// one of our items
			long ourLineIndex = GetLineIndex (1 + n);//JLM
			item.index = ourLineIndex;
			
			switch (ourLineIndex) 
			{
				case I_STARTTIME:
					Secs2DateString2 (fDialogVariables.startTime, s);
					sprintf(text, "Start time: %s", s);
					break;
				case I_ENDTIME:
					durationHours = (fDialogVariables.duration) / 3600;
					sprintf(text, "Duration: %ld hours", durationHours);
					break;
				case I_COMPUTESTEP:
					sprintf(text, "Computational time step: %.2f hr", (float)(fDialogVariables.computeTimeStep / 3600.0));
					break;
//				case I_OUTPUTSTEP:
//					sprintf(text, "Output time step: %.2f hr", (float)(fOutputTimeStep / 3600.0));
//					if (!savePath[0])
//						strcat(text, " [No output until .SAV file is saved]");
//					break;
				case I_UNCERTAIN:
					item.indent++;
					item.bullet = IsUncertain() ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
					strcpy(text, "Include the Minimum Regret");
					break;
				case I_UNCERTAIN2:
					item.indent++;
					strcpy(text, "solution (RED SPLOTS on screen)");
					break;

				case I_DRAWLEMOVEMENT:
					item.indent++;
					item.bullet = this->fDrawMovement ?  BULLET_FILLEDBOX : BULLET_EMPTYBOX;
					//strcpy(text, "Show movement grid");
					strcpy(text, "Show Currents");
					break;

				case I_PREVENTLANDJUMPING:
					item.indent++;
					item.bullet = fDialogVariables.preventLandJumping ?  BULLET_FILLEDBOX : BULLET_EMPTYBOX;
					strcpy(text, "Prevent Land Jumping");
					break;
			
				case I_HINDCAST:
					item.indent++;
					item.bullet = this->bHindcast ?  BULLET_FILLEDBOX : BULLET_EMPTYBOX;
					//strcpy(text, "Show movement grid");
					strcpy(text, "Run Backwards");
					break;

				case I_DSTDISABLED:
					if (settings.daylightSavingsTimeFlag == DAYLIGHTSAVINGSOFF)
					{
					//item.indent++;
					//item.bullet = settings.preventLandJumping ?  BULLET_FILLEDBOX : BULLET_EMPTYBOX;
					strcpy(text, "Daylight Savings Time Disabled");
					}
					break;
			} // end switch
			
			return item;
		}
		n -= numModelSubItems;
	}
/////////////////////////////////////////////////
// JLM add wizard 
	if(fWizard) 
	{
		short numItems = fWizard->GetListLength();
		if(n < numItems) 
			return fWizard->GetNthListItem(n,indent,style,text);
		n -= numItems;
	}
/////////////////////////////////////////////////
		

	/////////////////////////////////////////////
	// Wind / universal movers
	////{
	if (n == 0) {
		item.bullet = uMoverOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		if (theMode < ADVANCEDMODE) 
		{
			strcpy(text, "Wind");
			item.index = I_WIND;
		}
		else 
		{
			strcpy(text, "Universal Movers");
			item.index = I_UMOVERS;
		}
		*style = bold;
		return item;
	}
	n -= 1; 
	
	if (uMoverOpen)
	{
		short indentOffset = (theMode < ADVANCEDMODE) ? 0:1;
		
		if (theMode < ADVANCEDMODE)
		{
			TWindMover*wind = GetWindMover(false); // don't create
			if(wind)
			{
				count = wind->GetListLength();
				if (count > n)
					return wind->GetNthListItem(n, indent + indentOffset, style, text);
				n -= count;
			}
		}
		else
		{	//ADVANCEDMODE
			uMoverList = uMap -> GetMoverList ();
			for (i = 0, m = uMoverList->GetItemCount() ; i < m ; i++) {
				uMoverList->GetListItem((Ptr)&thisMover, i);
				count = thisMover->GetListLength();
				if (count > n)
					return thisMover->GetNthListItem(n, indent + indentOffset, style, text);
				n -= count;
			}
		}
	}
	///}
	//////////////////////////////////////////////
	
	if (theMode == ADVANCEDMODE)
	{
		// check maps
	
		if (n == 0) {
			item.index = I_MAPS;
			item.bullet = mapsOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Maps");
			*style = bold;
	
			return item;
		}
		
		n -= 1; // it's not the Maps
		// is it in the open Maps guy	
		
		if (mapsOpen)
		{
			for (i = 0, m = mapList->GetItemCount() ; i < m ; i++) {
				mapList->GetListItem((Ptr)&thisMap, i);
				count = thisMap->GetListLength();
				if (count > n)
					return thisMap->GetNthListItem(n, indent + 1, style, text);
				
				n -= count;
			}
		}
	}

	if (n == 0) {
		item.index = I_LESETS;
		item.bullet = fSpillsOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		strcpy(text, "Spills");
		*style = bold;
		
		return item;
	}

	n -= 1;

	if (fSpillsOpen)
	{
		
		/////////////////////////////////////////////////
		short numLeSets,numForecastSpills = 0;
		short unitsForTotals = BARRELS; 
		for(numLeSets = LESetsList->GetItemCount(),i=numLeSets-1; i>=0 ; i--)
		{
			LESetsList -> GetListItem ((Ptr) &thisLEList, i);
			if(thisLEList -> GetLEType() != UNCERTAINTY_LE )
			{
				numForecastSpills++;
				unitsForTotals = thisLEList->GetMassUnits(); //return units for the first nonuncertaintly le list
			}
		}
		if(numForecastSpills > 1)
		{
			if (n == 0) {
				item.indent++;
				item.index = I_MASSBALANCETOTALS;
				item.bullet = bMassBalanceTotalsOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				strcpy(text, "Splot Mass Balance Totals (Best estimate)");
				return item;
			}
			n -= 1;
			
			if(bMassBalanceTotalsOpen)
			{
				double amtTotal,amtReleased,amtEvaporated,amtBeached,amtOffmap,amtFloating,amtDispersed=0,amtRemoved=0;
				short numDecPlaces;
				char ofTotalStr[64] = "",ofReleasedStr[64] = "";
				char infoStr [255];
				char unitsStr [64];
				
				GetTotalAmountStatistics(unitsForTotals,&amtTotal,&amtReleased,&amtEvaporated,&amtDispersed,&amtBeached,&amtOffmap,&amtFloating,&amtRemoved);
																																	
				GetLeUnitsStr(unitsStr,unitsForTotals);
				
				if(amtReleased < 100) numDecPlaces = 1;
				else numDecPlaces = 0;

				if (n == 0) {// released
					item.index = I_MASSBALANCELINE;
					StringWithoutTrailingZeros(infoStr,amtReleased,numDecPlaces); 
					sprintf(text, "Released: %s %s",infoStr,unitsStr);
					item.indent++;
					return item;
				}
				n -= 1;
			
				if (n == 0) {// floating
					item.index = I_MASSBALANCELINE;
					StringWithoutTrailingZeros(infoStr,amtFloating,numDecPlaces); 
					sprintf(text, "Floating: %s %s ",infoStr,unitsStr);
					item.indent++;
					return item;
				}
				n -= 1;
			
				if (n == 0) {// beached
					item.index = I_MASSBALANCELINE;
					StringWithoutTrailingZeros(infoStr,amtBeached,numDecPlaces); 
					sprintf(text, "Beached:  %s %s ",infoStr,unitsStr);
					item.indent++;
					return item;
				}
				n -= 1;
			
				if (n == 0) {// evaporated/dispersed
					item.index = I_MASSBALANCELINE;
					StringWithoutTrailingZeros(infoStr,amtEvaporated,numDecPlaces); 
					sprintf(text, "Evaporated and Dispersed: %s %s  ",infoStr,unitsStr);
					item.indent++;
					return item;
				}				
				n -= 1;

				if (n == 0) {// offmap
					item.index = I_MASSBALANCELINE;
					StringWithoutTrailingZeros(infoStr,amtOffmap,numDecPlaces); 
					sprintf(text, "Off map: %s %s ",infoStr,unitsStr);
					item.indent++;
					return item;
				}
				n -= 1;
			
			}
		}
		/////////////////////////////////////////////////

		
		/// add lines for every spill
		for (i = 0, m = LESetsList->GetItemCount() ; i < m ; i++) {
			LESetsList->GetListItem((Ptr)&thisLEList, i);
			count = thisLEList->GetListLength();
			if (count > n)
				return thisLEList->GetNthListItem(n, indent + 1, style, text);
			
			n -= count;
		}
	}

	/*if (theMode == ADVANCEDMODE)
	{
		if (n == 0) {
			item.index = I_WEATHERING;
			item.bullet = weatheringOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Weathering");
			*style = bold;
			
			return item;
		}

		n -= 1;
	
		if (weatheringOpen)		// check individual weatherers
		{
			TWeatherer	*thisWeatherer;
	
			for (i = 0, m = weatherList -> GetItemCount (); i < m; i++)
			{
				weatherList -> GetListItem ((Ptr) &thisWeatherer, i);
				count = thisWeatherer -> GetListLength ();
				if (count > n)
					return (thisWeatherer -> GetNthListItem (n, indent + 1, style, text));
				
				n -= count;
			}
		}
	}*/


	/////////////////////////////////////////////
	// Overlays
	////{
	//if (theMode == ADVANCEDMODE)	// only for noaa.ver ? or only if overlay has been loaded ?
	if (theMode == ADVANCEDMODE && (fOverlayList->GetItemCount() > 0 || gNoaaVersion))	// only for noaa.ver ? or only if overlay has been loaded ?
	{	
		// check overlays
	
		if (n == 0) {
			item.index = I_OVERLAYS;
			item.bullet = fOverlaysOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Overlays");
			*style = bold;
	
			return item;
		}		
		n -= 1; // it's not the Overlays toggle item

		// is it in the open Overlays guy			
		if (fOverlaysOpen)
		{
			TOverlay *thisOverlay;
			for (i = 0, m = fOverlayList->GetItemCount() ; i < m ; i++) {
				fOverlayList->GetListItem((Ptr)&thisOverlay, i);
				count = thisOverlay->GetListLength();
				if (count > n)
					return thisOverlay->GetNthListItem(n, indent + 1, style, text);
				
				n -= count;
			}
		}
	}
	/////////////////////	

	// item could not be identified
	item.owner = 0;
	
	return item;
}

Boolean TModel::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	long i, n;
	WorldRect bounds = voidWorldRect;
	TLEList *thisLEList;
	TMap *thisMap;
	OSErr err = 0;
	
	//JLM  add wizard
	if(item.owner == fWizard && fWizard) return fWizard->ListClick(item, inBullet, doubleClick);
	if(item.index == I_WIND && !inBullet && doubleClick) 
	{
		// if in wizard mode and we have a file, we might want them to get to 
		// the dialog where we let them choose wind type			TWindMover *wind = GetWindMover(false); // don't create
		TWindMover *wind = GetWindMover(false); // don't create
		if(wind) 
		{	// pretend it was a double click
			item.index = I_WINDNAME; 
			return wind->ListClick(item,inBullet,doubleClick);
		}
	}

	if ((item.index == I_LESETS || item.index == I_MAPS || item.index == I_OVERLAYS ||
		 item.index == I_WEATHERING || item.index == I_UMOVERS)
		 && doubleClick && !inBullet)
		return item.owner->AddItem(item);

	if (inBullet)
		switch (item.index) {
			case I_MODELSETTINGS: fSettingsOpen = !fSettingsOpen; return TRUE;
			case I_LESETS: fSpillsOpen = !fSpillsOpen; return TRUE;
			case I_MASSBALANCETOTALS: bMassBalanceTotalsOpen = !bMassBalanceTotalsOpen; return TRUE;
			case I_MAPS: mapsOpen = !mapsOpen; return TRUE;
			case I_OVERLAYS: fOverlaysOpen = !fOverlaysOpen; return TRUE;
			case I_UMOVERS: uMoverOpen = !uMoverOpen; return TRUE;
			case I_WIND: uMoverOpen = !uMoverOpen; return TRUE;
			case I_WEATHERING: weatheringOpen = !weatheringOpen; return TRUE;
			case I_UNCERTAIN:
				this->SetUncertain(!fDialogVariables.bUncertain);return TRUE;
			case I_PREVENTLANDJUMPING: fDialogVariables.preventLandJumping = !fDialogVariables.preventLandJumping; return TRUE;
			case I_HINDCAST:
				this->bHindcast = !this->bHindcast; 	
				model->NewDirtNotification(DIRTY_RUNBAR);
				return TRUE;
			case I_DRAWLEMOVEMENT: 
				this->fDrawMovement = !this->fDrawMovement;
//				InvalidateMapImage (); //JLM ??? 12/3/98
				InvalMapDrawingRect();
				return TRUE;
		}

	if (doubleClick)
		switch (item.index)
		{
			case I_DSTDISABLED: {PreferencesDialog(); return false;}
			case I_MODELSETTINGS:
			case I_STARTTIME:
			case I_ENDTIME:
			case I_COMPUTESTEP:
			case I_OUTPUTSTEP:
			case I_UNCERTAIN:
			case I_UNCERTAIN2:
			case I_DRAWLEMOVEMENT: 
			case I_PREVENTLANDJUMPING:
			case I_HINDCAST:
				ModelSettingsDialog(false);
				return false;
			case I_MASSBALANCETOTALS: 	// code goes here, hide this for regular GNOME
			{
				BudgetTableDataH totalBudgetTable = 0;
				short massVolUnits = GetMassUnitsForTotals();	// get units of first spill
				double amttotal = 0;
				GetTotalBudgetTableHdl(massVolUnits, &totalBudgetTable);
				GetTotalAmountSpilled(massVolUnits,&amttotal);
				BudgetTable(massVolUnits, amttotal, totalBudgetTable);
				if (totalBudgetTable) {DisposeHandle((Handle)totalBudgetTable); totalBudgetTable = 0;}
			}
				
			return FALSE;
		}

	return FALSE;
}

short TModel::GetMassUnitsForTotals()
{
	short i, unitsForTotals = BARRELS;
	char unitsStr [64];
	char infoStr [255];
	short numDecPlaces;
	short numLeSets;
	TLEList* thisLEList;
	
	for(numLeSets = LESetsList->GetItemCount(),i=numLeSets-1; i>=0 ; i--)
	{
		LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if(thisLEList -> GetLEType() != UNCERTAINTY_LE )
		{
			unitsForTotals = thisLEList->GetMassUnits(); //return units for the first nonuncertaintly le list
		}
	}
	return unitsForTotals;
}

Boolean TModel::FunctionEnabled(ListItem item, short buttonID)
{
	TWindMover *wind =nil;
	//JLM  add wizard
	if(item.owner == fWizard && fWizard) return fWizard->FunctionEnabled(item,buttonID);

	switch (buttonID) {
		case UPBUTTON:
		case DOWNBUTTON:
			return FALSE;
	}
	
	switch (item.index) {
		case I_WIND: 
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: 
					wind = GetWindMover(false); // don't create
					if(wind) return TRUE;
					else return FALSE;
				case DELETEBUTTON: return FALSE;
			}
			break;

		case I_MODELSETTINGS:
		case I_STARTTIME:
		case I_ENDTIME:
		case I_COMPUTESTEP:
		case I_OUTPUTSTEP:
		case I_UNCERTAIN:
		case I_UNCERTAIN2:
		case I_DRAWLEMOVEMENT:
		case I_PREVENTLANDJUMPING:
		case I_HINDCAST:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
			break;
		case I_LESETS:
		case I_MAPS:
		case I_OVERLAYS:
		case I_UMOVERS:
		case I_WEATHERING:
			switch (buttonID) {
				case ADDBUTTON: return TRUE;
				case SETTINGSBUTTON: return FALSE;
				case DELETEBUTTON: return FALSE;
			}
			break;
		case I_MASSBALANCETOTALS:
		case I_MASSBALANCELINE:
		case I_DSTDISABLED:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return FALSE; 
				case DELETEBUTTON: return FALSE;
			}
			break;
			
	}
	
	return FALSE;
}

OSErr M21bInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)

	return 0;
}

short M21bClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short	theType;

	switch (itemNum) {
		case M21bCANCEL: return M21bCANCEL;

		case M21bCREATE:
		case M21bLOAD:
			return itemNum;

		break;
	}

	return 0;
}

OSErr TModel::AddItem(ListItem item)
{
	//JLM  add wizard
	OSErr err = 0;
	if(item.owner == fWizard && fWizard) return fWizard->AddItem(item);

	switch (item.index) {
		case I_LESETS: return AddLESetDialog();
		
		case I_UMOVERS:
			uMap -> AddItem (item);		// add a mover to the universal map
		break;
		
		case I_MAPS: 
		{
			short dItem = MyModalDialog (M21b, mapWindow, 0, M21bInit, M21bClick);
			if (dItem == M21bLOAD)
			{
				//AddMapsDialog (); break;
				AddMapsDialog2 (); break;
			}
			else if (dItem == M21bCREATE)
			{
				err = CreateMapBox();
				break;
			}
			else //if (dItem == M21bCANCEL)
			{
				break;
			}
		}

		case I_WEATHERING: return AddWeatherDialog();

		case I_OVERLAYS: return AddOverlayDialog();
	}
	
	return 0;
}

OSErr TModel::SettingsItem(ListItem item)
{
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = ListClick(item,inBullet,doubleClick);
	
	return 0;
}

OSErr TModel::DeleteItem(ListItem item)
{
	return 0;
}

///////////////////////////////////////////////////////////////////////////
static Boolean sharedUseNextPreviousButtons =false;

static PopInfoRec daysPopTable[] = {
		{ M10, nil, M10STARTMONTH,   0, pMONTHS,       0, 1, FALSE, nil },
		{ M10, nil, M10STARTYEAR,    0,  pYEARS,       0, 1, FALSE, nil }
	};

static PopInfoRec daysPopTable2[] = {
		{ M10b, nil, M10bSTARTMONTH,   0, pMONTHS,       0, 1, FALSE, nil },
		{ M10b, nil, M10bSTARTYEAR,    0,  pYEARS,       0, 1, FALSE, nil },
		{ M10b, nil, M10bENDMONTH,   0, pMONTHS2,       0, 1, FALSE, nil },
		{ M10b, nil, M10bENDYEAR,    0,  pYEARS2,       0, 1, FALSE, nil }
	};

OSErr MSInit(DialogPtr dialog, VOIDPTR data)
{
	long		durationHours;
	DateTimeRec	time;
	char path[256];
	char str[64];

	if(UseExtendedYears())
		daysPopTable[1].menuID = pYEARS_EXTENDED;
	else
		daysPopTable[1].menuID = pYEARS;

	RegisterPopTable (daysPopTable, sizeof (daysPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (M10, dialog);

	if(sharedUseNextPreviousButtons)
	{
		GetWizButtonTitle_Next(str);
		MySetControlTitle(dialog,M10OK,str);
		GetWizButtonTitle_Previous(str);
		MySetControlTitle(dialog,M10CANCEL,str);
	}
	
	SetDialogItemHandle(dialog, M10HILITEDEFAULT, (Handle) FrameDefault);
	SetDialogItemHandle(dialog, M10FROST, (Handle) FrameEmbossed);
	
//	DisplayTime (dialog, M10STARTMONTH, model -> GetStartTime ());
//	DisplayTime (dialog, M10ENDMONTH, model -> GetEndTime ());

	SecondsToDate (model -> GetStartTime (), &time);
	// check that the year is in the range we allow
	{
		Boolean alertUser = true;
		if(time.year < FirstYearInPopup()) time.year = FirstYearInPopup();
		else if(time.year > LASTYEARINPOPUP) time.year = LASTYEARINPOPUP;
		else alertUser = false;
		if(alertUser) printWarning("The model start time was outside of the allowable year range and will be modified to be within the allowable range.  Be sure to check the year when setting the Start Date in the following dialog.");
	}
	SetPopSelection (dialog, M10STARTMONTH, time.month);
	SetPopSelection (dialog, M10STARTYEAR,  time.year - (FirstYearInPopup()  - 1));
	Long2EditText (dialog, M10STARTDAY, time.day);
	Long2EditText (dialog, M10STARTHOURS, time.hour);
	////////////
	// present the minutes with a leading zero if 9 or less, 1/11/99
	if(0 <= time.minute && time.minute <= 9)
	{
		strcpy(str,"00");
		str[1] = '0'+time.minute;
		mysetitext(dialog, M10STARTMINUTES, str);
	}
	else Long2EditText (dialog, M10STARTMINUTES, time.minute);
	//////////////////

	durationHours = model -> GetEndTime () - model -> GetStartTime ();
	durationHours /= 3600;
	Long2EditText (dialog, M10DURATIONDAYS, durationHours / 24);
	Long2EditText (dialog, M10DURATIONHOURS, durationHours % 24);
	
	Float2EditText (dialog, M10COMPUTESTEP, model -> GetTimeStep   () / 3600.0, 2);
	
	SetButton (dialog, M10UNCERTAIN,  model -> IsUncertain ());
	SetButton (dialog, M10PREVENTLANDJUMPING, model -> PreventLandJumping ());
	SetButton (dialog, M10SHOWCURRENTS, model -> fDrawMovement);
	SetButton (dialog, M10HINDCAST, model -> bHindcast);
	
	if (model -> GetModelMode () < ADVANCEDMODE)
	{
		ShowHideDialogItem(dialog, M10COMPUTESTEP,       false);
		ShowHideDialogItem(dialog, M10COMPUTELABEL,false);
		ShowHideDialogItem(dialog, M10COMPUTEHOURLABEL,false);
		ShowHideDialogItem(dialog, M10PREVENTLANDJUMPING,false);
		ShowHideDialogItem(dialog, M10HINDCAST,false);
	}
	

	MySelectDialogItemText(dialog, M10STARTDAY, 0, 255);

	return 0;
}

short MSClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	Seconds	theTime;
	long	menuID_menuItem;
	long	durationHours = 0, durationDays = 0; 
	char 	errStr[256] = "";
	OSErr 	err = 0;

	switch (itemNum) {
		
		case M10OK:
		{
			//TModelDialogVariables userVal = DefaultTModelDialogVariables(0);
			TModelDialogVariables userVal = model->GetDialogVariables(); // JLM 6/22/00
			Boolean drawMovement = model->fDrawMovement;
			Boolean runBackwards = model->bHindcast;
			
			// JLM note: month, day, year  hr month must be sequential from M10STARTMONTH
			userVal.startTime = RetrievePopTime (dialog, M10STARTMONTH,&err);
			if(err ) break;
			// convert hours to seconds and add to model-start-time to get end-time
			durationDays = EditText2Long (dialog, M10DURATIONDAYS);

			durationHours = EditText2Long (dialog, M10DURATIONHOURS);
			durationHours += durationDays * 24;
			err = model->CheckMaxModelDuration(durationHours,errStr);
			if(err) {printError(errStr); break;}
			
			userVal.duration = durationHours *3600;

			userVal.bUncertain = GetButton (dialog, M10UNCERTAIN);
			userVal.preventLandJumping = GetButton (dialog, M10PREVENTLANDJUMPING);
			drawMovement = GetButton (dialog, M10SHOWCURRENTS);
			runBackwards = GetButton (dialog,M10HINDCAST);
			
			userVal.computeTimeStep = (long) round(EditText2Float (dialog, M10COMPUTESTEP) * 3600);

			// point of no return
			model ->SetDialogVariables (userVal);
			model ->fDrawMovement = drawMovement;
			model ->bHindcast = runBackwards;

			VLUpdate (&objects);
			InvalMapWindow ();
			
			return M10OK;
		}
		case M10CANCEL: return M10CANCEL;
		
		case M10STARTMONTH:
		case M10STARTYEAR:
			PopClick(dialog, itemNum, &menuID_menuItem);
		break;
		
		case M10STARTDAY: //JLM
		case M10STARTHOURS:
		case M10STARTMINUTES:
		case M10DURATIONDAYS:
		case M10DURATIONHOURS:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;

		case M10COMPUTESTEP:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

		case M10UNCERTAIN:
			ToggleButton(dialog, M10UNCERTAIN);
			break;
		case M10HINDCAST:
			ToggleButton(dialog, M10HINDCAST);
			break;
		case M10PREVENTLANDJUMPING:
			ToggleButton(dialog, M10PREVENTLANDJUMPING);
			break;
		
		case M10SHOWCURRENTS:
			ToggleButton(dialog, M10SHOWCURRENTS);
			break;
		
	}

	return 0;
}

OSErr MSbInit(DialogPtr dialog, VOIDPTR data)
{
	long		durationHours;
	DateTimeRec	time, endTime;
	char path[256];
	char str[64];

	if(UseExtendedYears())
	{
		daysPopTable2[1].menuID = pYEARS_EXTENDED;
		daysPopTable2[3].menuID = pYEARS2_EXTENDED;
	}
	else
	{
		daysPopTable2[1].menuID = pYEARS;
		daysPopTable2[3].menuID = pYEARS2;
	}

	RegisterPopTable (daysPopTable2, sizeof (daysPopTable2) / sizeof (PopInfoRec));
	RegisterPopUpDialog (M10b, dialog);

	if(sharedUseNextPreviousButtons)
	{
		GetWizButtonTitle_Next(str);
		MySetControlTitle(dialog,M10OK,str);
		GetWizButtonTitle_Previous(str);
		MySetControlTitle(dialog,M10CANCEL,str);
	}
	
	SetDialogItemHandle(dialog, M10bHILITEDEFAULT, (Handle) FrameDefault);
	SetDialogItemHandle(dialog, M10bFROST, (Handle) FrameEmbossed);
	
//	DisplayTime (dialog, M10STARTMONTH, model -> GetStartTime ());
//	DisplayTime (dialog, M10ENDMONTH, model -> GetEndTime ());

	SecondsToDate (model -> GetStartTime (), &time);
	// check that the year is in the range we allow
	{
		Boolean alertUser = true;
		if(time.year < FirstYearInPopup()) time.year = FirstYearInPopup();
		else if(time.year > LASTYEARINPOPUP) time.year = LASTYEARINPOPUP;
		else alertUser = false;
		if(alertUser) printWarning("The model start time was outside of the allowable year range and will be modified to be within the allowable range.  Be sure to check the year when setting the Start Date in the following dialog.");
	}
	SetPopSelection (dialog, M10bSTARTMONTH, time.month);
	SetPopSelection (dialog, M10bSTARTYEAR,  time.year - (FirstYearInPopup()  - 1));
	Long2EditText (dialog, M10bSTARTDAY, time.day);
	Long2EditText (dialog, M10bSTARTHOURS, time.hour);

	SecondsToDate (model -> GetStartTime () + model -> GetDuration (), &endTime);
	SetPopSelection (dialog, M10bENDMONTH, endTime.month);
	SetPopSelection (dialog, M10bENDYEAR,  endTime.year - (FirstYearInPopup()  - 1));
	Long2EditText (dialog, M10bENDDAY, endTime.day);
	Long2EditText (dialog, M10bENDHOURS, endTime.hour);

	////////////
	// present the minutes with a leading zero if 9 or less, 1/11/99
	if(0 <= time.minute && time.minute <= 9)
	{
		strcpy(str,"00");
		str[1] = '0'+time.minute;
		mysetitext(dialog, M10bSTARTMINUTES, str);
	}
	else Long2EditText (dialog, M10bSTARTMINUTES, time.minute);
	if(0 <= endTime.minute && endTime.minute <= 9)
	{
		strcpy(str,"00");
		str[1] = '0'+endTime.minute;
		mysetitext(dialog, M10bENDMINUTES, str);
	}
	else Long2EditText (dialog, M10bENDMINUTES, endTime.minute);
	//////////////////

	//durationHours = model -> GetEndTime () - model -> GetStartTime ();
	//durationHours /= 3600;
	//Long2EditText (dialog, M10DURATIONDAYS, durationHours / 24);
	//Long2EditText (dialog, M10DURATIONHOURS, durationHours % 24);
	
	Float2EditText (dialog, M10bCOMPUTESTEP, model -> GetTimeStep   () / 3600.0, 2);
	
	SetButton (dialog, M10bUNCERTAIN,  model -> IsUncertain ());
	SetButton (dialog, M10bPREVENTLANDJUMPING, model -> PreventLandJumping ());
	SetButton (dialog, M10bSHOWCURRENTS, model -> fDrawMovement);
	SetButton (dialog, M10bHINDCAST, model -> bHindcast);
	
	if (model -> GetModelMode () < ADVANCEDMODE)
	{
		ShowHideDialogItem(dialog, M10bCOMPUTESTEP,       false);
		ShowHideDialogItem(dialog, M10bCOMPUTELABEL,false);
		ShowHideDialogItem(dialog, M10bCOMPUTEHOURLABEL,false);
		ShowHideDialogItem(dialog, M10bPREVENTLANDJUMPING,false);
		ShowHideDialogItem(dialog, M10bHINDCAST,false);
	}
	

	MySelectDialogItemText(dialog, M10bSTARTDAY, 0, 255);

	return 0;
}

short MSbClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	Seconds	theTime, endTime;
	long	menuID_menuItem;
	long	durationHours = 0, durationDays = 0; 
	Seconds durationSeconds;
	char 	errStr[256] = "";
	OSErr 	err = 0;

	switch (itemNum) {
		
		case M10OK:
		{
			//TModelDialogVariables userVal = DefaultTModelDialogVariables(0);
			TModelDialogVariables userVal = model->GetDialogVariables(); // JLM 6/22/00
			Boolean drawMovement = model->fDrawMovement;
			Boolean runBackwards = model->bHindcast;
			
			// JLM note: month, day, year  hr month must be sequential from M10STARTMONTH
			userVal.startTime = RetrievePopTime (dialog, M10bSTARTMONTH,&err);
			if(err ) break;
			
			endTime = RetrievePopTime (dialog, M10bENDMONTH,&err);
			if (endTime < userVal.startTime)
			{
				strcpy(errStr,"The Model Start Time cannot be later than the Model End Time. Please check your inputs.");
				printError(errStr);
				err = -1;
			}
			if(err ) break;
			
			durationSeconds = endTime - userVal.startTime;	// seconds
			durationHours = durationSeconds / 3600.;	// hours
			// convert hours to seconds and add to model-start-time to get end-time
			// here have end-time, need to calculate duration
			//durationDays = EditText2Long (dialog, M10DURATIONDAYS);

			//durationHours = EditText2Long (dialog, M10DURATIONHOURS);
			//durationHours += durationDays * 24;
			err = model->CheckMaxModelDuration(durationHours,errStr);
			if(err) {printError(errStr); break;}
			
			userVal.duration = durationHours *3600;

			userVal.bUncertain = GetButton (dialog, M10bUNCERTAIN);
			userVal.preventLandJumping = GetButton (dialog, M10bPREVENTLANDJUMPING);
			drawMovement = GetButton (dialog, M10bSHOWCURRENTS);
			runBackwards = GetButton (dialog,M10bHINDCAST);
			
			userVal.computeTimeStep = (long) round(EditText2Float (dialog, M10bCOMPUTESTEP) * 3600);

			// point of no return
			model ->SetDialogVariables (userVal);
			model ->fDrawMovement = drawMovement;
			model ->bHindcast = runBackwards;

			VLUpdate (&objects);
			InvalMapWindow ();
			
			return M10OK;
		}
		case M10CANCEL: return M10CANCEL;
		
		case M10bSTARTMONTH:
		case M10bSTARTYEAR:
		case M10bENDMONTH:
		case M10bENDYEAR:
			PopClick(dialog, itemNum, &menuID_menuItem);
		break;
		
		case M10bENDDAY: //JLM
		case M10bENDHOURS:
		case M10bENDMINUTES:
		case M10bSTARTDAY: //JLM
		case M10bSTARTHOURS:
		case M10bSTARTMINUTES:
		//case M10DURATIONDAYS:
		//case M10DURATIONHOURS:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;

		case M10bCOMPUTESTEP:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

		case M10bUNCERTAIN:
			ToggleButton(dialog, M10bUNCERTAIN);
			break;
		case M10bHINDCAST:
			ToggleButton(dialog, M10bHINDCAST);
			break;
		case M10bPREVENTLANDJUMPING:
			ToggleButton(dialog, M10bPREVENTLANDJUMPING);
			break;
		
		case M10bSHOWCURRENTS:
			ToggleButton(dialog, M10bSHOWCURRENTS);
			break;
		
	}

	return 0;
}

OSErr ModelSettingsDialog(Boolean useNextPreviousButtons)
{
	short item;
	sharedUseNextPreviousButtons = useNextPreviousButtons;
	if (model->bHindcast)
		item = MyModalDialog(M10b, mapWindow, 0, MSbInit, MSbClick);
	else
		item = MyModalDialog(M10, mapWindow, 0, MSInit, MSClick);
	useNextPreviousButtons =  false;
	if(item == M10CANCEL) return USERCANCEL; 
	model->NewDirtNotification();
	if(item == M10OK) return 0; // JLM 7/8/98
	else return -1;
}
/////////////////////////////////////////////////

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
#ifdef MAC
CGrafPtr GetColorImage(void (*drawProc)(void * object,WorldRect wRect,Rect drawRect), void * object,WorldRect wRect,Rect bitMapRect,OSErr * err)
{
	BitMap      	paintBits;
	MyGWorldRec		saveGWorld;
	CGrafPtr			colorImage = 0;	
	
	*err = 0;

	memset(&paintBits,0,sizeof(paintBits));
	memset(&saveGWorld,0,sizeof(MyGWorldRec));
	
	SaveGWorld (&saveGWorld);	// save original world settings
	
	colorImage = MakeNewGWorld (0, &bitMapRect);
	if (colorImage != nil)
	{
		BeginGWorldDraw (colorImage, false, true);
		(*drawProc)(object,wRect,bitMapRect);
		EndGWorldDraw (colorImage, &saveGWorld);	
	}
	else
	{
		*err = memFullErr;
	}
	
	return colorImage; 
}
/////////////////////////////////////////////////
#if TARGET_API_MAC_CARBON
BitMap GetBlackAndWhiteBitmap(void (*drawProc)(void * object,WorldRect wRect,Rect drawRect), void * object,WorldRect wRect,Rect bitMapRect,OSErr * err)
{
	BitMap      	paintBits;
	MyGWorldRec		saveGWorld;
	RgnHandle		newVis = 0;
	const BitMap *b;
	GrafPtr newPort;
	
	*err = 0;

	memset(&paintBits,0,sizeof(paintBits));
	memset(&saveGWorld,0,sizeof(MyGWorldRec));
	
	paintBits.bounds = bitMapRect;
	paintBits.rowBytes = (((paintBits.bounds.right - paintBits.bounds.left) >> 4) + 1) << 1;
	paintBits.baseAddr = _NewPtrClear((paintBits.bounds.bottom - paintBits.bounds.top)* paintBits.rowBytes);	
	
	if (!paintBits.baseAddr){
		*err = memFullErr;
		printError("Out of memory in GetBlackAndWhiteBitmap()");
	}
	else {
	
		*err = MyNewGWorld(mapWindow,1,&bitMapRect,&saveGWorld);	// this saves current port and device into GWorld
		newPort = MySetGWorld(&saveGWorld,false,true);
		ForeColor(blackColor);			
		BackColor(whiteColor);
		
		PrepareToDraw(bitMapRect, wRect, 0, 0);
		gRect = bitMapRect;
		(*drawProc)(object,wRect,bitMapRect);
		
		b = GetPortBitMapForCopyBits(newPort);
		if(b) {
			CopyBits(b,&paintBits, &paintBits.bounds,  &paintBits.bounds, srcCopy, nil);
		}
			
		if (saveGWorld . theGWorld != nil){
			SetGWorld ((CGrafPtr) saveGWorld . savePort, saveGWorld . saveGDevice);
			UnlockPixels (GetPortPixMap(saveGWorld . theGWorld));	/* locked when it was created */
			DisposeGWorld (saveGWorld . theGWorld);
			saveGWorld . theGWorld = nil;
		}
	}
	
	return paintBits; 
}
#else
/////////
BitMap GetBlackAndWhiteBitmap(void (*drawProc)(void * object,WorldRect wRect,Rect drawRect), void * object,WorldRect wRect,Rect bitMapRect,OSErr * err)
{
	BitMap      	paintBits;
	MyGWorldRec		saveGWorld;
	
	*err = 0;

	memset(&paintBits,0,sizeof(paintBits));
	memset(&saveGWorld,0,sizeof(MyGWorldRec));
	
	paintBits = OpenBlackAndWhiteBitMap(bitMapRect, &saveGWorld);
	
	if (!paintBits.baseAddr)	
	{
		*err = memFullErr;
		printError("Out of memory in GetBlackAndWhiteBitmap()");
	}
	else
	{
		PrepareToDraw(bitMapRect, wRect, 0, 0);
		gRect = bitMapRect;
		(*drawProc)(object,wRect,bitMapRect);
		
		CloseBlackAndWhiteBitMap (&saveGWorld);
	} 		
	
	return paintBits; 
	
}
#endif
/////////
#else ////////////////////
/////////////
//HDIB GetImageHelper(void (*drawProc)(void * object,WorldRect wRect,Rect drawRect), void * object,WorldRect wRect,Rect bitMapRect,OSErr * err,Boolean blackAndWhiteFlag,Boolean wantDIB)
HANDLE GetImageHelper(void (*drawProc)(void * object,WorldRect wRect,Rect drawRect), void * object,WorldRect wRect,Rect bitMapRect,OSErr * err,Boolean blackAndWhiteFlag,Boolean wantDIB)
{
	Boolean saveStatic = false;
   //HDC hScrDC = 0, hMemDC=0;           // screen DC and memory DC
   static HDC hScrDC = 0, hMemDC=0;           // screen DC and memory DC
   HBITMAP hBitmap = 0, hOldBitmap=0;  // handles to deice-dependent bitmaps
   int nX, nY, nX2, nY2;         // coordinates of rectangle to grab
   int nWidth, nHeight;          // DIB width and height
   int xScrn, yScrn;             // screen resolution
	Rect r,saveClip;
	HDIB hDib = 0;// the return value
	GrafPtr	savePort;
	long numBytes,bwidth,height;
	
	const short kUnableToCreateDC = -12;
	const short kUnableToBitmap = -13;

   *err = 0;
	
	//  create a DC for the screen and create
   //  a memory DC compatible to screen DC
	if (!hScrDC) // try creating it only the first time
  		hScrDC = CreateDC("DISPLAY", NULL, NULL, NULL);
   if(!hScrDC) {*err = kUnableToCreateDC; goto done;}
	if (!hMemDC) // try creating it only the first time
		hMemDC = CreateCompatibleDC(hScrDC);
   if(!hMemDC) {*err = kUnableToCreateDC; goto done;}
	
	MyOffsetRect(&bitMapRect,-bitMapRect.left,-bitMapRect.top);

//////////
	if(blackAndWhiteFlag)
	{/// create a mono-bitmap 
		hBitmap = CreateBitmap(bitMapRect.right,bitMapRect.bottom,1,1,NULL);
	}
	else // color
	{// create a bitmap compatible with the screen DC 
   	hBitmap = CreateCompatibleBitmap(hScrDC,bitMapRect.right,bitMapRect.bottom);
	}
///////////
	
	if(!hBitmap){*err = kUnableToBitmap; goto done;}

	//select new bitmap into memory DC 
	hOldBitmap = (HBITMAP)SelectObject(hMemDC, hBitmap);
	
	GetPort(&savePort);
	//SetPort(hMemDC);
	SetPort((HWND)hMemDC);
	
	saveClip = MyClipRect(bitMapRect);

	PrepareToDraw(bitMapRect, wRect, 0, 0);
	SetTempDrawingRect(&bitMapRect);
	gRect = bitMapRect;
	
	EraseRect(&bitMapRect);
	(*drawProc)(object,wRect,bitMapRect);
	
	SetTempDrawingRect(0);
	MyClipRect(saveClip);
	SetPort(savePort);
	
	//  select old bitmap back into memory DC to get handle to bitmap 
	hBitmap = (HBITMAP)SelectObject(hMemDC, hOldBitmap);

done:
	if (!saveStatic) {
	if(hScrDC) {DeleteDC(hScrDC); hScrDC = 0;} 
	if(hMemDC) {DeleteDC(hMemDC); hMemDC = 0;}
	}
	
	if(*err)
	{
		switch(*err)
		{
			case kUnableToCreateDC:
				printError("Unable to create device context in GetImageHelper"); 
				break;
				
			case kUnableToBitmap:
				printError("Unable to create bitmap in GetImageHelper"); 
				break;
		}
		if(hBitmap) {DeleteObject(hBitmap); hBitmap = 0;}
	}
	if(hBitmap && wantDIB) 
	{
		hDib = BitmapToDIB(hBitmap,0);
		DeleteObject(hBitmap); 
		hBitmap = 0;
	}
	if (wantDIB) return (HANDLE) hDib;

   return (HANDLE) hBitmap;
}
/////////////////////////////////////////////////
HDIB GetColorImageDIB(void (*drawProc)(void * object,WorldRect wRect,Rect drawRect), void * object,WorldRect wRect,Rect bitMapRect,OSErr * err)
{
	Boolean blackAndWhiteFlag = false;
	return (HDIB) GetImageHelper(drawProc,object,wRect,bitMapRect,err,blackAndWhiteFlag,true);

}
/////////////////////////////////////////////////
HBITMAP GetColorImageBitmap(void (*drawProc)(void * object,WorldRect wRect,Rect drawRect), void * object,WorldRect wRect,Rect bitMapRect,OSErr * err)
{
	Boolean blackAndWhiteFlag = false;
	return (HBITMAP) GetImageHelper(drawProc,object,wRect,bitMapRect,err,blackAndWhiteFlag,false);

}
/////////////////////////////////////////////////
HDIB GetBlackAndWhiteBitmap(void (*drawProc)(void * object,WorldRect wRect,Rect drawRect), void * object,WorldRect wRect,Rect bitMapRect,OSErr * err)
{
	Boolean blackAndWhiteFlag = true;
	return (HDIB) GetImageHelper(drawProc,object,wRect,bitMapRect,err,blackAndWhiteFlag,true);
}
#endif
/////////////////////////////////////////////////
/////////////////////////////////////////////////

/////////////////////////////////////////////////

void DrawBaseMap(void * object,WorldRect wRect,Rect r)
{
	TModel* theModel = (TModel*)object; // typecast 
	TMap 		*map = 0;
	long 		n;

	RGBForeColor(&backColors[settings.backgroundColor]);
	PaintRect(&r);
	RGBForeColor(&colors[BLACK]);

	if(sharedPrinting) StartThinLines();
	////////////////////////////////////
	
	if (settings.showLatLongLines && settings.llPosition == LL_BELOW)
		DrawLatLongLines(r, wRect);

	// draw each of the maps (in reverse order to show priority)		
	for (n = theModel -> mapList->GetItemCount() - 1; n >= 0 ; n--) {
		theModel -> mapList->GetListItem((Ptr)&map, n);
		map->Draw(r, wRect);
	}

	// draw the universal map's movers
	theModel -> uMap -> Draw(r, wRect);
	
	if (settings.showLatLongLines && settings.llPosition == LL_ABOVE)
		DrawLatLongLines(r, wRect);

	///////////////////////////
	if(sharedPrinting) StopThinLines();

}

/////////////////////////////////////////////////
void DrawCombinedImage(void * object,WorldRect wRect,Rect r)
{
	TModel* theModel = (TModel*)object; // typecast 
	TLEList 	*list = 0;
	TMover *mover = 0;
	TMap *map = 0;
	long i,n,j,m;
			
	if (!sharedPrinting && theModel->mapImage)
	{	// draw from saved bitmap
#ifdef MAC
		CopyWorldToScreen (theModel->mapImage, r, srcCopy); // actually copies to current GWorld, not screen
#else 
		DrawDIBImage(PURPLE,&theModel->mapImage,r); // NOTE: color is ignored since it is a color bitmap
		// note we use  &mapImage because the MAC and the way the function is used in TVectorMap
#endif
	}
	else 
	{	// draw from scratch
		DrawBaseMap(theModel,wRect,r);
	}
	
	// now draw on top of the base map
	//////////////////////////////////

	// code goes here, if drawingdependsontime (variable currents, winds, ...) draw movers on top (map shouldn't change)
	// problem is if multiple maps on top of each other all movers will show on top map, big bnas can slow down drawing a lot though
	// 
	// draw each of the maps (in reverse order to show priority)	
	if (model->DrawingDependsOnTime() && model->GetMapCount()==1)	
	{
		for (n = theModel -> mapList->GetItemCount() - 1; n >= 0 ; n--) {
			theModel -> mapList->GetListItem((Ptr)&map, n);
			//map->Draw(r, wRect);
			// draw each of the movers
			for (j = 0, m = map->moverList->GetItemCount() ; j < m ; j++) {
				map->moverList->GetListItem((Ptr)&mover, j);
				mover->Draw(r, wRect);
			}

		}
		// draw the universal map's movers
		for (i = 0, n = theModel->uMap->moverList->GetItemCount() ; i< n ; i++) {
			theModel->uMap->moverList->GetListItem((Ptr)&mover, i);
			mover->Draw(r, wRect);
		}
		//theModel -> uMap -> Draw(r, wRect);
	}
	//////////////////////////////////
	
	theModel->DrawOverlays(r,wRect);

	theModel->DrawLEMovement();

	// draw uncertain LEs if any
	if(theModel->IsUncertain())  //JLM 9/10/98
	{
		for (i = 0, n = theModel->LESetsList->GetItemCount() ; i < n ; i++) {
			theModel->LESetsList->GetListItem((Ptr)&list, i);
			if(list->GetLEType() == UNCERTAINTY_LE)
				list->Draw(r, wRect);
		}
	}

	// draw forecast LE's
	for (i = 0, n = theModel->LESetsList->GetItemCount() ; i < n ; i++) {
		theModel->LESetsList->GetListItem((Ptr)&list, i);
		if(list->GetLEType() == FORECAST_LE)
			list->Draw(r, wRect);
	}
	
	theModel->DrawLegends(r,wRect);
}

short PicFrameExtension(void)
{
	short ht = 16;
	#ifdef IBM
		FontInfo finfo;
		GetFontInfo(&finfo);
		ht =  finfo.ascent+finfo.descent+finfo.leading;
	#endif
	return ht;
}

/////////////////////////////////////////////////
void DrawCombinedImageWithDate(void * object,WorldRect wRect,Rect extendedRect)
{
	short extra = PicFrameExtension();
	Rect  rectWithoutLabel = extendedRect;
	Rect  labelRect = extendedRect;
	short ht;
	
	rectWithoutLabel.bottom -= extra; // compensate for the extra amount added for the labels
	labelRect.top = rectWithoutLabel.bottom;
	
#ifdef IBM	// JLM 3/15/02 this function is only used for making movie frames
	PrepareToDraw(rectWithoutLabel, wRect, 0, 0);
	SetTempDrawingRect(&rectWithoutLabel);
	gRect = rectWithoutLabel;
#endif

	DrawCombinedImage(object,wRect,rectWithoutLabel);
	//////////////////
	//now draw the labeling info
	{
		DateTimeRec time;
		char *p,s[256];
		char str[256];
		FontInfo finfo;
		EraseRect(&labelRect);
		MyMoveTo(labelRect.left,labelRect.top);
		MyLineTo(labelRect.right,labelRect.top);
		//
		SecondsToDate (model->GetModelTime(), &time);
		Date2String(&time, s);
		if (p = strrchr(s, ':')) p[0] = 0; // remove seconds
		sprintf(str,"   %s",s);
		#ifdef MAC
			ht = labelRect.bottom -4;
		#else
			GetFontInfo(&finfo);
			ht =  (labelRect.bottom + labelRect.top + finfo.ascent)/2;
		#endif
		MyMoveTo(labelRect.left,ht);
		drawstring(str);
		//
	}
}

/////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////


Boolean TModel::DrawingDependsOnTime(void)
{
	Boolean drawingDependsOnTime = FALSE;
	long n;
	TMap 		*map = 0;
	
	// ask each map 	
	for (n = model -> mapList->GetItemCount() - 1; n >= 0 || drawingDependsOnTime; n--) {
		model -> mapList->GetListItem((Ptr)&map, n);
		drawingDependsOnTime = map->DrawingDependsOnTime();
		if(drawingDependsOnTime) 
			return drawingDependsOnTime;
	}

	// ask the universal map
	drawingDependsOnTime = model -> uMap -> DrawingDependsOnTime();
	if(drawingDependsOnTime) 
		return drawingDependsOnTime;
	
	return drawingDependsOnTime;
	//return true;
}

/////////////////////////////////////////////////


void TModel::Draw(Rect r, WorldRect view)
{
	long 		i, n;
	TLEList 	*list = 0;
	TMap 		*map = 0;
	Boolean 	restoredImageSuccessfully = false;
	RGBColor	saveColor;
	Boolean makeNewMapImage = false;
	Boolean makeCombinedImage = false;
	OSErr err = 0;
	
#ifdef MAC
	MyGWorldRec saveGWorld;
	CGrafPtr	frameGWorld = nil;
	CGrafPtr combinedImage = 0;
#else
	//HDIB combinedImage = 0;
	HBITMAP combinedImage = 0;
#endif

	if(r.right <= r.left || r.bottom <= r.top) return; // JLM 2/5/99 , nothing to draw
	
	makeNewMapImage =  !sharedPrinting && (mapImage == nil || HasFrameMapListChanged() || r != mapImageRect);
	
	//if(this -> DrawingDependsOnTime())	
	if(this -> DrawingDependsOnTime() && (GetMapCount()>1 || GetMapCount()==0))	// universal map needs to be redrawn
		makeNewMapImage = true;
	
	if (makeNewMapImage)
	{
		// delete previous map image if any
		if(mapImage)
		{
			#ifdef MAC
				KillGWorld (mapImage);
			#else 
				DestroyDIB(mapImage);
			#endif
			mapImage = nil;
		}
		
		mapImage = GetColorImageDIB(DrawBaseMap,model,view,r,&err);
		mapImageRect = r;	// save rect used to make map image
		UpdateFrameMapList ();	// save the models map list for future comparison
	}
	
	makeCombinedImage = !sharedPrinting && mapImage; // no need trying for a combined image if we couldn't make a mapImage 
	
//#ifdef MAC
	if(makeCombinedImage)
	{
		combinedImage = GetColorImageBitmap(DrawCombinedImage,model,view,r,&err);	// 2/28/03 this was causing slowdown on the PC
	}
//#else
	// 2/28/03 GetColorImage calls GetImageHelper which calls PrepareToDraw(bitMapRect, wRect, 0, 0) 
	// and messes up the drawing rect so we need to preparetodraw again
	PrepareToDraw(r,view,0,0);
//#endif
	
	if (!sharedPrinting && combinedImage)
	{	// draw from saved bitmap
#ifdef MAC
		CopyWorldToScreen (combinedImage, r, srcCopy); // actually copies to current GWorld, not screen
#else 
		//DrawDIBImage(PURPLE,&combinedImage,r); // NOTE: color is ignored since it is a color bitmap
		DrawBitmapImage(PURPLE,&combinedImage,r); // NOTE: color is ignored since it is a color bitmap
		// note we use  &mapImage because the MAC and the way the function is used in TVectorMap
#endif
	}
	else 
	{	// draw from scratch
		DrawCombinedImage(model,view,r);
	}

	/// clean up temporary combined image
	if(combinedImage)
	{
		#ifdef MAC
			KillGWorld (combinedImage);
		#else 
			//DestroyDIB(combinedImage);
			DeleteObject(combinedImage);	// free the Bitmap
		#endif
		combinedImage = nil;
	}

}

////////////////////////////////////////////////////////////////////////////////

void TModel::UpdateFrameMapList ()
{
	short	i, n;
	TMap	*thisMap;
	
	if (frameMapList)
	{
		frameMapList->ClearList();
	
		for (i = 0, n = mapList->GetItemCount() ; i < n ; i++)
		{
			mapList->GetListItem((Ptr)&thisMap, i);
			frameMapList -> AppendItem ((Ptr) &thisMap);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////

Boolean TModel::HasFrameMapListChanged ()
{
	short	i, n;
	TMap	*modelMap, *frameMap;
	Boolean	bChanged = false;
	
	if (mapList == nil || frameMapList == nil) return true;

	// do we have the same number of maps as before?
	if (mapList->GetItemCount() != frameMapList->GetItemCount())
		bChanged = true;
	else
	{
		for (i = 0, n = mapList->GetItemCount(); i < n; i++)
		{
			mapList -> GetListItem ((Ptr) &modelMap, i);
			frameMapList -> GetListItem ((Ptr) &frameMap, i);
			if (modelMap != frameMap)
			{
				bChanged = true;
				break;
			}
		}
	}

	return bChanged;
}
////////////////////////////////////////////////////////////////////////////////

void TModel::SetModelMode (long theMode)
{
	modelMode = theMode;

	return;
}

////////////////////////////////////////////////////////////////////////////////

OSErr TModel::SaveModelLEs (Seconds forTime, short fileNumber)
{
	long 		n, i, count;
	BFPB 		LEFile;
	char 		LEFileName[256];
	double freeBytes;
	short vRefNum;
	long sizeNeeded = 0;
	OSErr err = 0;
	
	static Boolean sEnoughFreeSpaceLastTime = true; // so we don't keep telling the user
	
	LEFile.f = 0;
	
	err = GetTempLEOutputFilePathName (fileNumber, LEFileName,&vRefNum);
	if(err) {printNote("Unable to create temporary folder to store past positions of the splots for the runbar."); return err;}

	hdelete(0, 0, LEFileName);

	// get vRefNum for the file we will be creating
	err = FreeBytesOnVolume(vRefNum, &freeBytes, LEFileName); // MAC uses vRefNum, IBM uses first part of LEFileName
	if(err) {printNote("Error calculating free space on disk."); return err;}
	
	// calculate how much disk space is needed
	// tell the user if we have a problem
	/////////////////////////////////////////////////
	{
		long i, j, n, m, numLEs;
		TLEList *thisLEList;
		LETYPE leType;
		Handle h;
		
		sizeNeeded = 100000; // leave some extra space on the hard drive
		for (i = 0, n = LESetsList->GetItemCount() ; i < n ; i++) 
		{
			LESetsList->GetListItem((Ptr)&thisLEList, i);
			h = (Handle) thisLEList -> LEHandle;
			if(h) sizeNeeded+= _GetHandleSize(h);
		}
		
		if(freeBytes < sizeNeeded)
		{	// not enough disk space
			if(sEnoughFreeSpaceLastTime)
			{	// first time there was not enough space
				char msg[256];
				#ifdef MAC
					sprintf(msg,"Due to limited disk space on the Volume containing the System Preferences, past positions of the splots cannot be saved for the runbar.  %ld bytes required, %g available",sizeNeeded,freeBytes);
					//printNote("Due to limited disk space on the Volume containing the System Preferences, past positions of the splots cannot be saved for the runbar." );
					printNote(msg);
				#else
					sprintf(msg,"Due to limited disk space on the hard drive containing the Windows directory, past positions of the splots cannot be saved for the runbar.  %g bytes required, %g available",sizeNeeded,freeBytes);
					//printNote("Due to limited disk space on the hard drive containing the Windows directory, past positions of the splots cannot be saved for the runbar." );
					printNote(msg);
				#endif
			}
			sEnoughFreeSpaceLastTime = false;
			return noErr; // saving these files is optional
		}
		
		sEnoughFreeSpaceLastTime = true;
	}
	/////////////////////////////////////////////////

	if (err = hcreate(0, 0, LEFileName, '\?\?\?\?', 'BINA'))
	{	// try using the GNOME directory if windows won't allow user to write file
		err = GetTempLEOutputFilePathNameTake2 (fileNumber, LEFileName,&vRefNum);
		if(err) {printNote("Unable to create temporary folder to store past positions of the splots for the runbar."); return err;}
	
		hdelete(0, 0, LEFileName);
		if (err = hcreate(0, 0, LEFileName, '\?\?\?\?', 'BINA'))		
		{ /*TechError("SaveModelLEs()", "hcreate()", err); return err;*/return noErr; }
	}

	if (err = FSOpenBuf(0, 0, LEFileName, &LEFile, 100000, FALSE))
		{ TechError("SaveModelLEs()", "FSOpenBuf()", err); return err; }

	err = SaveModelLEs (&LEFile);
	if (!err)
	{
		LEFrameRec	thisFrame;

		strcpy (thisFrame.frameLEFName, LEFileName);
		thisFrame.frameTime = forTime;

		if (fileNumber >= LEFramesList -> GetItemCount ())
			err = LEFramesList -> AppendItem ((Ptr) &thisFrame);
		else
			LEFramesList -> SetListItem ((Ptr) &thisFrame, fileNumber);
	}
	
	if (LEFile.f) FSCloseBuf(&LEFile);
	
	if (err) printError("Error saving model splots to a file");
	return err;
}

////////////////////////////////////////////////////////////////////////////////

OSErr TModel::SaveModelLEs (BFPB *bfpb)
{
	TOLEList	*thisLEList;
	long		i;
	ClassID		id;
	OSErr err = 0;

	// write the number of LE sets that are to be written to this file
	err = WriteMacValue (bfpb, (long) LESetsList -> GetItemCount ());
	
	if (!err)
	{
		for (i = 0; i < LESetsList -> GetItemCount (); i++)
		{
			LESetsList -> GetListItem ((Ptr) &thisLEList, i);
			id = thisLEList->GetClassID();
			if (err = WriteMacValue (bfpb, id)) break;
			if (err = thisLEList -> Write (bfpb)) break;
		}
	}

	return err;
}

////////////////////////////////////////////////////////////////////////////////

Seconds TModel::ClosestSavedModelLEsTime(Seconds givenTime)
{
	Seconds bestTime = this->GetLastComputeTime();
	long i, timeDiff, minTimeDiff = TimeDifference(givenTime,bestTime);
	LEFrameRec	thisFrame;
	if (LEFramesList)
	{
		for (i = LEFramesList -> GetItemCount () - 1; i >= 0 && minTimeDiff > 0 ; --i)
		{
			LEFramesList -> GetListItem ((Ptr) &thisFrame, i);
			timeDiff = TimeDifference(thisFrame.frameTime,givenTime);
			if (timeDiff < minTimeDiff)
			{
				minTimeDiff = timeDiff;
				bestTime = thisFrame.frameTime;
			}
		}
	}
	return bestTime;
}

/////////////////////////////////////////////////

Seconds TModel::PreviousSavedModelLEsTime(Seconds givenTime)
{
	Seconds bestTime = givenTime;
	long i, timeDiff, minTimeDiff = 300 * 24 * 3600;	// 300 days
	LEFrameRec	thisFrame;
	if (LEFramesList)
	{
		for (i = LEFramesList -> GetItemCount () - 1; i >= 0 ; --i)
		{
			LEFramesList -> GetListItem ((Ptr) &thisFrame, i);
			if (thisFrame.frameTime < givenTime) 
			{	// after the givenTime
				timeDiff = TimeDifference(thisFrame.frameTime,givenTime);
				if (timeDiff < minTimeDiff)
				{
					minTimeDiff = timeDiff;
					bestTime = thisFrame.frameTime;
				}
			}
		}
	}
	return bestTime;
}

Seconds TModel::NextSavedModelLEsTime(Seconds givenTime)
{
	Seconds bestTime = givenTime;
	long i, timeDiff, minTimeDiff = 300 * 24 * 3600;	// 300 days
	LEFrameRec	thisFrame;
	if (LEFramesList)
	{
		for (i = LEFramesList -> GetItemCount () - 1; i >= 0 ; --i)
		{
			LEFramesList -> GetListItem ((Ptr) &thisFrame, i);
			if (thisFrame.frameTime > givenTime) 
			{	// after the givenTime
				timeDiff = TimeDifference(thisFrame.frameTime,givenTime);
				if (timeDiff < minTimeDiff)
				{
					minTimeDiff = timeDiff;
					bestTime = thisFrame.frameTime;
				}
			}
		}
	}
	return bestTime;
}

OSErr TModel::LoadModelLEs (Seconds forTime, Seconds *actualTime)
// note: if data is not available for the requested time, the closest frame is 
// returned.  actualTime contains the time for which LE data was loaded
{
	OSErr		err = noErr;
	long		bestFrameIndex = -1, i, timeDiff,minTimeDiff = 300 * 24 * 3600;	// 300 days
	LEFrameRec	thisFrame;
	BFPB 		LEFile;
		
	if (LEFramesList)
	{
		for (i = LEFramesList -> GetItemCount () - 1; i >= 0 ; --i)
		{
			LEFramesList -> GetListItem ((Ptr) &thisFrame, i);
			timeDiff = TimeDifference(thisFrame.frameTime,forTime);
			if (timeDiff < minTimeDiff)
			{
				minTimeDiff = timeDiff;
				bestFrameIndex = i; ///JLM 1/6/99
			}
		}
	}
	
	if (bestFrameIndex >= 0)
	{
		LEFramesList -> GetListItem ((Ptr) &thisFrame, bestFrameIndex);
		err = FSOpenBuf(0, 0, thisFrame.frameLEFName, &LEFile, 100000, FALSE);
		if (!err)
		{
			err = LoadModelLEs (&LEFile);
			if (!err)
			{
				SetModelTime (thisFrame.frameTime);
				//DisplayCurrentTime(false);
				*actualTime = thisFrame.frameTime;
			}
			
			FSCloseBuf (&LEFile);
		}
	}

	return err;
}

////////////////////////////////////////////////////////////////////////////////

OSErr TModel::LoadModelLEs (BFPB *bfpb)
{
	OSErr		err = noErr;
	TLEList		*thisLEList = nil;
	long		numLESets, i;
	ClassID 	id;

	// model LE's need to be disposed of before attempting to read new ones
	DisposeModelLEs ();

	if (err = ReadMacValue (bfpb, &numLESets)) return err;

	for (i = 0; i < numLESets; i++)
	{
		if (err = ReadMacValue (bfpb, &id)) return err;
		
		switch(id)
		{
			case TYPE_LELIST: thisLEList = new TLEList (); break;
			case TYPE_OSSMLELIST: thisLEList = new TOLEList (); break;
			case TYPE_SPRAYLELIST: thisLEList = new TSprayLEList (); break;
			case TYPE_CDOGLELIST: thisLEList = new CDOGLEList (); break;
			default: printError("Unrecognized LE List type in TModel::LoadModelLEs()."); return -1;
		}
			
		if (!thisLEList)
			err = memFullErr;
		else
		{
			err = thisLEList->Read(bfpb);
			if (!err)
			{
				err = AddLEList(thisLEList, 0);
			}
			else
			{
				thisLEList -> Dispose ();
				delete (thisLEList);
				break;
			}
		}
	}

	return err;
}

////////////////////////////////////////////////////////////////////////////////

void TModel::DisposeModelLEs ()
// note: this function does not dispose of the model's main LE-sets list, STH
{
	long	i, n;
	TLEList	*thisLEList;
	OSErr err = 0;

	if (LESetsList)
	{
		while (LESetsList -> GetItemCount() > 0)
		{
			// get the bottom-most LEList and drop & dispose of it
			LESetsList->GetListItem((Ptr)&thisLEList, 0);
			if (err = LESetsList->DeleteItem(0))
				{ TechError("TModel::DisposeModelLEs()", "DeleteItem()", err); return; }
			if(thisLEList)
				{thisLEList -> Dispose ();delete (thisLEList); thisLEList = 0;}
		}
	}
	
	return;
}

/////////////////////////////////////////////////

void GetFramePathFormatStr(char* folderPathWithDelimiter,char* str)
{
		strcpy(str,folderPathWithDelimiter);
#ifdef IBM
		strcat(str,"frame%03ld.bmp");
#else
		strcat(str,"frame%03ld.pic");
#endif
}

////////////////////////////////////////////////////////////////////////////////
// mac resource files have a size limit so changed to save the frames to a folder
OSErr TModel::SaveMovieFrame (WindowPtr mapWindow, Rect frameRect)
{
	OSErr		err = noErr;
	short		pictResID;
	Rect originalPicFrame = frameRect;
	Rect extendedPicFrame = originalPicFrame;
		
#ifdef MAC
	/// Label the picture
	// try putting label below the picture
	Rect saveClip;
	char *p,s[256];
	short i,f;
	long longZero = 0, longCount = 4;
	DateTimeRec time;
	PicHandle combinedPic = nil; 

	char filePathName[256];
	char formatStr[256];

	short		vRefNum;
	long		parDirID;
	
	// get the dir-spec info for preferences folder
	err = FindFolder (kOnSystemDisk, kPreferencesFolderType, kCreateFolder, &vRefNum, &parDirID);
	
	GetFramePathFormatStr(fMoviePicsPath,formatStr);
	sprintf(filePathName,formatStr,movieFrameIndex);

	(void)hdelete(0, 0, filePathName);
	if (err = hcreate(0, 0, filePathName, 'ttxt', 'PICT'))
		{ TechError("SaveMovieFrame()", "hcreate()", err); return err; }

	if (err = hopendf(0, 0, filePathName, fsCurPerm, &f))
		{ TechError("SaveMovieFrame()", "hopendf()", err); return err; }
	
	extendedPicFrame.bottom += PicFrameExtension();
	combinedPic = OpenPicture(&extendedPicFrame);
	saveClip = MyClipRect(extendedPicFrame);
	EraseRect(&extendedPicFrame);// JLM  7/2/99
	
	DrawMaps (mapWindow, MapDrawingRect(), settings.currentView, FALSE);
	
	MyMoveTo(originalPicFrame.left,originalPicFrame.bottom);
	MyLineTo(originalPicFrame.right,originalPicFrame.bottom);
	//
	SecondsToDate (this->GetModelTime(), &time);
	Date2String(&time, s);
	if (p = strrchr(s, ':')) p[0] = 0; // remove seconds
	MyMoveTo(extendedPicFrame.left+20,extendedPicFrame.bottom -4);
	drawstring(s);
	//
	ClosePicture();
	MyClipRect(saveClip);
	//////
	
	for (i = 0 ; i < 128 ; i++)
		FSWrite(f, &longCount, &longZero); // write blank PICT header
	
	longCount = _GetHandleSize((Handle)combinedPic);
	_HLock((Handle)combinedPic);
	err = FSWrite(f, &longCount, (Ptr)*combinedPic);
	_HUnlock((Handle)combinedPic);
	err |= FSClose(f);
	
	//DisposeHandle((Handle)p); // KillPicture(p);
	KillPicture(combinedPic); combinedPic = 0;
		
#else //IBM
	HDIB combinedImage = 0;
	WorldRect view = settings.currentView;
	extendedPicFrame.bottom += PicFrameExtension();
	combinedImage = GetColorImageDIB(DrawCombinedImageWithDate,this,view,extendedPicFrame,&err);
	if(combinedImage && !err)
	{	// write it out to a file in the frames folder
		char filePathName[256];
		char formatStr[256];
		GetFramePathFormatStr(fMoviePicsPath,formatStr);
		sprintf(filePathName,formatStr,movieFrameIndex);
		err = SaveDIB(combinedImage,filePathName);
	}
	if(combinedImage)
	{
		DestroyDIB(combinedImage);
		combinedImage = nil;
	}
#endif

	++movieFrameIndex;
	
	
	if (err) 
		printError("GNOME was unable to save the movie frame.");
	return err;
}

////////////////////////////////////////////////////////////////////////////////
// mac resource files have a size limit so changed to save the frames to a folder
OSErr TModel::OpenMovieFile ()
{
	OSErr		err = noErr;
	MySFReply 	reply;
	//char		name [255];
	static char	name [255];
	static long index = 1;
	Point 		where = CenteredDialogUpLeft(M55);
	char fullPath[256];

	movieFrameIndex = 0;

	if (!gCommandFileRun)	// command file sets fMoviePath from file
	{
		if (index==1) strcpy (name, "SpillMovie.mov");
#if TARGET_API_MAC_CARBON
		err = AskUserForSaveFilename("SpillMovie.mov",fullPath,".mov",TRUE);
		if (err) return USERCANCEL;
#else
#ifdef MAC
		sfputfile(&where, "Save movie as:", name, (DlgHookUPP)0, &reply);
#else
		sfpputfile(&where, "VOM.", name, (DlgHookUPP)0, &reply,
				   M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
#endif
		if (!reply.good) return USERCANCEL;
#endif
	}
	
#ifdef MAC
	short		vRefNum;
	long		parDirID;
	
	// get the dir-spec info for preferences folder
	err = FindFolder (kOnSystemDisk, kPreferencesFolderType, kCreateFolder, &vRefNum, &parDirID);
	
	if (!err)
	{
		char	folderPath [255];
		long	prefsDirID;
		OSType folderType;
		FSSpec dirFSpec;
		
		strcpy (folderPath, "GNOME Movie Frames");
		my_c2pstr (folderPath);
		
		err = DirCreate (vRefNum, parDirID, (ConstStr255Param) folderPath, &prefsDirID);
		if (!err || err == dupFNErr)
		{
			PathNameFromDirID (prefsDirID, vRefNum, folderPath);
			my_p2cstr ((StringPtr) folderPath);
			if (err == dupFNErr)
				strcat (folderPath, "GNOME Movie Frames:");
			err = noErr;
		}
		//mac needs a file name
		{
			char test[256];
			sprintf(test,"%sframe000.pic",folderPath);
			if(FileExists(0,0,test))
	#ifdef MAC
				err = DeleteFilesInFolderSpecifiedByFullPathWithDelimiter(test); // Mac needs a file name
	#else
				err = DeleteFilesInFolderSpecifiedByFullPathWithDelimiter(folderPath); // IBM needs a folder name
	#endif
		}
		//err = DeleteFilesInFolderSpecifiedByFullPathWithDelimiter(char* path)
		if(!err)
		{
			char movieFilePath[256],shortFileName[64];
			if(!gCommandFileRun)
			{
#if MACB4CARBON
				my_p2cstr(reply.fName);
				GetFullPath (reply.vRefNum, 0, (char *) "", movieFilePath);
				strcat (movieFilePath, ":");
				strcat (movieFilePath, (char *) reply.fName);
				strcpy(fMoviePath,movieFilePath); // record the file name chosen in this  structure
				strcpy (name, (char *) reply.fName);	// save users name
#else
				strcpy(movieFilePath,fullPath);
				strcpy(fMoviePath,movieFilePath); // record the file name chosen in this  structure
				SplitPathFile(fullPath,shortFileName);
				strcpy (name, shortFileName);	// save users name
#endif
			}
			strcpy(fMoviePicsPath,folderPath); // record the folder name chosen in this  structure
		}
	}
#else
	{
		char folderPath[255];
		char* p;
		long parentDirID = 0;
		long dirID = 0, len, nBufferLen = 255;
		char delimStr[32] ={DIRDELIMITER,0};
		
		my_p2cstr((StringPtr)reply.fName);

		// make a directory in the location where the movie will be made
		// it will be easiest to put the files in the preferences folder
		// since then we don't have to delete the folder when we are done, just the file in it

		len = GetTempPath(nBufferLen,folderPath);	// may need to try GNOME folder if this fails
		//GetWindowsDirectory(folderPath, 255);
		if (folderPath[strlen(folderPath) - 1] != '\\')
			strcat(folderPath, "\\"); // add backslash
		strcat(folderPath, "GNOME Movie Frames");
		err = AddFolderIfMissing(0,0,folderPath,&dirID);
		strcat(folderPath,delimStr);
		if(!err) (void) DeleteFilesInFolder(0,0,folderPath); // get rid of any previous frames
		
		if(!err)
		{
			if (!gCommandFileRun)
			{
				strcpy(fMoviePath,reply.fName); // record the file name chosen in this  structure
				SplitPathFile ((char*)reply.fName, name);	// remember users name
			}
			strcpy(fMoviePicsPath,folderPath); // record the folder name chosen in this  structure
		}
	}
#endif 

	if (err) printError("GNOME was unable to open the movie file.");
	index++;
	return err;
}



////////////////////////////////////////////////////////////////////////////////

// mac resource files have a size limit so changed to save the frames to a folder
OSErr TModel::CloseMovieFile ()
// closes the movie pictures resource file and generates the movie.
// NOTE:	make-movie flag is reset by this routine to prevent accidental overwriting of movie, STH
{
	OSErr	err = noErr;
	char formatStr[256];
	char folderName[256];
	char path[256];
	long startIndex,stopIndex;
	Rect bitMapRect = MapDrawingRect();
	bitMapRect.bottom += PicFrameExtension();
	MyOffsetRect(&bitMapRect,-bitMapRect.left,-bitMapRect.top);
	
	// make sure the files exist for the startIndex and stopIndex
	GetFramePathFormatStr(fMoviePicsPath,formatStr);
	
	// make sure stopIndex file exists
	for(stopIndex = movieFrameIndex-1;stopIndex>= 0;stopIndex--) 
	{
		sprintf(path,formatStr,stopIndex);
		if(FileExists(0, 0, path)) break;
	} 
	
	// make sure startIndex file exists
	for(startIndex = 0;startIndex <= stopIndex;startIndex++) 
	{
		sprintf(path,formatStr,startIndex);
		if(FileExists(0, 0, path)) break;
	} 

	if(stopIndex >= startIndex)
	{
		// call code to make movie from files in the movie folder
		///////////
	#ifdef IBM
		(void)hdelete(0,0,fMoviePath);// make sure file does not already exist - I think CreateMovieFile takes care of this (test on PC)
	#endif
		SetWatchCursor();
		err = PICStoMovie(fMoviePath,formatStr,startIndex,stopIndex,bitMapRect.top,bitMapRect.left,bitMapRect.bottom,bitMapRect.right);
		if (err) {printError("GNOME was unable to make the movie."); return err;}
	}

	// delete the files after the movie is made
	//////////
	strcpy(folderName,fMoviePicsPath);		
	sprintf(path,formatStr,startIndex);
	if(FileExists(0, 0, path))
	{
#ifdef MAC
		err = DeleteFilesInFolderSpecifiedByFullPathWithDelimiter(path); // Mac needs a file name
#else
		err = DeleteFilesInFolderSpecifiedByFullPathWithDelimiter(folderName); // IBM needs a folder name
#endif
	}
	
	bMakeMovie = false;
	
	if (err) printError("GNOME was unable to make the movie.");
	return err;
}

