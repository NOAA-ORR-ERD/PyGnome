
#include "Basics.h"
#include "Cross.h"
#include "TimUtils.h"
#include "EditCDOGProfilesDialog.h"
#include "Contdlg.h"
#include "NetCDFMover.h"
#include "netcdf.h"

#define kMaxNumCDOGInputFiles 500
/////////////////

typedef struct {

	Boolean  waitingForCDOG;
	#ifdef MAC
		ProcessSerialNumber	 processSN;
	#endif

} CDOGInfo;

CDOGInfo gCDOG;

Boolean gCDOGVersion = false; // set to true if CDOG is present

double depthCF[] = {1,.3048,1.8288};


//////

Boolean WaitingForCDOG(void)
{
	return gCDOGVersion && gCDOG.waitingForCDOG;
}

Boolean CDOGAvailable(void)
{
	return gCDOGVersion;
}

void GetCDogFolderPathWithDelimiter(char *path) {
	char  cDogFolderPath[256] = ":CDOG:";
	// find the path to CDOG
	ResolvePath(cDogFolderPath);
	strcpy(path,cDogFolderPath);
}

OSErr LaunchCDOG(void) 
{
	OSErr err = 0;
	#ifdef MAC
		ProcessSerialNumber processSN;
		//char  cDogPath[256] = ":CDOG:CDOG.app";
		char  cDogPath[256] = ":CDOG:CDOG";
	#else
		char  cDogPath[256] = ":CDOG:CDOG.exe";
	#endif
		// find the path to CDOG
		ResolvePath(cDogPath);
		if(!FileExists(0,0,cDogPath)) {
			printNote("Could not locate the CDOG application.  Make sure it is in a folder called CDOG inside of the GNOME directory.");
			return -1;
		}
	
	#ifdef MAC
		err = TryToLaunch(cDogPath,TRUE,&processSN);
	#else
		err = TryToLaunch(cDogPath,TRUE);
		// note we could try using WaitForInputIdle() here if we wanted to wait here until CDOG was done
	#endif

	if(!err) 
	{
		// record the fact that we are waiting on CDOG
		#ifdef MAC
			gCDOG.waitingForCDOG = TRUE;
			gCDOG.processSN = processSN;
		#else
			// code goes here, what do we do on the IBM ?
		#endif
	}
	else
	{
		printNote("An error occurred trying to launch CDOG.");
	}
	return err;
	
}

////////

OSErr AboutCDOGInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	ShowHideDialogItem(dialog, ABOUTCDOGCANCEL, false); 	// may want a cancel to disallow use of CDOG

	return 0;
}


short AboutCDOGClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	switch (itemNum) {
		case ABOUTCDOGCANCEL: return ABOUTCDOGCANCEL;

		case ABOUTCDOGOK:
			return ABOUTCDOGOK;

	}

	return 0;
}

short ShowCDOGInfo()
{
	short item = MyModalDialog(ABOUT_CDOG_DLGID, mapWindow, 0, AboutCDOGInit, AboutCDOGClick);
	return item;
}
/////////////////////////////////////////////////
OSErr DeleteFilesInFolderSpecifiedByFullPathWithDelimiter(char* path)
{
	FSSpec spec;
	OSErr err = 0;
	/////
	memset(&spec, 0,sizeof(spec));
#ifdef MAC
	my_c2pstr(path);
	err = FSMakeFSSpec(0,0,(StringPtr)path,&spec);
	my_p2cstr((StringPtr)path);
	if(err) return err;
#endif
	err = DeleteFilesInFolder(spec.vRefNum,spec.parID,path);
	
	return err;

}

void DoRunCDOGMenuItem(void) 
{
	OSErr err = 0;
	// perhaps we check on stuff here before launching...
	char cdogFolderPathWithDelimiter[256];
	char cdogOutputFolderPathWithDelimiter[256];
	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogOutputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"output",DIRDELIMITER);
	// get rid of the old cdog output before running
	{
		char test[256];
		sprintf(test,"%sSurfaceOil.dat",cdogOutputFolderPathWithDelimiter);
		if(FileExists(0,0,test))
#ifdef MAC
			err = DeleteFilesInFolderSpecifiedByFullPathWithDelimiter(test); // Mac needs a file name
#else
			err = DeleteFilesInFolderSpecifiedByFullPathWithDelimiter(cdogOutputFolderPathWithDelimiter); // IBM needs a folder name
#endif
	}
	err = LaunchCDOG();
}


/////////////////////////////////////////////////
void SetCDOGMenuItems(void)
{
	// nothing to do right now...
	
	//MenuHandle mHdl = GetMHandle(CDOGMENU);
	//if(!mHdl) return;
	MenuHandle mh = GetMenuHandle(CDOGMENU);
	if(mh)
	{
		//float arrowDepth;
		long mode = model->GetModelMode();
		//PtCurMap *map = GetPtCurMap();	
		Boolean enableOutputItems = (mode == ADVANCEDMODE/* && (model->ThereIsA3DMover(&arrowDepth))*/);
		Boolean enableOutputItems2 = (mode == ADVANCEDMODE /*&& (map && map->ThereIsADispersedSpill())*/);
		OSSMEnableMenuItem(mh, MODELSETTINGSITEM, enableOutputItems);
		OSSMEnableMenuItem(mh, SETMAPBOXITEM, enableOutputItems);
		OSSMEnableMenuItem(mh, CDOGSETTINGSITEM, enableOutputItems);
		OSSMEnableMenuItem(mh, EXPORTCDOGFILESITEM, enableOutputItems2);
		OSSMEnableMenuItem(mh, RUNCDOGITEM, enableOutputItems2);
		OSSMEnableMenuItem(mh, INPUTCDOGFILESITEM, enableOutputItems2);
		OSSMEnableMenuItem(mh, ABOUTCDOGITEM, enableOutputItems2);
	}
	
}
/////////////////////////////////////////////////


Boolean DoCDOGMenu(short menuCodedItemID, Boolean dontDoIt)
{ 	// does menu item (if dontDoIt is not true) and returns true if it is a CDOG item 
	// returns false it not a CDOG menuitem
	OSErr err = 0;
	if(!gCDOGVersion) return false;
	switch(menuCodedItemID)
	{
		case MODELSETTINGSITEM:
			if(!dontDoIt) {
				err = ModelSettingsDialog(false);
			}
			break;
		case SETMAPBOXITEM:
			if(!dontDoIt) {
				CreateMapBox();
			}
			break;
		case CDOGSETTINGSITEM:
			if(!dontDoIt) {
				CDOGLEList *thisCDOGLEList = GetCDOGSpill();
				if (thisCDOGLEList)
				{
					(void)CDOGSpillSettingsDialog(thisCDOGLEList);
					return true;
				}

				else 
				{
					err = CreateCDOGLESet();
					if (err) printError("Unable to create CDOG spill");
				}
			}
			break;
		case EXPORTCDOGFILESITEM:
			if (!dontDoIt) {
				CDOGLEList *thisCDOGLEList = GetCDOGSpill();
					if (thisCDOGLEList)
					{
						// deletefilesinfolder ??
						DisplayMessage("Exporting GNOME data to CDOG input folder");
						// should check everything has been set first
						MySpinCursor();
						if (thisCDOGLEList->GetMethodOfDeterminingHydrodynamics()==1)	// hand set constant, steady state profiles
						{
							if (err = thisCDOGLEList->ExportProfilesToCDOGInputFolder()) {printError("Error outputting profiles to CDOG input folder"); return true;}
						}
						else if (thisCDOGLEList->GetMethodOfDeterminingHydrodynamics()==2)	// data from netCDF model
						{
							long numDepthLevels=0;
							// check ThereIsA3DMover()
							TCurrentMover *possible3DMover = model->GetPossible3DCurrentMover();	// probably a better way to do this
							if (!possible3DMover) {printError("Please load a 3D current mover or select a different circulation option. Error outputting data to CDOG input folder."); return true;}
							if (possible3DMover->IAm(TYPE_NETCDFMOVER)) numDepthLevels = (dynamic_cast<NetCDFMover *>(possible3DMover))->GetNumDepthLevels();
							else {printError("Only currents in NetCDF format are allowed");return true;}
							if (numDepthLevels<=1)
							{
								printError("Please check your hydrodynamic file. There is a problem with the depth levels. Error outputting files to CDOG input folder"); return true;
							}
							MySpinCursor();
							if (err = thisCDOGLEList->ExportProfilesToCDOGInputFolder(possible3DMover)) {printError("Error outputting profiles to CDOG input folder");}
							if (err = thisCDOGLEList->ExportProfilesAsNetCDF(possible3DMover)) {printError("Error outputting profiles to netcdf file in CDOG input folder"); return true;}
							MySpinCursor();
						}
						//else if (thisCDOGLEList->GetMethodOfDeterminingHydrodynamics()==3)	// profiles already in CDOG input folder
						else if (thisCDOGLEList->GetMethodOfDeterminingHydrodynamics()==4)	// move hydrodynamic from other locations to CDOG input folder
						{
							if (thisCDOGLEList->GetMethodOfDeterminingTempSal()==2)	// select temp/sal files
							{
								if(!thisCDOGLEList->TemperatureProfilePathSet()) {printNote("Temperature file has not been selected.");	return true;}
								if (!thisCDOGLEList->SalinityProfilePathSet()) {printNote("Salinity file has not been selected.");	return true;}
								if (err = thisCDOGLEList->CopyTemperatureFilesToCDOGInputFolder()) {printError("Error copying temperature files to CDOG input folder"); return true;}
								if (err = thisCDOGLEList->CopySalinityFilesToCDOGInputFolder()) {printError("Error copying salinity files to CDOG input folder"); return true;}
							}
							if (thisCDOGLEList->GetMethodOfDeterminingCurrents()==1)
							{
								if (err = thisCDOGLEList->CopyHydrodynamicsFilesToCDOGInputFolder()) {printError("Error copying hydrodynamics files to CDOG input folder"); return true;}
							}
							//if (thisCDOGLEList->GetMethodOfDeterminingCurrents()==2)	// option to output u,v only from Gnome current data?
							//if (thisCDOGLEList->GetMethodOfDeterminingCurrents()==3)	// u,v already in CDOG input folder
						}
						if (err = thisCDOGLEList->ExportCDOGGeninf()) {printError("Unable to export CDOG parameters");return true;}	
						// also need to export grid I think?? or would it be with hydrodynamic info?
						if (err = thisCDOGLEList->ExportCDOGZeroWind()) {printError("Unable to create CDOG zero wind file");return true;}	
						DisplayMessage(0);
						MySpinCursor();
					}
					else printNote("There is no CDOG spill");
			}
			break;
		case RUNCDOGITEM:
			if (!dontDoIt) {
				DoRunCDOGMenuItem();
			}
			break;
		case INPUTCDOGFILESITEM:
			if (!dontDoIt) {
				CDOGLEList *thisCDOGLEList = GetCDOGSpill();
				if (thisCDOGLEList) 
				{
					DisplayMessage("Importing CDOG data");
					//MySpinCursor();	//might want to put this inside the read function, doesn't do anything here
					// also for reading save files...
					err = thisCDOGLEList->ReadCDOGOutputFiles();
					if (err) {printError("Error importing CDOG data"); return true;}
					DisplayMessage(0);
				}
				else printNote("You must create a CDOG spill before importing the data");
				// might create a CDOG spill if there isn't one
			}
			break;
		case ABOUTCDOGITEM:
			if (!dontDoIt) {
				ShowCDOGInfo();
			}
			break;
	
		default: return false;
	}
	return true; // it was handled
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

void CDOGTasks(void) 
{
	OSErr err = 0;
	if(!gCDOGVersion) return; // CDOG not present
	if(gCDOG.waitingForCDOG) {  
		// Hmmm... check to see if the CDOG process is still alive
		// should we wait for an event for when we come forward again ..
		// or just monitor for CDOG to quit ?
		// Maybe we need to do both.
#ifdef MAC
		ProcessInfoRec  info;
		memset(&info,0,sizeof(info));
		info.processInfoLength = sizeof(ProcessInfoRec);
		err = GetProcessInformation(&gCDOG.processSN,&info);
#endif
		if(err) {
			// CDOG is no longer running
			memset(&gCDOG,0,sizeof(gCDOG));
			// OK when CDOG quits... try to import the spill automatically ?
			printNote("Use your imagination...pretend we now import the CDOG LEs.");
		}
	}
	//////
	SetCDOGMenuItems();
}

/////////////////////////////////////////////////


void InitCDOGMenu(void)
{
	/// check if diagnostic version
	#ifdef MAC
		//char  cDogPath[256] = ":CDOG:CDOG.app";
		char  cDogPath[256] = ":CDOG:CDOG";
	#else
		char  cDogPath[256] = ":CDOG:CDOG.exe";
	#endif

#if TARGET_API_MAC_CARBON
	{	// on the MAC, we need to chop the package contents to get the folder the user "sees" as the MARPLOT "folder"
		char path[256];
		OSStatus err = 0;
		FSSpec spec;
		long len;
	if (err = hgetvol(TATFolder, &TATvRefNum, &TATdirID))	// I seem to need to get this again...
		{ TechError("CommonInit()", "hgetvol()", err); return; }
	
		GetFullPath(TATvRefNum, TATdirID, "", path);
		if(IsBundledApplication(path)) {
			ChopExtensionAndHiddenPackageFolders(path);
			// hmmm... we want the parent of the .app folder, try this...
			strcat(path,".app");
			my_c2pstr(path);
			err = FSMakeFSSpec(0, 0, (StringPtr)path, &spec);
			if(err) { TechError("CommonInit()", "FSMakeFSSpec()", err); return; }
			
			TATdirID = spec.parID;
			// for debugging, recontruct the path to make sure we've got it correct
			GetFullPath(TATvRefNum,TATdirID,"",path);
			len = strlen(path);
			if(len > 0 && path[len-1] == DIRDELIMITER)
				path[len-1] = 0;// chop this delimiter
			strcat(path,cDogPath);
			strcpy(cDogPath,path);
		}
	}
#else

	// find the path to CDOG
	ResolvePath(cDogPath);
#endif
	if(FileExists(0,0,cDogPath)) 
	{
		#ifdef MAC
			gCDOGVersion =true;
			GetAndInsertMenu(CDOGMENU, 0);
			//DrawMenuBar(); on the MAC, Draw MenuBar will be called by the calling function
		#else
		{
			MenuHandle hMenu = LoadMenu (hInst,"CDOG");
			if(hMenu) {
				gCDOGVersion = true;
				AppendMenu (GetMenu(hMainWnd), MF_POPUP,(UINT)hMenu ,"&CDOG");
				DrawMenuBar(hMainWnd);
			}
		}
		#endif
	}
}

////////////////

OSErr GetFilePath(char *path,char *fileName,short fileType)
{ // Note: returns USERCANCEL when user cancels
	//char 			path [256];
	short resID = M38h; //temperature default
	Point 			where;
//#if !TARGET_API_MAC_CARBON
	OSType 		typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 		reply;
//#endif
	OSErr			err = noErr;
	char 	tempPath[256];
	char prompt[256] = ""; // JLM 12/30/98
	
	if (fileType == 1) //temperature
	{
		resID = M38h;
	}
	if (fileType == 2) //salinity
	{
		resID = M38i;
	}
	if (fileType == 3) //currents
	{
		resID = M38j;	
	}
	else
	{
		resID = M38g;
		paramtext(prompt,"","","");
	}
		
	where = CenteredDialogUpLeft (resID);
		
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, resID, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);

		strcpy(tempPath, path);
		SplitPathFile(tempPath, fileName);
#else
	sfpgetfile(&where, "",
				(FileFilterUPP)0,
				-1, typeList,
				(DlgHookUPP)0,
				&reply, resID,
				(ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	if (!reply.good)  return USERCANCEL;
	
	my_p2cstr(reply.fName);
#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
		strcpy(fileName,(char*) reply.fName);
#else
		strcpy(path, reply.fName);
		strcpy(tempPath, path);
		SplitPathFile(tempPath, fileName);
#endif
#endif

	return err;
}



enum { I_CDOG_LEFIRSTLINE = 1, I_CDOG_ACTIVE, I_CDOG_WINDAGE, I_CDOG_LESHOWHIDE, I_CDOG_LERELEASE_TIMEPOSITION, I_CDOG_UNSPECIFIEDLINE, I_CDOG_LERELEASE_MASSBALANCE, I_CDOG_OILDISCHARGERATE, I_CDOG_GASDISCHARGERATE, I_CDOG_GOR };

//static CDOGParameters sDialogCDOGParameters;
static WorldRect sMapBoxBounds;
static PopInfoRec MapBoxPopTable[] = {
		{ MAPBOX_DLGID, nil, MAPBOXTOPLATDIR, 0, pNORTHSOUTH1, 0, 1, FALSE, nil },
		{ MAPBOX_DLGID, nil, MAPBOXLEFTLONGDIR, 0, pEASTWEST1, 0, 1, FALSE, nil },
		{ MAPBOX_DLGID, nil, MAPBOXBOTTOMLATDIR, 0, pNORTHSOUTH2, 0, 1, FALSE, nil },
		{ MAPBOX_DLGID, nil, MAPBOXRIGHTLONGDIR, 0, pEASTWEST2, 0, 1, FALSE, nil },
	};

void ShowHideRegion(DialogPtr dialog)
{
	Boolean show  = GetButton (dialog, MAPBOXBOUNDS); 

	SwitchLLFormatHelper(dialog, MAPBOXTOPLATDEGREES, MAPBOXDEGREES, show);
	SwitchLLFormatHelper(dialog, MAPBOXBOTTOMLATDEGREES, MAPBOXDEGREES, show); 
	
	ShowHideDialogItem(dialog, MAPBOXDEGREES, show); 
	ShowHideDialogItem(dialog, MAPBOXDEGMIN, show); 
	ShowHideDialogItem(dialog, MAPBOXDMS, show); 

	ShowHideDialogItem(dialog, MAPBOXTOPLATLABEL, show); 
	ShowHideDialogItem(dialog, MAPBOXLEFTLONGLABEL, show); 
	ShowHideDialogItem(dialog, MAPBOXBOTTOMLATLABEL, show); 
	ShowHideDialogItem(dialog, MAPBOXRIGHTLONGLABEL, show); 
}

short MapBoxClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
#pragma unused (data)
	WorldPoint p, p2;
	Boolean changed, tempUseBounds;
	WorldRect origBounds = emptyWorldRect;
	OSErr err = 0;

	StandardLLClick(dialog, itemNum, MAPBOXTOPLATDEGREES, MAPBOXDEGREES, &p, &changed);
	StandardLLClick(dialog, itemNum, MAPBOXBOTTOMLATDEGREES, MAPBOXDEGREES, &p2, &changed);

	/*{	// code goes here
		PtCurMap *map = GetPtCurMap();	// still could be 2D...
		if (map)
		{
			origBounds = map -> GetMapBounds();
		}
	}*/
	
	switch(itemNum)
	{
		case MAPBOX_OK:
			
			tempUseBounds = GetButton (dialog, MAPBOXBOUNDS);
			
			if(tempUseBounds)
			{
				//long oneSecond = (1000000/3600);
				// retrieve the extendedBounds
				if (err = EditTexts2LL(dialog, MAPBOXTOPLATDEGREES, &p, TRUE)) break;
				if (err = EditTexts2LL(dialog, MAPBOXBOTTOMLATDEGREES, &p2, TRUE)) break;

				// check extended bounds (oneSecond handles accuracy issue in reading from dialog)			
				/*if (p.pLat > origBounds.hiLat + oneSecond || p2.pLat < origBounds.loLat - oneSecond
					|| p.pLong < origBounds.loLong - oneSecond || p2.pLong > origBounds.hiLong + oneSecond)
				{
					printError("The map box cannot be greater than the map bounds."); 
					return 0; 
				}*/
				
				if (p.pLat < p2.pLat || p.pLong > p2.pLong)
				{
					printError("The map box bounds are not consistent (top < bot or left > right)."); 
					return 0; 
				}
				
				// just in case of round off
				/*p.pLat = _min(p.pLat,origBounds.hiLat);
				p.pLong = _max(p.pLong,origBounds.loLong);
				p2.pLat = _max(p2.pLat,origBounds.loLat);
				p2.pLong = _min(p2.pLong,origBounds.hiLong);*/
			}
	
			// restore to original bounds if uncheck box
			if (tempUseBounds)
			{
				sMapBoxBounds.loLat = p2.pLat;
				sMapBoxBounds.hiLat = p.pLat;
				sMapBoxBounds.loLong = p.pLong;
				sMapBoxBounds.hiLong = p2.pLong;
			}
			//else
				//sMapBoxBounds = origBounds;

			return MAPBOX_OK;
			
		case MAPBOX_CANCEL:
			return MAPBOX_CANCEL;
			break;
			
		case MAPBOXBOUNDS:
			// code goes here, decide what clicking box should mean
			ToggleButton(dialog, itemNum);
			ShowHideRegion(dialog);
			break;

		case MAPBOXDEGREES:
		case MAPBOXDEGMIN:
		case MAPBOXDMS:
				if (err = EditTexts2LL(dialog, MAPBOXTOPLATDEGREES, &p, TRUE)) break;
				if (err = EditTexts2LL(dialog, MAPBOXBOTTOMLATDEGREES, &p2, TRUE)) break;
				if (itemNum == MAPBOXDEGREES) settings.latLongFormat = DEGREES;
				if (itemNum == MAPBOXDEGMIN) settings.latLongFormat = DEGMIN;
				if (itemNum == MAPBOXDMS) settings.latLongFormat = DMS;
				//ShowHideRegion(dialog);
				SwitchLLFormatHelper(dialog, MAPBOXTOPLATDEGREES, MAPBOXDEGREES, true);
				SwitchLLFormatHelper(dialog, MAPBOXBOTTOMLATDEGREES, MAPBOXDEGREES, true); 
				LL2EditTexts(dialog, MAPBOXBOTTOMLATDEGREES, &p2);
				LL2EditTexts(dialog, MAPBOXTOPLATDEGREES, &p);
			break;

	}
	return 0;
}

OSErr MapBoxInit(DialogPtr dialog, VOIDPTR data)
{
	#pragma unused (data)
	WorldPoint wp;

	SetDialogItemHandle(dialog, MAPBOX_HILITE, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, MAPBOX_FROST1, (Handle)FrameEmbossed);

	RegisterPopTable (MapBoxPopTable, sizeof (MapBoxPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (MAPBOX_DLGID, dialog);
	
	SetButton (dialog, MAPBOXBOUNDS, true); // or should default be whole map...
	//wp.pLat = sMapBoxBounds.hiLat;
	//wp.pLong = sMapBoxBounds.loLong;
	wp.pLat = 	89*1e6;	//kWorldTop
	wp.pLong = 	-179*1e6;	//kWorldLeft;
	LL2EditTexts (dialog, MAPBOXTOPLATDEGREES, &wp);
	
	//wp.pLat = sMapBoxBounds.loLat;
	//wp.pLong = sMapBoxBounds.hiLong;
	wp.pLat = 	-89*1e6;	//kWorldBottom;
	wp.pLong = 	179*1e6;	//kWorldRight;
	LL2EditTexts (dialog, MAPBOXBOTTOMLATDEGREES, &wp);

	//ShowHideRegion(dialog);
	SwitchLLFormatHelper(dialog, MAPBOXTOPLATDEGREES, MAPBOXDEGREES, true);
	SwitchLLFormatHelper(dialog, MAPBOXBOTTOMLATDEGREES, MAPBOXDEGREES, true); 

	return 0;
}

OSErr MapBoxDialog(WorldRect *mapBoxBounds)
{
	short item;

	//if(info == nil) return -1;
	//sMapBoxBounds = *mapBoxBounds;

	item = MyModalDialog(MAPBOX_DLGID, mapWindow, 0, MapBoxInit, MapBoxClick);
	if (item == MAPBOX_OK) {
		*mapBoxBounds = sMapBoxBounds;
		model->NewDirtNotification(); 
	}
	if (item == MAPBOX_CANCEL) {return USERCANCEL;}
	return item == MAPBOX_OK? 0 : -1;
}
/////////////////////////////////////////////////
CDOGDiffusivityInfo sDialogDiffusivity;
OSErr CDOGDiffusivityInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	Float2EditText(dialog,CDOGHORIZDIFF, sDialogDiffusivity.horizDiff*10000, 2);
	Float2EditText(dialog,CDOGVERTDIFF, sDialogDiffusivity.vertDiff*10000, 2);
	Float2EditText(dialog,CDOGTIMESTEPINTERVAL, sDialogDiffusivity.timeStep/60., 2);	// minutes
	
	MySelectDialogItemText(dialog, CDOGTIMESTEPINTERVAL, 0, 255);

	return 0;
}


short CDOGDiffusivityClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	switch (itemNum) {
		case CDOGDIFFCANCEL: return CDOGDIFFCANCEL;

		case CDOGDIFFOK:
			sDialogDiffusivity.horizDiff = EditText2Float(dialog,CDOGHORIZDIFF)/10000.;	// m^2/s
			sDialogDiffusivity.vertDiff = EditText2Float(dialog,CDOGVERTDIFF)/10000.;	// m^2/s
			sDialogDiffusivity.timeStep = EditText2Float(dialog,CDOGTIMESTEPINTERVAL) * 60;	// minutes to seconds
			return itemNum;
			
		case CDOGTIMESTEPINTERVAL:
		case CDOGHORIZDIFF:
		case CDOGVERTDIFF:
			//CheckNumberTextItem(dialog, itemNum, TRUE);
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;

	}

	return 0;
}

OSErr CDOGDiffusivityDialog(CDOGDiffusivityInfo *diffusivityInfo,WindowPtr parentWindow)
{
	short item;
	sDialogDiffusivity = *diffusivityInfo;
	item = MyModalDialog(CDOGDIFFUSIVITYDLG, mapWindow, 0, CDOGDiffusivityInit, CDOGDiffusivityClick);
	if(item == CDOGDIFFCANCEL) return USERCANCEL; 
	model->NewDirtNotification();	// is this necessary ?
	if(item == CDOGDIFFOK) 
	{
		*diffusivityInfo = sDialogDiffusivity;
		return 0; 
	}
	else return -1;
}

static PopInfoRec HydrodynamicsPopTable[] = {
		{ CDOGHYDDLG, nil, CDOGHYDFOLDER, 0, pHYDRODYNAMICOPTIONS, 0, 1, FALSE, nil }
	};

CDOGHydrodynamicInfo sDialogHydrodynamics;
OSErr CDOGHydrodynamicsInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	RegisterPopUpDialog (CDOGHYDDLG, dialog);
	
	SetPopSelection (dialog, CDOGHYDFOLDER, 1);

	if (sDialogHydrodynamics.methodOfDeterminingCurrents==1)
	{
		ShowHideDialogItem(dialog, CDOGHYDFOLDERNAME, true); 
		mysetitext(dialog, CDOGHYDFOLDERNAME, sDialogHydrodynamics.hydrodynamicFilesFolderPath);
		SetPopSelection (dialog, CDOGHYDFOLDER, 1);
	}
	else if(sDialogHydrodynamics.methodOfDeterminingCurrents==2)
	{
		SetPopSelection (dialog, CDOGHYDFOLDER, 2);	// should save 
		ShowHideDialogItem(dialog, CDOGHYDFOLDERNAME, false); 
	}
	else	// first time
	{
		ShowHideDialogItem(dialog, CDOGHYDFOLDERNAME, false); 
		SetPopSelection (dialog, CDOGHYDFOLDER, 1);
	}
	return 0;
}


short CDOGHydrodynamicsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short	theType;
	long	menuID_menuItem;

	switch (itemNum) {
		case CDOGHYDCANCEL: return CDOGHYDCANCEL;

		case CDOGHYDOK:
			theType = GetPopSelection (dialog, CDOGHYDFOLDER);
			if (theType == 1)
				sDialogHydrodynamics.methodOfDeterminingCurrents = 1;	// select folder
			else if (theType == 2)
				sDialogHydrodynamics.methodOfDeterminingCurrents = 2;	// export Gnome currents
			return itemNum;
			
		case CDOGHYDFOLDER:
			PopClick(dialog, itemNum, &menuID_menuItem);
			theType = GetPopSelection (dialog, CDOGHYDFOLDER);
			if (theType == 1)
			{
				OSErr err = 0;
				CDOGHydrodynamicInfo hydrodynamicInfo;
				char hydName[32],hydFilePath[256],testFile[256];
				err = GetFilePath(hydFilePath,hydName,3);
				if (err) break;
				// just cut off when matches hydName...
				strncpy(hydrodynamicInfo.hydrodynamicFilesFolderPath,hydFilePath,strlen(hydFilePath)-strlen(hydName));
				hydrodynamicInfo.hydrodynamicFilesFolderPath[strlen(hydFilePath)-strlen(hydName)]=0;
				//Check that hydName is u00*.dat, v00*.dat, w00*.dat
				strcpy(testFile,hydrodynamicInfo.hydrodynamicFilesFolderPath);
				strcat(testFile,"U001.dat");
				if (!FileExists(0,0,testFile)) 
				{
					printNote("Current file names must conform to CDOG format - U001.dat, ..."); 
					break;
				}
				strcpy(sDialogHydrodynamics.hydrodynamicFilesFolderPath,hydrodynamicInfo.hydrodynamicFilesFolderPath);
				ShowHideDialogItem(dialog, CDOGHYDFOLDERNAME, true); 
				mysetitext(dialog, CDOGHYDFOLDERNAME, sDialogHydrodynamics.hydrodynamicFilesFolderPath);
			}
			else
			{
				ShowHideDialogItem(dialog, CDOGHYDFOLDERNAME, false); 
			}
			break;

	}

	return 0;
}

OSErr CDOGHydrodynamicsDialog(CDOGHydrodynamicInfo *hydrodynamicsInfo,WindowPtr parentWindow)
{
	short item;
	if(parentWindow == nil) parentWindow = mapWindow; // JLM 6/2/99, we need the parent on the IBM
	sDialogHydrodynamics = *hydrodynamicsInfo;

	PopTableInfo saveTable = SavePopTable();
	short j, numItems = 0;
	PopInfoRec combinedDialogsPopTable[10];
	// code to allow a dialog on top of another with pops
	for(j = 0; j < sizeof(HydrodynamicsPopTable) / sizeof(PopInfoRec);j++)
		combinedDialogsPopTable[numItems++] = HydrodynamicsPopTable[j];
	for(j= 0; j < saveTable.numPopUps ; j++)
		combinedDialogsPopTable[numItems++] = saveTable.popTable[j];
	
	RegisterPopTable(combinedDialogsPopTable,numItems);

	item = MyModalDialog(CDOGHYDDLG, parentWindow, 0, CDOGHydrodynamicsInit, CDOGHydrodynamicsClick);
	RestorePopTableInfo(saveTable);
	if(item == CDOGHYDCANCEL) return USERCANCEL; 
	model->NewDirtNotification();	// is this necessary ?
	if(item == CDOGHYDOK) 
	{	
		*hydrodynamicsInfo = sDialogHydrodynamics;
		return 0; 
	}
	else return -1;
}

/////////////////////////////////////////////////
Boolean sDialogOutputSubsurfaceFiles, sDialogOutputGasFiles;
OSErr CDOGOutputOptionsInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	SetButton (dialog, CDOGOUTPUTSUBSURFACEFILES, sDialogOutputSubsurfaceFiles);
	SetButton (dialog, CDOGOUTPUTGASFILES, sDialogOutputGasFiles);

	return 0;
}


short CDOGOutputOptionsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	switch (itemNum) {
		case CDOGOUTPUTOPTIONSCANCEL: return CDOGOUTPUTOPTIONSCANCEL;

		case CDOGOUTPUTOPTIONSOK:
			sDialogOutputSubsurfaceFiles = GetButton (dialog, CDOGOUTPUTSUBSURFACEFILES);
			sDialogOutputGasFiles = GetButton (dialog, CDOGOUTPUTGASFILES);
			return itemNum;
			
		case CDOGOUTPUTSUBSURFACEFILES:
		case CDOGOUTPUTGASFILES:
			ToggleButton(dialog, itemNum);
			//CheckNumberTextItem(dialog, itemNum, FALSE);
			break;

	}

	return 0;
}

OSErr CDOGOutputOptionsDialog(Boolean *outputSubsurfaceFiles,Boolean *outputGasFiles,WindowPtr parentWindow)
{
	short item;
	sDialogOutputSubsurfaceFiles = *outputSubsurfaceFiles;
	sDialogOutputGasFiles = *outputGasFiles;
	item = MyModalDialog(CDOGOUTPUTOPTIONSDLG, mapWindow, 0, CDOGOutputOptionsInit, CDOGOutputOptionsClick);
	if(item == CDOGOUTPUTOPTIONSCANCEL) return USERCANCEL; 
	model->NewDirtNotification();	// is this necessary ?
	if(item == CDOGOUTPUTOPTIONSOK) 
	{
		*outputSubsurfaceFiles = sDialogOutputSubsurfaceFiles;
		*outputGasFiles = sDialogOutputGasFiles;
		return 0; 
	}
	else return -1;
}

/////////////////////////////////////////////////
static PopInfoRec TempSalPopTable[] = {
		{ TEMPSALDLG, nil, TEMPSALFILENAME, 0, pTEMPSAL, 0, 1, FALSE, nil }
	};

static CDOGTempSalInfo sDialogTempSalInfo;
OSErr TempSalInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	//Float2EditText(dialog,TEMPSALFILENAME, model -> LEDumpInterval / 3600. , 2);
	//RegisterPopTable (TempSalPopTable, sizeof (TempSalPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (TEMPSALDLG, dialog);
	
	if (sDialogTempSalInfo.methodOfDeterminingTempSal==2)
	{
		ShowHideDialogItem(dialog, TEMPFILENAME, true); 
		ShowHideDialogItem(dialog, SALFILENAME, true); 
		mysetitext(dialog, TEMPFILENAME, sDialogTempSalInfo.temperatureFieldFilePath);
		mysetitext(dialog, SALFILENAME, sDialogTempSalInfo.salinityFieldFilePath);
		SetPopSelection (dialog, TEMPSALFILENAME, 2);
	}
	else
	{
		SetPopSelection (dialog, TEMPSALFILENAME, 1);	// should save 
		ShowHideDialogItem(dialog, TEMPFILENAME, false); 
		ShowHideDialogItem(dialog, SALFILENAME, false); 
	}
	//MySelectDialogItemText(dialog, TEMPSALFILENAME, 0, 255);

	return 0;
}


short TempSalClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	long	menuID_menuItem;

	switch (itemNum) {
		case TEMPSALCANCEL: return TEMPSALCANCEL;

		case TEMPSALOK:
			short	theType;
			char tempPath[kMaxNameLen], salPath[kMaxNameLen];
			theType = GetPopSelection (dialog, TEMPSALFILENAME);
			if (theType==2) 
			{
				sDialogTempSalInfo.methodOfDeterminingTempSal = theType;	
				//mygetitext(dialog, TEMPFILENAME, sDialogTempSalInfo.temperatureFieldFilePath, kMaxNameLen - 1);
				//mygetitext(dialog, SALFILENAME, sDialogTempSalInfo.salinityFieldFilePath, kMaxNameLen - 1);
				mygetitext(dialog, TEMPFILENAME,tempPath, kMaxNameLen - 1);
				if (tempPath[0]==0) {printNote("No temperature profile was selected"); break;}
				mygetitext(dialog, SALFILENAME, salPath, kMaxNameLen - 1);
				if (salPath[0]==0) {printNote("No salinity profile was selected"); break;}
				strcpy(sDialogTempSalInfo.temperatureFieldFilePath,tempPath);
				strcpy(sDialogTempSalInfo.salinityFieldFilePath,salPath);
			}
			return itemNum;
			
		case TEMPSALFILENAME:
			PopClick(dialog, itemNum, &menuID_menuItem);
			//CheckNumberTextItem(dialog, itemNum, TRUE);
			theType = GetPopSelection (dialog, TEMPSALFILENAME);
			if (theType == 2)
			{
				OSErr err = 0;
				CDOGTempSalInfo tempSalInfo;
				char tempName[32],salName[32];
				err = GetFilePath(tempSalInfo.temperatureFieldFilePath,tempName,1);
				if (err) break;
				err = GetFilePath(tempSalInfo.salinityFieldFilePath,salName,2);
				if (err) break;
				//tempSalInfo.methodOfDeterminingTempSal = 2;
				sDialogTempSalInfo.methodOfDeterminingTempSal = 2;
				//strcpy(sDialogTempSalInfo.temperatureFieldFilePath,tempSalInfo.temperatureFieldFilePath);
				//strcpy(sDialogTempSalInfo.salinityFieldFilePath,tempSalInfo.salinityFieldFilePath);
				ShowHideDialogItem(dialog, TEMPFILENAME, true); 
				ShowHideDialogItem(dialog, SALFILENAME, true); 
				//mysetitext(dialog, TEMPFILENAME, sDialogTempSalInfo.temperatureFieldFilePath);
				//mysetitext(dialog, SALFILENAME, sDialogTempSalInfo.salinityFieldFilePath);
				mysetitext(dialog, TEMPFILENAME, tempSalInfo.temperatureFieldFilePath);
				mysetitext(dialog, SALFILENAME, tempSalInfo.salinityFieldFilePath);
			}
			else
			{
				ShowHideDialogItem(dialog, TEMPFILENAME, false); 
				ShowHideDialogItem(dialog, SALFILENAME, false); 
			}

			break;

	}

	return 0;
}

OSErr TempSalDialog(CDOGTempSalInfo *tempSalInfo)
{
	short item;
	PopTableInfo saveTable = SavePopTable();
	short j, numItems = 0;
	PopInfoRec combinedDialogsPopTable[10];
	// code to allow a dialog on top of another with pops
	for(j = 0; j < sizeof(TempSalPopTable) / sizeof(PopInfoRec);j++)
		combinedDialogsPopTable[numItems++] = TempSalPopTable[j];
	for(j= 0; j < saveTable.numPopUps ; j++)
		combinedDialogsPopTable[numItems++] = saveTable.popTable[j];
	
	RegisterPopTable(combinedDialogsPopTable,numItems);

	sDialogTempSalInfo = *tempSalInfo;
	item = MyModalDialog(TEMPSALDLG, mapWindow, 0, TempSalInit, TempSalClick);
	RestorePopTableInfo(saveTable);
	if(item == TEMPSALCANCEL) return USERCANCEL; 
	model->NewDirtNotification();	
	if(item == TEMPSALOK) 
	{
		*tempSalInfo = sDialogTempSalInfo;
		return 0; 
	}
	else return -1;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

static PopInfoRec CDOGSpillPopTable[] = {
		{ CDOGSPILLDLG, nil, CDOGSPILLSLATDIR, 0, pNORTHSOUTH1, 0, 1, FALSE, nil },
		{ CDOGSPILLDLG, nil, CDOGSPILLSLONGDIR, 0, pEASTWEST1, 0, 1, FALSE, nil },
		//{ CDOGSPILLDLG, nil, CDOGSPILLPOLLUTANT, 0, pPOLLUTANTS, 0, 1, FALSE, nil },
		//{ CDOGSPILLDLG, nil, CDOGSPILLDRPOPUP, 0, pDISCHARGERATE, 0, 1, FALSE, nil },
		{ CDOGSPILLDLG, nil, CDOGSPILLSTARTMONTH, 0, pMONTHS, 0, 1, FALSE, nil },
		{ CDOGSPILLDLG, nil, CDOGSPILLSTARTYEAR, 0, pYEARS, 0, 1, FALSE, nil },
		{ CDOGSPILLDLG, nil, CDOGSPILLENDMONTH, 0, pMONTHS2, 0, 1, FALSE, nil },
		{ CDOGSPILLDLG, nil, CDOGSPILLENDYEAR, 0, pYEARS2, 0, 1, FALSE, nil },
		{ CDOGSPILLDLG, nil, CDOGSPILLDEPTHUNITS, 0, pDEPTHUNITS2, 0, 1, FALSE, nil }
	};

CDOGLEList* gCDOGLEListInDialog = 0;
DepthValuesSetH gEditDepthLevels = 0;
DischargeDataH gEditDischargeTimes = 0;
static CDOGParameters sCDOGParameters;
static CDOGSpillParameters sCDOGSpillParameters;
static LESetSummary sCdogDialogSharedSet;
static WindageRec sWindageInfo;
//static Seconds sTimeStep;
static CDOGHydrodynamicInfo sHydrodynamicInfo;
static CDOGDiffusivityInfo sDiffusivityInfo;
static CDOGTempSalInfo	sTempSalInfo;
static Boolean sOutputSubsurfaceFiles, sOutputGasFiles;
static DepthValuesSetH sdvals=0;
static DepthValuesSetH sEditDepthVals=0;
static DischargeDataH sDischargeData=0;
//static float sLengthOfGridCellInMeters;
//static long sNumGridCells;
static Boolean sHydrogenSulfide;
static CDOGUserUnits sCDOGUserUnits;
static short sDepthUnits;
static short sMethodOfDeterminingHydrodynamics;

/*void EnableCDOGEndTime(DialogPtr dialog, Boolean bEnable)
{
	Boolean show  = bEnable; //JLM
	CDOGSpillPopTable[6].bStatic = !bEnable;
	CDOGSpillPopTable[7].bStatic = !bEnable;
	
	ShowHideDialogItem(dialog, CDOGSPILLENDMONTH, show); 
	ShowHideDialogItem(dialog, CDOGSPILLENDYEAR, show); 
	
	ShowHideDialogItem(dialog, CDOGSPILLENDDAY, show);
	ShowHideDialogItem(dialog, CDOGSPILLENDHOURS, show);
	ShowHideDialogItem(dialog, CDOGSPILLENDMINUTES, show);

	ShowHideDialogItem(dialog, CDOGSPILLENDTIMECOLON, show);
	ShowHideDialogItem(dialog, CDOGSPILLENDTIMELABEL, show);

	if(show)
	{
		PopDraw(dialog, CDOGSPILLENDMONTH);
		PopDraw(dialog, CDOGSPILLENDYEAR);
	}
	
	EnableTextItem(dialog, CDOGSPILLENDDAY, bEnable);
	EnableTextItem(dialog, CDOGSPILLENDHOURS, bEnable);
	EnableTextItem(dialog, CDOGSPILLENDMINUTES, bEnable);
}*/

/*void ShowHideCDOGSpillDialogItems(DialogPtr dialog)
{
	//EnableCDOGEndTime (dialog, GetButton (dialog, CDOGSPILLWANTENDTIME));
	
	SwitchLLFormat(dialog, CDOGSPILLSLATDEGREES, CDOGSPILLDEGREES); // do it 2nd to set focus

}*/
			
/*void MyDisposeHandle(Handle *g)
{
	if(*g)DisposeHandle(*g);
	*g = 0;
}*/

void MySetHandle(Handle *h, Handle q)
{
	if(*h != 0)MyDisposeHandle((Handle *)h);
	*h = q;
}

short CDOGSpillClick (DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	Boolean changed;
	long menuID_menuItem;
	WorldPoint p;
	OSErr err = 0;
	char errStr[256] = "";
	//long amountMin = 0, amountMax;
	//double gasOilRatio = 200.;
	Boolean haveOpenWizFile = (model->fWizard && model->fWizard->HaveOpenWizardFile());
	DischargeDataH tempDialogDischargeData = 0;
	short depthUnits;
	
	LESetSummary tempSet;

	StandardLLClick(dialog, itemNum, CDOGSPILLSLATDEGREES, CDOGSPILLDEGREES, &p, &changed);

	switch (itemNum) {
		case CDOGSPILLOK:
			tempSet = sCdogDialogSharedSet;	// 6/22/00
			tempSet.numOfLEs = EditText2Float(dialog, CDOGSPILLNUMLES);
			//tempSet.pollutantType = GetPopSelection(dialog, CDOGSPILLPOLLUTANT);
			tempSet.pollutantType = OIL_CONSERVATIVE;
			//tempSet.totalMass = EditText2Float(dialog, CDOGSPILLGASOILRATIO);
			//tempSet.massUnits = GetPopSelection(dialog, CDOGSPILLDRPOPUP);
			tempSet.massUnits = CUBICMETERS;
			tempSet.totalMass = 500;	// will calculate from rates
			tempSet.z = EditText2Float(dialog, CDOGSPILLDEPTH);
			
			/////////////////////////////
			// enforce input limits on the amount 
			/////////////////////////////
			/*switch(tempSet.massUnits)
			{
				case GALLONS: amountMax = 20000000;break;
				case BARRELS: amountMax = 476000;break;
				case CUBICMETERS: amountMax = 76000;break;
				///
				case METRICTONS: amountMax = 70000;break;
				case SHORTTONS:  amountMax = 77000;break;
			 	case KILOGRAMS: amountMax = 70000000;break;
				default: amountMax =-1;break; // -1 means don't enforce a limit
			}
			if(tempSet.totalMass <= amountMin) strcpy(errStr,"The amount released must be greater than zero.");	
			else if(tempSet.totalMass > amountMax && amountMax > 0)
			{
				char unitsStr[64];
				GetLeUnitsStr(unitsStr,tempSet.massUnits);
				sprintf(errStr,"The amount released cannot exceed %ld %s.",amountMax,unitsStr);
			}
			
			if(errStr[0])
			{
				printError(errStr);
				MySelectDialogItemText(dialog, CDOGSPILLGASOILRATIO,0,100);
				break;
			}*/
			//tempSet.totalMass
			if(tempSet.numOfLEs > 50000) 
			{
				printError("The number of Splots cannot exceed 50000."); // CDOG limit
				break;
			}
			
			/////////////////////////////
			
			// JLM, 1/26/99
			// the user cannot set the depth field since it is not used
			// so for it to be zero here
			// code goes here, let user set depth 

			//tempSet.z  = 0;
			//
			// JLM 1/26/99, the density will be set by the oil type
			tempSet.density = GetPollutantDensity(tempSet.pollutantType); 
			//
			tempSet.ageInHrsWhenReleased = 0.;	// should always be zero
			//
//			tempSet.mass = VolumeMassToKilograms(tempSet.totalMass,tempSet.representativeLE.density,tempSet.massUnits)/ (float) tempSet.numOfLEs;
			/////////////////////////////////////////////////

				
			// get start release time
			tempSet.startRelTime = RetrievePopTime(dialog, CDOGSPILLSTARTMONTH,&err);
			if(err) break;
			
			// get end release time
			//tempSet.bWantEndRelTime = GetButton (dialog, CDOGSPILLWANTENDTIME);
			//tempSet.endRelTime = RetrievePopTime(dialog, CDOGSPILLENDMONTH,&err);
			//if(err) break;
			tempSet.endRelTime = tempSet.startRelTime + model->GetDuration();
			
			// code goes here, check if there are some discharge values set
			if (!gEditDischargeTimes || _GetHandleSize((Handle)gEditDischargeTimes)/sizeof(**gEditDischargeTimes)==0)
			{
				if (!sDischargeData)
				{
					printNote("The discharge rate has not been set.");
					break;
				}
			//also if variable discharge make sure there are more than 2
			// if constant discharge must have a duration, or continuous release
			}
			//tempSet.totalMass = (tempSet.endRelTime-tempSet.startRelTime)*oilDischargeRate;
			// code goes here, pull out all the unit conversion to separate routines
			// might want to do this calculation even if gEditDischargeTimes does not exist
			if (gEditDischargeTimes)
			{
				tempDialogDischargeData = gEditDischargeTimes;
			}
			else if (sDischargeData) tempDialogDischargeData = sDischargeData;
			else	// should be redundant but just in case
			{
				printNote("The discharge rate has not been set.");
				break;
			}
			
			{
				double totalMass=0, duration, gor;
				short dischargeType = sCDOGUserUnits.dischargeType, dischargeUnits = sCDOGUserUnits.dischargeUnits, gorUnits = sCDOGUserUnits.gorUnits;
				//long i, numDischarges = _GetHandleSize((Handle)gEditDischargeTimes)/sizeof(**gEditDischargeTimes);
				long i, numDischarges=0; 
				if (tempDialogDischargeData) numDischarges = _GetHandleSize((Handle)tempDialogDischargeData)/sizeof(**tempDialogDischargeData);
				if (numDischarges>1)
				{
					double dischargeTime1, dischargeRate1, dischargeTime2, dischargeRate2;
					for (i=0;i<numDischarges-1;i++)
					{
						//dischargeTime1 = INDEXH(gEditDischargeTimes,i).time;
						//dischargeRate1 = INDEXH(gEditDischargeTimes,i).q_oil;	// will need to deal with units, and gas vs oil
						//dischargeTime2 = INDEXH(gEditDischargeTimes,i+1).time;
						//dischargeRate2 = INDEXH(gEditDischargeTimes,i+1).q_oil;	// will need to deal with units, and gas vs oil
						
						dischargeTime1 = INDEXH(tempDialogDischargeData,i).time;
						dischargeRate1 = INDEXH(tempDialogDischargeData,i).q_oil;	// will need to deal with units, and gas vs oil
						dischargeTime2 = INDEXH(tempDialogDischargeData,i+1).time;
						dischargeRate2 = INDEXH(tempDialogDischargeData,i+1).q_oil;	// will need to deal with units, and gas vs oil
						
						if (dischargeType==1)	// input oil rate
						{	// if already in m3/s no issue, if in bopd convert
							if (dischargeUnits==2) {dischargeRate1 = dischargeRate1*(.158987/(3600*24));dischargeRate2 = dischargeRate2*(.158987/(3600*24));}
						}
						else	//input gas rate
						{	// if already in m3/s no issue, if in mscf convert
							//gor = INDEXH(gEditDischargeTimes,i).q_gas;
							gor = INDEXH(tempDialogDischargeData,i).q_gas;
							if (dischargeUnits==1 && gorUnits==1) {dischargeRate1 = dischargeRate1 / gor; dischargeRate2 = dischargeRate2 / gor;}
							if (dischargeUnits==2 && gorUnits==3) {dischargeRate1 = dischargeRate1 / (gor*28.32/.158987); dischargeRate2 = dischargeRate2 / (gor*28.32/.158987);}
							if (dischargeUnits==2 && gorUnits==2) {dischargeRate1 = dischargeRate1 / (gor*.02832/.158987); dischargeRate2 = dischargeRate2 / (gor*.02832/.158987);}
						}
						duration = 3600*(dischargeTime2 - dischargeTime1);
						totalMass = totalMass + ((dischargeRate1+dischargeRate2) / 2.) * duration;
					}
					
				}
				else if (numDischarges == 1)
				{
					if (sCDOGParameters.isContinuousRelease)
					{
						if (dischargeType==1)	// input oil rate
						{
							//totalMass = model->GetDuration() * INDEXH(gEditDischargeTimes,0).q_oil;
							totalMass = model->GetDuration() * INDEXH(tempDialogDischargeData,0).q_oil;
							if (dischargeUnits==2) {totalMass = totalMass*(.158987/(3600*24));}
						}
						else
						{
							//gor = INDEXH(gEditDischargeTimes,0).q_gas;
							gor = INDEXH(tempDialogDischargeData,0).q_gas;
						//	if (dischargeUnits==1 && gorUnits==1) {dischargeRate1 = dischargeRate1 / gor; dischargeRate2 = dischargeRate2 / gor;}
						//	if (dischargeUnits==2 && gorUnits==3) {dischargeRate1 = dischargeRate1 / (gor*28.32/.158987); dischargeRate2 = dischargeRate2 / (gor*28.32/.158987);}
						//	if (dischargeUnits==2 && gorUnits==2) {dischargeRate1 = dischargeRate1 / (gor*.02832/.158987); dischargeRate2 = dischargeRate2 / (gor*.02832/.158987);}
							totalMass = model->GetDuration() * INDEXH(tempDialogDischargeData,0).q_oil;
						}
					}
					else
					{
						if (dischargeType==1)	// input oil rate
						{
							//totalMass = sCDOGParameters.duration * INDEXH(gEditDischargeTimes,0).q_oil;
							totalMass = sCDOGParameters.duration * INDEXH(tempDialogDischargeData,0).q_oil;
							if (dischargeUnits==2) {totalMass = totalMass*(.158987/(3600*24));}
						}
						else
						{
							//gor = INDEXH(gEditDischargeTimes,0).q_gas;
							gor = INDEXH(tempDialogDischargeData,0).q_gas;
						//	if (dischargeUnits==1 && gorUnits==1) {dischargeRate1 = dischargeRate1 / gor; dischargeRate2 = dischargeRate2 / gor;}
						//	if (dischargeUnits==2 && gorUnits==3) {dischargeRate1 = dischargeRate1 / (gor*28.32/.158987); dischargeRate2 = dischargeRate2 / (gor*28.32/.158987);}
						//	if (dischargeUnits==2 && gorUnits==2) {dischargeRate1 = dischargeRate1 / (gor*.02832/.158987); dischargeRate2 = dischargeRate2 / (gor*.02832/.158987);}
							totalMass = model->GetDuration() * INDEXH(tempDialogDischargeData,0).q_oil;
						}
					}
				}
				tempSet.totalMass = totalMass;
			}

			// get start release position
			err = EditTexts2LL(dialog, CDOGSPILLSLATDEGREES, &tempSet.startRelPos,TRUE);
			if(err)break;

			// get end release position
			tempSet.bWantEndRelPosition = false;	// not an option 
			
			//// verify data before assigning
			//////////////////////////////////
			
			//if(tempSet.numOfLEs < 1)  { 
			//	printError("The number of LE's must be at least one."); 
			//	MySelectDialogItemText(dialog, CDOGSPILLNUMLES, 0, 255);  
			//	break;}
				
				// require a different end release time
			/*if(!tempSet.bWantEndRelTime || tempSet.startRelTime == tempSet.endRelTime) {
				printError("The release end time must be set."); 
				break;}*/

			/*if(tempSet.bWantEndRelTime && tempSet.startRelTime > tempSet.endRelTime) {
				printError("The release start time cannot be after the release end time."); 
				break;}*/
			
			if(!model->IsWaterPoint(tempSet.startRelPos)) {
				printError("The release start position must be in the water."); 
				break;}
				
			if(!model->IsAllowableSpillPoint(tempSet.startRelPos)){
				if(haveOpenWizFile)
					printError("This Location File has not been set up for spills in the area of your release start position."); 
				else
					printError("This map has not been set up for spills in the area of your release start position."); 
				break;}
			
			////////////////////////
			///// end verification
			
			/////
			// final checks   ///{
			if(tempSet.startRelTime != model -> GetStartTime())
			{
				short buttonSelected;
				if(model -> ThereIsAnEarlierSpill(tempSet.startRelTime,(TLEList*)gCDOGLEListInDialog))
				{	// there is an earlier spill
					if(tempSet.startRelTime < model -> GetStartTime())
					{
						// they are already in trouble, so they are on their own
					}
					else if (tempSet.startRelTime > model -> GetEndTime())
					{
						// point out that this spill is not in the time interval being modeled
						buttonSelected  = MULTICHOICEALERT(1690,"The Release Start Time is outside of the time interval being modeled.  Are you sure you want to continue?",TRUE);
						switch(buttonSelected){
							case 1:// continue
								break;  
							case 3: // cancel
								return 0;// stay at this dialog
								break;
						}
					}
				}
				else 
				{	// no other spill is earlier than this one
					buttonSelected  = MULTICHOICEALERT(1681,"",TRUE);
					switch(buttonSelected){
						case 1:// change
							model -> SetStartTime(tempSet.startRelTime);
							model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
							break;  
						case 3: // don't change
							break;
						case 4: // cancel
							return 0;// stay at this dialog
							break;
					}
				}
			}
			// get depth units
			depthUnits = GetPopSelection(dialog, CDOGSPILLDEPTHUNITS);
			if (tempSet.z <= 0) {printNote("The spill depth must be below the surface");break;}
			if (depthUnits==1 && tempSet.z > 12000)
			{
				printNote("The spill depth must be less than 12000 meters");
				break;
			}
			else if (tempSet.z > 39370)
			{
				printNote("The spill depth must be less than 39370 feet");
				break;
			}
			else if (depthUnits==3  && tempSet.z > 6561.7)
			{
				printNote("The spill depth must be less than 6561.7 fathoms");
				break;
			}
			// CDOG requires that the total water depth is greater than the depth of the oil/gas release point
			if (sEditDepthVals && sMethodOfDeterminingHydrodynamics== 2) 
			{	
				//long numProfileSets = _GetHandleSize((Handle)gEditDepthLevels)/sizeof(**gEditDepthLevels);
				//if (numProfileSets>0 && tempSet.z > INDEXH(gEditDepthLevels,numProfileSets-1).depth)
				long numProfileSets = _GetHandleSize((Handle)sEditDepthVals)/sizeof(**sEditDepthVals);
				if (numProfileSets>0 && tempSet.z*depthCF[depthUnits-1] > INDEXH(sEditDepthVals,numProfileSets-1).depth)
				{
					char infoStr[64];
					StringWithoutTrailingZeros(infoStr,INDEXH(sEditDepthVals,numProfileSets-1).depth,2); 
					sprintf(errStr,"The spill depth must be smaller than the total depth which is %s meters",infoStr);
					printNote(errStr);
					break;
				}
			}
			else if (sdvals && sMethodOfDeterminingHydrodynamics== 2) 
			{	
				long numProfileSets = _GetHandleSize((Handle)sdvals)/sizeof(**sdvals);
				if (numProfileSets>0 && tempSet.z*depthCF[depthUnits-1] > INDEXH(sdvals,numProfileSets-1).depth)
				{
					char infoStr[64];
					StringWithoutTrailingZeros(infoStr,INDEXH(sdvals,numProfileSets-1).depth,2); 
					sprintf(errStr,"The spill depth must be smaller than the total depth which is %s meters",infoStr);
					printNote(errStr);
					break;
				}
			}
			
			///////////////////////////// 
			//check what to do about an end time the sticks out past the time being modeled

			//if(tempSet.endRelTime > model -> GetEndTime())
			//{
				//Either
				//printNote("The End Release Time is outside of the interval being modeled");
				//return 0; // stay at this dialog
				//Or
				//short buttonSelected  = MULTICHOICEALERT(1687,"",TRUE);
				//switch(buttonSelected){
					//case 1:// change
						//model -> SetDuration(tempSet.endRelTime-tempSet.startRelTime);
						//break;  
					//case 3: // don't change
						//break;
					//case 4: // cancel
						//return 0;// stay at this dialog
						//break;
				//}
			//}
			/////////////////////////////
			
			///} end of final checks
			
			/////////////////////////////////////////////////
			/////////////////////////////////////////////////

			// code goes here, what to do about total mass with variable/constant discharge rate?
			//tempSet.totalMass = (tempSet.endRelTime-tempSet.startRelTime)*sCDOGSpillParameters.oilDischargeRate;
			sDepthUnits = depthUnits;
			mygetitext(dialog, CDOGSPILLNAME, tempSet.spillName, kMaxNameLen - 1);		// get the mover's nameStr
			sCdogDialogSharedSet = tempSet;

			if (sEditDepthVals && sEditDepthVals!=sdvals) 
			{
				MyDisposeHandle((Handle*)&sdvals);
				sdvals = sEditDepthVals;
			}

			if (gEditDischargeTimes && gEditDischargeTimes!=sDischargeData) 
			{
				MyDisposeHandle((Handle*)&sDischargeData);
				sDischargeData = gEditDischargeTimes;
			}

			return CDOGSPILLOK;

		case CDOGSPILLCANCEL: 
		{
			
			if (sEditDepthVals && sEditDepthVals!=sdvals) 
			{
				MyDisposeHandle((Handle*)&sdvals);
				//sdvals = sEditDepthVals;
			}
			return CDOGSPILLCANCEL;
		}

//		case CDOGSPILLSTARTMONTH:
		case CDOGSPILLSTARTDAY:
//		case CDOGSPILLSTARTYEAR:
		case CDOGSPILLSTARTHOURS:
		case CDOGSPILLSTARTMINUTES:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;

//		case CDOGSPILLENDMONTH:
		case CDOGSPILLENDDAY:
//		case CDOGSPILLENDYEAR:
		case CDOGSPILLENDHOURS:
		case CDOGSPILLENDMINUTES:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;

		case CDOGSPILLDEGREES:
		case CDOGSPILLDEGMIN:
		case CDOGSPILLDMS:
			err = EditTexts2LL(dialog, CDOGSPILLSLATDEGREES, &p,TRUE);
			if(err) break;
			if (itemNum == CDOGSPILLDEGREES) settings.latLongFormat = DEGREES;
			if (itemNum == CDOGSPILLDEGMIN) settings.latLongFormat = DEGMIN;
			if (itemNum == CDOGSPILLDMS) settings.latLongFormat = DMS;
			SwitchLLFormat(dialog, CDOGSPILLSLATDEGREES, CDOGSPILLDEGREES); // do it 2nd to set focus
			//ShowHideCDOGSpillDialogItems(dialog);//JLM
			LL2EditTexts(dialog, CDOGSPILLSLATDEGREES, &p);
			break;

		case CDOGSPILLNUMLES:
		case CDOGSPILLDEPTH:
			CheckNumberTextItem(dialog, itemNum, itemNum != CDOGSPILLNUMLES);
			break;

		//case CDOGSPILLWANTENDTIME:
			//ToggleButton(dialog, itemNum);
				//EnableCDOGEndTime (dialog, GetButton (dialog, itemNum));
			//ShowHideCDOGSpillDialogItems(dialog);//JLM
			//break;

		//case CDOGSPILLPOLLUTANT:
			//PopClick(dialog, itemNum, &menuID_menuItem);
			//break;

		//case CDOGSPILLDRPOPUP:
			//PopClick(dialog, itemNum, &menuID_menuItem);
			/*if (GetPopSelection(dialog, CDOGSPILLDRPOPUP)==1)
				mysetitext(dialog, CDOGSPILLDRLABEL, "Gas Discharge Rate");
			else
				mysetitext(dialog, CDOGSPILLDRLABEL, "Oil Discharge Rate");*/
			//break;

		case CDOGSPILLSTARTMONTH:
		case CDOGSPILLSTARTYEAR:
		case CDOGSPILLENDMONTH:
		case CDOGSPILLENDYEAR:
		case CDOGSPILLDEPTHUNITS:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
		case CDOGSPILLWINDAGE:
		{
			WindageRec windageData = sWindageInfo;
			err = WindageSettingsDialog(&windageData, GetDialogWindow(dialog));
			//RegisterPopTable (CDOGSpillPopTable, sizeof (CDOGSpillPopTable) / sizeof (PopInfoRec));
			if (!err) sWindageInfo = windageData;
			break;
		}
		case CDOGSPILLDIFFUSIVITY:
		{
			short methodOfDeterminingHydrodynamics = sMethodOfDeterminingHydrodynamics;
			err = CircInfoDialog(&methodOfDeterminingHydrodynamics);
			if (!err)
			{
				sMethodOfDeterminingHydrodynamics = methodOfDeterminingHydrodynamics;
			}
			break;
		}
		case CDOGSPILLOUTPUTOPTIONS:
		{
			Boolean outputSubsurfaceFiles = sOutputSubsurfaceFiles, outputGasFiles = sOutputGasFiles;
			err = CDOGOutputOptionsDialog(&outputSubsurfaceFiles,&outputGasFiles,mapWindow);
			//RegisterPopTable (CDOGSpillPopTable, sizeof (CDOGSpillPopTable) / sizeof (PopInfoRec));
			if (!err) 
			{
				sOutputSubsurfaceFiles = outputSubsurfaceFiles;
				sOutputGasFiles = outputGasFiles;
			}
			break;
		}
		case CDOGSPILLPARAMETERS:
		//case CDOGVARDISCHARGE:
		{
			Boolean hydrogenSulfide = sHydrogenSulfide;
			CDOGUserUnits userUnits = sCDOGUserUnits;
			gEditDischargeTimes = (DischargeDataH)_NewHandleClear(0);
			// code goes here, should init fContourLevelsH if nil
			if(sDischargeData)
			{
				gEditDischargeTimes = sDischargeData;
				// should only do this once ?
				if(_HandToHand((Handle *)&gEditDischargeTimes))
				{
					printError("Not enough memory to create temporary discharge data");
					break;
				}
			}
			CDOGParameters spillParameters = sCDOGParameters;
			CDOGSpillParameters spillParameters2 = sCDOGSpillParameters;
			err = CDOGVarDischargeDialog(&gEditDischargeTimes,&hydrogenSulfide,&userUnits,&spillParameters,&spillParameters2,GetDialogWindow(dialog));
			//RegisterPopTable (CDOGSpillPopTable, sizeof (CDOGSpillPopTable) / sizeof (PopInfoRec));
			if (!err) 
			{
				sDischargeData = gEditDischargeTimes;
				sCDOGParameters = spillParameters;
				sCDOGSpillParameters = spillParameters2;
				sHydrogenSulfide = hydrogenSulfide;
				sCDOGUserUnits = userUnits;
			}
			break;
		}
	}
	
	return 0;
}

void EnableCDOGEndTime(DialogPtr dialog, Boolean bEnable)
{
	Boolean show  = bEnable; //JLM
	CDOGSpillPopTable[4].bStatic = !bEnable;
	CDOGSpillPopTable[5].bStatic = !bEnable;
	

	ShowHideDialogItem(dialog, CDOGSPILLENDMONTH, show); 
	ShowHideDialogItem(dialog, CDOGSPILLENDYEAR, show); 
	
	ShowHideDialogItem(dialog, CDOGSPILLENDDAY, show);
	ShowHideDialogItem(dialog, CDOGSPILLENDHOURS, show);
	ShowHideDialogItem(dialog, CDOGSPILLENDMINUTES, show);

	ShowHideDialogItem(dialog, CDOGSPILLENDTIMECOLON, show);
	ShowHideDialogItem(dialog, CDOGSPILLENDTIMELABEL, show);

	ShowHideDialogItem(dialog, CDOGSPILLWANTENDTIME, show);
	ShowHideDialogItem(dialog, CDOGSPILLFROST2, show);
	/*if(show)
	{
		PopDraw(dialog, CDOGSPILLENDMONTH);
		PopDraw(dialog, CDOGSPILLENDYEAR);
	}*/
	
	EnableTextItem(dialog, CDOGSPILLENDDAY, bEnable);
	EnableTextItem(dialog, CDOGSPILLENDHOURS, bEnable);
	EnableTextItem(dialog, CDOGSPILLENDMINUTES, bEnable);
}

OSErr CDOGSpillInit (DialogPtr dialog, VOIDPTR data)
{
	WorldPoint 	p;
	DateTimeRec	time;
	
	SetDialogItemHandle(dialog, CDOGSPILLHILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, CDOGSPILLFROST1, (Handle) FrameEmbossed);
	SetDialogItemHandle(dialog, CDOGSPILLFROST2, (Handle) FrameEmbossed);
	
	if(UseExtendedYears()) {
		//CDOGSpillPopTable[5].menuID = pYEARS_EXTENDED;
		//CDOGSpillPopTable[7].menuID = pYEARS2_EXTENDED;
		CDOGSpillPopTable[3].menuID = pYEARS_EXTENDED;
		CDOGSpillPopTable[5].menuID = pYEARS2_EXTENDED;
	}
	else {
		//CDOGSpillPopTable[5].menuID = pYEARS;
		//CDOGSpillPopTable[7].menuID = pYEARS2;
		CDOGSpillPopTable[3].menuID = pYEARS;
		CDOGSpillPopTable[5].menuID = pYEARS2;
	}

	RegisterPopTable (CDOGSpillPopTable, sizeof (CDOGSpillPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (CDOGSPILLDLG, dialog);
	
	Float2EditText (dialog, CDOGSPILLNUMLES, sCdogDialogSharedSet.numOfLEs, 0);
	//SetPopSelection (dialog, CDOGSPILLPOLLUTANT, sCdogDialogSharedSet.pollutantType);
	 
	ShowHideDialogItem(dialog, CDOGSPILLMASSLABEL, false); // pollutant will always be conservative
	ShowHideDialogItem(dialog, CDOGSPILLPOLLUTANT, false); // pollutant will always be conservative
	// JLM 12/29/98
	// the user should have to enter the amount of oil spilled
	// there should be no default
	// we do this by making the intial amount negative and not displaying 
	// a negative amount here
	//if(sCdogDialogSharedSet.totalMass < 0) mysetitext(dialog, CDOGSPILLGASOILRATIO, ""); // BLANK
	//else Float2EditText (dialog, CDOGSPILLGASOILRATIO, sCdogDialogSharedSet.totalMass, 0);
	
	Float2EditText (dialog, CDOGSPILLDEPTH, sCdogDialogSharedSet.z, 0);

	SetPopSelection (dialog, CDOGSPILLDEPTHUNITS, sDepthUnits);	
//	DisplayTime (dialog, CDOGSPILLSTARTMONTH, sCdogDialogSharedSet.startRelTime);
	SecondsToDate (sCdogDialogSharedSet.startRelTime, &time);
	SetPopSelection (dialog, CDOGSPILLSTARTMONTH, time.month);
	SetPopSelection (dialog, CDOGSPILLSTARTYEAR,  time.year - (FirstYearInPopup()  - 1));
	Long2EditText (dialog, CDOGSPILLSTARTDAY, time.day);
	Long2EditText (dialog, CDOGSPILLSTARTHOURS, time.hour);
	Long2EditText (dialog, CDOGSPILLSTARTMINUTES, time.minute);

	if(!data)
	{
		LL2EditTexts (dialog, CDOGSPILLSLATDEGREES, &sCdogDialogSharedSet.startRelPos);
	}
	else LL2EditTexts (dialog, CDOGSPILLSLATDEGREES, (WorldPoint*)data);
	
	SecondsToDate (sCdogDialogSharedSet.endRelTime, &time);
	SetPopSelection (dialog, CDOGSPILLENDMONTH, time.month);
	SetPopSelection (dialog, CDOGSPILLENDYEAR,  time.year - (FirstYearInPopup()  - 1));
	Long2EditText (dialog, CDOGSPILLENDDAY, time.day);
	Long2EditText (dialog, CDOGSPILLENDHOURS, time.hour);
	Long2EditText (dialog, CDOGSPILLENDMINUTES, time.minute);
	// force end time
	//SetButton (dialog, CDOGSPILLWANTENDTIME, sCdogDialogSharedSet.bWantEndRelTime);

	mysetitext(dialog, CDOGSPILLNAME, sCdogDialogSharedSet.spillName);

	//ShowHideDialogItem(dialog, 4, false);
	
	if (model -> GetModelMode () < ADVANCEDMODE)
	{
		ShowHideDialogItem(dialog, CDOGSPILLLELABEL, false);
		ShowHideDialogItem(dialog, CDOGSPILLNUMLES,  false);
	}

	// set the hiliting of the end-time and end-position dialogs
	// according to the corresponding checkboxes
	//EnableCDOGEndTime (dialog, GetButton (dialog, CDOGSPILLWANTENDTIME));

	SwitchLLFormat(dialog, CDOGSPILLSLATDEGREES, CDOGSPILLDEGREES);
	//ShowHideCDOGSpillDialogItems(dialog);//JLM

	ShowHideDialogItem(dialog, CDOGSPILLWINDAGE, model->GetModelMode() == ADVANCEDMODE); // if there is a ptcurmap, should be 3D...
	//MySelectDialogItemText(dialog, CDOGSPILLGASOILRATIO, 0, 255);
	MySelectDialogItemText(dialog, CDOGSPILLNAME, 0, 255);
	
	EnableCDOGEndTime(dialog, false);
	return 0;
}

OSErr CDOGSpillSettingsDialog (CDOGLEList 	*thisLEList)
{
	short 		dialogItem;
	CDOGLEList 	*uncertaintyLEList=0;
	Boolean weCreatedThisLEList = false;
	Boolean weCreatedUncertaintyLEList = false;
	OSErr err = 0;
	
	if (!thisLEList) return -1; //JLM, this code is only called when the LE already exists
										// otherwise we would need code to add them to the list
	
	gCDOGLEListInDialog = thisLEList;
	sCdogDialogSharedSet = thisLEList -> fSetSummary;
	sWindageInfo = thisLEList -> GetWindageInfo();
	sCDOGParameters = thisLEList -> GetCDOGParameters();
	sDiffusivityInfo = thisLEList->GetCDOGDiffusivityInfo();
	sHydrodynamicInfo = thisLEList->GetCDOGHydrodynamicInfo();
	sTempSalInfo = thisLEList->GetCDOGTempSalInfo();
	sOutputSubsurfaceFiles = thisLEList->GetOutputSubsurfaceFiles();
	sOutputGasFiles = thisLEList->GetOutputGasFiles();
	sdvals = thisLEList->GetDepthValuesHandle();
	sDischargeData = thisLEList->GetDischargeDataHandle();
	sHydrogenSulfide = thisLEList->GetHydrogenSulfideInfo();
	sCDOGUserUnits = thisLEList->GetCDOGUserUnits();
	sDepthUnits = thisLEList->GetDepthUnits();
	sMethodOfDeterminingHydrodynamics = thisLEList->GetMethodOfDeterminingHydrodynamics();
	
	dialogItem = MyModalDialog(CDOGSPILLDLG, mapWindow, nil, CDOGSpillInit, CDOGSpillClick);
	if (dialogItem == CDOGSPILLCANCEL)
		return CDOGSPILLCANCEL;
	else if (dialogItem == CDOGSPILLOK)
	{
		
		if (!thisLEList)
		{
			thisLEList = new CDOGLEList ();
			weCreatedThisLEList = true;
			if (!thisLEList)
				{ TechError("CDOGSpillSettingsDialog()", "new CDOGLEList()", 0); err = -1; goto done; }
		}
		thisLEList -> SetWindageInfo(sWindageInfo);
		thisLEList -> SetCDOGParameters(sCDOGParameters);
		thisLEList -> SetCDOGDiffusivityInfo(sDiffusivityInfo);
		thisLEList -> SetCDOGHydrodynamicInfo(sHydrodynamicInfo);
		thisLEList -> SetCDOGTempSalInfo(sTempSalInfo);
		thisLEList -> SetOutputSubsurfaceFiles(sOutputSubsurfaceFiles);
		thisLEList -> SetOutputGasFiles(sOutputGasFiles);
		thisLEList -> SetDepthValuesHandle(sdvals);
		thisLEList -> SetDischargeDataHandle(sDischargeData);
		thisLEList -> SetHydrogenSulfideInfo(sHydrogenSulfide);
		thisLEList -> SetCDOGUserUnits(sCDOGUserUnits);
		thisLEList -> SetDepthUnits(sDepthUnits);
		thisLEList -> SetMethodOfDeterminingHydrodynamics(sMethodOfDeterminingHydrodynamics);

		uncertaintyLEList = (CDOGLEList*)model->GetMirroredLEList(thisLEList);
		if (uncertaintyLEList) uncertaintyLEList -> SetWindageInfo(sWindageInfo);
		if (uncertaintyLEList) uncertaintyLEList -> SetCDOGParameters(sCDOGParameters);
		if (uncertaintyLEList) uncertaintyLEList -> SetCDOGDiffusivityInfo(sDiffusivityInfo);
		if (uncertaintyLEList) uncertaintyLEList -> SetCDOGHydrodynamicInfo(sHydrodynamicInfo);
		if (uncertaintyLEList) uncertaintyLEList -> SetCDOGTempSalInfo(sTempSalInfo);
		if (uncertaintyLEList) uncertaintyLEList -> SetOutputSubsurfaceFiles(sOutputSubsurfaceFiles);
		if (uncertaintyLEList) uncertaintyLEList -> SetOutputGasFiles(sOutputGasFiles);
		if (uncertaintyLEList) uncertaintyLEList -> SetDepthValuesHandle(sdvals);
		if (uncertaintyLEList) uncertaintyLEList -> SetDischargeDataHandle(sDischargeData);
		if (uncertaintyLEList) uncertaintyLEList -> SetHydrogenSulfideInfo(sHydrogenSulfide);
		if (uncertaintyLEList) uncertaintyLEList -> SetCDOGUserUnits(sCDOGUserUnits);
		if (uncertaintyLEList) uncertaintyLEList -> SetDepthUnits(sDepthUnits);
		if (uncertaintyLEList) uncertaintyLEList -> SetMethodOfDeterminingHydrodynamics(sMethodOfDeterminingHydrodynamics);
	
		err = thisLEList -> Initialize (&sCdogDialogSharedSet,true);
		if(err) goto done;
		
		if(uncertaintyLEList)		
			err = uncertaintyLEList -> Initialize (&sCdogDialogSharedSet,true);
		if(err) goto done; 
		
		model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
		// since the model might be in negative time, in which case the LE's would be deleted and
		// readded from the files, which do not contain these changes
		model->NewDirtNotification();
	}
done:
	if(err)
	{
		if(weCreatedThisLEList && thisLEList) { thisLEList -> Dispose(); delete thisLEList; thisLEList = 0;} 
		if(weCreatedUncertaintyLEList && uncertaintyLEList) { uncertaintyLEList -> Dispose(); delete uncertaintyLEList; uncertaintyLEList = 0;} 
	}
	
	return err;
}

/////////////////////////////////////////////////
static PopInfoRec CircInfoPopTable[] = {
		{ CDOGCIRCINFODLG, nil, CDOGCIRCINPUTTYPEPOPUP, 0, pCIRCULATIONINFO, 0, 1, FALSE, nil }
	};

//static CDOGTempSalInfo sDialogTempSalInfo;
static short sCircDialogMethodOfDeterminingHydrodynamics;
void ShowHideCircButtons(DialogPtr dialog)
{
	Boolean showProfilesButton;
	short theType = GetPopSelection (dialog, CDOGCIRCINPUTTYPEPOPUP);
	if (theType == 4) showProfilesButton = false;
	else showProfilesButton = true;

	if (theType==4 || theType==1)
	{
		ShowHideDialogItem(dialog, CDOGCIRCINFOPROFILES, showProfilesButton); 
		ShowHideDialogItem(dialog, CDOGCIRCINFOHYD, !showProfilesButton); 
		ShowHideDialogItem(dialog, CDOGCIRCINFOTEMPSAL, !showProfilesButton); 
	}
	else
	{
		ShowHideDialogItem(dialog, CDOGCIRCINFOPROFILES, false); 
		ShowHideDialogItem(dialog, CDOGCIRCINFOHYD, false); 
		ShowHideDialogItem(dialog, CDOGCIRCINFOTEMPSAL, false); 
	}
	return;
}

OSErr CircInfoInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	//RegisterPopTable (CircInfoPopTable, sizeof (CircInfoPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (CDOGCIRCINFODLG, dialog);
	SetPopSelection (dialog, CDOGCIRCINPUTTYPEPOPUP, sCircDialogMethodOfDeterminingHydrodynamics);	
	
	ShowHideCircButtons(dialog);
	if (sCircDialogMethodOfDeterminingHydrodynamics==1)
		ShowHideDialogItem(dialog, CDOGCIRCINFOPROFILES, true); 
	else
		ShowHideDialogItem(dialog, CDOGCIRCINFOPROFILES, false); 
	
	return 0;
}


short CircInfoClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short	circInfoType;
	long	menuID_menuItem;
	OSErr err = 0;
	
	switch (itemNum) {
		case CDOGCIRCINFOCANCEL: 
		{
			if (sEditDepthVals) 
			{
				MyDisposeHandle((Handle*)&sEditDepthVals);
			}
			return CDOGCIRCINFOCANCEL;
		}

		case CDOGCIRCINFOOK:
			circInfoType = GetPopSelection (dialog, CDOGCIRCINPUTTYPEPOPUP);
			if (circInfoType == 1)
			{
				// bring up grid and profiles dialog
				if (!sEditDepthVals || _GetHandleSize((Handle)sEditDepthVals)/sizeof(**sEditDepthVals)==0) {printError("No profile has been set up"); break;}
			}
			
			sCircDialogMethodOfDeterminingHydrodynamics = circInfoType;

			if (circInfoType == 1)
			{
				if (sEditDepthVals && sEditDepthVals!=sdvals) 
				{
					MyDisposeHandle((Handle*)&sdvals);
					sdvals = sEditDepthVals;
				}
			}
			else
			{
				if (sEditDepthVals) 
				{
					MyDisposeHandle((Handle*)&sEditDepthVals);
				}
			}	
			return itemNum;
			

		case CDOGCIRCINFODIFF:
		{
			CDOGDiffusivityInfo diffusivityInfo = sDiffusivityInfo;
			err = CDOGDiffusivityDialog(&diffusivityInfo, GetDialogWindow(dialog));
			if (!err) sDiffusivityInfo = diffusivityInfo;
			break;
		}
		case CDOGCIRCINFOHYD:
		{
			CDOGHydrodynamicInfo hydrodynamicsInfo = sHydrodynamicInfo;
			err = CDOGHydrodynamicsDialog(&hydrodynamicsInfo,mapWindow);
			if (!err) sHydrodynamicInfo = hydrodynamicsInfo;
			break;
		}
		case CDOGCIRCINFOTEMPSAL:
		{
			CDOGTempSalInfo tempSalInfo = sTempSalInfo;
			err = TempSalDialog(&tempSalInfo);
			if (!err)
			{
				sTempSalInfo = tempSalInfo;
			}
			break;
		}
		case CDOGCIRCINFOPROFILES:
		{
			gEditDepthLevels = (DepthValuesSetH)_NewHandleClear(0);
			// code goes here, should init fContourLevelsH if nil
			//if(sdvals)
			if(sEditDepthVals)
			{
				//gEditDepthLevels = sdvals;
				gEditDepthLevels = sEditDepthVals;
				// should only do this once ?
				if(_HandToHand((Handle *)&gEditDepthLevels))
				{
					printError("Not enough memory to create temporary contour levels");
					break;
				}
			}
			//ContourDialog(&gEditdepthLevels);
			//Boolean outputSubsurfaceFiles = sOutputSubsurfaceFiles, outputGasFiles = sOutputGasFiles;
			//DepthValuesSetH dvals = sdvals;
			//float cellLength = sLengthOfGridCellInMeters;
			//long  numCells = sNumGridCells;
			err = EditCDOGProfilesDialog(&gEditDepthLevels/*,&cellLength,&numCells*/,mapWindow);
			//RegisterPopTable (CDOGSpillPopTable, sizeof (CDOGSpillPopTable) / sizeof (PopInfoRec));
			if (!err) 
			{
				//sdvals = dvals;
				//sdvals = gEditDepthLevels;
				sEditDepthVals = gEditDepthLevels;
			//	sLengthOfGridCellInMeters = cellLength;
			//	sNumGridCells = numCells;
			}
			break;
		}

		case CDOGCIRCINPUTTYPEPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowHideCircButtons(dialog);
			circInfoType = GetPopSelection (dialog, CDOGCIRCINPUTTYPEPOPUP);
			if (circInfoType == 1)
			{
				// bring up grid and profiles dialog
				gEditDepthLevels = (DepthValuesSetH)_NewHandleClear(0);
				// code goes here, should init fContourLevelsH if nil
				//if(sdvals)
				if(sEditDepthVals)
				{
					//gEditDepthLevels = sdvals;
					gEditDepthLevels = sEditDepthVals;
					// should only do this once ?
					if(_HandToHand((Handle *)&gEditDepthLevels))
					{
						printError("Not enough memory to create temporary contour levels");
						break;
					}
				}
				//ContourDialog(&gEditdepthLevels);
				//Boolean outputSubsurfaceFiles = sOutputSubsurfaceFiles, outputGasFiles = sOutputGasFiles;
				//DepthValuesSetH dvals = sdvals;
				//float cellLength = sLengthOfGridCellInMeters;
				//long  numCells = sNumGridCells;
				err = EditCDOGProfilesDialog(&gEditDepthLevels,/*&cellLength,&numCells,*/mapWindow);
				//RegisterPopTable (CDOGSpillPopTable, sizeof (CDOGSpillPopTable) / sizeof (PopInfoRec));
				if (!err) 
				{
					//sdvals = dvals;
					//sdvals = gEditDepthLevels;
					sEditDepthVals = gEditDepthLevels;
					//sLengthOfGridCellInMeters = cellLength;
					//sNumGridCells = numCells;
					//sOutputSubsurfaceFiles = outputSubsurfaceFiles;
					//sOutputGasFiles = outputGasFiles;
				}
				MySetControlTitle(dialog, CDOGCIRCINFOTEMPSAL, "Temperature and Salinity Profiles");	// button gets overwritten in profiles dialog
				break;
			}
			else if (circInfoType == 2)
			{
				//printNote("This option has not been implemented yet");
			}
			else
			{
				// bring up ??
			}
			break;

	}

	return 0;
}

OSErr CircInfoDialog(short *methodOfDeterminingHydrodynamics)
{	// code goes here, need to pass everything in to allow user cancel 
	short item;
	PopTableInfo saveTable = SavePopTable();
	short j, numItems = 0;
	PopInfoRec combinedDialogsPopTable[9];
	// code to allow a dialog on top of another with pops
	for(j = 0; j < sizeof(CircInfoPopTable) / sizeof(PopInfoRec);j++)
		combinedDialogsPopTable[numItems++] = CircInfoPopTable[j];
	for(j= 0; j < saveTable.numPopUps ; j++)
		combinedDialogsPopTable[numItems++] = saveTable.popTable[j];
	
	RegisterPopTable(combinedDialogsPopTable,numItems);

	sEditDepthVals = (DepthValuesSetH)_NewHandleClear(0);
	if(sdvals)
	{
		sEditDepthVals = sdvals;
		if(_HandToHand((Handle *)&sEditDepthVals))
		{
			printError("Not enough memory to create temporary edit values");
			return 0;
		}
	}
	sCircDialogMethodOfDeterminingHydrodynamics = *methodOfDeterminingHydrodynamics;
	item = MyModalDialog(CDOGCIRCINFODLG, mapWindow, 0, CircInfoInit, CircInfoClick);
	RestorePopTableInfo(saveTable);
	if(item == CDOGCIRCINFOCANCEL) return USERCANCEL; 
	model->NewDirtNotification();	
	if(item == CDOGCIRCINFOOK) 
	{
		*methodOfDeterminingHydrodynamics = sCircDialogMethodOfDeterminingHydrodynamics;
		return 0; 
	}
	else return -1;
}

OSErr CreateCDOGLESet()
{
	short	item;
	CDOGLEList 	*list=nil,*uncertaintyLEList = nil;
	OSErr err = 0;
	
	long numMaps = 0;
	
	numMaps = model -> mapList -> GetItemCount();
	
	// allow this ? have to allow for all spills then...
	if (numMaps <= 0) {
		printError("You cannot create a spill because there is no map.");
		return -1;
	}

	
	memset(&sCdogDialogSharedSet,0,sizeof(sCdogDialogSharedSet));
	sCdogDialogSharedSet.numOfLEs = 10000;
	sCdogDialogSharedSet.pollutantType = OIL_CONSERVATIVE;
	sCdogDialogSharedSet.totalMass = -1;// JLM 12/29/98, force user to enter a value
	sCdogDialogSharedSet.massUnits = BARRELS;// CJ asked that this be barrels
	sCdogDialogSharedSet.startRelPos = WorldRectCenter (settings.currentView);
	sCdogDialogSharedSet.endRelPos = sCdogDialogSharedSet.startRelPos;
	sCdogDialogSharedSet.startRelTime = model -> GetStartTime ();
	//sCdogDialogSharedSet.endRelTime = sCdogDialogSharedSet.startRelTime + 24*3600;	// default is 1 day release
	sCdogDialogSharedSet.endRelTime = sCdogDialogSharedSet.startRelTime + model->GetDuration();	// default is continuous release
	//sCdogDialogSharedSet.bWantEndRelTime = true;
	sCdogDialogSharedSet.bWantEndRelTime = false;	// there will always be an end release time, but it's handled via duration
	sCdogDialogSharedSet.bWantEndRelPosition = false;

	sCdogDialogSharedSet.z = 0;
	sCdogDialogSharedSet.density = GetPollutantDensity(sCdogDialogSharedSet.pollutantType);
	sCdogDialogSharedSet.ageInHrsWhenReleased = 0.;

	sWindageInfo.windageA = .01;
	sWindageInfo.windageB = .04;
	sWindageInfo.persistence = .25;	// in hours

	//sCDOGSpillParameters.oilDischargeRate = .01829;
	//sCDOGSpillParameters.gasDischargeRate = 3.659;
	//sCDOGSpillParameters.dischargeRateType = 1;

	//sCDOGParameters.orificeDiameter = .1;
	//sCDOGParameters.temp = 80;
	//sCDOGParameters.density = 842.5;	//kg/m^3
	sCDOGParameters.equilibriumCurves = 2;
	sCDOGParameters.bubbleRadius = .0025;
	sCDOGParameters.molecularWt = .0191;
	sCDOGParameters.hydrateDensity = 920;
	sCDOGParameters.separationFlag = 1;
	sCDOGParameters.hydrateProcess = 1;
	sCDOGParameters.dropSize = 0;
	
	//sCDOGParameters.dischargeRateType = 1;				// 1 oil, 2 gas on popup
	sCDOGParameters.duration = 3600;
	sCDOGParameters.isContinuousRelease = true;

	sDiffusivityInfo.horizDiff = 10.; // m^2/s
	sDiffusivityInfo.vertDiff = .0001;	// m^2/s
	sDiffusivityInfo.timeStep = .1 * 3600;
	
	//sHydrodynamicInfo.timeInterval = .333; // hrs
	sHydrodynamicInfo.timeInterval = 9999; // hrs
	//sHydrodynamicInfo.period = 1000;	// hrs
	sHydrodynamicInfo.period = 9999;	// hrs
	sHydrodynamicInfo.methodOfDeterminingCurrents = 1;	// select folder
	
	sTempSalInfo.methodOfDeterminingTempSal = 1;
	
	sOutputSubsurfaceFiles = false;
	sOutputGasFiles = false;
	
	//if (sdvals)
	sdvals = 0;
	sDischargeData = 0;

	//sLengthOfGridCellInMeters = 500;
	//sNumGridCells = 1;
	
	sHydrogenSulfide = false;
	
	sCDOGUserUnits.temperatureUnits = kDegreesC;
	sCDOGUserUnits.densityUnits = kKgM3;
	sCDOGUserUnits.diameterUnits = kMeters;
	sCDOGUserUnits.dischargeType = 1;
	sCDOGUserUnits.gorUnits = 1;
	sCDOGUserUnits.dischargeUnits = 1;	// m3/s
	sCDOGUserUnits.molWtUnits = 2;	// kg/mol
	
	sDepthUnits = 1;	// meters
	
	sMethodOfDeterminingHydrodynamics = 1;	// hand set profiles
	
	item = MyModalDialog(CDOGSPILLDLG, mapWindow, 0, CDOGSpillInit, CDOGSpillClick);
	if (item == CDOGSPILLCANCEL) 
		return noErr;
	else if (item == CDOGSPILLOK)
	{
		model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
		// since the model might be in negative time, in which case the LE's would be deleted and
		// readded from the files, which do not contain these new LE sets
		////
		// note: since the model time is going to be reset when the user adds LEs
		// we need to do this before Initialize() is called 
		// since Initialize() uses GetModelTime() -- JLM 10/18/00

		list = new CDOGLEList();
		if (!list) { TechError("CreateCDOGLESet()", "new CDOGLEList()", 0); err= -1; goto done;}
			
		uncertaintyLEList = new CDOGLEList ();
		if (!uncertaintyLEList) { TechError("CreateCDOGLESet()", "new CDOGLEList()", 0); err = -1; goto done; }
			
		list -> SetWindageInfo(sWindageInfo); 
		uncertaintyLEList -> SetWindageInfo(sWindageInfo); 

		err = list->Initialize(&sCdogDialogSharedSet,true);
		if(err) goto done;

		uncertaintyLEList->fLeType = UNCERTAINTY_LE;
		uncertaintyLEList->fOwnersUniqueID = list->GetUniqueID();
		err = uncertaintyLEList->Initialize(&sCdogDialogSharedSet,true);
		if(err) goto done;


		list->SetCDOGParameters(sCDOGParameters);
		uncertaintyLEList->SetCDOGParameters(sCDOGParameters);
		//list->SetCDOGSpillParameters(sCDOGSpillParameters);
		//uncertaintyLEList->SetCDOGSpillParameters(sCDOGSpillParameters);
		list->SetCDOGDiffusivityInfo(sDiffusivityInfo);
		uncertaintyLEList->SetCDOGDiffusivityInfo(sDiffusivityInfo);
		list->SetCDOGHydrodynamicInfo(sHydrodynamicInfo);
		uncertaintyLEList->SetCDOGHydrodynamicInfo(sHydrodynamicInfo);
		list->SetCDOGTempSalInfo(sTempSalInfo);
		uncertaintyLEList->SetCDOGTempSalInfo(sTempSalInfo);
		list->SetOutputSubsurfaceFiles(sOutputSubsurfaceFiles);
		uncertaintyLEList->SetOutputSubsurfaceFiles(sOutputSubsurfaceFiles);
		list->SetOutputGasFiles(sOutputGasFiles);
		uncertaintyLEList->SetOutputGasFiles(sOutputGasFiles);
		list->SetDepthValuesHandle(sdvals);
		uncertaintyLEList->SetDepthValuesHandle(sdvals);
		list->SetDischargeDataHandle(sDischargeData);
		uncertaintyLEList->SetDischargeDataHandle(sDischargeData);
		//list->SetCDOGTimeStep(sTimeStep);
		//uncertaintyLEList->SetCDOGTimeStep(sTimeStep);
		//list->SetGridInfo(sLengthOfGridCellInMeters,sNumGridCells);
		//uncertaintyLEList->SetGridInfo(sLengthOfGridCellInMeters,sNumGridCells);
		list->SetHydrogenSulfideInfo(sHydrogenSulfide);
		uncertaintyLEList->SetHydrogenSulfideInfo(sHydrogenSulfide);
		list->SetCDOGUserUnits(sCDOGUserUnits);
		uncertaintyLEList->SetCDOGUserUnits(sCDOGUserUnits);
		list->SetDepthUnits(sDepthUnits);
		uncertaintyLEList->SetDepthUnits(sDepthUnits);
		list->SetMethodOfDeterminingHydrodynamics(sMethodOfDeterminingHydrodynamics);
		uncertaintyLEList->SetMethodOfDeterminingHydrodynamics(sMethodOfDeterminingHydrodynamics);
		model->NewDirtNotification();

		err = model->AddLEList(list, 0);
		if(err) goto done;
		err = model->AddLEList(uncertaintyLEList, 0);
		if(err) goto done;
		
	}
			
done:
	if(err)
	{
		if(list) {list -> Dispose(); delete list; list = 0;}
		if(uncertaintyLEList) {uncertaintyLEList -> Dispose(); delete uncertaintyLEList; uncertaintyLEList = 0;}
	}
	
	return noErr;
}

CDOGLEList* GetCDOGSpill()
{
	TLEList *thisLEList;
	long i,numLESets = model->LESetsList->GetItemCount();
	CMyList	*theLESetsList = model->GetLESetsList ();
	LETYPE leType;
	for (i = 0; i < numLESets; i++) 
	{
		theLESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE) continue;
		if(thisLEList -> IAm(TYPE_CDOGLELIST))
		{
			return (CDOGLEList*)thisLEList; // typecast
		}
	}
	return nil;
}


OSErr ReadCDOGLEFileToSetReleaseTimes(char *cdogOutputFolderPathWithDelimiter,long fileNumber, long numLEs, LERecH array, Seconds startCDogReleaseTimeInSeconds)
{
	// this function sets the release time field for those LEs 
	//	that are marked as NEW_SURFACE in the given CDOG file
	//
	// fileNumber - number of file to be read
	
	// startCDogReleaseTimeInSeconds
	
	char  s[256], leType[64], pollutantTypeStr[64], status[64];
	long n, id;
	float depth,zeroDepth = 0.0; 
	float density, ageInHrs, longF, latF;
	double mass;
	BFPB ms4, ms5;
	LERec leRec,leRec2;
	OSErr err = 0;
	char fileName4[256],fileName5[256],path[256];
	float timeOfCDogReleaseInHrs, timeSurfacedInHrs;
	long bestGuessIndex;
	
	////////

	if(!array)  return -6;
	
	sprintf(fileName4,"FILE4_%03ld.out",fileNumber);
	sprintf(fileName5,"FILE5_%03ld.out",fileNumber);
		
		
	ms4.f = 0;
	strcpy(path,cdogOutputFolderPathWithDelimiter);
	strcat(path,fileName4);
	//strcat(path,"FILE4_028.out");
	if (err = FSOpenBuf(0, 0, path, &ms4, 100000, FALSE))
		{ TechError("ReadLEFile()", "FSOpenBuf()", err); return -1; }
	
	ms5.f = 0;
	strcpy(path,cdogOutputFolderPathWithDelimiter);
	strcat(path,fileName5);
	//strcat(path,"FILE5_028.out");
	if (err = FSOpenBuf(0, 0, path, &ms5, 100000, FALSE))
		{ TechError("ReadLEFile()", "FSOpenBuf()", err); err = -1; goto done; }
	
	
	for (n = 0 ; ; n++) {
		Boolean isEvaporated;
		if (!myfgetsbuf(s, 255, &ms4)) // header line
			break; // done reading file
		// note: we could get the ID from this line, but for now we will trust it
		
		if (!myfgetsbuf(s, 255, &ms4)) // data line
			{ TechError("ReadLEFile()", "myfgetsbuf()", 0); err = -1; goto done; }
		
		memset(&leRec,0,sizeof(leRec));
		
		sscanf(s, "%f %f", &longF, &latF); // will this be slow ???
		leRec.p.pLong = longF * 1000000;
		leRec.p.pLat = latF * 1000000;
		
		if (!myfgetsbuf(s, 255, &ms5)) // data line
			{ TechError("ReadLEFile()", "myfgetsbuf()", 0); err = -1; goto done; }
		
		StringSubstitute(s, ',', ' ');
		sscanf(s, lfFix("%ld %s %s %f %lf %f %f %s %f %f"),
				  &id, leType, pollutantTypeStr, &depth, &mass, &density, &ageInHrs, status, &timeOfCDogReleaseInHrs, &timeSurfacedInHrs);
		
		leRec.leKey = id;
		leRec.leCustomData = 0;
		leRec.pollutantType = OldToNewPollutantCode(GetOldPollutantNumber(pollutantTypeStr),&isEvaporated);//should always be conservative 
		leRec.z = depth;
		leRec.mass = mass;
		leRec.density = density;
		leRec.clockRef = 0; // ??
		
		// From GNOME's point of view, an LE is "released" 
		// when GNOME is supposed to start moving the LE around.
		// This will be when CDOG is done with it, i.e. when CDOG says NEW_SURFACE.

		if(strcmpnocase(status,"NEW_SURFACE")) { 
			continue;  // it is not an LE we care about
			//we don't need to keep track of  WATERCOLUMN,  OLD_SURFACE etc... 
		}

		if(depth != zeroDepth) {
			err= -66; 
			printError("Depth for NEW_SURFACE LE was not zero in ReadCDOGLEFileToSetReleaseTimes");
			goto done;
		}

		leRec.releaseTime = startCDogReleaseTimeInSeconds + (long) (timeSurfacedInHrs*3600);
		//leRec.releaseTime = startCDogReleaseTimeInSeconds;
		leRec.ageInHrsWhenReleased = 0; // ?? this is for evaporation... how do we deal with that age ?? i.e. does it evaporate underwater like it does on the surface ??
		leRec.statusCode = OILSTAT_INWATER;
		
		// now find the LE this corresponds to the leKey.
		// The LE keys are probably the index 1...n or their negatives.
		// Will need to set up an index key to identify which LE in the LERecH this corresponds to
		bestGuessIndex = abs(leRec.leKey) -1;
		
		if(0 <= bestGuessIndex && bestGuessIndex < numLEs) {
			leRec2 = INDEXH(array, bestGuessIndex);
			if(leRec2.leKey ==leRec.leKey) {
				// life is good we found it
				INDEXH(array, bestGuessIndex) = leRec;
				continue;
			}
			else { // life is bad, I suppose we could search for it
				// for now just fall through to complaint below
			}
		
		}
		err= -67; 
		printError("Could not match leKey in ReadCDOGLEFileToSetReleaseTimes");
		goto done;
			
	}
	
done:

	if (ms4.f) FSCloseBuf(&ms4);
	if (ms5.f) FSCloseBuf(&ms5);
	
	return err;
}

OSErr ReadCDOGSubSurfaceGasFile(char *path, long numLEs, LERecH array, Seconds startCDogReleaseTimeInSeconds)
{
	// this function reads in a gas file
	
	char  s[256], pollutantTypeStr[64];
	long n, id;
	float depth,zeroDepth = 0.0; 
	double /*density, */longF, latF;
	double vol, gas_frac, hyd_frac, den_gas;
	BFPB ms4;
	LERec leRec/*,leRec2*/;
	OSErr err = 0;
	long bestGuessIndex, numScanned;
	
	////////

	if(!array)  return -6;
	
	ms4.f = 0;
	// ask for a file
	if(!FileExists(0,0,path)) 
	{
		printNote("There is no CDOG output to import"); 
		return -1;
	}
	if (err = FSOpenBuf(0, 0, path, &ms4, 100000, FALSE))
		{ TechError("ReadLEFile()", "FSOpenBuf()", err); return -1; }
	strcpy(pollutantTypeStr,"CONSERVATIVE");
	if (!myfgetsbuf(s, 255, &ms4)) {;}// header line
	if (!myfgetsbuf(s, 255, &ms4)) {;}// header line
	for (n = 0 ; ; n++) {
		Boolean isEvaporated;
		
		memset(&leRec,0,sizeof(leRec));
		
		//if (!myfgetsbuf(s, 255, &ms4) && n>0) // header line
		if (!myfgetsbuf(s, 255, &ms4)) // header line
			break; // done reading file
			//{ TechError("ReadLEFile()", "myfgetsbuf()", 0); err = -1; goto done; }

		StringSubstitute(s, ',', ' ');
		numScanned = sscanf(s, lfFix("%ld %lf %lf %lf %lf %lf %lf %lf"),
				  &id, &longF, &latF, &depth, &vol, &gas_frac, &hyd_frac, &den_gas);
		
		if (numScanned != 8) {printError("Error reading Gas***.dat file"); err = -1; goto done;}
		leRec.leKey = id;
		leRec.leCustomData = 0;
		leRec.pollutantType = OldToNewPollutantCode(GetOldPollutantNumber(pollutantTypeStr),&isEvaporated);//should always be conservative 
		leRec.p.pLong = longF * 1000000;
		leRec.p.pLat = latF * 1000000;
		//leRec.z = depth;
		leRec.z = zeroDepth;
		//leRec.mass = mass;
		//leRec.density = density;
		leRec.mass = 1;
		leRec.density = 1;
		leRec.clockRef = 0; // ??
		
		// From GNOME's point of view, an LE is "released" 
		// when GNOME is supposed to start moving the LE around.
		// This will be when CDOG is done with it, i.e. when CDOG says NEW_SURFACE.

		/*if(depth != zeroDepth) {
			err= -66; 
			printError("Depth for SURFACE_FLT LE was not zero in ReadCDOGSurfaceOilFileToSetReleaseTimes");
			goto done;
		}*/

		leRec.releaseTime = startCDogReleaseTimeInSeconds;
		//leRec.releaseTime = startCDogReleaseTimeInSeconds + timeSurfacedInHrs*3600;
		leRec.ageInHrsWhenReleased = 0; // ?? this is for evaporation... how do we deal with that age ?? i.e. does it evaporate underwater like it does on the surface ??
		leRec.statusCode = OILSTAT_INWATER;
		
		// now find the LE this corresponds to the leKey.
		// The LE keys are probably the index 1...n or their negatives.
		// Will need to set up an index key to identify which LE in the LERecH this corresponds to
		bestGuessIndex = abs(leRec.leKey) -1;
		
		// probably don't need to match in this case, just take what's there and reset the array size
		// for now don't check the LEs, will probably care about this if/when we look at the subsurface
		// files throughout the run, but since CDOG sets the LEs and we continue to increase the LEKeys
		// with each spill, these numbers may not match
		if(0 <= bestGuessIndex && bestGuessIndex < numLEs) {
			//leRec2 = INDEXH(array, bestGuessIndex);
			//if(leRec2.leKey ==leRec.leKey) {
				// life is good we found it
				INDEXH(array, bestGuessIndex) = leRec;
				continue;
			//}
			//else { // life is bad, I suppose we could search for it
				// for now just fall through to complaint below
			//}
		
		}
		err= -67; 
		printError("Could not match leKey in ReadCDOGSubSurfaceGasFile");
		goto done;
			
	}
	
done:

	if (ms4.f) FSCloseBuf(&ms4);
	
	return err;
}

/////////////////////////////////////////////////
OSErr ReadCDOGSurfaceOilFileToSetReleaseTimes(char *cdogOutputFolderPathWithDelimiter, long numLEs, LERecH array, Seconds startCDogReleaseTimeInSeconds)
{
	// this function sets the release time field for all the surface LEs 
	//	they are marked as SURFACE_FLT in the CDOG file SurfaceOil.dat
	//
	
	char  s[256], leType[64], pollutantTypeStr[64], status[64];
	long n, id;
	float depth,zeroDepth = 0.0; 
	float density, ageInHrs, longF, latF;
	double mass;
	BFPB ms4;
	LERec leRec/*,leRec2*/;
	OSErr err = 0;
	char fileName[256],path[256];
	float timeOfCDogReleaseInHrs, timeSurfacedInHrs;
	long bestGuessIndex, maxIndex = 0, numScanned;
	
	////////

	if(!array)  return -6;
	
	ms4.f = 0;
	strcpy(path,cdogOutputFolderPathWithDelimiter);
	strcat(path,"SurfaceOil.dat");
	if(!FileExists(0,0,path)) 
	{
		// couldn't find file in cdog output folder, allow option to pick a gas file or another SurfaceOil file
		err = GetFilePath(path,fileName,5);
		if (err)
		{
			printNote("There is no CDOG output to import"); 
			return -1;
		}
		if (!strncmpnocase(fileName,"SurfaceOil",10))
			goto ReadSurfaceOilFile;
		if (!strncmpnocase(fileName,"Gas",3))
		{
			err = ReadCDOGSubSurfaceGasFile(path, numLEs, array, startCDogReleaseTimeInSeconds);
		}
		else
		{
			printNote("File is not an allowable CDOG output file");	// maybe check FILE4,5
			err = -1;
		}
		return err;
	}

ReadSurfaceOilFile:
	if (err = FSOpenBuf(0, 0, path, &ms4, 100000, FALSE))
		{ TechError("ReadLEFile()", "FSOpenBuf()", err); return -1; }
	
	for (n = 0 ; ; n++) {
		Boolean isEvaporated;
		
		memset(&leRec,0,sizeof(leRec));
		
		if (!myfgetsbuf(s, 255, &ms4) /*&& n>0*/) // no header line
			break; // done reading file
			//{ TechError("ReadLEFile()", "myfgetsbuf()", 0); err = -1; goto done; }

		// note should check if CDOG is outputting lat/lon or lon/lat...
		StringSubstitute(s, ',', ' ');
		numScanned = sscanf(s, lfFix("%ld %s %s %f %f %f %lf %f %f %s %f %f"),
				  &id, leType, pollutantTypeStr, &longF, &latF, &depth, &mass, &density, &ageInHrs, status, &timeOfCDogReleaseInHrs, &timeSurfacedInHrs);
		
		if (numScanned != 12) {printError("Error reading SurfaceOil.dat file"); err = -1; goto done;}
		leRec.leKey = id;
		leRec.leCustomData = 0;
		leRec.pollutantType = OldToNewPollutantCode(GetOldPollutantNumber(pollutantTypeStr),&isEvaporated);//should always be conservative 
		leRec.p.pLong = longF * 1000000;
		leRec.p.pLat = latF * 1000000;
		leRec.z = depth;
		leRec.mass = mass;
		leRec.density = density;
		leRec.clockRef = 0; // ??
		
		// From GNOME's point of view, an LE is "released" 
		// when GNOME is supposed to start moving the LE around.
		// This will be when CDOG is done with it, i.e. when CDOG says NEW_SURFACE.

		if(strcmpnocase(status,"SURFACE_FLT")) { 
			continue;  // it is not an LE we care about, but this should be an error
			//we don't need to keep track of  WATERCOLUMN,  OLD_SURFACE etc... 
		}

		if(depth != zeroDepth) {
			//err= -66; 
			//printError("Depth for SURFACE_FLT LE was not zero in ReadCDOGSurfaceOilFileToSetReleaseTimes");
			printNote("Depth for SURFACE_FLT LE was not zero in ReadCDOGSurfaceOilFileToSetReleaseTimes");
			//goto done;
		}

		leRec.releaseTime = startCDogReleaseTimeInSeconds + (long)(timeSurfacedInHrs*3600);
		leRec.ageInHrsWhenReleased = 0; // ?? this is for evaporation... how do we deal with that age ?? i.e. does it evaporate underwater like it does on the surface ??
		leRec.statusCode = OILSTAT_INWATER;
		
		// now find the LE this corresponds to the leKey.
		// The LE keys are probably the index 1...n or their negatives.
		// Will need to set up an index key to identify which LE in the LERecH this corresponds to
		bestGuessIndex = abs(leRec.leKey) -1;
		
		// probably don't need to match in this case, just take what's there and reset the array size
		// for now don't check the LEs, will probably care about this if/when we look at the subsurface
		// files throughout the run, but since CDOG sets the LEs and we continue to increase the LEKeys
		// with each spill, these numbers may not match
		//if(0 <= bestGuessIndex && bestGuessIndex < numLEs) {
		if(0 <= bestGuessIndex && bestGuessIndex < numLEs && maxIndex==0) {
			//leRec2 = INDEXH(array, bestGuessIndex);
			//if(leRec2.leKey ==leRec.leKey) {
				// life is good we found it
				INDEXH(array, bestGuessIndex) = leRec;
				continue;
			//}
			//else { // life is bad, I suppose we could search for it
				// for now just fall through to complaint below
			//}
		}
		else
		{
			// find the max index so user can reset numLEs accordingly
			if (bestGuessIndex>maxIndex) {maxIndex = bestGuessIndex; err = -67; continue;}
			if (maxIndex>0) continue;
		}	
		err= -67; 
		printError("Could not match leKey in ReadCDOGSurfaceOilFileToSetReleaseTimes");
		goto done;
			
	}
	
done:

	if (maxIndex>0)
	{
		char errmsg[256];
		sprintf(errmsg,"The CDOG output file had %li LEs. Reset the number of LEs in the spill to import this file",maxIndex+1);
		printError(errmsg);
	}
	if (ms4.f) FSCloseBuf(&ms4);
	
	return err;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

OSErr CDOGLEList::ReadCDOGOutputFiles()
{
	OSErr err = 0;
	char cdogFolderPathWithDelimiter[256];
	char cdogOutputFolderPathWithDelimiter[256];
	//long i, numFiles = 28;
	long i, numFiles = 1;	// have an input option as to which file type...
	long numLEs = fSetSummary.numOfLEs;
	Seconds startCDogReleaseTimeInSeconds = fSetSummary.startRelTime;
	Seconds largeReleaseTime = startCDogReleaseTimeInSeconds + 3600 * 24 * 100;
	LERecH array = 0;  // rats !!
	LERec thisLE;
	CDOGLEList 	*uncertaintyLEList=0;
	//unsigned long  end, start;
	//char note[256];

	
	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogOutputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"output",DIRDELIMITER);
	
	// note, number of LEs is not the same as the number of particles that come from CDOG
	// code goes here, should allow to reset the array ?
	// set array inside subroutine
	if (!(array = (LERecH)_NewHandleClear(sizeof(LERec)*numLEs)))
		{ TechError("ReadCDOGOutputFiles()", "_NewHandleClear()", 0); err = -1; goto done; }

	memset(&thisLE,0,sizeof(thisLE));
	
	// to allow reset should specifically set default LE values, lat,lon, z of the source, so don't
	// get extra random blue dots if another data set is loaded fSetSummary.startRelPos.pLong, pLat, z
	// what about leKey?
	for (i=0;i<numLEs;i++)
	{
		this -> GetLE (i, &thisLE);
		//INDEXH(array,i) = thisLE;
		INDEXH(array,i).p = fSetSummary.startRelPos;
		INDEXH(array,i).z = fSetSummary.z;
		INDEXH(array,i).releaseTime = largeReleaseTime;
	}
			
	//start = TickCount();
	if(numFiles==1)
	{
		err = ReadCDOGSurfaceOilFileToSetReleaseTimes(cdogOutputFolderPathWithDelimiter, numLEs, array, startCDogReleaseTimeInSeconds);
		if(err) goto done;
	}
	else
	{
		for (i=1;i<=numFiles;i++)
		{
			err = ReadCDOGLEFileToSetReleaseTimes(cdogOutputFolderPathWithDelimiter,i,numLEs, array, startCDogReleaseTimeInSeconds);
			if(err) goto done;
		}
	}
	//end = TickCount();
	//sprintf(note,"took %f seconds",(float)(end-start)/60.0);
	//printNote(note);
	// need to set the releaseTimes to something large, so watercolumn LEs won't get released
	// startCDogReleaseTimeInSeconds + 3600 *24 * 100
	// delete initialLEs if it exists
	if (initialLEs)	// dispose of LE distribution array, if any, STH
	{
		initialLEs->Dispose();
		delete (initialLEs);
		initialLEs = nil;
	}
	this -> initialLEs = new CMyList (sizeof (InitialLEInfoRec));
	
	uncertaintyLEList = (CDOGLEList*)model->GetMirroredLEList(this);
	if (uncertaintyLEList) 
	{
		if (uncertaintyLEList -> initialLEs)	// dispose of LE distribution array, if any, STH
		{
			uncertaintyLEList -> initialLEs -> Dispose();
			delete (uncertaintyLEList -> initialLEs);
			uncertaintyLEList -> initialLEs = nil;
		}
	
		uncertaintyLEList -> initialLEs = new CMyList (sizeof (InitialLEInfoRec));
	}
	if (this -> initialLEs && uncertaintyLEList && uncertaintyLEList -> initialLEs)
	{
		InitialLEInfoRec thisSavedRec;
		LERec thisLERec;
		
		this -> initialLEs -> IList ();
		uncertaintyLEList -> initialLEs -> IList ();
		
		memset(&thisSavedRec,0,sizeof(thisSavedRec));
		
		//for (i = 0; i <= numLEs; ++i)
		for (i = 0; i < numLEs; ++i)
		{
			thisLERec = INDEXH(array, i);
			
			thisSavedRec.p = thisLERec.p;
			thisSavedRec.z = thisLERec.z;
			thisSavedRec.ageInHrsWhenReleased = thisLERec.ageInHrsWhenReleased;
			thisSavedRec.releaseTime = thisLERec.releaseTime;
			thisSavedRec.pollutantType = thisLERec.pollutantType;
			thisSavedRec.density = thisLERec.density;
			err = this -> initialLEs -> AppendItem ((Ptr) &thisSavedRec);
			if(err) goto done;
			err = uncertaintyLEList -> initialLEs -> AppendItem ((Ptr) &thisSavedRec);
			if(err) goto done;
		}
	}
	else
	{
		err = memFullErr;
		goto done;
	}
	this -> Reset(false);
	uncertaintyLEList -> Reset(false);
	//model -> NewDirtNotification();
	model -> NewDirtNotification(DIRTY_EVERYTHING);	// so resets the runbar in advanced mode too

done:

	if (err)
	{
		printError("Error importing CDOG output");
		if (initialLEs)	
		{
			initialLEs -> Dispose();
			delete (initialLEs);
			initialLEs = nil;
		}

		if (uncertaintyLEList && uncertaintyLEList -> initialLEs)	
		{
			uncertaintyLEList -> initialLEs -> Dispose();
			delete (uncertaintyLEList -> initialLEs);
			uncertaintyLEList -> initialLEs = nil;
		}

	}
	if (array) {DisposeHandle((Handle)array); array = 0;}
	return err;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
CDOGLEList::CDOGLEList()
{
	//
	memset(&fCDOGParameters,0,sizeof(fCDOGParameters));
	// set defaults...
	//fCDOGSpillParameters.oilDischargeRate = .01829;
	//fCDOGSpillParameters.gasDischargeRate = 3.659;
	//fCDOGSpillParameters.dischargeRateType = 1;

	//fCDOGParameters.orificeDiameter = .1;
	//fCDOGParameters.temp = 80;
	//fCDOGParameters.density = 842.5;	// kg/m^3
	fCDOGParameters.equilibriumCurves = 2;	// methane=1, natural gas=2
	fCDOGParameters.bubbleRadius = .0025;
	fCDOGParameters.molecularWt = .0191;
	fCDOGParameters.hydrateDensity = 920;
	fCDOGParameters.separationFlag = 1;
	fCDOGParameters.hydrateProcess = 1;	//
	fCDOGParameters.dropSize = 0;	//use default
	//fCDOGParameters.dischargeRateType = 1;
	
	fCDOGParameters.duration = 3600;
	fCDOGParameters.isContinuousRelease = true;

	fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath[0] = 0;	// folder name
	//fCDOGHydrodynamicInfo.timeInterval = .333;	// hrs
	//fCDOGHydrodynamicInfo.period = 1000;	// hrs
	fCDOGHydrodynamicInfo.timeInterval = 9999;	// hrs
	fCDOGHydrodynamicInfo.period = 9999;	// hrs
	fCDOGHydrodynamicInfo.methodOfDeterminingCurrents = 1;	// select a folder with all the currents

	fCDOGDiffusivityInfo.horizDiff = 10.;	// m^2/s
	fCDOGDiffusivityInfo.vertDiff = .0001;	// m^2/s
	fCDOGDiffusivityInfo.timeStep = .1 * 3600;	// seconds
	
	fCDOGTempSalInfo.temperatureFieldFilePath[0]=0;
	fCDOGTempSalInfo.salinityFieldFilePath[0]=0;
	fCDOGTempSalInfo.methodOfDeterminingTempSal = 1;

	fOutputSubsurfaceFiles = false;
	fOutputGasFiles = false;
	//
	fDepthValuesH = 0;
	fLengthOfGridCellInMeters = 500;	// calculate from the map
	fNumGridCells = 1;
	
	bIsHydrogenSulfide = false;
	
	fCDOGUserUnits.temperatureUnits = kDegreesC;
	fCDOGUserUnits.densityUnits = kKgM3;
	fCDOGUserUnits.diameterUnits = kMeters;
	fCDOGUserUnits.dischargeType = 1;	//oil
	fCDOGUserUnits.gorUnits = 1;	//
	fCDOGUserUnits.dischargeUnits = 1;	//
	fCDOGUserUnits.molWtUnits = 2;	// g/mol
	
	fDepthUnits = 1; // meters

	fMethodOfDeterminingHydrodynamics = 1;	// 1 = create profile, 2 = use GNOME netcdf file, 3 = already in CDOG folder, 4 = Move files

	fDischargeDataH = 0;
}

void CDOGLEList::Dispose ()
{
	/////
	if (fDepthValuesH) {DisposeHandle((Handle)fDepthValuesH); fDepthValuesH = 0;}
	if (fDischargeDataH) {DisposeHandle((Handle)fDischargeDataH); fDischargeDataH = 0;}
	TOLEList::Dispose ();
}

OSErr CDOGLEList::Reset(Boolean newKeys)
{
	OSErr err = TOLEList::Reset(newKeys);
	//TSprayLEList *sprayedHOwner = this;
	if(err) return err;
	
	/*
	if (this->fLeType == UNCERTAINTY_LE)
	{ // then we need to use the sprayed points kept in our certainty buddy
		sprayedHOwner =  (TSprayLEList *) model -> GetLEListOwner(this);
		if (!sprayedHOwner)
			return 0; // this will happen on the resets called before we are put in the model's list
	}
	
	// override the LE positions
	if(this -> LEHandle && sprayedHOwner && sprayedHOwner -> fSprayedH)
	{
		long i,j;
		WorldPoint wp;
		LERec leRec;
	
		for(i = 0; i < this -> numOfLEs; i++)
		{
			j = sprayedHOwner -> LEIndexToSprayedIndex(i);
			if(j < 0) 
				return -1; // there is some problem
			wp = INDEXH(sprayedHOwner -> fSprayedH ,j);
			leRec = INDEXH(this -> LEHandle,i);
			leRec.p = wp;
			INDEXH(this -> LEHandle,i) = leRec;
		}
	}*/

	return 0;
}

long CDOGLEList::GetListLength()
{
	long summaryLines = 0;
	
	Boolean owned = (fOwnersUniqueID.counter != 0);
	if(owned) return 0; // we will not be in the list
	
	summaryLines++; // toggle with oil type and amount
	
	if(bOpen)
	{
		if(model->GetModelMode() == ADVANCEDMODE) summaryLines++; // active button

		if(model->GetModelMode() == ADVANCEDMODE) summaryLines += 1;// windage
		if(model->GetModelMode() == ADVANCEDMODE) summaryLines += 1;// oil discharge rate
		if(model->GetModelMode() == ADVANCEDMODE) summaryLines += 1;// gas discharge rate
		if(model->GetModelMode() == ADVANCEDMODE) summaryLines += 1;// GOR

		summaryLines++; // show initial positions
		summaryLines++; // release time/position toggle
		
		if(bReleasePositionOpen)
		{
			summaryLines++; // release time
			summaryLines++; // duration
			//if (fSetSummary.bWantEndRelTime) summaryLines++; // release end time
			summaryLines++; // release position
			if (fSetSummary.bWantEndRelPosition) summaryLines++; // release end position
		}
		
		summaryLines++; // Mass Balance toggle
		
		if (bMassBalanceOpen)
		{
			summaryLines++;// released 
			summaryLines++;// floating
			summaryLines++;// beached
			summaryLines++;// evaporated/dispersed
			summaryLines++;// offmap
		}
				
	}
	
	return summaryLines;
}

ListItem CDOGLEList::GetNthListItem(long n, short indent, short *style, char *text)
{
	char dateStr [64],infoStr [255];
	//char polutantName [64];
	char unitsStr [64], valStr[32], valStr2[32], valStr3[32];
	char latString[20],longString[20];
	char roundLat,roundLong;
	LERec LE;
	ListItem item = { this, 0, indent, 0 };
	long numDischarges = GetNumDischarges();
	double dischargeRate=0, q_oil, q_gas, gor;

	// number  and type of LE's
	if (n == 0) {
		item.index = I_CDOG_LEFIRSTLINE;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		//GetPollutantName (fSetSummary.pollutantType, polutantName);
		GetLeUnitsStr(unitsStr,fSetSummary.massUnits);
		StringWithoutTrailingZeros(infoStr,fSetSummary.totalMass,6); 
		//sprintf(text, "%s : %s %s", polutantName,infoStr,unitsStr);
		//sprintf(text, "%s:  %s : %s %s", fSetSummary.spillName, polutantName,infoStr,unitsStr);
		sprintf(text, "%s: %s %s", fSetSummary.spillName, /*polutantName,*/infoStr,unitsStr);
		if(!bActive)*style = italic; // JLM 6/14/10
		return item;
	}
	n -= 1;
	
	if (bOpen)
	{
		if (model->GetModelMode() == ADVANCEDMODE) {
			if (n == 0) {
				item.indent++;
				item.index = I_CDOG_ACTIVE;
				item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
				strcpy(text, "Active");
				return item;
			}
			n -= 1;
	
			////
			if (n == 0)
			{
				item.index = I_CDOG_WINDAGE;
				StringWithoutTrailingZeros(valStr,fWindageData.windageA * 100,2);
				StringWithoutTrailingZeros(valStr2,fWindageData.windageB * 100,2);
				if (fWindageData.persistence==-1) 
					sprintf(text, "Windage: %s%% to %s%%, Persistence: Infinite", valStr, valStr2);
				else
				{
					StringWithoutTrailingZeros(valStr3,fWindageData.persistence,2);
					sprintf(text, "Windage: %s%% to %s%%, Persistence: %s hrs", valStr, valStr2, valStr3);
				}
				return item;
			}
			else
				n -= 1;
		}
		// code goes here, do something different for constant/variable discharge
		// issue if user has a units mismatch

		if (numDischarges>0) {dischargeRate = INDEXH(fDischargeDataH,0).q_oil; gor = INDEXH(fDischargeDataH,0).q_gas;}
		if (fCDOGUserUnits.dischargeType==2) {q_gas = dischargeRate; q_oil = q_gas / gor;if (fCDOGUserUnits.gorUnits==2 && fCDOGUserUnits.dischargeUnits==2) q_oil = q_oil *1000;}
		else {q_oil = dischargeRate; {q_gas = gor * q_oil;} if (fCDOGUserUnits.gorUnits==2 && fCDOGUserUnits.dischargeUnits==2) q_gas=q_gas/1000;}
		if (n == 0) {
			item.index = I_CDOG_OILDISCHARGERATE;			
		

			if (numDischarges==1) 
			{
				StringWithoutTrailingZeros(valStr,q_oil,4);
				if (fCDOGUserUnits.dischargeUnits==1) 
					sprintf(text, "Oil Discharge Rate: %s m**3/s", valStr);
				else
					sprintf(text, "Oil Discharge Rate: %s BOPD", valStr);
			}
			else if (numDischarges>1)
			{
				StringWithoutTrailingZeros(valStr,q_oil,4);
				if (fCDOGUserUnits.dischargeUnits==1) 
					sprintf(text, "Oil Discharge Rate: %s m**3/s", valStr);
				else
					sprintf(text, "Oil Discharge Rate: %s BOPD", valStr);
				//sprintf(text, "Variable discharge");
			}
			else
			{
				sprintf(text, "Oil Discharge Rate: not set");	// shouldn't happen
			}
			return item;
		}
		n -= 1;
		if (n == 0) {
			item.index = I_CDOG_GASDISCHARGERATE;
			if (numDischarges==1) 
			{
				StringWithoutTrailingZeros(valStr,q_gas,4);
				if (fCDOGUserUnits.dischargeUnits==1) 
					sprintf(text, "Gas Discharge Rate: %s m**3/s", valStr);
				else
					sprintf(text, "Gas Discharge Rate: %s MSCF", valStr);
			}
			else if (numDischarges>1)
			{
				StringWithoutTrailingZeros(valStr,q_gas,4);
				if (fCDOGUserUnits.dischargeUnits==1) 
					sprintf(text, "Gas Discharge Rate: %s m**3/s", valStr);
				else
					sprintf(text, "Gas Discharge Rate: %s MSCF", valStr);
				//sprintf(text, "Variable discharge");
			}
			else
			{
				sprintf(text, "Gas Discharge Rate: not set");	// shouldn't happen
			}
			return item;
		}
		n -= 1;
		if (n == 0) {
			item.index = I_CDOG_GOR;
			if (numDischarges==1) 
			{
				StringWithoutTrailingZeros(valStr,gor,4);
				if (fCDOGUserUnits.gorUnits==1) 
					sprintf(text, "Gas Oil Ratio: %s SI Units", valStr);
				else if (fCDOGUserUnits.gorUnits==2)
					sprintf(text, "Gas Oil Ratio: %s SCFD/BOPD", valStr);
				else
					sprintf(text, "Gas Oil Ratio: %s MSCF/BOPD", valStr);
			}
			else if (numDischarges>1)
			{
				StringWithoutTrailingZeros(valStr,gor,4);
				if (fCDOGUserUnits.gorUnits==1) 
					sprintf(text, "Gas Oil Ratio: %s SI Units", valStr);
				else if (fCDOGUserUnits.gorUnits==2)
					sprintf(text, "Gas Oil Ratio: %s SCFD/BOPD", valStr);
				else
					sprintf(text, "Gas Oil Ratio: %s MSCF/BOPD", valStr);
				//sprintf(text, "Variable discharge");
			}
			else
			{
				sprintf(text, "Gas Oil Ratio: not set");	// shouldn't happen
			}
			return item;
		}
		n -= 1;
		//n -= 1;
		
		item.indent++; // all items are indented
		
		if (n == 0) {
			item.index = I_CDOG_LESHOWHIDE;
			item.bullet = binitialLEsVisible ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Initial Positions");
			return item;
		}
		n -= 1;

		////
		if (n == 0) {
			item.index = I_CDOG_LERELEASE_TIMEPOSITION;
			item.bullet = bReleasePositionOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			sprintf(text, "Release Time/Position");
			return item;
		}
		n -= 1;


		/////////////////////////////////////////////////
		if(bReleasePositionOpen)
		{
			// release time
			if (n == 0) {
				//item.indent++;
				Secs2DateString2 (fSetSummary.startRelTime, dateStr);
				sprintf (text, "Start Time: %s", dateStr);
				//if(fSetSummary.bWantEndRelTime) sprintf (text, "Start Time: %s", dateStr);
				//else sprintf (text, "Time: %s", dateStr);
				return item;
			}
			n -= 1;
			
			//if (fSetSummary.bWantEndRelTime)
			{
				// make it duration
				double duration = GetSpillDuration() / 3600.;
				if (n == 0) {
					//item.indent++;
					//Secs2DateString2 (fSetSummary.endRelTime, dateStr);
					//sprintf (text, "End Time: %s", dateStr);
					StringWithoutTrailingZeros(valStr,duration,4);
					sprintf(text, "Duration: %s hours", valStr);
					return item;
				}
				n -= 1;
			}
			
			// release position
			if (n == 0) {
				//item.indent++;
				WorldPointToStrings2(fSetSummary.startRelPos, latString, &roundLat, longString, &roundLong);	
				SimplifyLLString(longString, 3, roundLong);
				SimplifyLLString(latString, 3, roundLat);
				StringWithoutTrailingZeros(valStr,fSetSummary.z,2);
				//if(fSetSummary.bWantEndRelPosition) sprintf(text, "Start Position: %s, %s", latString,longString);
				/*else*/ sprintf(text, "Position: %s, %s, %s m", latString,longString,valStr);
				return item;
			}
			n -= 1;
	
			//////
			/*if (fSetSummary.bWantEndRelPosition)
			{
				if (n == 0) {
					//item.indent++;
					WorldPointToStrings2(fSetSummary.endRelPos, latString, &roundLat, longString, &roundLong);	
					SimplifyLLString(longString, 3, roundLong);
					SimplifyLLString(latString, 3, roundLat);
					sprintf(text, "End Position: %s, %s", latString,longString);
					return item;
				}
				n -= 1;
			}*/
		}

		/////////////////////////////////////////////////
		
		////
		if (n == 0) {
			item.index = I_CDOG_LERELEASE_MASSBALANCE;
			//item.indent++;
			item.bullet = bMassBalanceOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			sprintf(text, "Splot Mass Balance (Best Estimate)");
			return item;
		}
		n -= 1;

		////
		if (bMassBalanceOpen) // released, evaporated/dispersed, offmap, beached
		{
			////////////
			if (n == 0) {// released
				this->GetMassBalanceLines(TRUE,text,nil,nil,nil,nil,nil,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (n == 0) {// floating
				this->GetMassBalanceLines(TRUE,nil,text,nil,nil,nil,nil,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (n == 0) {// beached
				this->GetMassBalanceLines(TRUE,nil,nil,text,nil,nil,nil,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (n == 0) {// evaporated/dispersed
				this->GetMassBalanceLines(TRUE,nil,nil,nil,text,nil,nil,nil);
				//item.indent++;
				return item;
			}			
			n -= 1;

			if (fDispersantData.bDisperseOil || fAdiosDataH)
			{
				if (n == 0) {// dispersed
					this->GetMassBalanceLines(TRUE,nil,nil,nil,nil,text,nil,nil);
					//item.indent++;
					return item;
				}
				n -= 1;
			}

			if (n == 0) {// offmap
				this->GetMassBalanceLines(TRUE,nil,nil,nil,nil,nil,text,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (fAdiosDataH)
			{
				if (n == 0) {// dispersed
					this->GetMassBalanceLines(TRUE,nil,nil,nil,nil,nil,nil,text);
					//item.indent++;
					return item;
				}
				n -= 1;
			}
		}

	}
	
	item.owner = 0;
	
	return item;
}

Boolean CDOGLEList::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet && item.index == I_CDOG_LEFIRSTLINE)
		{ bOpen = !bOpen; return TRUE; }
		
	if (inBullet && item.index == I_CDOG_LESHOWHIDE) {
		binitialLEsVisible = !binitialLEsVisible;
		model->NewDirtNotification(DIRTY_MAPDRAWINGRECT);
		return TRUE;
	}

	if (inBullet && item.index == I_CDOG_ACTIVE)
		{ 
			TOLEList 	*uncertaintyLEList = (TOLEList*)model->GetMirroredLEList(this);
			bActive = !bActive; 
			uncertaintyLEList->SetActive(bActive);
			model->NewDirtNotification(DIRTY_RUNBAR); model->NewDirtNotification(); return TRUE; 
		}

	if (inBullet && item.index == I_CDOG_LERELEASE_TIMEPOSITION)
		{ bReleasePositionOpen = !bReleasePositionOpen; return TRUE; }
		
	if (inBullet && item.index == I_CDOG_LERELEASE_MASSBALANCE)
		{ bMassBalanceOpen = !bMassBalanceOpen; return TRUE; }
	
	//JLM  9/18/98
	if(doubleClick)  return this -> SettingsItem (item); // do the settings
	
	// do other click operations...
	
	return FALSE;
}

Boolean CDOGLEList::FunctionEnabled(ListItem item, short buttonID)
{
	switch (buttonID) {
		case UPBUTTON:
		case DOWNBUTTON:
			return FALSE;
	}
	
	if (item.index == I_CDOG_LEFIRSTLINE)
		switch (buttonID) {
			case ADDBUTTON: return FALSE;
			case SETTINGSBUTTON: return TRUE;
			case DELETEBUTTON: return TRUE;
		}
	
	switch (buttonID) {
		case ADDBUTTON: return FALSE;
		case SETTINGSBUTTON: return TRUE;
		case DELETEBUTTON: return FALSE;
	}
	
	return FALSE;
}

OSErr CDOGLEList::SettingsItem(ListItem item)
{
	// JLM 9/17/99 with sohail
	// for LE sets from a file, we don't have the dialog implemented yet
	// code goes here
	
	/*if(this -> initialLEs) {
		// then this guy was loaded from a file
		printNote("The settings dialog for LE sets loaded from a file is unimplemented.");
		return TRUE;
	}*/
	
	//if (item.index == I_LEWINDAGE)	// maybe have the windage dialog come up in this case
	(void)CDOGSpillSettingsDialog( (CDOGLEList *)item.owner);
	return TRUE;
}

OSErr CDOGLEList::DeleteItem(ListItem item)
{
	if (item.index == I_CDOG_LEFIRSTLINE)
		return model->DropLEList(this, false);
	
	return 0;
}

/////////////////////////////////////////////////
#define kCDOGLEListVersion 3
//#define kCDOGLEListVersion 2
//#define kCDOGLEListVersion 1

OSErr CDOGLEList::Write(BFPB *bfpb)
{
	long i, version = kCDOGLEListVersion;
	long numDischarges = GetNumDischarges(), numProfiles = GetNumProfiles();
	ClassID id = GetClassID ();
	OSErr	err = noErr;

	if(err = TOLEList::Write(bfpb)) return err;
	
	StartReadWriteSequence("CDOGLEList::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;

	//if (err = WriteMacValue(bfpb, fCDOGSpillParameters.oilDischargeRate)) return err;
	//if (err = WriteMacValue(bfpb, fCDOGSpillParameters.gasDischargeRate)) return err;

	//if (err = WriteMacValue(bfpb, fCDOGSpillParameters.dischargeRateType)) return err;

	//if (err = WriteMacValue(bfpb, fCDOGParameters.orificeDiameter)) return err;
	//if (err = WriteMacValue(bfpb, fCDOGParameters.temp)) return err;
	//if (err = WriteMacValue(bfpb, fCDOGParameters.density)) return err;
	if (err = WriteMacValue(bfpb, fCDOGParameters.equilibriumCurves)) return err;
	if (err = WriteMacValue(bfpb, fCDOGParameters.bubbleRadius)) return err;
	if (err = WriteMacValue(bfpb, fCDOGParameters.molecularWt)) return err;
	if (err = WriteMacValue(bfpb, fCDOGParameters.hydrateDensity)) return err;
	if (err = WriteMacValue(bfpb, fCDOGParameters.separationFlag)) return err;
	if (err = WriteMacValue(bfpb, fCDOGParameters.hydrateProcess)) return err;
	if (err = WriteMacValue(bfpb, fCDOGParameters.dropSize)) return err;
	//fCDOGParameters.dischargeRateType = 1;
	if (err = WriteMacValue(bfpb, fCDOGParameters.duration)) return err;
	if (err = WriteMacValue(bfpb, fCDOGParameters.isContinuousRelease)) return err;
	//fCDOGParameters.duration = 3600;
	//fCDOGParameters.isContinuousRelease = true;
	if (err = WriteMacValue(bfpb, fDepthUnits)) return err;
	// fDepthUnits

	if (err = WriteMacValue(bfpb, fCDOGDiffusivityInfo.timeStep)) return err;
	if (err = WriteMacValue(bfpb, fCDOGDiffusivityInfo.horizDiff)) return err;
	if (err = WriteMacValue(bfpb, fCDOGDiffusivityInfo.vertDiff)) return err;

	if (err = WriteMacValue(bfpb, fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fCDOGHydrodynamicInfo.timeInterval)) return err;
	if (err = WriteMacValue(bfpb, fCDOGHydrodynamicInfo.period)) return err;
	if (err = WriteMacValue(bfpb, fCDOGHydrodynamicInfo.methodOfDeterminingCurrents)) return err;

	if (err = WriteMacValue(bfpb, fCDOGTempSalInfo.temperatureFieldFilePath, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fCDOGTempSalInfo.salinityFieldFilePath, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fCDOGTempSalInfo.methodOfDeterminingTempSal)) return err;

	if (err = WriteMacValue(bfpb, fOutputSubsurfaceFiles)) return err;
	if (err = WriteMacValue(bfpb, fOutputGasFiles)) return err;

	if (err = WriteMacValue(bfpb, fLengthOfGridCellInMeters)) return err;	//may want to edit these down the road
	if (err = WriteMacValue(bfpb, fNumGridCells)) return err;

	if (err = WriteMacValue(bfpb, bIsHydrogenSulfide)) return err;	
	if (err = WriteMacValue(bfpb, fMethodOfDeterminingHydrodynamics)) return err;

	if (err = WriteMacValue(bfpb, fCDOGUserUnits.temperatureUnits)) return err;
	if (err = WriteMacValue(bfpb, fCDOGUserUnits.densityUnits)) return err;
	if (err = WriteMacValue(bfpb, fCDOGUserUnits.diameterUnits)) return err;
	if (err = WriteMacValue(bfpb, fCDOGUserUnits.dischargeType)) return err;
	if (err = WriteMacValue(bfpb, fCDOGUserUnits.gorUnits)) return err;
	if (err = WriteMacValue(bfpb, fCDOGUserUnits.dischargeUnits)) return err;
	if (err = WriteMacValue(bfpb, fCDOGUserUnits.molWtUnits)) return err;
	
	
	if (err = WriteMacValue(bfpb, numDischarges)) return err;
	if (numDischarges>0)
	{
		for (i=0;i<numDischarges;i++)
		{
			DischargeData dischargeData = INDEXH(fDischargeDataH,i);
			if (err = WriteMacValue(bfpb, dischargeData.time)) return err;
			if (err = WriteMacValue(bfpb, dischargeData.q_oil)) return err;
			if (err = WriteMacValue(bfpb, dischargeData.q_gas)) return err;
			if (err = WriteMacValue(bfpb, dischargeData.temp)) return err;
			if (err = WriteMacValue(bfpb, dischargeData.diam)) return err;
			if (err = WriteMacValue(bfpb, dischargeData.rho_oil)) return err;
			if (err = WriteMacValue(bfpb, dischargeData.n_den)) return err;
			if (err = WriteMacValue(bfpb, dischargeData.output_int)) return err;
		}
	}
		
	if (err = WriteMacValue(bfpb, numProfiles)) return err;
	if (numProfiles>0)
	{
		for (i=0;i<numProfiles;i++)
		{
			DepthValuesSet depthProfile = INDEXH(fDepthValuesH,i);
			if (err = WriteMacValue(bfpb, depthProfile.depth)) return err;
			if (err = WriteMacValue(bfpb, depthProfile.value.u)) return err;
			if (err = WriteMacValue(bfpb, depthProfile.value.v)) return err;
			if (err = WriteMacValue(bfpb, depthProfile.w)) return err;
			if (err = WriteMacValue(bfpb, depthProfile.temp)) return err;
			if (err = WriteMacValue(bfpb, depthProfile.sal)) return err;
		}
	}

	return 0;
}

OSErr CDOGLEList::Read(BFPB *bfpb)
{
	char 	c;
	long 	i, version, numDischarges, numProfiles;
	ClassID id;
	OSErr	err = noErr;
	DischargeData dischargeData;
	DepthValuesSet depthProfile;
	
	this -> Dispose();

	if(err = TOLEList::Read(bfpb)) return err;
	
	StartReadWriteSequence("CDOGLEList::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("CDOGLEList::Read()", "id != GetClassID", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > kCDOGLEListVersion || version < 1) { printSaveFileVersionError(); return -1; }

	//if (err = ReadMacValue(bfpb, &fCDOGSpillParameters.oilDischargeRate)) return err;
	//if (err = ReadMacValue(bfpb, &fCDOGSpillParameters.gasDischargeRate)) return err;
	
	//if (err = ReadMacValue(bfpb, &fCDOGSpillParameters.dischargeRateType)) return err;

	//if (err = ReadMacValue(bfpb, &fCDOGParameters.orificeDiameter)) return err;
	//if (err = ReadMacValue(bfpb, &fCDOGParameters.temp)) return err;
	//if (err = ReadMacValue(bfpb, &fCDOGParameters.density)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGParameters.equilibriumCurves)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGParameters.bubbleRadius)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGParameters.molecularWt)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGParameters.hydrateDensity)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGParameters.separationFlag)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGParameters.hydrateProcess)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGParameters.dropSize)) return err;
	//fCDOGParameters.dischargeRateType = 1;
	if (version > 2)
	{
	if (err = ReadMacValue(bfpb, &fCDOGParameters.duration)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGParameters.isContinuousRelease)) return err;
	//fCDOGParameters.duration = 3600;
	//fCDOGParameters.isContinuousRelease = true;
	}
	if (version > 2) {if (err = ReadMacValue(bfpb, &fDepthUnits)) return err;}

	if (err = ReadMacValue(bfpb, &fCDOGDiffusivityInfo.timeStep)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGDiffusivityInfo.horizDiff)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGDiffusivityInfo.vertDiff)) return err;

	if (err = ReadMacValue(bfpb, fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath, kMaxNameLen)) return err;
	ResolvePath(fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath); // JLM 6/3/10
	if (err = ReadMacValue(bfpb, &fCDOGHydrodynamicInfo.timeInterval)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGHydrodynamicInfo.period)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGHydrodynamicInfo.methodOfDeterminingCurrents)) return err;

	if (err = ReadMacValue(bfpb, fCDOGTempSalInfo.temperatureFieldFilePath, kMaxNameLen)) return err;
	ResolvePath(fCDOGTempSalInfo.temperatureFieldFilePath); // JLM 6/3/10
	if (err = ReadMacValue(bfpb, fCDOGTempSalInfo.salinityFieldFilePath, kMaxNameLen)) return err;
	ResolvePath(fCDOGTempSalInfo.salinityFieldFilePath); // JLM 6/3/10
	if (err = ReadMacValue(bfpb, &fCDOGTempSalInfo.methodOfDeterminingTempSal)) return err;

	//if (version>1)
	//{
		if (err = ReadMacValue(bfpb, &fOutputSubsurfaceFiles)) return err;
		if (err = ReadMacValue(bfpb, &fOutputGasFiles)) return err;
	//}
	
	if (err = ReadMacValue(bfpb, &fLengthOfGridCellInMeters)) return err;
	if (err = ReadMacValue(bfpb, &fNumGridCells)) return err;


	if (err = ReadMacValue(bfpb, &bIsHydrogenSulfide)) return err;
	if (err = ReadMacValue(bfpb, &fMethodOfDeterminingHydrodynamics)) return err;

	if (err = ReadMacValue(bfpb, &fCDOGUserUnits.temperatureUnits)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGUserUnits.densityUnits)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGUserUnits.diameterUnits)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGUserUnits.dischargeType)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGUserUnits.gorUnits)) return err;
	if (err = ReadMacValue(bfpb, &fCDOGUserUnits.dischargeUnits)) return err;
	if (version>1)
		if (err = ReadMacValue(bfpb, &fCDOGUserUnits.molWtUnits)) return err;

	if (err = ReadMacValue(bfpb, &numDischarges)) return err;
	if (numDischarges>0)
	{
		fDischargeDataH = (DischargeDataH)_NewHandle(sizeof(DischargeData)*numDischarges);
		if(!fDischargeDataH) {TechError("CDOGLEList::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
		for (i=0;i<numDischarges;i++)
		{
			if (err = ReadMacValue(bfpb, &dischargeData.time)) return err;
			if (err = ReadMacValue(bfpb, &dischargeData.q_oil)) return err;
			if (err = ReadMacValue(bfpb, &dischargeData.q_gas)) return err;
			if (err = ReadMacValue(bfpb, &dischargeData.temp)) return err;
			if (err = ReadMacValue(bfpb, &dischargeData.diam)) return err;
			if (err = ReadMacValue(bfpb, &dischargeData.rho_oil)) return err;
			if (err = ReadMacValue(bfpb, &dischargeData.n_den)) return err;
			if (err = ReadMacValue(bfpb, &dischargeData.output_int)) return err;
			INDEXH(fDischargeDataH,i) = dischargeData;
		}
	}
		
	if (err = ReadMacValue(bfpb, &numProfiles)) return err;
	if (numProfiles>0)
	{
		fDepthValuesH = (DepthValuesSetH)_NewHandle(sizeof(DepthValuesSet)*numProfiles);
		if(!fDepthValuesH) {TechError("CDOGLEList::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
		for (i=0;i<numProfiles;i++)
		{
			if (err = ReadMacValue(bfpb, &depthProfile.depth)) return err;
			if (err = ReadMacValue(bfpb, &depthProfile.value.u)) return err;
			if (err = ReadMacValue(bfpb, &depthProfile.value.v)) return err;
			if (err = ReadMacValue(bfpb, &depthProfile.w)) return err;
			if (err = ReadMacValue(bfpb, &depthProfile.temp)) return err;
			if (err = ReadMacValue(bfpb, &depthProfile.sal)) return err;
			INDEXH(fDepthValuesH,i) = depthProfile;
		}
	}

done:
	if(err)
	{
		TechError("CDOGLEList::Read(char* path)", " ", 0); 
		if(fDischargeDataH) {DisposeHandle((Handle)fDischargeDataH); fDischargeDataH=0;}
		if(fDepthValuesH) {DisposeHandle((Handle)fDepthValuesH); fDepthValuesH=0;}
	}
	return err;
}

CDOGHydrodynamicInfo CDOGLEList::GetCDOGHydrodynamicInfo ()
{
	CDOGHydrodynamicInfo	info;

	memset(&info,0,sizeof(info));
	strcpy(info.hydrodynamicFilesFolderPath,this -> fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
	info.timeInterval = this -> fCDOGHydrodynamicInfo.timeInterval;
	info.period = this -> fCDOGHydrodynamicInfo.period;
	info.methodOfDeterminingCurrents = this -> fCDOGHydrodynamicInfo.methodOfDeterminingCurrents;

	return info;
}

void CDOGLEList::SetCDOGHydrodynamicInfo (CDOGHydrodynamicInfo info)
{
	strcpy(this -> fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath,info.hydrodynamicFilesFolderPath);	
	this -> fCDOGHydrodynamicInfo.timeInterval = info.timeInterval;
	this -> fCDOGHydrodynamicInfo.period = info.period;
	this -> fCDOGHydrodynamicInfo.methodOfDeterminingCurrents = info.methodOfDeterminingCurrents;

	return;
}

CDOGDiffusivityInfo CDOGLEList::GetCDOGDiffusivityInfo ()
{
	CDOGDiffusivityInfo	info;

	memset(&info,0,sizeof(info));
	info.horizDiff = this -> fCDOGDiffusivityInfo.horizDiff;
	info.vertDiff = this -> fCDOGDiffusivityInfo.vertDiff;
	info.timeStep = this -> fCDOGDiffusivityInfo.timeStep;

	return info;
}

void CDOGLEList::SetCDOGDiffusivityInfo (CDOGDiffusivityInfo info)
{
	this -> fCDOGDiffusivityInfo.horizDiff = info.horizDiff;
	this -> fCDOGDiffusivityInfo.vertDiff = info.vertDiff;
	this -> fCDOGDiffusivityInfo.timeStep = info.timeStep;

	return;
}

Boolean	CDOGLEList::TemperatureProfilePathSet()
{
	if (fCDOGTempSalInfo.temperatureFieldFilePath[0]==0)
	return false;
	else return true;
}

Boolean	CDOGLEList::SalinityProfilePathSet()
{
	if (fCDOGTempSalInfo.salinityFieldFilePath[0]==0)
	return false;
	else return true;
}

CDOGTempSalInfo CDOGLEList::GetCDOGTempSalInfo ()
{
	CDOGTempSalInfo info;

	memset(&info,0,sizeof(info));
	strcpy(info.temperatureFieldFilePath,this -> fCDOGTempSalInfo.temperatureFieldFilePath);
	strcpy(info.salinityFieldFilePath,this -> fCDOGTempSalInfo.salinityFieldFilePath);
	info.methodOfDeterminingTempSal = this -> fCDOGTempSalInfo.methodOfDeterminingTempSal;

	return info;
}

void CDOGLEList::SetCDOGTempSalInfo (CDOGTempSalInfo info)
{
	strcpy(fCDOGTempSalInfo.temperatureFieldFilePath,info.temperatureFieldFilePath);
	strcpy(fCDOGTempSalInfo.salinityFieldFilePath,info.salinityFieldFilePath);
	this -> fCDOGTempSalInfo.methodOfDeterminingTempSal = info.methodOfDeterminingTempSal;

	return;
}

CDOGParameters CDOGLEList::GetCDOGParameters ()
{
	CDOGParameters	params;

	memset(&params,0,sizeof(params));
	//params.orificeDiameter = this -> fCDOGParameters.orificeDiameter;
	//params.temp = this -> fCDOGParameters.temp;
	//params.density = this -> fCDOGParameters.density;
	params.equilibriumCurves = this -> fCDOGParameters.equilibriumCurves;
	params.bubbleRadius = this -> fCDOGParameters.bubbleRadius;
	params.molecularWt = this -> fCDOGParameters.molecularWt;
	params.hydrateDensity = this -> fCDOGParameters.hydrateDensity;
	params.separationFlag = this -> fCDOGParameters.separationFlag;
	params.hydrateProcess = this -> fCDOGParameters.hydrateProcess;
	params.dropSize = this -> fCDOGParameters.dropSize;
	//params.dischargeRateType = this -> fCDOGParameters.dischargeRateType;
	params.duration = this -> fCDOGParameters.duration;
	params.isContinuousRelease = this -> fCDOGParameters.isContinuousRelease;

	return params;
}

void CDOGLEList::SetCDOGParameters (CDOGParameters params)
{
	//this -> fCDOGParameters.orificeDiameter = params.orificeDiameter;
	//this -> fCDOGParameters.temp = params.temp;
	//this -> fCDOGParameters.density = params.density;
	this -> fCDOGParameters.equilibriumCurves = params.equilibriumCurves;
	this -> fCDOGParameters.bubbleRadius = params.bubbleRadius;
	this -> fCDOGParameters.molecularWt = params.molecularWt;
	this -> fCDOGParameters.hydrateDensity = params.hydrateDensity;
	this -> fCDOGParameters.separationFlag = params.separationFlag;
	this -> fCDOGParameters.hydrateProcess = params.hydrateProcess;
	//this -> fCDOGParameters.dischargeRateType = params.dischargeRateType;
	this -> fCDOGParameters.duration = params.duration;
	this -> fCDOGParameters.isContinuousRelease = params.isContinuousRelease;

	return;
}

CDOGUserUnits CDOGLEList::GetCDOGUserUnits ()
{
	CDOGUserUnits	units;

	memset(&units,0,sizeof(units));
	units.temperatureUnits = this -> fCDOGUserUnits.temperatureUnits;
	units.densityUnits = this -> fCDOGUserUnits.densityUnits;
	units.diameterUnits = this -> fCDOGUserUnits.diameterUnits;
	units.dischargeType = this -> fCDOGUserUnits.dischargeType;
	units.gorUnits = this -> fCDOGUserUnits.gorUnits;
	units.dischargeUnits = this -> fCDOGUserUnits.dischargeUnits;
	units.molWtUnits = this -> fCDOGUserUnits.molWtUnits;

	return units;
}

void CDOGLEList::SetCDOGUserUnits (CDOGUserUnits units)
{
	this -> fCDOGUserUnits.temperatureUnits = units.temperatureUnits;
	this -> fCDOGUserUnits.densityUnits = units.densityUnits;
	this -> fCDOGUserUnits.diameterUnits = units.diameterUnits;
	this -> fCDOGUserUnits.dischargeType = units.dischargeType;
	this -> fCDOGUserUnits.gorUnits = units.gorUnits;
	this -> fCDOGUserUnits.dischargeUnits = units.dischargeUnits;
	this -> fCDOGUserUnits.molWtUnits = units.molWtUnits;

	return;
}

/*CDOGSpillParameters CDOGLEList::GetCDOGSpillParameters ()
{
	CDOGSpillParameters	params;

	memset(&params,0,sizeof(params));
	params.oilDischargeRate = this -> fCDOGSpillParameters.oilDischargeRate;
	params.gasDischargeRate = this -> fCDOGSpillParameters.gasDischargeRate;
	//params.dischargeRateType = this -> fCDOGSpillParameters.dischargeRateType;

	return params;
}*/

/*void CDOGLEList::SetCDOGSpillParameters (CDOGSpillParameters params)
{
	this -> fCDOGSpillParameters.oilDischargeRate = params.oilDischargeRate;
	this -> fCDOGSpillParameters.gasDischargeRate = params.gasDischargeRate;
	//this -> fCDOGSpillParameters.dischargeRateType = params.dischargeRateType;

	return;
}*/
void CDOGLEList::SetDepthValuesHandle(DepthValuesSetH dvals)
{
	if(fDepthValuesH && dvals != fDepthValuesH)DisposeHandle((Handle)fDepthValuesH);
	fDepthValuesH=dvals;
}

void CDOGLEList::SetDischargeDataHandle(DischargeDataH dischargevals)
{
	if(fDischargeDataH && dischargevals != fDischargeDataH)DisposeHandle((Handle)fDischargeDataH);
	fDischargeDataH=dischargevals;
}


/////////////////////////////////////////////////
WorldPoint GetLowerLeftPointOfMap(void)
{// code goes here, check for multiple maps, maybe should allow user to input after all...
	long i,n;
	WorldRect mapBounds;
	WorldPoint wp = {0,0};
	TMap *map;
	n = model -> mapList->GetItemCount() ;
	for (i=0; i<n; i++)
	{
		model -> mapList->GetListItem((Ptr)&map, i);
		if (map->IAm(TYPE_MAP)) 
		{
			mapBounds = map->GetMapBounds();
			wp.pLong = mapBounds.loLong;
			wp.pLat = mapBounds.loLat;
			return wp;
		}
	}
	return wp;
}
/////////////////////////////////////////////////
WorldRect GetCDOGMapBounds(void)
{	// code goes here, check for multiple maps, maybe should allow user to input after all...
	long i,n;
	WorldRect mapBounds = emptyWorldRect;
	TMap *map;
	n = model -> mapList->GetItemCount() ;
	for (i=0; i<n; i++)
	{
		model -> mapList->GetListItem((Ptr)&map, i);
		if (map->IAm(TYPE_MAP)) 
		{
			mapBounds = map->GetMapBounds();
			return mapBounds;
		}
	}
	return mapBounds;
}



/*OSErr CDOGLEList::ExportCDOGParameters()	
{
	char path[256];
	char cdogFolderPathWithDelimiter[256];
	char cdogInputFolderPathWithDelimiter[256];
	char buffer[512],paramStr[128];
	OSErr err = 0;
	BFPB bfpb;
	WorldPoint mapRef = GetLowerLeftPointOfMap();

	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogInputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"input",DIRDELIMITER);

	strcpy(path,cdogInputFolderPathWithDelimiter);
	strcat(path,"Buojet.dat");

	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }

	// new CDOG version number
	strcpy(buffer,"CDOGG2.1\t (version number)");	// header line
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	// file format has changed, starts with a -1 flag for x,y in lat,lon, then lat,lon of the origin
	// followed by lat,lon of the spill position (5 lines in place of the first 2)
	// question of how to set the origin - CDOG requires x,y,z to be positive throughout
	// may want 0 to 360 degrees, but make sure to convert back when LEs are read in from CDOG
	// x coordinate of spill position [m]  - fSetSummary.startRelPos - converted from lat/lon
	// y coordinate of spill position [m]
	/////
	strcpy(buffer,"-1\t a flag to indicate the units of the coordinates (1 is km, -1 is lat/lon)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	// Next 2 lines are only included when lat/lon coordinates are used - reference point, lower left corner
	StringWithoutTrailingZeros(paramStr,mapRef.pLong/1e6,2);	// line 2
	strcpy(buffer,paramStr);
	strcat(buffer,"\tx coordinate of reference point [long]");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,mapRef.pLat/1e6,2);	// line 3
	strcpy(buffer,paramStr);
	strcat(buffer,"\ty coordinate of reference point [lat]");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fSetSummary.startRelPos.pLong/1e6,2);	// line 4
	strcpy(buffer,paramStr);
	strcat(buffer,"\tx coordinate of spill position [long or km]");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fSetSummary.startRelPos.pLat/1e6,2);	// line 5
	strcpy(buffer,paramStr);
	strcat(buffer,"\ty coordinate of spill position [lat or km]");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fSetSummary.z,2);	// line 6
	strcpy(buffer,paramStr);
	strcat(buffer,"\tdepth of spill position [m]");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fCDOGSpillParameters.oilDischargeRate,4);	// line 7
	strcpy(buffer,paramStr);
	strcat(buffer,"\toil discharge rate [m3/s]");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fCDOGSpillParameters.gasDischargeRate,4);	// line 8
	strcpy(buffer,paramStr);
	strcat(buffer,"\tgas discharge rate [Nm3/s]");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	// this line will have 2 extras added if natural gas is simulated
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.equilibriumCurves,0);	// line 9a,b,c
	strcpy(buffer,paramStr);
	strcat(buffer,"\t equilibrium curves");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////////////////////////////////////////////////
	if (fCDOGParameters.equilibriumCurves==2)
	{
		strcpy(buffer,".64995");
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		strcpy(buffer,"6.69021");
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	/////
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.orificeDiameter,4);	// line 10
	strcpy(buffer,paramStr);
	strcat(buffer,"\t diameter of the orifice [m]");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.temp,1);	// line 11
	strcpy(buffer,paramStr);
	strcat(buffer,"\t temperature of discharged mixture [degree C]");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.density,1);	// line 12
	strcpy(buffer,paramStr);
	strcat(buffer,"\tdensity of oil at an average ambient water temperature [kg/m3]");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	// line 13  - a flag to indicate the type of released liquid, this will always be 1 for oil
	strcpy(buffer,"1\t a flag to indicate the type of released liquid (1 is oil, 0 is water)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.bubbleRadius,4); // line 14
	strcpy(buffer,paramStr);
	strcat(buffer,"\tinitial gas bubble radius [m]");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.molecularWt,4);	// line 15
	strcpy(buffer,paramStr);
	strcat(buffer,"\tmolecular weight of gas");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.hydrateDensity,1);	// line 16
	strcpy(buffer,paramStr);
	strcat(buffer,"\tdensity of gas hydrate");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	// not sure what to use here, line 17 
	strcpy(buffer,"100\t number of output files within a period = int[prd_tid/dt_cur_inp]+1");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.separationFlag,0);	// line 18
	strcpy(buffer,paramStr);
	strcat(buffer,"\tflag for separation of gas from bent plume");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	// not sure what to use here, line 19
	strcpy(buffer,"100\t output frequency with respect to time step (in the plume phase)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 

done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		// the user has already been told there was a problem
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}*/
/////////////////////////////////////////////////
long CDOGLEList::GetNumProfiles()
{
	long numProfiles = 0;
	if (fDepthValuesH) 
	{
		numProfiles = _GetHandleSize((Handle)fDepthValuesH)/sizeof(**fDepthValuesH);
	}
	return numProfiles;
}

long CDOGLEList::GetNumDischarges()
{
	long numDischarges = 0;
	if (fDischargeDataH) 
	{
		numDischarges = _GetHandleSize((Handle)fDischargeDataH)/sizeof(**fDischargeDataH);
	}
	return numDischarges;
}

Seconds	CDOGLEList::GetSpillDuration()
{
	Seconds duration = 0;
	//return fSetSummary.endRelTime - fSetSummary.startRelTime;
	long numDischarges = GetNumDischarges();

	if (numDischarges == 0) return 0;
	if (numDischarges == 1)
	{
		if (!fCDOGParameters.isContinuousRelease) return fCDOGParameters.duration;
		else duration = model->GetDuration()/3600.;
	}
	if (numDischarges>1)
	{
		double dischargeTime = INDEXH(fDischargeDataH,numDischarges-1).time;
		duration = 3600*dischargeTime;
	}

	return duration;
}

OSErr CDOGLEList::ExportCDOGGeninf()	
{
	char path[256];
	char cdogFolderPathWithDelimiter[256];
	char cdogInputFolderPathWithDelimiter[256];
	char buffer[512],paramStr[128];
	OSErr err = 0;
	BFPB bfpb;
	WorldPoint mapRef = GetLowerLeftPointOfMap();
	long i, val = 1, numDischarges=0/*, n_den, output_int*/;
	double q_oil, q_gas, rho_oil, temp, diam, dischargeTime, gor, dischargeRate;
	Seconds simulationDuration = model->GetDuration(), duration=0;
	Seconds spillDuration = GetSpillDuration();	// no longer used since now have variable discharge option - should dialog lose the end rel time?
	short dischargeUnits, gorUnits;

	if (spillDuration==0) {printError("There is no discharge rate set. CDOG cannot be run."); return -1;}
	
	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogInputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"input",DIRDELIMITER);

	strcpy(path,cdogInputFolderPathWithDelimiter);
	strcat(path,"Geninf.dat");

	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }

	strcpy(buffer,"CDOGG2.1\t (version number)");	// header line
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	StringWithoutTrailingZeros(paramStr,fOutputSubsurfaceFiles,0);	// line 2
	strcpy(buffer,paramStr);
	strcat(buffer,"\t (a flag to indicate whether to output subsurface data : 1 is yes, 0 is no)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	StringWithoutTrailingZeros(paramStr,fOutputGasFiles,0);	// line 3
	strcpy(buffer,paramStr);
	strcat(buffer,"\t (a flag to indicate whether to output gas particles : 1 is yes, 0 is no)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	//StringWithoutTrailingZeros(paramStr,fOutputGasFiles,0);	// line 4
	//strcpy(buffer,paramStr);
	//strcat(buffer,"\t a flag to indicate data format (>=0 is native CDOG, < 0 is netCDF)");
	strcpy(buffer,"1\t (a flag to indicate data format : >=0 is native CDOG, < 0 is netCDF)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	//StringWithoutTrailingZeros(paramStr,fOutputGasFiles,0);	// line 5
	//strcpy(buffer,paramStr);
	//strcat(buffer,"\t a flag to indicate whether constant ambient condition and horizontal and flat ocean bottom (>= 0 is yes, < 0 is no)");
	strcpy(buffer,"1\t (a flag to indicate whether constant ambient condition and horizontal and flat ocean bottom : >= 0 is yes, < 0 is no)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.hydrateProcess,0);	// line 6
	strcpy(buffer,paramStr);
	strcat(buffer,"\t (hydrate process 0 suppress, 1 do not suppress)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.separationFlag,0);	// line 7
	strcpy(buffer,paramStr);
	strcat(buffer,"\t (flag for separation of gas from bent plume)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	// file format has changed, starts with a -1 flag for x,y in lat,lon, then lat,lon of the origin
	// followed by lat,lon of the spill position (5 lines in place of the first 2)
	// question of how to set the origin - CDOG requires x,y,z to be positive throughout
	// may want 0 to 360 degrees, but make sure to convert back when LEs are read in from CDOG
	// x coordinate of spill position [m]  - fSetSummary.startRelPos - converted from lat/lon
	// y coordinate of spill position [m]
	/////
	strcpy(buffer,"-1\t (a flag to indicate the units of the coordinates : 1 is km, -1 is lat/lon)");	// line 8
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	strcpy(buffer,"1\t (index_drop, >=0 let model to calculate dropsize, <0,use user inputs)");	// line 9
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	// CDOG ignores this line if previous line is >=0, but not sure which to choose if line 9 is < 0
	strcpy(buffer,"1\t (>=0 input size (diameter) distribution, <0 input volume distribution)");	// line 10
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,simulationDuration/3600,1);	// line 11
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(Simulation duration [hr])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fCDOGHydrodynamicInfo.timeInterval,2);	//  line 12
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(Time interval to input ambient data [hr])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fCDOGHydrodynamicInfo.period,2);	// line 13 
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(Period of ambient data variation [hr])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	StringWithoutTrailingZeros(paramStr,fSetSummary.numOfLEs,0);	// line 14
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(Total number of released oil parcels - max 50000)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	// Next 2 lines are only included when lat/lon coordinates are used - reference point, lower left corner
	StringWithoutTrailingZeros(paramStr,mapRef.pLong/1e6,2);	// line 15
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(x coordinate of reference point [long])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,mapRef.pLat/1e6,2);	// line 16
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(y coordinate of reference point [lat])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fSetSummary.startRelPos.pLong/1e6,2);	// line 16a
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(x coordinate of spill position [long or km])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fSetSummary.startRelPos.pLat/1e6,2);	// line 16b
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(y coordinate of spill position [lat or km])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fSetSummary.z*depthCF[fDepthUnits-1],2);	// line 17
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(depth of spill position [m])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	// this line will have 2 extras added if natural gas is simulated
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.equilibriumCurves,0);	// line 18a,b,c
	strcpy(buffer,paramStr);
	strcat(buffer,"\t (equilibrium curves)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////////////////////////////////////////////////
	if (fCDOGParameters.equilibriumCurves==2)
	{
		strcpy(buffer,".64995");
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		strcpy(buffer,"6.69021");
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	///// 
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.bubbleRadius,4); // line 19
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(initial gas bubble radius [m])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	if (fCDOGUserUnits.molWtUnits==2)	// kg/mol
		StringWithoutTrailingZeros(paramStr,fCDOGParameters.molecularWt,4);	// line 20
	else	// g/mol
		StringWithoutTrailingZeros(paramStr,fCDOGParameters.molecularWt/1000.,4);	// line 20
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(molecular weight of gas [kg/mol])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fCDOGParameters.hydrateDensity,1);	// line 21
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(density of gas hydrate [kg/m^3])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	// not sure what to use here, line 22 
	strcpy(buffer,"100\t (number of output files within a period = int[prd_tid/dt_cur_inp]+1)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,fCDOGDiffusivityInfo.timeStep/60,1);	// line 23
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(Time step of advection/turbulent diffusion simulation [min])");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	StringWithoutTrailingZeros(paramStr,val,0);	// line 24
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(Output skip with respect to time step)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	/////
	if (fDischargeDataH) 
	{
		numDischarges = _GetHandleSize((Handle)fDischargeDataH)/sizeof(**fDischargeDataH);
		if (numDischarges == 1)
		{
			if (!fCDOGParameters.isContinuousRelease) duration = fCDOGParameters.duration;
			else duration = model->GetDuration()/3600.;
		}
	}
	if (numDischarges==1 && duration > 0)
		StringWithoutTrailingZeros(paramStr,numDischarges+1,0);	// line 25
	else
		StringWithoutTrailingZeros(paramStr,numDischarges,0);	// line 25
	strcpy(buffer,paramStr);
	strcat(buffer,"\t(Number of points on the discharge-time curve)");	// 0 should be error
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	if (numDischarges==0) {printError("No discharge rate has been set. CDOG cannot be run."); return -1;}
	
	dischargeUnits = fCDOGUserUnits.dischargeUnits;
	gorUnits = fCDOGUserUnits.gorUnits;
	for (i=0; i<numDischarges; i++) 
	{
		dischargeRate = INDEXH(fDischargeDataH,i).q_oil;
		gor = INDEXH(fDischargeDataH,i).q_gas;
		temp = INDEXH(fDischargeDataH,i).temp;
		diam = INDEXH(fDischargeDataH,i).diam;
		rho_oil = INDEXH(fDischargeDataH,i).rho_oil;
		dischargeTime = INDEXH(fDischargeDataH,i).time;
		if (fCDOGUserUnits.temperatureUnits==kDegreesF) temp = (temp - 32) * 5. / 9.;
		if (fCDOGUserUnits.densityUnits==kAPI) rho_oil = 141.5/(rho_oil + 131.5) * 1000.;
		if (fCDOGUserUnits.diameterUnits==kCentimeters) diam = diam / 100;
		else if (fCDOGUserUnits.diameterUnits==kInches) diam = diam / 2.54;
		if (fCDOGUserUnits.dischargeType==2) 
		{
			q_gas = dischargeRate; 
			if (dischargeUnits==2) {q_gas = q_gas*(28.32/(3600*24));} 
			if (gorUnits==2) {gor = gor*.02832/.158987;} 	// factor of 5.62 roughly
			//if (gorUnits==3) {gor = gor*(28.32)/(.158987/(3600*24))} // check this, 
			if (gorUnits==3) {gor = gor*28.32/.158987;} // check this, 
			q_oil = q_gas / gor;
		}
		else 
		{
			q_oil = dischargeRate; 
			if (dischargeUnits==2) {q_oil = q_oil*(.158987/(3600*24));} 
			if (gorUnits==2) {gor = gor*.02832/.158987;} 	// factor of 5.62 roughly
			//if (gorUnits==3) {gor = gor*(28.32)/(.158987/(3600*24))} // check this, 
			if (gorUnits==3) {gor = gor*28.32/.158987;} // check this, 
			q_gas = gor * q_oil;
		}
		//n_den = 1;
		//output_int = 1000;
		// Part of the new variable discharge rate time varying input data 
		StringWithoutTrailingZeros(paramStr,dischargeTime,2);	// line 26+
		strcpy(buffer,paramStr);
		strcat(buffer,"\t");
		StringWithoutTrailingZeros(paramStr,q_oil,4);	// line 26+
		strcat(buffer,paramStr);
		//strcat(buffer,"\toil discharge rate [m3/s]");
		//strcat(buffer,NEWLINESTRING);
		//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		/////
		StringWithoutTrailingZeros(paramStr,q_gas,4);	// line 26+
		//strcpy(buffer,paramStr);
		strcat(buffer,"\t");
		strcat(buffer,paramStr);
		//strcat(buffer,"\tgas discharge rate [m3/s]");
		//strcat(buffer,NEWLINESTRING);
		//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		/////
		// Part of the new variable discharge rate time varying input data 
		StringWithoutTrailingZeros(paramStr,temp,1);	// line 26+
		strcat(buffer,"\t");
		strcat(buffer,paramStr);
		//strcat(buffer,"\t temperature of discharged mixture [degree C]");
		//strcat(buffer,NEWLINESTRING);
		//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		/////
		// Part of the new variable discharge rate time varying input data 
		StringWithoutTrailingZeros(paramStr,diam,4);	// line 26+
		strcat(buffer,"\t");
		strcat(buffer,paramStr);
		//strcat(buffer,"\t diameter of the orifice [m]");
		//strcat(buffer,NEWLINESTRING);
		//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		/////
		// Part of the new variable discharge rate time varying input data 
		StringWithoutTrailingZeros(paramStr,rho_oil,1);	// line 26+
		strcat(buffer,"\t");
		strcat(buffer,paramStr);
		//strcat(buffer,"\tdensity of oil at an average ambient water temperature [kg/m3]");
		//strcat(buffer,NEWLINESTRING);
		//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		/////
		// Part of the new variable discharge rate time varying input data 
		// line 13  - a flag to indicate the type of released liquid, this will always be 1 for oil
		strcat(buffer,"\t1");
		//strcpy(buffer,"1\t a flag to indicate the type of released liquid (1 is oil, 0 is water)");
		//strcat(buffer,NEWLINESTRING);
		//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		/////
		// Part of the new variable discharge rate time varying input data 
		// not sure what to use here, line 19
		strcat(buffer,"\t100\t ()");
		//strcpy(buffer,"100\t (output frequency with respect to time step in the plume phase)");
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		///// 
	}
	if (numDischarges==1 && duration > 0)
	{	// constant discharge requires an end time with all the same discharge info
		dischargeTime += duration;
		StringWithoutTrailingZeros(paramStr,dischargeTime,2);	// line 26+
		strcpy(buffer,paramStr);
		strcat(buffer,"\t");
		StringWithoutTrailingZeros(paramStr,q_oil,4);	// line 26+
		strcat(buffer,paramStr);
		StringWithoutTrailingZeros(paramStr,q_gas,4);	// line 26+
		strcat(buffer,"\t");
		strcat(buffer,paramStr);
		StringWithoutTrailingZeros(paramStr,temp,1);	// line 26+
		strcat(buffer,"\t");
		strcat(buffer,paramStr);
		StringWithoutTrailingZeros(paramStr,diam,4);	// line 26+
		strcat(buffer,"\t");
		strcat(buffer,paramStr);
		StringWithoutTrailingZeros(paramStr,rho_oil,1);	// line 26+
		strcat(buffer,"\t");
		strcat(buffer,paramStr);
		strcat(buffer,"\t1");
		strcat(buffer,"\t100\t ()");
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
		strcpy(buffer,NEWLINESTRING);
		strcat(buffer,"Time of discharge [hrs]");
		strcat(buffer,NEWLINESTRING);
		strcat(buffer,"oil discharge rate [m3/s]");
		strcat(buffer,NEWLINESTRING);
		strcat(buffer,"gas discharge rate [Nm3/s]");
		strcat(buffer,NEWLINESTRING);
		strcat(buffer,"temperature of discharged mixture [degree C]");
		strcat(buffer,NEWLINESTRING);
		strcat(buffer,"diameter of the orifice [m]");
		strcat(buffer,NEWLINESTRING);
		strcat(buffer,"density of oil at an average ambient water temperature [kg/m3]");
		strcat(buffer,NEWLINESTRING);
		strcat(buffer,"flag to indicate the type of released liquid (1 is oil, 0 is water)");
		strcat(buffer,NEWLINESTRING);
		strcat(buffer,"output frequency with respect to time step in the plume phase");
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		// the user has already been told there was a problem
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}
/////////////////////////////////////////////////
OSErr CDOGLEList::ExportCDOGZeroWind()	
{
	char path[256];
	char cdogFolderPathWithDelimiter[256];
	char cdogInputFolderPathWithDelimiter[256];
	char buffer[512]/*,paramStr[128]*/;
	OSErr err = 0;
	BFPB bfpb;

	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogInputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"input",DIRDELIMITER);

	strcpy(path,cdogInputFolderPathWithDelimiter);
	strcat(path,"Wind.dat");

	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }

	strcpy(buffer,"CDOGG2.1\t (version number)");	// header line
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	///// 
	//StringWithoutTrailingZeros(paramStr,fOutputSubsurfaceFiles,0);	// line 2
	//strcpy(buffer,paramStr);
	strcpy(buffer,"NOAA uses zero winds");	// key word NOAA to let CDOG know the file won't have any wind values
	//strcat(buffer,"\t (a flag to indicate whether to output subsurface data : 1 is yes, 0 is no)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	// code goes here, might want to add the wind.dat file structure for zero wind
	///// 
done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		// the user has already been told there was a problem
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}
/////////////////////////////////////////////////
OSErr CDOGLEList::ExportProfilesAsNetCDF(TCurrentMover *hydrodynamicData)
{
	OSErr err = 0;	
	int status, ncid, lat_dim, lon_dim, time_dim, depth_dim, u_id, v_id, w_id, h_id; 
	int temp_id, sal_id, depth_id, hyd_dimids[4], dx_id, difh_id, difv_id/*, old_fill_mode*/;
	int lat_id, lon_id, time_id,/* reftime_id, timelen_dim,*/ dimid[1], hdimid[2];
	float fillVal = 1e-34 ,dxVal;
	//double rh_range[] = {0.0, 100.0};
	char path[256];
	char title[] = "Gnome netCDF dataset for CDOG";
	static size_t time_index[1],diff_index[1],h_index[2];
	float lat_vals[2];
	float lon_vals[2];
	//static char reftime[] = {"1992 03 04 12:00"};

	DepthValuesSet profileSet;
	long i,j,k,numProfileSets = 0,numValues,numFiles=1,timeIndex=0,numTimes=1,sigmaLength=0;	// numFiles - for now not allowing time dependence
	char cdogFolderPathWithDelimiter[256], cdogInputFolderPathWithDelimiter[256];
	float *u_vals,*v_vals,*w_vals,*temp_vals,*sal_vals=0,*depth_vals=0,diff_val/*,*time_vals=0*/;
	WorldRect mapBounds;
	double degreeLat, degreeLon, totDepth, timeInHrs;
	DepthValuesSetH profilesH = 0;
	Seconds timeValInSecs;

	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogInputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"input",DIRDELIMITER);
	strcpy(path,cdogInputFolderPathWithDelimiter);
	strcat(path,"ambient.nc");
	//if (fDepthValuesH) numProfileSets = _GetHandleSize((Handle)fDepthValuesH)/sizeof(**fDepthValuesH);
	// might want to store in the handle
	numValues = (fNumGridCells+1)*(fNumGridCells+1);

	mapBounds = GetCDOGMapBounds();
	degreeLat = (mapBounds.hiLat - mapBounds.loLat) / 1000000.;
	degreeLon = (mapBounds.hiLong - mapBounds.loLong) / 1000000.;
	fLengthOfGridCellInMeters = (degreeLat * METERSPERDEGREELAT) / fNumGridCells;	// always one giant grid cell at this point
	
	lat_vals[0] = mapBounds.loLat/1000000.;
	lat_vals[1] = mapBounds.hiLat/1000000.;
	lon_vals[0] = mapBounds.loLong/1000000.;
	lon_vals[1] = mapBounds.hiLong/1000000.;
	// decide which is larger? or which is smaller?
	//fSetSummary.startRelPos.pLat/1e6
	//fSetSummary.endRelPos.pLat/1e6
	//if (possible3DMover->IAm(TYPE_NETCDFMOVER)) ((NetCDFMover*)possible3DMover)->GetNumDepthLevels();
	// also need to get all the velocities and temp, sal, use DepthValuesSetH
	
	if (hydrodynamicData->IAm(TYPE_NETCDFMOVER)) numTimes = /*OK*/(dynamic_cast<NetCDFMover *>(hydrodynamicData))->GetNumTimesInFile();
	else return -1;
	if (numTimes < 1) return -1;

	sigmaLength = (dynamic_cast<NetCDFMover *>(hydrodynamicData))->GetNumDepthLevels();
	timeValInSecs = /*OK*/ (dynamic_cast<NetCDFMover *>(hydrodynamicData))->GetTimeValue(numTimes-1);
	if (numTimes>1 && timeValInSecs < model->GetEndTime()) {printNote("The hydrodynamic data runs out before the model end time. This may affect CDOG run.");}

	// for some reason the nc_create command gives an error, have to use nccreate with no error checking
	ncid = nccreate(path,NC_CLOBBER);
	//ncid = nc_create(path,NC_CLOBBER);
	//if (status != NC_NOERR) {err = -1; goto done;}
	numFiles = numTimes;
	// right now the grid file is getting written out numTimes (overwriting the same file each time). Should cut out the extra stuff
	// reuse the same array for u,v,temp,sal - check for w or set it to zero
	u_vals = new float[sigmaLength*numValues*numTimes];
	if (!u_vals) {err = memFullErr; goto done;}
	v_vals = new float[sigmaLength*numValues*numTimes];
	if (!v_vals) {err = memFullErr; goto done;}
	w_vals = new float[sigmaLength*numValues*numTimes];
	if (!w_vals) {err = memFullErr; goto done;}
	temp_vals = new float[sigmaLength*numValues*numTimes];
	if (!temp_vals) {err = memFullErr; goto done;}
	sal_vals = new float[sigmaLength*numValues*numTimes];
	if (!sal_vals) {err = memFullErr; goto done;}
	depth_vals = new float[sigmaLength*numValues];
	if (!depth_vals) {err = memFullErr; goto done;}
	//time_vals = new float[numTimes];
	//if (!time_vals) {err = memFullErr; goto done;}

	//status = nc_def_dim(ncid, "lat", 5L, &lat_dim);
	status = nc_def_dim(ncid, "LAT", 2L, &lat_dim);	//in this case do we have to repeat all values?
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_def_dim(ncid, "lon", 10L, &lon_dim);
	status = nc_def_dim(ncid, "LON", 2L, &lon_dim);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_def_dim(ncid, "DEPTH", numProfileSets, &depth_dim);
	status = nc_def_dim(ncid, "DEPTH", sigmaLength, &depth_dim);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_def_dim(ncid, "TIME", NC_UNLIMITED, &time_dim);
	if (status != NC_NOERR) {err = -1; goto done;}
    //status = nc_def_dim(ncid, "timelen", 20L, &timelen_dim);
	//if (status != NC_NOERR) {err = -1; goto done;}

	hyd_dimids[0] = time_dim;
	hyd_dimids[1] = depth_dim;
	hyd_dimids[2] = lat_dim;
	hyd_dimids[3] = lon_dim;
	
	status = nc_def_var(ncid, "U", NC_FLOAT, 4, hyd_dimids, &u_id);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_def_var(ncid, "V", NC_FLOAT, 4, hyd_dimids, &v_id);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_def_var(ncid, "W", NC_FLOAT, 4, hyd_dimids, &w_id);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_def_var(ncid, "TEMP", NC_FLOAT, 4, hyd_dimids, &temp_id);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_def_var(ncid, "SAL", NC_FLOAT, 4, hyd_dimids, &sal_id);
	if (status != NC_NOERR) {err = -1; goto done;}

 	dimid[0] = depth_dim;
	status = nc_def_var(ncid, "DEPTH", NC_FLOAT, 1, dimid, &depth_id);
	if (status != NC_NOERR) {err = -1; goto done;}

 	dimid[0] = depth_dim;
	status = nc_def_var(ncid, "DIFH", NC_FLOAT, 1, dimid, &difh_id);
	if (status != NC_NOERR) {err = -1; goto done;}
 	dimid[0] = depth_dim;
	status = nc_def_var(ncid, "DIFV", NC_FLOAT, 1, dimid, &difv_id);
	if (status != NC_NOERR) {err = -1; goto done;}

 	hdimid[0] = lat_dim;
	hdimid[1] = lon_dim;
	
	status = nc_def_var(ncid, "H", NC_DOUBLE, 2, hdimid, &h_id);
	if (status != NC_NOERR) {err = -1; goto done;}

 	dimid[0] = 1;
	//status = nc_def_var(ncid, "DX", NC_LONG, 0, dimid, &dx_id);
	status = nc_def_var(ncid, "DX", NC_FLOAT, 0, dimid, &dx_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_set_fill(ncid, NC_NOFILL, &old_fill_mode);
	//if (status != NC_NOERR) {err = -1; goto done;}

	//status = nc_put_att_double (ncid, u_id, "valid_range", NC_DOUBLE, 2, rh_range);
	//if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_put_att_float (ncid, u_id, "_FillValue", NC_FLOAT, 1, &fillVal);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_float (ncid, v_id, "_FillValue", NC_FLOAT, 1, &fillVal);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_float (ncid, w_id, "_FillValue", NC_FLOAT, 1, &fillVal);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_float (ncid, temp_id, "_FillValue", NC_FLOAT, 1, &fillVal);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_float (ncid, sal_id, "_FillValue", NC_FLOAT, 1, &fillVal);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_float (ncid, depth_id, "_FillValue", NC_FLOAT, 1, &fillVal);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_put_att_text (ncid, u_id, "long_name",/* NC_CHAR,*/ strlen("Eastward Water Velocity"), "Eastward Water Velocity");
	status = nc_put_att_text (ncid, v_id, "long_name",/* NC_CHAR,*/ strlen("Northward Water Velocity"), "Northward Water Velocity");
	status = nc_put_att_text (ncid, w_id, "long_name",/* NC_CHAR,*/ strlen("Vertical Water Velocity"), "Vertical Water Velocity");
	status = nc_put_att_text (ncid, temp_id, "long_name",/* NC_CHAR,*/ strlen("Temperature"), "Temperature");
	status = nc_put_att_text (ncid, sal_id, "long_name",/* NC_CHAR,*/ strlen("Salinity"), "Salinity");
	status = nc_put_att_text (ncid, depth_id, "long_name",/* NC_CHAR,*/ strlen("Depth"), "Depth");
	//status = nc_put_att_text (ncid, dx_id, "long_name",/* NC_CHAR,*/ strlen("Number of Grid Cells"), "Number of Grid Cells");
	status = nc_put_att_text (ncid, dx_id, "long_name",/* NC_CHAR,*/ strlen("Grid Cell Length"), "Grid Cell Length");

	status = nc_put_att_text (ncid, u_id, "units",/* NC_CHAR,*/ strlen("m/s"), "m/s");
	status = nc_put_att_text (ncid, v_id, "units",/* NC_CHAR,*/ strlen("m/s"), "m/s");
	status = nc_put_att_text (ncid, w_id, "units",/* NC_CHAR,*/ strlen("m/s"), "m/s");
	status = nc_put_att_text (ncid, temp_id, "units",/* NC_CHAR,*/ strlen("Celsius"), "Celsius");
	status = nc_put_att_text (ncid, sal_id, "units",/* NC_CHAR,*/ strlen("ppt"), "ppt");
	status = nc_put_att_text (ncid, depth_id, "units",/* NC_CHAR,*/ strlen("meters"), "meters");
		//lat:units = "degrees_north" ;
	dimid[0] = lat_dim;
   status = nc_def_var (ncid, "LAT", NC_DOUBLE, 1, dimid, &lat_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, lat_id, "long_name",/* NC_CHAR,*/ strlen("latitude"), "latitude");
	status = nc_put_att_text (ncid, lat_id, "units",/* NC_CHAR,*/ strlen("degrees_north"), "degrees_north");
	dimid[0] = lon_dim;
   status = nc_def_var (ncid, "LON", NC_DOUBLE, 1, dimid, &lon_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, lon_id, "long_name",/* NC_CHAR,*/ strlen("longitude"), "longitude");
	status = nc_put_att_text (ncid, lon_id, "units",/* NC_CHAR,*/ strlen("degrees_east"), "degrees_east");
   dimid[0] = time_dim;
   status = nc_def_var (ncid, "TIME", NC_DOUBLE, 1, dimid, &time_id);
 	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, time_id, "long_name",/* NC_CHAR,*/ strlen("Time since model start"), "Time since model start");
	status = nc_put_att_text (ncid, time_id, "units",/* NC_CHAR,*/ strlen("hours"), "hours");
  // dimid[0] = timelen_dim;
  // status = nc_def_var (ncid, "reftime", NC_CHAR, 1, dimid, &reftime_id);
 	//if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_put_att_text (ncid, reftime_id, "long_name",/* NC_CHAR,*/ strlen("reference time"), "reference time");
	//if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_put_att_text (ncid, reftime_id, "units",/* NC_CHAR,*/ strlen("text_time"), "text_time");
	//if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_put_att_text (ncid, NC_GLOBAL, "title",/* NC_CHAR,*/ strlen(title), title);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_enddef(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	for (k=0;k<numTimes;k++)
	{
		timeIndex = k;
		if (hydrodynamicData->IAm(TYPE_NETCDFMOVER)) err = (dynamic_cast<NetCDFMover *>(hydrodynamicData))->GetDepthProfileAtPoint(fSetSummary.startRelPos,timeIndex,&profilesH);
		if (!err && profilesH) numProfileSets = _GetHandleSize((Handle)profilesH)/sizeof(**profilesH);
		else return -1;
		// code goes here, add an h.dat file  numValues * largest depth (or larger?)

		if (k==0)
		{	// should be the same every time
			totDepth = INDEXH(profilesH,numProfileSets-1).depth;
			if (totDepth < fSetSummary.z*depthCF[fDepthUnits-1]) {printError("The spill depth must be less than the total depth");err=-1;goto done;}	// or add a bottom set of values (same as above or zero?)
		}

		for (i=0; i<numProfileSets; i++)	// have 2x2 grid of the same values, could instead  get values at each of the map corners?
		{
			profileSet = INDEXH(profilesH,i);
			for (j=0; j<numValues;j++)
			{
				if (k==0) 
				{
					depth_vals[j*numProfileSets+i] = profileSet.depth;
					if (j==0)
					{
						diff_index[0]=i;
						diff_val = fCDOGDiffusivityInfo.horizDiff;
						nc_put_var1_float(ncid,difh_id,diff_index,&diff_val);
						diff_val = fCDOGDiffusivityInfo.vertDiff;
						nc_put_var1_float(ncid,difv_id,diff_index,&diff_val);
					}
					if (i==0)
					{
						if (j<2)
						{
							h_index[0]=0;
							h_index[1]=j;
						}
						else
						{
							h_index[0]=1;
							h_index[1]=j-2;
						}
						nc_put_var1_double(ncid,h_id,h_index,&totDepth);						
					}
				}
				u_vals[i*numValues+j + k*numProfileSets*numValues] = profileSet.value.u;
				v_vals[i*numValues+j+ k*numProfileSets*numValues] = profileSet.value.v;
				w_vals[i*numValues+j+ k*numProfileSets*numValues] = 0.;
				temp_vals[i*numValues+j+ k*numProfileSets*numValues] = profileSet.temp;
				sal_vals[i*numValues+j+ k*numProfileSets*numValues] = profileSet.sal;
			}
		}
		time_index[0] = timeIndex;
		timeValInSecs = /*OK*/ (dynamic_cast<NetCDFMover *>(hydrodynamicData))->GetTimeValue(timeIndex);
		timeInHrs = (timeValInSecs - model->GetStartTime()) / 3600.;
		nc_put_var1_double(ncid,time_id,time_index,&timeInHrs);
		
		if (profilesH)  {DisposeHandle((Handle)profilesH); profilesH=0;}
	}

	status = nc_put_var_float(ncid, u_id, u_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_var_float(ncid, v_id, v_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_var_float(ncid, w_id, w_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_var_float(ncid, temp_id, temp_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_var_float(ncid, sal_id, sal_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_var_float(ncid, depth_id, depth_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	// store lat 
	status = nc_put_var_float(ncid, lat_id, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}

	// store lon 
	status = nc_put_var_float(ncid, lon_id, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}

	//status = nc_put_var_long(ncid, dx_id, &fNumGridCells);
	// are we exporting cell size or number of cells?
	dxVal = fLengthOfGridCellInMeters/1000.;
	status = nc_put_var_float(ncid, dx_id, &dxVal);
	if (status != NC_NOERR) {err = -1; goto done;}

	// store reftime 
	//status = nc_put_var_text(ncid, reftime_id, reftime);
	//if (status != NC_NOERR) {err = -1; goto done;}

	// store rh_vals
	//status = nc_put_var_double(ncid, rh_id, rh_vals);
	//if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

done:
	if (err)
	{
		printError("Error writing out netcdf file");
	}
	if (u_vals) delete [] u_vals;
	if (v_vals) delete [] v_vals;
	if (w_vals) delete [] w_vals;
	if (temp_vals) delete [] temp_vals;
	if (sal_vals) delete [] sal_vals;
	if (depth_vals) delete [] depth_vals;
	//if (time_vals) delete [] time_vals;
	if (profilesH)  {DisposeHandle((Handle)profilesH); profilesH=0;}
	return err;
}

/////////////////////////////////////////////////

void GetFileName(long index, char *fileName)
{
	switch(index)
	{
		case 1:
			strcpy(fileName,"Grid");
			break;
		case 2:
			strcpy(fileName,"U");
			break;
		case 3:
			strcpy(fileName,"V");
			break;
		case 4:
			strcpy(fileName,"W");
			break;
		case 5:
			strcpy(fileName,"Temp");
			break;
		case 6:
			strcpy(fileName,"Sal");
			break;
		case 7:
			strcpy(fileName,"H");
			break;
		default:
			strcpy(fileName,"");
			break;
	}
	return;
}
/////////////////////////////////////////////////
OSErr CDOGLEList::ExportProfilesToCDOGInputFolder(TCurrentMover *hydrodynamicData)
{
	BFPB bfpb;
	OSErr err = 0;
	DepthValuesSet profileSet;
	long i,j,k,numProfileSets = 0,numValues,numFiles=1,timeIndex=0,numTimes=1;	// numFiles - for now not allowing time dependence
	char cdogFolderPathWithDelimiter[256], cdogInputFolderPathWithDelimiter[256];
	char fileName[32], outputStr[32], headerStr[32], numCellStr[32];
	float dataToExport[6]/*,previousDepth=0*/;
	char cdogFilePathForInput[256], buffer[512];
	char cdogUFilePathForInput[256], cdogVFilePathForInput[256], cdogWFilePathForInput[256], cdogTempFilePathForInput[256], cdogSalFilePathForInput[256];
	char uFileName[256], vFileName[256], wFileName[256], tempFileName[256], salFileName[256], fileNum[32];
	//char uFilePath[256], vFilePath[256], wFilePath[256];
	WorldRect mapBounds;
	double degreeLat, degreeLon, totDepth, timeInHrs;
	DepthValuesSetH profilesH = 0;
	Seconds timeValInSecs;

	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogInputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"input",DIRDELIMITER);
	//if (fDepthValuesH) numProfileSets = _GetHandleSize((Handle)fDepthValuesH)/sizeof(**fDepthValuesH);
	// might want to store in the handle
	numValues = (fNumGridCells+1)*(fNumGridCells+1);

	mapBounds = GetCDOGMapBounds();
	degreeLat = (mapBounds.hiLat - mapBounds.loLat) / 1000000.;
	degreeLon = (mapBounds.hiLong - mapBounds.loLong) / 1000000.;
	fLengthOfGridCellInMeters = (degreeLat * METERSPERDEGREELAT) / fNumGridCells;	// always one giant grid cell at this point
	// decide which is larger? or which is smaller?
	//fSetSummary.startRelPos.pLat/1e6
	//fSetSummary.endRelPos.pLat/1e6
	//if (possible3DMover->IAm(TYPE_NETCDFMOVER)) ((NetCDFMover*)possible3DMover)->GetNumDepthLevels();
	// also need to get all the velocities and temp, sal, use DepthValuesSetH
	
	if (hydrodynamicData->IAm(TYPE_NETCDFMOVER)) numTimes = /*OK*/ (dynamic_cast<NetCDFMover *>(hydrodynamicData))->GetNumTimesInFile();
	else return -1;
	if (numTimes < 1) return -1;
	timeValInSecs = /*OK*/ (dynamic_cast<NetCDFMover *>(hydrodynamicData))->GetTimeValue(numTimes-1);
	if (numTimes>1 && timeValInSecs < model->GetEndTime()) {printNote("The hydrodynamic data runs out before the model end time. This may affect CDOG run.");}

	numFiles = numTimes;
	// right now the grid file is getting written out numTimes (overwriting the same file each time). Should cut out the extra stuff
	for (k=0;k<numTimes;k++)
	{
		timeIndex = k;
	if (hydrodynamicData->IAm(TYPE_NETCDFMOVER)) err = (dynamic_cast<NetCDFMover *>(hydrodynamicData))->GetDepthProfileAtPoint(fSetSummary.startRelPos,timeIndex,&profilesH);
	if (!err && profilesH) numProfileSets = _GetHandleSize((Handle)profilesH)/sizeof(**profilesH);
	else return -1;
	// code goes here, add an h.dat file  numValues * largest depth (or larger?)
	// code goes here, delete all u,v,w,temp,sal files in folder

	if (k==0)
	{	// should be the same every time
		totDepth = INDEXH(profilesH,numProfileSets-1).depth;
		//if (totDepth < fSetSummary.z*depthCF[fDepthUnits-1]) {printError("The spill depth must be less than the total depth");err=-1;goto done;}	// or add a bottom set of values (same as above or zero?)
		if (totDepth < fSetSummary.z*depthCF[fDepthUnits-1]) {printError("The spill depth must be less than the total depth");err=-1;return err;}	// or add a bottom set of values (same as above or zero?)
	}
	for (j=0;j<6;j++)
	{
		// figure out path name here
		// make sure any extra files have been deleted (e.g. u002.dat)
		if (j==0 && k>0) continue;	// depth is not time dependent
		GetFileName(j+1,fileName);
		strcpy(cdogFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogFilePathForInput,fileName);
		if (j>0 && j<6) 
		{
			sprintf(fileNum,"%03ld",k+1);
			strcat(cdogFilePathForInput,fileNum);
		}
		strcat(cdogFilePathForInput,".DAT");

		(void)hdelete(0, 0, cdogFilePathForInput);
		if (err = hcreate(0, 0, cdogFilePathForInput, 'ttxt', 'TEXT'))
			{ TechError("WriteToPath()", "hcreate()", err); return err; }
		if (err = FSOpenBuf(0, 0, cdogFilePathForInput, &bfpb, 100000, FALSE))
			{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
		// new CDOG version number
		strcpy(buffer,"CDOGG2.1\t (version number)");	// header line
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		// each file needs a header (0.0 usually)
		if (j==0)
		{
			// need grid information
			sprintf(numCellStr,"%ld%c%ld%c",fNumGridCells,',',fNumGridCells,',');
			strcpy(headerStr,numCellStr);
			StringWithoutTrailingZeros(outputStr,numProfileSets-1,1);
			strcat(headerStr,outputStr);
			strcat(headerStr,",");
			StringWithoutTrailingZeros(outputStr,fLengthOfGridCellInMeters / 1000.,1);	// convert to km
			strcat(headerStr,outputStr);
			strcat(headerStr,"\t ()");	// start time
		}
		else
		{
			timeValInSecs = /*OK*/ (dynamic_cast<NetCDFMover *>(hydrodynamicData))->GetTimeValue(timeIndex);
			timeInHrs = (timeValInSecs - model->GetStartTime()) / 3600.;
			sprintf(headerStr,"%lf",timeInHrs);
			//strcpy(headerStr,"0.00");	// start time - need to get this from the netcdf file
			strcat(headerStr,"\t ()");	// start time
		}
		if (j<6)
		{
			strcpy(buffer,headerStr);
			strcat(buffer,NEWLINESTRING);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			if (j==0)
			{
				//double totDepth = INDEXH(fDepthValuesH,numProfileSets-1).depth;
				//double totDepth = INDEXH(profilesH,numProfileSets-1).depth;
				StringWithoutTrailingZeros(outputStr,totDepth,1);
				StringWithoutTrailingZeros(numCellStr,numValues,1);
				{
					strcpy(buffer,numCellStr);
					strcat(buffer,"*");
					strcat(buffer,outputStr);
				}
				strcat(buffer,"\t ()");	// start time
				strcat(buffer,NEWLINESTRING);
				if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			}
			for (i=0;i<numProfileSets;i++)
			{
				// the first one is a little different, depth has one less entry and it's not time varying
				// header is numCellsx,numCellsy,numCellsz,dx
				//profileSet = INDEXH(fDepthValuesH,i);
				profileSet = INDEXH(profilesH,i);
				//if (i>0) dataToExport[0] = profileSet.depth - previousDepth;
				dataToExport[0] = profileSet.depth;
				//previousDepth = profileSet.depth;
				dataToExport[1] = profileSet.value.u;
				dataToExport[2] = profileSet.value.v;
				dataToExport[3] = profileSet.w;
				dataToExport[4] = profileSet.temp;
				dataToExport[5] = profileSet.sal;
				//if (i==0 && j==0) continue;
				StringWithoutTrailingZeros(numCellStr,numValues,1);
				StringWithoutTrailingZeros(outputStr,dataToExport[j],4);
				/////
				// need number of horizontal grid points, 16*.2, etc
				if (j==0)
				{
					strcpy(buffer,outputStr);
					strcat(buffer,"\t");
					StringWithoutTrailingZeros(outputStr,fCDOGDiffusivityInfo.horizDiff,4);	// horizontal diffusion, get from TRandom?
					strcat(buffer,outputStr);
					//strcat(buffer,"\tHorizontal diffusivity at water surface (m^2/s)");
					//strcat(buffer,NEWLINESTRING);
					//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
					/////
					StringWithoutTrailingZeros(outputStr,fCDOGDiffusivityInfo.vertDiff,4);	// 
					strcat(buffer,"\t");
					strcat(buffer,outputStr);
					//strcat(buffer,"\tVertical diffusivity at water surface (m^2/s)");
					//strcat(buffer,NEWLINESTRING);
					//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
					/////
				}
				else
				{
					strcpy(buffer,numCellStr);
					strcat(buffer,"*");
					strcat(buffer,outputStr);
				}
				strcat(buffer,"\t ()");
				strcat(buffer,NEWLINESTRING);
				if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			}
		}
	FSCloseBuf(&bfpb);
	}
	if (profilesH)  {DisposeHandle((Handle)profilesH); profilesH=0;}
	}	
	// delete any u,v,w with larger file numbers in cdog input folder
	// cdog will use whatever files are there
	// cdog requires same number of u,v,w,temp,sal files I think
	for (i=numFiles;i<kMaxNumCDOGInputFiles;i++)
	{
		sprintf(uFileName,"U%03ld.dat",i+1);
		//strcpy(uFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		//strcat(uFilePath,uFileName);
		strcpy(cdogUFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogUFilePathForInput,uFileName);
		if (FileExists(0,0,cdogUFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogUFilePathForInput);
		}
		else
			break;
		
		sprintf(vFileName,"V%03ld.dat",i+1);
		//strcpy(vFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		//strcat(vFilePath,vFileName);
		strcpy(cdogVFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogVFilePathForInput,vFileName);
		if (FileExists(0,0,cdogVFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogVFilePathForInput);
		}

		sprintf(wFileName,"W%03ld.dat",i+1);
		//strcpy(wFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		//strcat(wFilePath,wFileName);
		strcpy(cdogWFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogWFilePathForInput,wFileName);
		if (FileExists(0,0,cdogWFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogWFilePathForInput);
		}
		// should also handle temp, sal files
		sprintf(tempFileName,"Temp%03ld.dat",i+1);
		strcpy(cdogTempFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogTempFilePathForInput,tempFileName);
		if (FileExists(0,0,cdogTempFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogTempFilePathForInput);
		}
		sprintf(salFileName,"Sal%03ld.dat",i+1);
		strcpy(cdogSalFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogSalFilePathForInput,salFileName);
		if (FileExists(0,0,cdogSalFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogSalFilePathForInput);
		}
	}

done:
	//FSCloseBuf(&bfpb);
	if(err) {	
		// the user has already been told there was a problem
		FSCloseBuf(&bfpb);
		(void)hdelete(0, 0, cdogFilePathForInput); // don't leave them with a partial file
	}
	if (profilesH)  {DisposeHandle((Handle)profilesH); profilesH=0;}
	return err;
}
/////////////////////////////////////////////////
OSErr CDOGLEList::ExportProfilesToCDOGInputFolder()
{
	BFPB bfpb;
	OSErr err = 0;
	DepthValuesSet profileSet;
	long i,j,numProfileSets = 0,numValues,numFiles=1;	// numFiles - for now not allowing time dependence
	char cdogFolderPathWithDelimiter[256], cdogInputFolderPathWithDelimiter[256];
	char fileName[32], outputStr[32], headerStr[32], numCellStr[32];
	float dataToExport[6],previousDepth=0;
	char cdogFilePathForInput[256], buffer[512];
	char cdogUFilePathForInput[256], cdogVFilePathForInput[256], cdogWFilePathForInput[256], cdogTempFilePathForInput[256], cdogSalFilePathForInput[256];
	char uFileName[256], vFileName[256], wFileName[256], tempFileName[256], salFileName[256];
	//char uFilePath[256], vFilePath[256], wFilePath[256];
	WorldRect mapBounds;
	double degreeLat, degreeLon;

	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogInputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"input",DIRDELIMITER);
	if (fDepthValuesH) numProfileSets = _GetHandleSize((Handle)fDepthValuesH)/sizeof(**fDepthValuesH);
	numValues = (fNumGridCells+1)*(fNumGridCells+1);

	mapBounds = GetCDOGMapBounds();
	degreeLat = (mapBounds.hiLat - mapBounds.loLat) / 1000000.;
	degreeLon = (mapBounds.hiLong - mapBounds.loLong) / 1000000.;
	fLengthOfGridCellInMeters = (degreeLat * METERSPERDEGREELAT) / fNumGridCells;	// always one giant grid cell at this point
	// decide which is larger? or which is smaller?

	// code goes here, add an h.dat file  numValues * largest depth (or larger?)
	// code goes here, delete all u,v,w,temp,sal files in folder
	for (j=0;j<6;j++)
	{
		// figure out path name here
		// make sure any extra files have been deleted (e.g. u002.dat)
		GetFileName(j+1,fileName);
		strcpy(cdogFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogFilePathForInput,fileName);
		if (j>0 && j<6) strcat(cdogFilePathForInput,"001");
		strcat(cdogFilePathForInput,".DAT");

		(void)hdelete(0, 0, cdogFilePathForInput);
		if (err = hcreate(0, 0, cdogFilePathForInput, 'ttxt', 'TEXT'))
			{ TechError("WriteToPath()", "hcreate()", err); return err; }
		if (err = FSOpenBuf(0, 0, cdogFilePathForInput, &bfpb, 100000, FALSE))
			{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
		// new CDOG version number
		strcpy(buffer,"CDOGG2.1\t (version number)");	// header line
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		// each file needs a header (0.0 usually)
		if (j==0)
		{
			// need grid information
			sprintf(numCellStr,"%ld%c%ld%c",fNumGridCells,',',fNumGridCells,',');
			strcpy(headerStr,numCellStr);
			StringWithoutTrailingZeros(outputStr,numProfileSets-1,1);
			strcat(headerStr,outputStr);
			strcat(headerStr,",");
			StringWithoutTrailingZeros(outputStr,fLengthOfGridCellInMeters / 1000.,1);	// convert to km
			strcat(headerStr,outputStr);
			strcat(headerStr,"\t ()");	// start time
		}
		else
		{
			strcpy(headerStr,"0.00");	// start time
			strcat(headerStr,"\t ()");	// start time
		}
		if (j<6)
		{
			strcpy(buffer,headerStr);
			strcat(buffer,NEWLINESTRING);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			if (j==0)
			{
				double totDepth = INDEXH(fDepthValuesH,numProfileSets-1).depth;
				StringWithoutTrailingZeros(outputStr,totDepth,1);
				StringWithoutTrailingZeros(numCellStr,numValues,1);
				{
					strcpy(buffer,numCellStr);
					strcat(buffer,"*");
					strcat(buffer,outputStr);
				}
				strcat(buffer,"\t ()");	// start time
				strcat(buffer,NEWLINESTRING);
				if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			}

			for (i=0;i<numProfileSets;i++)
			{
				// the first one is a little different, depth has one less entry
				// header is numCellsx,numCellsy,numCellsz,dx
				profileSet = INDEXH(fDepthValuesH,i);
				//if (i>0) dataToExport[0] = profileSet.depth - previousDepth;
				dataToExport[0] = profileSet.depth;
				//previousDepth = profileSet.depth;
				dataToExport[1] = profileSet.value.u;
				dataToExport[2] = profileSet.value.v;
				dataToExport[3] = profileSet.w;
				dataToExport[4] = profileSet.temp;
				dataToExport[5] = profileSet.sal;
				//if (i==0 && j==0) continue;
				StringWithoutTrailingZeros(numCellStr,numValues,1);
				StringWithoutTrailingZeros(outputStr,dataToExport[j],4);
				/////
				// need number of horizontal grid points, 16*.2, etc
				if (j==0)
				{
					strcpy(buffer,outputStr);
					strcat(buffer,"\t");
					StringWithoutTrailingZeros(outputStr,fCDOGDiffusivityInfo.horizDiff,4);	// horizontal diffusion, get from TRandom?
					strcat(buffer,outputStr);
					//strcat(buffer,"\tHorizontal diffusivity at water surface (m^2/s)");
					//strcat(buffer,NEWLINESTRING);
					//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
					/////
					StringWithoutTrailingZeros(outputStr,fCDOGDiffusivityInfo.vertDiff,4);	// 
					strcat(buffer,"\t");
					strcat(buffer,outputStr);
					//strcat(buffer,"\tVertical diffusivity at water surface (m^2/s)");
					//strcat(buffer,NEWLINESTRING);
					//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
					/////
				}
				else
				{
					strcpy(buffer,numCellStr);
					strcat(buffer,"*");
					strcat(buffer,outputStr);
				}
				strcat(buffer,"\t ()");
				strcat(buffer,NEWLINESTRING);
				if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			}
		}
	FSCloseBuf(&bfpb);
	}
		/*else if (j==6)
		{
			StringWithoutTrailingZeros(outputStr,previousDepth,1);
			{
				strcpy(buffer,numCellStr);
				strcat(buffer,"*");
				strcat(buffer,outputStr);
			}
			strcat(buffer,NEWLINESTRING);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}*/
	// delete any u,v,w with larger file numbers in cdog input folder
	// cdog will use whatever files are there
	for (i=numFiles;i<kMaxNumCDOGInputFiles;i++)
	{
		sprintf(uFileName,"U%03ld.dat",i+1);
		//strcpy(uFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		//strcat(uFilePath,uFileName);
		strcpy(cdogUFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogUFilePathForInput,uFileName);
		if (FileExists(0,0,cdogUFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogUFilePathForInput);
		}
		else
			break;
		
		sprintf(vFileName,"V%03ld.dat",i+1);
		//strcpy(vFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		//strcat(vFilePath,vFileName);
		strcpy(cdogVFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogVFilePathForInput,vFileName);
		if (FileExists(0,0,cdogVFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogVFilePathForInput);
		}

		sprintf(wFileName,"W%03ld.dat",i+1);
		//strcpy(wFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		//strcat(wFilePath,wFileName);
		strcpy(cdogWFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogWFilePathForInput,wFileName);
		if (FileExists(0,0,cdogWFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogWFilePathForInput);
		}
		// should also handle temp, sal files
		sprintf(tempFileName,"Temp%03ld.dat",i+1);
		strcpy(cdogTempFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogTempFilePathForInput,tempFileName);
		if (FileExists(0,0,cdogTempFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogTempFilePathForInput);
		}
		sprintf(salFileName,"Sal%03ld.dat",i+1);
		strcpy(cdogSalFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogSalFilePathForInput,salFileName);
		if (FileExists(0,0,cdogSalFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogSalFilePathForInput);
		}
	}
done:
	//FSCloseBuf(&bfpb);
	if(err) {	
		FSCloseBuf(&bfpb);
		// the user has already been told there was a problem
		(void)hdelete(0, 0, cdogFilePathForInput); // don't leave them with a partial file
	}
	return err;
}
/////////////////////////////////////////////////
/*OSErr CDOGLEList::ExportProfilesToCDOGInputFolder_old()
{
	BFPB bfpb;
	OSErr err = 0;
	DepthValuesSet profileSet;
	long i,j,numProfileSets = 0,numValues;
	char cdogFolderPathWithDelimiter[256], cdogInputFolderPathWithDelimiter[256];
	char fileName[32], outputStr[32], headerStr[32], numCellStr[32];
	float dataToExport[6],previousDepth=0;
	char cdogFilePathForInput[256], buffer[512];

	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogInputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"input",DIRDELIMITER);
	if (fDepthValuesH) numProfileSets = _GetHandleSize((Handle)fDepthValuesH)/sizeof(**fDepthValuesH);
	numValues = (fNumGridCells+1)*(fNumGridCells+1);

	// code goes here, add an h.dat file  numValues * largest depth (or larger?)
	for (j=0;j<7;j++)
	{
		// figure out path name here
		// make sure any extra files have been deleted (e.g. u002.dat)
		GetFileName(j+1,fileName);
		strcpy(cdogFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogFilePathForInput,fileName);
		if (j>0 && j<6) strcat(cdogFilePathForInput,"001");
		strcat(cdogFilePathForInput,".DAT");

		(void)hdelete(0, 0, cdogFilePathForInput);
		if (err = hcreate(0, 0, cdogFilePathForInput, 'ttxt', 'TEXT'))
			{ TechError("WriteToPath()", "hcreate()", err); return err; }
		if (err = FSOpenBuf(0, 0, cdogFilePathForInput, &bfpb, 100000, FALSE))
			{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
		// each file needs a header (0.0 usually)
		if (j==0)
		{
			// need grid information
			sprintf(numCellStr,"%ld%c%ld%c",fNumGridCells,',',fNumGridCells,',');
			strcpy(headerStr,numCellStr);
			StringWithoutTrailingZeros(outputStr,numProfileSets-1,1);
			strcat(headerStr,outputStr);
			strcat(headerStr,",");
			StringWithoutTrailingZeros(outputStr,fLengthOfGridCellInMeters / 1000.,1);	// convert to km
			strcat(headerStr,outputStr);
		}
		else
		{
			strcpy(headerStr,"0.00");	// start time
		}
		if (j<6)
		{
			strcpy(buffer,headerStr);
			strcat(buffer,NEWLINESTRING);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		//}
		for (i=0;i<numProfileSets;i++)
		{
			// the first one is a little different, depth has one less entry
			// header is numCellsx,numCellsy,numCellsz,dx
			profileSet = INDEXH(fDepthValuesH,i);
			if (i>0) dataToExport[0] = profileSet.depth - previousDepth;
			previousDepth = profileSet.depth;
			dataToExport[1] = profileSet.value.u;
			dataToExport[2] = profileSet.value.v;
			dataToExport[3] = profileSet.w;
			dataToExport[4] = profileSet.temp;
			dataToExport[5] = profileSet.sal;
			if (i==0 && j==0) continue;
			StringWithoutTrailingZeros(numCellStr,numValues,1);
			StringWithoutTrailingZeros(outputStr,dataToExport[j],1);
			/////
			// need number of horizontal grid points, 16*.2, etc
			if (j==0)
				strcpy(buffer,outputStr);
			else
			{
				strcpy(buffer,numCellStr);
				strcat(buffer,"*");
				strcat(buffer,outputStr);
			}
			strcat(buffer,NEWLINESTRING);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
		}
		else if (j==6)
		{
			StringWithoutTrailingZeros(outputStr,previousDepth,1);
			{
				strcpy(buffer,numCellStr);
				strcat(buffer,"*");
				strcat(buffer,outputStr);
			}
			strcat(buffer,NEWLINESTRING);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
done:
		FSCloseBuf(&bfpb);
		if(err) {	
			// the user has already been told there was a problem
			(void)hdelete(0, 0, cdogFilePathForInput); // don't leave them with a partial file
		}
	}
	return err;
}*/
/////////////////////////////////////////////////
OSErr CDOGLEList::CopyTemperatureFilesToCDOGInputFolder()	
{
	OSErr err = 0;
	char cdogFolderPathWithDelimiter[256], cdogInputFolderPathWithDelimiter[256];
	char cdogTempFilePathForInput[256];
	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogInputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"input",DIRDELIMITER);
	strcpy(cdogTempFilePathForInput,cdogInputFolderPathWithDelimiter);
	strcat(cdogTempFilePathForInput,"TEMP001.DAT");
	// check that the file paths have been set and move them to the CDOG input folder
//OSErr MyCopyFile(short vRefNumFrom, long dirIDFrom, CHARPTR nameFrom,
				 //short vRefNumTo, long dirIDTo, CHARPTR nameTo)
				 
	//check they exist at some point and conform to the format
	if (!fCDOGTempSalInfo.temperatureFieldFilePath[0] || !FileExists(0,0,fCDOGTempSalInfo.temperatureFieldFilePath))
		strcpy(fCDOGTempSalInfo.temperatureFieldFilePath,"MacintoshHD:Desktop Folder:CDOG:Expl-input2:TEMP001.DAT");
	err = MyCopyFile(0,0,fCDOGTempSalInfo.temperatureFieldFilePath,0,0,cdogTempFilePathForInput);
	return err;
}
/////////////////////////////////////////////////
OSErr CDOGLEList::CopySalinityFilesToCDOGInputFolder()	
{
	OSErr err = 0;
	char cdogFolderPathWithDelimiter[256], cdogInputFolderPathWithDelimiter[256];
	char cdogSalFilePathForInput[256];
	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogInputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"input",DIRDELIMITER);
	strcpy(cdogSalFilePathForInput,cdogInputFolderPathWithDelimiter);
	strcat(cdogSalFilePathForInput,"SAL001.DAT");
	// check that the file paths have been set and move them to the CDOG input folder
//OSErr MyCopyFile(short vRefNumFrom, long dirIDFrom, CHARPTR nameFrom,
				 //short vRefNumTo, long dirIDTo, CHARPTR nameTo)
				 
	//check they exist at some point and conform to the format
	if (!fCDOGTempSalInfo.salinityFieldFilePath[0] || !FileExists(0,0,fCDOGTempSalInfo.salinityFieldFilePath))	
		return -1;
		//strcpy(fCDOGTempSalInfo.salinityFieldFilePath,"MacintoshHD:Desktop Folder:CDOG:Expl-input2:SAL001.DAT");
	err = MyCopyFile(0,0,fCDOGTempSalInfo.salinityFieldFilePath,0,0,cdogSalFilePathForInput);
	// delete any higher numbered temp/sal files in input folder
	return err;
}
/////////////////////////////////////////////////
OSErr CDOGLEList::ExportGNOMECurrentsForCDOG()
{
	OSErr err = 0;
	// Will need to set grid size, find velocity in each grid box
	// export in CDOG format
	// vertically - pick a profile, and vertical step
	// if time dependent, how to connect to time step?
	// grid must match temp,sal profiles, and number of time steps
	return err;
}
/////////////////////////////////////////////////
OSErr CDOGLEList::CopyHydrodynamicsFilesToCDOGInputFolder()	
{
	OSErr err = 0;
	long i,numFiles=kMaxNumCDOGInputFiles;
	char cdogFolderPathWithDelimiter[256], cdogInputFolderPathWithDelimiter[256];
	char cdogUFilePathForInput[256], cdogVFilePathForInput[256], cdogWFilePathForInput[256];
	char uFileName[256], vFileName[256], wFileName[256];
	char uFilePath[256], vFilePath[256], wFilePath[256];
	GetCDogFolderPathWithDelimiter(cdogFolderPathWithDelimiter);
	sprintf(cdogInputFolderPathWithDelimiter,"%s%s%c",cdogFolderPathWithDelimiter,"input",DIRDELIMITER);

	// will need to loop over the number of files in the folder
	// check that these file paths have been set and move them to the CDOG input folder
	// time series of u,v,w files, possibly created by GNOME from 2D current and profile or 3D current
	//if (fCDOGHydrodynamicInfo.methodOfDeterminingCurrents==1)
	if (!fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath[0])
		return -1;
		//strcpy(fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath,"MacintoshHD:Desktop Folder:CDOG:Expl-input2:");
	
	// need to keep going until there are no more files
	// loop through max number of files for now, and break out when done
	for (i=0;i<kMaxNumCDOGInputFiles;i++)
	{
		sprintf(uFileName,"U%03ld.dat",i+1);
		strcpy(uFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		strcat(uFilePath,uFileName);
		strcpy(cdogUFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogUFilePathForInput,uFileName);
		
		sprintf(vFileName,"V%03ld.dat",i+1);
		strcpy(vFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		strcat(vFilePath,vFileName);
		strcpy(cdogVFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogVFilePathForInput,vFileName);

		sprintf(wFileName,"W%03ld.dat",i+1);
		strcpy(wFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		strcat(wFilePath,wFileName);
		strcpy(cdogWFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogWFilePathForInput,wFileName);

		if (!FileExists(0,0,uFilePath)) 
		{	
			numFiles = i;
			break; 
			// also error if the first file does not exist for u,v,or w
			// always assume the same number of u,v,w files
		}
		if (err = MyCopyFile(0,0,uFilePath,0,0,cdogUFilePathForInput)) return err;
		if (err = MyCopyFile(0,0,vFilePath,0,0,cdogVFilePathForInput)) return err;
		if (err = MyCopyFile(0,0,wFilePath,0,0,cdogWFilePathForInput)) return err;
	}
	// delete any u,v,w with larger file numbers in cdog input folder
	// cdog will use whatever files are there
	for (i=numFiles;i<kMaxNumCDOGInputFiles;i++)
	{
		sprintf(uFileName,"U%03ld.dat",i+1);
		//strcpy(uFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		//strcat(uFilePath,uFileName);
		strcpy(cdogUFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogUFilePathForInput,uFileName);
		if (FileExists(0,0,cdogUFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogUFilePathForInput);
		}
		else
			break;
		
		sprintf(vFileName,"V%03ld.dat",i+1);
		//strcpy(vFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		//strcat(vFilePath,vFileName);
		strcpy(cdogVFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogVFilePathForInput,vFileName);
		if (FileExists(0,0,cdogVFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogVFilePathForInput);
		}

		sprintf(wFileName,"W%03ld.dat",i+1);
		//strcpy(wFilePath,fCDOGHydrodynamicInfo.hydrodynamicFilesFolderPath);
		//strcat(wFilePath,wFileName);
		strcpy(cdogWFilePathForInput,cdogInputFolderPathWithDelimiter);
		strcat(cdogWFilePathForInput,wFileName);
		if (FileExists(0,0,cdogWFilePathForInput)) 	
		{
			(void)hdelete(0, 0, cdogWFilePathForInput);
		}
	}

	// check for a 501 ? above the limit ...

	return err;
}
/**************************************************************************************************/

void CDOGLEList::Draw(Rect r, WorldRect view)
{
	Point start,end;
	
	#ifdef IBM
		short xtraOffset = 1;
	#else
		short xtraOffset = 0;
	#endif
	short offset = round(2*PixelsPerPoint()); // 2 points each way

	if (model -> UserIsEditingSplots() && ! this -> UserIsEditingMeInMapDrawingRect())
		return; // so we don't confuse the user about which splots are being edited
	
	end = start = WorldToScreenPoint(fSetSummary.startRelPos, settings.currentView, MapDrawingRect());
	
	/// figure out the rectangle to draw when beached,inWater,etc.
	/// Note: in Windows,  currentHDC can have  600 dpi or more
	/// so we need to use a rectangle in all cases
		
	// for now always draw the plus...	
	if (/*binitialLEsVisible &&*/ initialLEs && !(this->fLeType == UNCERTAINTY_LE)) // don't draw initial position for uncertainty LEs
		//DrawLEPositionMarker(&start,&end);// draw position marker, TOLEList won't draw this for initialLE list
		
	{
		// draw plus at start
		MyMoveTo(start.h-offset,start.v);MyLineTo(start.h+offset+xtraOffset,start.v);
		MyMoveTo(start.h,start.v-offset);MyLineTo(start.h,start.v+offset+xtraOffset);
		// draw plus at end
		MyMoveTo(end.h-offset,end.v);MyLineTo(end.h+offset+xtraOffset,end.v);
		MyMoveTo(end.h,end.v-offset);MyLineTo(end.h,end.v+offset+xtraOffset);
		// draw line from start to end
		MyMoveTo(start.h,start.v);MyLineTo(end.h,end.v);
	}
	
	TOLEList::Draw(r,view);
	
	return;
	
}
