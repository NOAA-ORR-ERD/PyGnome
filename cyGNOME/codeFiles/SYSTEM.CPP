
#include "CROSS.h"
#include "MakeMovie.h"
#include "myRandom.h"
#include "NetCDFMover.h"

#if MACB4CARBON
	#include <Balloons.h>
#endif
#include "Wizard.h"

#include "contdlg.h"

//#include "HtmlHelp.h"

#ifdef MAC
	#pragma segment SYSTEM // JLM 
    #include <unistd.h>
#endif


#ifdef IBM

WindowPtr hMainWnd, toolWnd, messageWnd;

static Boolean inMenu = FALSE;
static UINT timer = 0;

#endif

void SetShowGrid(Boolean set)
{
#ifdef MAC
	//UncheckAllItems(VIEWMENU);
	Boolean currentState = ItemChecked(ANALYSISMENU,SHOWGRIDITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck(ANALYSISMENU,SHOWGRIDITEM-ANALYSISMENU);
#else
	//UncheckAllItems2(GetSubMenu(GetMenu(mapWindow),2));
	Boolean currentState = ItemChecked2(GetSubMenu(GetMenu(mapWindow),4),SHOWGRIDITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWGRIDITEM-ANALYSISMENU);
#endif
}
void SetShowSelectedTriangles(Boolean set)
{
#ifdef MAC
	Boolean currentState = ItemChecked(ANALYSISMENU,SHOWSELECTEDTRIITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck(ANALYSISMENU,SHOWSELECTEDTRIITEM-ANALYSISMENU);
#else
	Boolean currentState = ItemChecked2(GetSubMenu(GetMenu(mapWindow),4),SHOWSELECTEDTRIITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWSELECTEDTRIITEM-ANALYSISMENU);
#endif
}

void SetDepthContoursCheckMark(Boolean set)
{
#ifdef MAC
	Boolean currentState = ItemChecked(ANALYSISMENU,SHOWCONTOURSITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck(ANALYSISMENU,SHOWCONTOURSITEM-ANALYSISMENU);
#else
	Boolean currentState = ItemChecked2(GetSubMenu(GetMenu(mapWindow),4),SHOWCONTOURSITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWCONTOURSITEM-ANALYSISMENU);
#endif
}

void SetDepthLegendCheckMark(Boolean set)
{
#ifdef MAC
	Boolean currentState = ItemChecked(ANALYSISMENU,SHOWCONTOURLABELSITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck(ANALYSISMENU,SHOWCONTOURLABELSITEM-ANALYSISMENU);
#else
	Boolean currentState = ItemChecked2(GetSubMenu(GetMenu(mapWindow),4),SHOWCONTOURLABELSITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWCONTOURLABELSITEM-ANALYSISMENU);
#endif
}

void SetDosageCheckMarks(Boolean set)
{
#ifdef MAC
	Boolean currentState = ItemChecked(ANALYSISMENU,CALCDOSAGEITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck(ANALYSISMENU,CALCDOSAGEITEM-ANALYSISMENU);
	currentState = ItemChecked(ANALYSISMENU,SHOWDOSAGEITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck(ANALYSISMENU,SHOWDOSAGEITEM-ANALYSISMENU);
#else
	Boolean currentState = ItemChecked2(GetSubMenu(GetMenu(mapWindow),4),CALCDOSAGEITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),CALCDOSAGEITEM-ANALYSISMENU);
	currentState = ItemChecked2(GetSubMenu(GetMenu(mapWindow),4),SHOWDOSAGEITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWDOSAGEITEM-ANALYSISMENU);
#endif
}
// code goes here, could combine some of these that are only called to initialize
void SetSuppressDrawingCheckMark(Boolean set)
{
#ifdef MAC
	Boolean currentState = ItemChecked(ANALYSISMENU,SUPPRESSDRAWINGITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck(ANALYSISMENU,SUPPRESSDRAWINGITEM-ANALYSISMENU);
#else
	Boolean currentState = ItemChecked2(GetSubMenu(GetMenu(mapWindow),4),SUPPRESSDRAWINGITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SUPPRESSDRAWINGITEM-ANALYSISMENU);
#endif
}

void SetShowMaxTriCheckMark(Boolean set)
{
#ifdef MAC
	Boolean currentState = ItemChecked(ANALYSISMENU,SHOWMAXTRIITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck(ANALYSISMENU,SHOWMAXTRIITEM-ANALYSISMENU);
#else
	Boolean currentState = ItemChecked2(GetSubMenu(GetMenu(mapWindow),4),SHOWMAXTRIITEM-ANALYSISMENU);
	if(set != currentState)
		ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWMAXTRIITEM-ANALYSISMENU);
#endif
}


/////////////////////////////////////////////////
Boolean HaveMapBounds(void)
{
	WorldRect bounds = model->GetMapBounds();
	if(EqualWRects(voidWorldRect,bounds)) return false; // no bounds
	return true;
}


Boolean LimitZoomIn(void)
{
	if(!HaveMapBounds()) return false; // nothing to limit 
	if(model->GetModelMode() <  ADVANCEDMODE) return true;
	return false; // advanced users can zoom to their hearts content
}

Boolean LimitZoomOut(void)
{
	if(!HaveMapBounds()) return false; // nothing to limit 
	return true; // might as well limit all users
}

float MinKilometersPerScreenInch(void)
{
	if(model && model->fWizard)
	{
		Boolean haveMinKilometerValue;
		float minKilometers = -1;
		haveMinKilometerValue = model->fWizard->GetMinKilometersPerScreenInch(&minKilometers);
		if(haveMinKilometerValue && minKilometers > 0)
			return minKilometers;
	}

	return 1.0; // default is 1 kilometer
}

long MostZoomedInScale(void)
{ // JLM
	//const double kMinNumGridCellsPerSiteRect = 1.8;
	long scale = MOST_ZOOMED_IN_SCALE;
	long currentScale = CurrentScaleDenominator();
	if(LimitZoomIn())
	{
		// want to have at least 1 km per screen inch
		short numPix = PixelsPerInchCurrent();
		// find the distance in km of a screen inch
		long dLat = ScreenToWorldDistance(numPix);
		double latDistInKilometers = LatToDistance(dLat); 
		double kMinDistanceInKilometers = MinKilometersPerScreenInch();
		
		if(latDistInKilometers > 0.0) 
			scale = kMinDistanceInKilometers*currentScale/latDistInKilometers;
	}
	return scale;
}


long MostZoomedOutScale(void)
{ // JLM
	long scale = MOST_ZOOMED_OUT_SCALE;
	if(LimitZoomOut())
	{
		// JLM 6/10/99 we want the map to fill 1/3 of the screen either horizontally or vertically
		// a rectangles screen height is given by dLat/(PixelsPerMilDeg() * scale)
		// the screenHeightInPix = screen_dLat/(PixelsPerMilDeg() * scale)
		// or scale = screen_dLat/(PixelsPerMilDeg()*screenHeightInPix)
		// thus the scale where the maps dLat is 1/3 the screen's dLat
		// 
		Rect drawingRect = MapDrawingRect();
		Rect r = WorldToScreenRect(model->GetMapBounds(), settings.currentView,drawingRect);
		short drH = RectHeight(drawingRect);
		short drW = RectWidth(drawingRect);
		long rH = RectHeight(r);
		long rW = RectWidth(r);
		double factorH = 1.0;
		double factorW = 1.0;
		double factor = 1.0;
		// we want the rect to be no less than 1/3 of the drawing rect
		// if the map rect is equal to the 1/3 drawing rect we take the current scale
		// if the map rect is  equal to the 2/3 drawing rect we can zoom in once, max scale is twice the current scale
		// if the map rect is  equal to the 4/3 drawing rect we can zoom in twice, max scale is 4x the current scale
		if(rH > 0 && rW && drH > 0 && drW > 0)
		{
			factorH = 3*rH/(double)drH;
			factorW = 3*rW/(double)drW;
			factor = _max(factorH,factorW);
			scale = factor*CurrentScaleDenominator();
			if(scale > MOST_ZOOMED_OUT_SCALE) scale = MOST_ZOOMED_OUT_SCALE;
			if(scale < MOST_ZOOMED_IN_SCALE) scale = MOST_ZOOMED_IN_SCALE;
		}
	}
	return scale;
}

/////////////////////////////////////////////////


Boolean IsLeftSideButton(short base, short index) //JLM 7/27/98
{
	switch(base)
	{
		case DISABLEDBUTTONPICTBASE:
		case BUTTONPICTBASE:
		case DEPRESSEDBUTTONPICTBASE:
			if(OPENBUTTON <= index && index <= DOWNBUTTON) 
				return true;;
	}
	return false;
}


#ifdef MAC  ///{
/////////////////////////////////////////////////
/////////////////////////////////////////////////

MenuHandle gMyHelpMenu =nil;
Boolean SystemHelpAvailable(void)
{
#if TARGET_API_MAC_CARBON 
	return false;
#else
	// starting in Sys 8, they put it to the right of our menu
	long sysVersion;
	
	OSErr err = Gestalt(gestaltSystemVersion, &sysVersion);
	if(err)return false;
	if(0x0500 <= sysVersion && sysVersion < 0x0800) return false;
	return true; // sytem 8 or some higher system
#endif
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
#endif ///}


//////////////////////////////// COMMON CODE //////////////////////////////////
static long startno =-1,oldSegNum = -1,middleno=-1,endno=-1;
short buttons[] = { OPENBUTTON, SAVEBUTTON, -1,
					UPBUTTON, DOWNBUTTON, ADDBUTTON, SETTINGSBUTTON, DELETEBUTTON, -2 };

//short tools[] = { ARROWTOOL, ZOOMPLUSTOOL, ZOOMMINUSTOOL, HANDTOOL, SPILLTOOL, SPRAYCANTOOL, ERASETOOL, RULERTOOL, -1, -2 };
short tools[] = { ARROWTOOL, ZOOMPLUSTOOL, ZOOMMINUSTOOL, HANDTOOL, SPILLTOOL, SPRAYCANTOOL, ERASETOOL, RULERTOOL, -1, -2 };
static short clickNo = 0;
long NUMTOOLS = 8;

Boolean OKToQuit()
{
	OSErr err=0;
	if(UserIsInEditMode()) {
		// leave edit mode
		err = model -> EditObjectInMapDrawingRect(nil);
		if(err) return FALSE;	// an object insists on being edited
	}
	return TRUE;
}


OSErr InitSettings()
{
	long size, i;
	Boolean useDefaults = true;
	OSErr err =  0;
	
	if (!MyGetFileSize(TATvRefNum, TATdirID, "SETTINGS.OSM", &size))
		if (size != sizeof(settings))
			hdelete(TATvRefNum, TATdirID, "SETTINGS.OSM");
	
	err = ReadFileContents(NONTERMINATED,TATvRefNum, TATdirID,"SETTINGS.OSM", (char *)&settings, sizeof(Settings), 0);

	if(!err) 
	{
		// verify the file is OK
		if(settings.platform == PLATFORM 
			&& settings.settingsVersion == SETTINGS_VERSION 
			&& M1TAB1 <= settings.preferencesScreen 
			&&  settings.preferencesScreen <= M1TABMAX)
			useDefaults = false; // file is OK
	}
						 
						 
						 
	if(useDefaults)
	{
		memset(&settings,0,sizeof(Settings)); 
		settings.platform = PLATFORM;			
		settings.settingsVersion = SETTINGS_VERSION;
		/////
		SetWorldRect(&settings.currentView, -90000000, -180000000, 90000000, 180000000);
		settings.listWidth = ButtonGroupWidth()+SPLITBARWIDTH+TOOLSIDE; //add 1 extra TOOLSIDE just so we have a little more space
//		settings.textMode = SCREENMODE;
		settings.screenMode = COLORMODE; // actually its always color
		settings.printMode = BANDWMODE;
		settings.massUnits = KILOGRAMS;
		settings.areaUnits = SQUAREKILOMETERS;
		settings.distanceUnits = KILOMETERS;
		settings.backgroundColor = BACKWHITE;
		settings.showLatLongLines = TRUE;
		//settings.llPosition = LL_BELOW; //JLM
		settings.llPosition = LL_ABOVE;//JLM
		settings.latLongFormat = DEGMIN;
		settings.showIntermediateLines = FALSE;
		settings.customGrid = FALSE;
		settings.longLineSpace = 10;
		settings.longLineUnits = MINUTES;
		settings.longLabelSpace = 10;
		settings.longLabelUnits = MINUTES;
		settings.latLineSpace = 10;
		settings.latLineUnits = MINUTES;
		settings.latLabelSpace = 10;
		settings.latLabelUnits = MINUTES;
		settings.preferencesScreen = M1TAB1;
//		settings.settingsScreen = 1;
		settings.modelStartMode = NOVICEMODE;
		settings.sprayCanSize = SPRAY_SIZE_MEDIUM;
		
		for (i = 0 ; i < 5 ; i++) GetStdCaveatLine(settings.caveat[i],i);

		settings.daylightSavingsTimeFlag = DAYLIGHTSAVINGSON;
		settings.omitFooter = 0;
		//for (i = 0 ; i < 20 ; i++) settings.reserved[i] = 0;
		//for (i = 0 ; i < 19 ; i++) settings.reserved[i] = 0;
		for (i = 0 ; i < 18 ; i++) settings.reserved[i] = 0;

	}
	
	settings.currentTool = ARROWTOOL;
	settings.inBackground = FALSE;
	settings.colorQDAvailable = TRUE;
	settings.undoCode = UNDO_NOTHING;
	settings.doNotPrintError = FALSE;
	settings.quitting = FALSE;
	
	return 0;
}

short MapRectLeft()
{
	return settings.listWidth;
}

OSErr InitList()
{
	Rect r, client = GetWindowPortRect(mapWindow);
	GrafPtr savePort;
	OSErr err = 0;
	
	r.top = TOOLBARHEIGHT;
	r.left = 0;
	r.bottom = client.bottom - MESSAGEHEIGHT;
	r.right = MapRectLeft();
	
#ifdef MAC
	r.top++;
	r.bottom--;
#endif
	
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	TextFontSizeFace(kFontIDGeneva,9,normal);
	//err = VLNewInWindow(mapWindow, 0, &r, 0, DrawObjectsList, &objects);	//carbon here
	err = VLNewInWindow(mapWindow, &r, 0, DrawObjectsList2, &objects);
	objects.hRect = HiliteObjectsRect;		// hilites only to right-edge of selected text
	SetPortGrafPort(savePort);
	
	return err;
}

Boolean gNoaaVersion = FALSE;
Boolean gMearnsVersion = FALSE;
Boolean gDispersedOilVersion = FALSE;
Boolean gDrawBitmapInBlackAndWhite = TRUE;
Boolean gSavingOrPrintingPictFile = FALSE;
OSType gMySignature = 'COSM';

/////////////////////////////////////////////////
#ifdef MAC

/// JLM HandleODOC

OSErr RequiredCheck(const AppleEvent  * ae)
{
	OSErr err = noErr;
	Size actualSize;
	DescType actualType;
	
	err = AEGetAttributePtr(ae, keyMissedKeywordAttr, typeWildCard,
							&actualType, 0L, 0, &actualSize);
	if (err == errAEDescNotFound) return noErr;
	if (err == noErr) return errAEEventNotHandled;
	return err;
}

pascal_ifMac OSErr HandleODOC(const AppleEvent  * ae, AppleEvent  * reply, long refCon)
{
#pragma unused(reply, refCon)
	FSSpec  myFSS;
	AEDescList docList;
	long	itemsInList,i;
	AEKeyword  theAEKeyword;
	DescType	typeCode;
	Size actualSize;
	OSErr	err = RequiredCheck(ae);
	
	err = AEGetParamDesc(ae,keyDirectObject,typeAEList,&docList);
	if(err == noErr)
	{
		FInfo fndrInfo;
		
		err = AECountItems(&docList,&itemsInList);
		
		for(i= 1;i<=itemsInList &&err == noErr;i++)
		{
			err = AEGetNthPtr(&docList,i,typeFSS,&theAEKeyword,&typeCode,(Ptr)&myFSS,sizeof(myFSS),&actualSize);
			if(err == noErr)
			{
				err = FSpGetFInfo(&myFSS,&fndrInfo);
				if(err == noErr)
				{
					// get the full path name
					char pathName[256] = "";
					PathNameFromDirID(myFSS.parID,myFSS.vRefNum,pathName);
					my_p2cstr((StringPtr)pathName);
					
					// make sure the last char is a directory delimiter
					if(pathName[strlen(pathName) -1] != ':') strcat(pathName,":"); 
					
					my_p2cstr((StringPtr)myFSS.name);
					strcat(pathName,(char*)myFSS.name);
					
					if(pathName[0] != 0)
					{
						err = true; // in case it is a file we do not recognize
						switch(fndrInfo.fdType)
						{
							case 'TEXT':
								if(IsCommandFile(pathName))
								{
									err = DoCommandFile(pathName);
									if(err == noErr || err == USERCANCEL) return noErr;
								}
								break;
							case 'SAVE':
							case '.SAV':
							{
								long savedMode = model -> GetModelMode();
								if (model -> GetModelMode() != ADVANCEDMODE) 
									model -> SetModelMode (ADVANCEDMODE);	// must be in advanced mode to make SAVE files
								err = OpenSaveFileFromPath(pathName);
								if(err == noErr || err == USERCANCEL) return noErr;
								if (err && savedMode != ADVANCEDMODE) model -> SetModelMode (savedMode);
								break;
							}
							case WIZARDSAVEFILETYPE:
							case WIZARDFILETYPE:
								err = OpenSaveFileFromPath(pathName);
								if(err == noErr || err == USERCANCEL) return noErr;
								break;
						}
					}
				}
			}
		}
			
		printNote("Please use the Open menu item to open this file.");
	}
	return err;
}

pascal_ifMac OSErr HandleOAPP(const AppleEvent  * ae, AppleEvent  * reply, long refCon)
{
#pragma unused(reply, refCon)
	
	return RequiredCheck(ae);
}

pascal_ifMac OSErr HandlePDOC(const AppleEvent  * ae, AppleEvent  * reply, long refCon)
{
	OSErr	err = RequiredCheck(ae);
#pragma unused(reply, refCon)
	
	printNote("Please print using the Print menu item.");
	return err;
}

pascal_ifMac OSErr HandleQUIT(const AppleEvent  * ae, AppleEvent  * reply, long refCon)
{	// This routine is called when the user choses Shutdown from in the Finder.
	OSErr err = RequiredCheck(ae);
	OSErr dontQuitErr = userCanceledErr; //-128
	Boolean userChangedMind = false;
#pragma unused(reply, refCon)
	
	if (!OKToQuit())
		return dontQuitErr;
		
	if(model->GetModelMode()  <  ADVANCEDMODE)  
		userChangedMind = model->fWizard->QuitMenuHit();
	else 
		userChangedMind = CloseSaveFile(true,true);
	if (userChangedMind) 
		return dontQuitErr;

	DoQuit();
	
	settings.quitting = true; // this will cause us to quit in the main event loop
	
	return(noErr);
}

void InstallEventHandlers(void)
{
#if MACB4CARBON
	AEInstallEventHandler(kCoreEventClass, kAEOpenDocuments,
						  (AEEventHandlerUPP)MakeUPP((ProcPtr)HandleODOC, uppAEEventHandlerProcInfo), 0, false);

	AEInstallEventHandler(kCoreEventClass, kAEQuitApplication,
						  (AEEventHandlerUPP)MakeUPP((ProcPtr)HandleQUIT, uppAEEventHandlerProcInfo), 0, false);
	AEInstallEventHandler(kCoreEventClass, kAEPrintDocuments,
						  (AEEventHandlerUPP)MakeUPP((ProcPtr)HandlePDOC, uppAEEventHandlerProcInfo), 0, false);
	AEInstallEventHandler(kCoreEventClass, kAEOpenApplication,
						  (AEEventHandlerUPP)MakeUPP((ProcPtr)HandleOAPP, uppAEEventHandlerProcInfo), 0, false);
#else
	AEInstallEventHandler(kCoreEventClass, kAEOpenDocuments,
						  MakeAEEventHandlerUPP(HandleODOC), 0, false);
	AEInstallEventHandler(kCoreEventClass, kAEQuitApplication,
						  MakeAEEventHandlerUPP(HandleQUIT), 0, false);
	AEInstallEventHandler(kCoreEventClass, kAEPrintDocuments,
						  MakeAEEventHandlerUPP(HandlePDOC), 0, false);
	AEInstallEventHandler(kCoreEventClass, kAEOpenApplication,
						  MakeAEEventHandlerUPP(HandleOAPP), 0, false);
	//AEInstallEventHandler('NOAA', 'AEVT',
						 // MakeAEEventHandlerUPP(HandleNOAAEvent), 0, false);
#endif
}

#endif
/////////////////////////////////////////////////

void InitAnalysisMenu(void)
{
	/// check if diagnostic version

	if (gDispersedOilVersion) return;

#ifdef MAC
	gDispersedOilVersion =true;
	GetAndInsertMenu(ANALYSISMENU, 0);
	//DrawMenuBar(); on the MAC, Draw MenuBar will be called by the calling function
#else
{
	MenuHandle hMenu = LoadMenu (hInst,"ANALYSIS");
	if(hMenu) {
		gDispersedOilVersion = true;
		//AppendMenu (GetMenu(hMainWnd), MF_POPUP,(UINT)hMenu ,"&Analysis");
		// code goes here, change all the 4's to kZeroRelativeAnalysisMenuPosition or something
		InsertMenu (GetMenu(hMainWnd), 4, MF_POPUP | MF_BYPOSITION,(UINT)hMenu ,"&Analysis");
		DrawMenuBar(hMainWnd);
	}
}
#endif

	NUMTOOLS = 10;
	tools[8] = LASSOTOOL;
	tools[9] = SHORTSELTOOL;
	tools[10] = -1;
	tools[11] = -2;
	//gMearnsVersion = true;	// temp for old .SAV files, remove this?
	// the check marks need to be initted after the menu
	SetShowGrid(0);
	SetShowSelectedTriangles(0);
	SetDosageCheckMarks(0);
	SetDepthContoursCheckMark(0);
	SetDepthLegendCheckMark(0);
	SetSuppressDrawingCheckMark(0);
	SetShowMaxTriCheckMark(0);	
}


OSErr CommonInit()
{
	char s[256];
	Seconds time;
	DateTimeRec tTime;
	OSErr err = 0;
	
	//objects.port = 0;
	
	if (!(rainyDayReserve = _NewHandle(35000))) return -1; // memory to be given up in an emergency
	
// hgetvol is getting the Current Working Directory on Windows
	if (err = hgetvol(TATFolder, &TATvRefNum, &TATdirID))
		{ TechError("CommonInit()", "hgetvol()", err); return err; }
		
	gApplicationFolder[0] = 0;
	gWorkingDirectory[0] = 0;

#ifdef MAC
	{	// on the MAC, we need to chop the package contents to get the folder the user "sees" as the MARPLOT "folder"
		char path[256];
		OSStatus err = 0;
		FSSpec spec;
		GetFullPath(TATvRefNum, TATdirID, "", path);
		if(IsBundledApplication(path)) {
			ChopExtensionAndHiddenPackageFolders(path);
			// hmmm... we want the parent of the .app folder, try this...
			strcat(path,".app");
			my_c2pstr(path);
			err = FSMakeFSSpec(0, 0, (StringPtr)path, &spec);
			if(err) { TechError("CommonInit()", "FSMakeFSSpec()", err); return err; }
			
			TATdirID = spec.parID;
			// for debugging, recontruct the path to make sure we've got it correct
			GetFullPath(TATvRefNum,TATdirID,"",path);
			strcpy(gApplicationFolder,path);
		}
		else { // if not bundled, MARPLOT is in the folder the user sees as the MARPLOT "folder"
			//gMarplotFolderDirID = MARPLOT_hgetvol_dirID;
		}
	}
	#else
		// in Windows, the GNOME.exe is always in the GNOME "folder"
	#endif

	InitSettings();
	ResetAllRandomSeeds();
	
	GetDateTime(&time);
	SecondsToDate (time, &tTime);
	DateToSeconds (&tTime, &time);
	SecondsToDate (time, &tTime);
	DateToSeconds (&tTime, &time);
	
	GetDateTime(&time);
	SecondsToDate (time, &tTime);

	// check that the year is in the range we allow
	{
		if(tTime.year < FirstYearInPopup() || tTime.year > LASTYEARINPOPUP)
			printNote("The time specified by your computer clock is outside of GNOME's allowable year range for the Start Date of a spill.  Check that your computer clock is set correctly.");
	}
	/////////////////////////////////////////////////

	tTime.second = 0;
	DateToSeconds (&tTime, &time);
	
	model = new TModel(time);
	if (!model)
		{ TechError("CommonInit()", "new TModel()", 0); return -1; }
	
	if (err = model->InitModel())
		{ TechError("CommonInit()", "InitModel()", err); return err; }
	
	model->SetModelMode(settings.modelStartMode);
	if(settings.modelStartMode < ADVANCEDMODE)	settings.preferencesScreen = M1TAB1;

#ifndef NO_GUI
	if (err = InitList()) return err;
	InvalListLength();
#endif

#ifdef NO_GUI
	if (model -> GetModelMode() != ADVANCEDMODE) 
	model -> SetModelMode (ADVANCEDMODE);	// can't do standard mode with no gui
#endif

	// the check marks need to be initted after the menu
	/*SetShowGrid(0);
	SetShowSelectedTriangles(0);
	SetDosageCheckMarks(0);
	SetDepthContoursCheckMark(0);
	SetDepthLegendCheckMark(0);
	SetSuppressDrawingCheckMark(0);
	SetShowMaxTriCheckMark(0);	*/

#ifdef MAC
	InstallEventHandlers(); //JLM HandleODOC
#endif

/*	if (err = InitIAC("OSSM", mapWindow))
	{
		printError("Error starting IAC.");
		bIACInited = false;
		err = noErr;			// reset error code after setting no-IAC flag
	}
	else
		bIACInited = true;

	if (err = AddFriends(TATvRefNum, TATdirID)) return err;
	GreetFriend("HOLA", 0);
*/	
	
	/// check if noaa version
	if(FileExists(TATvRefNum, TATdirID,"noaa.ver"))
	{
		gNoaaVersion =true;
		//printNote("GNOME has found a noaa.ver file. Spills will be allowed outside the spillable area.");
		InitAnalysisMenu();
	}
	
	/// check if noaa version
	if(FileExists(TATvRefNum, TATdirID,"mearns.txt"))
	{
		gMearnsVersion = true;	// use this until all the .SAV files have been fixed
		InitAnalysisMenu();	// let Alan have the menus from the start, maybe a better way to do this...
		//printNote("GNOME has found a noaa.ver file. Spills will be allowed outside the spillable area.");
	}
	
	return 0;
}

void CommonQuit()
{
	OSErr err = 0;
	settings.doNotPrintError = FALSE;
	
	err = WriteFileContents(TATvRefNum, TATdirID, "SETTINGS.OSM", APPL_SIG, 'SETS',
						 		(char *)&settings, sizeof(Settings), 0);
	//if(err) printError(GS1(113, 0)); // Error saving settings.
	
	model->Dispose();
	InvalidateMapImage();
	CleanupMovieStuff();// JLM  7/7/99
	
//	GreetFriend("BYE ", 0);
//	CloseIAC();
}

OSErr AddFriends(short vRefNum, long dirID)
{
	return 0;
}


OSErr GreetFriend(char *message, char *signature) 
{
	return 0;
}

void DisplayMessage(char *msg)
{
#ifdef NO_GUI
	//OSErr err = WriteErrorToCommandFileErrorLog(msg,9999);
	return;
#endif
	Rect r;
#ifdef MAC
	Rect saveClip;
#endif 
	GrafPtr oldPort;
	static Boolean nextMessageTemp = FALSE;
	static char prevMessage[256] = "", currentMessage[256];
	
	if (!mapWindow) return;
    
	if (msg) {
		if (!strcmp(msg, "NEXTMESSAGETEMP")) { nextMessageTemp = TRUE; return; }
		if (!strcmp(msg, currentMessage)) return;
		strcpy(currentMessage, msg);
	}
	else
		if (!strcmp(currentMessage, prevMessage)) return;
	
	GetPortGrafPtr(&oldPort);
#ifdef MAC
	r = MessageRect();
	SetPortWindowPort(mapWindow);
	saveClip = MyClipRect(r);
#else
	if (!IsWindow(messageWnd)) return;
	
	r = GetWindowPortRect(messageWnd);
	r.top++;
	SetPortWindowPort(messageWnd);
#endif
	TextSizeSmall();
	EraseRect(&r);
	#if TARGET_API_MAC_CARBON
		//  before OS X, DrawGrowIcon drew lines, now we have to do it ourselves
		MyMoveTo(r.left,r.top); MyLineTo(r.right-1,r.top);
	#endif

#ifdef MAC
	MyMoveTo(r.left + 5, r.bottom - 3);
#else
	MyMoveTo(r.left + 5, r.bottom - 4);
#endif
	if (settings.colorQDAvailable)
		RGBForeColor(&colors[BLACK]);
	else
		ForeColor(blackColor);
	
#ifdef MAC
	TextFont(kFontIDGeneva);
#else
	TextFont(kFontIDTimes);
#endif
	TextSizeSmall();
	if (msg) {
		drawstring(msg);
		if (strncmp(msg, GS3(129, 0), 7) && !nextMessageTemp) // Drawing
			strcpy(prevMessage, msg);
	}
	else {
		drawstring(prevMessage);
		strcpy(currentMessage, prevMessage);
	}
	nextMessageTemp = FALSE;
#ifdef MAC
	MyClipRect(saveClip);
#endif
	SetPortGrafPort(oldPort);
}

void DisplayNumber(long n)
{
	char s[20];
	
	sprintf(s, "%ld", n);
	DisplayMessage(s);
}

Rect FullScreenMapWindowRect()
{
	Rect r;
	
	#if TARGET_API_MAC_CARBON
		GetAvailableWindowPositioningBounds(GetMainDevice(), &r); // excludes menubar and dock
	#else
		r = GetScreenBounds();
	#endif
	
	//MyInsetRect(&r, 5, 5);
	MyInsetRect(&r, TITLEBARHEIGHT+5,TITLEBARHEIGHT+ 5); // JLM , allow more room
#ifdef MAC
	r.top += TITLEBARHEIGHT + 20;
//	r.right -= 50;
#endif
	
	return r;
}

void FullScreenMapWindow()
{
	Rect r = FullScreenMapWindowRect();
	
	PlaceWindow(mapWindow, r, FALSE);
}

Rect ToolBarRect(void)
{
#ifdef MAC
	Rect ToolPaletteRect();
	
	return ToolPaletteRect();
#else
	return GetWindowPortRect(toolWnd);
#endif
}

WindowPtr ToolBarWindow(void)
{
#ifdef MAC
	return mapWindow;
#else
	return toolWnd;
#endif
}

Rect SplitBarRect()
{
	short left = MapRectLeft();
	Rect r, toolBar = ToolBarRect();
	
	MySetRect(&r, left - SPLITBARWIDTH, toolBar.top, left, toolBar.bottom);
	MyOffsetRect(&r, 2, 0);
#ifdef MAC
	r.right++;
#else
	r.bottom--;
	MyOffsetRect(&r, -1, 0);
#endif
	
	return r;
}

short TimeLineLength(void)
{
	// we need to make sure the time line is large enough
	// so that the time string in DrawTimeString fits over it
	static short sFirstCalculation = 0;
	if(sFirstCalculation> 0) return sFirstCalculation;
	if(!mapWindow) return MINTIMELINELENGTH; // in case called before the window exists
	
	short lenInPixels = MINTIMELINELENGTH;
	short w;
	GrafPtr savePort;
	char *p, s[64];
	DateTimeRec time;
	// see how long the sample string is
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	TextFontSize(kFontIDGeneva,LISTTEXTSIZE);
	/// an arbitrary date, try for wide chars
	time.month = 12;
	time.day = 29;
	time.year = 2000;
	time.hour = 23;
	time.minute = 59;
	time.second = 0;
	Date2String(&time, s);
	if (p = strrchr(s, ':')) p[0] = 0; // remove seconds
	lenInPixels =  _max(lenInPixels, stringwidth(s));
	// try again with a nmonth that might have a leading 0
	time.month = 9;
	Date2String(&time, s);
	if (p = strrchr(s, ':')) p[0] = 0; // remove seconds
	lenInPixels =  _max(lenInPixels, stringwidth(s));
	///
	SetPortGrafPort(savePort);
	sFirstCalculation = lenInPixels;
	return lenInPixels;
}

Rect PlayerRect()
{
	short i, offset = MapRectLeft();
	Rect r;
	
	for (i = 0 ; tools[i] != -2 ; i++) {
		if (tools[i] != -1)
			offset += TOOLSIDE + TOOLSPACE;
		else
			offset += TOOLSPACE * 3;
	}
	
	r.left = offset;
	r.right = r.left + TOOLSPACE * 4 + PLAYERSIDE * 3 + TimeLineLength();
	
	r.top = (TOOLBARHEIGHT - TOOLSIDE) / 2;
	r.bottom = r.top + TOOLSIDE;
	
#ifdef IBM
	r.bottom--;
#endif
	
	return r;
}

Boolean PtInPlayerRect(Point p)
{
	Rect r = PlayerRect();
	if (MyPtInRect (p, &r))
		return TRUE;
	return FALSE;
}

Rect TimeLineRect()
{
	Rect r = PlayerRect();
	
	r.left += TOOLSPACE * 2 + PLAYERSIDE;
	r.right -= TOOLSPACE * 3 + PLAYERSIDE * 2;
	
	r.bottom -= TOOLSPACE + 3;
	r.top = r.bottom - TOOLSPACE * 1.5;
	
#ifdef MAC
	r.top -= 2;
	MyOffsetRect(&r, 0, -2);
#else
	MyOffsetRect(&r, 0, 2);// JLM 1/21/99 move it down a touch
	r.bottom--; // account for frame rect ?
#endif
	
	return r;
}


Rect TimeTriangleRect(short h)
{
	Rect 	triRect, r = TimeLineRect();
	//checks on h
	if (h > r.right) h = r.right;
	if (h < r.left + 1) h = r.left + 1;
	//
	triRect.top = r.bottom+1;
	triRect.bottom = triRect.top + 6;
	triRect.left = h-6;
	triRect.right = h+5;
	return triRect;
}


Rect TimeBarTriangleRect (void)
{
	Rect 	r = TimeLineRect(), triRect;
	Seconds	time;
	float	ratio;
	
	r.left++;
	r.top++;
#ifdef MAC
	r.right--;
	r.bottom--;
#endif

	// JLM fix drawing bug, though the real cure would be elsewhere 
	// in the model when they set the start,end and model times
	{
		Seconds endTime 		= model -> GetEndTime ();
		Seconds startTime 		= model -> GetStartTime ();
		Seconds modelTime 		= model -> GetModelTime ();

		if( startTime <= modelTime  && startTime < endTime) ratio = (float) (modelTime - startTime) / (float) (endTime - startTime);
		else ratio = 0;
		if(ratio>1) ratio= 1;
	}
	
	// draw the current-time triangle
	// to indicate the current model time
	triRect = TimeTriangleRect(r.left + RectWidth(r) * ratio);
	
	return triRect;
}


enum { GRAYBAR, TRIANGLE};
void DrawTimeBarPartAtTime(short partCode,Seconds theTime)
{
	Seconds endTime = model -> GetEndTime ();
	Seconds startTime = model -> GetStartTime ();
	float ratio;
	short saveLeft, saveRight;
	Point	triPoint;
	Rect r = TimeLineRect();
	r.left++;
	r.top++;
#ifdef MAC
	r.right--;
	r.bottom--;
#endif
	
	if( startTime <= theTime  && startTime < endTime) ratio = (float) (theTime - startTime) / (float) (endTime - startTime);
	else ratio = 0;
	if(ratio>1) ratio= 1;
	
	switch(partCode)
	{
		case GRAYBAR:
			if (model->bHindcast)
			{
				saveRight = r.right;
				saveLeft  = r.left;
				r.left = r.left + RectWidth(r) * ratio;
				RGBForeColor(&colors[GRAY]);
				PaintRect(&r);
				r.right = r.left;
				r.right = saveLeft;
				RGBForeColor(&colors[LIGHTGRAY]);
				PaintRect(&r);
				RGBForeColor(&colors[BLACK]);
			}
			else
			{
				saveRight = r.right;
				saveLeft  = r.left;
				r.right = r.left + RectWidth(r) * ratio;
				RGBForeColor(&colors[GRAY]);
				PaintRect(&r);
				r.left = r.right;
				r.right = saveRight;
				RGBForeColor(&colors[LIGHTGRAY]);
				PaintRect(&r);
				RGBForeColor(&colors[BLACK]);
				}
			break;
		
		case TRIANGLE:
			DrawTimeBarTriangle (r.left + RectWidth(r) * ratio, true);
			break;
	}
}

void DrawTimeString(Seconds desiredTime)
{
	char *p, s[64];
	DateTimeRec time;
	Rect r = TimeLineRect();

	SecondsToDate (desiredTime, &time);
	Date2String(&time, s);
	if (p = strrchr(s, ':')) p[0] = 0; // remove seconds

	TextMode(patCopy);// so it erases previous string
	TextFontSize(kFontIDGeneva,LISTTEXTSIZE);
	RGBBackColor(&colors[LIGHTGRAY]);
	MyMoveTo((r.left + r.right - stringwidth(s)) / 2, r.top - 3);
	drawstring(s);
	RGBBackColor(&colors[WHITE]);
	TextMode(patOr);
}

long EditStatus(void)
{
	if (model -> UserIsEditingSplots())
		return EDITINGSPLOTS;
	
	if (IsEditTool(settings.currentTool))
		return EDITTOOLSELECTED;
	
	return NOT_EDITING;
}

Boolean UserIsInEditMode(void)
{
	long status = EditStatus();
	return status != NOT_EDITING;
}

void DrawEditStatus(void)
{
	long status = EditStatus();
	char str[64] = "";
	char str2[64] = "";
	Rect pr = PlayerRect();
	short h,v,w,lineHt;
	FontInfo		fInfo;

	switch (status)
	{
		case EDITTOOLSELECTED:
			strcpy(str,"Select splots in summary list");
			strcpy(str2,"(select arrow tool to leave edit mode)");
			break;
		case EDITINGSPLOTS:
			strcpy(str,"In Editing Splots Mode");
			strcpy(str2,"(select arrow tool to leave edit mode)");
			break;
		default:
			return; // nothing to draw
	}
	
	TextFontSize(kFontIDGeneva,LISTTEXTSIZE);
	GetFontInfo(&fInfo);
	lineHt = fInfo.ascent + fInfo.descent;

	if(str[0]) {
		w = stringwidth(str);
		h = (pr.left + pr.right -w)/2;
		v = (2*pr.top + pr.bottom)/3 + lineHt/2;
		MyMoveTo(h,v);
		drawstring(str);
	}

	if(str2[0]) {
		w = stringwidth(str2);
		h = (pr.left + pr.right -w)/2;
		v = (pr.top + 2*pr.bottom)/3 + lineHt/2;
		MyMoveTo(h,v);
		drawstring(str2);
	}
}

void DrawPlayer(void)
{
	Rect r = PlayerRect();
	long status = EditStatus();

	MyFrameRect(&r);

	if(status == NOT_EDITING)
		DisplayCurrentTime(TRUE);
	else
		DrawEditStatus();
}

void DisplayCurrentTime(Boolean bDrawTriangle)
{
	WindowPtr	toolWindow;
	Rect saveClip, wPortRect;
	GrafPtr savePort;
	long status = EditStatus();
	Rect r = TimeLineRect();

	GetPortGrafPtr(&savePort);
	toolWindow = ToolBarWindow ();
	SetPortWindowPort(toolWindow);
	wPortRect = GetWindowPortRect (toolWindow);
	saveClip = MyClipRect (wPortRect);
	
	if(status != NOT_EDITING) return;

	MyFrameRect(&r);
	DrawTimeString(model->GetModelTime());
	
	DrawTimeBarPartAtTime(GRAYBAR,model->GetLastComputeTime());
	if (bDrawTriangle) DrawTimeBarPartAtTime(TRIANGLE,model->GetModelTime());
	
	MyClipRect (saveClip);
	SetPortGrafPort(savePort);
	
	DrawTools(PLAYERPICTBASE, RESETBUTTON);
	DrawTools(PLAYERPICTBASE, PLAYBUTTON);
	DrawTools(PLAYERPICTBASE, STEPBUTTON);
}


WindowPtr ListWindow()
{
#ifdef MAC
	return mapWindow;
#else
	//return objects.port;
	return objects.window;
#endif
}

Rect DrawingRect(short leftOffset, short rightOffset)
{
	Rect r = GetWindowPortRect(mapWindow);
#ifdef IBM
	static Rect prevValidRect = { 0, 0, 0, 0 };
#endif
	
	r.top += TOOLBARHEIGHT;
	r.left += leftOffset;
	r.bottom -= MESSAGEHEIGHT;
	r.right -= rightOffset;
#ifdef IBM
	if (IsIconic(mapWindow)) return prevValidRect;
	if (!IsWindowVisible(mapWindow)) return FullScreenMapWindowRect();
	prevValidRect = r;
#endif
	
	return r;
}

Rect ListDrawingRect()
{
	Rect r = GetWindowPortRect(mapWindow);
	
	return DrawingRect(0, RectWidth(r) - MapRectLeft());
}

static Rect tempDrawingRect;
static Boolean drawingToScreen = TRUE;

void SetTempDrawingRect(Rect *r)
{
	if (!r) { drawingToScreen = TRUE; return; }
	
	drawingToScreen = FALSE;
	tempDrawingRect = *r;
}

Boolean GetTempDrawingRect(Rect *r)
{
	if (!drawingToScreen) (*r) = tempDrawingRect;
	
	return !drawingToScreen;
}

Rect MapDrawingRect()
{
	if (!drawingToScreen) return tempDrawingRect;
	
#ifdef MAC
	return DrawingRect(MapRectLeft() + 1, RIGHTBARWIDTH);
#else
	return DrawingRect(MapRectLeft(), RIGHTBARWIDTH);
#endif
}

void InvalMapDrawingRect()
{
	InvalRectInWindow(MapDrawingRect(), mapWindow);
#ifdef MAC
	InvalRectInWindow(MessageRect(), mapWindow);
#else
	InvalidateRect(messageWnd, 0, TRUE);
#endif
}

void InvalMapWRect(WorldRect wr)
{
#ifdef MAC
	Rect r, r2, r3;
	
	r = WorldToScreenRect(wr, settings.currentView, MapDrawingRect());
	r2 = MapDrawingRect();
	SectRect(&r, &r2, &r3);
	InvalRectInWindow(r3, mapWindow);
#else
	InvalRectInWindow(WorldToScreenRect(wr, settings.currentView, MapDrawingRect()), mapWindow);
#endif
}

void InvalMapWindowBorders()
{
#ifdef MAC
	InvalRectInWindow(ListDrawingRect(), mapWindow);
	InvalRectInWindow(ScrollAreaRight(), mapWindow);
	InvalRectInWindow(ScrollAreaBottom(), mapWindow);
#else
	InvalRectInWindow(GetWindowPortRect(messageWnd), messageWnd);
	InvalRectInWindow(GetWindowPortRect(toolWnd), toolWnd);
#endif
}

void InvalMapWindow()
{
#ifdef IBM
	if (!IsWindowVisible(mapWindow)) return;
#endif
	
	InvalRectInWindow(GetWindowPortRect(mapWindow), mapWindow);
	InvalMapWindowBorders();
}


void ResizeChildWindows()
{
	short h;
	Rect r, client = GetWindowPortRect(mapWindow);
	Rect r1,r2,r3,r4;
	short topMessageWnd = _max(TOOLBARHEIGHT,client.bottom - MESSAGEHEIGHT); 
	
#ifdef IBM
	MoveWindow(messageWnd, 0, topMessageWnd, client.right, MESSAGEHEIGHT, TRUE);
	MoveWindow(toolWnd, 0, 0, client.right, TOOLBARHEIGHT, TRUE);
#endif
	// the list "window"
	h = _max(0,(client.bottom - client.top) - (TOOLBARHEIGHT + MESSAGEHEIGHT));
//	h = (h / objects.lineHeight) * objects.lineHeight;
	MySetRect(&r, -1, TOOLBARHEIGHT - 1, MapRectLeft(), TOOLBARHEIGHT + h + 1);
#ifdef MAC
#if TARGET_API_MAC_CARBON
	InvalRectInWindow(ToolBarRect(),mapWindow);
#endif
	InvalRectInWindow(r, mapWindow);
	r.top++;
	r.bottom--;
#endif
	
	VLResize(&objects, r);
	
	PrintDrawingRect(TRUE, &r1,&r2,&r3,&r4);	// this call is causing bug in Windows 7, is it necessary?
}

short  ButtonGroupWidth(void)
{
	short i, offset = TOOLSPACE;
	for (i = 0 ; buttons[i] != -2 ; i++) {
		switch (buttons[i]) 
		{
			case -1: offset += TOOLSPACE * 3;break; // -1 is code for 3 space widths
			
			case UPBUTTON: break; // shares the same place as the downbutton
				
			case DOWNBUTTON: offset += UPDOWNWIDTH + TOOLSPACE; break;
			
			default: offset += TOOLSIDE + TOOLSPACE; break;
		}
	}
	
	return offset;

}

void ForEachTool(void (*proc)(short, short, short, short))
{
	short i, vOffset, offset = 0;
	short minWidth = ButtonGroupWidth() + SPLITBARWIDTH;
	
	if (settings.listWidth < minWidth)
		offset = settings.listWidth - minWidth;
	
	for (i = 0 ; buttons[i] != -2 ; i++) {
		if (buttons[i] != -1) {
			vOffset = 0;
			if (buttons[i] == DOWNBUTTON) vOffset += UPDOWNHEIGHT + UPDOWNSPACE;
			proc(BUTTONPICTBASE,
				buttons[i],
				 offset + TOOLSPACE,
				 (TOOLBARHEIGHT - TOOLSIDE) / 2 + vOffset);
			switch (buttons[i]) {
				case UPBUTTON: break;
				case DOWNBUTTON: offset += UPDOWNWIDTH + TOOLSPACE; break;
				default: offset += TOOLSIDE + TOOLSPACE; break;
			}
		}
		else
			offset += TOOLSPACE * 3;
	}
	
	offset = MapRectLeft();
	

	for (i = 0 ; tools[i] != -2 ; i++) {
		if (tools[i] != -1) {
			proc(TOOLPICTBASE,			//CHANGED JLE 5/21/98
				 i,                   //tools[i], ---  this should be  index of tool not tool id itself
				 offset + TOOLSPACE,
				 (TOOLBARHEIGHT - TOOLSIDE) / 2);
			offset += TOOLSIDE + TOOLSPACE;
		}
		else
			offset += TOOLSPACE * 3;
	}
	
	proc(PLAYERPICTBASE, RESETBUTTON,
		 offset + TOOLSPACE,
		 (TOOLBARHEIGHT - PLAYERSIDE) / 2); 
	proc(PLAYERPICTBASE, PLAYBUTTON,
		 offset + TOOLSPACE * 2 + PLAYERSIDE + TimeLineLength(),
		 (TOOLBARHEIGHT - PLAYERSIDE) / 2); 
	proc(PLAYERPICTBASE, STEPBUTTON,
		 offset + TOOLSPACE * 3 + PLAYERSIDE * 2 + TimeLineLength(),
		 (TOOLBARHEIGHT - PLAYERSIDE) / 2); 
}

static short sharedOverBase, sharedOverIndex, sharedDrawBase, sharedDrawIndex;
static Point sharedPoint;


void DrawPictureAtPt(short resId, short x,short y,Boolean center)
{
#ifdef MAC
	Rect frame;
	PicHandle p;
	if (p = GetPicture(resId)) {
		RgnHandle	saveClip = NewRgn ();

		//frame = (**p).picFrame;
		frame = GetPICTFrame(p);
		MyOffsetRect(&frame, -frame.left, -frame.top);
		if(center) MyOffsetRect(&frame, -(frame.left+frame.right)/2, -(frame.top+frame.bottom)/2);
		MyOffsetRect(&frame, x, y);

		GetClip (saveClip);
		ClipRect (&frame);
		DrawPicture(p, &frame);
		SetClip (saveClip);
		DisposeRgn (saveClip);
	}
#else
	HBITMAP b;
	if (b = LoadBitmap(hInst, MAKEINTRESOURCE(resId))) {
		if(center)
		{
			BITMAP bm;
			GetObject(b, sizeof(BITMAP), (LPSTR)&bm);
			x -= bm.bmWidth/2;
			y -= bm.bmHeight/2;
		}
		DrawBitmap(b, x, y, SRCCOPY);
		DeleteObject(b);
	}
#endif
}


void DrawTool(short base, short index, short x, short y)
{
	Boolean editing = UserIsInEditMode();

	// don't draw run-bar buttons during spray-splot-edit-mode
	if (editing) {
		if (index == RESETBUTTON) return;
		if (index == PLAYBUTTON) return;
		if (index == STEPBUTTON) return;
	}

	switch (sharedDrawBase) {
		case 0:
			if (base == TOOLPICTBASE && index == settings.currentTool)
				base = SELECTEDTOOLPICTBASE;
			break;
		case BUTTONPICTBASE:
		case DEPRESSEDBUTTONPICTBASE:
			if (base != BUTTONPICTBASE || (sharedDrawIndex != -1 && index != sharedDrawIndex)) return;
			base = sharedDrawBase;
			break;
		case TOOLPICTBASE:
		case DEPRESSEDTOOLPICTBASE:
			if (base != TOOLPICTBASE || (sharedDrawIndex != -1 && index != sharedDrawIndex)) return;
			base = sharedDrawBase;
			break;
		case PLAYERPICTBASE:
		case DEPRESSEDPLAYERPICTBASE:
			if (base != PLAYERPICTBASE || (sharedDrawIndex != -1 && index != sharedDrawIndex)) return;
			base = sharedDrawBase;
			break;
		case SELECTEDTOOLPICTBASE:
			if (base != TOOLPICTBASE || index != sharedDrawIndex) return;
			base = sharedDrawBase;
			break;
	}
	
	if (base == BUTTONPICTBASE && !OSSMButtonEnabled(index))
		base = DISABLEDBUTTONPICTBASE;
	
	if (base == PLAYERPICTBASE && !OSSMButtonEnabled(index))
		base = DISABLEDPLAYERPICTBASE;
	
	if (base == PLAYERPICTBASE && index == PLAYBUTTON && model->fRunning == TRUE)
		index += PAUSEPICTOFFSET;
	
	if (base == PLAYERPICTBASE || base == DEPRESSEDPLAYERPICTBASE || base == DISABLEDPLAYERPICTBASE)
		index -= RESETBUTTON;
	

	///{ JLM 7/27/98
	if(model->GetModelMode() <  ADVANCEDMODE
		&& IsLeftSideButton(base,index)) 
		return; // don't draw tools on left in Novice mode
	////////}
	
	DrawPictureAtPt(base+index,x,y,false);


}

void DrawTools(short base, short index)
{
	Rect r,toolBarRect = ToolBarRect();
	Rect saveClip,wPortRect;
	GrafPtr savePort;
#ifdef MAC
	WindowPtr w = mapWindow;
#else
	WindowPtr w = toolWnd;
#endif
	
	if (base == 0) {
	
		///////JLM 2/3/99  ////{
		wPortRect = GetWindowPortRect(w);
		saveClip = MyClipRect(wPortRect);
		//////////}//

		PenNormal();
		RGBForeColor(&colors[LIGHTGRAY]);
		PaintRect(&toolBarRect);
		
		//void FillRectWith3DBackground(Rect r);	// could replace previous 2 lines to match better
		//FillRectWith3DBackground(toolBarRect);
		RGBForeColor(&colors[BLACK]);

		//////////////////////////
		// in the lowest modes, the GNOME icon and helpful text 
		// replaces the left side buttons
		if(model->GetModelMode() <  ADVANCEDMODE)
		{	
			Rect leftHandRect = toolBarRect;
			Rect iconRect;
			short h,v;
			short lineHt = 12;
			short barHeight = toolBarRect.bottom - toolBarRect.top;// TOOLBARHEIGHT is 40
			
			TextFontSizeFace(kFontIDGeneva,9,normal);
			
			r = SplitBarRect();
			leftHandRect.right = r.left;
			saveClip = MyClipRect(leftHandRect);
			
			// draw icon here
			iconRect = leftHandRect;
			iconRect.right = iconRect.left+barHeight;
			DrawPictureAtPt(GNOME_GOING_RIGHT_ID,leftHandRect.left+(barHeight/2),leftHandRect.top+(barHeight/2),true);
			
			//
			h = iconRect.right+barHeight;
			v = leftHandRect.top + 11;
			MyMoveTo(h,v);
			TextFace(bold);
			drawstring("Summary List");
			TextFace(normal);
			h = iconRect.right + 4;
			v = leftHandRect.top + 23;
			MyMoveTo(h,v);
			drawstring("- Double click item to access its settings");
			v = leftHandRect.top + 35;
			MyMoveTo(h,v);
			drawstring("- Toggle carat to display list");

			MyClipRect (saveClip); //JLM 11/25/98
		
		}
		
	}
	else {
		GetPortGrafPtr(&savePort);
		SetPortWindowPort(w);
		///////JLM 11/25/98  ////{
		wPortRect = GetWindowPortRect(w);
		saveClip = MyClipRect(wPortRect);
		//////////}//
	}
	
	sharedDrawBase = base;
	sharedDrawIndex = index;
	ForEachTool(DrawTool);
	
	if (base == 0) {
		PenNormal();
		r = SplitBarRect();
#ifdef MAC
		Rect frame;
		PicHandle p;
	
		p = GetPicture(SPLITBARPICT);
		if (!p) { SysBeep(1); return; }
	
		//frame = (**p).picFrame;
		frame = GetPICTFrame(p);
		MyOffsetRect(&frame, r.left - 1, r.top - 1);
		DrawPicture(p, &frame);
#else
		SetPenPat(GRAY);
		PaintRect(&r);
		SetPenPat(BLACK);
		MyFrameRect(&r);
#endif	

		DrawPlayer();
		
		PenNormal();
		r = ToolBarRect();
		MyMoveTo(r.left, r.bottom - 1);
		MyLineTo(r.right, r.bottom - 1);
		// RGBForeColor(&colors[GRAY]);
		// MyMoveTo(r.left, r.top);
		// MyLineTo(r.right, r.top);
		// RGBForeColor(&colors[WHITE]);
		// MyMoveTo(r.left, r.top + 1);
		// MyLineTo(r.right, r.top + 1);
		// RGBForeColor(&colors[BLACK]);
	}
	else
	{
		MyClipRect (saveClip); //JLM 11/25/98
		SetPortGrafPort(savePort);
	}
}

void IsMouseAboveTool(short base, short index, short x, short y)
{
	short xSide = TOOLSIDE, ySide = TOOLSIDE;
	Point p;
	
	///{ JLM 7/27/98
	if(model->GetModelMode() <  ADVANCEDMODE
		&& IsLeftSideButton(base,index)) 
		return; // don't use tools on left in Novice mode
	////////}
	p.h = x;
	p.v = y;
	if (PtInPlayerRect(p) && UserIsInEditMode())
		return; // we are using the player rect as a status indicator

	if (base == PLAYERPICTBASE)
		xSide = ySide = PLAYERSIDE;
	
	if (base == BUTTONPICTBASE && index >= UPBUTTON) {
		xSide = UPDOWNWIDTH;
		ySide = UPDOWNHEIGHT;
	}
	
	if (sharedPoint.h > x &&
		sharedPoint.h < (x + xSide) &&
		sharedPoint.v > y &&
		sharedPoint.v < (y + ySide))
		{ sharedOverBase = base; sharedOverIndex = index; }
}

Boolean ShowToolInstructions(Point p)
{
	short toolShown, stringID = 168;
	static short lastToolShown = -1;
	Rect pRect = PlayerRect();
	
#ifdef IBM
	if (inMenu || FrontWindow() != mapWindow) return FALSE;
#endif
	
	sharedOverBase = -1;
	sharedOverIndex = -1;
	sharedPoint = p;
	ForEachTool(IsMouseAboveTool);
	toolShown = sharedOverBase + sharedOverIndex;
	
	if (!settings.inBackground && toolShown > 0) {
		if (toolShown != lastToolShown) {
			DisplayMessage("NEXTMESSAGETEMP");
			switch (sharedOverBase) {
				case BUTTONPICTBASE:
					if (sharedOverIndex >= UPBUTTON)
						stringID = 189 + (sharedOverIndex - UPBUTTON);
					else
						stringID = 168 + sharedOverIndex;
					break;
				case TOOLPICTBASE: stringID = 173 + sharedOverIndex;break;
				case PLAYERPICTBASE: stringID = 184 + (sharedOverIndex - RESETBUTTON); break;
			}
			DisplayMessage(GS1(stringID, 0)); // instructions for using tool
			lastToolShown = toolShown;
		}
		
		return TRUE;
	}
if (gDispersedOilVersion)
{
	// show elapsed time if over run bar but not on a button
	//if (!settings.inBackground && toolShown<=0 && MyPtInRect(p, &PlayerRect()))
	if (!settings.inBackground && toolShown<=0 && MyPtInRect(p, &pRect))
	{
		long i, n;
		TLEList *thisLEList;
		OSErr err = 0;
		ClassID thisClassID;
		Boolean bSomeSpillIsDispersed = false, bSomeSpillHasStarted = false;
		Seconds currentTime = model->GetModelTime();
		char msg[256];
		float time1,time2;
	
		for (i = 0, n = model->LESetsList->GetItemCount() ; i < n ; i++) {
			model->LESetsList->GetListItem((Ptr)&thisLEList, i);
			if(thisLEList -> GetLEType() == UNCERTAINTY_LE ) continue;
			thisClassID = thisLEList -> GetClassID();
			if(thisClassID == TYPE_OSSMLELIST || thisClassID == TYPE_SPRAYLELIST )
			{
				TOLEList *thisOLEList = (TOLEList*)thisLEList; // typecast
				Seconds thisStartTime = thisOLEList->fSetSummary.startRelTime;
				if (currentTime>thisStartTime) 
				{
					time1 = (currentTime - thisStartTime)/3600.;
					bSomeSpillHasStarted = true;
				}
				else continue;
				if (thisOLEList->fDispersantData.bDisperseOil || thisOLEList->fAdiosDataH) 
				{
					if (thisOLEList->fDispersantData.timeToDisperse/3600. < time1)
					{
						bSomeSpillIsDispersed = true;
						time2 = time1 - thisOLEList->fDispersantData.timeToDisperse/3600.;
					}
				}
			}
		}
		
		if(bSomeSpillHasStarted) 
		{
			char time1Str[64],time2Str[64];
			StringWithoutTrailingZeros(time1Str,time1,2);
			if (bSomeSpillIsDispersed)
			{
				StringWithoutTrailingZeros(time2Str,time2,2);
				sprintf(msg,"Elapsed time = %s hrs, Dispersed %s hrs ago",time1Str,time2Str);
			}
			else
				sprintf(msg,"Elapsed time = %s hrs", time1Str);
			DisplayMessage("NEXTMESSAGETEMP");
			DisplayMessage(msg); 
			lastToolShown = -1;
			return true;
		}
			
	}
}
	if (lastToolShown != -1) {
		DisplayMessage(0);
		lastToolShown = -1;
	}
	
	return FALSE;
}

void InvalToolBarRect(void)
{
	InvalRectInWindow(ToolBarRect(), ToolBarWindow());
}

OSErr EnforceToolBehaviorOnToolSelection(long newTool)
{	// returns the current tool selectin if the tool selection is being rejected
	
	long numEditableSplotObjects = model -> NumEditableSplotObjects();
	long numMaps = 0;
	OSErr err = 0;
	
	if(!model) return 0;
	if(!model -> mapList)  return 0;
	
	numMaps = model -> mapList -> GetItemCount();
	
	if(newTool == settings.currentTool)
		return 0; // we have already have that tool

	if(newTool == SPILLTOOL) {
		if (numMaps <= 0) {
			printError("You cannot create a spill because there is no map.");
			return -1;// refuse the new tool
		}
	}
	
	// can't select eraser tool if there are no splots to edit
	if(newTool == ERASETOOL) {
		if(numEditableSplotObjects <= 0) {
			printError("The are no spray can splot sets to edit.  Select the spray can tool to create a spray can splot set.");
			return -1; // refuse the new tool
		}
		else
			model -> SelectAnEditableSplotObject(); // be nice to the user
	}
	
	// first section of of spray can makes a new Splot set
	if(newTool == SPRAYCANTOOL) {
		if(numEditableSplotObjects <= 0 || !(model->EditableSplotObjectIsSelected())) {	// if user has selected a spill edit that one, else create a new one
			// create a spray can splot set
			long saveCurrentTool = settings.currentTool;
			settings.currentTool = newTool; // to prevent the IBM from redrawing with the old tool (when we put up the next dialog)
			err = CreateSprayLESet();
			if(err) {
				settings.currentTool = saveCurrentTool;
				return err;
			}
			
			SprayCanSettingsDialog();// first time put up the spray can size dialog too
		}
		else
			// need to check if user wants to start a new spill
				model -> SelectAnEditableSplotObject(); // be nice to the user
	}
	
	
	// if the user is selecting the spray can or the erase and these is only one editable splot set
	// it would be nice to select that set automatically for the user
	
	
	return 0; // accept the tool change
}


void SetTool(long tool)
{
	OSErr err = 0;
	long prevTool = settings.currentTool;

	if (0 <= tool && tool < NUMTOOLS)
	{

		// draw the 2 tools
		if (mapWindow) {
			if (tool != prevTool && 0 <= prevTool && prevTool < NUMTOOLS)	
				DrawTools(TOOLPICTBASE, prevTool); // i.e. previous tool is unselected
			DrawTools(SELECTEDTOOLPICTBASE, tool); // i.e. selected
		}
		
		if(tool != prevTool)
		{
			err = EnforceToolBehaviorOnToolSelection(tool);
			if (err) {
				DrawTools(0,0);// redraw the tools to the correct highlighting
				return;
			}
			
			settings.currentTool = tool;
		}
		
		
		// tell objects about the new tool
		if (tool != prevTool)	
		{	// new tool has being selected
			char toolNumStr[32];
			
			// code goes here, M_NEWTOOLSELECTED could now be eliminated because of CheckEditModeChange 

			sprintf (toolNumStr, "PREVIOUSTOOL\t%ld", prevTool);
			if (model)
				err = model->BroadcastMessage(M_NEWTOOLSELECTED, "MODEL", toolNumStr, 0);
		}
	}
}


Boolean ClickOnTool(Point p, Boolean doubleClick)
{
	Boolean overTool, depressed = FALSE;
	short toolClicked, base, index, dX, dY;
	Rect toolBar, splitBar, dragBar;
	RgnHandle dragRgn;
	static short lastToolClicked = -1;
	Rect saveClip;
	GrafPtr savePort;
	Rect triRect = TimeBarTriangleRect();
	Boolean firstTimeThoughLoop;
	long prevTool,newTool;
	
	if (PtInPlayerRect(p) && UserIsInEditMode())
		return FALSE; // we are using the player rect as a status indicator

	if(MyPtInRect (p, &triRect))
	{
		WindowPtr	toolWindow;
		Rect wPortRect;
		Seconds endTime   		= model -> GetEndTime ();
		Seconds startTime 		= model -> GetStartTime ();
		Seconds lastComputeTime = model -> GetLastComputeTime ();
		Seconds newTime,saveModelTime,computeTime;
		Seconds newTimeLastTimeThruLoop = 0;
		Rect movementRectForTriangle, timeLineRect = TimeLineRect();
		Rect playerRect = PlayerRect();
		short extraHeight = triRect.bottom - triRect.top;// so they can be a little sloppy
		Point	localPt, lastPt;
		float	ratio = 0;
		Seconds originalModelTime = model -> GetModelTime ();

		movementRectForTriangle.left = playerRect.left;//timeLineRect.left - extraHeight;
		movementRectForTriangle.right = playerRect.right;//timeLineRect.right + extraHeight;
		movementRectForTriangle.top = triRect.top - extraHeight;
		movementRectForTriangle.bottom = triRect.bottom + extraHeight;
		
		GetPortGrafPtr (&savePort);
		//SetPort (mapWindow);
		toolWindow = ToolBarWindow ();
		SetPortWindowPort(toolWindow);
		wPortRect = GetWindowPortRect (toolWindow);
		saveClip = MyClipRect (wPortRect);
		
		
		SetPt (&lastPt, -10, -10);
		model->SuppressDirt(DIRTY_EVERYTHING);
		while (StillDown ())
		{
			GetMouse (&localPt);
			if (localPt != lastPt)
			{					
				lastPt = localPt;
				
				if(!MyPtInRect(localPt,&movementRectForTriangle))
				{	// user has dragged the mouse ouside of the progressBar, 
					// they are trying to cancel their movement
					DrawTimeBarPartAtTime(TRIANGLE,originalModelTime);
					newTime = originalModelTime;
				}
				else 
				{	// find the time the user is indicating from the mouse position
					ratio = DrawTimeBarTriangle (localPt.h, true);// true, erase old triangle
					newTime = startTime + (long) ((float) (endTime - startTime) * ratio);
					// round to nearest multiple of computational time step
					computeTime = model->GetTimeStep();
					newTime = startTime + computeTime * (long) round((endTime - startTime) * ratio/computeTime);
				}
				
				if(newTime != newTimeLastTimeThruLoop) // check to see if we have changed the time to be shown
				{
					newTimeLastTimeThruLoop = newTime; // record the change
					if (newTime <= lastComputeTime && !(model->bHindcast))
					{
						model->SetModelToPastTime (newTime);
						DrawTimeString(model->GetModelTime()); 
					}
					else if (newTime >= lastComputeTime && model->bHindcast)
					{	
						model->SetModelToPastTime (newTime);
						DrawTimeString(model->GetModelTime()); 
					}
					else
					{	// we will let the user set the runTill time to a future time.
						model->TemporarilyShowFutureTime (newTime);
						DrawTimeString(newTime); 
					}
				}
			}
		}
		model->SuppressDirt(0);
		(void)MyClipRect(saveClip);
		SetPortGrafPort (savePort);
		
		
		// we suppressed the dirt
		// so now we have to update those parts which have not been updated
		// that would include the left hand list and redrawing the run bar buttons
		if(originalModelTime != model -> GetModelTime())
		{
			model->NewDirtNotification(DIRTY_LIST);
			DisplayCurrentTime(true); // makes sure time is right and redraws the run bar buttons
		}
		else
		{
			// JLM 3/12/99
			// In cases where the user has loaded a save file
			// It is possible there are no past saved times.
			// In this case the triangle can be left somewhere weird along the
			// runbar even though the originalModelTime is equal to model -> GetModelTime()
			DisplayCurrentTime(true); // makes sure time is right and redraws the run bar buttons
		}
		
		// run forward until desired time
		if (newTime > lastComputeTime)
			model -> RunTill (newTime);
		
		return false;
	}
	
	splitBar = SplitBarRect();
	if (MyPtInRect(p, &splitBar)) {
		dragBar = splitBar;
		MyOffsetRect (&dragBar, -1, 0);
		dragBar.right -= 1;
		dragBar.top -= 1;
		dragRgn = NewRgn();		
		dragRgn = MyRectRgn(dragRgn, &dragBar);
		toolBar = ToolBarRect();
		toolBar.right -= 20 + RIGHTBARWIDTH;
		MyDragGrayRgn(dragRgn, p, nil, &dX, &dY, &splitBar, &toolBar);
		toolBar.right += 20 + RIGHTBARWIDTH;
		DisposeRgn(dragRgn);
		
		if (dX) {
			settings.listWidth += dX;
			ResizeChildWindows();
			InvalRectInWindow(toolBar, ToolBarWindow());
			ChangeCurrentView(settings.currentView, FALSE, FV_SAMESCALE);
		}
		
		return FALSE;
	}
	
	sharedOverBase = -1;
	sharedOverIndex = -1;
	sharedPoint = p;
	ForEachTool(IsMouseAboveTool);
	base = sharedOverBase;
	index = sharedOverIndex;
	
	
	//CHANGED JLE 5/21/98
	// tool id's may not be in contiguous order
	// toolClicked = sharedOverIndex; --- change this to
	//toolClicked = sharedOverIndex != -1 ? tools[sharedOverIndex] : -1;
	
	//CHANGED JLE 6/5/98 change back
	// changed close hand id
	toolClicked = sharedOverIndex;
	
	if (toolClicked >= 0) {
//		ClearPendingMHITs();	// IAC task removed for classossm
		ShowToolInstructions(GetMouseLocal(ToolBarWindow()));
		
		if (base == BUTTONPICTBASE && !OSSMButtonEnabled(index)) return TRUE;
		if (base == PLAYERPICTBASE && !OSSMButtonEnabled(index)) return TRUE;
		
		/// JLM 1/11/99
		// quick clicks tend to not get noticed
		// a quick click on a tool should count as clicking the tool
		// add firstTime check to make this work 
	
		//while (Button()) 
		firstTimeThoughLoop = true;
		while (StillDown() || firstTimeThoughLoop) 
		{
			sharedOverBase = -1;
			sharedOverIndex = -1;
			/////
			//sharedPoint = GetMouseLocal(ToolBarWindow());
			if(!firstTimeThoughLoop) sharedPoint = GetMouseLocal(ToolBarWindow());
			firstTimeThoughLoop = false;
			/////
			ForEachTool(IsMouseAboveTool);
			overTool = sharedOverBase == base && sharedOverIndex == index;
			if (overTool != depressed) {
			
				depressed = overTool;
				switch (base) {
					case BUTTONPICTBASE:
						if (depressed)
							DrawTools(DEPRESSEDBUTTONPICTBASE, index);
						else
							DrawTools(BUTTONPICTBASE, index);
						break;
					case TOOLPICTBASE:
						if (depressed)
							DrawTools(DEPRESSEDTOOLPICTBASE, index);
						else
							DrawTools(index == settings.currentTool ?
										SELECTEDTOOLPICTBASE : TOOLPICTBASE, index);
						break;
					case PLAYERPICTBASE:
						if (depressed)
							DrawTools(DEPRESSEDPLAYERPICTBASE, index);
						else
							DrawTools(PLAYERPICTBASE, index);
						break;
				}
			}
		}
		if (!depressed) return TRUE; // the user did not select a new tool
		
		switch (base) {
			case BUTTONPICTBASE:
			{
				Boolean dontDoIt = UserIsInEditMode();
				DrawTools(BUTTONPICTBASE, index);
				switch (index) {
					case OPENBUTTON: 
						if (dontDoIt) goto StopUser;
						OpenSaveFile(); break;
					case SAVEBUTTON: 
						if (dontDoIt) goto StopUser;
						SaveAs(); break;
					case UPBUTTON: 
						if (dontDoIt) goto StopUser;
						UpItem(); break;
					case DOWNBUTTON: 
						if (dontDoIt) goto StopUser;
						DownItem(); break;
					case ADDBUTTON: 
						if (dontDoIt) goto StopUser;
						AddItem(); break;
					case SETTINGSBUTTON: 
						SettingsItem(); break;
					case DELETEBUTTON: 
						DeleteItem(); break;
				}
				break;
			}
			case TOOLPICTBASE:
				SetTool(toolClicked);
				if (toolClicked == lastToolClicked && doubleClick)
					ToolPreferencesDialog();
				lastToolClicked = toolClicked;
				break;
			case PLAYERPICTBASE:
				DrawTools(PLAYERPICTBASE, index);
				switch (index) {
					case RESETBUTTON: model->Reset(); break;
					case PLAYBUTTON: 
					{
						if (model->bHindcast)
							model->Run(model->GetStartTime()); 
						else
							model->Run(model->GetEndTime());
						break;
					}
					case STEPBUTTON: 
					{
						if (model->bHindcast)
							model->StepBackwards(); 
						else
							model->Step(); 
						break;
					}
				}
				break;
		}
	}
	
	return toolClicked >= 0;

StopUser:
	printError("You must finish editing the splots before using this button.  Choose the arrow tool when you are done editing the splots.");
	return toolClicked >= 0;

}

Boolean ClickedPause()
{
	Point p;
	Boolean overButton;
	Boolean  hadMouseEvent;
	Boolean useOldCode = false;
	Boolean overPauseBtn = false;
	
	// JLM 3/11/99
	// there was an annoying feature where 
	// when the user clicked pause and then moved the mouse from over the button 
	// it did not register as a pause request.
	//
	// I.e. we should use the point the user clicked not the current mouse location
	// HOWEVER, there seems to be a bug on the IBM where after the user initiates a 
	// a save file series, the mouse location comes back with a really large y coordinate
	// As a work around , on the IBM we check to see if the mouse is over the pause button when we 
	// see that there has been a click
	/////////////////////////////
	if(useOldCode)
	{
		hadMouseEvent = MouseDownEvent();
		p = GetMouseLocal(ToolBarWindow());
	}
	else
	{ // new improved code
		#ifdef MAC
			EventRecord event;
			hadMouseEvent = GetNextEvent(mDownMask, &event);
			if(hadMouseEvent){
				p = event.where;
				GlobalToLocal(&p);
			}
		#else
			MSG message;
			hadMouseEvent = PeekMessage(&message, toolWnd, WM_LBUTTONDOWN, WM_LBUTTONDOWN, PM_REMOVE);
			if(hadMouseEvent) {
				POINT P = message.pt;
				ScreenToClient(toolWnd, &P);
				MakeMacPoint(&P, &p);
			}
		#endif
	}
	
	/////////////////////////////
	
	if (hadMouseEvent) {
		sharedOverBase = -1;
		sharedOverIndex = -1;
		sharedPoint = p;
		ForEachTool(IsMouseAboveTool);
		
		overPauseBtn = ( sharedOverBase == PLAYERPICTBASE && (sharedOverIndex == PAUSEBUTTON ||
													sharedOverIndex == RESETBUTTON));	
	#ifdef IBM
		// bug work around JLM 3/12/99
		if(!overPauseBtn) {
			// check the current mouse position as well
			sharedPoint = GetMouseLocal(ToolBarWindow());
			ForEachTool(IsMouseAboveTool);
			overPauseBtn = ( sharedOverBase == PLAYERPICTBASE && (sharedOverIndex == PAUSEBUTTON ||
														sharedOverIndex == RESETBUTTON));	
		}
	#endif
	
		return overPauseBtn;
	}
	
	return FALSE;
}


OSErr DeleteList()
{
	/*if (objects.port)*/ VLDispose(&objects);
	
	return 0;
}

void ShowCoordinates(Point p)
{
	char s[256], latS[20], longS[20],uStr[32],sStr[32];
	long i, j, m, n;
	double lengthU, lengthS;
	WorldPoint3D wp = {0,0,0.};
	VelocityRec velocity = {0.,0.};
	TMap *map;
	TMover *mover;
	OSErr err = 0;
	
	wp.p = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
#ifdef IBM
	if (inMenu || FrontWindow() != mapWindow) return;
#endif
	
	if(settings.inBackground) return; //JLM
	
	WorldPointToStrings(wp.p, latS, longS);
	sprintf(s, "%s  %s", latS, longS);
	
	if(model->GetModelMode() ==  ADVANCEDMODE) 
	{	// JLM, add a diagnostic message for CJ
		char diagnosticStr[256] = "";
		model->MovementString(wp, diagnosticStr);

		// On Land diagnostic
		if(ControlKeyDown()) //JLM
		{
			for (j = 0, m = model -> GetMapCount (); j < m; j++) 
			{
				model -> mapList -> GetListItem ((Ptr) &map, j);
				if (!map->InMap(wp.p)) continue;
				if(map->OnLand(wp.p)) strcat(s," [on land]");
				else strcat(s," [on water]");
			}
		}
		// extended to handle all current movers 2/25/03
		if(!diagnosticStr[0])
		{ // original CATS mover diagnostic code
			for (j = 0, m = model -> GetMapCount (); j < m; j++) 
			{
				model -> mapList -> GetListItem ((Ptr) &map, j);
				if (!map->InMap(wp.p)) continue;
				
				// CATS-mover diagnostic, added Component mover 9/28/01
				for (i = 0, n = map->moverList->GetItemCount() ; i < n ; i++) 
				{
					map->moverList->GetListItem((Ptr)&mover, i);
					if (!mover->bActive) continue;
					//if (mover->GetClassID() == TYPE_NETCDFMOVERTRI) continue; // need to set up function for this, else goes to base class
					// show first current mover on first map that has a current mover only
					if (mover->VelocityStrAtPoint(wp, diagnosticStr)) 
					{
						strcat(s,diagnosticStr);
						DisplayMessage(s);
						return;
					}

				}
			}
		}

		strcat(s,diagnosticStr);
	}
	
	DisplayMessage(s);
}

void OSSMEnableMenuItem(MenuHandle m, short item, Boolean enable)
{
	if(!m) return;
	#ifdef MAC
		item = item%100; // remove the encoded menu part
		if (item>50) item = item%50; // remove the encoded menu part
	#endif
	MyEnableMenuItem(m,item,enable);
}


void CommonSetCursor(void) //JLM
{
	Point mousePt;
	WorldPoint	WStartPoint;
	GrafPtr oldPort;
	
	
	switch (settings.currentTool) {
		case ARROWTOOL: 
			InitCursor();
			break;
		case ZOOMPLUSTOOL: 
			if (OptionKeyDown()) MySetCursor(1000 + ZOOMMINUSTOOL );
			else MySetCursor(1000 + ZOOMPLUSTOOL );
			break;
		case ZOOMMINUSTOOL:
			if (OptionKeyDown()) MySetCursor(1000 + ZOOMPLUSTOOL );
			else MySetCursor(1000 + ZOOMMINUSTOOL );
			break;
		case HANDTOOL:	
			MySetCursor(1000 + HANDTOOL);
			break;
		case RULERTOOL:	
			MySetCursor(1005);
			break;
		case SPRAYCANTOOL:
		{
			if(CanSpray(nil))
			{
				switch(settings.sprayCanSize) 
				{
					case SPRAY_SIZE_SMALL:
						MySetCursor(203);
						break;
					case SPRAY_SIZE_MEDIUM:
						MySetCursor(205);
						break;
					case SPRAY_SIZE_LARGE:
						MySetCursor(207);
						break;
				}
			}
			else 
				MySetCursor( 1008 ); // spill not allowed
				
			break;
		}
		case ERASETOOL:
			MySetCursor (307);
			break;
		case SPILLTOOL:
		{
			GetPortGrafPtr(&oldPort);
			SetPortWindowPort(mapWindow);
			GetMouse(&mousePt);
			SetPortGrafPort(oldPort);
			WStartPoint = ScreenToWorldPoint(mousePt, MapDrawingRect(), settings.currentView);
			if(model->IsAllowableSpillPoint(WStartPoint) || (gNoaaVersion && model->bHindcast)) MySetCursor( 1005 ); // spill allowed
			else MySetCursor( 1008 ); // spill not allowed
			break;
		}
	}
}

void DoTasks()
{
	Boolean itemSelected, messageGiven = FALSE;
	char s[256];
	long n = 0;
	short	style = normal;
	Point p;
	Rect r = MapDrawingRect();
	GrafPtr oldPort;
	ClassID id;
	ListItem item;
	MenuHandle mh;
	static long sPreviousStatus = FALSE;
	long status;
#ifdef MAC
	short tool;
	Rect r2;
	Rect r3;
#else
	char hcMessage[20];
	extern long _handleCount;
#endif

	CDOGTasks();
	
	if(WeHaveAModalDialogUp()) return; // JLM 10/15/99, don't be doing tasks on the IBM when a dialog is up
	if (gInWizard) return; // don't do taks until we are out of the wizard
	
	TellPlayersAboutNewDirt(); //JLM 10/20/98
	ValidateListLength(); // JLM
	TellPlayersAboutNewListSelection(); 
	model -> CheckEditModeChange();

	
	////////////
	// monitor the splot editing mode
	// since they can't use the runbar while editing
	// use the runbar as a user feedback area
	status = EditStatus();
	if ( sPreviousStatus != status)
	{
		sPreviousStatus = status;
		InvalRectInWindow(PlayerRect(),ToolBarWindow());
	}
	///////////
	
	GetPortGrafPtr(&oldPort);
	SetPortWindowPort(mapWindow);
	GetMouse(&p);
#ifndef IBM ///////
	{	// MAC code to handle "windowshade"
		// where mac users can double click the title bar to collapse the window to have no content
		// in this case the port rect is still a big rect, so we use FindWindow
		WindowPtr whichWindow;
		short partCode;
		Boolean overMapWindow;
		Point globalCoordP = p;
		LocalToGlobal(&globalCoordP);
		partCode = FindWindow(globalCoordP, &whichWindow);
		overMapWindow = (whichWindow == mapWindow && partCode == inContent);
		if(!overMapWindow)
			SetPt(&p,-32000,-32000); // so that we know the mouse is not over our window
	}
#endif ///////
	SetPortGrafPort(oldPort);
	
	settings.doNotPrintError = FALSE;
	ExitReadOnly(0);
	
#ifdef MAC
{
	if (settings.currentTool == 0 
		|| !MyPtInRect(p, &r) // not in map drawing rect
		) {
		Rect triRect = TimeBarTriangleRect(); // JLM 1/21/99
		r2 = SplitBarRect();
		if (MyPtInRect(p, &r2) || (MyPtInRect(p, &triRect) && !UserIsInEditMode()))
			MySetCursor(STRETCHLISTCURSOR);
		else
			InitCursor();
	}
	else {
		CommonSetCursor();
	}
}
#else
	if (GetActiveWindow() == mapWindow && GetFocus() != mapWindow)
		SetFocus(mapWindow);
#endif
	
	if (itemSelected = VLGetSelect(&n, &objects)) {
		item = model->GetNthListItem(n, 0, &style, s);
		if (!item.owner) itemSelected = FALSE;
	}
	
	mh = GetMenuHandle(FILEMENU); // JLM
	if(mh)
	{
		if(model->GetModelMode() <  ADVANCEDMODE)
		{
			Boolean haveOpenWizFile = model->fWizard->HaveOpenWizardFile();
			OSSMEnableMenuItem(mh, CLOSEITEM, haveOpenWizFile);
			OSSMEnableMenuItem(mh, SAVEASITEM, haveOpenWizFile);
		}
		else
		{
			OSSMEnableMenuItem(mh, CLOSEITEM, TRUE);
			OSSMEnableMenuItem(mh, SAVEASITEM, TRUE);
		}
	}
	
	mh = GetMenuHandle(EDITMENU);
	if(mh)
	{
		OSSMEnableMenuItem(mh, UNDOITEM, settings.undoCode != UNDO_NOTHING);
		OSSMEnableMenuItem(mh, CUTITEM, FALSE);
		OSSMEnableMenuItem(mh, COPYITEM, itemSelected);
		OSSMEnableMenuItem(mh, PASTEITEM, FALSE);
	}
	
	mh = GetMenuHandle(ITEMMENU);
	if(mh)
	{
		Boolean flag ;
		flag = OSSMButtonEnabled(ADDBUTTON);
		flag = itemSelected && item.owner->FunctionEnabled(item, ADDBUTTON);

		OSSMEnableMenuItem(mh, WIZARDITEM, true); // always on	

		OSSMEnableMenuItem(mh, ADDITEM, itemSelected && item.owner->FunctionEnabled(item, ADDBUTTON));
		OSSMEnableMenuItem(mh, SETTINGSITEM, itemSelected && item.owner->FunctionEnabled(item, SETTINGSBUTTON));
		OSSMEnableMenuItem(mh, DELETEITEM, itemSelected && item.owner->FunctionEnabled(item, DELETEBUTTON));
		OSSMEnableMenuItem(mh, UPITEM, itemSelected && item.owner->FunctionEnabled(item, UPBUTTON));
		OSSMEnableMenuItem(mh, DOWNITEM, itemSelected && item.owner->FunctionEnabled(item, DOWNBUTTON));
	}

	mh = GetMenuHandle(RUNMENU);
	if(mh)
	{
		OSSMEnableMenuItem(mh, RESETITEM, OSSMButtonEnabled(RESETBUTTON));
		OSSMEnableMenuItem(mh, RUNITEM, OSSMButtonEnabled(PLAYBUTTON));
		
//		OSSMEnableMenuItem(mh, RUNTILLITEM, OSSMButtonEnabled(PLAYBUTTON)); 
// should no longer be disabled since they can go back using this menu too, STH

		OSSMEnableMenuItem(mh, STEPITEM, OSSMButtonEnabled(STEPBUTTON));
	}

	// NOTE: CDOG menusitems are handled in CDOGTasks()
	
	//hide until 3D current file is opened...
	mh = GetMenuHandle(ANALYSISMENU);
	if(mh)
	{
		float arrowDepth;
		long mode = model->GetModelMode();
		PtCurMap *map = GetPtCurMap();	
		Boolean enableOutputItems = (mode == ADVANCEDMODE && (model->ThereIsA3DMover(&arrowDepth)));
		Boolean enableOutputItems2 = (mode == ADVANCEDMODE && (map && map->ThereIsADispersedSpill()));
		Boolean enableOutputItems3 = (mode == ADVANCEDMODE && (map && map->ThereAreTrianglesSelected()));
		OSSMEnableMenuItem(mh, SHOWGRIDITEM, enableOutputItems);
		if (!enableOutputItems) SetShowGrid(false);
		OSSMEnableMenuItem(mh, SHOWSELECTEDTRIITEM, enableOutputItems3);
		if (!enableOutputItems3) SetShowSelectedTriangles(false);
		OSSMEnableMenuItem(mh, DESELECTITEM, enableOutputItems3);
		OSSMEnableMenuItem(mh, DISPERSEOILITEM, enableOutputItems);
		OSSMEnableMenuItem(mh, SETCONTOURSITEM, enableOutputItems2);
		OSSMEnableMenuItem(mh, CONCATPTITEM, enableOutputItems2);
		OSSMEnableMenuItem(mh, CONCATPLUMEITEM, enableOutputItems2);
		OSSMEnableMenuItem(mh, SHOWPLOTSITEM, enableOutputItems2);
		OSSMEnableMenuItem(mh, DEPTHCONTOURSITEM, enableOutputItems);
		OSSMEnableMenuItem(mh, SHOWCONTOURSITEM, enableOutputItems);
		OSSMEnableMenuItem(mh, SHOWCONTOURLABELSITEM, enableOutputItems);
		OSSMEnableMenuItem(mh, SMOOTHINGPARAMETERSITEM, enableOutputItems2);
		OSSMEnableMenuItem(mh, SCALEDEPTHSITEM, enableOutputItems);
		OSSMEnableMenuItem(mh, CALCDOSAGEITEM, enableOutputItems2);
		OSSMEnableMenuItem(mh, SHOWDOSAGEITEM, enableOutputItems2);
		OSSMEnableMenuItem(mh, SHOWMAXTRIITEM, enableOutputItems2);
		OSSMEnableMenuItem(mh, SUPPRESSDRAWINGITEM, enableOutputItems);
		OSSMEnableMenuItem(mh, SETSELECTEDBEACH, enableOutputItems);
		OSSMEnableMenuItem(mh, CLEARSELECTEDBEACH, enableOutputItems);
		OSSMEnableMenuItem(mh, CONCENTRATIONONBEACH, enableOutputItems);	// here should require a spill but not dispersed

	}

	if (!ShowToolInstructions(p)) {
		if (MyPtInRect(p, &r))
			ShowCoordinates(p);
		else {
			if (itemSelected) {
				id = item.owner->GetClassID();
				id = (id / 100) * 100;
				// double clicking maps now give the map setting dialog
				//if (id == TYPE_MAP && item.index == I_MAPNAME) {
				//	DisplayMessage("NEXTMESSAGETEMP");
				//	DisplayMessage("Double-click to view map area.");
				//	messageGiven = TRUE;
				//}
				if (id == TYPE_MODEL && item.index == I_LESETS) {
					DisplayMessage("NEXTMESSAGETEMP");
					//DisplayMessage("Double-click to view area of all LE lists.");
					DisplayMessage("Double-click to add a spill (or use spill tool).");
					messageGiven = TRUE;
				}
				if (id == TYPE_MODEL && item.index == I_MAPS) {
					DisplayMessage("NEXTMESSAGETEMP");
					//DisplayMessage("Double-click to view area of all maps.");
					DisplayMessage("Double-click to load a new map.");
					messageGiven = TRUE;
				}

				if (id == TYPE_MODEL && item.index == I_UMOVERS) {
					DisplayMessage("NEXTMESSAGETEMP");
					//DisplayMessage("Double-click to view area of all universal movers.");
					DisplayMessage("Double-click to add a universal mover.");
					messageGiven = TRUE;
				}
			}
			if (!messageGiven)
				DisplayMessage("");
		}
	}
	
#ifdef IBM
	sprintf(hcMessage, "%ld", _handleCount);
	// DisplayMessage(hcMessage);
#endif
}

RgnHandle ClipToDrawingRect()
{
#ifdef MAC
	Rect r = MapDrawingRect();
	RgnHandle saveClip = NewRgn();
	
	GetClip(saveClip);
	ClipRect(&r);
	
	return saveClip;
#else
	return 0;
#endif
}

void UnClip(RgnHandle saveClip)
{
#ifdef MAC
	SetClip(saveClip);
	DisposeRgn(saveClip);
#else
#endif
}

void RefreshMapWindowContents()
{
	long ticks1, ticks2;
	Rect saveClip, r = GetWindowPortRect(mapWindow);
	
	EraseRect(&r); // JLM, this causes the map to flash, it might be better to erase each part just before drawing it
#ifdef MAC
	MyDrawGrowIcon();
	DrawTools(0, 0);
	VLUpdate(&objects);
	DrawControls(mapWindow);
#endif
	
	saveClip = MyClipRect(MapDrawingRect()); // ClipToDrawingRect();
	ticks1 = TickCount();
	SetWatchCursor();
	DrawMaps(mapWindow, MapDrawingRect(), settings.currentView, FALSE);
	DrawMarkers(FALSE);
	InitCursor();
	ticks2 = TickCount();
	sprintf(mssg, GS1(133, 0), ((float)ticks2 - (float)ticks1) / 60.0); // Draw time in seconds: %f
	MyClipRect(saveClip); // UnClip(saveClip);
	
#ifdef MAC
	DisplayMessage(ControlKeyDown() ? mssg : 0);
#else
	DisplayMessage(KeyDown(VK_RBUTTON) ? mssg : 0);
#endif
}

///////////////////////////////////////////////////////////////////////////

void MagnifyTool(Point p, short tool)
{
	WorldRect newView;
	WorldPoint wp = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
	long width = WRectWidth(settings.currentView), height = WRectHeight(settings.currentView);
	Boolean checkOption = tool < 0;
	
	// JLM , zoom in clicks are not zooming in to maxZoom
	// so force the scale to be 1/2 what it was
	long originalScale = CurrentScaleDenominator();// JLM
	long mostZoomedOutScale;

	if (tool < 0) tool = -tool;
	
	if ((tool == ZOOMPLUSTOOL) ^ (checkOption && OptionKeyDown())) {
		if (width < MINWRECTDIST || height < MINWRECTDIST) { SysBeep(1); return; }
		width /= 4;
		height /= 4;
		SetWorldRect(&newView, wp.pLat - height, wp.pLong - width,
						   wp.pLat + height, wp.pLong + width);
		ChangeCurrentView(newView, TRUE, TRUE);
		SetCurrentScaleDenominator(originalScale/2);// JLM
	}
	else
	{ // zoom out
		if (width > MAXWRECTDIST || height > MAXWRECTDIST) { SysBeep(1); return; }
		
		mostZoomedOutScale = MostZoomedOutScale();
		if(originalScale >=  (long)(0.99*mostZoomedOutScale)) // JLM 12/22/97
		{ SysBeep(1); return;}

		SetWorldRect(&newView, wp.pLat - height, wp.pLong - width,
						   wp.pLat + height, wp.pLong + width);
		ChangeCurrentView(newView, TRUE, TRUE);
		SetCurrentScaleDenominator(originalScale*2);// JLM
	}
}

void ZoomRectAction(Rect *r, Boolean fromCenter)
{
	Rect r2 = MapDrawingRect();
	short sHeight = RectHeight(r2), sWidth = RectWidth(r2),
		  rHeight = abs(RectHeight(*r)), rWidth = abs(RectWidth(*r)),
		  desiredWidth, dWidth, desiredHeight, dHeight;
	float dummy,
		  sRatio = (float)sWidth / (float)sHeight,
		  rRatio =  (float)rWidth / (float)rHeight;
#pragma unused(fromCenter)
	
	dummy = sRatio; // to avoid compiler bug
	dummy = rRatio; // to avoid compiler bug
	
	if (rRatio > sRatio) { // r is flatter than screen
		desiredHeight = rWidth / sRatio, dHeight = (desiredHeight - rHeight) / 2;
		if (r->top > r->bottom) dHeight = -dHeight;
		r->top -= dHeight;
		r->bottom += dHeight;
	}
	else { // r is taller than screen
		desiredWidth = rHeight * sRatio, dWidth = (desiredWidth - rWidth) / 2;
		if (r->left > r->right) dWidth = -dWidth;
		r->left -= dWidth;
		r->right += dWidth;
	}
	
}

void DoZoomPlusTool(Point p)
{
	Point where;
	Rect r;
	WorldRect check;
	long currentScale = CurrentScaleDenominator();
	long mostZoomedInScale = MostZoomedInScale();
	
	
	r = DefineGrayRect(p, ZoomRectAction, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE);
	
	if(currentScale <= (long)(1.01*mostZoomedInScale)) // JLM 12/19/97
	{ SysBeep(1); return;}

	if (RectWidth(r) > 2 && RectHeight(r) > 2) {
		check = ScreenToWorldRect(r, MapDrawingRect(), settings.currentView);
		if (WRectWidth(check) < MINWRECTDIST || WRectHeight(check) < MINWRECTDIST) { SysBeep(1); return ; }
		if (WRectWidth(check) > MAXWRECTDIST || WRectHeight(check) > MAXWRECTDIST) { SysBeep(1); return ; }
		ChangeCurrentView(check, TRUE, TRUE);
	}
	else
		if (r.right > 1) {
			where.h = r.left;
			where.v = r.top;
			MagnifyTool(where, -ZOOMPLUSTOOL);
		}
	
	//return r;
}	

Rect DoHandTool(Point p)
{
	RgnHandle dragRgn = NewRgn();
	Rect r = MapDrawingRect();
	short dh, dv;
	WorldRect check;
	
	dragRgn = MyRectRgn(dragRgn, &r);
	//CHANGED JLE
	// SOMEDAY WE HAVE TO GET RID OF THIS  TOOLPICTBASE STUFF
	//MySetCursor(TOOLPICTBASE + CLOSEDHAND);
	
	MySetCursor(CLOSEDHAND);
	
	MyDragGrayRgn(dragRgn, p, nil, &dh, &dv, 0, 0);
	MySetCursor(TOOLPICTBASE + HANDTOOL);
	
	if (dh || dv) {
		MyOffsetRect(&r, -dh, -dv);
		check = ScreenToWorldRect(r, MapDrawingRect(), settings.currentView);
		if (WRectTouchesWRect(&check, &theWorld))
			ChangeCurrentView(check, FALSE, TRUE);
		else
			SysBeep(1);
	}
	
	DisposeRgn(dragRgn);
	
	return r;
}	


void DoSpillTool(Point startPoint)
{
	WorldPoint	WStartPoint = ScreenToWorldPoint(startPoint, MapDrawingRect(), settings.currentView);
	WorldPoint WEndPoint;
	Point endPoint,oldEndPoint=startPoint;
	Boolean		bMoved = false;
	Point MousePoint;
	Boolean haveOpenWizFile = (model->fWizard && model->fWizard->HaveOpenWizardFile());
	
	WStartPoint = RoundWorldPoint(WStartPoint);// JLM 12/9/98
	
	if(!model->IsAllowableSpillPoint(WStartPoint) && (!gNoaaVersion && !(model->bHindcast))) goto BadStartPoint; 

	// keep looping until mouse moves past tolerence-pixels or button is released
	bMoved = false;
	endPoint = startPoint;
	while (StillDown ())
	{
		GetMouse (&MousePoint);
		if (abs (MousePoint.h - startPoint.h) > 4 ||
			abs (MousePoint.v - startPoint.v) > 4)
		{
			bMoved = true;
			break;
		}
	}
	
	if(bMoved)
	{
		PenMode(patXor);
		while(StillDown())
		{
			GetMouse(&endPoint);

			if(endPoint.h != oldEndPoint.h ||  endPoint.v != oldEndPoint.v)
			{
				if(startPoint.h != oldEndPoint.h ||  startPoint.v != oldEndPoint.v)
				{// erase the old one
					MyMoveTo(startPoint.h,startPoint.v);
					MyLineTo(oldEndPoint.h,oldEndPoint.v);
				}
				MyMoveTo(startPoint.h,startPoint.v);
				MyLineTo(endPoint.h,endPoint.v);
				oldEndPoint = endPoint;
				
				///////////////{
				// JLM, 12/24/98 add cursor feedback code
				WEndPoint = ScreenToWorldPoint(endPoint, MapDrawingRect(), settings.currentView);
				WEndPoint = RoundWorldPoint(WEndPoint);
				if(model->IsAllowableSpillPoint(WEndPoint)) MySetCursor( 1005 ); // spill allowed
				else MySetCursor( 1008 ); // spill not allowed
				///////////}
			}
		}
		if(startPoint.h != oldEndPoint.h ||  startPoint.v != oldEndPoint.v)
		{	// erase the line, it will drawn later
			MyMoveTo(startPoint.h,startPoint.v);// erase the old one
			MyLineTo(oldEndPoint.h,oldEndPoint.v);
		}
		PenMode(patCopy);
	}
	
	
	if(endPoint.h != startPoint.h ||  endPoint.v != startPoint.v)
	{
		WEndPoint = ScreenToWorldPoint(endPoint, MapDrawingRect(), settings.currentView);
		WEndPoint = RoundWorldPoint(WEndPoint);// JLM 12/9/98
		
		if(!model->IsAllowableSpillPoint(WEndPoint)) goto BadEndPoint; 
		
		CreateLESet(&WStartPoint,&WEndPoint);
	}
	else CreateLESet(&WStartPoint,(WorldPoint*)nil);
	
	return;

BadStartPoint:
	if(model->IsWaterPoint(WStartPoint))
	{
		char msg[256];
		if(haveOpenWizFile)
			//printError("This Location File has not been set up for spills in this area.  The cursor appears as a plus over allowable spill locations."); 
			strcpy(msg,"This Location File has not been set up for spills in this area.  The cursor appears as a plus over allowable spill locations."); 
		else
			strcpy(msg,"This map has not been set up for spills in this area.  The cursor appears as a plus over allowable spill locations.");
			//printError("This map has not been set up for spills in this area.  The cursor appears as a plus over allowable spill locations."); 
		if (model->HaveAllowableSpillLayer(WStartPoint) && model->GetModelMode()==ADVANCEDMODE) strcat(msg," To access this area, make the Spillable Area inactive in the left hand list.");
		printError(msg);
	}
	else // on Land
		SysBeep(5); //printError("Spills can only occur in the water."; 
	return;

BadEndPoint:
	if(model->IsWaterPoint(WEndPoint))
	{
		char msg[256];
		if(haveOpenWizFile)
			strcpy(msg,"This Location File has not been set up for spills in this area.  The cursor appears as a plus over allowable spill locations."); 
			//printError("This Location File has not been set up for spills in this area.  The cursor appears as a plus over allowable spill locations."); 
		else
			strcpy(msg,"This map has not been set up for spills in this area.  The cursor appears as a plus over allowable spill locations.");
			//printError("This map has not been set up for spills in this area.  The cursor appears as a plus over allowable spill locations."); 
		if (model->HaveAllowableSpillLayer(WEndPoint) && model->GetModelMode()==ADVANCEDMODE) strcat(msg," To access this area, make the Spillable Area inactive in the left hand list.");
		printError(msg);
	}
	else // on Land
		SysBeep(5); //printError("Spills can only occur in the water."; 
	return;

	
}

Rect DoArrowTool(Point p)
{
	short dh, dv;
	long i, j, n, numWindMovers;
	Rect r = MapDrawingRect(), frame = { 0, 0, 0, 0 }, frame2 = { 0, 0, 0, 0 }, frame3 = { 0, 0, 0, 0 };
	RgnHandle dragRgn;
	//TVectorMap *map = (TVectorMap*)model->GetMap(TYPE_VECTORMAP);	// not using the ESI stuff at this point
	PtCurMap *map = GetPtCurMap();
	TWindMover *windMover = nil;
	TTriGridVel3D *triGrid = 0;
	TriCurMover *mover = (TriCurMover*)(model->GetMover(TYPE_TRICURMOVER));
	NetCDFMover *mover2 = (NetCDFMover*)(model->GetMover(TYPE_NETCDFMOVER));
	//TCurrentMover *mover = map->Get3DCurrentMover();
	//TWindMover *windMover = model->GetWindMover(false);
	//if (map==nil && windMover==nil) return r;
	numWindMovers = model->GetNumWindMovers();
	if (map==nil && numWindMovers==0) return r;
	if (map) 
	{
		if (map->IAm(TYPE_COMPOUNDMAP))
		{	// get grid and mover for priority map that includes p
			WorldPoint wp = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
			long mapIndex = ((TCompoundMap*)map)->WhichMapIsPtIn(wp);
			triGrid = ((TCompoundMap*)map)->GetGrid3DFromMapIndex(mapIndex);
			// get mover2 here if there is one from the compoundmap
		}
		else
			triGrid = map -> GetGrid3D(false);
		frame = (*map).fLegendRect;
	}
	if (triGrid) frame2 = (*triGrid).fLegendRect;
	//if (mover->IAm(TYPE_TRICURMOVER)) frame3 = ((TriCurMover*)mover)->fLegendRect;
	if (mover) frame3 = mover->fLegendRect;
	else if (mover2) frame3 = mover2->fLegendRect;
	//if (windMover) frame2 = (*windMover).fWindBarbRect;
#ifdef IBM
	// convert the frame to the map drawing rect coordinate system
	if (map) MyOffsetRect(&frame,r.left,r.top);
	if (triGrid) MyOffsetRect(&frame2,r.left,r.top);
	if (mover && mover->IAm(TYPE_TRICURMOVER)) MyOffsetRect(&frame3,r.left,r.top);
	if (mover2 && mover2->IAm(TYPE_NETCDFMOVER)) MyOffsetRect(&frame3,r.left,r.top);
	//if (windMover) MyOffsetRect(&frame2,r.left,r.top);
#endif
	if (map && MyPtInRect(p, &frame))
	{	
		dragRgn = NewRgn();
		dragRgn = MyRectRgn(dragRgn, &frame);
		MyDragGrayRgn(dragRgn, p, nil, &dh, &dv, &frame, &r);
		DisposeRgn(dragRgn);
		
		if (dh == 0 && dv == 0) return frame;
		
		MyInsetRect(&frame, -10, -10);
		InvalMapWRect(ScreenToWorldRect(frame, MapDrawingRect(), settings.currentView));
		MyOffsetRect(&frame, dh, dv);
		InvalMapWRect(ScreenToWorldRect(frame, MapDrawingRect(), settings.currentView));
		MyInsetRect(&frame, 10, 10);
		
#ifdef IBM
		MyOffsetRect(&frame,-r.left,-r.top);
#endif
		(*map).fLegendRect = frame;
		
		InvalidateMapImage();// invalidate the offscreen bitmap
		//InvalMapDrawingRect();
		return frame;
	}
	else if (triGrid && MyPtInRect(p, &frame2))
	{	
		dragRgn = NewRgn();
		dragRgn = MyRectRgn(dragRgn, &frame2);
		MyDragGrayRgn(dragRgn, p, nil, &dh, &dv, &frame2, &r);
		DisposeRgn(dragRgn);
		
		if (dh == 0 && dv == 0) return frame2;
		
		MyInsetRect(&frame2, -10, -10);
		InvalMapWRect(ScreenToWorldRect(frame2, MapDrawingRect(), settings.currentView));
		MyOffsetRect(&frame2, dh, dv);
		InvalMapWRect(ScreenToWorldRect(frame2, MapDrawingRect(), settings.currentView));
		MyInsetRect(&frame2, 10, 10);
		
#ifdef IBM
		MyOffsetRect(&frame2,-r.left,-r.top);
#endif
		(*triGrid).fLegendRect = frame2;
		
		InvalidateMapImage();// invalidate the offscreen bitmap
		//InvalMapDrawingRect();
		return frame2;
	}
	else if (mover && mover->IAm(TYPE_TRICURMOVER) && MyPtInRect(p, &frame3))
	{	
		dragRgn = NewRgn();
		dragRgn = MyRectRgn(dragRgn, &frame3);
		MyDragGrayRgn(dragRgn, p, nil, &dh, &dv, &frame3, &r);
		DisposeRgn(dragRgn);
		
		if (dh == 0 && dv == 0) return frame3;
		
		MyInsetRect(&frame3, -10, -10);
		InvalMapWRect(ScreenToWorldRect(frame3, MapDrawingRect(), settings.currentView));
		MyOffsetRect(&frame3, dh, dv);
		InvalMapWRect(ScreenToWorldRect(frame3, MapDrawingRect(), settings.currentView));
		MyInsetRect(&frame3, 10, 10);
		
#ifdef IBM
		MyOffsetRect(&frame3,-r.left,-r.top);
#endif
		((TriCurMover*)mover)->fLegendRect = frame3;
		
		InvalidateMapImage();// invalidate the offscreen bitmap
		//InvalMapDrawingRect();
		return frame3;
	}
	else if (mover2 && mover2->IAm(TYPE_NETCDFMOVER) && MyPtInRect(p, &frame3))
	{	
		dragRgn = NewRgn();
		dragRgn = MyRectRgn(dragRgn, &frame3);
		MyDragGrayRgn(dragRgn, p, nil, &dh, &dv, &frame3, &r);
		DisposeRgn(dragRgn);
		
		if (dh == 0 && dv == 0) return frame3;
		
		MyInsetRect(&frame3, -10, -10);
		InvalMapWRect(ScreenToWorldRect(frame3, MapDrawingRect(), settings.currentView));
		MyOffsetRect(&frame3, dh, dv);
		InvalMapWRect(ScreenToWorldRect(frame3, MapDrawingRect(), settings.currentView));
		MyInsetRect(&frame3, 10, 10);
		
#ifdef IBM
		MyOffsetRect(&frame3,-r.left,-r.top);
#endif
		((NetCDFMover*)mover2)->fLegendRect = frame3;
		
		InvalidateMapImage();// invalidate the offscreen bitmap
		//InvalMapDrawingRect();
		return frame3;
	}
	//else if (windMover && MyPtInRect(p, &frame2))
	else if (numWindMovers>0)
	{	// need to think about possibility of more than 1 wind mover
		for (i=0;i<numWindMovers;i++)
		{
			windMover = model->GetNthWindMover(i);
			if (windMover) frame2 = (*windMover).fWindBarbRect;
		#ifdef IBM
			// convert the frame to the map drawing rect coordinate system
			if (windMover) MyOffsetRect(&frame2,r.left,r.top);
		#endif
			if (windMover && MyPtInRect(p, &frame2))
			{
				dragRgn = NewRgn();
				dragRgn = MyRectRgn(dragRgn, &frame2);
				MyDragGrayRgn(dragRgn, p, nil, &dh, &dv, &frame2, &r);
				DisposeRgn(dragRgn);
				
				if (dh == 0 && dv == 0) return frame2;
				
				MyInsetRect(&frame2, -10, -10);
				InvalMapWRect(ScreenToWorldRect(frame2, MapDrawingRect(), settings.currentView));
				MyOffsetRect(&frame2, dh, dv);
				InvalMapWRect(ScreenToWorldRect(frame2, MapDrawingRect(), settings.currentView));
				MyInsetRect(&frame2, 10, 10);
				
		#ifdef IBM
				MyOffsetRect(&frame2,-r.left,-r.top);
		#endif
				(*windMover).fWindBarbRect = frame2;
				
				InvalidateMapImage();// invalidate the offscreen bitmap
				InvalMapDrawingRect();	// in case legend is outside map 
				return frame2;
			}
		}
	}
	if (map)
	{
		//return MapDrawingRect();
		// select triangles to do 3D concentration vs time analysis on
		// first should check whether there is a ptcurmap, 
		// eventually replace with a lasso tool
		long trinum;
		Point newPoint;
		r = MapDrawingRect();
		TTriGridVel3D* triGrid = 0;	
		TMover *mover = 0;
		Boolean drewToScreen = false, needToRefresh = false;
	
		// Figure out if this map has a PtCurMover or TCATSMover current
		// code goes here, allow for NetCDFMovers too
		// use a separate function, this is getting messy...

		if (map->IAm(TYPE_COMPOUNDMAP))
		{	// get grid and mover for priority map that includes p
			WorldPoint wp = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
			long mapIndex = ((TCompoundMap*)map)->WhichMapIsPtIn(wp);
			triGrid = ((TCompoundMap*)map)->GetGrid3DFromMapIndex(mapIndex);
			// get mover2 here if there is one from the compoundmap
		}

		mover = map->GetMover(TYPE_PTCURMOVER);
		if (mover)
		{
			if (!((PtCurMover *)mover) -> fVar.bShowGrid && !OptionKeyDown()) 
				return r;	// only select triangles if grid is showing
			triGrid = (TTriGridVel3D*)(((PtCurMover *)mover) -> fGrid);
		}
		else	
		{
			mover = map->GetMover(TYPE_CATSMOVER3D);
			if (mover)
			{
				if (!((TCATSMover3D *)mover) -> bShowGrid && !OptionKeyDown()) return r; // only select triangles if grid is showing
				if (((TCATSMover3D *)mover) -> fRefinedGrid)
					triGrid = ((TCATSMover3D *)mover) -> fRefinedGrid;
				else
					triGrid = (TTriGridVel3D*)(((TCATSMover3D *)mover) -> fGrid);
			}
			else
			{
				mover = map->GetMover(TYPE_TRICURMOVER);
				if (mover)
				{
					if (!((TriCurMover *)mover) -> fVar.bShowGrid && !OptionKeyDown()) 
						return r;	// only select triangles if grid is showing
					triGrid = (TTriGridVel3D*)(((TriCurMover *)mover) -> fGrid);
				}
				else
				{
					mover = map->GetMover(TYPE_NETCDFMOVER);
					if (mover)
					{
						if (!((NetCDFMover *)mover) -> fVar.bShowGrid && !OptionKeyDown()) 
							return r;	// only select triangles if grid is showing
						if (!(((NetCDFMover *)mover) -> fNumDepthLevels > 1))
							return r;
						triGrid = (TTriGridVel3D*)(((NetCDFMover *)mover) -> fGrid);
					}
				}
			}
		}
	
		if (!triGrid) return r; 
	
		if (OptionKeyDown())
		{
			trinum = triGrid -> FindTriNearClick(p);
			return map->DoArrowTool(trinum);
		}
		if(!ShiftKeyDown()) {
			if (triGrid -> ThereAreTrianglesSelected()) needToRefresh = true;
			drewToScreen = true;
			triGrid -> ClearTriSelection();	// delete the handle
			//triGrid -> DeselectAll();
		}

		Boolean **triSelected = triGrid -> GetTriSelection(true);	// initialize if necessary
		if (!triSelected) return r;
	
		if((trinum = triGrid -> FindTriNearClick(p)) > -1)
		{	
			drewToScreen = true;
#ifdef MAC
			triGrid -> DrawTriangle3D(&r,trinum,FALSE,TRUE); 
#endif
			triGrid -> ToggleTriSelection(trinum);		
			//TriDisplayInfoStr(trinum,infostr);	// may want to display something	
			//DisplayMessage(infostr);
			while(StillDown()) 
			{
				GetMouse(&newPoint);
				if((newPoint.h != p.h || newPoint.v != p.v) && (trinum = triGrid -> FindTriNearClick(newPoint)) > -1)
				{
					p = newPoint;
					if(!(*triSelected)[trinum])
					{
#ifdef MAC
						triGrid -> DrawTriangle3D(&r,trinum,FALSE,TRUE);
#endif
						triGrid -> ToggleTriSelection(trinum);		
						//TriDisplayInfoStr(trinum,infostr);	
						//DisplayMessage(infostr);
					}
				}
			}
		}
		else
		{
			triGrid -> ClearTriSelection();
			drewToScreen = true;
		}
		
		if(drewToScreen) {
			model->NewDirtNotification(DIRTY_LIST);
			InvalidateMapImage();// invalidate the offscreen bitmap
#ifdef IBM
			InvalMapDrawingRect();
//#endif
#else
			if (needToRefresh) InvalMapDrawingRect();	// don't flash the screen unless old triangles need to be removedInvalMapDrawingRect()
#endif
		}
		
		return r;
	}

	return MapDrawingRect();
}

void DoSprayCanTool(Point p)
{
	model -> BroadcastToSelectedItem (M_TOOLDOWN,0,0);
}

void DoSpraySetCursor(Point p)
{
	model -> BroadcastToSelectedItem (M_TOOLDOWN,0,0); // code goes here
}

void DoEraseTool(Point p)
{
	model -> BroadcastToSelectedItem (M_TOOLDOWN,0,0);
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
Boolean AppendToLONGH(LONGH *h, long val)
{
	long npts;
	if(*h == 0)
	{
		if(!(*h=(LONGH)_NewHandleClear(0)))return FALSE;
	}
	
	npts = _GetHandleSize((Handle)(*h))/sizeof(long);
	//npts = GetNumLONGHItems(*h);
	_SetHandleSize((Handle)*h, (npts+1)*sizeof(long));
	if (_MemError()) 
		{printError("Out of memory."); return false; }
	(**h)[npts] = val;
	return true;
}
         
Boolean AppendToWORLDPOINTH(WORLDPOINTH *wh, WorldPoint *wp)
{
	long npts;
	if(*wh == 0)
	{
		if(!(*wh=(WORLDPOINTH)_NewHandleClear(0)))return false;
	}
	
	npts = _GetHandleSize((Handle)(*wh))/sizeof(WorldPoint);
	_SetHandleSize((Handle)*wh, (npts+1)*sizeof(WorldPoint));
	if (_MemError()) 
		{printError("Out of memory."); return false; }
	(**wh)[npts] = *wp;
	return true;
}

void DoSelectLEs(Point p)
{
	long i, j, n, numsegs, numLEs, count = 0;
	Point newPoint,startPoint;
	WorldPoint w;
	WORLDPOINTH wh=0;
	TLEList *thisLEList;
	LETYPE leType;
	ClassID thisClassID;
	Boolean foundLEList = 0, needToRefresh = false;
	SEGMENTH poly = 0;
	//double x, effectiveness = 100.;
	OSErr err = 0;

	// code goes here, lasso should apply to all LEs it captures, which may be multiple spills
	// code goes here, bring up a dialog to ask user if he wants to remove LEs not in lasso from spill
	for (i = 0, n = model->LESetsList->GetItemCount() ; i < n && !err; i++) 
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !model->IsUncertain()) continue;
		
		thisClassID = thisLEList -> GetClassID();
		if(thisClassID == TYPE_OSSMLELIST || thisClassID == TYPE_SPRAYLELIST )
		{
			TOLEList *thisOLEList = (TOLEList*)thisLEList; // typecast
			foundLEList = true;
			break;
		}
	}
	if (!foundLEList) { printNote("Set a spill before using lasso tool to hindcast"); return;}

	startPoint = p;
	MyMoveTo(p.h,p.v);
	w = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
	AppendToWORLDPOINTH(&wh,&w);	
	while(StillDown()) 
	{
		GetMouse(&newPoint);
		if((newPoint.h != p.h || newPoint.v != p.v))
		{
			MyLineTo(newPoint.h, newPoint.v);
			p = newPoint;
			w = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
			if(!AppendToWORLDPOINTH(&wh,&w))goto Err;
		}
	}
	w = ScreenToWorldPoint(startPoint, MapDrawingRect(), settings.currentView);
	AppendToWORLDPOINTH(&wh,&w);
	poly = WPointsToSegments(wh,_GetHandleSize((Handle)(wh))/sizeof(WorldPoint),&numsegs);
	if (!poly) goto Err;

	// loop over the LEs and check if they are in the polygon, if so mark to disperse
	// disperse immediately (at current time) and instantaneously, though maybe over an hour to test
	for (i = 0, n = model->LESetsList->GetItemCount() ; i < n && !err; i++) 
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !model->IsUncertain()) continue;
		
		thisClassID = thisLEList -> GetClassID();
		// always with a spray can ? then do this in the TSprayLEList...
		if(/*thisClassID == TYPE_OSSMLELIST || */thisClassID == TYPE_SPRAYLELIST )
		{
			WORLDPOINTH newHdl = ((TSprayLEList*)thisLEList)->fSprayedH;
			long numToSave=0, k;
			//Boolean alreadyUsingLasso = false;
			WorldPoint w,wp;
			LERec theLE;
			if (poly != 0)
			{
				TSprayLEList *sprayedHOwner = ((TSprayLEList*)thisLEList);
				sprayedHOwner =  (TSprayLEList *) model -> GetLEListOwner(((TSprayLEList*)thisLEList));
				// bring up dialog to set api, maybe dispersion duration, effectiveness
				numLEs = thisLEList->GetLECount();
				for(j=0; j < numLEs; j++) // make this numLEs
				{
					k = ((TSprayLEList*)thisLEList) -> LEIndexToSprayedIndex(j);
					wp = INDEXH(newHdl ,k);
					thisLEList -> GetLE (j, &theLE);
					//code goes here, should total all LEs selected at different times to get amount
					// also if selecting LEs at earlier time than others, deselect the later ones?
					// will eventually need to be able to edit the lassoed regions
					w.pLong = theLE.p.pLong;
					w.pLat = theLE.p.pLat;
					if (PointInPolygon(w,poly,numsegs,true))	// true -> holes ??
					{
						//count++;
						//if (count==1) {GetScaleFactorFromUser("Input dispersant effectiveness as a decimal (0 to 1)",&effectiveness);
						//if (effectiveness > 1) effectiveness = 1;}
						//x = GetRandomFloat(0, 1.0);
						//if (x <= effectiveness)
						//{
						//theLE.leCustomData = 5;	// but gets reset
						//theLE.beachTime = model->GetModelTime();	// use for time to disperse
						//INDEXH(newHdl,numToSave) = w;
						INDEXH(newHdl,numToSave) = wp;
 						numToSave++;
						//thisLEList -> SetLE (j, &theLE);
						//}
					}
				}
				//if (dispInfo.lassoSelectedLEsToDisperse) alreadyUsingLasso = true;
				if (numToSave>0)
				{
					needToRefresh = true;
					((TSprayLEList*)thisLEList)->fNumSprayedPts = numToSave;
					if (numToSave > 0) {
						_SetHandleSize((Handle)newHdl,numToSave * sizeof(**newHdl));
					}
					else { // no more points, they were all erased
						((TSprayLEList*)thisLEList) -> DisposeSprayedPoints();
					}
								//((TOLEList*)thisLEList) -> SetDispersionInfo(dispInfo);	
					/*for(j=0; j < numLEs; j++) // make this numLEs
					{
						thisLEList -> GetLE (j, &theLE);
						w = theLE.p;
						if (theLE.leCustomData == 5)	
							theLE.leCustomData = 0;
						else theLE.releaseTime = theLE.releaseTime + 864000;
						thisLEList -> SetLE (j, &theLE);
					}*/
				}
			}
		}
	}
Err:
	if(wh) {DisposeHandle((Handle)wh); wh=0;}	// may want to save this to draw or whatever
	model->NewDirtNotification(DIRTY_LIST);
	if (needToRefresh)
	{	// this doesn't do anything, really need to reset the model so LEs go to initial position
		InvalidateMapImage();// invalidate the offscreen bitmap
		InvalMapDrawingRect();
	}
	return;
}

void DoLassoTool(Point p)
{
	Boolean triSelected = false, gridShowing = false, needToRefresh = false;
	Point newPoint,startPoint;
	WorldPoint w;
	WORLDPOINTH wh=0;
	startPoint = p;
	//PrepareToDraw(MapDrawingRect(),settings.currentView,0,0);

	PtCurMap *map = GetPtCurMap();
	if (map==nil) return;
	TTriGridVel3D* triGrid = 0;	
	TMover *mover = 0;

	// if triangles not showing use lasso to draw circle around points that will be dispersed
	// code goes here, should circle be permanent? should dots turn color? area highlighted?
	// option key and control key are used to have lasso tool select area or LEs to disperse, otherwise selects triangles to track
	mover = map->GetMover(TYPE_PTCURMOVER);
	if (mover)
	{
		//if (!((PtCurMover *)mover) -> fVar.bShowGrid && !OptionKeyDown() && !ControlKeyDown()) return;	// only select triangles if grid is showing
		if (((PtCurMover *)mover) -> fVar.bShowGrid) gridShowing = true;	// only select triangles if grid is showing
		triGrid = (TTriGridVel3D*)(((PtCurMover *)mover) -> fGrid);
	}
	else	
	{
		mover = map->GetMover(TYPE_CATSMOVER3D);
		if (mover)
		{
			//if (!((TCATSMover3D *)mover) -> bShowGrid && !OptionKeyDown() && !ControlKeyDown()) return; // only select triangles if grid is showing
			if (((TCATSMover3D *)mover) -> bShowGrid) gridShowing = true; // only select triangles if grid is showing
			if (((TCATSMover3D *)mover) -> fRefinedGrid)
				triGrid = ((TCATSMover3D *)mover) -> fRefinedGrid;
			else
				triGrid = (TTriGridVel3D*)(((TCATSMover3D *)mover) -> fGrid);
		}
	}

	//if (!triGrid) return; 
	
	if (!triGrid) 
	{
		DoSelectLEs(p);	// would need to let lasso appear even without 3D grid...
		return; 
	}

	// other uses for the lasso - maybe bring up a dialog??
	/////////////////////////////////////////////////
	if (OptionKeyDown())
	{
		map->MarkRect(p); // select a rectangle to disperse
		return; 
	}
	if (ControlKeyDown() || !gridShowing)	// default if grid is not on
	{
		map->DoLassoTool(p); // select LEs to disperse
		return; 
	}
	/////////////////////////////////////////////////
	if(!ShiftKeyDown()) triGrid -> ClearTriSelection();	// delete it

	Boolean **triSelection = triGrid -> GetTriSelection(true);	// initialize if necessary
	if (!triSelection) return;

	//if(!ShiftKeyDown()) triGrid -> DeselectAll();

	MyMoveTo(p.h,p.v);
	w = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
	AppendToWORLDPOINTH(&wh,&w);	
	while(StillDown()) 
	{
		GetMouse(&newPoint);
		if((newPoint.h != p.h || newPoint.v != p.v))
		{
			MyLineTo(newPoint.h, newPoint.v);
			p = newPoint;
			w = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
			if(!AppendToWORLDPOINTH(&wh,&w))goto Err;
		}
	}
	w = ScreenToWorldPoint(startPoint, MapDrawingRect(), settings.currentView);
	AppendToWORLDPOINTH(&wh,&w);
	triSelected = triGrid->SelectTriInPolygon(wh, &needToRefresh);	

Err:
	if(wh) {DisposeHandle((Handle)wh); wh=0; needToRefresh = true;}
	model->NewDirtNotification(DIRTY_LIST);
	if(triSelected){	
		triGrid->bShowSelectedTriangles = true;
		SetShowSelectedTriangles(true);
		//InvalidateMapImage();// invalidate the offscreen bitmap
		//InvalMapDrawingRect();
	}
	if (needToRefresh)
	{
		InvalidateMapImage();// invalidate the offscreen bitmap
		InvalMapDrawingRect();
	}
	if (triSelection && !triGrid->ThereAreTrianglesSelected2()) triGrid->ClearTriSelection();
}

char *GetDistanceString(char *s, float value,short units)
{
	//char format[] = "%.2f";
	char format[] = "%.4f";
	
	if (units == FEET || units == YARDS || units == METERS)
		format[2] = '0';
	
	sprintf(s, format, value);
	
	strcat(s, " ");
	AppendUnits(s, units);
	
	
	return s;
}

OSErr DoRulerTool(Point pstart)
{
	PenState pensave;
	OSErr err = 0;
	WorldPoint anchorWP,endWP;
	double lastMeasuredDistanceInKm;
	char s[64],s2[64],s3[64];

	Point anchorPoint,lastPoint,endPoint;
	Rect r = MapDrawingRect(), saveClip;
	Boolean firstTime = true;

	GetPenState(&pensave); PenNormal();
	PenMode(patXor); 
	RGBForeColor(&colors[BLACK]);
#ifdef MAC
	//PenPat(&DKGRAY_BRUSH);
	PenPatQDGlobalsDarkGray();
#else
	PenStyle(GRAY, 1);
#endif

	MySetCursor(1005);
	anchorWP = ScreenToWorldPoint(pstart, MapDrawingRect(), settings.currentView);
	
	anchorPoint = lastPoint = endPoint = pstart;
	
	firstTime = true;

	while (StillDown()) 
	{
		GetMouse(&endPoint);
		endWP = ScreenToWorldPoint(endPoint, MapDrawingRect(), settings.currentView);

		if (firstTime || !EqualPoints(endPoint, lastPoint)) 
		{
			GetPenState(&pensave);
			
			lastMeasuredDistanceInKm = DistanceBetweenWorldPoints(anchorWP, endWP);
			GetDistanceString(s,lastMeasuredDistanceInKm,KILOMETERS);
			strcat(s,"   ");
			GetDistanceString(s2,lastMeasuredDistanceInKm/MILESTOKILO,MILES);
			strcat(s,s2);
			strcat(s,"   ");
			GetDistanceString(s3,lastMeasuredDistanceInKm*MILESTONAUTS/MILESTOKILO,NAUTICALMILES);
			strcat(s,s3);
			//DisplayMessage("NEXTMESSAGETEMP");
			
			DisplayMessage(s);
			
			SetPenState(&pensave);

			if(!firstTime)
			{
				// erase last line
				MyMoveTo(anchorPoint.h, anchorPoint.v); 
				MyLineTo(lastPoint.h, lastPoint.v);
			}
			firstTime = false;

			// draw the new line
			MyMoveTo(anchorPoint.h, anchorPoint.v); 
			MyLineTo(endPoint.h, endPoint.v);

			lastPoint  = endPoint;
		}
	}
	if(!firstTime)
	{	
		// erase last line
		MyMoveTo(anchorPoint.h, anchorPoint.v); 
		MyLineTo(lastPoint.h, lastPoint.v);
	}

	SetPenState(&pensave);
	//DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(0);
	
	return 0;
}

void ResetBoundaryTool()
{
	startno=endno=middleno=-1;
	clickNo = 0;
}

void DoBoundarySelectTool(Boolean usingShortSelTool, Point p)
{
		long segNo,verNum,v1,v2,numver;
		double clockwisePath,counterclockwisePath;
		Rect r=MapDrawingRect();
		GrafPtr		savePort;
		Rect saveClip;
	 	Boolean selectionDirection;
		short POINTSELECTFLAG =10;
		
		PtCurMap *map = GetPtCurMap();
		if (map==nil) return;
		TTriGridVel3D *triGrid = map -> GetGrid3D(false);

		map->FindNearestBoundary(p,&verNum,&segNo); 
		if(verNum > -1)
		{
			//if(!ShiftKeyDown()) triGrid -> ClearPtsSelection();	// delete it - should this be pts?
		
			/*if(!ShiftKeyDown()) 
			{
				triGrid -> ClearPtsSelection();	// delete it - should this be pts?
				map->ClearSegmentHdl();
				model->NewDirtNotification();	
				InvalMapDrawingRect();
			}*/
		
			Boolean **triSelection = triGrid -> GetPtsSelection(true);	// initialize if necessary
			if (!triSelection) return;
		
			//if(ControlKeyDown())
			//{
				//InvertBoundarySelection();
				//return;
			//}
			
			GetPortGrafPtr(&savePort);
			SetPortWindowPort(mapWindow);
			saveClip = MyClipRect(MapDrawingRect()); // ClipToDrawingRect();
			PrepareToDraw(MapDrawingRect(),settings.currentView,0,0);

		
			switch(clickNo)
			{
				case 0: // first click
					//MySetCursor(1008);
					//if(!ShiftKeyDown()) {map->ClearSegmentHdl();				model->NewDirtNotification();	}
					if(!ShiftKeyDown())
					{
						model->NewDirtNotification();	
						triGrid->DeselectAllPoints();
						// also dispose of fSegSelected?
						map->ClearSegmentHdl();
						InvalMapDrawingRect();
						triGrid -> ClearPtsSelection();	
						triSelection = 0;
					}

					//Boolean **triSelection = triGrid -> GetPtsSelection(true);	// initialize if necessary
					triSelection = triGrid -> GetPtsSelection(true);	// initialize if necessary
					if (!triSelection) return;
					//triGrid->DeselectAllPoints();
					triGrid->TogglePointSelection(verNum);
					triGrid->DrawPointAt(&r,verNum,POINTSELECTFLAG);
					startno = verNum;
					oldSegNum = segNo;
					clickNo = 1;
					break;
				case 1: // second click
					if(segNo == oldSegNum)
					{
						//MySetCursor(1009);
						clickNo = 2;
						map->AddSegmentToSegHdl(startno);
						//AppendToLONGH(&gSegSelected,startno);
						middleno = verNum;
						if(middleno == startno) // select all points in boundary
						{
							endno = map->PrevPointOnSeg(segNo,startno);
							map->InitBoundaryIter(segNo!=0,segNo,startno,endno);
							while(map->MoreBoundarySegments(&v1,&v2))		
							{
								triGrid->TogglePointSelection(v2);
								triGrid->DrawPointAt(&r,v2,POINTSELECTFLAG);
								map->AddSegmentToSegHdl(v2);
								//AppendToLONGH(&gSegSelected,v2);
							}
							//AppendToLONGH(&gSegSelected,startno);	// connect first and last boundary points
							// in CATS didn't allow selecting entire boundary - might want to here
							map->AddSegmentToSegHdl(startno);
							map->AddSegmentToSegHdl(-1);
							//AppendToLONGH(&gSegSelected,-1);
							startno = middleno=endno = -1;
							clickNo = 0;
						}
						else
						{
							numver = map->GetNumPointsInBoundarySeg(segNo);
							endno = verNum;

							counterclockwisePath = map->PathLength(TRUE,segNo,startno,endno);
							clockwisePath = map->PathLength(FALSE,segNo,startno,endno);
							
 
							selectionDirection = 
								usingShortSelTool?
									counterclockwisePath < clockwisePath:
									counterclockwisePath > clockwisePath;
							

							triGrid->DeselectAllPoints();
							triGrid->TogglePointSelection(startno);
							triGrid->DrawPointAt(&r,startno,POINTSELECTFLAG);
							
						
							map->AddSegmentToSegHdl(startno);
							//AppendToLONGH(&gSegSelected,startno);
							map->InitBoundaryIter(selectionDirection, segNo,  startno, endno);
							while(map->MoreBoundarySegments(&v1,&v2))		
							{
								triGrid->TogglePointSelection(v2);
								triGrid->DrawPointAt(&r,v2,POINTSELECTFLAG);
								map->AddSegmentToSegHdl(v2);
								//AppendToLONGH(&gSegSelected,v2);
							}
							map->AddSegmentToSegHdl(-1);
							//AppendToLONGH(&gSegSelected,-1);
							startno = middleno=endno = -1;
							clickNo = 0;
						}
					}
					else {SysBeep(5);break;}
					break;
			}
			
			MyClipRect(saveClip); // UnClip(saveClip);
			SetPortGrafPort(savePort);
		}
}

void ToolClick(EventRecord *event, short tool)
{
	Point where;
	long when;
	Rect saveClip;
	GrafPtr savePort;
	
	// event->where has been localized
	
	where = GetEventWhere(*event);
	when = GetEventWhen(*event);
	
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	saveClip = MyClipRect(MapDrawingRect()); // ClipToDrawingRect();
	switch (tool) {
		case ARROWTOOL: DoArrowTool(where); break;
		case HANDTOOL: DoHandTool(where); break;
		case ZOOMPLUSTOOL: DoZoomPlusTool(where); break;
		case ZOOMMINUSTOOL: MagnifyTool(where, -ZOOMMINUSTOOL); break;
		case SPILLTOOL:DoSpillTool(where);break;
		case SPRAYCANTOOL:DoSprayCanTool(where);break;
		case ERASETOOL:DoEraseTool(where);break;
		case LASSOTOOL:DoLassoTool(where);break;
		case RULERTOOL: DoRulerTool(where); break;
		case SHORTSELTOOL:
			DoBoundarySelectTool(tool == SHORTSELTOOL,where);
			break;
	}
	MyClipRect(saveClip); // UnClip(saveClip);
	SetPortGrafPort(savePort);
}


void  GoEntryViewMenuItemHit(void)
{
	WorldRect boundingRect =  model->GetMapBounds();
	if(!EqualWRects(boundingRect,voidWorldRect)) 
		ChangeCurrentView(boundingRect, TRUE, TRUE);
}

OSErr ToolPreferencesDialog() 
{ 
	// the chosen tool was double clicked
	OSErr err = 0;
	switch (settings.currentTool) 
	{
		//case ZOOMPLUSTOOL://fall through
		case ZOOMMINUSTOOL:
			GoEntryViewMenuItemHit(); break;
		case SPRAYCANTOOL:
		{
			SprayCanSettingsDialog(); 
			break;
		}
		case SPILLTOOL:
			err = CreateLESet((WorldPoint*)nil,(WorldPoint*)nil);
			break;
	}

	return 0; 
}

/////////////////////////////////////////////////


Boolean CommonDOMENU(long menuCodedItemID)
{
	Boolean handled = true;
	OSErr err = 0;
	Boolean dontDoIt = UserIsInEditMode();
	
	switch (menuCodedItemID) {
		case OPENITEM: 
			if (dontDoIt) goto StopUser;
			OpenSaveFile();
			break;
		case CLOSEITEM: 
			if (dontDoIt) goto StopUser;
			if(model->GetModelMode()  <  ADVANCEDMODE) 
			{
				err = model->fWizard->CloseMenuHit();
				if(err == USERCANCEL)
				{
					err = 0;// user cancel is not a real error
				}
				else
				{
					// close the rest of the stuff as well
					err = CloseSaveFile(FALSE,FALSE);
				}
			}
			else 
			{	// ask user then change files
				err = CloseSaveFile(TRUE,FALSE); 
				if(err != USERCANCEL) model->fWizard->CloseMenuHit();// close any wizard file also
				else err = 0; // user cancel is not a real error
			}
			break;
		case SAVEASITEM:
			if (dontDoIt) goto StopUser;
			SaveAs(); 
			break;
		case PRINTSETUPITEM: 
			PageSetUp(FALSE); break;
		case PRINTITEM: 
			if (dontDoIt) goto StopUser;
			Print(); break;
		case PREFERENCESITEM: 
			PreferencesDialog(); break;

		case UNDOITEM: 
			Undo(); break;
		case CUTITEM: 
			Cut(); break;
		case COPYITEM: 
			CopyMapViewToClipboard(); break;
		case PASTEITEM: 
			Paste(); break;

		case WIZARDITEM: 
			if (dontDoIt) goto StopUser;
			model->fWizard->InvokeWizardMenuHit(); break;
		//case MOVEITEM:
		case ADDITEM: 
			if (dontDoIt) goto StopUser;
			AddItem(); break;
		case SETTINGSITEM: 
			SettingsItem(); break;
		case DELETEITEM: 
			DeleteItem(); break;
		case UPITEM:
			if (dontDoIt) goto StopUser;
			UpItem(); break;
		case DOWNITEM:
			if (dontDoIt) goto StopUser;
			DownItem(); break;

		case RUNITEM:
			if (dontDoIt) goto StopUser;
			if (model->bHindcast)
				model->Run(model->GetStartTime()); 
			else
				model->Run(model->GetEndTime());
			break;
		case RUNTILLITEM: 
			if (dontDoIt) goto StopUser;
			model->RunTill(); break;
		case STEPITEM: 
			if (dontDoIt) goto StopUser;
			if (model->bHindcast)
				model->StepBackwards();
			else
				model->Step(); 
			break;
		case RESETITEM: 
			if (dontDoIt) goto StopUser;
			model->Reset(); break;
			// add some more menu items - deselect triangles, show grid

		case SHOWGRIDITEM:
		case SHOWSELECTEDTRIITEM:
		case DESELECTITEM:
		case SHOWMAXTRIITEM:
		case SETCONTOURSITEM:	// bring up map dialog
		case CONCATPTITEM:	// show grid afterwards
		case CONCATPLUMEITEM: 	// may want to deselect triangles
		case SHOWPLOTSITEM:
		case DEPTHCONTOURSITEM:
		case SHOWCONTOURSITEM:
		case SHOWCONTOURLABELSITEM:
		case DISPERSEOILITEM:
		case SMOOTHINGPARAMETERSITEM:
		case SCALEDEPTHSITEM:
		case CALCDOSAGEITEM:
		case SHOWDOSAGEITEM:
		case SUPPRESSDRAWINGITEM:
		case SETSELECTEDBEACH:
		case CLEARSELECTEDBEACH:
		case CONCENTRATIONONBEACH:
			if (dontDoIt) goto StopUser;
			{
				PtCurMap *map = GetPtCurMap();
				if (map==nil) return 0;		// this shouldn't happen since menu options won't be available
				if (err = map -> DoAnalysisMenuItems(menuCodedItemID)) return 0;
			}
			break;

		
		//////
		default: handled = false;
	}
	
	if(!handled) {
		Boolean isCDOGMenuItem = DoCDOGMenu(menuCodedItemID,dontDoIt);
		if(isCDOGMenuItem) {
			if(dontDoIt) goto StopUser; // tell the user why we didn't do it
			else handled = true;
		}
	}

	return handled;

StopUser:
	printError("You must finish editing the splots before using this menu item.  Choose the arrow tool when you are done editing the splots.");
	return handled;
}


////////////////////////////////// MAC CODE ////////////////////////////////////

#ifdef MAC

Rect ScrollAreaRight()
{
	Rect r = GetWindowPortRect(mapWindow);
	
	r.left = r.right - 15;
	r.top = TOOLBARHEIGHT;

	return r;
}

Rect ScrollAreaBottom()
{
	Rect r = GetWindowPortRect(mapWindow);
	
	r.top = r.bottom - 15;
	
	return r;
}

Rect MessageRect()
{
	Rect r = GetWindowPortRect(mapWindow);
	
	//r.top = r.bottom - 14;
	#if TARGET_API_MAC_CARBON
		r.top = r.bottom - MESSAGEHEIGHT;
	#else
		r.top = r.bottom - 14; // note: this is MESSAGEHEIGHT - 1
	#endif
	r.right -= 15;
	
	return r;
}

Rect ToolPaletteRect()
{
	Rect r, portRect = GetWindowPortRect(mapWindow);
	
	MySetRect(&r, 0, 0, portRect.right, TOOLBARHEIGHT);
	
	return r;
}

void MyDrawGrowIcon()
{
	Point p;
	Rect r = GetWindowPortRect(mapWindow);
	GrafPtr oldPort;
	RgnHandle clip = NewRgn();
	
	GetPortGrafPtr(&oldPort);
	SetPortWindowPort(mapWindow);
	GetClip(clip);
	r.top += TOOLBARHEIGHT;
	ClipRect(&r);
#if MACB4CARBON
	SetPt(&p, 0, 0); // correct for bug in System 7
	LocalToGlobal(&p);
	if (p.h < 0) 
	{
		mapWindow->portRect.bottom--;
	}
#endif
	DrawGrowIcon(mapWindow);
	#if TARGET_API_MAC_CARBON
	{	//  before OS X, DrawGrowIcon drew lines, now we have to do it ourselves
		//r = RightSideBarRect();
		r = ScrollAreaRight();
		MyMoveTo(r.left,r.top);
		MyLineTo(r.left,r.bottom -1);
		// note: DisplayMessage will also the the top line for the MessageRect 
		r = MessageRect();
		MyMoveTo(r.left,r.top); MyLineTo(r.right-1,r.top);
	}
	#endif
#if MACB4CARBON
	if (p.h < 0) 
	{
		mapWindow->portRect.bottom++;
	}
#endif
	SetClip(clip);
	DisposeRgn(clip);
	SetPortGrafPort(oldPort);
}

void UpdateMapWindow()
{
	GrafPtr savePort;
	
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	InvalRectInWindow(MessageRect(), mapWindow);
	BeginUpdate(mapWindow);
	RefreshMapWindowContents();
	EndUpdate(mapWindow);
	SetPortGrafPort(savePort);
}

void ClickWindow(WindowPtr w, EventRecord *event)
{
	Rect toolBar = ToolBarRect(), r = MapDrawingRect(), r2 = ListDrawingRect();
	//Rect r3 = /*TimeLineRect()*/PlayerRect();
	Point where;
	long when, dT;
	static long lastClick = 0;
	
	GlobalToLocal(&event->where);
	where = GetEventWhere(*event);
	when = GetEventWhen(*event);
	dT = when - lastClick;
	lastClick = when;
	
	if (w != mapWindow) return;
	
	if (MyPtInRect(where, &toolBar))
		if (ClickOnTool(where, dT <= GetDblTime())) return;
	
	if (MyPtInRect(where, &r))
		ToolClick(event, settings.currentTool);
	
	
	if (MyPtInRect(where, &r2))
		ListClick(VLClick(where, &objects));

	// check contour legend double click
	{
		PtCurMap *map = GetPtCurMap();	// still could be 2D...
		if (map && map->ThereIsADispersedSpill() && map->bShowLegend)
		{
			// MAC only code, need to find IBM code...
			if (MyPtInRect(where, &((*map).fLegendRect)) && dT <= GetDblTime())
				ContourDialog(&(map->fContourLevelsH),0);
			//if (MyPtInRect(where, &r3) && dT <= GetDblTime())	// for now asking user every time, would have to show everything in the dialog
				//OutputOptionsDialog(0);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////

void DrawGreetContents(WindowPtr w)
{
 	char s[256];
	short i;
	Rect frame, r = GetWindowPortRect(w);
	PicHandle p = (PicHandle)GetResource('PICT', 128);
	
	frame = GetPICTFrame(p);
	
	OffsetRect(&frame, (RectWidth(r) - RectWidth(frame)) / 2,
					   (RectHeight(r) - RectHeight(frame)) / 2);
	
	DrawPicture(p, &frame);
	ReleaseResource((Handle)p);
	
	if (FALSE) {
		PenMode(patBic);	
		//PenPat(&DKGRAY_BRUSH); // gray
		PenPatQDGlobalsGray();
		PaintRect(&r);
		PenNormal();
	}
	
	TextFontSizeFace(kFontIDTimes,18,bold); // outline + shadow
	
	for (i = 1 ; i <= 10 ; i++) {
		getindstring(s, 128, i);
		if (!strcmp(s, "END")) { i++; break; }
		MoveTo(40, (i + 1) * 25);
		drawstring(s);
	}
	
	TextSize(12);
	getindstring(s, 128, i);
	MoveTo(r.right - (40 + stringwidth(s)), r.bottom - 50);
	drawstring(s);
	
	PenNormal(); TextFontSizeFace(0,12,0);
}

OSErr Greet()
{
#ifdef NO_GUI
	return 0;
#endif
	WindowPtr w;
	GrafPtr oldPort;
	EventRecord	e;
	
	GetPortGrafPtr(&oldPort);
	
	w = MyGetNewCWindow(128, nil, (WindowPtr)-1);
	SetPortWindowPort(w);
	//CenterDialog(w, CENTER);
	CenterWindow(w, CENTER);
	
	InitCursor();
	
	if (!settings.inBackground) {
		MyShowWindow(w);
		for ( ; ; ) {
			//WaitNextEvent(mDownMask + updateMask + keyDownMask, &e, 15, nil);
			WaitNextEvent(mDownMask | mUpMask | activMask | updateMask | keyDownMask, &e, 15, nil);
			if (e.what == updateEvt && (WindowPtr)e.message == w) {
				BeginUpdate(w);
				DrawGreetContents(w);
				EndUpdate(w);
			}
			//if (e.what == mouseDown) break;
			if (e.what == mouseDown || e.what == mouseUp) break;
			if (e.what == keyDown)
			{
				char	msgChar = 0;
				msgChar = e.message & 0xff;
				if (msgChar == ENTER || msgChar == RETURN) break;
			}
		}
	}
	
	DisposeWindow(w);
	
	SetPortGrafPort(oldPort);
	
	return 0;
}


Boolean DOMENU(long menudata)
{
  	short menuID, menuItem;
	char pStr[100];
	Boolean	quit = FALSE;
	void DoSystemHelpMenuItem(short oneRelativeItemNum); // cheap way out
	
	menuID = HiWord(menudata);
	menuItem = LoWord(menudata);
	switch (menuID) {
		case kHMHelpMenuID:
			DoSystemHelpMenuItem(menuItem);
			break;
		case APPLEMENU:
			if ((menuItem + APPLEMENU) == APPLEABOUTTATITEM)
				Greet();
			else {
				GetMenuItemText(GetMenuHandle(APPLEMENU), menuItem, (unsigned char *)pStr);
				#if MACB4CARBON
				OpenDeskAcc((unsigned char *)pStr);	// not supported in carbon - what is this?
				#endif
			}
			break;
		
		case FILEMENU:
			switch (menuItem + FILEMENU) {
				default: 
					(void)CommonDOMENU(menuItem + menuID);
					break;
				case QUITITEM:
					if (OKToQuit()) 
					{
						Boolean userChangedMind ;
						if(model->GetModelMode()  <  ADVANCEDMODE)  userChangedMind = model->fWizard->QuitMenuHit();
						else userChangedMind = CloseSaveFile(true,true);
						if (userChangedMind) break;
						quit = TRUE;
						DoQuit();
					}
					break;
			}
			break;
		
		case RUNMENU:
		case ITEMMENU:
		case EDITMENU:
		case CDOGMENU:
		case ANALYSISMENU:
			(void)CommonDOMENU(menuItem + menuID);
			break;
		

		case SHARINGMENU:
			switch (menuItem + SHARINGMENU) {
				case ABOUTSHARINGITEM:
					if (GetHelp("¥ Sharing menu", FALSE) == HELP_TOPICS)
						GetHelpTopics("¥ Sharing menu", FALSE);
					break;
			}
			break;
		
		case HELPMENU:
			GetHelpTopics(0, FALSE);
			break;
	}

//	IAC task removed for classossm	
//	if (menuID >= 50 && menuID < APPLEMENU) SendMHIT(menuID, menuItem);
	
	HiliteMenu(0);
	
	return quit;
}

void DoSystemHelpMenuItem(short oneRelativeItemNum)
{
	short ourItemNum = 0;
	long menudata = HELPMENU;

#if TARGET_API_MAC_CARBON
	ourItemNum = oneRelativeItemNum;
#else
	char ourStr[256],theirStr[256];
	short numMyItems,i;
	MenuHandle sysHelpMenu = nil;
	OSErr err = HMGetHelpMenuHandle(&sysHelpMenu);

	if(err || !gMyHelpMenu || !sysHelpMenu) return;
	
	getmenuitemtext(sysHelpMenu,oneRelativeItemNum,theirStr);
	
	numMyItems = CountMenuItems(gMyHelpMenu);
	for(i = 1; i<= numMyItems; i++)
	{
		getmenuitemtext(gMyHelpMenu,i,ourStr);
		if(!strcmp(ourStr,theirStr))
		{ // we found it
			//long menudata = HELPMENU;
			//menudata = menudata<<16;
			//menudata+=i;
			//DOMENU(menudata);
			ourItemNum = i;
			break;
		}
	}
#endif

	if(ourItemNum > 0){
		menudata = HELPMENU;
		menudata = menudata<<16;
		menudata+=ourItemNum;
		DOMENU(menudata);
	}
}

#if MACB4CARBON
void MyDragWindow2(WindowPtr whichWindow, Rect *dragLimit, EventRecord *event)
{
	GrafPtr		savePort, newPort;
	//RgnHandle 	dragRgn;
	long 		result;
	Point 		move = event->where;
	Rect 		portBounds;
	
	if (WaitMouseUp()) {
		RgnHandle 	dragRgn =  MyNewRgn();
		if(!dragRgn) return;
		GetPortGrafPtr(&savePort);
		GetWMgrPort(&newPort);
		SetPortWindowPort(newPort);
		SetClip((RgnHandle)GetGrayRgn());
		
		// ensure the drag outline does not get drawn through the windows above whichWindow
		ClipAbove(whichWindow);
		
	#if MACB4CARBON
		CopyRgn(((WindowPeek)whichWindow)->strucRgn, dragRgn = NewRgn());
	#else
		err = GetWindowRegion (whichWindow,kWindowStructureRgn,dragRgn = NewRgn());
		if(err) goto done; // this should never happen
	#endif
		
		// drag the outline of the window around the screen
		result = DragGrayRgn(dragRgn, move, dragLimit, dragLimit, noConstraint, nil);
		
		move.v = result >> 16;			// HiWord(result)
		move.h = result & 0xFFFF;		// LoWord(result)
		
		// if(move.v != 0x8000) 
		if (move.v != -32768) {
			// the mouse button has been released inside dragLimit
			// see 'Inside Macintosh Volume I,' pages 294-295, for expalnation of 0x8000
			portBounds = SafeWindowBounds(whichWindow);
			move.v += whichWindow->portRect.top - portBounds.top;
			move.h += whichWindow->portRect.left - portBounds.left;
			
			MyMoveWindow(whichWindow, move.h, move.v, false);
			// if (!EmptyRgn(((WindowPeek)mapWindow)->updateRgn)) settings.screenImageValid = FALSE;
		}
		DisposeRgn(dragRgn);
		SetPortGrafPort(savePort);
	}
}
#endif
void MyDragWindow(WindowPtr w, EventRecord *event)
{
	Rect dragLimit = GetScreenBounds();
	
	dragLimit.top += GetMBarHeight();
	
	// if (event->modifiers & cmdKey) { MyDragWindow2(w, &dragLimit, event); return; }
	// if (w != mapWindow) BringToFront(w); // update palette before it is dragged?
	
	#if MACB4CARBON
		MyDragWindow2(w, &dragLimit, event);
	#else
		Rect r = MapDrawingRect();
		DragWindow(w,event->where,&dragLimit); // JLM, actually we will probably stop implementing our own dragging of the window in OS X
	#endif
}

Boolean DealWithEvent(EventRecord event, Boolean dialog)
{
	WindowPtr		whichWindow;
	GrafPtr 		savePort;
	long			size;
  	short			partCode;
	char			c;
	Boolean			quit = FALSE;
	Rect			r, r2;
	double			xRatio, yRatio;
	WorldRect		newView;
	
	lastEvent = event;
	
	switch (event.what) {
		case kHighLevelEvent: AEProcessAppleEvent(&event); break; // JLM HandleODOC
		
		case MultFindEvt:
			settings.inBackground = !(BitAnd((long)event.message, (long)activeFlag) == (long)activeFlag);
			HiliteWindow(mapWindow, !settings.inBackground && mapWindow == FrontWindow());
			MyDrawGrowIcon();
			////
			///// JLM 11/3/06  when we come forward via IAC, if our Window has been minimized, we'd like our Window to unminimize
			if(!settings.inBackground && FrontWindow() == mapWindow) 
				SelectWindow(mapWindow);
			/////
			break;
		
		case updateEvt:
			if ((WindowPtr)event.message != mapWindow) break;
			UpdateMapWindow();
			break;
		
		case activateEvt:
			#if MACB4CARBON
				HiliteWindow(mapWindow, FrontWindow() == mapWindow);
			#endif
			
//			MyDrawGrowIcon();
			break;
	}
	
	if (dialog) return quit;
	
	switch (event.what) {
		case keyDown:
		case autoKey:
			c = BitAnd(event.message, charCodeMask);
			if (BitAnd((long)event.modifiers, (long)cmdKey) == (long)cmdKey) {
				if (c == '=' || c == '+')
					MagnifyTool(event.where, -ZOOMPLUSTOOL); // ToolClick(&event, ZOOMPLUSTOOL);
				else
					quit = DOMENU(MenuKey(c));
			}
			if (VListKeyEvent(&event, &objects)) 
			{
				DrawTools(BUTTONPICTBASE, -1);// JLM 1/4/99 because an up arrow can cause button highlighting to change
				break;
			}
			if (c == DEL) ; // DeleteSelectedObjects();
			break;
		
		case mouseDown:
			partCode = FindWindow(event.where, &whichWindow);
			switch (partCode) {
				case inMenuBar:
					InitCursor();
					quit = DOMENU(MenuSelect(event.where));
					break;
				
				case inDrag:
					InitCursor();
					MyDragWindow(whichWindow, &event);
						//r = MapDrawingRect();
						//InvalMapDrawingRect();
					break;
				
				case inGrow:
					InitCursor();
					SetRect(&r, MINIMUM_WINDOW_HEIGHT, MINIMUM_WINDOW_HEIGHT, RectWidth(GetScreenBounds()), RectHeight(GetScreenBounds()));
					size = GrowWindow(whichWindow, event.where, &r);
					if (size) {
						InvalMapWindowBorders();
						r = MapDrawingRect();
						#if TARGET_API_MAC_CARBON  // double check this
						{	// try erasing the newly exposed areas right away
							Rect newRect = GetWindowPortRect(whichWindow);
							GetPortGrafPtr(&savePort);
							SizeWindow(whichWindow, LoWord(size), HiWord(size), TRUE);
							newRect = GetWindowPortRect(whichWindow);
							MyClipRect(newRect);
							if(newRect.bottom > r.bottom) {// erase any area exposed below the map rect
								r2 = newRect;
								r2.top = r.bottom-1;
								MyEraseRect(&r2);
							}
							if(newRect.right > r.right) {// erase any area exposed below the map rect
								r2 = newRect;
								r2.left = r.right-1;
								MyEraseRect(&r2);
							}
						}
						#else
							SizeWindow(whichWindow, LoWord(size), HiWord(size), TRUE);
						#endif
						//SizeWindow(whichWindow, LoWord(size), HiWord(size), TRUE);
						ResizeChildWindows();
						r2 = MapDrawingRect();
						xRatio = (double)RectWidth(r2) / (double)RectWidth(r);
						yRatio = (double)RectHeight(r2) / (double)RectHeight(r);
						newView = settings.currentView;
						newView.hiLong = newView.loLong + WRectWidth(newView) * xRatio;
						newView.loLat = newView.hiLat - WRectHeight(newView) * yRatio;
						ChangeCurrentView(newView, FALSE, FV_SAMESCALE);
						//ChangeCurrentView(settings.currentView, FALSE, FV_SAMESCALE);
						//ChangeCurrentView(settings.currentView, TRUE, FV_SAMESCALE);

						InvalMapWindowBorders();
						//InvalMapDrawingRect();
					}
					break;
				
				case inZoomIn:
				case inZoomOut:
					if (TrackBox(whichWindow, event.where, partCode)) {
						GetPortGrafPtr(&savePort);
						SetPortWindowPort(whichWindow);
						//r = MapDrawingRect();
					  	ZoomWindow(whichWindow, partCode, TRUE);
						ResizeChildWindows();
						ChangeCurrentView(settings.currentView, FALSE, FV_SAMESCALE);
						/*r2 = MapDrawingRect();
						xRatio = (double)RectWidth(r2) / (double)RectWidth(r);
						yRatio = (double)RectHeight(r2) / (double)RectHeight(r);
						newView = currentView;
						newView.hiLong = newView.loLong + WRectWidth(newView) * xRatio;
						newView.loLat = newView.hiLat - WRectHeight(newView) * yRatio;
						ChangeCurrentView(newView, TRUE, TRUE, TRUE);
						InvalMapWindow();*/
						//InvalMapWindow();
						SetPortGrafPort(savePort);
					}
					break;
				
				case inContent:
					if (whichWindow != mapWindow && whichWindow != FrontWindow())
						BringToFront(whichWindow);
					GetPortGrafPtr(&savePort);
					SetPortWindowPort(whichWindow);
					ClickWindow(whichWindow, &event);
					SetPortGrafPort(savePort);
					break;
				
				case inGoAway:
					 if (TrackGoAway(whichWindow, event.where)) {
					 	InitCursor();
					 	//if (CautionAlert(4000, nil) == 2) 
							ExitToShell();
					 }
					break;
			}
			break;
	}
	
	return quit;
}

/////////////////////////////////////////////////////////////////////////////////////

//#pragma segment Main //JLM
#pragma segment SYSTEM //JLM


/////////////////////////////////////////////////////////////////////////////////////



#if TARGET_API_MAC_CARBON
OSStatus ConvertTraditionalPathToUnixPath(const char * inPath, char * outPath, int outPathMaxLen) 
{
	CFStringRef  inStr = 0;
	CFURLRef url = 0;
	CFStringRef outStr = 0;
	Boolean gotStr;
	OSStatus err =  0;
	
	inStr = CFStringCreateWithCString(kCFAllocatorDefault, inPath ,kCFStringEncodingMacRoman);
	if (inStr == 0){err = -1; goto done;}
	
	url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, inStr, kCFURLHFSPathStyle,0);
	if (url == 0){err = -1; goto done;}
	
	outStr = CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle);
	if (outStr == 0){err = -1; goto done;}
	
	gotStr = CFStringGetCString(outStr, outPath, outPathMaxLen, kCFURLPOSIXPathStyle);
	if (!gotStr){err = -1; goto done;}

done:
	if(outStr) CFRelease(outStr);
	if(url) CFRelease(url);
	if(inStr) CFRelease(inStr);
	return err;
}

OSStatus ConvertUnixPathToTraditionalPath(const char * inPath, char * outPath, int outPathMaxLen) 
{
	CFStringRef  inStr = 0;
	CFURLRef url = 0;
	CFStringRef outStr = 0;
	Boolean gotStr;
	OSStatus err =  0;
	
	inStr = CFStringCreateWithCString(kCFAllocatorDefault, inPath ,kCFStringEncodingMacRoman);
	if (inStr == 0){err = -1; goto done;}
	
	url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, inStr, kCFURLPOSIXPathStyle,0);
	if (url == 0){err = -1; goto done;}
	
	outStr = CFURLCopyFileSystemPath(url, kCFURLHFSPathStyle);
	if (outStr == 0){err = -1; goto done;}
	
	gotStr = CFStringGetCString(outStr, outPath, outPathMaxLen, kCFURLHFSPathStyle);
	if (!gotStr){err = -1; goto done;}

done:
	if(outStr) CFRelease(outStr);
	if(url) CFRelease(url);
	if(inStr) CFRelease(inStr);
	return err;
}

#endif

///////////////////////////
////////////////////////////


/////////////////////////////////////////////////

#if TARGET_API_MAC_CARBON
static void SetNumMenuItemsTo256(MenuHandle m)
{
	// this function exists to increase the number of menu items to 256 
	short numItems;
	
	if(m== 0) return;
	
	numItems = CountMenuItems(m);
	for(; numItems < 256; numItems ++) {
		AppendMenu(m,"\px"); // any name will do
	}
}
#endif

OSErr InitMacStuff()
{
	char fontName[50];
#ifdef VERSION881
	SysEnvRec theWorld;
#endif
	
#ifndef NO_GUI
	BasicInit();
#endif
	
#ifdef VERSION881
	SysEnvirons(curSysEnvVers, &theWorld);
	if (!theWorld.hasFPU) return -1;
#endif
	
	strcpy("", ""); // load the STDCLIB segment
	sprintf(fontName, "dummy"); // load the STDIOMRG segment (which contains the
								// the STDIO, SANELIB, INTENV, and SADEV segments)
	
	//////// normal menus
#ifndef NO_GUI
	GetAndInsertMenu(APPLEMENU, 1);
	GetAndInsertMenu(FILEMENU, 0);
	GetAndInsertMenu(EDITMENU, 0);
	GetAndInsertMenu(ITEMMENU, 0);
	GetAndInsertMenu(RUNMENU, 0);
	//InitAnalysisMenu();	// may want to move to after loading a 3D mover
	InitCDOGMenu();
	
//	GetAndInsertMenu(SHARINGMENU, 0);		// IAC removed for cossm

	#ifdef MPW  ///{
	//	printNote("JLM debug startup");
	#endif  ///}
	
	
	if(SystemHelpAvailable()) // JLM 2/25/98
	{	// put our items under the system  help menu
		short numMyItems, i;
		Str255 pStr;
		MenuHandle sysHelpMenu  = nil;
		OSErr err = 0;
		
		#if MACB4CARBON
			err = HMGetHelpMenuHandle(&sysHelpMenu);
			//err = HMGetHelpMenu(&sysHelpMenu,NULL);
		#endif
		
		if(!err && sysHelpMenu)
		{
			gMyHelpMenu = GetMenu(HELPMENU);
			numMyItems = CountMenuItems(gMyHelpMenu);
			for(i = 0; i < numMyItems; i++)
			{
				GetMenuItemText(gMyHelpMenu,i+1,pStr); 
				AppendMenu(sysHelpMenu,pStr);
			}
		}
		else
		{	// use my own help menu
			GetAndInsertMenu(HELPMENU, 0);
		}
	}
	else
	{	// use my own help menu
		GetAndInsertMenu(HELPMENU, 0);
	}

	
	GetAndInsertMenu(pFILETYPES, -1);
	GetAndInsertMenu(pNOVICESAVEFILETYPES, -1);	// code goes here, remove this?
	GetAndInsertMenu(pINTERMEDIATESAVEFILETYPES, -1);	
	GetAndInsertMenu(pADVANCEDSAVEFILETYPES, -1);
	GetAndInsertMenu(pDISPERSEDOILSAVEFILETYPES, -1);
	GetAndInsertMenu(pLONGLINEUNITS, -1);
	GetAndInsertMenu(pLONGLABELUNITS, -1);
	GetAndInsertMenu(pLATLINEUNITS, -1);
	GetAndInsertMenu(pLATLABELUNITS, -1);
	GetAndInsertMenu(pMASSUNITS, -1);
	GetAndInsertMenu(pAREAUNITS, -1);

	GetAndInsertMenu(pLEMASSUNITS, -1);
	GetAndInsertMenu(pNORTHSOUTH1, -1);
	GetAndInsertMenu(pEASTWEST1, -1);
	GetAndInsertMenu(pNORTHSOUTH2, -1);
	GetAndInsertMenu(pEASTWEST2, -1);
	GetAndInsertMenu(pPOLLUTANTS, -1);
	GetAndInsertMenu(pPOLLUTANTS2, -1);	// with chemicals

	GetAndInsertMenu(pMOVERTYPES, -1);
	GetAndInsertMenu(pMOVERTYPES2, -1);
	GetAndInsertMenu(pSCALEBYTYPES, -1);
	GetAndInsertMenu(pMAPTYPES, -1);
	GetAndInsertMenu(pTIMEFILETYPES, -1);

	GetAndInsertMenu(pRANDOMINPUTTYPE, -1);
	GetAndInsertMenu(pDISPERSANTMODE, -1);
	GetAndInsertMenu(pDISPTIMETYPE, -1);
	GetAndInsertMenu(pWAVEHEIGHT, -1);
	GetAndInsertMenu(pWAVEHTINPUT, -1);
	GetAndInsertMenu(pMIXEDLAYERDEPTH, -1);
	GetAndInsertMenu(pWATERDENSITY, -1);
	GetAndInsertMenu(pDEPTHUNITS, -1);
	GetAndInsertMenu(pHEIGHTUNITS, -1);
	GetAndInsertMenu(pDIAGNOSTICSTRTYPE, -1);
	GetAndInsertMenu(pLEVELOFCONCERN, -1);
	GetAndInsertMenu(pSPECIES, -1);
	GetAndInsertMenu(pCHECKBOXTYPE, -1);

	GetAndInsertMenu(pHYDROLOGYINFO, -1);
	GetAndInsertMenu(pTRANSPORTUNITS, -1);
	GetAndInsertMenu(pPERSISTENCE, -1);
	
	GetAndInsertMenu(pANGLEUNITS, -1);
	GetAndInsertMenu(pTIMEZONES, -1);

	GetAndInsertMenu(pHYDPROCESS, -1);
	GetAndInsertMenu(pEQCURVES, -1);
	GetAndInsertMenu(pSEPFROMPLUME, -1);
	GetAndInsertMenu(pTEMPSAL, -1);
	GetAndInsertMenu(pDROPLET, -1);
	GetAndInsertMenu(pDISCHARGERATE, -1);
	GetAndInsertMenu(pHYDRODYNAMICOPTIONS, -1);

	GetAndInsertMenu(pDISCHARGETYPE, -1);
	GetAndInsertMenu(pCIRCULATIONINFO, -1);
	GetAndInsertMenu(pYESNO, -1);

	GetAndInsertMenu(pDIAMETERUNITS, -1);
	GetAndInsertMenu(pTEMPUNITS, -1);
	GetAndInsertMenu(pDENSITYUNITS, -1);
	GetAndInsertMenu(pGOR, -1);
	GetAndInsertMenu(pOILDISCHARGE, -1);
	GetAndInsertMenu(pGASDISCHARGE, -1);
	GetAndInsertMenu(pDEPTHUNITS2, -1);
	GetAndInsertMenu(pMOLWT, -1);

	GetAndInsertMenu(pNOVICELETYPES, -1);
	GetAndInsertMenu(pINTERMEDIATELETYPES, -1);
	GetAndInsertMenu(pADVANCEDLETYPES, -1);
	GetAndInsertMenu(pADVANCEDLETYPES2, -1);
	GetAndInsertMenu(pTIMEAMOUNTTYPES, -1);
	
	GetAndInsertMenu(pSPEEDUNITS, -1);
	GetAndInsertMenu(pSPEEDUNITS2, -1);
	GetAndInsertMenu(pDIRECTIONS, -1);
	GetAndInsertMenu(pDURATIONDAYS, -1);
	GetAndInsertMenu(pMONTHS, -1);
	GetAndInsertMenu(pMONTHS2, -1);
	GetAndInsertMenu(pYEARS, -1);
	GetAndInsertMenu(pYEARS2, -1);
	GetAndInsertMenu(pYEARS_EXTENDED, -1);
	GetAndInsertMenu(pYEARS2_EXTENDED, -1);

	//DefineMenu(pBACKGROUNDCOLORS, 18, 60, FALSE, 1, (ProcPtr)0, (ProcPtr)0, nil);
	/*DefineMenu(pBACKGROUNDCOLORS, 18, 60, FALSE, 1,
			   (SAMenuActionUPP)MakeUPP((ProcPtr)DrawBackColorItem, uppSAMenuActionProcInfo),
			   (SAMenuActionUPP)MakeUPP((ProcPtr)InvertColorItem, uppSAMenuActionProcInfo),
			   nil);*/
	DefineMenu(pBACKGROUNDCOLORS, 18, 60, FALSE, 1,
			   MakeSAMenuActionUPP(DrawBackColorItem),
			   MakeSAMenuActionUPP(InvertColorItem),
			   nil);	// carbon
	GetAndInsertMenu(pBACKGROUNDCOLORS, -1);	
	#if TARGET_API_MAC_CARBON
	{
		OSStatus err = 0;
		MenuDefSpec saMenu_defSpec;
		MenuRef m = 0;
		
		memset(&saMenu_defSpec,0,sizeof(saMenu_defSpec));
		saMenu_defSpec.defType = kMenuDefProcPtr;
		saMenu_defSpec.u.defProc = NewMenuDefUPP( SaMenu_MDEF );
		
		m = GetMenuHandle(pBACKGROUNDCOLORS); err = SetMenuDefinition(m,&saMenu_defSpec);		
	}
	#endif
	
	DrawMenuBar();
#endif
	return 0;
}

//void main()
int main(int argc, char *argv[])
{
	//#pragma unused (argc,argv)
	
	Boolean gotEvent, done = FALSE;
	EventRecord e;
	OSErr err = 0;
	// PG = 0;
	// if (!InitPerf(&PG, 5, 2, 1, 1, "\pCODE", 0, "", 0, 0, 0, 0)) { printError2("performance tools", 0); return; }
	
#ifndef NO_GUI 
	if (InitMacStuff()) { Alert(9002, 0); return -1; }
#endif
	
	if (!System7Available()) { Alert(9000, 0); return -1; }
	
#ifndef NO_GUI 
	#if TARGET_API_MAC_CARBON
	{
   		WindowClass windowClass = kDocumentWindowClass;
		WindowAttributes attributes =  kWindowFullZoomAttribute | kWindowCollapseBoxAttribute | kWindowResizableAttribute | kWindowCloseBoxAttribute;
		Rect contentBounds;
		GetAvailableWindowPositioningBounds(GetMainDevice(), &contentBounds); // excludes menubar and dock
		InsetRect(&contentBounds,20,20); // NOTE: if we make the window too close to the zoomed state, the system sets the standard state to a weird size 
		contentBounds.top += 20; // allow space for the title bar
		err = CreateNewWindow(windowClass,attributes, &contentBounds, &mapWindow);
	}
	#else
	mapWindow = MyGetNewCWindow(MAPWINDOW, nil, (WindowPtr)-1);
	#endif
	if(!mapWindow) {return -1;}
	SetPortWindowPort(mapWindow); //JLM 2/24/99 set the port at the beginning of the program
  	progressDialog = GetNewDialog(M29, nil, (WindowPtr)-1);
	if(!progressDialog) {return -1;}
#endif
	// HSetVol(0, 0, 0); // put the system in "hierarchical mode"
	//InitPrinting();	// will do this differently for carbon
	
	if (err = CommonInit()) return -1;
	
#ifndef NO_GUI 
	#if !TARGET_API_MAC_CARBON
		//MyMoveWindow(mapWindow, 5, 25 + 18, FALSE);	// not sure about this stuff
		//SizeMapWindowForDrawingRect(FullScreenMapDrawingRect());
	#endif
	FullScreenMapWindow();
	ResizeChildWindows();
	MyShowWindow(mapWindow);
#endif

	/// check for command file on command line or in folder	
	if (argc>1)
	{
		char commandFileName[kMaxNameLen], commandFilePath[kMaxNameLen],unixPath[kMaxNameLen],cCurrentPath[kMaxNameLen], fullPath[kMaxNameLen];
		long i;
		OSErr err = 0;
		Boolean gnomeStartedByDoubleClick = false;

		strcpy(commandFileName,argv[1]);
		// need to deal with garbage in argv when start Gnome from double click
		if (!strncmp("-psn",commandFileName,4))
			{gnomeStartedByDoubleClick = true; goto CheckForCommandFile;}
		for (i=2;i<argc;i++)
		{	
			strcat(commandFileName," ");
			strcat(commandFileName,argv[i]);
		}
		if (commandFileName[0] == '/')
		{
			strcpy(unixPath, commandFileName);
			err = ConvertUnixPathToTraditionalPath((const char *)unixPath, commandFilePath, kMaxNameLen);
		}
		else
		{
			// want to save the working directory for later
			// also should check for leading '....'
			if(!getcwd(cCurrentPath, sizeof(cCurrentPath)))	// unix directory
			{	
				StringSubstitute(commandFileName, '/', ':');
				GetFullPath(TATvRefNum, TATdirID,commandFileName, commandFilePath);
			}
			else 
			{
				char workingDirectoryPath[kMaxNameLen], classicPath[kMaxNameLen];
				err = ConvertUnixPathToTraditionalPath((const char *)cCurrentPath, workingDirectoryPath, kMaxNameLen); 
				strcpy(gWorkingDirectory,workingDirectoryPath);	
				if (ConvertIfUnixPath(commandFileName, classicPath)) strcpy(commandFileName,classicPath);
				ResolvePartialPathFromThisFolderPath(commandFileName,workingDirectoryPath);
				strcpy(commandFilePath,commandFileName);
			}
		}
		if (FileExists(0,0,commandFilePath))
		{
			if(IsCommandFile(commandFilePath))
			{
		#ifndef NO_GUI 
				UpdateMapWindow();// so the tool bar draws etc
		#endif
				err = DoCommandFile(commandFilePath);
				goto skipInterface; // we assume they will want the program to quit when finished
			}
			else
			{
				char msg[256];
				sprintf(msg,"Selected file is not a command file %s",commandFilePath);
				printNote(msg);
			}
		}
		else
		{	
			char msg[256];
			sprintf(msg,"Command file not found at %s",commandFilePath);
			printNote(msg);
		}
	}
CheckForCommandFile:
	if(FileExists(TATvRefNum, TATdirID,STDCOMMANDFILENAME))
	{
		char commandFilePath[256];
		GetFullPath(TATvRefNum, TATdirID,STDCOMMANDFILENAME, commandFilePath);
		if(IsCommandFile(commandFilePath))
		{
	#ifndef NO_GUI 
			UpdateMapWindow();// so the tool bar draws etc
	#endif
			err = DoCommandFile(commandFilePath);
			goto skipInterface; // we assume they will want the program to quit when finished
		}
	}

#ifndef NO_GUI 
	if (Greet()) return -1;	
	//if (gNoaaVersion && model->GetModelMode() == ADVANCEDMODE) printNote("GNOME has found a noaa.ver file. Spills will be allowed outside the spillable area.");
#endif

	// JLM 9/7/00
	// When the user drags a LOC file onto the application
	// we don't want to ask less than advanced users to select a file
	// because they already have.  
	// The only way we will know if they have done this is to 
	// process the events in our queue.
	// So process any events waiting at this point 
	while (!done && !settings.quitting) {
		gotEvent = WaitNextEvent(everyEvent, &e, 15, nil);
		if (gotEvent) {
			done = DealWithEvent(e, FALSE);
		}
		else
			break;
	}
	//////////////
#ifdef NO_GUI // allow for drag and drop with no gui
			goto skipInterface; // we assume they will want the program to quit when finished
#endif
	
	
	if(model->fWizard->StartUp())  return -1;
	
	// BringTATForward();
	
	while (!done && !settings.quitting) {
//		ProcessDDEQueue();
		ShowToolInstructions(GetMouseLocal(mapWindow));
		DoTasks();
		gotEvent = WaitNextEvent(everyEvent, &e, 15, nil);
		if (gotEvent) {
			done = DealWithEvent(e, FALSE);
		}
	}
	
	skipInterface:
		
		CommonQuit();
		
		DeleteList();
		DisposeWindow(mapWindow);
		DisposeDialog(progressDialog);
		
	return 0;
}

#else

////////////////////////////////// WINDOWS CODE ////////////////////////////////////

LRESULT CALLBACK DrawIconProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	PAINTSTRUCT ps;
	GrafPtr savePort;
	
	switch (message) {
		case WM_PAINT:
			GetPortGrafPtr(&savePort);
			SetPortBP(hWnd, &ps);
			SetPortEP(hWnd, &ps);
			SetPortWindowPort(GetParent(hWnd));
			// DrawAIIcon(GetParent(hWnd), M42ICON);
			SetPortGrafPort(savePort);
			break;
		
		default: return DefWindowProc(hWnd, message, wParam, lParam);
	}
	
	return 0;
}

LRESULT CALLBACK SplashProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	PAINTSTRUCT ps;
	WindowPtr savePort;
	Rect r;
	HBITMAP b;
	short i;
	char str[100];
	
	switch (message) {
		case WM_PAINT:
			GetPortGrafPtr(&savePort);
			SetPortBP(hWnd, &ps);
			r = GetWindowPortRect(hWnd);
			if (b = LoadBitmap(hInst, MAKEINTRESOURCE(128))) {
				
				DrawBitmap(b, (r.left + r.right) / 2 - 125,
							  (r.top + r.bottom) / 2 - 125,
							  SRCCOPY);
				DeleteObject(b);
			}
			TextFontSize(kFontIDTimes,14);
			
			for (i = 0 ; i < 16 ; i++) {
				MyMoveTo(r.left + 40, r.top + 35 + (i + 1) * 23);
				getindstring(str, 128, i);
				if (!strcmp(str, "END")) break;
				drawstring(str);
			}
			SetPortEP(savePort, &ps);
			break;
		
		default: return DefWindowProc(hWnd, message, wParam, lParam);
	}
	
	return 0;
}

LRESULT CALLBACK GreetProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message) {
		case WM_INITDIALOG:
			InitCursor();
			CenterWindow(hDlg, GetDesktopWindow());
			return TRUE;
		
		case WM_COMMAND:
			if (LOWORD(wParam) == IDOK) {
				EndDialog(hDlg, TRUE);
				return TRUE;
			}
	}
	
	return 0;
}

OSErr Greet()
{
#ifndef NO_GUI
	DialogBox(hInst, "Greet", mapWindow, (DLGPROC)GreetProc);
#endif	
	return 0;
}

char* HelpFileName(char* helpPath)
{
	helpPath[0] = 0;
	if(model->fWizard) model->fWizard->WizardHelpFilePath(helpPath);
	if(!helpPath[0]) strcpy(helpPath,HELPFILENAME);
	return helpPath;
}

void DOMENU(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	int itemID = LOWORD(wParam), wmEvent = HIWORD(wParam);
	Boolean modify = FALSE;
	char helpFilePath[256];
	Boolean handled = CommonDOMENU(itemID);
	
	if(handled) 
		return;
	
	switch (itemID) {
		///// File menu
		case QUITITEM:
			if (OKToQuit()) 
			{
				Boolean userChangedMind ;
				if(model->GetModelMode()  <  ADVANCEDMODE)  userChangedMind = model->fWizard->QuitMenuHit();
				else userChangedMind = CloseSaveFile(true,true);
				if (userChangedMind) break;
				SendMessage(mapWindow, WM_USER, 0, 0);// short-cut close without confirmation
			}
			break;
		
		///// Sharing menu
		case ABOUTSHARINGITEM:
			//if (!HtmlHelp(hWnd, HelpFileName(helpFilePath), HH_DISPLAY_TOPIC, (DWORD)(LPSTR)GS1(192, 0))) // sharing with other programs
			if (!WinHelp(hWnd, HelpFileName(helpFilePath), HELP_KEY, (DWORD)(LPSTR)GS1(192, 0))) // sharing with other programs
				printError(GS1(136, 0)); // Unable to activate help.
			break;
		
		///// Help menu
		case ABOUTTATITEM: Greet(); break;
		case HELPCONTENTSITEM:
			//if (!HtmlHelp(hWnd, HelpFileName(helpFilePath), HH_DISPLAY_TOPIC, (DWORD)(LPSTR)GS1(192, 0))) // sharing with other programs
			//if (!WinHelp(hWnd, HELPFILENAME, HELP_KEY, (DWORD)(LPSTR)GS1(191, 0))) // help index
			//if (!WinHelp(hWnd, HelpFileName(helpFilePath), HELP_FINDER, (DWORD)(LPSTR)GS1(191, 0))) // help index
			if (!WinHelp(hWnd, HelpFileName(helpFilePath), HELP_CONTENTS, (DWORD)(LPSTR)GS1(191, 0))) // help index
				printError(GS1(136, 0)); // Unable to activate help.
			break;
		case HELPSEARCHITEM:
			//if (!HtmlHelp(hWnd, HelpFileName(helpFilePath), HH_DISPLAY_TOPIC, (DWORD)(LPSTR)GS1(192, 0))) // sharing with other programs
			if (!WinHelp(hWnd, HelpFileName(helpFilePath), HELP_PARTIALKEY, (DWORD)(LPSTR)""))
				printError(GS1(136, 0)); // Unable to activate help.
			break;
		case HELPHELPITEM:
			//if (!HtmlHelp(hWnd, HelpFileName(helpFilePath), HH_HELP_CONTEXT, 0)) // sharing with other programs
			if (!WinHelp(hWnd, (LPSTR)NULL, HELP_HELPONHELP, 0))
				printError(GS1(136, 0)); // Unable to activate help.
			break;
	}

//	IAC task removed for classossm	
//	if (itemID >= 10000 && itemID < 20000) SendMHIT(itemID, 0);
}

void FillWithPattern(HDC hdc, RECT *R, short index)
{
	HBITMAP hBitMap = 0;
	HBRUSH hBrush = 0;
	static MonoPatBits bits[] = {
		{ 0x00, 0, 0x00, 0, 0x00, 0, 0x00, 0, 0x00, 0, 0x00, 0, 0x00, 0, 0x00, 0 }, // black
		{ 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0 }, // white
	 // { 0x22, 0, 0x88, 0, 0x22, 0, 0x88, 0, 0x22, 0, 0x88, 0, 0x22, 0, 0x88, 0 }, // dark gray
		{ 0xAA, 0, 0x55, 0, 0xAA, 0, 0x55, 0, 0xAA, 0, 0x55, 0, 0xAA, 0, 0x55, 0 }, // gray
		{ 0xDD, 0, 0x77, 0, 0xDD, 0, 0x77, 0, 0xDD, 0, 0x77, 0, 0xDD, 0, 0x77, 0 }, // light gray
		{ 0x7F, 0, 0xF7, 0, 0x7F, 0, 0xF7, 0, 0x7F, 0, 0xF7, 0, 0x7F, 0, 0xF7, 0 }, // lighter gray
		{ 0x77, 0, 0x77, 0, 0x77, 0, 0x77, 0, 0x77, 0, 0x77, 0, 0x77, 0, 0x77, 0 }, // vert stripes
		{ 0x00, 0, 0xFF, 0, 0x00, 0, 0xFF, 0, 0x00, 0, 0xFF, 0, 0x00, 0, 0xFF, 0 }, // horiz stripes
		{ 0xBB, 0, 0xDD, 0, 0xEE, 0, 0x77, 0, 0xBB, 0, 0xDD, 0, 0xEE, 0, 0x77, 0 }, // downstipes
		{ 0x77, 0, 0xEE, 0, 0xDD, 0, 0xBB, 0, 0x77, 0, 0xEE, 0, 0xDD, 0, 0xBB, 0 }, // upstripes
		{ 0xEE, 0, 0xEE, 0, 0x00, 0, 0xEE, 0, 0xEE, 0, 0xEE, 0, 0x00, 0, 0xEE, 0 }  // boxes
	};
	
	if (hBitMap = CreateBitmap(8, 8, 1, 1, (LPSTR)bits[index]))
		if (hBrush = CreatePatternBrush(hBitMap))
			FillRect(hdc, R, hBrush);
	
	if (hBitMap) DeleteObject(hBitMap);
	if (hBrush) DeleteObject(hBrush);
}

COLORREF Get3DBackgroundColor()
{
	if (RunningWindows95()) return GetSysColor(COLOR_3DFACE);
	
	if (RunningOnWin32()) return colors[LIGHTGRAY];
	
	return colors[WHITE];
}

void FillRectWith3DBackground(Rect r)
{
	RECT R;
	HBRUSH brush;
	
	MakeWindowsRect(&r, &R);
	
	if (brush = CreateSolidBrush(Get3DBackgroundColor())) {
		FillRect(currentHDC, &R, brush);
		DeleteObject(brush);
	}
}

// shared between WinMain and the window procs
static MSG msg;
Boolean gResizing = FALSE;

LRESULT CALLBACK ToolProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	Point where;
	POINT p;
	Rect r,triRect;
	RECT R,triR;
	WindowPtr savePort;
	PAINTSTRUCT ps;
	
	switch (message) {
		case WM_PAINT:
			//if (gResizing) break; JLM 2/9/99, this is quick so draw it
			GetPort(&savePort);
			SetPortBP(hWnd, &ps);
			DrawTools(0, 0);
			SetPortEP(savePort, &ps);
			break;
		
		case WM_LBUTTONDOWN:
		case WM_LBUTTONDBLCLK:
			GetPort(&savePort);
			SetPort(hWnd);
			MakeMacPoint(&msg.pt, &where);
			GlobalToLocal(&where);
			// for now user is asked every time
			//if (MyPtInRect(where,&(PlayerRect())) && /*message == WM_LBUTTONDBLCLK*/ShiftKeyDown())
				//OutputOptionsDialog(0);
			//else
				ClickOnTool(where, message == WM_LBUTTONDBLCLK);
			SetPort(savePort);
			break;
		
		case WM_SETCURSOR: // keep Windows from resetting the arrow cursor within the client rect
			r = SplitBarRect();
			MakeWindowsRect(&r, &R);
			triRect = TimeBarTriangleRect(); // JLM 1/21/99
			MakeWindowsRect(&triRect, &triR);
			GetCursorPos(&p);
			ScreenToClient(hWnd, &p);
			if (PtInRect(&R, p) || (PtInRect(&triR, p) && !UserIsInEditMode()))
				MySetCursor(STRETCHLISTCURSOR);
			else
				InitCursor();
			break;
		
		default: return DefWindowProc(hWnd, message, wParam, lParam);
	}
	
	return 0;
}


LRESULT CALLBACK MessageProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	PAINTSTRUCT ps;
	Rect r;
	WindowPtr savePort;
	
	switch (message) {
		case WM_PAINT:
			//if (gResizing) break; JLM 2/9/99, this is quick so draw it
			GetPortGrafPtr(&savePort);
			SetPortBP(hWnd, &ps);
			DisplayMessage(0);
			PenNormal();
			r = GetWindowPortRect(hWnd);
			MyMoveTo(r.left, r.top);
			MyLineTo(r.right, r.top);
			SetPortEP(savePort, &ps);
			break;
		
		case WM_SETCURSOR: // keep Windows from resetting the arrow cursor within the client rect
			InitCursor();
			break;

		default: return DefWindowProc(hWnd, message, wParam, lParam);
	}
	
	return 0;
}

void UpdateMapWindow()
{
	GrafPtr savePort;
	PAINTSTRUCT ps;
	
	GetPortGrafPtr(&savePort);
	/*
	SetPort(mapWindow);
	SetPort(savePort);
	*/
	
	SetPortBP(mapWindow, &ps);
	RefreshMapWindowContents();
	SetPortEP(savePort, &ps);
}

void PlaceMapWindow(Boolean get, int nCmdShow)
{
	char s[100], *e = "";
	RECT r;
	
	if (get) {
		if ( !GetPrivateProfileString("window", "position", e, s, 99, "OSSM.INI")) {
			FullScreenMapWindow();
			ShowWindow(mapWindow, SW_SHOWMAXIMIZED); // nCmdShow
		}
		else {
			Boolean windowOffScreen = FALSE;
			Rect defaultRect = FullScreenMapWindowRect();
			sscanf(s, "%ld, %ld, %ld, %ld", &r.left, &r.top, &r.right, &r.bottom);
			// changed this logic 1/7/98
			if (r.left < defaultRect.left) windowOffScreen = TRUE;
			if (r.top < defaultRect.top) windowOffScreen = TRUE;
			if (r.right > defaultRect.right) windowOffScreen = TRUE;
			if (r.bottom > defaultRect.bottom) windowOffScreen = TRUE;
			if (windowOffScreen) {
				FullScreenMapWindow();
				//ShowWindow(mapWindow, SW_SHOWMAXIMIZED); // nCmdShow
				ShowWindow(mapWindow, nCmdShow); // JLM 1/7/98
			}
			else {
				MoveWindow(mapWindow, r.left, r.top, r.right - r.left, r.bottom - r.top, FALSE);
				ShowWindow(mapWindow, nCmdShow);
			}
		}
	}
	else {
		GetWindowRect(mapWindow, &r);
		sprintf(s, "%ld, %ld, %ld, %ld", r.left, r.top, r.right, r.bottom);
		WritePrivateProfileString("window", "position", s, "OSSM.INI");
	}
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	char itemText[100];
	short tool;
	int nTextX, nTextY;
	Point where;
	Rect r;
	MenuHandle m;
	WindowPtr savePort;
    SIZE size;
    WORD wCheckX;
    DWORD dwCheckXY;
	POINT p;
	RECT client, R, itemRect;
	HBRUSH colorBrush;
    HFONT hfontOld;
    HDC hdc;
	PAINTSTRUCT ps;
    LPMEASUREITEMSTRUCT lpmis;
    LPDRAWITEMSTRUCT lpdis;
	COLORREF crText, crBkgnd;
    static COLORREF crSelText, crSelBkgnd;  // text and background color of selected item
 	static Boolean appWindowMayBeUp = FALSE,
 				   currentlyUpdating = FALSE, sGotWMPaintWhileResizing = FALSE;
	static Rect currentMapRect = { 0, 0, 0, 0 };
	extern void ReleaseStaticPorts();
	extern long TranslateSharingKey(short key);
	
	DefineEvent(&lastEvent, hWnd, message, wParam, lParam);
	
	// DisplayMessage("WndProc() start");
	switch (message) {
	    case WM_CREATE:
			m = GetMenu(hWnd);
		    crSelText = GetSysColor(COLOR_HIGHLIGHTTEXT);
		    crSelBkgnd = GetSysColor(COLOR_HIGHLIGHT);
		    return 0;
		
		case WM_ACTIVATEAPP:
			settings.inBackground = (wParam == 0);
			break;
		
		case WM_MOUSEACTIVATE:
			if (!settings.inBackground || LOWORD(lParam) != HTCLIENT)
				return DefWindowProc(hWnd, message, wParam, lParam);
			else
				return MA_ACTIVATEANDEAT;
		
		case WM_SETFOCUS:
			break;
		
		case WM_KILLFOCUS:
			//CommonFlashCursor(TRUE);
			break;
		
		case WM_MEASUREITEM:
            lpmis = (LPMEASUREITEMSTRUCT)lParam;
			if (lpmis->CtlType != ODT_BUTTON) { // a menu/combo-box item
				hdc = GetDC(hWnd);
				hfontOld = (HFONT)SelectObject(hdc, GetStockObject(SYSTEM_FONT));
				dwCheckXY = GetMenuCheckMarkDimensions();
				strcpy(itemText, "XXX Green");
				GetTextExtentPoint(hdc, itemText, strlen(itemText), &size);
	        	lpmis->itemWidth = size.cx + LOWORD(dwCheckXY);
	        	lpmis->itemHeight = size.cy;
				SelectObject(hdc, hfontOld);
				ReleaseDC(hWnd, hdc);
			}
			else { // a tab button
				// r = GetDialogItemBox(, lpmis->CtlID);
	        	// lpmis->itemWidth = 90;
	        	// lpmis->itemHeight = 32;
			}
			
			return TRUE;
		
        case WM_DRAWITEM:
            lpdis = (LPDRAWITEMSTRUCT)lParam;
			if (lpdis-> CtlType != ODT_BUTTON) { // a menu/combo-box item
	            if (lpdis->itemState & ODS_CHECKED) {
	            	itemRect = lpdis->rcItem;
					itemRect.left += 2;
					itemText[0] = '>';
		            ExtTextOut(lpdis->hDC, itemRect.left, itemRect.top,
		            		   0, &itemRect, &itemText[0], 1, NULL);
				}
				
	            dwCheckXY = GetMenuCheckMarkDimensions();
	            wCheckX = LOWORD(dwCheckXY);
	            nTextX = wCheckX + lpdis->rcItem.left;
	            nTextY = lpdis->rcItem.top;
				
				hfontOld = (HFONT)SelectObject(lpdis->hDC, GetStockObject(SYSTEM_FONT));
	            if (lpdis->itemState & ODS_SELECTED) {
	                crText = SetTextColor(lpdis->hDC, crSelText);
	                crBkgnd = SetBkColor(lpdis->hDC, crSelBkgnd);
	            }
				
				strcpy(itemText, "XXX ");
				GetTextExtentPoint(lpdis->hDC, itemText, strlen(itemText), &size);
				nTextX += size.cx;
				lpdis->rcItem.left = nTextX;
				LoadString(hInst, lpdis->itemID + 1, itemText, 99);
	            ExtTextOut(lpdis->hDC, nTextX, nTextY, ETO_OPAQUE, &lpdis->rcItem,
	            		   itemText, strlen(itemText), NULL);
				
				lpdis->rcItem.right = lpdis->rcItem.left - wCheckX / 2;
				lpdis->rcItem.left -= size.cx;
				lpdis->rcItem.top += 2;
				lpdis->rcItem.bottom -= 2;
				FrameRect(lpdis->hDC, &lpdis->rcItem, (HBRUSH)GetStockObject(BLACK_BRUSH));
				colorBrush = CreateSolidBrush(backColors[lpdis->itemID + 1]);
				if (colorBrush) {
					InflateRect(&lpdis->rcItem, -1, -1);
					FillRect(lpdis->hDC, &lpdis->rcItem, colorBrush);
					DeleteObject(colorBrush);
				}
				
				if (lpdis->itemState & ODS_SELECTED) {
	                SetTextColor(lpdis->hDC, crText);
	                SetBkColor(lpdis->hDC, crBkgnd);
				}
				
	            SelectObject(lpdis->hDC, hfontOld);
			}
 			else { // a tab button
				GetPort(&savePort);
				SetPort(lpdis->hwndItem);
				DrawTab(GetParent(lpdis->hwndItem),
						lpdis->CtlID,
						lpdis->itemState & ODS_FOCUS);
				SetPort(savePort);
			}
			
            return TRUE;
		
		case WM_SYSCOMMAND:
			//CommonFlashCursor(TRUE);
			return DefWindowProc(hWnd, message, wParam, lParam);
		
		case WM_COMMAND: // message: command from application menu
			DOMENU(hWnd, message, wParam, lParam);
			break;
		
        case WM_MOUSEMOVE:
			SetMouse(MAKEPOINTS(lParam));
            break;

		case 0x020A: // WM_MOUSEWHEEL
			GetPort(&savePort);
			SetPort(hWnd);
			MakeMacPoint(&msg.pt, &where);
			GlobalToLocal(&where);
			r = ListDrawingRect();
			if (MyPtInRect(where, &r)){
				short scrollValue = HIWORD(wParam);
				if(scrollValue < 0 ) {
					if(objects.curRec + objects.viewLines > objects.numItems) // no need to scroll down
						break; 
					wParam = SB_LINEDOWN;
				}
				else if (scrollValue > 0) {
					if(objects.curRec <= 0) // no need to scroll up
						break; 
					wParam = SB_LINEUP;
				}
				if(IsWindowEnabled(objects.window)) {
					VListWndProc(objects.window,WM_VSCROLL, wParam, 0);
				}	
			}
			SetPort(savePort);
			break;
        
		case WM_LBUTTONDOWN:
		case WM_LBUTTONDBLCLK:
			SetCapture(hWnd);
			SetMouseButton(TRUE);
			GetPort(&savePort);
			SetPort(hWnd);
			MakeMacPoint(&msg.pt, &where);
			GlobalToLocal(&where);
			MakeWindowsPoint(&where, &msg.pt);
			r = MapDrawingRect();
			if (MyPtInRect(where, &r))
			{
				ToolClick(&msg, settings.currentTool);
				{
					PtCurMap *map = GetPtCurMap();
					Boolean dblClick = (message == WM_LBUTTONDBLCLK);
					if (map && map->ThereIsADispersedSpill() && map->bShowLegend && settings.currentTool == ARROWTOOL)
					{
						Rect frame = (*map).fLegendRect;
						MyOffsetRect(&frame,r.left,r.top);
						if (MyPtInRect(where,&frame) /*&& ShiftKeyDown()*/ && dblClick)	// doesn't seem to recognize the double click
							ContourDialog(&(map->fContourLevelsH),0);
					}
				}
			}
			else {
				r = ListDrawingRect();
				r.right -= GetSystemMetrics(SM_CYVTHUMB); // don't count click on scroll bar
				if (MyPtInRect(where, &r))
					ListClick(message == WM_LBUTTONDBLCLK);
			}
			SetPort(savePort);
			break;
		
		case WM_LBUTTONUP:
			SetMouseButton(FALSE);
			ReleaseCapture();
			break;
		
		case WM_SETCURSOR: // keep Windows from resetting the arrow cursor within the client rect
			GetClientRect(hWnd, &client);
			GetCursorPos(&p);
			ScreenToClient(hWnd, &p);
			if (!PtInRect(&client, p) || inMenu || hWnd != GetActiveWindow())
				return DefWindowProc(hWnd, message, wParam, lParam);
			CommonSetCursor();
			break;
		
		case WM_TIMER:
		{ 
			int static sHandlingTimer = false;
//			ProcessDDEQueue();
			if (sHandlingTimer || (model && model->fRunning)) break;
			sHandlingTimer = TRUE; // to avoid infinite DoTasks 
			if (!inMenu && !settings.inBackground)
				ShowToolInstructions(GetMouseLocal(toolWnd));
			DoTasks();
			if (!Button() && GetActiveWindow() == mapWindow) {
				SetCapture(mapWindow); // force cursor to update
				ReleaseCapture();	   // even if mouse hasn't moved
			}
			sHandlingTimer = FALSE;
			break;
		}
		
		case WM_ENTERMENULOOP:
			inMenu = TRUE;
			return 0;
		
		case WM_EXITMENULOOP:
			inMenu = FALSE;
			return 0;
		
        case WM_ENTERSIZEMOVE:
			gResizing = TRUE;
			sGotWMPaintWhileResizing = FALSE;
			return DefWindowProc(hWnd, message, wParam, lParam);
		
        case WM_EXITSIZEMOVE:
			gResizing = FALSE;
			if (sGotWMPaintWhileResizing) {
				sGotWMPaintWhileResizing = FALSE;
				InvalMapWindow();
			}
			break;
		
		case WM_SYSKEYDOWN:
		case WM_KEYDOWN:
			if (wParam == VK_MENU) appWindowMayBeUp = TRUE;
			if (wParam == VK_DELETE) {
				// DOMENU(hWnd, message, MAKEWPARAM(CLEARITEM, 0), 0);
				break;
			}

//			IAC tasks removed for classossm
//			if (ControlKeyDown() && TranslateSharingKey(wParam) != -1)
//				DOMENU(hWnd, message, MAKEWPARAM(TranslateSharingKey(wParam), 0), 0);
//			if (wParam >= VK_PRIOR && wParam <= VK_DOWN)
//				return SendMessage(objects.port, message, wParam, lParam);
			return DefWindowProc(hWnd, message, wParam, lParam);
		
		case WM_SYSKEYUP:
		case WM_KEYUP:
			if (wParam == VK_MENU) appWindowMayBeUp = FALSE;
			return DefWindowProc(hWnd, message, wParam, lParam);
		
		case WM_PAINT:
			if (currentlyUpdating) break;
			if (gResizing) {
				HDC hDC;
				if (hDC = BeginPaint(mapWindow, &ps))
				{
					SelectClipRgn(hDC,0); // make sure the clip region is cleared, JLM 2/9/99
					EndPaint(mapWindow, &ps);
				}
				sGotWMPaintWhileResizing = TRUE;
				break;
			}
			SendMessage(toolWnd, WM_PAINT, 0, 0);
			SendMessage(messageWnd, WM_PAINT, 0, 0);
			currentlyUpdating = TRUE;
			UpdateMapWindow();
			currentlyUpdating = FALSE;
			break;
		
		case WM_SIZE:
			if (wParam == SIZE_MINIMIZED) break;
			// JLM 2/5/99 allow window to be smaller because of NT auto draw while resizing
			//if (LOWORD(lParam) < MINIMUM_WINDOW_HEIGHT || HIWORD(lParam) < MINIMUM_WINDOW_HEIGHT) {
			//	GetWindowRect(mapWindow, &R);
			//	MoveWindow(mapWindow, R.left, R.top, // will cause another WM_SIZE message to be sent
			//			   _max(R.right - R.left, MINIMUM_WINDOW_HEIGHT), _max(R.bottom - R.top, MINIMUM_WINDOW_HEIGHT), TRUE);
			//	break;
			//}
			ResizeChildWindows();
			ChangeCurrentView(settings.currentView, FALSE, FV_SAMESCALE);
			break;
		
//		case WM_IAC:
//			HandleIACMessage((short)wParam, (long)lParam);
//			break;
		
		case WM_CLOSE:
			if (!CHOICEALERT(M75, 0, FALSE)) break;
			// fall through
			
		case WM_USER: // short-cut close without confirmation
			if (!OKToQuit()) break;
			if (timer) { KillTimer(mapWindow, timer); timer = 0; }
			PageSetUp(TRUE);
			CommonQuit();
			DestroyWindow(hWnd);
			break;
		
        case WM_DESTROY:
			DoQuit();
			PlaceMapWindow(FALSE, 0);
			WinHelp(mapWindow, HELPFILENAME, HELP_QUIT, 0);
			//if (!HtmlHelp(mapWindow, HELPFILENAME, HH_CLOSE_ALL, 0)) // sharing with other programs
			if (timer) { KillTimer(mapWindow, timer); timer = 0; }
			PostQuitMessage(0);
			settings.quitting = TRUE;
			ReleaseStaticPorts();
            break;
		
        default: // pass it on if unprocessed
			return DefWindowProc(hWnd, message, wParam, lParam);
	}
	
	return 0;
}

BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
	// Create a main window for this application instance.
	if (!(hMainWnd = CreateWindow(
	        CLASSNAME,				// See RegisterClass() call.
	        BASIC_WINDOW_TITLE_STR,					// Text for window title bar.
	        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,	// Window style.
	        CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, // Use default positioning
	        NULL,					// Overlapped windows have no parent.
	        NULL,					// Use the window class menu.
	        hInstance,				// This instance owns this window.
	        NULL					// We don't use any data in our WM_CREATE
	))) return FALSE;
	
	if (!(toolWnd = CreateWindow(
			"ToolClass",			// class
	        "",						// no title
	        WS_CHILD | WS_VISIBLE,	// style
	        0, 0, 0, 0,				// position
	        hMainWnd,				// parent
	        (HMENU)1,				// child ID
	        hInstance,				// owning instance
	        NULL					// no data
	))) return FALSE;
	
	if (!(messageWnd = CreateWindow(
			"MessageClass",			// class
	        "",						// no title
	        WS_CHILD | WS_VISIBLE,	// style
	        0, 0, 0, 0,				// position
	        hMainWnd,				// parent
	        (HMENU)3,				// child ID
	        hInstance,				// owning instance
	        NULL					// no data
	))) return FALSE;
	
	SetPort(hMainWnd);
	
	return TRUE;
}

BOOL InitApplication(HINSTANCE hInstance)
{
	WNDCLASS  wc;
	LRESULT CALLBACK PlotProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	LRESULT CALLBACK PlotProc2(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	
	BasicInit();
	
	wc.style         = CS_BYTEALIGNCLIENT | CS_HREDRAW | CS_VREDRAW | CS_OWNDC | CS_DBLCLKS; // Class style(s).
	wc.lpfnWndProc   = (WNDPROC)WndProc;       // Window Procedure
	wc.cbClsExtra    = 0;                      // No per-class extra data.
	wc.cbWndExtra    = 0;
	wc.hInstance     = hInstance;              // Owner of this class
	wc.hIcon         = LoadIcon (hInstance, "ICONOSSM"); // Icon name from .RC
	wc.hCursor       = LoadCursor(NULL, IDC_ARROW); // Cursor
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); // (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName  = MENURESOURCENAME;              // Menu name from .RC
	wc.lpszClassName = CLASSNAME;              // Name to register as
	
	RegisterClass(&wc);
	
	wc.style         = CS_HREDRAW | CS_VREDRAW | CS_OWNDC | CS_DBLCLKS; // Class style(s).
	wc.lpfnWndProc   = (WNDPROC)ToolProc;       // Window Procedure
	wc.cbClsExtra    = 0;                      // No per-class extra data.
	wc.cbWndExtra    = 0;
	wc.hInstance     = hInstance;              // Owner of this class
	wc.hIcon         = 0;						// Icon name from .RC
	wc.hCursor       = LoadCursor(NULL, IDC_ARROW); // Cursor
	wc.hbrBackground = (HBRUSH)GetStockObject(LTGRAY_BRUSH); // (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName  = NULL;              		// Menu name from .RC
	wc.lpszClassName = "ToolClass";				// Name to register as
	
	RegisterClass(&wc);
	
	wc.style         = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; // Class style(s).
	wc.lpfnWndProc   = (WNDPROC)MessageProc;	// Window Procedure
	wc.cbClsExtra    = 0;                      // No per-class extra data.
	wc.cbWndExtra    = 0;
	wc.hInstance     = hInstance;              // Owner of this class
	wc.hIcon         = 0;						// Icon name from .RC
	wc.hCursor       = LoadCursor(NULL, IDC_ARROW); // Cursor
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); // (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName  = NULL;              		// Menu name from .RC
	wc.lpszClassName = "MessageClass";			// Name to register as
	
	RegisterClass(&wc);
	
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
	wc.lpfnWndProc = SplashProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInst;
	wc.hIcon = NULL;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); // (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = "Splash";
	
	RegisterClass(&wc);
	
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
	wc.lpfnWndProc = PlotProc2;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInstance;
	wc.hIcon = NULL;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); // (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = "PlotClass2";
	
	RegisterClass(&wc);

	wc.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
	wc.lpfnWndProc = PlotProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInstance;
	wc.hIcon = NULL;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); // (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = "PlotClass";
	
	RegisterClass(&wc);

	wc.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
	wc.lpfnWndProc = DrawIconProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInst;
	wc.hIcon = NULL;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); // (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = "DrawIcon";
	
	RegisterClass(&wc);
	
	return TRUE;
}

int CALLBACK EnumMARPLOTFont(LOGFONT *f, TEXTMETRIC *m, DWORD type, LPARAM lParam)
{
	return 0;
}

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
					 LPSTR lpCmdLine, int nCmdShow)
{
	HANDLE hAccelTable;
	WindowPtr w, w2;
	Boolean done = FALSE;
	GrafPtr oldPort;
	OSErr err = 0;
	
	hInst = hInstance;
	GetPort(&oldPort);
	
	int winArgc;
	LPWSTR *winArgv = CommandLineToArgvW(GetCommandLineW(), &winArgc);
	
	// comment out this if statement to allow multiple instances on PC 3/10/05
	/*if (w = FindWindow(CLASSNAME, 0)) {
		w2 = GetLastActivePopup(w);
		BringWindowToTop(w);
		if (IsIconic(w2))
			ShowWindow(w2, SW_RESTORE);
		else
			SetForegroundWindow(w2);
		
		return FALSE;
	}*/
	//
	if (hPrevInstance) { // should never be non-null
		MessageBox(0, GS1(137, 0), 0, MB_ICONSTOP | MB_OK); // MARPLOT is already running.
		return FALSE;
	}
	
	if (!InitApplication(hInstance)) return FALSE;
	if (!InitInstance(hInstance, nCmdShow)) return FALSE;
	
	hAccelTable = LoadAccelerators(hInstance, MENURESOURCENAME);
	
	mapWindow = hMainWnd;
	
	if (err = _InitAllHandles())
		{ TechError("WinMain()", "_InitAllHandles()", err); return FALSE; }
	
	if (err = InitDirectoryTable())
		{ TechError("WinMain()", "InitDirectoryTable()", err); return FALSE; }
	
	if (CommonInit()) { DestroyWindow(mapWindow); goto skipInterface; }
#ifndef NO_GUI
	//InitAnalysisMenu();	// may want to move to after loading a 3D mover
	InitCDOGMenu();
	PlaceMapWindow(TRUE, nCmdShow);
	ResizeChildWindows();
#endif
	
	PageSetUp(-1); // JLM moved before Greet 
	
	/// check for command file on command line or in folder	
	if (winArgc>1)
	{	// what might be input on command line? full path, filename only, .\, ..\ etc, partial path, any unix?
		char commandFilePath[kMaxNameLen], commandFileName[kMaxNameLen], workingDirectoryPath[kMaxNameLen];
		strcpy(commandFilePath,lpCmdLine);
		GetDirectoryFromID(TATdirID, workingDirectoryPath);
		strcpy(gWorkingDirectory,workingDirectoryPath);
		RemoveLeadingTrailingQuotes(commandFilePath);	// do we only want to do this for full path?
		if (!IsWindowsAbsolutePath(commandFilePath))
		{
			if (IsPartialPath(commandFilePath))
			{
				//ResolvePath(commandFilePath);
				ResolvePartialPathFromThisFolderPath(commandFilePath,workingDirectoryPath);
			}
			else
			{
				strcpy(commandFileName,"\\");
				strcat(commandFileName,commandFilePath);
				ResolvePartialPathFromThisFolderPath(commandFileName,workingDirectoryPath);
				strcpy(commandFilePath,commandFileName);
			}
		}
		if (FileExists(0,0,commandFilePath))
		{
			if(IsCommandFile(commandFilePath))
			{
		#ifndef NO_GUI 
				UpdateMapWindow();// so the tool bar draws etc
		#endif
				err = DoCommandFile(commandFilePath);
				goto skipInterface; // we assume they will want the program to quit when finished
			}
			else
			{
				char msg[256];
				sprintf(msg,"Selected file is not a command file %s",commandFilePath);
				printNote(msg);
			}
		}
		else
		{
			char msg[256];
			sprintf(msg,"Command file not found at %s",commandFilePath);
			printNote(msg);
		}
	}
	/// check for command file 
	if(FileExists(TATvRefNum, TATdirID,STDCOMMANDFILENAME))
	{
		char commandFilePath[256];
		GetDirectoryFromID(TATdirID, commandFilePath);
		strcpy(gWorkingDirectory,commandFilePath);
		strcat(commandFilePath, "COMMAND.TXT"); // JLM 1/14/03 for some reason my compiler didn't like strcat(commandFilePath, STDCOMMMANDFILENAME);
		if(IsCommandFile(commandFilePath))
		{
		#ifndef NO_GUI
			UpdateWindow(mapWindow);// so the tool bar draws etc
		#endif
			err = DoCommandFile(commandFilePath);
			goto skipInterface; // we assume they will want the program to quit when finished
		}
	}

#ifndef NO_GUI
	if (Greet()) done = TRUE;
#endif

	if(!done)
	{
		if(model->fWizard->StartUp())  done = TRUE;
	}
	
	// FullScreenMapWindow();
	// ShowWindow(mapWindow, nCmdShow);
#ifndef NO_GUI
	UpdateWindow(mapWindow);
#endif
	//timer = SetTimer(mapWindow, 1, 250, NULL);
	timer = SetTimer(mapWindow, 1, 1000, NULL);	// code goes here, should make sure timer is longer than DoubleClickTime
	
	if (!timer)
		printNote(GS1(138, 0)); // MARPLOT will be unable to flash the Focus Point
								// because too many clocks/timers are in use.
	
	//if (gNoaaVersion && model->GetModelMode() == ADVANCEDMODE) printNote("GNOME has found a noaa.ver file. Spills will be allowed outside the spillable area.");

	while (!done && !settings.quitting &&
		   (FALSE /* PeekMessage(&msg, NULL, WM_DDE_FIRST, WM_DDE_LAST, TRUE) */ ||
		    GetMessage(&msg, NULL, 0, 0))) {
		if (GetFocus() == mapWindow &&
			msg.message != WM_MOUSEMOVE && msg.message != WM_NCMOUSEMOVE)
			DoTasks();
		//if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg)) {
		if (!TranslateAccelerator(msg.hwnd, (HACCEL)hAccelTable, &msg)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}
	
	skipInterface:
		
		DeleteList();
		SetPort(oldPort);
		ReleaseDirectoryTable();
		_DeleteAllHandles();
		
		return done ? TRUE : msg.wParam;
}

#endif
