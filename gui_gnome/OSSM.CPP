
#ifdef MAC
void MyDebugSysBeep(short x)
{
	SysBeep(5);
}
#endif

#include "CROSS.h"
#include "Wizard.h"
#include "Overlays.h"


#ifdef MAC
#ifdef MPW
#pragma SEGMENT OSSM
#endif
#endif

//Boolean makingMovie = FALSE;
char TATFolder[32], mssg[256];
char gApplicationFolder[256];
char gWorkingDirectory[256];
short TATvRefNum;
long TATdirID,
	 sixteen = 16, eight = 8, five = 5, four = 4, three = 3, two = 2, one = 1, zero[] = { 0, 0, 0, 0 };
//OSErr err;

WorldPoint focusPoint = { -1, -1 };
WorldRect previousView;
Settings settings;

WindowPtr mapWindow;
DialogPtr progressDialog = 0;
Handle rainyDayReserve = 0;
VList objects;

#ifndef pyGNOME
TModel *model = nil;
#endif

/////////////////////////////////////////////////////////////////////////////

Boolean UseExtendedYears(void)
{
	if(gNoaaVersion &&  model->GetModelMode() == ADVANCEDMODE)
		return true;
	else
		return false;
	return true;	
}

short FirstYearInPopup(void)
{
	if(UseExtendedYears())
		return FIRSTYEARINPOPUP_EXTENDED;
	else
		return FIRSTYEARINPOPUP;
}



static Boolean	gInvalidListLength = true;
void InvalListLength(void)
{
	// use of this function reduced flashing , JLM
	gInvalidListLength = true;
}

void ValidateListLength(void)
{
	if(gInvalidListLength)
		UpdateListLength(false);
	gInvalidListLength = false;
}

void UpdateListLength(Boolean autoScroll)
{	// use oldLength = -2 to indicate no autoscroll
	long n, newLength = model->GetListLength();
	gInvalidListLength = false; // we have validated it
	
	Boolean haveSelection = VLGetSelect(&n, &objects);
	
	// JLM  2/10/99
	// we use VLResetNR instead of VLReset for two reasons
	// (1) we don't want force a reset of L->curRec
	// and (2) we don't need to draw becaue we invalidate the rectangle below
	//VLReset(&objects, newLength);
	VLResetNR(&objects, newLength);
	if(haveSelection) // JLM 2/10/99
	{
		VLSetSelect(n, &objects);
		if(autoScroll) VLAutoScroll(&objects);
	}
	
	InvalRectInWindow(ListDrawingRect(), mapWindow);
	// JLM 2/11/99, on the IBM the VList is a separate window
	// and it needs and invalidate, i.e. we should use VLInvalidate !!!
	VLInvalidate(&objects);
	
}

Boolean EqualListItems(ListItem item1, ListItem item2)
{
	return item1.owner == item2.owner && item1.index == item2.index;
}

void SelectListItem(ListItem item)
{
	char s[256];
	long i;
	short style = normal;
	ListItem item2;
	
	for (i = 0 ; i < objects.numItems ; i++) {
		item2 = model->GetNthListItem(i, 0, &style, s);
		if (EqualListItems(item2, item)) {
			VLSetSelect(i, &objects);
			break;
		}
	}
}


Boolean SelectedListItem(ListItem *item)
{
	long n;
	short style = normal;
	char s[256];
	Boolean itemSelected;
	ListItem localItem;

	memset(&localItem,0,sizeof(localItem));
	
	itemSelected = VLGetSelect(&n, &objects);
	if (!itemSelected) 
		return FALSE;
	
	localItem = model->GetNthListItem(n, 0, &style, s);
	if (!localItem.owner) 
		return FALSE;
	
	if (item) *item = localItem;	
	return TRUE;
}

Boolean SelectListItemOfOwner(TClassID *owner)
{	// returns true if an item is found and selected
	long n;
	short style = normal;
	char s[256];
	Boolean itemSelected;
	ListItem localItem;
	long numItems;
	
	if(gInWizard)
		return TRUE; // when the wizard calls , fake them out
		
	// we need to make sure the spills item is open if we are going to select the item
	// of the added spill, and similarly for the maps
	if(owner->IAm(TYPE_LELIST))
		model->fSpillsOpen = true;
	if(owner->IAm(TYPE_MAP))
		model->mapsOpen = true;	
	
	ValidateListLength(); // make sure our list is up to date

	memset(&localItem,0,sizeof(localItem));
	//  first check the selected item
	itemSelected = VLGetSelect(&n, &objects);
	if (itemSelected)  {
		localItem = model->GetNthListItem(n, 0, &style, s);
		if (localItem.owner == owner) 
			return TRUE;
	}

	// then go through the list until we find an item with the right owner
	numItems = model -> GetListLength();
	for(n= 0; n <numItems; n++)
	{
		localItem = model->GetNthListItem(n, 0, &style, s);
		if (localItem.owner == owner) {
			VLSetSelect(n, &objects);
			InvalToolBarRect();// so the buttons show the correct enabling
			return TRUE;
		}
	}
	return FALSE;// no items by that owner
}


void ListClick(Boolean doubleClick)
{
	Boolean inBullet;
	char s[256];
	long n, oldLength;
	Point p;
	Rect box, r = { 0, 0, 0, 0 };
	ListItem item;
	short style = normal;
	
	GetMouse(&p);
	
	DrawTools(BUTTONPICTBASE, -1);
	
	TellPlayersAboutNewListSelection();
	
	if (!VLGetSelect(&n, &objects)) return;
	
	item = model->GetNthListItem(n, 0, &style, s);
	if (!item.owner) return;
	
	DrawBullet(item, &r, FALSE, &box);
	
	inBullet = p.h >= (box.left - 4) && p.h <= (box.right + 4);
	
	oldLength = objects.numItems;
	
	// code ges here, with the new method, we no longer need this code checking UserIsInEditMode, right ??
	//if(doubleClick && UserIsInEditMode())
	//{ // don't let them get to settings, add objects etc
		//printError("You must finish editing the splots before double clicking summary list items.  Choose the arrow tool when you are done editing the splots.");
		//doubleClick = false; // let it be prcessed as a single click
	//}
	
	if (item.owner->ListClick(item, inBullet, doubleClick))
	{
		//UpdateListLength(oldLength);
		model->NewDirtNotification(DIRTY_LIST);// JLM, 1/6/99 can we get away with dirty list ?
	}
}

Boolean OSSMButtonEnabled(short buttonID)
{
	Boolean itemSelected;
	char s[256];
	long n;
	ListItem item;
	short	style = normal;
	
	switch (buttonID) {
		case RESETBUTTON: 
		{
			if (!(model->bHindcast))
				return model -> GetModelTime () != model -> GetStartTime ();
			else
				return model -> GetModelTime () != model -> GetEndTime ();
		}
		case PLAYBUTTON:
		case STEPBUTTON:  
		{
			if (!(model->bHindcast))
				return model -> GetModelTime () < model -> GetEndTime ();
			else
				return model -> GetModelTime () > model -> GetStartTime ();
		}
 	}

	itemSelected = VLGetSelect(&n, &objects);
	
	if (itemSelected) {
		item = model->GetNthListItem(n, 0, &style, s);
		if (!item.owner) return FALSE;
	}
	
	switch (buttonID) {
		case ADDBUTTON:
		case SETTINGSBUTTON:
		case DELETEBUTTON:
			return itemSelected && item.owner->FunctionEnabled(item, buttonID);
		case UPBUTTON:
		case DOWNBUTTON:
			return itemSelected && item.owner->FunctionEnabled(item, buttonID);
	}
	
	return TRUE;
}

OSErr UpItem()
{
	char s[256];
	long n;
	ListItem item;
	short	style = normal;
	
	if (!OSSMButtonEnabled(UPBUTTON)) return 0;
	
	if (VLGetSelect(&n, &objects)) {
		item = model->GetNthListItem(n, 0, &style, s);
		if (item.owner)
			return item.owner->UpItem(item);
	}
	
	return 0;
}

OSErr DownItem()
{
	char s[256];
	long n;
	ListItem item;
	short	style = normal;
	
	if (!OSSMButtonEnabled(DOWNBUTTON)) return 0;
	
	if (VLGetSelect(&n, &objects)) {
		item = model->GetNthListItem(n, 0, &style, s);
		if (item.owner)
			return item.owner->DownItem(item);
	}
	
	return 0;
}

OSErr AddItem()
{
	char s[256];
	long n;
	ListItem item;
	short	style = normal;
	
	if (!OSSMButtonEnabled(ADDBUTTON)) return 0;
	
	if (VLGetSelect(&n, &objects)) {
		item = model->GetNthListItem(n, 0, &style, s);
		if (item.owner)
		{
			model->NewDirtNotification();// JLM
			return item.owner->AddItem(item);
		}
	}
	
	return 0;
}

OSErr SettingsItem()
{
	char s[256];
	long n;
	ListItem item;
	short	style = normal;
	
	if (!OSSMButtonEnabled(SETTINGSBUTTON)) return 0;
	
	if (VLGetSelect(&n, &objects)) {
		item = model->GetNthListItem(n, 0, &style, s);
		if (item.owner)
		{
			model->NewDirtNotification(DIRTY_LIST);// JLM 1/6/98 , just dirty list ??
			return item.owner->SettingsItem(item);
		}
	}
	
	return 0;
}

OSErr DeleteItem()
{
	char s[256];
	long n;
	ListItem item;
	short	style = normal;
	
	if (!OSSMButtonEnabled(DELETEBUTTON)) return 0;
	
	if (VLGetSelect(&n, &objects)) {
		item = model->GetNthListItem(n, 0, &style, s);
		if (item.owner && CHOICEALERT(M76, 0, FALSE))
			if (!item.owner->DeleteItem(item)) {
				item.owner->Dispose();
				delete item.owner;
				model->NewDirtNotification();// JLM
			}
			else
				return -1;
	}
	
	return 0;
}

///////////////////////////////////////////////////////////////////////

char savePath[256] = "";

void printSaveFileVersionError(void)
{
	printError("This version of GNOME cannot read this save file.");
}

void ResetWindowTitle(char *path)
{
	char s[256], fileName[32], title[256];
	
	if (path != savePath)
		strcpy(savePath, path);
	
	strcpy(title, BASIC_WINDOW_TITLE_STR);
	// Note: we could use ABBREVIATED_WINDOW_TITLE_STR
	if (savePath[0]) {
		strcpy(s, savePath);
		SplitPathFile(s, fileName);
		strcat(title, " - ");
		strcat(title, fileName);
	}
	setwtitle(mapWindow, title);
}

OSErr SaveSaveFile(char *path)
{
	OSErr err = 0;
	char s[256];
	
	SetWatchCursor();
	sprintf(s, "Writing file: %s...", path);
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(s);
	
	if (err = model->WriteToPath(path)) goto done;
	
	done:
		
		if (err)
			hdelete(0, 0, path);
		
		DisplayMessage(0);
		InitCursor();
		
		return err;
}


OSErr NewTATFile() { return 0; }

Boolean IsGnomeSaveFile(char* path)
{ // Check first two values - classId and version number for now
	OSErr	err = noErr;
	long	line;
	char	firstPartOfFile [256];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		long   *classIdP;
		long   *versionP;
		long 	classId;
		long	version;
		
		classIdP = (long*)firstPartOfFile;		
		classId = (long)SwapLong(*(unsigned long *)classIdP);
		classIdP = &classId; //IBM insists on two steps
		if( *classIdP != TYPE_MODEL)
			return false;

		versionP = (long*)(firstPartOfFile+sizeof(long));
		version = (long)SwapLong(*(unsigned long *)versionP);
		versionP = &version;
		if(*versionP < 1)
			return false;		
	}
	
	return true;
}

OSErr GetFilePath(char *path)
{ // Note: returns USERCANCEL when user cancels
	short 			item;
	short resID = M38g; 
	Point 			where = CenteredDialogUpLeft (resID);
	OSType 		typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 		reply;
	OSErr			err = noErr;
	
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, resID, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);
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
#else
	strcpy(path, reply.fName);
#endif
#endif
	return err;
}


OSErr OpenSaveFile(void)
{
	short resID = M38g; //JLM 12/30/98,  was M58;
	char path[256];
	Point where = CenteredDialogUpLeft(resID);
	OSType typeList[] = { WIZARDSAVEFILETYPE, WIZARDFILETYPE, 'SAVE', '.SAV' };
	MySFReply reply;
	char prompt[256] = ""; // JLM 12/30/98
	OSErr err = 0;
	long numTypes = 4;
	
#ifdef MAC
	if (model->GetModelMode() == ADVANCEDMODE)
	{
		numTypes = -1;
	}
	//else numTypes = 2;	// let standard users open .SAV files
#else
	if (model->GetModelMode() == ADVANCEDMODE && gNoaaVersion)
	{
		numTypes = -1;
	}
#endif

#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", numTypes, typeList,
				   (MyDlgHookUPP)0, &reply, resID,MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);
#else
	paramtext(prompt,"","","");
	sfpgetfile(&where, "",
			   (FileFilterUPP)0,
			   numTypes, typeList,
			   (DlgHookUPP)0,
			   &reply, resID,
			   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	
	if (!reply.good) return USERCANCEL;
	
	my_p2cstr(reply.fName);
#ifdef MAC
	GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
#else
	strcpy(path, reply.fName);
#endif
#endif	
	err = OpenSaveFileFromPath(path);  /// JLM HandleODOC
	return err;
}

//////////////////////////////
////////////////////////////

char gCurrentPathOfSaveFile[256]; // JLM 6/3/10

void ResolvePathFromSavedFile(char* pathToResolveFromSaveFile) // JLM 6/3/10
{
	if(gCurrentPathOfSaveFile[0])
		ResolvePathFromInputFile(gCurrentPathOfSaveFile,pathToResolveFromSaveFile);
}	

OSErr OpenSaveFileFromPath(char * path) /// JLM HandleODOC
{
	char m[300],fileName[256],shortName[256];
	Boolean askUserAboutSaving, bCloseSaveFile = true;
	long nameLen;
	OSErr err = 0;
	Boolean haveOpenWizFile = (model->fWizard && model->fWizard->HaveOpenWizardFile());
	
	if(!FileExists(0, 0, path))
		return -2;

	if(IsWizardFile(path)) 
	{
		model->fWizard->OpenWizFile(path);
		return noErr;
	}
	
	if(IsCommandFile(path))
	{
		err = DoCommandFile(path);
		return err;
	}

	if(IsOverlayFile(path)) { // JLM 6/4/10, let the user open it any way they want
		err = AddOverlayFromFile(path);
		if(err) return err;
		if(NumFilesSelected() > 1) { 
			long i;
			long n = NumFilesSelected();
			for(i=2; i <= n; i++) {
				strcpyNthFile(path,i); // let's assume they are all of the correct type
				err = AddOverlayFromFile(path);
				if(err) return err;
			}
		}
		return err;
	}
	
	if(!IsGnomeSaveFile(path))
	{
		printError("This file is not a GNOME save file");
		return -1;
	}

	strcpy(gCurrentPathOfSaveFile,path); // JLM 6/3/10, so we can resolve full paths in the save file using ResolvePathFromSavedFile()
	
	askUserAboutSaving = TRUE;
	
	if (haveOpenWizFile)
	{
		char locationFileNameStr[64];
		strcpy(fileName,path);
		SplitPathFile(fileName, shortName);
		model->fWizard->GetLocationFileName(locationFileNameStr,true);
		if (strstr(shortName,locationFileNameStr))
			bCloseSaveFile = false;
	}

	//if (!haveOpenWizFile)	// need to check that SAV file matches the LF
	if (bCloseSaveFile)
		{if (err = CloseSaveFile(askUserAboutSaving,FALSE)) return err;}
	//if (!gSuppressDrawing)
#ifndef NO_GUI
	SetWatchCursor();
	sprintf(m, "Loading file: %s...", path);
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(m);
#endif
	err = 0;
	{
		char *extStr = "", *asvStr="ASV";
		Boolean isDispersedOilFile = false;
		//GetStrExtension (path, extStr);
		//if (!strncmp(extStr,"ASV",strlen("ASV"))) isDispersedOilFile = true;
		if (strstr(path,asvStr)) isDispersedOilFile = true;
		//if (err = model->ReadFromPath(path)) goto done;
		// code goes here, if err try again, maybe it's an ERA .SAV file
		if (isDispersedOilFile || (err = model->ReadFromPath(path)))  // check model version > 3, then isn't old ERA SAV file
		{
			CloseSaveFile(FALSE,FALSE);
			gMearnsVersion =  true; InitAnalysisMenu();
	#ifndef NO_GUI
		SetWatchCursor();
		sprintf(m, "Loading file: %s...", path);
		DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage(m);
	#endif
			err = model->ReadFromPath(path);
			gMearnsVersion =  false;
			if (err) goto done;
		}
	} 
	if (gMearnsVersion || gDispersedOilVersion)	// code goes here, get rid of mearns version...
	{	// but these may not be set yet..., do below
		strcpy(fileName,path);
		SplitPathFile(fileName, shortName);
		nameLen = strlen(shortName);
		// Chop off extension
		if(strlen(shortName) >= 4) 
		{
			long i;
			for(i=1; i<=4; i++)
			{
				if(shortName[nameLen-i] == '.')
				{	
					shortName[nameLen-i]=0;
					break; // only take off last extension
				}
			}
		}
		strcpy(settings.headerSPILLID,shortName);
	}
	//if (model->GetModelMode() == NOVICEMODE && !bCloseSaveFile)	// need to check that SAV file matches the LF
	if (model->GetModelMode() == NOVICEMODE && bCloseSaveFile)	// need to check that SAV file matches the LF
	{
		printNote("The file you just opened is a Diagnostic Mode .SAV file. Since you are working in GNOME's standard mode you will not be able to interact with the map or the currents. But you will be able to start spills in the domain.");
 	}
	model->SetSaveFileName(path);
	
#ifndef NO_GUI
	ResetWindowTitle(path);
	InvalListLength();
	InvalMapWindow();
#endif

	gCurrentPathOfSaveFile[0] = 0; // done opening this save file
	
	done:
		
#ifndef NO_GUI
		DisplayMessage(0);
#endif
		// Reset Model in case some data was read in before read error occurred
		if(err) CloseSaveFile(FALSE,FALSE);
		//if (!gSuppressDrawing)
#ifndef NO_GUI
		InitCursor();
#endif
		
		return err;
}

OSErr CloseSaveFile(Boolean ask,Boolean quitting)
{
	long i;
	long dialogID = quitting ? M77Q : M77C;
	OSErr err = 0;
	
#ifndef NO_GUI
	if (ask)
	{
		long numMaps = 0;
		Boolean dirtyEnoughToAsk;
		
		numMaps = model -> mapList -> GetItemCount();
		
		dirtyEnoughToAsk = (numMaps > 0 && model -> IsDirty ());
		
		if (dirtyEnoughToAsk)
		{
			short itemHit = MULTICHOICEALERT(dialogID, 0, FALSE);
			switch(itemHit)
			{
				case 1: break; // NO, default button
				case 3: return USERCANCEL; //CANCEL button
				case 6:
					err = AdvancedSaveFileSaveAs(); //YES button
					if(err) return err;
					break;
			}
		}
	}
	
	ResetWindowTitle("");
	InvalListLength();
	InvalidateMapImage();  // JLM 11/30/98
	InvalMapWindow();
#endif

	model->Dispose();
	if (err = model->InitModel())
		{ TechError("CloseSaveFile()", "InitModel()", err); return err; }
	
	
	return 0;
}

OSErr Print()
{
	return PrintMap();
}

OSErr DoQuit() { return 0; }
OSErr Undo() { return 0; }
OSErr Cut() { return 0; }
OSErr Paste() { return 0; }

///////////////////////////////////////////////////////////////////////

Boolean IsDrifterDataFile(char *path)
{

	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{	//		GMTdatetime,float_id,depth,latitude,longitude // header line
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		if (strstrnocase(strLine,"float_id")) 
			bIsValid = true;
	}
	
	return bIsValid;
}

OSErr ReadOSSMLEFile(char *path, LERecH *array, long *numLEs, short *massUnits, Seconds *fileTime)
{
	Boolean doSwap = FALSE, isMOSS = FALSE;
	char *p, *suffix, s[256], leType[32], pollutantTypeStr[32], status[32];
	short f;
	long i, n=0, count, id;
	float depth, mass, density, ageInHrs, longF, latF;
	CHARH ms3, fH;
	Seconds seconds, ossmRefTimeInSecondsSince1904,currTimeInSecondsSince1904;
	long refYear;
	char refStr[32];
	BFPB ms4, ms5;
	LEHeaderRec header;
	LEPropRec lePropRec;
	LEPropH oldOssmStyleElements = 0;
	LERec leRec;
	LERecH elements = 0;
	OSErr err = 0;
	
	(*array) = 0;
	(*numLEs) = 0;
	(*massUnits) = KILOGRAMS;
	(*fileTime) = 0;
	
	if (suffix = strrchr(path, '.'))
		if (!strncmpnocase(suffix, ".MS", 3))
			//if (isdigit(suffix[3]))
			if (suffix[3]>='0' && suffix[3]<='9')	// for code warrior
				isMOSS = TRUE;
	
	// add option to read in an ascii file with lat,lon,z,time, then convert to LE formatx
	/*if (suffix = strrchr(path, '.'))
		if (!strncmpnocase(suffix, ".MS", 3))
			//if (isdigit(suffix[3]))
			if (suffix[3]>='0' && suffix[3]<='9')	// for code warrior
				isMOSS = TRUE;
	*/

	if (IsDrifterDataFile(path))
	{	// if file is in GMT need to have user input offset
		long float_id, numLines, numScanned, /*offsetFromGMT = -8*/ offsetFromGMT = 0;	// was hardcoded for Alaska, now users can at least use local time, should put in a dialog or file option for offset
		char GMTdateStr[32], GMTtimeStr[32], secStr[32], dateStr[32], drifterIdStr[32];
		if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &fH))
			{ TechError("TOSSMTimeValue::ReadTimeValues()", "ReadFileContents()", 0); goto done; }
			_HLock((Handle)fH);
		numLines = NumLinesInText(*fH);	// 1 header line
		//NthLineInTextOptimized(*fH, i, s, 512); // float_id,depth,GMTdatetime,latitude,longitude
		mass = 1.;
		density = 1.;
		ageInHrs = 0;
		if (!(elements = (LERecH)_NewHandle(0))) 
			{ TechError("ReadLEFile()", "_NewHandle()", 0); err = -1; goto done; }
		
		for (i = 0 ; i < numLines ; i++) {
			NthLineInTextOptimized(*fH, i, s, 512); // GMTdatetime,float_id,depth,latitude,longitude
			if (i==0) continue;
			if(i%200 == 0) MySpinCursor(); 
			RemoveLeadingAndTrailingWhiteSpace(s);
			if(s[0] == 0) continue; // it's a blank line, allow this and skip the line
			StringSubstitute(s, ',', ' ');
			//numScanned=sscanf(s, "%d %f %s %s %f %f",
						 // &float_id, &depth, &GMTdateStr,
						 // &GMTtimeStr, &latF, &longF) ;
			/*numScanned=sscanf(s, "%s %s %d %f %f %f",
						 &GMTdateStr, &GMTtimeStr,  
						 &float_id, &depth, &latF, &longF) ;*/
			numScanned=sscanf(s, "%s %s %s %f %f %f",
						 &GMTdateStr, &GMTtimeStr,  
						 &drifterIdStr, &depth, &latF, &longF) ;
			if (numScanned<6)		
				//{ err = -1; TechError("ReadOSSMLEFile()", "sscanf() < 6", 0); goto done; }
				continue;	// no data for the drifter at this time

				sscanf(GMTdateStr, "%04hd-%02hd-%02hd",
							   &header.currYear,
							   &header.currMonth,
							   &header.currDay);
				sscanf(GMTtimeStr, "%02hd:%02hd:%s",
							   &header.currHour,
							   &header.currMin,
							   &secStr);
				sprintf(dateStr,"%hd/%hd/%hd",header.currMonth,header.currDay,header.currYear);
				seconds = DateString2Secs(dateStr);
				seconds += header.currHour * 3600 + header.currMin * 60;
				seconds += offsetFromGMT*3600;	// need to check if offset changes year...
				currTimeInSecondsSince1904 = seconds;
				
				////////////
				// Bushy says the reference for an ossm file is the previous year divisible by 4
				refYear = (header.currYear - (header.currYear % 4))%100;
				sprintf(refStr,"1/1/%02ld",refYear);
				ossmRefTimeInSecondsSince1904 = DateString2Secs(refStr);
				seconds -= ossmRefTimeInSecondsSince1904;
				///////////////

				header.currTimeInHrsAfterOssmRefTime = seconds/3600.0;

			n = n+1;

			numScanned=sscanf(drifterIdStr, "%d", &float_id) ;
			if (numScanned<1)	// allow users to put whatever they like for an ID, only use in GNOME if it's a long	
				float_id = n;

			leRec.p.pLong = longF * 1000000;
			leRec.p.pLat = latF * 1000000;
			
			leRec.leKey = float_id;
			leRec.leCustomData = 0;
			leRec.pollutantType = OIL_CONSERVATIVE;
			//leRec.pollutantType = OldToNewPollutantCode(GetOldPollutantNumber(pollutantTypeStr),&isEvaporated); 
			if (depth>=0) leRec.z = depth;
			else leRec.z = 0;
			leRec.mass = mass;
			leRec.density = density;
			
			// code goes here, AGE, we would compute the age here
			// the MOSS file has the AGE of the LE at the time of the MOSS file
			// we need to set the release time to be the MOSS file time
			// (MOSS files do not contain unreleased LE's)
			// implemented 8/18/10
			leRec.releaseTime = currTimeInSecondsSince1904;
			leRec.ageInHrsWhenReleased = ageInHrs;
			
			leRec.clockRef = 0;
			// issue if on_land		
			leRec.statusCode = OILSTAT_INWATER;
			/*leRec.statusCode = GetStatusCode(status); // note we don't use isEvaporated because evaporated is not possible , right??, 
			if (leRec.statusCode == OILSTAT_ONLAND) {
				leRec.lastWaterPt = leRec.p; // assume point is beached right on shoreline
				leRec.beachTime = currTimeInSecondsSince1904; // assume point was just beached
			}*/
			
			//n++;
			
			_SetHandleSize((Handle)elements, n * sizeof(LERec));
			//_SetHandleSize((Handle)elements, (i+1) * sizeof(LERec));
			if (_MemError())
				{ TechError("ReadLEFile()", "_SetHandleSize()", 0); err = -1; goto done; }
			
			INDEXH(elements, n-1) = leRec;
			//INDEXH(elements, i) = leRec;

		}
		_HUnlock((Handle)fH);
		DisposeHandle((Handle)fH);
		//header.numRecords = i;
		header.numRecords = n;
		goto done;
	}
	
	if (isMOSS) {
		// set up the header
		header.currDay = 0;
		header.currMonth = 0;
		header.currYear = 0;
		header.currHour = 0;
		header.currMin = 0;
		header.currTimeInHrsAfterOssmRefTime = 0;
		header.version = 0;
		
		strcpy(suffix, ".MS3");
		if (!ReadFileContents(TERMINATED,0, 0, path, 0, 0, &ms3)) {
			_HLock((Handle)ms3);
			if (p = strstr(DEREFH(ms3), "0, VALIDFOR:")) {
				// 0430, 7/14/95
				sscanf(&p[12], "%02hd%02hd, %02hd/%02hd/%02hd",
							   &header.currHour,
							   &header.currMin,
							   &header.currMonth,
							   &header.currDay,
							   &header.currYear);
				seconds = DateString2Secs(&p[19]);
				seconds += header.currHour * 3600 + header.currMin * 60;
				currTimeInSecondsSince1904 = seconds;
				
				////////////
				// Bushy says the reference for an ossm file is the previous year divisible by 4
				refYear = (header.currYear - (header.currYear % 4))%100;
				sprintf(refStr,"1/1/%02ld",refYear);
				ossmRefTimeInSecondsSince1904 = DateString2Secs(refStr);
				seconds -= ossmRefTimeInSecondsSince1904;
				///////////////

				header.currTimeInHrsAfterOssmRefTime = seconds/3600.0;
			}
			_HUnlock((Handle)ms3);
			DisposeHandle((Handle)ms3);
		}
		
		ms4.f = 0;
		strcpy(suffix, ".MS4");
		if (err = FSOpenBuf(0, 0, path, &ms4, 100000, FALSE))
			{ TechError("ReadLEFile()", "FSOpenBuf()", err); return -1; }
		
		ms5.f = 0;
		strcpy(suffix, ".MS5");
		if (err = FSOpenBuf(0, 0, path, &ms5, 100000, FALSE))
			{ TechError("ReadLEFile()", "FSOpenBuf()", err); err = -1; goto done; }
		
		if (!(elements = (LERecH)_NewHandle(0))) 
			{ TechError("ReadLEFile()", "_NewHandle()", 0); err = -1; goto done; }
		
		for (n = 0 ; ; ) {
			Boolean isEvaporated;
			if (!myfgetsbuf(s, 255, &ms4)) // header line
				break; // done reading file
			
			if (!myfgetsbuf(s, 255, &ms4)) // data line
				{ TechError("ReadLEFile()", "myfgetsbuf()", 0); err = -1; goto done; }
			
			sscanf(s, "%f %f", &longF, &latF);
			leRec.p.pLong = longF * 1000000;
			leRec.p.pLat = latF * 1000000;
			
			if (!myfgetsbuf(s, 255, &ms5)) // data line
				{ TechError("ReadLEFile()", "myfgetsbuf()", 0); err = -1; goto done; }
			
			StringSubstitute(s, ',', ' ');
			sscanf(s, "%ld %s %s %f %f %f %f %s",
					  &id, leType, pollutantTypeStr, &depth, &mass, &density, &ageInHrs, status);
			
			leRec.leKey = id;
			leRec.leCustomData = 0;
			leRec.pollutantType = OldToNewPollutantCode(GetOldPollutantNumber(pollutantTypeStr),&isEvaporated); 
			leRec.z = depth;
			leRec.mass = mass;
			leRec.density = density;
			
			// code goes here, AGE, we would compute the age here
			// the MOSS file has the AGE of the LE at the time of the MOSS file
			// we need to set the release time to be the MOSS file time
			// (MOSS files do not contain unreleased LE's)
			// implemented 8/18/10
			leRec.ageInHrsWhenReleased = ageInHrs;
			leRec.releaseTime = currTimeInSecondsSince1904;
			
			leRec.clockRef = 0;
						
			leRec.statusCode = GetStatusCode(status); // note we don't use isEvaporated because evaporated is not possible , right??, 
			if (leRec.statusCode == OILSTAT_ONLAND) {
				leRec.lastWaterPt = leRec.p; // assume point is beached right on shoreline
				leRec.beachTime = currTimeInSecondsSince1904; // assume point was just beached
			}
			
			n++;
			
			_SetHandleSize((Handle)elements, n * sizeof(LERec));
			if (_MemError())
				{ TechError("ReadLEFile()", "_SetHandleSize()", 0); err = -1; goto done; }
			
			INDEXH(elements, n - 1) = leRec;
		}
		
		header.numRecords = n;
		
		goto done;
	}
	
	if (err = hopendf(0, 0, path, fsCurPerm, &f))
		{ TechError("ReadLEFile()", "hopendf()", err); goto done; }
	
	count = sizeof(LEHeaderRec);
	if (err = FSRead(f, &count, &header))
		{ TechError("ReadLEFile()", "FSRead()", err); FSClose(f); goto done; }
	
//#ifdef IBM
#ifdef SWAP_BINARY
	if (header.currDay >= 256) doSwap = TRUE;
	if (doSwap) {
		header.currDay = (short)SwapShort((unsigned short)header.currDay);
		header.currMonth = (short)SwapShort((unsigned short)header.currMonth);
		header.currYear = (short)SwapShort((unsigned short)header.currYear);
		header.currHour = (short)SwapShort((unsigned short)header.currHour);
		header.currMin = (short)SwapShort((unsigned short)header.currMin);
		SwapFloat(&header.currTimeInHrsAfterOssmRefTime);
		SwapFloat(&header.version);
		header.numRecords = (long)SwapLong(*(unsigned long *)&header.numRecords);
	}
#endif
	
/////////////////////////////////////////////////
// resolve the OSSM reference time
	////////////
	// Bushy says the reference for an ossm file is the previous year divisible by 4
	// We can also just subtract the currTimeInHrsAfterOssmRefTime field
	// the first method should be more acturate, but the ref time is not actually well defined
	// so we go with the other method
	sprintf(refStr,"%02ld/%02ld/%02ld",header.currMonth,header.currDay,header.currYear);
	seconds = DateString2Secs(refStr);
	seconds += header.currHour * 3600 + header.currMin * 60;
	currTimeInSecondsSince1904 = seconds;
	ossmRefTimeInSecondsSince1904 = currTimeInSecondsSince1904 - (long)(3600 * header.currTimeInHrsAfterOssmRefTime);
	///////////////
/////////////////////////////////////////////////

	count = header.numRecords * sizeof(LEPropRec);
	if (!(oldOssmStyleElements = (LEPropH)_NewHandle(count)))
		{ TechError("ReadLEFile()", "_NewHandle()", 0); FSClose(f); err = -1; goto done; }
	
	_HLock((Handle)oldOssmStyleElements);
	err = FSRead(f, &count, DEREFH(oldOssmStyleElements));
	_HUnlock((Handle)oldOssmStyleElements);
	FSClose(f);
	
	if (err)
		{ TechError("ReadLEFile()", "FSRead()", err); goto done; }
	
//#ifdef IBM
#ifdef SWAP_BINARY
	if (doSwap)
		for (i = 0 ; i < header.numRecords ; i++) {
			SwapFloat(&INDEXH(oldOssmStyleElements, i).pLat);
			SwapFloat(&INDEXH(oldOssmStyleElements, i).pLong);
			SwapFloat(&INDEXH(oldOssmStyleElements, i).releaseTimeInHrsAfterRefTime);
			SwapFloat(&INDEXH(oldOssmStyleElements, i).ageWhenReleasedInHrsAfterReleaseTime);
			SwapFloat(&INDEXH(oldOssmStyleElements, i).beachHeight);
			INDEXH(oldOssmStyleElements, i).nMap = (long)SwapLong((unsigned long)INDEXH(oldOssmStyleElements, i).nMap);
			INDEXH(oldOssmStyleElements, i).pollutant = (long)SwapLong((unsigned long)INDEXH(oldOssmStyleElements, i).pollutant);
			INDEXH(oldOssmStyleElements, i).windKey = (long)SwapLong((unsigned long)INDEXH(oldOssmStyleElements, i).windKey);
		}
#endif
	
//	if (!(elements = (LERecH)_NewHandle(count * sizeof(LERec))))
	if (!(elements = (LERecH)_NewHandle(header.numRecords * sizeof(LERec))))
		{ TechError("ReadLEFile()", "_NewHandle()", 0); err = -1; goto done; }

	for (i = 0 ; i < header.numRecords ; i++) {
		Boolean isEvaporated;
		INDEXH(elements, i).p.pLong = -INDEXH(oldOssmStyleElements, i).pLong * 1000000;
		INDEXH(elements, i).p.pLat  =  INDEXH(oldOssmStyleElements, i).pLat  * 1000000;
		
		INDEXH(elements, i).leKey = i + 1;
		INDEXH(elements, i).leCustomData = 0;
		INDEXH(elements, i).pollutantType = OldToNewPollutantCode(INDEXH(oldOssmStyleElements, i).pollutant,&isEvaporated);
		INDEXH(elements, i).z = 0;
		INDEXH(elements, i).mass = 1;
		//INDEXH(elements, i).density = 0.95; 
		INDEXH(elements, i).density = GetPollutantDensity(INDEXH(elements, i).pollutantType);//1/23/03
		
		/////////////////////////////////////////////////
		
		seconds = (long) (INDEXH(oldOssmStyleElements, i).releaseTimeInHrsAfterRefTime * 3600);
		seconds += ossmRefTimeInSecondsSince1904;
		INDEXH(elements, i).releaseTime = seconds; // releaseTime is in seconds since 1904
		
		/////////////////////////////////////////////////
		// deal with age and release time here
		{
			double releaseTimeInHrsAfterRefTime = INDEXH(oldOssmStyleElements, i).releaseTimeInHrsAfterRefTime;
			if (seconds > currTimeInSecondsSince1904) {
				// the LE is not yet released, use the release time in specifed in the file
				INDEXH(elements, i).releaseTime = seconds; // releaseTime is in seconds since 1904
				INDEXH(elements, i).ageInHrsWhenReleased = INDEXH(oldOssmStyleElements, i).ageWhenReleasedInHrsAfterReleaseTime;
			}
			else {
				// it was released in the past
				// use the file time as the release time
				INDEXH(elements, i).releaseTime = currTimeInSecondsSince1904; 
				INDEXH(elements, i).ageInHrsWhenReleased = (currTimeInSecondsSince1904 - seconds)/3600.0 + INDEXH(oldOssmStyleElements, i).ageWhenReleasedInHrsAfterReleaseTime;
			}
		}
		/////////////////////////////////////////////////

		
		INDEXH(elements, i).clockRef = 0;

		INDEXH(elements, i).statusCode = OILSTAT_INWATER;
		// old OSSM added 10 to the pollutant code to indicate it had evaporated
		if (isEvaporated) 
			INDEXH(elements, i).statusCode = OILSTAT_EVAPORATED;
		if (INDEXH(oldOssmStyleElements, i).nMap == OLD_OSSM_OFFMAPS)
			INDEXH(elements, i).statusCode = OILSTAT_OFFMAPS;
//		if (INDEXH(oldOssmStyleElements, i).releaseTimeInHrsAfterRefTime > header.currTimeInHrsAfterOssmRefTime)
		// compare release time to the model time and set released flag accordingly, STH 7/1/99
		if (INDEXH(elements, i).releaseTime > model -> GetModelTime())
			INDEXH(elements, i).statusCode = OILSTAT_NOTRELEASED;
		if (INDEXH(oldOssmStyleElements, i).beachHeight == OLD_OSSM_BEACHED) {
			INDEXH(elements, i).statusCode = OILSTAT_ONLAND;
			INDEXH(elements, i).lastWaterPt = INDEXH(elements, i).p; // assume point is beached right on shoreline
			INDEXH(elements, i).beachTime = currTimeInSecondsSince1904; // assume point was just beached
		}
	}

	done:

		if (isMOSS) {
			if (ms4.f) FSCloseBuf(&ms4);
			if (ms5.f) FSCloseBuf(&ms5);
		}
		else
			if (oldOssmStyleElements) DisposeHandle((Handle)oldOssmStyleElements);
		
		if (err) {
			if (elements) DisposeHandle((Handle)elements);
		}
		else {
			(*array) = elements;
			(*numLEs) = header.numRecords;
			(*fileTime) = currTimeInSecondsSince1904;
		}
		
		return err;
}

////////////////////////////////////////////////////////////////////////

#define		kUnknownStr		"Unknown"
#define		kGasStr			"Gasoline"
#define		kJetFuelsStr	"Kerosene / Jet Fuels"
#define		kDieselStr		"Diesel"
#define		k4Str			"Fuel Oil #4"
#define		kCrudeStr		"Medium Crude"
#define		k6Str			"Fuel Oil #6"
#define 	kUserPol1Str	"User Pollutant1"
#define 	kUserPol2Str	"User Pollutant2"
#define		kConservStr		"Non-Weathering"
#define		kCombinationStr	"Combination Oil"
#define		kChemicalStr	"Chemical"	// eventually separate this

OilType GetPollutantNumber(char *name)
{
	OilType	num = OIL_UNKNOWN;
	
	if (!strcmpnocase(name, kUnknownStr)) num = OIL_UNKNOWN;
	else if (!strcmpnocase(name, kGasStr)) num = OIL_GAS;
	else if (!strcmpnocase(name, kJetFuelsStr)) num = OIL_JETFUELS;
	else if (!strcmpnocase(name, kDieselStr)) num = OIL_DIESEL;
	else if (!strcmpnocase(name, k4Str)) num = OIL_4;
	else if (!strcmpnocase(name, kCrudeStr)) num = OIL_CRUDE;
	else if (!strcmpnocase(name, k6Str)) num = OIL_6;
	else if (!strcmpnocase(name, kUserPol1Str)) num = OIL_USER1;
	else if (!strcmpnocase(name, kUserPol2Str)) num = OIL_USER2;
	else if (!strcmpnocase(name, kConservStr)) num = OIL_CONSERVATIVE;
	else if (!strcmpnocase(name, kCombinationStr)) num = OIL_COMBINATION;
	else if (!strcmpnocase(name, kChemicalStr)) num = CHEMICAL;	// added for Bushy
	
	return num;
}

//JLE

// Densities in gm/cm^3
double GetPollutantDensity(OilType num)
{
	double density=1;
	switch(num)
	{
		case OIL_GAS:density=.75; break;
		case OIL_JETFUELS: density = .81; break;
		case OIL_DIESEL: density=.87; break;
		case OIL_4: density=.90; break;
		case OIL_CRUDE: density = .90;break;
		case OIL_6: density = .99; break;
		case OIL_CONSERVATIVE: density=1;break;
		case CHEMICAL: 	density=1; break;	// will need to override calls with input value, or calculated value
		default:printError("Pollutant type not found");
	}
	return density;
}
	   
void GetPollutantName(OilType num, char *name)
{
	switch (num) {
		case OIL_GAS: strcpy(name, kGasStr); break;
		case OIL_JETFUELS: strcpy(name, kJetFuelsStr); break;
		case OIL_DIESEL: strcpy(name, kDieselStr); break;
		case OIL_4: strcpy(name, k4Str); break;
		case OIL_CRUDE: strcpy(name, kCrudeStr); break;
		case OIL_6: strcpy (name, k6Str); break;
		case OIL_USER1: strcpy (name, kUserPol1Str); break;
		case OIL_USER2: strcpy (name, kUserPol2Str); break;
		case OIL_CONSERVATIVE: strcpy (name, kConservStr); break;
		case OIL_COMBINATION: strcpy (name, kCombinationStr); break;
		case CHEMICAL: strcpy (name, kChemicalStr); break;
		default: strcpy(name, kUnknownStr); break;
	}
}

short GetOldPollutantNumber(char *name)
{
	short num;
	
	if (!strcmpnocase(name, "GAS")) num = 1;
	else if (!strcmpnocase(name, "JP4")) num = 2;
	else if (!strcmpnocase(name, "DIESEL")) num = 3;
	else if (!strcmpnocase(name, "IFO")) num = 4;
	else if (!strcmpnocase(name, "MEDIUMCRUDE")) num = 5;
	else if (!strcmpnocase(name, "BUNKER")) num = 6;
	else num = 9; // CONSERVATIVE
	
	return num;
}

short OldToNewPollutantCode(short code, Boolean* isEvaporated)
{
	if(code >= OLD_OSSM_EVAPORATED)  {
		*isEvaporated = TRUE;
		code -= OLD_OSSM_EVAPORATED; // it was evaporated, so we sutract that code
	}
	else {
		*isEvaporated = FALSE;
	}
	
	switch (code) {
		case OSSMOIL_GAS: return OIL_GAS;
		case OSSMOIL_JETFUELS: return OIL_JETFUELS; // JP4
		case OSSMOIL_DIESEL: return OIL_DIESEL;
		case OSSMOIL_4: return OIL_4;
		case OSSMOIL_CRUDE: return OIL_CRUDE;
		case OSSMOIL_6: return OIL_6;
		case OSSMOIL_USER1: return OIL_USER1;
		case OSSMOIL_USER2: return OIL_USER2;
		case OSSMOIL_CONSERVATIVE: return OIL_CONSERVATIVE; // IFO
		
		//case OSSMOIL_COMBINATION: return OIL_COMBINATION; //JLM 3/1/99
		
		//case OSSMOIL_UNKNOWN: return OIL_COMBINATION; // there was no such thing as UNKNOWN in old OSSM
		default: return OIL_COMBINATION; // code goes here, is this what we want ?
	}
}

short NewToOldPollutantCode(short code)
{
	short	newCode;
	
	switch (code) {
		case OIL_GAS: newCode = OSSMOIL_GAS; break;
		case OIL_JETFUELS: newCode = OSSMOIL_JETFUELS; break;
		case OIL_DIESEL: newCode = OSSMOIL_DIESEL; break;
		case OIL_4: newCode = OSSMOIL_4; break;
		case OIL_CRUDE: newCode = OSSMOIL_CRUDE; break;
		case OIL_6: newCode = OSSMOIL_6; break;
		case OIL_USER1: newCode = OSSMOIL_USER1; break;
		case OIL_USER2: newCode = OSSMOIL_USER2; break;
		case OIL_CONSERVATIVE: newCode = OSSMOIL_CONSERVATIVE; break;
		case CHEMICAL: newCode = 10; break;	// this is for dispersed oil Gnome only
		//case OIL_COMBINATION: newCode = OSSMOIL_COMBINATION; break; //JLM 3/1/99
		
		default: 
			//printError("Programmer error in NewToOldPollutantCode"); // JLM 3/1/99
			newCode = OSSMOIL_CONSERVATIVE; 
	}
	
	return newCode;
}

#define		kNotReleasedStr	"NOTRELEASED"
#define		kEvaporatedStr	"EVAPORATED"
#define		kInWaterStr		"INWATER"
#define		kOnBeachStr		"ONBEACH"
#define		kOffMapStr		"OFFMAP"

short GetStatusCode(char *text)
{
	short code = OILSTAT_INWATER;
	
	if (!strcmpnocase(text, kNotReleasedStr)) code = OILSTAT_NOTRELEASED;
	else if (!strcmpnocase(text, kEvaporatedStr)) code = OILSTAT_EVAPORATED;
	else if (!strcmpnocase(text, kInWaterStr)) code = OILSTAT_INWATER;
	else if (!strcmpnocase(text, kOnBeachStr)) code = OILSTAT_ONLAND;
	else if (!strcmpnocase(text, kOffMapStr)) code = OILSTAT_OFFMAPS;
	
	return code;
}

void GetStatusText(short code, char *text)
{
	switch (code) {
		case OILSTAT_NOTRELEASED: strcpy(text, kNotReleasedStr); break;
		case OILSTAT_EVAPORATED: strcpy(text, kEvaporatedStr); break;
		case OILSTAT_INWATER: strcpy(text, kInWaterStr); break;
		case OILSTAT_ONLAND: strcpy(text, kOnBeachStr); break;
		case OILSTAT_OFFMAPS: strcpy(text, kOffMapStr); break;
		default: strcpy(text, kInWaterStr); break;
	}
}

char *GetMassString(double mass, short units, char *s)
{	// mass is in kilograms
	char u[10];
	float m;
	
strcpy(s,"GetMassString is not supported");
//	switch (units) {
//		case TONS:
//			strcpy(u, "ton");
//			m = mass / 1000000;
//			break;
//		case POUNDS:
//			strcpy(u, "lb");
//			m = mass * KG2POUNDS;
//			break;
//		default:
//			strcpy(u, "kg");
//			m = mass;
//			break;
//	}
//	sprintf(s, "%f %s", m, u);
	
	return s;
}

char *GetDensityString(double density, char *s)
{
	float d = density;
	
	sprintf(s, "%f g/cc", d);
	
	return s;
}

char *GetStatusString(OilStatus	statusCode, char *s)
{
	switch (statusCode) {
		case OILSTAT_NOTRELEASED: sprintf(s, "Not released"); break;
		case OILSTAT_EVAPORATED: sprintf(s, "Evaporated and Dispersed"); break;
		case OILSTAT_INWATER: sprintf(s, "Floating"); break;
		case OILSTAT_ONLAND: sprintf(s, "Beached"); break;
		case OILSTAT_OFFMAPS: sprintf(s, "Off map(s)"); break;
	}
	
	return s;
}

char *GetDepthString(double z, char *s)
{
	float d = z;
	
	sprintf(s, "%f m", z);
	
	return s;
}


