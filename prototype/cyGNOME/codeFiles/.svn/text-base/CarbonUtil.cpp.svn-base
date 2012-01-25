

#ifdef MAC

#include  "CarbonUtil.h"

#include "CROSS.h"
#include "OSSM.h"


Pattern MyGetQDGlobalsGray(void)
{
	Pattern pat;
	#if TARGET_API_MAC_CARBON
		GetQDGlobalsGray(&pat);
	#else
		pat = qd.gray;
	#endif
	return pat;
}

Pattern MyGetQDGlobalsBlack(void)
{
	Pattern pat;
	#if TARGET_API_MAC_CARBON
		GetQDGlobalsBlack(&pat);
	#else
		pat = qd.black;
	#endif
	return pat;
}

Pattern MyGetQDGlobalsLightGray(void)
{
	Pattern pat;
	#if TARGET_API_MAC_CARBON
		GetQDGlobalsLightGray(&pat);
	#else
		pat = qd.ltGray;
	#endif
	return pat;
}

Pattern MyGetQDGlobalsDarkGray(void)
{
	Pattern pat;
	#if TARGET_API_MAC_CARBON
		GetQDGlobalsDarkGray(&pat);
	#else
		pat = qd.dkGray;
	#endif
	return pat;
}

Pattern MyGetQDGlobalsWhite(void)
{
	Pattern pat;
	#if TARGET_API_MAC_CARBON
		GetQDGlobalsWhite(&pat);
	#else
		pat = qd.white;
	#endif
	return pat;
}

void FillRectWithQDGlobalsGray(Rect * rect)
{
	Pattern pat = MyGetQDGlobalsGray();
	FillRect(rect,&pat);
}

////

void PenPatQDGlobalsGray(void)
{
	Pattern pat = MyGetQDGlobalsGray();
	PenPat(&pat);
}

void PenPatQDGlobalsDarkGray(void)
{
	Pattern pat = MyGetQDGlobalsDarkGray();
	PenPat(&pat);
}

void PenPatQDGlobalsBlack(void)
{
	Pattern pat = MyGetQDGlobalsBlack();
	PenPat(&pat);
}

void PenPatQDGlobalsWhite(void)
{
	Pattern pat = MyGetQDGlobalsWhite();
	PenPat(&pat);
}

///
void MyLUpdateVisRgn(DialogRef theDialog, ListHandle listHdl)
{
	#if TARGET_API_MAC_CARBON
		RgnHandle   visRgn = NewRgn();
		if(visRgn) GetPortVisibleRegion(GetDialogPort(theDialog), visRgn);
		LUpdate(visRgn,listHdl);
		if(visRgn) DisposeRgn(visRgn);	
	#else
		LUpdate(theDialog->visRgn,listHdl);
	#endif
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

/////////////////////////////////////////////////
/////// my version of old MAC functions
/////////////////////////////////////////////////
#if TARGET_API_MAC_CARBON
void UpperText(char* textPtr, short len)
{
	UppercaseText(textPtr,len,smSystemScript);
}


#endif


OSErr MyGetVInfo(short drvNum, StringPtr volNamePtr, short *vRefNumPtr, double *freeBytesPtr)
{
	OSErr err = 0;
	
	if(volNamePtr) volNamePtr[0] = 0;
	if(vRefNumPtr) *vRefNumPtr = 0;
	if(freeBytesPtr) *freeBytesPtr = 0;
	#if TARGET_API_MAC_CARBON
	{
		FSVolumeRefNum volume = kFSInvalidVolumeRefNum; // to index through the volumes
		ItemCount volumeIndex = drvNum; // // hmmm... maybe the drvNum is the index into the drives
		FSVolumeRefNum actualVolume;
		FSVolumeInfoBitmap whichInfo;
		FSVolumeInfo info;
		HFSUniStr255 * volumeName = 0; // we don't ask for this info
		FSRef rootDirectoryFSRef; 

		
		// only ask for the sizes if we need to
		if(freeBytesPtr)
			whichInfo = kFSVolInfoSizes;
		else
			whichInfo = kFSVolInfoNone;
		
		memset(&info,0,sizeof(info));
		err = FSGetVolumeInfo (volume,volumeIndex,&actualVolume,whichInfo,&info,volumeName,&rootDirectoryFSRef);
		if(!err) {
			if(vRefNumPtr) *vRefNumPtr = actualVolume;
			if(freeBytesPtr) *freeBytesPtr = info.freeBytes;
			if(volNamePtr){
				FSSpec fsSpec;
				err = FSGetCatalogInfo(&rootDirectoryFSRef, 0,0,0,&fsSpec,0)  ;
				if(!err) {
					mypstrcpyJM(volNamePtr,(char*)fsSpec.name);
				}
			}
		}
	}
	#else // MACB4CARBON
	{
		char pStr[256];
		short vNum;
		long fBytes;
		
		err = GetVInfo(drvNum,(StringPtr)pStr,&vNum,&fBytes);
		if(!err) {
			if(volNamePtr) mypstrcpyJM(volNamePtr,pStr);
			if(vRefNumPtr) *vRefNumPtr = vNum;
			if(freeBytesPtr) *freeBytesPtr = fBytes;
		}
	}
	#endif
	
	return err;
}



/////////////////////////////////////////////////
/////// my version of lower case functions
/////////////////////////////////////////////////
#if TARGET_API_MAC_CARBON
void drawstring(const char *s)
{
	if(!s) return;
	DrawText(s,0,strlen(s));
}

short stringwidth(char *s)
{
	short w = 0;
	char localStr[256] ="";
	if(s) {
		strcpy(localStr,s);
		my_c2pstr(localStr);
		w = StringWidth((StringPtr)localStr);
	}
	return w;
}

void getindstring(char *theString, short strListID, short index)
{
	GetIndString((StringPtr)theString, strListID, index);
	my_p2cstr(theString);
}

void numtostring(long theNum, char *theString)
{
	NumToString(theNum,(StringPtr)theString);
	my_p2cstr(theString);
}

void stringtonum(char *theString, long *theNum)
{
	char localStr[256] ="";
	strcpy(localStr,theString);
	my_c2pstr(localStr);
	StringToNum((StringPtr)localStr,theNum);
}

Handle getnamedresource(ResType theType, const char *name)
{
	char localStr[256];
	strcpy(localStr,name);
	my_c2pstr(localStr);
	return GetNamedResource(theType,(StringPtr)localStr);
}

void  setwtitle(WindowRef window, char* title)
{
	char localStr[256] ="";
	strcpy(localStr,title);
	my_c2pstr(localStr);
	SetWTitle(window,(StringPtr)localStr);
}

void getwtitle(WindowRef window, char* cStr)
{
	cStr[0] = 0;
	if(!window) return;
	GetWTitle(window,(StringPtr)cStr);
	my_p2cstr(cStr);
}

void paramtext(char* p0,char* p1,char* p2,char* p3)
{
	char localStr0[256] = "";
	char localStr1[256] = "";
	char localStr2[256] = "";
	char localStr3[256] = "";
	if(p0) {strncpy(localStr0,p0,255);localStr0[255] = 0;} 
	if(p1) {strncpy(localStr1,p1,255);localStr1[255] = 0;} 
	if(p2) {strncpy(localStr2,p2,255);localStr2[255] = 0;} 
	if(p3) {strncpy(localStr3,p3,255);localStr3[255] = 0;} 
	// now call the mac param text
	my_c2pstr(localStr0);
	my_c2pstr(localStr1);
	my_c2pstr(localStr2);
	my_c2pstr(localStr3);
	ParamText((StringPtr)localStr0,(StringPtr)localStr1,(StringPtr)localStr2,(StringPtr)localStr3);
}

void getitem(MenuRef menu, short item, char *itemString)
{
	GetMenuItemText(menu,item,(StringPtr)itemString);
	my_p2cstr(itemString);
}

void appendmenu(MenuRef menu, const char *data)
{
	char localStr[256] ="";
	strcpy(localStr,data);
	my_c2pstr(localStr);
	AppendMenu(menu,(StringPtr)localStr);
}

void insmenuitem(MenuRef theMenu, const char *itemString, short afterItem) // insertmenuitem
{
	char localStr[256] ="";
	strcpy(localStr,itemString);
	my_c2pstr(localStr);
	InsertMenuItem(theMenu,(StringPtr)localStr,afterItem);
}

void getfontname(short familyID, char *theName)
{
	GetFontName(familyID,(StringPtr)theName);
	my_p2cstr(theName);
}
	
void getfnum(char *name, short *num)
{
	char localStr[256] ="";
	strcpy(localStr,name);
	my_c2pstr(localStr);
	GetFNum((StringPtr)localStr,num);
}


#endif



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

#if TARGET_API_MAC_CARBON
	PMPrintSession  gPrintSession = 0;
	PMPageFormat  gPageFormat = kPMNoPageFormat;
	PMPrintSettings gPrintSettings = kPMNoPrintSettings;
	Boolean gSessionDocumentIsOpen = false;
#else
	THPrint gPrRecHdl = 0;
#endif

/////////////////////////////////////////////////
OSStatus  OpenPrinterAndValidate(void)
{
	OSStatus			err = noErr;

	#if TARGET_API_MAC_CARBON
	/////////////////////////////////////////////////
		err =  PMCreateSession(&gPrintSession);
		if(err) return err;

		// get and validate the page formatting
		if (gPageFormat == kPMNoPageFormat) {   // Set up a valid PageFormat object
			err = PMCreatePageFormat(&gPageFormat);
			//  Note that PMPageFormat is not session-specific, but calling
			//  PMSessionDefaultPageFormat assigns values specific to the printer
			//  associated with the current printing session.
			if ((err == noErr) && (gPageFormat != kPMNoPageFormat))
				err = PMSessionDefaultPageFormat(gPrintSession, gPageFormat);
		}
		else{
			err = PMSessionValidatePageFormat(gPrintSession, gPageFormat, kPMDontWantBoolean);
		}
		if(err) return err;
		/////////////////////////////////////////////////
		// let's also get and validate the print settings
		if (gPrintSettings == kPMNoPrintSettings)
		{
			err = PMCreatePrintSettings(&gPrintSettings);	
			// Note that PMPrintSettings is not session-specific, but calling
			// PMSessionDefaultPrintSettings assigns values specific to the printer
			// associated with the current printing session.
			if ((err == noErr) && (gPrintSettings != kPMNoPrintSettings))
				err = PMSessionDefaultPrintSettings(gPrintSession, gPrintSettings);
		}
		else {
			err = PMSessionValidatePrintSettings(gPrintSession, gPrintSettings, kPMDontWantBoolean);
		}

		return err;
	
	#else //// MACB4CARBON ////////////////////////////////////////
	/////////////////////////////////////////////////
		PrOpen();
		err = PrError();
		if(err) return err;		
	   
		// always use the global
		if(gPrRecHdl == nil){
			gPrRecHdl  = (THPrint)_NewHandleClear(sizeof(TPrint));
			if(!gPrRecHdl)return memFullErr; 
			PrintDefault(gPrRecHdl);
		}
	
		PrValidate(gPrRecHdl); 	
		// We ignore the returned value  
		
		return(noErr);
	#endif
} 

/////////////////////////////////////////////////

void ClosePrinter(void)
{
#if TARGET_API_MAC_CARBON
	//if(gPageFormat) {PMRelease(gPageFormat); gPageFormat = 0;} 
	// we want to save the pageFormat so the user can get the settings they set in Page Setup 
	
	if(gPrintSettings) {PMRelease(gPrintSettings); gPrintSettings = 0;} 
	if(gPrintSession) {PMRelease(gPrintSession); gPrintSession = 0;} 
	// Apple says... "By not saving print settings between calls to the Print dialog, you ensure that the dialog displays with the appropriate default settings, which is the recommended behavior."
#else 
	PrClose();
#endif
}

/////////////////////////////////////////////////
#if TARGET_API_MAC_CARBON
OSStatus	DetermineNumberOfPagesInDoc(PMPageFormat pageFormat, UInt32* numPages);
#endif


OSStatus DoJobPrintDialog(char * jobNameStr) 
{
	Boolean accepted = false;

#if TARGET_API_MAC_CARBON
	OSStatus  err = 0;
	UInt32 realNumberOfPagesinDoc;
	CFStringRef jobNameRef = 0;

    // Set the job name
	if(jobNameStr) {
		jobNameRef = CFStringCreateWithCString(NULL, jobNameStr,kCFStringEncodingMacRoman);
		if (jobNameRef){
			(void)PMSetJobNameCFString (gPrintSettings,jobNameRef);
			CFRelease (jobNameRef); jobNameRef = 0;
		}
	}

    // Calculate the number of pages required to print the entire document.
   //  err = DetermineNumberOfPagesInDoc(gPageFormat, &realNumberOfPagesinDoc);
	if(err) goto done;

    // Set a valid page range before displaying the Print dialog
    err = PMSetPageRange(gPrintSettings, 1, realNumberOfPagesinDoc);
	if(err) goto done;

    //	Display the Print dialog.
    err = PMSessionPrintDialog(gPrintSession, gPrintSettings, gPageFormat, &accepted);

#else
	#pragma unused(jobNameStr)
	accepted = PrJobDialog(gPrRecHdl);
#endif

done:
	if(!accepted) return -1;
	return noErr;
}

/////////////////////////////////////////////////

void MyPMSessionSetError(OSStatus err)
{
#if TARGET_API_MAC_CARBON
	PMSessionSetError(gPrintSession,err);
#else
	PrSetError(err);
#endif
}

/////////////////////////////////////////////////

#if TARGET_API_MAC_CARBON
OSStatus My_PMSessionBeginDocument(void)
{
	OSStatus err = 0;
	if(!gSessionDocumentIsOpen){
		err = PMSessionBeginDocument(gPrintSession, gPrintSettings, gPageFormat);
	}
	gSessionDocumentIsOpen = true; // it is not clear if we should check for the error
	return err;
}

OSStatus My_PMSessionEndDocument(void)
{
	OSStatus err = 0;
	if(gSessionDocumentIsOpen) {
		err = PMSessionEndDocument(gPrintSession);
	}
	gSessionDocumentIsOpen = false; 
	return err;
}

Rect GetPrinterPageRect(void)
{
	OSStatus err = 0;
   	PMRect pmPageRect;
	Rect  pageRect = {0,0,0,0};
	
	err = PMGetAdjustedPageRect (gPageFormat,&pmPageRect);
	
	pageRect.top = pmPageRect.top;
	pageRect.bottom = pmPageRect.bottom;
	pageRect.left = pmPageRect.left;
	pageRect.right = pmPageRect.right;
   
   return pageRect;
}
#endif

/////////////////////////////////////////////////
/////////////////////////////////////////////////

long MyPutScrap(long length, ResType theType, void *source)
{

	#if TARGET_API_MAC_CARBON
		ScrapRef scrapRef;
		ScrapFlavorType flavorType = theType;
		ScrapFlavorFlags flavorFlags =  kScrapFlavorMaskNone;
		
		OSStatus err = GetCurrentScrap(&scrapRef);
		if(err) return 0;
		
		return PutScrapFlavor(scrapRef,flavorType,flavorFlags,length,source);
	#else
		return PutScrap(length,theType,source);
		
	#endif
}

long MyGetScrapLength(ResType theType)
{
	return MyGetScrap(0,theType);
}

long MyGetScrap(Handle hDest, ResType theType)
{
	// returns the length
	// If the handle is not nil, it sizes handle to hold the data and fills in the data 
	#if TARGET_API_MAC_CARBON
		ScrapRef scrapRef;
		ScrapFlavorType flavorType  = theType;
		Size byteCount,numBytesAllocated;
		OSStatus err = GetCurrentScrap (&scrapRef);
		if(err) return 0;
		//
		err = GetScrapFlavorSize(scrapRef,flavorType,&byteCount);
		if(err) return 0;
		if(byteCount == 0) return 0;
		
		if(hDest) { // then the caller wants the data filled in (
			_SetHandleSize(hDest,byteCount);
			numBytesAllocated = _GetHandleSize(hDest); // _MemError();
			if(byteCount != numBytesAllocated){// we had a memory err, the requested space was not allocated
				SysBeep(5);return(0);
			}
			
			_HLock(hDest);
			err = GetScrapFlavorData(scrapRef,flavorType,&byteCount,*hDest);
			_HUnlock(hDest);
			
			if(err) return 0;
		}
		
		return byteCount;
	#else
		long offset; // note: returned value is not used
		return GetScrap(hDest, theType, &offset);
	#endif
}

long MyZeroScrap(void)
{
	#if TARGET_API_MAC_CARBON
		ClearCurrentScrap();
		return 0;
	#else
		return ZeroScrap();
	#endif
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////


#if TARGET_API_MAC_CARBON

////////

NavEventUPP gMyNavEventHandlerPtr;

static pascal_ifMac OSErr AddMyNavDialogItems (DialogRef dialog, short ditlResID, NavDialogRef context)
{
	OSErr err = noErr;
	
	Handle ditl = GetResource ('DITL',ditlResID);
	
	if (!ditl) {
		//MESSAGE("Resource not found");
		return -1;
	}

	if (context)
		err = NavCustomControl (context, kNavCtlAddControlList, ditl);
	else
		AppendDITL (dialog, ditl, appendDITLBottom);
	
	ReleaseResource (ditl);
	
	return err;
}

//////

static pascal_ifMac void MyNavEventCallback(
    NavEventCallbackMessage message, 
    NavCBRecPtr  param, 
    NavCallBackUserData   userDataPtr)
{
	OSErr err = noErr;
	MyCustomNavItemsData *myDataPtr = (MyCustomNavItemsData *)userDataPtr;

	switch (message)
	{
		case kNavCBEvent :
		{
			short firstCustomItem;
			short myItemHit;
			DialogRef dialog = GetDialogFromWindow(param->window);
			err = NavCustomControl (param->context, kNavCtlGetFirstControlID, &firstCustomItem);
			myItemHit = param->eventData.itemHit  - firstCustomItem;
			
			if(myDataPtr->clickProc && myItemHit > 0)
				myDataPtr->clickProc(dialog,myItemHit);
		}
		break;

		case kNavCBCustomize :
			if(param->customRect.bottom == 0) {
				// get the size from the first item of the custom DITL
				param->customRect.bottom = param->customRect.top + myDataPtr->customHeight;
				param->customRect.right = param->customRect.left + myDataPtr->customWidth;
			}
		break;

		case kNavCBStart :
			err = AddMyNavDialogItems (GetDialogFromWindow(param->window), myDataPtr->ditlResID, param->context);
			if(err) return;
			if(myDataPtr->initProc)
				myDataPtr->initProc(GetDialogFromWindow(param->window));
			
		break;

		case kNavCBTerminate :
		break;
	}

	if (err) SysBeep (-1);
} 


////////

static NavEventUPP GetMyNavEventHandlerUPP(void)
{
	if(!gMyNavEventHandlerPtr)
		gMyNavEventHandlerPtr = NewNavEventUPP( MyNavEventCallback );
	return gMyNavEventHandlerPtr; 
}


///

Boolean AskUserForPutFileName(char* promptStr, char* defaultName, char* pathOfFileSelected, short maxPathLength, FSSpec *specPtr, MyCustomNavItemsData *myCustomItemsDataPtr)
{
	OSStatus err = 0;
	
	NavDialogCreationOptions  dialogOptions;
   	OSType inFileType = 'TEXT'; // what will happen if we always just say TEXT and change our mind later ? (perhaps it will add .txt) ?
   	OSType inFileCreator = kNavGenericSignature;
	NavEventUPP inEventProc = 0;
	Ptr inClientData = (Ptr)myCustomItemsDataPtr;
	NavDialogRef navDialogRef = 0;
	NavReplyRecord replyRecord;
	Boolean gotReply = false;
	
	FSSpec fsSpec;
	char pathName[256]="";
	char theFileName[256]="";
	
	CFStringRef windowTitleCFStrRef = 0;
	CFStringRef clientNameCFStrRef = 0;
	CFStringRef saveFileNameCFStrRef = 0;
	
	if(specPtr) memset(specPtr,0,sizeof(specPtr));
	if(pathOfFileSelected && maxPathLength > 0) pathOfFileSelected[0] = 0;
	
	err = NavGetDefaultDialogCreationOptions(&dialogOptions);
	if(err) return false;
	
	// for now don't customize the dialogs, may want to do this later
	/*if(myCustomItemsDataPtr && myCustomItemsDataPtr->ditlResID) {
		inEventProc = GetMyNavEventHandlerUPP();
	}*/

	if(promptStr && promptStr[0]) {
		windowTitleCFStrRef = CFStringCreateWithCString(NULL, promptStr,kCFStringEncodingMacRoman);
		dialogOptions.windowTitle = windowTitleCFStrRef;
	}
	
	// set the client name so the system can remember the last place the user looked
	clientNameCFStrRef  = CFStringCreateWithCString(NULL,"GNOME",kCFStringEncodingMacRoman);
	dialogOptions.clientName = clientNameCFStrRef;
	
	if(defaultName && defaultName[0]) {
		saveFileNameCFStrRef = CFStringCreateWithCString(NULL, defaultName,kCFStringEncodingMacRoman);
		dialogOptions.saveFileName = saveFileNameCFStrRef;
	}
	
	//err = NavCreatePutFileDialog(&dialogOptions,inFileType,inFileCreator,inEventProc,inClientData,&navDialogRef);
	err = NavCreatePutFileDialog(&dialogOptions,inFileType,inFileCreator,inEventProc,NULL,&navDialogRef);
	if(err) return false;
	
	InitCursor();
	err = NavDialogRun(navDialogRef);
	SetWatchCursor();
	if(err) goto done;
	
	err = NavDialogGetReply(navDialogRef,&replyRecord);
	if(err) goto done;
	
	gotReply = true;
	
	err = AEGetNthPtr(&(replyRecord.selection), 1, typeFSS, NULL, NULL, &fsSpec, sizeof(fsSpec), NULL);
	if(err) goto done;
	
	(void)PathNameFromDirID(fsSpec.parID,fsSpec.vRefNum,pathName); // GetPathWithDelimiterFromDirID
	mypstrcatJM((StringPtr)pathName,fsSpec.name);
	my_p2cstr(pathName);
	
	(void)CFStringGetCString(replyRecord.saveFileName,theFileName,256,kCFStringEncodingMacRoman);
	
	if( (strlen(pathName) + 1 + strlen(theFileName)) < maxPathLength){
		strcpy(pathOfFileSelected,pathName);
		strcat(pathOfFileSelected,":");
		strcat(pathOfFileSelected,theFileName);
	}
	else  {
		err = -1; // path is too long to return
	}
	
	if(!err && specPtr)  *specPtr = fsSpec;
	
done:
	if(windowTitleCFStrRef) { CFRelease(windowTitleCFStrRef);  windowTitleCFStrRef = 0;}
	if(clientNameCFStrRef) { CFRelease(clientNameCFStrRef);  clientNameCFStrRef = 0;}
	if(saveFileNameCFStrRef) { CFRelease(saveFileNameCFStrRef);  saveFileNameCFStrRef = 0;}

	if(gotReply) NavDisposeReply(&replyRecord);
	
	NavDialogDispose(navDialogRef);
	
	if(gotReply && !err) return true;
	return false;
 	
}

/////////////////////////////////////////////////


static NavTypeListHandle MyCreateNavTypeListHandle(OSType applicationSignature, short numTypes, OSType typeList[])
{
  NavTypeListHandle hdl = NULL;
  
  if ( numTypes > 0 ) {
    hdl = (NavTypeListHandle) _NewHandle(sizeof(NavTypeList) + numTypes * sizeof(OSType));
  
    if ( hdl != NULL ){
      (*hdl)->componentSignature = applicationSignature;
      (*hdl)->osTypeCount    = numTypes;
      BlockMoveData(typeList, (*hdl)->osType, numTypes * sizeof(OSType));
    }
  }
  
  return hdl;
}

/////////////////////////////////////////////////

Boolean AskUserForGetFileName(char* prompt_string,    
							short numTypes, 
							OSType typeList[],
							//
							char* pathOfFileSelected,  
							short maxPathLength,
							FSSpec *specPtr,
							MyCustomNavItemsData *myCustomItemsDataPtr)
{
	OSStatus err = 0;
	NavDialogCreationOptions  dialogOptions;
	NavTypeListHandle inTypeList = 0;
	NavEventUPP inEventProc = 0;
	NavPreviewUPP inPreviewProc = 0;
	NavObjectFilterUPP inFilterProc = 0;
	Ptr inClientData = (Ptr)myCustomItemsDataPtr;
	NavDialogRef navDialogRef = 0;
	NavReplyRecord replyRecord;
	Boolean gotReply = false;
	
	FSSpec fsSpec;
	char pathName[256]="";
	
	CFStringRef windowTitleCFStrRef = 0;
	CFStringRef clientNameCFStrRef = 0;
	
	if(specPtr) memset(specPtr,0,sizeof(specPtr));
	if(pathOfFileSelected && maxPathLength > 0) pathOfFileSelected[0] = 0;

	err = NavGetDefaultDialogCreationOptions(&dialogOptions);
	if(err) return false;
	
	// for now don't customize the dialogs, may want to do this later
	/*if(myCustomItemsDataPtr && myCustomItemsDataPtr->ditlResID) {
		inEventProc = GetMyNavEventHandlerUPP();
	}*/

	if(prompt_string && prompt_string[0]) {
		windowTitleCFStrRef = CFStringCreateWithCString(NULL, prompt_string, kCFStringEncodingMacRoman);
		dialogOptions.windowTitle = windowTitleCFStrRef;
	}
	
	// set the client name so the system can remember the last place the user looked
	clientNameCFStrRef  = CFStringCreateWithCString(NULL, "GNOME", kCFStringEncodingMacRoman);
	dialogOptions.clientName = clientNameCFStrRef;
	
	//inTypeList = MyCreateNavTypeListHandle(gMySignature,numTypes,typeList);	// for now allow all files to be shown, filtering is messed up on 10.6
	
	//err = NavCreateChooseFileDialog (&dialogOptions,inTypeList,inEventProc,inPreviewProc,inFilterProc,inClientData,&navDialogRef);
	err = NavCreateChooseFileDialog (&dialogOptions,inTypeList,inEventProc,inPreviewProc,inFilterProc,NULL,&navDialogRef);
	if(err) return false;
	
	InitCursor();
	err = NavDialogRun(navDialogRef);
	SetWatchCursor();
	if(err) goto done;
	
	err = NavDialogGetReply(navDialogRef,&replyRecord);
	if(err) goto done;
	
	gotReply = true;
	
	err = AEGetNthPtr(&(replyRecord.selection), 1, typeFSS, NULL, NULL, &fsSpec, sizeof(fsSpec), NULL);
	if(err) goto done;
	
	(void)PathNameFromDirID(fsSpec.parID,fsSpec.vRefNum,pathName);//GetPathWithDelimiterFromDirID
	mypstrcatJM((StringPtr)pathName,fsSpec.name);
	my_p2cstr(pathName);
	
	if(strlen(pathName) < maxPathLength){
		strcpy(pathOfFileSelected,pathName);
	}
	else  {
		err = -1; // path is too long to return
	}
		
	if(!err && specPtr)  *specPtr = fsSpec;
	
done:
	if(inTypeList) {DisposeHandle((Handle)inTypeList); inTypeList = 0;}
	if(windowTitleCFStrRef) { CFRelease(windowTitleCFStrRef);  windowTitleCFStrRef = 0;}
	if(clientNameCFStrRef) { CFRelease(clientNameCFStrRef);  clientNameCFStrRef = 0;}

	if(gotReply) NavDisposeReply(&replyRecord);
	
	NavDialogDispose(navDialogRef);
	
	if(gotReply && !err) return true;
	return false;
	
} /* end MyGetMacFile() */

#endif
///////////////////

short DetermineCancelButtonItemNumByLookingAtButtons(DialogRef theDialog);
void SetDefaultItemBehavior(DialogRef dialog)
{
	#if TARGET_API_MAC_CARBON
		short cancelButtonItemNum;
		(void)SetDialogTracksCursor (dialog,true); 
		SetDialogDefaultItem(dialog,1);
		cancelButtonItemNum = DetermineCancelButtonItemNumByLookingAtButtons(dialog);
		if(cancelButtonItemNum < 1) cancelButtonItemNum = 1; // make the default button respond to the "escape key" i.e. equivalent to the cancel button
		SetDialogCancelItem(dialog,cancelButtonItemNum); 
	#else
		#pragma unused(dialog)
		// nothing to do, the  modalfilter proc takes care of the behavior
	#endif
}

void MySelectDialogItemText(DialogRef theDialog, short editTextitemNum, short strtSel, short endSel)
{
#if TARGET_API_MAC_CARBON
	// with the new MAC code, the edit text item with the focus was not showing the text 
	// it acted like the text was invisible... like it had drawn it and didn't care to redraw it.
	// I don't know what is up with that, but Hiding and Showing the item seems to do the trick
	// even though InvalDialogItemRect didn't help
	if(!IsWindowVisible(GetDialogWindow(theDialog))) {
		// then this is being called when the window is being set up
		// and so we need to use our trick
		HideDialogItem(theDialog,editTextitemNum);
		ShowDialogItem(theDialog,editTextitemNum);
	}
#endif

	SelectDialogItemText(theDialog,editTextitemNum,strtSel,endSel);
}

#if TARGET_API_MAC_CARBON
static OSStatus strcpyFileSystemRepresentationFromClassicPath(char *nativePath, char * classicPath, long nativePathMaxLength )
{
		CFURLRef   fileAsCFURLRef = 0;
		Boolean    resolveAgainstBase = true;
		FSRef fileAsFSRef;
		Boolean gotPath;
		char pathPStr[256];
		FSSpec spec;
		OSStatus err = 0;
		
		strcpy(pathPStr,classicPath);
		my_c2pstr(pathPStr);
		err = FSMakeFSSpec(0, 0, (StringPtr)pathPStr,&spec);
		if(err == fnfErr) {
			// just means the file does not exist yet
			err = FSpCreate (&spec,'MPW ','TEXT',smSystemScript); // we should be able to use any creator and type here
		}
		if(err) return err;
		
		err = FSpMakeFSRef(&spec,&fileAsFSRef);
		if(err) return err;

		 // Convert the reference to the file to a CFURL
		fileAsCFURLRef = CFURLCreateFromFSRef(NULL, &fileAsFSRef);
		if(fileAsCFURLRef) {
			gotPath = CFURLGetFileSystemRepresentation(fileAsCFURLRef,resolveAgainstBase,(UInt8 *)nativePath,nativePathMaxLength);
			CFRelease(fileAsCFURLRef); fileAsCFURLRef = 0;
		}
		
		if(gotPath) 
			return noErr;
		
		return -1; // did not get the path

}
#endif

FILE *my_fopen (char *filePath,char *mode)
{
	#ifdef __GNUC__
		// XCode
		char nativePath[1024];
		OSStatus err = strcpyFileSystemRepresentationFromClassicPath(nativePath,filePath,1024);
		if(err) return NULL;
		
		return fopen(nativePath,mode);
		
	#else // WIN and CodeWarrior
		return fopen(filePath,mode);
	#endif
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////

#endif


