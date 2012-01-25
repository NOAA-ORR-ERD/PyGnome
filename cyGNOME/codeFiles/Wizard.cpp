#include "cross.h"
#include "OUtils.h"
#include "Wizard.h"
#include "EditWindsDialog.h"
//#include "HtmlHelp.h"
#ifdef MAC
#pragma segment WIZARD
#endif
/////////////////////////////////////////////////
/////////////////////////////////////////////////
Boolean gInWizard = FALSE;

char gCommandFilePath[kMaxNameLen];


/////////////////////////////////////////////////
////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

void GetWizButtonTitle_Next(char *str)
{
	strcpy(str,"Next >>");	
}

void GetWizButtonTitle_Previous(char *str)
{
	strcpy(str,"<< Back");	
}

void GetWizButtonTitle_Done(char *str)
{
	//strcpy(str,"Done");	
	strcpy(str,"To the Map Window");	//12/27/99
}

void GetWizButtonTitle_Cancel(char *str)
{
	strcpy(str,"Cancel");	
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

WizardFile * gCurrentWizardFile = nil;

#ifdef MAC
pascal_ifMac Boolean WizSTDFilter(DialogPtr theDialog, EventRecord *theEvent, short *itemHit)
{
	GrafPtr oldPort;
	
	GetPortGrafPtr(&oldPort);
	SetPortDialogPort(theDialog);
	
	if (KeyEvent(theEvent, ESC, ESC) || CommandKeyEvent(theEvent, '.', '.')) 
	{
		FlashItem(theDialog, WIZ_CANCEL);
		*itemHit = WIZ_CANCEL;
		SetPortGrafPort(oldPort);
		return TRUE;
	}
	//else if(theEvent->what == keyDown || theEvent->what == autoKey)
	//{
	//   char key = (theEvent->message & 0xff);
	//	if(key )
	//}
	SetPortGrafPort(oldPort);
	return STDFilter(theDialog,theEvent,itemHit);
}

#else // IBM code

typedef struct
{
	short resNum;
	long	dialogFlags;
	CHARH	prevAnswers;
	CHARH	*userAnswers;
	CHARH	*messages;
} IBMDialogVar;
IBMDialogVar gIBMDialogVar;



BOOL CALLBACK WizSTDFilter(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	short  itemHit;
	Boolean done;
	HCURSOR arrow;
	
	
	switch (message) {
		case WM_INITDIALOG:
			SetPort(hWnd);
			CenterDialog(hWnd, 0);
			SetFocus(GetDlgItem(hWnd, WIZ_OK)); // set focus to OK button by default, but do before possibly highlighting an edit text box
			gCurrentWizardFile->InitWizardDialog(gIBMDialogVar.resNum,hWnd,gIBMDialogVar.dialogFlags,gIBMDialogVar.prevAnswers);
			//SetFocus(GetDlgItem(hWnd, WIZ_OK)); // set focus to OK button by default
			if (arrow = LoadCursor(NULL, IDC_ARROW))
			SetCursor(arrow);
			return FALSE;
		
    case WM_PAINT      :              /* the dialog is being painted */
      	gCurrentWizardFile->WM_Paint(hWnd);
			return TRUE;
		
		case WM_DESTROY:
			SetPort(0);
			return TRUE;
		
		case WM_COMMAND:
			if (!IsWindowVisible(hWnd)) return FALSE;
			itemHit = LOWORD(wParam);
			settings.doNotPrintError = false;// allows dialogs to come up more than once
			if (!lParam && !HIWORD(wParam)) { // from a non-existent item
				if (itemHit == 2) 
				{ // "2" is standard windows "cancel" item
					EndDialog(hWnd, WIZ_CANCEL);
					return TRUE;
				}
				done = gCurrentWizardFile->DoWizardItemHit(hWnd,itemHit,gIBMDialogVar.userAnswers,gIBMDialogVar.messages);		
				if (done) EndDialog(hWnd, itemHit);
				return TRUE;
			}

			switch (HIWORD(wParam)) {
				case EN_CHANGE:
					if ((WindowPtr)lParam != GetFocus()) break;
				case BN_CLICKED:
				case EN_SETFOCUS:
				case EN_KILLFOCUS:
				case CBN_SELCHANGE:
					itemHit = LOWORD(wParam);
					done = gCurrentWizardFile->DoWizardItemHit(hWnd,itemHit,gIBMDialogVar.userAnswers,gIBMDialogVar.messages);		
					if (done) EndDialog(hWnd, itemHit);
					return TRUE;
			}
			return FALSE;
		
	}
	
	return FALSE;
}
#endif

/////////////////////////////////////////////////

void ParsingErrorAlert(char*errStr,char* offendingLine)
{	
	char msg[256];
	char temp[64];
	long len;
	strnzcpy(msg,errStr,150);
	
	if(offendingLine)
	{
		strcat(msg,NEWLINESTRING);
		strcat(msg,"Offending line:");
		strcat(msg,NEWLINESTRING);
		len = strlen(offendingLine);
		if(len > 50)
		{ // we need to cut it off (without doing damage to the input strings !!)
			strnzcpy(temp,offendingLine,50);
			strcat(temp,"...");
			strcat(msg,temp);
		}
		else
		{
			strcat(msg,offendingLine);
		}
	}
	settings.doNotPrintError = false;// allows dialogs to come up more than once
	printNote(msg);
}

/////////////////////////////////////////////////
void WizardOutOfMemoryAlert(void)
{
	printError("There is not enough memory allocated to the program.  Error in Wizard.");
}
/////////////////////////////////////////////////

Boolean IsWizardLocaleFile(char* fullPathName)
{
	Boolean isWizFile = false;
	CHARH orderResource = nil;
	#ifdef IBM
	{
		OSErr err = 0;
		HINSTANCE instDLL=0;
		// In NT there is an annoying error alert 
		// if you call LoadLibrary on something that is not an exe or DLL
		// so we first have to see if this file is an exe or dll.
		//
		// EXE's and DLL's have a header at the top of the file
		// 78 binary chars  followed by
		// a string saying the program cannot be run in DOS mode
	 	char fileStart[16] = {	0x4D, 0x5A, 0x90, 0x00, 
										0x03, 0x00, 0x00, 0x00, 
										0x04, 0x00, 0x00, 0x00, 
										0xFF, 0xFF, 0x00, 0x00};
	
		char *magicStr = "This program cannot be run in DOS mode";
		
		// read in the first part of the file
		char strLine[256];
		char firstPartOfFile[256];
		long i,lenToRead,fileLength,line;
		Boolean matched16Chars = true;
		
		err = MyGetFileSize(0,0,fullPathName,&fileLength);
		if(err) return false;
		
		lenToRead = _min(256,fileLength);
		
		err = ReadSectionOfFile(0,0,fullPathName,0,lenToRead,firstPartOfFile,0);
		if(err) return false;
		
		// check first 16 chars for a match
		for(i = 0; i < 16; i++)
		{
			if(firstPartOfFile[i] != fileStart[i])
				matched16Chars = false;
		}
		if(!matched16Chars) 
		/*{	// check for a text file that drives a SAV file
			firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
			NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
			if (strstr(strLine,"[ORDER]"))
					isWizFile = true;
			else
				return false;
		}*/
		return false;
		
		// I think the magic string always starts at char 78
		if(strncmpnocase(&firstPartOfFile[78],magicStr,strlen(magicStr)))
		{	// the magic string is not there
			// so it is not an EXE or DLL
			return false;
		}
		
		// the first 16 chars and the magic string match
		// so it should be safe to call load library	
		instDLL = LoadLibrary(fullPathName);
		if (instDLL) 
		{ 
			HRSRC hResInfo = FindResource(instDLL,"#10000","TEXT");
			if(hResInfo) isWizFile = true;
			FreeLibrary(instDLL);
		}
	}
	#else
	{
		//long f = openresfile(fullPathName);
		Str255 fpname;
		CopyCStringToPascal(fullPathName,fpname);
		long f = HOpenResFile(0,0,fpname,fsCurPerm);
		CHARH r =nil;
		if(f == -1)  return false; // openresfile() failure
		r = Get1Resource('TEXT',10000);
		if(r) { isWizFile = true; ReleaseResource(r); r = 0;}
		
		else
		{
			OSErr err = 0;
			long line;
			char	strLine [512];
			char	firstPartOfFile [512];
			long lenToRead,fileLength;
			
			err = MyGetFileSize(0,0,fullPathName,&fileLength);
			if(err) return false;
			
			lenToRead = _min(512,fileLength);
			
			err = ReadSectionOfFile(0,0,fullPathName,0,lenToRead,firstPartOfFile,0);
			firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
			if (!err)
			{
				NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
				if (strstr(strLine,"[ORDER]"))
					isWizFile = true;
			}
		}
		
		CloseResFile(f);
	}
	#endif
	return isWizFile;
}
/////////////////////////////////////////////////


Boolean IsWizardSaveFile(char* path)
{

	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [256];
	char	firstPartOfFile [256];
	long lenToRead,fileLength;
	char *firstLineStartingWord = "WIZARDSAVEFILE";
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 256);
		RemoveTrailingWhiteSpace(strLine);
		if (!strncmpnocase(strLine,firstLineStartingWord,strlen(firstLineStartingWord)))
			bIsValid = true;
	}
	
	return bIsValid;
}

/////////////////////////////////////////////////

Boolean IsWizardFile(char* fullPathName)
{
	if(model->fWizard)
	{
		// if we have a current file
		// we need to check to see if the user picked the same file again
		char currentPathName[256];
		model->fWizard->GetLocationFileFullPathName(currentPathName);
		if(!strcmp(fullPathName,currentPathName))
		{
			// they chose the same file as they selected before
			return true;
		}
	}
	/////////////////////////////////////////////////
	
	if(IsWizardLocaleFile(fullPathName)) return true;
	if(IsWizardSaveFile(fullPathName)) return true;
	return false;
}

/////////////////////////////////////////////////

void	DebugHandleMessage(CHARH h)
{
	long hLen;
	OSErr err = 0;
	
	if(!h) printError("DebugHandleMessage got nil handle");
	else
	{
		hLen = _GetHandleSize((Handle)h);
		if((*h)[hLen-1] == 0)
		{ // everything is great it is null terminated
			printError(*h);
		}
		else
		{	// we have to grow the handle to null terminate it
			_SetHandleSize((Handle)h,hLen+1);
			err = _MemError();
			if(err) printError("DebugHandleMessage got MemError");
			else
			{
				(*h)[hLen] = 0;
				printError(*h);
				_SetHandleSize((Handle)h,hLen);
			}
		}
	}
}
	


/////////////////////////////////////////////////



OSErr AskForWizardFileHelper(char *pathName, Boolean allowGnomeSaveFile,Boolean allowWizardSaveFile)
{
	char path[256]="", m[300];
	long index, numTypes = 4;
	char prompt[256] = "";
	
	Point where = {50,50};
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', WIZARDFILETYPE };
	MySFReply reply;
	OSErr err = 0;
	
	if(allowWizardSaveFile) typeList[2] = WIZARDSAVEFILETYPE;
	if(allowGnomeSaveFile) {typeList[0] ='.SAV';typeList[1] = 'SAVE';}
	
	where = CenteredDialogUpLeft(M38g);
	
	if(pathName[0])
	{
		// code goes here
		// we should use this file name as a default if we can
	}
	
	// for now don't ask if they want to save
	//if (CHOICEALERT(M77, 0, FALSE)) return -1;
	paramtext(prompt,"","","");

#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, prompt, numTypes, typeList,
				   (MyDlgHookUPP)0, &reply, M38g, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);
#else
	sfpgetfile(&where, prompt,
				(FileFilterUPP)0,
				4, typeList,
				(DlgHookUPP)0,
				&reply, M38g,
				(ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));

	if (!reply.good) return USERCANCEL;
	
	my_p2cstr(reply.fName);

#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
#else
		strcpy(path, reply.fName);
#endif	
#endif	
	InitCursor();
	strcpy(pathName,path);
	return err;
}

OSErr AskForWizardFile(char *pathName, Boolean allowGnomeSaveFile)
{
	Boolean allowWizardSaveFile = true;
	pathName[0] = 0;
	return AskForWizardFileHelper(pathName,allowGnomeSaveFile,allowWizardSaveFile);
}

OSErr AskForSpecificWizardLocationFile(char *pathName)
{
	Boolean allowWizardSaveFile = false;
	Boolean allowGnomeSaveFile = false;
	return AskForWizardFileHelper(pathName,allowGnomeSaveFile,allowWizardSaveFile);
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
// Utility functions
///////////
typedef struct
{
	MenuHandle origHdl;
	short menuID;
} OriginalMenuInfo;

#define kMAXNUMPOPUPS  10
PopInfoRec gWizardPopTable[kMAXNUMPOPUPS];
OriginalMenuInfo gOriginalOssmMenus[kMAXNUMPOPUPS];
	
void ClearWizardPopupTable(void)
{
	memset((void *)gWizardPopTable, 0, kMAXNUMPOPUPS*sizeof(PopInfoRec));
}


#ifdef IBM
	long GetWinHandleSize(HANDLE h)
	{
		long size = 0;
		if(GlobalFlags(h) == GMEM_DISCARDED) size = 0;
		else size = GlobalSize(h);
		return size;
	}
#endif

MenuHandle LoadMenuHandle(short menuID)
{ 	
	MenuHandle m = 0;
#ifndef IBM
	m = GetMenuHandle(menuID);
	if(m) DeleteMenu(menuID);
#endif
	GetAndInsertMenu(menuID, -1);
	return m;
}

#ifdef IBM
void LoadComboBoxItems(HINSTANCE hInstance,DialogPtr theDialog, short comboboxID)
{	// expects strings to start at the same ID as the combobox , e.g. 10040
	HANDLE itemHandle = GetDlgItem (theDialog, comboboxID);
	short stringResID = 0,i = 0;
	char buffer[256];
	
	if(!itemHandle) return;

   for(i = 0; i <100; i++) // more than 100 items would probably be a mistake
   {
      buffer[0] = 0;
      LoadString(hInstance, comboboxID+i, buffer, 255);
      if (!strlen(buffer))
         return;  //Successfully found end of list
      else
      {
         //SendMessage (itemHandle, CB_INSERTSTRING, i, (LPARAM)(LPCSTR)buffer);
         SendMessage ((HWND)itemHandle, CB_INSERTSTRING, i, (LPARAM)(LPCSTR)buffer);
         stringResID++;	// not used
      }
   }
}
#endif 

void RestoreOSSMMenuHdls(void)
{
#ifndef IBM
	MenuHandle ourMenuHdl = 0,origHdl;
	for(int i = 0; i<kMAXNUMPOPUPS;i++)
	{
		short menuID = gOriginalOssmMenus[i].menuID;
		origHdl = gOriginalOssmMenus[i].origHdl;
		if(menuID > 0) ourMenuHdl = GetMenuHandle(menuID);
		if(ourMenuHdl) DeleteMenu(menuID);
		if(origHdl) InsertMenu(origHdl,-1);
		gOriginalOssmMenus[i].menuID = 0;
		gOriginalOssmMenus[i].origHdl= 0;
	}
#endif
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
#ifdef MAC

pascal_ifMac void NonCppDrawingRoutine(DialogPtr theDialog, short itemNum) 
{
	gCurrentWizardFile->DrawingRoutine(theDialog,itemNum);
}
#endif

/*Rect GetDialogItemBox(DialogPtr theDialog, short item)
{
	short				itemType;
	Handle				itemHandle;
	Rect				itemBox = {0,0,0,0};
	
	GetDialogItem(theDialog, item, &itemType, &itemHandle, &itemBox);
	return itemBox;
}*/

short NumDialogItems(DialogPtr theDialog)
{
	short numItems = 0;
	#ifdef MAC
		//numItems = 1 + (**((short**)((DialogPeek)theDialog)->items));
		// should use Gestalt to be sure this option is available, or MACB4CARBON
		numItems = CountDITL(theDialog);
	#else 
		numItems = 30; // just guess too big
	#endif
	return numItems;
}



void SafeSetDialogItemBox(DialogPtr theDialog, short item, Rect* newBox)
{	
	short		itemType;
	Handle		itemHandle;
	Rect		itemBox;
	if(item < 1 || item > NumDialogItems(theDialog)) return;
	GetDialogItem(theDialog, item, &itemType, &itemHandle, &itemBox);
	SetDialogItem(theDialog, item, itemType, itemHandle, newBox);
}

void SafeSetDialogItemHandle(DialogPtr theDialog, short item, Handle itemProc)
{	
	short		itemType;
	Handle		itemHandle;
	Rect		itemBox;
	UserItemUPP uPP = 0;
	
	if(item < 1 || item > NumDialogItems(theDialog)) return;
	//uPP = (UserItemUPP)MakeUPP((ProcPtr)itemProc, uppUserItemProcInfo);
	if(itemProc) uPP = MakeUserItemUPP((UserItemProcPtr)itemProc);

	GetDialogItem(theDialog, item, &itemType, &itemHandle, &itemBox);
	SetDialogItem(theDialog, item, itemType, (Handle)uPP, &itemBox);
}

void 	SafeGetDialogItem(DialogPtr theDialog,short item,short* itemType,Handle* itemHandle,Rect* itemRect)
{
	if(item < 1 || item > NumDialogItems(theDialog))
	{
		*itemType = 0;
		*itemHandle = 0;
		MySetRect(itemRect,0,0,0,0);
		return;
	}
	GetDialogItem(theDialog,item,itemType,itemHandle,itemRect);

}

void SafeSelectDialogItemText(DialogPtr theDialog, short itemNum, short begin, short end)
{
	short itemType;
	Rect itemRect;
	Handle itemHandle;
	
	if(itemNum < 1 || itemNum > NumDialogItems(theDialog))return;
	GetDialogItem(theDialog,itemNum,&itemType,&itemHandle,&itemRect);
	switch(itemType)
	{
	#ifdef IBM
		case 0: // on the IBM, we don't have itemType working, right now it is always returning 0
	#else
		case	editText:
		case (editText+itemDisable):
	#endif
			MySelectDialogItemText(theDialog,itemNum,begin,end);
			break;
	}
}

void SafeSetDialogItem(DialogPtr theDialog,short item,short	newItemType,Handle newHdl,Rect *newRect)
{	
	short		itemType;
	Handle		itemHandle;
	Rect		itemBox;
	
	if(item < 1 || item > NumDialogItems(theDialog))return;
	GetDialogItem(theDialog, item, &itemType, &itemHandle, &itemBox);
	SetDialogItem(theDialog,item,newItemType,newHdl,newRect);
}

#ifndef IBM
void SetDlgItemText(DialogPtr theDialog,short itemNum,char* str)
{
	short itemType;
	Rect itemRect;
	Handle itemHandle;
	
	if(itemNum < 1 || itemNum > NumDialogItems(theDialog))return;
	GetDialogItem(theDialog,itemNum,&itemType,&itemHandle,&itemRect);
	my_c2pstr (str);
	switch(itemType)
	{
		case	editText:
		case (editText+itemDisable):
		case statText:
		case (statText+itemDisable):
			SetDialogItemText(itemHandle,(StringPtr)str);
			break;
		case kButtonDialogItem:
		case (kButtonDialogItem+kItemDisableBit):
			if(itemHandle) SetControlTitle((ControlHandle)itemHandle,(StringPtr)str);
			break;
	}
	my_p2cstr((StringPtr)str);
}
void GetDlgItemText(DialogPtr theDialog,short itemNum, char* str, long maxChar)
{
	short itemType;
	Rect itemRect;
	Handle itemHandle;
	char localStr[256] ="";
	
	SafeGetDialogItem(theDialog,itemNum,&itemType,&itemHandle,&itemRect);
	my_c2pstr(localStr);
	switch(itemType)
	{
		case	editText:
		case (editText+itemDisable):
		case statText:
		case (statText+itemDisable):
			GetDialogItemText(itemHandle,(StringPtr)localStr);
			break;
		case kButtonDialogItem:
		case (kButtonDialogItem+kItemDisableBit):
			if(itemHandle) GetControlTitle((ControlHandle)itemHandle,(StringPtr)localStr);
			break;
	}
	my_p2cstr((StringPtr)localStr);
	strncpy(str,localStr,maxChar);
	str[maxChar-1] = 0;
}
#endif
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

Boolean GetLineFromStr(long lineNum1Relative,char* str, char* answerStr,long maxNumChars)
{	// returns true if it gets the line
	// returns false when lineNum1Relative > number of lines
	long currentLineNum = 1;
	long i,j,lineStart = 0;
	long strLen = strlen(str);
		
	answerStr[0] = 0;
	// move to the line number requested
	while(currentLineNum <lineNum1Relative && lineStart < strLen)
	{
		if(lineStart >= strLen || str[lineStart] == 0) 
		{
			return false; // we ran out of lines
		}
		if(str[lineStart] == RETURN || str[lineStart] == LINEFEED) 
		{
			if(str[lineStart] == RETURN && str[lineStart+1] ==LINEFEED) lineStart++; // skip that char also
			currentLineNum++;
		}
		lineStart++;
	}
	////////////
	j =0,i = lineStart;
	
	// copy till the end delimiter
	for(/*i is already set*/; j< maxNumChars-1 && i < strLen; i++)
	{
		if(str[i] == 0 || str[i] == RETURN || str[i] == LINEFEED) break;
		answerStr[j++] = str[i];
	}
	answerStr[j++] = 0;
	return true; // we got the line

}

char* StartOfFirstWord(char* str,long *lenOfWord)
{	//returns ptr to the end of the string if string is only white space
	char* startOfWord = str;
	long len = 0;
	*lenOfWord = 0;
	//move past any leading white space
	while(*str == ' ' || *str == '\t') str++;
	startOfWord = str;
	//count chars till white space or the end
	while(*str != ' ' && *str != '\t' && *str) {str++;len++;}
	*lenOfWord = len;
	return startOfWord;
}


Boolean GetLineFromHdlHelper(long lineNum1Relative,long lineStart, long *nextLineStart, CHARH paramListHdl, char* answerStr,long maxNumChars,Boolean removeLeadingAndTrailingWhiteSpace)
{	// returns true if it get the line
	// returns false when lineNum1Relative > number of lines
	char *paramListPtr;
	long currentLineNum = 1;
	long i,j;
	long hdlLen;
	long lastNonWhiteSpaceJ;
	
	answerStr[0] = 0;
	if(lineNum1Relative<1) 
	{
		printError("Programmer error: first parameter of GetLineFromHdl() is 1-relative");
		return false;
	}
	
	if(!paramListHdl) return false; // there are no lines to get

	hdlLen = _GetHandleSize((Handle)paramListHdl);
	_HLock((Handle)paramListHdl);
	paramListPtr = *paramListHdl;
	
	// move to the line number requested
	while(currentLineNum <lineNum1Relative && lineStart < hdlLen)
	{
		if(paramListPtr[lineStart] == 0) 
		{	// we ran out of lines
			_HUnlock((Handle)paramListHdl);
			return false; 
		}
		if(paramListPtr[lineStart] == RETURN || paramListPtr[lineStart] == LINEFEED) 
		{
			if(paramListPtr[lineStart] == RETURN && paramListPtr[lineStart+1] ==LINEFEED) lineStart++; // skip that char also
			currentLineNum++;
		}
		lineStart++;
	}
	////
	if(lineStart >= hdlLen) 
	{	// we ran out of lines
		_HUnlock((Handle)paramListHdl);
		return false; 
	}
	////////////
	j =0,i = lineStart;
	
	// copy till the end delimiter
	lastNonWhiteSpaceJ = 0;
	for(/*i is already set*/; i < hdlLen; i++)
	{
		if(paramListPtr[i] == 0 || paramListPtr[i] == RETURN || paramListPtr[i] == LINEFEED) 
		{	// set the next line start index
			break;
		}
		if(j< maxNumChars-1)
		{	// deal with this char
			answerStr[j++] = paramListPtr[i];
			if(removeLeadingAndTrailingWhiteSpace)
			{
				if(paramListPtr[i] == ' ' || paramListPtr[i] == '\t')
				{ // white space
					if(j == 1) {
						// it was the first char and it was white space
						j--;// so that we copy over this leading white space
					}
				}
				else 
					lastNonWhiteSpaceJ = j;
			}
		}
	}
	
	// set the next line start index
	if(i < hdlLen) {
		// i is the char that stopped the scan, i.e. the first delimiter for a line
		if(paramListPtr[i] == RETURN && paramListPtr[i] == LINEFEED) *nextLineStart = i+2;
		else *nextLineStart = i+1; // a single char delimiter (either 0 or RETURN)
	}
	else {
		*nextLineStart = hdlLen; // so that when looping the next call will end the loop
	}
	
	answerStr[j++] = 0;
	
	if(removeLeadingAndTrailingWhiteSpace)
		answerStr[lastNonWhiteSpaceJ] = 0; // chops off trailing white space

	_HUnlock((Handle)paramListHdl);

	return true; // we got the line
}

Boolean GetLineFromHdl(long lineNum1Relative,CHARH paramListHdl, char* answerStr,long maxNumChars)
{	// returns true if it gets the line
	// returns false when lineNum1Relative > number of lines
	long lineStart = 0;
	long nextLineStart;
	return GetLineFromHdlHelper(lineNum1Relative,lineStart,&nextLineStart,paramListHdl,answerStr,maxNumChars,false);
}

Boolean GetCleanedUpLineFromHdl(long lineNum1Relative,CHARH paramListHdl, char* answerStr,long maxNumChars)
{ // no leading or trailing white space
	// returns true if it gets the line
	// returns false when lineNum1Relative > number of lines
	long lineStart = 0;
	long nextLineStart;
	return GetLineFromHdlHelper(lineNum1Relative,lineStart,&nextLineStart,paramListHdl,answerStr,maxNumChars,true);
}

void WizardGetPhraseFromLine(long phraseNum1Relative,char* line, char* answerStr,long maxNumChars)
{	// returns the part between the ';' chars
	long i,j;
	char* ptr = line;
	
	answerStr[0] = 0;
	
	// starting with i = 1, move past ';' chars until i == phraseNum1Relative
	i = 1;
	while(i < phraseNum1Relative)
	{
		if(*ptr == 0) return; // end of the line
		if(*ptr == ';') i++; // 
		ptr++;
	}
	// we found the start of the phrase
	// move past spaces and tabs
	while(*ptr == ' ' || *ptr == '\t')ptr++;
	// now copy till the end of the line or the next ';' char
	// copy the rest
	for(/*i is already set*/ j = 0; j< maxNumChars-1 ; ptr++)
	{
		if(*ptr == 0 || *ptr == ';') break;
		answerStr[j++] = *ptr;
	}
	answerStr[j++] = 0;
	RemoveTrailingWhiteSpace(answerStr);

}

void WizardGetParameterString(char* key,char* line, char* answerStr,long maxNumChars)
{
	long i,j,lenKey = strlen(key);
	char phrase[256];
	answerStr[0] = 0;
	
	for(i = 0;true;i++)
	{
		WizardGetPhraseFromLine(i,line,phrase,256);
		if(!phrase[0]) return; // no more phrases
		if(phrase[lenKey] != ' ' && phrase[lenKey] != '\t') continue; // it can't be the key followed by white space
		if(strncmpnocase(phrase, key,lenKey)) continue; //it does not match
		break;// we found it
	}
	///////////
	//move past spaces and tabs
	i = lenKey;
	while(phrase[i] == ' ' || phrase[i] == '\t')i++;
	// copy the rest
	for(/*i is already set*/ j = 0; j< maxNumChars-1 ; i++)
	{
		if(phrase[i] == 0) break;
		answerStr[j++] = phrase[i];
	}
	answerStr[j++] = 0;
	RemoveTrailingWhiteSpace(answerStr);

}

void WizardGetParameterFromStringLine(char* key,long lineNum1Relative,char* str, char* answerStr,long maxNumChars)
{
	char line[512];
	if(maxNumChars > 512) printError("Programmer error: maxNumChars > 512");
	GetLineFromStr(lineNum1Relative,str,line,512);
	WizardGetParameterString(key,line,answerStr,maxNumChars);
}

void WizardGetParameterFromHdlLine(char* key,long lineNum1Relative,CHARH paramListHdl, char* answerStr,long maxNumChars)
{
	char line[512];
	answerStr[0] = 0;
	if(paramListHdl)
	{
		char* ptr;
		if(maxNumChars > 512) printError("Programmer error: maxNumChars > 512");
		GetLineFromHdl(lineNum1Relative,paramListHdl,line,512);
		WizardGetParameterString(key,line,answerStr,maxNumChars);
	}
}

/////////////////////////////////////////////////
TModelDialogVariables DefaultModelSettingsForWizard(void)
{
	Seconds start;
	DateTimeRec tTime;
	TModelDialogVariables  modelSettings;
	GetDateTime(&start);	// Mac recognizes daylight saving time, PC does not
	SecondsToDate (start, &tTime);
	tTime.second = 0;
	tTime.minute = 0; // use the previous hour
	DateToSeconds (&tTime, &start);
	modelSettings = DefaultTModelDialogVariables(start);
	return modelSettings;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

#define MAXNUMPARAMCHARS 64
OSErr WizardGetParameterAsDouble(char * key, char* line, double * val)
{
	// we use the wizard get parameter functions
	char localStr[MAXNUMPARAMCHARS];
	double localVal;
	OSErr err = -1;
	WizardGetParameterString(key,line,localStr,MAXNUMPARAMCHARS);
	if(localStr[0]) err =  StringToDouble(localStr,&localVal);
	else err = -1;
	if(!err) *val = localVal;
	return err;
}

/////////////////////////////////////////////////
OSErr WizardGetParameterAsLong(char * key, char* line, long * val)
{
	// we use the wizard get parameter functions
	char localStr[MAXNUMPARAMCHARS];
	long localVal;
	OSErr err = -1;
	WizardGetParameterString(key,line,localStr,MAXNUMPARAMCHARS);
	if(localStr[0]) 
	{
		long numScanned = sscanf(localStr,"%ld",&localVal);
		if(numScanned ==1) 
		{
			*val = localVal;
			return noErr;
		}
	}
	return err;
}
/////////////////////////////////////////////////
OSErr WizardGetParameterAsShort(char * key, char* line, short * val)
{
	// we use the wizard get parameter functions
	long localLong = 0;
	OSErr err = WizardGetParameterAsLong(key,line,&localLong);
	if(!err) *val = (short)localLong;
	return err;
}
/////////////////////////////////////////////////
OSErr WizardGetParameterAsBoolean(char * key, char* line, Boolean * val)
{// returns error if the data is not a single number
	char localStr[MAXNUMPARAMCHARS];
	WizardGetParameterString(key,line,localStr,MAXNUMPARAMCHARS);
	switch(localStr[0])
	{
		case 'T':
		case 't':
		case '1':
			*val = true; return 0;
		case 'F':
		case 'f':
		case '0':
			*val = false; return 0;
		default: 
			return -1; //error
	}
	return -1;
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////


OSErr WriteWizardModelSettings(BFPB *bfpb)
{
	char s[512],temp[256];
	OSErr err = 0;
	long count;
	TModelDialogVariables var = model->GetDialogVariables();
	
	strcpy(s,"WSF_TYPE MODELSETTINGS;");
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

	sprintf(s,"startTime %ld;", var.startTime);
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

	// to support GNOME's before 12/9/99, write out endTime rather than the duration
	sprintf(s,"endTime %ld;", var.startTime + var.duration);
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

	sprintf(s,"computeTimeStep %ld;", var.computeTimeStep);
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

	sprintf(s,"bUncertain %s;", (var.bUncertain ? "TRUE":"FALSE"));
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

	strcpy(s,NEWLINESTRING);
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

	return err;
	
writeError:
	TechError("WriteWizardModelSettings()", "FSWriteBuf()", err); 
	return err;
	
}
/////////////////////////////////////////////////

OSErr SetWizardModelSettings(char* line)
{
	OSErr err = 0;
	TModelDialogVariables var = model->GetDialogVariables();
	long tempLong;
		
	err = WizardGetParameterAsLong("startTime",line,&tempLong);
	if(!err) var.startTime = tempLong;
	
	// look for old style keyword ,GNOME's before 12/9/99
	err = WizardGetParameterAsLong("endTime",line,&tempLong);	
	if(!err) var.duration = tempLong - var.startTime;
	
	err = WizardGetParameterAsLong("computeTimeStep",line,&tempLong);	
	if(!err) var.computeTimeStep = tempLong;
	err = WizardGetParameterAsBoolean("bUncertain",line, (Boolean*) &(var.bUncertain));

	model->SetDialogVariables(var); // do this even if there is an error, so we can see how far we got

	return noErr; // Parameters are optional

}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
	
OSErr WriteWizardWind(BFPB *bfpb)
{
	char s[512],temp[256];
	OSErr err = 0;
	long count,i;
	
	TWindMover *wind = model->GetWindMover(false); // don't create
	if(wind) 
	{
	
		strcpy(s,"WSF_TYPE WIND;");
		count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
		sprintf(s,"fSpeedScale %g;", wind->fSpeedScale);
		count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
		sprintf(s,"fAngleScale %g;", wind->fAngleScale);
		count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
		//sprintf(s,"windageA %g;", wind->windageA);
		//count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
		//sprintf(s,"windageB %g;", wind->windageB);
		//count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
		sprintf(s,"fDuration %g;", wind->fDuration);
		count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
		sprintf(s,"fUncertainStartTime %ld;", wind->fUncertainStartTime);
		count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
		sprintf(s,"fIsConstantWind %s;", (wind->fIsConstantWind ? "TRUE":"FALSE"));
		count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

		sprintf(s,"fConstantValue.u %g;", wind->fConstantValue.u);
		count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
		sprintf(s,"fConstantValue.v %g;", wind->fConstantValue.v);
		count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
		if(wind->timeDep)
		{
			short userUnits = wind->timeDep -> GetUserUnits();
			sprintf(s,"userUnits %hd;", userUnits);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
		}

		strcpy(s,NEWLINESTRING);
		count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
		/////////////////////////////////////////////////
		// now write a line for each record of the timeDep
		///
		if(wind->timeDep)
		{
			long n = wind->timeDep->GetNumValues (); 
			TimeValuePair pair;
		
			for(i = 0; i < n; i++)
			{
				strcpy(s,"WSF_TYPE WINDRECORD;");
				count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
				pair = INDEXH(wind->timeDep -> GetTimeValueHandle (), i);
				
				sprintf(s,"time %ld;", pair.time);
				count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

				sprintf(s,"u %g;", pair.value.u);
				count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
				
				sprintf(s,"v %g;", pair.value.v);
				count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

				strcpy(s,NEWLINESTRING);
				count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
			}
		}
	}

	return err;
	
writeError:
	TechError("WriteWizardWind()", "FSWriteBuf()", err); 
	return err;
	
}

/////////////////////////////////////////////////

OSErr SetWizardWind(char* line)
{
	OSErr err = 0;
	EWDialogNonPtrFields var ;
	short userUnits;
	TWindMover *wind = model->GetWindMover(true); // create if needed

	memset(&var,0,sizeof(var));
		
	if(wind) 
	{
		var = GetEWDialogNonPtrFields(wind);
		var.bActive = true;
		
		err = WizardGetParameterAsDouble("fSpeedScale",line,&var.fSpeedScale);
		err = WizardGetParameterAsDouble("fAngleScale",line,&var.fAngleScale);
		//err = WizardGetParameterAsDouble("windageA",line,&var.windageA);
		//err = WizardGetParameterAsDouble("windageB",line,&var.windageB);
		err = WizardGetParameterAsDouble("fDuration",line,&var.fDuration);
		err = WizardGetParameterAsLong("fUncertainStartTime",line,(long*) &var.fUncertainStartTime);
		err = WizardGetParameterAsBoolean("fIsConstantWind",line,(Boolean*) &(var.fIsConstantWind));
		err = WizardGetParameterAsDouble("fConstantValue.u",line,(double*) &(var.fConstantValue.u));
		err = WizardGetParameterAsDouble("fConstantValue.v",line,(double*) &(var.fConstantValue.v));
	
		err = WizardGetParameterAsShort("userUnits",line,&userUnits);
		if(!err && wind->timeDep ) wind->timeDep -> SetUserUnits(userUnits);
		
		//if(err) printError("Unable to read parameters in SetWizardWind()");
	
		SetEWDialogNonPtrFields(wind,&var); // do this even if there is an error, so we can see how far we got
	}

	return noErr; // we'll let the parameters be optional

}

/////////////////////////////////////////////////

OSErr AddWizardWindRecord(char* line)
{
	OSErr err = 0;
	TWindMover *wind = model->GetWindMover(true); // create if needed
	if(wind && wind->timeDep) 
	{
		TimeValuePair pair;
		
		err = WizardGetParameterAsLong("time",line,(long*) &pair.time);
		if(!err) err = WizardGetParameterAsDouble("u",line,&pair.value.u);
		if(!err) err = WizardGetParameterAsDouble("v",line,&pair.value.v);
	
		if(err) printError("Unable to read parameters in AddWizardWindRecord()");
		else 
		{	// append to list
			long len,newLen;
			TimeValuePairH tvalh = wind->timeDep->GetTimeValueHandle();
			if(tvalh) len = _GetHandleSize((Handle)tvalh);
			else len = 0;
			newLen = len +sizeof(pair);
			if(tvalh)	_SetHandleSize((Handle)tvalh,newLen);
			else
			{ // create and record the handle 
				tvalh = (TimeValuePairH)_NewHandleClear(newLen);
				wind->timeDep->SetTimeValueHandle(tvalh);
			}
			err = _MemError();
			if(tvalh && !err && newLen == _GetHandleSize((Handle)tvalh))
			{
				long n = len/sizeof(pair);
				INDEXH(tvalh, n) = pair;
			}
			else printError("Out of memory in AddWizardWindRecord()");
		}
		
	}
	return err;
}

/////////////////////////////////////////////////



/////////////////////////////////////////////////
/////////////////////////////////////////////////
	
OSErr WriteWizardSpills(BFPB *bfpb)
{
	char s[512],temp[256];
	OSErr err = 0;
	long count,i,numLESets;
	TOLEList *thisLEList;
	LETYPE leType;
	
	numLESets = model->LESetsList->GetItemCount();
	for(i = 0; i< numLESets; i++)
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == FORECAST_LE )
		{	//we only need to write out the forecast LE's

			long theClassID = thisLEList -> GetClassID();
		
			strcpy(s,"WSF_TYPE SPILL;");
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			// there is more than one type of LE set as of 10/24/99
			// write out the type
			{
				switch (theClassID) {
					case TYPE_OSSMLELIST:
						sprintf(s,"classID OSSMLELIST;");
						count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
						break;
					case TYPE_SPRAYLELIST:
						sprintf(s,"classID SPRAYLELIST;");
						count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
						break;
					default:
						printError("Unrecognized LE type inWriteWizardSpills");
						return -1;
						break;
				}
			}

			sprintf(s,"numOfLEs %ld;", thisLEList->fSetSummary.numOfLEs);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"pollutantType %ld;", thisLEList->fSetSummary.pollutantType);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"totalMass %g;", thisLEList->fSetSummary.totalMass);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"massUnits %d;", thisLEList->fSetSummary.massUnits);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"startRelTime %ld;", thisLEList->fSetSummary.startRelTime);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"endRelTime %ld;", thisLEList->fSetSummary.endRelTime);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"startRelPos.pLat %ld;", thisLEList->fSetSummary.startRelPos.pLat);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"startRelPos.pLong %ld;", thisLEList->fSetSummary.startRelPos.pLong);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"endRelPos.pLat %ld;", thisLEList->fSetSummary.endRelPos.pLat);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"endRelPos.pLong %ld;", thisLEList->fSetSummary.endRelPos.pLong);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"bWantEndRelTime %s;", (thisLEList->fSetSummary.bWantEndRelTime ? "TRUE":"FALSE"));
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"bWantEndRelPosition %s;", (thisLEList->fSetSummary.bWantEndRelPosition ? "TRUE":"FALSE"));
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"z %g;", thisLEList->fSetSummary.z);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			sprintf(s,"density %g;", thisLEList->fSetSummary.density);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
			
			//sprintf(s,"windageA %g;", thisLEList->fWindageData.windageA);
			//count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
			
			//sprintf(s,"windageB %g;", thisLEList->fWindageData.windageB);
			//count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
			
			//sprintf(s,"persistence %g;", thisLEList->fWindageData.persistence);
			//count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
			
			// age (at release) was added as a summary parameter 10/24/99
			sprintf(s,"age %lf;", thisLEList->fSetSummary.ageInHrsWhenReleased);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
			
			sprintf(s,"name %s;", thisLEList->fSetSummary.spillName);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;

			strcpy(s,NEWLINESTRING);
			count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
			
			if(theClassID ==TYPE_SPRAYLELIST)
			{	// write out the sprayed points
				TSprayLEList * sprayedList = (TSprayLEList *)thisLEList ;
				long i,n = 0;
				WorldPoint wp;
				if (sprayedList -> fSprayedH)
					n = sprayedList -> fNumSprayedPts;
				for (i = 0; i < n ; i++)
				{
					wp = INDEXH(sprayedList -> fSprayedH,i);
					sprintf(s,"WSF_TYPE SPRAYEDPT; pLat %ld; pLong %ld;%s",wp.pLat,wp.pLong,NEWLINESTRING);
					count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
				}
			}
		}
	}

	return err;
	
writeError:
	TechError("WriteWizardSpills()", "FSWriteBuf()", err); 
	return err;
	
}
/////////////////////////////////////////////////

 
OSErr AddWizardSprayedPt(char* line)
{	// add the sprayed point the the last Sprayed LE set in the list
	OSErr err = 0;
	TSprayLEList 	*forecastLEList=0;
	WorldPoint wp;
	
	err = WizardGetParameterAsLong("pLat",line,&wp.pLat);
	if (!err) err = WizardGetParameterAsLong("pLong",line,&wp.pLong);

	if(err) {
		printError("Unable to read parameters in AddWizardSprayedPt()");
		return err;
	}
	
		
	///////////
	// get forecastLEList
	// check all the LE sets
	{
		long i,n;
		TLEList *thisLEList;
		long classOfObject;
		long leType;

		n = model -> LESetsList -> GetItemCount ();

		for (i = n -1 ; i >= 0; i--)
		{
			model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
			classOfObject = thisLEList -> GetClassID();
			leType = thisLEList -> fLeType;
			if (classOfObject == TYPE_SPRAYLELIST && leType == FORECAST_LE) {
				// we've found the last one in the list
				forecastLEList = (TSprayLEList*) thisLEList;
				break;
			}
		}
	}
	/////////////////
	
	if (forecastLEList)
		 forecastLEList -> AddSprayPoint(wp);
	else {
		printError("Unable to find TSpillLEList object in AddWizardSprayedPt()");
		return err;
	}
		
	return err;
	 
}

/////////////////////////////////////////////////


OSErr AddWizardSpill(char* line)
{
	OSErr err = 0;
	LESetSummary  summary;
	TOLEList 	*forecastLEList=0;
	TOLEList 	*uncertaintyLEList=0;
	long classID;
	
	memset(&summary,0,sizeof(summary));
	
	// 10/24/99 we added the classID
	// before that all sets were TYPE_OSSMLELIST
	{
		char typeStr[64];
		WizardGetParameterString("classID",line,typeStr,63);
		if (typeStr[0] == 0 || !strcmpnocase(typeStr,"OSSMLELIST")) // typeStr[0] == 0 would be a pre 10/24/99 file
			classID = TYPE_OSSMLELIST;
		else if (!strcmpnocase(typeStr,"SPRAYLELIST"))
			classID = TYPE_SPRAYLELIST;
		else {
			printError("Unrecognized classID parameter in AddWizardSpill");
			return -1;
		}
	}

	err = WizardGetParameterAsLong("numOfLEs",line,&summary.numOfLEs);
	if(!err) err = WizardGetParameterAsShort("pollutantType",line,&summary.pollutantType);
	if(!err) err = WizardGetParameterAsDouble("totalMass",line,&summary.totalMass);
	if(!err) err = WizardGetParameterAsShort("massUnits",line,&summary.massUnits);
	if(!err) err = WizardGetParameterAsLong("startRelTime",line,(long*)&summary.startRelTime);
	if(!err) err = WizardGetParameterAsLong("endRelTime",line,(long*)&summary.endRelTime);
	if(!err) err = WizardGetParameterAsLong("startRelPos.pLat",line,&summary.startRelPos.pLat);
	if(!err) err = WizardGetParameterAsLong("startRelPos.pLong",line,&summary.startRelPos.pLong);
	if(!err) err = WizardGetParameterAsLong("endRelPos.pLat",line,&summary.endRelPos.pLat);
	if(!err) err = WizardGetParameterAsLong("endRelPos.pLong",line,&summary.endRelPos.pLong);
	if(!err) err = WizardGetParameterAsBoolean("bWantEndRelTime",line,&summary.bWantEndRelTime);
	if(!err) err = WizardGetParameterAsBoolean("bWantEndRelPosition",line,&summary.bWantEndRelPosition);
	if(!err) err = WizardGetParameterAsDouble("z",line,&summary.z);
	if(!err) err = WizardGetParameterAsDouble("density",line,&summary.density);

	// windage was moved from windmover to LEs as a summary parameter 6/18/01 
	/*if (!err)
	{
		OSErr windageErr = WizardGetParameterAsDouble("windageA",line,&summary.windageA);
		if (windageErr) summary.windageA = .01;
		windageErr = WizardGetParameterAsDouble("windageB",line,&summary.windageB);
		if (windageErr) summary.windageB = .04;
		windageErr = WizardGetParameterAsDouble("persistence",line,&summary.persistence);
		if (windageErr) summary.persistence = .25;	// in hours
	}*/
	// age (at release) was added as a summary parameter 10/24/99 
	// if not present, assume 0
	if(!err) {
		OSErr ageErr = WizardGetParameterAsDouble("age",line,&summary.ageInHrsWhenReleased);
		if(ageErr || classID == TYPE_OSSMLELIST)	// old save files have random numbers for age
																// since the field wasn't being initialized to zero 11/1/00
																// this field is not used for point/line spill LEs so set to zero
			summary.ageInHrsWhenReleased = 0;
	}
	
	if (!err)
	{	
		char nameStr[kMaxNameLen];
		WizardGetParameterString("name",line,nameStr,kMaxNameLen-1);
		if(nameStr[0]) 
		{
			strcpy(summary.spillName,nameStr);
		}
		/*else
		{
			if (classID==TYPE_OSSMLELIST)
				strcpy(summary.spillName,"Point Source");
			else
				strcpy(summary.spillName,"Overflight");
		}*/
	}
	
	if(err) printError("Unable to read parameters in AddWizardSpill()");
	else
	{	// add the LE set
		
		// now add LE sets
		switch (classID)
		{
			case TYPE_OSSMLELIST:
				forecastLEList = new TOLEList ();
				uncertaintyLEList = new TOLEList ();
				break;
			case TYPE_SPRAYLELIST:
				forecastLEList = (TOLEList*) new TSprayLEList ();
				uncertaintyLEList =(TOLEList*) new TSprayLEList ();
				break;
		}
		if (!forecastLEList || ! uncertaintyLEList) goto memErrCode; 
		err = forecastLEList->Initialize(&summary,true);
		if(err) goto done;

		uncertaintyLEList->fLeType = UNCERTAINTY_LE;
		uncertaintyLEList->fOwnersUniqueID = forecastLEList->GetUniqueID();
		err = uncertaintyLEList->Initialize(&summary,true);
		if(err) goto memErrCode;

		err = model->AddLEList(forecastLEList, 0);
		if(err) goto memErrCode;
		err = model->AddLEList(uncertaintyLEList, 0);
		if(err) goto memErrCode;
		
	}

done:
	return err;
	
memErrCode:
	if(forecastLEList) {forecastLEList->Dispose(); delete forecastLEList; forecastLEList = 0;}
	if(uncertaintyLEList) {uncertaintyLEList->Dispose(); delete uncertaintyLEList; uncertaintyLEList = 0;}
	printError("Out of memory in AddWizardSpill()");
	return memFullErr;
	
}

/////////////////////////////////////////////////

void AddDelimiterAtEndIfNeeded(char* str)
{	// add delimiter at end if the last char is not a delimiter
	long len = strlen(str);
	if(str[len-1] != DIRDELIMITER) 
	{
		str[len] = DIRDELIMITER;
		str[len+1] = 0;
	}
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

OSErr GetLocaleFilePath(CHARH f,char* saveFilePath, char* locationFilePath)
{
	OSErr err = 0;
	// note we don't use WizardGetParameterString
	// in case the path contains the ';' char
	char* path = nil;
	char* wfLineStart = "WSF_TYPE LOCATIONFILEPATH;";
	long wfLen = strlen(wfLineStart);
	char* thisFileLineStart = "WSF_TYPE THISFILEPATH;";
	long tfLen = strlen(thisFileLineStart);
	char* appLineStart = "WSF_TYPE APPLICATIONFOLDER;";
	long appLen = strlen(appLineStart);
	long lineStart;
	char line[1024];

	char oldLocationFilePath[256] ="";
	char oldThisFilePath[256] ="";
	char oldAppFolder[256] ="";
	
	char oldFolder[256] ="";
	//char oldThisFileFolder[256] ="";
	
	//char commonFolderAncestor[256];

	char shortLocationFileName[256];
	char possiblePath[256];
	char tempStr[256];
	long len,len1,len2;
	
	Boolean gotLine;
	Boolean thisFileWasMovedOrRenamed;
	long i;
	
	locationFilePath[0] = 0;
	
	// process the lines

	// since we are looping through the lines
	// use the more efficient GetLineFromHdlHelper
	//for(lineNum1Relative = 1; !err;lineNum1Relative++) // forever
	for(lineStart = 0; !err;)
	{
		//gotLine = GetLineFromHdl(lineNum1Relative,f,line,1023);
		gotLine = GetLineFromHdlHelper(1,lineStart,&lineStart,f,line,1023,true);
		
		if(!gotLine) break;// no more lines
		if(!line[0]) continue; // blank line

		if(!strncmpnocase(line,wfLineStart,wfLen))
		{
			path = line + wfLen;
			strcpy(oldLocationFilePath,path);
		}
		if(!strncmpnocase(line,appLineStart,appLen))
		{
			path = line + appLen;
			strcpy(oldAppFolder,path);
			AddDelimiterAtEndIfNeeded(oldAppFolder);
		}
		if(!strncmpnocase(line,thisFileLineStart,tfLen))
		{
			path = line + tfLen;
			strcpy(oldThisFilePath,path);
		}
	}
	
	// now see if the old path is still good
	strcpy(possiblePath,oldLocationFilePath);
	if(possiblePath[0] && FileExists(0, 0, possiblePath)) 
	{
		strcpy(locationFilePath,possiblePath);
		return noErr;
	}
		
	/////////
	// OK, the old location file path is no longer good
	// try other placers we can think of
	/////////
	
	// get the short name of the location file we are looking for
	strcpy(possiblePath,oldLocationFilePath);
	SplitPathFile(possiblePath,shortLocationFileName);

	////////////////////////////////
	// the location file may be in the same directory as the current application
	// look there !!! (Note: this works for Spill conference setup )
	////////////////////////////////
	MyGetFolderName(TATvRefNum,TATdirID,TRUE,possiblePath);
	AddDelimiterAtEndIfNeeded(possiblePath);
	strcat(possiblePath,shortLocationFileName);
	if(possiblePath[0] && FileExists(0, 0, possiblePath)) 
	{
		strcpy(locationFilePath,possiblePath);
		return noErr;
	}
	
	////////////////////////////////
	// the location file may be in the same directory as this save file
	////////////////////////////////
	strcpy(possiblePath,saveFilePath);
	SplitPathFile(possiblePath,tempStr);
	AddDelimiterAtEndIfNeeded(possiblePath);
	strcat(possiblePath,shortLocationFileName);
	if(possiblePath[0] && FileExists(0, 0, possiblePath)) 
	{
		strcpy(locationFilePath,possiblePath);
		return noErr;
	}
	
	////////////////////////////////
	// maybe the location file and the save file were in a common folder that got renamed or moved
	// so that the leadingPart of the the path has changed but the endingPart is unchanged.
	////////////////////////////////
	strcpy(possiblePath,saveFilePath);
	SplitPathFile(possiblePath,tempStr);
	AddDelimiterAtEndIfNeeded(possiblePath);
	strcpy(oldFolder,oldThisFilePath);
	SplitPathFile(oldFolder,tempStr);
	AddDelimiterAtEndIfNeeded(oldFolder);
	len1 = strlen(possiblePath);
	len2 = strlen(oldFolder);
	len = _min(len1,len2);
	//  move back as far as the strings are equal
	for(i = 0; i < len; i++)
	{
		if(possiblePath[len1-1-i] != oldFolder[len2-1-i]) break;
	}
	possiblePath[len1-i] = 0; // cut after first different character
	oldFolder[len2-i] = 0; // cut after first different character
	// try substituting possiblePath for oldFolder in the oldLocationFilePath
	if(!strncmpnocase(oldLocationFilePath,oldFolder,len2-i))
	{
		strcat(possiblePath,oldLocationFilePath+len2-i);
		if(possiblePath[0] && FileExists(0, 0, possiblePath)) 
		{
			strcpy(locationFilePath,possiblePath);
			return noErr;
		}
	}
	/////////////////////////////////////////////////

	
	////////////////////////////////
	// maybe the location file and the application were in a common folder that got renamed or moved
	// so that the leadingPart of the the path has changed but the endingPart is unchanged.
	////////////////////////////////
	MyGetFolderName(TATvRefNum,TATdirID,TRUE,possiblePath);
	AddDelimiterAtEndIfNeeded(possiblePath);
	strcpy(oldFolder,oldAppFolder);
	len1 = strlen(possiblePath);
	len2 = strlen(oldFolder);
	len = _min(len1,len2);
	//  move back as far as the strings are equal
	for(i = 0; i < len; i++)
	{
		if(possiblePath[len1-1-i] != oldFolder[len2-1-i]) break;
	}
	possiblePath[len1-i] = 0; // cut after first different character
	oldFolder[len2-i] = 0; // cut after first different character
	// try substituting possiblePath for oldFolder in the oldLocationFilePath
	if(!strncmpnocase(oldLocationFilePath,oldFolder,len2-i))
	{
		strcat(possiblePath,oldLocationFilePath+len2-i);
		if(possiblePath[0] && FileExists(0, 0, possiblePath)) 
		{
			strcpy(locationFilePath,possiblePath);
			return noErr;
		}
	}
	/////////////////////////////////////////////////

	
	// otherwise ask the user
	sprintf(line,"This save file references a location file which cannot be found.  Please find the file \"%s\".",shortLocationFileName);
	printNote(line);
	err = AskForSpecificWizardLocationFile(oldLocationFilePath);
	if(!err) strcpy(locationFilePath,oldLocationFilePath);
	return err;

}


OSErr WizardFile::SetWizardFileStuff(char* line,char* answerStr)
{
	OSErr err = 0;
	Boolean b;
	short index;
	char str[512];
	Boolean isWind = !strcmpnocase(answerStr,"USEVARIABLEWIND") ?true:false;
	Boolean isAnswer = !strcmpnocase(answerStr,"ANSWER") ?true:false;
	Boolean isMessage = !strcmpnocase(answerStr,"MESSAGE") ?true:false;
	long len;
	CHARH h = 0;
	char *lineStart = 0;
	
	
	if(isWind){
		err = WizardGetParameterAsBoolean("VALUE",line,&b);
		if(!err) this->fUseVariableWind  = b;
	}
	else  if(isAnswer || isMessage)
	{
		char *valueLabel = "VALUE ";
		err = WizardGetParameterAsShort("INDEX",line,&index);
		// because the VALUE has ';' chars in it, we can't use WizardGetParameterString here
		// WizardGetParameterString("VALUE",line,str,500);
		// move past the VALUE ###; then strcpy
		lineStart = strstr(line,valueLabel);
		if(!lineStart) { printError("VALUE param not found in SetWizardStuff");return -1; }
		lineStart+= strlen(valueLabel); // move past the label
		strncpy(str,lineStart,500);
		str[500] = 0;
		
		if(!err && str[0])
		{	// put it in a handle 
			// note: answers have returns at the end of the line and are null terminated
			if(index < 0 || index >= kMAXNUMWIZARDDIALOGS)
				{ printError("Index out of range in SetWizardStuff");return -1; }
			strcat(str,NEWLINESTRING);
			// NOTE: the messages can be multiple lines, in which case we have to concat the lines
			if(isAnswer) h = this->fAnswer[index];
			else if(isMessage) h = this->fMessage[index];
			if(!h) 
			{	// new to make a new handle
				len = strlen(str) + 1;
				h = (CHARH) _NewHandle(len);
				if(!h) { WizardOutOfMemoryAlert(); return memFullErr;}
				strcpy(*h,str);
			}
			else
			{ // we need to concat them
				long newLen;
				len = _GetHandleSize((Handle)h);
				if(len == 0) len = 1;// pretend it was a string with a null terminator
				newLen = len + strlen(str);
				_SetHandleSize((Handle)h,newLen);
				if(newLen != _GetHandleSize((Handle)h)) { WizardOutOfMemoryAlert(); return memFullErr;}
				strcpy((*h) + (len -1) ,str);
			}
			// reassign in case we assigned h above
			if(isAnswer)this->fAnswer[index] = h;
			else if(isMessage)this->fMessage[index] = h;
		}
	}
	return err;
}

OSErr WizardFile::SaveFileGoThoughDialogs(CHARH f)
{
	OSErr err = 0;
	long lineStart;
	char line[2048];
	char answerStr[128];
	
	// read file and process the lines

	// since we are looping through the lines
	// use the more efficient GetLineFromHdlHelper
	//for(lineNum1Relative = 1; !err;lineNum1Relative++) // forever
	for(lineStart = 0; !err;)
	{
		//Boolean gotLine = GetLineFromHdl(lineNum1Relative,f,line,2047);
		Boolean gotLine = GetLineFromHdlHelper(1,lineStart,&lineStart,f,line,2047,true);
		
		if(!gotLine) break; // out of for loop, end of hdl
		if(!line[0]) continue; //  blank line
		WizardGetParameterString("WSF_TYPE",line,answerStr,127);
		
		if(!strcmpnocase(answerStr,"WIND")) err = SetWizardWind(line);
		else if(!strcmpnocase(answerStr,"MODELSETTINGS")) err = SetWizardModelSettings(line);
		else if(!strcmpnocase(answerStr,"WINDRECORD")) err = AddWizardWindRecord(line);
		else if(!strcmpnocase(answerStr,"USEVARIABLEWIND")) err = this->SetWizardFileStuff(line,answerStr);
		else if(!strcmpnocase(answerStr,"ANSWER")) err = this->SetWizardFileStuff(line,answerStr);
		else if(!strcmpnocase(answerStr,"MESSAGE")) err = this->SetWizardFileStuff(line,answerStr);
		//////////
		// note we don't do spill lines here, we do them in SaveFileAddSpills
		//else if(!strcmpnocase(answerStr,"SPILL")) err = AddWizardSpill(line);
		//else if(!strcmpnocase(answerStr,"SPRAYEDPT")) err = AddWizardSprayedPt(line);
		//////////////
		else if(!strcmpnocase(answerStr,"VERSION"))
		{	// JLM, 3/9/99 check the version numbers
			short num;
			char idStr[64];
			WizardGetParameterString("LOCATIONFILEIDSTR",line,idStr,64);
			if(strcmpnocase(idStr,this->fLocationFileIdStr))
			{
				printError("This location save file cannot be used.  The ID string of the referenced location file does not match the value in this save file.");
				return -1;
			}
			/////
			err = WizardGetParameterAsShort("LOCATIONFILEFORMAT",line,&num);
			if(err)
			{ 
				printError("LOCATIONFILEFORMAT is missing from the location save file");
				return -1;
			}
			else
			{
				if(this->fLocationFileFormat != num)
				{ // it is the wrong version
					printError("This location save file cannot be used.  The format of the referenced location file does not match the value in this save file.");	
					return -1;
				}
			}
			/////
			err = WizardGetParameterAsShort("LOCATIONFILEVERSION",line,&num);
			if(err)
			{ 
				printError("LOCATIONFILEVERSION is missing from the location save file");
				return -1;
			}
			else
			{
				if(this->fLocationFileVersion != num)
				{ // it is the wrong version
					printError("This location save file cannot be used.  The version number of the referenced location file does not match the value in this save file.");	
					return -1;
				}
			}
		}
	}
	
done:
	if(err)
	{  // should we deal with partially added stuff ??
		// close file or leave partialy added stuff ??
		// Maybe it's better to see how far we got
	}
	
	return err;
}
	
OSErr WizardFile::SaveFileAddSpills(CHARH f)
{
	OSErr err = 0;
	long lineStart;
	char line[1024];
	char answerStr[128];
	
	// read file and process the lines

	// since we are looping through the lines
	// use the more efficient GetLineFromHdlHelper
	//for(lineNum1Relative = 1; !err;lineNum1Relative++) // forever
	for(lineStart = 0; !err;)
	{
		//Boolean gotLine = GetLineFromHdl(lineNum1Relative,f,line,1023);
		Boolean gotLine = GetLineFromHdlHelper(1,lineStart,&lineStart,f,line,1023,true);
		
		if(!gotLine) break; // out of for loop, end of hdl
		if(!line[0]) continue; //  blank line
		WizardGetParameterString("WSF_TYPE",line,answerStr,127);
		
		if(!strcmpnocase(answerStr,"SPILL")) err = AddWizardSpill(line);
		else if(!strcmpnocase(answerStr,"SPRAYEDPT")) err = AddWizardSprayedPt(line);
	}
	
	model -> Reset(); // so that the sprayed LE sets are reset
	
	return err;
}
	
	
	

/////////////////////////////////////////////////
/////////////////////////////////////////////////


OSErr WizardFile::WriteWizardSaveFile(BFPB *bfpb,char * path)
{
	char s[512],temp[256];
	OSErr err = 0;
	long count,i;
	long lfsFormat = 0; // can change this when we support format greater than 0, JLM 3/10/99
	
	//write out version line
	strcpy(s,"WIZARDSAVEFILE");
	if(lfsFormat > 0)
	{	// if greater than 0, new style add the format version number to the first line
		sprintf(temp," %ld",lfsFormat);
	}
	strcat(s,NEWLINESTRING);
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
	// JLM, 3/9/99 write out the location file version numbers
	sprintf(s,"WSF_TYPE VERSION;LOCATIONFILEIDSTR %s;LOCATIONFILEFORMAT %ld;LOCATIONFILEVERSION %ld;",this->fLocationFileIdStr,this->fLocationFileFormat,this->fLocationFileVersion);
	strcat(s,NEWLINESTRING);
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
	// write out Wizard file path
	strcpy(s,"WSF_TYPE LOCATIONFILEPATH;");
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	this->GetPathName(s); // note! just the path name , no key
	strcat(s,NEWLINESTRING);
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
	// write out this file's path
	strcpy(s,"WSF_TYPE THISFILEPATH;");
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	strcpy(s,path); // note! just the path name , no key
	strcat(s,NEWLINESTRING);
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
	// write out GNOME path (so we can do partial paths)
	strcpy(s,"WSF_TYPE APPLICATIONFOLDER;");
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	MyGetFolderName(TATvRefNum,TATdirID,TRUE,s);
	AddDelimiterAtEndIfNeeded(s);
	strcat(s,NEWLINESTRING);
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
	// write out the flag for variable vs constant wind
	sprintf(s,"WSF_TYPE USEVARIABLEWIND;VALUE %s;", (this->fUseVariableWind ? "TRUE":"FALSE"));
	strcat(s,NEWLINESTRING);
	count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
	
	// write out answers and messages
	for(i = 0; i < kMAXNUMWIZARDDIALOGS; i++)
	{
		CHARH h = this->fAnswer[i];
		long hLen,hStrLen;
		if(h) 
		{
			// note: answers have returns at the end of the line and are null terminated
			/*hLen = _GetHandleSize((Handle)h);
			if(hLen > 2) // must have at least return and null char
			{
				hStrLen = strlen(*h);
				if(hStrLen> 0)
				{
					sprintf(s,"WSF_TYPE ANSWER;INDEX %ld;VALUE ",i);
					count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
					_HLock((Handle)h);
					count = hStrLen; err = FSWriteBuf(bfpb, &count, *h); if(err) goto writeError;
					_HUnlock((Handle)h);
				}
			}*/
			// note answers have returns at the end of the line and are null terminated
			// but may be multiple lines 9/24/03
			char line[1024];
			long lineLen,lineStart;

			for(lineStart = 0; !err;)
			{
				//GetLineFromHdl(lineNum1Relative,h,line,1020); // 1020 to allow room for NEWLINESTRING
				Boolean gotLine = GetLineFromHdlHelper(1,lineStart,&lineStart,h,line,1020,true);
				if(!gotLine) break;// no more lines
				if(!line[0]) continue; // blank line
				strcat(line,NEWLINESTRING);
				lineLen = strlen(line);
				sprintf(s,"WSF_TYPE ANSWER;INDEX %ld;VALUE ",i);
				count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
				count = lineLen;err = FSWriteBuf(bfpb, &count, line); if(err) goto writeError;
			}
		}
		////////////////////////////////////////////////
		h = this->fMessage[i];
		if(h)  
		{
			// note messages have returns at the end of the line and are null terminated
			// but may be multiple lines
			char line[1024];
			long lineLen,lineStart;

			for(lineStart = 0; !err;)
			{
				//GetLineFromHdl(lineNum1Relative,h,line,1020); // 1020 to allow room for NEWLINESTRING
				Boolean gotLine = GetLineFromHdlHelper(1,lineStart,&lineStart,h,line,1020,true);
				if(!gotLine) break;// no more lines
				if(!line[0]) continue; // blank line
				strcat(line,NEWLINESTRING);
				lineLen = strlen(line);
				sprintf(s,"WSF_TYPE MESSAGE;INDEX %ld;VALUE ",i);
				count = strlen(s); err = FSWriteBuf(bfpb, &count, s); if(err) goto writeError;
				count = lineLen;err = FSWriteBuf(bfpb, &count, line); if(err) goto writeError;
			}
		}
		/////////////////////////////////////////
	}
	
	return err;
	
writeError:
	TechError("WriteWizardSaveFile()", "FSWriteBuf()", err); 
	return err;
	
}



void WizardFile::InitWizardFile(void)
{
	long i;
	this->fPathName[0] = 0;
	this->fIsOpen = false;
	#ifdef IBM
		this->fInstDLL = 0;
	#else
		this->fResRefNum = 0;
	#endif
	this->Clear_Dialog();
	this->fText10000Hdl = 0;
	
	for(i = 0; i< kMAXNUMWIZARDDIALOGS;i++)
	{
		this->fAnswer[i] = 0;
		this->fMessage[i] = 0;
		this->fSavedAnswer[i] = 0;
		this->fSavedMessage[i] = 0;
	}
	
	this->fUseVariableWind = true; 
	this->fUseVariableWindSaved = 0; 
	
	this->fLocationFileIdStr[0] = 0;
	this->fLocationFileFormat = 0;
	this->fLocationFileVersion = 0;
	this->fLocationFileExpirationDate = 0;
	
	this->fMinKilometersPerInch = 0; // zero or negative means not limited (by the location file)
	
	this->ClearCommandVariables();
}
/////////////////////////////////////////////////
WizardFile::WizardFile(void)
{// constructor
	this->InitWizardFile();
}
/////////////////////////////////////////////////
WizardFile::WizardFile(char *pathName)
{// constructor
	this->InitWizardFile(); 
	strcpy(this->fPathName,pathName);
	this->ClearLeftHandSideAnswers();
}
/////////////////////////////////////////////////
WizardFile::~WizardFile(void)
{// destructor
	this->Dispose();
}
/////////////////////////////////////////////////
Boolean WizardFile::GetMinKilometersPerScreenInch(float *minKilometers)
{
	if(this->fMinKilometersPerInch > 0) // zero or negative means not limited (by the location file))
	{
		*minKilometers = this->fMinKilometersPerInch;
		return true;
	}
	return false;
}
/////////////////////////////////////////////////

void WizardFile::ClearLeftHandSideAnswers(void)
{ // clears if tracker and variables
	short i;
	//
	for(i = 0; i < MAX_NUM_LHS_ANSWERS; i++)
	{
		memset(&this->fLHSAnswer[i],0,sizeof(this->fLHSAnswer[i]));
	}
}
/////////////////////////////////////////////////
OSErr WizardFile::GetFreeIndexLHSAnswers(short* index)
{ 
	short i;
	for(i = 0; i < MAX_NUM_LHS_ANSWERS; i++)
	{
		if(this->fLHSAnswer[i].text[0] == 0) 
		{
			*index = i;
			return noErr;
		}
	}
	return -1; // no free space
}
/////////////////////////////////////////////////
short WizardFile::NumLeftHandSideAnswersInList(void)
{ // count those that have text and the boolean set
	short i = 0,num = 0;
	//
	for(i = 0; i < MAX_NUM_LHS_ANSWERS; i++)
	{	
		if(this->fLHSAnswer[i].lhs && this->fLHSAnswer[i].text[0] != 0) num++;
	}
	return num;
}
/////////////////////////////////////////////////
short WizardFile::NumLeftHandSideAnswersOnPrintout(void)
{ // count those that have text and the boolean set
	short i = 0,num = 0;
	for(i = 0; i < MAX_NUM_LHS_ANSWERS; i++)
	{	
		if(this->fLHSAnswer[i].print && this->fLHSAnswer[i].text[0] != 0) num++;
	}
	return num;
}

/////////////////////////////////////////////////
void	WizardFile::LeftHandSideTextForListOrPrintoutHelper(short desired1RelativeLineNum,char* text,Boolean forPrintout)
{	
	short i = 0,num = 0;
	text[0] = 0;
	for(i = 0; i < MAX_NUM_LHS_ANSWERS; i++)
	{	
		if((forPrintout && this->fLHSAnswer[i].print) || (!forPrintout && this->fLHSAnswer[i].lhs))
		{
			if(this->fLHSAnswer[i].text[0] != 0) 
			{
				num++;
				if(num == desired1RelativeLineNum)
					{ strcpy(text,this->fLHSAnswer[i].text); return; }
			}
		}
	}
}
void	WizardFile::LeftHandSideTextForList(short desired0RelativeLineNum,char* text)
{
	short desired1RelativeLineNum = desired0RelativeLineNum+1;
	this->LeftHandSideTextForListOrPrintoutHelper(desired1RelativeLineNum,text,false);
}
/////////////////////////////////////////////////
void	WizardFile::LeftHandSideTextForPrintout(short desired0RelativeLineNum,char* text)
{
	short desired1RelativeLineNum = desired0RelativeLineNum+1;
	this->LeftHandSideTextForListOrPrintoutHelper(desired1RelativeLineNum,text,true);
}
/////////////////////////////////////////////////

Boolean	WizardFile::LeftHandSideAnswerSettingsItemHit(short i)
{ // handle user clicking on a left hand side item
	short settingsDialogResNum = 0;
	if(i < MAX_NUM_LHS_ANSWERS)
		settingsDialogResNum = this->fLHSAnswer[i].settingsDialogResNum;
	
	if(settingsDialogResNum>0)
	{ // do that dialog
		// code could go here if we only wanted to do that dialog
		// but we would have to deal with the changing of he left hand side etc
		// we'll save this for another time
		//return true;// we handled it
	}
	return false;// we did not handle it
}
/////////////////////////////////////////////////
void WizardFile::Dispose(void)
{// destructor
	this->CloseFile();
	this->Dispose_Dialog();
	this->Dispose_Answers();
	this->Dispose_SavedAnswers();
	
}
/////////////////////////////////////////////////
void WizardFile::ClearCommandVariables(void)
{ // clears if tracker and variables
	short i;
	this->fIfLevel = 0;	
	for(i = 0; i <= MAX_IF_LEVEL; i++) 
		this->fIfValue[i] = true;
	//
	for(i = 0; i < MAX_NUM_WIZ_VARIABLES; i++)
	{
		this->fCommandVariable[i].name[0] = 0;
		this->fCommandVariable[i].value[0] = 0;
	}
}

/////////////////////////////////////////////////
void	WizardFile::Clear_Dialog(void)
{
	memset((void *)this->fItem, 0, kMAXNUMWIZARDDIALOGITEMS*sizeof(WizDialogItemInfo));
	fDialogResNum = 0;
}
/////////////////////////////////////////////////
void	WizardFile::Dispose_Dialog(void)
{
	short i;
	for(i = 0; i< kMAXNUMWIZARDDIALOGITEMS; i++)
		if(this->fItem[i].hdl) { DisposeHandle((Handle)this->fItem[i].hdl);this->fItem[i].hdl = 0; }
	this->Clear_Dialog();
}
/////////////////////////////////////////////////
OSErr WizardFile::SaveAnswers(void)
{
	OSErr err = 0;
	CHARH h;
	long i;
	this->Dispose_SavedAnswers(); // get rid of the old, just in case
	
	this->fUseVariableWindSaved = this->fUseVariableWind; 

	for(i = 0; !err && i< kMAXNUMWIZARDDIALOGS;i++)
	{
		h = this->fAnswer[i];
		if(h) 
		{ 
			err = _HandToHand((Handle*)&h);
			if(err) break;
			this->fSavedAnswer[i] = h;
		}
		h = this->fMessage[i];
		if(h) 
		{ 
			err = _HandToHand((Handle*)&h);
			if(err) break;
			this->fSavedMessage[i] = h;
		}
	}
	if(err) 
	{
		this->Dispose_SavedAnswers();
		WizardOutOfMemoryAlert();
	}
	return err;
}
/////////////////////////////////////////////////
void WizardFile::RestoreAnswers(void)
{
	long i;
	this->Dispose_Answers(); // get rid of the old, just in case
	for(i = 0; i< kMAXNUMWIZARDDIALOGS;i++)
	{
		this->fAnswer[i] = this->fSavedAnswer[i]; this->fSavedAnswer[i] = nil;
		this->fMessage[i] = this->fSavedMessage[i]; this->fSavedMessage[i] = nil;
	}

	this->fUseVariableWind = this->fUseVariableWindSaved; this->fUseVariableWindSaved = 0;

}
/////////////////////////////////////////////////
void WizardFile::Dispose_SavedAnswers(void)
{
	long i;
	for(i = 0; i< kMAXNUMWIZARDDIALOGS;i++)
	{
		if(this->fSavedAnswer[i]) { DisposeHandle((Handle)this->fSavedAnswer[i]);this->fSavedAnswer[i] = 0; }
		if(this->fSavedMessage[i]) { DisposeHandle((Handle)this->fSavedMessage[i]);this->fSavedMessage[i] = 0; }
	}

	this->fUseVariableWindSaved = 0; //(not really a dispose issue, but what the heck)
}
/////////////////////////////////////////////////
void WizardFile::Dispose_Answers(void)
{
	long i;
	for(i = 0; i< kMAXNUMWIZARDDIALOGS;i++)
	{
		if(this->fAnswer[i]) { DisposeHandle((Handle)this->fAnswer[i]);this->fAnswer[i] = 0; }
		if(this->fMessage[i]) { DisposeHandle((Handle)this->fMessage[i]);this->fMessage[i] = 0; }
	}
	
	this->fUseVariableWind = 0; //(not really a dispose issue, but what the heck)
}
/////////////////////////////////////////////////
OSErr WizardFile::SetPathName(char *pathName)
{
	this->CloseFile();
	strcpy(this->fPathName,pathName);
	return noErr;
}
/////////////////////////////////////////////////
void WizardFile::GetPathName(char *pathName)
{
	strcpy(pathName,this->fPathName);
}
/////////////////////////////////////////////////

Boolean IsSupportedFormat(short locationFileFormat)
{
	if(locationFileFormat == 0) return true;
	else return false;
}

/////////////////////////////////////////////////
OSErr WizardFile::OpenFile(void)
{
	this->CloseFile(); // just to be sure
	#ifdef IBM
	{
		HINSTANCE instDLL = LoadLibrary(this->fPathName);
		if(!instDLL) return -1; // failure
		this->fIsOpen = true;
		this->fInstDLL = instDLL;
	}
	#else
	{
#if MACB4CARBON
		long f = openresfile(this->fPathName);
		if(f == -1)  return -1; // openresfile() failure
		this->fIsOpen = true;
		this->fResRefNum = f;
#else
		Str255 fpname;
		CopyCStringToPascal(this->fPathName,fpname);
		long f = HOpenResFile(0,0,fpname,fsCurPerm);
		//long f = openresfile(this->fPathName);
		if(f == -1)  return -1; // openresfile() failure
		this->fIsOpen = true;
		this->fResRefNum = f;
#endif
	}
	#endif
	
	// get the order resource from TEXT 10000
	if(this->fText10000Hdl) DisposeHandle((Handle)this->fText10000Hdl);
	this->fText10000Hdl = this->GetResource('TEXT',10000,true); 
	if(!this->fText10000Hdl) // resource failure
	{
		OSErr err = 0;
		if (err = ReadFileContents(TERMINATED,0, 0, this->fPathName, 0, 0, &(this->fText10000Hdl)))
		{	
		this->CloseFile();
		return -1;
		}
	}
	
	// get the version numbers
	(void) this->DoCommandBlock("[VERSION]"); // this sets the version fields

	if(!IsSupportedFormat(this->fLocationFileFormat))
	{ 	// if format is not supported tell the user
		printError("This location file's format is not supported by this version of GNOME.");
		this->CloseFile();
		return -1;
	}
	
	return noErr;
}
/////////////////////////////////////////////////
OSErr WizardFile::OpenFile(char *pathName)
{
	this->SetPathName(pathName);
	return this->OpenFile();
}
/////////////////////////////////////////////////
void WizardFile::CloseFile(void)
{
	if(this->fIsOpen)
	{
		#ifdef IBM
			FreeLibrary(this->fInstDLL);
			this->fInstDLL = 0;
		#else
			CloseResFile(this->fResRefNum);
			this->fResRefNum = 0;
		#endif
	}
	this->fIsOpen = false;
	if(this->fText10000Hdl) DisposeHandle((Handle)this->fText10000Hdl);
	this->fText10000Hdl = 0;
}
/////////////////////////////////////////////////
CHARH WizardFile::GetResource(OSType type,short resNum,Boolean terminate)
{
	return this->GetResourceHelper(type,resNum,false,terminate);
}
/////////////////////////////////////////////////
Boolean WizardFile::ResourceExists(OSType type,short resNum)
{
	long size; 
	return this->ResourceExists(type,resNum,&size);
}
/////////////////////////////////////////////////
Boolean WizardFile::ResourceExists(OSType type,short resNum,long* size)
{
	long resSize = (long)(this->GetResourceHelper(type,resNum,true,false));
	if(resSize < 0) 
	{
		*size = 0;
		return false;
	}
	*size = resSize;
	return true;
}
/////////////////////////////////////////////////
CHARH WizardFile::GetResourceHelper(OSType type,short resNum,Boolean returnResSize,Boolean terminate)
{	
	// If returnResSize is true, this function returns the resource size
	// it returns -1 if the resource does not exist
	//
	// NOTE: if returnResSize is false then this function
	// returns a *COPY* of the resource handle
	// not the resource handle itself
	//////////////////////////////
	CHARH h = 0;
	OSErr err = 0;
	Boolean wasClosed = !this->fIsOpen;
	long  len = 0;
	long extraLen = 0;
	if(terminate)  extraLen = 1;

	if(!this->fIsOpen) 
	{
		err = this->OpenFile();
		if(err) return h;
	}

	#ifdef IBM
	{
		char numStr[32];
		char typeStr[5];
		long i;
		HRSRC hResInfo =0;
		HGLOBAL r = 0;
		sprintf(numStr,"#%ld",resNum);
		
		// copy the OSType into a string
		for(i = 0; i < 4;i++)
			typeStr[i] = ((char*)&type)[3 - i];
		typeStr[4] = 0;
		
		hResInfo = FindResource(this->fInstDLL,numStr,typeStr);
		if(hResInfo) len = SizeofResource(this->fInstDLL,hResInfo); // fixes windows 2000 bug 7/18/00

		if(returnResSize)
		{	// just asking for the file size
			if(!hResInfo) h =  (CHARH)(-1); // resource does not exist
			else h = (CHARH)(len+extraLen); // resource exists if len > 0
		}
		else
		{	// get the handle
			if(hResInfo) r = LoadResource(this->fInstDLL,hResInfo);
			if(r) 
			{
				h = (CHARH) _NewHandle(len + extraLen);
				if(h)
				{
					char* p = (char*) LockResource(r);
					_HLock((Handle)h);
					memcpy(*h, p, len);
					if(terminate) (*h)[len] = 0;
					// WIN32 applications do not have to unlock resources locked by LockResource
					// I don't think we free something gotten from a resource
					_HUnlock((Handle)h);
				}
			}
		}
	}
	#else
	{
		// MAC code
		Handle r = Get1Resource(type,resNum);
		if(r) len = _GetHandleSize((Handle)r);
		if(returnResSize)
		{	// just asking for the file size
			if(!r) h =  (CHARH)(-1); // resource does not exist
			else h = (CHARH)(len+extraLen); // resource exists if len > 0
			if(r) ReleaseResource(r);
		}
		else if(r)
		{	// get the handle
			h = (CHARH)_NewHandle(len + extraLen);
			if(h)
			{
				_HLock(r);
				_HLock(h);
				memcpy(*h, *r, len);
				if(terminate) (*h)[len] = 0;
				_HUnlock(r);
				_HUnlock(h);
			}
			ReleaseResource(r);
		}
	}
	#endif
	done:
	if(wasClosed) this->CloseFile(); // don't leave it open
	return h;
}

/////////////////////////////////////////////////
CHARH WizardFile::GetWMSG(short dialogResNum)
{	// we used to get this out of a resource, 
	// this function now uses TEXT 10000
	CHARH msgH = nil;
	long i,len;
	long lineStart = 0;
	
	if(this->fText10000Hdl)
	{	
		char tempLine[512]= "";
		Boolean bInRightWmsgSection = false;
		char lineToMatch[64];
		
		len = _GetHandleSize((Handle)this->fText10000Hdl);
		msgH = (CHARH) _NewHandle(len);
		if(!msgH) {WizardOutOfMemoryAlert(); return nil;}
		_HLock((Handle)msgH);
		**msgH = 0; // terminate the cString
		sprintf(lineToMatch,"[WMSG %d]",dialogResNum);
		
		// get the line corresponding to the dialog number 
		// from the text 10000 resource
		// look for keyword [ORDER] then count lines
		
		// since we are looping through the lines
		// use the more efficient GetLineFromHdlHelper
		//for(i = 1; true; i++)
		for(lineStart = 0; true;)
		{
			//Boolean gotLine= GetCleanedUpLineFromHdl(i,this->fText10000Hdl,tempLine,512);			
			Boolean gotLine = GetLineFromHdlHelper(1,lineStart,&lineStart,this->fText10000Hdl,tempLine,512,true);
			
			if(!gotLine) break; // out of for loop, end of resource
			if(!tempLine[0]) continue; // blank line
			if(tempLine[0] == '-' || tempLine[0] == '#' || tempLine[0] == '/') continue; // comment line
			if(tempLine[0] == '[')
			{
				//special line
				bInRightWmsgSection = false;
				if(!strcmpnocase(tempLine,lineToMatch) )
					bInRightWmsgSection = true;
				continue;// get next line
			}
			if(bInRightWmsgSection)
			{ // copy the line to the msgH
				strcat(*msgH,tempLine);
				strcat(*msgH,NEWLINESTRING);
			}
		}
	}
	if(msgH)
	{	
		_HUnlock((Handle)msgH);
		len = strlen(*msgH);
		if(len == 0)// no lines were found
		{
			DisposeHandle((Handle)msgH); 
			msgH = nil;
		}
		else
			_SetHandleSize((Handle)msgH,len+1);
	}
	
	return msgH;
}


/////////////////////////////////////////////////
long WizardFile::DialogResNum(long dialogNum,long*specialFlag)
{	// NOTE: dialogNum is 1 relative
	long lineStart,orderLineNum = 0,resNum=0;
	if(specialFlag) *specialFlag = 0;
	if(this->fText10000Hdl)
	{	// new style 
		// use 'ORDR' resource
		char line[64]="",tempLine[256]= "";
		char paramStr[32];
		char* ptr;
		Boolean gotLine,bInOrderSection = false;
		
		// get the line corresponding to the dialog number 
		// from the text 10000 resource
		// look for keyword [ORDER] then count lines
		
		// since we are looping through the lines
		// use the more efficient GetLineFromHdlHelper
		//for(i = 1; true; i++)
		for(lineStart = 0; line[0] == 0;)
		{
			//gotLine = GetCleanedUpLineFromHdl(i,this->fText10000Hdl,tempLine,256);
			gotLine = GetLineFromHdlHelper(1,lineStart,&lineStart,this->fText10000Hdl,tempLine,256,true);
			
			if(!gotLine) break; // out of for loop, end of resource
			if(!tempLine[0]) continue; // blank line
			if(tempLine[0] == '-' || tempLine[0] == '#' || tempLine[0] == '/') continue; // comment line
			switch(tempLine[0])
			{
				case '[':
					// special line
					if(!strncmpnocase("[ORDER]", tempLine,7)) bInOrderSection = true;
					else bInOrderSection = false;
					break;
				case '1': //start of number
				case '2': //start of number
				case '3': //start of number
				case '4': //start of number
				case '5': //start of number
				case '6': //start of number
				case '7': //start of number
				case '8': //start of number
				case '9': //start of number
				case 'w': case 'W': // Wind
				//case 't': case 'T': // WndType
				case 'm': case 'M': // Model
				case 'a': case 'A': // AlmostDone
					if(bInOrderSection) 
					{
						orderLineNum++;
						if(orderLineNum == dialogNum) 
						{
							strncpy(line,tempLine,64);
							line[63] = 0;
						}
					}
					break;
			}
		}
		
		////////////////////////////////////////////////
		
		if(line[0])
		{	// check for keywords
		
			// check for WIND
			// NOTE: WIND need not provide a resource number since the wizard message is known
			if(!strcmpnocase(line,"WIND") || strstr(line,"WIND"))  // backwardly compatible to WIND <resNum>
			{
				if(specialFlag) *specialFlag = WINDFLAG;
				strcpy(line,paramStr);
			}
			else if(!strcmpnocase(line,"WNDTYPE"))
			{	// check for WNDTYPE
				// NOTE: WNDTYPE need not provide a resource number since the wizard message is known
				if(specialFlag) *specialFlag = WNDTYPEFLAG;
				strcpy(line,"1665");// the resource number for wind type is 1665
			}
			else if(!strcmpnocase(line,"10010"))
			{	// check for WIND TYPE
				// NOTE: WIND TYPE resource number must be 10010 or else require WNDTYPE flag
#ifdef MAC /*TARGET_API_MAC_CARBON*/
				if(specialFlag) *specialFlag = WNDTYPEFLAG;
				strcpy(line,"1665");// the resource number for wind type is 1665
				//if (this->fLocationFileVersion == 0) printNote();
#endif
			}
			//else if(!strcmpnocase(line,"10002"))
			else if(!strcmpnocase(line,"WELCOME"))
			{	// check for WELCOME
				
#ifdef MAC 
				// put in a check so old location files still use built in resources
				if(specialFlag) *specialFlag = WELCOMEFLAG;
				strcpy(line,"9920");// the resource number for welcome dialog is 9920
				//if (this->fLocationFileVersion == 0) printNote();
#endif
			}
			else if(!strcmpnocase(line,"MODEL"))
			{	// check for MODEL
				// NOTE: MODEL need not provide a resource number since the wizard message is known
				if(specialFlag) *specialFlag = MODELFLAG;
				strcpy(line,"0");// we'll set the resource number to 0
			}
			else if(!strcmpnocase(line,"ALMOSTDONE"))
			{	// check for ALMOSTDONE
				if(specialFlag) *specialFlag = ALMOSTDONEFLAG;
				strcpy(line,"9990"); // the resource number for almost done is fixed as 9990
			}
			// line should contain the resource number
			stringtonum(line,&resNum);
			if(resNum < 0 )  resNum = 0;
		}
	}

	return resNum;
}
/////////////////////////////////////////////////
Boolean WizardFile::DialogExists(long dialogNum)
{
	long specialFlag = 0;
	long resNum = this->DialogResNum(dialogNum,&specialFlag);
	// special cases
	//if(specialFlag == WINDFLAG || specialFlag == WNDTYPEFLAG || specialFlag == MODELFLAG || specialFlag == ALMOSTDONEFLAG) return true;
	if(specialFlag == WELCOMEFLAG || specialFlag == WINDFLAG || specialFlag == WNDTYPEFLAG || specialFlag == MODELFLAG || specialFlag == ALMOSTDONEFLAG) return true;
	#ifdef IBM
		char numStr[32];
		HRSRC hResInfo =0;
		sprintf(numStr,"#%ld",resNum);
		hResInfo = FindResource(this->fInstDLL,numStr,RT_DIALOG);
		if(hResInfo) return true;
	#else 
		return this->ResourceExists('DLOG',resNum);
	#endif
	return false;
}
/////////////////////////////////////////////////
long WizardFile::NumWizardDialogs(void)
{
	long numDialogs = 0;
	while(this->DialogExists(numDialogs+1)) 
		numDialogs++;
	return numDialogs;
}

Boolean WizardFile::ModelDialogIncludedInFile(void)
{
	long i;
	for(i = 1; true; i++)	
	{
		long resNum,specialFlag = 0;
		if(!this->DialogExists(i)) return false;
		resNum = this->DialogResNum(i,&specialFlag);
		if(specialFlag == MODELFLAG) return true;
	}
	return false;
}



#ifdef MAC
void WizardFile::DrawingRoutine(DialogPtr theDialog, short itemNum)  
{
	#pragma unused(itemNum)
	short       	itemtype;
	Handle     		handle;
	Rect				rect;
	GrafPtr			savePort;
	
	GetPortGrafPtr(&savePort);
	SetPortDialogPort(theDialog);
	PenNormal();
	// outline the default button
	////////////////////////
	GetDialogItem(theDialog,1,&itemtype,&handle,&rect);
	PenSize(3,3);
	InsetRect(&rect,-4,-4);
	FrameRoundRect(&rect,16,16);
	PenNormal();
	SetPortGrafPort(savePort);
}
#endif



void WizardFile::GetPhraseFromLine(long phraseNum1Relative,char* line, char* answerStr,long maxNumChars)
{	// returns the part between the ';' chars
	 WizardGetPhraseFromLine(phraseNum1Relative,line,answerStr,maxNumChars);
}

void WizardFile::GetParameterString(char* key,char* line, char* answerStr,long maxNumChars)
{
	WizardGetParameterString(key,line,answerStr,maxNumChars);
}

void WizardFile::GetParameterString(char* key,long lineNum1Relative,char* str, char* answerStr,long maxNumChars)
{
	WizardGetParameterFromStringLine(key,lineNum1Relative,str,answerStr,maxNumChars);
}

void WizardFile::GetParameterString(char* key,long lineNum1Relative,CHARH paramListHdl, char* answerStr,long maxNumChars)
{
	WizardGetParameterFromHdlLine(key,lineNum1Relative,paramListHdl,answerStr,maxNumChars);
}


OSErr SubstituteString(char* str,long maxStrLen, long oldPartStart,long oldPartLen, char*substitutionString)
{
	OSErr err = 0;
	long oldStrLen,newStrLen,newPartLen,i;

	oldStrLen = strlen(str);
	newPartLen = strlen(substitutionString);
	
	if(oldPartLen == newPartLen)
	{
		for(i = 0; i< newPartLen; i++)
		{
			str[oldPartStart+i] = substitutionString[i];
		}
	}
	else if(oldPartLen < newPartLen)
	{
		// the string will be longer, so check maxStrLen
		long addedLength = newPartLen - oldPartLen; 
		long newStartOfTextAfterSubstitution = oldPartStart+newPartLen;
		
		newStrLen = oldStrLen+addedLength;
		if(newStrLen >= maxStrLen)
		{
			printError("maxStrLen exceeded in SubstituteString");
			return -1;
		}

		// move over the rest of the characters
		// to the right
		for(i = newStrLen-1; i >= newStartOfTextAfterSubstitution; i--)
		{
			str[i] = str[i-addedLength];
		}
		// copy in the new chars
		for(i = 0; i< newPartLen; i++)
		{
			str[oldPartStart+i] = substitutionString[i];
		}
		// terminate the string
		str[newStrLen] = 0;
	
	}
	else //oldLen > newLen
	{
		long subtractedLength = oldPartLen - newPartLen; 
		long newStartOfTextAfterSubstitution = oldPartStart+newPartLen;
		newStrLen = oldStrLen-subtractedLength;
		// copy in the new chars
		for(i = 0; i< newPartLen; i++)
		{
			str[oldPartStart+i] = substitutionString[i];
		}
		// now move over the rest of the characters
		// to the left
		for(i = newStartOfTextAfterSubstitution;i < newStrLen; i++)
		{
			str[i] = str[i+subtractedLength];
		}
		// cut off the string
		str[newStrLen] = 0;

	}
	
	return noErr;

}

OSErr SubstituteStringInHdl(CHARH msgH, long oldPartStart,long oldLen, char*substitutionString)
{
	OSErr err = 0;
	long hdlLen,newHdlLen,newLen,i;

	if(!msgH) return -1;
	hdlLen = _GetHandleSize((Handle)msgH);
	newLen = strlen(substitutionString);
	
	if(oldLen == newLen)
	{
		for(i = 0; i< oldLen; i++)
		{
			(*msgH)[oldPartStart+i] = substitutionString[i];
		}
	}
	else if(oldLen < newLen)
	{
		// grow the handle
		long addedLength = newLen - oldLen; 
		long oldHdlLen = hdlLen;
		long newStartOfNonVariableText = oldPartStart+newLen;
		newHdlLen = hdlLen+addedLength;
		_SetHandleSize((Handle)msgH,newHdlLen);
		err = _MemError();
		if(err) { WizardOutOfMemoryAlert(); return err;}
		// move over the rest of the characters
		// to the right
		for(i = newHdlLen-1; i >= newStartOfNonVariableText; i--)
		{
			(*msgH)[i] = (*msgH)[i-addedLength];
		}
		// copy in the new chars
		for(i = 0; i< newLen; i++)
		{
			(*msgH)[oldPartStart+i] = substitutionString[i];
		}
	
	}
	else //oldLen > newLen
	{
		long subtractedLength = oldLen - newLen; 
		long newStartOfNonVariableText = oldPartStart+newLen;
		newHdlLen = hdlLen-subtractedLength;
		// copy in the new chars
		for(i = 0; i< newLen; i++)
		{
			(*msgH)[oldPartStart+i] = substitutionString[i];
		}
		// now move over the rest of the characters
		// to the left
		for(i = newStartOfNonVariableText;i < newHdlLen; i++)
		{
			(*msgH)[i] = (*msgH)[i+subtractedLength];
		}
		// shrink handle
		_SetHandleSize((Handle)msgH,newHdlLen);
	}
	
	return err;

}

OSErr WizardFile::EvaluateBasicBlock(char* str,long maxStrLen,EvaluationType evaluationType)
{	// this is a strong evaluation
	// if the block is not recognized, it is viewed as an error
	double z= 0;
	double val,val2;
	long numScanned,len,i;
	OSErr err = noErr;
	// a STRICT evaluation is one that requires the block to be recognized
	// and evaluated.  Numbers evaluate to themselves.  
	// Evaluating white-space returns an error under STRICT evaluation.
	// Evaluating text which is not a variable is an error under STRICT evaluation.
	Boolean strict = (evaluationType & STRICT_EVALUATION); 
	
	Boolean hasInternalWhiteSpace = false;
	Boolean hasMultiplication = false;
	Boolean hasAddition = false;
	Boolean hasDivision = false;
	Boolean startsWithMinusSign = false;
	Boolean hasMinusSign = false;
	Boolean hasExponent = false;
	Boolean hasMathEqualityOrInequalityOperator = false;


	RemoveLeadingAndTrailingWhiteSpace(str);
	len = strlen(str); 
	if(len == 0) 
	{	// it collapsed to nothing
		if(strict) 
			{ ParsingErrorAlert("Found only white-space in a strict evaluation",0); return -1;}
		return noErr; 
	}
	
	// in order to avoid a lot of sscanf calls
	// determine if it has white space, math operation chars etc
	
	for(i = 0; i<len;i++)
	{
		switch(str[i])
		{
			// white space
			case ' ': case '\t': 
				hasInternalWhiteSpace = true; break;
			
			// math binary operators 
			case '*':  
				hasMultiplication = true; break;
			case '+':  
				hasAddition = true; break;
			case '/': 
				hasDivision = true; break;
				
			//the minus sign can be a unary or a binary operators
			//and is difficult to out guess .  
			// remember we can have things like "IsPositive -200"
			// I think the only thing we can be sure of are those starting with a minus sign
			case '-':
				if(i != 0) hasMinusSign = true;
				if(i == 0)  
					startsWithMinusSign = true;
				break;

			case '^': 
				hasExponent = true; break;

			// equality, inequality
			case '<': case '>': 
			case '=': // first part of ==
			case '!': // first part of !=
				hasMathEqualityOrInequalityOperator = true; break;
				
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////
	
	// This should be taken care of at the end... 5/11/00 
	/*if(startsWithMinusSign && !hasMinusSign && !hasAddition && !hasMultiplication && !hasInternalWhiteSpace && !hasDivision && !hasExponent && !hasMathEqualityOrInequalityOperator)/////////////////////////////////////////////////
	{// it must be a simple negative number  // Must also consider cases -a+b, -a-b,...  (may not just be -a)
		// numbers evaluate to themselves
		err = StringToDouble(str,&val);
		if(!err) return noErr;
		goto unrecognized;
	}*/
	//// 
	
	if(hasMultiplication)/////////////////////////////////////////////////
	{	// watch out for white space
		numScanned = sscanf(str,lfFix("%lf*%lf"),&val,&val2);
		if(numScanned == 2) { z = val*val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf *%lf"),&val,&val2);
		if(numScanned == 2) { z = val*val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf* %lf"),&val,&val2);
		if(numScanned == 2) { z = val*val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf * %lf"),&val,&val2);
		if(numScanned == 2) { z = val*val2; goto evalZ;}
		goto unrecognized;
	}
	
	if(hasAddition)/////////////////////////////////////////////////
	{	// watch out for white space
		numScanned = sscanf(str,lfFix("%lf+%lf"),&val,&val2);
		if(numScanned == 2) { z = val+val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf +%lf"),&val,&val2);
		if(numScanned == 2) { z = val+val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf+ %lf"),&val,&val2);
		if(numScanned == 2) { z = val+val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf + %lf"),&val,&val2);
		if(numScanned == 2) { z = val+val2; goto evalZ;}
		goto unrecognized;
	}
	
	if(hasDivision)/////////////////////////////////////////////////
	{	// watch out for white space
		numScanned = sscanf(str,lfFix("%lf/%lf"),&val,&val2);
		if(numScanned == 2) { z = val/val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf /%lf"),&val,&val2);
		if(numScanned == 2) { z = val/val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf/ %lf"),&val,&val2);
		if(numScanned == 2) { z = val/val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf / %lf"),&val,&val2);
		if(numScanned == 2) { z = val/val2; goto evalZ;}
		goto unrecognized;
	}
	
	if(hasExponent)/////////////////////////////////////////////////
	{	// watch out for white space
		numScanned = sscanf(str,lfFix("%lf^%lf"),&val,&val2);
		if(numScanned == 2) { z = pow(val,val2); goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf ^%lf"),&val,&val2);
		if(numScanned == 2) { z = pow(val,val2); goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf^ %lf"),&val,&val2);
		if(numScanned == 2) { z = pow(val,val2); goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf / %lf"),&val,&val2);
		if(numScanned == 2) { z = pow(val,val2); goto evalZ;}
		goto unrecognized;
	}
	
	if(hasMathEqualityOrInequalityOperator)/////////////////////////////////////////////////
	{	// watch out for white space  { z =  (val > 0) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf>%lf"),&val,&val2);
		if(numScanned == 2) { z = (val>val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf >%lf"),&val,&val2);
		if(numScanned == 2) { z = (val>val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf> %lf"),&val,&val2);
		if(numScanned == 2) { z = (val>val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf > %lf"),&val,&val2);
		if(numScanned == 2) { z = (val>val2) ? 1:0; goto evalZ;}
		
		numScanned = sscanf(str,lfFix("%lf>=%lf"),&val,&val2);
		if(numScanned == 2) { z = (val>=val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf >=%lf"),&val,&val2);
		if(numScanned == 2) { z = (val>=val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf>= %lf"),&val,&val2);
		if(numScanned == 2) { z = (val>=val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf >= %lf"),&val,&val2);
		if(numScanned == 2) { z = (val>=val2) ? 1:0; goto evalZ;}
		
		numScanned = sscanf(str,lfFix("%lf<%lf"),&val,&val2);
		if(numScanned == 2) { z = (val<val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf <%lf"),&val,&val2);
		if(numScanned == 2) { z = (val<val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf< %lf"),&val,&val2);
		if(numScanned == 2) { z = (val<val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf < %lf"),&val,&val2);
		if(numScanned == 2) { z = (val<val2) ? 1:0; goto evalZ;}
		
		numScanned = sscanf(str,lfFix("%lf<=%lf"),&val,&val2);
		if(numScanned == 2) { z = (val<=val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf <=%lf"),&val,&val2);
		if(numScanned == 2) { z = (val<=val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf<= %lf"),&val,&val2);
		if(numScanned == 2) { z = (val<=val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf <= %lf"),&val,&val2);
		if(numScanned == 2) { z = (val<=val2) ? 1:0; goto evalZ;}
		
		numScanned = sscanf(str,lfFix("%lf==%lf"),&val,&val2);
		if(numScanned == 2) { z = (val==val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf ==%lf"),&val,&val2);
		if(numScanned == 2) { z = (val==val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf== %lf"),&val,&val2);
		if(numScanned == 2) { z = (val==val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf == %lf"),&val,&val2);
		if(numScanned == 2) { z = (val==val2) ? 1:0; goto evalZ;}
		
		numScanned = sscanf(str,lfFix("%lf!=%lf"),&val,&val2);
		if(numScanned == 2) { z = (val!=val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf !=%lf"),&val,&val2);
		if(numScanned == 2) { z = (val!=val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf!= %lf"),&val,&val2);
		if(numScanned == 2) { z = (val!=val2) ? 1:0; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf != %lf"),&val,&val2);
		if(numScanned == 2) { z = (val!=val2) ? 1:0; goto evalZ;}
		
		goto unrecognized;
	}
	
	// check subtraction 
	if(hasMinusSign)
	{
		numScanned = sscanf(str,lfFix("%lf-%lf"),&val,&val2);
		if(numScanned == 2) { z = val-val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf -%lf"),&val,&val2);
		if(numScanned == 2) { z = val-val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf- %lf"),&val,&val2);
		if(numScanned == 2) { z = val-val2; goto evalZ;}
		numScanned = sscanf(str,lfFix("%lf - %lf"),&val,&val2);
		if(numScanned == 2) { z = val-val2; goto evalZ;}
		// note: it could be a negative sign in one of our built in variables
		// so we can't "goto unrecognized;"   
	}

	
	if(hasInternalWhiteSpace) /////////////////////////////////////////////////
	{	// we eliminated the possibility of the math operations above
		// so it must be one of our built in functions
		// or something involving subtraction
		char firstWord[32],param1[32] = "",param2[32] ="";
		long lenOfFirstWord,lenOfSecondWord,lenOfThirdWord;
		char *firstWordStart = StartOfFirstWord(str,&lenOfFirstWord);
		char *secondWordStart,*thirdWordStart;
		Boolean numParam = 0;
		// get the first word, or at least the first part of it

		//
		if(lenOfFirstWord< 32)
		{ // then it could be one of our functions
			
			strnzcpy(firstWord,firstWordStart,lenOfFirstWord);
			
			// get the second word, that would be our 1st parameter
			secondWordStart = StartOfFirstWord(firstWordStart+lenOfFirstWord,&lenOfSecondWord);
			if(0 < lenOfSecondWord && lenOfSecondWord < 32)
			{
				strnzcpy(param1,secondWordStart,lenOfSecondWord);
				// get the third word, that would be our 2nd parameter
				thirdWordStart = StartOfFirstWord(secondWordStart+lenOfSecondWord,&lenOfThirdWord);
				if(0 < lenOfThirdWord && lenOfThirdWord < 32)
					strnzcpy(param2,thirdWordStart,lenOfThirdWord);
			}
			
			// the params must be numbers for any of our functions
			if(param1[0])
			{
				err = StringToDouble(secondWordStart,&val);
				if(!err)
				{
					numParam++;
					if(param2[0])
					{ 
						//err = StringToDouble(secondWordStart,&val2); 
						err = StringToDouble(thirdWordStart,&val2); //5/10/00 
						if(!err) numParam++;
					}
				}
			}
			
			if(!err && numParam == 1)
			{ 

				switch(firstWord[0])
				{
					case 'A': case 'a':
						if(!strcmpnocase(firstWord,"abs")) { z = abs(val); goto evalZ;}
						if(!strcmpnocase(firstWord,"acos")) { z = acos(val); goto evalZ;}
						if(!strcmpnocase(firstWord,"asin")) { z = asin(val); goto evalZ;}
						if(!strcmpnocase(firstWord,"atan")) { z = atan(val); goto evalZ;}
						break;
						
					case 'C': case 'c':
						if(!strcmpnocase(firstWord,"ceil")) { z = ceil(val); goto evalZ;}
						if(!strcmpnocase(firstWord,"cos")) { z = cos(val); goto evalZ;}
						if(!strcmpnocase(firstWord,"cosh")) { z = cosh(val); goto evalZ;}
						break;
						
					case 'E': case 'e':
						if(!strcmpnocase(firstWord,"exp")) { z = exp(val); goto evalZ;}
						break;
						
					case 'F': case 'f':
						if(!strcmpnocase(firstWord,"fabs")) { z = fabs(val); goto evalZ;}
						if(!strcmpnocase(firstWord,"floor")) { z = floor(val); goto evalZ;}
						break;
						
					case 'I': case 'i': // these probably won't get used much anymore, now that we have logical operators
						if(!strcmpnocase(firstWord,"IsPositive")) 	{ z = (val > 0) ? 1:0; goto evalZ;}
						if(!strcmpnocase(firstWord,"IsNegative")) 	{ z = (val < 0) ? 1:0; goto evalZ;}
						if(!strcmpnocase(firstWord,"IsNonPositive")) { z = (val <= 0) ? 1:0; goto evalZ;}
						if(!strcmpnocase(firstWord,"IsNonNegative")) 	{ z = (val >= 0) ? 1:0; goto evalZ;}
						if(!strcmpnocase(firstWord,"IsZero")) 	{ z = (val == 0) ? 1:0; goto evalZ;}
						if(!strcmpnocase(firstWord,"IsNonZero")) 	{ z = (val != 0) ? 1:0; goto evalZ;}
						break;

					case 'L': case 'l':
						if(!strcmpnocase(firstWord,"log10")) { z = log10(val); goto evalZ;}
						if(!strcmpnocase(firstWord,"log")) { z = log(val); goto evalZ;}
						break;

					case 'R': case 'r':
						if(!strcmpnocase(firstWord,"ROUND")) { z = round(val); goto evalZ;}
						break;

					case 'S': case 's':
						if(!strcmpnocase(firstWord,"sinh")) { z = sinh(val); goto evalZ;}
						if(!strcmpnocase(firstWord,"sin")) { z = sin(val); goto evalZ;}
						if(!strcmpnocase(firstWord,"sqrt")) { z = sqrt(val); goto evalZ;}
						break;

					case 'T': case 't':
						if(!strcmpnocase(firstWord,"tanh")) { z = tanh(val); goto evalZ;}
						if(!strcmpnocase(firstWord,"tan")) { z = tan(val); goto evalZ;}
						break;
				}
			}
			else if(numParam == 2)
			{
				switch(firstWord[0])
				{
					case 'F': case 'f':
						if(!strcmpnocase(firstWord,"fmod")) { z = fmod(val,val2); goto evalZ;}
						break;
							
					//////////
					//case 'R': case 'r':
					//	if(!strcmpnocase(firstWord,"ROUND")) 
					//	{ 
					////////////////
					// an extension of round where the second param is the number of decimal places
					////////////////////
					//		long n = round(val2); // must be an integer by def
					//		if(0 <= n && n <= 9)
					//		{	//sprintf rounds
					//			sprintf()
					//		}
					//		z = round(val); goto evalZ;
					//	}
					//	break;
					////////////////////

					case 'P': case 'p':	// redundant now that we have ^ operator, 5/11/00
						if(!strcmpnocase(firstWord,"pow")) { z = pow(val,val2); goto evalZ;}
						break;
				}
			}
		}
		
		// if we get here
		// it has an internal space 
		// but is not one of our functions followed by a number
		// nothing else to try
		goto unrecognized;
	}
	else
	{	/////////////////////////////////////////////////
		// it must be a single word
		/////////////////////////////////////////////////
		//////////
		// built-in variables
		//////////
		if(	
			#ifdef MAC
					!strcmpnocase(str,"MAC")
			#else
					!strcmpnocase(str,"IBM")
			#endif
				|| !strcmpnocase(str,"TRUE")
				|| !strcmpnocase(str,"YES")
			)
		{
			strcpy(str,"1"); 
			return noErr;
		}
		/////
	
		if(	
			#ifdef IBM
					!strcmpnocase(str,"MAC")
			#else
					!strcmpnocase(str,"IBM")
			#endif
				|| !strcmpnocase(str,"FALSE")
				|| !strcmpnocase(str,"NO")
			)
		{
			strcpy(str,"0"); 
			return noErr;
		}
		
		/////
		//////////
		// user defined command variables
		/////////
		for(i = 0; i < MAX_NUM_WIZ_VARIABLES; i++) {
			if(!this->fCommandVariable[i].name[0]) break; // a blank variable, i.e. no more variables 
			if(!strcmpnocase(str,this->fCommandVariable[i].name)) 
			{ // it matches, return the value
				len = strlen(this->fCommandVariable[i].value);
				if(len >= maxStrLen) goto maxStrLenErr; // this is not going to happen, but we check anyway
				strcpy(str,this->fCommandVariable[i].value);
				return noErr;
			}
		}
		
		// last chance, it must be a number
		// Note: numbers evaluate to themselves
		err = StringToDouble(str,&val);
		if(!err) return noErr; 
		
		goto unrecognized;
		
	}
	
	
unrecognized:
	// if we get here, it was not a recognizable basic block
	if(strict){
 		ParsingErrorAlert("Unable to evaluate expression.",str); 
 		return -1;
	}
	return noErr;
	
evalZ:
	char tempStr[64];
	sprintf(tempStr,lfFix("%lf"),z); // avoid scientific notation 5/10/00
	ChopEndZeros(tempStr);
	len = strlen(tempStr);
	if(len >= maxStrLen) goto maxStrLenErr; // this is not going to happen, but we check anyway
	strcpy(str,tempStr);
	return noErr; // evaluated
	
maxStrLenErr:
		ParsingErrorAlert("Programmer err: EvaluateBasicBlock exceeded maxStrLen",str);
		return -1;
}



OSErr WizardFile::EvaluateBooleanString(char* evaluationString,long maxStrLen,Boolean * bVal)
{	// this is a strong evaluation
	// if the final phrase is not recognized, it is viewed as an error
	OSErr err = 0;
	*bVal = 0;
	
	err = this->EvaluateAndTrimString(evaluationString,maxStrLen,STRICT_EVALUATION | FORCE_EVALUATION);
	if(err) return err;

	// check to see if it has been reduced to a Boolean value
	// it should be either a number or "true" or "false"
	
	if(!strcmpnocase(evaluationString,"true")) 
		*bVal = true;
	else if(!strcmpnocase(evaluationString,"false"))
		*bVal = false;
	else if(!strcmpnocase(evaluationString,"yes")) 
		*bVal = true;
	else if(!strcmpnocase(evaluationString,"no"))
		*bVal = false;
	else
	{ 
		// make sure it is a number or report an error to the user
		double val;
		err = StringToDouble(evaluationString,&val);
		if(err) 
		{
			ParsingErrorAlert("Unable to recognize this as a Boolean expression",evaluationString);
			return err;
		}
		*bVal = (val != 0); // do we have to worry about the number being really small but not zero??
	}
	
	return err;
}


OSErr WizardFile::EvaluateNumberString(char* evaluationString,long maxStrLen,double * dVal)
{	// this is a strong evaluation
	// if the final phrase is not recognized, it is viewed as an error
	OSErr err = 0;
	double localVal;
	*dVal = 0;
	
	err = this->EvaluateAndTrimString(evaluationString,maxStrLen,STRICT_EVALUATION | FORCE_EVALUATION);
	if(err) return err;

	// check to see if it has been reduced to a valid number
		
	// or report an error to the user
	err = StringToDouble(evaluationString,&localVal);
	if(err) 
	{
		ParsingErrorAlert("Unable to recognize this as a number expression",evaluationString);
		return err;
	}
	*dVal = localVal;
	
	return err;
}

/////////////////////////////////////////////////

OSErr WizardFile::EvaluateAndTrimString(char* str,long maxStrLen,EvaluationType evaluationType)
{	// evaluates all expressions in the string delimited by { } or << >>
	OSErr err = 0;
	Boolean foundStartOfExpression;
	Boolean foundEndOfExpression;
	long substitutionStart,oldStrLen,subLen,i,j;
	char basicBlockString[VALSTRMAX];
	long delimiterType;
	Boolean oneMoreEvaluationNeeded =false;
	Boolean mustBeEvaluatedIfNotSurrounded = (evaluationType & FORCE_EVALUATION);

	enum { kNewStyle, kOldStyle};
	
	if(mustBeEvaluatedIfNotSurrounded)
	{ 
		// i.e. check to see if the first non-white space char is the start of an evaluation
		long lenOfFirstWord;
		char* firstWordStart = StartOfFirstWord(str,&lenOfFirstWord);
		if(lenOfFirstWord== 0) {
			ParsingErrorAlert("Nothing but white-space in a forced evaluation",0);
			return -1;
		}
		if(firstWordStart[0] == '{') 
			oneMoreEvaluationNeeded = false; 
		else if(firstWordStart[0] == '<' && firstWordStart[1] == '<') // note this barely is ok when i == oldStrLen -1;
			oneMoreEvaluationNeeded = false; 
		else 
			oneMoreEvaluationNeeded = true; // not surrounded by an evaluation
	}
	
	for(;;) // forever
	{
		basicBlockString[0] = 0;
		subLen = 0;
		foundStartOfExpression =false;
		foundEndOfExpression = false;
		oldStrLen = strlen(str); // note we do need to do this every time !!
			
		// look for the beginning of a basic block (expression)
		///////////////////
		for(i = 0;i<oldStrLen;i++) 
		{
			
			if(str[i] == '{') 
			{ // new style delimiter
				FoundNewStyle: //Label
				delimiterType = kNewStyle;
				substitutionStart = i;
				foundStartOfExpression = true;
				i++;// move past the char
				break;
			}
			if(str[i] == '<' && str[i+1] == '<') // note this barely is ok when i == oldStrLen -1;
			{
				FoundOldStyle: //Label
				delimiterType = kOldStyle;
				substitutionStart = i;
				foundStartOfExpression = true;
				i++;i++;// move past the two chars
				break;
			}
		} ////end for i loop
		///////////////////
		
		////////////////////////
		if(!foundStartOfExpression) 
		{	// there are no more starting delimiters
			// this is the normal exit out of our loop
			RemoveLeadingAndTrailingWhiteSpace(str);
			if(oneMoreEvaluationNeeded)
			{ // we require that what is left is a basic block
				err = this->EvaluateBasicBlock(str,maxStrLen,evaluationType);
				return err;
			}
			return noErr;
		}
		/////////////
		
		// we found the start of an expression
		// copy the next chars to basicBlockString
		//////////////////////////
		for(j = 0; j < VALSTRMAX-1 &&  i<oldStrLen;i++) 
		{
			if(str[i] == '{')  goto FoundNewStyle; // another new style delimiter
			if(str[i] == '<' && str[i+1] == '<') goto FoundOldStyle; // another old style delimiter
			
			if(str[i] == '}' ) 
			{
				if(delimiterType == kOldStyle){
					ParsingErrorAlert("Miss-matched  << }",str);
					return -1;
				}
				foundEndOfExpression = true;
				subLen = i - substitutionStart + 1;
				break; // out of for loop on j
			}
			if(str[i] == '>' && str[i+1] == '>')
			{ 
				if(delimiterType == kNewStyle){
					ParsingErrorAlert("Miss-matched  { >>",str);
					return -1;
				}
				foundEndOfExpression = true;
				subLen = (i+1) - substitutionStart + 1;
				break; // out of for loop on j
			}
			
			// continue to copy
			basicBlockString[j++] = str[i];
			basicBlockString[j] = 0;
			
		} // end of copy loop on j
		////////////////////////
			
		if(!foundEndOfExpression){
			ParsingErrorAlert("End of expression not found in EvaluateAndTrimString()",str);
			return -1;
		}

		err = this->EvaluateBasicBlock(basicBlockString,VALSTRMAX,STRICT_EVALUATION);
		if(err) return err;
		err = SubstituteString(str,maxStrLen,substitutionStart,subLen,basicBlockString);
		if(err) return err;
	} 
	
	return err;
}

OSErr WizardFile::SubstituteDollarVariable(CHARH msgH,Boolean *hdlChanged,DialogPtr theDialog)
{
	OSErr err = 0;
	long hdlLen,newHdlLen,substitutionStart,newLen,oldLen,i,itemNum,valueNum;
	char substitutionString[VALSTRMAX];
	const short kReturnPopupItemNum = -333;
	
	*hdlChanged = false;
	if(!msgH) return -1;
	hdlLen = _GetHandleSize((Handle)msgH);
	// look through handle for '$' char
	for(i = 0;i<hdlLen;i++)
	{
		if((*msgH)[i] == '$') break;
	}
	if(i < hdlLen)
	{
		//we found a variable
		// which variable is it ??
		// find valueNum and itemNum
		char c;
		substitutionStart = i;
		if(i >= hdlLen) goto endHdlError;
		c = (*msgH)[i++]; // get the first '$' char
		oldLen = 1;
		
		c = (*msgH)[i++];// get next char
		if( ('A'<= c && c <= 'F' )
			|| (c == 'V' || c == 'v') //form $V2 etc, added 3/2/99 JLM
			)
		{	// of the form $D2
			oldLen++; // another char in the $ variable
			if(c == 'V' || c == 'v') valueNum = kReturnPopupItemNum; // special code used below
			else valueNum = c -'A' + 1 ; //'"A" is valueNum 1
			if(i >= hdlLen) goto endHdlError;
			c = (*msgH)[i++];// get next char
		}
		else if('0'<= c && c <= '9') valueNum = 1; // of form $2 etc
		else
		{ // bad $ variable
			char tempStr[64] = "$";
			short j;
			tempStr[1] = c;
			for(j = 2; j<50 && i <hdlLen;j++)
			{
				c = (*msgH)[i++];// get next char
				tempStr[j] = c; //copy to our string
			}
			tempStr[j] = 0;
			ParsingErrorAlert("Bad $ syntax found.",tempStr);
			return -1;
		}

		// now find the item number
		itemNum = 0;
		for(;;)
		{
			if('0'<= c && c <= '9')
			{
				oldLen++; // another char in the $ variable
				itemNum = itemNum*10 + (c-'0');
				if(i >= hdlLen) 
				{
					// if we have already gotten the item number
					// it is not an error to get to the end of the handle
					if(itemNum > 0 ) break;
					else goto endHdlError;
				}
				c = (*msgH)[i++];// get next char
				continue;
			}
			break; // no more numbers
		}
		

		if(1<= itemNum && itemNum < kMAXNUMWIZARDDIALOGITEMS)
		{ // look up the value;
			if(valueNum == kReturnPopupItemNum) 
			{	// JLM 3/2/99
				// special code to return the number of the item selected
				// check the type to see if there is an error, this is only for popups
				// we need to get the popup value for this puppy
				short itemSelected;
				switch(this->fItem[itemNum].type)
				{
					case WIZ_POPUP:
					case WIZ_UNITS:
					case WIZ_WINDPOPUP:
						itemSelected = this->GetPopupItemSelected(theDialog,itemNum);
						numtostring(itemSelected,substitutionString);
						break;
					
					default:
						ParsingErrorAlert("$V can only be used with popup items",0);
						return -1;
				}
			}
			else
				this->GetPhraseFromLine(valueNum,this->fItem[itemNum].valStr,substitutionString,VALSTRMAX);
		}
		
		*hdlChanged = true; // we will be attempting to change the handle 
		err = SubstituteStringInHdl(msgH,substitutionStart,oldLen,substitutionString);

	}
	
	return err;

	endHdlError:
		printError("End of Handle error in SubstituteDollarVariable()");
		return -1;
	
}




OSErr WizardFile::RetrieveValueStrings(DialogPtr theDialog)
{
	short i,k,numDialogItems = NumDialogItems(theDialog);
	OSErr err = 0,scanErr;
	char line[VALSTRMAX];

	for(i = 1;i<= numDialogItems && i < kMAXNUMWIZARDDIALOGITEMS;i++)
	{
		char valStr[VALSTRMAX] = "";
		char str[256];
		short editID = 0;
		
		if(this->fItem[i].hidden) continue; // don't retrieve hidden text
		
		if(	fItem[i].type == WIZ_POPUP
			|| fItem[i].type == WIZ_UNITS
			|| fItem[i].type == WIZ_WINDPOPUP
		)
		{ // 
			short numScanned = 0;
			long item[10]; // 10 items max
			long lineNumInResource;
			short itemSelected = this->GetPopupItemSelected(theDialog,i);

			// note item lines follow the first line so we add 1 to the itemSelected
			lineNumInResource = itemSelected+1;
			GetLineFromStr(lineNumInResource,this->fItem[i].str,line,VALSTRMAX);
			
			if(fItem[i].type == WIZ_POPUP || fItem[i].type == WIZ_WINDPOPUP)
			{ // read in value string(s)
				int valNum =2;
				this->GetParameterString("VALUE",line,valStr,VALSTRMAX);
				// strcat on VALUE1,VALUE2,.. VALUE9, etc
				if(valStr[0] == 0) 
				{ // we did not find the VALUE parameter
					// use VALUE1 is a synonym for VALUE
					valNum = 1;
				}
				else 
				{	// we found a VALUE parameter
					valNum = 2;// start looking at VALUEB
					
					if(fItem[i].type == WIZ_WINDPOPUP)
					{	// this type of item is special 
						// we use the VALUE parameter to set fUseVariableWind
						// so we know which kind of WIND dialog to put up
						if(!strcmpnocase(valStr,"VARIABLEWIND")) this->fUseVariableWind = true;
						if(!strcmpnocase(valStr,"CONSTANTWIND")) this->fUseVariableWind = false;
						else {}; // we leave it as the default
					}

				}
				for( ; valNum<7;valNum++) 
				{	
					char oldKey[32],newKey[32]; 
					strcpy(oldKey,"VALUE*");
					strcpy(newKey,"$*");
					//char *oldKey = "VALUE*";
					//char *newKey = "$*";
					char temp[32];
					oldKey[5] = 'A'-1+valNum; // change last char of key
					newKey[1] = 'A'-1+valNum; // change last char of key
					// try new style
					this->GetParameterString(newKey,line,temp,32);
					// if we didn't get it, then old style
					if(!temp[0]) this->GetParameterString(oldKey,line,temp,32);
					// if we didn't get it,then just call it zero as a place holder
					if(!temp[0]) strcpy(temp,"0"); // just a number place holder
					if(valNum > 1) strcat(valStr,";"); // use the phrase separator so we can use GetPhraseFromLine
					strcat(valStr,temp);
				}
				
			}
			else
			{ // WIZ_UNITS
				double scalar = 1.0;
				double value,userValue = 0;
				//short editID = this->fItem[i].editID;
				editID = this->fItem[i].editID;
				char scalarStr[32];
				
				if(this->fItem[editID].hidden)
				{  // error the edit text is hidden
					sprintf(str,"Wizard resource error. The edit text item %d referred to by UNITS item %d of dialog %d is hidden.",editID,i,this->fDialogResNum);
					printError(str);
					err = -1;
					goto done;
				}
				// check edit text is in range
				GetDlgItemText(theDialog,editID,str,256);
				
				
				// convert to number
				err =  StringToDouble(str,&userValue);
				if(err)  
				{ // error
					printError("The text you entered does not represent a number.");
					SafeSelectDialogItemText(theDialog, editID, 0, 255);
					goto done;
				}
				/////////////
				// check against minimum
				this->GetParameterString("USERMIN",line,scalarStr,32);
				if(scalarStr[0]) 
				{
					scanErr =  StringToDouble(scalarStr,&scalar);
					if(!scanErr) 
					{ 
						if(userValue < scalar)
						{
							sprintf(str,"You must enter a number greater than %s.",scalarStr);
							printError(str);
							SafeSelectDialogItemText(theDialog, editID, 0, 255);
							err = -1;
							goto done;
						}
					}
					else err = 0; // MIN is optional
				}
				//////////
				// check against max
				this->GetParameterString("USERMAX",line,scalarStr,32);
				if(scalarStr[0]) 
				{
					scanErr =  StringToDouble(scalarStr,&scalar);
					if(!scanErr) 
					{ 
						if(userValue > scalar)
						{
							sprintf(str,"You must enter a number less than %s.",scalarStr);
							printError(str);
							SafeSelectDialogItemText(theDialog, editID, 0, 255);
							err = -1;
							goto done;
						}
					}
				}
				////////////
				// get scalar
				this->GetParameterString("SCALAR",line,scalarStr,32);
				if(scalarStr[0]) 
				{
					scanErr =  StringToDouble(scalarStr,&scalar);
					if(scanErr) 
					{ // error
						scalar = 1.0;
					}
				}
				else
				{
					// no SCALAR string
					sprintf(str,"Wizard resource error. No SCALAR parameter found on line %d of item %d of dialog %d",lineNumInResource,i,this->fDialogResNum);
					printError(str);
					SafeSelectDialogItemText(theDialog, editID, 0, 255);
					err = -1;
					goto done;
				}
				
				value = userValue*scalar;
				StringWithoutTrailingZeros(valStr,value,9);
			}
		}
		strncpy(this->fItem[i].valStr,valStr,VALSTRMAX);
		this->fItem[i].valStr[VALSTRMAX-1] = 0;
		if (editID>0) strncpy(this->fItem[editID].valStr,valStr,VALSTRMAX);	// allow to set using the item # instead of the units popup item #
		if (editID>0) this->fItem[editID].valStr[VALSTRMAX-1] = 0;
	}
	
	done:
	
	return err;
}

/////////////////////////////////////////////////
CHARH WizardFile::RetrieveUserAnswer(DialogPtr theDialog)
{
	short i,k,numDialogItems = NumDialogItems(theDialog);
	OSErr err = 0;
	#define kMaxAnswerLength 1000
	long ansLen = 0,len,newLen;
	CHARH ansH = (CHARH)_NewHandle(kMaxAnswerLength);

	err = _MemError();
	if(err) goto done;
	
	(*ansH)[0] = 0;

	for(i = 1;i<= numDialogItems && i < kMAXNUMWIZARDDIALOGITEMS;i++)
	{
		char str[256] = "";
		
		if(this->fItem[i].hidden) continue; // don't retrieve hidden text
		
		switch(fItem[i].type)
		{
			case WIZ_WINDPOPUP:
			{
				long itemSelected = this->GetPopupItemSelected(theDialog,i);
				sprintf(str,"WIZ_WINDPOPUP %ld %ld;",i,itemSelected);
				break;
			}
			case WIZ_POPUP:
			{
				long itemSelected = this->GetPopupItemSelected(theDialog,i);
				sprintf(str,"WIZ_POPUP %ld %ld;",i,itemSelected);
				break;
			}
			case WIZ_UNITS:
			{
				long itemSelected = this->GetPopupItemSelected(theDialog,i);
				sprintf(str,"WIZ_UNITS %ld %ld;",i,itemSelected);
				break;
			}
			case WIZ_EDIT:
			{
				char userStr[64];
				GetDlgItemText(theDialog, i, userStr, 64);
				sprintf(str,"WIZ_EDIT %ld %s;",i,userStr);
				break;
			}
		}

		if(str[0])
		{
			strcat(str,NEWLINESTRING);
			len = strlen(str);
			newLen = ansLen + len;
			if(newLen < kMaxAnswerLength)	
			{
				strcat(*ansH,str);
				ansLen = newLen;
			}
			else
			{
				printError("kMaxAnswerLength exceeded in Wizard");
				err = true;
				goto done;
			}
		}
	} // end of for
	
	done:
	
	if(ansH)
	{	
		_HUnlock((Handle)ansH);
		len = strlen(*ansH);
		if(len == 0)// no lines were found
		{
			DisposeHandle((Handle)ansH); 
			ansH = nil;
		}
		else
			_SetHandleSize((Handle)ansH,len+1);
	}
	if(err)
	{
		if(ansH) DisposeHandle((Handle)ansH); ansH = 0;
		if(err == memFullErr) WizardOutOfMemoryAlert();
	}
	
	//if(ansH) printNote(*ansH); // debug
	
	return ansH;
}
/////////////////////////////////////////////////

long WizStringToMessageCode(char* messageTypeStr)
{
	long messageCode = 0;
	
	if(!messageTypeStr[0]) return 0;
	StrToUpper(messageTypeStr);
	
	if(!strcmp(messageTypeStr,"SETFIELD")) messageCode = M_SETFIELD;
	else if(!strcmp(messageTypeStr,"CREATEMOVER")) messageCode = M_CREATEMOVER;
	else if(!strcmp(messageTypeStr,"CREATEMAP")) messageCode = M_CREATEMAP;
	else if(!strcmp(messageTypeStr,"CREATESPILL")) messageCode = M_CREATESPILL;
	///
	// mesages to support TAP/Command files
	else if(!strcmp(messageTypeStr,"RUNSPILL")) messageCode = M_RUNSPILL;
	else if(!strcmp(messageTypeStr,"RUN")) messageCode = M_RUN;
	else if(!strcmp(messageTypeStr,"OPEN")) messageCode = M_OPEN;
	else if(!strcmp(messageTypeStr,"CLOSE")) messageCode = M_CLOSE;
	else if(!strcmp(messageTypeStr,"SAVE")) messageCode = M_SAVE;
	else if(!strcmp(messageTypeStr,"QUIT")) messageCode = M_QUIT;
	else if(!strcmp(messageTypeStr,"RESET")) messageCode = M_RESET;
	else if(!strcmp(messageTypeStr,"CLEARSPILLS")) messageCode = M_CLEARSPILLS;
	else if(!strcmp(messageTypeStr,"CLEARWINDS")) messageCode = M_CLEARWINDS;
	///
	// messages to support development/testing
	else if(!strcmp(messageTypeStr,"STARTTIMER")) messageCode = M_STARTTIMER;
	else if(!strcmp(messageTypeStr,"STOPTIMER")) messageCode = M_STOPTIMER;

	return messageCode;
}

/////////////////////////////////////////////////


/////////////////////////////////////////////////
//void WizardFile::DoWizardBeforeAfterCommands(Boolean preDialogs)
OSErr WizardFile::DoCommandBlock(char* blockName)
{
	char line[512];
	char messageTypeStr[64];
	Boolean terminate = true;
	long lineStart;
	Boolean doCommand = false;
	short dialogResNum = -1;
	OSErr err=0;
	
	this->ClearCommandVariables();
	if(this->fText10000Hdl)
	{
			
		// since we are looping through the lines
		// use the more efficient GetLineFromHdlHelper
		//for(lineNum1Relative = 1; !err; lineNum1Relative++)
		for(lineStart = 0; !err;)
		{
			//Boolean gotLine = GetCleanedUpLineFromHdl(lineNum1Relative,this->fText10000Hdl,line,512);
			Boolean gotLine = GetLineFromHdlHelper(1,lineStart,&lineStart,this->fText10000Hdl,line,512,true);
			
			if(!gotLine) break;// no more lines
			if(!line[0]) continue;// blank line
			if(line[0] == '-' || line[0] == '#' || line[0] == '/') continue; // comment line
			if(line[0] == '[')
			{
				//special line
				doCommand = false;
				if(!strncmpnocase(line,blockName,strlen(blockName))) 
				{	// note use strncmpnocase in case they add comments after the block header
					doCommand = true;
				}
			}
			else if(doCommand) 
			{
				MySpinCursor(); // JLM 8/4/99
				err = this->DoWizardCommand(line,512);
			}
		}
	}
	else err = -1;

	return err;
}



OSErr WizardFile::DoWizardCommand(char* line, long maxExpansionLen)
{
	// disect the lines 
	// example line:
	// MESSAGE setfield;TO name of current pattern;fieldName theValue;
	
	// this code is free to destroy the input line
	// The input line can be expanded to maxExpansionLen chars
	// The input string has no leading or trailing white space
	char messageTypeStr[64];
	char targetName[kMaxNameLen];
	char dataStr[64];
	long messageCode = 0;
	long lenOfFirstWord,lenOfSecondWord,lenOfThirdWord;
	char *firstWordStart,*secondWordStart,*thirdWordStart;
	char* commandKeyWord = "";
	char evaluationString[512];
	long len,i,lenKey;
	short versionNum;
	char* paramStart;
	Boolean bVal;
	char errStr[256] = "";
	char tempStr[68] ="";
	char*originalLineStart = line;
	char firstWord[32];
	Boolean bExecuteStatements;
	OSErr err = 0;
	
	/// first determine if the statements will be executed
	bExecuteStatements = true;
	// Note: we are in an IF block when this->fIfLevel >= 1
	// We only execute commands if we are in TRUE for every part of the IF level
	for(i = 1; i <= this->fIfLevel;i++){
		if(!this->fIfValue[i]) 
		{
			bExecuteStatements = false;
			break;
		}
	}

	if(bExecuteStatements)
	{
		err = this->EvaluateAndTrimString(line,maxExpansionLen,WEAK_EVALUATION);
		if(err) return err;
	}
	
	if(!line[0]) return noErr; // empty line

	// get the first word of the command
	firstWordStart = StartOfFirstWord(line,&lenOfFirstWord);
	secondWordStart = StartOfFirstWord(firstWordStart+lenOfFirstWord,&lenOfSecondWord);
	thirdWordStart = StartOfFirstWord(secondWordStart+lenOfSecondWord,&lenOfThirdWord);

	len = _min(31,lenOfFirstWord);
	strncpy(firstWord,firstWordStart,len);
	firstWord[len] = 0; 
	
	// we need to do the IF control block statements first 
	// IMPORTANT !!!
	// We need to keep track of the ifLevel  
	// even when we are not supposed to execute statements
	////////////////////////////////////

	//////
	if(!strcmpnocase(firstWord,"IF"))
	{	// IF ** statement
		if(this->fIfLevel >= MAX_IF_LEVEL) // safety check
			{ ParsingErrorAlert("Number of IF statements exceeds MAX_IF_LEVEL",originalLineStart); err = -1; goto done;}
		this->fIfLevel++;
		this->fIfValue[this->fIfLevel] = false;
		if(bExecuteStatements)
		{	// evaluate the boolean expression
			err = this->EvaluateBooleanString(secondWordStart,maxExpansionLen-(long)(secondWordStart - originalLineStart),&bVal);
			if(!err) this->fIfValue[this->fIfLevel] = bVal;
		}
		return err;
	}
	//////
	if(!strcmpnocase(firstWord,"ELSE"))
	{	//  ELSE statement
		if(this->fIfLevel <= 0) // safety check
			{	ParsingErrorAlert("Found unexpected ELSE statement",originalLineStart); err = -1; goto done;}
		// flip boolean value
		this->fIfValue[this->fIfLevel] = !this->fIfValue[this->fIfLevel];
		return noErr;
	}
	//////
	if(!strcmpnocase(firstWord,"ENDIF"))
	{	//  ENDIF statement
		if(this->fIfLevel <= 0)// safety check
			{	ParsingErrorAlert("Found unexpected ENDIF statement",originalLineStart); err = -1; goto done;}
		this->fIfLevel--;
		return noErr;
	}
	//////
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	// from here on , screen against the IF blocks
	///////////////////////////////
	if(!bExecuteStatements) return noErr;
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	
	// check to see what command it is
	if(!strcmpnocase(firstWord,"SET"))
	{	// starts with SET
		// the second word is the variable name
		char variableName[WIZ_VARIABLE_NAME_LENGTH];
		
		if(!secondWordStart[0]) 
			{ ParsingErrorAlert("The variable of SET statement is missing.",originalLineStart);err = -1; goto done;}
		if(lenOfSecondWord > WIZ_VARIABLE_NAME_LENGTH -1)
			{ ParsingErrorAlert("The variable of SET statement exceeds WIZ_VARIABLE_NAME_LENGTH -1.",originalLineStart); err = -1; goto done;}

		strnzcpy(variableName,secondWordStart,lenOfSecondWord);
		
		// everything from the third word on is the value of the variable
		if(!thirdWordStart[0]) 
			{ ParsingErrorAlert("The value for the variable is missing in SET statement:",originalLineStart); err = -1; goto done;}
		// now evaluate the variable one more time
		// note that the variable could be text, so we use a weak evaluation
		err = this->EvaluateAndTrimString(thirdWordStart,maxExpansionLen-(long)(thirdWordStart - originalLineStart),WEAK_EVALUATION | FORCE_EVALUATION);
		if(err) goto done;
		len = strlen(thirdWordStart);
		if(len > WIZ_VARIABLE_VALUE_LENGTH -1)
			{ ParsingErrorAlert("The value of SET statement exceeds WIZ_VARIABLE_VALUE_LENGTH -1.",originalLineStart); err = -1; goto done;}
		////////////
		// add this variable to the list
		//////////
		// check to see if it is already in the list
		// while looking for a place to add it
		for(i = 0; i< MAX_NUM_WIZ_VARIABLES; i++)
		{
			if(this->fCommandVariable[i].name[0] == 0) 
				break;	 // we found the first opening in the list
			if(!strcmpnocase(variableName,this->fCommandVariable[i].name)) 
				break;

		}
		if(i >= MAX_NUM_WIZ_VARIABLES)
			{ ParsingErrorAlert("The number of variables exceeds MAX_NUM_WIZ_VARIABLES.",originalLineStart); err = -1; goto done;}
		// record the variable	
		strcpy(this->fCommandVariable[i].name,variableName);
		strcpy(this->fCommandVariable[i].value,thirdWordStart);
		goto done;
	}
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	//  LHSTEXT
	/////////////////////////////////////////////////
	if(!strcmpnocase(firstWord,"LHSTEXT"))
	{
		char paramStr[32];
		LHSAnswer  lhs;
		Boolean bTemp;
		double dTemp;
		short freeIndex;
		
		memset(&lhs,0,sizeof(lhs));
		lhs.lhs = true; // default, include on left hand side
		lhs.print = true; // default, include on printout
		lhs.settingsDialogResNum = 0; // default, 0 means use entire wizard sequence of dialogs
		
		err = this->GetFreeIndexLHSAnswers(&freeIndex);
		if(err) { ParsingErrorAlert("MAX_NUM_LHS_ANSWERS has been exceeded",originalLineStart); goto done;}
		
		this->GetParameterString("LHSTEXT",line,lhs.text,LHSANSWER_TEXT_LENGTH);
		if(!lhs.text[0]) 
			{ ParsingErrorAlert("LHSTEXT cannot be white-space",originalLineStart); err = -1; goto done;}
			// optional parameters
		this->GetParameterString("PRINT",line,paramStr,32);
		if(paramStr[0])
		{ // they are trying to tell us something, we demand that we can evaluate it as a Boolean
			err = this->EvaluateBooleanString(paramStr,32,&bTemp);
			if(err) goto done;
			lhs.print = bTemp;
		}
		//		
		this->GetParameterString("LHS",line,paramStr,32);
		if(paramStr[0])
		{ // they are trying to tell us something, we demand that we can evaluate it as a Boolean
			err = this->EvaluateBooleanString(paramStr,32,&bTemp);
			if(err) goto done;
			lhs.lhs = bTemp;
		}
		//		
		this->GetParameterString("settingsDialogResNum",line,paramStr,32);
		if(paramStr[0])
		{ // they are trying to tell us something, we demand that we can evaluate it as a Boolean
			err = this->EvaluateNumberString(paramStr,32,&dTemp);
			if(err) goto done;
			// JLM 2/2/99
			// dialogs with resource number less than 10000 are in the application resources
			// e.g. the almost done dialog
			lhs.settingsDialogResNum = round(dTemp);
		}
		//
		this->fLHSAnswer[freeIndex] = lhs;
		goto done;
	}

	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	if(!strcmpnocase(firstWord,"LOCATIONFILEIDSTR"))
	{
		this->GetParameterString("LOCATIONFILEIDSTR",line,this->fLocationFileIdStr,64);
		goto done;
	}
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	if(!strcmpnocase(firstWord,"LOCATIONFILEFORMAT"))
	{
		err = WizardGetParameterAsShort("LOCATIONFILEFORMAT",line,&versionNum);
		if(!err) this->fLocationFileFormat = versionNum;
		else this->fLocationFileFormat = 0;
		goto done;
	}
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	if(!strcmpnocase(firstWord,"LOCATIONFILEVERSION"))
	{
		err = WizardGetParameterAsShort("LOCATIONFILEVERSION",line,&versionNum);
		if(!err) this->fLocationFileVersion = versionNum;
		else this->fLocationFileVersion = 0;
		goto done;
	}
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	if(!strcmpnocase(firstWord,"EXPIRATIONDATE"))
	{
		long expDate;
		err = WizardGetParameterAsLong("EXPIRATIONDATE",line,&expDate);
		if(!err) this->fLocationFileExpirationDate = expDate;
		else this->fLocationFileExpirationDate = 0;
		goto done;
	}
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	if(!strcmpnocase(firstWord,"MESSAGE"))
	{
		this->GetParameterString("MESSAGE",line,messageTypeStr,64);
		messageCode = WizStringToMessageCode(messageTypeStr);
		if(messageCode > 0)
		{
			this->GetParameterString("TO",line,targetName,64);
			err = model->BroadcastMessage(messageCode,targetName,line,nil);
		}
		goto done;
	}
	
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	if(!strcmpnocase(firstWord,"MINKILOMETERSPERINCH"))
	{
		double minKiloPerInch;
		err = WizardGetParameterAsDouble("MINKILOMETERSPERINCH",line,&minKiloPerInch);
		if(!err && minKiloPerInch > 0)
		{
			this->fMinKilometersPerInch = (float)minKiloPerInch;
		}
		goto done;
	}
	
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	if(!strcmpnocase(firstWordStart,"BEEP"))
	{	//  BEEP statement (for testing)
		SysBeep(5);
		goto done;
	}
	//

	commandKeyWord = "DEBUGSTR";
	lenKey = strlen(commandKeyWord);
	if(!strncmpnocase(firstWordStart,commandKeyWord,lenKey))
	{
		ParsingErrorAlert(firstWordStart,nil); 
		/*err = -1;*/ goto done;
	}
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////

	/// unrecognized command
	ParsingErrorAlert("Unrecognized command line:",originalLineStart);
	err = -1;

done:
	return err;

concatProblemLine:
	ParsingErrorAlert(errStr,originalLineStart);
	return -1; 
	
}

void WizardFile::DoWizardWmsgCommands(short flags)
{
	char line[512];
	char messageTypeStr[64];
	char dataStr[64];
	CHARH dataHdl = 0;
	long messageCode;
	long dialogNum;
	CHARH h;
	OSErr err = 0;
	
	Boolean tellWind = (flags == 0 || flags == WINDFLAG);
	Boolean tellOthers = (flags == 0 );
	
	/////////////////////////////////////////////////
	
	//send the other messages
	/////////////////////////////////////////////////
	
	for(dialogNum = 0 ; !err && tellOthers && dialogNum < kMAXNUMWIZARDDIALOGS;dialogNum++)
	{
		long lineStart;
		this->ClearCommandVariables();
		h = this->fMessage[dialogNum];
		// read the lines of the message handles
		for(lineStart = 0; !err;)
		{
			//GetLineFromHdl(lineNum,h,line,512);
			Boolean gotLine = GetLineFromHdlHelper(1,lineStart,&lineStart,h,line,512,true);
			if(!gotLine) break;// no more lines
			if(!line[0]) continue; // blank line
			err = this->DoWizardCommand(line,512);
		}
		//if (this->fIfLevel) printError("No ENDIF to match IF statement");
	}
	if(tellWind)
	{
	
		TWindMover *wind = model->GetWindMover(false); // don't create
		if(wind) wind->SetActive(true);
	}
	
	model->NewDirtNotification(); // JLM 8/14
	
}



/////////////////////////////////////////////////
void WizardFile::SetPopupItemSelected(DialogPtr theDialog, short itemNum, short itemSelected)
{
#ifdef MAC
	// look in the popTable
	// on the MAC, popupNum is the index into the poptable
	short popupNum;
	if(0< itemNum && itemNum < kMAXNUMWIZARDDIALOGITEMS)
	{
		popupNum = this->fItem[itemNum].popupNum;
		if(0 <= popupNum && popupNum < kMAXNUMPOPUPS)
			 gWizardPopTable[popupNum].lastItemSelected = itemSelected;
	}
#else
	// get the value from the combo box
	// on the IBM, popupNum is the dialog item for the combo box
	HANDLE itemHandle;
	short comboboxID;
	if(0< itemNum && itemNum < kMAXNUMWIZARDDIALOGITEMS)
	{
		comboboxID = this->fItem[itemNum].popupNum;
		itemHandle = GetDlgItem (theDialog, comboboxID);
		if(itemHandle)
			//SendMessage (itemHandle, CB_SETCURSEL, itemSelected-1, 0L);  //Combo box item ID's begin at 0
			SendMessage ((HWND)itemHandle, CB_SETCURSEL, itemSelected-1, 0L);  //Combo box item ID's begin at 0
	}
#endif
}


short WizardFile::GetPopupItemSelected(DialogPtr theDialog, short itemNum)
{
#ifdef MAC
	// look in the popTable
	// on the MAC, popupNum is the index into the poptable
	short popupNum;
	if(0< itemNum && itemNum < kMAXNUMWIZARDDIALOGITEMS)
	{
		popupNum = this->fItem[itemNum].popupNum;
		if(0 <= popupNum && popupNum < kMAXNUMPOPUPS)
			return gWizardPopTable[popupNum].lastItemSelected;
	}
	return 0;
#else
	// get the value from the combo box
	// on the IBM, popupNum is the dialog item for the combo box
	HANDLE itemHandle;
	short comboboxID;
	if(0< itemNum && itemNum < kMAXNUMWIZARDDIALOGITEMS)
	{
		comboboxID = this->fItem[itemNum].popupNum;
		itemHandle = GetDlgItem (theDialog, comboboxID);
		if(itemHandle) 
		{
			//short lastItemSelected = (SendMessage (itemHandle, CB_GETCURSEL, 0, 0L)+1);  //Combo box item ID's begin at 0
			short lastItemSelected = (SendMessage ((HWND)itemHandle, CB_GETCURSEL, 0, 0L)+1);  //Combo box item ID's begin at 0
			return lastItemSelected;
		}
	}
	return 0;
#endif
}

void WizardFile::DoPopupShowHide(DialogPtr theDialog)
{
	short i,k,numDialogItems = NumDialogItems(theDialog);
	short popupNum = 0;
	Boolean haveSelectedText = false;
	for(i = 1;i<= numDialogItems && i < kMAXNUMWIZARDDIALOGITEMS;i++)
	{
		if(fItem[i].type == WIZ_POPUP || fItem[i].type == WIZ_WINDPOPUP)
		{ // 
			char str[256];
			short numScanned,itemNum;
			long item[10]; // 10 items max
			long lineNumInResource;
			short itemSelected = GetPopupItemSelected(theDialog,i);

			//note item lines follow the first line so we add 1 to the itemSelected
			lineNumInResource = itemSelected+1;
			this->GetParameterString("HIDE",lineNumInResource,this->fItem[i].str,str,256);
			///
			if(str[0])
			{
				numScanned = sscanf(str,"%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld",
										&item[0],&item[1],&item[2],&item[3],&item[4],
										&item[5],&item[6],&item[7],&item[8],&item[9]);
				///
				for(k = 0; k< numScanned; k++)
				{
					if(item[k] >0 && item[k] <= numDialogItems)
					{
						if(item[k] < kMAXNUMWIZARDDIALOGITEMS) this->fItem[item[k]].hidden = true;
						HideDialogItem(theDialog,StaticTextIDToComboBoxID(item[k]));
					}
				}
			}
			this->GetParameterString("SHOW",lineNumInResource,this->fItem[i].str,str,256);
			///////
			if(str[0])
			{
				numScanned = sscanf(str,"%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld",
										&item[0],&item[1],&item[2],&item[3],&item[4],
										&item[5],&item[6],&item[7],&item[8],&item[9]);
				///////
				for(k = 0; k< numScanned; k++)
				{
					if(item[k] >0 && item[k] <= numDialogItems)
					{
						ShowDialogItem(theDialog,StaticTextIDToComboBoxID(item[k]));
						if(item[k] < kMAXNUMWIZARDDIALOGITEMS) 
						{
							Boolean wasHidden = this->fItem[item[k]].hidden;
							this->fItem[item[k]].hidden = false;
							if(!haveSelectedText && wasHidden && this->fItem[item[k]].type == WIZ_EDIT)
							{	// we are showing text that was previously hidden
								// the user must have just changed the popup to show this item
								// so select the text
								SafeSelectDialogItemText(theDialog, item[k], 0, 255);
								haveSelectedText = true;
							}
						}
					}
				}
			}
		}
	}
}


OSErr WizardFile::InitWizardDialog(short dialogResNum,DialogPtr theDialog,long dialogFlags,CHARH prevAnswers)
{
	OSErr err=0;
	short numDialogItems = NumDialogItems(theDialog);
	long i;
	char line[256];
	Boolean bCheckDialogResNumbers = (dialogResNum < 10000) ? false:true; // dialogs less than 10000 are an application resource
	#ifdef IBM
		HINSTANCE hInstance = (dialogResNum < 10000) ? hInst:this->fInstDLL;
	#endif
	
	#ifdef MAC
	{
		Rect rect;
		const short kUpdateUserItemNum = 2; // fixed in stone
		if(numDialogItems >1)
		{
			rect = GetDialogItemBox(theDialog,1);
			InsetRect(&rect,-4,-4);
			SafeSetDialogItem(theDialog,kUpdateUserItemNum,userItem + itemDisable,nil,&rect);
			SafeSetDialogItemHandle(theDialog,kUpdateUserItemNum,(CHARH)NonCppDrawingRoutine);
		}
	}
	#endif
	short numPopups = 0;
	Boolean onlyDialog = (
									(dialogFlags & ONLYDIALOG )
									||((dialogFlags & LASTDIALOG) &&(dialogFlags &FIRSTDIALOG))
								);
								
	Boolean changeButtons = (dialogFlags & CHANGEBUTTONS);
	
	CenterDialog(theDialog, CENTER);
	
	ClearWizardPopupTable();
	this->Clear_Dialog();
	
	this->fDialogResNum = dialogResNum;
	
	// set buttons
	if(changeButtons)
	{
		if(onlyDialog)
		{
			SetDlgItemText(theDialog,WIZ_OK,"OK");
			SetDlgItemText(theDialog,WIZ_CANCEL,"Cancel");
		}
		else
		{
			char str[64],welcomeStr[256], refStr[64];
			GetWizButtonTitle_Next(str);
			SetDlgItemText(theDialog,WIZ_OK,str);
			GetWizButtonTitle_Previous(str);
			SetDlgItemText(theDialog,WIZ_CANCEL,str);
			if(dialogFlags & LASTDIALOG) 
			{
				GetWizButtonTitle_Done(str);
				SetDlgItemText(theDialog,WIZ_OK,str);
			}
			if(dialogFlags & FIRSTDIALOG) 
			{
				GetWizButtonTitle_Cancel(str);
				SetDlgItemText(theDialog,WIZ_CANCEL,str);

				// should only do this if one of the newer style
				if (dialogResNum==9920)
				{
					char pathName[256], titleStr[64], locationFileNameStr[32], shortName[32];
					GetPathName(pathName);
					strcpy(titleStr,"Welcome to ");
						//strcpy(fileName,pathName);
						SplitPathFile(pathName, shortName);
					//model->fWizard->GetLocationFileName(locationFileNameStr,true);
					strcat(titleStr,shortName);
					mygetitext(theDialog,WELCOME_TEXT,welcomeStr,255);
					strcat(welcomeStr," for ");
					strcat(welcomeStr,shortName);
					strcat(welcomeStr," :");
					mysetitext(theDialog,WELCOME_TEXT,welcomeStr);
					// code goes here, would need to make button bigger to accomodate this...
					//strcpy(refStr,shortName);
					//strcat(refStr," References");
					//SetDlgItemText(theDialog,WELCOME_REFERENCES,refStr);
					setwtitle(GetDialogWindow(theDialog),titleStr);
				}
			}
		}
	
	}
	
	// classify the dialog items
	for(i = 1;i<= numDialogItems && i < kMAXNUMWIZARDDIALOGITEMS;i++)
	{
		char str[256],wizTypeStr[32];
		
		
		GetDlgItemText(theDialog,i,str,256);
		GetLineFromStr(1,str,line,256);
		this->GetParameterString("WIZTYPE",line,wizTypeStr,32);
		if(wizTypeStr[0] == 0) continue;
		
		if(!strcmp(wizTypeStr,"POPUP")) this->fItem[i].type = WIZ_POPUP;
		else if(!strcmp(wizTypeStr,"UNITS")) this->fItem[i].type = WIZ_UNITS;
		else if(!strcmp(wizTypeStr,"HELPBUTTON")) this->fItem[i].type = WIZ_HELPBUTTON;
		else if(!strcmp(wizTypeStr,"BMP")) this->fItem[i].type = WIZ_BMP;
		else if(!strcmp(wizTypeStr,"WINDPOPUP")) this->fItem[i].type = WIZ_WINDPOPUP;
		else
		{ // unrecognized type
				sprintf(str,"Wizard Resource error.  Unrecognized WIZTYPE %s for item %d of dialog %d",wizTypeStr,i,dialogResNum);
				this->fItem[i].type = 0;
				this->fItem[i].str[0] = 0;
				printError(str);
				continue;
		}
			
		if(this->fItem[i].type)	strcpy(this->fItem[i].str,str);
		
		switch(this->fItem[i].type)
		{	
			case  WIZ_UNITS:
			{
				long editID = 0;
				char defaultStr[32]="";
				char editIDStr[32]="";
				this->GetParameterString("EDITID",line,editIDStr,32);
				if(editIDStr[0]) stringtonum(editIDStr, &editID);
				if(editID < 1) 
				{	// error edit ID not specified
					sprintf(str,"Wizard Resource error.  EDIT not specified for item %d of dialog %d",i,dialogResNum);
					this->fItem[i].type = 0;
					this->fItem[i].str[0] = 0;
					printError(str);
					continue;// to ease development
				}
				if(editID > numDialogItems) 
				{	// error menu ID not specified
					sprintf(str,"Wizard Resource error. Item %d of dialog %d has a EDITID of %d but there are only %d dialog items.",i,dialogResNum,editID,numDialogItems);
					this->fItem[i].type = 0;
					this->fItem[i].str[0] = 0;
					printError(str);
					continue; // to ease development
				}
				this->fItem[i].editID = editID;
				this->fItem[editID].type = WIZ_EDIT;// do we need to mark it as on edit text 
				// fall through
			}
			case  WIZ_WINDPOPUP:
			case  WIZ_POPUP:
			{
				// add menu to the popuptable
				long defaultSelection = 1;
				long menuResID = 0;
				char menuIDStr[32]="";
				char defaultStr[32]="";
				this->GetParameterString("MENUID",line,menuIDStr,32);
				if(menuIDStr[0]) stringtonum(menuIDStr, &menuResID);
				if(menuResID < 1) 
				{	// error menu ID not specified
					sprintf(str,"Wizard Resource error.  MENUID not specified for item %d of dialog %d",i,dialogResNum);
					this->fItem[i].type = 0;
					this->fItem[i].str[0] = 0;
					printError(str);
					continue;// to ease development
				}
				if(bCheckDialogResNumbers && (menuResID < 10000 || menuResID >= 20000) )
				{	// error menu ID not specified
					sprintf(str,"Wizard Resource error. Item %d of dialog %d has a MENUID of %d.  It must be in the range 10,000 to 19,999.",i,dialogResNum,menuResID);
					this->fItem[i].type = 0;
					this->fItem[i].str[0] = 0;
					printError(str);
					continue;// to ease development
				}
				// check menu exits if using the MAC
				// the IBM uses a combo box of ID menuResID
			#ifdef MAC
				// code goes here, make sure we can use menus from main GNOME resources
				if(!this->ResourceExists('MENU',menuResID) && !(menuResID==10050))
				{	// error menu ID 
					sprintf(str,"Wizard Resource error.  MENU %d does not exist. Specified in item %d of dialog %d",menuResID,i,dialogResNum);
					this->fItem[i].type = 0;
					this->fItem[i].str[0] = 0;
					printError(str);
					continue;// to ease development
				}
			#else
				if(!GetDlgItem (theDialog, menuResID))
				{	// on the IBM, we use a combo box with that ID
					sprintf(str,"Wizard Resource error.  ComboBox %d does not exist. Specified in item %d of dialog %d",menuResID,i,dialogResNum);
					this->fItem[i].type = 0;
					this->fItem[i].str[0] = 0;
					printError(str);
					continue;// to ease development
				}
			#endif
				
				this->GetParameterString("DEFAULT",line,defaultStr,32);
				if (defaultStr[0])
				{
					if(!strcmpnocase(defaultStr,"month"))
					{	// to let default depend on which month user chose
						DateTimeRec date;
						SecondsToDate (model->GetStartTime(), &date);
						defaultSelection = date.month;
					}
					else stringtonum(defaultStr, &defaultSelection);
				}
				//if(defaultStr[0]) stringtonum(defaultStr, &defaultSelection);
				if(defaultSelection < 1) defaultSelection = 1;
				
				if(numPopups < kMAXNUMPOPUPS -1)
				{
					#ifdef MAC
					// on the MAC, popupNum is the index into the poptable
						this->fItem[i].popupNum = numPopups;
						
						gOriginalOssmMenus[numPopups].origHdl = LoadMenuHandle(menuResID);
						gOriginalOssmMenus[numPopups].menuID = menuResID;
						
						gWizardPopTable[numPopups].dialogID = dialogResNum; 
						gWizardPopTable[numPopups].popupItemNum = i; 
						gWizardPopTable[numPopups].titleItemNum = 0; // I don't think we want to use this 
						gWizardPopTable[numPopups].menuID = menuResID;
						gWizardPopTable[numPopups].lastItemSelected = defaultSelection; 
					#else
						// on the IBM, popupNum is the dialog item for the combo box
						this->fItem[i].popupNum = menuResID;
						LoadComboBoxItems(hInstance,theDialog,menuResID);
						this->SetPopupItemSelected(theDialog,i,defaultSelection);
						HideDialogItem(theDialog,i);// hide the static text holding the wizard info
					#endif
					numPopups++;
				}
				else printError("Too many wizard popups");

				break;
			}
			case WIZ_HELPBUTTON:
			{
				// set the button title to the chopped off string
				int k;
				for(k = 0; k < strlen(str); k++)
				{
					if(str[k] == ';'|| str[k] == RETURN ||str[k] == LINEFEED)
					{
						str[k] = 0; 
						break; // for loop
					}
				}
				SetDlgItemText(theDialog,i,str);
				break;// switch
			}
			
			#ifdef IBM
				case WIZ_BMP:
				{
					char bmpIDStr[32];
					long bmpResID = 0;
					this->GetParameterString("BMPID",line,bmpIDStr,32);
					if(bmpIDStr[0]) stringtonum(bmpIDStr, &bmpResID);
					if(bmpResID < 1) 
					{	// error menu ID not specified
						sprintf(str,"Wizard Resource error.  BMPID not specified for item %d of dialog %d",i,dialogResNum);
						this->fItem[i].type = 0;
						this->fItem[i].str[0] = 0;
						printError(str);
						continue;// to ease development
					}
					if(bCheckDialogResNumbers && (bmpResID < 10000 || bmpResID >= 20000) )
					{	// error menu ID not specified
						sprintf(str,"Wizard Resource error. Item %d of dialog %d has a BMPID of %d.  It must be in the range 10,000 to 19,999.",i,dialogResNum,bmpResID);
						this->fItem[i].type = 0;
						this->fItem[i].str[0] = 0;
						printError(str);
						continue;// to ease development
					}
					
					// on the IBM, for WIZ_BMP guys, popupNum is the bitmap resource number 
					this->fItem[i].popupNum = bmpResID; // 
					HideDialogItem(theDialog,i);// hide the static text holding the wizard info
					break;
				}
			#endif
		
		}
	}
	 
	// done getting items
#ifdef MAC
	if(numPopups > 0)
	{
		RegisterPopTable(gWizardPopTable,numPopups);
		RegisterPopUpDialog(dialogResNum,theDialog);
	}
#endif
	
	if(prevAnswers)
	{	// set the previous answers
		char valStr[64];
		long lineNum = 1;
		long wizType = 0;
		for(lineNum = 1; true;lineNum++)
		{
			long itemNum = 0,selectionNum = 0;
			long popupNum,numScanned;
			
			GetLineFromHdl(lineNum,prevAnswers,line,256);
			if(!line[0]) break;
			
			/// determine type
			wizType = 0;
			if(wizType == 0)
			{
				this->GetParameterString("WIZ_POPUP",line,valStr,64);
				if(valStr[0]) wizType = WIZ_POPUP;
			}
			if(wizType == 0)
			{
				this->GetParameterString("WIZ_UNITS",line,valStr,64);
				if(valStr[0]) wizType = WIZ_UNITS;
			}
			if(wizType == 0)
			{
				this->GetParameterString("WIZ_EDIT",line,valStr,64);
				if(valStr[0]) wizType = WIZ_EDIT;
			}
			if(wizType == 0)
			{
				this->GetParameterString("WIZ_WINDPOPUP",line,valStr,64);
				if(valStr[0]) wizType = WIZ_WINDPOPUP;
			}
			////////////////////////////
			
			
			
			switch(wizType)
			{
				case WIZ_WINDPOPUP:
				case WIZ_POPUP:
				case WIZ_UNITS:
				{
					numScanned = sscanf(valStr,"%ld %ld",&itemNum,&selectionNum);
					if(numScanned == 2 
						&& 0<= itemNum && itemNum <= numDialogItems && itemNum < kMAXNUMWIZARDDIALOGITEMS
						&& selectionNum >= 0 ) // should we check against the max value as well ?
					{
						if(this->fItem[itemNum].type == wizType)
						{ // go ahead and set the popup
							this->SetPopupItemSelected(theDialog,itemNum,selectionNum);
						}
					}
					break;
				}
				case WIZ_EDIT:
				{
					// string should be the number followed by a space and then the text
					char*  userStr = 0;
					long len = strlen(valStr);
					Boolean foundNum = false;
					// find the space after the number 
					numScanned = sscanf(valStr,"%ld",&itemNum);
					if(numScanned == 1 
						&& 0<= itemNum && itemNum <= numDialogItems && itemNum < kMAXNUMWIZARDDIALOGITEMS)
					{
						if(this->fItem[itemNum].type == wizType)
						{ // go ahead and set the edit text
							for(i = 0; i<len; i++)
							{
								if(foundNum && valStr[i] == ' ')
								{ // we found the space after the number
									valStr[i] = 0;
									userStr = valStr+i+1;
									break;
								}
								if('0'<= valStr[i] && valStr[i] <= '9')
									foundNum = true;
							}
							if(userStr)
							{
								// we found the string
								len = strlen(userStr);
								// chop off string at the ';' char
								for(i = 0; i<len; i++)
								{
									if(userStr[i] == ';') userStr[i] = 0;
								}
								SetDlgItemText(theDialog,itemNum,userStr);
							}
						}
					}
					break;
				}
			}
			/////
		} 
	}
	
	this->DoPopupShowHide(theDialog);
	
	// select the text for the first non-hidden edit text item
	for(i = 1;i<= numDialogItems && i < kMAXNUMWIZARDDIALOGITEMS;i++)
	{
		if(this->fItem[i].type == WIZ_EDIT && this->fItem[i].hidden == false)
		{
 			SafeSelectDialogItemText(theDialog, i, 0, 255);
			break;
		}
	}


	return 0;
}

short	WizardFile::ComboBoxIDToStaticTextID(short itemHit)
{
#ifdef IBM
	long i;
	if(itemHit < kMAXNUMWIZARDDIALOGITEMS) 
		return itemHit; // it's a standard item
	for(i = 0; i< kMAXNUMWIZARDDIALOGITEMS; i++)
	{
		short type = this->fItem[i].type;
		if(type == WIZ_POPUP || type == WIZ_UNITS || type == WIZ_WINDPOPUP)
		{
			if(this->fItem[i].popupNum == itemHit) 
				return i;
		}
	}
#endif
	return itemHit;
}

short	WizardFile::StaticTextIDToComboBoxID(short itemHit)
{
#ifdef IBM
	//if(0 <= itemHit < kMAXNUMWIZARDDIALOGITEMS) 
	if(0 <= itemHit && itemHit < kMAXNUMWIZARDDIALOGITEMS) 
	{
		short type = this->fItem[itemHit].type;
		if(type == WIZ_POPUP || type == WIZ_UNITS || type == WIZ_WINDPOPUP)
		{
			if(this->fItem[itemHit].popupNum > kMAXNUMWIZARDDIALOGITEMS) 
				return this->fItem[itemHit].popupNum;
		}
	}
#endif
	return itemHit;
}


      
#ifdef IBM
void WizardFile::WM_Paint(HWND theDialog)
{
	PAINTSTRUCT ps;
	HDC hdc,hdcM;
	long i;
	hdc = BeginPaint(theDialog,&ps);
	hdcM = CreateCompatibleDC(hdc);
	for(i = 0; i < kMAXNUMWIZARDDIALOGITEMS; i++)
	{	
		if(gCurrentWizardFile->fItem[i].type == WIZ_BMP)
		{
			short bmpResID = this->fItem[i].popupNum;// on the IBM, for WIZ_BMP guys, popupNum is the bitmap resource number 
			char numStr[32];
			HBITMAP oldBitmapHdl=0,newBitmapHdl=0;
			BITMAP bMapS;
			Rect rect;
			HINSTANCE hResInst = gCurrentWizardFile->fInstDLL;
			
			sprintf(numStr,"#%ld",bmpResID);
			if(hResInst) 
				newBitmapHdl = LoadBitmap(hResInst, MAKEINTRESOURCE(bmpResID));
			
			if(!newBitmapHdl)
			{	// check the application resources
				newBitmapHdl = LoadBitmap(hInst, MAKEINTRESOURCE(bmpResID));
			}
			
			rect = GetDialogItemBox(theDialog,i);
			if(newBitmapHdl) 
			{
				oldBitmapHdl = (HBITMAP)SelectObject(hdcM,newBitmapHdl);
				GetObject(newBitmapHdl,sizeof(BITMAP),(LPSTR)&bMapS);
				BitBlt(hdc,rect.left,rect.top,bMapS.bmWidth,bMapS.bmHeight,hdcM,0,0,SRCCOPY);
				SelectObject(hdcM,oldBitmapHdl);
				DeleteObject(newBitmapHdl);
			}
		}
	}
	DeleteDC(hdcM);
	EndPaint(theDialog,&ps);
}
#endif


void WizardFile::WizardHelpFilePath(char* helpPath)
{
	helpPath[0] = 0;
	#ifdef IBM
	{
		// we use the hlp file of the same name if it exists
		this->GetPathName(helpPath);
		if(helpPath)
		{ 	// change extension to .hlp and see if the file exists
			//
			short i,len = strlen(helpPath);
			for(i = 1; i <= 4; i++)
			{
				if(helpPath[len-i] == '.')
				{
					helpPath[len-i] = 0;
					break;
				}
			}
			strcat(helpPath,".HLP");
			if(FileExists(0, 0, helpPath))
			{ 
				// use this associated help file
			}
			else helpPath[0] = 0; // the file does not exist so it cannot be used
		}
	}
	#endif
}

void WizardFile::DoWizardHelp(char* title)
{	
	if(title[0])
	{
	
		#ifdef IBM
		{
			// we use the hlp file of the same name if it exists
			char helpPath[256];
			this->WizardHelpFilePath(helpPath);
			if(!helpPath[0]) {
				(void) HelpFileName(helpPath); // we'll have to use the main helps
			}
			if(helpPath[0])
			{ // use this associated help file
					// strcat(title,".html");
					//if (!HtmlHelp(hMainWnd, helpPath, HH_DISPLAY_TOPIC, (DWORD)(LPSTR)title)) {
					if (!WinHelp(hMainWnd, helpPath, HELP_KEY, (DWORD)(LPSTR)title)) {
						MessageBox(hMainWnd, GS1(136, 0), 0, MB_ICONHAND); // Unable to activate help.
					}
			}
			else GetHelp(title, TRUE);// use the application help file
		}
		#else // MAC
		{
			if (GetHelp(title, TRUE) == HELP_TOPICS)
				GetHelpTopics(title, TRUE);
		}
		#endif
	}
}

Boolean	WizardFile::DoWizardItemHit(DialogPtr theDialog, short itemHit,CHARH *userAnswers, CHARH *messages)
{
	//This routine handles the item hit in the dialog.
	// It returns true when the dialog is over.  I.e. Either the 
	// user hits OK and all the inputs have passed the restrictions
	// or the  user has hit cancel.
	long         err;
	Boolean 		done;// the returned variable
	char			str[256];
	long menuID_menuItem;
	CHARH msgH = 0;
	Boolean hdlChanged = 0;
	
	done = false;

	// JLM 8/22/11, should not inialize the two return values here.
	// There were already initalized in DoWizardDialogResNum,
	// and it leads to a bug in Windows when the user 
	// uses the enter key rather than clicking on the Next button.
	// A KillFocus event passes through here after the button event, 
	// clearing the values we had just set set.
	////
	//*userAnswers = 0;
	//*messages = 0;
	////////////////////////
	
	itemHit = ComboBoxIDToStaticTextID(itemHit);
	
	if(itemHit < kMAXNUMWIZARDDIALOGITEMS)
	{
		short type = this->fItem[itemHit].type;
		if(type == WIZ_POPUP || type == WIZ_UNITS || type == WIZ_WINDPOPUP)
		{
			#ifdef MAC
				PopClick(theDialog, itemHit, &menuID_menuItem);
			#else
				// on the IBM, we are using combo boxes
				// and the click has already happened
			#endif
			
			// deal with hiding and showing of items
			this->DoPopupShowHide(theDialog);
		}
		else if(type == WIZ_HELPBUTTON)
		{
			char title[256]="";
			this->GetParameterString("TOPIC",1,this->fItem[itemHit].str,title,256);
			if(title[0]) this->DoWizardHelp(title);
		}
	}
	/////////////////////////////////////////////////

	switch(itemHit)
	{
		case WIZ_OK: // OK button
		{

			#ifdef IBM //////////////////////////////{
			// On the IBM, it turns out that if the user hits the OK button quickly
			// the dialog can be dismissed before it actually appears.  It seems a good 
			// idea to prevent this behavior.
			if(IsWindowVisible(theDialog) == false)
			{
				ShowWindow(theDialog,SW_SHOW);
				UpdateWindow(theDialog);
			}
			#endif ////////////////////////////////}
			
			/// Pull data, check it all, and save it all
			err = this->RetrieveValueStrings(theDialog);
			if(err)  break;
			
			*userAnswers = this->RetrieveUserAnswer(theDialog);
			
			
			// get the WMSG message strings so that we can substitute the dialog variables
			// while we have that information
			msgH = this->GetWMSG(this->fDialogResNum);
			
			// Note: msgH can be many lines of commands
			// what we have to do is substitute the values of the popups into these commands
			// i.e replace the $variables and 
			if(msgH)
			{
				// substitute values for the "$" variables
				for(;;)
				{
					err = this->SubstituteDollarVariable(msgH,&hdlChanged,theDialog);
					if(err)
					{
						DisposeHandle((Handle)msgH); msgH = nil;
						if(*userAnswers) {DisposeHandle((Handle)*userAnswers); *userAnswers = nil;} // 8/22/11
						return false; // not done, user must cancel out of the dialog
					}
					if(!hdlChanged) break; //no more variables
				}
				////////////
				
				// we can no longer evaluate the messages here
				// the lines might contain variables and we need to execute the lines get 
				// the variables defined
				// so save the Evaluation for DoWizardCommand()
				// JLM 3/2/99
				
				if(err)
				{
					DisposeHandle((Handle)msgH); msgH = nil;
					if(*userAnswers) {DisposeHandle((Handle)*userAnswers); *userAnswers = nil;} // 8/22/11
					return false; // not done, user must cancel out of the dialog
				}
				// pass back the messages
				*messages = msgH;
			}
			//
			
			
			done= true;
			break;
		}
		
		case WIZ_CANCEL: // cancel
		{
			done= true;
			break;
		}
	}// end of switch
	
	return(done);// return the flag which is "true" if the user hit cancel or if the
	// user hit OK and all of the data checks out.  Otherwise the flag is "false".


}

OSErr WizardFile::CallWindDialog(long dialogFlags, Boolean *userCancel, CHARH *userAnswers, CHARH *messages)
{
	TWindMover * windMover = model->GetWindMover(true); // create if necessary
	OSErr err = 0;
	Boolean onlyDialog = (
									(dialogFlags & ONLYDIALOG )
									||((dialogFlags & LASTDIALOG) &&(dialogFlags &FIRSTDIALOG))
								);
	Boolean useNextPreviousButtons = !onlyDialog;
	*userAnswers = 0; // we keep the answers in a separate field
	
								
	if (windMover)  
	{
		Boolean settingsForcedAfterDialog = false; // don't force them in the wizard
		windMover -> SetIsConstantWind(!this->fUseVariableWind);
		if(!this->fUseVariableWind) 
			windMover->SetClassName("Constant Wind"); // 4/28/00
		else
			windMover->SetClassName("Variable Wind"); // 4/28/00
		err=EditWindsDialog(windMover,model->GetStartTime(),useNextPreviousButtons,settingsForcedAfterDialog);
	}
	else err = true;
	
	if(err == USERCANCEL) 
	//if(err) 
	{
		*userCancel = true; // they return user cancel as an error
		err = 0;
	}
	else *userCancel = false;
	
	*messages = 0;

	return err;
}


OSErr WizardFile::CallModelDialog(long dialogFlags, Boolean *userCancel, CHARH *userAnswers, CHARH *messages)
{
	OSErr err;
	*userCancel = false;
	*userAnswers = 0;
	*messages = 0;
	
	err = ModelSettingsDialog(true);
	if(!err)
	{	// user accepted the dialog values
		*userCancel = false;
	}
	else
	{	
		if(err == USERCANCEL)
		{
			err = noErr;
			*userCancel = true;
		}
		else
		{
			// a real error
			// code goes here
			// should we report this to the user ?
		}
	}
	return err;
}

OSErr WizardFile::DoWizardDialogResNum(long resNum, long dialogFlags, Boolean *userCancel, CHARH prevAnswers,CHARH *userAnswers, CHARH *messages)
{ 	// returns error code if dialog fails.
	// returns no err if user cancels or completes dialog (check parameters).
	short itemHit;
	*userCancel = false;
	*userAnswers = 0;
	*messages = 0;
	
	
	#ifdef MAC
	{	// macintosh code
		GrafPtr		savePort;
		DialogPtr theDialog;
		Boolean done = false;
		GetPortGrafPtr(&savePort);
 		theDialog=GetNewDialog(resNum,(Ptr)nil,(WindowPtr)-1); 
		if(theDialog)
		{
			SetPortDialogPort(theDialog);
			gCurrentWizardFile = this; 
			this->InitWizardDialog(resNum,theDialog,dialogFlags,prevAnswers);
			SetDefaultItemBehavior(theDialog); 
			ShowWindow(GetDialogWindow(theDialog));
			done = false;
			while(done == false)
			{
				//ModalDialog((ModalFilterUPP)MakeUPP((ProcPtr)WizSTDFilter, uppModalFilterProcInfo),&itemHit);
				ModalDialog(MakeModalFilterUPP(WizSTDFilter), &itemHit);
				done = this->DoWizardItemHit(theDialog,itemHit,userAnswers,messages);
			}	
			gCurrentWizardFile = nil; 
			DisposeDialog(theDialog);
		}
		else return -1; 
		SetPortGrafPort(savePort);
	}
#else
	{	// IBM code
		GrafPtr oldPort;
		WindowPtr parent = FrontWindow();// mapWindow ??
		
		HINSTANCE inst = (resNum < 10000) ? hInst:this->fInstDLL;
		// JLM 2/2/99
		// dialogs with resource number less than 10000 are in the application resources
		// e.g. the almost done dialog
		
		GetPortGrafPtr(&oldPort);
		
		//lpData = data;
		//gInitProc = initProc;
		//gClickProc = clickProc;
		
		CmdPeriod(); // ignore pending ESC
		
		//PushModalGlobals(frontWindow);
		// set globals for our proc
		gCurrentWizardFile = this;
		memset((void *)&gIBMDialogVar, 0, sizeof(gIBMDialogVar));
		gIBMDialogVar.resNum = resNum;
		gIBMDialogVar.dialogFlags = dialogFlags;
		gIBMDialogVar.prevAnswers = prevAnswers;
		gIBMDialogVar.userAnswers = userAnswers;
		gIBMDialogVar.messages = messages;
		
		itemHit = DialogBox(inst, MAKEINTRESOURCE(resNum), parent, (DLGPROC)WizSTDFilter);
		
		//PopModalGlobals();
		gCurrentWizardFile = 0;
		memset((void *)&gIBMDialogVar, 0, sizeof(gIBMDialogVar));
		
		SetPortGrafPort(oldPort);
		
		if(itemHit == -1) return itemHit;  // error in dialog
	}
	#endif
	
	ClearWizardPopupTable();
	RestoreOSSMMenuHdls();
	this->Dispose_Dialog();

	
	// set return variables
	if(itemHit == WIZ_CANCEL) *userCancel = true;
	

	return noErr;
}

OSErr WizardFile::DoWizardDialog(short dialogNum,long dialogFlags, Boolean *userCancel, CHARH prevAnswers,CHARH *userAnswers, CHARH *messages)
{ 	// returns error code if dialog fails.
	// returns no err if user cancels or completes dialog (check parameters).
	short itemHit;
	*userCancel = false;
	*userAnswers = 0;
	*messages = 0;
	
	long specialFlag = 0;
	long resNum = this->DialogResNum(dialogNum,&specialFlag);
	// special cases
	if(specialFlag == WINDFLAG)
		return this->CallWindDialog(dialogFlags,userCancel,userAnswers,messages);
	else if(specialFlag == MODELFLAG)
		return this->CallModelDialog(dialogFlags,userCancel,userAnswers,messages);
	
	// check dialog exists
	if(!this->DialogExists(dialogNum)) return -1;
	
	return DoWizardDialogResNum(resNum,dialogFlags,userCancel,prevAnswers,userAnswers,messages);
}
	
OSErr WizardFile::CheckExpirationDate()
{
	char msg[256]="";
	if (fLocationFileExpirationDate==0)
	{
		if ((!strncmpnocase(fLocationFileIdStr,"Kaneohe Bay",strlen("Kaneohe Bay")) && fLocationFileVersion == 0) 
			|| (!strncmpnocase(fLocationFileIdStr,"San Juan PR",strlen("San Juan PR")) && fLocationFileVersion == 0)
			|| (!strncmpnocase(fLocationFileIdStr,"Boston and vicinity",strlen("Boston and vicinity")) && fLocationFileVersion == 0) 
			|| (!strncmpnocase(fLocationFileIdStr,"Apra Harbor",strlen("Apra Harbor")) && fLocationFileVersion == 0))
		{
			sprintf(msg,"The Location File you have chosen is outdated. Please download the recent version from the NOAA OR&R web site:  http://www.response.restoration.noaa.gov/software/\ngnome/locfiles.html .");
		}
#ifdef MAC /* TARGET_API_MAC_CARBON*/ 
		if ((!strncmpnocase(fLocationFileIdStr,"Columbia River Estuary",strlen("Columbia River Estuary")) && fLocationFileVersion == 0) 
			|| (!strncmpnocase(fLocationFileIdStr,"Galveston Bay",strlen("Galveston Bay")) && fLocationFileVersion == 0)
			|| (!strncmpnocase(fLocationFileIdStr,"Lower Mississippi River",strlen("Lower Mississippi River")) && fLocationFileVersion == 0) 
			|| (!strncmpnocase(fLocationFileIdStr,"Mobile Bay",strlen("Mobile Bay")) && fLocationFileVersion == 0)
			|| (!strncmpnocase(fLocationFileIdStr,"Strait of Juan de Fuca",strlen("Strait of Juan de Fuca")) && fLocationFileVersion == 0))
		{
			sprintf(msg,"The Location File you have chosen is outdated. Please download the recent version from the NOAA OR&R web site:  http://www.response.restoration.noaa.gov/software/gnome/locfiles.html .");
		}
#endif
	}
	else
	{
		if (fLocationFileExpirationDate < LASTYEARINPOPUP)
		{
			sprintf(msg,"The Location File you have chosen is outdated. Please download the recent version from the NOAA OR&R web site:  http://www.response.restoration.noaa.gov/software/\ngnome/locfiles.html .");
		}

	}
	if (msg[0])
	{
		CloseFile();
		short buttonSelected  = MULTICHOICEALERT(1687,msg,FALSE);
		switch(buttonSelected){
			case 1:// ok
				return -1;// don't load location file
				//break;  
			case 3: // help
				return -1;// don't load location file
				//break;
		}
	}
	return 0;
}

/////////////////////////////////////////////////
OSErr WizardFile::GoThroughDialogs(Boolean * userCancelFlag,Boolean firstTimeThroughFile)
{	// returns an error code, but code has already alerted the user
	long i;
	OSErr err = noErr;
	Boolean userCancel = false;
	*userCancelFlag = false;

	if(!this->fIsOpen) this->OpenFile();
	if(!this->fIsOpen) return -1; //  unable to open the file 
		
	if(!this->DialogExists(1))	
	{	// no dialogs
		// opened the file, but it is not a valid wizard file
		printError("This file contains no wizard dialogs");
		return -1;
	}

	this->SaveAnswers();
	
	for(i = 1; true; i++)
	{
		long dialogFlags = CHANGEBUTTONS;
		CHARH userAnswers = 0;
		CHARH messages = 0;
		Boolean thisDialogExists = this->DialogExists(i);
		Boolean nextDialogExists = this->DialogExists(i+1);
		if(!thisDialogExists) 
		{
			break;
		}
		if(!nextDialogExists)  dialogFlags |= LASTDIALOG;
		if(i == 1) dialogFlags |= FIRSTDIALOG;
		if(firstTimeThroughFile)  dialogFlags |= FIRSTTIMETHROUGHFILE;
		
		err = this->DoWizardDialog(i,dialogFlags,&userCancel,this->fAnswer[i],&userAnswers,&messages);
		if(!err && !userCancel)
		{
			if(this->fAnswer[i]) DisposeHandle((Handle)this->fAnswer[i]);this->fAnswer[i] = nil;
			this->fAnswer[i] = userAnswers;
			this->fMessage[i] = messages;
			continue;// on to the next dialog
		}
		else
		{	// err or cancel , dispose just to be sure
			if(userAnswers) DisposeHandle((Handle)userAnswers); userAnswers = nil;
			if(messages) DisposeHandle((Handle)messages); messages = nil;
		}
		if(err) break;
		if(userCancel)
		{	// back up a dialog
			if(i == 1) 
			{	// they backed out of first dialog
				*userCancelFlag = true;
				break; 
			}
			i-=2;  // backs up one after incrementing
			continue;
		} 
	}
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////


	if(err || userCancel)
	{
		this->RestoreAnswers();
	}
	else
	{
		// we have gotten through all the dialogs
		// we don't need the saved answers anymore
		this->Dispose_SavedAnswers();
	}
	/////////////////////////////////////////////////
	if(err)
	{
		char str[256];
		sprintf(str,"Wizard returned error %d",err);
		printError(str);
	}

	//// note: on the MAC
	//we must leave the file open so the help texts are available
	
	return err;
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////


/////////////////////////////////////////////////
/////////////////////////////////////////////////


/////////////////////////////////////////////////
/////////////////////////////////////////////////


enum { WIZ_MAINWIZARDITEM = 0,WIZ_LOCALENAME,WIZ_LHSANSWER};
/////////////////////////////////////////////////
/////////////////////////////////////////////////

/////////////////////////////////////////////////
// constuctor	
LocaleWizard::LocaleWizard()
{
	this->fCurrentFile = 0;
	strcpy(this->className,"Location Wizard");
}

// destructor
LocaleWizard::~LocaleWizard(void)
{
	this->Dispose(); 
}

/////////////////////////////////////////////////


void LocaleWizard::Dispose()
{
	this->DisposeCurrentFile();
}


void LocaleWizard::DisposeCurrentFile(void)
{
	if(this->fCurrentFile) delete this->fCurrentFile;
	this->fCurrentFile = 0;
}

/////////////////////////////////////////////////
Boolean LocaleWizard::GetMinKilometersPerScreenInch(float *minKilometers)
{
	if(this->fCurrentFile) 
	{
		return this->fCurrentFile->GetMinKilometersPerScreenInch(minKilometers);
	}
	return false;
}

long LocaleWizard::GetListLength()
{
	long listLength;
	ListItem item;
	long n = 0;// dummy
	short indent = 0;// dummy
	short style = 0;// dummy
	char text[256];// dummy
	
	item =  this->GetNthListItemOrListLength(n,indent,&style,text,&listLength);

	return listLength;
}

/////////////////////////////////////////////////
ListItem LocaleWizard::GetNthListItem(long n, short indent, short *style, char *text)
{
	return this->GetNthListItemOrListLength(n,indent,style,text,nil);
}

void LocaleWizard::GetLocationFileName(char* shortFileName, Boolean chopExtension)
{
	if(!this->fCurrentFile) strcpy(shortFileName, "");
	else 
	{
		char name[256] ="";
		// use short file name
		if(!name[0])
		{
			char fullPath[256];
			this->fCurrentFile->GetPathName(fullPath);
			SplitPathFile(fullPath,name);	
		}
		strcpy(shortFileName,name);
#ifdef IBM
		if (chopExtension)
		{
			if(strlen(shortFileName) >= 4) 
			{
				long i,nameLen;
				nameLen = strlen(shortFileName);
				for(i=1; i<=4; i++)
				{
					if(shortFileName[nameLen-i] == '.')
					{	
						shortFileName[nameLen-i]=0;
						break; // only take off last extension
					}
				}
			}
		}
#endif

	}
}

void LocaleWizard::GetLocationFileFullPathName(char *pathName)
{
	pathName[0] = 0;
	if(this->fCurrentFile)
		this->fCurrentFile->GetPathName(pathName);
}
/////////////////////////////////////////////////


void LocaleWizard::WizardHelpFilePath(char* helpPath)
{
	helpPath[0] = 0;
	if(this->fCurrentFile) 
		this->fCurrentFile->WizardHelpFilePath(helpPath); 
}


short LocaleWizard::NumLeftHandSideAnswersOnPrintout(void)
{
	short num = 0;
	if(this->fCurrentFile) num = this->fCurrentFile->NumLeftHandSideAnswersOnPrintout();
		
	return num;
}
void LocaleWizard::LeftHandSideTextForPrintout(short desired0RelativeLineNum,char* text)
{
	text[0] = 0;
	if(this->fCurrentFile) 
		this->fCurrentFile->LeftHandSideTextForPrintout(desired0RelativeLineNum,text);
}


ListItem LocaleWizard::GetNthListItemOrListLength(long n, short indent, short *style, char *text,long *listLength)
{
	// call with listLength nil to get nth Item
	// having listLength non-nil just counts the list
	//long i, m, count;
	ListItem item = { this, 0, indent, 0 };
	Boolean returnTheItem = (listLength == nil);
	long theMode = model->GetModelMode();
	short numLHSAns = 0;

	if(listLength) *listLength = 0;
	
	///////{ 
	if(theMode >= ADVANCEDMODE && !this->fCurrentFile) 
	{
		// we will only put the wizard in the list in advanced mode 
		// if the user has used a wizard file
		item.owner = 0;
		return item;
	} 
	//////////////////}
	
	if(this->fCurrentFile)
		numLHSAns = this->fCurrentFile->NumLeftHandSideAnswersInList();
	else
		numLHSAns = 0;
	
	if (n == 0 && returnTheItem) {
		item.index = WIZ_MAINWIZARDITEM;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		strcpy(text, "Location File");
		*style = bold;
		return item;
	}
	n -= 1; if(listLength) (*listLength)++;
	
	if (bOpen) 
	{
		if (n == 0 && returnTheItem) 
		{ 	// Locale name
			char localeName[64];
			item.index = WIZ_LOCALENAME;
			item.indent = indent;
			if(numLHSAns == 0)
			{
				// then it must be an old file not supporting LHSAnswers
				// show the file name on line 1			
				if(!this->fCurrentFile) strcpy(localeName, "Select location using Open menu item");
				else this->GetLocationFileName(localeName,false);
				strcpy(text,localeName);
			}
			else this->fCurrentFile->LeftHandSideTextForList(0,text);
			return item;
		}
		n -= 1; if(listLength) (*listLength)++;
		//
		//item 2 would go here
		
		if(numLHSAns > 1)
		{
			// add the remaining lines
			short numLinesAddedHere = numLHSAns-1;
			if(listLength) (*listLength) += numLinesAddedHere;
			if( n < numLinesAddedHere)
			{
				item.index = WIZ_LHSANSWER+n;
				item.indent = indent;
				this->fCurrentFile->LeftHandSideTextForList(n+1,text); // add 1 because we already used the first line above
				return item;
			}
			n-=numLinesAddedHere;
		}
		
	}
	
	// if we get here, we didn't find the item
	item.owner = 0;

done:	
	return item;
}

Boolean LocaleWizard::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
	{
		switch (item.index) 
		{
			case WIZ_MAINWIZARDITEM: 
				this->bOpen = !this->bOpen; 
				return TRUE;
		}
	}
	
	else if(doubleClick)
	{
		(void)this->SettingsItem(item);// double click is same as settings
		return TRUE; // ?? to update ??
		//return FALSE; //  ?? did not change bullet
	}
	

	
	return FALSE;
}

OSErr LocaleWizard::SettingsItem(ListItem item)
{	// return error code
	OSErr err = 0;
	switch (item.index) 
	{
		case WIZ_MAINWIZARDITEM:
			this->OpenMenuHit();  
			return FALSE;
		case WIZ_LOCALENAME:
			this->InvokeWizardMenuHit();
			return FALSE;
	}
	if(item.index >= WIZ_LHSANSWER)
	{
		short i = item.index - WIZ_LHSANSWER;
		Boolean handled = this->fCurrentFile->LeftHandSideAnswerSettingsItemHit(i);
		if(!handled) this->InvokeWizardMenuHit();
		return false;
	}

	return err;
}


Boolean LocaleWizard::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	

	switch (item.index) {
		case WIZ_MAINWIZARDITEM:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
			break;
		default:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
			break;
	}
	
	return false;
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////


/////////////////////////////////////////////////
#define STARTSTRING "RESNUM "

Boolean PathIsWizardResourceHelper(char* path,long* resNum)
{
	if(path && path[0])
	{
		if(!strncmpnocase(path,STARTSTRING,strlen(STARTSTRING)))
		{
			// get the file contents from a TEXT resource
			*resNum = atol(path+strlen(STARTSTRING));
			if(*resNum > 0)
			{
				return true;
			}
		}
	}
	return false;
}

Boolean LocaleWizard::PathIsWizardResource(char* path)
{
	long resNum;
	return PathIsWizardResourceHelper(path,&resNum);
}


OSErr LocaleWizard::ReadFileContentsFromResource(char* path,CHARHP handle,Boolean terminate)
{	//// get the file contents from a TEXT resource
	long resNum;
	if(!handle) return -1;
	*handle = nil;
	if(!PathIsWizardResourceHelper(path,&resNum)) return -1;
	*handle = this->fCurrentFile->GetResource('TEXT',resNum,terminate);
	if(*handle == nil) return -1;
	return noErr;
}

OSErr LocaleWizard::ReadSectionOfFileFromResource(char* path,char* ptr,long maxLength,long offset)
{
	CHARH h = nil;
	long resNum,i;
	Boolean terminate = false;
	if(!ptr) return -1;
	*ptr = 0;
	if(!PathIsWizardResourceHelper(path,&resNum)) return -1;
	h = this->fCurrentFile->GetResource('TEXT',resNum,terminate);
	if(h)
	{	// copy maxLength chars
		long hLen = _GetHandleSize((Handle)h);
		for(i = 0; i < maxLength; i++) 
		{
			//if(i< hLen) ptr[i] = INDEXH(h,i);
			if(i+offset< hLen) ptr[i] = INDEXH(h,i+offset);
			else ptr[i] = 0;
		}
		DisposeHandle((Handle)h);
		return 0;
	}
	return -1;
	
}

OSErr LocaleWizard::MyGetFileSizeFromResource(CHARPTR pathName, LONGPTR size)
{
	long resNum;
	Boolean exists = false;
	*size = 0;
	if(!PathIsWizardResourceHelper(pathName,&resNum)) return -1;
	exists =  this->fCurrentFile->ResourceExists('TEXT',resNum,size);
	if(!exists) { *size = 0; return -1;} 
	else return noErr;
}
/////////////////////////////////////////////////


void LocaleWizard::OpenMenuHit(void)
{	
	this->OffToSeeTheWizard(true,nil);
}

void LocaleWizard::OpenWizFile(char* path)
{	// called in advanced mode when user picks a wizard file	
	this->OffToSeeTheWizard(true,path);
}

long LocaleWizard::CloseMenuHit(void)
{
	// the user has selected the Close menu 
	// In novice mode, this would mean to stop using the locale file
	if(model->GetModelMode() <  ADVANCEDMODE)
	{
		// code goes here
		// ?? should we ask if they want to save changes ?
		// I think we should remove the key board short cut for this menu item
		
		// we can return USERCANCEL if the user wants to cancel
		//printNote("Asking to save changes before closing is unimplemented in the Wizard");
		//if() return USERCANCEL;
	}
	else
	{
		// they are in advanced mode, don't ask if they want to 
		// wizard answers
	}
	this->DisposeCurrentFile();
	InvalListLength();
	InvalMapDrawingRect();
	return 0;
}


Boolean LocaleWizard::OKToChangeWizardMode(long oldMode,long newMode,Boolean * closeFile)
{
	// returns true if OK to leave wizard mode
	Boolean bOKToChange = true;
	long dialogFlags =0; //i.e.  don't change buttons
	CHARH userAnswers = 0;
	CHARH messages = 0;
	Boolean userCancel = false;
	short dialogID = 0;
	Boolean needToCloseFile = false;
	*closeFile = false;
	OSErr err = 0;

	if(newMode == ADVANCEDMODE && oldMode != ADVANCEDMODE)
	{ // they are switching to advanced mode with an open wizard file
		// we must tell them they cannot get back
		if(this->fCurrentFile) {
			dialogID = 9950;
		}
	}
	else if(newMode != ADVANCEDMODE && oldMode == ADVANCEDMODE)
	{	// they are switching to a lower mode
		// and have a wizard file open, we must close this file to 
		// ensure everything matches
		
		// JLM 3/3/99  CJ thinks we should always clear everything, 
		// even if the stuff did not come from a wizard file.
		//if(this->fCurrentFile) 
		{
			dialogID = 9960;
			needToCloseFile = true;
		}
	}
	
	if(dialogID)
	{
		// do dialog
		// don't change button names.
		WizardFile *fakeLocaleFile = new WizardFile(); // fake because we use resources in the application
		if(!fakeLocaleFile) 
		{ // memory error
			WizardOutOfMemoryAlert();
			bOKToChange = false;
		}
		else
		{
			err = fakeLocaleFile->DoWizardDialogResNum(dialogID,dialogFlags,&userCancel,nil,&userAnswers,&messages);
			if(err  || userCancel) bOKToChange = false;// they canceled
			fakeLocaleFile->Dispose();
			delete fakeLocaleFile;
			fakeLocaleFile = 0;
		}
	}
	
	if(bOKToChange && needToCloseFile)
	{
		*closeFile = true;
	}
	return bOKToChange; 
}

OSErr LocaleWizard::SaveAsMenuHit(void)
{
	char s[512],temp[256];
	char path[256];
	OSErr err = 0;
	BFPB bfpb;
	long count,i;
	
	if(!this->fCurrentFile )
	{	// the user cannot save a file if there is not an open wizard file
		printError("There is no location file open. There are no answers to save.");
		return -1;
	}
	
	model -> GetSaveFileName (path);
	err = AskUserForSaveFilename(path,path,".LFS",true);
	if(err) return USERCANCEL; // user cancel
	
	model -> SetSaveFileName (path);
	
	hdelete(0, 0, path); // get rid of old file
	err = hcreate(0, 0, path, APPL_SIG, WIZARDSAVEFILETYPE);
	if (err)	{ sprintf(s,"Unable to create file %s", path ); printError(s); goto done; }
	err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE);
	if(err)	{ sprintf(s,"Unable to open file %s", path ); printError(s); goto done;  }
	
	SetWatchCursor();
	sprintf(s, "Writing file: %s...", path);
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(s);
	
	err = this->fCurrentFile->WriteWizardSaveFile(&bfpb,path);
	if(err) goto closeFile;
	
	err = WriteWizardModelSettings(&bfpb);
	if(err) goto closeFile;
	
	err = WriteWizardWind(&bfpb);
	if(err) goto closeFile;
	
	err = WriteWizardSpills(&bfpb);
	if(err) goto closeFile;

closeFile:
	FSCloseBuf(&bfpb);
	
done:
	if (err)
		hdelete(0, 0, path);
	
	DisplayMessage(0);
	InitCursor();
	return err;
}

 
Boolean LocaleWizard::QuitMenuHit(void)
{
	Boolean userChangedMind = false;
	// the user has selected the Quit menu 
	
	// code goes here
	// ?? should we ask if they want to save changes ?
	return userChangedMind;
}

Boolean LocaleWizard::StartUp(void)
{	// returns TRUE if program should quit 
	Boolean quitFlag = false;
	OSErr err = 0;
	// the program has started up
	if(model->GetModelMode() < ADVANCEDMODE && !HaveOpenWizardFile())
	{
		// go through dialogs 9900,9901, etc till there are no more
		// don't change button names.
		long i,startDialogNum = 9900;
		WizardFile *fakeLocaleFile = new WizardFile(); // fake because we use resources in the application
		if(!fakeLocaleFile) 
		{ // memory error
			WizardOutOfMemoryAlert();
			quitFlag = true;
			return quitFlag;
		}
		for(i = startDialogNum; i <= startDialogNum+10; i++)
		{
			long dialogFlags =0; //i.e.  don't change buttons
			CHARH userAnswers = 0;
			CHARH messages = 0;
			Boolean userCancel = false;
			
			err = fakeLocaleFile->DoWizardDialogResNum(i,dialogFlags,&userCancel,nil,&userAnswers,&messages);
			if(!err && !userCancel)
			{
				continue;// on to the next dialog
			}
			
			if(err) 
			{
				// probably just a missing dialog
				// so end of the line
				break;
			}
			if(userCancel)
			{	// back up a dialog
				if(i == startDialogNum) 
				{	// they backed out of first dialog
					quitFlag = true;
					break; 
				}
				i-=2;  // backs up one after incrementing
				continue;
			} 
		}
		
		fakeLocaleFile->Dispose();
		delete fakeLocaleFile;
		fakeLocaleFile = 0;
		
		if(i> startDialogNum)
		{
			// they must have hit the select file dialog
			this->OpenMenuHit();
			// if the user does not select a file
			// should we let them go on, (to get to advanced mode)
			// or should we quit ?
		}
	}
	return quitFlag;
}


void LocaleWizard::InvokeWizardMenuHit(void)
{
	this->OffToSeeTheWizard(false,nil);
}



void LocaleWizard::OffToSeeTheWizard(Boolean ask,char* providedPath)
{
	char pathName[256] = "",str[256];
	char currentPathName[256] = "";
	char wizSaveFilePath[256] = "";
	OSErr err = 0;
	Boolean userCancel = false;
	WizardFile *wfp = 0;
	Boolean firstTimeFile = false;
	TModelDialogVariables  savedSettings = model->GetDialogVariables();
	TWindMover *wind = nil,*windClone = nil;
	TMap	*saveMoverMap;
	Boolean bCloseCurrentFile = false;
	CHARH saveFileContentsHdl = 0;
	Boolean bForceCloseFile = false;
	long numMaps = model -> mapList -> GetItemCount();
	Boolean askUserAboutSaving;

	gInWizard = TRUE;
	
	pathName[0] = 0;
	if(providedPath && providedPath[0])
	{	// use the name provided
		strcpy(pathName,providedPath);
	}
	else if(!this->fCurrentFile || ask) 
	{	// ask the user
		Boolean newFile = true;
		Boolean allowGnomeSaveFile = false;
		pathName[0] = 0;
		err = AskForWizardFile(pathName,allowGnomeSaveFile);
		if(err) goto done;  // user probably canceled
	}
	else 
	{	// use the current file
		this->fCurrentFile->GetPathName(pathName); 
	}
	
	if(IsWizardSaveFile(pathName)) 
	{ 	// we need to get the localeFile path from the file
		// and short circuit GoThroughDialogs() below 
		strcpy(wizSaveFilePath,pathName);
		bForceCloseFile = true;// so that we get exactly what was in the file
		err = ReadFileContents(TERMINATED,0, 0, wizSaveFilePath, 0, 0, &saveFileContentsHdl);
		if(err){printError("Unable to read file contents into memory.");goto done; }
		err = GetLocaleFilePath(saveFileContentsHdl,wizSaveFilePath,pathName);
		if(err) goto done; // can't find the file
	} 
		
	
	// JLM 9/1/98
	firstTimeFile = true; // initialize to true
	if(this->fCurrentFile)
	{	// if we have a current file
		// we need to check to see if the user picked the same file again
		this->fCurrentFile->GetPathName(currentPathName);
		if(!strcmp(pathName,currentPathName) && !bForceCloseFile)
		{
			// they chose the same file as they selected before
			firstTimeFile =  false;
			wfp = this->fCurrentFile;
		}
		else
		{
			// they have chosen a different file
			// we need to close the current file so the resources do not get confused
			bCloseCurrentFile = true;
		}
	}
	
/////////////////////////////////////////////////
/////////////////////////////////////////////////
	// if they had a wizard file open, and the new is different, ask if they want to save changes
	askUserAboutSaving = (firstTimeFile && numMaps > 0 && model -> IsDirty ());
	if(askUserAboutSaving)
	{
		long dialogID = M77C;
		short itemHit = MULTICHOICEALERT(dialogID, 0, FALSE);
		switch(itemHit)
		{
			case 1: break; // NO, default button
			case 3: goto done;; //CANCEL button
			case 6://YES button
				if (model->GetModelMode() >= ADVANCEDMODE)
					err = AdvancedSaveFileSaveAs(); 
				else
					err = model->fWizard->SaveAsMenuHit();
				if(err) goto done; // might be user cancel
				break;
		}
	}
	// do we need to call CloseSaveFile() here if it is a different file ????
	// I.e. do the [BEFORE] messages do anything that will freak
/////////////////////////////////////////////////
/////////////////////////////////////////////////

	
	if(firstTimeFile)
	{
		wfp = new WizardFile(pathName);
		if(!wfp){ WizardOutOfMemoryAlert(); goto done;}
	}
	
	if(!wfp) goto done; // programmer error, this should not happen
	
	if(bCloseCurrentFile && this->fCurrentFile) 
		this->fCurrentFile->CloseFile();
	
	if(firstTimeFile)
	{
		wfp->OpenFile();
		// check expiration date
		if (err = wfp->CheckExpirationDate()) {err = -1; goto done;}
		model->SetDialogVariables(DefaultModelSettingsForWizard());
		wind = model->GetWindMover(true); // create one if it does not exist
		if(wind) err = wind->MakeClone((TClassID**)&windClone);
		if(wind) wind->ClearWindValues();// force them to re-enter the winds, 12/18/98
		err = wfp->DoCommandBlock("[BEFORE]"); // pre-dialogs
	}

	if (!firstTimeFile && model->GetModelMode() >= ADVANCEDMODE)
	{
		wind = model->GetWindMover(false); // should already exist, get before closing file or it will be deleted, 3/29/00
		if(wind) err = wind->MakeClone((TClassID**)&windClone); // in case they back out of dialogs
	}

	if(wizSaveFilePath[0]) err = wfp->SaveFileGoThoughDialogs(saveFileContentsHdl);
	else err = wfp->GoThroughDialogs(&userCancel,firstTimeFile);
	
	if(err || userCancel) 
	{
		// reset the model variables
		model->SetDialogVariables(savedSettings);
		if(wind) wind->BecomeClone(windClone);// reset the wind
		if(wfp != this->fCurrentFile) delete wfp; // only delete it if it is different
		if(this->fCurrentFile && bCloseCurrentFile)  
			err = this->fCurrentFile->OpenFile();// we need to reopen our wizard file
		goto done;
	}
	
	// they did not cancel, so we don't need the saved clones
	if(windClone) {windClone->Dispose(); delete windClone; windClone = nil;}
	
	savedSettings = model->GetDialogVariables();	// already gone through dialogs ? need to keep info for SAV file option
	
	// record globals
	if(wfp != this->fCurrentFile && this->fCurrentFile) delete this->fCurrentFile; // only delete it if it is different
	this->fCurrentFile = wfp;
	
	// Clear the left hand side answers so that they are not duplicated
	this->fCurrentFile->ClearLeftHandSideAnswers();

	// read in the save file only the first time so we don't replace any entered spills
	if(firstTimeFile || model->GetModelMode() >= ADVANCEDMODE)
	{
		CMyList* saveLESets=0;

		// read in the save file
		// get the savefile handle
		
		// before we have the model reset, we need to
		// remove ourselves so we don't get deleted
		LocaleWizard *saveWizard = model->fWizard;
		model->fWizard = nil;
		
		// save spills - save handle of LELists, put nil in handle's place 
		if (!firstTimeFile)
		{
			saveLESets = model->LESetsList;
			model->LESetsList = 0;
		}
		
		savedSettings = model->GetDialogVariables();
		
		// save the user entered wind data using clones
		if(wind) err = wind->MakeClone((TClassID**)&windClone);
		
		// if loaded a SAV file don't want to delete
		CloseSaveFile(false,false);// don't ask if they want to save file
		
		// now put the stuff back
		if(saveWizard && model->fWizard) 
		{
			model->fWizard->Dispose(); delete model->fWizard; model->fWizard = nil;
			model->fWizard = saveWizard; 
		}
		
		// restore spills  
		if (saveLESets)
		{
			if(model->LESetsList) // this is the new one created in Init()
				{model->LESetsList->Dispose(); delete model->LESetsList; model->LESetsList =0;}
			
			model->LESetsList = saveLESets;
		}

		model->SetDialogVariables(savedSettings);
		model->Reset(); // to force run time to startTime
		
		// create Diffusion mover
		(void)model->GetDiffusionMover(true);// create one if it does not exist
		
		// create/get the new wind mover
		wind = model->GetWindMover(true); // create one if it does not exist
		
		saveMoverMap = wind->GetMoverMap(); // save and restore map pointer
		if(wind) wind->BecomeClone(windClone);
		wind->SetMoverMap(saveMoverMap);
		
		// this new call replaces the savefile
		// and is basically a command file
		DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage("Setting up the location.");// it could take a while to create the maps and movers
		err = this->fCurrentFile->DoCommandBlock("[AFTER]");// post-dialogs
		// code goes here, special code to set uncertainty in currents
		if(saveFileContentsHdl) this->fCurrentFile->SaveFileAddSpills(saveFileContentsHdl);
		
	}
	
	// Send Messages to the model objects
	this->fCurrentFile->DoWizardWmsgCommands(0);
	// need to save model time if using a SAV file
		model->SetDialogVariables(savedSettings);
	
	if(firstTimeFile ) // set the cursor to the spill tool the first time through a wizard file
		SetTool(SPILLTOOL);
	
done:
	gInWizard = FALSE;
	DisplayMessage(0);
	if(saveFileContentsHdl) {DisposeHandle((Handle)saveFileContentsHdl); saveFileContentsHdl = 0;}
	if(windClone) {windClone->Dispose(); delete windClone; windClone = nil;}

}


void LocaleWizard::GetPhraseFromLine(long phraseNum1Relative,char* line, char* answerStr,long maxNumChars)
{	// returns the part between the ';' chars
	 WizardGetPhraseFromLine(phraseNum1Relative,line,answerStr,maxNumChars);
}

void LocaleWizard::GetParameterString(char* key,char* line, char* answerStr,long maxNumChars)
{
	WizardGetParameterString(key,line,answerStr,maxNumChars);
}

void LocaleWizard::GetParameterString(char* key,long lineNum1Relative,char* str, char* answerStr,long maxNumChars)
{
	WizardGetParameterFromStringLine(key,lineNum1Relative,str,answerStr,maxNumChars);
}

void LocaleWizard::GetParameterString(char* key,long lineNum1Relative,CHARH paramListHdl, char* answerStr,long maxNumChars)
{
	WizardGetParameterFromHdlLine(key,lineNum1Relative,paramListHdl,answerStr,maxNumChars);
}

/////////////////////////////////////////////////

void GetOurAppPathName(char* path)
{	
#ifndef IBM 
	{
#if MACB4CARBON
		short vRef;
		//getvol(nil, &vRef);
		vRef = TATvRefNum;
		PathNameFromWD(vRef, path);
		mypstrcat(path, (char*)LMGetCurApName());
		my_p2cstr((StringPtr)path);
#else
		strcpy(path,"");//clear it
		if (gApplicationFolder[0]) strcpy(path,gApplicationFolder);
		else
		{
			PathNameFromDirID(TATdirID,TATvRefNum,path);
			mypstrcat(path, (char*)LMGetCurApName());
			my_p2cstr((StringPtr)path);
		}
#endif
	}
#else 
	{
		/// IBM code
		strcpy(path, _pgmptr);
		RemoveTrailingSpaces(path);
	}
#endif
}

Boolean IsClassicAbsolutePath(char* path)
{
	// classic paths use ':' delimiters and full paths start with drive name
	if (IsClassicPath(path) && path[0]!=':' && path[0]!='.') return true;
	return false;
}

Boolean IsUnixAbsolutePath(char* path)
{
	if (path[0]=='/') return true;
	return false;
}

Boolean IsWindowsAbsolutePath(char* path)
{
	// check for mapped drive 
	if (path[1]==':' && path[2]=='\\') return true;
	// check for unmapped drive
	if (path[1]=='\\' && path[2]=='\\') return true;
	// at some point switch the leading \ to be full path rather than partial, have to figure out the drive
	return false;
}

Boolean IsWindowsPath(char* path)
{
	long i,len;
	
	//If leads with a {drive letter}:\ it's a full path, this is covered below
	// unmapped drive \\, also covered below
	if (IsWindowsAbsolutePath(path)) return true;

	// if has '\' anywhere in path it's Windows (though Mac allows '\' in filenames, for now assuming it's a delimiter)
	len = strlen(path);
	for(i = 0; i < len  && path[i]; i++)
	{
		if(path[i] == '\\')
			return true;
	}
	
	return false;	// is filename only true or false...
}

Boolean IsUnixPath(char* path)
{
	long i,len;
	
	//If leads with a '/' it's a full path, this is covered below
	//if (IsUnixAbsolutePath(path) return true;

	// if has '/' anywhere in path it's unix (though Mac allows '/' in filenames, for now assuming it's a delimiter)
	len = strlen(path);
	for(i = 0; i < len  && path[i]; i++)
	{
		if(path[i] == '/')
			return true;
	}
	
	return false;	// is filename only true or false...
}

Boolean IsClassicPath(char* path)
{
	long i,len;
	if (IsWindowsAbsolutePath(path))
		return false;
	// if has ':' anywhere in path it's classic (Windows and Mac don't allow ':' in filenames)
	len = strlen(path);
	for(i = 0; i < len  && path[i]; i++)
	{
		if(path[i] == ':')
			return true;
	}
	
	return false;	// is filename only true or false...
}

Boolean ConvertIfUnixPath(char* path, char* classicPath)
{
	OSErr err = 0;
	
	if (IsWindowsPath(path)) return false;
	if (IsClassicPath(path)) return false;
	 
#ifdef MAC
	if (IsUnixAbsolutePath(path)) 
	{
		err = ConvertUnixPathToTraditionalPath((const char *)path, classicPath, kMaxNameLen); 
		return true;
	}
#endif	
	if (IsUnixPath(path)) 	// partial path
	{
		StringSubstitute(path,'/',':'); 
		if (path[0]==':' || path[0]=='.') strcpy(classicPath,path);
		else {strcpy(classicPath,":"); strcat(classicPath,path);}	// unix partial paths shouldn't start with '/'
		return true;
	}
	// assume if doesn't have any file delimiters it's a filename, for now it needs a leading delimiter
	// may want a separate way to handle this case
	strcpy(classicPath,":");
	strcat(classicPath,path);
	return true;

	//return false;
}

Boolean IsFullPath(char* path)
{
	if (IsWindowsAbsolutePath(path)) return true;
	if (IsClassicAbsolutePath(path)) return true;
	if (IsUnixAbsolutePath(path)) return true;
	return false;
}

Boolean IsPartialPath(char* relativePath)
{
	// To simplify scripting, we allow the relative path to use either 
	// platform delimiter.
	char macDelimiter = ':';
	char ibmDelimiter = '\\';
	char unixDelimiter = '/';	// handle ./, ../, etc
	char unixDirectoryUp = '.';	// handle ./, ../, etc
	char delimiter = DIRDELIMITER;
	char otherDelimiter;
	Boolean isRelativePath;
	
	if (IsFullPath(relativePath)) return false;
	
	if(delimiter == macDelimiter)
		otherDelimiter = ibmDelimiter;
	else
		otherDelimiter = macDelimiter;

	
	isRelativePath = 	(relativePath[0] == macDelimiter || relativePath[0] == ibmDelimiter || relativePath[0] == unixDirectoryUp);

	return(isRelativePath);
}

void ResolvePartialPathFromThisFolderPath(char* relativePath,char * thisFolderPath)
{
	// To simplify scripting, we allow the relative path to use either 
	// platform delimiter.
	char macDelimiter = ':';
	char ibmDelimiter = '\\';
	char unixDelimiter = '/';
	char unixDirectoryUp = '.';	// handle ./, ../, etc
	char delimiter = DIRDELIMITER;
	char otherDelimiter;
	long numChops = 0;
	long len,i;
	char* p;
	Boolean isRelativePath;
	char fullFolderPath[256];
	
	
	if(!IsPartialPath(relativePath))
		return; 
		
	strcpy(fullFolderPath,thisFolderPath);

	if(delimiter == macDelimiter)
		otherDelimiter = ibmDelimiter;
	else
		otherDelimiter = macDelimiter;

	// substitute to the appropriate delimiter
	// be careful of the IBM  "C:\" type stings
	
	len = strlen(relativePath);
	for(i = 0; i < len  && relativePath[i]; i++)
	{
		if(relativePath[i] == otherDelimiter && relativePath[i+1] != delimiter)
			relativePath[i] = delimiter;
	}
	
	if (relativePath[0]==unixDirectoryUp)
	{
		// count the number of directories to chop (# delimiters - 1)
		for(i = 1; i < len  && relativePath[i] == unixDirectoryUp; i++)
		{
			numChops++;
		}
	}
	else
	{
		// count the number of directories to chop (# delimiters - 1), old style
		for(i = 1; i < len  && relativePath[i] == delimiter; i++)
		{
			numChops++;
		}
	}
	
	// to be nice, we will be flexible about whether or not fullFolderPath ends in a DIRDELIMITER.
	// so chop the delimiter if there is one
	len = strlen(fullFolderPath);
	if(len > 0 && fullFolderPath[len-1] == DIRDELIMITER)
		fullFolderPath[len-1] = 0;// chop this delimiter
	
	for(i = 0; i < numChops; i++)
	{
		// chop the support files directory, i.e. go up one directory
		p = strrchr(fullFolderPath,DIRDELIMITER);
		if(p) *p = 0;
	}
	// add the relative part 
	if (relativePath[0]==unixDirectoryUp)
		strcat(fullFolderPath,relativePath + numChops + 1);
	else
	{
		if (relativePath[0]!=delimiter)	// allow for filenames
			AddDelimiterAtEndIfNeeded(fullFolderPath);
		strcat(fullFolderPath,relativePath + numChops);
	}
	
	// finally copy the path back into the input variable 
	strcpy(relativePath,fullFolderPath);
}

void ResolvePathFromApplication(char* relativePath)
{
	// To simplify scripting, we allow the relative path to use either 
	// platform delimiter.
	char* p;
	char applicationFolderPath[256] ="";
	char possiblePath[256] ="";
	
	
	// Note: if it is a path that starts with "Resnum", don't do anything
	if(!strncmpnocase(relativePath,"RESNUM",strlen("RESNUM")))
		return; // special case of path being used to indicate a Wizard resource number

	if(!IsPartialPath(relativePath))
		return; 
		
	PathNameFromDirID(TATdirID,TATvRefNum,applicationFolderPath);
	my_p2cstr((StringPtr)applicationFolderPath);
	strcpy(possiblePath,relativePath);
	ResolvePartialPathFromThisFolderPath(possiblePath,applicationFolderPath);
	strcpy(relativePath,possiblePath);

}

void ResolvePathFromInputFile(char *pathOfTheInputFile, char* pathToResolve) // JLM 6/8/10
{
	// Chris has asked that the input files can use a relative path from the input file.
	// Previously he was forced to use absolute paths. 
	// So now, sometimes the path is saved in an input file will just be a file name, 
	// and sometimes it will have been an absolute path, but the absolute path may have been broken when the file is moved.
	// Often the referenced files are in a folder with the input file and it is just that the folder has been moved.
	// This function helps look for these referenced files and changes the input parameter pathToResolveFromInputFile
	// if it can find the file.
	// Otherwise pathToResolveFromInputFile will be unchanged.
	char pathToTry[2*kMaxNameLen] = "",pathToTest[kMaxNameLen], classicPath[kMaxNameLen];
	char pathToTearApart[kMaxNameLen] = "";
	char delimiter = DIRDELIMITER;
	char directoryOfSaveFile[kMaxNameLen];
	char *p,*q;
	int i,numDelimiters;

	if(pathOfTheInputFile == NULL)
		return;

	if(pathOfTheInputFile[0] == NULL)
		return;

	if(pathToResolve == NULL)
		return;

	if(pathToResolve[0] == NULL)
		return;

	// Note: if it is a path that starts with "Resnum", don't do anything
	if(!strncmpnocase(pathToResolve,"RESNUM",strlen("RESNUM")))
		return; // special case of path being used to indicate a Wizard resourse number

	RemoveLeadingAndTrailingWhiteSpace(pathToResolve);

	// need to check for unix paths and convert	
	if (ConvertIfUnixPath(pathToResolve, classicPath)) strcpy(pathToResolve,classicPath);

	if (IsFullPath(pathToResolve))
	{
		if(FileExists(0,0,pathToResolve)) {
			// no problem, the file exists at the path given
			//strcpy(pathToResolve,pathToTest);
			return;
		}
	}
	
	/*if(IsPartialPath(pathToResolve)) {
		ResolvePathFromApplication(pathToResolve); // this resolves the partial path relative to the application
	}

	if(FileExists(0,0,pathToResolve)) {
		// no problem, the file exists at the path given
		return;
	}*/

	// otherwise we have to try to find it
	//////////////////////////////

	///////////////

	// get the directory of the save file
	strcpy(directoryOfSaveFile,pathOfTheInputFile);
	p = strrchr(directoryOfSaveFile,DIRDELIMITER);
	if(p) *(p+1) = 0; // chop off the file name, leave the delimiter

	// First try to resolve relative path to the SaveFile (or whatever has been designated)
	strcpy(pathToTest,pathToResolve);
	if(IsPartialPath(pathToTest)) {
		ResolvePartialPathFromThisFolderPath(pathToTest,directoryOfSaveFile);
	}

	if(FileExists(0,0,pathToTest)) {
		// no problem, the file exists at the path given
		strcpy(pathToResolve,pathToTest);
		return;
	}

	// this one probably isn't necessary...
	strcpy(pathToTest,pathToResolve);
	if(IsPartialPath(pathToTest)) {
		ResolvePathFromApplication(pathToTest); // this resolves the partial path relative to the application 
	}

	if(FileExists(0,0,pathToTest)) {
		// no problem, the file exists at the path given
		strcpy(pathToResolve,pathToTest);
		return;
	}

	// typically the files are either in directoryOfSaveFile or down one level, but we will try any number of levels
	q = pathToResolve;
	for(;;) { // forever
		// find the next delimiter from left to right
		// and append that path onto the directoryOfSaveFile
		strcpy(pathToTry,directoryOfSaveFile);
		strcat(pathToTry,q);
		if(strlen(pathToTry) < kMaxNameLen) { // don't try paths that we know are too long for Windows and Mac pascal strings
			if(FileExists(0,0,pathToTry)) {
				// we found the file
				strcpy(pathToResolve,pathToTry);
				return;
			}
		}
		// find the next part of the path to try
		p = strchr(q,DIRDELIMITER);
		if(p == 0){
			break;// no more delimiters
		}
		//
		q = p+1; // the char after the delimiter
	}	
	return;	// file not found - may want to return a Boolean or error
}

OSErr ResolvePathFromCommandFile(char* pathToResolve) // JLM 6/8/10
{
	char *p, directoryOfCommandFile[kMaxNameLen], resolvedFilePath[kMaxNameLen], commandFileName[64];
	long len, numChops = 0, numPrefixes = 0, i; 
	char macDelimiter = ':';
	char ibmDelimiter = '\\';
	char unixDelimiter = '/';
	char unixDirectoryUp = '.';	// handle ./, ../, etc
	char delimiter = DIRDELIMITER;
	char otherDelimiter;
	OSErr err = 0;
	
	if(gCommandFilePath[0])
		strcpy(directoryOfCommandFile,gCommandFilePath);
	else 
		return -1;


	if(!IsPartialPath(pathToResolve))
		return noErr; 
		
	if(delimiter == macDelimiter)
		otherDelimiter = ibmDelimiter;
	else
		otherDelimiter = macDelimiter;

	// substitute to the appropriate delimiter
	// be careful of the IBM  "C:\" type stings
	
	len = strlen(pathToResolve);
	for(i = 0; i < len  && pathToResolve[i]; i++)
	{
		if(pathToResolve[i] == otherDelimiter && pathToResolve[i+1] != delimiter)
			pathToResolve[i] = delimiter;
	}
	
	for(i = 0; i < len  && pathToResolve[i] == unixDirectoryUp; i++)	// first one is current directory
	{
		numPrefixes++;
	}
	for(i = 1; i < len  && pathToResolve[i] == unixDirectoryUp; i++)	// first one is current directory
	{
		numChops++;
	}
	
	// chop off the file name, and delimiter
	p = strrchr(directoryOfCommandFile,DIRDELIMITER); 
	if(p) *(p) = 0;
	
	for(i = 0; i < numChops; i++)
	{	// chop the support files directory, i.e. go up one directory
		p = strrchr(directoryOfCommandFile,DIRDELIMITER); 
		if(p) *(p) = 0;
	}
	// check for '.' at start of path, chop directoryOfCommandFile accordingly
	strcat(directoryOfCommandFile,pathToResolve+numPrefixes);
	
	strcpy(pathToResolve,directoryOfCommandFile);
	
	return err;
}

void ResolvePath(char* pathToResolve) // JLM 6/9/10
{
	char basePath[kMaxNameLen], classicPath[kMaxNameLen], pathToTest[kMaxNameLen];

	if(pathToResolve == NULL)
		return;

	if(pathToResolve[0] == NULL)
		return;

	// Note: if it is a path that starts with "Resnum", don't do anything
	if(!strncmpnocase(pathToResolve,"RESNUM",strlen("RESNUM")))
		return; // special case of path being used to indicate a Wizard resource number

	RemoveLeadingAndTrailingWhiteSpace(pathToResolve);

	// need to check for unix paths and convert	
	if (ConvertIfUnixPath(pathToResolve, classicPath)) strcpy(pathToResolve,classicPath);

	// make sure path is not getting trashed in any of the failed attempts	
	strcpy(pathToTest,pathToResolve);
	
	if (IsFullPath(pathToTest))
	{
		if(FileExists(0,0,pathToTest)) {
			// no problem, the file exists at the path given
			strcpy(pathToResolve,pathToTest);
			return;
		}
	}
	
	if(gCommandFilePath[0])
	{
		if(IsPartialPath(pathToTest)) {
			char *p, directoryOfCommandFile[kMaxNameLen];
			
			strcpy(directoryOfCommandFile,gCommandFilePath);
			// chop off the file name, and delimiter
			p = strrchr(directoryOfCommandFile,DIRDELIMITER); 
			if(p) *(p) = 0;	
			ResolvePartialPathFromThisFolderPath(pathToTest,directoryOfCommandFile);
		}
		else
			ResolvePathFromInputFile(gCommandFilePath,pathToTest);
	}

	if(FileExists(0,0,pathToTest)) {
		// no problem, the file exists at the path given
		strcpy(pathToResolve,pathToTest);
		return;
	}

	strcpy(pathToTest,pathToResolve);

	if(IsPartialPath(pathToTest)) {
		ResolvePathFromApplication(pathToTest); // this resolves the partial path relative to the application (actually working directory for Windows...)
	}

	if(FileExists(0,0,pathToTest)) {
		// no problem, the file exists at the path given
		strcpy(pathToResolve,pathToTest);
		return;
	}

	/////////////
	// OK... so try various places
	// it does not hurt to try everywhere
	strcpy(pathToTest,pathToResolve);
	ResolvePathFromSavedFile(pathToTest);
	if(FileExists(0,0,pathToTest)) {
		// no problem, the file exists at the path given
		strcpy(pathToResolve,pathToTest);
		return;
	}

	if(gCurrentWizardFile) {
		gCurrentWizardFile->GetPathName(basePath);
		ResolvePathFromInputFile(basePath,pathToResolve);
		if(FileExists(0,0,pathToResolve)) return;
	}

	// try resolving from the application
	GetOurAppPathName(basePath);
	ResolvePathFromInputFile(basePath,pathToResolve);
	if(FileExists(0,0,pathToResolve)) return;

	// try resolving from the working directory - I don't think we are about this except for the command file on the command line
	if(gWorkingDirectory[0])
	{
		ResolvePathFromInputFile(gWorkingDirectory,pathToResolve);
		if(FileExists(0,0,pathToResolve)) return;
	}
	// may want to return a Boolean or an error here

}


Boolean IsCommandFile(char *path)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	strLine [256];
	char	firstPartOfFile [256];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		char key[] = "[GNOME COMMAND FILE]";
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 256);
		if (!strncmpnocase (strLine, key, strlen(key))) 
			bIsValid = true;
	}
	
	return bIsValid;
}

Boolean	gSuppressDrawing = FALSE;
Boolean	gCommandFileRun = FALSE;
char gCommandFileErrorLogPath[256]; 
static Boolean sStopCommandExecution = FALSE;

OSErr WriteErrorToCommandFileErrorLog(char *msg, long dialogID)
{
	// returns noErr only if the error log file is specified and the 
	// message is successfully written
	OSErr err = 0;
	char str[64] ="";
	long offset;
	
	// ifdef NO_GUI should require an error log file
	if(!gCommandFileErrorLogPath[0])
		err = -1; // no error log file specifed
	else
	{
		/*if(!FileExists(0,0,gCommandFileErrorLogPath)) {
			//err = WriteFileContents(0, 0,gCommandFileErrorLogPath, 'ttxt', 'TEXT',"", 0,0);
			//err = WriteFileContents(0, 0,gCommandFileErrorLogPath, 'ttxt', 'TEXT',0, 0,0);	// code warrior doesn't like the ""
			//Mac is getting an errror trying to write nothing to the file - move below
		}*/
			
		// add blank lines and say whether it is an error, note or warning
		strcpy(str,NEWLINESTRING);
		strcat(str,NEWLINESTRING);
		strcat(str,NEWLINESTRING);
		switch(dialogID)
		{
			case  8000: 
				strcat(str,"*** ERROR ***");
				sStopCommandExecution = TRUE;
				break;
			case  8001: strcat(str,"*** NOTE ***"); 
				break;
			case  8002: 
				strcat(str,"*** WARNING ***"); 
				break;
			default: 
				strcat(str,"*** MESSAGE ***"); 
				break;
		}
		strcat(str,NEWLINESTRING);

		if(!FileExists(0,0,gCommandFileErrorLogPath)) {
			//err = WriteFileContents(0, 0,gCommandFileErrorLogPath, 'ttxt', 'TEXT',"", 0,0);
			err = WriteFileContents(0, 0,gCommandFileErrorLogPath, 'ttxt', 'TEXT', str, strlen(str), 0);	// code warrior doesn't like the ""
		}
		else	
		//if(!err)
			err = AddToFileContents(0, 0, gCommandFileErrorLogPath, 'ttxt', 'TEXT', str, strlen(str), 0, &offset);
		if(!err)
			err = AddToFileContents(0, 0, gCommandFileErrorLogPath, 'ttxt', 'TEXT', msg, strlen(msg), 0, &offset);
			
		if(err)  {
			SysBeep(5); // just to make sure the user knows we are upset
			SysBeep(5);
		}
	}
	return err; 

}



OSErr DoCommandFile(char *path)
{
	// COMMAND file commands and WIZARD commands are overlapping but distinct sets of commands
	// COMMAND files primarily contain "MESSAGE" lines 
	// the first line is "[GNOME COMMAND FILE]"
	
	OSErr err = noErr;
	char line[512];
	char messageTypeStr[64];
	long lineStart;
	Boolean doCommand = false;
	long messageCode;
	char targetName[kMaxNameLen];
	
	CHARH f = 0;

	if (!path || !path[0]) return -1;
	
	#ifndef NO_GUI 
	SetWatchCursor();
	#endif

	sprintf(line, "Loading command file: %s...", path);
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(line);

	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("DoCommandFile()", "ReadFileContents()", err);
		goto done;
	}
	_HLock((Handle)f); // JLM 8/4/99

	strcpy(gCommandFilePath,path); // so it is available for ResolvePath()
	
	sprintf(line, "Executing command file: %s...", path);
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(line);
	
	// set the model to be in advanced mode and NOAA version (to be nice to the TAP power users)
	gNoaaVersion = true; // must be a noaa user
	if (model -> GetModelMode() != ADVANCEDMODE) 
		model -> SetModelMode (ADVANCEDMODE);	// must be in advanced mode

	if(f)
	{
		// since we are looping through the lines
		// use the more efficient GetLineFromHdlHelper
		Boolean gotLine;
		char key[] = "[GNOME COMMAND FILE]";
		lineStart = 0;
		gotLine = GetLineFromHdlHelper(1,lineStart,&lineStart,f,line,512,true);
		if(strncmpnocase (line, key, strlen(key)))
			err = true; // the first line should be ""
			
		sStopCommandExecution = FALSE;
		gSuppressDrawing = TRUE; 
		gCommandFileRun = TRUE;
		
		while(!err && !sStopCommandExecution)
		{
			gotLine = GetLineFromHdlHelper(1,lineStart,&lineStart,f,line,512,true);
			if(!gotLine) break;// no more lines
			if(!line[0]) continue;// blank line
			if(line[0] == '-' || line[0] == '#' || line[0] == '/') continue; // comment line
			//if (!gSuppressDrawing)
			MySpinCursor(); 
			WizardGetParameterString("MESSAGE",line,messageTypeStr,64);
			messageCode = WizStringToMessageCode(messageTypeStr);
			if(messageCode > 0)
			{
				WizardGetParameterString("TO",line,targetName,64);
				err = model->BroadcastMessage(messageCode,targetName,line,nil);
				if (err==-7) sStopCommandExecution = TRUE;	// special TAP error to stop execution on command period
			}
			else
			{
				/// unrecognized command
				ParsingErrorAlert("Unrecognized command line:",line);
				err = true;
			}
		}
		
		if(sStopCommandExecution)
		{
			printNote("Execution of Command File Terminated");
			sStopCommandExecution = FALSE;
		}

	}
done:
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}
	
	// restore the basic model operating behavior
	// in case the command file invoked special features
	gSuppressDrawing = FALSE; // set the model to draw as normal
	gCommandFileErrorLogPath[0] = 0;// set the model to no longer override the normal error message system
	gCommandFileRun = FALSE;

	gCommandFilePath[0] = 0;

#ifndef NO_GUI
	model->NewDirtNotification();// cause a redraw etc...
	InitCursor();
#endif

	return err;	

}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
//////////////////////////////////////////////////



/////////////////////////////////////////////////
/////////////////////////////////////////////////


