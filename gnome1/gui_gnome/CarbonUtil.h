


#ifdef MAC

#ifndef __CARBONUTIL__
#define __CARBONUTIL__
/////////////////////////////////////////////////
/////////////////////////////////////////////////
extern OSType gMySignature ; // needs to be defined in the application


#ifdef	MPW
	#define TRUE 1
	#define FALSE 0
	#define  SetPortDialogPort(x)  SetPortWindowPort(GetDialogWindow(x))
	#define GetPortBitMapForCopyBits(x)  (&(((GrafPtr)x)->portBits))
	#define GetDialogFromWindow(x)  (x)
	#define GetPortPixMap(x) ((x)->portPixMap)
	#define GetWindowFromPort(x)  ((WindowRef)(x))
	#define GetMenuWidth(x)  ((**x).menuWidth)
	/////
#else
	// //in CodeWarrior, define the old names  so we can minimize changes to the code
#ifdef CODEWARRIOR
	#define GetPortPixMap(x) ((x)->portPixMap)
	#define GetPortBitMapForCopyBits(x)  (&(((GrafPtr)x)->portBits))
	#define GetWindowFromPort(x)  ((WindowRef)(x))
	#define GetMenuWidth(x)  ((**x).menuWidth)
//#endif
	#define SetCtlValue(theControl, theValue) SetControlValue(theControl, theValue)
	#define SetCtlMax(theControl, maxValue) SetControlMaximum(theControl, maxValue)
	#define GetCtlMax(theControl) GetControlMaximum(theControl)
	#define SetCtlMin(theControl, minValue) SetControlMinimum(theControl, minValue)
	#define GetCtlMin(theControl) GetControlMinimum(theControl)
	#define SetCTitle(theControl, title) SetControlTitle(theControl, title)
	#define GetCTitle(theControl, title) GetControlTitle(theControl, title)
	#define SetCRefCon(theControl, data) SetControlReference(theControl, data)
	#define GetCtlValue(theControl) GetControlValue(theControl)
	#define GetCRefCon(theControl) GetControlReference(theControl)
	//
	#define GetIText(item, text) GetDialogItemText(item, text)
	#define ShowDItem(theDialog, itemNo) ShowDialogItem(theDialog, itemNo)
	#define HideDItem(theDialog, itemNo) HideDialogItem(theDialog, itemNo)
	#define DelMenuItem(theMenu, item) DeleteMenuItem(theMenu, item)
	#define TextBox(text, length, box, just) TETextBox(text, length, box, just)
	//
	//#define EnableItem EnableMenuItem
	//#define DisableItem DisableMenuItem
	#define CheckItem CheckMenuItem
#endif
#endif
#if TARGET_API_MAC_CARBON
	#define EnableItem EnableMenuItem
	#define DisableItem DisableMenuItem
#endif

#define GRAY_BRUSH  	MyGetQDGlobalsGray()  
#define BLACK_BRUSH  	MyGetQDGlobalsBlack()
#define LTGRAY_BRUSH  MyGetQDGlobalsLightGray()  
#define DKGRAY_BRUSH  MyGetQDGlobalsDarkGray()
#define WHITE_BRUSH   MyGetQDGlobalsWhite()


Pattern MyGetQDGlobalsGray(void);  
Pattern MyGetQDGlobalsBlack(void);
Pattern MyGetQDGlobalsLightGray(void);  
Pattern MyGetQDGlobalsDarkGray(void);
Pattern MyGetQDGlobalsWhite(void);

void FillRectWithQDGlobalsGray(Rect * rect);

void PenPatQDGlobalsGray(void);
void PenPatQDGlobalsDarkGray(void);
void PenPatQDGlobalsBlack(void);
void PenPatQDGlobalsWhite(void);

void MyLUpdateVisRgn(DialogRef theDialog, ListHandle listHdl);

#if TARGET_API_MAC_CARBON
	void UpperText(char* textPtr, short len);
#endif


OSErr MyGetVInfo(short drvNum, StringPtr volNamePtr, short *vRefNumPtr, double *freeBytesPtr);

#if TARGET_API_MAC_CARBON
	// my versions of the old style lower case functions
	void drawstring(const char *s);
	short stringwidth(char *s);
	void getindstring(char *theString, short strListID, short index);
	void numtostring(long theNum, char *theString);
	void stringtonum(char *theString, long *theNum);
	Handle getnamedresource(ResType theType, const char *name);
	void  setwtitle(WindowRef window, char* title);
	void getwtitle(WindowRef window, char* cStr);
	void paramtext(char* p0,char* p1,char* p2,char* p3);
	void getitem(MenuRef menu, short item, char *itemString);
	void appendmenu(MenuRef menu, const char *data);
	void insmenuitem(MenuRef theMenu, const char *itemString, short afterItem); // insertmenuitem
	void getfontname(short familyID, char *theName);
	void getfnum(char *name, short *num);
#endif

/////////////////////////////////////////////////
#ifdef MAC
	#if TARGET_API_MAC_CARBON
		extern PMPrintSession  gPrintSession;
		extern PMPageFormat  gPageFormat;
		extern PMPrintSettings gPrintSettings;
		extern Boolean gSessionDocumentIsOpen;
	#else
		extern THPrint gPrRecHdl;
	#endif
#endif

OSStatus  OpenPrinterAndValidate(void);
OSStatus DoJobPrintDialog(char * jobNameStr); 
void ClosePrinter(void);
void MyPMSessionSetError(OSStatus err);

#if TARGET_API_MAC_CARBON
	OSStatus My_PMSessionBeginDocument(void);
	OSStatus My_PMSessionEndDocument(void);
	Rect GetPrinterPageRect(void);
#endif

long MyPutScrap(long length, ResType theType, void *source);
long MyGetScrapLength(ResType theType);
long MyGetScrap(Handle hDest, ResType theType);
long MyZeroScrap(void);

#if TARGET_API_MAC_CARBON
	typedef struct {
		short ditlResID ;
		short customHeight;
		short customWidth; 
		void (*initProc)(DialogRef);
		void (*clickProc)(DialogRef, short);
	} MyCustomNavItemsData;  

	Boolean AskUserForPutFileName(char* promptStr, char* defaultName, char* pathOfFileSelected, short maxPathLength, FSSpec *specPtr, MyCustomNavItemsData *myCustomItemsDataPtr);
	Boolean AskUserForGetFileName(char* prompt_string, short numTypes, OSType typeList[], char* pathOfFileSelected, short maxPathLength, FSSpec *specPtr, MyCustomNavItemsData *myCustomItemsDataPtr);
#endif

void SetDefaultItemBehavior(DialogRef dialog);
void MySelectDialogItemText(DialogRef theDialog, short editTextitemNum, short strtSel, short endSel);

FILE *my_fopen (char *filePath,char *mode);

/////////////////////////////////////////////////
#endif
#endif


