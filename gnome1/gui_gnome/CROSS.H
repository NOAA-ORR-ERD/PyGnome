
/// TYPES ///////////////////////////////////////////////////////////////////////
#ifndef __CROSSH__
#define __CROSSH__

#include "Earl.h"
#include "TypeDefs.h"

//#ifdef MAC
//extern THPrint gPrRecHdl;
//#endif

#ifndef pyGNOME
extern RGBColor colors[];

enum { BLACK = 1, WHITE, DARKGRAY, GRAY, LIGHTGRAY,
	BROWN, LIGHTBROWN, OLIVE = 7, DARKGREEN, GREEN,
	LIGHTBLUE, BLUE, DARKBLUE, PURPLE, PINK, RED, YELLOW, OTHERCOLOR }; // colors

#endif

///// MISC ///////////////////////////////////////////////////////////////////////////

void BasicInit();
unsigned long MyTickCount();
void MyHLock(Handle h);
Handle MySetHandleSize(Handle h, long newSize);
short GetTextFont(void);
char GetDegreeChar(short fontNum);
char GetSmartDoubleQuoteChar(short fontNum);
short MyGetFontNum (char *name);
void MyTextFont (char *fontName);
void SetWatchCursor();
void MySetCursor(short id);
Point GetEventWhere(EventRecord e);
long GetEventWhen(EventRecord e);
Boolean MouseDownEvent();
Point GetMouseLocal(WindowPtr w);
void GlobalToLocalRect (Rect *globalRect);
Boolean WindowObscured(WindowPtr w);
Boolean FloatingWindowAbove(WindowPtr w);
Boolean WindowNeedsUpdating(WindowPtr w);

#ifdef MAC
void InitPrinting();
OSErr DoPrintFunction(short function, Boolean *accepted);
#endif
OSErr PrintFred(Ptr title, long iconNum, Handle textHnd, short printingFont, short printingSize);
OSErr PrintText(char *title, Handle textHnd, short ICONid);
//Boolean FileExists(short vRefNum, long dirID, CHARPTR fileName);
Boolean FileExists(short vRefNum, long dirID, const char *fileName);
Boolean FolderExists(short vRefNum, long dirID, CHARPTR folderName);
OSErr AddFolderIfMissing(long vRefNum, long parentDirID, char *name, long *dirID);

Boolean EqualPoints(Point p1, Point p2);
Boolean EqualRects(Rect r1, Rect r2);
Boolean EqualRGBColors(RGBColor c1, RGBColor c2);
char *	GS1(short id, char *s);
char *	GS2(short id, char *s);
char *	GS3(short id, char *s);
char *ZapGremlins(char *s);

#ifdef MAC
	#if TARGET_RT_LITTLE_ENDIAN
		#define SWAP_BINARY  // swap on the intel MAC
	#endif
#else // IBM
	#define SWAP_BINARY // swap in MS Windows
#endif

unsigned short SwapShort(unsigned short n);
unsigned long SwapLong(unsigned long n);
void SwapFloat(float *f);
void TextSizeSmall();
void TextSizeTiny();

///// HFILES ///////////////////////////////////////////////////////////////////////////

OSErr hgetvol(CHARPTR volName, SHORTPTR vRefNum, LONGPTR dirID);
OSErr hsetvol(CHARPTR volName, short vRefNum, long dirID);
OSErr hopen(short vRefNum, long dirID, CHARPTR fileName, char permission, SHORTPTR refNum);
OSErr hopendf(short vRefNum, long dirID, CHARPTR fileName, char permission, SHORTPTR refNum);
OSErr hopenrf(short vRefNum, long dirID, CHARPTR fileName, char permission, SHORTPTR refNum);
short hopenresfile(short vRefNum, long dirID, CHARPTR fileName, char permission);
OSErr hcreate(short vRefNum, long dirID, CHARPTR fileName, OSType creator, OSType fileType);
void hcreateresfile(short vRefNum, long dirID, CHARPTR fileName);
OSErr dircreate(short vRefNum, long parentDirID, CHARPTR directoryName, LONGPTR createdDirID);
OSErr hdelete(short vRefNum, long dirID, CHARPTR fileName);
OSErr hgetfinfo(short vRefNum, long dirID, CHARPTR fileName, FINFOPTR fndrInfo);
OSErr hsetfinfo(short vRefNum, long dirID, CHARPTR fileName, FINFOPTR fndrInfo);
OSErr hsetflock(short vRefNum, long dirID, CHARPTR fileName);
OSErr hrstflock(short vRefNum, long dirID, CHARPTR fileName);
OSErr hrename(short vRefNum, long dirID, CHARPTR oldName, CHARPTR newName);
OSErr catmove(short vRefNum, long dirID, CHARPTR oldName, long newDirID, CHARPTR newName);

///// GRAPHICS ///////////////////////////////////////////////////////////////////////////

#ifdef MAC
#define GetRValue(c) (c.red / 256)
#define GetGValue(c) (c.green / 256)
#define GetBValue(c) (c.blue / 256)
#endif


Boolean DragRectInRect (WindowPtr theWindow, Point clickPoint, Rect *startRect, Rect *limitRect, Rect *endRect);
RGBColor MyPickColor(RGBColor color, WindowPtr w);
Rect GetScreenBounds();
short PixelsPerInchCurrent();
short PixelsPerInchScreen();
short PixelsPerCentimeter();
double PixelsPerMilDeg();
long PixelDiscrepancy(WorldRect wr);
long LongToPixels(long dLong, WorldPoint center);
Boolean MyPtInRect(Point p, RECTPTR r);
void MyMoveTo(short x, short y);
void MyLineTo(short x, short y);
void MyFrameRect(RECTPTR r);
void MyFillRect(RECTPTR r, Pattern p);
void MyInvertRect(RECTPTR r);
void MyOffsetRect(RECTPTR r, short dh, short dv);
void MyInsetRect(RECTPTR r, short h, short v);
void MyPaintRgn(RgnHandle r);
void MyFillRgn(RgnHandle r, Pattern p);
void MyFrameRgn(RgnHandle r);
RgnHandle MyRectRgn(RgnHandle rgn, RECTPTR r);
void TextAngle(long angle);
void MyUnionRect(Rect *src1, Rect *src2, Rect *dstRect); // NOTE: this has mac parameter ordering
#ifdef IBM
	Boolean EmptyRect(Rect *r);// reproduces mac call
#endif
Rect MyClipRect(Rect r);
void ClearClipRect(void);
void MyShowWindow(WindowPtr w);
void DrawPictureAtPt(short resId, short x,short y,Boolean center);

Rect GetResBitmapRect(short resID);
void DrawResBitmap(short resID, Rect frame, unsigned long mode);
void DrawResBitmap2(short resID, Rect *iconRect, unsigned long mode);
OSErr MyOpenPicture(RECTPTR r, PICTUREHP p);
void MyClosePicture(PICTUREH h);
void MyDrawPicture(PICTUREH h, RECTPTR r);
void MyKillPicture(PICTUREH h);
void ClipRectPlus(Rect r);
void ClipRectMinus(Rect r);
void InvalRectInWindow(Rect r, WindowPtr w);
void ValidRectInWindow(Rect r, WindowPtr w);
void InvalRgnInWindow(RgnHandle rgn, WindowPtr w);
void MyDragGrayRgn(RgnHandle r, Point p, void (*action)(Point),
				   SHORTPTR dh, SHORTPTR dv,
				   RECTPTR rgnBounds, RECTPTR dragBounds);
Rect DefineGrayRect(Point p, void (*action)(RECTPTR, Boolean), Boolean gray,
					Boolean fromCenter, Boolean optionTogglesCenter,
					Boolean ellipse, Boolean optionTogglesEllipse, Boolean escape);

///// KEYS, ETC. ////////////////////////////////////////////////////////////////////////

Boolean KeyDown(short keyBit);
Boolean OptionKeyDown();
Boolean ShiftKeyDown();
Boolean ControlKeyDown();
Boolean KeyEvent(EVENTRECORDPTR e, char key1, char key2);
Boolean CommandKeyEvent(EVENTRECORDPTR e, char key1, char key2);
Boolean CmdPeriodEvent(EVENTRECORDPTR e);
Boolean CmdPeriod();
Boolean CmdPeriodAny();

///// MAC LISTS ////////////////////////////////////////////////////////////////////////

#ifdef MAC
Boolean ListKeyEvent(EVENTRECORDPTR e, ListHandle L);
void MakeListCellVisible(Point cell, ListHandle L);
void DeselectAllCells(ListHandle L);
#endif

///// MENUS ///////////////////////////////////////////////////////////////////////////

Boolean GetAndInsertMenu(short menuID, short special);
void MyEnableMenuItem(MenuHandle m, short item, Boolean enable);
MenuHandle EnsureMenu(short menuID, short position);
void ClearAllMarks(short menuID);
void ClearAllMarks2(MenuHandle m);
short CheckedItem(short menuID);
short CheckedItem2(MenuHandle m);
void UncheckAllItems(short menuID);
void UncheckAllItems2(MenuHandle m);
Boolean ItemChecked(short menuID, short item);
Boolean ItemChecked2(MenuHandle m, short item);
void ToggleMenuCheck(short menuID, short item);
void ToggleMenuCheck2(MenuHandle m, short item);
CHARPTR MyGetMenuTitle(short menuID, CHARPTR title);
CHARPTR MySetMenuTitle(short menuID, CHARPTR title);

///// IAC //////////////////////////////////////////////////////////////////////////////

typedef struct {
			char signature[8];
			char pSig[8];
			char name[32];
			char appPath[256];
			char docPath[256];
			long menuID;
			short running;
			short pendingMenuID;	// if not 0 these represent an
			short pendingMenuItem;	// outstanding MHIT item for the friend
			// reference to .VWR file
			short vRefNum;
			long dirID;
			char fileName[256];
		} Friend, **FRIENDH;

typedef struct {
		char keyWord[5];
		long type;
		Ptr ptr;
		long size;
	} AEdata;

extern char xtraString[];
extern FRIENDH friends;
extern short iacWatchCount;
extern Boolean bIACInited;

#ifdef IBM
typedef Handle AppleEvent;
#define typeChar 1
#define WM_IAC (WM_USER + 1)
#define NE_GET_ALL_MESSAGES 1
#define NE_APP_IS_RUNNING 2
#define NE_TRANSFER_MESSAGE 3
OSErr HandleIACMessage(short wParam, long lParam);
#endif

OSErr ProcessDDEQueue();
long SigToFriendIndex(char *sig);
long PSigToFriendIndex(char *pSig);
Boolean FriendIsRunning(char *pSig);
OSErr InitIAC(char *signature, WindowPtr w);
void CloseIAC(void);
OSErr myAESend(char *message, AEdata dataArray[], char *toSig, Boolean beep, char *fromSig, char *fromPSIG);
OSErr myAEGetParamPtr(AppleEvent *ae, char *keyWord, char *ptr, long maxSize, Boolean beep);
long myAEGetParamPtrSize(AppleEvent *ae, char *keyWord);
OSErr myAEGetParamIn_NewPtr(AppleEvent *ae, char *keyWord, Ptr *ptr, Boolean beep);

///// SYSTEM ///////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////
#ifdef IBM
	#define  GetDialogWindow(x)  (x)
	void SetPortGrafPort(GrafPtr w);
	#define SetPortWindowPort(x) SetPortGrafPort(x)
	#define SetPortDialogPort(x) SetPortGrafPort(x)
#else
	#define SetPortGrafPort(x) SetPort(x)
	#ifdef MPW
		#define  SetPortDialogPort(x)  SetPortWindowPort(GetDialogWindow(x))
	#endif
#endif

/////////////////////////////////////////////////
void TextFontSize(long font,long size); 
void TextFontSizeFace(long font,long size,long style) ; 

Boolean SigToProcessInfo(char *sig, ProcessInfoRec *pInfo, char *name, FSSpec *spec);
Boolean AppIsRunning(char *sig);
Boolean CurrentProcessInBackground();

//#ifdef MAC
//short NumToolboxTraps();
//TrapType GetTrapType(short trap);
//Boolean TrapAvailable(short trap);
//#endif

char *CurrentApplicationName(char *name);
OSErr DeleteHandleItem(CHARH h, long i, long size);
//long MyTempMaxMem();
Boolean TempMemCallsAvailable();
Handle MyNewHandleTemp(long size);
Boolean System7Available();
Boolean ColorQDAvailable();
Boolean WindowHasColor(WindowPtr w);
short NumColorBits();
Rect GetPICTFrame(PicHandle hPICT);
WindowPtr MyGetNewCWindow(short ID, Ptr p, WindowPtr behind);
void MyMoveWindow(WindowPtr w, short h, short v, Boolean bringToFront);
void PlaceWindow(WindowPtr w, Rect r, Boolean redraw);
void GetWindowLocation(WindowPtr w, RECTPTR r);
//Rect GetPortRect(WindowPtr w);
Rect GetWindowPortRect(WindowRef w);
#define  GetDialogPortRect(x)  GetWindowPortRect(GetDialogWindow(x))

Rect SafeWindowBounds(WindowPtr w);
UniversalProcPtr MakeUPP(ProcPtr p, ProcInfoType pInfo);

///// FILES ///////////////////////////////////////////////////////////////////////////

Boolean ScanVolumes(Boolean initialize, Boolean cdOnly, short *changes);
Boolean VolumesChanged(Boolean cdOnly);
Boolean VolumeOnLine(char *volumeName, char *drive, short *vRefNum, Boolean cdOnly);
CHARPTR PathNameFromDirID(long dirID, short vRefNum, CHARPTR pathName);
#if MACB4CARBON
CHARPTR PathNameFromWD(long vRefNum, CHARPTR pathName);
#endif
CHARPTR GetFullPath(short vRefNum, long dirID, CHARPTR fileName, CHARPTR path);
void AddFileSuffix(char *fileName, char *suffix, short vRefNum);
OSErr FindFileFromPath(char *path, char *suffix, char *name, short dialogID, Boolean exeFiles);
CHARPTR MyGetFolderName(short vRefNum, long dirID, Boolean fullPath, CHARPTR folderName);
long GetFolderID(short vRefNum, long dirID, CHARPTR folder);
long GetFolderNumFiles(short vRefNum, long dirID, CHARPTR pathName);
long ParentFolderID(short vRefNum, long dirID, CHARPTR pathName);
OSErr MyGetFileSize(short vRefNum, long dirID, CHARPTR pathName, LONGPTR size);
OSErr GetFileModDate(short vRefNum, long dirID, CHARPTR pathName, LONGPTR seconds);
OSErr GetVolumeName(short vRefNum, char *volName);
OSErr FreeBytesOnVolume(short vRefNum, double* dFreeBytes, CHARPTR volName);
OSErr ForEachItemInFolder(short vRefNum, long dirID, CHARPTR pathName,
						  OSErr (*proc)(CINFOPBRECPTR), Boolean (*filter)(CINFOPBRECPTR));
OSErr DeleteFilesInFolder(short vRefNum, long dirID, CHARPTR pathName);
OSErr DeleteFilesInFolderSpecifiedByFullPathWithDelimiter(char* path);
OSErr CreateFoldersInDirectoryPath(char *path);
#ifdef MAC
Boolean IsBundledApplication(char *path);
void ChopExtensionAndHiddenPackageFolders(char *fullPath);
OSErr ForEachMountedVolume(OSErr (*proc)(HParamBlockRec *));
#else
OSErr ForEachMountedVolume(OSErr (*proc)(char));
#endif
#ifdef __cplusplus
extern "C" {
#endif
int ComparePaths(const void *rec1, const void *rec2);
#ifdef __cplusplus
}
#endif
OSErr QuickSpec(char *path, short *vRefNum, long *dirID);

///// LAUNCHING APPLICATIONS ///////////////////////////////////////////////////////////////////////
#ifdef MAC
OSErr TryToLaunch(char* appPath, Boolean bringToFront, ProcessSerialNumber *launchedProcessSN);
#else
OSErr TryToLaunch(char* appPath, Boolean bringToFront);
#endif


///// DIALOGS ///////////////////////////////////////////////////////////////////////

//void MyTEKey(char key, WindowPtr dialog);
//void EnableTextItem(WindowPtr dialog, short item, Boolean enabled);
//Boolean DisabledStatTextItem(WindowPtr dialog, short item);
void MyTEKey(char key, DialogPtr dialog);
void EnableTextItem(DialogPtr dialog, short item, Boolean enabled);
Boolean DisabledStatTextItem(DialogPtr dialog, short item);
void MyShowHide(WindowPtr w, Boolean show);
Boolean DragDuringDialog(DialogPtr theDialog, EVENTRECORDPTR theEvent);
pascal_ifMac Boolean STDFilter(DialogPtr theDialog, EVENTRECORDPTR theEvent, SHORTPTR itemHit);
void CenterDialog(DialogPtr theDialog, short position);
void CenterWindow(WindowPtr theWindow, short position);
void PushModalGlobals(WindowPtr parent);
void PopModalGlobals();
#ifdef IBM
BOOL CALLBACK STDWinDlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
#endif

Boolean WeHaveAModalDialogUp(void);
short MyModalDialog(short dialogID, WindowPtr frontWindow, VOIDPTR data,
					OSErr (*initProc)(DialogPtr dialog, VOIDPTR data),
					short (*clickProc)(DialogPtr dialog, short itemNum,
									   long lParam, VOIDPTR data));
short MyModalDialogF(short dialogID, WindowPtr frontWindow, VOIDPTR data,
					 OSErr (*initProc)(DialogPtr dialog, VOIDPTR data),
#ifdef MAC
					 pascal_ifMac Boolean (*filterProc)(DialogPtr dialog, EventRecord *e, short *itemNum),
#else
					 DLGPROC   filterProc, 
#endif
					 short (*clickProc)(DialogPtr dialog, short itemNum,
									    long lParam, VOIDPTR data));
Boolean CHOICEALERT(short dialogID, CHARPTR text, Boolean bParent);
short MULTICHOICEALERT(short dialogID, CHARPTR text, Boolean bParent);
short MULTICHOICEALERT2(short dialogID, CHARPTR text, CHARPTR text2, Boolean bParent);
long REQUEST(char *message, char *defaultAnswer, char *answer);
void ActivateParentDialog(Boolean bActivate);
void EraseUserItem(DialogPtr theDialog, short item);
void DrawUserItem(DialogPtr theDialog, short item);
void DoFrameDefault(DialogPtr theDialog, short frameItem);
pascal_ifMac void FrameDefault(DialogPtr theDialog, short frameItem);
void DoFrameEmbossed(Rect r);
pascal_ifMac void FrameEmbossed(DialogPtr theDialog, short frameItem);
void CenterDialogUpLeft(DialogPtr theDialog);
Point CenteredDialogUpLeft(int dialogID);
Boolean ItemHidden(DialogPtr theDialog, short item);
void FillEmptyItems(DialogPtr theDialog, SHORTPTR items, CHARPTR newValue);
void MyEnableControl(DialogPtr dialog, short item, Boolean enable);
Boolean ControlEnabled(DialogPtr dialog, short item);
ControlHandle GetControlItemHandle(DialogPtr dialog, short item);
void MyGetControlTitle(DialogPtr theDialog, short item, CHARPTR title);
void MySetControlTitle(DialogPtr theDialog, short item, CHARPTR title);
void FlashItem(DialogPtr theDialog, short item);
pascal_ifMac void FrameBlack(DialogPtr theDialog, short itemNum);
pascal_ifMac void FrameGray(DialogPtr theDialog, short itemNum);
short GetHelp(CHARPTR title, Boolean bParent);
short GetHelpTopics(char *prevTopic, Boolean bParent);

///// APPLICATION HOOKS //////////////////////////////////////////////////////////////

// It is assumed that the app provides these functions.

extern Boolean DealWithEvent(EventRecord event, Boolean dialog);
extern pascal_ifMac OSErr HandleNOAAEvent(const AppleEvent *ae, AppleEvent *reply, long refCon);
extern void DisplayMessage(char *msg);

///// MACUTIL ///////////////////////////////////////////////////////////////////////
///// MACUTIL ///////////////////////////////////////////////////////////////////////
///// MACUTIL ///////////////////////////////////////////////////////////////////////
///// MACUTIL ///////////////////////////////////////////////////////////////////////
///// MACUTIL ///////////////////////////////////////////////////////////////////////

// in order to link with the Quicktime library, we need to avoid MAC-like names on the IBM
#ifdef MAC
	#define GetPortGrafPtr GetPort
	#define MyPenMode PenMode
	#define MyPenNormal PenNormal
	#define MyInitCursor InitCursor
	#define MyGetPenState GetPenState
	#define MyGetPen GetPen
	#define MySetPenState SetPenState
	#define MyPenSize PenSize
	#define MyMove Move
	#define MyLine Line
	#define MyForeColor ForeColor
	#define MySectRect SectRect
	#define MyPaintRect PaintRect
	#define MyEraseRect EraseRect
	#define MyFrameOval FrameOval
	#define MyPaintOval PaintOval
	#define MyFrameArc FrameArc
	#define MyPaintArc PaintArc
	#define MyNewRgn NewRgn
	#define MyDisposeRgn DisposeRgn
	#define MySetPt SetPt
	#define MyLocalToGlobal LocalToGlobal
	#define MyGlobalToLocal GlobalToLocal
	#define MyMapPt MapPt
	#define MyRGBForeColor RGBForeColor
	#define MyRGBBackColor RGBBackColor
	#define MyGetForeColor GetForeColor
	#define MyGetPicture GetPicture
	#define MyMemError MemError
	#define MyNewHandle NewHandle
	//#define MyNewHandleClear NewHandleClear
	//#define MyRecoverHandle RecoverHandle
	//#define MyNewPtr NewPtr
	//#define MyNewPtrClear NewPtrClear
	//#define MyHLock HLock
	//#define MyHUnlock HUnlock
	//#define MyTempNewHandle TempNewHandle
	//#define MyDisposePtr DisposePtr
	//#define MyGetPtrSize GetPtrSize
	//#define MySetHandleSize SetHandleSize
	//#define MyGetHandleSize GetHandleSize
	#define MyBlockMove BlockMove
	//#define MyHandToHand HandToHand
	//#define MyKillPicture KillPicture
#endif

#ifdef IBM

typedef struct {
		WindowPtr w;
		HDC hDC;
		HBRUSH hBrush;
		HPEN hPen;
		HFONT hFont;
		long pattern;
		long penStyle;
		long penWidth;
		long font;
		long size;
		long style;
		long textMode;
		long textAngle;
		long rgb;
		long clipRgn;
		Point penPos;
	} PortSettings;

typedef unsigned char MonoPatBits[16];

extern HINSTANCE hInst;
extern WindowPtr hMainWnd,toolWnd, currentWindow;
extern HDC currentHDC;
extern PortSettings currentPort;
extern char SFSaveDisk[];
extern Boolean makingMetaFile, trueTypeMode,
			   scalePrintFonts, doCompleteIsWindowCheck;

/////////// MACUTIL: GRAPHICS

void SetPort(WindowPtr w);
void SetPortBP(WindowPtr w, PAINTSTRUCT *ps);
void SetPortEP(WindowPtr w, PAINTSTRUCT *ps);
void GetPort(WindowPtr *w);
void GetPortGrafPtr(WindowPtr *w);
WindowPtr FrontWindow();
void HiliteWindow(WindowPtr w, Boolean hilite);
void SelectWindow(WindowPtr w);
void HideWindow(WindowPtr w);
void SizeWindow(WindowPtr w, short width, short height, Boolean update);
BOOL RegisterStandardWindows(void);
void SetClip(RgnHandle rgn);
void ClipRect(RECTPTR r);
void ValidRect(RECTPTR r);
void GetMouse(POINTPTR p);
Boolean StillDown();
Boolean Button();
void SetMouse(POINTS p);
void SetMouseButton(Boolean button);
void GetPen(POINTPTR p);
void GetPenState(PenState *ps);
void SetPenState(PenState *ps);
void GlobalToLocal(POINTPTR p);
void LocalToGlobal(POINTPTR p);
void PenMode(int drawMode);
void FillPat(long pattern);
short MacToWinStyle(short linePattern, short lineWidth);
void WinToMacStyle(long style, short *linePattern, short *lineWidth);
void PenStyle(long style, long width);
void PenSize(long h, long v);
void PenNormal();
void GetForeColor(RGBColor *c);
void RGBForeColor(RGBColor *c);
void ForeColor(short color);
void RGBBackColor(RGBColor *c);
void TextMode(int mode);
void TextFont(long font);
void getfnum(char *name, short *num);
void SetOutlinePreferred(Boolean trueType);
void TextSize(long size);
void TextFace(long style);
void GetFontInfo(FontInfo *fInfo);
PicHandle GetPicture(short id);
void DrawPicture(PicHandle h, RECTPTR r);
void KillPicture(PicHandle h);
void DrawBitmap(HBITMAP hBitmap, short x, short y, DWORD dwROP);
void Move(short dx, short dy);
void Line(short dx, short dy);
void FrameArc(RECTPTR r, short startAngle, short arcAngle);
void PaintArc(RECTPTR r, short startAngle, short arcAngle);
void FrameOval(RECTPTR r);
void PaintOval(RECTPTR r);
void PaintRect(RECTPTR r);
void EraseRect(RECTPTR r);
RgnHandle NewRgn();
void DisposeRgn(RgnHandle rgn);
void drawstring(CHARPTR s);
void DrawChar(char c);
int CharWidth(char ch);
short stringwidth(CHARPTR s);
short StringWidth(CHARPTR s);
void MyInsetRect(RECTPTR r, short h, short v);
Boolean SectRect(RECTPTR r1, RECTPTR r2, RECTPTR r3);
void MapPt(POINTPTR pt, RECTPTR srcRect, RECTPTR dstRect);

/////////// MACUTIL: FILES

OSErr FSRead(int f, LONGPTR count, VOIDPTR buffer);
OSErr FSWrite(int f, LONGPTR count, VOIDPTR buffer);
OSErr GetFPos(int f, LONGPTR posOff);      
OSErr SetFPos(int f, short posMode, long posOff);
OSErr GetEOF (int refNum, LONGPTR logEOF);
OSErr SetEOF (int refNum, long logEOF);
OSErr FSClose(int f);
Boolean GetModDateLong(short f, LONGPTR seconds);
Boolean NameStartsAtIBMRoot(char* fileName);
Boolean NameIsIBMRoot(char* fileName);
short VRefNumFromIBMName(char* fileName);
OSErr NetworkGetVolumeName(short vRefNum, char *volName);//JLM 7/8/98
OSErr FSMakeFSSpec(short check, long dirID, CHARPTR fileName, FSSpec *spec);
void sfpputfile(POINTPTR where, CHARPTR rType, CHARPTR defaultName,
            DlgHookProcPtr hook, SFREPLYPTR reply, short dialogID,
            ModalFilterProcPtr filter);
void sfpgetfile(POINTPTR where, CHARPTR defaultName, FileFilterProcPtr fileFilter,
            short numTypes, SFTYPELISTPTR typeList,
             DlgHookProcPtr hook, SFREPLYPTR reply, short dialogID,
            ModalFilterProcPtr filter);

long NumFilesSelected(void); // JLM 6/4/10
void strcpyNthFile(char *szFile,long n);// JLM 6/4/10

#endif

/////////// MACUTIL: DIALOGS

#ifdef IBM

void paramtext(CHARPTR s0, CHARPTR s1, CHARPTR s2, CHARPTR s3);
void GetDialogItem(DialogPtr dialog, short itemNum, SHORTPTR type, HANDLEPTR h, RECTPTR r);
void SetDialogItem(DialogPtr dialog, short itemNum, short type, Handle h, RECTPTR r);
void setdialogitemtext(HWND w, CHARPTR text);
void SetDialogItemText(HWND w, CHARPTR text);
void getdialogitemtext(HWND w, CHARPTR text);
void GetDialogItemText(HWND w, CHARPTR text);
void SelectDialogItemText(DialogPtr dialog, short itemNum, short begin, short end);
void ShowDialogItem(DialogPtr dialog, short itemNum);
void HideDialogItem(DialogPtr dialog, short itemNum);
void DrawDialog(DialogPtr dialog);
DialogPtr GetNewDialog (int dialogID, Ptr dStorage, Ptr WindowPtr);
BOOL CALLBACK ModelessDlgProc (HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
void DisposeDialog(DialogPtr theDialog) ;

void MySelectDialogItemText(DialogPtr theDialog, short editTextitemNum, short strtSel, short endSel);
/////////// MACUTIL: OTHER

void debugstr(CHARPTR s);
void DebugStr(CHARPTR s);
void SysBeep(short duration);
void SetResLoad(Boolean load);
MenuHandle GetMenuHandle(short menuID);
short GetItemMark(MenuHandle m, short item, short *mark);
void SetItemMark(MenuHandle m, short item, char mark);
void DisableItem(MenuHandle m, short item);
void EnableItem(MenuHandle m, short item);
short CountMenuItems(MenuHandle m);
void AddResMenu(MenuHandle theMenuHdl, OSType type);
void MyDrawMenuBar();
void getwtitle(WindowPtr w, CHARPTR title);
void setwtitle(WindowPtr w, CHARPTR title);
void numtostring(long n, CHARPTR s);
void stringtonum(CHARPTR s, LONGPTR n);
void getstring(CHARPTR s, short id);
void getindstring(CHARPTR s, short groupID, short index);
void InitCursor(void);
void SetWRefCon(WindowPtr w, long value);
long GetWRefCon(WindowPtr w);
short GetControlValue(ControlHandle h);
void SetControlValue(ControlHandle h, short value);
short GetControlMinimum(ControlHandle h);
short GetControlMaximum(ControlHandle h);
void SetControlMinimim(ControlHandle h, short minVal);
void SetControlMaximim(ControlHandle h, short maxVal);
long GetDblTime(void);

unsigned long TickCount();
double rint(double value);
double anint(double num1);
//char *p2cstr (char *string);
//void c2pstr (char *string);

////////////////////////// IBM-SPECIFIC FUNCTIONS //////////////////////////

void MakeWindowsRect(RECTPTR macRect, RECT *windowsRect);
void MakeMacRect(RECT *windowsRect, RECTPTR macRect);
void MakeWindowsPoint(POINTPTR macPoint, POINT *windowsPoint);
void MakeMacPoint(POINT *windowsPoint, POINTPTR macPoint);
POINT MakePOINT(long x, long y);
void DefineEvent(EventRecord *e, HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
Boolean IsEGA();
Boolean MyIsWindow(HWND w);
BOOL RunningOnWin32(void);
BOOL RunningWindows95(void);
BOOL LongFileNames(short vRefNum);
BOOL CenterWindow (HWND hwndChild, HWND hwndParent);
void SubstituteParamText(HWND hWnd);
BOOL CALLBACK ChoiceAlertProc (HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
OSErr InitDirectoryTable();
OSErr ReleaseDirectoryTable();
Boolean GetIDFromPath (long *dirID, char *pathName, Boolean check);
Boolean GetDirectoryFromID (long dirID, char *pathName);
Boolean ResolveFileDirectory(short vRefNum, long dirID, CHARPTR fileName, CHARPTR pathName);
OSErr RenameDirectoryInTable(char *oldPathName, char *newPathName);
HMETAFILE MyGetMetaFile(LPSTR path);
OSErr MakeMetaFilePlaceable(LPSTR path, Rect frame);

// device-independent bitmap utilities

#define PW_WINDOW        1
#define PW_CLIENT        2
#define PW_BESTFIT       1
#define PW_STRETCHTOPAGE 2
#define PW_SCALE         3

extern short gDIBDestRectWidth, gDIBDestRectHeight;

HBITMAP ResizeBitmap(HBITMAP hDDB, short width, short height);
HDIB ResizeDIB(HDIB hDIB, short width, short height);

HDIB      FAR  BitmapToDIB (HBITMAP hBitmap, HPALETTE hPal);
HDIB      FAR  ChangeBitmapFormat (HBITMAP  hBitmap,
                                   WORD     wBitCount,
                                   DWORD    dwCompression,
                                   HPALETTE hPal);
HDIB      FAR  ChangeDIBFormat (HDIB hDIB, WORD wBitCount,
                                DWORD dwCompression);
HBITMAP   FAR  CopyScreenToBitmap (LPRECT);
HDIB      FAR  CopyScreenToDIB (LPRECT);
HBITMAP   FAR  CopyWindowToBitmap (HWND, WORD);
HDIB      FAR  CopyWindowToDIB (HWND, WORD);
HPALETTE  FAR  CreateDIBPalette (HDIB hDIB);
HDIB      FAR  CreateDIB(DWORD, DWORD, WORD);
WORD      FAR  DestroyDIB (HDIB);
void      FAR  DIBError (int ErrNo);
DWORD     FAR  DIBHeight (LPSTR lpDIB);
WORD      FAR  DIBNumColors (LPSTR lpDIB);
HBITMAP   FAR  DIBToBitmap (HDIB hDIB, HPALETTE hPal);
DWORD     FAR  DIBWidth (LPSTR lpDIB);
LPSTR     FAR  FindDIBBits (LPSTR lpDIB);
HPALETTE  FAR  GetSystemPalette (void);
HDIB      FAR  LoadDIB (LPSTR);
BOOL      FAR  PaintBitmap (HDC, LPRECT, HBITMAP, LPRECT, HPALETTE);
BOOL      FAR  PaintDIB (HDC, LPRECT, HDIB, LPRECT, HPALETTE);
int       FAR  PalEntriesOnDevice (HDC hDC);
WORD      FAR  PaletteSize (LPSTR lpDIB);
WORD      FAR  PrintDIB (HDIB, WORD, WORD, WORD, LPSTR);
WORD      FAR  PrintScreen (LPRECT, WORD, WORD, WORD, LPSTR);
WORD      FAR  PrintWindow (HWND, WORD, WORD, WORD, WORD, LPSTR);
WORD      FAR  SaveDIB (HDIB, LPSTR);

Rect GetDIBFrame(HDIB hDib);

#endif

long NumFilesSelected(void); // JLM 6/4/10
void strcpyNthFile(char *szFile,long n);// JLM 6/4/10

///////////////////////////////////////////// off-screen bitmap utilities

typedef struct {
#ifdef MAC
		WindowPtr w;
		CGrafPtr newPort;
		PixMapHandle newPixMap;
		PixMapHandle oldPixMap;
		Handle pixelsHandle;
#else
		HDC hMemDC;
		HBITMAP hOldBitmap;
#endif
		Rect r;
	} PortPixRecord, *PortPixRecordP, **PortPixRecordH;

OSErr OSNewPortPix(PortPixRecordP pixRec, WindowPtr w, Rect r);
void OSSetPort(PortPixRecordP pixRec);
void OSResetPort(PortPixRecordP pixRec);
void OSCopyToWindow(PortPixRecordP pixRec, WindowPtr w, Rect rSrc, Rect rDst, short mode);
void OSDisposePortPix(PortPixRecordP pixRec);

#ifdef MAC
void 			CopyBitsToGWorld(PixMapHandle bits, CGrafPtr destGWorld, Rect srcRect, Rect dstRect);
void 			CopyBitsFromGWorld(CGrafPtr sourceGWorld, PixMapHandle bits, Rect srcRect, Rect dstRect);
void			CopyBitsFromWindow(WindowPtr w, PixMapHandle bits, Rect srcRect, Rect dstRect);
#else
void			CopyBitsFromWindow(WindowPtr w, PixMapHandle *bits, Rect srcRect, Rect dstRect);
void			CopyBitsFromWindowB(WindowPtr w, PixMapHandle *bits, Rect srcRect, Rect dstRect);
#endif

void			CopyBitsToWindow(PixMapHandle bits, WindowPtr w, Rect srcRect, Rect dstRect);
void			BitMapDispose(PixMapHandle bits, WindowPtr w);

#ifdef IBM
void			CopyBitsToWindowB(PixMapHandle bits, WindowPtr w, Rect srcRect, Rect dstRect);
void			BitMapDisposeB(PixMapHandle bits, WindowPtr w);
#endif

#ifdef MAC

// a kludge for now until the offscreen bitmap stuff can be eliminated
struct MyGrafPort {
  short               device;                 /* not available in Carbon*/
  BitMap              portBits;               /* in Carbon use GetPortBitMapForCopyBits or IsPortColor*/
  Rect                portRect;               /* in Carbon use Get/SetPortBounds*/
  RgnHandle           visRgn;                 /* in Carbon use Get/SetPortVisibleRegion*/
  RgnHandle           clipRgn;                /* in Carbon use Get/SetPortClipRegion*/
  Pattern             bkPat;                  /* not available in Carbon all GrafPorts are CGrafPorts*/
  Pattern             fillPat;                /* not available in Carbon all GrafPorts are CGrafPorts*/
  Point               pnLoc;                  /* in Carbon use GetPortPenLocation or MoveTo*/
  Point               pnSize;                 /* in Carbon use Get/SetPortPenSize*/
  short               pnMode;                 /* in Carbon use Get/SetPortPenMode*/
  Pattern             pnPat;                  /* not available in Carbon all GrafPorts are CGrafPorts*/
  short               pnVis;                  /* in Carbon use GetPortPenVisibility or Show/HidePen*/
  short               txFont;                 /* in Carbon use GetPortTextFont or TextFont*/
  StyleField          txFace;                 /* in Carbon use GetPortTextFace or TextFace*/
                                              /*StyleField occupies 16-bits, but only first 8-bits are used*/
  short               txMode;                 /* in Carbon use GetPortTextMode or TextMode*/
  short               txSize;                 /* in Carbon use GetPortTextSize or TextSize*/
  Fixed               spExtra;                /* in Carbon use GetPortSpExtra or SpaceExtra*/
  long                fgColor;                /* not available in Carbon */
  long                bkColor;                /* not available in Carbon*/
  short               colrBit;                /* not available in Carbon*/
  short               patStretch;             /* not available in Carbon*/
  Handle              picSave;                /* in Carbon use IsPortPictureBeingDefined*/
  Handle              rgnSave;                /* not available in Carbon*/
  Handle              polySave;               /* not available in Carbon*/
  QDProcsPtr          grafProcs;              /* not available in Carbon all GrafPorts are CGrafPorts*/
};
typedef struct MyGrafPort                 MyGrafPort;
typedef MyGrafPort *                      MyGrafPtr;

//
typedef struct {
		GrafPtr 		savePort;
#if TARGET_API_MAC_CARBON
		//MyGrafPtr		newPort;
#else
		GrafPort		newPort;
#endif
		BitMap			savePortBits;
		RgnHandle		saveVisRgn;
		CGrafPtr 		sourcePort;
		GDHandle 		saveGDevice;
		CGrafPtr 		theGWorld;
		Rect 			theGWorldRect;
		PaletteHandle 	savePortPalHdl;
	} MyGWorldRec, *MyGWorldRecPtr;

//PixMapHandle PixMapCreate(CWindowPtr CWPtr, Rect *theBounds, Handle *PixMapHdlPtr, OSErr *ErrCodePtr);
PixMapHandle PixMapCreate(WindowPtr CWPtr, Rect *theBounds, Handle *PixMapHdlPtr, OSErr *ErrCodePtr);
Handle AllocPixMapMem(PixMapHandle thePixels);
short GetScrDepth();
PixMapHandle GetPortPix();
PicHandle GetScreenPicture (WindowPtr wPtr, Rect *picRect);

CGrafPtr MakeNewGWorld (short worldDepth, Rect *theRect);
void 	 SaveGWorld (MyGWorldRecPtr gWorldRecPtr);
void	 BeginGWorldDraw (CGrafPtr theGWorld, Boolean bCopyScreen, Boolean bErase);
void 	 EndGWorldDraw (CGrafPtr theGWorld, MyGWorldRecPtr gWorldRecPtr);
void 	 CopyWorldToScreen (CGrafPtr theGWorld, Rect theRect, short transferMode);
void 	 KillGWorld (CGrafPtr theGWorld);

OSErr   MyNewGWorld(WindowPtr SourceWPtr, short WDepth, Rect *WorldRectPtr, MyGWorldRecPtr GWorldRecPtr);
GrafPtr MySetGWorld(MyGWorldRecPtr GWorldRecPtr, Boolean bCopyScreen, Boolean bErase);
void	MyRestoreGWorld(MyGWorldRecPtr GWorldRecPtr);
void	MyBlitGWorld(MyGWorldRecPtr GWorldRecPtr, WindowPtr DumpWPtr, short transferMode);
void	MyKillGWorld(MyGWorldRecPtr GWorldRecPtr);

BitMap	OpenBlackAndWhiteBitMap(Rect theBounds, MyGWorldRec *saveGWorld);
void	CloseBlackAndWhiteBitMap (MyGWorldRec *saveGWorld);
void	DrawBlackAndWhiteBitMap(BitMap *bitmap, Rect m);
void 	DisposeBlackAndWhiteBitMap (BitMap *theBitMap);

#endif

OSErr CurrentUncertaintyDialog(CurrentUncertainyInfo *info, WindowPtr parentWindow, Boolean *uncertaintyChanged);


void mysfpputfile(POINTPTR where, CHARPTR rType, CHARPTR defaultName,
            MyDlgHookUPP hook, MySFReplyPtr reply, short dialogID,
            ModalFilterUPP filter);
void mysfpgetfile(POINTPTR where, CHARPTR defaultName, 
            short numTypes, OSType typeList[],
             MyDlgHookUPP hook, MySFReplyPtr reply, short dialogID,
            ModalFilterUPP filter);
extern OSType gMySignature ;

////////////////////////// INCLUDE OTHER HEADERS //////////////////////////
#include "CompFunctions.h"
#include "StringFunctions.h"
#include "GENUTIL.h"
#include "SAMENU.h"
#include "VLIST.h"
#include "GEOMETRY.h"

#include "CLASSES.h"
#include "OSSM.h"

#ifdef MAC
	#include "CarbonUtil.h"
#endif
#endif	 // __CROSSH__

