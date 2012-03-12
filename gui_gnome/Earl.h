/*
 *  Earl.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Earl__
#define __Earl__

#include "Basics.h"

#ifdef MAC
#ifdef VERSION881
#pragma load "MACHEADERS881"
#endif
#ifdef VERSION68K
#pragma load "MACHEADERS68K"
#endif
#else
#pragma warning(disable : 4761)
#endif

#ifdef MAC
#if TARGET_API_MAC_CARBON
#define MACB4CARBON 0 
#else
#define MACB4CARBON 1 
#endif
#else // IBM
#define MACB4CARBON 0 
#endif

#ifdef IBM
//	#define _MAX_DIR 128		// defined in win stdlib.h
#define OEMRESOURCE
#include <Windows.h>
#include <WindowsX.h>
#include <ddeml.h>
#include <stdlib.h>
#include <stdio.h>
#include <io.h>
//	#include <dos.h>			// changed for Codewarrior
#include <math.h>
#include <direct.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <float.h>	// for isnan
#include <shellapi.h>	// for command line argc/argv
//	#include <StandardFile.h>	// changed for Codewarrior
#endif

#ifdef MAC //JLM
#ifdef MPW //JLM
#define PI pi // JLM
#define _min(a,b) ((a) < (b) ? (a) : (b))  //JLM
#define _max(a,b) ((a) > (b) ? (a) : (b)) ///JLM
#define _min(a,b) ((a) < (b) ? (a) : (b))  //AH
#define _max(a,b) ((a) > (b) ? (a) : (b)) ///AH
#define kFontIDNewYork    2
#define kFontIDGeneva     3
#define kFontIDMonaco     4
#define kFontIDTimes     20
#define kFontIDHelvetica 21
#define kFontIDCourier   22
#define kFontIDSymbol    23
#define kFontIDMarplot   -1			
#endif
#endif

// get around WIN compiler warning about pascal, JLM 3/25/09
#ifdef MAC
#define pascal_ifMac  pascal
#else
#define pascal_ifMac
#endif


#ifdef IBM
#define unused(v) warning ( disable : 4137 )
typedef BOOL Boolean;
typedef BYTE Byte;
typedef double extended;
typedef unsigned long OSType;
typedef long OSErr;
typedef HWND ControlHandle;
typedef HWND GrafPtr;
typedef HWND WindowPtr;
typedef HWND DialogPtr;
typedef HWND WindowRef;
typedef HWND DialogRef;
typedef HWND WindowPeek;
typedef HRGN RgnHandle;
typedef HMENU MenuHandle;
typedef LPOFNHOOKPROC DlgHookProcPtr;
typedef LPOFNHOOKPROC MyDlgHookProcPtr;
typedef char *StringPtr;
typedef int (*ProcPtr)();
#define fsCurPerm 0
#define fsRdPerm 1
#define fsWrPerm 2
#define fsRdWrPerm 3
#define fsRdWrShPerm 4
#define fsAtMark 0
#define fsFromStart 1
#define fsFromLEOF 2
#define fsFromMark 3
#define kControlUpButtonPart 20
#define kControlDownButtonPart 21
#define kControlPageUpPart 22
#define kControlPageDownPart 23
#define kControlIndicatorPart 129
#define ioErr -36
#define fnfErr -43
#define eofErr -39
#define memFullErr -108
#define nilHandleErr -109
#define dirNFErr -120	//Directory not found
#define normal 0
#define bold 1
#define italic 2
#define underline 4
#define outline 8
#define shadow 16
#define srcCopy      R2_COPYPEN
#define srcOr        R2_MASKPEN
#define srcXor       R2_NOTXORPEN
#define srcBic       R2_WHITE
#define notSrcCopy   R2_NOTCOPYPEN
#define notSrcOr     R2_NOTMASKPEN
#define notSrcXor    R2_XORPEN
#define notSrcBic    R2_BLACK
#define patCopy      R2_COPYPEN
#define patOr        R2_MASKPEN
#define patXor       R2_NOTXORPEN
#define patBic       R2_WHITE
#define notPatCopy   R2_NOTCOPYPEN
#define notPatOr     R2_NOTMASKPEN
#define notPatXor    R2_XORPEN
#define notPatBic    R2_BLACK
#define systemFont   	  0
#define applFont     	  1
#define kFontIDNewYork    2
#define kFontIDGeneva     3
#define kFontIDMonaco     4
#define kFontIDTimes     20
#define kFontIDHelvetica 21
#define kFontIDCourier   22
#define kFontIDSymbol    23
#define kFontIDMarplot   -1			
#define teJustLeft   0
#define teJustCenter 1
#define teJustRight -1
#define teForceLeft -2
#define noMark 0
#define checkMark 0x12
#define blackColor   33
#define whiteColor   30
#define redColor     205
#define greenColor   341
#define blueColor    409
#define cyanColor    273
#define magentaColor 137
#define yellowColor  69
#define LW_DATA 0
#define LW_INDX 0
#define LW_MCTR 0
#define LW_TEXT 0
#define LW_CLNK 0
typedef struct {
	short v;
	short h;
} Point;
typedef MSG EventRecord;
typedef struct {
	short top;
	short left;
	short bottom;
	short right;
} Rect;
typedef struct {
	long picSize;
	Rect picFrame;
	HDC hdcMeta;
	HMETAFILE hmf;
} Picture;
typedef struct {
	OSType fdType;
	OSType fdCreator;
	unsigned short fdFlags;
	Point fdLocation;
	short fdFldr;
} FInfo;
typedef struct {
	short ascent;
	short descent;
	short widMax;
	short leading;
} FontInfo;
typedef COLORREF RGBColor;
typedef struct {
	Point pnLoc;
	Point pnSize;
	short pnMode;
	short pnPat;
} PenState;
typedef struct {
	short year;
	short month;
	short day;
	short hour;
	short minute;
	short second;
	short dayOfWeek;
} DateTimeRec;
typedef struct {
	short vRefNum;
	long dirID;
	char pathName[_MAX_DIR];
} ParamBlockRec;
typedef struct {
	short vRefNum;
	long dirID;
	char pathName[_MAX_DIR];
} HFileInfo;
typedef struct {
	struct {
		short ioVRefNum;
		long ioFlParID;
		char *ioNamePtr;
	} hFileInfo;
	struct {
		long ioDrDirID;
	} dirInfo;
} CInfoPBRec;
typedef struct {
	short vRefNum;
	long parID;
	char name[256];
} FSSpec;
typedef struct {
	long lo;
	long hi;
} ProcessInfoRec;
typedef struct {
	Boolean good;
	short vRefNum;
	char fName[_MAX_DIR];
} SFReply;
typedef OSType SFTypeList[];
enum { QD_PATTERN = 32, QD_PENWIDTH = 36,
	QD_FONT = 40, QD_SIZE = 44, QD_STYLE = 48, QD_TEXTMODE = 52,
	QD_RGB = 56, QD_CLIPRGN = 60 };
#endif

#if TARGET_API_MAC_CARBON
typedef struct {
	Boolean good;
	////
	//short vRefNum; ///code goes here.. can we get rid of this ?
	// char fName[256];///code goes here.. can we get rid of this ?
#ifdef MAC
	OSType fType;
#endif
	/////
	char fullPath[256];
} MySFReply;
#else
typedef SFReply MySFReply;
#endif

typedef Boolean *BOOLEANPTR;
typedef Byte *BYTEPTR;
typedef Point *POINTPTR, **POINTH;
typedef Rect *RECTPTR, **RECTH;
typedef Picture *PICTUREPTR, **PICTUREH;
typedef PICTUREH *PICTUREHP;
typedef EventRecord *EVENTRECORDPTR;
typedef FInfo *FINFOPTR;
typedef RGBColor *RGBCOLORPTR;
typedef HFileInfo *HFILEINFOPTR;
typedef ParamBlockRec *PARMBLKPTR;
typedef CInfoPBRec *CINFOPBRECPTR;
typedef MySFReply *MySFReplyPtr;

#ifdef IBM
typedef SFReply *SFREPLYPTR;	// should get rid of this altogether
typedef SFTypeList SFTYPELISTPTR;
typedef PICTUREH PicHandle;
typedef HANDLE HDIB;
typedef HDIB PixMapHandle;
typedef long SysEnvRec;
typedef short Pattern;
typedef long Size;
#define LoWord LOWORD
#define HiWord HIWORD
#define true TRUE
#define false FALSE
#define DIRDELIMITER '\\'
#define OPPOSITEDIRDELIMITER ':'
typedef Boolean (pascal_ifMac *ModalFilterProcPtr)(DialogPtr, EVENTRECORDPTR, SHORTPTR);
typedef Boolean (pascal_ifMac *FileFilterProcPtr)(PARMBLKPTR);
#define _min(a,b) ((a) < (b) ? (a) : (b))  //AH
#define _max(a,b) ((a) > (b) ? (a) : (b)) ///AH

#else
#ifdef MPW
#define _MAX_DIR 128
#endif

#define _min(a,b) ((a) < (b) ? (a) : (b))
#define _max(a,b) ((a) > (b) ? (a) : (b))
#define DIRDELIMITER ':'
#define OPPOSITEDIRDELIMITER '\\'
#define LW_DATA 'DATA'
#define LW_INDX 'INDX'
#define LW_MCTR 'MCTR'
#define LW_TEXT 'TEXT'
#define LW_CLNK 'CLNK'
#endif

#ifdef MAC
#define _HLock(h) MyHLock(h)
#define WITHOUTERRORS(c) c
#else
#define WITHOUTERRORS(c) { UINT e; e = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX); c SetErrorMode(e); }
#endif


/*#if MACB4CARBON
 extern QDGlobals qd; 	
 #endif*/

#define MyWindowPeek WindowPeek

#ifdef MAC
// these are unused on the MAC
typedef long MyDlgHookUPP;
typedef long MyDlgHookProcPtr;
#endif

#ifdef IBM  
#define UniversalProcPtr ProcPtr
typedef long ProcInfoType;

#define UserItemProcPtr  ProcPtr

#define ModalFilterUPP ModalFilterProcPtr
#define UserItemUPP UserItemProcPtr
#define MyDlgHookUPP MyDlgHookProcPtr
#endif
/////////////////////////////////////////////////
/////////////////////////////////////////////////

ModalFilterUPP MakeModalFilterUPP(ModalFilterProcPtr p);
UserItemUPP MakeUserItemUPP(UserItemProcPtr p);

MyDlgHookUPP MakeDlgHookUPP(MyDlgHookProcPtr p);

#ifndef VERSIONPPC
#ifndef SYMANTEC
#ifdef MPW
#define QDTextProcPtr Ptr
#define QDLineProcPtr Ptr
#define QDRectProcPtr Ptr
#define QDRRectProcPtr Ptr
#define QDOvalProcPtr Ptr
#define QDArcProcPtr Ptr
#define QDPolyProcPtr Ptr
#define QDRgnProcPtr Ptr
#define QDBitsProcPtr Ptr
#define QDCommentProcPtr Ptr
#define QDTxMeasProcPtr Ptr
#define QDGetPicProcPtr Ptr
#define QDPutPicProcPtr Ptr
#define QDOpcodeProcPtr Ptr
#endif
#ifdef IBM
#define FileFilterUPP FileFilterProcPtr
#define DlgHookUPP DlgHookProcPtr
#define uppModalFilterProcInfo 0
#define uppUserItemProcInfo 0
#endif
#endif
#endif

#ifdef MAC
AEEventHandlerUPP MakeAEEventHandlerUPP(AEEventHandlerProcPtr p);
ControlActionUPP MakeControlActionUPP(ControlActionProcPtr p);
#endif

#define  MakeSAMenuActionUPP(p)  (SAMenuActionUPP)MakeUPP((ProcPtr) p,uppSAMenuActionProcInfo)
UniversalProcPtr MakeUPP(ProcPtr p, ProcInfoType pInfo);


#ifdef MAC
#define GetWindowGrafPort(w)  ((GrafPtr)GetWindowPort(w))
#else
#define GetWindowGrafPort(w)  (w)
#define GetPortPixMap(x) ((x)->portPixMap)
#endif

///// CONSTANTS, VARIABLES ///////////////////////////////////////////////////////////

#define DLG_STRINGS 10000
// include strings with these index numbers in DLG_STRINGS
enum { CANCEL_STRING = 1, HELP_STRING, PAGE_STRING, OF_STRING,
	HEADING_STRING, CLOSE_STRING = 14, DONE_STRING = 15 };

#define DIMMED 255
#define NOTDIMMED 0

enum { TOP, CENTER, BOTTOM, LEFT, RIGHT, LEFTTOP, RIGHTTOP, LEFTBOT, RIGHTBOT };

enum { JOBDIALOG, STYLEDIALOG, JUSTGETPRRECORD };

#define HELP_DIALOG 134
enum { HELP_TOPICS = 1, HELP_OUTLINE, HELP_CANCEL, HELP_COPY, HELP_PRINT, HELP_USERITEM };
#define HELP_TOPICS_DIALOG 135
enum { TOPICS_SELECT = 1, TOPICS_OUTLINE, TOPICS_CANCEL, TOPICS_USERITEM };
#define TOPICS_TEXT 101

#define CONCENTRATION_TABLE 136
enum { CONC_OK = 1, CONC_FROST, CONC_CANCEL, CONC_COPY, CONC_PRINT, CONC_USERITEM };

#define HELP_ICON_ID 128
#define GNOME_GOING_LEFT_ID 130
#define GNOME_GOING_RIGHT_ID 131
#define BIG_GNOME_GOING_RIGHT_ID 133
#define BLACKANDWHITE_GNOME_GOING_RIGHT_ID 134

extern EventRecord lastEvent;
extern Handle rainyDayReserve;
extern Boolean resetCursor, inBackground;

enum { VERTSTRIPES = 6, HORIZSTRIPES, UPSTRIPES, DOWNSTRIPES, BOXES }; // patterns
enum { ANTSH1 = 14, ANTSH2, ANTSH3, ANTSH4, ANTSH5, ANTSH6, ANTSH7, ANTSH8,
	ANTSV1, ANTSV2, ANTSV3, ANTSV4, ANTSV5, ANTSV6, ANTSV7, ANTSV8 }; // ant patterns
enum { TWOPOINT = 3, THREEPOINT, FOURPOINT, DOTS, DASHDOTDOT, DASHDOT, DASHES }; // Windows line patterns


#endif
