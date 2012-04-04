
#include "Earl.h"
#include "TypeDefs.h"

///// CONSTANTS /////////////////////////////////////////////////////////////////

#ifdef MAC
#ifdef MPW
#define NEWLINESTRING "\n"
#define UNIXNEWLINESTRING "\r"
#define MACNEWLINESTRING "\n"
#define IBMNEWLINESTRING "\n\r"
#else
#define NEWLINESTRING "\r"
#define UNIXNEWLINESTRING "\n"
#define MACNEWLINESTRING "\r"
#define IBMNEWLINESTRING "\r\n"
#endif
#else
#define NEWLINESTRING "\r\n"
#define UNIXNEWLINESTRING "\n"
#define MACNEWLINESTRING "\r"
#define IBMNEWLINESTRING "\r\n"
#endif

#ifndef MYKEYS
#define MYKEYS
enum { ENTER = 3, DEL = 8, TAB = 9, LINEFEED = 10, RETURN = 13, ESC = 27,
	   LEFTARROW = 28, RIGHTARROW = 29, UPARROW = 30, DOWNARROW = 31 };
enum { ESCAPEKEY = 0X35, ENTERKEY = 0X4C, RETURNKEY = 0X24, PERIODKEY = 0X2F,
	   UPKEY = 0X7E, UPKEYPLUS = 0X4D, DOWNKEY = 0X7D, DOWNKEYPLUS = 0X48,
	   PAGEUPKEY = 116, PAGEDOWNKEY = 121, HOMEKEY = 115, ENDKEY = 119,
	   AKEY = 0X00, IKEY = 0X22 };
#endif

#ifdef MAC
enum { ON = 1, OFF = 0 };
#define _min(a, b) ((a) < (b) ? (a) : (b))
#define _max(a, b) ((a) > (b) ? (a) : (b))
#endif

#define round(n) floor((n) + 0.5)
#define abs(n) ((n) >= 0 ? (n) : -(n))
#define RectWidth(r) ((r).right - (r).left)
#define RectHeight(r) ((r).bottom - (r).top)
#define TOPLEFT(r) (POINTPTR)(&(r).top)
#define BOTRIGHT(r) (POINTPTR)(&(r).bottom)

#define MAX_MACPAINT_SIZE 53000
#define MAX_MACPAINT_LINES 720
#define MACPAINT_BYTE_UNPACK 72

#define PI 3.14159265359
#define ROOT2 1.4142136562373
#define KG2POUNDS 2.2

#define FRAME_ITEM 2

#define MultFindEvt 15

extern RGBColor WHITECOLOR, BLACKCOLOR;
extern Rect bigRect;

///// MISC ///////////////////////////////////////////////////////////////////////////

void debugnum(CHARPTR title, long n);
float max4(float f1, float f2, float f3, float f4);
float min4(float f1, float f2, float f3, float f4);
double logB(double b, double x);
float hypotenuse(float a, float b);
double myfabs(double x);
void DrawQDSegment(Point p1, Point p2);
void SplitDMS(float n, SHORTPTR deg, SHORTPTR min, FLOATPTR sec);
void SetSign(FLOATPTR n, short code);
short ScaleToShort(long n);
long GetRandom(long low, long high);
float GetRandomFloat(float low, float high);
void GetRandomVectorInUnitCircle(float *u,float *v);
char *SwapN(char *s, short n);
long Assoc(long key, LONGPTR table, short n);
void SwitchShorts(SHORTPTR a, SHORTPTR b);
void SwitchLongs(LONGPTR a, LONGPTR b);
void SwitchStrings(CHARPTR a, CHARPTR b);
short NumDecimals(CHARPTR str);
long antol(CHARPTR s, short n);
short RGBDistance(RGBColor c1, RGBColor c2);
short ClosestMatchingColor(RGBColor color);
Boolean EarlierThan(Seconds time1, Seconds time2);
Boolean LaterThan(Seconds time1, Seconds time2);

///// STRINGS ///////////////////////////////////////////////////////////////////////////
char* lfFix(char* str);
OSErr StringToDouble(char* str,double* val);
void StringWithoutTrailingZeros(char* str,double val,short maxNumDecimalPlaces); //JLM
void ChopEndZeros(CHARPTR cString); //JLM
CHARPTR RemoveTrailingSpaces(CHARPTR s);

void RemoveLeadingWhiteSpace(CHARPTR s);//JLM
void RemoveTrailingWhiteSpace(CHARPTR s);//JLM
void RemoveLeadingAndTrailingWhiteSpace(CHARPTR s);//JLM

void RemoveLeadingTrailingQuotes(CHARPTR s);

long CountSetInString(CHARPTR s, CHARPTR set);
long CountSetInString(CHARPTR s, CHARPTR set, CHARPTR stop);
void RemoveSetFromString(CHARPTR strin, CHARPTR set, CHARPTR strout);
CHARPTR AfterSetInString(CHARPTR s, CHARPTR set, long count);
CHARPTR StringSubstitute(CHARPTR s, char old, char newC);
CHARPTR strnzcpy(CHARPTR to, CHARPTR from, short n);
CHARPTR strnztrimcpy(CHARPTR to, CHARPTR from, short n);
CHARPTR strtrimcpy(CHARPTR to, CHARPTR from);
char mytolower(char c);
CHARPTR StrToLower(CHARPTR s);
char mytoupper(char c);
CHARPTR StrToUpper(CHARPTR s);
char *strstrnocase(CHARPTR s1, CHARPTR s2);
short strcmpnocase(CHARPTR s1, CHARPTR s2);
short strncmpnocase(CHARPTR s1, CHARPTR s2, short n);
short strnblankcmp(CHARPTR s1, CHARPTR s2, short n);
Boolean strcmptoreturn(CHARPTR s1, CHARPTR s2);
Boolean strcmptoreturnnocase(CHARPTR s1, CHARPTR s2);
void AbbreviateString(CHARPTR s, short maxWidth);
CHARPTR strcatres(CHARPTR s, short resID, short index);
CHARPTR mypstrcat(CHARPTR dest, CHARPTR src);
CHARPTR mypstrcpy(CHARPTR dest, CHARPTR src);
void mypstrcatJM(void* dest_asVoidPtr, void* src_asVoidPtr);
void mypstrcpyJM(void* dest_asVoidPtr, void* src_asVoidPtr);

long NumLinesInText(CHARPTR text);
CHARPTR NthLineInTextOptimized(CHARPTR text, long n, CHARPTR line, long maxLen);
CHARPTR NthLineInTextNonOptimized(CHARPTR text, long n, CHARPTR line, long maxLen);
CHARPTR LineInText(CHARPTR text, CHARPTR line);
CHARPTR LineInTextNoCase(CHARPTR text, CHARPTR line);
CHARPTR AddLineToText(CHARPTR text, CHARPTR line);
CHARPTR IntersectLinesInText(CHARPTR text1, CHARPTR text2, CHARPTR intersection);
CHARPTR strcpyToDelimeter(CHARPTR target, CHARPTR source, char delimiter);
CHARPTR strcpyWithDelimeter(CHARPTR target, CHARPTR source, char delimiter);
Boolean ForceStringNumber(CHARPTR s);
Boolean ForceStringNumberAllowingNegative(CHARPTR s);//JLM
Boolean DecForceStringNumber(CHARPTR s);
Boolean DecForceStringNumberAllowingNegative(CHARPTR s); //JLM
Boolean DecForceStringDirection(CHARPTR s);
char *RGBColorToString(RGBColor color, char *s);
RGBColor StringToRGBColor(char *s);
Boolean ValidRGBString(char *s);

///// RECTS ///////////////////////////////////////////////////////////////////////////

void MySetRect(RECTPTR r, short left, short top, short right, short bottom);
Point RectCenter(Rect r);
Boolean RectInRect(RECTPTR r1, RECTPTR r2);
Rect ForceRectIntoRect(Rect r, Rect r2);
void NormalizeRect(RECTPTR r);
void FrameRectEllipse(RECTPTR r, Boolean fromCenter, Boolean ellipse, Point p);
void LocalToGlobalRect(Rect *r);
Boolean RectOffScreen(WindowPtr w, Rect r);

///// DIALOGS ////////////////////////////////////////////////////////////////////////

void SetDialogItemHandle(DialogPtr theDialog, short item, Handle itemProc);
Handle GetDialogItemHandle(DialogPtr theDialog, short item);
Rect GetDialogItemBox(DialogPtr theDialog, short item);
void SetDialogItemBox(DialogPtr theDialog, short item, Rect box);
void InvalDItemBox(DialogPtr d, short item);
short GetDialogItemType(DialogPtr theDialog, short item);
void SetDialogItemType(DialogPtr theDialog, short item, short type);
void ToggleButton(DialogPtr theDialog, short item);
void SetButton(DialogPtr theDialog, short item, short value);
short GetButton(DialogPtr theDialog, short item);
void mysetitext(DialogPtr theDialog, short item, CHARPTR text);
CHARPTR mygetitext(DialogPtr theDialog, short item, CHARPTR text, short n);
CHARPTR mygetitextexact(DialogPtr theDialog, short item, CHARPTR text, short n);
Boolean setitextIfChanged(DialogPtr theDialog, short item, CHARPTR text);
void ShowHideDialogItem(DialogPtr theDialog, short item, Boolean bShow);
Boolean CheckDirectionTextItem(DialogPtr theDialog, short item);
Boolean CheckNumberTextItem(DialogPtr theDialog, short item, Boolean decimal);
Boolean CheckNumberTextItemAllowingNegative(DialogPtr theDialog, short item, Boolean decimal);
void Float2EditText(DialogPtr theDialog, short item, double n, short decimals);
void OriginalFloat2EditText(DialogPtr theDialog, short item, float n, short decimals);
void Long2EditText(DialogPtr theDialog, short item, long n);
long EditText2Long(DialogPtr theDialog, short item);
double EditText2Float(DialogPtr theDialog, short item);
void RemoveSpacesAndCommas(DialogPtr theDialog, SHORTPTR items);
void FillScreenWithDialog(WindowPtr w, short titleHeight);
short ClickAnyButtonDialog(short dialogID, WindowPtr parent);

///// FILES ///////////////////////////////////////////////////////////////////////////

long myfgets(CHARPTR buffer, long n, short f);
OSErr RenameFolder(short vRefNum, long dirID, char *newName);
OSErr AppendFile(short f1, short f2);
OSErr FilterFile(long fromID, short fromVRefNum, CHARPTR from,
				 long toID, short toVRefNum, CHARPTR to, OSType creator, OSType type,
				 long recLength, Boolean (*filterFunc)(VOIDPTR), void (*progress)(short));

///// FQSORT ///////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif
int CCompare(const void *rec1, const void *rec2);
#ifdef __cplusplus
}
#endif
OSErr MergeSortedFiles(long dirID, short vRefNum, CHARPTR file1, CHARPTR file2,
					   CHARPTR to, OSType creator, OSType type,
					   int (*compare)(const void *rec1, const void *rec2), long numRecs1, long numRecs2, long recLength, long memAvail,
					   void (*display)(char *), void (*progress)(short));
OSErr fqsort(long fromID, short fromVRefNum, CHARPTR from,
			 long toID, short toVRefNum, CHARPTR to, OSType creator, OSType type,
			 int (*compare)(const void *rec1, const void *rec2), long recLength,
			 void (*display)(char *), void (*progress)(short));

///// BUFFERED FILES ///////////////////////////////////////////////////////////////////

void EnterReadOnly();
void ExitReadOnly(OSErr (*f)(void));
OSErr FSOpenBuf(short vRefNum, long dirID, CHARPTR fName, BFPBP bfpb, long bufSize, Boolean temp);
OSErr FSReopenBuf(short f, BFPBP bfpb);
OSErr FSWriteIfDirtyBuf(BFPBP bfpb);
OSErr FSResizeBuf(BFPBP bfpb, long newSize, Boolean temp);
OSErr FSPrepareBuf(BFPBP bfpb, long count);
OSErr FSReadBuf(BFPBP bfpb, LONGPTR count, CHARPTR dest);
OSErr FSWriteBuf(BFPBP bfpb, LONGPTR count, CHARPTR src);
OSErr GetFPosBuf(BFPBP bfpb, LONGPTR filePos);
OSErr SetFPosBuf(BFPBP bfpb, long filePos);
OSErr WriteAtPosBuf(BFPBP bfpb, LONGPTR count, CHARPTR src, long pos);
OSErr FSTransferBuf(BFPBP from, BFPBP to, long start, long end);
OSErr FSCloseBuf(BFPBP bfpb);

///// BLOCK FILES //////////////////////////////////////////////////////////////////////
enum {TERMINATED = 2,NONTERMINATED};
OSErr ReadFileContents(short terminationFlag, short vRefNum, long dirID, CHARPTR name,
					   VOIDPTR ptr, long length, CHARHP handle);
OSErr ReadSectionOfFile(short vRefNum, long dirID, CHARPTR name,
						long offset, long length, VOIDPTR ptr, CHARHP handle);
OSErr WriteFileContents(short vRefNum, long dirID, CHARPTR name, OSType creator, OSType fileType,
						VOIDPTR ptr, long length, CHARH handle);
OSErr AddToFileContents(short vRefNum, long dirID, CHARPTR name, OSType creator, OSType fileType,
						VOIDPTR ptr, long length, CHARH handle, LONGPTR offset);
OSErr MyCopyFile(short vRefNumFrom, long dirIDFrom, CHARPTR nameFrom,
			   short vRefNumTo, long dirIDTo, CHARPTR nameTo);
OSErr TransferFileSection(short from, short to, long start, long end);
long myfgetsbuf(CHARPTR buffer, long n, BFPBP file);
OSErr DeleteFileSection(short vRefNum, long dirID, CHARPTR name, long offset, long length);

///// DATABASES ////////////////////////////////////////////////////////////////////////

OSErr OpenDataBase(DataBaseP db, CHARPTR fileName, short vRefNum, long dirID,
				   OSType creator, IndexFileP indexList, long bufSize);
OSErr NthByIndex(DataBaseP db, long n, short indexNum, LONGPTR offset);
OSErr SeekByIndex(DataBaseP db, short indexNum, VOIDPTR keyRecord, long recLength,
				  LONGPTR offset, LONGPTR indexN);
OSErr ShiftFileSection(short f, long start, long d);
OSErr ReadRecord(DataBaseP db, long offset, long length, VOIDPTR record);
long FileSize(DataBaseP db);
OSErr InsertRecordIntoIndex(DataBaseP db, short indexNum, VOIDPTR record, long recLength,
							long offset);
OSErr DeleteRecordFromIndex(DataBaseP db, short indexNum, VOIDPTR record, long recLength);
OSErr AppendRecord(DataBaseP db, VOIDPTR record, long length);
OSErr OverwriteRecord(DataBaseP db, VOIDPTR record, long length, long offset);
OSErr CloseDataBase(DataBaseP db);
