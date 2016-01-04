
#include "CROSS.h"
#include <fstream>
#include <ios>

using std::fstream;
using std::ios;
using std::ifstream;

// #define NOTEMPMEM

#ifdef MAC
#ifdef MPW
#pragma SEGMENT GENUTIL
#endif
#endif


#ifdef MAC
RGBColor WHITECOLOR = {65535, 65535, 65535}, BLACKCOLOR = {0, 0, 0};
#else
RGBColor WHITECOLOR = RGB(255, 255, 255), BLACKCOLOR = RGB(0, 0, 0);	
#endif


///// MISC ///////////////////////////////////////////////////////////////////////////

void debugnum(CHARPTR title, long n)
{
	char s[256], t[30];
	
	strcpy(s, title);
	numtostring(n, (CHARPTR) t);
	strcat(s, " ");
	strcat(s, t);
	my_c2pstr(s);
	DebugStr((StringPtr)s);
}


void DrawQDSegment(Point p1, Point p2)
{
	MyMoveTo(p1.h, p1.v); MyLineTo(p2.h, p2.v);
}

void SplitDMS(float n, SHORTPTR deg, SHORTPTR min, FLOATPTR sec)
{
	*deg = (short)n;
	n -= (short)n;
	n *= 60;
	*min = (short)n;
	n -= (short)n;
	*sec = n * 60;
}



short RGBDistance(RGBColor c1, RGBColor c2)
{
	short r1 = GetRValue(c1), g1 = GetRValue(c1), b1 = GetRValue(c1),
		  r2 = GetRValue(c2), g2 = GetRValue(c2), b2 = GetRValue(c2);
	
	return abs(r2 - r1) + abs(g2 - g1) + abs(b2 - b1);
}

short ClosestMatchingColor(RGBColor color)
{
	short i, c;
	long d, minD = 0;
	
	for (i = 1 ; i <= 16 ; i++) {
		d = RGBDistance(colors[i], color);
		if (i == 1 || d < minD) {
			minD = d;
			c = i;
		}
	}
	
	return c;
}



void AbbreviateString(CHARPTR s, short maxWidth)
{
	if (stringwidth(s) <= maxWidth) return;
	
	maxWidth -= stringwidth("...") + 2;
	my_c2pstr(s);
	while (s[0] && StringWidth((StringPtr)s) > maxWidth)
		s[0]--;
	my_p2cstr((StringPtr)s);
	strcat(s, "...");
}

CHARPTR strcatres(CHARPTR s, short resID, short index)
{
	char addition[100];
	
	getindstring(addition, resID, index);
	strcat(s, addition);
	return s;
}

/////////////////////////////////////////////////

char *RGBColorToString(RGBColor color, char *s)
{
	short red = GetRValue(color), blue = GetBValue(color), green = GetGValue(color);
	
	sprintf(s, "R%hdG%hdB%hd", red, green, blue);
	
	return s;
}

RGBColor StringToRGBColor(char *s)
{
	short red, green, blue;
	RGBColor color = colors[BLACK];
	
	if (sscanf(s, "R%hdG%hdB%hd", &red, &green, &blue) == 3) {
#ifdef MAC
		color.red = red * 256;
		color.green = green * 256;
		color.blue = blue * 256;
#else
		color = RGB(red, green, blue);
#endif
	}
	
	return color;
}

Boolean ValidRGBString(char *s)
{
	short red, green, blue;
	
	return sscanf(s, "R%hdG%hdB%hd", &red, &green, &blue) == 3;
}

///// RECTS ///////////////////////////////////////////////////////////////////////////


Rect ForceRectIntoRect(Rect r, Rect r2)
{
	if (RectWidth(r) > RectWidth(r2) || RectHeight(r) > RectHeight(r2))
		return r;
	
	if (r.left < r2.left) MyOffsetRect(&r, r2.left - r.left, 0);
	if (r.right > r2.right) MyOffsetRect(&r, r2.right - r.right, 0);
	if (r.top < r2.top) MyOffsetRect(&r, 0, r2.top - r.top);
	if (r.bottom > r2.bottom) MyOffsetRect(&r, 0, r2.bottom - r.bottom);
	
	return r;
}

void FrameRectEllipse(RECTPTR r, Boolean fromCenter, Boolean ellipse, Point p)
{
	Rect r2 = *r;
	
	if (fromCenter)
		MySetRect(&r2, p.h - RectWidth(*r), p.v - RectHeight(*r),
					   p.h + RectWidth(*r), p.v + RectHeight(*r));
	
	if (ellipse)
		FrameOval(&r2);
	else
		MyFrameRect(&r2);
}

void LocalToGlobalRect(Rect *r)
{
	LocalToGlobal((Point *)&r->top);
	LocalToGlobal((Point *)&r->bottom);
}

Boolean RectOffScreen(WindowPtr w, Rect r)
{
	Point p1, p2, p3, p4;
	Rect screen = GetScreenBounds();
	GrafPtr savePort;
	Boolean result = FALSE;
	
#ifdef IBM
	if (IsZoomed(w)) return FALSE;
#endif
	
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(w);
	
	SetPt(&p1, r.left, r.top); SetPt(&p2, r.left, r.bottom);
	SetPt(&p3, r.right, r.top); SetPt(&p4, r.right, r.bottom);
	
	LocalToGlobal(&p1); LocalToGlobal(&p2); LocalToGlobal(&p3); LocalToGlobal(&p4);
	
	if (!MyPtInRect(p1, &screen) || !MyPtInRect(p2, &screen) ||
		!MyPtInRect(p3, &screen) || !MyPtInRect(p4, &screen)) result = TRUE;
	
	SetPortGrafPort(savePort);
	
#ifdef MAC
	{ // deal with WindowShade
		WindowPtr w2;
		
		if (!result) {
			p1.h = (p1.h + p3.h) / 2;
			p1.v = (p1.v + p2.v) / 2;
			FindWindow(p1, &w2);
			if (w2 != w)
				result = TRUE;
		}
	}
#endif
	
	return result;
}

///// DIALOGS ///////////////////////////////////////////////////////////////////////////

void SetDialogItemHandle(DialogPtr theDialog, short item, Handle itemProc)
{
	short itemType;
	Handle itemHandle;
	Rect itemBox;
	UserItemUPP uPP = 0;
	
	//uPP = (UserItemUPP)MakeUPP((ProcPtr)itemProc, uppUserItemProcInfo);
	if(itemProc) uPP = MakeUserItemUPP((UserItemProcPtr)itemProc);

//	uPP = NewUserItemProc ((ProcPtr)itemProc);
	
	GetDialogItem(theDialog, item, &itemType, &itemHandle, &itemBox);
	SetDialogItem(theDialog, item, itemType, (Handle)uPP, &itemBox);
}

Handle GetDialogItemHandle(DialogPtr theDialog, short item)
{
	short		itemType;
	Handle	itemHandle;
	Rect		itemBox;
	
	GetDialogItem(theDialog, item, &itemType, &itemHandle, &itemBox);
	return itemHandle;
}

Rect GetDialogItemBox(DialogPtr theDialog, short item)
{
	short		itemType;
	Handle	itemHandle;
	Rect		itemBox;
	
	GetDialogItem(theDialog, item, &itemType, &itemHandle, &itemBox);
	return itemBox;
}

short GetDialogItemType(DialogPtr theDialog, short item)
{
	short		itemType;
	Handle	itemHandle;
	Rect		itemBox;
	
	GetDialogItem(theDialog, item, &itemType, &itemHandle, &itemBox);
	return itemType;
}

void SetDialogItemBox(DialogPtr theDialog, short item, Rect box)
{
	short		itemType;
	Handle	itemHandle;
	Rect		itemBox;
	
	GetDialogItem(theDialog, item, &itemType, &itemHandle, &itemBox);
	SetDialogItem(theDialog, item, itemType, itemHandle, &box);
}

void SetDialogItemType(DialogPtr theDialog, short item, short type)
{
	short		itemType;
	Handle	itemHandle;
	Rect		itemBox;
	
	GetDialogItem(theDialog, item, &itemType, &itemHandle, &itemBox);
	SetDialogItem(theDialog, item, type, itemHandle, &itemBox);
}

void ToggleButton(DialogPtr theDialog, short item)
{
	ControlHandle h = (ControlHandle)GetDialogItemHandle(theDialog, item);
	
	SetControlValue(h, !GetControlValue(h));
}

void SetButton(DialogPtr theDialog, short item, short value)
{
	ControlHandle h = (ControlHandle)GetDialogItemHandle(theDialog, item);
	
	SetControlValue(h, value);
}

short GetButton(DialogPtr theDialog, short item)
{
	ControlHandle h = (ControlHandle)GetDialogItemHandle(theDialog, item);
	
	return GetControlValue(h);
}

void mysetitext(DialogPtr theDialog, short item, CHARPTR text)
{
	char localStr[256];
	strcpy(localStr,text);
	my_c2pstr(localStr);
#ifdef MAC
	//char localStr[256];
	//strcpy(localStr,text);
	//my_c2pstr(localStr);
	SetDialogItemText(GetDialogItemHandle(theDialog, item), (StringPtr)localStr);
	//SetDialogItemText(GetDialogItemHandle(theDialog, item), (StringPtr)text);
	#if TARGET_API_MAC_CARBON
	{	// In OS X we seem to need to use both use SetControlTitle and SetDialogItemText
		ControlRef theControl = 0;
		OSErr err = GetDialogItemAsControl(theDialog,item,&theControl) ;
		if(!err &&  theControl){
			SetControlTitle(theControl,(StringPtr)localStr); 
		}
	}
	#endif
#else
	//SetDialogItemText((HWND)GetDialogItemHandle(theDialog, item), (StringPtr)text);
	SetDialogItemText((HWND)GetDialogItemHandle(theDialog, item), (StringPtr)localStr);
#endif
	//my_p2cstr((StringPtr)text);
}

CHARPTR mygetitext(DialogPtr theDialog, short item, CHARPTR text, short n)
{
	char s[256];
#ifdef MAC	
	GetDialogItemText(GetDialogItemHandle(theDialog, item), (StringPtr)s);
#else
	GetDialogItemText((HWND)GetDialogItemHandle(theDialog, item), (StringPtr)s);
#endif
	my_p2cstr((StringPtr)s);
	strnztrimcpy(text, s, n);
	
	return text;
}

CHARPTR mygetitextexact(DialogPtr theDialog, short item, CHARPTR text, short n)
{
	char s[256];
#ifdef MAC
	GetDialogItemText(GetDialogItemHandle(theDialog, item), (StringPtr)s);
#else
	GetDialogItemText((HWND)GetDialogItemHandle(theDialog, item), (StringPtr)s);
#endif
	my_p2cstr((StringPtr)s);
	strnzcpy(text, s, n);
	
	return text;
}

Boolean setitextIfChanged(DialogPtr theDialog, short item, CHARPTR text)
{
	char oldText[256];
	
	mygetitext(theDialog, item, oldText, 255);
	if (strcmp(text, oldText)) {
		mysetitext(theDialog, item, text);
		
		return TRUE;
	}
	
	return FALSE;
}

void ShowHideDialogItem(DialogPtr theDialog, short item, Boolean bShow)
{
	if (bShow) ShowDialogItem(theDialog, item);
	else HideDialogItem(theDialog, item);
}

Boolean CheckDirectionTextItem(DialogPtr theDialog, short item)
{
	Boolean changed;
	char s[256];
	
	mygetitextexact(theDialog, item, s, 255);
	if (strlen(s) == 0) return FALSE;
	
	if (changed = DecForceStringDirection(s))
		mysetitext(theDialog, item, s); // s has been modified
	
	return changed;
}

Boolean CheckNumberTextItem(DialogPtr theDialog, short item, Boolean decimal)
{
	Boolean changed;
	char s[256];
	
	mygetitextexact(theDialog, item, s, 255);
	if (strlen(s) == 0) return FALSE;
	
	if (changed = (decimal ? DecForceStringNumber(s) : ForceStringNumber(s)))
		mysetitext(theDialog, item, s); // s has been modified
	
	return changed;
}

Boolean CheckNumberTextItemAllowingNegative(DialogPtr theDialog, short item, Boolean decimal)
{	// JLM
	Boolean changed;
	char s[256];
	
	mygetitextexact(theDialog, item, s, 255);
	if (strlen(s) == 0) return FALSE;
	
	if (changed = (decimal ? DecForceStringNumberAllowingNegative(s) : ForceStringNumberAllowingNegative(s)))
		mysetitext(theDialog, item, s); // s has been modified
	
	return changed;
}

void OriginalFloat2EditText(DialogPtr theDialog, short item, float n, short decimals)
{
	char str[32];
	char format[10];
	sprintf(format, "%%.%df", decimals);
	sprintf(str, format, n);

	ChopEndZeros(str);

	mysetitext(theDialog, item, str);
}

void Float2EditText(DialogPtr theDialog, short item, double n, short decimals)
{	// JLM changed to a double for accuracy 8/31/99
	char str[32];
	//char format[10];
	//sprintf(format, "%%.%df", decimals);
	//sprintf(str, format, n);

	// JLM, 10/2/98, ignore decimal request, show as many as were entered, up to a max of 6
	// chopping the trailing zeros
	short maxNumDecimals = 6;
	StringWithoutTrailingZeros(str,n,maxNumDecimals);

	mysetitext(theDialog, item, str);
}

void Long2EditText(DialogPtr theDialog, short item, long n)
{
	char str[30];
	
	sprintf(str, "%ld", n);
	mysetitext(theDialog, item, str);
}

long EditText2Long(DialogPtr theDialog, short item)
{
	char str[256];
	short count;
	long n;
	
	mygetitext(theDialog, item, str, 255);
	count = sscanf(str, "%ld", &n);
	return (count == 1) ? n : 0;
}

double EditText2Float(DialogPtr theDialog, short item)
{
	char str[256];
	short count;
	double n;
	
	mygetitext(theDialog, item, str, 255);
	count = sscanf(str,lfFix("%lf"), &n);
	return (count == 1) ? n : 0;
}

void RemoveSpacesAndCommas(DialogPtr theDialog, SHORTPTR items)
{
	char str[256], str2[256];
	short i = 0;
	
	while (items[i] != 0) {
		mygetitext(theDialog, items[i], str, 255);
		RemoveSetFromString(str, " ,", str2);
		mysetitext(theDialog, items[i], str2);
		i++;
	}
} 

void FillScreenWithDialog(WindowPtr w, short titleHeight)
{
	Rect screen = GetScreenBounds();
	
	MyMoveWindow(w, 15, 25 + titleHeight, FALSE);
	SizeWindow(w, RectWidth(screen) - 60,
				  RectHeight(screen) - (25 + titleHeight + 15), FALSE);
}

short CABClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	return itemNum;
}

OSErr CABInit(DialogPtr dialog, VOIDPTR data)
{
	return 0;
}

short ClickAnyButtonDialog(short dialogID, WindowPtr parent)
{
	return MyModalDialog(dialogID, parent, 0, CABInit, CABClick);
}

///// FILES ///////////////////////////////////////////////////////////////////////////

long myfgets(CHARPTR buffer, long n, short f)
{
	long count = 1, i, start;
	
	GetFPos(f, &start);
	FSRead(f, &n, buffer);
	buffer[n] = 0;
	for (i = 0 ; i < n ; i++) if (buffer[i] == RETURN) { /* i++; */ break; }
	SetFPos(f, fsFromStart, start + i + (buffer[i] == RETURN ? 1 : 0));
	buffer[i] = 0;
	
	return i;
}

OSErr RenameFolder(short vRefNum, long dirID, char *newName)
{
	char count, oldFolderName[256], newFolderName[256], suffix[] = " #";
	long end, parentID = ParentFolderID(vRefNum, dirID, 0);
	
	MyGetFolderName(vRefNum, dirID, FALSE, oldFolderName);
	
#ifdef IBM
		suffix[0] = '_';
		end = LongFileNames(vRefNum) ? 255 : 8;
#else
		end = 31;
#endif
	
	strnzcpy(newFolderName, newName, end);
	ZapGremlins(newFolderName);
	
	if (!strcmpnocase(oldFolderName, newFolderName)) return 0;
	
	end = _max(0, _min(end - 2, (long)strlen(newFolderName)));
	for (count = '2' ; count <= '9' && FolderExists(vRefNum, parentID, newFolderName) ; count++) {
		newFolderName[end] = 0;
		suffix[1] = count;
		strcat(newFolderName, suffix);
	}
	
	if (count > '9')
		return -1;
	
	return hrename(vRefNum, parentID, oldFolderName, newFolderName);
}

/*OSErr AppendFile(short f1, short f2)
{
	long bufSize = _max(MyTempMaxMem(), _MaxBlock()) - 100000, err = 0;
	CHARH buf;
	
	if (!(buf = (CHARH)MyNewHandleTemp(bufSize))) return -1;
	
	_HLock((Handle)buf);
	while (!err) {
		err = FSRead(f2, &bufSize, DEREFH(buf));
		err |= FSWrite(f1, &bufSize, DEREFH(buf));
	}
	_HUnlock((Handle)buf);
	DisposeHandle((Handle)buf);
	
	return (err == eofErr) ? 0 : err;
}

OSErr FilterFile(long fromID, short fromVRefNum, CHARPTR from,
					  long toID, short toVRefNum, CHARPTR to, OSType creator, OSType type,
					  long recLength, Boolean (*filterFunc)(VOIDPTR), void (*progress)(short))
{
	long	sizeAvail = (_max(MyTempMaxMem(), _MaxBlock()) - 100000) / 2,
			bufSizeInRecs = (sizeAvail / recLength),
			bufSize = bufSizeInRecs * recLength,
			numRecs, fileLength, count, recsInBuf, i, nextUnfilledRec,
			err = -1;
	short fFrom = 0, fTo = 0;
	Boolean keep;
	CHARH buf = 0;
	
	if (!(buf = (CHARH)MyNewHandleTemp(bufSize))) goto done;
	
	if (err = hopendf(fromVRefNum, fromID, from, fsCurPerm, &fFrom)) goto done;
	if (err = GetEOF(fFrom, &fileLength)) goto done;
	numRecs = fileLength / recLength;
	
	hdelete(toVRefNum, toID, to);
	if (err = hcreate(toVRefNum, toID, to, creator, type)) goto done;
	if (err = hopendf(toVRefNum, toID, to, fsCurPerm, &fTo)) goto done;
	
	while (numRecs) {
		if (progress) (*progress)((fileLength - numRecs * recLength) / _max(fileLength / 100, 1));
		recsInBuf = _min(numRecs, bufSizeInRecs);
		count = recsInBuf * recLength;
		_HLock((Handle)buf); err = FSRead(fFrom, &count, DEREFH(buf)); _HUnlock((Handle)buf);
		if (err) goto done;
		
		_HLock((Handle)buf);
		nextUnfilledRec = 0;
		for (i = 0 ; i < recsInBuf ; i++) {
			keep = (*filterFunc)(DEREFH(buf) + i * recLength);
			if (!keep) continue;
			if (i != nextUnfilledRec)
				_BlockMove(DEREFH(buf) + i * recLength,
							 DEREFH(buf) + nextUnfilledRec * recLength,
							 recLength);
			nextUnfilledRec++;
		}
		count = nextUnfilledRec * recLength;
		err = FSWrite(fTo, &count, DEREFH(buf));
		_HUnlock((Handle)buf);
		if (err) goto done;
		
		numRecs -= recsInBuf;
	}
	
	done:
		if (buf) DisposeHandle((Handle)buf);
		if (fFrom) FSClose(fFrom);
		if (fTo) FSClose(fTo);
		
		return err;
}

///// FQSORT ///////////////////////////////////////////////////////////////////////////

OSErr RefillBuffer(short f, long bufRecs, long numRecs, long recLength, CHARH buf)
{
	long count = _min(numRecs, bufRecs) * recLength;
	
	return FSRead(f, &count, DEREFH(buf));
}

OSErr MergeSortedFiles(long dirID, short vRefNum, CHARPTR file1, CHARPTR file2,
							  CHARPTR to, OSType creator, OSType type,
							  int (*compare)(const void *rec1, const void *rec2), long numRecs1, long numRecs2, long recLength, long memAvail,
							  void (*display)(char *), void (*progress)(short))
{
	short f, f1, f2, err = -1, err2, c;
	long bufRecs = (memAvail / (3 * recLength)), bufIndex1 = 0, bufIndex2 = 0, bufIndex3 = 0,
		 total = numRecs1 + numRecs2, bufSize = bufRecs * recLength, writeSize, n;
	CHARH buf1 = 0, buf2 = 0, buf3 = 0;
	char s[100], format[100];
	
	if (display) {
		getindstring(format, 1000, 2); // "Merging files (%ld + %ld records)."
		sprintf(s, format, numRecs1, numRecs2);
		(*display)(s);
	}
	
	hdelete(vRefNum, dirID, "merge1"); hdelete(vRefNum, dirID, "merge2");
	if (err = hrename(vRefNum, dirID, file1, "merge1")) goto done;
	if (err = hrename(vRefNum, dirID, file2, "merge2")) goto done;
	
	if (err = hopendf(vRefNum, dirID, "merge1", fsCurPerm, &f1)) goto done;
	if (err = hopendf(vRefNum, dirID, "merge2", fsCurPerm, &f2)) goto done;
	
	hdelete(vRefNum, dirID, to);
	if (err = hcreate(vRefNum, dirID, to, creator, type)) goto done;
	if (err = hopendf(vRefNum, dirID, to, fsCurPerm, &f)) goto done;
	
	buf1 = (CHARH)MyNewHandleTemp(bufSize);
	buf2 = (CHARH)MyNewHandleTemp(bufSize);
	buf3 = (CHARH)MyNewHandleTemp(bufSize);
	while (!buf1 || !buf2 || !buf3) {
		if (buf1) { DisposeHandle((Handle)buf1); buf1 = 0; }
		if (buf2) { DisposeHandle((Handle)buf2); buf2 = 0; }
		if (buf3) { DisposeHandle((Handle)buf3); buf3 = 0; }
		bufSize -= recLength;
		bufRecs--;
		if (bufSize < 10000) { err = -1; goto done; }
		buf1 = (CHARH)MyNewHandleTemp(bufSize);
		buf2 = (CHARH)MyNewHandleTemp(bufSize);
		buf3 = (CHARH)MyNewHandleTemp(bufSize);
	}
	
	if (err = RefillBuffer(f1, bufRecs, numRecs1, recLength, buf1)) goto done;
	if (err = RefillBuffer(f2, bufRecs, numRecs2, recLength, buf2)) goto done;
	
	_HLock((Handle)buf1); _HLock((Handle)buf2); _HLock((Handle)buf3);
	while (numRecs1 || numRecs2) {
		n = total - (numRecs1 + numRecs2);
		if (progress) if (n < 10 || (n % 50) == 0) (*progress)((n * 100) / total);
		err2 = 0;
		c = (!numRecs2 ? -1 : (!numRecs1 ? 1 : 0));
		if (!c) c = (*compare)(DEREFH(buf1) + bufIndex1 * recLength,
									  DEREFH(buf2) + bufIndex2 * recLength);
		if (c <= 0) {
			_BlockMove(DEREFH(buf1) + bufIndex1 * recLength,
						 DEREFH(buf3) + bufIndex3 * recLength,
						 recLength);
			bufIndex3++;
			bufIndex1++;
			numRecs1--;
		}
		if (c > 0) {
			_BlockMove(DEREFH(buf2) + bufIndex2 * recLength,
						 DEREFH(buf3) + bufIndex3 * recLength,
						 recLength);
			bufIndex3++;
			bufIndex2++;
			numRecs2--;
		}
		if (bufIndex1 == bufRecs) {
			err2 |= RefillBuffer(f1, bufRecs, numRecs1, recLength, buf1);
			bufIndex1 = 0;
		}
		if (bufIndex2 == bufRecs) {
			err2 |= RefillBuffer(f2, bufRecs, numRecs2, recLength, buf2);
			bufIndex2 = 0;
		}
		if (bufIndex3 == bufRecs || !(numRecs1 || numRecs2)) {
			writeSize = bufIndex3 * recLength;
			err2 |= FSWrite(f, &writeSize, DEREFH(buf3));
			bufIndex3 = 0;
		}
		if (err2) {
			_HUnlock((Handle)buf1); _HUnlock((Handle)buf2); _HUnlock((Handle)buf3);
			err = err2; goto done;
		}
	}
	_HUnlock((Handle)buf1); _HUnlock((Handle)buf2); _HUnlock((Handle)buf3);
	FSClose(f); FSClose(f1); FSClose(f2);
	
	err = 0;
	
	done:
		if (buf1) DisposeHandle((Handle)buf1);
		if (buf2) DisposeHandle((Handle)buf2);
		if (buf3) DisposeHandle((Handle)buf3);
		hdelete(vRefNum, dirID, "merge1");
		hdelete(vRefNum, dirID, "merge2");
		
		return err;
}

static int (*CPPCompare)(const void *rec1, const void *rec2);

int CCompare(const void *rec1, const void *rec2)
{
	return CPPCompare(rec1, rec2);
}

OSErr SortFromTo(long fromID, short fromVRefNum, CHARPTR from,
				 long toID, long toVRefNum, CHARPTR to, OSType creator, OSType type,
				 int (*compare)(const void *rec1, const void *rec2), long recLength, long startRec, long numRecs,
				 long bufSize, void (*display)(char *), void (*progress)(short))
{
	char s[100], format[100];
	short f, err = -1;
	CHARH buf = 0;
#pragma unused(progress)
	
	if (display) {
		getindstring(format, 1000, 1); // "Sorting '%s' from record %ld to record %ld."
		sprintf(s, format, from, startRec + 1, startRec + 1 + numRecs);
		(*display)(s);
	}
	
	buf = (CHARH)MyNewHandleTemp(bufSize);
	if (!buf) goto done;
	
	if (err = hopendf(fromVRefNum, fromID, from, fsCurPerm, &f)) goto done;
	if (err = SetFPos(f, fsFromStart, startRec * recLength)) goto done;
	_HLock((Handle)buf); err = FSRead(f, &bufSize, DEREFH(buf)); _HUnlock((Handle)buf);
	if (err) goto done;
	FSClose(f);
	
	CPPCompare = compare;
	_HLock((Handle)buf);
	qsort((void *)DEREFH(buf), numRecs, recLength, CCompare);
	_HUnlock((Handle)buf);
	
	hdelete(toVRefNum, toID, to);
	if (err = hcreate(toVRefNum, toID, to, creator, type)) goto done;
	if (err = hopendf(toVRefNum, toID, to, fsCurPerm, &f)) goto done;
	_HLock((Handle)buf); err = FSWrite(f, &bufSize, DEREFH(buf)); _HUnlock((Handle)buf);
	if (err) goto done;
	FSClose(f);
	
	err = 0;
	
	done:
		if (buf) DisposeHandle((Handle)buf);
		return err;
}

OSErr fqsortsection(long fromID, short fromVRefNum, CHARPTR from,
						  long toID, short toVRefNum, CHARPTR to, OSType creator, OSType type,
						  int (*compare)(const void *rec1, const void *rec2), long recLength, long startRec, long numRecs,
						  short level, void (*display)(char *), void (*progress)(short))
{
	short err;
	long halfRecs,
		  sizeNeeded = numRecs * recLength,
	     bufSize = _max(0, _max(MyTempMaxMem(), _MaxBlock()) - 100000);
	char nameA[20], nameB[20];
	
	bufSize = _min(bufSize, recLength * 16000); // only sort up to 16000 at a time to conserve stack space
	
	if (sizeNeeded <= bufSize) {
		if (err = SortFromTo(fromID, fromVRefNum, from,
									toID, toVRefNum, to, creator, type,
									compare, recLength, startRec, numRecs, sizeNeeded,
									display, progress)) return err;
	}
	else {
		halfRecs = numRecs / 2;
		sprintf(nameA, "srttempA.%hd", level);
		sprintf(nameB, "srttempB.%hd", level);
		if (err = fqsortsection(fromID, fromVRefNum, from,
										toID, toVRefNum, nameA, creator, type,
										compare, recLength, startRec, halfRecs, level + 1,
										display, progress)) return err;
		if (err = fqsortsection(fromID, fromVRefNum, from,
										toID, toVRefNum, nameB, creator, type,
										compare, recLength, startRec + halfRecs, numRecs - halfRecs,
										level + 1, display, progress)) return err;
		if (err = MergeSortedFiles(toID, toVRefNum, nameA, nameB, to, creator, type, compare, halfRecs, numRecs - halfRecs, recLength, bufSize, display, progress)) return err;
	}
	
	return 0;
}

OSErr fqsort(long fromID, short fromVRefNum, CHARPTR from,
				 long toID, short toVRefNum, CHARPTR to, OSType creator, OSType type,
				 int (*compare)(const void *rec1, const void *rec2), long recLength,
				 void (*display)(char *), void (*progress)(short))
{
	short f, err;
	long fromLength;
	
	if (err = hopendf(fromVRefNum, fromID, from, fsCurPerm, &f)) return err;
	if (err = GetEOF(f, &fromLength)) return err;
	FSClose(f);
	
	return fqsortsection(fromID, fromVRefNum, from,
								toID, toVRefNum, to, creator, type,
								compare, recLength, 0, fromLength / recLength, 0, display, progress);
}*/

///// BUFFERED FILES ///////////////////////////////////////////////////////////////////

static short savedPermission, readOnlyCount = 0, currentPermission = fsCurPerm;

void EnterReadOnly()
{
	readOnlyCount++;
	if (readOnlyCount > 1) return;
	
	savedPermission = currentPermission;
	currentPermission = fsRdPerm;
}

void ExitReadOnly(OSErr (*f)(void))
{
	if (f == 0) readOnlyCount = 0; // special reset flag
	
	if (readOnlyCount > 0) readOnlyCount--;
	if (readOnlyCount > 0) return;
	
	currentPermission = savedPermission;
	if (f) (*f)();
}

long myMin(long a, long b)
{
	if (a < b) return a;
	return b;
}

OSErr FSOpenBuf(short vRefNum, long dirID, CHARPTR fName, BFPBP bfpb, long bufSize, Boolean temp)
{
	char fNameOnCD[256];
	long err;

	if(bufSize < 1000000) bufSize = 1000000; // JLM , 5/3/10 memory is now readily available, we do this here so we don't have to change the code everywhere
	
	if (fName) {
		strcpy(fNameOnCD, fName);
		strcat(fNameOnCD, ";1");
		if ((err = hopendf(vRefNum, dirID, fName, currentPermission, &(bfpb->f))) &&
			(err = hopendf(vRefNum, dirID, fNameOnCD, currentPermission, &(bfpb->f))))
			{ bfpb->f = 0; return err; }
		if (err = GetEOF(bfpb->f, &(bfpb->fileLength)))
			{ FSClose(bfpb->f); bfpb->f = 0; return err; }
	}
	bfpb->buf = 0;
	bfpb->bufSize = 0;
	bfpb->base = 0;
	bfpb->index = -1;
	bfpb->bufModified = 0;
	if (!(bfpb->buf = (temp ? (CHARH)MyNewHandleTemp(bufSize) : (CHARH)_NewHandle(bufSize))))
		 { FSClose(bfpb->f); bfpb->f = 0; return -1; }
	bfpb->bufSize = bufSize;
	
	return 0;
}

OSErr FSReopenBuf(short f, BFPBP bfpb)
{
	long err;
	
	bfpb->f = f;
	if (err = GetEOF(bfpb->f, &(bfpb->fileLength))) return err;
	bfpb->base = 0;
	bfpb->index = -1;
	bfpb->bufModified = 0;
	
	return 0;
}

OSErr FSWriteIfDirtyBuf(BFPBP bfpb)
{
	long err = 0, writeCount;
	
	if (bfpb->index == -1) return 0;
	
	if (bfpb->bufModified) {
		err = SetFPos(bfpb->f, fsFromStart, bfpb->base);
		writeCount = myMin(bfpb->bufSize, bfpb->fileLength - bfpb->base);
		err |= FSWrite(bfpb->f, &writeCount, DEREFH(bfpb->buf));
		bfpb->bufModified = 0;
	}
	
	return err;
}

OSErr FSResizeBuf(BFPBP bfpb, long newSize, Boolean temp)
{
	long err, savePos;
	
	if (bfpb->buf && bfpb->bufSize == newSize) return 0;
	
	if (err = GetFPosBuf(bfpb, &savePos)) return err;
	
	if (err = FSWriteIfDirtyBuf(bfpb)) return err;
	
	if (bfpb->buf) {
		if (_GetHandleSize((Handle)(bfpb->buf)) == newSize) return 0;
		DisposeHandle((Handle)(bfpb->buf));
		bfpb->buf = 0;
		bfpb->bufSize = 0;
	}
	if (!(bfpb->buf = (temp ? (CHARH)MyNewHandleTemp(newSize) : (CHARH)_NewHandle(newSize)))) return -1;
	bfpb->bufSize = newSize;
	bfpb->base = 0;
	bfpb->index = -1;
	bfpb->bufModified = 0;
	
	if (err = SetFPosBuf(bfpb, savePos)) return err;
	
	return 0;
}

OSErr FSPrepareBuf(BFPBP bfpb, long count)
{
	long readCount, err = 0;
	
	if (bfpb->index == -1 || (bfpb->index + count) > bfpb->bufSize) {
		_HLock((Handle)(bfpb->buf));
		err = FSWriteIfDirtyBuf(bfpb);
		if (bfpb->index != -1) bfpb->base += bfpb->index;
		readCount = bfpb->bufSize;
		err |= SetFPos(bfpb->f, fsFromStart, bfpb->base);
		err |= FSRead(bfpb->f, &readCount, DEREFH(bfpb->buf));
		if (err == eofErr) err = 0;
		bfpb->index = 0;
		_HUnlock((Handle)(bfpb->buf));
	}
	
	return err;
}

OSErr FSReadBuf(BFPBP bfpb, LONGPTR count, CHARPTR dest)
{
	long err = 0;
	
	if (*count > bfpb->bufSize) return -1;
	if (err = FSPrepareBuf(bfpb, *count)) return err;
	
	_BlockMove(DEREFH(bfpb->buf) + bfpb->index, dest, *count);
	
	bfpb->index += *count;
	if ((bfpb->base + bfpb->index) > bfpb->fileLength) {
		err = eofErr;
		*count -= ((bfpb->base + bfpb->index) - bfpb->fileLength);
	}
	bfpb->index = myMin(bfpb->index, bfpb->fileLength - bfpb->base);
	
	return err;
}

OSErr FSWriteBuf(BFPBP bfpb, LONGPTR count, CHARPTR src)
{
	long err = 0;
	
	if (*count > bfpb->bufSize) return -1;
	if (err = FSPrepareBuf(bfpb, *count)) return err;

	_BlockMove(src, DEREFH(bfpb->buf) + bfpb->index, *count);
	bfpb->index += *count;
	bfpb->bufModified = 1;
	if ((bfpb->base + bfpb->index) > bfpb->fileLength)
		bfpb->fileLength = bfpb->base + bfpb->index;
	
	return err;
}

OSErr GetFPosBuf(BFPBP bfpb, LONGPTR filePos)
{
	*filePos = bfpb->base + ((bfpb->index == -1) ? 0 : bfpb->index);
	
	return 0;
}

OSErr SetFPosBuf(BFPBP bfpb, long filePos)
{
	long maxPos = myMin(bfpb->fileLength, bfpb->base + bfpb->bufSize), err = 0;
	
	if (filePos < 0) filePos = 0;
	
	if (bfpb->index == -1 || filePos < bfpb->base || filePos > maxPos) {
		err = FSWriteIfDirtyBuf(bfpb);
		bfpb->index = -1;
		bfpb->base = filePos;
	}
	else bfpb->index = filePos - bfpb->base;
	
	return err;
}

OSErr WriteAtPosBuf(BFPBP bfpb, LONGPTR count, CHARPTR src, long pos)
{
	long err, oldPos;
	
	if (err = GetFPosBuf(bfpb, &oldPos)) return err;
	if (err = SetFPosBuf(bfpb, pos)) return err;
	if (err = FSWriteBuf(bfpb, count, src)) return err;
	if (err = SetFPosBuf(bfpb, oldPos)) return err;
	return 0;
}

OSErr FSTransferBuf(BFPBP from, BFPBP to, long start, long end)
{
	long length = end - start, err, count;
	char buffer[1000];
	
	if (err = SetFPosBuf(from, start)) return err;
	while (length) {
		count = _min(length, 1000);
		if (err = FSReadBuf(from, &count, buffer)) return err;
		if (err = FSWriteBuf(to, &count, buffer)) return err;
		length -= count;
	}
	return 0;
}

long myfgetsbuf(CHARPTR buffer, long n, BFPBP file)
{
	long i, start, count = 1;
	
	GetFPosBuf(file, &start);
	FSReadBuf(file, &n, buffer);
	buffer[n] = 0;
	for (i = 0 ; i < n ; i++)
		if (buffer[i] == RETURN || buffer[i] == LINEFEED) { /* i++; */ break; }
	if (buffer[i] == RETURN || buffer[i] == LINEFEED) start++;
	if (buffer[i] == RETURN && buffer[i + 1] == LINEFEED) start++;
	SetFPosBuf(file, start + i);
	buffer[i] = 0;
	
	return i;
}

OSErr FSCloseBuf(BFPBP bfpb)
{
	long err = FSWriteIfDirtyBuf(bfpb);
	
	FSClose(bfpb->f);
	bfpb->f = 0;
	DisposeHandle((Handle)(bfpb->buf));
	bfpb->buf = 0;
	bfpb->index = -1;
	bfpb->base = bfpb->fileLength = bfpb->bufSize = bfpb->bufModified = 0;
	
	return err;
}

///// BLOCK FILES //////////////////////////////////////////////////////////////////////

OSErr ReadFileContents(short terminationFlag, short vRefNum, long dirID, CHARPTR name,
					   VOIDPTR ptr, long length, CHARHP handle) {
	char c;
	int x = 0, i = 0;

	long err = 0;
	char filePath[256], tempFilePath[256];
	Boolean terminate;
	
	if(handle) *handle = 0; //JLM,7/22/98	
	switch(terminationFlag)
	{
		case TERMINATED:
			terminate  = true; break;
		case NONTERMINATED:
			terminate  = false; break;
		default:
			printError("Bad flag in ReadFileContents");return -1;
	}
	
	/////
	if(model && model->fWizard && model->fWizard->PathIsWizardResource(name))
	{	// wizard support, JLM 10/2/98
		err = model->fWizard->ReadFileContentsFromResource(name,handle,terminate);
		return err;
	}
	////////
	if (vRefNum!=0 || dirID!=0) 
	{
		sprintf(tempFilePath, "%s%s", TATFolder, name);
	}
	else
		strcpy(tempFilePath, name);
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) tempFilePath, filePath, kMaxNameLen) ;
#else
		strcpy(filePath,tempFilePath);
#endif
	
	/*FILE *fp;
	long len=0;
	char *buf=0;
	fp=fopen(filePath,"rb");
	fseek(fp,0,SEEK_END); //go to end
	len=ftell(fp); //get position at end (length)
	fseek(fp,0,SEEK_SET); //go to beg.
	//len = len + (terminate ? 1 : 0);
	buf=(char *)malloc(len); //malloc buffer
	//buf = new char[len];
	fread(buf,len,1,fp); //read into buffer
	fclose(fp);	
	
	if(handle) {
		*handle = _NewHandle(len + (terminate ? 1 : 0));
		for(; i < len; i++)
		{
			DEREFH(*handle)[i] = buf[i];
			c = buf[i];
		}
		if (terminate) DEREFH(*handle)[len] = 0;
	} 
	else {
		for(; i < len; i++)
			((char *)ptr)[i] = buf[i];
		if (terminate) ((char *)ptr)[len] = 0;
	}
	if (buf) {free(buf); buf = 0;}
	return noErr;	// check for errors
	*/
	try {
		fstream *_ifstream = new fstream(filePath, ios::in);
		for(; _ifstream->get(c); x++);
		delete _ifstream;
		if(!(x > 0))
			throw("empty file.\n");
		_ifstream = new fstream(filePath, ios::in);
		if(x > length && length != 0)
		    x = length;
		if(handle) {
			*handle = _NewHandle(x + (terminate ? 1 : 0));
			for(; i < x && _ifstream->get(c); i++)
				DEREFH(*handle)[i] = c;
			if (terminate) DEREFH(*handle)[x] = 0;
		} 
		else {
			for(; i < x && _ifstream->get(c); i++)
				((char *)ptr)[i] = c;
			if (terminate) ((char *)ptr)[x] = 0;
		}
		delete _ifstream;
	}
	catch(...) {
		//printError("We are unable to open or read from the file. \nBreaking from ReadSectionOfFile().\n");
		return true;
	}
	return false;
}
/*OSErr ReadFileContents(short terminationFlag, short vRefNum, long dirID, CHARPTR name,
							 VOIDPTR ptr, long length, CHARHP handle)
{	// note: "length" does not include the termination char
	Boolean terminate;
	char nameOnCD[256];
	short f;
	long err = 0;
	
	if(handle) *handle = 0; //JLM,7/22/98	
	switch(terminationFlag)
	{
		case TERMINATED:
			terminate  = true; break;
		case NONTERMINATED:
			terminate  = false; break;
		default:
			printError("Bad flag in ReadFileContents");return -1;
	}
	
	/////
	if(model && model->fWizard && model->fWizard->PathIsWizardResource(name))
	{	// wizard support, JLM 10/2/98
		err = model->fWizard->ReadFileContentsFromResource(name,handle,terminate);
		return err;
	}
	////////
	
	strcpy(nameOnCD, name);
	strcat(nameOnCD, ";1");
	
	if ((err = hopendf(vRefNum, dirID, name, fsRdPerm, &f)) == 0 ||
		(err = hopendf(vRefNum, dirID, nameOnCD, fsRdPerm, &f)) == 0) {
		if (handle) {
			if ((err = GetEOF(f, &length)) == 0) {
				if (*handle = (CHARH)_NewHandle(length + (terminate ? 1 : 0))) {
					_HLock((Handle)(*handle));
					err = FSRead(f, &length, DEREFH(*handle));
					if (terminate) DEREFH(*handle)[length] = 0;
					_HUnlock((Handle)(*handle));
				}
				else err = memFullErr;
			}
		}
		else {
			err = FSRead(f, &length, ptr);
			if (terminate) ((char *)ptr)[length] = 0;
		}
		err |= FSClose(f);
	}
//done:	
	if(err && handle && *handle){DisposeHandle((Handle)*handle); *handle = 0;}//JLM,7/22/98	
	return err;
}


OSErr ReadSectionOfFile(short vRefNum, long dirID, CHARPTR name,
						long offset, long length, VOIDPTR ptr, CHARHP handle)
{
	char nameOnCD[256];
	short f;
	long err = 0;
	
	if(handle) *handle = 0; //JLM,7/22/98	

	/////
	if(model && model->fWizard && model->fWizard->PathIsWizardResource(name))
	{	// wizard support, JLM 10/2/98
		err = model->fWizard->ReadSectionOfFileFromResource(name,(char*) ptr,length,offset);
		return err;
	}
	////////

	strcpy(nameOnCD, name);
	strcat(nameOnCD, ";1");
	
	if ((err = hopendf(vRefNum, dirID, name, fsRdPerm, &f)) == 0 ||
		(err = hopendf(vRefNum, dirID, nameOnCD, fsRdPerm, &f)) == 0) {
		if ((err = SetFPos(f, fsFromStart, offset)) == 0) {
			if (handle) {
				if (*handle = (CHARH)_NewHandle(length)) {
					_HLock((Handle)(*handle));
					err = FSRead(f, &length, DEREFH(*handle));
					_HUnlock((Handle)(*handle));
				}
				else err = -1;
			}
			else
				err = FSRead(f, &length, ptr);
		}
		err |= FSClose(f);
	}
	
	if(err && handle && *handle){DisposeHandle((Handle)*handle); *handle = 0;}//JLM,7/22/98	
	return err;
}*/

OSErr ReadSectionOfFile(short vRefNum, long dirID, CHARPTR name,
						long offset, long length, VOIDPTR ptr, CHARHP handle) {
	char c;
	int x = 0, i = 0;

	long err = 0;
	char filePath[256], tempFilePath[256];
	
	if(handle) *handle = 0; //JLM,7/22/98	

	/////
	if(model && model->fWizard && model->fWizard->PathIsWizardResource(name))
	{	// wizard support, JLM 10/2/98
		err = model->fWizard->ReadSectionOfFileFromResource(name,(char*) ptr,length,offset);
		return err;
	}
	////////
	if (vRefNum!=0 || dirID!=0) 
	{
		sprintf(tempFilePath, "%s%s", TATFolder, name);
	}
	else
		strcpy(tempFilePath, name);
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) tempFilePath, filePath, kMaxNameLen) ;
#else
		strcpy(filePath,tempFilePath);
#endif
	try {
		 ifstream *_ifstream = new ifstream(filePath, ios::in);
		 // get length of file:
		_ifstream->seekg (0, ios::end);
		 int filelength = _ifstream->tellg();
		//_ifstream = new fstream(filePath, ios::in);
		 if(!(filelength > 0))
			throw("empty file.\n");
		_ifstream->seekg (offset, ios::beg);
		if((filelength >= length+offset) && length != 0)
		{
			if(handle) {
				*handle = _NewHandle(length);
				for(; i < length && _ifstream->get(c); i++)
					DEREFH(*handle)[i] = c;
			} 
			else {
				for(; i < length && _ifstream->get(c); i++)
					((char *)ptr)[i] = c;
			}
		}
		else {
			throw("Unable to open file");
		}
		delete _ifstream;
	}
	/*try {
		fstream *_ifstream = new fstream(filePath, ios::in);
		for(; _ifstream->get(c); x++);
		delete _ifstream;
		if(!(x > 0))
			throw("empty file.\n");
		_ifstream = new fstream(filePath, ios::in);
		for(int k = 0; k < offset; k++) _ifstream->get(c); 
		if(x > offset+length && length != 0)
		    x = offset+length;
		if(handle) {
			*handle = _NewHandle(x-offset);
			for(; i < x-offset && _ifstream->get(c); i++)
				DEREFH(*handle)[i] = c;
		} 
		else {
			for(; i < x-offset && _ifstream->get(c); i++)
				((char *)ptr)[i] = c;
		}
		delete _ifstream;
	}*/
	catch(...) {
		//printError("We are unable to open or read from the file. \nBreaking from ReadSectionOfFile().\n");
		return true;
	}
	return false;
}

OSErr WriteFileContents(short vRefNum, long dirID, CHARPTR name, OSType creator, OSType fileType,
						VOIDPTR ptr, long length, CHARH handle)
{
	short f;
	long err, end;
	
	if (err = hopendf(vRefNum, dirID, name, fsCurPerm, &f)) {
		if (err = hcreate(vRefNum, dirID, name, creator, fileType)) return err;
		if (err = hopendf(vRefNum, dirID, name, fsCurPerm, &f)) return err;
	}
	
	if (handle) {
		// length = _GetHandleSize((Handle)handle);
		_HLock((Handle)handle);
		err = FSWrite(f, &length, DEREFH(handle));
		_HUnlock((Handle)handle);
	}
	else
		err = FSWrite(f, &length, ptr);
	err |= GetFPos(f, &end);
	err |= SetEOF(f, end);
	err |= FSClose(f);
	
	return err;
}

OSErr AddToFileContents(short vRefNum, long dirID, CHARPTR name, OSType creator, OSType fileType,
						VOIDPTR ptr, long length, CHARH handle, LONGPTR offset)
{
	short f;
	long err;
	
	if (err = hopendf(vRefNum, dirID, name, fsCurPerm, &f)) {
		if (err = hcreate(vRefNum, dirID, name, creator, fileType)) return err;
		if (err = hopendf(vRefNum, dirID, name, fsCurPerm, &f)) return err;
	}
	
	if (err = SetFPos(f, fsFromLEOF, 0)) return err;
	if (err = GetFPos(f, offset)) return err;
	
	if (handle) {
		// length = _GetHandleSize((Handle)handle);
		_HLock((Handle)handle);
		err = FSWrite(f, &length, DEREFH(handle));
		_HUnlock((Handle)handle);
	}
	else
		err = FSWrite(f, &length, ptr);
	err |= FSClose(f);
	
	return err;
}

OSErr MyCopyFile(short vRefNumFrom, long dirIDFrom, CHARPTR nameFrom,
				 short vRefNumTo, long dirIDTo, CHARPTR nameTo)
{
	short from, to;
	long size, err;
	FInfo fInfo;
	
	if (err = hgetfinfo(vRefNumFrom, dirIDFrom, nameFrom, &fInfo)) return err;
	if (err = MyGetFileSize(vRefNumFrom, dirIDFrom, nameFrom, &size)) return err;
	if (err = hopendf(vRefNumFrom, dirIDFrom, nameFrom, fsRdPerm, &from)) return err;
	
	hdelete(vRefNumTo, dirIDTo, nameTo);
	if (err = hcreate(vRefNumTo, dirIDTo, nameTo, fInfo.fdCreator, fInfo.fdType)) return err;
	if (err = hopendf(vRefNumTo, dirIDTo, nameTo, fsCurPerm, &to)) return err;
	
	if (err = TransferFileSection(from, to, 0, size)) return err;
	
	if (err = FSClose(from)) return err;
	if (err = FSClose(to)) return err;
	
	return 0;
}

OSErr TransferFileSection(short from, short to, long start, long end)
{
	long count, length = end - start, bufSize = 100000, err = 0;
	Ptr buffer = 0;
	
	while (!buffer && bufSize > 0) {
		if (buffer = _NewPtr(bufSize)) break;
		bufSize -= 5000;
	}
	
	if (!buffer) return -1;
	
	if (err = SetFPos(from, fsFromStart, start)) goto done;
	while (length) {
		count = _min(length, bufSize);
		if (err = FSRead(from, &count, buffer)) goto done;
		if (err = FSWrite(to, &count, buffer)) goto done;
		length -= count;
	}
	
	done:
		if (buffer) _DisposePtr(buffer);
		
		return err;
}

OSErr DeleteFileSection(short vRefNum, long dirID, CHARPTR name, long offset, long length)
{
	short from, to;
	long end, err;
	
	if (err = hopendf(vRefNum, dirID, name, fsCurPerm, &from)) return err;
	hdelete(vRefNum, dirID, "trans.tmp");
	if (err = hcreate(vRefNum, dirID, "trans.tmp", 'ttxt', 'TEXT')) return err;
	if (err = hopendf(vRefNum, dirID, "trans.tmp", fsCurPerm, &to)) return err;
	
	if (err = TransferFileSection(from, to, 0, offset)) return err;
	if (err = GetEOF(from, &end)) return err;
	if (err = TransferFileSection(from, to, offset + length, end - (offset + length))) return err;
	
	if (err = FSClose(from)) return err;
	if (err = FSClose(to)) return err;
	
	if (err = hdelete(vRefNum, dirID, name)) return err;
	if (err = hrename(vRefNum, dirID, "trans.tmp", name)) return err;
	
	return 0;
}

///// DATABASES ////////////////////////////////////////////////////////////////////////

/*OSErr OpenDataBase(DataBaseP db, CHARPTR fileName, short vRefNum, long dirID,
						 OSType creator, IndexFileP indexList, long bufSize)
{
	long err;
	short i;
	
	err = hcreate(vRefNum, dirID, fileName, creator, 'DATA');
	if (err = FSOpenBuf(vRefNum, dirID, fileName, &db->bfpb, bufSize, FALSE)) return err;
	for (i = 0 ; indexList[i].name[0] ; i++) {
		err = hcreate(vRefNum, dirID, indexList[i].name, creator, 'INDX');
		if (err = hopendf(vRefNum, dirID, indexList[i].name, fsCurPerm, &indexList[i].fp)) return err;
	}
	strcpy(db->name, fileName);
	db->indexList = indexList;
	
	return 0;
}

OSErr NthByIndex(DataBaseP db, long n, short indexNum, LONGPTR offset)
{
	IndexFileP index = &db->indexList[indexNum];
	long ndxLength = 4 + index->previewLength, four = 4, err;
	
	if (err = SetFPos(index->fp, fsFromStart, ndxLength * n)) return err;
	if (err = FSRead(index->fp, &four, (char *)offset)) {
		if (err = eofErr) { *offset = -1; return 0; }
		return err;
	}
	
	return 0;
}

OSErr SeekByIndex(DataBaseP db, short indexNum, VOIDPTR keyRecord, long recLength,
						LONGPTR offset, LONGPTR indexN)
{
	IndexFileP index = &db->indexList[indexNum];
	long ndxLength = 4 + index->previewLength,
		  err, numIndexRecs, length, low, high, result = 0;
	char ndxBuf[256], recBuf[256];
	
	if (err = GetEOF(index->fp, &length)) return err;
	numIndexRecs = length / ndxLength;
	
	low = -1;
	high = numIndexRecs;
	recLength = _min(256, recLength); // only allow comparisons based on first 256 bytes
	
	for ( ; ; ) {
		if ((high - low) <= 1) {
			*offset = -1;
			if (result == 0) *indexN = 0;
			if (result < 0) (*indexN)++;
			return 0;
		}
		
		*indexN = (low + high) / 2;
		if (err = SetFPos(index->fp, fsFromStart, ndxLength * *indexN)) return err;
		if (err = FSRead(index->fp, &ndxLength, ndxBuf)) return err;
		
		// use index's ComparePreviewData() function against buffer[4]
		
		*offset = *(long *)ndxBuf;
		if (err = ReadRecord(db, *offset, recLength, recBuf)) return err;
		
		result = (*index->compare)(recBuf, keyRecord);
		if (result == 0) return 0;
		if (result > 0) high = *indexN;
		if (result < 0) low = *indexN;
	}
	
	return 0;
}

OSErr ReadRecord(DataBaseP db, long offset, long length, VOIDPTR record)
{
	long err;
	
	if (err = SetFPosBuf(&db->bfpb, offset)) return err;
	if (err = FSReadBuf(&db->bfpb, &length, (char *)record)) return err;
	
	return 0;
}

long FileSize(DataBaseP db)
{
	return db->bfpb.fileLength;
}

OSErr ShiftFileSection(short f, long start, long d)
{
	long bufSize = _max(MyTempMaxMem(), _MaxBlock()) - 100000, err = 0,
		  sectionStart, sectionEnd, end, count;
	CHARH buf;
	
	if (!(buf = (CHARH)MyNewHandleTemp(bufSize))) return -1;
	_HLock((Handle)buf);
	
	if (d > 0) {
		if (err = GetEOF(f, &sectionEnd)) goto done;
		while (sectionEnd > start) {
			sectionStart = _max(sectionEnd - bufSize, start);
			count = sectionEnd - sectionStart;
			if (err = SetFPos(f, fsFromStart, sectionStart)) goto done;
			if (err = FSRead(f, &count, DEREFH(buf))) goto done;
			if (err = SetFPos(f, fsFromStart, sectionStart + d)) goto done;
			if (err = FSWrite(f, &count, DEREFH(buf))) goto done;
			sectionEnd = sectionStart;
		}
	}
	else {
		if (err = GetEOF(f, &end)) goto done;
		sectionStart = start;
		while (sectionStart < end) {
			sectionEnd = _min(start + bufSize, end);
			count = sectionEnd - sectionStart;
			if (err = SetFPos(f, fsFromStart, sectionStart)) goto done;
			if (err = FSRead(f, &count, DEREFH(buf))) goto done;
			if (err = SetFPos(f, fsFromStart, sectionStart + d)) goto done;
			if (err = FSWrite(f, &count, DEREFH(buf))) goto done;
			sectionStart = sectionEnd;
		}
		if (err = SetEOF(f, end + d)) goto done;
	}
	
	done:
		
		_HUnlock((Handle)buf);
		DisposeHandle((Handle)buf);
		
		return err;
}

OSErr InsertRecordIntoIndex(DataBaseP db, short indexNum, VOIDPTR record, long recLength,
									 long offset)
{
	long indexN, dummy, four = 4, err;
	long ndxLength = 4 + db->indexList[indexNum].previewLength;

	if (err = SeekByIndex(db, indexNum, record, recLength, &dummy, &indexN)) return err;
	
	if (err = ShiftFileSection(db->indexList[indexNum].fp, indexN * ndxLength, ndxLength)) return err;
	if (err = SetFPos(db->indexList[indexNum].fp, fsFromStart, indexN * ndxLength)) return err;
	if (err = FSWrite(db->indexList[indexNum].fp, &four, (char *)&offset)) return err;
	// FOR LATER: add preview data
	
	return 0;
}

OSErr DeleteRecordFromIndex(DataBaseP db, short indexNum, VOIDPTR record, long recLength)
{
	long indexN, offset, err;
	long ndxLength = 4 + db->indexList[indexNum].previewLength;

	if (err = SeekByIndex(db, indexNum, record, recLength, &offset, &indexN)) return err;
	if (offset == -1) return 0; // index not found
	
	if (err = ShiftFileSection(db->indexList[indexNum].fp, (indexN + 1) * ndxLength, -ndxLength)) return err;
	
	return 0;
}

OSErr AppendRecord(DataBaseP db, VOIDPTR record, long length)
{
	short i;
	long err, offset = db->bfpb.fileLength;
	
	if (err = SetFPosBuf(&db->bfpb, offset)) return err;
	if (err = FSWriteBuf(&db->bfpb, &length, (char *)record)) return err;
	
	for (i = 0 ; db->indexList[i].name[0] ; i++)
		if (err = InsertRecordIntoIndex(db, i, record, length, offset)) return err;
	
	return 0;
}

OSErr OverwriteRecord(DataBaseP db, VOIDPTR record, long length, long offset)
{
	short i;
	long err;
	
	if (err = SetFPosBuf(&db->bfpb, offset)) return err;
	if (err = FSWriteBuf(&db->bfpb, &length, (char *)record)) return err;
	
	for (i = 0 ; db->indexList[i].name[0] ; i++) {
		if (err = DeleteRecordFromIndex(db, i, record, length)) return err;
		if (err = InsertRecordIntoIndex(db, i, record, length, offset)) return err;
	}
	
	return 0;
}

OSErr CloseDataBase(DataBaseP db)
{
	short i;
	long err;

	if (err = FSCloseBuf(&db->bfpb)) return err;
	for (i = 0 ; (db->indexList)[i].name ; i++)
		if (err = FSClose((db->indexList)[i].fp)) return err;

	return 0;
}*/
