
#include "CROSS.h"
#include "EditWindsDialog.h"

#ifdef MAC

#include <mach/mach_time.h>

	#ifdef MPW //JLM
		QDGlobals qd; // JLM
	#endif
	//#pragma segment Main // JLM 
	#pragma segment CROSS // JLM 
#endif

///// MISC ///////////////////////////////////////////////////////////////////////////

void BasicInit()
#ifdef MAC
{
	void InitColors();
#if MACB4CARBON  
	short i;
	Ptr stackTop;
	// JLMextern _DataInit();
	
	stackTop = GetApplLimit();
	
	
	SetApplLimit((void *)(stackTop - (Ptr)30000));
	// SetApplLimit((void *)(*(long *)csb - 30000));
	MaxApplZone();
	for (i = 1 ; i <= 16 ; i++) MoreMasters();
	
	InitGraf(&qd.thePort);
	InitFonts();
	InitWindows();
	InitMenus();
	TEInit();
	InitDialogs(nil);
	FlushEvents(everyEvent, 0);
#endif
	InitCursor();
	
	InitColors();
}
#else
{
	RegisterStandardWindows();
	RegisterBullsEye();
}
#endif

// Note: on the Mac, a tick is every 60th of a second.
// On the IBM, it is the number of mSecs since the system was booted.
// so to convert, Nticks = Nms * 60 / 1000
// since TickCount() is a carbon function we'll switch to mach_absolute_time() (in nanoseconds) and convert to old style ticks for now
unsigned long MyTickCount()
{
	unsigned long ticks = 0;
#ifdef IBM
   ticks = (unsigned long)(GetTickCount() * .06);
#else
    uint64_t nowTime;
    uint64_t nowTimeNano;
	double conversion;
    static mach_timebase_info_data_t    sTimebaseInfo = {0,0};
	nowTime = mach_absolute_time();
    if ( sTimebaseInfo.denom == 0 ) {
        (void) mach_timebase_info(&sTimebaseInfo);
    }

	conversion = (double) sTimebaseInfo.numer / (double) sTimebaseInfo.denom;
	conversion = 6e-08 * conversion;
	conversion = conversion * (double) nowTime;
	ticks = (unsigned long) conversion;
	
#endif   
   return ticks;
}

#ifdef MAC
	#pragma segment UTILS
#endif

Handle MySetHandleSize(Handle h, long newSize)
{
	_SetHandleSize(h, newSize);
	
	return h;
}

void SetWatchCursor()
{
#ifdef MAC
	if (!settings.inBackground)
		SetCursor(*GetCursor(watchCursor));
#else
	HCURSOR  watchCursorHdl = LoadCursor(NULL, IDC_WAIT);
	
	if (!settings.inBackground)
		if (watchCursorHdl) SetCursor(watchCursorHdl);
#endif
}

void MySetCursor(short id)
#ifdef MAC
{
	SetCursor(*GetCursor(id));
}
#else
{
	HCURSOR c = LoadCursor(hInst, MAKEINTRESOURCE(id));
	
	if (c) SetCursor(c);
}
#endif

Point GetEventWhere(EventRecord e)
{
#ifdef MAC
	return e.where;
#else
	Point p;

	MakeMacPoint(&e.pt, &p);

	return p;
#endif
}

long GetEventWhen(EventRecord e)
{
#ifdef MAC
	return e.when;
#else
	return (long)(e.time * 0.06);
#endif
}

Boolean MouseDownEvent()
#ifdef MAC
{
	EventRecord event;
	
	return GetNextEvent(mDownMask, &event);
	// code goes here, JLM note : this removed the message from the que, I think that is bad !!!
}
#else
{
	MSG message;
	Boolean down;
	
	down = PeekMessage(&message, NULL, WM_LBUTTONDOWN, WM_LBUTTONDOWN, PM_REMOVE);
	if (!down)
		if (PeekMessage(&message, NULL, WM_KEYDOWN, WM_KEYDOWN, PM_REMOVE))
			if (message.wParam == VK_RETURN)
				down = TRUE;
	if (down) SetMouseButton(TRUE);
	
	return down;
}
#endif

Point GetMouseLocal(WindowPtr w)
{
	GrafPtr savePort;
	Point p;
	
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(w);
	GetMouse(&p);
	SetPortGrafPort(savePort);
	
	return p;
}

Boolean WindowObscured(WindowPtr w)
#ifdef MAC
{
#pragma unused(w)
	
	return FALSE;
}
#else
{
	WindowPtr top;
	RECT rW, rC, rAbove, rSect;
	long b;
	
	GetWindowRect(w, &rW);
	GetClientRect(w, &rC);
	b = ((rW.right - rW.left) - (rC.right - rC.left)) / 2;
	rW.left += b;
	rW.right -= b;
	b = ((rW.bottom - rW.top) - (rC.bottom - rC.top)) / 2;
	rW.top += b;
	rW.bottom -= b;
	
	for (top = GetTopWindow(0) ; top && top != w; top = GetWindow(top, GW_HWNDNEXT)) {
		if (!IsWindowVisible(top)) continue;
		GetWindowRect(top, &rAbove);
		IntersectRect(&rSect, &rW, &rAbove);
		if (!IsRectEmpty(&rSect)) return TRUE;
	}
	
	return FALSE;
}
#endif

Boolean FloatingWindowAbove(WindowPtr w)
#ifdef MAC
{
#pragma unused(w)
	
	return FALSE;
}
#else
{
	WindowPtr top;
	RECT Rw;
	
	GetWindowRect(w, &Rw);
	for (top = GetTopWindow(0) ; top && top != w; top = GetWindow(top, GW_HWNDNEXT)) {
		if (!IsWindowVisible(top)) continue;
		return TRUE;
	}
	
	return FALSE;
}
#endif

Boolean WindowNeedsUpdating(WindowPtr w)
#ifdef MAC
{
	#if MACB4CARBON
		return !EmptyRgn(((WindowPeek)w)->updateRgn);
	#else
		RgnHandle windowUpdateRgn = NewRgn();
		OSStatus err = 0;
		Boolean needsUpdating = false;
		if(windowUpdateRgn){
			 err = GetWindowRegion (w,kWindowUpdateRgn,windowUpdateRgn);
			if(!err) needsUpdating = !EmptyRgn(windowUpdateRgn);
			DisposeRgn(windowUpdateRgn); windowUpdateRgn = 0;
		}
		return needsUpdating;
		//return IsWindowUpdatePending(w);
	#endif
}
#else
{
	// EventRecord e;
	
	// return PeekMessage(&e, w, WM_PAINT, WM_PAINT, FALSE);
	RECT R;
	
	return GetUpdateRect(w, &R, FALSE);
}
#endif

Boolean gUseColonIn24HrTime = true; // JLM likes the colon

void Secs2DateStrings(unsigned long seconds,
					  CHARPTR dateLong, CHARPTR dateShort,
					  CHARPTR time24, CHARPTR time12)
#ifdef MAC
{
	Intl0Hndl iZero;
	Intl1Hndl iOne;
	Intl0Rec saveZero;
	Intl1Rec saveOne;
	
	iZero = (Intl0Hndl) GetIntlResource(0);
	iOne  = (Intl1Hndl) GetIntlResource(1);
	
	if (!iZero || !iOne) { SysBeep(1); return; }
	
	saveZero = (**iZero);
	saveOne = (**iOne);
	
	(**iOne).suppressDay = 255; // suppress day
	//if (dateLong) IUDatePString(seconds, longDate, (StringPtr)dateLong, (Handle)iOne);
	if (dateLong) DateString(seconds, longDate, (StringPtr)dateLong, (Handle)iOne);
	
	(**iZero).shrtDateFmt |= dayLdingZ; // use leading zeros
	(**iZero).shrtDateFmt &= ~century; // suppress century
	//if (dateShort) IUDatePString(seconds, shortDate, (StringPtr)dateShort, (Handle)iZero);
	if (dateShort) DateString(seconds, shortDate, (StringPtr)dateShort, (Handle)iZero);
	
	//if (time12) IUTimeString(seconds, FALSE, (StringPtr)time12);
	if (time12) TimeString(seconds, FALSE, (StringPtr)time12, (Handle)iZero);
	
	(**iZero).timeCycle = 0; // 24 hour
	(**iZero).timeFmt |= hrLeadingZ | minLeadingZ; // use leading zeros
	
	if(gUseColonIn24HrTime) (**iZero).timeSep = ':';
	else  (**iZero).timeSep = 0; // no ':'
	
	
	//if (time24) IUTimePString(seconds, FALSE, (StringPtr)time24, (Handle)iZero);
	if (time24) TimeString(seconds, FALSE, (StringPtr)time24, (Handle)iZero);
	
	(**iZero) = saveZero;
	(**iOne) = saveOne;
	
	if (dateLong) my_p2cstr((StringPtr)dateLong);
	if (dateShort) my_p2cstr((StringPtr)dateShort);
	if (time24) my_p2cstr((StringPtr)time24);
	if (time12) my_p2cstr((StringPtr)time12);
}
#else
{
	struct tm *time;
	
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
	time = localtime((long *)&seconds); // gmtime(&seconds);
	if (!time) { SysBeep(1); return; }
	
	if (time->tm_isdst == 1)
	{ 	// we want to show standard time so we need to fake out the daylight savings time
		if(time->tm_hour > 0)
			time->tm_hour--;
		else
		{	
			seconds -= 3600;
			time = localtime((long *)&seconds);
			if(time->tm_isdst == 1) {
				// life is good, both times were daylight savings time
			}
			else
				printError("Programmer error in Secs2DateStrings");
		}
	}
	
	if (dateLong) strftime(dateLong, 32, "%B %d, %Y", time);
	if (dateShort) strftime(dateShort, 32, "%m/%d/%y", time);
	if (time24) 
	{
		if(gUseColonIn24HrTime) strftime(time24, 32, "%H:%M", time);
		else strftime(time24, 32, "%H%M", time);
	}
	if (time12) strftime(time12, 32, "%#I:%M %p", time);
}
#endif

void Secs2DateString(unsigned long seconds, CHARPTR s)
#ifdef MAC
{
	DateTimeRec DTR;
	
	SecondsToDate (seconds, &DTR);
	DTR.year = DTR.year %100;// year 2000 fix , JLM 1/25/99
	sprintf(s, "%02hd/%02hd/%02hd", DTR.month, DTR.day, DTR.year);
}
#else
{
	struct tm *newTime;

	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
	newTime = localtime((long *)&seconds); // gmtime(&seconds)
	newTime->tm_year = newTime->tm_year %100;// year 2000 fix , JLM 1/25/99
	if (newTime)
	{
		if (newTime->tm_isdst == 1)
		{ 	// we want to show standard time so we need to fake out the daylight savings time
			if(newTime->tm_hour > 0)
				newTime->tm_hour--;
			else
			{	
				seconds -= 3600;
				newTime = localtime((long *)&seconds);
				if(newTime->tm_isdst == 1) {
					newTime->tm_year = newTime->tm_year %100;// year 2000 fix , JLM 1/25/99
					// life is good, both times were daylight savings time
				}
				else
					printError("Programmer error in Secs2DateString");
			}
		}
		sprintf(s, "%02ld/%02ld/%02ld", (long)newTime->tm_mon + 1, (long)newTime->tm_mday, (long)newTime->tm_year);
	}
	else
		{ strcpy(s, "???"); SysBeep(1); }
}
#endif

void Secs2DateString2(unsigned long seconds, CHARPTR s)
{ // returns in format: January 2, 1998 
	short day ; //1-31
	short month; // 1-12
	short year4; // 4 digit year
	short hour; // 0-23
	short minute;// 0-59 //
	char str[255];
#ifdef MAC
	DateTimeRec DTR;
	SecondsToDate (seconds, &DTR);
	month = DTR.month;
	day = DTR.day;
	year4 = DTR.year;
	hour = DTR.hour;
	minute = DTR.minute;
#else
	// IBM
	struct tm *newTime;
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
	newTime = localtime((long *)&seconds); // gmtime(&seconds)
	if (newTime)
	{
		if (newTime->tm_isdst == 1)
		{	// we want to show standard time so we need to fake out the daylight savings time
			if(newTime->tm_hour > 0)
				newTime->tm_hour--;
			else
			{	
				seconds -= 3600;
				newTime = localtime((long *)&seconds);
				if(newTime->tm_isdst == 1) {
					// life is good, both times were daylight savings time
				}
				else
					printError("Programmer error in Secs2DateStrings2");
			}
		}
		
		month = newTime->tm_mon + 1;
		day = newTime->tm_mday;
		year4 = newTime->tm_year+1900; // year 2000 OK
		hour = newTime->tm_hour;
		minute = newTime->tm_min;
	}
	else
	{ strcpy(s, "???"); return; }

#endif
	switch (month)
	{
		case  1: strcpy (s, "January "); break;
		case  2: strcpy (s, "February "); break;
		case  3: strcpy (s, "March "); break;
		case  4: strcpy (s, "April "); break;
		case  5: strcpy (s, "May "); break;
		case  6: strcpy (s, "June "); break;
		case  7: strcpy (s, "July "); break;
		case  8: strcpy (s, "August "); break;
		case  9: strcpy (s, "September "); break;
		case 10: strcpy (s, "October "); break;
		case 11: strcpy (s, "November "); break;
		case 12: strcpy (s, "December "); break;
	}

	sprintf(str, "%02hd, %hd ", day, year4);
	strcat (s, str);
	sprintf (str, "%2.2d:%2.2d", hour, minute);
	strcat (s, str);
}

unsigned long DateString2Secs(CHARPTR s)
#ifdef MAC
{
	DateTimeRec DTR;
	unsigned long seconds;
	
	sscanf(s, "%hd/%hd/%hd", &DTR.month, &DTR.day, &DTR.year);
	if(DTR.year < 40) DTR.year += 2000;// year 2000 solution, JLM 1/25/99
	if (DTR.year < 200) DTR.year += 1900;
	DTR.hour = DTR.minute = DTR.second = 0;
	
	DateToSeconds (&DTR, &seconds);
	
	return seconds;
}
#else
{
	short month, day, year;
	unsigned long seconds;
	long long_secs;
	struct tm newTime;

	sscanf(s, "%hd/%hd/%hd", &month, &day, &year);
	newTime.tm_sec = newTime.tm_min = newTime.tm_hour = 0;
	newTime.tm_mday = day;
	newTime.tm_mon = month - 1;
	if(year < 40) year+= 100;// year 2000 solution, JLM 1/25/99
	newTime.tm_year = (year < 200) ? year : year - 1900;
	//newTime.tm_isdst = -1; // let mktime() determine if it's daylight savings time
	newTime.tm_isdst = 0; // use standard time to avoid duplicates at changeover 4/12/01

	long_secs = mktime(&newTime);
	//seconds = mktime(&newTime);
	//if (seconds == -1) SysBeep(1);
	if (long_secs==-1) 
	{
		long_secs=0; 
		printNote("The Windows function localtime() does not accept dates earlier than January 1,  00:00 GMT");
		SysBeep(1);
	}	// here check if year < 1970 and convert
	seconds = long_secs;
	seconds += 2082816000L; // convert from seconds since 1970 to seconds since 1904
	
	return seconds;
}
#endif

char *Date2String(DateTimeRec *time, char *s)
{
	sprintf(s, "%02hd/%02hd/%02hd %02hd:%02hd:%02hd",
			   time->month, time->day, time->year,
			   time->hour, time->minute, time->second);
	
	return s;
}

#ifdef MAC

//THPrint gPrRecHdl = 0;

/*void InitPrinting()
{
	// gPrRecHdl = 0; // (THPrint)Get1Resource('PRNT', 1000);
	
	if (!gPrRecHdl) {
		if (!(gPrRecHdl = (THPrint)_NewHandle(sizeof(TPrint)))) { SysBeep(1); return; }
		PrOpen(); // if (err = PrError()) { SysBeep(1); return; }
		PrintDefault(gPrRecHdl);
		PrClose();
		// addresource((Handle)gPrRecHdl, 'PRNT', 1000, "");
	}
	
	PrOpen(); // if (err = PrError()) { SysBeep(1); return; }
	PrValidate(gPrRecHdl); 	
	PrClose();
}*/

/*OSErr DoPrintFunction(short function, Boolean *accepted)
{
	(*accepted) = FALSE;
	
	InitCursor();
	PrOpen(); // if (err = PrError()) return err;
	PrValidate(gPrRecHdl); 	
	switch (function) {
		case STYLEDIALOG:
			if (PrStlDialog(gPrRecHdl)) {
				(*accepted) = TRUE;
				// ChangedResource((Handle)gPrRecHdl);
			}
			break;
		case JOBDIALOG:
			if (PrJobDialog(gPrRecHdl)) {
				(*accepted) = TRUE;
				// ChangedResource((Handle)gPrRecHdl);
			}
			break;
	}
	PrClose();
	
	return 0;
}*/

OSErr PrintFred(Ptr title, long iconNum, Handle textHnd,short printingFont,short printingSize)
{
  	Boolean     	accepted;
  	OSErr     		err;
	FontInfo		Finfo;
	long			firstLineOnNextPage;
	long			h;
	short			i;
	Rect			iconRect;
	Handle			iconHdl;
	int				lineHt;
	short			*lineStarts;
	//TPPrPort		myPrPort;
	short			numLinesPerPage;
	short			numLinesToPrint;
	short			numOfPages;
	long			pageNum;
	char			pageNumStr[20];
	//TPrStatus		prStatus;
	short			saveFont;
	short			saveSize;
	GrafPtr  		savePort;
	const long		spaceAtTop = 60;
	const long 		spaceAtBottom = 40; // 0
	const long		spaceAtRight = 40;
	const long 		spaceAtLeft = 40;
	char			str[20];
	long			stringLength;
	TEHandle		textTEHdl;
	long			textLen;
	Ptr				textPtr;
	Rect			textRect;
	long			width;
	#if MACB4CARBON
		TPPrPort      	myPrPort = 0;
		TPrStatus     	prStatus;
	#else
		Boolean sessionDocumentWasAlreadyOpen = gSessionDocumentIsOpen;
	#endif
	
	InitCursor();
	
 	GetPortGrafPtr(&savePort);
#if TARGET_API_MAC_CARBON
	saveFont = GetPortTextFont(savePort);
	saveSize = GetPortTextSize(savePort);
#else
	saveFont = savePort->txFont;
	saveSize = savePort->txSize;
#endif
 	
	err = OpenPrinterAndValidate();
	if (err) goto closePrinter;
	
	err = DoJobPrintDialog(title);
	if (err) goto closePrinter;
	
	SetWatchCursor(); 
#if TARGET_API_MAC_CARBON
	if(!sessionDocumentWasAlreadyOpen) {
		err = My_PMSessionBeginDocument();
		if(err) goto closeDocument;
	}
	textRect = GetPrinterPageRect();
#else
	myPrPort = PrOpenDoc(gPrRecHdl, nil, nil);
	err = PrError();
	if(err) goto closeDocument;
	textRect = (*gPrRecHdl)->prInfo.rPage;
#endif	
	TextFontSize(printingFont,printingSize);
	GetFontInfo(&Finfo);
	lineHt = Finfo.ascent + Finfo.descent + Finfo.leading;
	
	textRect.top += spaceAtTop;
	textRect.bottom -= spaceAtBottom;	
	textRect.bottom -= (textRect.bottom-textRect.top) % lineHt;
	textRect.left += spaceAtLeft;
	textRect.right -= spaceAtRight;
	
	_HLock(textHnd);
	textPtr = *textHnd;
	stringLength = strlen(textPtr);
	textLen = _GetHandleSize(textHnd);
	
	/* Because we are not sure if the text handle includes */
	/* a null terminator, we use the minimum of these numbers. */
	/* If it did not include the null terminator, stringLength is too big. */
	/* If it did contain the null terminator then textLen is too big. */
	
	if (stringLength < textLen) textLen = stringLength;
	
	textTEHdl = TENew(&textRect,&textRect);
	if (textTEHdl != nil) {
		TESetText(textPtr,textLen,textTEHdl);
		_HLock((Handle)textTEHdl);
		
		numLinesPerPage = (textRect.bottom - textRect.top) / lineHt;
		numLinesToPrint = (*textTEHdl)->nLines;
		numOfPages = ceil(((float) numLinesToPrint)/numLinesPerPage);
		lineStarts = &((*textTEHdl)->lineStarts[0]);
		
		/* Draw the rest of the pages in reverse order so they come out in the correct order. */
		pageNum = 0;
		
		for (i = 0 ; i < numLinesToPrint ; i += numLinesPerPage) 
		{
			#if	TARGET_API_MAC_CARBON
				err = PMSessionError(gPrintSession);
			#else
				err = PrError();
			#endif
			if (err == noErr) {
				#if	TARGET_API_MAC_CARBON
					err = PMSessionBeginPage(gPrintSession,gPageFormat, NULL);
					if (!err)
					{
						CGrafPtr printingContext = NULL;
						err = PMSessionGetGraphicsContext (gPrintSession,kPMGraphicsContextQuickdraw,(void**)&printingContext);
						if(err) goto closePage;
						SetPort ((CGrafPtr) printingContext);
					}
				#else
				PrOpenPage(myPrPort,nil);
					err = PrError();
					if(err) goto closePage;
				#endif
				pageNum = pageNum + 1;

				/* Draw title, icon and page__of__ at the top, and then do textbox. */

				/* Draw the title at the top left-hand corner of page. */
				//TextFont(systemFont);		/* chicago font */
				//TextSize(0);				/* default value 0 is fontsize 12 */
				// note: Chicago font is not supported on lazer writers
				TextFontSizeFace(kFontIDGeneva,12,bold);
			
				MoveTo(textRect.left,textRect.top-8);
				if (title != nil) drawstring(title);					
				
				/* Draw a line across the top of the page. */
				MoveTo(textRect.left, textRect.top-1);
				LineTo(textRect.right, textRect.top-1);
				
				/* Draw the "Page__of__" string */
				getindstring(pageNumStr, DLG_STRINGS, PAGE_STRING); // "Page "
				numtostring(pageNum, (CHARPTR) str);
				strcat(pageNumStr, str);
				getindstring(str, DLG_STRINGS, OF_STRING); // " of "
				strcat(pageNumStr, str);
				numtostring(numOfPages, (CHARPTR) str);
				strcat(pageNumStr, str);
						
				/* Move pen to correct position according to "Page__of__" string width */
				TextFontSizeFace(kFontIDMonaco,9,normal);
				width = stringwidth(pageNumStr);	
				h = textRect.right-(8)-(32+10+width); 	/* for "Page__of__" on far-right side */
				MoveTo(h, textRect.top-8);
				drawstring(pageNumStr);  
				
				if (iconNum) {
					iconHdl = GetIcon(iconNum);
					if ( iconHdl != nil) {
						MySetRect(&iconRect, textRect.right-(8)-(32), textRect.top-(4+32),
												 textRect.right-(8), textRect.top-4);
						_HLock(iconHdl); 
						//PlotIcon(&iconRect, iconHdl);
						// For some reason PlotIcon was not working in OS X but PlotIconHandle worked
						PlotIconHandle(&iconRect,  kAlignAbsoluteCenter, kTransformNone, iconHdl);
						_HUnlock(iconHdl); 		   
					}
				}
				
				firstLineOnNextPage = i+numLinesPerPage;
				if (firstLineOnNextPage > numLinesToPrint) {
					/* Set it equal to the number of lines to print. */
					firstLineOnNextPage = numLinesToPrint;
				}
				TextFontSize(printingFont,printingSize);
				TETextBox(&textPtr[lineStarts[i]],lineStarts[firstLineOnNextPage] - lineStarts[i],&textRect,teJustLeft);
			closePage:	
				#if	TARGET_API_MAC_CARBON
					err = PMSessionEndPage(gPrintSession);
				#else
				PrClosePage(myPrPort);
				#endif
				if(err) break;
			}
		}
	}
						
closeDocument:
						
	#if	TARGET_API_MAC_CARBON
		// Close the print job.  This dismisses the progress dialog on Mac OS X.
		if(!sessionDocumentWasAlreadyOpen) err = My_PMSessionEndDocument();
	#else
	PrCloseDoc(myPrPort);
	
	if (((*gPrRecHdl)->prJob.bJDocLoop == bSpoolLoop) && PrError() == noErr)
	{
		PrPicFile(gPrRecHdl,nil,nil,nil,&prStatus);
			err = PrError();
		}
	#endif
closePrinter:
			
	ClosePrinter();
	

	InitCursor();
			
//PrClose();
	
	_HUnlock(textHnd);
	if (textTEHdl != nil) {
		_HUnlock((Handle)textTEHdl);
		TEDispose(textTEHdl);
	}
	
	SetPortGrafPort(savePort);
	TextFontSize(saveFont,saveSize);
	return(noErr);
}
#endif

OSErr PrintText(char *title, Handle textHnd, short ICONid)
#ifdef MAC
{
	return PrintFred(title, ICONid, textHnd, kFontIDGeneva, 9);
}
#else
{
	// (MIKE) print the text with 'title' and icon 'ICONDid' at the top of the page

	return 0;
}
#endif

Boolean EqualPoints(Point p1, Point p2)
{
	return p1.h == p2.h && p1.v == p2.v;
}

Boolean EqualRects(Rect r1, Rect r2)
{
	return r1.top == r2.top && r1.left == r2.left &&
		   r1.bottom == r2.bottom && r1.right == r2.right;
}

Boolean EqualRGBColors(RGBColor c1, RGBColor c2)
{
#ifdef MAC
	return c1.red == c2.red && c1.green == c2.green && c1.blue == c2.blue;
#else
	return c1 == c2;
#endif
}

char *GS1(short id, char *s)
{
	static char s2[256];
	
#ifdef MAC
	getindstring(s ? s : s2, 25000 + (id / 100) * 100, (id % 100) + 1);
#else
	getindstring(s ? s : s2, 25000, id);
#endif
	
    return s ? s : s2;
}

char *GS2(short id, char *s)
{
	static char s2[256];
	
#ifdef MAC
	getindstring(s ? s : s2, 25000 + (id / 100) * 100, (id % 100) + 1);
#else
	getindstring(s ? s : s2, 25000, id);
#endif
    
    return s ? s : s2;
}

char *GS3(short id, char *s)
{
	static char s2[256];
	
#ifdef MAC
	getindstring(s ? s : s2, 25000 + (id / 100) * 100, (id % 100) + 1);
#else
	getindstring(s ? s : s2, 25000, id);
#endif
    
    return s ? s : s2;
}

char *ZapGremlins(char *s)
#ifdef MAC
{
	if (!s) return 0;
	
	StringSubstitute(s, ':', '-');
	
	return s;
}
#else
{
	if (!s) return 0;
	
	StringSubstitute(s, '"', '\'');
	StringSubstitute(s, ' ', '_');
	StringSubstitute(s, '\\', '_');
	StringSubstitute(s, '/', '_');
	StringSubstitute(s, ';', '_');
	StringSubstitute(s, ':', '_');
	StringSubstitute(s, '*', '_');
	StringSubstitute(s, '?', '_');
	StringSubstitute(s, '|', '_');
	StringSubstitute(s, '+', '_');
	StringSubstitute(s, '=', '_');
	StringSubstitute(s, '[', '\'');
	StringSubstitute(s, ']', '\'');
	StringSubstitute(s, '>', '\'');
	StringSubstitute(s, '<', '\'');
	
	return s;
}
#endif

unsigned short SwapShort(unsigned short n)
{
///#ifdef MAC
	//return n;
//#else
#ifdef SWAP_BINARY
	//unsigned char hi = LOBYTE(n), low = HIBYTE(n);
	
	//return (hi << 8) + low;
	char ch;
	char *p = (char*)&n;

	ch = p[0];
	p[0] = p[1];
	p[1] = ch;
#endif
	return n;
}

unsigned long SwapLong(unsigned long n)
{
//#ifdef MAC
	//return n;
//#else
#ifdef SWAP_BINARY
	//unsigned long hi = SwapShort(LOWORD(n)), low = SwapShort(HIWORD(n));
	
	//return (hi << 16) + low;
	char ch;
	char *p = (char*)&n;

	ch = p[0];
	p[0] = p[3];
	p[3] = ch;
	
	ch = p[1];
	p[1] = p[2];
	p[2] = ch;
#endif
	return n;
}

void SwapFloat(float *f)
{
//#ifdef MAC
//#pragma unused(f)
//#else
#ifdef SWAP_BINARY
	char c, *p = (char *)f;
	
	c = p[0];
	p[0] = p[3];
	p[3] = c;
	
	c = p[1];
	p[1] = p[2];
	p[2] = c;
#else
	#pragma unused(f)
#endif
}

void TextSizeSmall()
{
#ifdef IBM
	/*
 	GetWindowRect(GetDesktopWindow(), &R);
	TextSize((R.right - R.left) < 800 ? 7 : 9); // use smaller font in low-res
	*/
	TextSize(11);
#else
	TextSize(9);
#endif
}

void TextSizeTiny()
{
#ifdef IBM
	//TextSize(8);
	TextSize(6);	// to mimic cats for contour labeling
#else
	TextSize(9);
#endif
}


///// GRAPHICS ///////////////////////////////////////////////////////////////////////////

#ifdef MAC

RGBColor colors[] = {
	// THE VALUES IN THIS TABLE ARE BOGUS,
	// SEE THE clut 1000 RESOURCE FOR THE ACTUAL COLORS
	//red		green		blue
	{ 0,		0,			0 },		// dummy
	{ 0, 		0, 			0 },		// black
	{ 65535, 	65535, 		65535 },	// white
	{ 50000, 	50000, 		50000 },	// dark gray
	{ 42831, 	43263, 		43263 },	// gray
	{ 56050, 	56050, 		56050 },	// light gray
	{ 30975, 	18796, 		12449 },	// brown
	{ 57343, 	41929, 		27043 },	// light brown (olive)
	{ 7891, 	24063, 		12421 },	// dark green
	{ 11487,	44543,		8268 },		// green
	{ 14680,	54114,		56831 },	// light blue
	{ 11993, 	9715, 		53503 },	// blue
	{ 14680,	54114,		56831 },	// dark blue
	{ 23529,	7709,		33023 },	// purple
	{ 65535,	9830,		59157 },	// pink
	{ 53247,	4257,		4764 },		// red
	{ 65535,	64593,		13498 },	// yellow
	{ 0,		0,			0 }			// most recent other
};

void InitColors()
{
	short i;
	CTabHandle colorTable;
	
	colorTable = (CTabHandle)GetResource('clut', 1000);
	if (!colorTable)
		{ SysBeep(1); return; }
	
	_HLock((Handle)colorTable);
	for (i = 1 ; i <= 16 ; i++)
		colors[i] = (**colorTable).ctTable[i - 1].rgb;
	_HUnlock((Handle)colorTable);
	ReleaseResource((Handle)colorTable);
}

RGBColor MyPickColor(RGBColor color, WindowPtr w)
{
	char prompt[] = "";
	Point where = { 0, 0 };
	RGBColor outColor;
#pragma unused(w)
	
	if (GetColor(where, (unsigned char const *)prompt, &color, &outColor)) return outColor;
	
	return color;
}

#else

RGBColor colors[] = {
	//red		green		blue	// IBM Color
	RGB(0,		0,			0),		// dummy
	RGB(0,		0,			0),		// black
	RGB(255,	255,		255),	// white
	RGB(96,		96,			96),	// dark gray
	RGB(128,	128,		128),	// gray
	RGB(192,	192,		192),	// light gray
	RGB(128,	0,			0),		// brown
	RGB(220,	201,		169),	// light brown (olive)
	RGB(0,		128,		0),		// dark green
	RGB(0,		255,		0),		// green
	RGB(0,		255,		255),	// light blue
	RGB(0,		0,			255),	// blue
	RGB(0,		0,			128),	// dark blue
	RGB(128,	0,			128),	// purple
	RGB(255,	0,			255),	// pink
	RGB(255,	0,			0),		// red
	RGB(255,	255,		0),		// yellow
	RGB(0,		0,			0)		// most recent other
};

RGBColor customColors[] = {
	RGB(0,		0,			0),		// black
	RGB(255,	255,		255),	// white
	RGB(96,		96,			96),	// dark gray
	RGB(128,	128,		128),	// gray
	RGB(192,	192,		192),	// light gray
	RGB(128,	0,			0),		// brown
	RGB(220,	201,		169),	// light brown (olive)
	RGB(0,		128,		0),		// dark green
	RGB(0,		255,		0),		// green
	RGB(0,		255,		255),	// light blue
	RGB(0,		0,			255),	// blue
	RGB(0,		0,			128),	// dark blue
	RGB(128,	0,			128),	// purple
	RGB(255,	0,			255),	// pink
	RGB(255,	0,			0),		// red
	RGB(255,	255,		0),		// yellow
};

RGBColor MyPickColor(RGBColor color, WindowPtr w)
{
	CHOOSECOLOR cc;
	
    cc.lStructSize = sizeof(CHOOSECOLOR);
    cc.hwndOwner = w;
    cc.hInstance = 0;
    cc.rgbResult = color;
    cc.lpCustColors = customColors;
    cc.Flags = CC_FULLOPEN | CC_RGBINIT;
    cc.lCustData = 0;
    cc.lpfnHook = 0;
    cc.lpTemplateName = 0;
	
	if (ChooseColor(&cc)) return cc.rgbResult;
	
	return color;
}

#endif

#ifdef MAC

void MyMoveTo(short x, short y) { MoveTo(x, y); }
void MyLineTo(short x, short y) { LineTo(x, y); }
void MyFrameRect(RECTPTR r) { FrameRect(r); }
void MyFillRect(RECTPTR r, Pattern p) { FillRect(r, (ConstPatternParam)&p); }
void MyInvertRect(RECTPTR r) { InvertRect(r); }
void MyInsetRect(RECTPTR r, short h, short v) { InsetRect(r, h, v); }
void MyPaintRgn(RgnHandle r) { PaintRgn(r); }
void MyFillRgn(RgnHandle r, Pattern p) { FillRgn(r, (ConstPatternParam)&p); }
void MyFrameRgn(RgnHandle r) { FrameRgn(r); }
void MyShowWindow(WindowPtr w) { ShowWindow(w); }

void TextAngle(long angle)
{
#pragma unused(angle)
}

#endif

Rect GetScreenBounds()
#ifdef MAC
{
	Rect bounds;
	//return qd.screenBits.bounds;
	#if TARGET_API_MAC_CARBON
		BitMap qdScreenBits;
		GetQDGlobalsScreenBits(&qdScreenBits);
		bounds = qdScreenBits.bounds;
	#else
		bounds = qd.screenBits.bounds;
	#endif	
	
	return bounds;
}
#else
{
	RECT R;
	Rect r;

	GetWindowRect(GetDesktopWindow(), &R);
	MakeMacRect(&R, &r);

	return r;
}
#endif

void GlobalToLocalRect (Rect *globalRect)
{
	Point pt1,pt2;
	SetPt(&pt1,globalRect -> left,globalRect -> top);
	SetPt(&pt2,globalRect -> right,globalRect -> bottom);
	GlobalToLocal (&pt1);
	GlobalToLocal (&pt2);
	MySetRect(globalRect,pt1.h,pt1.v,pt2.h,pt2.v);
}

short PixelsPerInchScreen()
#ifdef MAC
{
	short h, v;
	
	ScreenRes(&h, &v);
	
	return h;
}
#else
{
	HDC hDC = GetDC(GetDesktopWindow());
	short result = GetDeviceCaps(hDC, LOGPIXELSX);

	ReleaseDC(GetDesktopWindow(), hDC);

	return result;
}
#endif

short PixelsPerInchCurrent()
#ifdef MAC
{
	short h, v;
	
	ScreenRes(&h, &v);
	
	return h;
}
#else
{
	return makingMetaFile ? PixelsPerInchScreen() :
							GetDeviceCaps(currentHDC, LOGPIXELSX);
}
#endif

void MyOffsetRect(RECTPTR r, short dh, short dv)
#ifdef MAC
{
	OffsetRect(r, dh, dv);
}
#else
{
	r->left += dh;
	r->right += dh;
	r->top += dv;
	r->bottom += dv;
}
#endif

RgnHandle MyRectRgn(RgnHandle rgn, RECTPTR r)
#ifdef MAC
{
	RectRgn(rgn, r);
	
	return rgn;
}
#else
{
	DisposeRgn(rgn);

	return CreateRectRgn(r->left, r->top, r->right, r->bottom);
}
#endif

/////////////
void MyUnionRect(Rect *src1, Rect *src2, Rect *dstRect)
{ 
#ifdef IBM
	dstRect -> top    = _max (src1 -> top,    src2 -> top);
	dstRect -> right  = _max (src1 -> right,  src2 -> right);
	dstRect -> bottom = _min (src1 -> bottom, src2 -> bottom);
	dstRect -> left   = _min (src1 -> left,   src2 -> left);
#else
	UnionRect(src1,src2,dstRect);
#endif
	
	
}
/////////////////
#ifdef IBM
Boolean EmptyRect(Rect *r)
{
	Boolean emptyRect = false;
	
	//if (r -> top <= r -> bottom)
	if (r -> bottom <= r -> top)	// CMO 8/3/01
		emptyRect = true;
	
	if (r -> right <= r -> left)
		emptyRect = true;
	
	return (emptyRect);
}
#endif 

//////////////////////
static Rect currentClipRect = { -32001, -32001, -32001, -32001 };
Rect bigRect = { -32001, -32001, 32001, 32001 };
#ifdef MAC
	static GrafPtr sPreviousClipPort = 0; // JLM 2/3/99
#else
	static HDC sPreviousClipPort = 0; // JLM 2/3/99
#endif
void ClearClipRect(void)
{	// clears clip rect on IBM
	// sets it to a big rect on the MAC
	//sPreviousClipPort = 0; //  to force the clip past the check
	MyClipRect(bigRect);
}

#ifdef MAC
Rect MyClipRect(Rect r)
{
	Rect oldClipRect;
	GrafPtr theCurrentPort;

	// JLM 2/3/99, don't return unless you are sure the port is the same as well !!
	//if (EqualRects(r, currentClipRect)) return currentClipRect;
	GetPortGrafPtr(&theCurrentPort);
	if (sPreviousClipPort == theCurrentPort && EqualRects(r, currentClipRect)) return currentClipRect;
	
	if (r.left >= -32000)
		ClipRect(&r);
	else {
		ClipRect(&bigRect);
	}
	
	oldClipRect = currentClipRect;
	currentClipRect = r;
	sPreviousClipPort = theCurrentPort; // JLM 2/3/99
	
	return oldClipRect;
}
#else
Rect MyClipRect(Rect r)
{
	Rect oldClipRect;
	RgnHandle rgn;
	
	// if (!MyIsWindow(currentWindow)) return r;
	
	// JLM 2/3/99, don't return unless you are sure the port is the same as well !!
	//if (EqualRects(r, currentClipRect)) return currentClipRect;
	if (sPreviousClipPort == currentHDC && EqualRects(r, currentClipRect)) return currentClipRect;
	
	if (r.left >= -32000) {
		rgn = CreateRectRgn(r.left, r.top, r.right, r.bottom);
		SelectClipRgn(currentHDC, rgn);
		DeleteObject(rgn);
	}
	else
		SelectClipRgn(currentHDC, 0);
	
	oldClipRect = currentClipRect;
	currentClipRect = r;
	sPreviousClipPort = currentHDC; // JLM 2/3/99
	
	return oldClipRect;
}
#endif

/////////////////////////////////////////////////
/////////////////////////////////////////////////

Rect GetResBitmapRect(short resID)
{
	 Rect frame = {0,0,0,0};
	 short width = 0,height = 0;
#ifdef MAC
	PicHandle p = GetPicture(resID);
#else
	HBITMAP p = LoadBitmap(hInst, MAKEINTRESOURCE(resID));
#endif

	if(p)
	{
		#ifdef MAC
			//frame = (*p)->picFrame;
			frame = GetPICTFrame(p);
		#else
			BITMAP b;
			GetObject(p, sizeof(BITMAP), (LPSTR)&b);
			frame.right = b.bmWidth;
			frame.bottom = b.bmHeight;
			DeleteObject(p); p = 0;
		#endif
		
	}
	return frame;
}

// JLE 1/5/99 created new routine DrawResBitMap2 to  draw pict using its frame size
// return bounding rect of pict in iconRect
void DrawResBitmap2(short resID, Rect *iconRect, unsigned long mode)
{
	 short width = 0,height = 0;
#ifdef MAC
	#pragma unused(mode)
	PicHandle p = GetPicture(resID);
#else
	HBITMAP p = LoadBitmap(hInst, MAKEINTRESOURCE(resID));
#endif

	if(p)
	{
		#ifdef MAC
			Rect frame = GetPICTFrame(p);
			//width = RectWidth((*p)->picFrame);
			//height = RectHeight((*p)->picFrame);
			width = RectWidth(frame);
			height = RectHeight(frame);
		#else
			BITMAP b;
			GetObject(p, sizeof(BITMAP), (LPSTR)&b);
			width = b.bmWidth;
			height = b.bmHeight;
		#endif
		
	}
	
	iconRect->right = iconRect->left + width;
	iconRect->bottom = iconRect->top + height;

	if(p)
	{
		#ifdef MAC
			DrawPicture(p, iconRect);
		#else
			if(sharedPrinting)
			{
				RECT DCRect,DDBRect;
				
				DDBRect.left = DDBRect.top = 0;
				DDBRect.right = width;
				DDBRect.bottom = height;
				
				// get destination size in printer pixels
				DCRect.top = DCRect.left = 0;
				DCRect.right = GetSystemMetrics(SM_CXICON);
				DCRect.bottom =GetSystemMetrics(SM_CYICON);
				
				PaintBitmap(currentHDC,&DCRect,p,&DDBRect,0);
			
			}
			else
				DrawBitmap(p, iconRect->left, iconRect->top, mode);
			
			DeleteObject(p); p = 0;
		#endif
	}
}


void DrawResBitmap(short resID, Rect frame, unsigned long mode)
#ifdef MAC
{
	PicHandle p;
#pragma unused(mode)
	
	if (p = GetPicture(resID))
		DrawPicture(p, &frame);
	else
	{
		//FillRect(&frame, (ConstPatternParam)&qd.gray);
		FillRectWithQDGlobalsGray(&frame);
	}
}
#else
{
	HBITMAP p;
	
	if (p = LoadBitmap(hInst, MAKEINTRESOURCE(resID))) {
		DrawBitmap(p, frame.left, frame.top, mode);
		DeleteObject(p);
	}
	else
		MyFillRect(&frame, GRAY);
}
#endif

OSErr MyOpenPicture(RECTPTR r, PICTUREHP p)
#ifdef MAC
{
	*p = OpenPicture(r);
	
	return (*p) ? 0 : -1;
}
#else
{
	PICTUREH h = (PICTUREH)_NewHandle(sizeof(Picture));
	
	*p = h;
	
	DEREFH(h)->picFrame = *r;
	DEREFH(h)->hdcMeta = CreateMetaFile(NULL);
	DEREFH(h)->hmf = 0;
	
	return 0;
}
#endif

void MyClosePicture(PICTUREH h)
#ifdef MAC
{
#pragma unused(h)
	
	ClosePicture();
}
#else
{
	DEREFH(h)->hmf = CloseMetaFile(DEREFH(h)->hdcMeta);
}
#endif

void MyDrawPicture(PICTUREH h, RECTPTR r)
#ifdef MAC
{
	DrawPicture(h, r);
}
#else
{
	PlayMetaFile(GetDC(FrontWindow()), DEREFH(h)->hmf);
}
#endif

void MyKillPicture(PICTUREH h)
#ifdef MAC
{
	KillPicture(h);
}
#else
{
	DeleteMetaFile(DEREFH(h)->hmf);
}
#endif

void ClipRectPlus(Rect r)
#ifdef MAC
{
	RgnHandle clip = NewRgn(), addition = NewRgn();
	
	GetClip(clip);
	RectRgn(addition, &r);
	SectRgn(clip, addition, clip);
	SetClip(clip);
	DisposeRgn(clip);
	DisposeRgn(addition);
}
#else
{
	// (MIKE) add the rect r to the current clip region
}
#endif

void ClipRectMinus(Rect r)
#ifdef MAC
{
	RgnHandle clip = NewRgn(), addition = NewRgn();
	
	GetClip(clip);
	RectRgn(addition, &r);
	DiffRgn(clip, addition, clip);
	SetClip(clip);
	DisposeRgn(clip);
	DisposeRgn(addition);
}
#else
{
	// (MIKE) subtract the rect r from the current clip region
}
#endif


void InvalRectInWindow(Rect r, WindowPtr w)
#ifdef MAC
{
	#if TARGET_API_MAC_CARBON
		InvalWindowRect(w,&r);
	#else
		GrafPtr oldPort;
		
		GetPortGrafPtr(&oldPort);
		SetPortWindowPort(w);
		InvalRect(&r);
		SetPortGrafPort(oldPort);
	#endif
}
#else
{
	RECT R;
	
	if (!IsWindowVisible(w)) return;
	
	MakeWindowsRect(&r, &R);
	InvalidateRect(w, &R, TRUE);
}
#endif

void ValidRectInWindow(Rect r, WindowPtr w)
#ifdef MAC
{
	#if TARGET_API_MAC_CARBON
		ValidWindowRect(w,&r);
	#else
		GrafPtr oldPort;
		
		GetPortGrafPtr(&oldPort);
		SetPortWindowPort(w);
		ValidRect(&r);
		SetPortGrafPort(oldPort);
	#endif
}
#else
{
	RECT R;
	
	if (!IsWindowVisible(w)) return;
	
	MakeWindowsRect(&r, &R);
	ValidateRect(w, &R);
}
#endif

void InvalRgnInWindow(RgnHandle rgn, WindowPtr w)
#ifdef MAC
{
	#if TARGET_API_MAC_CARBON
		InvalWindowRgn(w,rgn);
	#else
		GrafPtr oldPort;
		
		GetPortGrafPtr(&oldPort);
		SetPortWindowPort(w);
		InvalRgn(rgn);
		SetPortGrafPort(oldPort);
	#endif
}
#else
{
	if (!IsWindowVisible(w)) return;
	
	InvalidateRgn(w, rgn, TRUE);
}
#endif

void MyDragGrayRgn(RgnHandle r, Point p, void (*action)(Point),
				   SHORTPTR dh, SHORTPTR dv,
				   RECTPTR rgnBounds, RECTPTR dragBounds)
{
	short dX, dY;
	Rect check, check2;
	Point prevPoint = p, nextPoint = p;
	PenState ps;
	
	GetPenState(&ps);
	PenNormal();
	PenMode(patXor);
#ifdef MAC
	//PenPat((ConstPatternParam)&qd.gray);
	//PenPat((ConstPatternParam)&GRAY_BRUSH);
	PenPatQDGlobalsGray();
#else
	FillPat(DARKGRAY);
	PenStyle(DARKGRAY, 2);
#endif
	
	MyFrameRgn(r);
	while(StillDown()) {
		GetMouse(&nextPoint);
		if (rgnBounds && dragBounds) {
			dX = nextPoint.h - p.h;
			dY = nextPoint.v - p.v;
			check = *rgnBounds;
			MyOffsetRect(&check, dX, dY);
			check2 = ForceRectIntoRect(check, *dragBounds);
			nextPoint.h += check2.left - check.left;
			nextPoint.v += check2.top - check.top;
		}
		dX = nextPoint.h - prevPoint.h;
		dY = nextPoint.v - prevPoint.v;
		if (dX || dY) {
			PenNormal();
			PenMode(patXor);
#ifdef MAC
			//PenPat((ConstPatternParam)&GRAY_BRUSH);
			PenPatQDGlobalsGray();
#else
			FillPat(DARKGRAY);
			PenStyle(DARKGRAY, 2);
#endif
			MyFrameRgn(r);
			OffsetRgn(r, dX, dY);
			MyFrameRgn(r);
			prevPoint = nextPoint;
			if (action) (*action)(nextPoint);
		}
	}
	MyFrameRgn(r);
	
	*dh = nextPoint.h - p.h;
	*dv = nextPoint.v - p.v;
	
	SetPenState(&ps);
}

Rect DefineGrayRect(Point p, void (*action)(RECTPTR, Boolean), Boolean gray,
					Boolean fromCenter, Boolean optionTogglesCenter,
					Boolean ellipse, Boolean optionTogglesEllipse, Boolean escape)
{
	Point prevPoint = p, nextPoint = p;
	PenState ps;
	Rect r, r2, oldR2;
	Boolean shiftWasDown = ShiftKeyDown(), optionWasDown = OptionKeyDown(), oldOWD;
	
	GetPenState(&ps);
	PenNormal(); PenMode(patXor);
#ifdef MAC
	//if (gray) PenPat((ConstPatternParam)&qd.gray);
	//if (gray) PenPat((ConstPatternParam)&GRAY_BRUSH);
	if (gray) PenPatQDGlobalsGray();
#else
	if (gray) PenStyle(GRAY, 1);
#endif
	
	MySetRect(&r, p.h, p.v, p.h + 1, p.v + 1);
	r2 = r;
	FrameRectEllipse(&r2, (optionTogglesCenter && optionWasDown) ? !fromCenter : fromCenter,
						  (optionTogglesEllipse && optionWasDown) ? !ellipse : ellipse, p);
	while (StillDown()) {
		if (escape && CmdPeriod()) { MySetRect(&r, 0, 0, 0, 0); break; }
		
		GetMouse(&nextPoint);
		if (!EqualPoints(nextPoint, prevPoint) ||
			shiftWasDown != ShiftKeyDown() || optionWasDown != OptionKeyDown()) {
			oldR2 = r2;
			oldOWD = optionWasDown;
			shiftWasDown = ShiftKeyDown();
			optionWasDown = OptionKeyDown();
			MySetRect(&r, p.h, p.v, nextPoint.h, nextPoint.v);
			if (action)
				(*action)(&r, fromCenter ^ (optionTogglesCenter && optionWasDown));
			r2 = r;
			NormalizeRect(&r2);
			PenNormal(); PenMode(patXor);
#ifdef MAC
			//if (gray) PenPat((ConstPatternParam)&qd.gray);
			//if (gray) PenPat((ConstPatternParam)&GRAY_BRUSH);
			if (gray) PenPatQDGlobalsGray();
#else
			if (gray) PenStyle(GRAY, 1);
#endif
			FrameRectEllipse(&oldR2, (optionTogglesCenter && oldOWD) ? !fromCenter : fromCenter,
									 (optionTogglesEllipse && oldOWD) ? !ellipse : ellipse, p);
			FrameRectEllipse(&r2, (optionTogglesCenter && optionWasDown) ? !fromCenter : fromCenter,
								  (optionTogglesEllipse && optionWasDown) ? !ellipse : ellipse, p);
			prevPoint = nextPoint;
		}
	}
	FrameRectEllipse(&r2, (optionTogglesCenter && optionWasDown) ? !fromCenter : fromCenter,
						  (optionTogglesEllipse && optionWasDown) ? !ellipse : ellipse, p);
	NormalizeRect(&r);
	if (fromCenter ^ (optionTogglesCenter && optionWasDown))
		MySetRect(&r, p.h - RectWidth(r), p.v - RectHeight(r),
					  p.h + RectWidth(r), p.v + RectHeight(r));
	
	SetPenState(&ps);
	
	return r;
}

///// KEYS ////////////////////////////////////////////////////////////////////////////

Boolean KeyDown(short key)
#ifdef MAC
{
	KeyMap theKeys;
	
	GetKeys(theKeys);
	
	return(BitTst((Ptr)theKeys, key));
}
#else
{
	return GetAsyncKeyState(key) < 0;
}
#endif

Boolean OptionKeyDown()
#ifdef MAC
{
	return KeyDown(61);
}
#else
{
	return GetAsyncKeyState(VK_CONTROL) < 0;
}
#endif

Boolean ShiftKeyDown()
#ifdef MAC
{
	return KeyDown(63);
}
#else
{
	return GetAsyncKeyState(VK_SHIFT) < 0;
}
#endif

Boolean ControlKeyDown()
#ifdef MAC
{
	return KeyDown(60);
}
#else
{
	return GetAsyncKeyState(VK_CONTROL) < 0;
}
#endif

Boolean KeyEvent(EVENTRECORDPTR e, char key1, char key2)
#ifdef MAC
{
	return (  (e->what == keyDown || e->what == autoKey)
			    &&
		       (!key1 || ((e->message & 0xff) == key1) || ((e->message & 0xff) == key2))  );
}
#else
{
	return (e->message == WM_CHAR || e->message == WM_KEYDOWN) &&
		   (!key1 || (e->wParam == (UINT)key1) || e->wParam == (UINT)key2);
}
#endif

Boolean CommandKeyEvent(EVENTRECORDPTR e, char key1, char key2)
#ifdef MAC
{
	return (KeyEvent(e, key1, key2) && (e->modifiers & cmdKey));
}
#else
{
	Boolean result;
	
	if (e->message == WM_CHAR) {
		if (!(e->wParam < ' ')) return FALSE;
		
		e->wParam += 0x60; // map control keys to normal lower-case keys
		result = KeyEvent(e, key1, key2);
		e->wParam -= 0x60; // map back
		
		return result;
	}
	
	if (e->message == WM_KEYDOWN)
		return e->wParam != VK_CONTROL &&
			   (/* KeyDown(VK_CONTROL) || */ GetKeyState(VK_CONTROL) < 0);
	
	return FALSE;
}
#endif

Boolean CmdPeriodEvent(EVENTRECORDPTR e)
{
	return KeyEvent(e, ESC, ESC) || CommandKeyEvent(e, '.', '.');
}

Boolean CmdPeriod2(Boolean anyKey)
#ifdef MAC
{
#if TARGET_API_MAC_CARBON
	if(CheckEventQueueForUserCancel()) return TRUE;
	if(anyKey) {
		EventRecord event;
		if(EventAvail(mDownMask,&event)) {
			return TRUE; // we have a mouse down event waiting
		}
		if(EventAvail(keyDownMask,&event)) {
			if(CommandKeyEvent(&event, 0, 0))
				return TRUE; // we have a command key event waiting
		}
	}
	return FALSE;
	
#else
	EventRecord	*e, event;
	QHdrPtr header = GetEvQHdr();
	QElemPtr events = header->qHead;
	
	while (events) {
		e = (EventRecord *)&events->qData;
		if (anyKey)
			if (e->what == mouseDown || CommandKeyEvent(e, 0, 0))
				return TRUE;
		if (KeyEvent(e, ESC, ESC) || CommandKeyEvent(e, '.', '.')) {
			while (GetNextEvent(keyDownMask | keyUpMask, &event)) ;
			
			return TRUE;
		}
		events = events->qLink;
	}
	
	return FALSE;
#endif
}
#else
{
	EventRecord e;
	Boolean broke = FALSE;
	DWORD ticks;
	long limit = settings.inBackground ? 0 : 500;
	static DWORD prevTicks = 0;
	
	ticks = GetTickCount();
	if ((ticks - prevTicks) < (DWORD)limit) return FALSE;
	prevTicks = ticks;
	
	// JLM 8/15/98, check Keystate for quicker stop during the model run
	// but not when in the background
	if(!settings.inBackground && HIWORD(GetKeyState(VK_ESCAPE))) 
		broke = TRUE;

	//JLM, PeekMessage(&e, 0, WM_DDE_FIRST, WM_DDE_LAST, FALSE); // process DDE
	
	if (anyKey)
		if (PeekMessage(&e, 0, WM_LBUTTONDOWN, WM_LBUTTONDOWN, FALSE) ||
			PeekMessage(&e, 0, WM_NCLBUTTONDOWN, WM_NCLBUTTONDOWN, FALSE))
			broke = TRUE;
	
	if (anyKey)
		while (PeekMessage(&e, 0, WM_KEYDOWN, WM_KEYUP, FALSE)) {
			if (e.message == WM_KEYDOWN) {
				if (CommandKeyEvent(&e, 0, 0)) {
					broke = TRUE;
					break;
				}
				else {
					if (KeyEvent(&e, ESC, ESC) || CommandKeyEvent(&e, '.', '.'))
						broke = TRUE;
					PeekMessage(&e, 0, WM_KEYDOWN, WM_KEYUP, TRUE);
				}
			}
			else {
				if (KeyEvent(&e, ESC, ESC) || CommandKeyEvent(&e, '.', '.'))
					broke = TRUE;
				PeekMessage(&e, 0, WM_KEYDOWN, WM_KEYUP, TRUE);
			}
		}
	
	if (!anyKey)
		while (PeekMessage(&e, 0, WM_KEYDOWN, WM_KEYUP, TRUE)) // this will remove misc key strokes
			if (KeyEvent(&e, ESC, ESC) || CommandKeyEvent(&e, '.', '.'))
				broke = TRUE;
	
	//if (settings.inBackground)
		//if (WindowNeedsUpdating(hMainWnd))
			//broke = 2; // special code: nested paint message sent during update in background
	
	return broke;
}
#endif

Boolean CmdPeriod()
{
	return CmdPeriod2(FALSE);
}

Boolean CmdPeriodAny()
{ // seems to be any command key or mouse down 
	return CmdPeriod2(TRUE);
}

///// MAC LISTS ////////////////////////////////////////////////////////////////////////

#ifdef MAC
Boolean ListKeyEvent(EVENTRECORDPTR e, ListHandle L)
{
	char c = (e->message & keyCodeMask) >> 8;
	short d = 0;
	Point cell;
	
	if (e->what != keyDown && e->what != autoKey) return false;
	SetPt(&cell, 0, 0);
	if (!LGetSelect(true, &cell, L)) return false;
	
	if (c == UPKEY || c == UPKEYPLUS) {
		if (cell.v > 0) d = -1;
		else return true;
	}
	
	if (c == DOWNKEY || c == DOWNKEYPLUS) {
		if (cell.v < ((**L).dataBounds.bottom - 1)) d = 1;
		else return true;
	}
	
	if (!d) return false;
	
	LSetSelect(false, cell, L);
	cell.v += d;
	LSetSelect(true, cell, L);
	MakeListCellVisible(cell, L);
	return true;
}
#endif

#ifdef MAC
void MakeListCellVisible(Point cell, ListHandle L)
{
	while (cell.v < (**L).visible.top) LScroll(0, -1, L);
	while (cell.v >= (**L).visible.bottom) LScroll(0, 1, L);
}
#endif

#ifdef MAC
void DeselectAllCells(ListHandle L)
{
	Point theCell;
	
	SetPt(&theCell, 0, 0);
	while (LGetSelect(true, &theCell, L)) {
		LSetSelect(false, theCell, L);
		SetPt(&theCell, 0, 0);
	}
}
#endif

///// MENUS ///////////////////////////////////////////////////////////////////////////

Boolean GetAndInsertMenu(short menuID, short special)
#ifdef MAC
{
	MenuHandle m = GetMenu(menuID);
	
	if (!m) { SysBeep(1); return FALSE; }
	
	if (special == 1) AppendResMenu (m, 'DRVR');
	if (special == 2) { InsertResMenu (m, 'FONT', 0); special = -1; }
	InsertMenu(m , (special == -1) ? -1 : 0);	// use -1 for font menus and 0 for all others
	
	return TRUE;
}
#else
{
	return TRUE;
}
#endif

void MyEnableMenuItem(MenuHandle m, short item, Boolean enable)
#ifdef MAC
{
	if (enable)
		EnableItem(m, item);
		//EnableMenuItem(m, item);
	else
		DisableItem(m, item);
		//DisableMenuItem(m, item);
}
#else
{
	EnableMenuItem(m, item, MF_BYCOMMAND | (enable ? MF_ENABLED : MF_GRAYED));
}
#endif

MenuHandle EnsureMenu(short menuID, short position)
#ifdef MAC
{
	MenuHandle h;
	
	if (h = GetMenuHandle(menuID)) return h;
	
	if (h = GetMenu(menuID)) {
		InsertMenu(h, position);
		return h;
	}
	
	SysBeep(1);
	return nil;
}
#else
{
#pragma unused(position)
	
	return GetMenuHandle(menuID);
}
#endif

void ClearAllMarks(short menuID)
{
	MenuHandle		theMenu;
	short			i, c;
	
	for (i = 1, theMenu = GetMenuHandle(menuID), c = CountMenuItems(theMenu); i <= c ; i++)
		SetItemMark(theMenu, i, noMark);
}

void ClearAllMarks2(MenuHandle m)
{
	short i, c;
	
	for (i = 1, c = CountMenuItems(m); i <= c ; i++)
		SetItemMark(m, i, noMark);
}

short CheckedItem(short menuID)
#ifdef MAC
{
	MenuHandle theMenu;
	short i, c, mark, cmd;
	
	for (i = 1, theMenu = GetMenuHandle(menuID), c = CountMenuItems(theMenu); i <= c ; i++) {
		GetItemMark(theMenu, i, &mark);
		GetItemCmd(theMenu, i, &cmd);
		if (mark != noMark && cmd != 0x1B)	// 0x1B cmd key = hierarchical (check = menu ID)
			return i;
	}
	
	return 0;
}
#else
{
	MenuHandle theMenu;
	short i, c, mark;
	
	for (i = 1, theMenu = GetMenuHandle(menuID), c = CountMenuItems(theMenu); i <= c ; i++) {
		GetItemMark(theMenu, i, &mark);
		if (mark == checkMark)
			return i;
	}
	
	return 0;
}
#endif

short CheckedItem2(MenuHandle m)
#ifdef MAC
{
	short i, c, mark, cmd;
	
	for (i = 1, c = CountMenuItems(m); i <= c ; i++) {
		GetItemMark(m, i, &mark);
		GetItemCmd(m, i, &cmd);
		if (mark != noMark && cmd != 0x1B)	// 0x1B cmd key = hierarchical (check = menu ID)
			return i;
	}
	
	return 0;
}
#else
{
	short i, c, mark;
	
	for (i = 1, c = CountMenuItems(m); i <= c ; i++) {
		GetItemMark(m, i, &mark);
		if (mark == checkMark)
			return i;
	}
	
	return 0;
}
#endif

void UncheckAllItems(short menuID)
{
	short i;
	
	while (i = CheckedItem(menuID))
		ToggleMenuCheck(menuID, i);
}

void UncheckAllItems2(MenuHandle m)
{
	short i;
	
	while (i = CheckedItem2(m))
		ToggleMenuCheck2(m, i);
}

Boolean ItemChecked(short menuID, short item)
{
	short markChar;
	
	GetItemMark(GetMenuHandle(menuID), item, &markChar);
	
	return (markChar != noMark);
}

Boolean ItemChecked2(MenuHandle m, short item)
{
	short markChar;
	
	GetItemMark(m, item, &markChar);
	
	return (markChar != noMark);
}

void ToggleMenuCheck(short menuID, short item)
{
	SetItemMark(GetMenuHandle(menuID), item, ItemChecked(menuID, item) ? noMark : checkMark);
}

void ToggleMenuCheck2(MenuHandle m, short item)
{
	SetItemMark(m, item, ItemChecked2(m, item) ? noMark : checkMark);
}

CHARPTR MyGetMenuTitle(short menuID, CHARPTR title)
#ifdef MAC
{
	MenuHandle m = GetMenuHandle(menuID);
	
	title[0] = 0;
	
	if (m) {
		_HLock((Handle)m);
		mypstrcpy(title, (*(CHARH)m) + 14);
		_HUnlock((Handle)m);
		
		my_p2cstr((StringPtr)title);
	}
	
	return title;
}
#else
{
	// (MIKE) what is the title of the menu menuID?
	title[0] = 0;

	return title;
}
#endif

CHARPTR MySetMenuTitle(short menuID, CHARPTR title)
#ifdef MAC
{
	MenuHandle m = GetMenuHandle(menuID);
	
	my_c2pstr(title);
	
	_HLock((Handle)m);
	mypstrcpy((*(CHARH)m) + 14, title);
	_HUnlock((Handle)m);
	
	return title;
}
#else
{
	// (MIKE) set the title of the menu menuID

	return title;
}
#endif

///// SYSTEM ///////////////////////////////////////////////////////////////////////////

