
#include "CROSS.h"
#include "TextRect.h"
#include "EditWindsDialog.h"
#include "MakeMovie.h"
#include "NetCDFMover.h"
#include "NetCDFStore.h"
#include "NetCDFWindMover.h"
#include "NetCDFWindMoverCurv.h"
//#include <Navigation.h>
//#include <MacTextEditor.h>
//#include <Scrap.h>

Boolean bOmitFooter = settings.omitFooter;
/////////////////////////////////////////////////
float PixelsPerPoint(void)
{ // A point  is 1/72 of an inch
	#ifdef MAC
		// on the mac everything is in points = 1/72 inch
		return 1;
	#else
		long widthInMillimeters =  GetDeviceCaps(currentHDC,HORZSIZE);
		long widthInPixs = GetDeviceCaps(currentHDC,HORZRES);
		float widthInRealInches = widthInMillimeters/25.4;
		float widthInPoints = 72.0*widthInRealInches;
		float pixelsPerPoint = widthInPixs/widthInPoints;
		return pixelsPerPoint;
	#endif 
}

////////////////////////////////////////////////////////////////////////////////
char sharedTTime[20] = "", sharedTDate[20] = "",
			sharedPTime[20] = "", sharedPDate[20] = "";

void GetStdCaveatLine(char* str , long i)
{
	str[0] = 0;
	switch(i)
	{
		/*case 0: strcpy(str,"This trajectory was created using climatological currents from a GNOME Location File,");break;
		case 1: strcpy(str,"and is unlikely to represent conditions existing at any particular time at the depicted location.");break;
		case 2: strcpy(str,"Use Location Files only to create spill scenarios for training and educational purposes,");break;
		case 3: strcpy(str,"not for actual spill response.");break;
		case 4: strcpy(str,"");break;*/
		//case 0: strcpy(str,"This trajectory was produced by GNOME (General NOAA Operational Modeling Environment), and should");break;
		//case 1: strcpy(str,"be used for educational and planning purposes only--not for a real response. In the event of an oil");break;
		//case 2: strcpy(str,"or chemical spill in U.S. waters, contact the U.S. Coast Guard National Response Center at 1-800-424-8802.");break;
		//case 0: strcpy(str,"This trajectory was produced by GNOME (General NOAA Operational Modeling Environment), and");break;
		//case 1: strcpy(str,"should be used for educational and planning purposes only--not for a real response. In the event of");break;
		case 0: strcpy(str,"This trajectory was produced by GNOME (General NOAA Operational Modeling Environment),");break;
		case 1: strcpy(str,"and should be used for educational and planning purposes only--not for a real response. In the");break;
		case 2: strcpy(str,"event of an oil or chemical spill in U.S. waters, contact the U.S. Coast Guard National Response");break;
		case 3: strcpy(str,"Center at 1-800-424-8802.");break;
		case 4: strcpy(str,"");break;
	}
}

void GetDateTimeStrings(char *trajectoryTime, char *trajectoryDate,
						char *preparedTime, char *preparedDate)
{
	unsigned long seconds;
	DateTimeRec daterec;
	
	// JLM, I think the trajectoryTime should just be the model time
	seconds = model->GetModelTime();
	Secs2DateStrings(seconds, nil,trajectoryDate,trajectoryTime,nil);
	//Secs2Date(seconds, &daterec);
	//sprintf(trajectoryTime, "%02hd%02hd", daterec.hour, daterec.minute);
	//sprintf(trajectoryDate, "%hd/%hd/%02hd", daterec.month, daterec.day, daterec.year % 100);
	
	GetDateTime(&seconds);
	Secs2DateStrings(seconds, nil,preparedDate,preparedTime,nil);
	//Secs2Date(seconds, &daterec);
	//sprintf(preparedTime, "%02hd%02hd", daterec.hour, daterec.minute);
	//sprintf(preparedDate, "%hd/%hd/%02hd", daterec.month, daterec.day, daterec.year % 100);
}

short ECClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short i;
	char str[256];
#pragma unused(lParam)
#pragma unused(data)
	
	switch (itemNum) {
		case M53COK:
			mygetitext(dialog, M53CLINE1, settings.caveat[0], 120);
			mygetitext(dialog, M53CLINE2,  settings.caveat[1], 120);
			mygetitext(dialog, M53CLINE3,  settings.caveat[2], 120);
			mygetitext(dialog, M53CLINE4,  settings.caveat[3], 120);
			mygetitext(dialog, M53CLINE5,  settings.caveat[4], 120);
			return M53COK;
			
		case M53RESET:
			GetStdCaveatLine(str,0); mysetitext(dialog, M53CLINE1,str);
			GetStdCaveatLine(str,1); mysetitext(dialog, M53CLINE2,str);
			GetStdCaveatLine(str,2); mysetitext(dialog, M53CLINE3,str);
			GetStdCaveatLine(str,3); mysetitext(dialog, M53CLINE4,str);
			GetStdCaveatLine(str,4); mysetitext(dialog, M53CLINE5,str);
			break;		
		
		case M53CCANCEL: return M53CCANCEL;
		
	}
	
	return 0;
}

OSErr ECInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	SetDialogItemHandle(dialog, M53CHILITEDEFAULT, (Handle)FrameDefault);
	
	mysetitext(dialog, M53CLINE1,  settings.caveat[0]);
	mysetitext(dialog, M53CLINE2,  settings.caveat[1]);
	mysetitext(dialog, M53CLINE3, settings.caveat[2]);
	mysetitext(dialog, M53CLINE4, settings.caveat[3]);
	mysetitext(dialog, M53CLINE5,  settings.caveat[4]);
	
	MySelectDialogItemText(dialog, M53CLINE1, 0, 128);
	
	return 0;
}

Boolean gPMShowPrintButtons = true;
Boolean gPMShowTrajectoryTime = true;

OSErr PMInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	char trajectoryTime[20], trajectoryDate[20],
		 preparedTime[20], preparedDate[20];
	short i;
	
#ifdef MAC
	SetDialogItemHandle(dialog, M53HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M53FROST1, (Handle)FrameBlack);
	SetDialogItemHandle(dialog, M53FROST2, (Handle)FrameBlack);
	SetDialogItemHandle(dialog, M53FROST3, (Handle)FrameBlack);
	SetDialogItemHandle(dialog, M53FROST4, (Handle)FrameBlack);
	SetDialogItemHandle(dialog, M53FROST5, (Handle)FrameBlack);
	SetDialogItemHandle(dialog, M53FROST6, (Handle)FrameBlack);
	SetDialogItemHandle(dialog, M53FROST7, (Handle)FrameBlack);
#endif	
	
	mysetitext(dialog, M53TITLE, settings.headerSPILLID);
	mysetitext(dialog, M53FROM, settings.headerFROM);
	mysetitext(dialog, M53CONTACT, settings.headerCONTACT);
	
	GetDateTimeStrings(trajectoryTime, trajectoryDate,
					   preparedTime, preparedDate);
	mysetitext(dialog, M53TRAJECTORYTIME, trajectoryTime);
	mysetitext(dialog, M53TRAJECTORYDATE, trajectoryDate);
#ifdef SAVEUSERPREPAREDDATETIME	
	mysetitext(dialog, M53PREPAREDTIME, sharedPTime[0] ? sharedPTime : preparedTime);
	mysetitext(dialog, M53PREPAREDDATE, sharedPDate[0] ? sharedPDate : preparedDate);
#else
	mysetitext(dialog, M53PREPAREDTIME, preparedTime);
	mysetitext(dialog, M53PREPAREDDATE, preparedDate);
#endif
	
	if(!gPMShowTrajectoryTime)
	{	// when saving a MOSS file series, we don't let the user set these values
		// since they are different for every file
		ShowHideDialogItem(dialog, M53TRAJECTORYTIME,false);	
		ShowHideDialogItem(dialog, M53TRAJECTORYDATE,false);
		//
		ShowHideDialogItem(dialog, M53FROST2,false);
		ShowHideDialogItem(dialog, M53TRAJECTORYTIMELABEL,false);
		ShowHideDialogItem(dialog, M53TRAJECTORYDATELABEL,false);
		#ifdef MAC
			ShowHideDialogItem(dialog, M53TRAJECTORYDATETIMEFROSTINGLABEL,false);
		#endif
	}
	
	SetButton(dialog, M53WINDOW, settings.pictureSize == WINDOWSIZE);
	SetButton(dialog, M53PAGE, settings.pictureSize == PAGESIZE);
		
	
	SetButton(dialog, M53BANDW, settings.printMode == BANDWMODE);
	// we are not currently supporting grayscale 
	//SetButton(dialog, M53GRAYSCALE, settings.printMode == GRAYSCALEMODE);
	ShowHideDialogItem(dialog, M53GRAYSCALE,false);	
	if(settings.printMode == GRAYSCALEMODE) settings.printMode = COLORMODE;
	//////
	SetButton(dialog, M53COLOR, settings.printMode == COLORMODE);
	
	SetButton(dialog, M53OMITFOOTER, settings.omitFooter);

	if(!gPMShowPrintButtons)
	{	// hide all the buttons if not printing
		ShowHideDialogItem(dialog, M53BANDW,false);	
		ShowHideDialogItem(dialog, M53GRAYSCALE,false);	
		ShowHideDialogItem(dialog, M53COLOR,false);
		// hide the outline box and title
		ShowHideDialogItem(dialog, M53FROST7,false);
		#ifdef MAC
			ShowHideDialogItem(dialog, M53MODELABEL,false);
		#endif
	}
	
	if(gPMShowTrajectoryTime)
		MySelectDialogItemText(dialog, M53TRAJECTORYTIME, 0, 100);
	else
		MySelectDialogItemText(dialog, M53PREPAREDTIME, 0, 100);
	
	return 0;
}

short PMClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
#pragma unused(lParam)
#pragma unused(data)
	
	switch (itemNum) {
		case M53OK:
		{
			mygetitext(dialog, M53TITLE, settings.headerSPILLID, 99);
			mygetitext(dialog, M53FROM, settings.headerFROM, 99);
			mygetitext(dialog, M53CONTACT, settings.headerCONTACT, 99);
			mygetitext(dialog, M53TRAJECTORYTIME, sharedTTime, 19);
			mygetitext(dialog, M53TRAJECTORYDATE, sharedTDate, 19);
			mygetitext(dialog, M53PREPAREDTIME, sharedPTime, 19);
			mygetitext(dialog, M53PREPAREDDATE, sharedPDate, 19);
			if (GetButton(dialog, M53BANDW)) settings.printMode = BANDWMODE;
			// we are not currently supporting grayscale 
			//if (GetButton(dialog, M53GRAYSCALE)) settings.printMode = GRAYSCALEMODE;
			if (GetButton(dialog, M53COLOR)) settings.printMode = COLORMODE;
			if (GetButton(dialog, M53WINDOW)) settings.pictureSize = WINDOWSIZE;
			if (GetButton(dialog, M53PAGE)) settings.pictureSize = PAGESIZE;
			
			if (GetButton(dialog, M53OMITFOOTER)) settings.omitFooter = 1; else settings.omitFooter = 0;

			return M53OK;
		}
		
		case M53CANCEL: return M53CANCEL;
		
		case M53BANDW:
		case M53GRAYSCALE:
		case M53COLOR:
			SetButton(dialog, M53BANDW, itemNum == M53BANDW);
			SetButton(dialog, M53GRAYSCALE, itemNum == M53GRAYSCALE);
			SetButton(dialog, M53COLOR, itemNum == M53COLOR);
			break;

		case M53OMITFOOTER:
			ToggleButton(dialog, M53OMITFOOTER);
			break;
		
		case M53EDITCAVEAT:
			PushModalGlobals(GetDialogWindow(dialog));
			MyModalDialog(M53C, GetDialogWindow(dialog), 0, ECInit, ECClick);
			PopModalGlobals();
			break;
	}
		
	
	return 0;
}


OSErr PrintMap(void)
{
	OSErr err = 0;
	short item;
	// hmmm we are occasionally crasing in OS X, try setting the port here, JLM 7/10/06
	GrafPtr savePort = 0;
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	////
#ifdef MAC
	// JLM 4/19/06, we will open the printer to get the page size
	err = OpenPrinterAndValidate();
	if (err) goto closePrinter;
#endif	
	gPMShowPrintButtons = true;
	gPMShowTrajectoryTime = true;
	item = MyModalDialog(M53, mapWindow, 0, PMInit, PMClick);
	if(item == -1) err = -1;
	if (item == M53OK) {
		err = PrintMapToPrinter();
		DisplayMessage(0);
	}
#ifdef MAC
closePrinter:
	ClosePrinter();
#endif
	SetPortGrafPort(savePort);
	
	return err;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////



short  GetFontSpacing(short fontSize, short fontid)
{
	TextFontSize(fontid,fontSize);
	
	FontInfo finfo;
	GetFontInfo(&finfo);
	return finfo.ascent+finfo.descent+finfo.leading;
}

short  GetFontLeading(short fontSize, short fontid)
{
	TextFontSize(fontid,fontSize);
	
	FontInfo finfo;
	GetFontInfo(&finfo);
	return finfo.leading;
}

/////////////////////////////////////////////////
#define kPrinterOutputFont kFontIDTimes
#define kHeaderFontSize 12
#define kFooterFontSize 11
#define kCaveatFontSize 11
#define kSeparationFontSize 10


short LittleSeparation(void)
{	// about 1/2 line of text
	short ht;
	short fontSize = kSeparationFontSize;
	ht =  GetFontSpacing(fontSize,kPrinterOutputFont)/2;
	return ht;
}

short HeaderHeight(void)
{	// space for 3 lines of text, no space top and bottom
	// or icon height, whichever is more
	short ht,iconHt;
	Rect iconRect;
	short fontSize = kHeaderFontSize;
	ht =   3 * GetFontSpacing(fontSize,kPrinterOutputFont);
	//iconRect = GetResBitmapRect(BIG_GNOME_GOING_RIGHT_ID);
	iconRect = GetResBitmapRect(GNOME_GOING_RIGHT_ID);
	iconHt = RectHeight(iconRect);
	return _max(ht,iconHt);
}

short CaveatHeight(void)
{	// space for 5 lines of text, plus one more for space top and bottom
	short ht;
	short fontSize = kCaveatFontSize;
	ht = (5+1) * GetFontSpacing(fontSize,kPrinterOutputFont);
	return ht;
}

short FooterHeight(void)
{	// space for 7 lines of text, plus one more for space top and bottom
	short ht = 0;
	short fontSize = kFooterFontSize;
	//if (!settings.omitFooter) ht =  (7+1) * GetFontSpacing(fontSize,kPrinterOutputFont);
	if (!bOmitFooter) ht =  (7+1) * GetFontSpacing(fontSize,kPrinterOutputFont);
	return ht;
}
/////////////////////////////////////////////////

void DrawFontTest(Rect r)
{
	short extraSpacing = 0;
	short leftOffsetPercent = 20;
	short fontSize = 9;
	Rect rect;
	#define kNumLinesInTest  10
	short i = 0;
	char sampleString[256];
	typedef struct{
		short fontID;
		char *str;
	} TestFont;
	TestFont test[kNumLinesInTest] = {
			{ kFontIDMonaco,"Monaco,Courier New"},
			{ kFontIDGeneva,"Geneva,MS Sans Serif"},
			{ kFontIDNewYork,"NewYork,MS Serif"},
			{ kFontIDTimes,"Times,Times New Roman"},
			{ kFontIDCourier,"Courier,Courier New"},
			{ kFontIDSymbol,"Symbol,Symbol"},
			{ kFontIDHelvetica,"Helvetica,Arial"},
			{ 0,""} };
	
	
	
	for(i = 0;i< kNumLinesInTest && test[i].fontID != 0;i++)
	{
		rect  = r; //reset
		rect.top += (i/(float)kNumLinesInTest)*RectHeight(r);
		TextRect textRect(rect,test[i].fontID,fontSize,extraSpacing,extraSpacing);
		strcpy(sampleString,test[i].str);
		strcat(sampleString," abcdefghijklmnopqrstuvwxyz"); 
		textRect.PrintLine(sampleString,leftOffsetPercent);
	}
}


/////////////////////////////////////////////////

#ifdef IBM
void PrintBitmap(short resID, Rect r)
{
	POINT pSize, pOrg;
	HDC memDC = 0;
	HBITMAP p=0;
	BITMAP b;
	
	p = LoadBitmap(hInst, MAKEINTRESOURCE(resID));
	memDC = CreateCompatibleDC(currentHDC);
	
	if (p && memDC) 
	{
		SelectObject(memDC, p);
		SetMapMode(memDC, GetMapMode(currentHDC));
		
		GetObject(p, sizeof(BITMAP), (LPSTR)&b);
	
		StretchBlt(currentHDC, // dest DC
				   r.left, // dest x
				   r.top, // dest y
				   RectWidth(r), // dest width
				   RectHeight(r), // dest height
				   memDC, // source DC
				   0, 0, // source x, y
				   b.bmWidth, // source width
				   b.bmHeight, // source height
				   SRCCOPY); // ROP mode
	}
	if(memDC) DeleteDC(memDC);
	if(p) DeleteObject(p);
}

Rect PrintBitmapAtPoint(short resID, short h, short v)
{
	POINT pSize, pOrg;
	HDC memDC = 0;
	HBITMAP p=0;
	BITMAP b;
	Rect dstRect;
	short dstWidth = 0,dstHeight = 0;
	
	p = LoadBitmap(hInst, MAKEINTRESOURCE(resID));
	memDC = CreateCompatibleDC(currentHDC);
	
	if (p && memDC) 
	{
		SelectObject(memDC, p);
		SetMapMode(memDC, GetMapMode(currentHDC));
		
		GetObject(p, sizeof(BITMAP), (LPSTR)&b);
		
		///// compensate for printer resolution //////////////{
		{ 
			long widthInMillimeters =  GetDeviceCaps(currentHDC,HORZSIZE);
			long widthInPixs = GetDeviceCaps(currentHDC,HORZRES);
			float widthInRealInches = widthInMillimeters/25.4;
			short pixPerRealInch = widthInPixs/widthInRealInches;
	
			dstWidth = round((b.bmWidth*pixPerRealInch)/72.0);
			dstHeight = round((b.bmHeight*pixPerRealInch)/72.0);
			
			dstRect.top = v;
			dstRect.left = h;
			dstRect.right = dstRect.left +  dstWidth;
			dstRect.bottom = dstRect.top + dstHeight;
		}
		////////////////////////////////////////}
	
		StretchBlt(currentHDC, // dest DC
				   h, // dest x
				   v, // dest y
				   dstWidth, // dest width
				   dstHeight, // dest height
				   memDC, // source DC
				   0, 0, // source x, y
				   b.bmWidth, // source width
				   b.bmHeight, // source height
				   SRCCOPY); // ROP mode
	}
	if(memDC) DeleteDC(memDC);
	if(p) DeleteObject(p);
	
	dstRect.right = dstRect.left + dstWidth;
	dstRect.bottom = dstRect.top + dstHeight;
	return dstRect;
}

void PaintIcon(short resID,Rect* iconRect)
{
	HDC hMemDC = 0;
	HBITMAP hBob = 0;
	HBITMAP oldBitmap = 0;
	BOOL err;
	HICON hIcon = 0;
	short iconWidth = GetSystemMetrics(SM_CXICON);
	short iconHeight = GetSystemMetrics(SM_CYICON);
	short dstWidth,dstHeight;
	
	hMemDC = CreateCompatibleDC(currentHDC);
	hIcon = LoadIcon(hInst,MAKEINTRESOURCE(resID));
	
	if(hIcon != nil)
	{
	
		hBob = CreateCompatibleBitmap(currentHDC,iconWidth,iconHeight);
		
		oldBitmap = (HBITMAP)SelectObject(hMemDC, hBob);
		
		// now paint the newbitmap white (clean slate)
		PatBlt (hMemDC, 0, 0,iconWidth,iconHeight, WHITENESS);
		
		err =  DrawIcon (hMemDC,0,0, hIcon);  // draw icon onto bm
		
		///// compensate for printer resolution //////////////{
		{ 
			long widthInMillimeters =  GetDeviceCaps(currentHDC,HORZSIZE);
			long widthInPixs = GetDeviceCaps(currentHDC,HORZRES);
			float widthInRealInches = widthInMillimeters/25.4;
			short pixPerRealInch = widthInPixs/widthInRealInches;

			dstWidth = round((iconWidth*pixPerRealInch)/72.0);
			dstHeight = round((iconHeight*pixPerRealInch)/72.0);
			
			iconRect->right = iconRect->left +  dstWidth;
			iconRect->bottom = iconRect->top + dstHeight;
		}
		////////////////////////////////////////}
		
		  err = StretchBlt (currentHDC,                                                                    // dest DC,
							iconRect->left,  // Dest X,
							iconRect->top, //Dest Y
							dstWidth, //Ext X
							dstHeight, //Ext Y
							hMemDC,
							0,0,         // SrcX, Src Y
							iconWidth,            // Srx Xext
							iconHeight,            // Src Yext
							SRCCOPY);                // wUsage, dwRop
		
	
		(void)SelectObject(hMemDC,oldBitmap);
	}
	if(hMemDC) DeleteDC(hMemDC);
	if(hBob) DeleteObject(hBob);
	if(hIcon) DeleteObject(hIcon);  // ???
}
#endif

/////////////////////////////////////////////////
/////////////////////////////////////////////////

void DrawHeader(Rect r, Boolean bitmapIcons)
{
	char s[256],str[128];
	short left,fontSize=kHeaderFontSize;
	Rect iconRect,textRect;
	short extraSpacingLeft;
	short extraSpacingTop;
	short widestNum = 0, widestNumLeft = 0;
	Boolean bDrawBlackAndWhite = settings.printMode == BANDWMODE;
	
//Draw the gnome
	MySetRect(&iconRect, r.left,r.top,r.left,r.top); 
	
	//bounding rect of pict is returned in iconRect
#ifdef MAC
	if(bDrawBlackAndWhite) DrawResBitmap2(BLACKANDWHITE_GNOME_GOING_RIGHT_ID, &iconRect, 0);
	else DrawResBitmap2(GNOME_GOING_RIGHT_ID, &iconRect, 0);
#else 
	{
		Boolean hasColor ;
		short nPlanes = GetDeviceCaps(currentHDC, PLANES);
		short  nBitsPixel = GetDeviceCaps(currentHDC, BITSPIXEL);

		hasColor =  (nPlanes * nBitsPixel) > 1;
		
		// print bitmaps to fill left side of header
		iconRect.bottom = r.bottom; 
		iconRect.right = iconRect.top+RectHeight(iconRect); 
		if(hasColor && !bDrawBlackAndWhite) PrintBitmap(GNOME_GOING_RIGHT_ID,iconRect);
		else PrintBitmap(BLACKANDWHITE_GNOME_GOING_RIGHT_ID,iconRect);
		
		// draw bitmap at size
		//if(hasColor && !bDrawBlackAndWhite) iconRect = PrintBitmapAtPoint(GNOME_GOING_RIGHT_ID,iconRect.left,iconRect.top);
		//else iconRect = PrintBitmapAtPoint(BLACKANDWHITE_GNOME_GOING_RIGHT_ID,iconRect.left,iconRect.top);

		// use icons
		//if(hasColor && !bDrawBlackAndWhite) PaintIcon(GNOME_GOING_RIGHT_ID, &iconRect); 
		//else PaintIcon(BLACKANDWHITE_GNOME_GOING_RIGHT_ID, &iconRect); 
	}
#endif
	
	left = iconRect.right; // Note: this Starts at left hand at zero if pict was not read in
	MySetRect(&textRect,left,r.top,(left+r.right)/2,r.bottom);
	
	//fontSize=kHeaderFontSize;
	extraSpacingLeft = GetFontSpacing(fontSize,kPrinterOutputFont)/2;

//Print first column in header	
	
	TextRect leftColumn(textRect,kPrinterOutputFont,fontSize,extraSpacingLeft,0);  
	
	sprintf(s,"Model Mode: %s",model->GetModelModeStr(str));
	leftColumn.PrintLine(s);
	
	sprintf(s, "Estimate for: %s %s",sharedTTime, sharedTDate);
	leftColumn.PrintLine(s);
	
	widestNumLeft = stringwidth(s) + extraSpacingLeft; // I think this line will always be the longest
	
	sprintf(s, "Prepared: %s %s", sharedPTime, sharedPDate);
	leftColumn.PrintLine(s);


//Print second column in header
	// code goes here, figure out length of longest header line (what if zero?) and move right column all the way over
	//MySetRect(&textRect,(left+r.right)/2+1,r.top,r.right,r.bottom);
	//TextRect rightColumn(textRect,kPrinterOutputFont,fontSize,extraSpacingLeft,0);
	
	widestNum = stringwidth(settings.headerSPILLID)+stringwidth("Scenario Name: ");
	if ((stringwidth(settings.headerFROM)+stringwidth("Prepared by: ")) > widestNum) widestNum = stringwidth(settings.headerFROM)+stringwidth("Prepared by: ");
	if ((stringwidth(settings.headerCONTACT)+stringwidth("Contact Phone: ")) > widestNum) widestNum = stringwidth(settings.headerCONTACT)+stringwidth("Contact Phone: ");

	// code goes here, check if they have a superwide contact field, and don't shift too far left, maybe limit in dialog box...
	if (widestNum>0 && ((r.right-widestNum) > (left+widestNumLeft+extraSpacingLeft))) 
	{
		MySetRect(&textRect,(r.right-widestNum),r.top,r.right,r.bottom);
		//TextRect rightColumn(textRect,kPrinterOutputFont,fontSize,extraSpacingLeft,0);
		TextRect rightColumn(textRect,kPrinterOutputFont,fontSize,0,0);
	sprintf(s, "Scenario Name: %s",settings.headerSPILLID);
	rightColumn.PrintLine(s);
	
	sprintf(s, "Prepared by: %s", settings.headerFROM);
	rightColumn.PrintLine(s);
	
	sprintf(s, "Contact Phone: %s",settings.headerCONTACT);
	rightColumn.PrintLine(s);
	}
	else
	{
		//MySetRect(&textRect,(left+r.right)/2+1,r.top,r.right,r.bottom);
		MySetRect(&textRect,(left+widestNumLeft+extraSpacingLeft),r.top,r.right,r.bottom);
		TextRect rightColumn(textRect,kPrinterOutputFont,fontSize,extraSpacingLeft,0);
	sprintf(s, "Scenario Name: %s",settings.headerSPILLID);
	rightColumn.PrintLine(s);
	
	sprintf(s, "Prepared by: %s", settings.headerFROM);
	rightColumn.PrintLine(s);
	
	sprintf(s, "Contact Phone: %s",settings.headerCONTACT);
	rightColumn.PrintLine(s);
	}

/*	sprintf(s, "Scenario Name: %s",settings.headerSPILLID);
	rightColumn.PrintLine(s);
	
	sprintf(s, "Prepared by: %s", settings.headerFROM);
	rightColumn.PrintLine(s);
	
	sprintf(s, "Contact Phone: %s",settings.headerCONTACT);
	rightColumn.PrintLine(s);*/
}

/////////////////////////////////////////////////
	
void DrawHeader2(Rect r, Boolean bitmapIcons)
{
	char s[256],str[128];
	short left,fontSize=kHeaderFontSize;
	Rect iconRect,textRect;
	short extraSpacingLeft;
	short extraSpacingTop;
	Boolean bDrawBlackAndWhite = settings.printMode == BANDWMODE;
	unsigned long seconds;
	char modelStartTime[20], modelStartDate[20];
//Draw the gnome
	MySetRect(&iconRect, r.left,r.top,r.left,r.top); 
	// check if these are already set?
	GetDateTimeStrings(sharedTTime, sharedTDate,
				   sharedPTime, sharedPDate);

	//bounding rect of pict is returned in iconRect
#ifdef MAC
	if(bDrawBlackAndWhite) DrawResBitmap2(BLACKANDWHITE_GNOME_GOING_RIGHT_ID, &iconRect, 0);
	else DrawResBitmap2(GNOME_GOING_RIGHT_ID, &iconRect, 0);
#else 
	{
		Boolean hasColor ;
		short nPlanes = GetDeviceCaps(currentHDC, PLANES);
		short  nBitsPixel = GetDeviceCaps(currentHDC, BITSPIXEL);

		hasColor =  (nPlanes * nBitsPixel) > 1;
		
		// print bitmaps to fill left side of header
		iconRect.bottom = r.bottom; 
		iconRect.right = iconRect.top+RectHeight(iconRect); 
		if(hasColor && !bDrawBlackAndWhite) PrintBitmap(GNOME_GOING_RIGHT_ID,iconRect);
		else PrintBitmap(BLACKANDWHITE_GNOME_GOING_RIGHT_ID,iconRect);
		
		// draw bitmap at size
		//if(hasColor && !bDrawBlackAndWhite) iconRect = PrintBitmapAtPoint(GNOME_GOING_RIGHT_ID,iconRect.left,iconRect.top);
		//else iconRect = PrintBitmapAtPoint(BLACKANDWHITE_GNOME_GOING_RIGHT_ID,iconRect.left,iconRect.top);

		// use icons
		//if(hasColor && !bDrawBlackAndWhite) PaintIcon(GNOME_GOING_RIGHT_ID, &iconRect); 
		//else PaintIcon(BLACKANDWHITE_GNOME_GOING_RIGHT_ID, &iconRect); 

	}
#endif
	
	left = iconRect.right; // Note: this Starts at left hand at zero if pict was not read in
	MySetRect(&textRect,left,r.top,(left+r.right)/2,r.bottom);
	
	//fontSize=kHeaderFontSize;
	extraSpacingLeft = GetFontSpacing(fontSize,kPrinterOutputFont)/2;

//Print first column in header	
	
	TextRect leftColumn(textRect,kPrinterOutputFont,fontSize,extraSpacingLeft,0);  
	
	//sprintf(s,"Model Mode: %s",model->GetModelModeStr(str));

	seconds = model->GetStartTime();
	Secs2DateStrings(seconds, nil,modelStartDate,modelStartTime,nil);

	sprintf(s, "Scenario Name: %s",settings.headerSPILLID);
	leftColumn.PrintLine(s);

	sprintf(s, "Model Start Time: %s %s",modelStartTime,modelStartDate);
	leftColumn.PrintLine(s);
	
	sprintf(s, "Estimate for: %s %s",sharedTTime, sharedTDate);
	leftColumn.PrintLine(s);
	
//Print second column in header
	MySetRect(&textRect,(left+r.right)/2+1,r.top,r.right,r.bottom);
	TextRect rightColumn(textRect,kPrinterOutputFont,fontSize,extraSpacingLeft,0);
	
	sprintf(s, "Prepared by: %s", settings.headerFROM);
	rightColumn.PrintLine(s);
	
	sprintf(s, "Prepared on: %s %s", sharedPDate, sharedPTime);
	rightColumn.PrintLine(s);
	//sprintf(s, "Contact Phone: %s",settings.headerCONTACT);
	//rightColumn.PrintLine(s);
}

/////////////////////////////////////////////////
	
void DrawCaveat(Rect r, Boolean bitmapIcons)
{
	short fontSize=kCaveatFontSize;
	Rect textRect;
	short extraSpacing; 
	
	//fontSize=kCaveatFontSize;
	textRect = r;
	
	extraSpacing = GetFontSpacing(fontSize,kPrinterOutputFont)/2;

	TextRect caveat(textRect,kPrinterOutputFont,fontSize,extraSpacing,extraSpacing);
	
	caveat.FrameBounds();
	caveat.PrintLine(settings.caveat[0]);
	caveat.PrintLine(settings.caveat[1]);
	caveat.PrintLine(settings.caveat[2]);
	caveat.PrintLine(settings.caveat[3]);
	caveat.PrintLine(settings.caveat[4]);
}


/////////////////////////////////////////////////

void DrawFooter(Rect r, Boolean bitmapIcons)
{
	char s[256],str[128];
	short i, offset,top,left,fontSize = kFooterFontSize;
	Rect textRect;
	short ht;
	short extraSpacing;
	OSErr err = 0;
	
	MySetRect(&textRect,r.left,r.top,(r.left+r.right)/2,r.bottom);

	extraSpacing = GetFontSpacing(fontSize,kPrinterOutputFont)/2;

	TextRect leftFooter(textRect,kPrinterOutputFont,fontSize,extraSpacing,extraSpacing);
	leftFooter.FrameBounds();
	// code goes here, what if number of lines is greater than the 7 allotted?
	if(model->fWizard->HaveOpenWizardFile())
	{
		short numWizLines = model->fWizard->NumLeftHandSideAnswersOnPrintout();
		for(i = 0; i< numWizLines;i++)
		{
			model->fWizard->LeftHandSideTextForPrintout(i,s);
			if (i==0) strcat(s," Location File");
			leftFooter.PrintLine(s);
		}
	}
	
	// wind info
	TWindMover *mover = model->GetWindMover(false);
	if(mover)
	{
		VelocityRec vel;
		if(mover->fIsConstantWind)
		{
			vel=mover->fConstantValue;
			UV2RThetaStrings(vel.u,vel.v,mover->timeDep -> GetUserUnits(),str);
			//sprintf(s, "Wind: Constant %s",str);
			sprintf(s, "Constant Wind: %s",str);
			leftFooter.PrintLine(s);
		}
		else
		{
			DateTimeRec time;
			char timeStr[20];
			SecondsToDate (model->GetModelTime(), &time);
			Date2String(&time, timeStr);
			err = mover->GetTimeValue(model->GetModelTime(),&vel);
			UV2RThetaStrings(vel.u,vel.v,mover->timeDep -> GetUserUnits(),str);
			
			//sprintf(s,"Wind: Variable %s",str);
			strcpy(s,"Variable Wind");
			leftFooter.PrintLine(s);
			
			//sprintf(s, "at %s",timeStr);
			//leftFooter.PrintLine(s,5);
		}
	}
	

	// move to left side of footer
	if (model->IsUncertain())
		leftFooter.PrintLine("Black Splots: Best Estimate, Red Splots: Uncertainty");
	else
		leftFooter.PrintLine("Black Splots: Best Estimate");


	//sprintf(s,"Number of Spills: %ld",model->GetNumForecastSpills());
	//leftFooter.PrintLine(s);
	
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////


	MySetRect(&textRect,(r.left+r.right)/2+1,r.top,r.right,r.bottom);
	TextRect rightFooter(textRect,kPrinterOutputFont,fontSize,extraSpacing,extraSpacing);
	rightFooter.FrameBounds();
	
	/*if (model->IsUncertain())
		rightFooter.PrintLine("Black Splots: Best Estimate, Red Splots: Uncertainty");
	else
		rightFooter.PrintLine("Black Splots: Best Estimate");*/

	/////////////////////////////////////////////////
	 // spill totals code
	
	double amtReleased=0,amtEvap=0,amtBeached=0,amtOffmap=0, amtFloating=0,amtTotal=0,amtDisp=0,amtRemoved=0;
	short unitsForTotals = BARRELS;
	char unitsStr [64];
	char infoStr [255];
	short numDecPlaces;
	short numLeSets;
	TLEList* thisLEList;
	
	for(numLeSets = model->LESetsList->GetItemCount(),i=numLeSets-1; i>=0 ; i--)
	{
		model->LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if(thisLEList -> GetLEType() != UNCERTAINTY_LE )
		{
			unitsForTotals = thisLEList->GetMassUnits(); //return units for the first nonuncertaintly le list
		}
	}
	// get the totals
	model->GetTotalAmountStatistics(unitsForTotals,&amtTotal,&amtReleased,&amtEvap,&amtDisp,&amtBeached,&amtOffmap,&amtFloating,&amtRemoved);
	
	GetLeUnitsStr(unitsStr,unitsForTotals);
				
	if(amtReleased < 100) numDecPlaces = 1;
	else numDecPlaces = 0;

	sprintf(s,"Splot Mass Balance Totals (Best estimate):");
	rightFooter.PrintLine(s);


	// code goes here, try to do this a little better
	// might want stringwithtrailingzeros so numbers line up better?
	StringWithoutTrailingZeros(infoStr,amtReleased,numDecPlaces); 
	sprintf(s, "Released:                   %s %s",infoStr,unitsStr);
	//sprintf(s, "Released:\t %s %s",infoStr,unitsStr);
	rightFooter.PrintLine(s,5);
	
	StringWithoutTrailingZeros(infoStr,amtEvap,numDecPlaces); 
	sprintf(s, "Evaporated/Dispersed: %s %s",infoStr,unitsStr);
	rightFooter.PrintLine(s,5);

	StringWithoutTrailingZeros(infoStr,amtBeached,numDecPlaces); 
	sprintf(s, "Beached:                     %s %s",infoStr,unitsStr);
	//sprintf(s, "Beached:    %s %s",infoStr,unitsStr);
	rightFooter.PrintLine(s,5);

	StringWithoutTrailingZeros(infoStr,amtOffmap,numDecPlaces); 
	sprintf(s, "Off Map:                     %s %s",infoStr,unitsStr);
	rightFooter.PrintLine(s,5);

	StringWithoutTrailingZeros(infoStr,amtFloating,numDecPlaces); 
	sprintf(s, "Floating:                    %s %s",infoStr,unitsStr);
	rightFooter.PrintLine(s,5);

	/////////////////////////////////////////////////
	/////////////////////////////////////////////////


}
/////////////////////////////////////////////////

void DrawFooter2(Rect r, Boolean bitmapIcons)
{
	char s[256],str[128];
	short i, offset,top,left,fontSize = kFooterFontSize;
	Rect textRect;
	short ht;
	short extraSpacing;
	OSErr err = 0;
	
	MySetRect(&textRect,r.left,r.top,(r.left+r.right)/2,r.bottom);
	extraSpacing = GetFontSpacing(fontSize,kPrinterOutputFont)/2;

	TextRect leftFooter(textRect,kPrinterOutputFont,fontSize,extraSpacing,extraSpacing);
	leftFooter.FrameBounds();
	if(model->fWizard->HaveOpenWizardFile())
	{
		short numWizLines = model->fWizard->NumLeftHandSideAnswersOnPrintout();
		for(i = 0; i< numWizLines;i++)
		{
			model->fWizard->LeftHandSideTextForPrintout(i,s);
			leftFooter.PrintLine(s);
		}
	}
		
	// wind info
	TWindMover *mover = model->GetWindMover(false);
	if(mover)
	{
		VelocityRec vel;
		if(mover->fIsConstantWind)
		{
			vel=mover->fConstantValue;
			UV2RThetaStrings(vel.u,vel.v,mover->timeDep -> GetUserUnits(),str);
			sprintf(s, "Wind: Constant %s",str);
			leftFooter.PrintLine(s);
		}
		else
		{
			DateTimeRec time;
			char timeStr[20];
			SecondsToDate (model->GetModelTime(), &time);
			Date2String(&time, timeStr);
			err = mover->GetTimeValue(model->GetModelTime(),&vel);
			UV2RThetaStrings(vel.u,vel.v,mover->timeDep -> GetUserUnits(),str);
			
			sprintf(s,"Wind: Variable %s",str);
			leftFooter.PrintLine(s);
			
			//sprintf(s, "at %s",timeStr);
			//leftFooter.PrintLine(s,5);
		}
	}
	
	if (model->GetNumForecastSpills()>1)
	{
		sprintf(s,"Number of Spills: %ld",model->GetNumForecastSpills());
		leftFooter.PrintLine(s);
	}
	{
		PtCurMap *map = GetPtCurMap();
		if (map && map -> ThereIsADispersedSpill())
		{
			// check for mixed layer depth, breaking wave height
			sprintf(s,"Mixed Layer depth : %g m", map->fMixedLayerDepth);
			leftFooter.PrintLine(s);
			//sprintf(s,"Breaking Wave Height : %g m", map->fBreakingWaveHeight);
			// need to check if calculated from wind speed, then different every step if variable wind
			//if (map->fWaveHtInput)
			sprintf(s,"Breaking Wave Height : %g m", map->GetBreakingWaveHeight());
			//else sprintf(s,"Breaking Wave Height computed from wind speed");
			leftFooter.PrintLine(s);
			if (map->fContourDepth1==-2)
				strcpy(s,"Contour depth : Bottom meter");
			else
				sprintf(s,"Contour depth : %g to %g m",map->fContourDepth1,map->fContourDepth2);
			leftFooter.PrintLine(s);
			if (map->fDiagnosticStrType==DEPTHATCENTERS)
				sprintf(s,"Diagnostic : Triangle depth in meters");
			else if (map->fDiagnosticStrType==NUMLESINTRIANGLE)
				sprintf(s,"Diagnostic : Number of LEs in Triangle");
			else if (map->fDiagnosticStrType==TRIANGLEAREA)
				sprintf(s,"Diagnostic : Triangle Area in km^2");
			else if (map->fDiagnosticStrType==CONCENTRATIONLEVEL)
				sprintf(s,"Diagnostic : Concentration Level in mg/L");
			else if (map->fDiagnosticStrType==SUBSURFACEPARTICLES)
				sprintf(s,"Diagnostic : Subsurface Particles");
			if (s[0]=='D') leftFooter.PrintLine(s);
		}
	}
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////


	MySetRect(&textRect,(r.left+r.right)/2+1,r.top,r.right,r.bottom);
	TextRect rightFooter(textRect,kPrinterOutputFont,fontSize,extraSpacing,extraSpacing);
	rightFooter.FrameBounds();
	
	//rightFooter.PrintLine("Black Splots: Best Guess, Red Splots: Uncertainty");
	
	/////////////////////////////////////////////////
	 // spill totals code
	
	double amtReleased=0,amtEvap=0,amtDisp=0,amtBeached=0,amtOffmap=0,amtFloating=0,amtRemoved=0,amtTotal=0;
	short unitsForTotals = BARRELS;
	char unitsStr [64];
	char infoStr [255];
	short numDecPlaces;
	short numLeSets;
	TLEList* thisLEList;
	DispersionRec dispInfo;
	Boolean thereisaSubsurfaceSpill = false;
	//AdiosInfoRecH adiosBudgetTable = ((TOLEList *)theLEList) -> GetAdiosInfo();
	
	for(numLeSets = model->LESetsList->GetItemCount(),i=numLeSets-1; i>=0 ; i--)
	{
		model->LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if(thisLEList -> GetLEType() != UNCERTAINTY_LE )
		{
			// check for dispersed oil and total number of LE sets
			dispInfo = ((TOLEList*)thisLEList) -> GetDispersionInfo();
			if (dispInfo.bDisperseOil)
			{
				sprintf(s,"Disperse %g%% after %g hrs",dispInfo.amountToDisperse*100,(float)dispInfo.timeToDisperse/3600.);
				leftFooter.PrintLine(s);
			}
			if (((TOLEList*)thisLEList)->fAdiosDataH)
			{
				leftFooter.PrintLine("Natural dispersion from ADIOS");
				sprintf(s,"%s",((TOLEList*)thisLEList)->fOilTypeName);
				leftFooter.PrintLine(s);
			}
			if (((TOLEList*)thisLEList)->fSetSummary.z > 0)	thereisaSubsurfaceSpill = true;
			unitsForTotals = thisLEList->GetMassUnits(); //return units for the first nonuncertaintly le list
		}
	}
	// get the totals
	model->GetTotalAmountStatistics(unitsForTotals,&amtTotal,&amtReleased,&amtEvap,&amtDisp,&amtBeached,&amtOffmap,&amtFloating,&amtRemoved);
	
	GetLeUnitsStr(unitsStr,unitsForTotals);
				
	if(amtReleased < 100) numDecPlaces = 1;
	else numDecPlaces = 0;

	sprintf(s,"Splot Mass Balance Totals (Best guess):");
	rightFooter.PrintLine(s);

	StringWithoutTrailingZeros(infoStr,amtReleased,numDecPlaces); 
	sprintf(s, "Released: %s %s",infoStr,unitsStr);
	rightFooter.PrintLine(s,5);
	
	StringWithoutTrailingZeros(infoStr,amtEvap,numDecPlaces); 
	sprintf(s, "Evaporated: %s %s",infoStr,unitsStr);
	rightFooter.PrintLine(s,5);

	StringWithoutTrailingZeros(infoStr,amtDisp,numDecPlaces); 
	if (thereisaSubsurfaceSpill)
		sprintf(s, "Subsurface: %s %s",infoStr,unitsStr);
	else
		sprintf(s, "Dispersed: %s %s",infoStr,unitsStr);
	rightFooter.PrintLine(s,5);

	StringWithoutTrailingZeros(infoStr,amtBeached,numDecPlaces); 
	sprintf(s, "Beached: %s %s",infoStr,unitsStr);
	rightFooter.PrintLine(s,5);

	StringWithoutTrailingZeros(infoStr,amtOffmap,numDecPlaces); 
	sprintf(s, "Off Map: %s %s",infoStr,unitsStr);
	rightFooter.PrintLine(s,5);

	if (amtRemoved>0)
	{
		StringWithoutTrailingZeros(infoStr,amtRemoved,numDecPlaces); 
		sprintf(s, "Removed: %s %s",infoStr,unitsStr);
		rightFooter.PrintLine(s,5);
	}
	
	StringWithoutTrailingZeros(infoStr,amtFloating,numDecPlaces); 
	sprintf(s, "Floating: %s %s",infoStr,unitsStr);
	rightFooter.PrintLine(s,5);



	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
}
/////////////////////////////////////////////////


static short thinCount = 0;
#ifdef MAC
static Boolean macPrintError, highResolution = TRUE;
#endif

Boolean AreUsingThinLines(void)
{
	return (sharedPrinting && thinCount > 0);
}

void StartThinLines()
{
	if (thinCount != 0) SysBeep(1);
	thinCount++;

#ifdef MAC
	POINTH width = (POINTH)_NewHandle(4);
	
	
	PenNormal();
	if (highResolution) {
		INDEXH(width, 0).h = 100; INDEXH(width, 0).v = 1;
		PicComment(182, 4, (Handle)width); // SetLineWidth
	}
	PenSize(1, 1);
	DisposeHandle((Handle)width);
#else
	// force the pen size to reset
	short saveWidth = currentPort.penWidth;
	currentPort.penWidth = 0;
	PenStyle(currentPort.penStyle,saveWidth);
#endif
}

void StopThinLines()
{
	if (thinCount != 1) SysBeep(1);
	thinCount--;

#ifdef MAC
	POINTH width = (POINTH)_NewHandle(4);
	
	PenNormal();
	if (highResolution) {
		INDEXH(width, 0).h = 1; INDEXH(width, 0).v = 100;
		PicComment(182, 4, (Handle)width); // SetLineWidth
		PenSize(1, 1);
		INDEXH(width, 0).h = 1; INDEXH(width, 0).v = 1;
		PicComment(182, 4, (Handle)width); // SetLineWidth
	}
	PenSize(1, 1);
	DisposeHandle((Handle)width);
#else
	// force the pen size to reset
	short saveWidth = currentPort.penWidth;
	currentPort.penWidth = 0;
	PenStyle(currentPort.penStyle,saveWidth);
#endif
}


//////////////////////////// MAC CODE ///////////////////////////////////////////

#ifdef MAC

////////////////////// THE PRINTING OF THE PICTURE
Rect PrintDrawingRect(short reset, RECTPTR pageRect, RECTPTR headerRect, RECTPTR caveatRect, RECTPTR footerRect)
{
	short indentH, indentV, width, height;
	float pRatio, dRatio;
	short footHt;
	Rect rDraw;
	static Rect rPrint = { 0, 0, 1000, 1000 };
	
	if (reset) {
	//#if TARGET_API_MAC_CARBON
		if (reset==3)
		{
			width = PixelsPerInchScreen()*8.5;
			height = PixelsPerInchScreen()*11.;
			rPrint.bottom = height;
			rPrint.right = width;
			rPrint.top = 0;
			rPrint.left = 0;
		}
		else
	#if TARGET_API_MAC_CARBON
			rPrint = GetPrinterPageRect();	// this is ok for printing, not saving
	#else
		rPrint = (*gPrRecHdl)->prInfo.rPage; // pixel rectangle of printable page area
	#endif
		if (pageRect) (*pageRect) = rPrint;
		indentH = 24; // (*gPrRecHdl)->prInfo.iHRes / 3
		indentV = 24; // (*gPrRecHdl)->prInfo.iVRes / 3
		MyInsetRect(&rPrint, indentH, indentV);

		*headerRect = rPrint; // top,left,right
		headerRect->bottom = headerRect->top + HeaderHeight();
	
		*caveatRect = rPrint; // left and right
		caveatRect->top = headerRect->bottom + LittleSeparation();
		caveatRect->bottom = caveatRect->top + CaveatHeight();

		footHt  = FooterHeight();
		
		rPrint.top = caveatRect->bottom +LittleSeparation();  
		rPrint.bottom -= footHt+LittleSeparation(); // leave room for footer
		
		rDraw = MapDrawingRect();
		dRatio = (float)RectWidth(rDraw) / (float)RectHeight(rDraw),
		pRatio = (float)RectWidth(rPrint) / (float)RectHeight(rPrint);
		
		if (dRatio > pRatio) // rDraw flatter than rPrint
			rPrint.bottom = rPrint.top + round(RectWidth(rPrint) / dRatio);
		else // rDraw taller than rPrint
			rPrint.right = rPrint.left + round(RectHeight(rPrint) * dRatio);
			
		*footerRect = *headerRect; // left and right
		footerRect->top = rPrint.bottom + LittleSeparation();
		footerRect->bottom = footerRect->top + footHt;

	}
	
	return rPrint;
}

void DrawPage(GrafPtr port)
{
	Rect rPrint, rPage;
	RgnHandle saveClip;
	Rect saveClipRect;
	Rect headerRect,caveatRect,footerRect;
	OSErr err = 0;
	#define iPrAbort 128  // the value in old Mac documentation
	
	rPrint = PrintDrawingRect(TRUE, &rPage,&headerRect,&caveatRect,&footerRect);
	saveClip = NewRgn();
	GetClip(saveClip);
	MyClipRect(rPrint);
	//StartThinLines(); JLM moved this , because we only want this to apply to the map
	gSavingOrPrintingPictFile = true;
	SetTempDrawingRect(&rPrint);
	err = DrawMaps(GetWindowFromPort(port), rPrint, settings.currentView, TRUE);
	SetTempDrawingRect(0);
	//StopThinLines(); JLM moved this , because we only want this to apply to the map
	gSavingOrPrintingPictFile = false;
	SetClip(saveClip);
	DisposeRgn(saveClip);
	
	if (err)
		{ MyPMSessionSetError(iPrAbort); macPrintError = TRUE; return; }
	
	FrameRect(&rPrint);
	
	DrawHeader(headerRect, FALSE);
	DrawCaveat(caveatRect, FALSE);
	if (!settings.omitFooter) DrawFooter(footerRect, FALSE);
	
	//DrawFontTest(rPrint);
}

Boolean PrintMapToPrinter()
{
	Boolean accepted;
	//short copies, printerType;
	//TPPrPort printPort;
	//TPrStatus printStatus;
	char titleStr[256];
	GrafPtr printerGrafPtr = 0;
	//TGetRslBlk getResolutionBlock;
	GrafPtr savePort;
	Rect saveClipRect;
	Boolean bPrintedSuccessfully = false;
	OSErr err = 0;
	
	GetPortGrafPtr(&savePort);	
	saveClipRect = MyClipRect(MapDrawingRect());

	#if MACB4CARBON
		TPPrPort      	printPort = 0;
		TPrStatus     	printStatus;
		short copies;
	#else
		Boolean sessionDocumentWasAlreadyOpen = gSessionDocumentIsOpen;
	#endif
		
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(GS1(603, 0)); // Printing.  Press ESC to Cancel.	
	
	// SetOutlinePreferred(TRUE);
	macPrintError = FALSE;
	
	// note: we opened the printer in PrintMap()
	
	getwtitle(mapWindow,titleStr);
	err = DoJobPrintDialog(titleStr);
	if (err) goto done;

	SetWatchCursor();

	#if TARGET_API_MAC_CARBON
		/*err = PMGetFirstPage(gPrintSettings, &firstPage); // get the first page the user wants to print
		if (err) goto done;
		err = PMGetLastPage(gPrintSettings, &lastPage); // get the last page the user wants to print
		if (err) goto done;*/
		if(!sessionDocumentWasAlreadyOpen) {
			err = My_PMSessionBeginDocument();
			if(err) goto closeDocument;
		}
	#else
		printPort = PrOpenDoc(gPrRecHdl,nil,nil);
		err = PrError();
		if(err || !printPort) goto closeDocument;
	#endif

	//printPort = PrOpenDoc(gPrRecHdl, nil, nil);
	//if (!PrError()) {
		//getResolutionBlock.iOpCode = getRslDataOp;
		//PrGeneral((Ptr)&getResolutionBlock);
		//printerType = (**gPrRecHdl).prStl.wDev >> 8;
		// highResolution = printerType == 3; // bDevLaser
		
		SetWatchCursor();
		/*PrOpenPage(printPort, nil);
		if (!PrError() && !macPrintError)
			DrawPage((GrafPtr)printPort);
		PrClosePage(printPort);*/
		/////////////////////////////////////////////////
		#if	TARGET_API_MAC_CARBON
			err = PMSessionBeginPage(gPrintSession,gPageFormat, NULL);
			if (!err)
			{
				CGrafPtr printingContext = NULL;
				err = PMSessionGetGraphicsContext (gPrintSession,kPMGraphicsContextQuickdraw,(void**)&printingContext);
				if (!err){
					SetPortGrafPort ((CGrafPtr) printingContext);
					printerGrafPtr = (GrafPtr)printingContext;
					//GetPortBounds (printingContext, &pageRect);
				}
			}
		#else
			PrOpenPage(printPort,nil);
			err = PrError();
			if(!err) {
				printerGrafPtr = (GrafPtr)printPort;
			}
		#endif
		/////////////////////////////////////////////////
		
		if (!err && !macPrintError) {
			DrawPage(printerGrafPtr);
		//	StartThinLines();
		//	DrawPage(printerGrafPtr, sectionRect, pageNum,
		//			 i == 0 && j == 0, i == 0 && j == (g_col - 1));
		//	StopThinLines();
		}
		
		#if	TARGET_API_MAC_CARBON
			err = PMSessionEndPage(gPrintSession);
		#else
			PrClosePage(printPort);
		#endif
	//}
closeDocument:
	SetWatchCursor();
	#if	TARGET_API_MAC_CARBON
		// Close the print job.  This dismisses the progress dialog on Mac OS X.
		if(!sessionDocumentWasAlreadyOpen) err = My_PMSessionEndDocument();
	#else
	PrCloseDoc(printPort);
	
	if (!macPrintError)
	{
		for (copies = 1 ; copies <= (**gPrRecHdl).prJob.iCopies ; copies++)
		{
			if ((**gPrRecHdl).prJob.bJDocLoop == bSpoolLoop && PrError() == noErr) {
				SetWatchCursor();
				PrPicFile(gPrRecHdl, nil, nil, nil, &printStatus);
					err = PrError();
			}
		}
	}
	#endif
done:	
	// note: we will close the printer in PrintMap()
	SetPortGrafPort(savePort);	
	(void)MyClipRect(saveClipRect);
	InitCursor();
	return bPrintedSuccessfully;
}

#define HEADERBUFFER 55

/////////////////////////////////////////////////
void DrawPlotPage(GrafPtr port, Rect r,
						long startP, long endP, short plotType,
						Boolean drawGrid, Rect grid, ExRect bounds, Boolean isConcPlot)
{
	RgnHandle saveClip;
	OSErr err = 0;
	
	Rect rPrint, rPage, headerRect, caveatRect, footerRect;
	
	rPrint = PrintDrawingRect(TRUE, &rPage,&headerRect,&caveatRect,&footerRect);
	saveClip = NewRgn();
	GetClip(saveClip);
	//ClipRect(&r);
	ClipRect(&rPrint);
	//StartThinLines();
	//err = ConcPlot(&r, startP, endP, plotType, drawGrid, TRUE, &grid, &bounds);
	if (isConcPlot)
		err = ConcPlot(&rPrint, startP, endP, plotType, drawGrid, TRUE, &grid, &bounds);
	else
		err = OiledShorelinePlot(&rPrint, startP, endP, plotType, drawGrid, TRUE, &grid, &bounds);
	//StopThinLines();
	SetClip(saveClip);
	DisposeRgn(saveClip);
	
	if (err)
		{ MyPMSessionSetError(iPrAbort); macPrintError = TRUE; return; }
	
	//FrameRect(&r);
	FrameRect(&rPrint);
	DrawHeader2(headerRect, false);	// GNOME version
	DrawCaveat(caveatRect, false);
	DrawFooter2(footerRect, false);
}

// see PrintMapToPrinter()
OSErr PrintPlot(char *path, Boolean print, Rect r,
					 long startP, long endP, short plotType,
					 Boolean drawGrid, Rect grid, ExRect bounds, Boolean isConcPlot)
{	// this code is for printing or saving the concentration plots
	Boolean accepted;
	short i, f, copies, printerType;
	long longZero = 0, longCount = 4;
	Rect saveClip;
	GrafPtr oldPort;
	GrafPtr printerGrafPtr = 0;
	PicHandle p = 0;
#if MACB4CARBON
	TPPrPort printPort = 0;
	TPrStatus printStatus;
	//TGetRslBlk getResolutionBlock;
#else
	Boolean sessionDocumentWasAlreadyOpen = gSessionDocumentIsOpen;
#endif
	OSErr err = 0;
//#pragma unused(freeBytes)
	
	if (!print) {
		Rect rPrint, rPage;
		Rect headerRect,caveatRect,footerRect;
		GetDateTimeStrings(sharedTTime, sharedTDate,
					   sharedPTime, sharedPDate);
		//somewhere may want to allow user to set spill name

		//Rect headerRect = r;
		/*{
			short item = MyModalDialog(M53, mapWindow, 0, PMInit, PMClick);
			if(item == -1) {err = -1;return err;}
			if (item == M53OK) {
				DisplayMessage(0);
			}
		}*/
		//rPrint = PrintDrawingRect(TRUE, &rPage,&headerRect,&caveatRect,&footerRect);
#if TARGET_API_MAC_CARBON
		rPrint = PrintDrawingRect(3, &rPage,&headerRect,&caveatRect,&footerRect);
		hdelete(0, 0,path);
		if (err = hcreate(0, 0, path, 'ttxt', 'PICT'))
			{ TechError("SavePICTFile()", "hcreate()", err); return err; }
#else
		rPrint = PrintDrawingRect(TRUE, &rPage,&headerRect,&caveatRect,&footerRect);
#endif		
		if (err = hopendf(0, 0, path, fsCurPerm, &f))
			{ TechError("PrintPlot()", "hopendf()", err); return err; }
		
		GetPortGrafPtr(&oldPort);
		SetPortWindowPort(mapWindow);
		//saveClip = MyClipRect(rPrint);
		p = OpenPicture(&rPage);
		saveClip = MyClipRect(rPage);
		//StartThinLines();
		EraseRect(&rPage);	

		FillRectWithWhite(&rPage); // we do this so that the background will be white in OS X

		if (isConcPlot)
			err = ConcPlot(&rPrint, startP, endP, plotType, drawGrid, TRUE, &grid, &bounds);
		else 
			err = OiledShorelinePlot(&rPrint, startP, endP, plotType, drawGrid, TRUE, &grid, &bounds);
		//MyClipRect(saveClip);	// include header/footer in pict?
		//StopThinLines();
		MyFrameRect(&rPrint);
		DrawHeader2(headerRect, false);	// GNOME version
		DrawCaveat(caveatRect, false);	// GNOME version
		DrawFooter2(footerRect, false);	// GNOME version
		ClosePicture();
		MyClipRect(saveClip);	// include header/footer in pict?
		
		SetPortGrafPort(oldPort);
		
		if (err) {
			if (p) DisposeHandle((Handle)p); // KillPicture(p);
			FSClose(f);
			hdelete(0, 0, path);
			
			return err;
		}
		
		for (i = 0 ; i < 128 ; i++)
			FSWrite(f, &longCount, &longZero); // write blank PICT header
		
		longCount = _GetHandleSize((Handle)p);
		_HLock((Handle)p);
		err = FSWrite(f, &longCount, (Ptr)*p);
		_HUnlock((Handle)p);
		err |= FSClose(f);
		
		DisposeHandle((Handle)p); // KillPicture(p);
		
		if (err) {
			printError(GS1(118, 0)); // Error writing PICT to disk.
			hdelete(0, 0, path);
		}
	}
	else {
		short item;
		char titleStr[256];
	// hmmm we are occasionally crasing in OS X, try setting the port here, JLM 7/10/06
	GrafPtr savePort = 0;
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
		err = OpenPrinterAndValidate();
		if(err) goto done;

		item = MyModalDialog(M53, mapWindow, 0, PMInit, PMClick);
		if(item == -1) {err = -1;return err;}
		if (item == M53OK) {
			DisplayMessage(0);
		}	// for some reason this causes the cursor to go to watch
	getwtitle(mapWindow,titleStr);
	err = DoJobPrintDialog(titleStr);
	if (err) goto done;
		
		macPrintError = FALSE;	// move to OUTPUT where this is defined
		
		//PrOpen();
		//if (PrError()) return -1;
		
		SetWatchCursor();
		//GetPortGrafPtr(&oldPort);
	#if TARGET_API_MAC_CARBON
		/*err = PMGetFirstPage(gPrintSettings, &firstPage); // get the first page the user wants to print
		if (err) goto done;
		err = PMGetLastPage(gPrintSettings, &lastPage); // get the last page the user wants to print
		if (err) goto done;*/
		if(!sessionDocumentWasAlreadyOpen) {
			err = My_PMSessionBeginDocument();
			if(err) goto closeDocument;
		}
	#else
		printPort = PrOpenDoc(gPrRecHdl,nil,nil);
		err = PrError();
		if(err || !printPort) goto closeDocument;
	#endif

		//printPort = PrOpenDoc(gPrRecHdl, nil, nil);
		//if (!PrError()) {
			//getResolutionBlock.iOpCode = getRslDataOp;
		//	PrGeneral((Ptr)&getResolutionBlock);
			//printerType = (**gPrRecHdl).prStl.wDev >> 8;
			// highResolution = printerType == 3; // bDevLaser
			
			SetWatchCursor();
			/*PrOpenPage(printPort, nil);
			if (!PrError() && !macPrintError)
				DrawPlotPage((GrafPtr)printPort, r,
								  startP, endP, plotType, drawGrid, grid, bounds, isConcPlot);
			PrClosePage(printPort);*/
		/////////////////////////////////////////////////
		#if	TARGET_API_MAC_CARBON
			err = PMSessionBeginPage(gPrintSession,gPageFormat, NULL);
			if (!err)
			{
				CGrafPtr printingContext = NULL;
				err = PMSessionGetGraphicsContext (gPrintSession,kPMGraphicsContextQuickdraw,(void**)&printingContext);
				if (!err){
					SetPortGrafPort ((CGrafPtr) printingContext);
					printerGrafPtr = (GrafPtr)printingContext;
					//GetPortBounds (printingContext, &pageRect);
				}
			}
		#else
			PrOpenPage(printPort,nil);
			err = PrError();
			if(!err) {
				printerGrafPtr = (GrafPtr)printPort;
			}
		#endif
			if (!err && !macPrintError)
				DrawPlotPage(printerGrafPtr, r,
								  startP, endP, plotType, drawGrid, grid, bounds, isConcPlot);
		/////////////////////////////////////////////////
		
		//}
		#if	TARGET_API_MAC_CARBON
			err = PMSessionEndPage(gPrintSession);
		#else
			PrClosePage(printPort);
		#endif

closeDocument:
		SetWatchCursor();
	#if	TARGET_API_MAC_CARBON
		// Close the print job.  This dismisses the progress dialog on Mac OS X.
		if(!sessionDocumentWasAlreadyOpen) err = My_PMSessionEndDocument();
	#else
		PrCloseDoc(printPort);
		
		if (!macPrintError)
			for (copies = 1 ; copies <= (**gPrRecHdl).prJob.iCopies ; copies++)
				if ((**gPrRecHdl).prJob.bJDocLoop == bSpoolLoop && PrError() == noErr) {
					SetWatchCursor();
					PrPicFile(gPrRecHdl, nil, nil, nil, &printStatus);
				}
		
	#endif
done:
		ClosePrinter();
		SetPortGrafPort(savePort);
		
		InitCursor();
	}
	
	return err ? -1 : 0;
}
/////////////////////////////////////////////////

OSErr SavePlot(char *path, Boolean print, Rect r)
{	// for saving the map with header and footer to a file
	Boolean accepted;
	short i, f, copies, printerType, width;
	long longZero = 0, longCount = 4;
	Rect saveClipRect;
	RgnHandle saveClip;
	GrafPtr oldPort;
	GrafPtr printerGrafPtr = 0;
	PicHandle p = 0;
#if MACB4CARBON
	TPPrPort printPort;
	TPrStatus printStatus;
	//TGetRslBlk getResolutionBlock;
#else
	Boolean sessionDocumentWasAlreadyOpen = gSessionDocumentIsOpen;
#endif
	OSErr err = 0;
//#pragma unused(freeBytes)
		
	//return SavePICTFile(path,0,print);
	if (!print) 
	{
		Rect rPrint, rPage;
		Rect headerRect,caveatRect,footerRect;

		if (!(model->bSaveSnapshots))
		{// probably want to bring up dialog for first snapshot
		short item = MyModalDialog(M53, mapWindow, 0, PMInit, PMClick);
		if(item == -1) {err = -1;return err;}
		if (item == M53OK) {
			DisplayMessage(0);
		}
		}
		bOmitFooter = false;
		//err = OpenPrinterAndValidate();
		//if (err) ClosePrinter();
//#if TARGET_API_MAC_CARBON
		rPrint = PrintDrawingRect(3, &rPage,&headerRect,&caveatRect,&footerRect);
//#else
//		rPrint = PrintDrawingRect(TRUE, &rPage,&headerRect,&caveatRect,&footerRect);
//#endif
		//footerRect.bottom = footerRect.top + (7+1) * GetFontSpacing(kFooterFontSize,kPrinterOutputFont);
		bOmitFooter = settings.omitFooter;
	
#if TARGET_API_MAC_CARBON
		hdelete(0, 0,path);
		if (err = hcreate(0, 0, path, 'ttxt', 'PICT'))
			{ TechError("SavePICTFile()", "hcreate()", err); return err; }
#endif		
		if (err = hopendf(0, 0, path, fsCurPerm, &f))
			{ TechError("SavePlot()", "hopendf()", err); return err; }
		GetPortGrafPtr(&oldPort);
		//SetPortWindowPort(mapWindow);

		//RGBBackColor(&backColors[settings.backgroundColor]); // somehow the text background is black in preview
		//EraseRect(&rPage);	
		//FillRectWithWhite(&rPage); // we do this so that the background will be white in OS X
		saveClipRect = MyClipRect(rPage);
		//saveClipRect = MyClipRect(rPrint);
		p = OpenPicture(&rPage);
		EraseRect(&rPage);	
		//saveClipRect = MyClipRect(rPage);
		//saveClipRect = MyClipRect(rPrint);
		saveClip = NewRgn();
		GetClip(saveClip);
		MyClipRect(rPrint);
		
		//p = OpenPicture(&rPage);
		gSavingOrPrintingPictFile = true;
		SetTempDrawingRect(&rPrint);
		EraseRect(&rPage);	
		FillRectWithWhite(&rPage); // we do this so that the background will be white in OS X
		DrawMaps (mapWindow, rPrint, settings.currentView, FALSE);
		SetTempDrawingRect(0);
	
		MyClipRect(saveClipRect);
		SetClip(saveClip);
		DisposeRgn(saveClip);
		
		gSavingOrPrintingPictFile = false;
		MyFrameRect(&rPrint);
	
		//RGBBackColor(&backColors[settings.backgroundColor]); // somehow the text background is black in preview
		EraseRect(&headerRect);
		DrawHeader2(headerRect, false);	
		EraseRect(&caveatRect);
		DrawCaveat(caveatRect, false);
		EraseRect(&footerRect);
		//FillRectWithWhite(&footerRect); // we do this so that the background will be white in OS X
		DrawFooter2(footerRect, false);
	
		ClosePicture();
		
		//ClosePrinter();
		SetPortGrafPort(oldPort);
		//MyClipRect(saveClipRect);
		
		if (err) {
			//if (p) DisposeHandle((Handle)p); // KillPicture(p);
			if (p) {KillPicture(p); p = 0;}
			FSClose(f);
			hdelete(0, 0, path);
			
			return err;
		}
		
		for (i = 0 ; i < 128 ; i++)
			FSWrite(f, &longCount, &longZero); // write blank PICT header
		
		longCount = _GetHandleSize((Handle)p);
		_HLock((Handle)p);
		err = FSWrite(f, &longCount, (Ptr)*p);
		_HUnlock((Handle)p);
		err |= FSClose(f);
		
		//DisposeHandle((Handle)p); // KillPicture(p);
		KillPicture(p); p = 0;
		
		if (err) {
			printError(GS1(118, 0)); // Error writing PICT to disk.
			hdelete(0, 0, path);
		}
	}
	else {	// this code isn't used
		short item;
		char titleStr[256];
	// hmmm we are occasionally crasing in OS X, try setting the port here, JLM 7/10/06
	GrafPtr savePort = 0;
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
		err = OpenPrinterAndValidate();
		if(err) goto done;

		item = MyModalDialog(M53, mapWindow, 0, PMInit, PMClick);
		if(item == -1) {err = -1;return err;}
		if (item == M53OK) {
			DisplayMessage(0);
		}	// for some reason this causes the cursor to go to watch
	getwtitle(mapWindow,titleStr);
	err = DoJobPrintDialog(titleStr);
	if (err) goto done;
		
		macPrintError = FALSE;	// move to OUTPUT where this is defined
		
		//PrOpen();
		//if (PrError()) return -1;
		
		SetWatchCursor();
		//GetPortGrafPtr(&oldPort);
	#if TARGET_API_MAC_CARBON
		/*err = PMGetFirstPage(gPrintSettings, &firstPage); // get the first page the user wants to print
		if (err) goto done;
		err = PMGetLastPage(gPrintSettings, &lastPage); // get the last page the user wants to print
		if (err) goto done;*/
		if(!sessionDocumentWasAlreadyOpen) {
			err = My_PMSessionBeginDocument();
			if(err) goto closeDocument;
		}
	#else
		printPort = PrOpenDoc(gPrRecHdl,nil,nil);
		err = PrError();
		if(err || !printPort) goto closeDocument;
	#endif

		/*printPort = PrOpenDoc(gPrRecHdl, nil, nil);
		if (!PrError()) {
			getResolutionBlock.iOpCode = getRslDataOp;
			PrGeneral((Ptr)&getResolutionBlock);*/
			//printerType = (**gPrRecHdl).prStl.wDev >> 8;
			// highResolution = printerType == 3; // bDevLaser
			
			SetWatchCursor();
			/*PrOpenPage(printPort, nil);
			if (!PrError() && !macPrintError)
				DrawPage((GrafPtr)printPort);
			PrClosePage(printPort);*/
		/////////////////////////////////////////////////
		#if	TARGET_API_MAC_CARBON
			err = PMSessionBeginPage(gPrintSession,gPageFormat, NULL);
			if (!err)
			{
				CGrafPtr printingContext = NULL;
				err = PMSessionGetGraphicsContext (gPrintSession,kPMGraphicsContextQuickdraw,(void**)&printingContext);
				if (!err){
					SetPortGrafPort ((CGrafPtr) printingContext);
					printerGrafPtr = (GrafPtr)printingContext;
					//GetPortBounds (printingContext, &pageRect);
				}
			}
		#else
			PrOpenPage(printPort,nil);
			err = PrError();
			if(!err) {
				printerGrafPtr = (GrafPtr)printPort;
			}
		#endif
			if (!err && !macPrintError)
				DrawPage(printerGrafPtr);
		/////////////////////////////////////////////////
		
		//}
		#if	TARGET_API_MAC_CARBON
			err = PMSessionEndPage(gPrintSession);
		#else
			PrClosePage(printPort);
		#endif

closeDocument:
		SetWatchCursor();
	#if	TARGET_API_MAC_CARBON
		// Close the print job.  This dismisses the progress dialog on Mac OS X.
		if(!sessionDocumentWasAlreadyOpen) err = My_PMSessionEndDocument();
	#else
		PrCloseDoc(printPort);
		
		if (!macPrintError)
			for (copies = 1 ; copies <= (**gPrRecHdl).prJob.iCopies ; copies++)
				if ((**gPrRecHdl).prJob.bJDocLoop == bSpoolLoop && PrError() == noErr) {
					SetWatchCursor();
					PrPicFile(gPrRecHdl, nil, nil, nil, &printStatus);
				}
		
	#endif
done:
		ClosePrinter();
		SetPortGrafPort(savePort);
		
		InitCursor();
	}

	return err ? -1 : 0;
}
/////////////////////////////////////////////////

OSErr PageSetUp(Boolean quitting)
{
	// macintosh code
	OSStatus      	err = 0;
	Boolean accepted;
	// hmmm we are occasionally crasing in OS X, try setting the port here, JLM 7/10/06
	GrafPtr savePort = 0;
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(mapWindow);
	////
#pragma unused(quitting)
	Rect r1,r2,r3,r4;
	
	err = OpenPrinterAndValidate();
	if(err) goto done;

#if TARGET_API_MAC_CARBON
	err = PMSessionPageSetupDialog(gPrintSession, gPageFormat, &accepted);
#else
	accepted = PrStlDialog(gPrRecHdl);
	err = PrError();
#endif

	PrintDrawingRect(TRUE, &r1,&r2,&r3,&r4);
	
done:
	if(err) SysBeep(5); // give the user some indication that we had trouble
	ClosePrinter();
	SetPortGrafPort(savePort);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////

OSErr SavePICTFile(char *path, short type, Boolean print)
{
	short			i, f, width;
	long			longZero = 0, longCount = 4;
	Rect			saveClip, rPage, rPrint, headerRect, caveatRect, footerRect;
	WorldRect		bounds;
	GrafPtr			oldPort;
	PicHandle		p = 0;
	PortPixRecord	pixRecord;
	OSErr err = 0;
	
	bOmitFooter = false;
	rPrint = PrintDrawingRect(2, &rPage,&headerRect,&caveatRect,&footerRect);
	//footerRect.bottom = footerRect.top + (7+1) * GetFontSpacing(kFooterFontSize,kPrinterOutputFont);
	bOmitFooter = settings.omitFooter;
	GetDateTimeStrings(sharedTTime, sharedTDate,
				   sharedPTime, sharedPDate);

	/*if (settings.pictureSize == WINDOWSIZE) {	// no way to set this
		width = RectWidth(rPrint);
		rPage = rPrint = MapDrawingRect();
		rPage.top -= HEADERBUFFER;
		rPage.bottom += FOOTERBUFFER;
		rPage.right = rPage.left + width;
	}*/
	
	if (err = hopendf(0, 0, path, fsCurPerm, &f))
		{ TechError("SavePICTFile()", "hopendf()", err); return err; }
	
	gSavingOrPrintingPictFile = true;
	GetPortGrafPtr(&oldPort);
	SetPortWindowPort(mapWindow);
	//if (type == BITMAPSAVEFILE) {
	if (type == FT_PICTSAVEFILE) {
		if (err = OSNewPortPix(&pixRecord, mapWindow, rPage))
			{ TechError("SavePICTFile()", "OSNewPortPix()", err); goto done; }
		OSSetPort(&pixRecord);
		EraseRect(&rPage);
	}
	//saveClip = MyClipRect(rPrint);
	//if (type != BITMAPSAVEFILE)
	if (type != FT_PICTSAVEFILE)
		p = OpenPicture(&rPage);
	saveClip = MyClipRect(rPage);
	SetTempDrawingRect(&rPrint);
		//EraseRect(&rPage);	

		//FillRectWithWhite(&rPage); // we do this so that the background will be white in OS X

	err = DrawMaps(mapWindow, rPrint, settings.currentView, print);
	bounds = ScreenToWorldRect(rPage,MapDrawingRect(),settings.currentView);
	SetTempDrawingRect(0);
	//MyClipRect(saveClip);
	FrameRect(&rPrint);
	
	DrawHeader2(headerRect,false);
	DrawCaveat(caveatRect,false);
	DrawFooter2(footerRect,false);
	
	//if (type == BITMAPSAVEFILE) {
	if (type == FT_PICTSAVEFILE) {
		p = OpenPicture(&rPage);
		//OSCopyToWindow(&pixRecord, (GrafPtr)pixRecord.newPort, rPage, rPage, srcCopy);
		OSCopyToWindow(&pixRecord, GetWindowFromPort(pixRecord.newPort), rPage, rPage, srcCopy);
		ClosePicture();
		OSDisposePortPix(&pixRecord);
	}
	else
		ClosePicture();
	MyClipRect(saveClip);
	
	done:
	
	SetPortGrafPort(oldPort);
	gSavingOrPrintingPictFile = false;
	
	if (err) {
		//if (p) DisposeHandle((Handle)p); // KillPicture(p);
		if (p) {KillPicture(p); p = 0;}
		FSClose(f);
		hdelete(0, 0, path);
		
		return err;
	}
	
	for (i = 0 ; i < 128 ; i++)
		FSWrite(f, &longCount, &longZero); // write blank PICT header
	
	longCount = _GetHandleSize((Handle)p);
	_HLock((Handle)p);
	err = FSWrite(f, &longCount, (Ptr)*p);
	_HUnlock((Handle)p);
	err |= FSClose(f);
	
	//DisposeHandle((Handle)p); // KillPicture(p);
	KillPicture(p); p = 0;
	
	if (!err)
		err = AddpPrmResource(0, 0, path, bounds);
	
	if (err) {
		printError(GS1(118, 0)); // Error writing PICT to disk.
		hdelete(0, 0, path);
	}
	
	return err ? -1 : 0;
}


OSErr CopyLLToClipboard()
{ // copy lat lng from current mouse position
	return noErr;
}

OSErr CopyMapViewToClipboard()
{
	Rect saveClip, rDraw = MapDrawingRect();
	GrafPtr oldPort;
	PicHandle p;
	OSErr err = 0;
	
	SetWatchCursor();
	
	GetPortGrafPtr(&oldPort);
	SetPortWindowPort(mapWindow);
	saveClip = MyClipRect(rDraw);
	p = OpenPicture(&rDraw);
	err = DrawMaps(mapWindow, rDraw, settings.currentView, FALSE);
	MyClipRect(saveClip);
	ClosePicture();
	SetPortGrafPort(oldPort);
	if (err)
		{ DisposeHandle((Handle)p); return err; } // KillPicture(p);
	
	MyZeroScrap();
	_HLock((Handle)p);
	MyPutScrap(_GetHandleSize((Handle)p), 'PICT', (Ptr)*p);
	_HUnlock((Handle)p);
 	KillPicture(p);
	
	return 0;
}

////////////////////////////////////////////////////////////////////////////////

OSErr CopyObjectsToPICT()
{
	return 0;
}

//////////////////////////// WINDOWS CODE ///////////////////////////////////////

#else

static Boolean abortPrint;
static float winPrintScaleX, winPrintScaleY;
// static Rect g_pageRect = { 0, 0, 1000, 1000 };
static HANDLE devMode = 0;
static DialogPtr printCancelDialog;
static PRINTDLG pd;

// return the number of pixels in two vertical screen inches
short TwoScreenInches()
{
	short pixPerInch;
	HDC hDC;
	
	hDC = GetDC(mapWindow);
	pixPerInch = GetDeviceCaps(hDC, LOGPIXELSY);
	ReleaseDC(mapWindow, hDC);
	
	return pixPerInch * 2;
}


void DrawPage(GrafPtr port)
{
	Rect rPrint, rPage;
	Rect headerRect,caveatRect,footerRect;
	Rect saveClip;
	OSErr err = 0;
	
	rPrint = PrintDrawingRect(2, &rPage,&headerRect,&caveatRect,&footerRect);
	saveClip = MyClipRect(rPrint);
	gSavingOrPrintingPictFile = true;
	SetTempDrawingRect(&rPrint);
	err = DrawMaps(port, rPrint, settings.currentView, TRUE);
	SetTempDrawingRect(0);
	gSavingOrPrintingPictFile = false;
	
	rPage.right++;
	MyClipRect(rPage);
	MyFrameRect(&rPrint);
	
	DrawHeader(headerRect, FALSE);
	DrawCaveat(caveatRect, FALSE);
	//DrawFooter(footerRect, FALSE);
	if (!settings.omitFooter) DrawFooter(footerRect, FALSE);
	
	//DrawFontTest(rPrint);

	(void)MyClipRect(saveClip);
}

BOOL CALLBACK PrintDialogProc(HWND dialog, UINT message, WPARAM wParam, LPARAM lParam)
{
	SetWatchCursor();
	switch (message) {
		case WM_INITDIALOG :
			CenterDialog(dialog, 0);
			abortPrint = FALSE;
			return TRUE;
		
		case WM_COMMAND:
			mysetitext(dialog, 3, GS1(121, 0)); // Printing aborted.
			abortPrint = TRUE;
			return TRUE;
		
		case WM_SETCURSOR: // keep the cursor a watch
			return TRUE;
	}
	
	return FALSE;
}

BOOL CALLBACK AbortProc(HDC hdc, short code)
{
	MSG msg;
	
	while (!abortPrint && PeekMessage(&msg, printCancelDialog, 0, 0, PM_REMOVE)) {
		if (!printCancelDialog || IsDialogMessage(printCancelDialog, &msg)) {
           TranslateMessage(&msg);
           DispatchMessage(&msg);
		}
	}
	
	return !abortPrint;
}

Boolean PrintMapToPrinter()
{
	static Boolean firstTime = TRUE;
	GrafPtr savePort;
	Boolean result;
	//char str[50];
	char str[128];	// 50 is not long enough, causes Gnome to crash
	DOCINFO di;
	
	SetWatchCursor();
	
	memset((void *)&pd, 0, sizeof(PRINTDLG));
	pd.lStructSize = sizeof(PRINTDLG);
	pd.hwndOwner = FrontWindow();
	pd.Flags = PD_RETURNDC | PD_NOSELECTION | PD_USEDEVMODECOPIES;
	pd.hDevMode = devMode;
	// if (firstTime) { pd.nFromPage = pd.nToPage = pd.nCopies = 1; firstTime = FALSE; }
	
	result = PrintDlg(&pd);
	devMode = pd.hDevMode;
	if (pd.hDevNames) GlobalFree(pd.hDevNames);
	
	if (!result) return FALSE;

	if (!pd.hDC) { TechError("PrintMapToPrinter()", "PrintDlg()", 0); return FALSE; }
	
	printCancelDialog = CreateDialog(hInst, "PrintCancel", mapWindow, (DLGPROC)PrintDialogProc);
	SetAbortProc(pd.hDC, (ABORTPROC)AbortProc);	
	EnableWindow(mapWindow, FALSE);
	
	GetPortGrafPtr(&savePort);
	doCompleteIsWindowCheck = TRUE;
	SetPortWindowPort((WindowPtr)pd.hDC);
	
	if (abortPrint) goto exitError;
	
	di.cbSize = sizeof(DOCINFO);
	getwtitle(mapWindow, str);
	di.lpszDocName = str;
	di.lpszOutput = (LPTSTR)NULL;
	di.lpszDatatype = 0;
	di.fwType = 0;

	//if (StartDoc(pd.hDC, &di) == SP_ERROR)
	if (StartDoc(pd.hDC, &di) <= 0)
		{ if (!abortPrint) TechError("PrintMapToPrinter()", "StartDoc()", SP_ERROR); goto exitError; }
	
	if (StartPage(pd.hDC) <= 0)
		{ if (!abortPrint) TechError("PrintMapToPrinter()", "StartPage()", -1); goto exitError; }
	DrawPage((WindowPtr)pd.hDC);
	if (EndPage(pd.hDC) <= 0)
		{ if (!abortPrint) TechError("PrintMapToPrinter()", "EndPage()", -1); goto exitError; }
	
	if (EndDoc(pd.hDC) <= 0)
		{ if (!abortPrint) TechError("PrintMapToPrinter()", "EndDoc()", -1); goto exitError; }
	
	exitError:
		EnableWindow(mapWindow, TRUE);
		DestroyWindow(printCancelDialog);
		doCompleteIsWindowCheck = FALSE;
		SetPortGrafPort(savePort);
		DeleteDC(pd.hDC);
		InitCursor();
		
		return TRUE;
}

/*OSErr PrintPlot(char *path, Boolean print, Rect r,
					 long startP, long endP, short plotType,
					 Boolean drawGrid, Rect grid, ExRect bounds, Boolean isConcPlot)
{
	short header = print ? 55 : 0; // PixelsPerInch() / 1.5
	Rect blank;
	HDC hScrDC, hMemDC;
	HBITMAP hDDB, hOldBitmap;
	PixMapHandle hDIB = 0;
	GrafPtr savePort;
	Rect headerRect = r, footerRect = r;
	OSErr err = 0;
	
	if (!print)
		hdelete(0, 0, path);
	
	hDDB = CreateCompatibleBitmap(currentHDC, RectWidth(r), RectHeight(r) + header);
	if (!hDDB)
		{ TechError("SavePICTFile()", "CreateCompatibleBitmap()", 0); return -1; }
	
	hScrDC = CreateDC("DISPLAY", NULL, NULL, NULL);
	hMemDC = CreateCompatibleDC(hScrDC);
	
	hOldBitmap = (HBITMAP)SelectObject(hMemDC, hDDB);
	
	SetViewportOrgEx(hMemDC, -r.left, header - r.top,  0);
	
	GetPort(&savePort);
	SetPort((WindowPtr)hMemDC);
	
	EraseRect(&r);
	if (isConcPlot)
		err = ConcPlot(&r, startP, endP, plotType, drawGrid, TRUE, &grid, &bounds);
	else
		err = OiledShorelinePlot(&r, startP, endP, plotType, drawGrid, TRUE, &grid, &bounds);
	
	if (print) {
		MySetRect(&blank, r.left, r.top - header, r.right, r.top);
		EraseRect(&blank);
		//r.right--;
		//r.bottom--;
		r.right--;
		r.bottom-=3;
		r.top+=3;
		MyFrameRect(&r);
		headerRect.bottom = r.top;
		headerRect.top = r.top - 55;
		footerRect.bottom = r.bottom;
		footerRect.top = r.bottom + 55;
		DrawHeader(headerRect, false);
		DrawFooter(footerRect, false);
		//DrawHeader(r, r.top - header, header);
	}
	
	SetPort(savePort);
	
	hDDB = (HBITMAP)SelectObject(hMemDC, hOldBitmap);
	hDIB = BitmapToDIB(hDDB, 0);
	DeleteObject(hDDB);
	
	DeleteDC(hScrDC);
	DeleteDC(hMemDC);
	
	if (!hDIB)
		{ TechError("SavePICTFile()", "BitmapToDIB()", 0); return -1; }
	
	if (print)
		err = PrintDIB(hDIB, PW_BESTFIT, 0, 0, "DOGS image"); // err = SaveDIB(hDIB, "c:\\out.bmp");
	else
		err = SaveDIB(hDIB, path);
	
	DestroyDIB(hDIB);
	
	if (err)
		{ TechError("SavePICTFile()", "SaveDIB()", 0); return -1; }
	
	return 0;
}*/

OSErr PrintPlot(char *path, Boolean print, Rect r,
					 long startP, long endP, short plotType,
					 Boolean drawGrid, Rect grid, ExRect bounds, Boolean isConcPlot)
{
	short width, header = print ? 55 : 0; // PixelsPerInch() / 1.5
	Rect blank;
	HDC hScrDC, hMemDC;
	HBITMAP hDDB, hOldBitmap;
	PixMapHandle hDIB = 0;
	GrafPtr savePort;
	Rect headerRect = r, footerRect = r, caveatRect = r, rPrint, rPage;
	OSErr err = 0;
	
	if (!print)
		hdelete(0, 0, path);
	
	// use this to save pict which is the right size to print
	bOmitFooter = false;
	rPrint = PrintDrawingRect(3, &rPage,&headerRect,&caveatRect,&footerRect);
	//footerRect.bottom = footerRect.top + (7+1) * GetFontSpacing(kFooterFontSize,kPrinterOutputFont);
	bOmitFooter = settings.omitFooter;
/////////////////////////////////////////////////

	// for header
	GetDateTimeStrings(sharedTTime, sharedTDate,
				sharedPTime, sharedPDate);
	
	if (settings.pictureSize == WINDOWSIZE && !print) {
		width = RectWidth(rPrint);
		rPage = rPrint = MapDrawingRect();
		rPage.top -= TwoScreenInches();
		rPage.bottom += TwoScreenInches() * 0.50;
		rPage.right = rPage.left + width;
	}
	
	MyOffsetRect(&rPrint, -rPage.left, -rPage.top);
	MyOffsetRect(&headerRect, -rPage.left, -rPage.top);
	MyOffsetRect(&caveatRect, -rPage.left, -rPage.top);
	MyOffsetRect(&footerRect, -rPage.left, -rPage.top);
	MyOffsetRect(&rPage, -rPage.left, -rPage.top);
	
	hDDB = CreateCompatibleBitmap(currentHDC, RectWidth(rPage), RectHeight(rPage));
	if (!hDDB)
		{ TechError("SavePICTFile()", "CreateCompatibleBitmap()", 0); return -1; }
	
	hScrDC = CreateDC("DISPLAY", NULL, NULL, NULL);
	hMemDC = CreateCompatibleDC(hScrDC);
	
	hOldBitmap = (HBITMAP)SelectObject(hMemDC, hDDB);
	
	//SetViewportOrgEx(hMemDC, -r.left, header - r.top,  0);
	
	GetPort(&savePort);
	SetPort((WindowPtr)hMemDC);
	
	EraseRect(&rPage);
	SetTempDrawingRect(&rPrint);
	
	if (isConcPlot)
		err = ConcPlot(&rPrint, startP, endP, plotType, drawGrid, TRUE, &grid, &bounds);
	else
		err = OiledShorelinePlot(&rPrint, startP, endP, plotType, drawGrid, TRUE, &grid, &bounds);
	
	/*if (print) {
		MySetRect(&blank, r.left, r.top - header, r.right, r.top);
		EraseRect(&blank);
		//r.right--;
		//r.bottom--;
		r.right--;
		r.bottom-=3;
		r.top+=3;
		MyFrameRect(&r);
		headerRect.bottom = r.top;
		headerRect.top = r.top - 55;
		footerRect.bottom = r.bottom;
		footerRect.top = r.bottom + 55;
		DrawHeader(headerRect, false);
		DrawFooter(footerRect, false);
		//DrawHeader(r, r.top - header, header);
	}*/
	
	SetTempDrawingRect(0);
	// MyClipRect(saveClip);
	
	// erase left, above, right, and below the image
	MySetRect(&r, rPage.left, rPage.top, rPrint.left, rPage.bottom);
	EraseRect(&r);
	MySetRect(&r, rPage.left, rPage.top, rPage.right, rPrint.top);
	EraseRect(&r);
	MySetRect(&r, rPrint.right, rPage.top, rPage.right, rPage.bottom);
	EraseRect(&r);
	MySetRect(&r, rPage.left, rPrint.bottom, rPage.right, rPage.bottom);
	EraseRect(&r);
	
	rPrint.right = rPage.right - 1;
	rPrint.bottom--;
	MyFrameRect(&rPrint);
	footerRect.bottom--;
	
	EraseRect(&headerRect);
	DrawHeader2(headerRect,false);
	EraseRect(&caveatRect);
	DrawCaveat(caveatRect,false);
	EraseRect(&footerRect);
	DrawFooter2(footerRect,false);

	SetPort(savePort);
	
	hDDB = (HBITMAP)SelectObject(hMemDC, hOldBitmap);
	hDIB = BitmapToDIB(hDDB, 0);
	DeleteObject(hDDB);
	
	DeleteDC(hScrDC);
	DeleteDC(hMemDC);
	
	if (!hDIB)
		{ TechError("SavePICTFile()", "BitmapToDIB()", 0); return -1; }
	
	if (print)
		err = PrintDIB(hDIB, PW_BESTFIT, 0, 0, "DOGS image"); // err = SaveDIB(hDIB, "c:\\out.bmp");
	else
		err = SaveDIB(hDIB, path);
	
	DestroyDIB(hDIB);
	
	if (err)
		{ TechError("SavePICTFile()", "SaveDIB()", 0); return -1; }
	
	return 0;
}

OSErr SavePlot(char *path, Boolean print, Rect r)
{
	short header = print ? 55 : 0; // PixelsPerInch() / 1.5
	Rect blank;
	HDC hScrDC, hMemDC;
	HBITMAP hDDB, hOldBitmap;
	PixMapHandle hDIB = 0;
	GrafPtr savePort;
	Rect rPrint, rPage;
	Rect headerRect,caveatRect,footerRect;
	Rect saveClip;
	OSErr err = 0;
	
	return SavePICTFile(path,0,print);
	if (!print)
		hdelete(0, 0, path);
	
	hDDB = CreateCompatibleBitmap(currentHDC, RectWidth(r), RectHeight(r) + header);
	if (!hDDB)
		{ TechError("SavePlot()", "CreateCompatibleBitmap()", 0); return -1; }
	
	hScrDC = CreateDC("DISPLAY", NULL, NULL, NULL);
	hMemDC = CreateCompatibleDC(hScrDC);
	
	hOldBitmap = (HBITMAP)SelectObject(hMemDC, hDDB);
	
	SetViewportOrgEx(hMemDC, -r.left, header - r.top,  0);
	
	GetPort(&savePort);
	SetPort((WindowPtr)hMemDC);
	
	// want header, footer either way
	//rPrint = PrintDrawingRect(2, &rPage,&headerRect,&caveatRect,&footerRect);
	//saveClip = MyClipRect(rPrint);
	gSavingOrPrintingPictFile = true;
	SetTempDrawingRect(&rPrint);
	err = DrawMaps(mapWindow, rPrint, settings.currentView, FALSE);
	SetTempDrawingRect(0);
	gSavingOrPrintingPictFile = false;
	
	rPage.right++;
	MyClipRect(rPage);
	MyFrameRect(&rPrint);
	
	DrawHeader2(headerRect,false);
	DrawCaveat(caveatRect,false);
	DrawFooter2(footerRect,false);
	(void)MyClipRect(saveClip);
	
	/*if (print) {
		MySetRect(&blank, r.left, r.top - header, r.right, r.top);
		EraseRect(&blank);
		//r.right--;
		//r.bottom--;
		r.right--;
		r.bottom-=3;
		r.top+=3;
		MyFrameRect(&r);
		headerRect.bottom = r.top;
		headerRect.top = r.top - 55;
		footerRect.bottom = r.bottom;
		footerRect.top = r.bottom + 55;
		DrawHeader(headerRect, false);
		DrawFooter(footerRect, false);
		//DrawHeader(r, r.top - header, header);
	}*/
	
	SetPort(savePort);
	
	hDDB = (HBITMAP)SelectObject(hMemDC, hOldBitmap);
	hDIB = BitmapToDIB(hDDB, 0);
	DeleteObject(hDDB);
	
	DeleteDC(hScrDC);
	DeleteDC(hMemDC);
	
	if (!hDIB)
		{ TechError("SavePlot()", "BitmapToDIB()", 0); return -1; }
	
	if (print)
		err = PrintDIB(hDIB, PW_BESTFIT, 0, 0, "GNOME image"); // err = SaveDIB(hDIB, "c:\\out.bmp");
	else
		err = SaveDIB(hDIB, path);
	
	DestroyDIB(hDIB);
	
	if (err)
		{ TechError("SavePlot()", "SaveDIB()", 0); return -1; }
	
	return 0;
}

HDC GetPrinterDC()
{
	char profile[100], *device, *driver, *output;
	LPDEVMODE p;
	HDC result;
	
	p = (LPDEVMODE)GlobalLock(devMode);
	
	GetProfileString("windows", "device", "", profile, 64);
	if ((device = strtok(profile, ",")) &&
		(driver = strtok(0, ", ")) &&
		(output = strtok(0, ", "))) {
		result = CreateDC(driver, device, output, p);
		GlobalUnlock(devMode);
		
		return result;
	}
	
	return 0;
}

Rect PrintDrawingRect(short reset, RECTPTR pageRect, RECTPTR headerRect, RECTPTR caveatRect, RECTPTR footerRect)
{
	short width, height;
	short logicalPixsX,logicalPixsY,pixPerInchCurrent;
	float dRatio, pRatio;
	Rect rPage, rDraw = MapDrawingRect();
	HDC hDC;
	short footHt;
	static Rect rPrint = { 0, 0, 1000, 1000 };
	GrafPtr savePort;
	
	// make sure we have a devMode
	if (!devMode) {
		memset((void *)&pd, 0, sizeof(PRINTDLG));
		pd.lStructSize = sizeof(PRINTDLG);
		pd.hwndOwner = mapWindow;
		pd.Flags = PD_RETURNDEFAULT;
		PrintDlg(&pd);
		devMode = pd.hDevMode;
		if (pd.hDevNames) GlobalFree(pd.hDevNames);
		// Windows reports error, so we don't have to.
		if (!devMode) return rPrint; // TechError("PrintMap()", "PrintDlg()", 0);
	}
	
	if (reset) {
		if (reset==3)
		{
			if (!(hDC = GetDC(mapWindow)))
				{ TechError("PrintDrawingRect()", "GetDC(mapWindow)", 0); return rPrint; }
		}
		else
		{
			if (!(hDC = GetPrinterDC()))
				{ TechError("PrintDrawingRect()", "GetPrinterDC()", 0); return rPrint; }
		}
		
		GetPortGrafPtr(&savePort);
		SetPortGrafPort((GrafPtr)hDC);
		
		// pixel rectangle of printable page area scaled to match pixel size of screen ...
		logicalPixsX = GetDeviceCaps(hDC, LOGPIXELSX);
		logicalPixsY = GetDeviceCaps(hDC, LOGPIXELSY);
		pixPerInchCurrent = PixelsPerInchCurrent();
		winPrintScaleX = (float)pixPerInchCurrent / (float)logicalPixsX;
		winPrintScaleY = (float)pixPerInchCurrent / (float)logicalPixsY;
		width = GetDeviceCaps(hDC, HORZRES);
		height = GetDeviceCaps(hDC, VERTRES);
		if (reset==3)
		{
			width = TwoScreenInches()*3.25;
			height = TwoScreenInches()*4.25;
		}
		//  rPage is inset 1/3 of a logical inch
		MySetRect(&rPage, logicalPixsX / 3,
						  logicalPixsY / 3,
						  width - logicalPixsX / 3,
						  height - logicalPixsY / 3);
		// ... except don't do scaling if reset == 2 (special flag)
		if (reset != 2) {
			rPage.left *= winPrintScaleX;
			rPage.right *= winPrintScaleX;
			rPage.top *= winPrintScaleY;
			rPage.bottom *= winPrintScaleY;
		}
		if (pageRect) (*pageRect) = rPage;
		
		rPrint = rPage;
		
		*headerRect = rPrint; // top,left,right
		headerRect->bottom = headerRect->top + HeaderHeight();
	
		*caveatRect = rPrint; // left and right
		caveatRect->top = headerRect->bottom + LittleSeparation();
		caveatRect->bottom = caveatRect->top + CaveatHeight();

		footHt  = FooterHeight();
		
		rPrint.top = caveatRect->bottom +LittleSeparation();  
		rPrint.bottom -= footHt+LittleSeparation(); // leave room for footer
		
		dRatio = (float)RectWidth(rDraw) / (float)RectHeight(rDraw),
		pRatio = (float)RectWidth(rPrint) / (float)RectHeight(rPrint);
		
		if (dRatio > pRatio) // rDraw flatter than rPage
			rPrint.bottom = rPrint.top + round(RectWidth(rPrint) / dRatio);
		else // rDraw taller than rPage
			rPrint.right = rPrint.left + round(RectHeight(rPrint) * dRatio);

		*footerRect = *headerRect; // left and right
		footerRect->top = rPrint.bottom + LittleSeparation();
		footerRect->bottom = footerRect->top + footHt;
		
		SetPortGrafPort(savePort);
		if (reset==3) ReleaseDC(mapWindow,hDC);
		else
		DeleteDC(hDC);
	}
	
	return rPrint;
}


OSErr PageSetUp(Boolean quitting)
{
	PRINTDLG pd;
	DEVMODE *p;
	Boolean result;
	Rect r1,r2,r3,r4;
	OSErr err = 0;
	
	if (quitting == -1) {
		/*
		if (err = MyGetFileSize(MARPLOTvRef, MARPLOTdirID, "PRINTER.PLT", &size)) return 0;
		if (devMode = GlobalAlloc(GHND, size)) {
			p = (DEVMODE *)GlobalLock(devMode);
			err = ReadFileContents(NONTERMINATED,MARPLOTvRef, MARPLOTdirID,
								   "PRINTER.PLT", (char *)p, size, 0);
			GlobalUnlock(devMode);
			if (err) {
				printError("Error reading printer information.");
				GlobalFree(devMode);
				devMode = 0;
				return err;
			}
		}
		*/
		return 0;
	}
	
	if (quitting == TRUE) {
		err = 0;
		if (devMode) {
			p = (DEVMODE *)GlobalLock(devMode);
			/*
			if (err = WriteFileContents(MARPLOTvRef, MARPLOTdirID, "PRINTER.PLT", 'MRP1', 'PRNT',
								 		(char *)p, GlobalSize(devMode), 0))
				printError("Error saving printer information.");
			*/
			GlobalUnlock(devMode);
			devMode = GlobalFree(devMode);
		}
		return err;
	}
	
	memset((void *)&pd, 0, sizeof(PRINTDLG));
	pd.lStructSize = sizeof(PRINTDLG);
	pd.hwndOwner = mapWindow;
	pd.Flags = PD_PRINTSETUP;
	pd.hDevMode = devMode;
	result = PrintDlg(&pd);
	devMode = pd.hDevMode;
	if (pd.hDevNames) GlobalFree(pd.hDevNames);
	
	PrintDrawingRect(TRUE, &r1,&r2,&r3,&r4);
	
	return 0;
}


extern short PicFrameExtension(void);
extern void DrawCombinedImage(void * object,WorldRect wRect,Rect extendedRect);
void DrawCombinedImageWithHeader(void * object,WorldRect wRect,Rect extendedRect)
{
	short extra = PicFrameExtension();
	Rect  rectWithoutLabel = extendedRect;
	Rect  labelRect = extendedRect;
	short ht;
	
	rectWithoutLabel.bottom -= extra; // compensate for the extra amount added for the labels
	labelRect.top = rectWithoutLabel.bottom;
	
#ifdef IBM	// JLM 3/15/02 this function is only used for making movie frames
	PrepareToDraw(rectWithoutLabel, wRect, 0, 0);
	SetTempDrawingRect(&rectWithoutLabel);
	gRect = rectWithoutLabel;
#endif

	DrawCombinedImage(object,wRect,rectWithoutLabel);
	//////////////////
	//now draw the labeling info
	{
		DateTimeRec time;
		char *p,s[256];
		char str[256];
		FontInfo finfo;
		EraseRect(&labelRect);
		MyMoveTo(labelRect.left,labelRect.top);
		MyLineTo(labelRect.right,labelRect.top);
		//
		SecondsToDate (model->GetModelTime(), &time);
		Date2String(&time, s);
		if (p = strrchr(s, ':')) p[0] = 0; // remove seconds
		sprintf(str,"   %s",s);
		#ifdef MAC
			ht = labelRect.bottom -4;
		#else
			GetFontInfo(&finfo);
			ht =  (labelRect.bottom + labelRect.top + finfo.ascent)/2;
		#endif
		MyMoveTo(labelRect.left,ht);
		drawstring(str);
		//
	}
}

/////////////////////////////////////////////////
OSErr SavePICTFile(char *path, short type, Boolean print)
{
	short  width;
	Rect r, rPage, rPrint;
	HDC hScrDC, hMemDC;
	HBITMAP hDDB, hOldBitmap;
	PixMapHandle hDIB = 0;
	// HMETAFILE hmf;
	GrafPtr savePort;
#pragma unused(type)
	
	Rect headerRect,caveatRect,footerRect;
	OSErr err = 0;
	
	// use this to save pict which is the right size to print
	bOmitFooter = false;
	rPrint = PrintDrawingRect(3, &rPage,&headerRect,&caveatRect,&footerRect);
	//footerRect.bottom = footerRect.top + (7+1) * GetFontSpacing(kFooterFontSize,kPrinterOutputFont);
	bOmitFooter = settings.omitFooter;
/////////////////////////////////////////////////

	// dinosaur - user to save the screen as is
	/*rPage = rPrint = MapDrawingRect();
	rPage.top -= HeaderHeight() + CaveatHeight() + LittleSeparation();
	rPage.bottom += FooterHeight() + LittleSeparation();
	width = _max(RectWidth(rPage),3.25*TwoScreenInches());
	rPage.right = rPage.left + width;
	headerRect = rPage;
	headerRect.bottom = headerRect.top + HeaderHeight();
	caveatRect = headerRect;
	caveatRect.top = headerRect.bottom;
	caveatRect.bottom = rPrint.top - LittleSeparation();
	footerRect = rPage;
	footerRect.top = rPrint.bottom + LittleSeparation();
	*/
	/////////////////////////////////////////////////

	// for header
	GetDateTimeStrings(sharedTTime, sharedTDate,
				sharedPTime, sharedPDate);
	
	if (settings.pictureSize == WINDOWSIZE && !print) {
		width = RectWidth(rPrint);
		rPage = rPrint = MapDrawingRect();
		rPage.top -= TwoScreenInches();
		rPage.bottom += TwoScreenInches() * 0.50;
		rPage.right = rPage.left + width;
	}
	
	MyOffsetRect(&rPrint, -rPage.left, -rPage.top);
	MyOffsetRect(&headerRect, -rPage.left, -rPage.top);
	MyOffsetRect(&caveatRect, -rPage.left, -rPage.top);
	MyOffsetRect(&footerRect, -rPage.left, -rPage.top);
	MyOffsetRect(&rPage, -rPage.left, -rPage.top);
	
	if (!print)
		hdelete(0, 0, path);
	
	hDDB = CreateCompatibleBitmap(currentHDC, RectWidth(rPage), RectHeight(rPage));
	if (!hDDB)
		{ TechError("SavePICTFile()", "CreateCompatibleBitmap()", 0); return -1; }
	
	hScrDC = CreateDC("DISPLAY", NULL, NULL, NULL);
	hMemDC = CreateCompatibleDC(hScrDC);
	
	hOldBitmap = (HBITMAP)SelectObject(hMemDC, hDDB);
	
	//gSavingOrPrintingPictFile = true;
	GetPort(&savePort);
	//doCompleteIsWindowCheck = TRUE;
	SetPort((WindowPtr)hMemDC);
	
	EraseRect(&rPage);
	// saveClip = MyClipRect(rPrint);
	SetTempDrawingRect(&rPrint);
	err = DrawMaps((WindowPtr)hMemDC, rPrint, settings.currentView, print);
	SetTempDrawingRect(0);
	// MyClipRect(saveClip);
	
	// erase left, above, right, and below the image
	MySetRect(&r, rPage.left, rPage.top, rPrint.left, rPage.bottom);
	EraseRect(&r);
	MySetRect(&r, rPage.left, rPage.top, rPage.right, rPrint.top);
	EraseRect(&r);
	MySetRect(&r, rPrint.right, rPage.top, rPage.right, rPage.bottom);
	EraseRect(&r);
	MySetRect(&r, rPage.left, rPrint.bottom, rPage.right, rPage.bottom);
	EraseRect(&r);
	
	rPrint.right = rPage.right - 1;
	rPrint.bottom--;
	MyFrameRect(&rPrint);
	footerRect.bottom--;
	
	EraseRect(&headerRect);
	DrawHeader2(headerRect,false);
	EraseRect(&caveatRect);
	DrawCaveat(caveatRect,false);
	EraseRect(&footerRect);
	DrawFooter2(footerRect,false);
	//doCompleteIsWindowCheck = FALSE;
	SetPort(savePort);
	//gSavingOrPrintingPictFile = false;
	
	hDDB = (HBITMAP)SelectObject(hMemDC, hOldBitmap);
	hDIB = BitmapToDIB(hDDB, 0);
	DeleteObject(hDDB);
	
	DeleteDC(hScrDC);
	DeleteDC(hMemDC);
	
	if (!hDIB)
		{ TechError("SavePICTFile()", "BitmapToDIB()", 0); return -1; }
	
	if (print)
		err = PrintDIB(hDIB, PW_BESTFIT, 0, 0, "GNOME Analyst image");
	else
		err = SaveDIB(hDIB, path);
	
	DestroyDIB(hDIB);
	
	if (err)
		{ TechError("SavePICTFile()", "SaveDIB()", 0); return -1; }
	
	
	return 0;
}

void MyCopyBitsFromWindow(WindowPtr w, PixMapHandle *bits, Rect r)
{
	RECT R;
	
	(*bits) = 0;
	
	MakeWindowsRect(&r, &R);
	MapWindowPoints(w, GetDesktopWindow(), (POINT *)&R, 2);
	(*bits) = CopyScreenToDIB(&R);
}


OSErr CopyLLToClipboard()
{ // copy lat lng from current mouse position
	return noErr;
}

OSErr CopyMapViewToClipboard()
{
	PixMapHandle hDIB = 0;
	
	SetWatchCursor();
	MyCopyBitsFromWindow(mapWindow, &hDIB, MapDrawingRect());
	
	if (!hDIB)
		{ TechError("Copy()", "MyCopyBitsFromWindow()", 0); return -1; }
	
	if (!OpenClipboard(mapWindow)) {
		TechError("Copy()", "OpenClipboard()", 0);
		DestroyDIB(hDIB);
		
		return -1;
	}
	EmptyClipboard();
	SetClipboardData(CF_DIB, hDIB);
	CloseClipboard();
	
	return 0;
}

#endif

////////////////////////////////////////////////////////////////////////////////
// MOSS FILES


char gMossPath[256];
char gKMLPath[256];
Boolean gSaveMossFiles = false;
Boolean gSaveKMLFile = false;

/////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

OSErr AskUserForSaveFilename(char *previousPath,char *chosenPath,char * defaultExtension, Boolean changeExtension)
{
	char path[256] = "untitled.SAV", name[64];
	Point where = CenteredDialogUpLeft(M55);
	MySFReply reply;
	char ibmTypeStr[32] = "";
	char ibmBackwardsTypeStr[32] = "";
	long len,i;
	OSErr err = 0;
	
	if(defaultExtension)
	{
		len = strlen(defaultExtension);
		for(i=0;i<len;i++) ibmBackwardsTypeStr[i] = defaultExtension[len-1-i];
		ibmBackwardsTypeStr[len] = 0;
	}
	
	if(previousPath) strcpy(path, previousPath); 
	/////////
	// JLM 1/22/99 
	// try to get them to use the default extension
	// replace any trailing .3 extensions with the preferred one
	if(changeExtension && defaultExtension)
	{
		len = strlen(path);
		if(len >= 4) 
		{
			if(path[len-4] == '.')
			{	// replace the extension
				strcpy(&path[len-4],defaultExtension);
			}
		}
	}
	////////////
	
	SplitPathFile (path, name);
	
#if TARGET_API_MAC_CARBON
		mysfpputfile(&where, "Name:", name, (MyDlgHookUPP)0, &reply, 
	           M55, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path,reply.fullPath);
#else
#ifdef MAC
	//sfpputfile(&where, "Name:", name, (DlgHookUPP)0, &reply,
	//		   M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	// JLM 1/6/99 -- use standard sfput file
	sfputfile(&where, "Name:", name, (DlgHookUPP)0, &reply);
	
#else
	//JLM sfpputfile(&where, "MSO.", name, (MyDlgHookProcPtr)0, &reply,
	sfpputfile(&where, ibmBackwardsTypeStr, name, (MyDlgHookProcPtr)0, &reply,
	           M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
#endif
if (!reply.good) return USERCANCEL;
	
	my_p2cstr(reply.fName);
	
#ifdef MAC
//	GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
	GetFullPath (reply.vRefNum, 0, (char *) "", path);
	strcat (path, ":");
	strcat (path, (char *) reply.fName);
#else
	strcpy(path, reply.fName);
#endif
#endif	
	strcpy(chosenPath,path);
	return 0;//successful

}

///////////////////////////////////////////////////////////////////////////

static short sM22ChosenFileType;

static PopInfoRec FileTypesPopTable[] = {
		{ M22, nil, M22TYPESITEM, 0, 0, 0, 1, FALSE, nil }
	};

OSErr M22Init(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	switch(model->GetModelMode())
	{
		//default: case NOVICEMODE: FileTypesPopTable[0].menuID = pNOVICESAVEFILETYPES; break;
		default: case NOVICEMODE: FileTypesPopTable[0].menuID = pINTERMEDIATESAVEFILETYPES; break;
		case INTERMEDIATEMODE: FileTypesPopTable[0].menuID = pINTERMEDIATESAVEFILETYPES; break;
		case ADVANCEDMODE: 
		{
			if (gDispersedOilVersion) FileTypesPopTable[0].menuID = pDISPERSEDOILSAVEFILETYPES;
			else FileTypesPopTable[0].menuID = pADVANCEDSAVEFILETYPES; 
			break;
		}
	}
	
	RegisterPopTable(FileTypesPopTable, sizeof(FileTypesPopTable) / sizeof(PopInfoRec));
	RegisterPopUpDialog(M22, dialog);
	SetPopSelection (dialog, M22TYPESITEM, 1); 
#ifdef IBM
	SetFocus(GetDlgItem(dialog, M22TYPESITEM));
#endif

	return 0;
}

short ItemNumToFileType(short itemNum, short userMode)
{
	switch(userMode)
	{
		default: case NOVICEMODE: 
		case INTERMEDIATEMODE: 
			switch(itemNum)
			{
				case 1: return FT_LOCATIONSAVEFILE; 
				case 2: return FT_MOVIEFILE;
				//
				case 3: return FT_MOSSSAVEFILE;
				case 4: return FT_MOSSSAVEFILESERIES; 
				case 5: return FT_TATSAVEFILE; 
				case 6: return FT_TATSAVEFILESERIES; 
				case 7: return FT_NETCDFLEFILESNAPSHOT;
				case 8: return FT_NETCDFLEFILE;
				case 9: return FT_KMLLEFILESNAPSHOT;
				case 10: return FT_KMLLEFILE;
				case 11: return FT_TAPLEFILE;
				case 12: return FT_SAVEBNAMAP; 
				case 13: return FT_SAVEWIND; 
			}
		case ADVANCEDMODE:
			switch(itemNum)
			{
				case 1: return FT_ADVANCEDSAVEFILE; 
				case 2: return FT_MOVIEFILE; 
				case 3: return FT_MOSSSAVEFILE;
				case 4: return FT_MOSSSAVEFILESERIES; 
				case 5: return FT_TATSAVEFILE; 
				case 6: return FT_TATSAVEFILESERIES; 
				case 7: return FT_NETCDFLEFILESNAPSHOT;
				case 8: return FT_NETCDFLEFILE;
				case 9: return FT_KMLLEFILESNAPSHOT;
				case 10: return FT_KMLLEFILE;
				case 11: return FT_TAPLEFILE; 
				case 12: return FT_SAVEBNAMAP; 
				case 13: return FT_SAVEWIND; 
				//case 9: if (gDispersedOilVersion) return FT_OUTPUTCONC;  else return FT_SAVETOPOLOGY;
				case 14: return FT_SAVETOPOLOGY; 	// code goes here, add export topology to dispersed oil version...
				case 15: return FT_OUTPUTCONC; 	// code goes here, add export topology to dispersed oil version...
				case 16: return FT_BUDGETTABLE;
				case 17: return FT_PICTSAVEFILE;
				case 18: return FT_PICTSAVEFILESERIES;
			}
			break;
	}
	return -1;
}

short M22Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short	menuItemChosen;
	long	menuID_menuItem;

	switch (itemNum) {
		case M22CANCEL: return M22CANCEL;

		case M22OK:
			menuItemChosen = GetPopSelection (dialog, M22TYPESITEM);
			sM22ChosenFileType = ItemNumToFileType(menuItemChosen,model->GetModelMode());
			return itemNum;

		case M22TYPESITEM:
			PopClick(dialog, itemNum, &menuID_menuItem);
		break;
	}

	return 0;
}

/////////////////////////////////////////////////
OSErr AskUserForSaveFileType(short * fileType)
{
	short type, dItem,err=0;
	dItem = MyModalDialog (M22, mapWindow, 0, M22Init, M22Click);
	switch(dItem)
	{	
		case M22OK:
			*fileType = sM22ChosenFileType;
			return noErr;
		case M22CANCEL:
			return USERCANCEL;
	}
	return -1; 
}
/////////////////////////////////////////////////

OSErr CantSaveIfNoSpill(void)
{
	OSErr err = 0;
	long numSpills = model -> GetNumForecastSpills();
	if(numSpills > 0) return 0;
	
	printError("You must specify at least one spill before using this save option.");
	return -1;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

void BNAMapInit(DialogPtr dialog, VLISTPTR L)
{
	
	VLSetSelect(0, L); // automatic
}

Boolean BNAMapClick(DialogPtr dialog, VLISTPTR L, short dialogItem, long *listItem,
				Boolean doubleClick)
{
	//TCATSMover *mover;
	
	switch (dialogItem) {
		case BNA_OK:
			//sharedMoverList->GetListItem((Ptr)&mover, *listItem);
			//strcpy(sharedFileName, mover->className);
			
			return TRUE;
		case BNA_CANCEL: return TRUE;
		case BNA_LIST: 
			if(doubleClick) return TRUE;
	}
	
	return FALSE;
}



void DrawBNAMapItem(DialogPtr dialog, Rect *r, long item0Relative)
{
	char s[256] ="";
	VectorMap_c *map = GetNthVectorMap(item0Relative);
	
	TextFontSize(kFontIDGeneva,LISTTEXTSIZE);
	
	//sharedMoverList->GetListItem((Ptr)&mover, item);
	//strcpy(s, mover->className);
	
	if(map)
		map -> GetClassName(s);
		
	// MyMoveTo
	drawstring(s);	
	
	TextFontSize(0,12);
}

/////////////////////////////////////////////////

int ncFileSave(char *path) {
	int err;
	strncpy(path, "Untitled.nc", 256);
	err = AskUserForSaveFilename(path, path, ".nc", true);
	if(err)
		DisplayMessage(0);
	InitCursor();
#if TARGET_API_MAC_CARBON
	ConvertTraditionalPathToUnixPath(path, model->ncPath, 256);
	ConvertTraditionalPathToUnixPath(path, model->ncPathConfidence, 256);
#else
	strncpy(model->ncPath, path, 256);
	strncpy(model->ncPathConfidence, path, 256);
#endif
	int tLen = strlen(model->ncPath);
	if(!(tLen <= 256-11)) {
		strncpy(&model->ncPath[tLen-11], ".nc", 4);
		strncpy(&model->ncPathConfidence[tLen-11], "_uncert.nc", 11);
	}
	else {
		if(strcmp(&model->ncPath[tLen-3], ".nc") != 0) {
			printError("Please specify the extension of the file you wish to save.");
			return(-1);
		}
		else {
			strncpy(&model->ncPath[tLen-3], ".nc", 4);
			strncpy(&model->ncPathConfidence[tLen-3], "_uncert.nc", 11);
		}
	}
	return(0);
}


OSErr SaveAsHelper(Boolean bOnlyAdvancedSaveFile)
{
	char path[256];
	OSErr err = 0;
	short fileType;
	short item;
	Seconds saveOutputTimeStep;
	
	if(bOnlyAdvancedSaveFile)
	{
		fileType = FT_ADVANCEDSAVEFILE;
	}
	else
	{	// ask the user what they want to save
		err = AskUserForSaveFileType(&fileType);
		if(err) goto done; // note: might be user cancel
	}
	
	switch(fileType)
	{
		
		case FT_ADVANCEDSAVEFILE:
			// JLM 1/7/99
			// If the user has set the model back before the lastComputeTime
			// we cannot save, because the model is not in sync
			// e.g. the uncertainty durations would be messed up.
			if(model->GetModelTime() < model->GetLastComputeTime())
			{
				printError("You cannot save a diagnostic save file while the model time is less than the last computed time.");
				err = -1;
				goto done;
			}
			model -> GetSaveFileName (path);
			err = AskUserForSaveFilename(path,path,".SAV",true);
			if(err) goto done; // note: might be user cancel
			err = SaveSaveFile(path);
			if (!err) model -> SetSaveFileName (path);
			ResetWindowTitle(path);
			if (err) hdelete(0, 0, path);
			break;
			
		case FT_LOCATIONSAVEFILE:
			err = model->fWizard->SaveAsMenuHit();
			if(err) goto done;
			break;
		
		case FT_MOVIEFILE:
			if(!CanMakeMovie()) {
				printError("Unable to make a movie.  Check that Quicktime is properly installed.");
				break;
			}
			if(model->GetModelMode() < ADVANCEDMODE) err = CantSaveIfNoSpill();
			if(err) goto done;
			// may want to save the dump interval and restore after making the movie
			// also if at end of run will need to get rid of saved run bar files
			saveOutputTimeStep = model->LEDumpInterval;
			if (OutputOptionsDialog(1)) return -1;	// allow to change the number of movie frames
			err = model->OpenMovieFile();
			if(err) goto done; // note: might be user cancel
			// update the window before running, since we just invalidated part of the window 
			// with our dialog box
			#ifdef IBM
				UpdateWindow(mapWindow); // causes WM_PAINT to be sent
			#else
				UpdateMapWindow();
			#endif
			/////
			//model->Reset(); // start movie from the beginning
			model->SetModelToPastTime (model->GetStartTime());
			if (model->bHindcast) 
				model->SetModelToPastTime (model->GetEndTime());
			else
				model->SetModelToPastTime (model->GetStartTime());
			//if (OutputOptionsDialog(0)) return -1;	// may have to reset ?
			if ((!(model->bHindcast) && model->GetModelTime() != model->GetStartTime()) || ( model->bHindcast && model->GetModelTime() != model->GetEndTime()) || saveOutputTimeStep != model->LEDumpInterval)
			{
				model->Reset();
			}
			//if (model->GetModelTime() != model->GetStartTime() || saveOutputTimeStep != model->LEDumpInterval) model->Reset();	// so SAV files at the end of a run will reset
			//
			model->bMakeMovie = true;
			//model->Run(model->GetEndTime());
			if (!(model->bHindcast)) 
			{
				model->Run(model->GetEndTime());
			}
			else
			{
				model->Run(model->GetStartTime());
			}
			model->bMakeMovie = false;
			model->CloseMovieFile();
			// reset the LEDumpInterval for now
			model->LEDumpInterval = saveOutputTimeStep;
			break;
		
			
		case FT_MOSSSAVEFILE:
		case FT_MOSSSAVEFILESERIES:
			err = CantSaveIfNoSpill();
			if(err) goto done; 
			/*if (!model->IsUncertain()) // this error message was due to a bug in the GNOME Arc Tool which has been fixed 12/1/08
			{	
				MULTICHOICEALERT(1692,"",TRUE);	// ok is only option
				err = -1; goto done;
			}*/
			// force them through the header screen for the moss 3 file info
			gPMShowPrintButtons = false;
			if(fileType == FT_MOSSSAVEFILESERIES) gPMShowTrajectoryTime = false;
			else gPMShowTrajectoryTime = true;// let user  set trajectory time for FT_MOSSSAVEFILE
			item = MyModalDialog(M53, mapWindow, 0, PMInit, PMClick);
			gPMShowPrintButtons = true;
			gPMShowTrajectoryTime = true;
			if (item != M53OK) {err = -1;goto done;} // note: might be user cancel
			//
			if(!gMossPath[0]) strcpy(gMossPath,"moss");
			err = AskUserForSaveFilename(gMossPath,path,".MOS",false);
			if(err) goto done; // note: might be user cancel
			//
			// update the window before running, since we just invalidated part of the window 
			// with our dialog box
			#ifdef IBM
				UpdateWindow(mapWindow); // causes WM_PAINT to be sent
			#else
				UpdateMapWindow();
			#endif
			/////
			strcpy(gMossPath,path);
			switch(fileType)
			{
				case FT_MOSSSAVEFILE: // single time
					err = model->SaveMossLEFile(model->GetModelTime(), -1); // a negative fileNumber means use the file name the user selected
					//err = model->SaveKmlLEFile(model->GetModelTime(), -1); // a negative fileNumber means use the file name the user selected
					break;
				case FT_MOSSSAVEFILESERIES:
				{
					// we need to save the series from the beginning
					// but we want to use the saved series if we can so that we don't have to reset
					Boolean haveToReset = false;
					saveOutputTimeStep = model->GetOutputStep();	// this is actually not doing anything I think...
					if (OutputOptionsDialog(0)) return -1;	// may have to reset ?
					model->SuppressDirt(DIRTY_EVERYTHING);// so non-advanced users don't have the runbar reset
					if (model->bHindcast) 
						model->SetModelToPastTime (model->GetEndTime());
					else
						model->SetModelToPastTime (model->GetStartTime());
					//if (OutputOptionsDialog(0)) return -1;	// may have to reset ?
					if( (!(model->bHindcast) && model->GetModelTime() != model->GetStartTime()) || ( model->bHindcast && model->GetModelTime() != model->GetEndTime()) || saveOutputTimeStep != model->GetOutputStep())
					{	// in cases where the past times are not available 
						// the time would have been set to the closest available saved time.
						// Not equaling model->GetStartTime() means the past times are not available
						// so we have to reset and re-run the model to save the files
						haveToReset = true;
					}
					model->SuppressDirt(0);
					if(haveToReset) model->Reset();
				//
					gSaveMossFiles = true;
					//model->Run(model->GetEndTime());
					if (!(model->bHindcast)) 
					{
						model->Run(model->GetEndTime());
					}
					else
					{
						model->Run(model->GetStartTime());
					}
					//model->FinishKmlFile();
					// reset the OutputTimeStep for now
					model->SetOutputStep(saveOutputTimeStep);
					gSaveMossFiles = false;
					break;
				}
			}
			break;
			
		
		case FT_KMLLEFILESNAPSHOT:
		case FT_KMLLEFILE:
			err = CantSaveIfNoSpill();
			if(err) goto done; 
			/*if (!model->IsUncertain()) // this error message was due to a bug in the GNOME Arc Tool which has been fixed 12/1/08
			 {	
			 MULTICHOICEALERT(1692,"",TRUE);	// ok is only option
			 err = -1; goto done;
			 }*/
			// force them through the header screen for the moss 3 file info
			/*gPMShowPrintButtons = false;
			if(fileType == FT_KMLLEFILE) gPMShowTrajectoryTime = false;
			else gPMShowTrajectoryTime = true;// let user  set trajectory time for FT_MOSSSAVEFILE
			item = MyModalDialog(M53, mapWindow, 0, PMInit, PMClick);
			gPMShowPrintButtons = true;
			gPMShowTrajectoryTime = true;
			if (item != M53OK) {err = -1;goto done;} // note: might be user cancel
			*/
			//
			if(!gKMLPath[0]) strcpy(gKMLPath,"moss");
			err = AskUserForSaveFilename(gKMLPath,path,".MOS",false);
			if(err) goto done; // note: might be user cancel
			//
			// update the window before running, since we just invalidated part of the window 
			// with our dialog box
#ifdef IBM
			UpdateWindow(mapWindow); // causes WM_PAINT to be sent
#else
			UpdateMapWindow();
#endif
			/////
			strcpy(gKMLPath,path);
			switch(fileType)
		{
			case FT_KMLLEFILESNAPSHOT: // single time
				//err = model->SaveMossLEFile(model->GetModelTime(), -1); // a negative fileNumber means use the file name the user selected
				err = model->SaveKmlLEFile(model->GetModelTime(), -1); // a negative fileNumber means use the file name the user selected
				break;
			case FT_KMLLEFILE:
			{
				// we need to save the series from the beginning
				// but we want to use the saved series if we can so that we don't have to reset
				Boolean haveToReset = false;
				saveOutputTimeStep = model->GetOutputStep();	// this is actually not doing anything I think...
				if (OutputOptionsDialog(0)) return -1;	// may have to reset ?
				model->SuppressDirt(DIRTY_EVERYTHING);// so non-advanced users don't have the runbar reset
				if (model->bHindcast) 
					model->SetModelToPastTime (model->GetEndTime());
				else
					model->SetModelToPastTime (model->GetStartTime());
				//if (OutputOptionsDialog(0)) return -1;	// may have to reset ?
				if( (!(model->bHindcast) && model->GetModelTime() != model->GetStartTime()) || ( model->bHindcast && model->GetModelTime() != model->GetEndTime()) || saveOutputTimeStep != model->GetOutputStep())
				{	// in cases where the past times are not available 
					// the time would have been set to the closest available saved time.
					// Not equaling model->GetStartTime() means the past times are not available
					// so we have to reset and re-run the model to save the files
					haveToReset = true;
				}
				model->SuppressDirt(0);
				if(haveToReset) model->Reset();
				//
				gSaveKMLFile = true;
				//model->Run(model->GetEndTime());
				if (!(model->bHindcast)) 
				{
					model->Run(model->GetEndTime());
				}
				else
				{
					model->Run(model->GetStartTime());
				}
				model->FinishKmlFile();
				// reset the OutputTimeStep for now
				model->SetOutputStep(saveOutputTimeStep);
				gSaveKMLFile = false;
				break;
			}
		}
		break;
			
			
		case FT_TATSAVEFILE:
		case FT_TATSAVEFILESERIES:
		{
			Boolean saveBool;
			err = CantSaveIfNoSpill();
			if(err) goto done; 
			model -> GetOutputFileName (path);
			err = AskUserForSaveFilename(path,path,".TAT",false);
			if(err) goto done; // note: might be user cancel
			//
			// update the window before running, since we just invalidated part of the window 
			// with our dialog box
			#ifdef IBM
				UpdateWindow(mapWindow); // causes WM_PAINT to be sent
			#else
				UpdateMapWindow();
			#endif
			/////
			model -> SetOutputFileName (path);
			switch(fileType)
			{
				case FT_TATSAVEFILE: // single time
					err = model->SaveOSSMLEFile(model->GetModelTime(), -1); // a negative fileNumber means use the file name the user selected
					break;
				case FT_TATSAVEFILESERIES:
					// we need to save the series from the beginning
					// but we want to use the saved series if we can so that we don't have to reset
					Boolean haveToReset = false;
					saveOutputTimeStep = model->GetOutputStep();
					if (OutputOptionsDialog(0)) return -1;	// may have to reset ?
					model->SuppressDirt(DIRTY_EVERYTHING);// so non-advanced users don't have the runbar reset
					//model->SetModelToPastTime (model->GetStartTime());
					if (model->bHindcast) 
						model->SetModelToPastTime (model->GetEndTime());
					else
						model->SetModelToPastTime (model->GetStartTime());
					//if (OutputOptionsDialog(0)) return -1;	// may have to reset ?
					if( (!(model->bHindcast) && model->GetModelTime() != model->GetStartTime()) || ( model->bHindcast && model->GetModelTime() != model->GetEndTime()) || saveOutputTimeStep != model->GetOutputStep())
					//if( model->GetModelTime() != model->GetStartTime() || saveOutputTimeStep != model->GetOutputStep())
					{	// in cases where the past times are not available 
						// the time would have been set to the closest available saved time.
						// Not equaling model->GetStartTime() means the past times are not available
						// so we have to reset and re-run the model to save the files
						haveToReset = true;
					}
					model->SuppressDirt(0);
					if(haveToReset) model->Reset();
					//
					saveBool = model->WantOutput(); // oh... just in case
					model->SetWantOutput(true);
					//model->Run(model->GetEndTime());
					if (!(model->bHindcast)) 
					{
						model->Run(model->GetEndTime());
					}
					else
					{
						model->Run(model->GetStartTime());
					}
					model->SetWantOutput(saveBool);
					// reset the OutputTimeStep for now
					model->SetOutputStep(saveOutputTimeStep);
					break;
			}
			break;
		}
		case FT_NETCDFLEFILESNAPSHOT:
		{
			if(CantSaveIfNoSpill() != 0)
				break;
			int outputStepsCount = model->outputStepsCount;
			model->outputStepsCount = 1;
			if(-ncFileSave(path))
				break;
			model->ncSnapshot = true;
			err = NetCDFStore::Create(model->ncPath, true, &model->ncID);
			if (err) break;
			err = NetCDFStore::Define(model, false, &model->ncVarIDs, &model->ncDimIDs);
			if (err) break;
			err = NetCDFStore::Capture(model, false, &model->ncVarIDs, &model->ncDimIDs);
			if (err) break;
			err = NetCDFStore::fClose(model->ncID);
			if (err) break;
			if(model->IsUncertain()) {
				err = NetCDFStore::Create(model->ncPathConfidence, true, &model->ncID_C);
				if (err) break;
				err = NetCDFStore::Define(model, true, &model->ncVarIDs_C, &model->ncDimIDs_C);
				if (err) break;
				err = NetCDFStore::Capture(model, true, &model->ncVarIDs_C, &model->ncDimIDs_C);
				if (err) break;
				err = NetCDFStore::fClose(model->ncID_C);
				if (err) break;
			}
			model->outputStepsCount = outputStepsCount;
			model->ncSnapshot = false;
			break;
		}
		case FT_NETCDFLEFILE:
		{
			if(CantSaveIfNoSpill() != 0)
				break;
			if(-ncFileSave(path))
				break;
			model->writeNC = true;
			model->Reset();
			saveOutputTimeStep = model->GetOutputStep();
			model->SetOutputStep(model->GetTimeStep());	// default to model time step??
			if (OutputOptionsDialog(0)) return -1;	// may have to reset ?
			if (!(model->bHindcast)) 
			{
				model->Run(model->GetEndTime());
			}
			else
			{
				model->Run(model->GetStartTime());
			}
			//model->Run(model->GetEndTime());
			model->writeNC = false;
			model->SetOutputStep(saveOutputTimeStep);
			break;
		}
			
		case FT_TAPLEFILE:
		{
			Boolean saveBool;
			err = CantSaveIfNoSpill();
			if(err) goto done; 
			model -> GetOutputFileName (path);	// maybe save this and put it back??
			err = AskUserForSaveFilename(path,path,".TAT",false);
			if(err) goto done; // note: might be user cancel
			//
			// update the window before running, since we just invalidated part of the window 
			// with our dialog box
			#ifdef IBM
				UpdateWindow(mapWindow); // causes WM_PAINT to be sent
			#else
				UpdateMapWindow();
			#endif
			/////


			// create and open the output file
			memset(&gRunSpillForecastFile,0,sizeof(gRunSpillForecastFile));
			if(path[0]) {
				hdelete(0, 0, path);
				if (err = hcreate(0, 0, path, '\?\?\?\?', 'BINA'))
					{ TechError("SaveAsHelper()", "hcreate()", err); goto done ; }
				
				if (err = FSOpenBuf(0, 0, path, &gRunSpillForecastFile, 50000, FALSE))
					{ TechError("SaveAsHelper()", "FSOpenBuf()", err); goto done ; }
			}
		
	
			//model -> SetOutputFileName (path);
			//switch(fileType)
			{
				// we need to save the series from the beginning
				// but we want to use the saved series if we can so that we don't have to reset
				Boolean haveToReset = false;
				saveOutputTimeStep = model->GetOutputStep();
				if (OutputOptionsDialog(0)) return -1;	// may have to reset ?
				model->SuppressDirt(DIRTY_EVERYTHING);// so non-advanced users don't have the runbar reset
				//model->SetModelToPastTime (model->GetStartTime());
				if (model->bHindcast) 
					model->SetModelToPastTime (model->GetEndTime());
				else
					model->SetModelToPastTime (model->GetStartTime());
				//if (OutputOptionsDialog(0)) return -1;	// may have to reset ?
				if( (!(model->bHindcast) && model->GetModelTime() != model->GetStartTime()) || ( model->bHindcast && model->GetModelTime() != model->GetEndTime()) || saveOutputTimeStep != model->GetOutputStep())
				//if( model->GetModelTime() != model->GetStartTime() || saveOutputTimeStep != model->GetOutputStep())
				{	// in cases where the past times are not available 
					// the time would have been set to the closest available saved time.
					// Not equaling model->GetStartTime() means the past times are not available
					// so we have to reset and re-run the model to save the files
					haveToReset = true;
				}
				model->SuppressDirt(0);
				if(haveToReset) model->Reset();
				//
				//saveBool = model->WantOutput(); // oh... just in case
				//model->SetWantOutput(true);
				//model->Run(model->GetEndTime());
				if (!(model->bHindcast)) 
				{
					model->Run(model->GetEndTime());
				}
				else
				{
					model->Run(model->GetStartTime());
				}
				//model->SetWantOutput(saveBool);
				// reset the OutputTimeStep for now
				model->SetOutputStep(saveOutputTimeStep);
				// close the output file
				if (gRunSpillForecastFile.f)  FSCloseBuf(&gRunSpillForecastFile);
	
				memset(&gRunSpillForecastFile,0,sizeof(gRunSpillForecastFile));
				break;
			}
			break;
		}
		case FT_SAVEBNAMAP:
		{
			long vectorMapCount;
			long i,n;
			TMap *map;
			long numVectorMaps = 0;
			short buttonSelected;
			long listItemSelected;
			VectorMap_c* vMap;
			char suggestedFileName[256];

			n = model -> mapList->GetItemCount() ;
			for (i=0; i<n; i++)
			{
				model -> mapList->GetListItem((Ptr)&map, i);
				if (map->IAm(TYPE_VECTORMAP)) 
				{
					numVectorMaps++;
					vMap = dynamic_cast<TVectorMap*>(map);
				}
			}


			if (numVectorMaps==0)
			{
				printNote("There is no BNA map to export");
				err = -1;
				goto done;
			}
			
			if (numVectorMaps>1)
			{
				listItemSelected = SelectFromVListDialog(SELECTBNAMAP, BNA_LIST, numVectorMaps, BNAMapInit, 0, 0, DrawBNAMapItem, BNAMapClick, TRUE, &buttonSelected);
				switch(buttonSelected)
				{
					case BNA_OK:
					case BNA_LIST:
						break; // they chose listItemSelected
					default:
						err = -2;
						goto done;
						break;
					case BNA_CANCEL: 
						err = USERCANCEL;
						goto done;
						break;
	
				}
				vMap = GetNthVectorMap(listItemSelected);
			}
			
			if(vMap) 
			{
				dynamic_cast<TVectorMap *>(vMap) -> GetSuggestedFileName(suggestedFileName,".BNA");
				
				err = AskUserForSaveFilename(suggestedFileName,path,".BNA",TRUE);
				if(err) goto done; // note: might be user cancel
			
				err = dynamic_cast<TVectorMap *>(vMap) ->ExportAsBNAFileForGnomeAnalyst(path);
				goto done;
			}
			break;
		}
		
		case FT_SAVEWIND:
		{
			char* suggestedFileName = "WindData.SAV";
			TWindMover *wind = nil;
			wind = model -> GetWindMover(false); // don't create one
			if(!wind || wind->fIsConstantWind) 
			{
				printError("There is no variable wind to export");
				err = -1;
				goto done;
			}
			if(wind) 
			{
				err = AskUserForSaveFilename(suggestedFileName,path,".WND",TRUE);
				if(err) goto done; // note: might be user cancel
			
				err = wind ->ExportVariableWind(path);
				goto done;
			}
			break;
		}


		case FT_SAVETOPOLOGY:
		{	// write out topology and dagtree - only netcdf and ptcur have function implemented
			char* suggestedFileName = "Topology.DAT";
			TCurrentMover *mover = 0;
		
			// Figure out if there is a current
			mover = (TCurrentMover *)(model->GetMover(TYPE_CURRENTMOVER));
			if (mover)	// code goes here, check for netcdf or ptcur only
			{	
				err = AskUserForSaveFilename(suggestedFileName,path,".DAT",TRUE);
				if(err) goto done; // note: might be user cancel
				err = mover -> ExportTopology(path);
				if (err == 2) printNote("There is no netCDF or ptCur current mover. No topology will be exported");
				goto done;
			}
			else	
			{
				NetCDFWindMoverCurv *windMover = (NetCDFWindMoverCurv *)(model->GetMover(TYPE_NETCDFWINDMOVERCURV));
				if (windMover)
				{
					err = AskUserForSaveFilename(suggestedFileName,path,".DAT",TRUE);
					if(err) goto done; // note: might be user cancel
					err = windMover -> ExportTopology(path);
					if (err == 2) printNote("There is no curvilinear grid netCDF wind mover. No topology will be exported");
					goto done;
				}
				else
				{
					Map3D *map3D = 0;
					map3D = (Map3D*)(model->GetMap(TYPE_MAP3D));
					if (map3D) 
					{
						err = AskUserForSaveFilename(suggestedFileName,path,".DAT",TRUE);
						if(err) goto done; // note: might be user cancel
						err = map3D->ExportTopology(path);
						goto done;
					}
				}
				printNote("There is no current mover. No topology will be exported");
				err = -1;
				goto done;
			}
			break;
		}

		case FT_OUTPUTCONC:
		{
			//char* suggestedFileName = "OilConcentration.DAT";
			char* suggestedFileName = "TriArea.dat";
			PtCurMap *map = GetPtCurMap();
			if (!map) return -1;
			TTriGridVel3D* triGrid = map->GetGrid3D(true);	// used refined grid if there is one	
			if (!triGrid) return -1; 
		
			outputData **oilConcHdl = triGrid -> GetOilConcHdl();	// don't init
			double **areaHdl = triGrid -> GetTriAreaHdl();
			if (!oilConcHdl && !areaHdl)
			{
				printError("There is no output data to export");
				err = -1;
				goto done;
			}
			else 
			{
				err = AskUserForSaveFilename(suggestedFileName,path,".DAT",TRUE);
				if(err) goto done; // note: might be user cancel
			
				if (areaHdl) err = triGrid ->ExportTriAreaHdl(path, map->GetNumContourLevels());
				else if (oilConcHdl) err = triGrid ->ExportOilConcHdl(path);
				goto done;
			}
			break;
		}

		case FT_BUDGETTABLE:
		{
			long i, n;
			TLEList *thisLEList;
			static char suggestedFileName[256];
			
			if (suggestedFileName[0]==0) strcpy(suggestedFileName,"BudgetTable.dat");
		
			err = AskUserForSaveFilename(suggestedFileName,path,".DAT",TRUE);
			if(err) goto done; // note: might be user cancel
			model->ExportBudgetTableHdl(path);
			SplitPathFile (path, suggestedFileName);	// save users name

			/*for (i = 0, n = model->LESetsList->GetItemCount() ; i < n ; i++) {
				model->LESetsList->GetListItem((Ptr)&thisLEList, i);
				if(thisLEList -> GetLEType() == UNCERTAINTY_LE ) continue;
				((TOLEList*)thisLEList) -> ExportBudgetTableHdl(path);	// total all stats ??
				break;
			}*/
			break;

		}

		case FT_PICTSAVEFILE:	// map window plus header, footer
		{
			char path[256];
			Point where;
			Rect r = MapDrawingRect();
			MySFReply reply;
			char ibmBackwardsTypeStr[32] = "";
			Boolean changeExtension = false;	// for now
			char previousPath[256]="",defaultExtension[3]="";
			where = CenteredDialogUpLeft(M55);
#if TARGET_API_MAC_CARBON
			err = AskUserForSaveFilename("snapshot.pic",path,".pic",true);
			if (err) goto done;
#else
#ifdef MAC
			sfputfile(&where, "Name:", "snapshot.pic", (DlgHookUPP)0, &reply);
#else
			sfpputfile(&where, ibmBackwardsTypeStr, "snapshot.bmp", (MyDlgHookProcPtr)0, &reply,
						  M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
#endif
			if (!reply.good) break;
			
			my_p2cstr(reply.fName);
			hdelete(reply.vRefNum, 0, (char *)reply.fName);
			
			if (err = hcreate(reply.vRefNum, 0, (char *)reply.fName, 'ttxt', 'PICT'))
				{ TechError("PLOTDLG_CLICK()", "hcreate()", err); return err; }
			
#ifdef MAC
			GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
#else
			strcpy(path, reply.fName);
#endif
#endif
			err = SavePlot(path,0,r);
			break;
		}

		case FT_PICTSAVEFILESERIES:	// map window plus header, footer
		{
			char path[256];
			long nameLen;
			Boolean haveToReset = false;
			Point where;
			Rect r = MapDrawingRect();
			MySFReply reply;
			Boolean changeExtension = false;	// for now
			char previousPath[256]="",defaultExtension[3]="";
			char ibmBackwardsTypeStr[32] = "";
			if (OutputOptionsDialog(2)) return -1;	// allow to shift the start time of the snapshot series

			where = CenteredDialogUpLeft(M55);

#if TARGET_API_MAC_CARBON
			err = AskUserForSaveFilename("snapshot.pic",path,".pic",true);
			if (err) goto done;
#else
#ifdef MAC
			sfputfile(&where, "Name:", "snapshot.pic", (DlgHookUPP)0, &reply);
#else
			sfpputfile(&where, ibmBackwardsTypeStr, "snapshot.bmp", (MyDlgHookProcPtr)0, &reply,
						  M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
#endif
			if (!reply.good) break;
			
			my_p2cstr(reply.fName);
			hdelete(reply.vRefNum, 0, (char *)reply.fName);
			
			if (err = hcreate(reply.vRefNum, 0, (char *)reply.fName, 'ttxt', 'PICT'))
				{ TechError("PLOTDLG_CLICK()", "hcreate()", err); return err; }
			
#ifdef MAC
			GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
#else
			strcpy(path, reply.fName);
#endif
#endif
			hdelete(0, 0, path);

			nameLen=strlen(path);	
			// Chop off extension
			if(nameLen >= 4) 
			{
				long i;
				for(i=1; i<=4; i++)
				{
					if(path[nameLen-i] == '.')
					{	
						path[nameLen-i]=0;
						break; // only take off last extension
					}
				}
			}
			//if (OutputOptionsDialog(2)) return -1;	// allow to shift the start time of the snapshot series
			//err = AskUserForSaveFilename(suggestedFileName,path,".pic",TRUE);
			// strip the extension
			strcpy(model->fSnapShotFileName,path);
			//model->SetSnapshotFileName(path);
			model->SuppressDirt(DIRTY_EVERYTHING);// so non-advanced users don't have the runbar reset
			model->SetModelToPastTime (model->GetStartTime());
			if( model->GetModelTime() != model->GetStartTime())
			{	// in cases where the past times are not available 
				// the time would have been set to the closest available saved time.
				// Not equaling model->GetStartTime() means the past times are not available
				// so we have to reset and re-run the model to save the files
				haveToReset = true;
			}
			model->SuppressDirt(0);
			if(haveToReset) model->Reset();
			// make movie at the same time?
			model->bSaveSnapshots = true;
			model->Run(model->GetEndTime());
			if (model->bSaveSnapshots)
			{
				char dataFilePath[256];
				strcpy(dataFilePath,model->fSnapShotFileName);	// strip any extension and tack on a dat
				strcat(dataFilePath,".dat");
				PtCurMap *map = GetPtCurMap();
				if (!map) return -1;
				TTriGridVel3D* triGrid = map->GetGrid3D(true);	// used refined grid if there is one	
				if (!triGrid) return -1; 
				triGrid -> ExportAllDataAtSetTimes(dataFilePath);
				model->bSaveSnapshots = false;
			}
			break;
		}
		
		default:
			printError("This file type is unimplemented");
			err = USERCANCEL;
			goto done;
			break;
	}
	
done:

	DisplayMessage(0);
	InitCursor();
	
	return err;
}

OSErr AdvancedSaveFileSaveAs(void)
{
	Boolean bOnlyAdvancedSaveFile = true;
	return SaveAsHelper(bOnlyAdvancedSaveFile);
}

OSErr SaveAs()
{
	Boolean bOnlyAdvancedSaveFile = false;
	return SaveAsHelper(bOnlyAdvancedSaveFile);
}


