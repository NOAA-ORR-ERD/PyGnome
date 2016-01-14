
#include "CROSS.h"
#include "ObjectUtilsPD.h"
//#include "HtmlHelp.h"


short PixelsPerCentimeter()
{
	return PixelsPerInchScreen() / CENTPERINCH;
}

double PixelsPerMilDeg()
{
	// how many millionths of a degree latitude in a pixel?
	
	return 1000000 / (double)(MilesPerDegreeLat() * INPERMILE * PixelsPerInchScreen());
}


long PixelDiscrepancy(WorldRect wr)
{
	long topPixels, bottomPixels;
	WorldPoint center;
	
	center.pLong = wr.loLong;
	center.pLat = wr.hiLat;
	topPixels = LongToPixels(wr.hiLong - wr.loLong, center);
	center.pLat = wr.loLat;
	bottomPixels = LongToPixels(wr.hiLong - wr.loLong, center);
	
	return abs(topPixels - bottomPixels);
}

long LongToPixels(long dLong, WorldPoint center)
{
	float baseLat = center.pLat / 1000000.0, degreesLat;
	long pixels;
	
	degreesLat = ((dLong / 1000000.0) * LongToLatRatio(baseLat)) * 1000000.0;
	pixels = (degreesLat / PixelsPerMilDeg()) / CurrentScaleDenominator();
	
	return pixels;
}

/////////////////////////////////////////////////

void TextFontSize(long font,long size)  
{ // this should be faster on the IBM
#ifdef MAC
	TextFont(font);TextSize(size);
#else
	
	if (font == currentPort.font && size == currentPort.size) return;
	
	if (size > 100) size = 100;
	if (size < -100) size = -100;
	currentPort.size = size;
	currentPort.font = -100;// force it to reset the font in TextFont()
	TextFont(font); // reset the font to use the new size
#endif
}

void TextFontSizeFace(long font,long size,long style)  
{ // this should be faster on the IBM
#ifdef MAC
	TextFont(font);TextSize(size);TextFace(style);  
#else
	if (font == currentPort.font 
		&& size == currentPort.size
		&& style == currentPort.style) return;
	
	if (size > 100) size = 100;
	if (size < -100) size = -100;
	currentPort.size = size;
	currentPort.style = style;
	currentPort.font = -100;// force it to reset the font in TextFont()
	TextFont(font); // reset the font to use the new size
#endif
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////


#ifdef MAC
Boolean System7Available()
{
	long sysVersion;
	
	// if (!TrapAvailable(_GestaltDispatch)) return false;
	if (!Gestalt(gestaltSystemVersion, &sysVersion)) return sysVersion >= 0x0700;
	return false;
}
#else
Boolean System7Available()
{
	return TRUE;
}
#endif

Boolean ColorQDAvailable()
#ifdef MAC
{
	long qdVersion;

	if (!Gestalt(gestaltQuickdrawVersion, &qdVersion)) return qdVersion >= 0x0100;
	return false;
}
#else
{
	// (MIKE) return whether color is available on the current monitor, if possible
	// otherwise just return TRUE
	HDC hDC = GetDC(GetDesktopWindow());
	short nPlanes = GetDeviceCaps(hDC, PLANES),
		  nBitsPixel = GetDeviceCaps(hDC, BITSPIXEL);

	ReleaseDC(GetDesktopWindow(), hDC);

	return (nPlanes * nBitsPixel) >= 4;
}
#endif

Boolean WindowHasColor(WindowPtr w)
#ifdef MAC
{
#if TARGET_API_MAC_CARBON
	return IsPortColor(GetWindowPort(w));
#else
	return (((CGrafPtr)w)->portVersion & 0xC000) != 0;
#endif
}
#else
{
	HDC hDC = GetDC(w);
	short nPlanes = GetDeviceCaps(hDC, PLANES),
		  nBitsPixel = GetDeviceCaps(hDC, BITSPIXEL);

	ReleaseDC(w, hDC);

	///return (nPlanes * nBitsPixel) >= 1;//JLM
	return (nPlanes * nBitsPixel) > 1;
}
#endif

short NumColorBits()
#ifdef MAC
{
	GDHandle device = GetGDevice();
	
	return (**(**device).gdPMap).pixelSize;
}
#else
{
	HDC hDC = GetDC(GetDesktopWindow());
	short nPlanes = GetDeviceCaps(hDC, PLANES),
		  nBitsPixel = GetDeviceCaps(hDC, BITSPIXEL);

	ReleaseDC(GetDesktopWindow(), hDC);

	return nPlanes * nBitsPixel;
}
#endif

Rect GetPICTFrame(PicHandle hPICT)
{
//#ifdef MAC
	//return (**hPICT).picFrame;
#ifdef MAC
	Rect frame = (**hPICT).picFrame;
	#ifdef SWAP_BINARY
		frame.left = SwapShort(frame.left);
		frame.top = SwapShort(frame.top);
		frame.right = SwapShort(frame.right);
		frame.bottom = SwapShort(frame.bottom);
	#endif
	return frame;
#else
	return (**hPICT).picFrame;
	// return meta-file bounds
#endif
}

WindowPtr MyGetNewCWindow(short ID, Ptr p, WindowPtr behind)
#ifdef MAC
{
	return ColorQDAvailable() ? GetNewCWindow(ID, p, behind) :
										 GetNewWindow(ID, p, behind);
}
#else
{
	return 0;
}
#endif

void MyMoveWindow(WindowPtr w, short h, short v, Boolean bringToFront)
#ifdef MAC
{
	MoveWindow(w, h, v, bringToFront);
}
#else
{
	RECT R;
	
	GetWindowRect(w, &R);
	MoveWindow(w, h, v, R.right - R.left, R.bottom - R.top, TRUE);
}
#endif

void PlaceWindow(WindowPtr w, Rect r, Boolean redraw)
{
	Rect d;
	
	d = GetScreenBounds();
	if (r.right < d.left || r.left > d.right ||
		r.bottom < d.top || r.top > d.bottom)
		// r.left < d.left || r.right > d.right ||
		// r.top < d.top || r.bottom > d.bottom
		MyOffsetRect(&r, r.left - d.left, r.top - d.top);
#ifdef MAC
	MoveWindow(w, r.left, r.top, FALSE);
	SizeWindow(w, RectWidth(r), RectHeight(r), redraw);
#else
	MoveWindow(w, r.left, r.top, RectWidth(r), RectHeight(r), redraw);
#endif
}

// window rect in global coordinates
/*void GetWindowLocation(WindowPtr w, RECTPTR r)
#ifdef MAC
{
	RgnHandle rgn = ((WindowPeek)w)->contRgn;
	
	(*r) = (*rgn)->rgnBBox;
	// (*r) =  w->portRect;
}
#else
{
	RECT R;
	
	GetWindowRect(w, &R);
	MakeMacRect(&R, r);
}
#endif*/

// client rect in local coordinates
/*Rect GetPortRect(WindowPtr w)
#ifdef MAC
{
	return w->portRect;
}
#else
{
	RECT R;
	Rect r;

	if (MyIsWindow(w)) {
		GetClientRect(w, &R);
		MakeMacRect(&R, &r);
	}
	else
		MySetRect(&r, 0, 0, GetDeviceCaps((HDC)w, HORZRES), GetDeviceCaps((HDC)w, VERTRES));
	
	return r;
}
#endif*/

// client rect in local coordinates
Rect GetWindowPortRect(WindowRef w)
#ifdef MAC
{
	Rect bounds = {0,0,0,0};
	if(w){
		CGrafPtr port = GetWindowPort(w);
		#if TARGET_API_MAC_CARBON
		if(port) GetPortBounds(port,&bounds);
		#else
		if(port) bounds = port->portRect;
		#endif	
	}
	return bounds;
}
#else
{
	RECT R;
	Rect r;

	if (MyIsWindow(w)) {
		GetClientRect(w, &R);
		MakeMacRect(&R, &r);
	}
	else
		MySetRect(&r, 0, 0, GetDeviceCaps((HDC)w, HORZRES), GetDeviceCaps((HDC)w, VERTRES));
	
	return r;
}
#endif

// client rect in global coordinates
Rect SafeWindowBounds(WindowPtr w)
#ifdef MAC
{
	Rect bounds = {0,0,0,0};
	
	if (WindowHasColor(w))  {
		/// color port
		PixMapHandle portPixMap = 0;
		#if TARGET_API_MAC_CARBON
			portPixMap = GetPortPixMap(GetWindowPort(w));
		#else
			portPixMap = ((CGrafPtr)w)->portPixMap;
		#endif
		bounds = (**portPixMap).bounds;
	}
	else {
		// old-style port
		#if TARGET_API_MAC_CARBON
			bounds = GetPortBitMapForCopyBits(GetWindowPort(w))->bounds;
		#else
			bounds = w->portBits.bounds;
		#endif
	}
	
	return bounds;
}
#else
{
	Rect r;
	POINT P1, P2;
	
	r = GetWindowPortRect(w);
	
	P1.x = r.left;
	P1.y = r.top;
	P2.x = r.right;
	P2.y = r.bottom;
	
	ClientToScreen(w, &P1);
	ClientToScreen(w, &P2);
	
	r.left = P1.x;
	r.top = P1.y;
	r.right = P2.x;
	r.bottom = P2.y;
	
	return r;
}
#endif

ModalFilterUPP MakeModalFilterUPP(ModalFilterProcPtr p)
{
#ifdef MAC
	#define kMaxNumModalFilterProcs 6
	static ModalFilterProcPtr sProcs[kMaxNumModalFilterProcs];
	static ModalFilterUPP sUPP[kMaxNumModalFilterProcs];
	long i;
	ModalFilterUPP upp = 0;
	
	for(i = 0; i < kMaxNumModalFilterProcs; i++) {
		if (sProcs[i] == 0)
			break; // we got to the end of our list
		if (sProcs[i] == p)
			return sUPP[i]; // we found it in the list
	}
	// if we get here, we need to create the UPP
	#if MACB4CARBON
		upp = NewModalFilterProc(p);
	#else
		upp = NewModalFilterUPP(p);
	#endif
	if(i < kMaxNumModalFilterProcs) {
		sUPP[i]= upp;
		sProcs[i]= p;
	} 
	else {
		printNote("Need to allocate more space in MakeModalFilterUPP");
	}
	return upp;
#else
	return p;
#endif
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////
UserItemUPP MakeUserItemUPP(UserItemProcPtr p)
{
#ifdef MAC
	#define kMaxNumUserItemProcs 30
	static UserItemProcPtr sProcs[kMaxNumUserItemProcs];
	static UserItemUPP sUPP[kMaxNumUserItemProcs];
	long i;
	UserItemUPP upp = 0;
	
	for(i = 0; i < kMaxNumUserItemProcs; i++) {
		if (sProcs[i] == 0)
			break; // we got to the end of our list
		if (sProcs[i] == p)
			return sUPP[i]; // we found it in the list
	}
	// if we get here, we need to create the UPP
	#if MACB4CARBON
		upp = NewUserItemProc(p);
	#else
		upp = NewUserItemUPP(p);
	#endif
	if(i < kMaxNumUserItemProcs) {
		sUPP[i]= upp;
		sProcs[i]= p;
	} 
	else {
		printNote("Need to allocate more space in MakeUserItemUPP");
	}
	return upp;
#else
	return p;
#endif
}
/////////////////////////////////////////////////
#ifdef MAC
AEEventHandlerUPP MakeAEEventHandlerUPP(AEEventHandlerProcPtr p)
{
	#define kMaxNumAEEventProcs 6
	static AEEventHandlerProcPtr sProcs[kMaxNumAEEventProcs];
	static AEEventHandlerUPP sUPP[kMaxNumAEEventProcs];
	long i;
	AEEventHandlerUPP upp = 0;
	
	for(i = 0; i < kMaxNumAEEventProcs; i++) {
		if (sProcs[i] == 0)
			break; // we got to the end of our list
		if (sProcs[i] == p)
			return sUPP[i]; // we found it in the list
	}
	// if we get here, we need to create the UPP
	#if MACB4CARBON
		upp = NewAEEventHandlerProc(p);
	#else
		upp = NewAEEventHandlerUPP(p);
	#endif
	if(i < kMaxNumAEEventProcs) {
		sUPP[i]= upp;
		sProcs[i]= p;
	} 
	else {
		printNote("Need to allocate more space in MakeAEEventHandlerUPP");
	}
	return upp;
}

ControlActionUPP MakeControlActionUPP(ControlActionProcPtr p)
{
	#define kMaxNumControlActionProcs 6
	static ControlActionProcPtr sProcs[kMaxNumControlActionProcs];
	static ControlActionUPP sUPP[kMaxNumControlActionProcs];
	long i;
	ControlActionUPP upp = 0;
	
	for(i = 0; i < kMaxNumControlActionProcs; i++) {
		if (sProcs[i] == 0)
			break; // we got to the end of our list
		if (sProcs[i] == p)
			return sUPP[i]; // we found it in the list
	}
	// if we get here, we need to create the UPP
	#if MACB4CARBON
		upp = NewControlActionProc(p);
	#else
		upp = NewControlActionUPP(p);
	#endif
	if(i < kMaxNumControlActionProcs) {
		sUPP[i]= upp;
		sProcs[i]= p;
	} 
	else {
		printNote("Need to allocate more space in MakeControlActionUPP");
	}
	return upp;
}
#endif

/////////////////////////////////////////////////
MyDlgHookUPP MakeDlgHookUPP(MyDlgHookProcPtr p)
{
#ifdef MAC
	// this function is no longer used on the Mac
	MyDlgHookUPP upp = 0;
	return upp;
#else
	return p;
#endif
}

UniversalProcPtr MakeUPP(ProcPtr p, ProcInfoType pInfo)
#if MACB4CARBON	// double check this
//#ifdef VERSIONPPC
{
	long i;
	UniversalProcPtr uPP;
	static long count = 0;
	static ProcPtr procs[100];
	static RoutineDescriptor routines[100];
	// static UniversalProcPtr routines[100];
	
	if(p == nil) return 0;//JLM	 , 8/13/98

	if (count == 100)
		{ SysBeep(1); return NewRoutineDescriptor(p, pInfo, GetCurrentISA()); } // 0
	
	for (i = 0 ; i < count ; i++)
		if (procs[i] == p)
			break;
	
	if (i == count) {
		procs[i] = p;
		// routines[i] = NewRoutineDescriptor(p, pInfo, GetCurrentISA());
		uPP = NewRoutineDescriptor(p, pInfo, GetCurrentISA());
		routines[i] = *uPP;
		DisposeRoutineDescriptor(uPP);
		count++;
	}
	
	return &routines[i];
}
#else
{
#pragma unused(pInfo)
	
	return (UniversalProcPtr) p;
}
#endif

///// DIALOGS ///////////////////////////////////////////////////////////////////////

Boolean resetCursor = TRUE;
EventRecord lastEvent;

//void MyTEKey(char key, WindowPtr dialog)
void MyTEKey(char key, DialogPtr dialog)
#ifdef MAC
{
	TEHandle textH = 0;
#if TARGET_API_MAC_CARBON
	textH = GetDialogTextEditHandle(dialog);
#else
	textH = ((DialogPeek)dialog)->textH;
#endif
	TEKey(key, textH);
	//TEKey(key, ((DialogPeek)dialog)->textH);
}
#else
{
	// (MIKE) insert key key into the current text edit item of dialog
}
#endif

//void EnableTextItem(WindowPtr dialog, short item, Boolean enabled)
void EnableTextItem(DialogPtr dialog, short item, Boolean enabled)
#ifdef MAC
{
	short type = enabled ? editText : statText + itemDisable;
	Rect r;
	GrafPtr oldPort;
	
	if (GetDialogItemType(dialog, item) == type) return;
	
	GetPortGrafPtr(&oldPort);
	SetPortDialogPort(dialog);
	SetDialogItemType(dialog, item, type);
	r = GetDialogItemBox(dialog, item);
	MyInsetRect(&r, -4, -4);
	EraseRect(&r);
	InvalRectInWindow(r, GetDialogWindow(dialog));
	SetPortGrafPort(oldPort);
}
#else
{
	MyEnableControl(dialog, item, enabled);
}
#endif

//Boolean DisabledStatTextItem(WindowPtr dialog, short item)
Boolean DisabledStatTextItem(DialogPtr dialog, short item)
#ifdef MAC
{
	return GetDialogItemType(dialog, item) == (statText + itemDisable);
}
#else
{
	// (MIKE) is item item of dialog dialog a disabled static text item?
	return FALSE;
}
#endif

void MyShowHide(WindowPtr w, Boolean show)
#ifdef MAC
{
	if (show)
		ShowWindow(w);
	else
		HideWindow(w);
}
#else
{
	ShowWindow(w, show ? SW_SHOW : SW_HIDE);
}
#endif

#ifdef MAC
Boolean DragDuringDialog(DialogPtr theDialog, EVENTRECORDPTR theEvent)
{
	WindowPtr		windowHit;
	//long			message, result;
	Rect			r;
	Rect			screenRect = GetScreenBounds();
	
	if (theEvent->what != mouseDown)
		return false;
	
	switch (FindWindow(theEvent->where, &windowHit)) {
		//case inMenuBar:
		//	result = MenuSelect(theEvent->where);
		//	message = DOMENU(stptr, result);
		//	if (message == notHandled)
		//		menu_handler(stptr, result);
		//	HiliteMenu(0);
		//	return true;
		
		case inDrag:
			if (windowHit == GetDialogWindow(theDialog)) {
				MySetRect(&r, screenRect.left + 4,
								screenRect.top + 20 + 4,
								screenRect.right - 4,
								screenRect.bottom - 4);
				DragWindow(windowHit, theEvent->where, &r);
				return true;
			}
	}
	
	return false;
}
#endif

#ifdef MAC
void FlashItem(DialogPtr theDialog, short item)
{
	ControlHandle	button;
//#ifdef MPW //JLM
	//long	finalTicks;
//#else
	unsigned long	finalTicks;
//#endif
	
	button = (ControlHandle)GetDialogItemHandle(theDialog, item);
	HiliteControl(button, 1);
	Delay(8, &finalTicks);
	HiliteControl(button, 0);
}
//#endif

void HideButtonOutlinePicture7000(DialogRef theDialog)
{
	// hide the dialog item that is outlining the default button in 
	short i;
	short numItems = CountDITL(theDialog); 
	short itemType;
	if(numItems > 5) numItems = 5; // just look through the first 5 items
	for(i=2; i <= numItems; i++) {
		itemType = GetDialogItemType(theDialog,i);
		if(itemType == picItem   || itemType == (picItem + itemDisable)){
			HideDialogItem(theDialog,i);	
		}
	}
}


/////////////////////////////////////////////////

short DetermineCancelButtonItemNumByLookingAtButtons(DialogRef theDialog)
{
	char theTitle[256], officalCancelSpelling[256], officalCloseSpelling[256];
	short    itemType,numItems;
	short		buttonNum;
	
	#if TARGET_API_MAC_CARBON
		HideButtonOutlinePicture7000(theDialog);
	#endif

	getindstring(officalCancelSpelling, DLG_STRINGS, CANCEL_STRING); // "Cancel"
	getindstring(officalCloseSpelling, DLG_STRINGS, CLOSE_STRING); // "Close"

	numItems = CountDITL(theDialog); 
	if(numItems > 3) numItems = 3; // we only need to check the first 3
	for(buttonNum = 1; buttonNum<=numItems;buttonNum++){
		itemType = GetDialogItemType(theDialog,buttonNum);
		if(itemType==ctrlItem+btnCtrl)  
		{
			MyGetControlTitle(theDialog,buttonNum,theTitle);
			if(
				strcmp(officalCancelSpelling,theTitle) == 0
				||  strcmp(officalCloseSpelling,theTitle) == 0
			)
			{
				return ( buttonNum );
			}
		}
	}
	
	return 1; // return the default item
}
#endif

void CenterDialog(DialogPtr dialog, short position)
#ifdef MAC
{
	Rect	screenRect, portRect;
	Rect	dialogPortRect = GetDialogPortRect(dialog);
	short	screenWidth, screenHeight, dialogWidth, dialogHeight;
	short h, v;
	
	screenRect = GetScreenBounds();
	screenWidth = screenRect.right - screenRect.left;
	screenHeight = screenRect.bottom - screenRect.top;
	
	//dialogWidth = dialog->portRect.right - dialog->portRect.left;
	//dialogHeight = dialog->portRect.bottom - dialog->portRect.top;
	dialogWidth = dialogPortRect.right - dialogPortRect.left;
	dialogHeight = dialogPortRect.bottom - dialogPortRect.top;
	
	h = (screenWidth - dialogWidth) / 2;
	v = (screenHeight - dialogHeight) / 2 + 20;
	
	switch (position) {
		case LEFTTOP:
		case RIGHTTOP:
		case TOP:    v = GetMBarHeight() + 20; break;
		case LEFTBOT:
		case RIGHTBOT:
		case BOTTOM: v = (screenHeight - 10) - dialogHeight; break;
	}
	
	switch (position) {
		case LEFTTOP:
		case LEFTBOT:
		case LEFT:	 h = 10; break;
		case RIGHTTOP:
		case RIGHTBOT:
		case RIGHT:  h = (screenWidth - 10) - dialogWidth; break;
	}
	
	MoveWindow(GetDialogWindow(dialog), h, v, false);
}
#else
{
  RECT   rect, theScreen;
  short x, y;
  int widthP,heightP,widthD,heightD;

  GetWindowRect(dialog, (LPRECT)&rect);
  GetWindowRect(GetDesktopWindow(), (LPRECT)&theScreen);

  widthP  = theScreen.right  - theScreen.left;
  heightP = theScreen.bottom - theScreen.top;
  widthD  = rect.right  - rect.left;
  heightD = rect.bottom - rect.top;

  x = (widthP/2  + theScreen.left) - widthD/2;
  y = (heightP/2 + theScreen.top)  - heightD/2;

  MoveWindow(dialog, x, y, widthD, heightD, FALSE);
}
#endif

void CenterWindow(WindowPtr window, short position)
#ifdef MAC
{
	Rect	screenRect, portRect;
	Rect	windowPortRect = GetWindowPortRect(window);
	short	screenWidth, screenHeight, windowWidth, windowHeight;
	short h, v;
	
	screenRect = GetScreenBounds();
	screenWidth = screenRect.right - screenRect.left;
	screenHeight = screenRect.bottom - screenRect.top;
	
	//GetWindowPortBounds((WindowPtr)dialog,&portRect);
	windowWidth = windowPortRect.right - windowPortRect.left;
	windowHeight = windowPortRect.bottom - windowPortRect.top;
	//dialogWidth = dialog->portRect.right - dialog->portRect.left;
	//dialogHeight = dialog->portRect.bottom - dialog->portRect.top;
	
	h = (screenWidth - windowWidth) / 2;
	v = (screenHeight - windowHeight) / 2 + 20;
	
	switch (position) {
		case LEFTTOP:
		case RIGHTTOP:
		case TOP:    v = GetMBarHeight() + 20; break;
		case LEFTBOT:
		case RIGHTBOT:
		case BOTTOM: v = (screenHeight - 10) - windowHeight; break;
	}
	
	switch (position) {
		case LEFTTOP:
		case LEFTBOT:
		case LEFT:	 h = 10; break;
		case RIGHTTOP:
		case RIGHTBOT:
		case RIGHT:  h = (screenWidth - 10) - windowWidth; break;
	}
	
	MoveWindow(window, h, v, false);
}
#else
{
  RECT   rect, theScreen;
  short x, y;
  int widthP,heightP,widthD,heightD;

  GetWindowRect(window, (LPRECT)&rect);
  GetWindowRect(GetDesktopWindow(), (LPRECT)&theScreen);

  widthP  = theScreen.right  - theScreen.left;
  heightP = theScreen.bottom - theScreen.top;
  widthD  = rect.right  - rect.left;
  heightD = rect.bottom - rect.top;

  x = (widthP/2  + theScreen.left) - widthD/2;
  y = (heightP/2 + theScreen.top)  - heightD/2;

  MoveWindow(window, x, y, widthD, heightD, FALSE);
}
#endif

#if TARGET_API_MAC_CARBON
void GetWindowTitleFromOldDialogResource(short dialogID, char * title)
{
	DialogRef dialog = 0;
	GrafPtr savePort;
	title[0] = 0;
	GetPortGrafPtr(&savePort);// just in case
	dialog = GetNewDialog(dialogID, nil, (WindowRef)-1);
	if(dialog) {
		getwtitle(GetDialogWindow(dialog),title);
		DisposeDialog(dialog);
		dialog = 0;
	}
	SetPortGrafPort(savePort);
}
#endif


#if TARGET_API_MAC_CARBON


void MyCustomNavItemHit(DialogRef dialog, short itemHit)
{
	if(!dialog) return;
	if(itemHit <= 0) return;
	
	// standard help button behavior
	{
		char buttonTitle[256], help[20], *b;
		char windowTitle[256];
		GrafPtr oldPort;
	
		if( (GetDialogItemType(dialog, itemHit) & 0x7) == ctrlItem) 
		{
			MyGetControlTitle(dialog, itemHit, buttonTitle);
			getindstring(help, DLG_STRINGS, HELP_STRING); // "Help..."
				b = buttonTitle;
				if (b[0] == '&') b++;
				if (!strcmp(help, b)
					|| !strcmpnocase("Help", b)
					|| !strcmpnocase("Help...", b)
				) 
				{
					//getwtitle(hWnd, windowTitle);
				//	GetHelp(windowTitle, TRUE);
					//return TRUE;
				//}
			//if (IsHelpButton(buttonTitle)) {
				GetPortGrafPtr(&oldPort);
				getwtitle(GetDialogWindow(dialog), windowTitle);
				if (GetHelp(windowTitle, TRUE) == HELP_TOPICS)
					GetHelpTopics(windowTitle, TRUE);
				SetPortGrafPort(oldPort);
				return;
			}
		}
	}

	switch(itemHit) {
		default:
			SysBeep(5); break;
	}
}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/*static void SetCustomNavItemHitM58dButtons(DialogRef dialog)
{
	SetButton(dialog, 3, gImportOperationM58dButtonID == M58dREPLACE);
	SetButton(dialog, 4, gImportOperationM58dButtonID == M58dADD);
	SetButton(dialog, 5, gImportOperationM58dButtonID == M58dSUBSEARCH);
}
///////////////

void MyCustomNavInitM58d(DialogRef dialog)
{
	SetCustomNavItemHitM58dButtons(dialog);
}

void MyCustomNavItemHitM58d(DialogRef dialog, short itemHit)
{
	switch(itemHit) {
		case 3:
			gImportOperationM58dButtonID = M58dREPLACE;
			SetCustomNavItemHitM58dButtons(dialog);
			return;
		case 4:
			gImportOperationM58dButtonID = M58dADD;
			SetCustomNavItemHitM58dButtons(dialog);
			return;
		case 5:
			gImportOperationM58dButtonID = M58dSUBSEARCH;
			SetCustomNavItemHitM58dButtons(dialog);
			return;
	}
	
	MyCustomNavItemHit(dialog,itemHit); // handles the Help BUtton
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////


void MyCustomNavItemHitM58(DialogRef dialog, short itemHit)
{
	switch(itemHit) {
		case 1:
			ImportOptionsDialog(dialog);
			return;
	}
	
	MyCustomNavItemHit(dialog,itemHit); // handles the Help BUtton
}*/
/////////////////////////////////////////////////
/////////////////////////////////////////////////


MyCustomNavItemsData GetCustomNavItems(short dialogID)
{
	Handle ditl = 0;
	short customDitlResID;
	MyCustomNavItemsData myCustomItemsData;
	enum {
		kMinimumDocumentedWidth    = 400,
		kMinimumDocumentedHeight  = 40 };


	memset(&myCustomItemsData,0,sizeof(myCustomItemsData));
	
	// we check for a custom item DITL resource which 20000 more than the given ID
/*	if( dialogID > 12000) return myCustomItemsData; // dialogID is too big for our encoded custom items ID method
	customDitlResID = 20000 + dialogID;
	ditl = GetResource ('DITL',customDitlResID);
	if (!ditl) return myCustomItemsData; // no custom items if there is not a DITL 
	ReleaseResource (ditl);
	
	myCustomItemsData.ditlResID = customDitlResID;
	
	// simplest way to retro fit MARPLOT is to fill in the stucture here in this function
	switch(dialogID) {
		case M58d:	// Load Search Collection
		case 10000:	// Find Application File
		case 10001:	// Find Document
			myCustomItemsData.customHeight = 100; 
			myCustomItemsData.customWidth = kMinimumDocumentedWidth; 
			break;
		default:
			myCustomItemsData.customHeight = kMinimumDocumentedHeight; 
			myCustomItemsData.customWidth = kMinimumDocumentedWidth; 
			break;
	}
	
	switch(dialogID) {
		case M58:	// Import
			myCustomItemsData.clickProc = MyCustomNavItemHitM58;
			break;
		case M58d:	// Load Search Collection
			myCustomItemsData.clickProc = MyCustomNavItemHitM58d;
			myCustomItemsData.initProc = MyCustomNavInitM58d;
			break;
		default:
			myCustomItemsData.clickProc = MyCustomNavItemHit;
			break;
	}*/
	
	return myCustomItemsData;
}
#endif

/////////////////////////////////////////////////
void mysfpputfile(POINTPTR where, CHARPTR rType, CHARPTR defaultName,
            MyDlgHookUPP hook, MySFReplyPtr reply, short dialogID,
            ModalFilterUPP filter)
{	
#if TARGET_API_MAC_CARBON
	#pragma unused (where,rType,hook,filter)
#endif

	//PushDialogID(dialogID);
	memset(reply,0,sizeof(*reply));

	#ifdef MAC
		#if TARGET_API_MAC_CARBON
		{
			FSSpec fsSpec;
			char pathOfFileSelected[256] = "";
			char promptStr[256]; 
			Boolean gotFile;
			//MyCustomNavItemsData myCustomItemsData = GetCustomNavItems(dialogID);
			MyCustomNavItemsData myCustomItemsData;
			// TRICK: get the promptStr from the old resource dialog
			GetWindowTitleFromOldDialogResource(dialogID,promptStr);
			/////
			gotFile = AskUserForPutFileName(promptStr,defaultName,pathOfFileSelected,256,&fsSpec,&myCustomItemsData);
			if(gotFile){
				reply->good = true; 
				reply->fType = 0;
				strcpy(reply->fullPath,pathOfFileSelected);
			}
		}
		#else
		{
			/*MySFReply sfReply;
			memset(&sfReply,0,sizeof(sfReply));
			sfpputfile(where,rType,defaultName,hook,&sfReply,dialogID,filter);
			reply->good = sfReply.good;
			if(sfReply.good) {
				reply->good = sfReply.good;
				reply->fType = sfReply.fType;
				// fill in the full path name
				PathNameFromWD(sfReply.vRefNum, reply->fullPath);
				mypstrcat(reply->fullPath,(char*)sfReply.fName);
				my_p2cstr(reply->fullPath);
			}*/
		}
		#endif
	#else
		//sfpputfile(where,rType,defaultName,hook,reply,dialogID,filter);
	#endif
	//PopDialogID();
}

/////////////////////////////////////////////////

void mysfpgetfile(POINTPTR where, CHARPTR defaultName,
            short numTypes, OSType typeList[],
             MyDlgHookUPP hook, MySFReplyPtr reply, short dialogID,
            ModalFilterUPP filter)
{	// for now this is not used, may redo file i/o later
#if TARGET_API_MAC_CARBON
	#pragma unused (where,defaultName,fileFilter,hook,dialogID,filter)
#endif
	//PushDialogID(dialogID);
	memset(reply,0,sizeof(*reply));
	#ifdef MAC
	{
	#if TARGET_API_MAC_CARBON
		FSSpec fsSpec;
		char promptStr[256]; 
		char pathOfFileSelected[256] = "";
		Boolean gotFile;
		MyCustomNavItemsData myCustomItemsData;
		//MyCustomNavItemsData myCustomItemsData = GetCustomNavItems(dialogID);
		// TRICK: get the promptStr from the old resource dialog
		GetWindowTitleFromOldDialogResource(dialogID,promptStr);
		gotFile = AskUserForGetFileName(promptStr,numTypes,typeList,pathOfFileSelected,256,&fsSpec,&myCustomItemsData); 
		if(gotFile){
			reply->good = true; 
			reply->fType = 0; 
			strcpy(reply->fullPath,pathOfFileSelected);
		}
		#else
		#endif
	}
	#else
		//CommonFileDialog(TRUE, (numTypes == -1) ? 0 : typeList[3],
						// 0, defaultName, hook, reply, dialogID, filter);
		//where->h = reply->good ? sharedFileType : -1;
	#endif
	
	//PopDialogID();
}


#ifdef MAC

void PushModalGlobals(WindowPtr parent)
{
#pragma unused(parent)
}

void PopModalGlobals() {}

#else

typedef struct {
		VOIDPTR data;
		OSErr (*initProc)(DialogPtr, VOIDPTR);
		short (*clickProc)(DialogPtr, short, long, VOIDPTR);
		WindowPtr parent;
	} ModalState; 
static VOIDPTR lpData;
static OSErr (*gInitProc)(DialogPtr dialog, VOIDPTR data);
static short (*gClickProc)(DialogPtr dialog, short itemNum,
               long lParam, VOIDPTR data);
static short numDialogsInAction = 0;
static ModalState modalStates[10];

void PushModalGlobals(WindowPtr parent)
{
	if (numDialogsInAction == 10) { SysBeep(1); return; }
	
	modalStates[numDialogsInAction].data = lpData;
	modalStates[numDialogsInAction].initProc = gInitProc;
	modalStates[numDialogsInAction].clickProc = gClickProc;
	modalStates[numDialogsInAction].parent = parent;
	numDialogsInAction++;
}

void PopModalGlobals()
{
	if (numDialogsInAction == 0) { SysBeep(1); return; }
	if (numDialogsInAction == 1) { numDialogsInAction--; return; }
	
	numDialogsInAction--;
	lpData = modalStates[numDialogsInAction - 1].data;
	gInitProc = modalStates[numDialogsInAction - 1].initProc;
	gClickProc = modalStates[numDialogsInAction - 1].clickProc;
}

BOOL CALLBACK STDWinDlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	char buttonTitle[20], help[20], windowTitle[100], cancel[32], close[32],done[32], *b;
	short result, i;
	HCURSOR arrow;
	short (*clickProc)(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data);
	
	clickProc = gClickProc;
	
	// we can get WM_KILLFOCUS and WM_SETFOCUS messages intended for the parent when
	// dialogs are nested, so here we check and send these messages to the parent
	if (numDialogsInAction > 1)
		if (hWnd == modalStates[numDialogsInAction - 1].parent)
			clickProc = modalStates[numDialogsInAction - 2].clickProc;
	
	DefineEvent(&lastEvent, hWnd, message, wParam, lParam);
	
	switch (message) {
		case WM_INITDIALOG:
			SetPort(hWnd);
			CenterDialog(hWnd, 0);
			SetFocus(GetDlgItem(hWnd, 1)); // set focus to OK button by default
			if (result = (*gInitProc)(hWnd, lpData))
				EndDialog(hWnd, result);
			SubstituteParamText(hWnd);
			PostMessage(hWnd, WM_USER, 0, 0);
 			if (!GetFocus())
				SetFocus(GetDlgItem(hWnd, 3)); // set to Cancel in case OK is disabled
			if (arrow = LoadCursor(NULL, IDC_ARROW))
				SetCursor(arrow);
			return FALSE;
		
		case WM_USER:
			// call Click-Proc with item id = 0 to initialize if necessary
			result = (*clickProc)(hWnd, 0, 0, lpData);  
			if (result > 0)
				EndDialog(hWnd, result);
			return TRUE;
		
		case WM_DESTROY:
			SetPort(0);
			return TRUE;
		
		case WM_COMMAND:
			if (!IsWindowVisible(hWnd)) return FALSE;
			if (!lParam && !HIWORD(wParam)) { // from a non-existent item
				if (LOWORD(wParam) == 2) 
				{ // "cancel" item
					getindstring(cancel, DLG_STRINGS, CANCEL_STRING); // "Cancel"
					getindstring(close, DLG_STRINGS, CLOSE_STRING); // "Close"
					getindstring(done, DLG_STRINGS, DONE_STRING); // "Done"
					for (i = 1 ; i <= 4 ; i++)
					{
						if (GetDlgItem(hWnd, i) && GetDlgItemText(hWnd, i, windowTitle, 99)) {
							if (!strcmp(windowTitle, cancel) || !strcmp(windowTitle, close))
							{
								EndDialog(hWnd, i);
								return TRUE;
							}
						}
					}
						
					// if there was no cancel button ,look to see if there is a done button
					// that is not the first button, JLM 9/1/98
					for (i = 2 ; i <= 4 ; i++)
					{
						if (GetDlgItem(hWnd, i) && GetDlgItemText(hWnd, i, windowTitle, 99)) {
							if (!strcmp(windowTitle, done))
							{
								EndDialog(hWnd, i);
								return TRUE;
							}
						}
					}
						
					return TRUE;
				}
				settings.doNotPrintError = false;//JLM 4/25/00, allows dialogs to come up more than once
				result = (*clickProc)(hWnd, LOWORD(wParam), 0, lpData);
				if (result > 0)
					EndDialog(hWnd, result);
				return TRUE;
			}

			// call Click-Proc with item id = 0 to reinitialize if necessary
			result = (*clickProc)(hWnd, 0, 0, lpData);
			if (result > 0)
				EndDialog(hWnd, result);
 			
			if (HIWORD(wParam) == BN_CLICKED) 
				settings.doNotPrintError = false;//JLM 5/10/00, allows dialogs to come up more than once
				
			if (HIWORD(wParam) == BN_CLICKED) {
				getindstring(help, DLG_STRINGS, HELP_STRING); // "Help..."
				GetWindowText((WindowPtr)lParam, buttonTitle, 19);
				b = buttonTitle;
				if (b[0] == '&') b++;
				if (!strcmp(help, b)
					|| !strcmpnocase("Help", b)
					|| !strcmpnocase("Help...", b)
				) 
				{
					getwtitle(hWnd, windowTitle);
					GetHelp(windowTitle, TRUE);
					return TRUE;
				}
			}
			
			switch (HIWORD(wParam)) {
				case EN_CHANGE:
					if ((WindowPtr)lParam != GetFocus()) break;
				case BN_CLICKED:
				case EN_SETFOCUS:
				case EN_KILLFOCUS:
				case CBN_SELCHANGE:
					result = (*clickProc)(hWnd, LOWORD(wParam), 0, lpData);
					if (result > 0)
						EndDialog(hWnd, result);
			        result = (*clickProc)(hWnd, 0, 0, lpData);
		        	if (result > 0)
				       EndDialog(hWnd, result);
					return TRUE;
			}
			return FALSE;
		
		case WM_HSCROLL:
		case WM_VSCROLL:
			result = (*clickProc)(hWnd, GetDlgCtrlID((HWND)lParam), wParam, lpData);
			if (result > 0)
				EndDialog(hWnd, result);
			return TRUE;
		
		case WM_MEASUREITEM:
		case WM_DRAWITEM:
			// we assume that the parent window will handle these messages and will
			// not call DefWindowProc()
			SendMessage(hMainWnd, message, wParam, lParam);
			return TRUE;
	}
	
	return FALSE;
}
#endif

short MyModalDialog(short dialogID, WindowPtr frontWindow, VOIDPTR data,
					OSErr (*initProc)(DialogPtr dialog, VOIDPTR data),
					short (*clickProc)(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data))
{
#ifdef MAC
	return MyModalDialogF(dialogID,frontWindow,data,
					 initProc,STDFilter,clickProc);
#else
	return MyModalDialogF(dialogID,frontWindow,data,
					 initProc,nil,clickProc);
#endif

}

#ifdef MAC
void SetDefaultFrameItem(DialogPtr dialog)
{
	//draw frame rather than use a picture
	short itemType = GetDialogItemType(dialog, FRAME_ITEM);
	switch(itemType) 
	{
		case picItem:
		case picItem + itemDisable:
		case userItem:
		case userItem + itemDisable:
			Rect defaultRect = GetDialogItemBox(dialog, 1);
			InsetRect(&defaultRect,-4,-4);
			SetDialogItemBox(dialog, FRAME_ITEM, defaultRect);
			SetDialogItemType(dialog, FRAME_ITEM, userItem + itemDisable);
			SetDialogItemHandle(dialog, FRAME_ITEM, (Handle) FrameDefault);
		break;
	}
}
void ShowHideDefaultFrameItem(DialogPtr dialog,Boolean hideFlag)
{
	//draw frame rather than use a picture
	short itemType = GetDialogItemType(dialog, FRAME_ITEM);
	switch(itemType) 
	{
		case picItem:
		case picItem + itemDisable:
		case userItem:
		case userItem + itemDisable:
			ShowHideDialogItem(dialog, FRAME_ITEM,hideFlag);
		break;
	}
}
#endif

long gModalDialogCount = 0; // used to know if we are in a Dialog

Boolean WeHaveAModalDialogUp(void) // JLM 10/15/99
{
	return (gModalDialogCount > 0);
}

short MyModalDialogF(short dialogID, WindowPtr frontWindow, VOIDPTR data,
					 OSErr (*initProc)(DialogPtr dialog, VOIDPTR data),
#ifdef MAC
					 pascal_ifMac Boolean (*filterProc)(DialogPtr dialog, EventRecord *e, short *itemNum),
#else
					 DLGPROC   filterProc,
#endif
					 short (*clickProc)(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data))
#ifdef MAC
{
	GrafPtr oldPort;
	DialogPtr dialog;
	short result = -1, itemHit;
	OSErr err;
	Point where = CenteredDialogUpLeft(dialogID);
#pragma unused(frontWindow)
	
	GetPortGrafPtr(&oldPort);
	gModalDialogCount++;	
	ActivateParentDialog(false);// JLM 12/24/98
	if (!(dialog = GetNewDialog(dialogID, nil, (WindowPtr)-1))) goto done;
	
	SetPortDialogPort(dialog);
	MoveWindow(GetDialogWindow(dialog), where.h, where.v, TRUE);
	SetDefaultFrameItem(dialog); // JLM 12/24/98
	if (initProc != nil)
		if (err = (*initProc)(dialog, data)) goto done;
	SetDefaultItemBehavior(dialog);	// os x only function
	MyShowHide(GetDialogWindow(dialog), TRUE);
	
	CmdPeriod(); // ignore pending ESC

	for ( ; ; ) {
		result = (*clickProc)(dialog, 0, 0, data);
		if (result > 0) break;
		//ModalDialog((ModalFilterUPP)MakeUPP((ProcPtr)filterProc, uppModalFilterProcInfo), &itemHit);
		ModalDialog(MakeModalFilterUPP(filterProc), &itemHit);
		result = (*clickProc)(dialog, itemHit, 0, data);
		if (result > 0) break;
		// if (result == -1) do default processing (no such thing on Mac)
	}
	
done:
	SetPortGrafPort(oldPort);
	DisposeDialog(dialog);
	ActivateParentDialog(true);// JLM 12/24/98
	gModalDialogCount--;	
	return result;
}
#else
{	// IBM code
	short result;
	GrafPtr oldPort;
	if(!filterProc) filterProc = (DLGPROC)STDWinDlgProc;
	
	GetPortGrafPtr(&oldPort);
	
	lpData = data;
	gInitProc = initProc;
	gClickProc = clickProc;
	
	CmdPeriod(); // ignore pending ESC
	
	gModalDialogCount++;	
	PushModalGlobals(frontWindow);
	result = DialogBox(hInst, MAKEINTRESOURCE(dialogID), frontWindow, filterProc);
	PopModalGlobals();
	gModalDialogCount--;	
	
	SetPortGrafPort(oldPort);
	
	return result; 
}
#endif

short MULTICHOICEALERT(short dialogID, CHARPTR text, Boolean bParent)
#ifdef MAC
{
	GrafPtr		savePort;
	DialogPtr	theDialog;
	short		itemHit = 0;
	
	if (text && text[0]) paramtext(text, "", "", "");
	if (!(theDialog = GetNewDialog(dialogID, (Ptr)nil, (WindowPtr)-1)))
		{ SysBeep(1); return TRUE; }
	GetPortGrafPtr(&savePort);
	gModalDialogCount++;	
	if (bParent) ActivateParentDialog(false);
	SetPortDialogPort(theDialog);
	CenterDialog(theDialog, CENTER);
	SetDefaultFrameItem(theDialog);//JLM 12/24/98
	SetDefaultItemBehavior(theDialog);
	MyShowHide(GetDialogWindow(theDialog), TRUE);
	InitCursor();
	// SysBeep(1);
	while (!itemHit)
		//ModalDialog((ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo), &itemHit);
		//ModalDialog((ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, NULL), &itemHit);
		ModalDialog(MakeModalFilterUPP(STDFilter), &itemHit);
	DisposeDialog(theDialog);
	SetPortGrafPort(savePort);
	if (bParent) ActivateParentDialog(true);
	gModalDialogCount--;	
	return itemHit;
}
#else
{
	short itemHit;
	
	gModalDialogCount++;	
	if (text && text[0]) paramtext(text, "", "", "");
	itemHit = (short)DialogBox(hInst, MAKEINTRESOURCE(dialogID), FrontWindow(), (DLGPROC)ChoiceAlertProc); 
	gModalDialogCount--;	
	
	return itemHit;
}
#endif

short MULTICHOICEALERT2(short dialogID, CHARPTR text, CHARPTR text2, Boolean bParent)
#ifdef MAC
{
	GrafPtr	savePort;
	DialogPtr	theDialog;
	short		itemHit = 0;
	
	/*if (text && text[0] && text2 && text2[0])*/ paramtext(text, text2, "", "");
	if (!(theDialog = GetNewDialog(dialogID, (Ptr)nil, (WindowPtr)-1)))
		{ SysBeep(1); return TRUE; }
	GetPortGrafPtr(&savePort);
	gModalDialogCount++;	
	if (bParent) ActivateParentDialog(false);
	SetPortDialogPort(theDialog);
	CenterDialog(theDialog, CENTER);
	SetDefaultFrameItem(theDialog);//JLM 12/24/98
	SetDefaultItemBehavior(theDialog);
	MyShowHide(GetDialogWindow(theDialog), TRUE);
	InitCursor();
	// SysBeep(1);
	while (!itemHit)
		//ModalDialog((ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo), &itemHit);
		//ModalDialog((ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, NULL), &itemHit);
		ModalDialog(MakeModalFilterUPP(STDFilter), &itemHit);
	DisposeDialog(theDialog);
	SetPortGrafPort(savePort);
	if (bParent) ActivateParentDialog(true);
	gModalDialogCount--;	
	return itemHit;
}
#else
{
	short itemHit;
	
	gModalDialogCount++;	
	if (text && text[0] && text2 && text2[0]) paramtext(text, text2, "", "");
	itemHit = (short)DialogBox(hInst, MAKEINTRESOURCE(dialogID), FrontWindow(), (DLGPROC)ChoiceAlertProc); 
	gModalDialogCount--;	
	
	return itemHit;
}
#endif

Boolean CHOICEALERT(short dialogID, CHARPTR text, Boolean bParent)
{
	short itemHit = MULTICHOICEALERT(dialogID, text, bParent);	
	return (itemHit == 1);
}

#ifdef MAC
void center_high(DialogPtr theDialog)
{
	Rect	screenRect;
	short	screenWidth;
	short	screenHeight;
	short	dialogWidth;
	short	dialogHeight;
	Point	corner;
	Rect 	portRect;
	
	screenRect = GetScreenBounds();
	screenWidth = screenRect.right -screenRect.left;
	screenHeight = screenRect.bottom -screenRect.top;
	
	//GetWindowPortBounds(GetDialogWindow(theDialog),&portRect);
	portRect  = GetDialogPortRect(theDialog);
	dialogWidth = portRect.right - portRect.left;
	dialogHeight = portRect.bottom - portRect.top;
	//dialogWidth = theDialog->portRect.right -theDialog->portRect.left;
	//dialogHeight = theDialog->portRect.bottom -theDialog->portRect.top;
	
	corner.h = (screenWidth - dialogWidth)/2;
	corner.v = (screenHeight - dialogHeight)/3;	/* above center */
	MoveWindow(GetDialogWindow(theDialog),corner.h,corner.v,false);
}

long message_height(Ptr str, WindowPtr window, Rect *dstRectPtr)
{
	TEHandle  	teHdl;
	long		messageHeight;
	Rect		portRect = GetWindowPortRect(window);
	
	teHdl = TENew(dstRectPtr,&portRect);
	if(teHdl != nil)
	{
		TESetText(str,strlen(str),teHdl);
		messageHeight = (**teHdl).nLines * (**teHdl).lineHeight;
		TEDispose(teHdl);
	}
	else
	{
		messageHeight = 0;
	}
	return(messageHeight);
}

pascal_ifMac void	draw_message(DialogPtr theDialog, short itemNum)
{
	/****  This routine draws the message into the itemBox  ****/
	/****  It is used as the procPtr for a user item that is over the ok button **/
	/****  The message is stored in the ref con  of the dialog*****/
	short			theType;
	Handle			itemHandle;
	Rect			itemBox;
	Ptr				str;

	GetDialogItem(theDialog,itemNum,&theType,&itemHandle,&itemBox);
	str = (Ptr)GetWRefCon(GetDialogWindow(theDialog));
	if (str != nil) TETextBox(str,strlen(str),&itemBox,normal);
}

long REQUEST(char *message, char *defaultAnswer, char *answer)
{	
	GrafPtr			savePort;
	DialogPtr 		theDialog;
	Boolean			moreToDo;
	short			itemHit;
	short			theType;
	Handle			itemHandle;
	Rect			itemBox, portRect;
	long			theMessageHeight;
	
	const	short	dialogResNum = 11702;	/// local constants
	const	short	okFrameItemNum = 3;	
	const	short	messageUserItem = 4;	
	const	short	editTextItemNum = 5;
	const	short	spaceForOK = 50;
	const	short	margin = 20;
	
	GetPortGrafPtr(&savePort);
	
	gModalDialogCount++;	
	theDialog = GetNewDialog(dialogResNum, nil, (WindowPtr)(-1));  
	if( theDialog  == nil || _MemError()){
		/** Problem with resource or memory error. **/
		/** In case our user forgets to check our error code, **/
		/** let's write something into the answer string.**/
		if(answer!= nil){
			strcpy(answer,"Resource or memory error occurred." );
		}
		itemHit = -1;	/** we return the value of -1 **/
	}
	else{
		SetPortDialogPort(theDialog);
		TextFont(0);//system font
		TextSize(0);// 12 point default
		
		//store the string ptr in the ref con for the user item
		SetWRefCon(GetDialogWindow(theDialog),(long)message);
		
		// determine the height of the message
		GetDialogItem(theDialog,messageUserItem,&theType,&itemHandle,&itemBox);
		theMessageHeight = message_height(message,GetDialogWindow(theDialog),&itemBox);
		
		portRect = GetDialogPortRect(theDialog);
		//GetWindowPortBounds(GetDialogWindow(theDialog),&portRect);
		SizeWindow(GetDialogWindow(theDialog), portRect.right - portRect.left,
						theMessageHeight + portRect.bottom - portRect.top,
						true);
		
		/***** setup the user item which draws the message  ***/
		GetDialogItem(theDialog,messageUserItem,&theType,&itemHandle,&itemBox);
		itemBox.bottom = itemBox.top + theMessageHeight;
		SetDialogItem(theDialog,messageUserItem,userItem + itemDisable,nil,&itemBox);
		SetDialogItemHandle(theDialog, messageUserItem, (Handle)draw_message);
		
		/***** Center the Window, above center ****/
		center_high(theDialog);
						
		GetDialogItem(theDialog,ok,&theType,&itemHandle,&itemBox);
		// move ok button to position for new window size
		OffsetRect(&itemBox,0,theMessageHeight);
		MoveControl((ControlHandle)itemHandle,itemBox.left,itemBox.top);
		SetDialogItem(theDialog,ok,theType,itemHandle,&itemBox);
		
		InsetRect(&itemBox,-4,-4);
		SetDialogItem(theDialog,okFrameItemNum,userItem + itemDisable,nil,&itemBox);
		SetDialogItemHandle(theDialog,okFrameItemNum,(Handle)FrameDefault);
		
		GetDialogItem(theDialog,cancel,&theType,&itemHandle,&itemBox);
		// move cancel button to position for new window size
		OffsetRect(&itemBox,0,theMessageHeight);
		MoveControl((ControlHandle)itemHandle,itemBox.left,itemBox.top);
		SetDialogItem(theDialog,cancel,theType,itemHandle,&itemBox);
		
		/**** move cancel button to position for new window size ****/
		GetDialogItem(theDialog,editTextItemNum,&theType,&itemHandle,&itemBox);
		OffsetRect(&itemBox,0,theMessageHeight);
		SetDialogItem(theDialog,editTextItemNum,theType,itemHandle,&itemBox);
		
		/****  setup and show the editText item  ****/
		/** Initialize editText to the default response **/
		if (defaultAnswer != nil)
			mysetitext(theDialog, editTextItemNum, defaultAnswer);
		/** Select the default so that the text will appear hilited **/
		MySelectDialogItemText(theDialog,editTextItemNum,0,32767);

		MyShowWindow(GetDialogWindow(theDialog));
	
		InitCursor();
		moreToDo = true;
		while (moreToDo) {
			//ModalDialog((ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo), &itemHit);
			//ModalDialog((ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, NULL), &itemHit);
			ModalDialog(MakeModalFilterUPP(STDFilter), &itemHit);
			switch (itemHit) {
				case ok:
				case cancel:
					moreToDo = false;
					break;
			}
		}
		/**** read the answer ****/
		if (answer != nil)
			mygetitext(theDialog, editTextItemNum, answer, 255);
		DisposeDialog(theDialog);
		SetPortGrafPort(savePort);
	}
	gModalDialogCount--;	

	
	return itemHit;
}
#else

char * gRequestMessage;
char * gRequestDefaultAnswer;
char * gRequestAnswer;

BOOL CALLBACK  RequestDialogProc (HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	HWND  hwndMsg;
	HWND hwndEdit;
	short end = -1;
	
	switch (message)
	{
		case WM_INITDIALOG:
		
			CenterDialog(hDlg, 0);
			hwndMsg = GetDlgItem (hDlg, 4);
			SetWindowText (hwndMsg, gRequestMessage);
			hwndEdit = GetDlgItem (hDlg, 5);
			if (gRequestDefaultAnswer != NULL)
			{
				SetWindowText (hwndEdit, gRequestDefaultAnswer);
				end = strlen(gRequestDefaultAnswer);
			}
			SetFocus (hwndEdit);
			SendMessage (hwndEdit, EM_SETSEL, 0, MAKELONG (0, end));
			ShowWindow (hDlg, SW_SHOW);
			return FALSE;
		
		case WM_COMMAND:
			switch (wParam)
			{
				case 1:
					hwndEdit = GetDlgItem (hDlg, 5);
					SendMessage (hwndEdit, WM_GETTEXT, 256, 
					(LONG)((LPSTR)gRequestAnswer));
					EndDialog (hDlg, TRUE);
					return TRUE;
				
				case 2:
					EndDialog (hDlg, FALSE);
					return TRUE;
			}
			return TRUE;   
	
	}
	return FALSE;
}



long REQUEST ( char * message,  char * defaultAnswer, char * answer)
{
   //FARPROC  lpfnRequestDialogProc;
   long  returnVal;
    
   //Assign global pointer values so we can use them in the dialog procedure
   gRequestMessage = message;
   gRequestDefaultAnswer = defaultAnswer;
   gRequestAnswer = answer;
   
   //lpfnRequestDialogProc = MakeProcInstance (RequestDialogProc, hInst);

   //returnVal = (long)DialogBox (hInst, "Request", mapWindow, lpfnRequestDialogProc);
   returnVal = (long)DialogBox (hInst, "Request", mapWindow, (DLGPROC)RequestDialogProc);

  // FreeProcInstance (lpfnRequestDialogProc);
   if (returnVal == TRUE)
      returnVal = 1;   //OK button was hit
   else
      returnVal = 2;   //Cancel button was hit
   return (returnVal);
}
#endif

#ifdef MAC
	#define kMaxNumButtons 10
	#define kMaxNumButtonInfo 5
	typedef struct {
		DialogPtr dialog;
		Boolean	origState[kMaxNumButtons];
	} ButtonInfo;
	short gButtonInfoIndex = 0;
	ButtonInfo  gButtonInfo[kMaxNumButtonInfo];
#endif

void ActivateParentDialog(Boolean bActivate)
#ifdef MAC
{
	WindowPtr	theWindow = FrontWindow();
	GrafPtr		oldPort;
	
	if (!theWindow) return;
	
	#if TARGET_API_MAC_CARBON
	{	// JLM 12/03/06, there is a bug when using the Help button on dialogs...
		// this routine deactivated all the items then reactivated all the items...
		// so that items which were originally disabled become enabled after the help dialog
		// Rather than fix the code in all those places to re-disable the appropriate items,
		// we can rely on OS X doing it automatically for us.
		// The only catch is that we still have to use this routine for our main window...
		DialogRef dialog = GetDialogFromWindow(theWindow);
		if(dialog)
			return; // it is a dialog and OS X can handle deactivating and reactiviating the controls
	}
	#endif
	
	GetPortGrafPtr(&oldPort);
	SetPortWindowPort(theWindow);
	
	#if TARGET_API_MAC_CARBON
	{
		ControlRef rootControl = 0;
		ControlHandle 		controlhdl;
		OSStatus err = GetRootControl(theWindow,&rootControl);
		if(!err && rootControl) {
			long i;
			UInt16 numControls;
			err = CountSubControls(rootControl, &numControls);
			for(i = 1; i <= numControls  && !err; i++) {
				err = GetIndexedSubControl(rootControl,i,&controlhdl);
				if(!err) {
					if(bActivate) ActivateControl(controlhdl);
					else  DeactivateControl(controlhdl);
				}
			}
		}
	}
	#else // MACB4CARBON
	{
		Handle		items, itemHandle;
		short		i, itemType;
		Rect		itemBox;
		short 		buttonIndex = 0;
		ControlRef	*controlList = 0;
		DialogRef theDialog = GetDialogFromWindow(theWindow);
	
		if(theWindow == mapWindow) return; 
		// JLM,12/29/98 if the frontWindow is a window not a dialog
		// we will crash.  The only window we have is the mapWindow
		
		if (!((WindowPeek)theDialog)->controlList) return;
		
		items = ((DialogPeek)theDialog)->items;
		i = *(short *)(*items) + 1;
		
		if(bActivate) {//  reactivating and restoring, check we have the right window
			if (gButtonInfoIndex == 0 || gButtonInfoIndex > kMaxNumButtonInfo || theDialog != gButtonInfo[gButtonInfoIndex-1].dialog)
				printNote("ActivateParentDialog error");
		}
		
		for ( ; i > 0 ; i-- ) {
			GetDialogItem(theDialog, i, &itemType, &itemHandle, &itemBox);
			if (itemType == (ctrlItem + btnCtrl)) {
				short newActivateVal = bActivate ? NOTDIMMED : DIMMED;
				Boolean wasEnabled;
				if(!bActivate) 
				{
					// save value
					wasEnabled = ControlEnabled(theDialog,i);
					if(buttonIndex < kMaxNumButtons)
						gButtonInfo[gButtonInfoIndex].origState[buttonIndex] = wasEnabled;
					else 
						printError("kMaxNumButtons is too small in ActivateParentDialog");
				}
				else 
				{
					// restore previous value
					wasEnabled = gButtonInfo[gButtonInfoIndex-1].origState[buttonIndex];
					newActivateVal = wasEnabled ? NOTDIMMED : DIMMED;
				}
				HiliteControl((ControlHandle)itemHandle, newActivateVal);
				buttonIndex++;
			}
		}
		
		ShowHideDefaultFrameItem(theDialog, bActivate); // JLM 12/24/98
		
		if(bActivate) 
		{ 	// done reactivating and restoring 
			gButtonInfo[gButtonInfoIndex-1].dialog = 0;
			gButtonInfoIndex--;
			if(gButtonInfoIndex < 0) 
			{
				printNote("ActivateParentDialog error");
				gButtonInfoIndex = 0;
			}
		}
		else 
		{ // deactivating and saving values
			gButtonInfo[gButtonInfoIndex].dialog = theDialog;
			gButtonInfoIndex++;
		}
	}
	#endif
	SetPortGrafPort(oldPort);
}
#else
{
	// in Windows it is not standard to deactivate the parent dialog
}
#endif

#ifdef MAC
void EraseUserItem(DialogPtr theDialog, short item)
{
	Rect r = GetDialogItemBox(theDialog, item);
	
	EraseRect(&r);
}
#endif

#ifdef MAC
void DrawUserItem(DialogPtr theDialog, short item)
{
	pascal_ifMac void (* itemFunction)(DialogPtr, short);
	
	itemFunction = (pascal_ifMac void (*)(DialogPtr, short))GetDialogItemHandle(theDialog, item);
	(* itemFunction)(theDialog, item);
}
#endif

#ifdef MAC
void CenterDialogUpLeft(DialogPtr theDialog)
{  // this will get you the same place as using CenteredDialogUpLeft() and MoveWindow()  
	Point corner ; 
	Rect	dialogRect = GetDialogPortRect(theDialog), screenRect;	
	//dialogRect = theDialog->portRect;
	dialogRect.top += 20;			// add for title bar
	screenRect = GetScreenBounds();
	corner.h = (RectWidth(screenRect) - RectWidth(dialogRect)) / 2;
	corner.v = (double)(RectHeight(screenRect) - RectHeight(dialogRect)) / 2.5;
	
	if (corner.v < (GetMBarHeight() + 20)) corner.v = GetMBarHeight() + 20;
	
	MoveWindow(GetDialogWindow(theDialog), corner.h, corner.v, TRUE);
}
#endif


Point CenteredDialogUpLeft(int dialogID)
#ifdef MAC
{
	Handle	theTemplate;
	Rect	dialogRect, screenRect;
	Point	corner;
	
	theTemplate = GetResource('DLOG', dialogID);
	if (theTemplate == nil)
		{ SysBeep(1); corner.h = corner.v = 0; return corner; }
	dialogRect = **(Rect **)theTemplate;
	dialogRect.top += 20;			// add for title bar
	ReleaseResource(theTemplate);

	screenRect = GetScreenBounds();
	corner.h = (RectWidth(screenRect) - RectWidth(dialogRect)) / 2;
	corner.v = (double)(RectHeight(screenRect) - RectHeight(dialogRect)) / 2.5;
	
	if (corner.v < (GetMBarHeight() + 20)) corner.v = GetMBarHeight() + 20;
	
	return corner;
}
#else
{
	Point corner = { 0, 0 };
	
	return corner;
}
#endif

void FillEmptyItems(DialogPtr theDialog, SHORTPTR items, CHARPTR newValue)
#ifdef MAC
{
	char	oldValue[5];
	short	i = 0;
	
	while (items[i] != 0) {
		mygetitext(theDialog, items[i], oldValue, 4);
		if (!oldValue[0])
			mysetitext(theDialog, items[i], newValue);
		i++;
	}
}
#else
{
	// (MIKE) fill any empty text-edit items with newValue
}
#endif

Boolean ItemHidden(DialogPtr theDialog, short item)
#ifdef MAC
{
	Rect r = GetDialogItemBox(theDialog, item);
	
	return r.left > 1000;
}
#else
{
	return !IsWindowVisible(GetDlgItem(theDialog, item));
}
#endif

void MyEnableControl(DialogPtr dialog, short item, Boolean enable)
#ifdef MAC
{
#if TARGET_API_MAC_CARBON
	ControlHandle c = (ControlHandle)GetDialogItemHandle(dialog, item);
	GrafPtr savePort;
	Boolean currentlyEnabled = false;
	
	if(!c) return;
	
	currentlyEnabled = ControlEnabled(dialog,item);
	
	if (currentlyEnabled != enable) {
		GetPortGrafPtr(&savePort);
		SetPortDialogPort(dialog);
		
		if(enable) ActivateControl(c);
		else  DeactivateControl(c);
		
		SetPortGrafPort(savePort);
	}
#else	// probably just MPW
	short newHilite = enable ? 0 : 255;
	ControlHandle c = (ControlHandle)GetDialogItemHandle(dialog, item);
	GrafPtr savePort;
	
	if ((**c).contrlHilite != newHilite) {
		GetPortGrafPtr(&savePort);
		SetPortDialogPort(dialog);
		HiliteControl(c, newHilite);
		SetPortGrafPort(savePort);
	}
#endif
}
#else
{
	HWND hCtl = GetDlgItem(dialog, item);
	Boolean wasDisabled = !IsWindowEnabled(hCtl);
	
	EnableWindow(hCtl, enable);
	// fix for a Windows default hilite bug
	// previous bad sequence:
	//		1) button 1 is the default but is disabled because no text is in the box
	//		2) user clicks button 2 for whatever reason, leaving it the default button
	//		3) user clicks in text box
	//		4) button 1 is now the "default" again but isn't boxed since it is disabled
	//		5) user types in text box, which enables button 1
	//		6) but due to the bug, button 1 does not get boxed even though it is enabled
	//			and is in fact the default
	// so the fix is just to reset the button to cause it to redraw boxed
	if (item == 1 && wasDisabled && enable)
		SendMessage(hCtl, BM_SETSTYLE, (WPARAM)BS_DEFPUSHBUTTON, MAKELPARAM(TRUE, 0));
}
#endif

Boolean ControlEnabled(DialogPtr dialog, short item)
#ifdef MAC
{
	ControlRef theControl = (ControlHandle)GetDialogItemHandle(dialog, item);
	
	if(!theControl) return false;
	
	#if TARGET_API_MAC_CARBON
		//return IsControlEnabled(theControl); // not available in carbon 1.x
		return  IsControlActive(theControl);
	#else
		return (**theControl).contrlHilite == 0;///255 is disabled setting
	#endif
}
#else
{
	HWND hCtl = GetDlgItem(dialog, item);
	
	return IsWindowEnabled(hCtl);
}	
#endif

ControlHandle GetControlItemHandle(DialogPtr dialog, short item)
#ifdef MAC
{
	return (ControlHandle)GetDialogItemHandle(dialog, item);
}
#else
{
	return GetDlgItem(dialog, item);
}
#endif

void MyGetControlTitle(DialogPtr theDialog, short item, CHARPTR title)
#ifdef MAC
{
	GetControlTitle((ControlHandle)GetDialogItemHandle(theDialog, item), (StringPtr)title);
	my_p2cstr((StringPtr)title);
}
#else
{
	getdialogitemtext((ControlHandle)GetDialogItemHandle(theDialog, item), title);
}
#endif

void MySetControlTitle(DialogPtr theDialog, short item, CHARPTR title)
#ifdef MAC
{
	char localStr[256];
	strcpy(localStr,title);
	//my_c2pstr(title);
	my_c2pstr(localStr);
	//SetControlTitle((ControlHandle)GetDialogItemHandle(theDialog, item), (StringPtr)title);
	SetControlTitle((ControlHandle)GetDialogItemHandle(theDialog, item), (StringPtr)localStr);
}
#else
{
	setdialogitemtext((ControlHandle)GetDialogItemHandle(theDialog, item), title);
}
#endif

#ifdef MAC
void DoFrameDefault(DialogPtr theDialog, short frameItem)
{
	Rect r;
	
	r = GetDialogItemBox(theDialog, frameItem);
	PenSize(3, 3);
	FrameRoundRect(&r, 16, 16);
	PenNormal();
}
#endif

pascal_ifMac void FrameDefault(DialogPtr theDialog, short frameItem)
#ifdef MAC
{
	if (GetDialogWindow(theDialog) == FrontWindow()) {
		PenNormal();
		DoFrameDefault(theDialog, frameItem);
	}
}
#else
{
}
#endif

pascal_ifMac void FrameBlack(DialogPtr theDialog, short itemNum)
#ifdef MAC
{
	Rect itemBox;
	
	itemBox = GetDialogItemBox(theDialog, itemNum);
	FrameRect(&itemBox);
}
#else
{
}
#endif

void DoFrameEmbossed(Rect itemBox)
{
	RGBForeColor(&colors[DARKGRAY]);
	MyFrameRect(&itemBox);
		
	RGBForeColor(&colors[WHITE]);
	MyMoveTo(itemBox.left+1,itemBox.top+1);
	MyLineTo(itemBox.right-2,itemBox.top+1);
	
	MyMoveTo(itemBox.right,itemBox.top+1);
	MyLineTo(itemBox.right,itemBox.bottom);
	MyLineTo(itemBox.left,itemBox.bottom);
	
	MyMoveTo(itemBox.left+1,itemBox.bottom-2);
	MyLineTo(itemBox.left+1,itemBox.top+1);
	RGBForeColor(&colors[BLACK]);
}

pascal_ifMac void FrameEmbossed(DialogPtr theDialog, short itemNum)
#ifdef MAC
{
	Rect itemBox;
	
	itemBox = GetDialogItemBox(theDialog, itemNum);
	DoFrameEmbossed(itemBox);

}
#else
{
}
#endif
pascal_ifMac void FrameGray(DialogPtr theDialog, short itemNum)
#ifdef MAC
{
	Rect itemBox;
	//Pattern gray, black;
	//MyGetQDGlobalsGray(&gray);
	//GetQDGlobalsBlack(&black);
	
	//PenPat((ConstPatternParam)&gray);
	//PenPat((ConstPatternParam)&GRAY_BRUSH);
	PenPatQDGlobalsGray();
	itemBox = GetDialogItemBox(theDialog, itemNum);
	FrameRect(&itemBox);
	//PenPat((ConstPatternParam)&BLACK_BRUSH);
	//PenPat((ConstPatternParam)&black);
	PenPatQDGlobalsBlack();
}
#else
{
}
#endif

#ifdef MAC
pascal_ifMac void HelpVSAction(ControlHandle c, short partCode)
{
    short delta, lineHt;
	Rect r;
	TEHandle textHandle;
	
	if (!partCode) return;
	
	textHandle = (TEHandle) GetControlReference(c);
	r = (**textHandle).viewRect;
	lineHt = (**textHandle).lineHeight;
	
	switch (partCode) {
		case kControlUpButtonPart:	 delta = -1; break;
		case kControlDownButtonPart: delta = 1; break;
		case kControlPageUpPart:	 delta = 1 - RectHeight(r) / lineHt; break;
		case kControlPageDownPart:	 delta = RectHeight(r) / lineHt - 1; break;
	}
	
	if ((GetControlValue(c) + delta) > GetControlMaximum(c)) delta = GetControlMaximum(c) - GetControlValue(c);
	if ((GetControlValue(c) + delta) < GetControlMinimum(c)) delta = GetControlMinimum(c) - GetControlValue(c);
	
	SetControlValue(c, GetControlValue(c) + delta);
	TEScroll(0, -delta * lineHt, textHandle);
}

static Rect sharedViewRect;
static TEHandle sharedTextHandle;

pascal_ifMac void DrawHelpText(DialogPtr theDialog, short item)
{
	Rect r = GetDialogItemBox(theDialog, item);
	
	InsetRect(&r, -1, -1);
	FrameRect(&r);
	TEUpdate(&sharedViewRect, sharedTextHandle);
}

short GetHelp2(CHARPTR title)
{
	GrafPtr oldPort;
	DialogPtr diawind;
	char heading[100];
	short itemHit, part, lineHt, oldvalue, height, result;
	Handle resHandle;
	Rect r, destRect, scrollRect;
	Point mouseLoc;
	Boolean done = FALSE;
	ControlHandle vScroll, c;
	Ptr resPtr;

	GetPortGrafPtr(&oldPort);

	if (!(diawind = GetNewDialog(HELP_DIALOG, (Ptr)nil, (WindowPtr)-1))) return -1;
	SetPortDialogPort(diawind);

	SetDialogItemHandle(diawind, HELP_OUTLINE, (Handle)FrameDefault);
	SetDialogItemHandle(diawind, HELP_USERITEM, (Handle)DrawHelpText);

	CenterDialog(diawind, CENTER);
	SetDefaultItemBehavior(diawind);
	MyShowHide(GetDialogWindow(diawind), TRUE);

	r = GetDialogItemBox(diawind, HELP_USERITEM);

	MySetRect(&scrollRect, r.right - 15, r.top - 1, r.right + 1, r.bottom + 1);
	destRect = r;
	destRect.left += 3;
	destRect.right -= 18;
	sharedViewRect = r;
	sharedViewRect.right -= 16;

	TextFont(kFontIDGeneva);
	TextSize(12); // 12

	sharedTextHandle = TENew(&destRect, &sharedViewRect);
	if (!sharedTextHandle) 
	{ 	SetPortGrafPort(oldPort); DisposeDialog(diawind); return -1; }
 	
 	getResource:
 	
 	resHandle = (Handle)getnamedresource('TEXT', title);
	if (!resHandle) {
		char msg[256];
		SetPortGrafPort(oldPort);
		DisposeDialog(diawind);
		TEDispose(sharedTextHandle);
		sprintf(msg,"Help topic \"%s\" not found.",title);
		printNote(msg);
		return -1;
	}
	if ((**resHandle) == '>') {
		strcpyToDelimeter(title, (*resHandle) + 1, '<');
		goto getResource;
	}
	
	_HLock(resHandle);
	resPtr = *resHandle;
	
	TESetText(resPtr, _GetHandleSize(resHandle), sharedTextHandle);
	lineHt = (*sharedTextHandle)->lineHeight;
	
	vScroll = (ControlHandle)NewControl(GetDialogWindow(diawind), &scrollRect, "\pVS", TRUE, 0,
										0, (*sharedTextHandle)->nLines - 1, scrollBarProc, 0);
	//vScroll = (ControlHandle)newcontrol((WindowPtr)diawind, &scrollRect, "VS", TRUE, 0,
										//0, (*sharedTextHandle)->nLines - 1, scrollBarProc, 0);
	
	height = sharedViewRect.bottom - sharedViewRect.top;
	HiliteControl(vScroll, ((**sharedTextHandle).nLines < (height / lineHt)) ? 255 : 0);
	
	SetControlReference(vScroll, (long)sharedTextHandle);
	SetWRefCon(GetDialogWindow(diawind), (long)sharedTextHandle);
	
	while (!done) {
		//ModalDialog((ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo), &itemHit);
		ModalDialog(MakeModalFilterUPP(STDFilter), &itemHit);
		switch (itemHit) {
			case HELP_CANCEL:
				done = TRUE;
				result = HELP_CANCEL;
				break;
			
			case HELP_COPY:
				SetWatchCursor();
				MyZeroScrap();
				_HLock((**sharedTextHandle).hText);
				MyPutScrap((**sharedTextHandle).teLength, 'TEXT',
						 *(**sharedTextHandle).hText);
				_HUnlock((**sharedTextHandle).hText);
				break;
			
			case HELP_PRINT:
				getindstring(heading, DLG_STRINGS, HEADING_STRING); // "<application> Help"
				PrintText(heading, (**sharedTextHandle).hText, HELP_ICON_ID);
				SetPortDialogPort(diawind);
				break;
			
			case HELP_USERITEM:
				GetMouse(&mouseLoc);
				//part = FindControl (mouseLoc, GetDialogWindow(diawind), &c);
				/////// we need to use TestControl for TARGET_API_MAC_CARBON
				// but it also works for MACB4CARBON
				part = TestControl(vScroll,mouseLoc);
				c = vScroll;
				///
				if (part == kControlIndicatorPart) {
					oldvalue = GetControlValue(c);
					if (TrackControl (c, mouseLoc, nil) == part)
						TEScroll(0, (oldvalue - GetControlValue(c)) * lineHt, sharedTextHandle);
				}
				else if (part)
					//part = TrackControl (c, mouseLoc, (ControlActionUPP)MakeUPP((ProcPtr)HelpVSAction, uppControlActionProcInfo));
					part = TrackControl(c, mouseLoc, MakeControlActionUPP(HelpVSAction));
				break;
			
			case HELP_TOPICS:
				done = TRUE;
				result = HELP_TOPICS;
				break;
		}
	}
	_HUnlock(resHandle);
	SetPortGrafPort(oldPort);
	DisposeDialog(diawind);
	TEDispose(sharedTextHandle);
	
	return result;
}
#endif

short GetHelp(CHARPTR title, Boolean bParent)
#ifdef MAC
{
	short result;
	
	if (bParent) ActivateParentDialog(FALSE);
	result = GetHelp2(title);
	if (bParent) ActivateParentDialog(TRUE);
	
	return result;
}
#else
{
	char path[256], *p;
	GetDirectoryFromID(TATdirID, path);

	if (p = strrchr(path, DIRDELIMITER))
		p[1] = 0;
	else
		p = path;
	strcat(path, HELPFILENAME);
	//strcat(title,".html");

	//if (!HtmlHelp(hMainWnd, path, HH_DISPLAY_TOPIC, (DWORD)(LPSTR)title)) {
	if (!WinHelp(hMainWnd, path, HELP_KEY, (DWORD)(LPSTR)title)) {
		MessageBox(hMainWnd, GS1(136, 0), 0, MB_ICONHAND); // Unable to activate help.
		return FALSE;
	}
	
	return TRUE;
}
#endif


#ifdef MAC

static Handle sharedTopics;
static long sharedPrevItem;

void DrawHTItem(DialogPtr dialog, Rect *r, long item)
{
	char title[100];
#pragma unused(dialog)
#pragma unused(r)
	
	TextFont(kFontIDGeneva); TextSize(9);
	NthLineInTextNonOptimized(*sharedTopics, item, title, 100);
	drawstring(title);
	TextFont(0); TextSize(12);
}

Boolean HTClick(DialogPtr dialog, VLISTPTR L, short dialogItem, long *listItem,
				Boolean doubleClick)
{
#pragma unused(dialog)
#pragma unused(L)
#pragma unused(listItem)
	
	if (doubleClick) dialogItem = TOPICS_SELECT;
	
	switch (dialogItem) {
		case TOPICS_SELECT: return TRUE;
		case TOPICS_CANCEL: return TRUE;
	}
	
	return FALSE;
}

void HTInit(DialogPtr dialog, VLISTPTR L)
{
#pragma unused(dialog)
	
	VLSetSelect(sharedPrevItem, L);
	VLAutoScroll(L);
}

short GetHelpTopics(char *prevTopic, Boolean bParent)
{
	long i, n, numLines;
	short item, dialogItem;
	char title[256], *t;
	VListSettings vSettings;
	
	if (!(sharedTopics = GetResource('TEXT', TOPICS_TEXT))) return -1;
	
	_HLock(sharedTopics); // JLM, so it can't be purged
	
	sharedPrevItem = 0;
	
	{ /////JLM fix bug when resource is not null terminated
		long len = _GetHandleSize(sharedTopics);
		if((*sharedTopics)[len-1] != 0)
		{ // the resource is not null terminated
			_SetHandleSize(sharedTopics,len+1);
			if(_GetHandleSize(sharedTopics) == len+1)
				(*sharedTopics)[len] = 0;
		}
	}
	
	numLines = NumLinesInText(*sharedTopics);

	if (prevTopic) {
		n = numLines;
		for (i = 0 ; i < n ; i++) {
			t = NthLineInTextOptimized(*sharedTopics, i, title, 256);
			while (t[0] == ' ' || t[0] == ',') t++;
			if (!strcmp(t, prevTopic)) {
				sharedPrevItem = i;
				break;
			}
		}
	}
	
doTheDialog:
	
	if (bParent) ActivateParentDialog(FALSE);
	SaveVListSettings(&vSettings);
	item = SelectFromVListDialog(HELP_TOPICS_DIALOG, TOPICS_USERITEM,
								 numLines,
					  			 HTInit, 0, 0, DrawHTItem, HTClick,
								 FALSE, &dialogItem);
	RestoreVListSettings(vSettings);
	if (bParent) ActivateParentDialog(TRUE);
	
	if (dialogItem == TOPICS_SELECT || dialogItem == TOPICS_USERITEM) {
		t = NthLineInTextNonOptimized(*sharedTopics, item, title, 256);
		while (t[0] == ' ' || t[0] == ',') t++;// move past any leading spaces
		if(!t[0]) goto doTheDialog; // they picked a blank line
		// for old locations files before the name change
		if (!strcmp(t, "General NOAA Oil Modeling Environment (GNOME)") || !strcmp(t, "GNOME (General NOAA Oil Modeling Environment)") ) strcpy(t,"General NOAA Operational Modeling Environment (GNOME)");
		if (GetHelp(t, TRUE) != HELP_CANCEL) //JLM
		{	// error or request for topics again
			// JLM 8/15/96, by the way, shouldn't this have been bParent rather than TRUE ?
			//GetHelpTopics(t, TRUE); // JLM ????
			//GetHelpTopics(t, bParent); 
			goto doTheDialog; // JLM, avoids recursion
		}
	}
	
	_HUnlock(sharedTopics); // JLM, now it can be purged
	
	return dialogItem;
}
#endif


////////////////
/*
If the user selects more than one file, 
the lpstrFile buffer returns the path to the current directory 
followed by the file names of the selected files. 
The nFileOffset member is the offset, in bytes or characters, 
to the first file name, and the nFileExtension member is not used. 
For Explorer-style dialog boxes, the directory and file name strings are NULL separated, 
with an extra NULL character after the last file name.
*/

typedef struct {
	char szFile[30000];
	short nFileOffset;
} MultipleFileInfo;

MultipleFileInfo gMultipleFileInfo;

long NumFilesSelected(void)
{
	MultipleFileInfo multipleFileInfo = gMultipleFileInfo;
	long i;
	long numFiles = 0;
	// look for the first double null, counting nulls along the way
	// start at the first file name
	for(i = multipleFileInfo.nFileOffset; true; i++) {
		if(multipleFileInfo.szFile[i] == 0) 
			numFiles++;
		if(multipleFileInfo.szFile[i] == 0 && multipleFileInfo.szFile[i+1] == 0)
			break; // end of the files
	}
	return numFiles;

}

void strcpyNthFile(char *szFile,long n)
{
	MultipleFileInfo multipleFileInfo = gMultipleFileInfo;
	long numFiles = NumFilesSelected();
	long i,k = 0;
	long nFileOffset = multipleFileInfo.nFileOffset;
	szFile[0] = 0;
	if(numFiles == 1 && n == 1) {
		// then there is not a null separater, we can just use strcpy 
		strcpy(szFile,multipleFileInfo.szFile);
	}
	else {
		if(1<= n && n <= numFiles) {
			strcpy(szFile,multipleFileInfo.szFile);
			szFile[nFileOffset-1] = DIRDELIMITER;
			szFile[nFileOffset] = 0;

			// count past  n NULL characters
			// count past  n NULL characters
			k = 1;
			for(i = nFileOffset; true; i++) {
				if(k == n) {
					strcat(szFile,multipleFileInfo.szFile+i);
					return;
				}
				if(multipleFileInfo.szFile[i] == 0) {
					k++;
				}
			}
		}
	}

}


///////////////

#ifdef IBM
				
/////////////////////////////////// MACUTIL ///////////////
///////////////////////////////////////////////////////// MACUTIL ///////////////
///////////////////////////////////////////////////////// MACUTIL ///////////////
///////////////////////////////////////////////////////// MACUTIL ///////////////
///////////////////////////////////////////////////////// MACUTIL ///////////////

PortSettings mapPort = { 0, 0 }, infoPort = { 0, 0 },
			toolPort = {0,0}, // JLM 1/21/99
			 messagePort = { 0, 0 }, currentPort = { 0, 0 };
			 


Boolean makingMetaFile = FALSE, trueTypeMode = FALSE,
		scalePrintFonts = FALSE, doCompleteIsWindowCheck = FALSE;
char SFSaveDisk[_MAX_DIR];
WindowPtr currentWindow = 0, currentPaintWindow = 0;
HDC currentHDC = 0, currentPaintHDC = 0;
HINSTANCE hInst;
extern WindowPtr messageWnd;

long totalInits = 0, totalDeletes = 0;

/////////// MACUTIL: GRAPHICS

BOOL RegisterStandardWindows(void)
{
	WNDCLASS wc;
	
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
	wc.lpfnWndProc = VListWndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInst;
	wc.hIcon = NULL;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); // (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = "VList";
	
	if (!RegisterClass(&wc)) SysBeep(1);
	
	return TRUE;
}

Boolean InitCurrentPort(Boolean painting)
{
	HBITMAP hBitMap;
	PLOGFONT plf;
	unsigned char black[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	
	if (!MyIsWindow(currentPort.w)) {
		if (!painting)
			currentPort.hDC = (HDC)(currentPort.w);
	}
	else {
		if (currentPort.w == mapPort.w) {
			currentPort = mapPort;
			currentWindow = mapPort.w;
			currentHDC = mapPort.hDC;
			
			return TRUE;
		}
		if (currentPort.w == toolPort.w) {
			currentPort = toolPort;
			currentWindow = toolPort.w;
			currentHDC = toolPort.hDC;
			
			return TRUE;
		}
		if (currentPort.w == infoPort.w) {
			currentPort = infoPort;
			currentWindow = infoPort.w;
			currentHDC = infoPort.hDC;
			
			return TRUE;
		}
		if (currentPort.w == messagePort.w) {
			currentPort = messagePort;
			currentWindow = messagePort.w;
			currentHDC = messagePort.hDC;
			
			return TRUE;
		}
		if (!painting)
			currentPort.hDC = GetDC(currentPort.w);
	}
	currentWindow = currentPort.w;
	currentHDC = currentPort.hDC;
	
	if (!(hBitMap = CreateBitmap(8, 8, 1, 1, (LPSTR)&black)))
		{ SysBeep(1); return FALSE; }
	currentPort.hBrush = CreatePatternBrush(hBitMap);
	DeleteObject(hBitMap);
	if (!currentPort.hBrush)
		{ SysBeep(1); return FALSE; }
	
	if (!(currentPort.hPen = CreatePen(PS_SOLID, 1, RGB(0, 0, 0))))
		{ SysBeep(1); DeleteObject(currentPort.hBrush); currentPort.hBrush = 0; return FALSE; }
	
	plf = (PLOGFONT)LocalAlloc(LPTR, sizeof(LOGFONT));
	lstrcpy(plf->lfFaceName, "MS Sans Serif");
	plf->lfHeight = 12;
	currentPort.hFont = CreateFontIndirect(plf);
	LocalFree((LOCALHANDLE)plf);
	if (!currentPort.hFont)
		{ SysBeep(1);
		  DeleteObject(currentPort.hBrush); DeleteObject(currentPort.hPen);
		  currentPort.hBrush = 0; currentPort.hPen = 0; return FALSE; }
	
	if (!SelectObject(currentPort.hDC, currentPort.hBrush) ||
		!SelectObject(currentPort.hDC, currentPort.hPen) ||
		!SelectObject(currentPort.hDC, currentPort.hFont)) {
		SelectObject(currentPort.hDC, GetStockObject(BLACK_BRUSH));
		SelectObject(currentPort.hDC, GetStockObject(BLACK_PEN));
		SelectObject(currentPort.hDC, GetStockObject(SYSTEM_FONT));
		DeleteObject(currentPort.hBrush);
		DeleteObject(currentPort.hPen);
		DeleteObject(currentPort.hFont);
		currentPort.hBrush = 0; currentPort.hPen = 0; currentPort.hFont = 0;
		SysBeep(1);
		
		return FALSE;
	}
	
	SetBkMode(currentPort.hDC, TRANSPARENT);
	
	currentPort.pattern = BLACK;
	currentPort.penStyle = BLACK;
	currentPort.penWidth = 1;
	currentPort.font = kFontIDGeneva;
	currentPort.size = 12;
	currentPort.style = normal;
	currentPort.textMode = patOr;
	currentPort.textAngle = 0;
	currentPort.rgb = 0;
	currentPort.clipRgn = 0;
	currentPort.penPos.h = currentPort.penPos.v = 0;
	
	if (currentPort.w == hMainWnd) mapPort = currentPort;
	if (currentPort.w == messageWnd) messagePort = currentPort;
	if (currentPort.w == toolWnd) toolPort = currentPort;// JLM 1/21/99

	totalInits++;
    
    return TRUE;
}

void DeletePortGraphics()
{
    HFONT hFont;
	HBRUSH hBrush;
	HPEN hPen;
	HRGN hRgn;
	
	if (hBrush = (HBRUSH)SelectObject(currentPort.hDC, GetStockObject(BLACK_BRUSH)))
		if (hBrush) DeleteObject(hBrush); // delete current custom brush
	if (hBrush != currentPort.hBrush)
		SysBeep(1);
	if (hPen = (HPEN)SelectObject(currentPort.hDC, GetStockObject(BLACK_PEN)))
		if (hPen) DeleteObject(hPen); // delete current custom pen
	if (hPen != currentPort.hPen)
		SysBeep(1);
	if (hFont = (HFONT)SelectObject(currentPort.hDC, GetStockObject(SYSTEM_FONT)))
		if (hFont) DeleteObject(hFont); // delete current custom font
	if (hFont != currentPort.hFont)
		SysBeep(1);
	if (hRgn = (HRGN)currentPort.clipRgn)
		DeleteObject(hRgn);
	currentPort.hBrush = 0; currentPort.hPen = 0; currentPort.hFont = 0;
	
	totalDeletes++;
}

void ReleasePort(Boolean releaseStatics)
{
	if (!releaseStatics && currentPort.hDC) {
		// keep these ports around
		if (currentPort.hDC == mapPort.hDC) { mapPort = currentPort; goto done; }
		if (currentPort.hDC == messagePort.hDC) { messagePort = currentPort; goto done; }
		if (currentPort.hDC == toolPort.hDC) { toolPort = currentPort; goto done; }
		if (currentPort.hDC == infoPort.hDC) { infoPort = currentPort; goto done; }
	}
	
	if (currentPort.hDC) {
		DeletePortGraphics();
		if (currentPort.hDC != currentPaintHDC && MyIsWindow(currentPort.w))
			if (!ReleaseDC(currentPort.w, currentPort.hDC))
				SysBeep(1);
	}
	
	done:
		currentPort.w = currentWindow = 0;
		currentPort.hDC = currentHDC = 0;
}

void ReleaseStaticPorts()
{
	currentPort = mapPort;
	ReleasePort(TRUE);
	currentPort = infoPort;
	ReleasePort(TRUE);
	currentPort = messagePort;
	ReleasePort(TRUE);
	currentPort = toolPort;
	ReleasePort(TRUE);
	return;
}

void SetPortGrafPort(GrafPtr w)
{
	if (w == currentPort.w) return;
	ReleasePort(FALSE);
	if (currentPort.w = w)
		InitCurrentPort(FALSE);
}

void SetPort(WindowPtr w)
{
	if (w == currentPort.w) return;
	ReleasePort(FALSE);
	if (currentPort.w = w)
		InitCurrentPort(FALSE);
}

void SetPortBP(WindowPtr w, PAINTSTRUCT *ps)
{
	if (currentPaintWindow) SysBeep(1); // nested calls to SetPortBP()/SetPortEP()
	ReleasePort(FALSE);
	currentPort.w = w;
	currentPort.hDC = BeginPaint(currentPort.w, ps);
	InitCurrentPort(TRUE);
	currentPaintWindow = currentPort.w;
	currentPaintHDC = currentPort.hDC;
	ClearClipRect(); // JLM 2/9/99, their should be no clip in a new port with Begin Paint
}

void SetPortEP(WindowPtr w, PAINTSTRUCT *ps)
{
	if (!currentPaintWindow) SysBeep(1); // no matching call to SetPortBP()
	ReleasePort(FALSE);
	EndPaint(currentPaintWindow, ps);
	currentPaintWindow = 0;
	currentPaintHDC = 0;
	currentPort.w = w;
	InitCurrentPort(FALSE);
}

void GetPort(WindowPtr *w)
{
	*w = currentPort.w;
}

void GetPortGrafPtr(WindowPtr *w)
{
	*w = currentPort.w;
}

void GetPenState(PenState *ps)
{
	GetPen(&ps->pnLoc);
	ps->pnSize.h = ps->pnSize.v = currentPort.penWidth;
	ps->pnMode = GetROP2(currentPort.hDC);
	ps->pnPat = currentPort.pattern;
}

void SetPenState(PenState *ps)
{
	MyMoveTo(ps->pnLoc.h, ps->pnLoc.v);
	PenSize(ps->pnSize.h, ps->pnSize.v);
	PenMode(ps->pnMode);
	FillPat(ps->pnPat);
}

void PenMode(int drawMode)
{
	// if (drawMode != patXor)
	// 	drawMode = R2_MERGENOTPEN;
	SetROP2(currentPort.hDC, drawMode);
}



void FillPat(long pattern)
{
	Boolean changeColor = FALSE;
	MonoPatBits *bits;
	HGDIOBJ oldBrush;
	HBITMAP hBitMap;
	HBRUSH hBrush;
	COLORREF rgb;
	static MonoPatBits screenBits[] = {
		{ 0x00, 0, 0x00, 0, 0x00, 0, 0x00, 0, 0x00, 0, 0x00, 0, 0x00, 0, 0x00, 0 }, // black
		{ 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0 }, // white
	 // { 0x22, 0, 0x88, 0, 0x22, 0, 0x88, 0, 0x22, 0, 0x88, 0, 0x22, 0, 0x88, 0 }, // dark gray
	 // { 0xAA, 0, 0x55, 0, 0xAA, 0, 0x55, 0, 0xAA, 0, 0x55, 0, 0xAA, 0, 0x55, 0 }, // gray
		{ 0x7F, 0, 0xF7, 0, 0x7F, 0, 0xF7, 0, 0x7F, 0, 0xF7, 0, 0x7F, 0, 0xF7, 0 }, // lighter gray
		{ 0xDD, 0, 0x77, 0, 0xDD, 0, 0x77, 0, 0xDD, 0, 0x77, 0, 0xDD, 0, 0x77, 0 }, // light gray
		{ 0xAA, 0, 0x55, 0, 0xAA, 0, 0x55, 0, 0xAA, 0, 0x55, 0, 0xAA, 0, 0x55, 0 }, // gray
		{ 0x77, 0, 0x77, 0, 0x77, 0, 0x77, 0, 0x77, 0, 0x77, 0, 0x77, 0, 0x77, 0 }, // vert stripes
		{ 0x00, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0x00, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0 }, // horiz stripes
#ifdef MARPLOT
		{ 0xBB, 0, 0xDD, 0, 0xEE, 0, 0x77, 0, 0xBB, 0, 0xDD, 0, 0xEE, 0, 0x77, 0 }, // downstipes
#else
		{ 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFB, 0, 0xFD, 0, 0xFE, 0 }, // stipple
#endif
		{ 0x77, 0, 0xEE, 0, 0xDD, 0, 0xBB, 0, 0x77, 0, 0xEE, 0, 0xDD, 0, 0xBB, 0 }, // upstripes
		{ 0xEE, 0, 0xEE, 0, 0x00, 0, 0xEE, 0, 0xEE, 0, 0xEE, 0, 0x00, 0, 0xEE, 0 }, // boxes
		
		{ 0xDD, 0, 0x77, 0, 0xDD, 0, 0x77, 0, 0xDD, 0, 0x77, 0, 0xDD, 0, 0x77, 0 }, // light gray
		{ 0x7F, 0, 0xEF, 0, 0xFD, 0, 0xBF, 0, 0xF7, 0, 0xFE, 0, 0xDF, 0, 0xFB, 0 }, // lighter gray
		{ 0x7F, 0, 0xFF, 0, 0xF7, 0, 0xFF, 0, 0x7F, 0, 0xFF, 0, 0xF7, 0, 0xFF, 0 }, // lightest gray
		
		{ 0xCC, 0, 0xCC, 0, 0xCC, 0, 0xCC, 0, 0xCC, 0, 0xCC, 0, 0xCC, 0, 0xCC, 0 }, // horiz ants 1
		{ 0x66, 0, 0x66, 0, 0x66, 0, 0x66, 0, 0x66, 0, 0x66, 0, 0x66, 0, 0x66, 0 }, // horiz ants 2
		{ 0x33, 0, 0x33, 0, 0x33, 0, 0x33, 0, 0x33, 0, 0x33, 0, 0x33, 0, 0x33, 0 }, // horiz ants 3
		{ 0x99, 0, 0x99, 0, 0x99, 0, 0x99, 0, 0x99, 0, 0x99, 0, 0x99, 0, 0x99, 0 }, // horiz ants 4
		{ 0xCC, 0, 0xCC, 0, 0xCC, 0, 0xCC, 0, 0xCC, 0, 0xCC, 0, 0xCC, 0, 0xCC, 0 }, // horiz ants 5
		{ 0x66, 0, 0x66, 0, 0x66, 0, 0x66, 0, 0x66, 0, 0x66, 0, 0x66, 0, 0x66, 0 }, // horiz ants 6
		{ 0x33, 0, 0x33, 0, 0x33, 0, 0x33, 0, 0x33, 0, 0x33, 0, 0x33, 0, 0x33, 0 }, // horiz ants 7
		{ 0x99, 0, 0x99, 0, 0x99, 0, 0x99, 0, 0x99, 0, 0x99, 0, 0x99, 0, 0x99, 0 }, // horiz ants 8
		
		{ 0xFF, 0, 0xFF, 0, 0x00, 0, 0x00, 0, 0xFF, 0, 0xFF, 0, 0x00, 0, 0x00, 0 }, // vert ants 1
		{ 0x00, 0, 0xFF, 0, 0xFF, 0, 0x00, 0, 0x00, 0, 0xFF, 0, 0xFF, 0, 0x00, 0 }, // vert ants 2
		{ 0x00, 0, 0x00, 0, 0xFF, 0, 0xFF, 0, 0x00, 0, 0x00, 0, 0xFF, 0, 0xFF, 0 }, // vert ants 3
		{ 0xFF, 0, 0x00, 0, 0x00, 0, 0xFF, 0, 0xFF, 0, 0x00, 0, 0x00, 0, 0xFF, 0 }, // vert ants 4
		{ 0xFF, 0, 0xFF, 0, 0x00, 0, 0x00, 0, 0xFF, 0, 0xFF, 0, 0x00, 0, 0x00, 0 }, // vert ants 5
		{ 0x00, 0, 0xFF, 0, 0xFF, 0, 0x00, 0, 0x00, 0, 0xFF, 0, 0xFF, 0, 0x00, 0 }, // vert ants 6
		{ 0x00, 0, 0x00, 0, 0xFF, 0, 0xFF, 0, 0x00, 0, 0x00, 0, 0xFF, 0, 0xFF, 0 }, // vert ants 7
		{ 0xFF, 0, 0x00, 0, 0x00, 0, 0xFF, 0, 0xFF, 0, 0x00, 0, 0x00, 0, 0xFF, 0 }  // vert ants 8
	};
	static MonoPatBits printBits[] = {
		{ 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0 }, // vert stripes (print)
		{ 0x00, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0 }, // horiz stripes (print)
#ifdef MARPLOT
		{ 0x7F, 0, 0xBF, 0, 0xDF, 0, 0xEF, 0, 0xF7, 0, 0xFB, 0, 0xFD, 0, 0xFE, 0 }, // downstipes (print)
#else
		{ 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFB, 0, 0xFD, 0, 0xFE, 0 }, // stipple
#endif
		{ 0xFE, 0, 0xFD, 0, 0xFB, 0, 0xF7, 0, 0xEF, 0, 0xDF, 0, 0xBF, 0, 0x7F, 0 }, // upstripes (print)
		{ 0x00, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0 } // boxes (print)
	};
	
	if (pattern < BLACK || pattern > ANTSV8) pattern = GRAY;
	
	if (pattern == currentPort.pattern) return;
	
	currentPort.pattern = pattern;
	
	rgb = currentPort.rgb;
	
	if (pattern == BLACK)
		hBrush = CreateSolidBrush(rgb);
	else {
		if (scalePrintFonts && pattern >= VERTSTRIPES && pattern <= BOXES) {
			pattern -= (VERTSTRIPES - 1);
			bits = printBits;
		}
		else
			bits = screenBits;
			
		if(sharedPrinting &&false) // Windows 95 only supports 8X8 bitmaps
		{ 
			// we need to adjust the penSize for high resolution devices
			// assume the width is in Points = 1/72 of an inch 
			// and convert it to pixels for the penSize
			short width = 8; // one char
			short orignalBitsPerLine = 32;// one WORD
			short numPixOnPrinter = _max(width,round(width*PixelsPerPoint()));
			// it might be cleaner to keep the pattern to a multiple of the original
			short multiple = round(numPixOnPrinter/width);
			numPixOnPrinter = width*multiple;
			// Note: each scan line in the bits must be word aligned, i.e. padded with zeros
			short numBitsPerWord = 8* sizeof(WORD);
			short numWordsPerLine = numPixOnPrinter/numBitsPerWord;
			if(numPixOnPrinter % numBitsPerWord) numWordsPerLine++;
			long numBytesPerLine = numWordsPerLine * sizeof(WORD);
			long numBitsPerLine = 8*numBytesPerLine;
			long numScanLines = numPixOnPrinter;
			char* originalBits = (char*)bits[pattern - 1];
			
			char*  tempPtr = (char*)_NewPtrClear(numBytesPerLine*numScanLines);
			if(tempPtr)
			{
				// stretch the bits
				long i,j,lineNum;
					// go through original bits
				for(lineNum = 0; lineNum < numPixOnPrinter; lineNum++)
				{
					for(i = 0; i < numPixOnPrinter; i++)
					{	// find the bit in the orignal
						long originalLine  = lineNum/multiple;
						long originalBit = i/multiple;
						long originalBitNum = originalLine*orignalBitsPerLine + originalBit;
						Boolean bitIsSet = BitTst(originalBits,originalBitNum);
						long newBitNum = lineNum*numBitsPerLine + i;
						if(bitIsSet) BitSet(tempPtr,newBitNum);
						else BitClr(tempPtr,newBitNum);
					}
				}
				
				hBitMap = CreateBitmap(numPixOnPrinter, numPixOnPrinter, 1, 1, (LPSTR)tempPtr);
				_DisposePtr(tempPtr);
			}
			
			if (!hBitMap) return;
			
		}
		else
		{ 	// normal code, for screen,  8X8 pattern
			if (!(hBitMap = CreateBitmap(8, 8, 1, 1, (LPSTR)bits[pattern - 1]))) return;
		}
		
		hBrush = CreatePatternBrush(hBitMap);
		DeleteObject(hBitMap);
	}
	if (!hBrush) return;
	
	if (oldBrush = (HBRUSH)SelectObject(currentPort.hDC, hBrush))
		if (currentPort.hBrush)
			{ DeleteObject(currentPort.hBrush); currentPort.hBrush = 0; }
	currentPort.hBrush = hBrush;
}

short MacToWinStyle(short linePattern, short lineWidth)
{
	switch (linePattern) {
		case BLACK:
			switch (lineWidth) {
				case 1: return BLACK;
				case 2: return TWOPOINT;
				case 3: return THREEPOINT;
				default: return FOURPOINT;
			}
		case WHITE: return WHITE;
		case DARKGRAY: return DASHDOT;
		case GRAY: return DOTS;
		case LIGHTGRAY: return DASHDOTDOT;
		case VERTSTRIPES:
		case HORIZSTRIPES:
		case UPSTRIPES:
		case DOWNSTRIPES: return DASHES;
		case BOXES: return BOXES;
	}
	
	return BLACK;
}

void WinToMacStyle(long style, short *linePattern, short *lineWidth)
{
	switch (style) {
		case BLACK: *linePattern = BLACK; *lineWidth = 1; break;
		case WHITE: *linePattern = WHITE; break;
		case TWOPOINT: *linePattern = BLACK; *lineWidth = 2; break;
		case THREEPOINT: *linePattern = BLACK; *lineWidth = 3; break;
		case FOURPOINT: *linePattern = BLACK; *lineWidth = 4; break;
		case DOTS: *linePattern = GRAY; break;
		case DASHDOTDOT: *linePattern = LIGHTGRAY; break;
		case DASHDOT: *linePattern = DARKGRAY; break;
		case DASHES: *linePattern = HORIZSTRIPES; break;
		case BOXES: *linePattern = BOXES; break;
	}
}

void PenStyle(long style, long width)
{
	Boolean changeColor = FALSE;
	long penStyle, penSize;
	COLORREF rgb;
	HPEN hPen, oldPen;
	
	if (style == currentPort.penStyle && width == currentPort.penWidth) return;
	
	currentPort.penStyle = style;
	if (width > 4) width = 4;

	currentPort.penWidth = width;
	
	penSize = 1;
	switch (style) { // map Mac pattern/size to Windows pen style
		case BLACK: penStyle = PS_SOLID; penSize = width; break;
		case WHITE: penStyle = PS_NULL; break;
		case DARKGRAY: penStyle = PS_DASHDOTDOT; break;
		case GRAY: penStyle = PS_DOT; break;
		case LIGHTGRAY: penStyle = PS_DASHDOT; break;
		case VERTSTRIPES:
		case HORIZSTRIPES:
		case UPSTRIPES:
		case DOWNSTRIPES: penStyle = PS_DASH; break;
		case BOXES: penStyle = PS_SOLID; break;
	}
	
	if(sharedPrinting && !AreUsingThinLines())
	{ // we need to adjust the penSize for high resolution devices
		// assume the width is in Points = 1/72 of an inch 
		// and convert it to pixels for the penSize
		penSize =max(width,round(width*PixelsPerPoint()));
	}
	
	rgb = currentPort.rgb;
	
	hPen = CreatePen(penStyle, penSize, rgb);
	if (!hPen) return;
	
	if (oldPen = (HPEN)SelectObject(currentPort.hDC, hPen))
		if (currentPort.hPen)
			{ DeleteObject(currentPort.hPen); currentPort.hPen = 0; }
	currentPort.hPen = hPen;
}

void PenSize(long h, long v)
{
	HGDIOBJ oldPen;
	HPEN hPen;
	COLORREF rgb;
#pragma unused(v)
	
	return; // On pre-NT Windows, a line thicker than 1 can't have a pattern.
			// So we'll ignore PenSize commands, and leave it to the "line style"
			// to create thin patterned or thick solid pens.
	
	if (h == currentPort.penWidth) return;
	
 	currentPort.penWidth = h;

	if (h > 1 && currentPort.pattern == BLACK) {
		// only SOLID pens can be thicker than 1 in Windows
		rgb = currentPort.rgb;

		if (!(hPen = CreatePen(PS_SOLID, h, rgb))) return;
		if (oldPen = (HPEN)SelectObject(currentPort.hDC, hPen))
			if (currentPort.hPen)
				{ DeleteObject(currentPort.hPen); currentPort.hPen = 0; }
		currentPort.hPen = hPen;
	} 
}

void PenNormal()
{
	PenMode(patCopy);
	FillPat(BLACK);
	PenStyle(BLACK, 1);
	PenSize(1, 1);
}

void GetForeColor(RGBColor *c)
{
	*c = (RGBColor)currentPort.rgb;
}

void RGBForeColor(RGBColor *c)
{
	long pattern;

	if (*c == (RGBColor)currentPort.rgb) return;
	
	currentPort.rgb = (long)*c;
	
	pattern = currentPort.pattern;
	currentPort.pattern = -1;
	FillPat(pattern); // reset the pattern to use the new colors in the brush
	currentPort.penStyle = -1;
	PenStyle(pattern, currentPort.penWidth); // reset the style to use the new colors in the pen
	SetTextColor(currentPort.hDC, *c);
}

void ForeColor(short color)
{
#pragma unused(color)
}

void RGBBackColor(RGBColor *c)
{
	SetBkColor(currentPort.hDC, *c);
}

void TextMode(int mode)
{
	if (mode == currentPort.textMode) return;

	SetBkMode(currentPort.hDC, mode == patCopy ? OPAQUE : TRANSPARENT);
	currentPort.textMode = mode;
}


void TextFont(long font)  
{ 

	HFONT hFont, oldFont;
	PLOGFONT plf;
	long style, height;
	
	if (!font || font == currentPort.font) return;
	
	currentPort.font = font;
	
	plf = (PLOGFONT)LocalAlloc(LPTR, sizeof(LOGFONT));
	plf->lfCharSet = 1; // 1 = default character set // SYMBOL_CHARSET
	if (abs(font) > 100)
		strcpy(plf->lfFaceName, (char *)font);
	else
		switch (font) {
			case applFont: lstrcpy(plf->lfFaceName, "System"); break;
			case kFontIDMonaco: lstrcpy(plf->lfFaceName, "Courier New"); break;
			default: //JLM
			case kFontIDGeneva: lstrcpy(plf->lfFaceName, "MS Sans Serif"); break;
			case kFontIDNewYork: lstrcpy(plf->lfFaceName, "MS Serif"); break;
			case kFontIDTimes: lstrcpy(plf->lfFaceName, "Times New Roman"); break;
			case kFontIDCourier: lstrcpy(plf->lfFaceName, "Courier New"); break;
			case kFontIDSymbol: lstrcpy(plf->lfFaceName, "Symbol"); break;
			case kFontIDHelvetica: lstrcpy(plf->lfFaceName, "Arial"); break;
			//case kFontIDMarplot:
			//	lstrcpy(plf->lfFaceName, trueTypeMode ? "MARPLOTP" : "MARPLOTD");
			//	break;
		}
	//height = round(abs(currentPort.size) * 1.33);
	//if(ibmSize) height = *ibmSize; //JLM
	long widthInMillimeters =  GetDeviceCaps(currentHDC,HORZSIZE);
	long widthInPixs = GetDeviceCaps(currentHDC,HORZRES);
	float widthInRealInches = widthInMillimeters/25.4;
	short pixPerRealInch = widthInPixs/widthInRealInches;

	height = round((currentPort.size*pixPerRealInch)/72.0); 
	if(height < 0) height = -height; // make sure height is positive
	
	plf->lfHeight = -height;
	style = currentPort.style;
	if (style & italic) plf->lfItalic = TRUE;
	if (style & underline) plf->lfUnderline = TRUE;
	if (style & bold) plf->lfWeight = FW_BOLD;
	
	plf->lfEscapement = currentPort.textAngle * 10;
	plf->lfOrientation = currentPort.textAngle * 10;
	
	hFont = CreateFontIndirect(plf);
	if (oldFont = (HFONT)SelectObject(currentPort.hDC, hFont))
		if (currentPort.hFont)
			{ DeleteObject(currentPort.hFont); currentPort.hFont = 0; }
	currentPort.hFont = hFont;
	
	LocalFree((LOCALHANDLE)plf);

} 

short MyGetFontNum (char *name)
{
	short fNum = 0;
	
	if (!strcmpnocase(name, "System")) fNum = applFont;
	if (!strcmpnocase(name, "Courier New")) fNum = kFontIDMonaco;
	if (!strcmpnocase(name, "MS Sans Serif")) fNum = kFontIDGeneva;
	if (!strcmpnocase(name, "MS Serif")) fNum = kFontIDNewYork;
	if (!strcmpnocase(name, "Times New Roman")) fNum = kFontIDTimes;
	if (!strcmpnocase(name, "Symbol")) fNum = kFontIDSymbol;
	if (!strcmpnocase(name, "Modern")) fNum = kFontIDHelvetica;
	if (!strcmpnocase(name, "MARPLOT")) fNum = kFontIDMarplot;
	if (!strcmpnocase(name, "MARPLOT Font")) fNum = kFontIDMarplot;
	if (!strcmpnocase(name, "MARPLOTD")) fNum = kFontIDMarplot;
	if (!strcmpnocase(name, "MARPLOTP")) fNum = kFontIDMarplot;
	
	return fNum;
}


void getfnum(char *name, short *num)
{
	*num = 0;
	
	if (!strcmpnocase(name, "System")) *num = applFont;
	if (!strcmpnocase(name, "Courier New")) *num = kFontIDMonaco;
	if (!strcmpnocase(name, "MS Sans Serif")) *num = kFontIDGeneva;
	if (!strcmpnocase(name, "MS Serif")) *num = kFontIDNewYork;
	if (!strcmpnocase(name, "Times New Roman")) *num = kFontIDTimes;
	if (!strcmpnocase(name, "Symbol")) *num = kFontIDSymbol;
	if (!strcmpnocase(name, "Modern")) *num = kFontIDHelvetica;
	if (!strcmpnocase(name, "MARPLOT")) *num = kFontIDMarplot;
	if (!strcmpnocase(name, "MARPLOT Font")) *num = kFontIDMarplot;
	if (!strcmpnocase(name, "MARPLOTD")) *num = kFontIDMarplot;
	if (!strcmpnocase(name, "MARPLOTP")) *num = kFontIDMarplot;
}

void SetOutlinePreferred(Boolean trueType)
{
	trueTypeMode = trueType;
}

void MyTextFont (char *fontName)
{
	long	theFontNum;
	
	theFontNum = MyGetFontNum (fontName);
	TextFont (theFontNum);
	
	return;
}

void TextSize(long size)
{
	if (size == currentPort.size) return;
	TextFontSize(currentPort.font,size);
}


void TextFace(long style)
{
	if (style == currentPort.style) return;
	TextFontSizeFace(currentPort.font,currentPort.size,style);  
}


 //////////////////

void TextAngle(long angle)
{
	long font;
	
	if (angle == currentPort.textAngle) return;
	
	currentPort.textAngle = angle;
	
	font = currentPort.font;
	currentPort.font = -100;
	TextFont(font); // reset the font to use the new angle
}

WindowPtr FrontWindow()
{
	return GetActiveWindow(); // GetFocus()
}

void HiliteWindow(WindowPtr w, Boolean hilite)
{
	if (hilite) SetFocus(w);
}

void SelectWindow(WindowPtr w)
{
	SetActiveWindow(w); // SetFocus(w)
}

void MyShowWindow(WindowPtr w)
{
	ShowWindow(w, SW_SHOW);
}

void HideWindow(WindowPtr w)
{
	ShowWindow(w, SW_HIDE);
	// ??? send a close message so that DoChildClose() gets called
	// ??? SendMessage(w, WM_CLOSE, 0, 0);
}

void SizeWindow(WindowPtr w, short width, short height, Boolean update)
{
	RECT W, C;
	short newWidth, newHeight;
	
	GetWindowRect(w, &W);
	GetClientRect(w, &C);
	newWidth = width + ((W.right - W.left) - (C.right - C.left));
	newHeight = height + ((W.bottom - W.top) - (C.bottom - C.top));
	
	MoveWindow(w, W.left, W.top, newWidth, newHeight, update);
}


void ValidRect(RECTPTR r)
{
	RECT R;

	MakeWindowsRect(r, &R);
	ValidateRect(currentWindow, &R);
}

void GlobalToLocal(POINTPTR p)
{
	POINT P;

	MakeWindowsPoint(p, &P);
	ScreenToClient(currentWindow, &P);
	MakeMacPoint(&P, p);
}

void LocalToGlobal(POINTPTR p)
{
	POINT P;

	MakeWindowsPoint(p, &P);
	ClientToScreen(currentWindow, &P);
	MakeMacPoint(&P, p);
}					  

void ModifyMouseState()
{
	MSG message;

	if (PeekMessage(&message, NULL, WM_MOUSEFIRST, WM_MOUSELAST, PM_NOREMOVE))
		switch (message.message) {
			case WM_LBUTTONDOWN: SetMouseButton(TRUE); break;
			case WM_MOUSEMOVE: SetMouse(MAKEPOINTS(message.lParam)); break;
			case WM_LBUTTONUP: SetMouseButton(FALSE); break;
		}
}

void GetMouse(POINTPTR p)
{
	POINT P;
	
	GetCursorPos(&P);
	MakeMacPoint(&P, p);
	GlobalToLocal(p);
}

void SetMouse(POINTS p)
{
	// currentMousePos.h = p.x;
	// currentMousePos.v = p.y;
}

Boolean StillDown()
{
	Boolean mouseSwapped = GetSystemMetrics(SM_SWAPBUTTON);
	int res;
	if(mouseSwapped) res= GetAsyncKeyState(VK_RBUTTON);
	else res= GetAsyncKeyState(VK_LBUTTON);
	
	return res < 0;
}

Boolean Button()
{
	return StillDown();// on the IBM, these are the same
}

void SetMouseButton(Boolean button)
{
	// currentButtonDown = button;
}

void GetPen(POINTPTR p)
{
	// Boolean result;
	// POINT P;

	// result = GetCurrentPositionEx(currentHDC, &P);
	// p->h = (short)P.x;
	// p->v = (short)P.y;
	(*p) = currentPort.penPos;
}

void GetFontInfo(FontInfo *fInfo)
{
	//long size = currentPort.size;
	//
	//fInfo->ascent = (size * 2) / 3;
	//fInfo->descent = (size * 2) / 9;
	//fInfo->leading = size - (fInfo->ascent + fInfo->descent);
	//fInfo->widMax = currentPort.size + 2;
	
	// JLM 1/29/99
	TEXTMETRIC	tm;
	GetTextMetrics(currentHDC,&tm);
	fInfo->ascent = tm.tmAscent;
	fInfo->descent = tm.tmDescent;
	fInfo->leading = tm.tmExternalLeading;
	fInfo->widMax = tm.tmMaxCharWidth;
}

PicHandle GetPicture(short id)
{
	/*
	HANDLE hResource;
	PicHandle p;

	hResource = LoadResource(hInst, FindResource(hInst, (LPCTSTR)id, "METAFILE"));
	if (!hResource) return 0;

	if (!(p = (PICTUREH)_NewHandle(sizeof(Picture)))) return 0;

 	LockResource(hResource);
	p->hmf = SetMetaFileBits(hResource);
	
	return p;
	*/

	return 0;
}

void DrawPicture(PicHandle h, RECTPTR r)
{
	// draw the meta-file in the rect
}

void KillPicture(PicHandle h)
{
	DeleteMetaFile((**h).hmf);
	DisposeHandle((Handle)h);
}

void DrawBitmap(HBITMAP hBitmap, short x, short y, DWORD dwROP)
{
	BITMAP b;
	HDC hdcMem;
	POINT pSize, pOrg;
	HBITMAP oldBitmap=0,hBob=0;
	
	
	
	hdcMem = CreateCompatibleDC(currentHDC);
	if(!hdcMem) return;//JLM 1/29/99
	
	GetObject(hBitmap, sizeof(BITMAP), (LPSTR)&b);

   	hBob = CreateCompatibleBitmap(currentHDC, b.bmWidth,b.bmHeight);
	if(!hBob) return;
	
	// now paint the newbitmap white (clean slate)
	PatBlt (hdcMem,0, 0,b.bmWidth,b.bmHeight,WHITENESS);

	
	oldBitmap = (HBITMAP)SelectObject(hdcMem, hBitmap);//JLM 1/29/99

	SetMapMode(hdcMem, GetMapMode(currentHDC));
	
	pSize.x = b.bmWidth;
	pSize.y = b.bmHeight;
	DPtoLP(currentHDC, &pSize, 1);
	
	pOrg.x = pOrg.y = 0;
	DPtoLP(hdcMem, &pOrg, 1);
	
	BitBlt(currentHDC, x, y, pSize.x, pSize.y, hdcMem, pOrg.x, pOrg.y, dwROP);
	
	oldBitmap = (HBITMAP)SelectObject(hdcMem, hBitmap);//JLM 1/29/99
	DeleteDC(hdcMem);
	DeleteObject(hBob);
}

void MyMoveTo(short x, short y)
{
   MoveToEx(currentHDC, x, y, NULL);
   currentPort.penPos.h = x;
   currentPort.penPos.v = y;
}

void MyLineTo(short x, short y)
{
   LineTo(currentHDC, x, y);
   currentPort.penPos.h = x;
   currentPort.penPos.v = y;
}

void Move(short dx, short dy)
{
	Point p;
	
	GetPen(&p);
	MyMoveTo(p.h + dx, p.v + dy);
}

void Line(short dx, short dy)
{
	Point p;
	Rect r;
	
	GetPen(&p);
	if (dx == 0 && dy == 0) {
		if (currentPort.penWidth == 1)
			MyLineTo(p.h + 1, p.v);
		else {
			MySetRect(&r, p.h, p.v, p.h + currentPort.penWidth, p.v + currentPort.penWidth);
			PaintRect(&r);
		}
	}
	else
		MyLineTo(p.h + dx, p.v + dy);
}

void ArcHelper(RECTPTR r, short sA, short aA, Boolean fill)
{
	int x1, y1, x2, y2, x3, y3, x4, y4, a, b, w;
	float ra = 57.29578F;
	
	// w     - utililty variable
	// a     - semi-major axis
	// b     - semi-minor axis
	// sA    - start angle
	// aA    - arc angle (angle through which arc subtends)
	// ra    - degrees per radian
	// rPt   - pointer to client rectangle (structure type of 4 components)
	// (see SDK reference vol. 2 pp 7-57)
	// arc() - windows function (SDK reference vol. 1 pp 4-14)
	
	w = abs(sA / 360);
	if (sA < 0)
		sA = (sA < -360) ? sA + 360*w : sA;
	else
		sA = (sA > 360) ? sA - 360*w : sA;
	
	if (aA < -360) aA = -360;
	if (aA > 360) aA = 360;
	
	a  = (r->right - r->left)/2;
	b  = (r->bottom- r->top)/2;
	x1 = r->left;
	y1 = r->top;
	x2 = r->right;
	y2 = r->bottom;
	x3 = (short) ( x1 + a * (1 + sin((sA+aA)/ra)) );
	y3 = (short) ( y1 + b * (1 - cos((sA+aA)/ra)) );
	x4 = (short) ( x1 + a * (1 + sin(sA/ra)) );
	y4 = (short) ( y1 + b * (1 - cos(sA/ra)) );
	
	if ((aA < 0 && sA > 0) || (sA <= 0 && aA < 0)) {
		w  = x3; x3 = x4; x4 = w;
		w  = y3; y3 = y4; y4 = w;
	}
	if (aA == 0 && y3 <= b)
		x3 += 1;
	else
		if (aA == 0)
			x3 -= 1;
	if (aA == 0 && x3 <= a)
		y3 -= 1;
	else
		if (aA == 0)
			y3 += 1;
	
	if (fill)
		Pie(currentHDC, x1, y1, x2, y2, x3, y3, x4, y4);
	else
		Arc(currentHDC, x1, y1, x2, y2, x3, y3, x4, y4);
}

void FrameArc(RECTPTR r, short startAngle, short arcAngle)
{
	ArcHelper(r, startAngle, arcAngle, FALSE);
}

void PaintArc(RECTPTR r, short startAngle, short arcAngle)
{
	ArcHelper(r, startAngle, arcAngle, TRUE);
}

void FrameOval(RECTPTR r)
{
	FrameArc(r, 0, 360); // note: we use FrameArc rather than Ellipse() 
}

void PaintOval(RECTPTR r)
{
	long pattern = currentPort.pattern;
	int currentROP2;
	HPEN currentPen;

 	if (pattern == BLACK) {
 		currentROP2 = GetROP2(currentHDC);
 		if (currentROP2 != patXor) PenMode(patCopy);
	}
	currentPen = (HPEN)SelectObject(currentHDC, GetStockObject(NULL_PEN));
	Ellipse(currentHDC, r->left, r->top, r->right, r->bottom);
	SelectObject(currentHDC, currentPen);
	if (pattern == BLACK) PenMode(currentROP2);
}

void MyFrameRect(RECTPTR r)
{
	/*
	HBRUSH currentBrush, nullBrush;
	RECT R;
	
	nullBrush = (HBRUSH)GetStockObject(NULL_BRUSH);
	if (currentBrush = (HBRUSH)SelectObject(currentHDC, nullBrush)) {
		SelectObject(currentHDC, currentBrush);
		MakeWindowsRect(r, &R);
		FrameRect(currentHDC, &R, currentBrush);
	}
	*/
	
	MyMoveTo(r->left, r->top);
	MyLineTo(r->right, r->top);
	MyLineTo(r->right, r->bottom);
	MyLineTo(r->left, r->bottom);
	MyLineTo(r->left, r->top);
}

void MyFillRect(RECTPTR r, Pattern p)
{
	long savePattern, saveStyle, saveWidth;

	savePattern = currentPort.pattern;
	saveStyle = currentPort.penStyle;
	saveWidth = currentPort.penWidth;
	FillPat(p);
	PenStyle(WHITE, 1);
	PaintRect(r);
	FillPat(savePattern);
	PenStyle(saveStyle, saveWidth);
}

void PaintRect(RECTPTR r)
{
	long pattern = currentPort.pattern;
	int currentROP2;
	HPEN currentPen;

 	if (pattern == BLACK) {
 		currentROP2 = GetROP2(currentHDC);
 		if (currentROP2 != patXor) PenMode(patCopy);
	}
	currentPen = (HPEN)SelectObject(currentHDC, GetStockObject(NULL_PEN));
	Rectangle(currentHDC, r->left, r->top, r->right + 1, r->bottom + 1);
	SelectObject(currentHDC, currentPen);
	if (pattern == BLACK) PenMode(currentROP2);
}

void MyInvertRect(RECTPTR r)
{
	RECT R;

	MakeWindowsRect(r, &R);
	InvertRect(currentHDC, &R);
}

void EraseRect(RECTPTR r)
{
	HBRUSH whiteBrush;
	RECT R;

	MakeWindowsRect(r, &R);
	whiteBrush = (HBRUSH)GetStockObject(WHITE_BRUSH);
	FillRect(currentHDC, &R, whiteBrush);
}

RgnHandle NewRgn()
{
	return CreateRectRgn(0, 0, 0, 0);
}

void DisposeRgn(RgnHandle rgn)
{
	if (rgn) DeleteObject(rgn);
}

void MyPaintRgn(RgnHandle r)
{
	long pattern = currentPort.pattern;
	int currentROP2;

 	if (pattern == BLACK) {
 		currentROP2 = GetROP2(currentHDC);
  		if (currentROP2 != patXor) PenMode(patCopy);
	}
	PaintRgn(currentHDC, r);
	if (pattern == BLACK) PenMode(currentROP2);
}

void MyFillRgn(RgnHandle r, Pattern p)
{
	long savePattern, saveStyle, saveWidth;

	savePattern = currentPort.pattern;
	saveStyle = currentPort.penStyle;
	saveWidth = currentPort.penWidth;
	FillPat(p);
	PenStyle(WHITE, 1);
	MyPaintRgn(r);
	FillPat(savePattern);
	PenStyle(saveStyle, saveWidth);
}

void MyFrameRgn(RgnHandle r)
{
	HBRUSH currentBrush, nullBrush;
	long size;
	
 	size = currentPort.penWidth;
	nullBrush = (HBRUSH)GetStockObject(NULL_BRUSH);
	if (currentBrush = (HBRUSH)SelectObject(currentHDC, nullBrush)) {
		SelectObject(currentHDC, currentBrush);
		FrameRgn(currentHDC, r, currentBrush, size, size);
	}
}

void drawstring(CHARPTR s)
{
	Point p;
	// short offset;
	UINT textAlign;
	
	GetPen(&p);
	/*
	if (currentPort.size < 0)
		offset = -currentPort.size;
	else
		switch (currentPort.size) { // since "size" != point-size
			case 7: offset = 11; break;
			case 9: offset = 13; break;
			case 24: offset = 24; break; 
			default: offset = 13; break; 
		}
	*/
	textAlign = GetTextAlign(currentHDC);
	SetTextAlign(currentHDC, TA_BASELINE);
	TextOut(currentHDC, p.h, p.v, s, strlen(s));
	SetTextAlign(currentHDC, textAlign);
	
	Move(stringwidth(s), 0);
}

void DrawChar(char c)
{
	Point p;
	UINT textAlign;
	
	GetPen(&p);
	
	if (c == checkMark) {
		Move(0, -1);
		Move(0, -3);
		Line(3, 3);
		Line(8, -8);
		Move(-1, 0);
		Line(-7, 7);
		Line(-3, -3);
		MyMoveTo(p.h + 10, p.v);
		return;
	}
	
	/*
	if (currentPort.size < 0)
		offset = -currentPort.size;
	else
		switch (currentPort.size) { // since "size" != point-size
			case 7: offset = 11; break;
			case 9: offset = 13; break;
			case 24: offset = 24; break; 
			default: offset = 13; break; 
		}
	*/
	textAlign = GetTextAlign(currentHDC);
	SetTextAlign(currentHDC, TA_BASELINE);
	TextOut(currentHDC, p.h, p.v, &c, 1);
	SetTextAlign(currentHDC, textAlign);
	
	Move(CharWidth(c), 0);
}

int CharWidth(char ch)
{
	// int width;
	// GetCharWidth32(currentHDC, ch, ch, &width);
	// return width;
	char s[2];
	
	if (ch == LINEFEED) return 0;
	
	s[0] = ch;
	s[1] = 0; // makingMetaFile ? 0 : ' '
	s[2] = 0;
	
	return stringwidth(s);
}

short stringwidth(CHARPTR s)
{
	RECT size = { 0, 0, 0, 0 };
	GrafPtr savePort;
	PortSettings cp;
	
	/*
	if (makingMetaFile) {
		cp = currentPort;
		GetPort(&savePort);
		SetPort(hMainWnd);
		TextFont(cp.font);
		TextSize(cp.size);
		TextFace(cp.style);
		TextAngle(cp.textAngle);
	}
	*/
	
	if (makingMetaFile)
		size.right = strlen(s) * currentPort.size * (2.0 / 3.0);
	else
		DrawText(currentHDC, s, strlen(s), &size, DT_SINGLELINE | DT_CALCRECT | DT_NOPREFIX);
	
	/*
	if (makingMetaFile)
		SetPort(savePort);
	*/
	
	return size.right;
}

short StringWidth(CHARPTR s)
{
	short result;
	
	my_p2cstr(s);
	result = stringwidth(s);
	my_c2pstr(s);
	
	return result;
}

void MyInsetRect(RECTPTR r, short h, short v)
{
   r->left += h;
   r->right -= h;
   r->top += v;
   r->bottom -= v;
}

Boolean SectRect(RECTPTR r1, RECTPTR r2, RECTPTR r3)
{
   RECT R1, R2, result;
   
   MakeWindowsRect(r1, &R1);
   MakeWindowsRect(r2, &R2);
   IntersectRect(&result, &R1, &R2);
   MakeMacRect(&result, r3);
   
   return !IsRectEmpty(&result);
}


void MapPt(POINTPTR pt, RECTPTR srcRect, RECTPTR dstRect)
{
   // this routine inputs the pt in relation to srcRect
   // and returns the corresponding point in  dstRect via 
   // the parameter pt
   float slope;
	
   slope = ((float)(dstRect->right - dstRect->left)/(float)(srcRect->right - srcRect->left));
   pt->h = (short)rint(slope*(pt->h - srcRect->left) + dstRect->left);
   slope = ((float)(dstRect->bottom - dstRect->top)/(float)(srcRect->bottom - srcRect->top));
   pt->v = (short)rint(slope*(pt->v - srcRect->top) + dstRect->top);
}

/////////// MACUTIL: FILES

OSErr FSRead(int f, LONGPTR count, VOIDPTR buffer)
{
   DWORD numRead, error, origCount = *count;
   BOOL success;

   success = ReadFile ((HANDLE)f, buffer, *count, &numRead, NULL);
   *count = numRead; 

   if (!success) {
      error = GetLastError();
      return -1;
   }
   if ((LONG)numRead < (long)origCount) return eofErr;
   return 0;
}

OSErr FSWrite(int f, LONGPTR count, VOIDPTR buffer)
{
  	DWORD numWritten;
  	BOOL success;
	
	success = WriteFile ((HANDLE)f, buffer, *count, &numWritten, NULL); 
	*count = numWritten;
	
	return success ? 0 : -1;
}

OSErr GetFPos(int f, LONGPTR posOff)
{
   *posOff = SetFilePointer ((HANDLE)f, 0, NULL, FILE_CURRENT);
   if (*posOff == 0xFFFFFFFF) return -1;
   return 0;

}
                                          
OSErr SetFPos(int f, short posMode, long posOff)
{
   long result;
   
   switch (posMode) {
        case fsAtMark:     result = SetFilePointer ((HANDLE)f, 0, NULL, FILE_CURRENT); break;
        case fsFromStart:  result = SetFilePointer ((HANDLE)f, posOff, NULL, FILE_BEGIN); break;
		case fsFromLEOF:   result = SetFilePointer ((HANDLE)f, -(posOff), NULL, FILE_END); break;
		case fsFromMark:   result = SetFilePointer ((HANDLE)f, posOff, NULL, FILE_CURRENT); break;
   }
   if (result == 0xFFFFFFFF) return -1;
   else return 0;
}

OSErr GetEOF (int refNum, LONGPTR logEOF)
{
   *logEOF = GetFileSize ((HANDLE)refNum, NULL);
   if (*logEOF == 0xFFFFFFFF ) return -1;
   else return 0;
} 

OSErr SetEOF (int refNum, long logEOF)
{
   LONG curPos;

   curPos = SetFilePointer ((HANDLE)refNum, 0, NULL, FILE_CURRENT);  //Save the current file position
   SetFilePointer ((HANDLE)refNum, logEOF, NULL, FILE_BEGIN);
   SetEndOfFile ((HANDLE)refNum);
   SetFilePointer ((HANDLE)refNum, curPos, NULL, FILE_BEGIN);  //Restore the file pointer position
	
	return 0;
}

OSErr FSClose(int f)
{
   if (CloseHandle((HANDLE)f) == TRUE) return 0;
   else return -1;
}

Boolean GetModDateLong(short f, LONGPTR seconds)
{
   FILETIME modTime;

   struct {
      unsigned day : 5;
      unsigned month : 4;
      unsigned year80 : 7;  } mydate;
   struct {
      unsigned sec2 : 5;
      unsigned minutes : 6;
      unsigned hours : 5;  } mytime;
   struct tm base;

   GetFileTime ((HANDLE)f, NULL, NULL, &modTime);
   FileTimeToDosDateTime (&modTime, (LPWORD)&mydate, (LPWORD)&mytime);
      
   base.tm_year = mydate.year80 + 80;
   base.tm_mon =  mydate.month - 1;
   base.tm_mday = mydate.day;
   base.tm_hour = mytime.hours;
   base.tm_min = mytime.minutes;
   base.tm_sec = mytime.sec2 * 2;
   
	// JLM 6/23/00, added code to set tm_isdst here
	base.tm_isdst = -1; // this asks the C-library to calculate the daylight savings time flag for us
   
   // adjustment factor to change time base from 1970 to 1904
   *seconds = (unsigned long)mktime(&base) + 2082816000L;
   
	/////////////////////////////////////////////////
	// JLM 6/23/00, I'm guessing this code no longer needs to be here
	/////////////////////////////////////////////////
   //	// see mktime bug in GlenUtl (6/16/92 JLM)
   //	if (base.tm_isdst == 1)
   //	   *seconds -= 3600; // subtract one hour
	/////////////////////////////////////////////////
    
   return TRUE;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
// JLM network stuff 7/8/98
#define MAXNUMNETWORKS 10
#define MAXROOTLENGTH 64
typedef struct 
{
	char root[MAXROOTLENGTH];
} NetworkRoot,**NetworkRootH;
NetworkRootH gNetworkRoot;

void StrcpyIBMRoot(char* root, char* fileName)
{
	char uch;
	strncpy(root,fileName,MAXROOTLENGTH);
	root[MAXROOTLENGTH-1] = 0;
	uch = mytoupper(root[0]);
	if('A' <= uch && uch <= 'Z' && root[1] == ':') // of the form A-Z:
		root[2] = 0; // chop
	else if(root[0] == '\\' && root[1] == '\\') // network path
	{
		// check that it has no other backslashes besides the initial 2
		long i,len = strlen(root);
		for(i = 2; i< len;i++)
		{
			if(root[i] == '\\') 
			{
				root[i] = 0; // chop
				break;
			}
		}
	}
}

Boolean NameStartsAtIBMRoot(char* fileName) //JLM 6/25/98
{ // returns true if IBM path starts with "C:" etc or "\\" (network path)
	char uch = mytoupper(fileName[0]);
	if (strlen(fileName) > 1)
	{
		if(	('A' <= uch && uch <= 'Z'  && fileName[1] == ':') // of the form A-Z:
			|| (fileName[0] == '\\' && fileName[1] == '\\')) // network path
		{
			return true;
		}
	}
	return false;
}

Boolean NameIsIBMRoot(char* fileName) //JLM 6/25/98
{ 	// returns true if IBM path equals with "A-Z:" etc or "\\name" (network path)
	long len = strlen(fileName);
	if (strlen(fileName) > 1)
	{
		char uch = mytoupper(fileName[0]);
		if(  ('A' <= uch && uch <= 'Z' && fileName[1] == ':') )
		{ // of the form A-Z:
			// it must be strlen 2
			return (len == 2);
		}
		else if(fileName[0] == '\\' && fileName[1] == '\\') // network path
		{
			// check that it has no other backslashes besides the initial 2
			long i;
			for(i = 2; i< len;i++)
				if(fileName[i] == '\\') return false;
			return true;
		}
	}
	return false;
}

OSErr NetworkGetVolumeName(short vRefNum, char *volName)//JLM 7/8/98 network stuff
{
	OSErr err = 0;
	NetworkRoot rt;
	if(vRefNum <26) return -1; // not a network volume
	if(gNetworkRoot == nil) gNetworkRoot = (NetworkRootH) _NewHandleClear(MAXNUMNETWORKS*sizeof(NetworkRoot));
	if(gNetworkRoot)
	{
		int i = vRefNum - 26 ;
		if( i < MAXNUMNETWORKS)
		{	// look for it in the list
			rt = INDEXH(gNetworkRoot,i);
			strcpy(volName,rt.root);
			return 0;
		}
	}
	
	return -1;// not a network volume
}

short VRefNumFromIBMName(char* fileName) //JLM 8/7/98
{	// returns 0-25 for drives A-Z and higher numbers for network paths starting with "\\"
	short vRefNum= -1;
	
	char uch = mytoupper(fileName[0]);
	if(	'A' <= uch && uch <= 'Z') vRefNum = uch -'A';
	else if(fileName[0] == '\\' && fileName[1] == '\\') 
	{
		// network path
		char  root[MAXROOTLENGTH];
		NetworkRoot rt;
		StrcpyIBMRoot(root,fileName);
		if(gNetworkRoot == nil) gNetworkRoot = (NetworkRootH) _NewHandleClear(MAXNUMNETWORKS*sizeof(NetworkRoot));
		if(gNetworkRoot)
		{
			int i, firstBlank = MAXNUMNETWORKS;
			for(i = 0; i < MAXNUMNETWORKS ; i++)
			{	// look for it in the list
				rt = INDEXH(gNetworkRoot,i);
				if(rt.root[0] == 0 && firstBlank >= MAXNUMNETWORKS) firstBlank = i; // we found a blank
				if(!strcmpnocase(rt.root,root))
				{	// we found it
					return (i + 26);
				}
			}
			// if we get here, it is not in the list
			if(firstBlank < MAXNUMNETWORKS)
			{
				strcpy(rt.root,root);
				INDEXH(gNetworkRoot,firstBlank) = rt;
				return (firstBlank + 26);
			}
		}
	}
	return vRefNum;
}

OSErr FSMakeFSSpec(short check, long dirID, CHARPTR fileName, FSSpec *spec)
{
	char *p, path[256];
	
	// NOTE: On the PC, what was the vRefNum parameter becomes the check parameter
	
	my_p2cstr(fileName);
	if (NameStartsAtIBMRoot(fileName))
		strcpy(path, fileName);
	else {
		GetDirectoryFromID(dirID, path);
		strcat(path, fileName);
	}
	my_c2pstr(fileName);
	
	p = strrchr(path, DIRDELIMITER);
	if (p)
		p++;
	else
		p = path;
	
	strcpy(spec->name, p);
	my_c2pstr(spec->name);
	
	*p = 0;
	spec->vRefNum = VRefNumFromIBMName(path);
	spec->parID = dirID;
	
	return GetIDFromPath(&spec->parID, path, check != 0) ? 0 : -1;
}

static short sharedFileType;
static MyDlgHookProcPtr userCFHook;

LRESULT CALLBACK CFHook(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	char buttonTitle[20], help[20], windowTitle[100], *b;
	
	if (message == WM_INITDIALOG) {
		CenterDialog(hWnd, CENTER);
		SubstituteParamText(hWnd);
		if (!userCFHook) return TRUE;
	}
	
	if (message == WM_COMMAND && HIWORD(wParam) == BN_CLICKED) {
		if (LOWORD(wParam) == 1) // OK
			sharedFileType = SendMessage(GetDlgItem(hWnd, 1136), CB_GETCURSEL, 0, 0);
		else {
			getindstring(help, DLG_STRINGS, HELP_STRING); // "Help..."
			GetWindowText((WindowPtr)lParam, buttonTitle, 19);
			b = buttonTitle;
			if (b[0] == '&') b++;
			if (
			!strcmpnocase(help, b)
			|| !strcmpnocase("Help", b)
			|| !strcmpnocase("Help...", b)
			) 
			{
				getwtitle(hWnd, windowTitle);
				GetHelp(windowTitle, TRUE);
				return TRUE;
			}
		}
	}
	
	if (userCFHook && (*userCFHook)(hWnd, message, wParam, lParam))
		return TRUE;
	
	return FALSE;
}

BOOL CALLBACK CFHookNew(HWND hDlg, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	char windowTitle[100];
	LPOFNOTIFY pofn;
	
	switch (uMsg) {
		case WM_INITDIALOG:
			SubstituteParamText(hDlg);
			getwtitle(hDlg, windowTitle);
			setwtitle(GetParent(hDlg), windowTitle);
			break;
		
		case WM_NOTIFY:
			pofn = (LPOFNOTIFY)lParam;
			switch (pofn->hdr.code) {
				case CDN_INITDONE : // JLM 2/2/99
					CenterDialog(GetParent(hDlg), CENTER);
					break;
				case CDN_FILEOK:
					SetWindowLong(hDlg, DWL_MSGRESULT, 1);
					sharedFileType = SendMessage(GetDlgItem(GetParent(hDlg), cmb1), CB_GETCURSEL, 0, 0);
					break;
				case CDN_HELP:
					getwtitle(hDlg, windowTitle);
					GetHelp(windowTitle, TRUE);
					break;
			}
			return FALSE;
	}
	
	if (userCFHook && (*userCFHook)(hDlg, uMsg, wParam, lParam))
		return TRUE;
	
	return FALSE;
}
/////////////////////////////////////////////////


/* Moved all this above so Mac could use it too */

/////////////////////////////////////////////////

////////////////
/*
If the user selects more than one file, 
the lpstrFile buffer returns the path to the current directory 
followed by the file names of the selected files. 
The nFileOffset member is the offset, in bytes or characters, 
to the first file name, and the nFileExtension member is not used. 
For Explorer-style dialog boxes, the directory and file name strings are NULL separated, 
with an extra NULL character after the last file name.
*/
/*
typedef struct {
	char szFile[30000];
	short nFileOffset;
} MultipleFileInfo;

MultipleFileInfo gMultipleFileInfo;

long NumFilesSelected(void)
{
	MultipleFileInfo multipleFileInfo = gMultipleFileInfo;
	long i;
	long numFiles = 0;
	// look for the first double null, counting nulls along the way
	// start at the first file name
	for(i = multipleFileInfo.nFileOffset; true; i++) {
		if(multipleFileInfo.szFile[i] == 0) 
			numFiles++;
		if(multipleFileInfo.szFile[i] == 0 && multipleFileInfo.szFile[i+1] == 0)
			break; // end of the files
	}
	return numFiles;

}

void strcpyNthFile(char *szFile,long n)
{
	MultipleFileInfo multipleFileInfo = gMultipleFileInfo;
	long numFiles = NumFilesSelected();
	long i,k = 0;
	long nFileOffset = multipleFileInfo.nFileOffset;
	szFile[0] = 0;
	if(numFiles == 1 && n == 1) {
		// then there is not a null separater, we can just use strcpy 
		strcpy(szFile,multipleFileInfo.szFile);
	}
	else {
		if(1<= n && n <= numFiles) {
			strcpy(szFile,multipleFileInfo.szFile);
			szFile[nFileOffset-1] = DIRDELIMITER;
			szFile[nFileOffset] = 0;

			// count past  n NULL characters
			k = 1;
			for(i = nFileOffset; true; i++) {
				if(k == n) {
					strcat(szFile,multipleFileInfo.szFile+i);
					return;
				}
				if(multipleFileInfo.szFile[i] == 0) {
					k++;
				}
			}
		}
	}

}


///////////////

*/

void CommonFileDialog(Boolean get, OSType type, CHARPTR sType, CHARPTR defaultName,
					  MyDlgHookProcPtr hook, SFREPLYPTR reply,
					  short dialogID, ModalFilterProcPtr filter)
{
	Boolean useModernDialog;
	char szFile[256] = "", szDirName[256] = "", szFileName[256] = "", szFilter[256];
	DWORD error;
	OPENFILENAME ofn;
	static short prevIndex = -1, prevDialogID = -1, prevMode = -1;
	static OSType prevType = '    ';
	short defaultIndex = 0;
	short numItems = 0;
	short mode = model->GetModelMode();
	Boolean useLatestStyleDialog = (gNoaaVersion && get && ShiftKeyDown()) || (dialogID == -1);

	Boolean allowMultiple = get && useLatestStyleDialog && (mode >= ADVANCEDMODE); // JLM 6/4/10
	
	if (mode < ADVANCEDMODE)
	{
		getindstring(szFilter, 10000, 17); // All Files (*.*)@*.*@
		numItems+=1; defaultIndex = 1; // first item
	}
	else
	{
		getindstring(szFilter, 10000, 6); // All Files (*.*)@*.*@Text Files (*.TXT)@*.TXT@
		numItems+=2; defaultIndex = 1; // first item
	}
	
	memset(&ofn, 0, sizeof(OPENFILENAME));
	
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.hwndOwner = FrontWindow(); // hMainWnd
	ofn.hInstance = hInst;
	
	
	if (type == '.BM1') {
		getindstring(szFilter, 10000, 7); // Bitmap Files (*.bmp)@*.bmp@
		numItems=1; defaultIndex = 1;
	}
	///
	if (type == '.BMP') {
		getindstring(&szFilter[strlen(szFilter)], 10000, 7); // Bitmap Files (*.bmp)@*.bmp@
		numItems+=1; defaultIndex = numItems; // this item
	}
	//
	if (type == '.GRF') {
		getindstring(&szFilter[strlen(szFilter)], 10000, 8); // Graphics (*.bmp, *.wmf)@*.bmp;*.wmf@
		numItems+=1; defaultIndex = numItems; // this item
	}
	//
	if (type == '.WMF') {
		getindstring(szFilter, 10000, 7); // Bitmap Files (*.bmp)@*.bmp@
		getindstring(&szFilter[strlen(szFilter)], 10000, 9); // Metafiles (*.wmf)@*.wmf@
		numItems=2; defaultIndex = 1; // first item
		if (RunningOnWin32())
		{
			getindstring(&szFilter[strlen(szFilter)], 10000, 14); // Enhanced Metafiles (*.emf)@*.emf@
			numItems+=1;
		}
	}
	//
	if (type == '.MOV') {
		getindstring(&szFilter[strlen(szFilter)], 10000, 10); // Movie Files (*.mov)@*.mov@
		numItems+=1; defaultIndex = numItems; // this item
	}
	//
	if (type == '.MOS') {
		getindstring(&szFilter[strlen(szFilter)], 10000, 11); // File Series (*.000)@*.@
		numItems+=1; defaultIndex = numItems; // this item
	}
	//
	if (type == '.TAT') {
		getindstring(&szFilter[strlen(szFilter)], 10000, 11); // File Series (*.000)@*.@
		numItems+=1; defaultIndex = numItems; // this item
	}
	//
	if (type == '.LFS') {
		getindstring(&szFilter[strlen(szFilter)], 10000, 12); // Save Files (*.lfs)@*.lfs@
		numItems+=1; defaultIndex = numItems; // this item
		if(/*get && */model->GetModelMode() <  ADVANCEDMODE)
		{	// add .loc files as a possibility
			getindstring(&szFilter[strlen(szFilter)], 10000, 16); // Location Files (*.loc)@*.loc@
			numItems+=1; 
		}
	}
	//
	/*if (type == '.SAV') {
		getindstring(&szFilter[strlen(szFilter)], 10000, 13); // Save Files (*.sav)@*.sav@
		numItems+=1; defaultIndex = numItems; // this item
		if(get && model->GetModelMode() ==  ADVANCEDMODE)
		{	// add .loc files as a possibility
			getindstring(&szFilter[strlen(szFilter)], 10000, 16); // Location Files (*.loc)@*.loc@
			numItems+=1; defaultIndex = numItems; // this item
			getindstring(&szFilter[strlen(szFilter)], 10000, 12); // Save Files (*.lfs)@*.lfs@
			numItems+=1; 
		}
	}*/
	//
	if (type == '.SAV') {
		getindstring(&szFilter[strlen(szFilter)], 10000, 16); // Location Files (*.loc)@*.loc@
		numItems+=1; defaultIndex = numItems; // this item
		if(/*get && */model->GetModelMode() ==  ADVANCEDMODE)
		{	// add .sav files as a possibility
			getindstring(&szFilter[strlen(szFilter)], 10000, 13); // Save Files (*.sav)@*.sav@
			numItems+=1; defaultIndex = numItems; // this item
		}
		// add .lfs files as a possibility
		getindstring(&szFilter[strlen(szFilter)], 10000, 12); // Save Files (*.lfs)@*.lfs@
		numItems+=1; 
	}
	//
	if (get && type == WIZARDFILETYPE) {
		getindstring(&szFilter[strlen(szFilter)], 10000, 16); // Location Files (*.loc)@*.loc@
		numItems+=1; defaultIndex = numItems; // this item
		if(model->GetModelMode() ==  ADVANCEDMODE)
		{	// add .sav files as a possibility
			getindstring(&szFilter[strlen(szFilter)], 10000, 13); //Save Files (*.sav)@*.sav@
			numItems+=1; 
		}
		else
		{	// add .lfs files as a possibility
			getindstring(&szFilter[strlen(szFilter)], 10000, 12); // Save Files (*.lfs)@*.lfs@
			numItems+=1; 
		}
	}
	////
	////////////////////////////////////////////
	
	if (prevIndex == -1 || prevDialogID != dialogID || prevType != type || prevMode != mode)
		prevIndex = defaultIndex;
	
	StringSubstitute(szFilter, '@', 0);
	ofn.lpstrFilter = szFilter;
	ofn.nFilterIndex = prevIndex;
	
	if (defaultName && defaultName[0]) strcpy(szFile, defaultName);
	ofn.lpstrFile = szFile;
	ofn.nMaxFile  = 255;
	ofn.lpstrFileTitle = szFileName;
	ofn.nMaxFileTitle = 255;
	// GetCurrentDirectory(255, szDirName);
	// ofn.lpstrInitialDir = szDirName;
	// ofn.lpstrTitle = NULL;
	ofn.lpstrDefExt = sType;
	useModernDialog = RunningOnWin32() && FindResource(0, MAKEINTRESOURCE(dialogID + 50), RT_DIALOG);
	ofn.lpfnHook = useModernDialog ? (LPOFNHOOKPROC)CFHookNew : (LPOFNHOOKPROC)CFHook;
	ofn.Flags = OFN_ENABLEHOOK | OFN_HIDEREADONLY | OFN_PATHMUSTEXIST | OFN_NONETWORKBUTTON |
				OFN_ENABLETEMPLATE | OFN_OVERWRITEPROMPT ;
	if (get)
		ofn.Flags |= OFN_SHOWHELP; // put file no longer has a help button, JLM  7/28/99
	
	if (useModernDialog)
		ofn.Flags |= OFN_EXPLORER;
	ofn.lpTemplateName = MAKEINTRESOURCE(useModernDialog ? dialogID + 50 : dialogID);
	
	
	ofn.Flags |= OFN_ENABLESIZING; // 5/26/10

	if(useLatestStyleDialog) { // JLM 5/26/10 don't use our smaller template... all it has is a help button and we don't need that.	
		ofn.lpTemplateName = NULL; 
		ofn.Flags = OFN_HIDEREADONLY | OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT ;
		ofn.lpfnHook = NULL; // we seem to need to turn off the hook to get the bigger nicer dialog box
		ofn.lpstrTitle = "Open a File";
	}

	userCFHook = hook;

	if(allowMultiple) {
		memset(&gMultipleFileInfo,0,sizeof(gMultipleFileInfo));
		ofn.lpstrFile = gMultipleFileInfo.szFile;
		ofn.nMaxFile  = 30000;
		ofn.Flags |= OFN_ALLOWMULTISELECT|OFN_EXPLORER;
	}

	reply->good = get ? GetOpenFileName(&ofn) : GetSaveFileName(&ofn);
	error = CommDlgExtendedError();

	if(reply->good && allowMultiple) {
		// undo our trick
		// and return the first file selected
		gMultipleFileInfo.nFileOffset = ofn.nFileOffset;
		strcpyNthFile(szFile,1); 
	}

	prevIndex = ofn.nFilterIndex;
	prevDialogID = dialogID;
	prevType = type; // JLM 7/26/99
	prevMode = mode; // CMO 6/26/00
	
	strcpy(reply->fName, szFile);
	my_c2pstr(reply->fName);
	CharUpper(szFile);
	reply->vRefNum = szFile[0] - 65;
}

void sfpputfile(POINTPTR where, CHARPTR rType, CHARPTR defaultName,
            MyDlgHookProcPtr hook, SFREPLYPTR reply, short dialogID,
            ModalFilterProcPtr filter)
{
	char sType[5] = "";
	OSType type = 0;
	
	if (rType && strlen(rType) >= 4) {
		type = *(OSType *)rType;
		if (type == '.WMF')
			strcpy(sType, "BMP");
		else {
			sType[0] = rType[2];
			sType[1] = rType[1];
			sType[2] = rType[0];
			sType[3] = 0;
		}
	}

	
	CommonFileDialog(FALSE, type, sType, defaultName, hook, reply, dialogID, filter);
	where->h = reply->good ? sharedFileType : -1;
}

void sfpgetfile(POINTPTR where, CHARPTR defaultName, FileFilterProcPtr fileFilter,
				short numTypes, SFTYPELISTPTR typeList,
				MyDlgHookProcPtr hook, SFREPLYPTR reply, short dialogID,
				ModalFilterProcPtr filter)
{
	CommonFileDialog(TRUE, (numTypes == -1) ? 0 : typeList[3],
					 0, defaultName, hook, reply, dialogID, filter);
	where->h = reply->good ? sharedFileType : -1;
}


/////////// MACUTIL: DIALOGS

static char GV_ParamText0[256] = "", GV_ParamText1[256] = "",
			GV_ParamText2[256] = "", GV_ParamText3[256] = "";

void paramtext(CHARPTR s0, CHARPTR s1, CHARPTR s2, CHARPTR s3)
{
   strcpy(GV_ParamText0, s0);
   strcpy(GV_ParamText1, s1);
   strcpy(GV_ParamText2, s2);
   strcpy(GV_ParamText3, s3);
}

void GetDialogItem(DialogPtr dialog, short itemNum, SHORTPTR type, HANDLEPTR h, RECTPTR r)
{
	HWND hWnd;
	RECT R = { 0, 0, 0, 0 };
	
	*type = 0; // for now, perhaps later we can retrieve the window's type
	hWnd = (HWND)GetDlgItem(dialog, itemNum);
	*h = (Handle)hWnd;
	if (hWnd != nil && MyIsWindow(currentWindow)) {
		GetWindowRect(hWnd, &R);
		// MapWindowPoints(GetDesktopWindow(), dialog, (POINT *)&R, 2);
		MapWindowPoints(GetDesktopWindow(), currentWindow, (POINT *)&R, 2);
		MakeMacRect(&R, r);
	}
	else
		MakeMacRect(&R, r);
}

void SetDialogItem(DialogPtr dialog, short itemNum, short type, Handle h, RECTPTR r)
{
	WindowPtr w = GetDlgItem(dialog, itemNum);
#pragma unused(type)
#pragma unused(h)
	
	if (w) {
		MoveWindow(w, r->left, r->top, RectWidth(*r), RectHeight(*r), TRUE);
		// perhaps later we can set the window's type (not used)
		// set the window procedure according to h (not used)
	}
}

void setdialogitemtext(HWND w, CHARPTR text)
{
   SendMessage(w, WM_SETTEXT, 0, (LPARAM)text);
}

void SetDialogItemText(HWND w, CHARPTR text)
{
	my_p2cstr(text);
	setdialogitemtext(w, text);
	my_c2pstr(text);
}

void getdialogitemtext(HWND w, CHARPTR text)
{  
   text[0] = 0; // JLM in case the dialog item does not exist
   SendMessage(w, WM_GETTEXT, 256, (LPARAM)text);
}

void GetDialogItemText(HWND w, CHARPTR text)
{
	getdialogitemtext(w, text);
	my_c2pstr(text);
}

void MySelectDialogItemText(DialogPtr dialog, short itemNum, short begin, short end)
{
   SetFocus(GetDlgItem(dialog, itemNum));
   SendDlgItemMessage(dialog, itemNum, EM_SETSEL, begin, end);
}

/*void SelectDialogItemText(DialogPtr dialog, short itemNum, short begin, short end)
{
   SetFocus(GetDlgItem(dialog, itemNum));
   SendDlgItemMessage(dialog, itemNum, EM_SETSEL, begin, end);
}*/

void ShowDialogItem(DialogPtr dialog, short itemNum)
{
   ShowWindow(GetDlgItem(dialog, itemNum), SW_SHOW);
}

void HideDialogItem(DialogPtr dialog, short itemNum)
{
   ShowWindow(GetDlgItem(dialog, itemNum), SW_HIDE);
}

void DrawDialog(DialogPtr dialog)
{
	RedrawWindow(dialog, 0, 0, RDW_INVALIDATE | RDW_UPDATENOW);
}

DialogPtr GetNewDialog(int dialogID, Ptr dStorage, Ptr WindowPtr)
{
   // This version of GetNewDialog will create a modeless dialog (only).

   HWND hDlg;

   hDlg = CreateDialog (hInst, MAKEINTRESOURCE(dialogID), NULL, (DLGPROC)ModelessDlgProc);
   ShowWindow (hDlg, SW_SHOW);
   return hDlg;
}

BOOL CALLBACK ModelessDlgProc (HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
   // MSG msg;

   switch (message) {
      case WM_INITDIALOG:
         CenterDialog (hWnd, 0);
         return FALSE;

      case WM_CLOSE:
         DestroyWindow (hWnd);
         return TRUE;
   }

   //if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
   //{
   //   TranslateMessage (&msg);
   //   DispatchMessage (&msg);
   //   return (FALSE);
   //}

   return FALSE;
}

void DisposeDialog(DialogPtr theDialog)
{
   if (theDialog)
      DestroyWindow (theDialog);
}

/////////// MACUTIL: OTHER

void debugstr(CHARPTR s)
{
	MessageBox(0, s, "debugstr", MB_ICONSTOP | MB_OK);
}

void DebugStr(CHARPTR s)
{
	my_p2cstr(s);
	debugstr(s);
	my_c2pstr(s);
}

void SysBeep(short duration)
{
   MessageBeep(0);
}

void SetResLoad(Boolean load)
{
#pragma unused(load)
}

MenuHandle GetMenuHandle(short menuID)
{
   return GetMenu(hMainWnd);
}

short GetItemMark(MenuHandle m, short item, short *mark)
{
	if (GetMenuState(m, item - 1, MF_BYPOSITION) & MF_CHECKED) {
		*mark = checkMark;
		return checkMark;
	}
	
	*mark = noMark;
	
	return noMark;
}

void SetItemMark(MenuHandle m, short item, char mark)
{
	CheckMenuItem(m, item - 1, MF_BYPOSITION | (mark == noMark ? MF_UNCHECKED : MF_CHECKED));
}

short CountMenuItems(MenuHandle m)
{
	return GetMenuItemCount(m);
}

//void AddResMenu(MenuHandle theMenuHdl, OSType type)
void AppendResMenu(MenuHandle theMenuHdl, OSType type)
{
   // probably do nothing
}

void MyDrawMenuBar()
{
	DrawMenuBar(hMainWnd);
}

void getwtitle(WindowPtr w, CHARPTR title)
{
	GetWindowText(w, title, 255);
}

void setwtitle(WindowPtr w, CHARPTR title)
{
   SendMessage(w, WM_SETTEXT, 0, (long)title);
}

void numtostring(long n, CHARPTR s)
{
   char s2[100];
   
   _ltoa(n, s2, 10);
   strcpy(s, s2);
}

void stringtonum(CHARPTR s, LONGPTR n)
{
   char s2[100];
   
   strcpy(s2, s);
   *n = atol(s2);
}

void getstring(CHARPTR s, short id)
{
   s[0] = 0;
   LoadString(hInst, id, s, 255);
   if (!s[0]) SysBeep(1);
}

void getindstring(CHARPTR s, short groupID, short index)
{
   s[0] = 0;
   LoadString(hInst, groupID + index, s, 255);
   
   if (!s[0]) SysBeep(1);
}

void InitCursor(void)
{
	char name[100];
	WNDCLASS aClass;
	// HCURSOR arrow;
   	// WindowPtr port;
	// Rect portRect;
	// Point mouse;
	
	// Microsoft says we should only set the cursor
	// when the cursor is in our client area or when we are capturing all input
	// ... but I think we should do it always
	
	GetClassName(currentWindow, name, 99);
	GetClassInfo(hInst, name, &aClass);
	SetCursor(aClass.hCursor);
	
	// if (arrow = LoadCursor(NULL, IDC_ARROW)) SetCursor(arrow);
	/*
	GetPort(&port);
	if (port) {
		portRect = GetPortRect(port);
		GetMouse(&mouse);
	
		if (MyPtInRect(mouse, &portRect))
			if (arrow = LoadCursor(NULL, IDC_ARROW))
				SetCursor(arrow);
	}
	*/
}

void SetWRefCon(WindowPtr w, long value)
{
   SetWindowLong(w, GWL_USERDATA, value);
}

long GetWRefCon(WindowPtr w)
{
   return GetWindowLong(w, GWL_USERDATA);
}

short GetControlValue(ControlHandle h)
{
	char aClass[50];
	
	GetClassName(h, aClass, 49);
	if (!strcmp(aClass, "ScrollBar"))
		return GetScrollPos(h, SB_CTL);
	
	return SendMessage(h, BM_GETCHECK, 0, 0L);
}

void SetControlValue(ControlHandle h, short value)
{
	char aClass[50];
	
	GetClassName(h, aClass, 49);
	if (!strcmp(aClass, "ScrollBar"))
		SetScrollPos(h, SB_CTL, value, TRUE);
	
	SendMessage(h, BM_SETCHECK, value, 0);
}

short GetControlMinimum(ControlHandle h)
{
   int  minVal, maxVal;
   
   GetScrollRange(h, SB_CTL, &minVal, &maxVal);
   
   return(minVal);
}

short GetControlMaximum(ControlHandle h)
{
   int  minVal, maxVal;
   
   GetScrollRange(h, SB_CTL, &minVal, &maxVal);
   
   return(maxVal);
}

void SetControlMinimum(ControlHandle h, short minVal)
{
   SetScrollRange(h, SB_CTL, (int)minVal, (int)GetControlMaximum(h), FALSE);
}

void SetControlMaximum(ControlHandle h, short maxVal)
{
   SetScrollRange(h, SB_CTL, (int)GetControlMinimum(h), (int)maxVal, FALSE);
}

long GetDblTime(void)
{
	return (long)(GetDoubleClickTime() * .06);
}

// Note: on the Mac, a tick is every 60th of a second.
// On the IBM, it is the number of mSecs since the system was booted.
// so to convert, Nticks = Nms * 60 / 1000
// try gettimeofday() or  mach_absolute_time()
unsigned long TickCount()
{
   unsigned long ticks = (unsigned long)(GetTickCount() * .06);
   
   return ticks;
}

double rint(double value)	  // function code commented back in for PC
{
   return anint(value);
}

// anint: rounds to nearnest whole integer...rounds
double anint(double num1)
{
   double a,b,c;

   c = myfabs(num1);
   a = floor(c); // get whole integer lower than num1
   b = c - a; // get fractional part
   if (b >= 0.5) a = a + 1.0; // check to see how big it is; if 1/2 or larger, round up
   if (num1 < 0.0) a = -a; // put sign back on
   
   return a;
}

/*char *p2cstr(char *string)
{
	short i, len;
	
	len = (unsigned char)string[0];
	for (i = 0 ; i < len ; i++)
		string[i] = string[i + 1];
	string[len] = 0;
	
	return string;
}

void c2pstr(char *string)
{
	short i, len;
	
	len = strlen(string);
	if (len > 255) len = 255;
	for (i = len ; i > 0 ; i--)
		string[i] = string[i - 1];
	string[0] = (unsigned char)len;
}*/

#endif

////////////////////////// IBM-SPECIFIC FUNCTIONS //////////////////////////

#ifdef IBM

void MakeWindowsRect(RECTPTR macRect, RECT *windowsRect)
{
	windowsRect->top = macRect->top;
	windowsRect->left = macRect->left;
	windowsRect->bottom = macRect->bottom;
	windowsRect->right = macRect->right;
}

void MakeMacRect(RECT *windowsRect, RECTPTR macRect)
{
	macRect->top = windowsRect->top;
	macRect->left = windowsRect->left;
	macRect->bottom = windowsRect->bottom;
	macRect->right = windowsRect->right;
}

void MakeWindowsPoint(POINTPTR macPoint, POINT *windowsPoint)
{
	windowsPoint->x = macPoint->h;
	windowsPoint->y = macPoint->v;
}

void MakeMacPoint(POINT *windowsPoint, POINTPTR macPoint)
{
	macPoint->h = windowsPoint->x;
	macPoint->v = windowsPoint->y;
}

POINT MakePOINT(long x, long y)
{
	POINT p = { x, y };
	
	return p;
}

void DefineEvent(EventRecord *e, HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	e->hwnd = hWnd;
	e->message = message;
	e->wParam = wParam;
	e->lParam = lParam;
	e->time = (DWORD)GetMessageTime(); // GetTickCount();
	e->pt.x = LOWORD(lParam);
	e->pt.y = HIWORD(lParam);
}

Boolean IsEGA()
{
   HDC hDC = GetDC(GetDesktopWindow());
   Boolean result = GetDeviceCaps(hDC, VERTRES) < 400;
   
   ReleaseDC(GetDesktopWindow(), hDC);
   
   return result;
}

Boolean MyIsWindow(HWND w)
{
	long contextNumber;
	
	if (!IsWindow(w)) return FALSE;
	
	// if (!doCompleteIsWindowCheck) return TRUE;
	
	// if it says it's a window, it may still be a device context;
	// to make sure it's really a window, we'll try a DC operation
	
	contextNumber = SaveDC((HDC)w);
	
	if (contextNumber == 0) return TRUE; // not a DC, must be a window
	
	RestoreDC((HDC)w, contextNumber);
	
	return FALSE; // it's a DC, not a window
}

short GetWinOpSysNumber()
{
	DWORD v;
	
	v = GetVersion();
	
	if (v < 0x80000000) return 4; // Windows NT
	
	if (LOBYTE(LOWORD(v)) < 4) return 3; // Win32s on Windows 3.1
	
	return 9; // Windows 95
}

BOOL RunningOnWin32(void)
{
	return GetWinOpSysNumber() != 3;
}

BOOL RunningWindows95(void)
{
	return GetWinOpSysNumber() == 9;
}

BOOL LongFileNames(short vRefNum)
{
	char volumeName[256], drive[10] = "x:";
	DWORD maxLength, flags;
	
	drive[0] = (char)vRefNum;
	if (GetVolumeInformation(drive, volumeName, 255, 0, &maxLength, &flags, 0, 0))
		if (maxLength >= 255)
			return TRUE;
	
	return FALSE;
}

// center one window over another
BOOL CenterWindow (HWND hwndChild, HWND hwndParent)
{
        RECT    rChild, rParent;
        int     wChild, hChild, wParent, hParent;
        int     wScreen, hScreen, xNew, yNew;
        HDC     hdc;

        // Get the Height and Width of the child window
        GetWindowRect (hwndChild, &rChild);
        wChild = rChild.right - rChild.left;
        hChild = rChild.bottom - rChild.top;

        // Get the Height and Width of the parent window
        GetWindowRect (hwndParent, &rParent);
        wParent = rParent.right - rParent.left;
        hParent = rParent.bottom - rParent.top;

        // Get the display limits
        hdc = GetDC (hwndChild);
        wScreen = GetDeviceCaps (hdc, HORZRES);
        hScreen = GetDeviceCaps (hdc, VERTRES);
        ReleaseDC (hwndChild, hdc);

        // Calculate new X position, then adjust for screen
        xNew = rParent.left + ((wParent - wChild) /2);
        if (xNew < 0) {
                xNew = 0;
        } else if ((xNew+wChild) > wScreen) {
                xNew = wScreen - wChild;
        }

        // Calculate new Y position, then adjust for screen
        yNew = rParent.top  + ((hParent - hChild) /2);
        if (yNew < 0) {
                yNew = 0;
        } else if ((yNew+hChild) > hScreen) {
                yNew = hScreen - hChild;
        }

        // Set it, and return
        return SetWindowPos (hwndChild, NULL,
                xNew, yNew, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
}

void CreateParamTextString (char* string, char *newString)
{
	char ptChar, *stringPos, *ptString, *lastPos = string;
	
	// replace any ^0's - ^3's with the GV_ParamText globals
	
	newString[0] = 0;
	
	while (stringPos = strchr(lastPos, '^')) {
		ptChar = *(stringPos + 1);
		if (ptChar >= '0' && ptChar <= '3') {
			strncat(newString, lastPos, stringPos - lastPos);
			stringPos += 2;
			lastPos = stringPos;
			switch (ptChar) {
				case '0': ptString = GV_ParamText0; break;
				case '1': ptString = GV_ParamText1; break;
				case '2': ptString = GV_ParamText2; break;
				case '3': ptString = GV_ParamText3; break;
			}
			strcat(newString, ptString);
		}
	}
	strcat(newString, lastPos);
} 

void SubstituteParamText(HWND hWnd)
{
	HWND hWndCtl;
	char className[256], text[512], newText[1024];
	short i;
	
	for (i = 0; i < 256 ; i++)
		if (hWndCtl = GetDlgItem(hWnd, i)) {
			GetClassName(hWndCtl, className, 255);
			if (!strcmpnocase("static", className)) {
				GetWindowText(hWndCtl, text, 511);
				CreateParamTextString(text, newText);
				SetWindowText(hWndCtl, newText);
			} 
		}
}

BOOL CALLBACK ChoiceAlertProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) 
{
	char windowTitle[128], cancel[32], close[32],done[32];
	short i;
	
	switch (message) {
		case WM_INITDIALOG:
			CenterDialog(hWnd, 0);
			SubstituteParamText(hWnd);
			SetFocus(GetDlgItem(hWnd, 1));
			return FALSE;
		
		case WM_COMMAND:
			if (!lParam && !HIWORD(wParam)) { // from a non-existent item
				if (LOWORD(wParam) == 2) 
				{ // "cancel" item
					getindstring(cancel, DLG_STRINGS, CANCEL_STRING); // "Cancel"
					getindstring(close, DLG_STRINGS, CLOSE_STRING); // "Close"
					getindstring(done, DLG_STRINGS, DONE_STRING); // "Done"
					for (i = 1 ; i <= 4 ; i++)
					{
						if (GetDlgItem(hWnd, i) && GetDlgItemText(hWnd, i, windowTitle, 99)) {
							if (!strcmp(windowTitle, cancel) || !strcmp(windowTitle, close))
							{
								EndDialog(hWnd, i);
								return TRUE;
							}
						}
					}
					// if there was no cancel button ,look to see if there is a done button
					// that is not the first button, JLM 9/1/98
					for (i = 2 ; i <= 4 ; i++)
					{
						if (GetDlgItem(hWnd, i) && GetDlgItemText(hWnd, i, windowTitle, 99)) {
							if (!strcmp(windowTitle, done))
							{
								EndDialog(hWnd, i);
								return TRUE;
							}
						}
					}
						
					return TRUE;
				}
			}
			EndDialog(hWnd, LOWORD(wParam));
			return TRUE;
	}
	
	return FALSE;
}

///////////////// THE DIRECTORY TABLE


// METHOD 2

typedef struct {
		short size;
		CHARH table;
		long count;
	} DirTablePart;

static DirTablePart directoryTable[5] = { { 16, 0, 0 }, { 32, 0, 0 },
										  { 128, 0, 0 }, { 256, 0, 0 },
										  { 0, 0, 0 } };

OSErr InitDirectoryTable()
{
	long i;
	
	for (i = 0 ; directoryTable[i].size ; i++)
		if (!(directoryTable[i].table = (CHARH)_NewHandle(0))) return -1;
	
	return 0;
}

OSErr ReleaseDirectoryTable()
{
	long i;
	
	for (i = 0 ; directoryTable[i].size ; i++)
		if (directoryTable[i].table)
			DisposeHandle((Handle)directoryTable[i].table);
	
	return 0;
}

Boolean GetIDFromPath(long *dirID, char *pathName, Boolean check)
{
	long i, j, k, len;
	CHARH h;
	
	len = strlen(pathName);
	
	if (len == 0) return FALSE;
	
	for (i = 0 ; directoryTable[i].size ; i++)
		if (len < directoryTable[i].size)
			break;
	
	if (!directoryTable[i].size) return FALSE;
	
	for (j = 0, k = 0 ; j < directoryTable[i].count ; j++, k += directoryTable[i].size)
		if (!strcmpnocase(pathName, &INDEXH(directoryTable[i].table, k))) {
			(*dirID) = (j << 4) + i;
			
			return TRUE;
		}
	
	if (check)
		if (!FolderExists(0, 0, pathName))
			return FALSE;
	
	h = (CHARH)MySetHandleSize((Handle)directoryTable[i].table,
							   directoryTable[i].size *
							   (directoryTable[i].count + 1));
	if (!h) return FALSE;
	
	directoryTable[i].table = h;
	j = directoryTable[i].count;
	k = directoryTable[i].size * j;
	directoryTable[i].count++;
	
	strcpy(&INDEXH(directoryTable[i].table, k), pathName);
	(*dirID) = (j << 4) + i;
	
	return TRUE;
}

Boolean GetDirectoryFromID(long dirID, char *pathName)
{
	long i, j, k, n, newID;
	
	pathName[0] = 0;
	
	i = dirID & 0x0F;
	for (n = 0 ; directoryTable[n].size ; n++) ;
	if (i >= n) return FALSE;
	
	j = dirID >> 4;
	if (j >= directoryTable[i].count) return FALSE;
	
	k = j * directoryTable[i].size;
	
	if (INDEXH(directoryTable[i].table, k) == 0) {
		// this path has been renamed
		newID = *(long *)&INDEXH(directoryTable[i].table, k + 4);
		
		return GetDirectoryFromID(newID, pathName);
	}
	
	strcpy(pathName, &INDEXH(directoryTable[i].table, k));
	
	if (pathName[strlen(pathName) - 1] != '\\')
		strcat(pathName, "\\"); // normalize path
	
	return TRUE;
}

Boolean ResolveFileDirectory(short vRefNum, long dirID, CHARPTR fileName, CHARPTR pathName)
{
	if (NameStartsAtIBMRoot(fileName)) {
		strcpy(pathName, fileName);
		return TRUE;
	}
	
	if (!GetDirectoryFromID(dirID, pathName)) return FALSE;
	strcat(pathName, fileName);
	
	return TRUE;
}

OSErr RenameDirectoryInTable(char *oldPathName, char *newPathName)
{
	char updatedPath[256];
	long i, j, k, len, newID;
	
	// A directory has been renamed and we need to update the global table
	// to reflect the change, in case the old directory name is part of one
	// or more of the paths in the table.
	
	// oldPathName and newPathName are assumed either both to be
	// backslash-terminated or neither to be backslash-terminated.
	
	len = strlen(oldPathName);
	
	for (i = 0 ; directoryTable[i].size ; i++)
		for (j = 0, k = 0 ; j < directoryTable[i].count ; j++, k += directoryTable[i].size)
			if (!strncmpnocase(&INDEXH(directoryTable[i].table, k), oldPathName, len)) {
				strcpy(updatedPath, newPathName);
				strcat(updatedPath, &INDEXH(directoryTable[i].table, k + len));
				if (strlen(updatedPath) < (unsigned)directoryTable[i].size)
					strcpy(&INDEXH(directoryTable[i].table, k), updatedPath);
				else {
					if (!GetIDFromPath(&newID, updatedPath, FALSE)) return -1;
					// store pointer to new path in old path area
					INDEXH(directoryTable[i].table, k) = 0; // flag
					*(long *)&INDEXH(directoryTable[i].table, k + 4) = newID; // value
				}
			}
	
	return 0;
}

///////////////// METAFILES

typedef struct tagOLDRECT {
		short left;
		short top;
		short right;
		short bottom;
	} OLDRECT;

typedef struct {
		DWORD   key;
		WORD	hmf;
		OLDRECT	bbox;
		WORD    inch;
		DWORD   reserved;
		WORD    checksum;
	} ALDUSMFHEADER;

#define ALDUSKEY 0x9AC6CDD7

HMETAFILE RenderPlaceableMetaFile(int f)
{
	int wBytesRead;
	long lBytesRead;
	// DWORD dwSize;
	LPSTR lpMFBits;
	ALDUSMFHEADER aldusMFHeader;
	METAHEADER mfHeader;
	HMETAFILE hMF;
	
	_llseek(f, 0, 0);
	
	// read the placeable header
	wBytesRead = _lread(f, (LPSTR)&aldusMFHeader, sizeof(ALDUSMFHEADER));
	if (wBytesRead == -1 || wBytesRead < sizeof(ALDUSMFHEADER)) return 0;
	
	// read the metafile header
	_llseek(f, sizeof(aldusMFHeader), 0);
	wBytesRead = _lread(f, (LPSTR)&mfHeader, sizeof(METAHEADER));
	
	// allocate memory for the metafile bits
	if (!(lpMFBits = (LPSTR)GlobalAllocPtr(GHND, mfHeader.mtSize * 2))) return 0;
	
	// read metafile bits 
	_llseek(f, sizeof(aldusMFHeader), 0);
	lBytesRead = _hread(f, lpMFBits, mfHeader.mtSize * 2);
	
	if (lBytesRead == -1) {
		GlobalFreePtr(lpMFBits);
		return 0;
	}
	
	// set the metafile bits to the memory that we allocated
	// dwSize = GlobalSizePtr(lpMFBits);
	if (!(hMF = (HMETAFILE)SetMetaFileBitsEx(mfHeader.mtSize * 2, (BYTE *)lpMFBits)))
		return 0;
	
	return hMF;
}

HMETAFILE MyGetMetaFile(LPSTR path)
{
	int f, wBytesRead;
	DWORD dwIsAldus;
	HMETAFILE hMF;
	
	if ((f = _lopen(path, OF_READ)) == -1) return 0;
	
	// read the first dword of the file to see if it is a placeable wmf
	wBytesRead = _lread(f, (LPSTR)&dwIsAldus, sizeof(dwIsAldus));
	_lclose(f);
	
	if (wBytesRead == -1 || wBytesRead < sizeof(dwIsAldus)) {
		_lclose(f);
		return 0;
	}
	
	if (dwIsAldus != ALDUSKEY) // a windows metafile
		return GetMetaFile(path);
	
	f = _lopen(path, OF_READ);
	hMF = RenderPlaceableMetaFile(f);
	_lclose(f);
	
	return hMF;
}

OSErr MakeMetaFilePlaceable(LPSTR path, Rect frame)
{
	char fileName[256], sPath[256], tempPath[256];
	short f, wBytesRead, i, *p, sum = 0;
	long lBytesRead, count, err;
	DWORD dwIsAldus;
	METAHEADER mfHeader;
	ALDUSMFHEADER aldusMFHeader;
	LPSTR lpMFBits;
	
	if ((f = _lopen(path, OF_READ)) == -1) return 0;
	
	// read the first dword of the file to see if it is a placeable wmf
	wBytesRead = _lread(f, (LPSTR)&dwIsAldus, sizeof(dwIsAldus));
	_lclose(f);
	
	if (wBytesRead == -1 || wBytesRead < sizeof(dwIsAldus)) return -1;
	
	if (dwIsAldus == ALDUSKEY) return 0; // already a placeable metafile
	
	// read the metafile header
	f = _lopen(path, OF_READ);	
	_llseek(f, 0, 0);
	wBytesRead = _lread(f, (LPSTR)&mfHeader, sizeof(METAHEADER));
	
	// allocate memory for the metafile bits
	if (!(lpMFBits = (LPSTR)GlobalAllocPtr(GHND, mfHeader.mtSize * 2)))
		{ _lclose(f); return 0; }
	
	// read metafile bits 
	_llseek(f, 0, 0);
	lBytesRead = _hread(f, lpMFBits, mfHeader.mtSize * 2);
	
	if (lBytesRead == -1)
		{ _lclose(f); GlobalFreePtr(lpMFBits); return -1; }
	
	_lclose(f);
	
	// fill in the fields of the placeable header
	
	aldusMFHeader.key = ALDUSKEY; // DWORD
	aldusMFHeader.hmf = 0; // WORD
	aldusMFHeader.bbox.left = frame.left; // OLDRECT
	aldusMFHeader.bbox.top = frame.top;
	aldusMFHeader.bbox.right = frame.right;
	aldusMFHeader.bbox.bottom = frame.bottom;
	aldusMFHeader.inch = PixelsPerInchScreen(); // WORD
	aldusMFHeader.reserved = 0; // DWORD
	// the checksum is the sum, using the XOR operator, of the first
	// ten words of the header
	p = (short *)&aldusMFHeader;
	for (i = 0 ; i < 10 ; i++) {
		sum = sum ^ (*p);
		p++;
	}
	aldusMFHeader.checksum = sum; // WORD
	
	// write a temp file that contains the placeable header, followed
	// by the lpMFBits
	
	strcpy(sPath, path);
	SplitPathFile(sPath, fileName);
	strcpy(tempPath, sPath);
	strcat(tempPath, "temp.wmf");
	hdelete(0, 0, tempPath);
	if (err = hcreate(0, 0, tempPath, 'ttxt', '.WMF'))
		{ GlobalFreePtr(lpMFBits); return -1; }
	if (err = hopendf(0, 0, tempPath, fsCurPerm, &f))
		{ GlobalFreePtr(lpMFBits); return -1; }
	
	count = sizeof(ALDUSMFHEADER);
	if (err = FSWrite(f, &count, (LPSTR)&aldusMFHeader))
		{ _lclose(f); GlobalFreePtr(lpMFBits); return -1; }		
	if (err = FSWrite(f, &lBytesRead, lpMFBits))
		{ _lclose(f); GlobalFreePtr(lpMFBits); return -1; }
	
	_lclose(f);
	GlobalFreePtr(lpMFBits);
	
	// delete the path file
	hdelete(0, 0, path);
	
	// rename the temp file as the path file
	hrename(0, 0, tempPath, path);
	
	return 0;
}

#endif

short GetTextFont(void)
{
#ifdef MAC
	GrafPtr currentPort;
	GetPortGrafPtr(&currentPort);
#if TARGET_API_MAC_CARBON
	if(currentPort) return  GetPortTextFont(currentPort);
#else
	if(currentPort) return  currentPort->txFont;
#endif
	return 0;
#else
	return currentPort.font;
#endif
}

char GetDegreeChar(short fontNum)
{
#ifdef MAC
	return 0xA1;
#else
	if(!sharedPrinting)// printers do font substitution, so don't get fancy
	{	// not printing, so we can get fancy	 
		switch(fontNum) 
		{
			case kFontIDMonaco: // "Courier New")
			case kFontIDTimes: // "Times New Roman"
			case kFontIDCourier: // "Courier New"
			case kFontIDHelvetica:// "Arial"
				return 0xBA; // this char is prettier
		}
	}
	return 0xB0; // this char works in any font
	// return 0x00B0; 	// unicode option, bug with degree symbol on computers with Chinese font defaults, but this doesn't fix problem...
#endif
}

char GetSmartDoubleQuoteChar(short fontNum)
{
#ifdef MAC
	return '"';
#else
	switch(fontNum)
	{
		case kFontIDMonaco: // "Courier New")
		case kFontIDTimes: // "Times New Roman"
		case kFontIDCourier: // "Courier New"
		case kFontIDHelvetica:// "Arial"
			return 0x94; // this char is prettier
	}
	return '"'; // this char works in any font
#endif
}

