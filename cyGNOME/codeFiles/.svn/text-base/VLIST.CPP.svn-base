
#include "CROSS.h"

#ifdef MAC
	#pragma segment VLIST
	
	short VLISTDummy() { return 0; }
#endif

void mySetControlMaximum(VLISTPTR L, ControlHandle c, long m)
{
	if (m < 0) m = 0;
	L->realControlMaximum = m;
#ifdef MAC
	SetControlMaximum (c, _min(m, 32000));
	if (m > 0) HiliteControl(c, 0);
#else
	if (m == 0) SetScrollPos(c, SB_CTL, 0, FALSE);
	SetScrollRange(c, SB_CTL, 0, _min(m, 32000), TRUE);
	if (L->realControlMaximum == 0) EnableWindow(c, FALSE);
	SendMessage(c, WM_PAINT, 0, 0);
#endif
}

void mySetControlRange(VLISTPTR L, ControlHandle c, long min, long max)
{
	// if (min > max) return;
	if (min > max) min = max = 0;
	L->realControlMaximum = max;
#ifdef MAC
	SetControlMinimum (c, min);
	mySetControlMaximum   (L, c, max);
#else
	SetScrollRange(c, SB_CTL, min, max, FALSE); 
	mySetControlMaximum (L, c, max);
	if (L->realControlMaximum == 0) EnableWindow(c, FALSE);
	SendMessage(c, WM_PAINT, 0, 0);
#endif
}

void mySetControlValue(VLISTPTR L, ControlHandle c, long v)
{
	L->realControlValue = v;
#ifdef MAC
	if (L->realControlMaximum > 32000)
		SetControlValue(c, (short)(32000.0 * ((float)v / (float)L->realControlMaximum)));
	else
		SetControlValue(c, v);
#else
	if (L->realControlMaximum > 32000)
		SetScrollPos(c, SB_CTL, (32000.0 * ((float)v / (float)L->realControlMaximum)), TRUE); 
	else {
		SetScrollPos(c, SB_CTL, v, TRUE);
		if (L->realControlMaximum == 0) EnableWindow(c, FALSE);
	}
	SendMessage(c, WM_PAINT, 0, 0);
#endif
}

long myGetControlValue(VLISTPTR L, ControlHandle c)
{
#pragma unused(c)
	
	return L->realControlValue;
}

#define VLHILITE(L) VLInvertRecord(L, TRUE, TRUE)
#define VLUNHILITE(L) VLInvertRecord(L, FALSE, TRUE)

void VLInvertRecord(VLISTPTR L, Boolean selected, Boolean drawText)
{
#ifdef MAC
	Rect r;
	GrafPtr savePort;
	RgnHandle saveClip;
	//SysEnvRec theWorld;
	static Boolean lockout = FALSE;
	void VLDrawLines(VLISTPTR L, short beginLine, short endLine);
#pragma unused(selected, drawText)
	
	if (lockout) return;
	
	if (L->selectedRec < L->curRec || L->selectedRec > (L->curRec + L->viewLines - 1))
		return; // record is not visible
	
	GetPortGrafPtr(&savePort);
	//SetPort(L->port);
	SetPortWindowPort(L->window);
	saveClip = NewRgn();
	GetClip(saveClip);
	r = L->viewRect;
	r.right -= 16;
	ClipRect(&r);
	
	r.left = L->viewRect.left + 1;
	r.right = L->viewRect.right - 16;
	r.top = L->viewRect.top + 2 + (L->selectedRec - L->curRec) * L->lineHeight;
	r.bottom = r.top + L->lineHeight - 1;
	//if (L->hRect) (*L->hRect)(L->port, &r, L->selectedRec);
	if (L->hRect) (*L->hRect)(GetDialogFromWindow(L->window), &r, L->selectedRec);
	
	if (r.left < r.right) {
		//SysEnvirons(curSysEnvVers, &theWorld);
		//if (theWorld.hasColorQD)
		if (ColorQDAvailable())
			LMSetHiliteMode(LMGetHiliteMode() & 0x7F); // clear 0th bit = pHiliteBit
		InvertRect(&r);
	}
	
	
	SetClip(saveClip);
	DisposeRgn(saveClip);
	SetPortGrafPort(savePort);
#else
	HDC hDC;
	RECT r;
	Rect r2;
	COLORREF color;
	HBRUSH hBrush;
	GrafPtr savePort;
	long line;
	void VLDrawLines(VLISTPTR, short, short);
	
	if (L->selectedRec < L->curRec || L->selectedRec > (L->curRec + L->viewLines - 1))
		return; // record is not visible
	
	r.left = L->viewRect.left + 1;
	r.right = L->viewRect.right - GetSystemMetrics(SM_CXVSCROLL);
	r.top = L->viewRect.top + 1 + (L->selectedRec - L->curRec) * L->lineHeight;
	r.bottom = r.top + L->lineHeight;
	MakeMacRect(&r, &r2);
	
	line = L->selectedRec - L->curRec;
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(L->window);
	if (selected) {
		if (L->hRect) {
			(*L->hRect)(L->window, &r2, L->selectedRec);
			MakeWindowsRect(&r2, &r);
		}
		if (r.left < r.right) {
			hDC = GetDC(L->window);
			color = GetSysColor(COLOR_HIGHLIGHT);
			if (hBrush = CreateSolidBrush(color)) {
				FillRect(hDC, &r, hBrush);
				DeleteObject(hBrush);
			}
			ReleaseDC(L->window, hDC);
		}
	}
	else {
		EraseRect(&r2);
		L->selectedRec = -1;
	}
	SetPortGrafPort(savePort);
	if (drawText) VLDrawLines(L, line, line);
#endif
}
#ifdef MAC
void FillRectWithWhite(Rect * r)
{
	RGBColor saveColor;
	MyGetForeColor(&saveColor);
	MyForeColor(whiteColor);
	MyPaintRect(r);
	MyRGBForeColor(&saveColor);
}
#endif
void VLDrawLines(VLISTPTR L, short beginLine, short endLine)
{
#ifdef MAC
	short i;
	GrafPtr savePort;
	Rect r;
	RgnHandle clip;
	Boolean eraseBottom = beginLine == 0 && endLine == L->viewLines - 1;
	
	//if (!((WindowPeek)L->port)->visible) return;
	if (!IsWindowVisible(L->window)) return;
	
	clip = NewRgn();
	
	GetPortGrafPtr(&savePort);
	//SetPort(L->port);
	SetPortWindowPort(L->window);
	
	GetClip(clip);
	
	r = L->viewRect;
	r.right -= 16;
	ClipRect(&r);
	r.left++;
	
	endLine = _min(endLine, (L->numItems - 1) - L->curRec);
	for (i = beginLine ; i <= endLine ; i++) {
		r.top = L->viewRect.top + 1 + i * L->lineHeight;
		r.bottom = r.top + L->lineHeight;
		EraseRect(&r);
		FillRectWithWhite(&r); // we do this so that the background will be white in OS X
		// if (L->selectedRec == (L->curRec + i)) VLInvertRecord(L, FALSE, FALSE);
		MoveTo(r.left + 4, r.bottom - 3);
		TextMode(srcOr);
		//(*L->draw)(L->port, &r, L->curRec + i);
		if(L->draw) (*L->draw)(GetDialogFromWindow(L->window), &r, L->curRec + i);
		else if(L->drawInWindow) (*L->drawInWindow)(L->window, &r, L->curRec + i);
		if (L->selectedRec == (L->curRec + i)) VLInvertRecord(L, FALSE, FALSE);
	}
	// JLM, I don't understand this if statement
	// we should always erase to the bottom of the rect when drawing all the lines
	//if ((L->curRec + i) == L->numItems && endLine < L->viewLines) {
	//	r.top = L->viewRect.top + 1 + i * L->lineHeight;
	//	r.bottom = L->viewRect.bottom - 1;
	//	EraseRect(&r);
	//	FillRectWithWhite(&r);  // we do this so that the background will be white in OS X
	//}
	if(eraseBottom) {
		r.top = r.bottom ;//note: r.bottom was set in previous for loop
		r.bottom = L->viewRect.bottom - 1;
		EraseRect(&r);
		FillRectWithWhite(&r);  // we do this so that the background will be white in OS X
	}

	r = L->viewRect;
	ClipRect(&r);
	Draw1Control(L->scroll);
	
	SetClip(clip);
	DisposeRgn(clip);
	SetPortGrafPort(savePort);
#else
	Rect r;
	short i;
	GrafPtr savePort;
	HRGN clip;
	COLORREF color, oldColor;
	Boolean eraseBottom = beginLine == 0 && endLine == L->viewLines - 1;
	
	if (!IsWindowVisible(L->window)) return;
	
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(L->window);
	r = L->viewRect;
	clip = CreateRectRgn(r.left, r.top, r.right - GetSystemMetrics(SM_CXVSCROLL), r.bottom);
	SelectClipRgn(currentHDC, clip);
	DeleteObject(clip);
	r.left++;
	r.right = L->viewRect.right - GetSystemMetrics(SM_CXVSCROLL);

	// JLM : erase below the last viewed line
	{ // JLM 6/12/10
		// find the bottom of the last line that is possible
		short numItemsPossible = L->numItems - L->curRec;
		short bottomOfLastPossible = L->viewRect.top + numItemsPossible * L->lineHeight;
		Rect r2;
		if(bottomOfLastPossible < L->viewRect.bottom) {
			r2 = L->viewRect;
			r2.bottom--;
			r2.top = bottomOfLastPossible;
			EraseRect(&r2);
		}
	}


	endLine = _min(endLine, (L->numItems - 1) - L->curRec);
	for (i = beginLine ; i <= endLine ; i++) {
		r.top = L->viewRect.top + 1 + i * L->lineHeight;
		r.bottom = r.top + L->lineHeight;
		EraseRect(&r);
		if (L->selectedRec == (L->curRec + i)) VLInvertRecord(L, TRUE, FALSE);
		MyMoveTo(r.left + 4, r.bottom - 5);
		TextMode(srcOr);
		GetForeColor(&oldColor);
		color = GetSysColor(L->selectedRec == (L->curRec + i) ?
								COLOR_HIGHLIGHTTEXT : 
								COLOR_WINDOWTEXT);
		RGBForeColor(&color);
		//(*L->draw)(GetParent(L->window), &r, L->curRec + i);
		if(L->draw) (*L->draw)(GetParent(L->window), &r, L->curRec + i);
		else if(L->drawInWindow) (*L->drawInWindow)(GetParent(L->window), &r, L->curRec + i);
		RGBForeColor(&oldColor);
	}
	// JLM, I don't understand this if statement
	// we should always erase to the bottom of the rect when drawing all the lines
	//if ((L->curRec + i) == L->numItems && endLine < L->viewLines) {
	//	r.top = L->viewRect.top + 1 + i * L->lineHeight;
	//	r.bottom = L->viewRect.bottom - 1;
	//	EraseRect(&r);
	//}
	if(eraseBottom) {
		r.top = r.bottom;//note: r.bottom was set in previous for loop
		r.bottom = L->viewRect.bottom - 1;
		EraseRect(&r);
	}
	
	SelectClipRgn(currentHDC, 0);
	SetPortGrafPort(savePort);
#endif
}

void VLInvalidate(VLISTPTR L)
{
	Rect r = L->viewRect;
	InvalRectInWindow(r,mapWindow);
	// JLM 2/11/99 on the IBM, L->port is not mapWindow
	// we should be invalidating in L->port !!!
	//InvalRectInWindow(r,L->port);
	InvalRectInWindow(r,(L->window));
}

void VLUpdate(VLISTPTR L)
{
	Rect r = L->viewRect;
	GrafPtr savePort;

	VLDrawLines(L, 0, L->viewLines - 1);
#ifdef MAC
	GetPortGrafPtr(&savePort);	// might not need port stuff here
	//SetPort(L->port);
	SetPortWindowPort(L->window);	//
	PenNormal();
	FrameRect(&r); // in Windows, rect is framed in WM_PAINT handler
	SetPortGrafPort(savePort);
#endif

}

void VLUpdateLine(VLISTPTR L, long record)
{
	if (record < L->curRec || record > (L->curRec + L->viewLines - 1))
		return; // record is not visible
	
	VLDrawLines(L, record - L->curRec, record - L->curRec);
}

void VLUpdateSelectedLine(VLISTPTR L)
{
	VLUpdateLine(L, L->selectedRec);
}

void VLSetSelect(long newSel, VLISTPTR L)
{
	if (newSel >= -1 && newSel < L->numItems) {
		if (L->selectedRec != -1) VLUNHILITE(L);
		L->selectedRec = newSel;
		if (newSel != -1) VLHILITE(L);
	}
}

Boolean VLGetSelect(LONGPTR theRecord, VLISTPTR L)
{
	*theRecord = L->selectedRec;
	return *theRecord != -1;
}

void VLScrollText(VLISTPTR L, short n)
{
#ifdef MAC
	Rect r = L->viewRect;
	RgnHandle rgn;
	
	if (n < 0) {
		n = _max(n, L->curRec + L->viewLines - L->numItems);
		if (n == 0) return;
	}
	else {
		n = _min(n, L->curRec);
		if (n == 0) return;
	}
	
	InsetRect(&r, 1, 1);
	r.right -= 15;
	
	rgn = NewRgn();
	ScrollRect(&r, 0, n * L->lineHeight, rgn);
	DisposeRgn(rgn);
	
	L->curRec -= n;
	mySetControlValue(L, L->scroll, L->curRec);
   
	if (n > 0)
		VLDrawLines(L, 0, _min(n - 1, L->viewLines - 1));
	else
		VLDrawLines(L, _max(0, L->viewLines + n), L->viewLines - 1);
#else
	RECT winRect;
	Rect r = L->viewRect;
	
	r.top += 1;
	r.bottom -= 1;
	r.left += 1;
	r.right -= GetSystemMetrics(SM_CXVSCROLL);
	
	if (n < 0) {
		n = _max(n, L->curRec + L->viewLines - L->numItems);
		if (n == 0) return;
	}
	else {
		n = _min(n, L->curRec);
		if (n == 0) return;
	}
	
	MakeWindowsRect(&r, &winRect);
	ScrollWindow(L->window, 0, n * L->lineHeight, &winRect, &winRect);
	
	L->curRec -= n;
	mySetControlValue(L, L->scroll, L->curRec);
	
	// UpdateWindow(L->port);
	if (n > 0)
		VLDrawLines(L, 0, _min(n - 1, L->viewLines - 1));
	else
		VLDrawLines(L, _max(0, L->viewLines + n), L->viewLines - 1);
	ValidateRect(L->window, 0);
#endif
}

void VLAutoScroll(VLISTPTR L)
{
	if (L->selectedRec < L->curRec || L->selectedRec > (L->curRec + L->viewLines - 1)) {
		L->curRec = L->selectedRec - L->viewLines / 2;
		L->curRec = _min(L->curRec, L->numItems - L->viewLines);
		L->curRec = _max(L->curRec, 0);
		mySetControlValue(L, L->scroll, L->curRec);
		VLDrawLines(L, 0, L->viewLines - 1);
	}
}

static VLISTPTR passList;

pascal_ifMac void VLScrollActionProc(ControlHandle scroll, short part)
{
#pragma unused(scroll)
	
	switch (part) {
		case kControlUpButtonPart:	 VLScrollText(passList, 1); break;
		case kControlDownButtonPart: VLScrollText(passList, -1); break;
		case kControlPageUpPart:	 VLScrollText(passList, passList->viewLines - 1); break;
		case kControlPageDownPart:	 VLScrollText(passList, -(passList->viewLines - 1)); break;
	}
}

void VLScrollClick(short thePart, Point thePt, VLISTPTR L)
{
	long v;
	
#ifdef MAC 
	if (thePart == kControlIndicatorPart) {
		thePart = TrackControl(L->scroll, thePt, nil);
		v = GetControlValue(L->scroll);
		if (L->realControlMaximum > 32000)
			v = (long) ((float)v * ((float)L->realControlMaximum / 32000.0));
		mySetControlValue(L, L->scroll, v);
		L->curRec =  myGetControlValue(L, L->scroll);
		VLDrawLines(L, 0, L->viewLines - 1);
	}  
	else {
		passList = L;
		//thePart = TrackControl(L->scroll, thePt, (ControlActionUPP)MakeUPP((ProcPtr)&VLScrollActionProc, uppControlActionProcInfo));
		thePart = TrackControl(L->scroll, thePt, MakeControlActionUPP(VLScrollActionProc));
	}
#else
	if (thePart == kControlIndicatorPart) {
		v = GetScrollPos(L->scroll, SB_CTL);
		if (L->realControlMaximum > 32000)
			v = (long)((float)v * ((float)L->realControlMaximum / 32000.0));
		mySetControlValue(L, L->scroll, v);
		L->curRec = myGetControlValue(L, L->scroll);
		VLDrawLines(L, 0, L->viewLines - 1);
	}
	else {
		passList = L;
		VLScrollActionProc(NULL, thePart);
	}
#endif
}

Boolean VLClick(Point pt, VLISTPTR L)
{
#ifdef MAC
	short thePart;
	long recHit, time = lastEvent.when; // TickCount()
	ControlHandle control;
	Boolean dblClick = FALSE;
	Boolean inScrollControl  = false;
	
	
	// when TARGET_API_MAC_CARBON the user item is a control 
	// and is on top of the scroll control so we have to use TestControl rather than FindControl
	// but TestControl also works for MACB4CARBON 
	if(L->scroll) {
		thePart = TestControl(L->scroll,pt);
		if(thePart)
			inScrollControl = true;
	}
	///////
	
	//if(clickedInMacScrollControlPtr) *clickedInMacScrollControlPtr = inScrollControl;

	if (inScrollControl) {
		VLScrollClick(thePart, pt, L);
		return dblClick;
	}
	
	/*if (thePart = FindControl(pt, L->port, &control))
	if (control == L->scroll) {
		VLScrollClick(thePart, pt, L);
		
		return dblClick;
	}*/
	
	if (pt.h < L->viewRect.left || pt.h > L->viewRect.right - 16) return dblClick;
	
	// handle click in view rect
	do {
		if (pt.v <= L->viewRect.top) {
			if (L->curRec > 0) {
				VLUNHILITE(L);
				VLScrollText(L, 1);
				L->selectedRec = L->curRec;
				VLHILITE(L);
			}
			else if (L->selectedRec != 0) {
				VLUNHILITE(L);
				L->selectedRec = 0;
				VLHILITE(L);
			}
			GetMouse(&pt);
			continue;
		}
		if (pt.v >= L->viewRect.bottom) {
			if ((L->curRec + L->viewLines) < L->numItems) {
				VLUNHILITE(L);
				VLScrollText(L, -1);
				L->selectedRec = L->curRec + L->viewLines - 1;
				VLHILITE(L);
			}
			else if (L->selectedRec != (L->numItems - 1)) {
				VLUNHILITE(L);
				L->selectedRec = L->numItems - 1;
 				VLHILITE(L);
			}
			GetMouse(&pt);
			continue;
		}
		
		recHit = (pt.v - L->viewRect.top) / L->lineHeight + L->curRec;
		if (recHit < L->curRec || recHit > L->curRec + L->viewLines - 1) {
			GetMouse(&pt);
			continue;
		}
		
		// Make sure there is actually a record at this position
		if (recHit >= 0 && recHit < L->numItems && L->numItems > 0) {
			dblClick = ((time - L->clickTime) <= GetDblTime()) &&
						recHit == L->selectedRec;
			
			if (dblClick)
				break;
			else L->clickTime = 0;
			
			if (recHit != L->selectedRec) {
				if (L->selectedRec != -1) VLUNHILITE(L);
				L->selectedRec = recHit;
				VLHILITE(L);
			}
		}
		else {   // don't hilite anything
			if (L->selectedRec != -1) VLUNHILITE(L);
			L->selectedRec = -1;
		}
		GetMouse(&pt);
	} while (StillDown());
	
	L->clickTime = time;
	
	return dblClick;
#else
	return FALSE; // clicks in the listbox are taken care of in VLIBMClick
#endif
}

Boolean VListKeyEventHelper(char c, VLISTPTR L)
{
	short d = 0;
	long curSelection;
	
	if (!VLGetSelect(&curSelection, L)) return FALSE;
	
	if (c == HOMEKEY) { VLSetSelect(0, L); VLAutoScroll(L); return TRUE; }
	
	if (c == ENDKEY) { VLSetSelect(L->numItems - 1, L); VLAutoScroll(L); return TRUE; }
	
	if (c == PAGEUPKEY) {
		if (curSelection >= 0) d = -(_min(curSelection, L->viewLines));
		else { VLAutoScroll(L); return TRUE; }
	}
	
	if (c == PAGEDOWNKEY) {
		if (curSelection < (L->numItems - 1))
			d = _min((L->numItems - 1) - curSelection, L->viewLines);
		else { VLAutoScroll(L); return TRUE; }
	}
	
	if (c == UPKEY || c == UPKEYPLUS) {
		if (curSelection > 0) d = -1;
		else { VLAutoScroll(L); return TRUE; }
	}
	
	if (c == DOWNKEY || c == DOWNKEYPLUS) {
		if (curSelection < (L->numItems - 1)) d = 1;
		else { VLAutoScroll(L); return TRUE; }
	}
	
	if (!d) return FALSE;
	
	VLSetSelect(curSelection + d, L);
	VLAutoScroll(L);
	
	return TRUE;
}

#ifdef MAC
Boolean VListKeyEvent(EVENTRECORDPTR e, VLISTPTR L)
{
	char c = (e->message & keyCodeMask) >> 8;
	
	if (e->what != keyDown && e->what != autoKey) return false;
	
	return VListKeyEventHelper(c, L);
}
#endif

Boolean VLAddRecordRowIsSelected(VLISTPTR L)
{
	long curSelection;
	Boolean  isLastItem;
	VLGetSelect(&curSelection,L);
	isLastItem = curSelection == -1 || curSelection == (*L).numItems-1;
	return isLastItem;
}


void ShowHideVList(DialogPtr dialog, VLISTPTR L, Boolean shown)
#ifdef MAC
{
	Rect r2, r = L->viewRect;
	
	if (shown && r.left < 0) {
		r.left += 1000;
		r.right += 1000;
		ShowControl(L->scroll);
	}
	if (!shown && r.left > 0) {
		r.left -= 1000;
		r.right -= 1000;
		HideControl(L->scroll);
	}
	
	if (!EqualRects(r, L->viewRect)) {
		r2 = (r.left > 0) ? r : L->viewRect;
		L->viewRect = r;
		InsetRect(&r2, -1, -1);
		EraseRect(&r2);
		InvalRectInWindow(r2, GetDialogWindow(dialog));
	}
}
#else
{
}
#endif

void VLReset2(VLISTPTR L, long numItems, Boolean draw)
{
#ifdef MAC
	GrafPtr savePort;
	Rect r = L->viewRect;
	
	GetPortGrafPtr(&savePort);
	//SetPort(L->port);
	SetPortWindowPort(L->window);
	
	L->numItems = numItems;
	// JLM 2/10/99
	// this code was reseting the curRec when the scroll bar was all 
	// the way down and the user hit step which called this routine  
	// So  I changed from >= to > 
//	if (draw || (L->curRec + L->viewLines) >= numItems) L->curRec = 0;
	if (draw || (L->curRec + L->viewLines) > numItems) L->curRec = 0;
	
	mySetControlMaximum(L, L->scroll, L->numItems - L->viewLines);
	mySetControlValue(L, L->scroll, L->curRec);
	
	if (L->selectedRec >= L->numItems)
		L->selectedRec = L->numItems - 1;
	
	if (draw) {
		InsetRect(&r, 1, 1);
		r.right -= 15;
		EraseRect(&r);
		VLDrawLines(L, 0, L->viewLines - 1);
	}
	
	SetPortGrafPort(savePort);
#else
	HDC hDC;
	Rect r = L->viewRect;
	RECT R;
	
	L->numItems = numItems;
	// JLM 2/10/99
	// this code was reseting the curRec when the scroll bar was all 
	// the way down and the user hit step which called this routine  
	// So  I changed from >= to > 
//	if (draw || (L->curRec + L->viewLines) >= numItems) L->curRec = 0;
	if (draw || (L->curRec + L->viewLines) > numItems) L->curRec = 0;
	
	mySetControlValue(L, L->scroll, L->curRec);
	mySetControlMaximum(L, L->scroll, L->numItems - L->viewLines);
	EnableWindow(L->scroll, L->numItems > L->viewLines);
	
	L->selectedRec = (L->numItems > 0) ? 0 : -1;
	r.top += 1;
	r.bottom -=1;
	r.left += 1;
	r.right -= GetSystemMetrics(SM_CXVSCROLL);
	
	if (draw) {
		hDC = GetDC(L->window);
		MakeWindowsRect(&r, &R);
		FillRect(hDC, &R, (HBRUSH)GetStockObject(WHITE_BRUSH));
		ReleaseDC(L->window, hDC);
		VLDrawLines(L, 0, L->viewLines - 1);
	}
#endif
}

void VLReset(VLISTPTR L, long numItems)
{
	VLReset2(L, numItems, TRUE);
}

void VLResetNR(VLISTPTR L, long numItems)
{
	VLReset2(L, numItems, FALSE);
}

void VLDeleteItem(long item, VLISTPTR L)
{
	long scrollMax;
	Rect r = L->viewRect;
#ifdef IBM
	HDC hDC;
	RECT R;
#endif
#pragma unused(item)
	
	if (!L->numItems) return;
	
	L->numItems--;
	if (L->selectedRec == L->numItems) L->selectedRec--;
	if (L->curRec && (L->numItems - L->curRec) < L->viewLines)
		L->curRec--;
	
	mySetControlMaximum(L, L->scroll, scrollMax = _max(L->numItems - L->viewLines, 0));
	mySetControlValue(L, L->scroll, _min(myGetControlValue(L, L->scroll), scrollMax));
	
	MyInsetRect(&r, 1, 1);
#ifdef MAC
   r.right -= 15;
   EraseRect(&r);
#else
	EnableWindow(L->scroll, L->numItems > L->viewLines);
	r.right -= GetSystemMetrics(SM_CXVSCROLL) - 1;
	hDC = GetDC(L->window);
	MakeWindowsRect(&r, &R);
	FillRect(hDC, &R, (HBRUSH)GetStockObject(WHITE_BRUSH));
	ReleaseDC (L->window, hDC);
#endif
}

void VLAddItem(long item, VLISTPTR L)
{
#pragma unused(item)
	
	L->numItems++;
	mySetControlMaximum(L, L->scroll, L->numItems - L->viewLines);
}

void VLResize(VLISTPTR L, Rect r)
{
	//if (L->port == 0) return;
	if (L->window == 0) return;
		
	L->viewLines = RectHeight(r) / L->lineHeight;
	mySetControlMaximum(L, L->scroll, L->numItems - L->viewLines);
	if (L->curRec > L->numItems - L->viewLines)
		L->curRec = _max(0, L->numItems - L->viewLines);
#ifdef MAC
	InsetRect(&r, -1, -1);
	L->viewRect = r;
	MoveControl(L->scroll, r.right - 16, r.top);
	SizeControl(L->scroll, 16, RectHeight(r));
//	SizeControl(L->scroll, 16, RectHeight(r) - 1);
#else
	L->viewRect.left = 0;
	L->viewRect.top = 0;
	L->viewRect.right = RectWidth(r);
	L->viewRect.bottom = RectHeight(r);
	MoveWindow(L->window, r.left, r.top, RectWidth(r), RectHeight(r), TRUE);
	MoveWindow(L->scroll, RectWidth(r) - GetSystemMetrics(SM_CXVSCROLL),
						  0, 
						  GetSystemMetrics(SM_CXVSCROLL),
						  RectHeight(r),
						  TRUE);
#endif
}

OSErr VLNewHelper(
		DialogRef diawind, short listItem,void (*draw)(DialogRef, RECTPTR, long), 
		WindowRef window, RECTPTR r, void (*drawInWindow)(WindowRef, RECTPTR, long),
		long numItems, VLISTPTR L)
{
// USAGE:
// for a list in a dialog : call with window and drawInWindow set to zero
// for a list in a window : call with diawind and draw set to zero

#ifdef MAC
	FontInfo finfo;
	Rect viewRect;
	
	memset(L,0,sizeof(*L)); // JLM 4/21/06
	
	if(diawind) window = GetDialogWindow(diawind);
	
	SetWRefCon(window, (long)L);
	
	L->numItems = numItems;
	L->curRec = 0;
	L->window = window;
	L->clickTime = L->keyTime = 0;
	L->draw = draw;
	L->drawInWindow = drawInWindow;
	L->hRect = 0;
	
	GetFontInfo(&finfo);
	L->lineHeight = finfo.ascent + finfo.descent + finfo.leading;
	viewRect = listItem ? GetDialogItemBox(diawind, listItem) : (*r);
	L->viewLines = RectHeight(viewRect) / L->lineHeight;
	if (listItem) // size view to hold an integral number of lines
		viewRect.bottom = viewRect.top + L->viewLines * L->lineHeight;
	else
		if (RectHeight(viewRect) != L->viewLines * L->lineHeight)
			L->viewLines++;
	InsetRect(&viewRect, -1, -1); // JLM, I think this compensates for the line we draw around the list
	L->viewRect = viewRect;
	
	viewRect.left = viewRect.right - 16;
	L->scroll = NewControl(window, &viewRect, "\p", true, 0, 0, 0, 16, 0);
	HiliteControl(L->scroll, L->numItems > L->viewLines ? 0 : 255);
	SetControlMinimum(L->scroll, 0);
	mySetControlMaximum(L, L->scroll, L->numItems - L->viewLines);
	mySetControlValue(L, L->scroll, 0);
	//SetCtlMin(L->scroll, 0);
	//mySetCtlMax(L, L->scroll, L->numItems - L->viewLines);
	//mySetCtlValue(L, L->scroll, 0);
	
	L->selectedRec = (L->numItems > 0) ? 0 : -1;
	VLDrawLines(L, 0, L->viewLines - 1);
	
	return 0;
#else
	HWND hWndList;
	Rect viewRect;
	RECT winRect;
	POINT pt[2];
	
	memset(L,0,sizeof(*L)); // JLM 4/21/06

	if(diawind) window = GetDialogWindow(diawind);
	
	SetWRefCon(window, (long)L);
	
	L->numItems = numItems;
	L->curRec = 0;
	L->clickTime = L->keyTime = 0;
	L->draw = draw;
	L->drawInWindow = drawInWindow;
	
	if (listItem)
		hWndList = GetDlgItem(diawind, listItem);
	else {
		hWndList = CreateWindow("VList", NULL, WS_CHILD | WS_VISIBLE,
								r->left, r->right, RectWidth(*r), RectHeight(*r),
								window, 0, hInst, NULL);
	}
	
	L->window = hWndList; // the window is the actual listbox itself
	L->lineHeight = 18;
	GetClientRect(hWndList, &winRect); // upper-left corner is (0, 0)
	
	MakeMacRect(&winRect, &viewRect); 
	L->viewLines = RectHeight(viewRect) / L->lineHeight;
	if (listItem) { // size view to hold an integral number of lines
		viewRect.bottom = viewRect.top + L->viewLines * L->lineHeight + 2;
		pt[0].x = 0;
		pt[0].y = 0;
		pt[1].x = viewRect.right;
		pt[1].y = viewRect.bottom;
		MapWindowPoints(hWndList, diawind, pt, 2);
		MoveWindow(hWndList, pt[0].x, pt[0].y,
				   pt[1].x - pt[0].x, pt[1].y - pt[0].y, FALSE); 
	}
	else
		if (RectHeight(viewRect) != L->viewLines * L->lineHeight)
			L->viewLines++;
	L->viewRect = viewRect;
	
	// add the scrollbar as a child to our user item box
	L->scroll = CreateWindow ("scrollbar", NULL, WS_CHILD | WS_VISIBLE | SBS_VERT,
							  viewRect.right - GetSystemMetrics(SM_CXVSCROLL), 
							  0, GetSystemMetrics(SM_CXVSCROLL), viewRect.bottom, 
							  hWndList, 0, hInst, NULL);
	EnableWindow(L->scroll, L->numItems > L->viewLines);   
	mySetControlRange(L, L->scroll, 0, L->numItems - L->viewLines);
	mySetControlValue(L, L->scroll, 0);
	
	L->selectedRec = (L->numItems > 0) ? 0 : -1;
	
	SetWRefCon(hWndList, (long)L); 
	
	VLDrawLines(L, 0, L->viewLines - 1);
	
	return 0;
#endif
}


OSErr VLNewInWindow(WindowRef window, RECTPTR r, long numItems,void (*drawInWindow)(WindowRef, RECTPTR, long), VLISTPTR L)
{
	return VLNewHelper(0,0,0, window, r, drawInWindow, numItems,L);
}

OSErr VLNew(DialogRef diawind, short listItem, RECTPTR r, long numItems,
			void (*draw)(DialogRef, RECTPTR, long), VLISTPTR L)
{
	return VLNewHelper(diawind,listItem,draw, 0, r, 0, numItems,L);
}

/*OSErr VLNew(DialogPtr diawind, short listItem, RECTPTR r, long numItems,
			void (*draw)(DialogPtr, RECTPTR, long), VLISTPTR L)
{
#ifdef MAC
	FontInfo finfo;
	Rect viewRect;
	
	SetWRefCon(diawind, (long)L);
	
	L->numItems = numItems;
	L->curRec = 0;
	L->port = diawind;
	L->clickTime = L->keyTime = 0;
	L->draw = draw;
	L->hRect = 0;
	
	GetFontInfo(&finfo);
//	L->lineHeight = finfo.ascent + finfo.descent + finfo.leading;
	L->lineHeight = 14;
	viewRect = listItem ? GetDialogItemBox(diawind, listItem) : (*r);
	L->viewLines = RectHeight(viewRect) / L->lineHeight;
	if (listItem) // size view to hold an integral number of lines
		viewRect.bottom = viewRect.top + L->viewLines * L->lineHeight;
	else
		if (RectHeight(viewRect) != L->viewLines * L->lineHeight)
			L->viewLines++;
	InsetRect(&viewRect, -1, -1);
	L->viewRect = viewRect;
	
	viewRect.left = viewRect.right - 16;
	L->scroll = NewControl(diawind, &viewRect, "\p", true, 0, 0, 0, 16, 0);
	HiliteControl(L->scroll, L->numItems > L->viewLines ? 0 : 255);
	SetControlMinimum(L->scroll, 0);
	mySetControlMaximum(L, L->scroll, L->numItems - L->viewLines);
	mySetControlValue(L, L->scroll, 0);
	
	L->selectedRec = (L->numItems > 0) ? 0 : -1;
	VLDrawLines(L, 0, L->viewLines - 1);
	
	return 0;
#else
	HWND hWndList;
	Rect viewRect;
	RECT winRect;
	POINT pt[2];
	
	SetWRefCon(diawind, (long)L);
	
	L->numItems = numItems;
	L->curRec = 0;
	L->clickTime = L->keyTime = 0;
	L->draw = draw;
	
	if (listItem)
		hWndList = GetDlgItem(diawind, listItem);
	else {
		hWndList = CreateWindow("VList", NULL, WS_CHILD | WS_VISIBLE,
								r->left, r->right, RectWidth(*r), RectHeight(*r),
								diawind, 0, hInst, NULL);
	}
	
	L->port = hWndList; // the port is the actual listbox itself
	L->lineHeight = 16;
	GetClientRect(hWndList, &winRect); // upper-left corner is (0, 0)
	
	MakeMacRect(&winRect, &viewRect); 
	L->viewLines = RectHeight(viewRect) / L->lineHeight;
	if (listItem) { // size view to hold an integral number of lines
		viewRect.bottom = viewRect.top + L->viewLines * L->lineHeight + 2;
		pt[0].x = 0;
		pt[0].y = 0;
		pt[1].x = viewRect.right;
		pt[1].y = viewRect.bottom;
		MapWindowPoints(hWndList, diawind, pt, 2);
		MoveWindow(hWndList, pt[0].x, pt[0].y,
				   pt[1].x - pt[0].x, pt[1].y - pt[0].y, FALSE); 
	}
	else
		if (RectHeight(viewRect) != L->viewLines * L->lineHeight)
			L->viewLines++;
	L->viewRect = viewRect;
	
	// add the scrollbar as a child to our user item box
	L->scroll = CreateWindow ("scrollbar", NULL, WS_CHILD | WS_VISIBLE | SBS_VERT,
							  viewRect.right - GetSystemMetrics(SM_CXVSCROLL), 
							  0, GetSystemMetrics(SM_CXVSCROLL), viewRect.bottom, 
							  hWndList, 0, hInst, NULL);
	EnableWindow(L->scroll, L->numItems > L->viewLines);   
	mySetControlRange(L, L->scroll, 0, L->numItems - L->viewLines);
	mySetControlValue(L, L->scroll, 0);
	
	L->selectedRec = (L->numItems > 0) ? 0 : -1;
	
	SetWRefCon(hWndList, (long)L); 
	
	VLDrawLines(L, 0, L->viewLines - 1);
	
	return 0;
#endif
}*/

#ifdef IBM
BOOL CALLBACK WinPropProc(HWND w, LPCSTR s)
{
	static int nProp = 1;    /* property counter           */
	TCHAR tchBuffer[1000]; /* buffer for expanded string */
	HDC hdc;                 /* device-context handle      */
	
	hdc = GetDC(w);
	sprintf(tchBuffer, "WinProp %d: %s", nProp++, s);
	MessageBox(0, s, "", MB_ICONSTOP | MB_OK);
	ReleaseDC(w, hdc);
	
	return TRUE;
}
#endif

void VLDispose(VLISTPTR L)
{
#ifdef MAC
   if (L->scroll) DisposeControl(L->scroll);
   L->scroll = 0; // JLM 12/10/98
#else
	// in Windows, the VList subwindows are disposed along with the parent window
	SetWRefCon(L->window, 0);
	SetWRefCon(GetParent(L->window), 0);
#endif
}

/////////////////////////////////////////////////////////////////////////////////

static pascal_ifMac Boolean (*userFilter)(DialogPtr, EVENTRECORDPTR, SHORTPTR);
static void (*userSearchText)(DialogPtr, long, CHARPTR);
static Boolean OKflag;

#ifdef IBM
static short sharedListItem, *sharedDialogItem, sharedItemHit;
static long sharedNumItems, sharedChosenItem;
static void (*sharedInit)(DialogPtr, VLISTPTR);
static void (*sharedDraw)(DialogPtr, RECTPTR, long);
static Boolean (*sharedClick)(DialogPtr, VLISTPTR, short, LONGPTR, Boolean);
static VList sharedVList;
#endif

void SaveVListSettings(VListSettings *vSettings)
{
	vSettings->userFilter = userFilter;
	vSettings->userSearchText = userSearchText;
	vSettings->OKflag = OKflag;
#ifdef IBM
	vSettings->sharedListItem = sharedListItem;
	vSettings->sharedDialogItem = sharedDialogItem;
	vSettings->sharedItemHit = sharedItemHit;
	vSettings->sharedNumItems = sharedNumItems;
	vSettings->sharedChosenItem = sharedChosenItem;
	vSettings->sharedInit = sharedInit;
	vSettings->sharedDraw = sharedDraw;
	vSettings->sharedClick = sharedClick;
	vSettings->sharedVList = sharedVList;
#endif
}

void RestoreVListSettings(VListSettings vSettings)
{
	userFilter = vSettings.userFilter;
	userSearchText = vSettings.userSearchText;
	OKflag = vSettings.OKflag;
#ifdef IBM
	sharedListItem = vSettings.sharedListItem;
	sharedDialogItem = vSettings.sharedDialogItem;
	sharedItemHit = vSettings.sharedItemHit;
	sharedNumItems = vSettings.sharedNumItems;
	sharedChosenItem = vSettings.sharedChosenItem;
	sharedInit = vSettings.sharedInit;
	sharedDraw = vSettings.sharedDraw;
	sharedClick = vSettings.sharedClick ;
	sharedVList = vSettings.sharedVList;
#endif
}

pascal_ifMac Boolean SFVLFilter(DialogPtr theDialog, EVENTRECORDPTR theEvent, SHORTPTR itemHit)
{
#ifdef MAC
	VLISTPTR L = (VLISTPTR)GetWRefCon(GetDialogWindow(theDialog));
#else
	VLISTPTR L = &sharedVList;
#endif
	char c, userText[256];
	long newTicks, oldLen, i, dummy;
	static long prevTicks = 0;
	static char key[256] = "";
	
#ifdef MAC
	if (userSearchText && theEvent->what == keyDown && !CommandKeyEvent(theEvent, 0, 0)) {
		c = BitAnd(theEvent->message, charCodeMask);
#else
	if (userSearchText && theEvent->message == WM_CHAR) {
		c = (char)theEvent->wParam;
#endif
		//if (isalnum(c)) {
		if ((c>='0' && c<='9') || (c>='A' && c<='Z') || (c>='a' && c<='z')) {	// for code warrior
			newTicks = GetEventWhen(*theEvent);
			if ((newTicks - prevTicks) > GetDblTime())
				key[0] = 0;
			if (strlen(key) < 10) {
				oldLen = strlen(key);
				key[oldLen] = c;
				key[oldLen + 1] = 0;
				for (i = 0 ; i < L->numItems ; i++) {
					strcpy(userText, key);
					(*userSearchText)(theDialog, i, userText);
					if (strncmpnocase(userText, key, strlen(key)) >= 0) {
						VLSetSelect(i, L);
						VLAutoScroll(L);
						break;
					}
				}
				if (i == L->numItems) { VLSetSelect(i - 1, L); VLAutoScroll(L); }
			}
			prevTicks = newTicks;
		}
	}
	
	if (userFilter && userFilter(theDialog, theEvent, itemHit)) return TRUE;
	
	if (!OKflag)
		MyEnableControl(theDialog, 1, OKflag || VLGetSelect(&dummy, L));
	
#ifdef MAC
	if (VListKeyEvent(theEvent, L)) { *itemHit = 0; return TRUE; }
	if (STDFilter(theDialog, theEvent, itemHit)) return TRUE;
#endif
	
	return FALSE;
}

#ifdef MAC

pascal_ifMac void SelectFromVListUpdate(DialogPtr theDialog, short itemNum)
{
#pragma unused(itemNum)
	
	TextFont(kFontIDGeneva); TextSize(9);
	VLUpdate((VLISTPTR)GetWRefCon(GetDialogWindow(theDialog)));
	TextFont(0); TextSize(12);
}

long SelectFromVListDialog(short dialogID, short listItem, long numItems,
						   void (*init)(DialogPtr, VLISTPTR),
						   void (*searchText)(DialogPtr, long, CHARPTR),
						   pascal_ifMac Boolean (*filter)(DialogPtr, EVENTRECORDPTR, SHORTPTR),
						   void (*draw)(DialogPtr, RECTPTR, long),
						   Boolean (*click)(DialogPtr, VLISTPTR, short, LONGPTR, Boolean),
						   Boolean OKanyTime, short *dialogItem)
{
	Point where = CenteredDialogUpLeft(dialogID);
	GrafPtr oldPort;
	DialogPtr diawind;
	short itemHit;
	long item = -1, err;
	Boolean doubleClick;
	VList L;
	
	*dialogItem = -1;
	
	GetPortGrafPtr(&oldPort);
	if (!(diawind = GetNewDialog(dialogID, nil, (WindowPtr)-1))) return -1;
	MoveWindow(GetDialogWindow(diawind), where.h, where.v, TRUE);
	SetPortDialogPort(diawind);
	
	TextFont(kFontIDGeneva); TextSize(9);
	if (err = VLNew(diawind, listItem, 0, numItems, draw, &L)) goto done;
	TextFont(0); TextSize(12);
	
	SetDialogItemHandle(diawind, 2, (Handle)FrameDefault);
	SetDialogItemHandle(diawind, listItem, (Handle)SelectFromVListUpdate);
	
	if (init) (*init)(diawind, &L);
	
	SetDefaultItemBehavior(diawind);
	ShowWindow(GetDialogWindow(diawind));
	
	userSearchText = searchText;
	userFilter = filter;
	OKflag = OKanyTime;
	
	CmdPeriod();
	
	for ( ; ; ) {
		//ModalDialog((ModalFilterUPP)MakeUPP((ProcPtr)SFVLFilter, uppModalFilterProcInfo), &itemHit);
		ModalDialog(MakeModalFilterUPP(SFVLFilter), &itemHit);
		GlobalToLocal(&lastEvent.where);
		doubleClick = FALSE;
		if (lastEvent.what == mouseDown && itemHit == listItem)
			doubleClick = VLClick(lastEvent.where, &L);
		VLGetSelect(&item, &L);
		
		if ((*click)(diawind, &L, itemHit, &item, doubleClick)) {
			*dialogItem = itemHit;
			break;
		}
	}
	
	done:
		VLDispose(&L);
		SetPortGrafPort(oldPort);
		DisposeDialog(diawind);
		
		return item;
}

#endif

/////////////////////////////////////////////////////////////////////////////////

#ifdef IBM

static Boolean processedByVLWindow;

LRESULT CALLBACK ChildKeyProc(HWND child, UINT message, WPARAM wParam, LPARAM lParam)
{
	BOOL (_stdcall * oldProc)(HWND, UINT, WPARAM, LPARAM);
	char c = 0;
	short itemHit = 0;
	long result;
	EventRecord thisEvent;
	static DWORD lastKeyDownTime;
	
	DefineEvent(&thisEvent, child, message, wParam, lParam);
	thisEvent.time = lastKeyDownTime;
	
	*(long *)&oldProc = GetWindowLong(child, GWL_USERDATA);
	switch (message) {
		case WM_DESTROY:
			SetWindowLong(child, GWL_WNDPROC, *(long *)&oldProc);
			SetWindowLong(child, GWL_USERDATA, 0);
			break;
		case WM_GETDLGCODE:
			if (GetAsyncKeyState(VK_MENU) < 0) break;
			result = (*oldProc)(child, message, wParam, lParam);
			return result | DLGC_WANTCHARS | DLGC_WANTARROWS;
		case WM_CHAR:
			SFVLFilter(GetParent(child), &thisEvent, &itemHit);
			return TRUE;
		case WM_KEYDOWN:
			lastKeyDownTime = GetMessageTime();
			switch (wParam) {
				case VK_UP: c = UPKEY; break;
				case VK_DOWN: c = DOWNKEY; break;
				case VK_PRIOR: c = PAGEUPKEY; break;
				case VK_NEXT: c = PAGEDOWNKEY; break;
				case VK_HOME: c = HOMEKEY; break;
				case VK_END: c = ENDKEY; break;
			}
			if (c) return SendMessage(sharedVList.window, message, wParam, lParam);
			break;
	}
	
	return (*oldProc)(child, message, wParam, lParam);	
}

BOOL CALLBACK SetChildKeyProc(HWND child, long lParam)
{
	long result, oldProc;
	
	result = SendMessage(child, WM_GETDLGCODE, 0, 0);
	if ((result & DLGC_BUTTON) || (result & DLGC_RADIOBUTTON)) {
		oldProc = GetWindowLong(child, GWL_WNDPROC);
		SetWindowLong(child, GWL_USERDATA, oldProc);
		SetWindowLong(child, GWL_WNDPROC, (long)ChildKeyProc);
	}
	
	return TRUE;
}

BOOL CALLBACK SVLWinDlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	char buttonTitle[20], help[20], windowTitle[100], *b;
	long err;
	RECT R;
	Boolean doClick = FALSE;
	EventRecord emptyEvent;
	HCURSOR arrow;
	
	switch (message) {
		case WM_INITDIALOG:
			DefineEvent(&lastEvent, hWnd, message, wParam, lParam); 
			SetPortWindowPort(hWnd);
			CenterDialog(hWnd, 0);
			SetFocus(GetDlgItem(hWnd, 1)); // set focus to OK button by default
			TextFontSize(kFontIDGeneva,9);
			if (err = VLNew(hWnd, sharedListItem, 0, sharedNumItems, sharedDraw,
							&sharedVList)) EndDialog(hWnd, err);
			// TextFont(0); TextSize(12);
			EnumChildWindows(hWnd, (WNDENUMPROC)SetChildKeyProc, 0);
			if (sharedInit) (*sharedInit)(hWnd, &sharedVList);
			SubstituteParamText(hWnd);
			sharedItemHit = 0;
			SFVLFilter(hWnd, &lastEvent, &sharedItemHit);
 			if (!GetFocus())
				SetFocus(GetDlgItem(hWnd, 3)); // set to Cancel in case OK is disabled
			if (arrow = LoadCursor(NULL, IDC_ARROW))
				SetCursor(arrow);
			return 0;
		
		case WM_DESTROY:
			VLDispose(&sharedVList);
			SetPortWindowPort(0);
			return TRUE;
		
		case WM_LBUTTONDBLCLK:
		case WM_LBUTTONDOWN:
		case WM_COMMAND:
		case WM_CHAR:
			DefineEvent(&lastEvent, hWnd, message, wParam, lParam); 
			if (message == WM_LBUTTONDBLCLK || message == WM_LBUTTONDOWN) {
				GetWindowRect(sharedVList.window, &R);
				MapWindowPoints(GetDesktopWindow(), hWnd, (POINT *)&R, 2);
				R.right -= GetSystemMetrics(SM_CYVTHUMB); // don't count click on scroll bar
				if (!PtInRect(&R, lastEvent.pt))
					return FALSE;
				sharedItemHit = sharedListItem;
				processedByVLWindow = TRUE;
			}
			else
				sharedItemHit = (message == WM_CHAR) ? 0 : LOWORD(wParam);
			VLGetSelect(&sharedChosenItem, &sharedVList);
			SFVLFilter(hWnd, &lastEvent, &sharedItemHit);
			if (message == WM_CHAR) break;
			
			if (message == WM_COMMAND && !lParam && !HIWORD(wParam)) { // from a menu item
				if (LOWORD(wParam) == 2) { // "cancel" from ESC key "menu item"
					*sharedDialogItem = 3; // Cancel must always be 3
					EndDialog(hWnd, 0);
				}
				if ((*sharedClick)(hWnd, &sharedVList, sharedItemHit,
								   &sharedChosenItem, FALSE)) { // message == WM_LBUTTONDBLCLK
					*sharedDialogItem = sharedItemHit;
					EndDialog(hWnd, 0);
				}
				return TRUE;
			}
			
			if (message == WM_LBUTTONDBLCLK || message == WM_LBUTTONDOWN)
				doClick = TRUE;
			
			if (message == WM_COMMAND && HIWORD(wParam) == BN_CLICKED && IsWindow((WindowPtr)lParam)) {
				getindstring(help, DLG_STRINGS, HELP_STRING); // "Help..."
				GetWindowText((WindowPtr)lParam, buttonTitle, 19);
				b = buttonTitle;
				if (b[0] == '&') b++;
				if (!strcmp(help, b)) {
					getwtitle(hWnd, windowTitle);
					GetHelp(windowTitle, TRUE);
					return TRUE;
				}
			}
			
			if (message == WM_COMMAND)
				switch (HIWORD(wParam)) {
					case EN_CHANGE:
						if ((WindowPtr)lParam != GetFocus()) break;
					case EN_SETFOCUS:
					case EN_KILLFOCUS:
					case CBN_SELCHANGE:
						lastEvent.pt.x = lastEvent.pt.y = 0;
						// fall through
					case BN_CLICKED:
						doClick = TRUE;
						break;
				}
			
			if (doClick) {
				if ((*sharedClick)(hWnd, &sharedVList, sharedItemHit,
								   &sharedChosenItem, (sharedChosenItem != -1) && message == WM_LBUTTONDBLCLK)) {
					*sharedDialogItem = sharedItemHit;
					EndDialog(hWnd, 0);
				}
				DefineEvent(&emptyEvent, hWnd, 0, 0, 0);
				sharedItemHit = 0; 
				SFVLFilter(hWnd, &emptyEvent, &sharedItemHit);
				return TRUE;
			}
			
			return FALSE;
		
		/*
		case WM_HSCROLL:
		case WM_VSCROLL:
			if ((*sharedClick)(hWnd, &sharedVList, GetDlgCtrlID((HWND)lParam),
							   (short *)&wParam, message == WM_LBUTTONDBLCLK))
				EndDialog(hWnd, 0);
				return TRUE;
		*/
		
		case WM_MEASUREITEM:
		case WM_DRAWITEM:
			// we assume that the parent window will handle these messages and will
			// not call DefWindowProc()
			SendMessage(hMainWnd, message, wParam, lParam);
			return TRUE;
	}
	
	return FALSE;
}

long SelectFromVListDialog(short dialogID, short listItem, long numItems,
						   void (*init)(DialogPtr, VLISTPTR),
						   void (*searchText)(DialogPtr, long, CHARPTR),
						   pascal_ifMac Boolean (*filter)(DialogPtr, EVENTRECORDPTR, SHORTPTR),
						   void (*draw)(DialogPtr, RECTPTR, long),
						   Boolean (*click)(DialogPtr, VLISTPTR, short, LONGPTR, Boolean),
						   Boolean OKanyTime, short *dialogItem)
{
	short *saveDialogItem;
	long result, saveChosenItem = sharedChosenItem;
	GrafPtr oldPort;
	
	GetPortGrafPtr(&oldPort);
	
	*dialogItem = -1;
	sharedChosenItem = -1;

	sharedListItem = listItem;
	sharedNumItems = numItems;
	sharedInit = init;
	userSearchText = searchText;
	userFilter = filter;
	sharedDraw = draw;
	sharedClick = click;
	OKflag = OKanyTime;
	saveDialogItem = sharedDialogItem;
	sharedDialogItem = dialogItem;
	
	result = DialogBox(hInst, MAKEINTRESOURCE(dialogID), FrontWindow(), (DLGPROC)SVLWinDlgProc);
	
	result = sharedChosenItem;
	sharedChosenItem = saveChosenItem;
	sharedDialogItem = saveDialogItem;
	
	SetPortGrafPort(oldPort);
	
	return result; 
}

Boolean VLIBMClick(Point pt, VLISTPTR L)
{
	long recHit;
	GrafPtr oldPort;
	
	if (pt.h < L->viewRect.left || pt.h > L->viewRect.right - GetSystemMetrics(SM_CXVSCROLL)) 
		return FALSE;
	
	GetPortGrafPtr(&oldPort);
	SetPortWindowPort(L->window);
	SetCapture(L->window);
	
	do {
		if (pt.v <= L->viewRect.top) {
			if (L->curRec > 0) {
				VLUNHILITE(L);
				VLScrollText(L, 1);
				L->selectedRec = L->curRec;
				VLHILITE(L);
			}
			else if (L->selectedRec != 0) {
				VLUNHILITE(L);
				L->selectedRec = 0;
				VLHILITE(L);
			}
			GetMouse(&pt);
		}
		
		if (pt.v >= L->viewRect.bottom) {  
			if ((L->curRec + L->viewLines) < L->numItems) {
				VLUNHILITE(L);
				VLScrollText(L, -1);
				L->selectedRec = L->curRec + L->viewLines - 1;
				VLHILITE(L);
			}
			else if (L->selectedRec != (L->numItems - 1)) {
				VLUNHILITE(L);
				L->selectedRec = L->numItems - 1;
				VLHILITE(L);
			}
			GetMouse(&pt);
		}
		
		recHit = (pt.v - L->viewRect.top) / L->lineHeight + L->curRec;
		if (recHit < L->curRec || recHit > L->curRec + L->viewLines - 1) {
			GetMouse(&pt);
			continue;
		}
		
		// Make sure there is actually a record at this position
		if (recHit >= 0 && recHit < L->numItems && L->numItems > 0) {
			if (recHit != L->selectedRec) {
				if (L->selectedRec != -1) VLUNHILITE(L);
				L->selectedRec = recHit;
				VLHILITE(L);
			}
		}
		else {   // don't hilite anything
			if (L->selectedRec != -1) VLUNHILITE(L);
			L->selectedRec = -1;
		}
		GetMouse(&pt);
	} while (GetAsyncKeyState(VK_LBUTTON) < 0);
	
	ReleaseCapture();
	SetPortGrafPort(oldPort);
	
	return FALSE;
}

LRESULT CALLBACK VListWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	HWND hWndParent = GetParent(hWnd);
	HDC hDC;
	VLISTPTR L;
	PAINTSTRUCT ps;
	RECT rect;
	POINT P;
	LPARAM newLP;
	Point thePoint;
	short thePart, itemHit = 0;
	char c = 0;
	EventRecord thisEvent;
	
	DefineEvent(&thisEvent, hWnd, message, wParam, lParam); 
	
	L = (VLISTPTR)GetWRefCon(hWnd);
	thePoint.v = 0;
	thePoint.h = 0;
	
	switch (message) {
		case WM_PAINT:      
			hDC = BeginPaint(hWnd, &ps);
			GetClientRect(hWnd, &rect);
			FrameRect(hDC, &rect, (HBRUSH)GetStockObject(GRAY_BRUSH));
			VLUpdate(L);
			EndPaint(hWnd, &ps);
			
			if (GetFocus() == hWnd) {
				GetWindowRect(hWnd, &rect);
				MapWindowPoints(GetDesktopWindow(), hWndParent, (POINT *)&rect, 2);
				if (rect.left >= 0 && rect.top >= 0) {
					InflateRect(&rect, 1, 1);
					hDC = GetDC(hWndParent);
					FrameRect(hDC, &rect, (HBRUSH)GetStockObject(GRAY_BRUSH));
					InflateRect(&rect, 1, 1);
					FrameRect(hDC, &rect, (HBRUSH)GetStockObject(GRAY_BRUSH));
					ReleaseDC(hWndParent, hDC);
				}
			}
			
			return TRUE;    
		
		case WM_GETDLGCODE:
			return DLGC_WANTCHARS | DLGC_WANTARROWS;
		
		case WM_CHAR:
			SendMessage(hWndParent, message, wParam, lParam);
			SendMessage(hWndParent, WM_COMMAND,
						MAKEWPARAM(GetDlgCtrlID(hWnd), CBN_SELCHANGE), (LPARAM)hWnd);
			break;
		
		case WM_KEYDOWN:
			switch (wParam) {
				case VK_UP: c = UPKEY; break;
				case VK_DOWN: c = DOWNKEY; break;
				case VK_PRIOR: c = PAGEUPKEY; break;
				case VK_NEXT: c = PAGEDOWNKEY; break;
				case VK_HOME: c = HOMEKEY; break;
				case VK_END: c = ENDKEY; break;
			}
			if (c) {
				VListKeyEventHelper(c, L);
				SendMessage(hWndParent, WM_COMMAND,
							MAKEWPARAM(GetDlgCtrlID(hWnd), CBN_SELCHANGE), (LPARAM)hWnd);
				return FALSE;
			}
			break;
		
		case WM_VSCROLL:
			SetFocus(hWnd);
			switch (LOWORD(wParam)) {
				case SB_LINEUP: thePart = kControlUpButtonPart; break;
				case SB_LINEDOWN: thePart = kControlDownButtonPart; break;
				case SB_PAGEUP: thePart = kControlPageUpPart; break;
				case SB_PAGEDOWN: thePart = kControlPageDownPart; break;
				case SB_THUMBPOSITION:
					SetScrollPos(L->scroll, SB_CTL, (int)HIWORD(wParam), TRUE);
					thePart = kControlIndicatorPart;
					break;
				default: return FALSE;
			}
			VLScrollClick(thePart, thePoint, L);
			return FALSE;
		
		case WM_LBUTTONDBLCLK:
		case WM_LBUTTONDOWN:
			SetFocus(hWnd);
			thePoint.h = LOWORD(lParam);
			thePoint.v = HIWORD(lParam);
			VLIBMClick(thePoint, L);
			GetWindowRect(hWnd, &rect);
			MapWindowPoints(GetDesktopWindow(), hWndParent, (POINT *)&rect, 2);
			P.x = LOWORD(lParam);
			P.y = HIWORD(lParam);
			MapWindowPoints(hWnd, hWndParent, &P, 1);
			newLP = ((long)P.y << 16) + P.x;
			processedByVLWindow = FALSE;
			SendMessage(hWndParent, message, wParam, newLP);
			if (processedByVLWindow) return TRUE;
			L->clickTime = message;
			SendMessage(hWndParent, WM_COMMAND,
						MAKEWPARAM(GetDlgCtrlID(hWnd), CBN_SELCHANGE), (LPARAM)hWnd);
			L->clickTime = 0;
			return TRUE;
		
		case WM_SETFOCUS:
		case WM_KILLFOCUS:
			GetWindowRect(hWnd, &rect);
			MapWindowPoints(GetDesktopWindow(), hWndParent, (POINT *)&rect, 2);
			if (rect.left < 0 || rect.top < 0) return TRUE;
			InflateRect(&rect, 1, 1);
			hDC = GetDC(hWndParent);
			FrameRect(hDC, &rect, (HBRUSH)GetStockObject(message == WM_SETFOCUS ?
													GRAY_BRUSH : LTGRAY_BRUSH));
			InflateRect(&rect, 1, 1);
			FrameRect(hDC, &rect, (HBRUSH)GetStockObject(message == WM_SETFOCUS ?
													GRAY_BRUSH : LTGRAY_BRUSH));
			ReleaseDC(hWndParent, hDC);
			return TRUE;
		
		case WM_SETCURSOR:
			if (hWndParent == hMainWnd) {
				InitCursor(); // keep Windows from resetting the arrow cursor within the client rect
				break;
			}
			// fall through
		
		default: return DefWindowProc(hWnd, message, wParam, lParam);
	}
	
	return 0;
}

#endif

