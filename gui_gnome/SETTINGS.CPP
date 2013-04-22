
#include "CROSS.h"
#include "OUtils.h"
#include "GridVel.h"
#include "NetCDFMover.h"
#include "Map3D.h"
#include "GridMap_c.h"

void	GetOWeatherValues (DialogPtr dialog, long item);
void 	SetOWeatherValues (DialogPtr dialog, long item);
void 	OWeatherInit(DialogPtr dialog, VLISTPTR L);
void 	DrawOWeatherItem(DialogPtr dialog, Rect *r, long item);
Boolean OWeatherClick(DialogPtr dialog, VLISTPTR L, short dialogItem, long *listItem,
				Boolean doubleClick);

static CMyList 		*sharedMoverList;
static char 		sharedFileName[256];

///////////////////////////////////////////////////////////////////////////

static DialogPtr currentPreferencesDialog;

void DrawTab(DialogPtr dialog, short itemNum, Boolean focus)
{
	long id = 0;
	Rect r = GetDialogItemBox(dialog, itemNum);
	void FillRectWith3DBackground(Rect r);
	
	if (dialog == currentPreferencesDialog)
		if (itemNum >= M1TAB1 && itemNum <= M1TABMAX) {
			id = PREFERENCESPICTBASE + (itemNum - M1TAB1);
			if (itemNum == settings.preferencesScreen)
				id += 1000;
		}
	
	if (focus)
#ifdef MAC
		;
#else
		id += 50;
#endif

#ifdef MAC
	if (id) DrawResBitmap(id, r, 0);
#else
	r.top = r.bottom - 32;
	FillRectWith3DBackground(r);
	if (id) DrawResBitmap(id, r, SRCAND);
	
	PenNormal();
	MyMoveTo(r.left + (dialog == currentPreferencesDialog ? 90 : 64), r.bottom - 2);
	MyLineTo(r.right, r.bottom - 2);
#endif
}

pascal_ifMac void DrawTabMac(DialogPtr dialog, short itemNum)
{
	DrawTab(dialog, itemNum, FALSE);
}

///////////////////////////////////////////////////////////////////////////

static PopInfoRec prefPopTable[] =
{
	// longitude line space units pop up
	{ M1, nil, D1LONGLINEUNITSPOPUP, 0, pLONGLINEUNITS, 0, 1, FALSE, nil },
	// longitude label space units pop up
	{ M1, nil, D1LONGLABELUNITSPOPUP, 0, pLONGLABELUNITS, 0, 1, FALSE, nil },
	// latitude line space units pop up
	{ M1, nil, D1LATLINEUNITSPOPUP, 0, pLATLINEUNITS, 0, 1, FALSE, nil },
	// latitude label space units pop up
	{ M1, nil, D1LATLABELUNITSPOPUP, 0, pLATLABELUNITS, 0, 1, FALSE, nil }
	// mass pop up
//	{ M1, nil, E1MASSPOPUP, E1MASSLABEL, pMASSUNITS, 0, 1, FALSE, nil },
	// area pop up
//	{ M1, nil, E1AREAPOPUP, E1AREALABEL, pAREAUNITS, 0, 1, FALSE, nil }
	// background color pop up
	//{ M1, nil, M1BACKCOLORPOPUP, M1BACKGROUNDLABEL, pBACKGROUNDCOLORS, 0, 1, FALSE, DrawBackColorItem },
};

float ZeroTo100(float value)
{
	if (value < 0) return 0;
	if (value > 100) return 100;
	
	return value;
}

void ShowPreferenceScreen(DialogPtr dialog, short oldScreen)
{
	char s[32], *p;
	short i;
	
#ifdef MAC
	switch (oldScreen) {
		case M1TAB2:
			//PopErase(dialog, M1BACKCOLORPOPUP);
			break;
		case M1DISPLAY:
			PopErase(dialog, D1LONGLINEUNITSPOPUP);
			PopErase(dialog, D1LONGLABELUNITSPOPUP);
			PopErase(dialog, D1LATLINEUNITSPOPUP);
			PopErase(dialog, D1LATLABELUNITSPOPUP);
			break;
//		case M1UNITS:
//			PopErase(dialog, E1MASSPOPUP);
//			PopErase(dialog, E1AREAPOPUP);
//			break;
	}
#endif
	
	for (i = 1 ; i <= A1NUMITEMS ; i++)
		ShowHideDialogItem(dialog, TAB1BASE + i,
					  settings.preferencesScreen == M1TAB1);
	for (i = 1 ; i <= B1NUMITEMS ; i++)
		ShowHideDialogItem(dialog, TAB2BASE + i,
					  settings.preferencesScreen == M1TAB2);
	for (i = 1 ; i <= C1NUMITEMS ; i++)
		ShowHideDialogItem(dialog, TAB3BASE + i,
					  settings.preferencesScreen == M1TAB3);
	for (i = 1 ; i <= D1NUMITEMS ; i++)
		ShowHideDialogItem(dialog, TAB4BASE + i,
					  settings.preferencesScreen == M1TAB4);
	for (i = 1 ; i <= E1NUMITEMS ; i++)
		ShowHideDialogItem(dialog, TAB5BASE + i,
					  settings.preferencesScreen == M1TAB5);
					  
	ShowHideDialogItem(dialog, M1TAB3, C1NUMITEMS);
	ShowHideDialogItem(dialog, M1TAB4, D1NUMITEMS);
	ShowHideDialogItem(dialog, M1TAB5, E1NUMITEMS);
	
	switch (settings.preferencesScreen) {
		case M1DISPLAY:
			if (settings.showLatLongLines && settings.customGrid)
				MySelectDialogItemText(dialog, D1LONGLINE, 0, 100);
			else
				MySelectDialogItemText(dialog, D1LONGLINE, 0, 0);
			break;
	}
	
#ifdef MAC
	switch (settings.preferencesScreen) {
		//case M1TAB2:
		//	PopDraw(dialog, M1BACKCOLORPOPUP);
		//	break;
		case M1DISPLAY:
			PopDraw(dialog, D1LONGLINEUNITSPOPUP);
			PopDraw(dialog, D1LONGLABELUNITSPOPUP);
			PopDraw(dialog, D1LATLINEUNITSPOPUP);
			PopDraw(dialog, D1LATLABELUNITSPOPUP);
			break;
//		case M1UNITS:
//			PopDraw(dialog, E1MASSPOPUP);
//			PopDraw(dialog, E1AREAPOPUP);
//			break;
	}
#endif
}

void EnableLatLongControls(DialogPtr dialog)
{
	Boolean s = GetButton(dialog, D1SHOWLATLONG),
			 c = GetButton(dialog, D1CUSTOM);
	
	MyEnableControl(dialog, D1DEG, s);
	MyEnableControl(dialog, D1DEGMIN, s);
	MyEnableControl(dialog, D1DEGMINSEC, s);
	MyEnableControl(dialog, D1ABOVE, s);
	MyEnableControl(dialog, D1BELOW, s);
	MyEnableControl(dialog, D1INTERMEDIATELINES, s && !c);
	MyEnableControl(dialog, D1CUSTOM, s);
	EnableTextItem(dialog, D1LONGLINE, s && c);
	EnableTextItem(dialog, D1LONGLABEL, s && c);
	EnableTextItem(dialog, D1LATLINE, s && c);
	EnableTextItem(dialog, D1LATLABEL, s && c);
	prefPopTable[0].bStatic = !(s && c);
	prefPopTable[1].bStatic = !(s && c);
	prefPopTable[2].bStatic = !(s && c);
	prefPopTable[3].bStatic = !(s && c);
	
	PopDraw(dialog, D1LONGLINEUNITSPOPUP);
	PopDraw(dialog, D1LONGLABELUNITSPOPUP);
	PopDraw(dialog, D1LATLINEUNITSPOPUP);
	PopDraw(dialog, D1LATLABELUNITSPOPUP);
	
	if (s && c)
		MySelectDialogItemText(dialog, D1LONGLINE, 0, 100);
	else
		MySelectDialogItemText(dialog, D1MACDUMMYTEXT, 0, 100);
}

long GetGridValue(DialogPtr dialog, short itemNum)
{
	long value;
	
	value = EditText2Float(dialog, itemNum);
	if (value > 59) value = 59;
	if (value < 0) value = 0;
	
	return value;
}

short PDClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short oldScreen;
	long menuID_menuItem;
	long newMode = ADVANCEDMODE;
	Boolean closeFile = false, bOkToChange, disableDST;
	
#pragma unused(lParam)
#pragma unused(data)
	
	switch (itemNum) {
		case M1OK:

			if (GetButton(dialog, M1USERWIZARD)) newMode = NOVICEMODE;
			//else if (GetButton(dialog, M1USERINTERMEDIATE)) newMode = INTERMEDIATEMODE;
			else if (GetButton(dialog, M1USERADVANCED)) newMode = ADVANCEDMODE;

			// if they are switching to modes, we must check with the wizard
			// if there is a wizard file, we must close it because they may have changed stuff
			// if they are switching to advanced mode and have have a wizard file open
			//	we need to tell them they cannot get back
			// without starting all over
			bOkToChange = model->fWizard->OKToChangeWizardMode(model -> GetModelMode (),newMode,&closeFile);
			if(!bOkToChange) break;
			if(closeFile) 
			{
				OSErr err = CloseSaveFile(FALSE,FALSE);// wizard gave them a chance to back out
				if(err == USERCANCEL) break;
			}
			
			// point of no return
			///////////
			settings.backgroundColor = 1; // white JLM ???
				
			settings.showLatLongLines = GetButton(dialog, D1SHOWLATLONG);
			if (GetButton(dialog, D1DEG)) settings.latLongFormat = DEGREES;
			if (GetButton(dialog, D1DEGMIN)) settings.latLongFormat = DEGMIN;
			if (GetButton(dialog, D1DEGMINSEC)) settings.latLongFormat = DMS;
			if (GetButton(dialog, D1ABOVE)) settings.llPosition = LL_ABOVE;
			if (GetButton(dialog, D1BELOW)) settings.llPosition = LL_BELOW;
			settings.showIntermediateLines = GetButton(dialog, D1INTERMEDIATELINES);
			settings.customGrid = GetButton(dialog, D1CUSTOM);
			settings.longLineSpace = GetGridValue(dialog, D1LONGLINE);
			settings.longLabelSpace = GetGridValue(dialog, D1LONGLABEL);
			settings.latLineSpace = GetGridValue(dialog, D1LATLINE);
			settings.latLabelSpace = GetGridValue(dialog, D1LATLABEL);
			settings.longLineUnits = GetPopSelection(dialog, D1LONGLINEUNITSPOPUP) - 1;
			settings.longLabelUnits = GetPopSelection(dialog, D1LONGLABELUNITSPOPUP) - 1;
			settings.latLineUnits = GetPopSelection(dialog, D1LATLINEUNITSPOPUP) - 1;
			settings.latLabelUnits = GetPopSelection(dialog, D1LATLABELUNITSPOPUP) - 1;
			
//			settings.massUnits = GetPopSelection(dialog, E1MASSPOPUP);
//			settings.areaUnits = GetPopSelection(dialog, E1AREAPOPUP);
		
			if (GetButton(dialog, M1USERWIZARD)) model -> SetModelMode (NOVICEMODE);
			//if (GetButton(dialog, M1USERINTERMEDIATE)) model -> SetModelMode (INTERMEDIATEMODE);
			if (GetButton(dialog, M1USERADVANCED)) model -> SetModelMode (ADVANCEDMODE);
			if (GetButton(dialog, M1STARTUPWIZARD)) settings.modelStartMode = NOVICEMODE;
			//if (GetButton(dialog, M1STARTUPINTERMEDIATE)) settings.modelStartMode = INTERMEDIATEMODE;
			if (GetButton(dialog, M1STARTUPADVANCED)) settings.modelStartMode = ADVANCEDMODE;
			
			disableDST = GetButton(dialog, C1DSTCHECKBOX);
			if (disableDST) settings.daylightSavingsTimeFlag = DAYLIGHTSAVINGSOFF;
			else settings.daylightSavingsTimeFlag = DAYLIGHTSAVINGSON;

			return M1OK;
		
		case M1CANCEL: return M1CANCEL;
		
		case M1TAB1:
		case M1TAB2:
		case M1TAB3:
		case M1TAB4:
		case M1TAB5:
			if (settings.preferencesScreen == itemNum)
				break;
			oldScreen = settings.preferencesScreen;
			settings.preferencesScreen = itemNum;
			DrawTab(dialog, oldScreen, FALSE);
			DrawTab(dialog, settings.preferencesScreen, TRUE);
			ShowPreferenceScreen(dialog, oldScreen);
			break;
		
		
		
		case D1SHOWLATLONG:
			ToggleButton(dialog, D1SHOWLATLONG);
			EnableLatLongControls(dialog);
			break;
		
		case D1DEG:
		case D1DEGMIN:
		case D1DEGMINSEC:
			SetButton(dialog, D1DEG, itemNum == D1DEG);
			SetButton(dialog, D1DEGMIN, itemNum == D1DEGMIN);
			SetButton(dialog, D1DEGMINSEC, itemNum == D1DEGMINSEC);
			break;
		
		case D1ABOVE:
		case D1BELOW:
			SetButton(dialog, D1ABOVE, itemNum == D1ABOVE);
			SetButton(dialog, D1BELOW, itemNum == D1BELOW);
			break;
		
		case D1INTERMEDIATELINES:
			ToggleButton(dialog, D1INTERMEDIATELINES);
			break;
		
		case D1CUSTOM:
			ToggleButton(dialog, D1CUSTOM);
			EnableLatLongControls(dialog);
			break;
		
		case D1LONGLINE:
		case D1LONGLABEL:
		case D1LATLINE:
		case D1LATLABEL:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;
		
		//case M1BACKCOLORPOPUP:
		case D1LONGLINEUNITSPOPUP:
		case D1LONGLABELUNITSPOPUP:
		case D1LATLINEUNITSPOPUP:
		case D1LATLABELUNITSPOPUP:
//		case E1MASSPOPUP:
//		case E1AREAPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;

		case M1USERWIZARD:
		//case M1USERINTERMEDIATE:
		case M1USERADVANCED:
			SetButton(dialog, M1USERWIZARD, itemNum == M1USERWIZARD);
			//SetButton(dialog, M1USERINTERMEDIATE, itemNum == M1USERINTERMEDIATE);
			SetButton(dialog, M1USERADVANCED, itemNum == M1USERADVANCED);
			break;
		
		case M1STARTUPWIZARD:
		//case M1STARTUPINTERMEDIATE:
		case M1STARTUPADVANCED:
			SetButton(dialog, M1STARTUPWIZARD, itemNum == M1STARTUPWIZARD);
			//SetButton(dialog, M1STARTUPINTERMEDIATE, itemNum == M1STARTUPINTERMEDIATE);
			SetButton(dialog, M1STARTUPADVANCED, itemNum == M1STARTUPADVANCED);
			break;
		
		case C1DSTCHECKBOX:
			ToggleButton(dialog, C1DSTCHECKBOX);
			break;
		
	}
	
	return 0;
}

OSErr PDInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	currentPreferencesDialog = dialog;
	
	RegisterPopTable(prefPopTable, 4);
	RegisterPopUpDialog(M1, dialog);
	
	SetDialogItemHandle(dialog, M1HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M1BOXLEFT, (Handle)FrameEmbossed);
	SetDialogItemHandle(dialog, M1BOXTOP, (Handle)FrameEmbossed);
	SetDialogItemHandle(dialog, M1BOXRIGHT, (Handle)FrameEmbossed);
	SetDialogItemHandle(dialog, M1BOXBOTTOM, (Handle)FrameEmbossed);
	SetDialogItemHandle(dialog, M1TAB1, (Handle)DrawTabMac);
	SetDialogItemHandle(dialog, M1TAB2, (Handle)DrawTabMac);
	SetDialogItemHandle(dialog, M1TAB3, (Handle)DrawTabMac);
	SetDialogItemHandle(dialog, M1TAB4, (Handle)DrawTabMac);
	SetDialogItemHandle(dialog, M1TAB5, (Handle)DrawTabMac);
	
	
	//SetPopSelection(dialog, M1BACKCOLORPOPUP, settings.backgroundColor);

	if (settings.daylightSavingsTimeFlag == DAYLIGHTSAVINGSON)
		SetButton(dialog, C1DSTCHECKBOX, false);
	else 
		SetButton(dialog, C1DSTCHECKBOX, true);
	
	SetButton(dialog, D1SHOWLATLONG, settings.showLatLongLines);
	SetButton(dialog, D1DEG, settings.latLongFormat == DEGREES);
	SetButton(dialog, D1DEGMIN, settings.latLongFormat == DEGMIN);
	SetButton(dialog, D1DEGMINSEC, settings.latLongFormat == DMS);
	SetButton(dialog, D1ABOVE, settings.llPosition == LL_ABOVE);
	SetButton(dialog, D1BELOW, settings.llPosition == LL_BELOW);
	SetButton(dialog, D1INTERMEDIATELINES, settings.showIntermediateLines);
	SetButton(dialog, D1CUSTOM, settings.customGrid);
	Float2EditText(dialog, D1LONGLINE, settings.longLineSpace, 0);
	Float2EditText(dialog, D1LONGLABEL, settings.longLabelSpace, 0);
	Float2EditText(dialog, D1LATLINE, settings.latLineSpace, 0);
	Float2EditText(dialog, D1LATLABEL, settings.latLabelSpace, 0);
	SetPopSelection(dialog, D1LONGLINEUNITSPOPUP, settings.longLineUnits + 1);
	SetPopSelection(dialog, D1LONGLABELUNITSPOPUP, settings.longLabelUnits + 1);
	SetPopSelection(dialog, D1LATLINEUNITSPOPUP, settings.latLineUnits + 1);
	SetPopSelection(dialog, D1LATLABELUNITSPOPUP, settings.latLabelUnits + 1);
	EnableLatLongControls(dialog);
	SetDialogItemHandle(dialog, D1BOX, (Handle)FrameEmbossed);
	SetDialogItemHandle(dialog, D1BOX2, (Handle)FrameEmbossed);
	
	long theMode = model -> GetModelMode ();
	//SetButton(dialog, M1USERWIZARD, theMode == NOVICEMODE);
	//SetButton(dialog, M1USERINTERMEDIATE, theMode == INTERMEDIATEMODE);
	SetButton(dialog, M1USERWIZARD, theMode == NOVICEMODE || theMode == INTERMEDIATEMODE);
	SetButton(dialog, M1USERADVANCED, theMode == ADVANCEDMODE);
	//SetButton(dialog, M1STARTUPWIZARD, settings.modelStartMode == NOVICEMODE);
	//SetButton(dialog, M1STARTUPINTERMEDIATE, settings.modelStartMode == INTERMEDIATEMODE);
	SetButton(dialog, M1STARTUPWIZARD, settings.modelStartMode == NOVICEMODE || settings.modelStartMode == INTERMEDIATEMODE);
	SetButton(dialog, M1STARTUPADVANCED, settings.modelStartMode == ADVANCEDMODE);

//	SetPopSelection(dialog, E1MASSPOPUP, settings.massUnits);
//	SetPopSelection(dialog, E1AREAPOPUP, settings.areaUnits);
	
	if (settings.preferencesScreen < M1TAB1 ||
		settings.preferencesScreen > M1TABMAX)
		settings.preferencesScreen = M1TAB1;
	
	ShowPreferenceScreen(dialog, 0);
	
	return 0;
}


///////////////////////////////////////////////////////////////////////////

OSErr PreferencesDialog()
{
	short item;
	Settings oldSettings = settings;
	
	currentPreferencesDialog = 0;
	
	item = MyModalDialog(M1, mapWindow, 0, PDInit, PDClick);
	
	if (item != M1OK)
		settings = oldSettings;
	else model->NewDirtNotification();//JLM	
	return item == -1 ? -1 : 0;
}

///////////////////////////////////////////////////////////////////////////

void M17Init(DialogPtr dialog, VLISTPTR L)
{
	long i;
	TCATSMover *mover;
	
	for (i = 0 ; i < L->numItems ; i++) {
		sharedMoverList->GetListItem((Ptr)&mover, i);
		if (!strcmp(mover->className, sharedFileName)) {
			VLSetSelect(i, L);
			VLAutoScroll(L);
			break;
		}
	}
}

Boolean M17Click(DialogPtr dialog, VLISTPTR L, short dialogItem, long *listItem,
				Boolean doubleClick)
{
	TCATSMover *mover;
	
	switch (dialogItem) {
		case M17OK:
			sharedMoverList->GetListItem((Ptr)&mover, *listItem);
			strcpy(sharedFileName, mover->className);
			
			return TRUE;
		case M17CANCEL: return TRUE;
	}
	
	return FALSE;
}

void M17Draw (DialogPtr dialog, Rect *r, long item)
{
	char s[256];
	TCATSMover *mover;
	
	TextFontSize(kFontIDGeneva,LISTTEXTSIZE);
	
	sharedMoverList->GetListItem((Ptr)&mover, item);
	strcpy(s, mover->className);
	
	drawstring(s);
	
	TextFontSize(0,12);
}

short ChooseWindMoverDialog(char* classNameOfSelectedGrid) //JLM added param
{
	short dialogItem = 0;
	long i, j, m, n;
	TMap *map;
	TMover *mover;
	OSErr err = 0;

	strcpy(sharedFileName,classNameOfSelectedGrid);//JLM 8/14/98

	sharedMoverList = new CMyList(sizeof(TMover *));
	if (!sharedMoverList)
		{ TechError("ChooseWindMoverDialog()", "new CMyList()", 0); return -1; }
	if (err = sharedMoverList->IList())
		{ TechError("ChooseWindMoverDialog()", "IList()", 0); return -1; }

	for (j = 0, m = model->mapList->GetItemCount() ; j < m ; j++) {
		model->mapList->GetListItem((Ptr)&map, j);

		for (i = 0, n = map->moverList->GetItemCount() ; i < n ; i++) {
			map->moverList->GetListItem((Ptr)&mover, i);
			if (mover->GetClassID() != TYPE_WINDMOVER) continue;
			if (err = sharedMoverList->AppendItem((Ptr)&mover))
				{ TechError("ChooseWindMoverDialog()", "AppendItem()", err); return -1; }
		}
	}

	// append any wind movers belonging to the universal map, STH
	map = model -> uMap;
	for (i = 0, n = map->moverList->GetItemCount() ; i < n ; i++) {
		map->moverList->GetListItem((Ptr)&mover, i);
		if (mover->GetClassID() != TYPE_WINDMOVER) continue;
		if (err = sharedMoverList->AppendItem((Ptr)&mover))
			{ TechError("ChooseWindMoverDialog()", "AppendItem()", err); return -1; }
	}

	if (sharedMoverList->GetItemCount() == 0)
	{
		printError("There are no wind movers loaded.");
		dialogItem = M17CANCEL;
	}
	else
		SelectFromVListDialog(M17, M17LIST, sharedMoverList->GetItemCount(),
							  M17Init, 0, 0, M17Draw, M17Click,
							  FALSE, &dialogItem);

	sharedMoverList->Dispose();
	delete sharedMoverList;
	strcpy(classNameOfSelectedGrid,sharedFileName);//JLM 8/14/98 return name through parameter

	return dialogItem;
}

short ChooseOtherGridDialog(TCATSMover	*connectingCMover,char* classNameOfSelectedGrid) //JLM added param
{
	short dialogItem = 0;
	long i, j, m, n;
	TMap *map;
	TMover *mover;
	OSErr err = 0;
	
	strcpy(sharedFileName,classNameOfSelectedGrid);//JLM 8/14/98

	sharedMoverList = new CMyList(sizeof(TMover *));
	if (!sharedMoverList)
		{ TechError("ChooseOtherGridDialog()", "new CMyList()", 0); return -1; }
	if (err = sharedMoverList->IList())
		{ TechError("ChooseOtherGridDialog()", "IList()", 0); return -1; }
	
	for (j = 0, m = model->mapList->GetItemCount() ; j < m ; j++) {
		model->mapList->GetListItem((Ptr)&map, j);
		
		for (i = 0, n = map->moverList->GetItemCount() ; i < n ; i++) {
			map->moverList->GetListItem((Ptr)&mover, i);
			if (mover->GetClassID() != TYPE_CATSMOVER) continue;
			if (!strcmp(mover->className, connectingCMover->className)) continue;// i.e. don't let it chooose one with the same name 
			if (err = sharedMoverList->AppendItem((Ptr)&mover))
				{ TechError("ChooseOtherGridDialog()", "AppendItem()", err); return -1; }
		}
	}
	
	if (sharedMoverList->GetItemCount() == 0)
	{
		printError("There are no other current grids loaded.");
		dialogItem = M17CANCEL;
	}
	else
		SelectFromVListDialog(M17, M17LIST, sharedMoverList->GetItemCount(),
							  M17Init, 0, 0, M17Draw, M17Click,
							  FALSE, &dialogItem);
	
	sharedMoverList->Dispose();
	delete sharedMoverList;
	strcpy(classNameOfSelectedGrid,sharedFileName);//JLM 8/14/98 return name through parameter
	
	return dialogItem;
}
/**************************************************************************************************/
OSErr AddMapsDialog()
{
	char 		path[256], nameStr [256], shortFileName[256], tempStr[256];
	OSErr		err = noErr;
	long 		n;
	Point 		where = CenteredDialogUpLeft(M38b);
	TVectorMap	*vMap;
	TOSSMMap 	*oMap;
	OSType 	typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 	reply;
	WorldRect	theRect = emptyWorldRect;
	short 		gridType;
	Boolean		isESI = false;
	float arrowDepth = 0;

#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, M38b, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);
		strcpy(tempStr,path);
		SplitPathFile(tempStr,shortFileName);
#else
	sfpgetfile(&where, "",
			   (FileFilterUPP)0,
			   -1, typeList,
			   (DlgHookUPP)0,
			   &reply, M38b,
			   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	if (!reply.good) return USERCANCEL;

	my_p2cstr(reply.fName);
	#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
		strcpy(shortFileName,(char*) reply.fName);
	#else
		strcpy(path, reply.fName);
		strcpy(tempStr,path);
		SplitPathFile(tempStr,shortFileName);
	#endif
#endif	
	if (IsVectorMap (path, &isESI))
	{
		if (isESI) {printNote("File is ESI segments not map polygons");err=-1; return err;}
		strcpy (nameStr, "Vector Map: ");
		strcat (nameStr, shortFileName);
	
		vMap = (TVectorMap*) new TVectorMap (nameStr, theRect);
		if (!vMap)
			{ TechError("AddMapsDialog()", "new TVectorMap()", 0); return -1; }

		if (err = vMap -> InitMap(path)) { delete vMap; return err; }
	
		if (err = model->AddMap(vMap, 0))
		{
			vMap -> Dispose ();
			delete vMap;
			err = -1;
		}
	}
	else if (IsGridMap (path))
	{
		strcpy (nameStr, "Grid Map: ");
		strcat (nameStr, shortFileName);

		oMap = new TOSSMMap(nameStr, voidWorldRect);
		if (!oMap)
			{ TechError("AddMapsDialog()", "new TOSSMMap()", 0); return -1; }
	
		if (err = ((TOSSMMap*) oMap) -> InitMap(path)) { delete oMap; return err; }
	
		if (err = model->AddMap(oMap, 0))
		{
			oMap -> Dispose ();
			delete oMap;
			err = -1;
		}
	}
	else if (IsPtCurFile (path))
	{
		TMap *newMap = 0;
		TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,"ptcurfile",&newMap);	// already have path
		
		if (newMover)
		{
			PtCurMover *ptCurMover = dynamic_cast<PtCurMover*>(newMover);
			err = ptCurMover -> SettingsDialog();
			if(err)	
			{ 
				newMover->Dispose(); delete newMover; newMover = 0;
				if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} 
			}
	
			if(newMover && !err)
			{
				Boolean timeFileChanged = false;
				if (!newMap) 
				{
					err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
				}
				else
				{
					err = model -> AddMap(newMap, 0);
					if (!err) err = AddMoverToMap(newMap, timeFileChanged, newMover);
					//if(!err) err = ((PtCurMap*)newMap)->MakeBitmaps();
					if (!err) newMover->SetMoverMap(newMap);
					if (model->ThereIsA3DMover(&arrowDepth)) InitAnalysisMenu();	// want to have it come and go?
					else 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
				}
			}
		}
	}
	else if (IsNetCDFFile (path, &gridType))
	{
		TMap *newMap = 0;
		TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,"NetCDFfile",&newMap);	// already have path
		
		if (newMover && gridType!=REGULAR && gridType!=REGULAR_SWAFS)	// should probably get rid of the SWAFS grid type and generalize a regular navy
		{
			NetCDFMover *netCDFMover = dynamic_cast<NetCDFMover *>(newMover);
			err = netCDFMover -> SettingsDialog();
			if(err)	
			{ 
				newMover->Dispose(); delete newMover; newMover = 0;
				if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} 
			}
	
			if(newMover && !err)
			{
				Boolean timeFileChanged = false;
				if (!newMap) 
				{
					err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
				}
				else
				{
					err = model -> AddMap(newMap, 0);
					if (!err) err = AddMoverToMap(newMap, timeFileChanged, newMover);
					// if (!err) err = ((PtCurMap*)newMap)->MakeBitmaps();
					if (!err) newMover->SetMoverMap(newMap);
					if (model->ThereIsA3DMover(&arrowDepth)) InitAnalysisMenu();	// want to have it come and go?
					else 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
				}
			}
		}
		else
		{
			err = true;
			sprintf(tempStr,"File %s is a current file and should be input as a universal mover.",shortFileName);
			printNote(tempStr);
		}
	}
	else if (IsTriCurFile (path))
	{
		TMap *newMap = 0;
		TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,"TriCurFile",&newMap);	// already have path
		
		if (newMover)
		{
			TriCurMover *triCurMover = dynamic_cast<TriCurMover *>(newMover);
			err = triCurMover -> SettingsDialog();
			if(err)	
			{ 
				newMover->Dispose(); delete newMover; newMover = 0;
				if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} 
			}
	
			if(newMover && !err)
			{
				Boolean timeFileChanged = false;
				if (!newMap) 
				{
					err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
				}
				else
				{
					err = model -> AddMap(newMap, 0);
					if (err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					err = AddMoverToMap(newMap, timeFileChanged, newMover);
					if(err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					/*err = ((PtCurMap*)newMap)->MakeBitmaps();
					if(err) 
					{
						newMap->Dispose(); delete newMap; newMap = 0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}*/
					newMover->SetMoverMap(newMap);
					if (model->ThereIsA3DMover(&arrowDepth)) InitAnalysisMenu();	// want to have it come and go?
				}
			}
		}
		else
		{
			err = true;
			sprintf(tempStr,"File %s is a current file and should be input as a universal mover.",shortFileName);
			printNote(tempStr);
		}
	}
	else if (IsCATS3DFile (path))
	{
		TMap *newMap = 0;
		char s[256],fileName[256];
		TCurrentMover *newMover = 0;
		Boolean	timeFileChanged = false;

		strcpy(s,path);
		SplitPathFile (s, fileName);
		strcat (nameStr, fileName);
		newMover = CreateAndInitCurrentsMover (model->uMap,false,path,fileName,&newMap);	// already have path
		
		if (newMover)
		{
			TCATSMover3D *catsMover3D = dynamic_cast<TCATSMover3D *>(newMover);
			err = CATSSettingsDialog (dynamic_cast<TCATSMover *>(newMover), model->uMap, &timeFileChanged);
			if(err)	
			{ 
				newMover->Dispose(); delete newMover; newMover = 0;
				if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} 
			}
	
			if(newMover && !err)
			{
				Boolean timeFileChanged = false;
				if (!newMap) 
				{
					err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
				}
				else
				{
					err = model -> AddMap(newMap, 0);
					if (err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					err = AddMoverToMap(newMap, timeFileChanged, newMover);
					if(err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					/*err = ((PtCurMap*)newMap)->MakeBitmaps();
					if(err) 
					{
						newMap->Dispose(); delete newMap; newMap = 0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}*/
					newMover->SetMoverMap(newMap);
					if (model->ThereIsA3DMover(&arrowDepth)) InitAnalysisMenu();	// want to have it come and go?
				}
			}
		}
	}
	else
	{
		err = true;
		sprintf(tempStr,"File %s is not a recognizable map file.",shortFileName);
		printError(tempStr);
	}

	if (!err)
	{
		model->NewDirtNotification();
	}

	return err;
}
OSErr AddMapsDialog2()
{
	char 		path[256], nameStr [256], shortFileName[256], tempStr[256];
	OSErr		err = noErr;
	long 		n;
	Point 		where = CenteredDialogUpLeft(M38b);
	TVectorMap	*vMap;
	TOSSMMap 	*oMap;
	Map3D 		*gMap;
	GridMap_c *gridMap = 0;
	OSType 	typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 	reply;
	WorldRect	theRect = emptyWorldRect;
	short 		gridType;
	Boolean		isESI = false;
	float arrowDepth = 0;

#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, M38b, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);
		strcpy(tempStr,path);
		SplitPathFile(tempStr,shortFileName);
#else
	sfpgetfile(&where, "",
			   (FileFilterUPP)0,
			   -1, typeList,
			   (DlgHookUPP)0,
			   &reply, M38b,
			   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	if (!reply.good) return USERCANCEL;

	my_p2cstr(reply.fName);
	#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
		strcpy(shortFileName,(char*) reply.fName);
	#else
		strcpy(path, reply.fName);
		strcpy(tempStr,path);
		SplitPathFile(tempStr,shortFileName);
	#endif
#endif	
	if (IsVectorMap (path, &isESI))
	{
		if (isESI) {printNote("File is ESI segments not map polygons");err=-1; return err;}
		strcpy (nameStr, "Vector Map: ");
		strcat (nameStr, shortFileName);
	
		vMap = (TVectorMap*) new TVectorMap (nameStr, theRect);
		if (!vMap)
			{ TechError("AddMapsDialog()", "new TVectorMap()", 0); return -1; }

		if (err = vMap -> InitMap(path)) { delete vMap; return err; }
	
		if (err = model->AddMap(vMap, 0))
		{
			vMap -> Dispose ();
			delete vMap;
			err = -1;
		}
	}
	else if (IsGridMap (path))
	{
		strcpy (nameStr, "Grid Map: ");
		strcat (nameStr, shortFileName);

		oMap = new TOSSMMap(nameStr, voidWorldRect);
		if (!oMap)
			{ TechError("AddMapsDialog()", "new TOSSMMap()", 0); return -1; }
	
		if (err = ((TOSSMMap*) oMap) -> InitMap(path)) { delete oMap; return err; }
	
		if (err = model->AddMap(oMap, 0))
		{
			oMap -> Dispose ();
			delete oMap;
			err = -1;
		}
	}
	else if (IsPtCurFile (path))
	{
		TMap *newMap = 0;
		TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,"ptcurfile",&newMap);	// already have path
		
		if (newMover)
		{
			PtCurMover *ptCurMover = dynamic_cast<PtCurMover*>(newMover);
			err = ptCurMover -> SettingsDialog();
			if(err)	
			{ 
				newMover->Dispose(); delete newMover; newMover = 0;
				if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} 
			}
	
			if(newMover && !err)
			{
				Boolean timeFileChanged = false;
				if (!newMap) 
				{
					err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
				}
				else
				{
					err = model -> AddMap(newMap, 0);
					if (!err) err = AddMoverToMap(newMap, timeFileChanged, newMover);
					//if(!err) err = ((PtCurMap*)newMap)->MakeBitmaps();
					if (!err) newMover->SetMoverMap(newMap);
					if (model->ThereIsA3DMover(&arrowDepth)) InitAnalysisMenu();	// want to have it come and go?
					else 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
				}
			}
		}
	}
	else if (IsNetCDFFile (path, &gridType))
	{
		TMap *newMap = 0;
		
		char s[256],fileName[256];
		WorldRect bounds = theWorld;

		strcpy(s,path);
		SplitPathFile (s, fileName);
		strcat (nameStr, fileName);
		gMap = CreateAndInitMap3D (path,bounds);	// need to fix bounds afterwards
		
		if (gMap && gridType!=REGULAR && gridType!=REGULAR_SWAFS)
		{
			FLOATH depthPtsH = 0;
			WORLDPOINTFH vertexPtsH=0;
			long numRows=0, numCols=0, numNodes=0, numTri=0, numBoundaryPts=0;
			char errmsg[256];
			gridMap = new GridMap_c();
			if (gridType == CURVILINEAR)
			{
				DOUBLEH maskH = 0;
				//err = gMap->GetPointsAndMask(path,&maskH,&vertexPtsH,&depthPtsH,&numRows, &numCols);	//Text read	
				//if (!err) err = gMap->SetUpCurvilinearGrid(maskH,numRows,numCols,vertexPtsH,depthPtsH,errmsg);	//Reorder points
				err = gridMap->GetPointsAndMask(path,&maskH,&vertexPtsH,&depthPtsH,&numRows, &numCols);	//Text read	
				if (!err) err = gridMap->SetUpCurvilinearGrid(maskH,numRows,numCols,vertexPtsH,depthPtsH,errmsg);	//Reorder points
				if(maskH) {DisposeHandle((Handle)maskH); maskH = 0;}
			}
			else if (gridType == TRIANGULAR)
			{	// check if topology is included
				LONGPTR bndry_indices=0, bndry_nums=0, bndry_type=0, tri_verts=0, tri_neighbors=0;
				//err = gMap->GetPointsAndBoundary(path,&vertexPtsH,&depthPtsH, &numNodes, &bndry_indices, &bndry_nums, &bndry_type, &numBoundaryPts, &tri_verts, &tri_neighbors, &numTri);	//Text read	
				err = gridMap->GetPointsAndBoundary(path,&vertexPtsH,&depthPtsH, &numNodes, &bndry_indices, &bndry_nums, &bndry_type, &numBoundaryPts, &tri_verts, &tri_neighbors, &numTri);	//Text read	
				if (!err) 
				{	// separate points and boundary
					if (numTri == 0)
						//err = gMap->SetUpTriangleGrid(numNodes,numTri,vertexPtsH,depthPtsH, bndry_indices, bndry_nums, bndry_type, numBoundaryPts);	//Reorder points
						err = gridMap->SetUpTriangleGrid(numNodes,numTri,vertexPtsH,depthPtsH, bndry_indices, bndry_nums, bndry_type, numBoundaryPts);	//Reorder points
					else
						//err = gMap->SetUpTriangleGrid2(numNodes,numTri,vertexPtsH,depthPtsH, bndry_indices, bndry_nums, bndry_type, numBoundaryPts,tri_verts, tri_neighbors);	//Reorder points
						err = gridMap->SetUpTriangleGrid2(numNodes,numTri,vertexPtsH,depthPtsH, bndry_indices, bndry_nums, bndry_type, numBoundaryPts,tri_verts, tri_neighbors);	//Reorder points
				
				}
				if (bndry_indices) delete [] bndry_indices;
				if (bndry_nums) delete [] bndry_nums;
				if (bndry_type) delete [] bndry_type;
				if (tri_verts) delete [] tri_verts;
				if (tri_neighbors) delete [] tri_neighbors;
			}
			if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;}
			if(depthPtsH) {DisposeHandle((Handle)depthPtsH); depthPtsH = 0;}
			if( !err) 
			{
				gMap->SetBoundarySegs(gridMap->GetBoundarySegs());
				gMap->SetWaterBoundaries(gridMap->GetWaterBoundaries());
				gMap->SetBoundaryPoints(gridMap->GetBoundaryPoints());
				gMap->SetGrid(gridMap->GetGrid());
				gMap->SetMapBounds(gridMap->GetMapBounds());
				err = gMap->MakeBitmaps();
			}
			if (!err) err = model->AddMap(gMap, 0);
			if (err)
			{
				gMap -> Dispose ();
				delete gMap;
				err = -1;
			}
		}
		else
		{
			err = true;
			sprintf(tempStr,"File %s is a current file and should be input as a universal mover.",shortFileName);
			printNote(tempStr);
		}
	}
	else if (IsTriCurFile (path))
	{
		TMap *newMap = 0;
		TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,"TriCurFile",&newMap);	// already have path
		
		if (newMover)
		{
			TriCurMover *triCurMover = dynamic_cast<TriCurMover *>(newMover);
			err = triCurMover -> SettingsDialog();
			if(err)	
			{ 
				newMover->Dispose(); delete newMover; newMover = 0;
				if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} 
			}
	
			if(newMover && !err)
			{
				Boolean timeFileChanged = false;
				if (!newMap) 
				{
					err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
				}
				else
				{
					err = model -> AddMap(newMap, 0);
					if (err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					err = AddMoverToMap(newMap, timeFileChanged, newMover);
					if(err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					/*err = ((PtCurMap*)newMap)->MakeBitmaps();
					if(err) 
					{
						newMap->Dispose(); delete newMap; newMap = 0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}*/
					newMover->SetMoverMap(newMap);
					if (model->ThereIsA3DMover(&arrowDepth)) InitAnalysisMenu();	// want to have it come and go?
				}
			}
		}
		else
		{
			err = true;
			sprintf(tempStr,"File %s is a current file and should be input as a universal mover.",shortFileName);
			printNote(tempStr);
		}
	}
	else if (IsCATS3DFile (path))	// for any CATS?
	{
		char s[256],fileName[256];
		WorldRect bounds = theWorld;

		strcpy(s,path);
		SplitPathFile (s, fileName);
		strcat (nameStr, fileName);
		gMap = CreateAndInitMap3D (path,bounds);	// need to fix bounds afterwards
		
		if (gMap)
		{
			gridMap = new GridMap_c();
			if (!gridMap) {err = gMap->ReadCATSMap(path);/*err = -1;*/}
			else
				//err = gMap->ReadCATSMap(path);
				err = gridMap->ReadCATSMap(path);	
			if(!err) 
			{	
				if (gridMap)
				{
					gMap->SetBoundarySegs(gridMap->GetBoundarySegs());
					gMap->SetWaterBoundaries(gridMap->GetWaterBoundaries());
					gMap->SetBoundaryPoints(gridMap->GetBoundaryPoints());
					gMap->SetGrid(gridMap->GetGrid());
					gMap->SetMapBounds(gridMap->GetMapBounds());
				}
				err = gMap->MakeBitmaps();
			}

			if (err = model->AddMap(gMap, 0))
			{
				gMap -> Dispose ();
				delete gMap;
				err = -1;
			}
		}
	}
	else
	{
		WorldRect bounds = theWorld;
		gMap = CreateAndInitMap3D (path,bounds);	// need to fix bounds afterwards
		if (gMap)
		{
			gridMap = new GridMap_c();
			if (!gridMap) {err = gMap -> ReadTopology(path);/*err = -1;*/}
			else 
				//err = gMap -> ReadTopology(path);
				err = gridMap -> ReadTopology(path);
			if(!err) 
			{
				if (gridMap)
				{
					gMap->SetBoundarySegs(gridMap->GetBoundarySegs());
					gMap->SetWaterBoundaries(gridMap->GetWaterBoundaries());
					gMap->SetBoundaryPoints(gridMap->GetBoundaryPoints());
					gMap->SetGrid(gridMap->GetGrid());
					gMap->SetMapBounds(gridMap->GetMapBounds());
				}
				err = gMap->MakeBitmaps();
			}
			if (err = model->AddMap(gMap, 0))
			{
				gMap -> Dispose ();
				delete gMap;
				err = -1;
			}
		}
		//err = true;
		if (err)
		{sprintf(tempStr,"File %s is not a recognizable map file.",shortFileName);
		printError(tempStr);}
	}

	if (!err)
	{
		model->NewDirtNotification();
	}

	return err;
}
