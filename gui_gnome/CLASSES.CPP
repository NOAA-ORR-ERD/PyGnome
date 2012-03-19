
#include "CROSS.h"
#include "MyRandom.h"
#include "GridVel.h"
#include "MapUtils.h"
#include "EditWindsDialog.h"
#include "OUtils.h"				
#include "Wizard.h"	
#include "Contdlg.h"
#include "GridCurMover.h"
#include "NetCDFMover.h"
#include "TideCurCycleMover.h"


#ifdef MAC
#ifdef MPW
#pragma SEGMENT CLASSES
#endif
#endif

OSErr M21Init(DialogPtr dialog, VOIDPTR data);
short M21Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data);
Boolean OilDecay (OilComponent *component, Seconds time);


///////////////////////////////////////////////////////////////////////////

static char readWriteProc[256];

void StartReadWriteSequence(char *procName)
{
	strcpy(readWriteProc, procName);
}

OSErr WriteValue(BFPB *bfpb, char *data, short length, Boolean swap)
{
	char s[500];
	long count = length;
	OSErr err = 0;

	memcpy(s, data, length);

#ifdef SWAP_BINARY
//#ifdef IBM
	if (swap) SwapN(s, length);
#endif

	if (err = FSWriteBuf(bfpb, &count, s))
		{ TechError(readWriteProc, "FSWriteBuf()", err); return -1; }

	return 0;
}

///////// {

static BFPB gTrickBFPBRec;
//static BFPB *gTrickBFPB= &gTrickBFPBRec;
// JLM 8/1/98  modify ReadValue to read from Handle for Wizard

BFPB* SetReadValueBFPBFTrick(CHARH h)
{
	// set handle to 
	if(!h)  {printError("Programmer error: SetReadValueBFPBFTrick called with nil");return nil;}
	memset(&gTrickBFPBRec,0,sizeof(gTrickBFPBRec));
	gTrickBFPBRec.buf = h;
	gTrickBFPBRec.fileLength = _GetHandleSize((Handle)h);
	gTrickBFPBRec.index = 0;// start of "file"
	return &gTrickBFPBRec;
}


void ClearReadValueBFPBFTrick(void)
{
	memset(&gTrickBFPBRec,0,sizeof(gTrickBFPBRec));
	// note: we do not dispose of the handle
	// the calling routine owns the handle
}


/////////////}

OSErr ReadValue(BFPB *bfpb, char *data, short length, Boolean swap)
{
	#define kMaxReadValueLen  512
	char s[kMaxReadValueLen];
	long count = length;
	OSErr err = 0;
	
	if(length > kMaxReadValueLen)
		{ printError("Programmer error: ReadValue called for more than kMaxReadValueLen chars"); return -1; }

	if(bfpb == &gTrickBFPBRec) //JLM
	{	// read from Handle
		long i,charsLeftToRead = gTrickBFPBRec.fileLength - gTrickBFPBRec.index; 
		if(count > charsLeftToRead)
		{
			err = eofErr;
			count = charsLeftToRead;
			for(i = count ; i< length;i++)
				s[i] = 0;// fill end with null terminators
		}
		_BlockMove(DEREFH(gTrickBFPBRec.buf) + gTrickBFPBRec.index, s, count);
		gTrickBFPBRec.index+=length;
	}
	else
	{	// original code, read from file
		if (err = FSReadBuf(bfpb, &count, s))
			{ TechError(readWriteProc, "ReadValue::FSReadBuf()", err); return err; }
	}
	
#ifdef SWAP_BINARY
//#ifdef IBM
	if (swap) SwapN(s, length);
#endif

	memcpy(data, s, length);

	return err;
}



static PopInfoRec M25PopTable[] = {
		{ M25, nil, M25TOPLATDIR, 0, pNORTHSOUTH1, 0, 1, FALSE, nil },
		{ M25, nil, M25LEFTLONGDIR, 0, pEASTWEST1, 0, 1, FALSE, nil },
		{ M25, nil, M25BOTTOMLATDIR, 0, pNORTHSOUTH2, 0, 1, FALSE, nil },
		{ M25, nil, M25RIGHTLONGDIR, 0, pEASTWEST2, 0, 1, FALSE, nil }
	};


void ShowHideBounds(DialogPtr dialog)
{
	Boolean show  = GetButton (dialog, M25WANTEDCHANGEBOUNDS); 

	SwitchLLFormatHelper(dialog, M25TOPLATDEGREES, M25DEGREES,show);
	SwitchLLFormatHelper(dialog, M25BOTTOMLATDEGREES, M25DEGREES,show); 
	
	ShowHideDialogItem(dialog, M25DEGREES, show); 
	ShowHideDialogItem(dialog, M25DEGMIN, show); 
	ShowHideDialogItem(dialog, M25DMS, show); 

	ShowHideDialogItem(dialog, M25TOPLATLABEL, show); 
	ShowHideDialogItem(dialog, M25LEFTLONGLABEL, show); 
	ShowHideDialogItem(dialog, M25BOTTOMLATLABEL, show); 
	ShowHideDialogItem(dialog, M25RIGHTLONGLABEL, show); 
	
}

TMap *gDialogTMap = 0;

OSErr M25Init(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	char name[256];
	WorldPoint wp;
	TVectorMap *dialogTVectorMap = dynamic_cast<TVectorMap *>(gDialogTMap);
	
	SetDialogItemHandle(dialog, M25HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M25FROST1, (Handle) FrameEmbossed);

	RegisterPopTable (M25PopTable, sizeof (M25PopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (M25, dialog);
	
	dialogTVectorMap -> GetMapName(name);
	mysetitext(dialog, M25NAME, name);
	
	Float2EditText(dialog,M25REFLOATTIME, dialogTVectorMap -> fRefloatHalfLifeInHrs , 4);
	
	SetButton (dialog, M25WANTEDCHANGEBOUNDS, dialogTVectorMap -> fUseExtendedBounds); // !EquaLWRect()

	//SetButton (dialog, M25REPLACEMAP, false); 
	
	wp.pLat = dialogTVectorMap -> fExtendedMapBounds.hiLat;
	wp.pLong = dialogTVectorMap -> fExtendedMapBounds.loLong;
	LL2EditTexts (dialog, M25TOPLATDEGREES, &wp);
	
	wp.pLat = dialogTVectorMap -> fExtendedMapBounds.loLat;
	wp.pLong = dialogTVectorMap -> fExtendedMapBounds.hiLong;
	LL2EditTexts (dialog, M25BOTTOMLATDEGREES, &wp);

	//if (dialogTVectorMap->HaveMapBoundsLayer()) 	// don't allow to change bounds with Map Bounds layer
	if (dialogTVectorMap->HaveMapBoundsLayer() && dialogTVectorMap->HaveLandWaterLayer()) 	// don't allow to change bounds with Map Bounds layer
	{
		ShowHideDialogItem(dialog,M25WANTEDCHANGEBOUNDS,false);
		//ShowHideDialogItem(dialog,M25FROST1,false);
	}
	if (dialogTVectorMap->HaveMapBoundsLayer() && !dialogTVectorMap->HaveLandWaterLayer()) 	// don't allow to change bounds with Map Bounds layer
	{
		ShowHideDialogItem(dialog,M25REPLACEMAP,false);
		//ShowHideDialogItem(dialog,M25FROST1,false);
	}
	ShowHideBounds(dialog);
	MySelectDialogItemText(dialog, M25REFLOATTIME, 0, 255);

	return 0;
}


short M25Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	Boolean changed, tempUseBounds;
	WorldPoint p, p2;
	WorldRect origBounds;
	float tempRefloat;
	TVectorMap *dialogTVectorMap = dynamic_cast<TVectorMap *>(gDialogTMap);
	OSErr err = 0;


	StandardLLClick(dialog, itemNum, M25TOPLATDEGREES, M25DEGREES, &p, &changed);
	StandardLLClick(dialog, itemNum, M25BOTTOMLATDEGREES, M25DEGREES, &p2, &changed);

	switch (itemNum) {
		case M25CANCEL: return M25CANCEL;


		case M25OK:
			
			tempRefloat = EditText2Float(dialog,M25REFLOATTIME);

			if (tempRefloat < 0)
			{
				printError("The refloat time must be nonnegative");
				break;
			}
			
			tempUseBounds = GetButton (dialog, M25WANTEDCHANGEBOUNDS);
			
			if(tempUseBounds)
			{
				long oneSecond = (1000000/3600);
				// retrieve the extendedBounds
				err = EditTexts2LL(dialog, M25TOPLATDEGREES, &p,TRUE);
				if(err)break;
				err = EditTexts2LL(dialog, M25BOTTOMLATDEGREES, &p2,TRUE);
				if(err)break;
				
				if (dialogTVectorMap -> HaveMapBoundsLayer()) {if (err = dialogTVectorMap -> ChangeMapBox(p,p2)) break; return itemNum;}

				// get original map bounds for comparison
				origBounds = dialogTVectorMap -> fMapBounds;  // if use GetMapBounds() may return extended bounds

				// check extended bounds (oneSecond handles accuracy issue in reading from dialog)			
				if (p.pLat < origBounds.hiLat - oneSecond || p2.pLat > origBounds.loLat + oneSecond
					|| p.pLong > origBounds.loLong + oneSecond || p2.pLong < origBounds.hiLong - oneSecond)
				{
					printError("The extended bounds must be larger than the original bounds."); 
					return 0; 
				}
				
				// just in case of round off
				p.pLat = _max(p.pLat,origBounds.hiLat);
				p.pLong = _min(p.pLong,origBounds.loLong);
				p2.pLat = _min(p2.pLat,origBounds.loLat);
				p2.pLong = _max(p2.pLong,origBounds.hiLong);
			}
						
			// point of no return
			/*if (GetButton (dialog, M25REPLACEMAP))
			{
				OSErr err;
				err = dialogTVectorMap -> ReplaceMap();
				return itemNum;	// what to do on error?
			}*/
			dialogTVectorMap -> fRefloatHalfLifeInHrs = EditText2Float(dialog,M25REFLOATTIME);
			dialogTVectorMap -> fUseExtendedBounds = GetButton (dialog, M25WANTEDCHANGEBOUNDS);

			if (dialogTVectorMap -> fUseExtendedBounds)
			{
					dialogTVectorMap -> fExtendedMapBounds.hiLat = p.pLat;
					dialogTVectorMap -> fExtendedMapBounds.loLong = p.pLong;
					dialogTVectorMap -> fExtendedMapBounds.loLat = p2.pLat;
					dialogTVectorMap -> fExtendedMapBounds.hiLong = p2.pLong;
					ChangeCurrentView(UnionWRect(settings.currentView, AddWRectBorders(dialogTVectorMap -> fExtendedMapBounds, 10)), TRUE, TRUE);
			}
			else
				dialogTVectorMap -> fExtendedMapBounds = dialogTVectorMap -> GetMapBounds();			
	
			return itemNum;
			
		case M25REFLOATTIME:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
		case M25WANTEDCHANGEBOUNDS:
			ToggleButton(dialog, itemNum);
			ShowHideBounds(dialog);
			break;
		case M25REPLACEMAP:
			//ToggleButton(dialog, itemNum);
			//if (GetButton (dialog, M25REPLACEMAP))
			//{
				//OSErr err;
				err = dialogTVectorMap -> ReplaceMap();
				if (err == USERCANCEL) break;	// stay at dialog
				
				return itemNum;	// what to do on error?
			//}
			break;

		case M25DEGREES:
		case M25DEGMIN:
		case M25DMS:
				err = EditTexts2LL(dialog, M25TOPLATDEGREES, &p,TRUE);
				if(err) break;
				err = EditTexts2LL(dialog, M25BOTTOMLATDEGREES, &p2,TRUE);
				if(err) break;
				if (itemNum == M25DEGREES) settings.latLongFormat = DEGREES;
				if (itemNum == M25DEGMIN) settings.latLongFormat = DEGMIN;
				if (itemNum == M25DMS) settings.latLongFormat = DMS;
				ShowHideBounds(dialog);
				//LL2EditTexts(dialog, M25TOPLATDEGREES, &p2);
				//LL2EditTexts(dialog, M25BOTTOMLATDEGREES, &p);
				LL2EditTexts(dialog, M25BOTTOMLATDEGREES, &p2);
				LL2EditTexts(dialog, M25TOPLATDEGREES, &p);
			break;

	}

	return 0;
}

static DOUBLEH	gEditContourLevels = 0;
static PopInfoRec M51PopTable[] = {
		{ M51, nil, M51WATERDENSITYPOPUP, 0, pWATERDENSITY, 0, 1, FALSE, nil },
		{ M51, nil, M51WAVEHTLABEL, 0, pWAVEHEIGHT, 0, 1, FALSE, nil },
		{ M51, nil, M51DEPTHLABEL, 0, pMIXEDLAYERDEPTH, 0, 1, FALSE, nil },
		{ M51, nil, M51DEPTHUNITS, 0, pDEPTHUNITS, 0, 1, FALSE, nil },
		{ M51, nil, M51WAVEHTUNITS, 0, pHEIGHTUNITS, 0, 1, FALSE, nil },
		{ M51, nil, M51DIAGNOSTICSTRTYPE, 0, pDIAGNOSTICSTRTYPE, 0, 1, FALSE, nil },
		{ M51, nil, M51WAVEHTINPUT, 0, pWAVEHTINPUT, 0, 1, FALSE, nil }};

void ShowHidePtCurMapDialogItems(DialogPtr dialog)
{
	Boolean contourBottom  = GetButton (dialog, M51CONTOURBOTTOM); 
	
	if (contourBottom)
	{
		ShowHideDialogItem(dialog, M51CONTOURDEPTH1, false);
		ShowHideDialogItem(dialog, M51CONTOURTO, false);
		ShowHideDialogItem(dialog, M51CONTOURDEPTH2, false);
		ShowHideDialogItem(dialog, M51CONTOURUNITS, false);
		ShowHideDialogItem(dialog, M51CONTOURBOTTOMLABEL, true);
		ShowHideDialogItem(dialog, M51CONTOURBOTTOMRANGE, true);
		ShowHideDialogItem(dialog, M51CONTOURBOTTOMUNITS, true);
	}
	else
	{
		ShowHideDialogItem(dialog, M51CONTOURDEPTH1, true); 
		ShowHideDialogItem(dialog, M51CONTOURTO, true); 
		ShowHideDialogItem(dialog, M51CONTOURDEPTH2, true); 
		ShowHideDialogItem(dialog, M51CONTOURUNITS, true); 
		ShowHideDialogItem(dialog, M51CONTOURBOTTOMLABEL, false); 
		ShowHideDialogItem(dialog, M51CONTOURBOTTOMRANGE, false); 
		ShowHideDialogItem(dialog, M51CONTOURBOTTOMUNITS, false); 
	}

	if(GetPopSelection(dialog,M51WATERDENSITYPOPUP)==4)
	{
		Long2EditText(dialog, M51WATERDENSITY, /*OK*/ (dynamic_cast<PtCurMap *>(gDialogTMap)) -> fWaterDensity);
		ShowHideDialogItem(dialog, M51WATERDENSITY, true);
	}
	else
		ShowHideDialogItem(dialog, M51WATERDENSITY, false); 

	if(GetPopSelection(dialog,M51WAVEHTINPUT)==1)
	{
		ShowHideDialogItem(dialog, M51WAVEHTLABEL, false); 
		ShowHideDialogItem(dialog, M51WAVEHT, false); 
		ShowHideDialogItem(dialog, M51WAVEHTUNITS, false); 
	}
	else
	{
		ShowHideDialogItem(dialog, M51WAVEHTLABEL, true); 
		ShowHideDialogItem(dialog, M51WAVEHT, true); 
		ShowHideDialogItem(dialog, M51WAVEHTUNITS, true); 
	}
}

OSErr M51Init(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	char name[256];
	Boolean contourBottom;
	
	RegisterPopTable (M51PopTable, sizeof (M51PopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (M51, dialog);
	
	if (/*OK*/ (dynamic_cast<PtCurMap *>(gDialogTMap)) -> fWaterDensity == 1020)
		SetPopSelection (dialog, M51WATERDENSITYPOPUP, 1);
	else if (/*OK*/ (dynamic_cast<PtCurMap *>(gDialogTMap)) -> fWaterDensity == 1010)
		SetPopSelection (dialog, M51WATERDENSITYPOPUP, 2);
	else if (/*OK*/ (dynamic_cast<PtCurMap *>(gDialogTMap)) -> fWaterDensity == 1000)
		SetPopSelection (dialog, M51WATERDENSITYPOPUP, 3);
	else 
		SetPopSelection (dialog, M51WATERDENSITYPOPUP, 4);

	//SetPopSelection (dialog, M51WAVEHTLABEL, 1);
	//SetPopSelection (dialog, M51WAVEHTLABEL, 2);	// significant wave height should be the default
	SetPopSelection (dialog, M51WAVEHTLABEL, /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fWaveHtInput == 0 ? 2 : (dynamic_cast<PtCurMap *>(gDialogTMap)) -> fWaveHtInput);	// significant wave height should be the default
	SetPopSelection (dialog, M51WAVEHTINPUT, /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fWaveHtInput == 0 ? 1 : 2 );	// significant wave height should be the default
	SetPopSelection (dialog, M51DEPTHLABEL, 1);
	SetPopSelection (dialog, M51DEPTHUNITS, 1);
	SetPopSelection (dialog, M51WAVEHTUNITS, 1);
	SetPopSelection (dialog, M51DIAGNOSTICSTRTYPE, /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fDiagnosticStrType + 1);

	gDialogTMap -> GetMapName(name);
	mysetitext(dialog, M51NAME, name);
	
	contourBottom = /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourDepth1 == BOTTOMINDEX;
	SetButton(dialog, M51CONTOURBOTTOM, contourBottom);
	//SetButton (dialog, M51REPLACEMAP, false); 
	
	Float2EditText(dialog,M51REFLOATTIME, gDialogTMap -> fRefloatHalfLifeInHrs , 4);
	
	if (contourBottom)
	{	// in case they change to a water column contour (maybe a new field bContourBottom ?)
		Float2EditText(dialog,M51CONTOURDEPTH1, 0, 4);
		Float2EditText(dialog,M51CONTOURDEPTH2, 5, 4);	// code goes here, use mixed layer depth?
	}
	else
	{
		Float2EditText(dialog,M51CONTOURDEPTH1, /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourDepth1, 4);
		Float2EditText(dialog,M51CONTOURDEPTH2, /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourDepth2, 4);
	}
	
	Float2EditText(dialog,M51CONTOURBOTTOMRANGE, /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fBottomRange, 4);	// have a default value
	//Long2EditText(dialog, M51WATERDENSITY, ((PtCurMap*)gDialogTMap) -> fWaterDensity);
	Float2EditText(dialog, M51DEPTH, /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fMixedLayerDepth, 2);
	if (/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fWaveHtInput == 1)
		Float2EditText(dialog, M51WAVEHT, /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fBreakingWaveHeight, 2);	// breaking wave height
	else
		Float2EditText(dialog, M51WAVEHT, /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fBreakingWaveHeight * 1.42, 2);	// significant wave height
	//MySelectDialogItemText(dialog, M51REFLOATTIME, 0, 255);
	MySelectDialogItemText(dialog, M51CONTOURDEPTH1, 0, 255);

	/*if(GetPopSelection(dialog,M51WATERDENSITYPOPUP)==4)
	{
		Long2EditText(dialog, M51WATERDENSITY, ((PtCurMap*)gDialogTMap) -> fWaterDensity);
		ShowHideDialogItem(dialog, M51WATERDENSITY, true);
	}
	else
		ShowHideDialogItem(dialog, M51WATERDENSITY, false); 
*/
	ShowHidePtCurMapDialogItems(dialog);

	return 0;
}


short M51Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	long	menuID_menuItem, waterDensity;
	short 	waterType, depthInputType, seaType, units, inputType;
	float	depth1, depth2, maxDepth, factor, bottomDepth;
	double	mixedLayerDepth, breakingWaveHeight;
	Boolean contourBottom  = GetButton (dialog, M51CONTOURBOTTOM); 
	OSErr err = 0;


	switch (itemNum) {
		case M51CANCEL: return M51CANCEL;

		case M51OK:
			// check input values
			inputType = GetPopSelection(dialog,M51WAVEHTINPUT);
			// if (inputType==1) skip the checks, sig wave height will be calculated when it's needed based on wind speed
			units = GetPopSelection(dialog, M51DEPTHUNITS);
			if (units==2)
				factor = .3048;
			else
				factor = 1.;
			depthInputType = GetPopSelection(dialog, M51DEPTHLABEL);
			if (depthInputType==1)	// non developed seas
				mixedLayerDepth = EditText2Float(dialog, M51DEPTH) * factor;
			else if (depthInputType==2)	// width between windrows 
				mixedLayerDepth = EditText2Float(dialog, M51DEPTH) * factor * 2.;

			units = GetPopSelection(dialog, M51WAVEHTUNITS);
			if (units==2)
				factor = .3048;
			else
				factor = 1.;
			seaType = GetPopSelection(dialog, M51WAVEHTLABEL);
			if (seaType==1)	// non developed seas
				breakingWaveHeight = EditText2Float(dialog, M51WAVEHT) * factor;
			else if (seaType==2)	// fully developed seas
				//breakingWaveHeight = EditText2Float(dialog, M51DEPTH) * factor / 1.42;
				breakingWaveHeight = EditText2Float(dialog, M51WAVEHT) * factor / 1.42;
				
			// should check which input types were chosen...
			if (breakingWaveHeight <= 0)
			{
				printError("The breaking wave height must be greater than zero");
				break;				
			}
			if (mixedLayerDepth < 1.5 * breakingWaveHeight)
			{
				printError("The mixed layer depth must be greater than or equal to 1.5 times the breaking wave height");
				break;				
			}
			
			if (!contourBottom)
			{
				depth1 = EditText2Float(dialog,M51CONTOURDEPTH1);
				depth2 = EditText2Float(dialog,M51CONTOURDEPTH2);
				if (depth1 >= depth2)
				{
					printError("Depth1 must be less than Depth2");
					break;
				}
				//maxDepth = ((PtCurMap*)gDialogTMap) -> GetMaxDepth2();	// use 3D stuff - should be able to combine...
				maxDepth = gDialogTMap -> GetMaxDepth2();	// use 3D stuff - should be able to combine...
				if (depth2 > maxDepth || mixedLayerDepth > maxDepth)
				{
					char errStr[64];
					sprintf(errStr,"The maximum depth of the region is %g meters.",maxDepth);
					printError(errStr);
					break;
				}
			}
			else
			{
				bottomDepth = EditText2Float(dialog,M51CONTOURBOTTOMRANGE);
			}
			waterType = GetPopSelection(dialog, M51WATERDENSITYPOPUP);
			if (waterType==1)	// oceanic
				waterDensity = 1020;
			else if (waterType==2)	// estuarine 
				waterDensity = 1010;
			else if (waterType==3)	// fresh water 
				waterDensity = 1000;
			else if (waterType==4)	// user input 
				waterDensity = EditText2Float(dialog, M51WATERDENSITY);
			
			if (waterDensity <= 0)
			{
				printError("The water density must be greater than zero");
				break;				
			}
			
			// point of no return
			/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fWaterDensity = waterDensity;
			/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fMixedLayerDepth = mixedLayerDepth;
			/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fBreakingWaveHeight = breakingWaveHeight;
			/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fWaveHtInput = inputType-1+seaType-1;

			gDialogTMap -> fRefloatHalfLifeInHrs = EditText2Float(dialog,M51REFLOATTIME);
			if (!contourBottom)
			{
				/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourDepth1 = depth1;
				/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourDepth2 = depth2;
			}
			else
			{
				/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourDepth1 = BOTTOMINDEX;
				/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourDepth2 = BOTTOMINDEX;
				/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fBottomRange = bottomDepth;
			}
			if (gEditContourLevels && gEditContourLevels != /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourLevelsH) 
			{
				MyDisposeHandle((Handle*)&(/*OK*/dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourLevelsH);
				/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourLevelsH = gEditContourLevels;
			}

			/*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fDiagnosticStrType = GetPopSelection(dialog, M51DIAGNOSTICSTRTYPE) - 1;
			return itemNum;
			
		case M51DEPTH:
		case M51WAVEHT:
		case M51WATERDENSITY:		
		case M51REFLOATTIME:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
		case M51CONTOURDEPTH1:
		case M51CONTOURDEPTH2:
		case M51CONTOURBOTTOMRANGE:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

		case M51REPLACEMAP:
			err = gDialogTMap -> ReplaceMap();
			if (err == USERCANCEL) break;	// stay at dialog
			return itemNum;	// what to do on error?
			break;

		case M51SETCONTOURS:
			// should dispose if exists ?
			gEditContourLevels = (DOUBLEH)_NewHandleClear(0);
			// code goes here, should init fContourLevelsH if nil
	/*OK*/ if((dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourLevelsH)
			{
				gEditContourLevels = /*OK*/(dynamic_cast<PtCurMap *>(gDialogTMap)) -> fContourLevelsH;
				if(_HandToHand((Handle *)&gEditContourLevels))
				{
					printError("Not enough memory to create temporary contour levels");
					break;
				}
			}
			err = ContourDialog(&gEditContourLevels,0);
			break;

		case M51WATERDENSITYPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			/*if(GetPopSelection(dialog,itemNum)==4)
				ShowHideDialogItem(dialog, M51WATERDENSITY, true); 
			else
				ShowHideDialogItem(dialog, M51WATERDENSITY, false); */
			ShowHidePtCurMapDialogItems(dialog);
			break;
		case M51CONTOURBOTTOM:
			ToggleButton(dialog, M51CONTOURBOTTOM);
			//sharedRMover -> bUseDepthDependentDiffusion = GetButton(dialog,M28bDEPTHDEPENDENT);
			ShowHidePtCurMapDialogItems(dialog);
			break;

		case M51WAVEHTINPUT:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowHidePtCurMapDialogItems(dialog);
			break;
			

		case M51DEPTHLABEL:
		case M51WAVEHTLABEL:
		case M51DEPTHUNITS:
		case M51WAVEHTUNITS:
		case M51DIAGNOSTICSTRTYPE:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
	}

	return 0;
}

OSErr M50Init(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	char name[256];
	
	gDialogTMap -> GetMapName(name);
	mysetitext(dialog, M50NAME, name);
	
	Float2EditText(dialog,M50REFLOATTIME, gDialogTMap -> fRefloatHalfLifeInHrs , 4);
	
	if (gDialogTMap ->IAm(TYPE_OSSMMAP)) 	// don't show refloat stuff since it varies with each grid cell
	{
		ShowHideDialogItem(dialog,M50REFLOATLABEL,false);
		ShowHideDialogItem(dialog,M50REFLOATTIME,false);
		ShowHideDialogItem(dialog,M50REFLOATLABEL2,false);
	}
	else
		MySelectDialogItemText(dialog, M50REFLOATTIME, 0, 255);

	return 0;
}


short M50Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	OSErr 	err = 0;

	switch (itemNum) {
		case M50CANCEL: return M50CANCEL;

		case M50OK:
			gDialogTMap -> fRefloatHalfLifeInHrs = EditText2Float(dialog,M50REFLOATTIME);
			return itemNum;
			
		case M50REFLOATTIME:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

		case M50REPLACEMAP:
			err = gDialogTMap -> ReplaceMap();
			if (err == USERCANCEL) break;	// stay at dialog
			return itemNum;	// what to do on error?
			break;

	}

	return 0;
}

OSErr TMapSettingsDialog(TMap* theMap)
{
	short item;
	gDialogTMap = theMap;
	if(theMap->IAm(TYPE_VECTORMAP) /*&& !theMap->HaveMapBoundsLayer()*/) // allow user to extend map bounds
		item = MyModalDialog(M25, mapWindow, 0, M25Init, M25Click);
	else if((theMap->IAm(TYPE_PTCURMAP) || theMap->IAm(TYPE_COMPOUNDMAP)) && /*OK*/(dynamic_cast<PtCurMap *>(theMap))->ThereIsADispersedSpill()) 
	{	// code goes here, make sure there is no way for casual users to see this 
		item = MyModalDialog(M51, mapWindow, 0, M51Init, M51Click);
	}
	else
		item = MyModalDialog(M50, mapWindow, 0, M50Init, M50Click);
	gDialogTMap =  0;
	if(item == M50CANCEL || item == M25CANCEL) return USERCANCEL; 
	model->NewDirtNotification();
	if(item == M50OK || item == M25OK) return 0; 
	else return -1;
}



/////////////////////////////////////////////////


UNIQUEID ZeroID(void)
{
	UNIQUEID uid;
	memset(&uid,0,sizeof(uid));
	return uid;
}

/////////////////////////////////////////////////


OSErr ReadMacValue(BFPB *bfpb, Seconds* val)
{
	return ReadValue(bfpb,(char*)val,sizeof(*val),TRUE);
}

OSErr ReadMacValue(BFPB *bfpb, char* val)
{
	return ReadValue(bfpb,val,sizeof(*val),FALSE);
}

OSErr ReadMacValue(BFPB *bfpb, long* val)
{
	return ReadValue(bfpb,(char*)val,sizeof(*val),TRUE);
}

OSErr ReadMacValue(BFPB *bfpb, short* val)
{
	return ReadValue(bfpb,(char*)val,sizeof(*val),TRUE);
}

OSErr ReadMacValue(BFPB *bfpb, float* val)
{
	return ReadValue(bfpb,(char*)val,sizeof(*val),TRUE);
}

OSErr ReadMacValue(BFPB *bfpb, double* val)
{
	return ReadValue(bfpb,(char*)val,sizeof(*val),TRUE);
}

OSErr ReadMacValue(BFPB *bfpb, Boolean* val)
{
	char c = 0;
	OSErr err = ReadValue(bfpb,&c,sizeof(c),FALSE);
	*val= c;
	return err;
}


OSErr ReadMacValue(BFPB *bfpb, UNIQUEID* val)
{
	OSErr err = ReadMacValue(bfpb,&val->ticksAtCreation);
	if(!err) err = ReadMacValue(bfpb,&val->counter);
	return err;
}

OSErr ReadMacValue(BFPB *bfpb, LongPoint *lp)
{
	OSErr err = 0;
	if (err = ReadMacValue(bfpb,&(lp->h))) return err;
	if (err = ReadMacValue(bfpb,&(lp->v))) return err;
	
	return 0;
}

OSErr ReadMacValue(BFPB *bfpb, WorldRect *wRect)
{
	OSErr err = 0;
	if (err = ReadMacValue(bfpb,&(wRect->hiLat))) return err;
	if (err = ReadMacValue(bfpb,&(wRect->loLat))) return err;
	if (err = ReadMacValue(bfpb,&(wRect->hiLong))) return err;
	if (err = ReadMacValue(bfpb,&(wRect->loLong))) return err;
	
	return 0;
}

OSErr ReadMacValue(BFPB *bfpb, Rect *theRect)
{
	OSErr err = 0;
	if (err = ReadMacValue(bfpb,&(theRect->left))) return err;
	if (err = ReadMacValue(bfpb,&(theRect->top))) return err;
	if (err = ReadMacValue(bfpb,&(theRect->right))) return err;
	if (err = ReadMacValue(bfpb,&(theRect->bottom))) return err;
	
	return 0;
}

OSErr ReadMacValue(BFPB *bfpb, WorldPoint *wp)
{
	OSErr err = 0;
	if (err = ReadMacValue(bfpb,&(wp->pLong))) return err;
	if (err = ReadMacValue(bfpb,&(wp->pLat))) return err;
	
	return 0;
}

OSErr ReadMacValue(BFPB *bfpb, char* str, long len)
{
	return ReadValue(bfpb,str,len,FALSE);
}


OSErr WriteMacValue(BFPB *bfpb, Seconds val)
{
	return WriteValue(bfpb,(char*)&val,sizeof(val),TRUE);
}

OSErr WriteMacValue(BFPB *bfpb, char val)
{
	return WriteValue(bfpb,&val,sizeof(val),FALSE);
}

OSErr WriteMacValue(BFPB *bfpb, long val)
{
	return WriteValue(bfpb,(char*)&val,sizeof(val),TRUE);
}

OSErr WriteMacValue(BFPB *bfpb, short val)
{
	return WriteValue(bfpb,(char*)&val,sizeof(val),TRUE);
}

OSErr WriteMacValue(BFPB *bfpb, float val)
{
	return WriteValue(bfpb,(char*)&val,sizeof(val),TRUE);
}

OSErr WriteMacValue(BFPB *bfpb, double val)
{
	return WriteValue(bfpb,(char*)&val,sizeof(val),TRUE);
}

OSErr WriteMacValue(BFPB *bfpb, Boolean val)
{
	char c = val;
	OSErr err = WriteValue(bfpb,&c,sizeof(c),FALSE);
	return err;
}

OSErr WriteMacValue(BFPB *bfpb, UNIQUEID val)
{
	OSErr err = WriteMacValue(bfpb,val.ticksAtCreation);
	if(!err) err = WriteMacValue(bfpb,val.counter);
	return err;
}

OSErr WriteMacValue(BFPB *bfpb, LongPoint lp)
{
	OSErr err = 0;
	if (err = WriteMacValue(bfpb,lp.h)) return err;
	if (err = WriteMacValue(bfpb,lp.v)) return err;
	
	return 0;
}
OSErr WriteMacValue(BFPB *bfpb, WorldRect wRect)
{
	OSErr err = 0;
	if (err = WriteMacValue(bfpb,wRect.hiLat)) return err;
	if (err = WriteMacValue(bfpb,wRect.loLat)) return err;
	if (err = WriteMacValue(bfpb,wRect.hiLong)) return err;
	if (err = WriteMacValue(bfpb,wRect.loLong)) return err;
	
	return 0;
}

OSErr WriteMacValue(BFPB *bfpb, Rect theRect)
{
	OSErr err = 0;
	if (err = WriteMacValue(bfpb,theRect.left)) return err;
	if (err = WriteMacValue(bfpb,theRect.top)) return err;
	if (err = WriteMacValue(bfpb,theRect.right)) return err;
	if (err = WriteMacValue(bfpb,theRect.bottom)) return err;
	
	return 0;
}

OSErr WriteMacValue(BFPB *bfpb, WorldPoint wp)
{
	OSErr err = 0;
	if (err = WriteMacValue(bfpb,wp.pLong)) return err;
	if (err = WriteMacValue(bfpb,wp.pLat)) return err;
	
	return 0;
}

OSErr WriteMacValue(BFPB *bfpb, char* str, long len)
{
	return WriteValue(bfpb,str,len,FALSE);
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////

/////////////////////////////////////////////////
TModelMessage::TModelMessage(long messageCode,char* targetName,char *dataStr)
{// constructor
	this->fMessageCode = messageCode;
	this->fTargetUniqueID = ZeroID();
	this->fTargetName = targetName;
	this->fDataStr = dataStr;
	this->fDataHdl = 0;
}
/////////////////////////////////////////////////
TModelMessage::TModelMessage(long messageCode,char* targetName,CHARH dataHdl)
{// constructor
	this->fMessageCode = messageCode;
	this->fTargetUniqueID = ZeroID();
	this->fTargetName = targetName;
	this->fDataStr = 0;
	this->fDataHdl = dataHdl;
}
/////////////////////////////////////////////////
TModelMessage::TModelMessage(long messageCode,UNIQUEID targetUniqueID,char *dataStr)
{// constructor
	this->fMessageCode = messageCode;
	this->fTargetUniqueID = targetUniqueID;
	this->fTargetName = nil;
	this->fDataStr = dataStr;
	this->fDataHdl = 0;
}
/////////////////////////////////////////////////
TModelMessage::TModelMessage(long messageCode,UNIQUEID targetUniqueID,CHARH dataHdl)
{// constructor
	this->fMessageCode = messageCode;
	this->fTargetUniqueID = targetUniqueID;
	this->fTargetName = nil;
	this->fDataStr = 0;
	this->fDataHdl = dataHdl;
}
/////////////////////////////////////////////////
TModelMessage::~TModelMessage()
{// destructor
	if(this->fDataHdl) DisposeHandle((Handle)this->fDataHdl);
	this->fDataHdl = 0;
}
/////////////////////////////////////////////////
Boolean TModelMessage::StringsMatch(char* str1,char* str2)
{
	Boolean matches = false;
	if(!str1 || !str1[0])
	{ 	// no string specified
		// i.e. matches nil and ""
		if(!str2) matches = true;
		else if(str2[0] == 0) matches = true;
		else matches = false;
	}
	else
	{	// 
		if(!str2) matches = false;
		else matches = (!strcmpnocase(str1,str2));
	} 
	return matches;
}
/////////////////////////////////////////////////
Boolean TModelMessage::IsMessage(long messageCode)
{
	if(this->fMessageCode != messageCode) return false;
	return true;
}
/////////////////////////////////////////////////
Boolean TModelMessage::IsMessage(char* targetName)
{
	if(!this->StringsMatch(this->fTargetName,targetName)) return false;
	return true;
}
/////////////////////////////////////////////////
Boolean TModelMessage::IsMessage(long messageCode,char* targetName)
{
	if(this->fMessageCode != messageCode) return false;
	if(!this->StringsMatch(this->fTargetName,targetName)) return false;
	return true;
}
/////////////////////////////////////////////////
Boolean TModelMessage::IsMessage(long messageCode,UNIQUEID targetUniqueID)
{
	if(this->fMessageCode != messageCode) return false;
	if(!EqualUniqueIDs(this->fTargetUniqueID,targetUniqueID)) return false;
	return true;
}


/////////////////////////////////////////////////


void TModelMessage::GetParameterString(char * key,char * answerStr,long maxNumChars)
{
	// we use the wizard get parameter functions
	answerStr[0] = 0;
	if(model->fWizard)
	{
		long lineNum1Relative = 1;
		if(this->fDataStr) 
		{
			model->fWizard->GetParameterString(key,lineNum1Relative,this->fDataStr, answerStr,maxNumChars);
		}
		if(this->fDataHdl) 
		{
			model->fWizard->GetParameterString(key,lineNum1Relative,this->fDataHdl, answerStr,maxNumChars);
		}
	}
}

#define MAXNUMPARAMCHARS 64
OSErr TModelMessage::GetParameterAsDouble(char * key,double * val)
{
	// we use the wizard get parameter functions
	char localStr[MAXNUMPARAMCHARS];
	double localVal;
	OSErr err = -1;
	this->GetParameterString(key,localStr,MAXNUMPARAMCHARS);
	if(localStr[0]) err =  StringToDouble(localStr,&localVal);
	else err = -1;
	if(!err) *val = localVal;
	return err;
}

/////////////////////////////////////////////////
OSErr TModelMessage::GetParameterAsLong(char * key,long * val)
{
	// we use the wizard get parameter functions
	char localStr[MAXNUMPARAMCHARS];
	long localVal;
	OSErr err = -1;
	this->GetParameterString(key,localStr,MAXNUMPARAMCHARS);
	if(localStr[0]) 
	{
		long numScanned = sscanf(localStr,"%ld",&localVal);
		if(numScanned ==1) 
		{
			*val = localVal;
			return noErr;
		}
	}
	return err;
}
/////////////////////////////////////////////////
OSErr TModelMessage::GetParameterAsBoolean(char * key,Boolean * val)
{// returns error if the data is not a single number
	char localStr[MAXNUMPARAMCHARS];
	this->GetParameterString(key,localStr,MAXNUMPARAMCHARS);
	switch(localStr[0])
	{
		case 'T':
		case 't':
		case '1':
			*val = true; return 0;
		case 'F':
		case 'f':
		case '0':
			*val = false; return 0;
		default: 
			return -1; //error
	}
	return -1;
}
/////////////////////////////////////////////////

OSErr TModelMessage::GetParameterAsWorldPoint(char * key,WorldPoint * val,Boolean checkForLLInputWithoutDirection)
{
	char str[256];
	WorldPoint p = {0,0};
	
	this-> GetParameterString(key,str,256);
	if(str && str[0]) 
	{
		char dirLat = 0, dirLong = 0;
		double degreesLat = 0, degreesLong = 0;
		long numscanned = sscanf(str,lfFix("%lf %c %lf %c"),&degreesLong,&dirLong,&degreesLat,&dirLat);
		if (numscanned==4)
		{
			if(dirLong == 'e') dirLong = 'E';
			else if(dirLong == 'w') dirLong = 'W';
			if(dirLat == 'n') dirLat = 'N';
			else if(dirLat == 'S') dirLat = 'S';
			if(/*numscanned == 4 
				&&*/ (dirLong == 'E' || dirLong == 'W') 
				&& (dirLat == 'N' ||dirLat == 'S') 
				)
			{
				DoublesToWorldPoint(degreesLat, degreesLong, dirLat, dirLong, &p);
				*val = p;
				return noErr;
			}
		}
		else
		{	
			if (checkForLLInputWithoutDirection)
			{
				numscanned = sscanf(str,lfFix("%lf %lf"),&degreesLong,&degreesLat);
				if (numscanned == 2)
				{
					if (degreesLong >= 0) dirLong = 'E';
					else {dirLong = 'W'; degreesLong *= -1;}
					if (degreesLat >= 0) dirLat = 'N';
					else {dirLat = 'S'; degreesLat *= -1;}
					DoublesToWorldPoint(degreesLat, degreesLong, dirLat, dirLong, &p);
					*val = p;
					return noErr;
				}
			}
		}
	}
	return -1;
}
/////////////////////////////////////////////////

OSErr TModelMessage::GetParameterAsSeconds(char * key,Seconds * val)
{	// accepts input in the form 
	// DD,MM,YYYY,HH,mm
	char str[256];
	unsigned long asSeconds;
	long numScanned;
	DateTimeRec time;
	
	this-> GetParameterString(key,str,256);
	if(str && str[0]) 
	{
		StringSubstitute(str, ',', ' ');
		
		numScanned=sscanf(str, "%hd %hd %hd %hd %hd",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute) ;
		if (numScanned!= 5)
			return -2;

		// check date is valid
		if (time.day<1 || time.day>31 || time.month<1 || time.month>12)
			return -3;

		if (time.year < 1900)					// two digit date, so fix it
		{
			if (time.year >= 40 && time.year <= 99)	// JLM
				time.year += 1900;
			else
				time.year += 2000;					// correct for year 2000 (00 to 40)
		}
		
		time.second = 0; 
		DateToSeconds (&time, &asSeconds);
		*val = asSeconds;
		return noErr;
	
	}
	return -1;
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
