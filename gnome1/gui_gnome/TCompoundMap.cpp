#include "Cross.h"
#include "Contdlg.h"
#include "NetCDFMoverCurv.h"
//#include "MapUtils.h"
//#include "OUtils.h"
//#include "EditWindsDialog.h"
//#include "Uncertainty.h"


/**static VList sgObjects;
 static CMyList *sMoverList=0;
 static CMyList *sDialogList=0;
 static short NAME_COL;
 static TCompoundMover	*sSharedDialogCompoundMover = 0;
 
 static CurrentUncertainyInfo sSharedCompoundUncertainyInfo; // used to hold the uncertainty dialog box info in case of a user cancel
 
 #define REPLACE true
 */


/**************************************************************************************************/
TCompoundMap* CreateAndInitCompoundMap(char *path, WorldRect bounds)
{
	char 		nameStr[256];
	OSErr		err = noErr;
	TCompoundMap 	*map = nil;
	char fileName[256],s[256];
	
	if (path[0])
	{
		strcpy(s,path);
		SplitPathFile (s, fileName);
		strcpy (nameStr, "Compound Map: ");
		strcat (nameStr, fileName);
	}
	else
		strcpy(nameStr,"Compound Map");
	map = new TCompoundMap(nameStr, bounds);
	if (!map)
	{ TechError("CreateAndInitCompoundMap()", "new TCompoundMap()", 0); return nil; }
	
	if (err = map->InitMap()) { delete map; return nil; }
	
	return map;
}

//TCompoundMap::TCompoundMap (char *name, WorldRect bounds) : TMap (name, bounds)
TCompoundMap::TCompoundMap (char *name, WorldRect bounds) : PtCurMap (name, bounds)
{
	mapList = 0;
	
	bMapsOpen = FALSE;
	
	return;
}

void TCompoundMap::Dispose ()
{
	long i, n;
	TMap *map;
	// may need to remove the compound current first
	//TMap::Dispose();
	//long i, n;
	TMover *mover;
	
	if (moverList != nil)
	{
		for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
			moverList->GetListItem((Ptr)&mover, i);
			mover->Dispose();
			delete mover;
		}
		
		moverList->Dispose();
		delete moverList;
		moverList = nil;
	}
	if (mapList != nil)
	{
		for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
			mapList->GetListItem((Ptr)&map, i);
			//map->Dispose();	// these maps have nothing on them
			//delete map;
		}
		
		mapList->Dispose();
		delete mapList;
		mapList = nil;
	}
	//TMap::Dispose();
	PtCurMap::Dispose();
}

OSErr TCompoundMap::InitMap ()
{
	OSErr err = 0;
	mapList = new CMyList(sizeof(TMap *));
	if (!mapList)
	{ TechError("TCompoundMap::InitMap()", "new CMyList()", 0); return -1; }
	if (err = mapList->IList())
	{ TechError("TCompoundMap::InitMap()", "IList()", 0); return -1; }
	
	//return TMap::InitMap();
	return PtCurMap::InitMap();
}

/*OSErr TCompoundMap::DeleteItem(ListItem item)
 {	// for now don't allow this
 if (item.index == I_COMPOUNDMAPNAME)
 return moverMap -> DropMap(this);
 
 return 0;
 }*/



OSErr TCompoundMap::DropMover(TMover *theMover)
{
	long 	i, numMovers;
	OSErr	err = noErr;
	TCurrentMover *mover = 0;
	TMover *thisMover = 0;
	
	if (moverList->IsItemInList((Ptr)&theMover, &i))
	{
		if (err = moverList->DeleteItem(i))
		{ TechError("TCompoundMap::DropMover()", "DeleteItem()", err); return err; }
	}
	numMovers = moverList->GetItemCount();
	mover = GetCompoundMover(); 
	if (numMovers==0) err = model->DropMap(this);
	
	if (!mover)
	{
		for (i = 0; i < numMovers; i++)
		{
			this -> moverList -> GetListItem ((Ptr) &thisMover, 0); // will always want the first item in the list
			if (err = this->DropMover(thisMover)) return err; // gets rid of first mover, moves rest up
		}
		err = model->DropMap(this);
	}
	SetDirty (true);
	
	return err;
}


Boolean TCompoundMap::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	switch (item.index) {
		case I_COMPOUNDMAPNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE; 
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (!model->mapList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (model->mapList->GetItemCount() - 1);
					}
					/*if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					 switch (buttonID) {
					 case UPBUTTON: return i > 0;
					 case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
					 }*/
			}
			break;
		case I_PMOVERS:
			switch (buttonID) {
				case ADDBUTTON: return TRUE;
				case SETTINGSBUTTON: return FALSE;
				case DELETEBUTTON: return FALSE;
			}
			break;
			/*case I_COMPOUNDCURRENT:	// allow to delete all currents here?
			 switch (buttonID) {
			 case ADDBUTTON: return FALSE;
			 case DELETEBUTTON: return TRUE;
			 case UPBUTTON:
			 case DOWNBUTTON:
			 if (!moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
			 switch (buttonID) {
			 case UPBUTTON: return i > 0;
			 case DOWNBUTTON: return i < (moverList->GetItemCount() - 1);
			 }
			 }
			 break;*/
		case I_PDRAWCONTOURS:
		case I_PSHOWLEGEND:
		case I_PSHOWSURFACELES:
		case I_PSETCONTOURS:
		case I_PCONCTABLE:
		case I_PWATERDENSITY:
		case I_PMIXEDLAYERDEPTH:
		case I_PBREAKINGWAVEHT:
		case I_PDIAGNOSTICSTRING:
			//case I_PDROPLETINFO:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
	}
	
	
	if (buttonID == SETTINGSBUTTON) return TRUE;
	
	//	return TCurrentMover::FunctionEnabled(item, buttonID);
	return FALSE;
}

OSErr TCompoundMap::UpItem(ListItem item)
{	//maybe CompoundMover should be based on TMover rather than TCurrentMover...
	/*long i;
	 OSErr err = 0;
	 
	 if (item.index == I_COMPOUNDCURRENT)
	 if (moverList->IsItemInList((Ptr)&item.owner, &i))
	 if (i > 0) {
	 if (err = moverList->SwapItems(i, i - 1))
	 { TechError("TCompoundMover::UpItem()", "moverList->SwapItems()", err); return err; }
	 SelectListItem(item);
	 UpdateListLength(true);
	 InvalidateMapImage();
	 InvalMapDrawingRect();
	 }
	 
	 return 0;*/
	return TMap::UpItem(item);
}

OSErr TCompoundMap::DownItem(ListItem item)
{
	/*long i;
	 OSErr err = 0;
	 
	 if (item.index == I_COMPOUNDCURRENT)
	 if (moverList->IsItemInList((Ptr)&item.owner, &i))
	 if (i < (moverList->GetItemCount() - 1)) {
	 if (err = moverList->SwapItems(i, i + 1))
	 { TechError("TCompoundMover::UpItem()", "moverList->SwapItems()", err); return err; }
	 SelectListItem(item);
	 UpdateListLength(true);
	 InvalidateMapImage();
	 InvalMapDrawingRect();
	 }
	 
	 return 0;*/
	return TMap::DownItem(item);
}

OSErr TCompoundMap::SettingsItem(ListItem item)
{
	return TMap::SettingsItem(item);
}

Boolean TCompoundMap::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	//TMap *newMap = 0;
	if (inBullet)
		switch (item.index) {
			case I_COMPOUNDMAPNAME: bOpen = !bOpen; return TRUE;
				/*case I_COMPOUNDMAPACTIVE:
				 {
				 bActive = !bActive;
				 InvalMapDrawingRect();
				 return TRUE;
				 }*/
			case I_COMPOUNDMAP:
				bMapsOpen = !bMapsOpen; return TRUE;
				//return TRUE;
			case I_COMPOUNDMOVERS: bMoversOpen = !bMoversOpen; return TRUE;
			case I_PSHOWSURFACELES:
				bShowSurfaceLEs = !bShowSurfaceLEs;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_PSHOWLEGEND:
				bShowLegend = !bShowLegend;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
		}
	
	if (doubleClick)
	{
		if (this -> FunctionEnabled(item, SETTINGSBUTTON)) {
			if (item.index == I_PSETCONTOURS || item.index == I_PSHOWLEGEND)
			{
				if (!fContourLevelsH) 
				{
					if (InitContourLevels()==-1) return TRUE;
				}
				ContourDialog(&fContourLevelsH,0);
				return TRUE;
			}
			if (item.index == I_PCONCTABLE)
			{
				TTriGridVel3D* triGrid = GetGrid3D(true);	// use refined grid if there is one
				
				if (!triGrid) return true; 
				outputData **oilConcHdl = triGrid -> GetOilConcHdl();	
				if (!oilConcHdl)
				{
					printError("There is no concentration data to plot");
					//err = -1;
					//return TRUE;
				}
				else
				{
					float depthRange1 = fContourDepth1, depthRange2 = fContourDepth2, bottomRange = fBottomRange;
					Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
					// compare current contourdepths with values at start of run
					// give warning if they are different, and send original to plot
					if (fContourDepth1AtStartOfRun != fContourDepth1 || fContourDepth2AtStartOfRun != fContourDepth2)
					{
						short buttonSelected;
						// code goes here, set up a new dialog with better wording
						buttonSelected  = MULTICHOICEALERT(1690,"The depth range has been changed. To see plots at new depth range rerun the model. Do you still want to see the old plots?",FALSE);
						switch(buttonSelected){
							case 1: // continue
								break;  
							case 3: // stop
								return 0; 
								break;
						}
						//printNote("The depth range has been changed. To see plots at new depth range rerun the model");	
						depthRange1 = fContourDepth1AtStartOfRun;
						depthRange2 = fContourDepth2AtStartOfRun;
					}
					if (triSelected) 	// tracked output at a specified area
						PlotDialog(oilConcHdl,fDepthSliceArray,depthRange1,depthRange2,bottomRange,true,false);
					else 	// tracked output following the plume
						PlotDialog(oilConcHdl,fDepthSliceArray,depthRange1,depthRange2,bottomRange,false,false);
				}
				//ConcentrationTable(oilConcHdl/*,fTriAreaArray,GetNumContourLevels()*/);
			}
			item.index = I_COMPOUNDMAPNAME;
			this -> SettingsItem(item);
			return TRUE;
		}
		if (item.index == I_PMOVERS)
		{
			item.owner -> AddItem (item);
			return TRUE;
		}
		
		//CompoundMoverSettingsDialog(this, this -> moverMap, &newMap);
	}
	
	return FALSE;
}

long TCompoundMap::GetListLength()
{
	long	i, n, listLength = 0;
	//TMap *map;
	TMover *mover = 0;
	TMap *map = 0;
	
	listLength = 1;
	
	if (bOpen)
	{
		//listLength += 1;		// add one for active box at top
		
		listLength += 1;		// one for title
		listLength += 1;		// one for title
		if (bMapsOpen)
		{	// here have a draw map and draw contours for each map, anything else? title
			n = mapList->GetItemCount() ;
			//listLength += n;
			for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
				mapList->GetListItem((Ptr)&map, i);
				//listLength += map->GetListLength();
				listLength += map->GetListLength();
			}
		}
		if(this->ThereIsADispersedSpill()) listLength++; // draw contours
		if(this->ThereIsADispersedSpill()) listLength++; // set contours
		if(this->ThereIsADispersedSpill()) listLength++; // draw legend
		if(this->ThereIsADispersedSpill()) listLength++; // concentration table
		
		if(this->ThereIsADispersedSpill()) listLength++; // surface LEs
		if(this->ThereIsADispersedSpill()) listLength++; // water density
		if(this->ThereIsADispersedSpill()) listLength++; // mixed layer depth
		if(this->ThereIsADispersedSpill()) listLength++; // breaking wave height
		
		if(this->ThereIsADispersedSpill()) listLength++; // diagnostic string info
		if(this->ThereIsADispersedSpill()) listLength++; // droplet data
		
		if (bMoversOpen)	// should be compound mover ?
			for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
				moverList->GetListItem((Ptr)&mover, i);
				listLength += mover->GetListLength();
			}
		
	}
	
	return listLength;
}

ListItem TCompoundMap::GetNthListItem (long n, short indent, short *style, char *text)
{
	ListItem item = { this, 0, indent, 0 };
	char *p, latS[20], longS[20], timeS[32],valStr[32],val2Str[32],unitStr[32];
	long i, m, count;
	TMap *map;
	TMover *mover = 0;
	
	if (n == 0) {
		item.index = I_COMPOUNDMAPNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Compound Map");
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}
	
	item.indent++;
	n -= 1;
	
	if (bOpen)
	{
		
		/*if (--n == 0) {
		 item.index = I_COMPOUNDACTIVE;
		 item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
		 strcpy(text, "Active");
		 
		 return item;
		 }*/
		
		
		if (n == 0) {
			item.index = I_COMPOUNDMAP;
			item.bullet = bMapsOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Embedded Maps");
			
			return item;
		}
		
		n -= 1;
		indent++;
	}	
	//n = n - 1;
	if (bMapsOpen)
	{
		for (i = 0, m = mapList->GetItemCount() ; i < m ; i++) {
			mapList->GetListItem((Ptr)&map, i);
			//strcpy(text, "Embedded Maps");
			//return item;
			count = map->GetListLength();
			if (count > n)
				return map->GetNthListItem(n, indent + 1, style, text);
			
			n -= count;
		}
		
		
	}
	//indent++;
	if(this ->ThereIsADispersedSpill())
	{
		item.indent--;
		if (n == 0) {
			//item.indent++;
			item.index = I_PDRAWCONTOURS;
			//item.bullet = bShowContours ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			if (this -> fContourDepth2 == 0) 
			{
				this -> fContourDepth2 = 5.;	// maybe use mixed layer depth
			}
			if (this -> fContourDepth1 == BOTTOMINDEX)
				//sprintf(text, "Draw Contours for Bottom Layer (1 meter)");
				sprintf(text, "Draw Contours for Bottom Layer (%g meters)",fBottomRange);
			else
				sprintf(text, "Draw Contours for %g to %g meters",this->fContourDepth1,this->fContourDepth2);
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			long numLevels;
			item.index = I_PSETCONTOURS;
			if (!fContourLevelsH) 
			{
				if (!InitContourLevels()) return item;
			}
			numLevels = GetNumDoubleHdlItems(fContourLevelsH);
			sprintf(text, "Contour Levels (mg/L) : Min %g  Max %g",(*fContourLevelsH)[0],(*fContourLevelsH)[numLevels-1]);
			return item;
		}
		n -= 1;
		if (n == 0) {
			//long numLevels;
			TTriGridVel3D* triGrid = GetGrid3D(true);	// use refined grid if there is one
			
			item.index = I_PCONCTABLE;
			// code goes here, only show if output values exist, change to button...
			// Following plume vs following point
			sprintf(text, "Show concentration plots");
			if (!triGrid) *style = normal; 
			else
			{
				outputData **oilConcHdl = triGrid -> GetOilConcHdl();	
				if (!oilConcHdl)
				{
					*style = normal;
					//printError("There is no concentration data to plot");
					//err = -1;
					//return TRUE;
				}
				else
				{
					Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
					if (triSelected)
						strcat(text," at selected triangles");
					else
						strcat(text," following the plume");
					*style = italic;
				}
			}
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			item.indent++;
			item.index = I_PSHOWLEGEND;
			item.bullet = bShowLegend ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Contour Legend");
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			item.indent++;
			item.index = I_PSHOWSURFACELES;
			item.bullet = bShowSurfaceLEs ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Surface LEs");
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			//item.indent++;
			item.index = I_PWATERDENSITY;
			sprintf(text, "Water Density : %ld (kg/m^3)",fWaterDensity);
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			//item.indent++;
			item.index = I_PMIXEDLAYERDEPTH;
			sprintf(text, "Mixed Layer Depth : %g m",fMixedLayerDepth);
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			//item.indent++;
			item.index = I_PBREAKINGWAVEHT;
			//sprintf(text, "Breaking Wave Height : %g m",fBreakingWaveHeight);
			sprintf(text, "Breaking Wave Height : %g m",GetBreakingWaveHeight());
			//if (fWaveHtInput==0) 	// user input value by hand, also show wind speed?				
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			//item.indent++;
			item.index = I_PDIAGNOSTICSTRING;
			if (fDiagnosticStrType == NODIAGNOSTICSTRING)
				sprintf(text, "No Grid Diagnostics");
			else if (fDiagnosticStrType == TRIANGLEAREA)
				sprintf(text, "Show Triangle Areas (km^2)");
			else if (fDiagnosticStrType == NUMLESINTRIANGLE)
				sprintf(text, "Show Number LEs in Triangles");
			else if (fDiagnosticStrType == CONCENTRATIONLEVEL)
				sprintf(text, "Show Concentration Levels");
			else if (fDiagnosticStrType == DEPTHATCENTERS)
				sprintf(text, "Show Depth at Triangle Centers (m)");
			else if (fDiagnosticStrType == SUBSURFACEPARTICLES)
				sprintf(text, "Show Subsurface Particles");
			else if (fDiagnosticStrType == SHORELINEPTNUMS)
				sprintf(text, "Show Selected Shoreline Point Numbers");
			
			return item;
		}
		n -= 1;
		/*if (n == 0) {
		 //item.indent++;
		 item.index = I_PDROPLETINFO;
		 sprintf(text, "Droplet Data");
		 
		 return item;
		 }
		 n -= 1;*/
	}
	
	if (bOpen) {
		indent++;
		if (n == 0) {
			item.index = I_COMPOUNDMOVERS;
			item.indent = indent;
			item.bullet = bMoversOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Movers");
			
			return item;
		}
		n -= 1;
		
		if (bMoversOpen)
			for (i = 0, m = moverList->GetItemCount() ; i < m ; i++) {
				moverList->GetListItem((Ptr)&mover, i);
				count = mover->GetListLength();
				if (count > n)
				{
					item =  mover->GetNthListItem(n, indent + 1, style, text);
					if (mover->bActive) *style = italic;
					return item;
					//return mover->GetNthListItem(n, indent + 1, style, text);
				}
				
				n -= count;
			}
	}
	item.owner = 0;
	
	return item;
}

void TCompoundMap::Draw(Rect r, WorldRect view)
{
	// loop through all the current movers and draw each
	// draw each of the movers
	long i, n;
	TMap *map;
	// draw each of the maps (in reverse order to show priority)		
	//	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
	for (n = mapList->GetItemCount()-1 ; n>=0; n--) {
		//mapList->GetListItem((Ptr)&map, i);
		mapList->GetListItem((Ptr)&map, n);
		map->Draw(r, view);
	}
	TMap::Draw(r, view);	// if put all maps together could do a PtCurMap::Draw(r, view); instead
	// code goes here, want to put together the bounds from any maps?
}


///////////////////////////////////////////////////////////////////////////
/*OSErr TCompoundMap::CheckAndPassOnMessage(TModelMessage *message)
 {	// JLM
 char ourName[kMaxNameLen];
 long lVal;
 OSErr err = 0;
 
 // see if the message is of concern to us
 this->GetClassName(ourName);
 
 if(message->IsMessage(M_SETFIELD,ourName))
 {
 double val;
 char str[256];
 OSErr err = 0;
 
 ////////////////
 }
 /////////////////////////////////////////////////
 //  pass on this message to our base class
 /////////////////////////////////////////////////
 return TMap::CheckAndPassOnMessage(message);
 }*/


/////////////////////////////////////////////////

/*short CompoundMapDlgClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
 {
 short 			item;
 OSErr 			err = 0;
 TCompoundMap	*map = sSharedDialogCompoundMap;
 TMap			*currentMap=0;
 long 			i,n,m,curSelection;
 Point 			pos;
 
 switch (itemNum) {
 
 case COMPOUNDMAPDLGOK:
 {
 if(sMapList)
 {
 //DepthValuesSetH dvalsh = sgDepthValuesH;
 n = sMapList->GetItemCount();
 if(n == 0)
 {	// no items are entered, tell the user
 char msg[512],buttonName[64];
 GetWizButtonTitle_Cancel(buttonName);
 sprintf(msg,"You have not entered any data values.  Either enter data values and use the 'Add New Record' button, or use the '%s' button to exit the dialog.",buttonName);
 printError(msg);
 break;
 }
 
 // check that all the values are in range - if there is some range
 // or may allow the user to change units
 for(i=0;i<n;i++)
 {
 char errStr[256] = "";
 err=sMapList->GetListItem((Ptr)&map,i);
 if(err) {SysBeep(5); break;}// this shouldn't ever happen
 }
 
 /////////////
 // point of no return
 //////////////
 m = sDialogMapList->GetItemCount() ;
 
 sDialogMapList->ClearList();
 for(i=0;i<n;i++)
 {
 if(err=sMapList->GetListItem((Ptr)&currentMap,i))return COMPOUNDMAPDLGOK;
 err=sDialogMapList->AppendItem((Ptr)&currentMap);
 if(err)return err;
 }
 }
 ////////////////////////////////////////////////////////////
 DisposeCOMPOUNDMAPDLGStuff();
 return COMPOUNDMAPDLGOK;
 }
 break;
 
 case COMPOUNDMAPDLGCANCEL:
 // delete any new patterns, restore original patterns
 //DeleteIfNotOriginalPattern(mover -> pattern1);
 //DeleteIfNotOriginalPattern(mover -> pattern2);
 //mover -> pattern1 = sSharedOriginalPattern1;
 //mover -> pattern2 = sSharedOriginalPattern2;
 
 DisposeCOMPOUNDMAPDLGStuff();
 return COMPOUNDMAPDLGCANCEL;
 break;
 
 
 case COMPOUNDMAPDLGDELETEALL:
 sMoverList->ClearList();
 VLReset(&sgObjects,1);
 UpdateDisplayWithCurSelection(dialog);
 break;
 case COMPOUNDMAPDLGMOVEUP:
 if (VLGetSelect(&curSelection, &sgObjects))
 {
 if (curSelection>0) 
 {
 sMapList->SwapItems(curSelection,curSelection-1);
 VLSetSelect(curSelection-1,&sgObjects);
 --curSelection;
 }
 }
 VLUpdate(&sgObjects);
 //VLReset(&sgObjects,1);
 UpdateDisplayWithCurSelection(dialog);
 break;
 case COMPOUNDMAPDLGDELETEROWS_BTN:
 if (VLGetSelect(&curSelection, &sgObjects))
 {
 sMapList->DeleteItem(curSelection);
 VLDeleteItem(curSelection,&sgObjects);
 if(sgObjects.numItems == 0)
 {
 VLAddItem(1,&sgObjects);
 VLSetSelect(0,&sgObjects);
 }
 --curSelection;
 if(curSelection >-1)
 {
 VLSetSelect(curSelection,&sgObjects);
 }
 VLUpdate(&sgObjects);
 }
 UpdateDisplayWithCurSelection(dialog);
 break;
 
 case COMPOUNDMAPDLGREPLACE:
 //err = RetrieveIncrementDepth(dialog);
 {
 TMap* curMap = sSharedDialogCompoundMap->AddNewMap(&err);
 //sMoverList = 
 if(err) break;
 if (VLGetSelect(&curSelection, &sgObjects))
 {
 //TMover* thisCurMover=0;
 //err=GetDepthVals(dialog,&dvals);
 if(err) break;
 
 if(curSelection==sMapList->GetItemCount())
 {
 // replacing blank record
 err = AddReplaceRecord(dialog,!REPLACE,(TMap*)curMap);
 if (!err) SelectNthRow(dialog, curSelection+1 ); 
 }
 else // replacing existing record
 {
 VLGetSelect(&curSelection,&sgObjects);
 sMapList->DeleteItem(curSelection);
 VLDeleteItem(curSelection,&sgObjects);		
 err = AddReplaceRecord(dialog,REPLACE,(TMap*)curMap);
 }
 }
 }
 break;
 
 case COMPOUNDMAPDLGLIST:
 // retrieve every time they click on the list
 // because clicking can cause the increment to be hidden
 // and we need to verify it before it gets hidden
 //err = RetrieveIncrementDepth(dialog);
 //if(err) break;
 ///////////
 pos=GetMouseLocal(GetDialogWindow(dialog));
 VLClick(pos, &sgObjects);
 VLUpdate(&sgObjects);
 VLGetSelect(&curSelection,&sgObjects);
 if(curSelection == -1 )
 {
 curSelection = sgObjects.numItems-1;
 VLSetSelect(curSelection,&sgObjects);
 VLUpdate(&sgObjects);
 }
 
 //ShowHideAutoIncrement(dialog,curSelection);
 // moved into UpdateDisplayWithCurSelection()
 
 //if (AddRecordRowIsSelected2())
 if (VLAddRecordRowIsSelected(&sgObjects))
 {
 //DepthValuesSet dvals;
 TMap* map=0;
 sMapList->GetListItem((Ptr)&map,sMapList->GetItemCount()-1);
 //err = RetrieveIncrementDepth(dialog);
 if(err) break;
 //IncrementDepth(dialog,dvals.depth);
 }
 UpdateDisplayWithCurSelection(dialog);
 break;
 
 
 }
 
 return 0;
 }*/

/*void DrawMapNameList(DialogPtr w, RECTPTR r, long n)
 {
 char s[256];
 //DepthValuesSet dvals;
 TMap* map = 0;
 
 if(n == sgObjects.numItems-1)
 {
 strcpy(s,"****");
 MyMoveTo(NAME_COL-stringwidth(s)/2,r->bottom);
 drawstring(s);
 //MyMoveTo(U_COL-stringwidth(s)/2,r->bottom);
 //drawstring(s);
 return; 
 }
 
 sMapList->GetListItem((Ptr)&map,n);
 if (map) 
 strcpy(s, map->className);
 else strcpy(s,"****");	// shouldn't happen
 //MyMoveTo(NAME_COL-stringwidth(s)/2,r->bottom);
 MyMoveTo(NAME_COL-stringwidth("Pattern Name")/2,r->bottom);
 drawstring(s);
 
 //StringWithoutTrailingZeros(s,dvals.depth,1);
 //MyMoveTo(DEPTH_COL-stringwidth(s)/2,r->bottom);
 //drawstring(s);
 
 
 return;
 }
 
 
 pascal_ifMac void MoverNamesListUpdate(DialogPtr dialog, short itemNum)
 {
 Rect r = GetDialogItemBox(dialog,COMPOUNDDLGLIST);
 
 VLUpdate(&sgObjects);
 }
 
 OSErr CompoundMapDlgInit(DialogPtr dialog, VOIDPTR data)
 {
 char 			s[256];
 TCompoundMap	*compoundMap;
 Rect r = GetDialogItemBox(dialog, COMPOUNDMAPDLGLIST);
 char blankStr[32];
 strcpy(blankStr,"");
 TMap *currentMap=0;
 long i,n;
 OSErr err = 0;
 short IBMoffset;
 
 compoundMap = sSharedDialogCompoundMap;
 if (!compoundMapr) return -1;
 sSharedCompoundUncertainyInfo = compoundMap -> GetCurrentUncertaintyInfo();
 
 memset(&sgObjects,0,sizeof(sgObjects));// JLM 12/10/98
 
 {
 sMapList = new CMyList(sizeof(TMover *));
 if(!sMapList)return -1;
 if(sMapList->IList())return -1;
 
 n = sDialogList->GetItemCount() ;
 if (n>0)
 {
 // copy list to temp list
 for(i=0;i<n;i++)
 {	// dvals is a list too in this case...
 //dvals=(*dvalsh)[i];
 sDialogMapList->GetListItem((Ptr)&currentMap, i);
 err=sMapList->AppendItem((Ptr)&currentMap);
 if(err)return err;
 }
 }
 else  n=0;
 
 n++; // Always have blank row at bottom
 
 err = VLNew(dialog, COMPOUNDMAPDLGLIST, &r,n, DrawMapNameList, &sgObjects);
 if(err) return err;
 }
 //RegisterPopTable (compoundIntDlgPopTable, sizeof (compoundIntDlgPopTable) / sizeof (PopInfoRec));
 //RegisterPopUpDialog (COMPOUNDDLG, dialog);
 
 SetDialogItemHandle(dialog, COMPOUNDMAPDLGHILITEDEFAULT, (Handle)FrameDefault);
 SetDialogItemHandle(dialog, COMPOUNDMAPDLGLIST, (Handle)MoverNamesListUpdate);
 
 #ifdef IBM
 IBMoffset = r.left;
 #else 
 IBMoffset = 0;
 #endif
 r = GetDialogItemBox(dialog, COMPOUNDMAPNAME_LIST_LABEL);NAME_COL=(r.left+r.right)/2-IBMoffset;
 
 //ShowUnscaledComponentValues(dialog,nil,nil);
 ShowHideTCompoundDialogItems(dialog);
 UpdateDisplayWithCurSelection(dialog);
 
 return 0;
 }*/

/*OSErr CompoundMapSettingsDialog(TCompoundMap *theMap)
 {
 short item;
 
 if (!theMap) return -1;
 
 sSharedDialogCompoundMap = theMap;
 sDialogMapList = theMap->maprList;
 item = MyModalDialog(COMPOUNDMAPDLG, mapWindow, 0, CompoundMapDlgInit, CompoundMapDlgClick);
 sSharedDialogCompoundMap = 0;
 
 if(COMPOUNDMAPDLGOK == item)	model->NewDirtNotification();// tell model about dirt
 return COMPOUNDMAPDLGOK == item ? 0 : -1;
 }*/

/*TMap* TCompoundMap::AddNewMap(OSErr *err)	
 {	// this should be used when Load is hit, or after ok
 *err = 0;
 
 {
 TMap *newMap = 0;
 TCurrentMover *newMover = CreateAndInitCurrentsMover (this->moverMap,true,0,0,&newMap);
 if (newMover)
 {
 switch (newMover->GetClassID()) 
 {
 case TYPE_CATSMOVER:
 case TYPE_TIDECURCYCLEMOVER:
 case TYPE_CATSMOVER3D:
 *err = CATSSettingsDialog ((TCATSMover*)newMover, this->moverMap, &timeFileChanged);
 break;
 case TYPE_NETCDFMOVER:
 case TYPE_NETCDFMOVERCURV:
 case TYPE_NETCDFMOVERTRI:
 case TYPE_GRIDCURMOVER:
 case TYPE_PTCURMOVER:
 case TYPE_TRICURMOVER:
 *err = newMover->SettingsDialog();
 break;
 default:
 printError("bad type in TCompoundMover::AddCurrent");
 break;
 }
 if(*err)	{ newMover->Dispose(); delete newMover; newMover = 0;}
 
 if (newMap) newMap->Dispose(); delete newMap; newMap = 0; //  need to track all maps for boundaries in a list
 if(newMover && !(*err))
 {	
 //if (!newMap) 
 {	// probably don't allow new map...
 //err = AddMoverToMap (this->moverMap, timeFileChanged, newMover);
 *err = AddMover (newMover,0);
 if (*err)
 {
 newMover->Dispose(); delete newMover; newMover = 0;
 return 0;
 }
 if (!(*err) && this->moverMap == model -> uMap)
 {
 //ChangeCurrentView(AddWRectBorders(((PtCurMover*)newMover)->fGrid->GetBounds(), 10), TRUE, TRUE);
 WorldRect newRect = newMover->GetGridBounds();
 ChangeCurrentView(AddWRectBorders(newRect, 10), TRUE, TRUE);	// so current loaded on the universal map can be found
 }
 newMover->bIAmPartOfACompoundMover = true;
 }
 else
 {
 float arrowDepth = 0;
 MySpinCursor();
 err = model -> AddMap(newMap, 0);
 if (!err) err = AddMoverToMap(newMap, timeFileChanged, newMover);
 //if (!err) err = ((PtCurMap*)newMap)->MakeBitmaps();
 if (!err) newMover->SetMoverMap(newMap);
 else
 {
 newMap->Dispose(); delete newMap; newMap = 0; 
 newMover->Dispose(); delete newMover; newMover = 0;
 return 0; 
 }
 //if (newMover->IAm(TYPE_CATSMOVER3D) || newMover->IAm(TYPE_TRICURMOVER)) InitAnalysisMenu();
 if (model->ThereIsA3DMover(&arrowDepth)) InitAnalysisMenu();	// want to have it come and go?
 MySpinCursor();
 }
 }	
 return newMover;
 }
 else {*err = -1; return 0;}
 //break;
 }
 
 
 //default: 
 //SysBeep (5); 
 //return -1;
 
 return 0;
 }*/

OSErr TCompoundMap::AddMap(TMap *theMap, short where)
{
	OSErr err = 0;
	if (!mapList) return -1;
	
	if (err = mapList->AppendItem((Ptr)&theMap))
	{ TechError("TCompoundMap::AddMap()", "AppendItem()", err); return err; }
	
	SetDirty (true);
	
	SelectListItemOfOwner(theMap);
	//SelectListItemOfOwner(this);
	
	return 0;
}

OSErr TCompoundMap::DropMap(TMap *theMap)
{
	long 	i;
	OSErr	err = noErr;
	
	if (mapList->IsItemInList((Ptr)&theMap, &i))
	{
		if (err = mapList->DeleteItem(i))
		{ TechError("TCompoundMap::DropMap()", "DeleteItem()", err); return err; }
	}
	SetDirty (true);
	
	return err;
}

#define TCompoundMap_FileVersion 1
OSErr TCompoundMap::Write(BFPB *bfpb)
{
	long i,n,version = TCompoundMap_FileVersion;
	ClassID id = GetClassID ();
	OSErr err = 0;
	char c;
	TMap		*thisMap;
	
	//if (err = TMap::Write(bfpb)) return err;
	if (err = PtCurMap::Write(bfpb)) return err;
	
	// save class fields
	
	StartReadWriteSequence("TCompoundMap::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	if (err = WriteMacValue(bfpb,bMapsOpen)) return err;
	
	n = mapList->GetItemCount();		// map count
	if (err = WriteMacValue(bfpb,n)) return err;
	
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) {
		mapList->GetListItem((Ptr)&thisMap, i);
		id = thisMap->GetClassID();
		if (err = WriteMacValue(bfpb,id)) return err;
		if (err = thisMap->Write(bfpb)) return err;
	}
	
	return 0;
}

OSErr TCompoundMap::Read(BFPB *bfpb)
{
	long i,n,version,numMaps,numMovers=0;
	OSErr err = 0;
	ClassID id;
	char c;
	TMap		*thisMap;
	TMover* compoundMover = 0;
	TMover* mover = 0;
	
	//if (err = TMap::Read(bfpb)) return err;
	if (err = PtCurMap::Read(bfpb)) return err;
	
	StartReadWriteSequence("TCompoundMap::Write()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TCompoundMap::Read()", "id != GetClassID", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > TCompoundMap_FileVersion) { printSaveFileVersionError(); return -1; }
	
	if (err = ReadMacValue(bfpb, &bMapsOpen)) return err;
	if (err = ReadMacValue(bfpb,&numMaps)) return err;
	
	compoundMover = GetMover(TYPE_COMPOUNDMOVER);
	if (compoundMover) numMovers = (dynamic_cast<TCompoundMover*>(compoundMover))->moverList->GetItemCount();
	for (i = 0 ; i < numMaps ; i++) {
		if (err = ReadMacValue(bfpb,&id)) return err;
		switch (id) {
				//case TYPE_MAP: thisMap = new TMap("", voidWorldRect); break;
				//case TYPE_OSSMMAP: thisMap = (TMap *)new TOSSMMap("", voidWorldRect); break;
				//case TYPE_VECTORMAP: thisMap = (TMap *)new TVectorMap ("", voidWorldRect); break;
			case TYPE_PTCURMAP: thisMap = (TMap *)new PtCurMap ("", voidWorldRect); break;
				
			default: printError("Unrecognized map type in TCompoundMap::Read()."); return -1;
		}
		if (!thisMap) { TechError("TCompoundMap::Read()", "new TMap", 0); return -1; };
		if (err = thisMap->InitMap()) return err;
		if (err = thisMap->Read(bfpb)) return err;
		if (err = AddMap(thisMap, 0))
		{ delete thisMap; TechError("TCompoundMap::Read()", "AddMap()", err); return err; };
		
		if (numMaps==numMovers)	// if not we will have to write out a mapping
		{
			(dynamic_cast<TCompoundMover*>(compoundMover))->moverList->GetListItem((Ptr)&mover, i);
			mover->SetMoverMap(thisMap);
		}
	}
	
	return 0;
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////

/*void DrawFilledWaterTriangles(void * object,WorldRect wRect,Rect r)
 {
 PtCurMap* ptCurMap = (PtCurMap*)object; // typecast
 TTriGridVel* triGrid = 0;	// use GetGrid??
 
 
 triGrid = ptCurMap->GetGrid(true);	// use refined grid if there is one
 if(triGrid) {
 // draw triangles as filled polygons
 triGrid->DrawBitMapTriangles(r);
 }
 return;
 }
 
 static Boolean drawingLandBitMap;
 void DrawWideLandSegments(void * object,WorldRect wRect,Rect r)
 {
 PtCurMap* ptCurMap = (PtCurMap*)object; // typecast
 
 // draw land boundaries as wide lines
 drawingLandBitMap = TRUE;
 ptCurMap -> DrawBoundaries(r);
 drawingLandBitMap = FALSE;
 }
 
 
 OSErr TCompoundMap::MakeBitmaps()
 {
 OSErr err = 0;
 TCurrentMover *mover=0;
 
 mover = Get3DCurrentMover();	
 if (!mover) return -1;	
 
 { // make the bitmaps etc
 Rect bitMapRect;
 long bmWidth, bmHeight;
 WorldRect wRect = this -> GetMapBounds();
 err = LandBitMapWidthHeight(wRect,&bmWidth,&bmHeight);
 if (err) goto done;
 MySetRect (&bitMapRect, 0, 0, bmWidth, bmHeight);
 fWaterBitmap = GetBlackAndWhiteBitmap(DrawFilledWaterTriangles,this,wRect,bitMapRect,&err);
 if(err) goto done;
 fLandBitmap = GetBlackAndWhiteBitmap(DrawWideLandSegments,this,wRect,bitMapRect,&err); 
 if(err) goto done;
 
 }
 done:	
 if(err)
 {
 #ifdef MAC
 DisposeBlackAndWhiteBitMap (&fWaterBitmap);
 DisposeBlackAndWhiteBitMap (&fLandBitmap);
 #else
 if(fWaterBitmap) DestroyDIB(fWaterBitmap);
 fWaterBitmap = 0;
 if(fLandBitmap) DestroyDIB(fLandBitmap);
 fLandBitmap = 0;
 #endif
 }
 return err;
 }
 */
OSErr TCompoundMap::AddMover(TMover *theMover, short where)
{
	OSErr err = 0;
	
	err = TMap::AddMover(theMover,where);
	return err;
}


/*OSErr TCompoundMap::ReplaceMap()	// code goes here, maybe not for NetCDF?
 {
 char 		path[256], nameStr [256];
 short 		gridType;
 OSErr		err = noErr;
 Point 		where = CenteredDialogUpLeft(M38b);
 TCompoundMap 	*map = nil;
 OSType 	typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
 MySFReply 	reply;
 
 #if TARGET_API_MAC_CARBON
 mysfpgetfile(&where, "", -1, typeList,
 (MyDlgHookUPP)0, &reply, M38b, MakeModalFilterUPP(STDFilter));
 if (!reply.good) return USERCANCEL;
 strcpy(path, reply.fullPath);
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
 #else
 strcpy(path, reply.fName);
 #endif
 #endif
 if (IsPtCurFile (path))
 {
 TMap *newMap = 0;
 TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,"ptcurfile",&newMap);	// already have path
 
 if (newMover)
 {
 PtCurMover *ptCurMover = (PtCurMover*)newMover;
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
 newMover->SetMoverMap(newMap);
 }
 }
 }
 map = (TCompoundMap *)newMap;
 }
 else if (IsNetCDFFile (path, &gridType))
 {
 TMap *newMap = 0;
 char s[256],fileName[256];
 strcpy(s,path);
 SplitPathFile (s, fileName);
 strcat (nameStr, fileName);
 TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,fileName,&newMap);	// already have path
 
 if (newMover && newMap)
 {
 NetCDFMover *netCDFMover = (NetCDFMover*)newMover;
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
 newMover->SetMoverMap(newMap);
 }
 }
 }
 else
 {
 printError("NetCDF file must include a map.");
 if (newMover) {newMover->Dispose(); delete newMover; newMover = 0;}
 if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} // shouldn't happen
 return USERCANCEL;
 }
 map = (TCompoundMap *)newMap;
 }
 else 
 {
 printError("New map must be of the same type.");
 return USERCANCEL;	// to return to the dialog
 }
 {
 // put movers on the new map and activate
 TMover *thisMover = nil;
 Boolean	timeFileChanged = false;
 long k, d = this -> moverList -> GetItemCount ();
 for (k = 0; k < d; k++)
 {
 this -> moverList -> GetListItem ((Ptr) &thisMover, 0); // will always want the first item in the list
 if (!thisMover->IAm(TYPE_PTCURMOVER) && !thisMover->IAm(TYPE_NETCDFMOVERCURV) && !IAm(TYPE_NETCDFMOVERTRI) )
 {
 if (err = AddMoverToMap(map, timeFileChanged, thisMover)) return err; 
 thisMover->SetMoverMap(map);
 }
 if (err = this->DropMover(thisMover)) return err; // gets rid of first mover, moves rest up
 }
 if (err = model->DropMap(this)) return err;
 model->NewDirtNotification();
 }
 
 return err;
 
 }*/


Boolean TCompoundMap::InMap(WorldPoint p)
{
	WorldRect ourBounds = this -> GetMapBounds(); 
	long i,n;
	TMap *map = 0;
	
	if (!WPointInWRect(p.pLong, p.pLat, &ourBounds))
		return false;
	
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) 
	{
		mapList->GetListItem((Ptr)&map, i);
		if (map -> InMap(p)) return true;
	}
	
	return false;
}


long TCompoundMap::GetLandType(WorldPoint p)
{
	// This isn't used at the moment
	long i,n,landType;
	TMap *map = 0;
	//	Boolean onLand = IsBlackPixel(p,ourBounds,fLandBitmap);
	//	Boolean inWater = IsBlackPixel(p,ourBounds,fWaterBitmap);
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) 
	{
		mapList->GetListItem((Ptr)&map, i);
		landType = map->GetLandType(p);
		if (!(landType==LT_UNDEFINED)) return landType;
	}
	return LT_UNDEFINED;
	
}


Boolean TCompoundMap::OnLand(WorldPoint p)
{	// here might want to check if in water in any map instead or onland or offmap for all maps
	WorldRect ourBounds = this -> GetMapBounds(); 
	Boolean onLand = false;
	Boolean onSomeMapsLand = false;
	Boolean inSomeMapsWater = false;
	//TTriGridVel* triGrid = GetGrid(false);	// don't think need 3D here
	//TDagTree *dagTree = triGrid->GetDagTree();
	long i,n;
	TMap *map = 0;
	
	if (!WPointInWRect(p.pLong, p.pLat, &ourBounds)) return false; // off map is not on land
	
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) 
	{
		mapList->GetListItem((Ptr)&map, i);
		if (map -> OnLand(p)) return true;	// need to check if in higher priority map 
		else if (map->InMap(p)) return false;
		//if (map -> OnLand(p)) onSomeMapsLand = true;
		//else if (((PtCurMap*)map)->InWater(p)) inSomeMapsWater = true;
	}
	
	//if (onSomeMapsLand && !inSomeMapsWater) onLand = true;	//  how to handle mismatched boundaries?
	return onLand;
}


WorldPoint3D	TCompoundMap::MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed)
{
	// check every pixel along the line it makes on the water bitmap
	// for any non-water point check the land bitmap as well and if it crosses a land boundary
	// force the point to the closest point in the bounds
	/*#ifdef MAC
	 BitMap bm = fWaterBitmap;
	 #else
	 HDIB bm = fWaterBitmap;
	 #endif*/
	
	// need to do this for each map and if offmap check the next map
	
	long i,n,fromPtMapIndex,toPtMapIndex,newMapIndex;
	double depthAtPt;
	TMap *map = 0;
	WorldPoint3D checkedPt, checkedPt2;
	
	toPtMapIndex = WhichMapIsPtIn(toWPt.p);
	fromPtMapIndex = WhichMapIsPtIn(fromWPt.p);
	
	Boolean LEsOnSurface = (fromWPt.z == 0 && toWPt.z == 0);
	
	if (toWPt.z == 0 && !isDispersed) LEsOnSurface = true;
	
	if (fromPtMapIndex == -1)
	{
		//this should be an error
		return toWPt;
	}
	if (LEsOnSurface)
	{
		mapList->GetListItem((Ptr)&map, fromPtMapIndex);
		//if (map->InMap(fromWPt) && map->InMap(toWPt))
		// if fromWPt is not in map ?
		checkedPt = (dynamic_cast<PtCurMap *>(map))->MovementCheck(fromWPt,toWPt,isDispersed);	// issue if have to reflect from one map to another...
		if (checkedPt.p.pLat == toWPt.p.pLat && checkedPt.p.pLong == toWPt.p.pLong)
			return checkedPt;
		/*else
		 {
		 if (map->OnLand(checkedPt.p))
		 {
		 if (this->OnLand(checkedPt.p))
		 return checkedPt;
		 }
		 return fromWPt;
		 }*/
		//return checkedPt;
	}
	if (toPtMapIndex == fromPtMapIndex && toPtMapIndex != -1)
	{	// staying on same map
		mapList->GetListItem((Ptr)&map, toPtMapIndex);
		//if (map->InMap(fromWPt) && map->InMap(toWPt))
		// if fromWPt is not in map ?
		checkedPt = (dynamic_cast<PtCurMap *>(map))->MovementCheck(fromWPt,toWPt,isDispersed);	// issue if have to reflect from one map to another...
		if (LEsOnSurface) return checkedPt;
		if (map -> InMap(checkedPt.p) && !map->OnLand(checkedPt.p)) 
		{
			if (!this->OnLand(checkedPt.p)) 
				return checkedPt;
			else
				return fromWPt;
		}
		else
			return fromWPt;
	}
	if (toPtMapIndex == -1 && fromPtMapIndex != -1)
	{
		mapList->GetListItem((Ptr)&map, fromPtMapIndex);
		//if (map->InMap(fromWPt) && map->InMap(toWPt))
		// if fromWPt is not in map ?
		checkedPt = (dynamic_cast<PtCurMap *>(map))->MovementCheck(fromWPt,toWPt,isDispersed);	// issue if have to reflect from one map to another...
		if (LEsOnSurface) return checkedPt;
		if (map -> InMap(checkedPt.p)  && !map->OnLand(checkedPt.p)) /*return checkedPt;*/
		{
			if (!this->OnLand(checkedPt.p)) 
				return checkedPt;
			else
				return fromWPt;
		}
		else
			return fromWPt;
	}
	if (toPtMapIndex != fromPtMapIndex && toPtMapIndex  != -1 && fromPtMapIndex != -1)
	{
		// code goes here, special version of MovementCheck that returns first point off map, then use that as fromWPt on next map
		// what if crossed over more than one map??
		
		// for subsurface points
		// if point is back in original map it has been reflected and can be returned
		// if point is on new map need to check depth
		// if off maps or beached something has gone wrong
		// for surface points, LEs should beach on original map or move to other map or cross off water boundary
		if (LEsOnSurface)
		{
			mapList->GetListItem((Ptr)&map, fromPtMapIndex);
			checkedPt = (dynamic_cast<PtCurMap *>(map))->MovementCheck(fromWPt,toWPt,isDispersed);	// issue if have to reflect from one map to another...
			return checkedPt;
		}
		
		else
		{
			mapList->GetListItem((Ptr)&map, fromPtMapIndex);
			checkedPt = (dynamic_cast<PtCurMap *>(map))->MovementCheck(fromWPt,toWPt,isDispersed);	// issue if have to reflect from one map to another...
			return checkedPt;
		}
		/*mapList->GetListItem((Ptr)&map, fromPtMapIndex);
		 checkedPt = ((PtCurMap*)map)->MovementCheck2(fromWPt,toWPt,isDispersed);	// issue if have to reflect from one map to another...
		 newMapIndex = WhichMapIsPtIn(checkedPt.p);
		 if (newMapIndex != -1)
		 {
		 mapList->GetListItem((Ptr)&map, newMapIndex);
		 checkedPt2 = ((PtCurMap*)map)->MovementCheck2(checkedPt,toWPt,isDispersed);	// issue if have to reflect from one map to another...
		 if (newMapIndex == toPtMapIndex)
		 {
		 }
		 else			
		 {
		 newMapIndex = WhichMapIsPtIn(checkedPt2.p);
		 mapList->GetListItem((Ptr)&map, newMapIndex);
		 checkedPt = ((PtCurMap*)map)->MovementCheck2(checkedPt2,toWPt,isDispersed);	// issue if have to reflect from one map to another...
		 }
		 }
		 else
		 {
		 //
		 }
		 */
		mapList->GetListItem((Ptr)&map, toPtMapIndex);
		if (LEsOnSurface) return toWPt;
		
		if (!LEsOnSurface &&  (!(dynamic_cast<PtCurMap *>(map))->InVerticalMap(toWPt) || toWPt.z == 0))  
		{
			/*double*/ depthAtPt = map->DepthAtPoint(toWPt.p);	// code goes here, a check on return value
			//if (depthAtPt < 0) 
			if (depthAtPt <= 0) 
			{
				//OSErr err = 0;
				return fromWPt;	// some sort of problem
			}
			//if (depthAtPt==0)
			//toWPt.z = .1;
			
			if (toWPt.z > depthAtPt) toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
			//if (toWPt.z > depthAtPt) toWPt.z = GetRandomFloat(.7*depthAtPt,.99*depthAtPt);
			
			if (toWPt.z <= 0) toWPt.z = GetRandomFloat(.01*depthAtPt,.1*depthAtPt);
			
			//movedPoint.z = fromWPt.z;	// try not changing depth
			//if (!InVerticalMap(toWPt))
			//toWPt.p = fromWPt.p;	// use original point - code goes here, need to find a z in the map
		}
		if (map->InMap(toWPt.p) && !map->OnLand(toWPt.p)) /*return toWPt;*/	// will need to check the transition from one map to the other, at least check z is in vertical map
			//if (map -> InMap(checkedPt.p)  && !map->OnLand(checkedPt.p)) return checkedPt;
		{
			if (!this->OnLand(toWPt.p)) 
				return toWPt;
			else
				return fromWPt;
		}
		else
			return fromWPt;
	}
	/*for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) 
	 {	// if maps overlap should be able to do crossing
	 mapList->GetListItem((Ptr)&map, i);
	 //if (map->InMap(fromWPt) && map->InMap(toWPt))
	 // if fromWPt is not in map ?
	 checkedPt = map->MovementCheck(fromWPt,toWPt,isDispersed);	// issue if have to reflect from one map to another...
	 if (map -> InMap(checkedPt.p)) return checkedPt;
	 }*/
	return toWPt;	// means off all maps	
}


void  TCompoundMap::FindNearestBoundary(Point where, long *verNum, long *segNo)
{
	long startVer = 0,i,jseg;
	WorldPoint wp = ScreenToWorldPoint(where, MapDrawingRect(), settings.currentView);
	WorldPoint wp2;
	LongPoint lp;
	long lastVer = GetNumBoundaryPts();
	//long nbounds = GetNumBoundaries();
	long nSegs = GetNumBoundarySegs();	
	float wdist = LatToDistance(ScreenToWorldDistance(4));
	LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
	if(!ptsHdl) return;
	*verNum= -1;
	*segNo =-1;
	for(i = 0; i < lastVer; i++)
	{
		//wp2 = (*gVertices)[i];
		lp = (*ptsHdl)[i];
		wp2.pLat = lp.v;
		wp2.pLong = lp.h;
		
		if(WPointNearWPoint(wp,wp2 ,wdist))
		{
			//for(jseg = 0; jseg < nbounds; jseg++)
			for(jseg = 0; jseg < nSegs; jseg++)
			{
				if(i <= (*fBoundarySegmentsH)[jseg])
				{
					*verNum  = i;
					*segNo = jseg;
					break;
				}
			}
		}
	} 
}

/**************************************************************************************************/
void TCompoundMap::DrawContourScale(Rect r, WorldRect view)
{	// new version combines triangle area and contours
	Point		p;
	short		h,v,x,y,dY,widestNum=0;
	RGBColor	rgb;
	Rect		rgbrect, legendRect = fLegendRect;
	char 		numstr[30],numstr2[30],numstr3[30],text[30],titleStr[40],unitStr[40];
	long 		i,numLevels,strLen;
	double	minLevel, maxLevel;
	double 	value, value2=0, totalArea = 0, htScale = 1., wScale = 1.;
	
	//TCurrentMover *mover = Get3DCurrentMover();
	TCurrentMover *mover = Get3DCurrentMoverFromIndex(0);	// do for all?
	//if ((mover) && mover->IAm(TYPE_TRICURMOVER)) {((TriCurMover*)mover)->DrawContourScale(r,view);}
	if ((mover) && mover->IAm(TYPE_NETCDFMOVER)) {(dynamic_cast<NetCDFMover*>(mover))->DrawContourScale(r,view);}
	// could just go back to PtCurMap after this...
	if (!this->ThereIsADispersedSpill()) return;
	SetRGBColor(&rgb,0,0,0);
	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
#ifdef IBM
	float mapWidth, mapHeight, rWidth, rHeight, legendWidth, legendHeight;
	TextFont(kFontIDGeneva); TextSize(6);
#endif
	if (!fContourLevelsH) 
	{
		if (InitContourLevels()==-1) return;
	}
	
	if (gSavingOrPrintingPictFile)	// on Windows, saving does not use this
	{
		Rect mapRect;
#ifdef MAC
		mapRect = DrawingRect(settings.listWidth + 1, RIGHTBARWIDTH);
		if (!EqualRects(r,mapRect))
		{
			Boolean bCloserToTop = (legendRect.top - mapRect.top) <= (mapRect.bottom - legendRect.bottom);
			Boolean bCloserToLeft = (legendRect.left - mapRect.left) <= (mapRect.right - legendRect.right);
			if (bCloserToTop)
			{
				legendRect.top = legendRect.top - mapRect.top + r.top;
				legendRect.bottom = legendRect.bottom - mapRect.top + r.top;
			}
			else
			{
				legendRect.top = r.bottom - (mapRect.bottom - legendRect.top);
				legendRect.bottom = r.bottom - (mapRect.bottom - legendRect.bottom);
			}
			if (bCloserToLeft)
			{
				legendRect.left = legendRect.left - mapRect.left + r.left;
				legendRect.right = legendRect.right - mapRect.left + r.left;
			}
			else
			{
				legendRect.left = r.right - (mapRect.right - legendRect.left);
				legendRect.right = r.right - (mapRect.right - legendRect.right);
			}
			// should check if legend is too big for size of printout
			if (legendRect.bottom > r.bottom) {legendRect.top -= (legendRect.bottom - r.bottom); legendRect.top -= (legendRect.bottom - r.bottom);}
			if (legendRect.right > r.right) {legendRect.left -= (legendRect.right - r.right); legendRect.right -= (legendRect.right - r.right);}
		}
#else
		mapRect = DrawingRect(0,settings.listWidth);
		mapRect.top = mapRect.top - TOOLBARHEIGHT;
		mapRect.bottom = mapRect.bottom - TOOLBARHEIGHT;
		mapWidth = mapRect.right - mapRect.left;
		rWidth = r.right - r.left;
		legendWidth = legendRect.right - legendRect.left;
		mapHeight = mapRect.bottom - mapRect.top;
		rHeight = r.bottom - r.top;
		legendHeight = legendRect.bottom - legendRect.top;
		if (!EqualRects(r,mapRect))
		{
			htScale = rHeight / mapHeight; wScale = rWidth / mapWidth;
			legendRect.left = r.left + (legendRect.left - mapRect.left) * rWidth / mapWidth;
			legendRect.right = legendRect.left + legendWidth * rWidth / mapWidth;
			legendRect.top = r.top + (legendRect.top - mapRect.top) * rHeight / mapHeight;
			legendRect.bottom = legendRect.top + legendHeight * rHeight / mapHeight;
			// should check if legend is too big for size of printout
			if (legendRect.bottom > r.bottom) {legendRect.top -= (legendRect.bottom - r.bottom); legendRect.top -= (legendRect.bottom - r.bottom);}
			if (legendRect.right > r.right) {legendRect.left -= (legendRect.right - r.right); legendRect.right -= (legendRect.right - r.right);}
		}
#endif
	}
	else
	{
		if (EmptyRect(&legendRect)||!RectInRect2(&legendRect,&r))	// otherwise printing or saving - set a global?
		{
			legendRect.top = r.top;
			legendRect.left = r.left;
			legendRect.bottom = r.top + 120*htScale;	// reset after contour levels drawn
			legendRect.right = r.left + 80*wScale;	// reset if values go beyond default width
			if (fTriAreaArray) legendRect.right += 20*wScale;	// reset if values go beyond default width
		}
	}
	rgbrect = legendRect;
	EraseRect(&rgbrect);
	
	x = (rgbrect.left + rgbrect.right) / 2;
	//dY = RectHeight(rgbrect) / 12;
	dY = 10*htScale;
	y = rgbrect.top + dY / 2;
	MyMoveTo(rgbrect.left+20*wScale,y+dY);
	if (fTriAreaArray) 
	{
		drawstring("Conc.");
		MyMoveTo(rgbrect.left+80*wScale,y+dY);			
		drawstring("Area");
		MyMoveTo(rgbrect.left+20*wScale,y+2*dY);
		drawstring(" ppm");
		MyMoveTo(rgbrect.left+80*wScale,y+2*dY);
		drawstring("km^2");
		widestNum = 80*wScale+stringwidth("Area");
	}
	else
	{
		drawstring("Conc. (ppm)");		
		widestNum = 20*wScale+stringwidth("Conc. (ppm)");
	}
	numLevels = GetNumDoubleHdlItems(fContourLevelsH);
	v = rgbrect.top+40*htScale;
	if (!fTriAreaArray) v -= 10*htScale;
	h = rgbrect.left;
	for (i=0;i<numLevels;i++)
	{
		//float colorLevel = .8*float(i)/float(numLevels-1);
		float colorLevel = float(i)/float(numLevels-1);
		value = (*fContourLevelsH)[i];
		
		if (fTriAreaArray) 
		{
			value2 = fTriAreaArray[i];
			totalArea += value2;
		}
		MySetRect(&rgbrect,h+4*wScale,v-9*htScale,h+14*wScale,v+1*htScale);
		
#ifdef IBM		
		rgb = GetRGBColor(colorLevel);
#else
		rgb = GetRGBColor(1.-colorLevel);
#endif
		//rgb = GetRGBColor(0.8-colorLevel);
		RGBForeColor(&rgb);
		PaintRect(&rgbrect);
		MyFrameRect(&rgbrect);
		
		MyMoveTo(h+20*wScale,v+.5*htScale);
		
		RGBForeColor(&colors[BLACK]);
		if (i<numLevels-1)
		{
			MyNumToStr(value,numstr);
			strcat(numstr," - ");
			MyNumToStr((*fContourLevelsH)[i+1],numstr2);
			strcat(numstr,numstr2);
		}
		else
		{
			strcpy(numstr,"> ");
			MyNumToStr(value,numstr2);
			strcat(numstr,numstr2);
		}
		//strcat(numstr,"    mg/L");
		//drawstring(MyNumToStr(value,numstr));
		drawstring(numstr);
		strLen = stringwidth(numstr);
		if (fTriAreaArray)
		{
			MyMoveTo(h+80*wScale,v+.5*htScale);
			MyNumToStr(value2,numstr3);
			//strcat(numstr,"		");
			//strcat(numstr,numstr3);
			//drawstring(numstr);
			drawstring(numstr3);
			strLen = 80*wScale + stringwidth(numstr3);
		}
		if (strLen>widestNum) widestNum = strLen;
		v = v+9*htScale;
	}
	if (fTriAreaArray)
	{
		MyMoveTo(h+20*wScale,v+5*htScale);
		strcpy(numstr,"Total Area = ");
		MyNumToStr(totalArea,numstr2);
		strcat(numstr,numstr2);
		drawstring(numstr);
		v = v + 9*htScale;
	}
	if (fContourDepth1==BOTTOMINDEX)
		//sprintf(text, "Depth: Bottom");
		sprintf(text, "Depth: Bottom %g m",fBottomRange);
	else
		sprintf(text, "Depth: %g to %g m",fContourDepth1,fContourDepth2);
	
	MyMoveTo(h+20*wScale, v+5*htScale);
	drawstring(text);
	if (stringwidth(text)+20*wScale > widestNum) widestNum = stringwidth(text)+20*wScale;
	v = v + 9*htScale;
	legendRect.bottom = v+3*htScale;
	if (legendRect.right<h+10*wScale+widestNum+4*wScale) legendRect.right = h+10*wScale+widestNum+4*wScale;
	else if (legendRect.right>legendRect.left+80*wScale && h+10*wScale+widestNum+4*wScale<=legendRect.right)
		legendRect.right = h+10*wScale+widestNum+4*wScale;	// may want to redraw to recenter the header
 	MyFrameRect(&legendRect);
	if (!gSavingOrPrintingPictFile)
		fLegendRect = legendRect;
	return;
}

void TCompoundMap::DrawDepthContourScale(Rect r, WorldRect view)
{
	Point		p;
	short		h,v,x,y,dY,widestNum=0;
	RGBColor	rgb;
	Rect		rgbrect;
	char 		numstr[30],numstr2[30],text[30];
	long 		i,numLevels;
	double	minLevel, maxLevel;
	double 	value;
	TTriGridVel3D* triGrid = GetGrid3DFromMapIndex(0);	
	
	// code goes here - use first priority only or do for all?
	triGrid->DrawContourScale(r,view);
	
	return;
}

/**************************************************************************************************/
#ifdef IBM
void TCompoundMap::EraseRegion(Rect r)
{
	long nSegs = GetNumBoundarySegs();	
	long theSeg,startver,endver,j,index;
	Point pt;
	Boolean offQuickDrawPlane = false;
	
	LongPointHdl ptsHdl = GetPointsHdl(false); // will use refined grid if there is one
	if(!ptsHdl) return;
	
	for(theSeg = 0; theSeg< nSegs; theSeg++)
	{
		startver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
		endver = (*fBoundarySegmentsH)[theSeg]+1;
		long numPts = endver - startver;
		POINT *pointsPtr = (POINT*)_NewPtr(numPts *sizeof(POINT));
		RgnHandle newClip=0;
		HBRUSH whiteBrush;
		
		for(j = startver; j < endver; j++)
		{
			if (fBoundaryPointsH)	// the reordered curvilinear grid
				index = (*fBoundaryPointsH)[j];
			else index = j;
			pt = GetQuickDrawPt((*ptsHdl)[index].h,(*ptsHdl)[index].v,&r,&offQuickDrawPlane);
			(pointsPtr)[j-startver] = MakePOINT(pt.h,pt.v);
		}
		
		newClip = CreatePolygonRgn((const POINT*)pointsPtr,numPts,ALTERNATE);
		whiteBrush = (HBRUSH)GetStockObject(WHITE_BRUSH);
		//err = SelectClipRgn(currentHDC,savedClip);
		FillRgn(currentHDC, newClip, whiteBrush);
		DisposeRgn(newClip);
		//DeleteObject(newClip);
		//SelectClipRgn(currentHDC,0);
		if(pointsPtr) {_DisposePtr((Ptr)pointsPtr); pointsPtr = 0;}
	}
	
}
#endif


/**************************************************************************************************/
void TCompoundMap::DrawContours(Rect r, WorldRect view)
{	
	long i,n;
	TMap *map = 0;
	
	// draw each of the maps contours (in reverse order to show priority)		
	for (n = mapList->GetItemCount()-1 ; n>=0; n--)
	{
		mapList->GetListItem((Ptr)&map, n);
		//map -> DrawContours(r,view);
		if ( (dynamic_cast<PtCurMap *>(map))->bDrawContours)
			this->DrawContoursFromMapIndex(r,view,n);
	}
	return;	
}

void TCompoundMap::DrawContoursFromMapIndex(Rect r, WorldRect view, long mapIndex)
{	// need all LELists
	long i, j, numOfLEs, numLESets, numTri, numDepths;
	LERec LE;
	Rect leRect, beachedRect, floatingRect;
	float beachedWidthInPoints = 3, floatingWidthInPoints = 2; // a point = 1/72 of an inch
	float pixelsPerPoint = PixelsPerPoint();
	short offset, massunits;
	Point pt;
	Boolean offQuickDrawPlane = false, bShowContours, bThereIsSubsurfaceOil = false;
	RGBColor saveColor, *onLandColor, *inWaterColor;
	LONGH numLEsInTri = 0;
	DOUBLEH massInTriInGrams = 0;
	double density, LEmass, massInGrams, halfLife;
	TopologyHdl topH = 0;
	TDagTree *dagTree = 0;
	TTriGridVel3D* triGrid = GetGrid3DFromMapIndex(mapIndex);	
	char countStr[64];
	long count=0;
	TLEList *thisLEList = 0;
	
	if (!triGrid) return; // some error alert, no depth info to check
	
	GetForeColor(&saveColor);
	
#ifdef IBM
	short xtraOffset = 1;
#else
	short xtraOffset = 0;
#endif
	
	offset = _max(1,(floatingWidthInPoints*pixelsPerPoint)/2);
	MySetRect(&floatingRect,-offset,-offset,offset,offset);
	offset = _max(1,(beachedWidthInPoints*pixelsPerPoint)/2);
	MySetRect(&beachedRect,-offset,-offset,offset,offset);
	
	dagTree = triGrid -> GetDagTree();
	if(!dagTree)	return;
	topH = dagTree->GetTopologyHdl();
	if(!topH)	return;
	numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
	numLEsInTri = (LONGH)_NewHandleClear(sizeof(long)*numTri);
	if (!numLEsInTri)
	{ TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
	massInTriInGrams = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
	if (!massInTriInGrams)
	{ TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
	
	numLESets = model->LESetsList->GetItemCount();
	for (i = 0; i < numLESets; i++)
	{
		model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if (thisLEList->fLeType == UNCERTAINTY_LE)	
			continue;	// don't draw uncertainty for now...
		if (!thisLEList->IsActive()) continue;
		numOfLEs = thisLEList->numOfLEs;
		// density set from API
		//density =  GetPollutantDensity(thisLEList->GetOilType());	
		density = (((TOLEList*)thisLEList)->fSetSummary).density;	
		halfLife = (*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.halfLife;
		massunits = thisLEList->GetMassUnits();
		
		// time has already been updated at this point
		bShowContours = (*(TOLEList*)thisLEList).fDispersantData.bDisperseOil && model->GetModelTime() - model->GetStartTime() - model->GetTimeStep() >= (*(TOLEList*)thisLEList).fDispersantData.timeToDisperse;
		// code goes here, should total LEs from all spills..., numLEsInTri as a permanent field, maybe should calculate during Step()
		bShowContours = bShowContours || (*(TOLEList*)thisLEList).fAdiosDataH;
		bShowContours = bShowContours || (*(TOLEList*)thisLEList).fSetSummary.z > 0;
		if (bShowContours) 
		{
			bThereIsSubsurfaceOil = true;
			for (j = 0 ; j < numOfLEs ; j++) {
				LongPoint lp;
				long triIndex;
				thisLEList -> GetLE (j, &LE);
				//if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
				//if (LE.statusCode == OILSTAT_EVAPORATED) continue;	// shouldn't happen, temporary for dissolved chemicals 
				if (!(LE.statusCode == OILSTAT_INWATER)) continue;// Windows compiler requires extra parentheses
				lp.h = LE.p.pLong;
				lp.v = LE.p.pLat;
				// will want to calculate individual LE mass for chemicals where particles will dissolve over time
				LEmass = GetLEMass(LE,halfLife);	// will only vary for chemical with different release end time
				massInGrams = VolumeMassToGrams(LEmass, density, massunits);	// need to do this above too
				if (fContourDepth1==BOTTOMINDEX)
				{
					double depthAtLE = DepthAtPoint(LE.p);	
					//if (depthAtLE <= 0) continue;	// occasionally dagtree is messed up
					//if (LE.z > (depthAtLE-1.) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map
					if (LE.z > (depthAtLE-fBottomRange) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map
					{
						triIndex = dagTree -> WhatTriAmIIn(lp);
						//if (triIndex>=0) (*numLEsInTri)[triIndex]++;
						if (triIndex>=0)  
						{
							(*numLEsInTri)[triIndex]++;
							(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
						}
						if (fDiagnosticStrType==SUBSURFACEPARTICLES)
						{
							pt = GetQuickDrawPt(LE.p.pLong,LE.p.pLat,&r,&offQuickDrawPlane);
							
							switch (LE.statusCode) {
								case OILSTAT_INWATER:
									RGBForeColor(&colors[BLACK]);
									leRect = floatingRect;
									MyOffsetRect(&leRect,pt.h,pt.v);
									PaintRect(&leRect);
									break;
							}
						}
					}
				}
				else if (LE.z>fContourDepth1 && LE.z<=fContourDepth2) 
				{
					triIndex = dagTree -> WhatTriAmIIn(lp);
					//if (triIndex>=0) (*numLEsInTri)[triIndex]++;
					if (triIndex>=0) 
					{
						(*numLEsInTri)[triIndex]++;
						(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
					}
					if (fDiagnosticStrType==SUBSURFACEPARTICLES)
					{
						pt = GetQuickDrawPt(LE.p.pLong,LE.p.pLat,&r,&offQuickDrawPlane);
						
						switch (LE.statusCode) {
							case OILSTAT_INWATER:
								RGBForeColor(&colors[BLACK]);
								leRect = floatingRect;
								MyOffsetRect(&leRect,pt.h,pt.v);
								PaintRect(&leRect);
								break;
						}
					}
				}
			}
		}
		
	}
	
	if (bThereIsSubsurfaceOil && !(fDiagnosticStrType==SUBSURFACEPARTICLES))	// draw LEs in a given layer
		//if (bShowContours && !(fDiagnosticStrType==SUBSURFACEPARTICLES))	// draw LEs in a given layer
	{
		double triArea, triVol, oilDensityInWaterColumn, prevMax=-1;
		long numLEsInTriangle, numLevels;
		double **dosageHdl = 0;
		double concInSelectedTriangles = 0;
		RGBColor col;
		
		if (!fContourLevelsH)
			if (!InitContourLevels()) return;
		numLevels = GetNumDoubleHdlItems(fContourLevelsH);
		// need to track here if want to display on legend
		if (fTriAreaArray)
		{delete [] fTriAreaArray; fTriAreaArray = 0;}
		fTriAreaArray = new double[numLevels];
		for (i=0;i<numLevels;i++)
			fTriAreaArray[i] = 0.;
		// code goes here, in order to smooth out the blips will have to allow
		// max conc to peak and start to decline, then cap it
		// from then on allow the saved max to get lower, but not higher
		// and stomp out blips by comparing to 
		// this needs to be max over all triangles, not just selected
		// and there may be other issues for the bottom			
		// first time max < prevMax, set global threshold at current max, some sort of flag to change the check
		// then check each time that max <= current max and reset current max to max
		// would also need to be done in the tracking section, hmm
		//if (bUseSmoothing)
		//prevMax = triGrid->GetMaxAtPreviousTimeStep(model->GetModelTime()-model->GetTimeStep());
		if (triGrid->bShowDosage)
		{
			dosageHdl = triGrid->GetDosageHdl(false);
		}
		for (i=0;i<numTri;i++)
		{
			float colorLevel,depthAtPt = -1, depthRange;
			long roundedDepth;
			//WorldPoint centroid = {0,0};
			triArea = (triGrid -> GetTriArea(i)) * 1000 * 1000;	// convert to meters
			//if (!triGrid->GetTriangleCentroidWC(i,&centroid))
			{
				//depthAtPt = DepthAtPoint(centroid);
				depthAtPt = DepthAtCentroidFromMapIndex(i,mapIndex);
				if (depthAtPt<=0) 
					roundedDepth = 0;
				//printError("Couldn't find depth at point");
				else
					roundedDepth = floor(depthAtPt) + (depthAtPt - floor(depthAtPt) >= .5 ? 1 : 0);
			}
			//else
			//roundedDepth = 0;
			//printError("Couldn't find centroid");
			if(!(fContourDepth1==BOTTOMINDEX))
			{
				if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt > 0) depthRange = depthAtPt - fContourDepth1;
				else depthRange = fContourDepth2 - fContourDepth1;
			}
			else
			{
				//if (depthAtPt<1 && depthAtPt>0) depthRange = depthAtPt;
				//else depthRange = 1.;	// for bottom layer will always use 1m
				if (depthAtPt<fBottomRange && depthAtPt>0) depthRange = depthAtPt;
				else depthRange = fBottomRange;	// for bottom layer will always use 1m
			}
			//if (depthAtPt < depthRange && depthAtPt != 0) depthRange = depthAtPt;
			//if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt != 0) depthRange = depthAtPt - fContourDepth1;
			triVol = triArea * depthRange; // code goes here, check this depth range is ok at all vertices
			//triVol = triArea * (fContourDepth2 - fContourDepth1); // code goes here, check this depth range is ok at all vertices
			numLEsInTriangle = (*numLEsInTri)[i];
			
			if (!(fContourDepth1==BOTTOMINDEX))		// need to decide what to do for bottom contour
				if (triGrid->CalculateDepthSliceVolume(&triVol,i,fContourDepth1,fContourDepth2)) goto done;
			
			oilDensityInWaterColumn = (*massInTriInGrams)[i] / triVol;	// units? milligrams/liter ?? for now gm/m^3
			
			//if (prevMax > 0 && prevMax < oilDensityInWaterColumn)	// change this to check global max, at each run reset to -1?
			//oilDensityInWaterColumn = prevMax;
			if (triGrid->bShowDosage && dosageHdl)	
			{
				double dosage = (*dosageHdl)[i];
				if (dosage > 2.) 	// need to get some threshold numbers from Alan
				{
					RGBForeColor (&colors[RED]);
					//triGrid->DrawTriangle(&r,i,TRUE,FALSE);	// fill triangles	
					triGrid->DrawTriangle3D(&r,i,TRUE,FALSE);	// fill triangles	
				}
			}
			else
			{
				if (numLEsInTriangle==0)
				{
					if (fDiagnosticStrType==DEPTHATCENTERS)	// uses refined grid to pick center, original to find depth
						//((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,roundedDepth);
						((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,depthAtPt);
					continue;
				}
				for (j=0;j<numLevels;j++)
				{
					colorLevel = float(j)/float(numLevels-1);
					if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
					{	// note: the lowest contour value is not included in count
						fTriAreaArray[j] = fTriAreaArray[j] + triArea/1000000;
#ifdef IBM		
						col = GetRGBColor(colorLevel);
#else
						col = GetRGBColor(1.-colorLevel);
#endif
						//col = GetRGBColor(0.8-colorLevel);
						RGBForeColor(&col);
						//if (!(fDiagnosticStrType==SUBSURFACEPARTICLES))
						//triGrid->DrawTriangle(&r,i,TRUE,FALSE);	// fill triangles	
						triGrid->DrawTriangle3D(&r,i,TRUE,FALSE);	// fill triangles	
						// draw concentration or #LEs as string centered in triangle
						if (fDiagnosticStrType==TRIANGLEAREA)
							((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,triArea/1000000);
						if (fDiagnosticStrType==NUMLESINTRIANGLE)
							((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,numLEsInTriangle);
						if (fDiagnosticStrType==CONCENTRATIONLEVEL)
							((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,oilDensityInWaterColumn);
						//if (fDiagnosticStrType==DEPTHATCENTERS)
						//((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,roundedDepth);
					}
				}
			}
			if (fDiagnosticStrType==DEPTHATCENTERS)	// uses refined grid to pick center, original to find depth
				//((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,roundedDepth);
				((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,depthAtPt);
			if (((TTriGridVel3D*)triGrid)->fMaxTri==i && ((TTriGridVel3D*)triGrid)->bShowMaxTri) 
			{
				//RGBForeColor (&colors[RED]);
				//triGrid->DrawTriangle(&r,i,TRUE,FALSE);	// fill triangles	
				((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,-1);
			}
		}
	}
	for ( i = 0; i < numLESets; i++)
	{
		model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if (thisLEList->fLeType == UNCERTAINTY_LE)	
			continue;	// don't draw uncertainty for now...
		if (!thisLEList->IsActive()) continue;
		
		numOfLEs = thisLEList->numOfLEs;
		
		for (j = 0 ; j < numOfLEs ; j++) {
			thisLEList -> GetLE (j, &LE);
			if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
			//if ((LE.z==0 && !bShowSurfaceLEs)) continue;	
			if ((LE.z==0 && bShowSurfaceLEs) || !bThereIsSubsurfaceOil )	// draw LEs colored based on depth, !LE.dispersionStatus==HAVE_DISPERSED
				//if ((LE.z==0 && bShowSurfaceLEs) || !bShowContours )	// draw LEs colored based on depth, !LE.dispersionStatus==HAVE_DISPERSED
			{
				if (!WPointInWRect(LE.p.pLong, LE.p.pLat, &view)) continue;
				
				switch(thisLEList->fLeType)
				{
					case UNCERTAINTY_LE:		// shouldn't happen...
						onLandColor  = &colors[RED];
						inWaterColor = &colors[RED];
						break;
					default:
						onLandColor  = &colors[BLACK];	
						inWaterColor = &colors[BLACK];	// surface LEs
						break;
				}
				
				pt = GetQuickDrawPt(LE.p.pLong,LE.p.pLat,&r,&offQuickDrawPlane);
				
				switch (LE.statusCode) {
					case OILSTAT_INWATER:
						RGBForeColor(inWaterColor);
						leRect = floatingRect;
						MyOffsetRect(&leRect,pt.h,pt.v);
						PaintRect(&leRect);
						break;
					case OILSTAT_ONLAND:	// shouldn't happen...
						RGBForeColor(onLandColor);
						leRect = beachedRect;
						MyOffsetRect(&leRect,pt.h,pt.v);
						// draw an "X"
						MyMoveTo(leRect.left,leRect.top);
						MyLineTo(leRect.right+xtraOffset,leRect.bottom+xtraOffset);
						MyMoveTo(leRect.left,leRect.bottom);
						MyLineTo(leRect.right+xtraOffset,leRect.top-xtraOffset);
						break;
				}
				/////////////////////////////////////////////////
				
			}
		}
		
	}
	
done:
	RGBForeColor(&saveColor);
	if(numLEsInTri) {DisposeHandle((Handle)numLEsInTri); numLEsInTri=0;}
	if(massInTriInGrams) {DisposeHandle((Handle)massInTriInGrams); massInTriInGrams=0;}
	return;
}

/////////////////////////////////////////////////
Rect TCompoundMap::DoArrowTool(long triNum)	
{	// show depth concentration profile at selected triangle
	long n,listIndex,numDepths=0;
	//TLEList	*thisLEList;
	Rect r = MapDrawingRect();
	//TTriGridVel3D* triGrid = GetGrid(false);
	TTriGridVel3D* triGrid = GetGrid3D(true);	// since output data is stored on the refined grid need to used it here
	OSErr err = 0;
	TMover *mover=0;
	double depthAtPt=0;
	Boolean needToRefresh=false;
	
	if (!triGrid) return r;
	
	mover = this->GetMover(TYPE_TRICURMOVER);
	if (mover)
	{
		numDepths = ( dynamic_cast<TriCurMover*>(mover)) -> CreateDepthSlice(triNum,&fDepthSliceArray);
		//numDepths = ((TriCurMover*)mover) -> CreateDepthSlice(triNum,fDepthSliceArray);
		if (numDepths > 0) goto drawPlot; else return r;
	}
	//for (listIndex = 0, n = model->LESetsList -> GetItemCount (); listIndex < n; listIndex++)
	//{
	//double depthAtPt=0;
	//WorldPoint centroid = {0,0};
	//model->LESetsList -> GetListItem ((Ptr) &thisLEList, listIndex);
	// check that a list is dispersed, if not no point
	//if (!(*(TOLEList*)thisLEList).fDispersantData.bDisperseOil && !(*(TOLEList*)thisLEList).fAdiosDataH && !(*(TOLEList*)thisLEList).fSetSummary.z > 0) 
	//continue;
	// also must be LEs in that triangle
	
	if (triNum < 0)
	{
		//if (GetDepthAtMaxTri(((TOLEList*)thisLEList),&triNum,&depthAtPt)) return r;
		if (GetDepthAtMaxTri(&triNum,&depthAtPt)) return r;
	}
	else
	{
		//if (!triGrid->GetTriangleCentroidWC(triNum,&centroid))
		{
			//depthAtPt = DepthAtPoint(centroid);
			depthAtPt = DepthAtCentroid(triNum);
			if (depthAtPt < 0) depthAtPt = 100;
		}
		//else return r;
	}
	// code goes here, at the bottom have to consider that LE.z could be greater than depth at triangle center
	numDepths = floor(depthAtPt)+1;	// split into 1m increments to track vertical slice
	
	//if (numDepths>0) err = CreateDepthSlice(thisLEList,triNum);
	if (numDepths>0) err = CreateDepthSlice(triNum,&fDepthSliceArray);
	//if (numDepths>0) err = CreateDepthSlice(triNum,fDepthSliceArray);
	if (!err) {triGrid -> fMaxTri = triNum; needToRefresh = true;/*triGrid -> bShowMaxTri = true;*/}
	// only do for first if more than one
	//if (!err && numDepths>0) break;
	//}
drawPlot:
	if (numDepths>0 && !err)
	{
		Boolean **triSelected = triGrid -> GetTriSelection(false);	// initialize if necessary
		outputData **oilConcHdl = triGrid -> GetOilConcHdl();	
		float depthRange1 = fContourDepth1, depthRange2 = fContourDepth2, bottomRange = fBottomRange;
		// should break out once a list is found and bring up the graph
		if (triSelected) 	// tracked output at a specified area
			PlotDialog(oilConcHdl,fDepthSliceArray,depthRange1,depthRange2,bottomRange,true,true);
		else 	// tracked output following the plume
			PlotDialog(oilConcHdl,fDepthSliceArray,depthRange1,depthRange2,bottomRange,false,true);
		if (needToRefresh == true)
		{
			InvalidateMapImage();// invalidate the offscreen bitmap
			InvalMapDrawingRect();
		}
	}
	else
		SysBeep(5);
	
	return r;
}

void TCompoundMap::MarkRect(Point p)
{
	Point where;
	Rect r;
	WorldRect check;
	long i,n,currentScale = CurrentScaleDenominator();
	long mostZoomedInScale = MostZoomedInScale();
	TLEList *thisLEList;
	LETYPE leType;
	ClassID thisClassID;
	Boolean foundLEList = 0;	
	DispersionRec dispInfo;
	
	MySetCursor(1005);
	for (i = 0, n = model->LESetsList->GetItemCount() ; i < n; i++) 
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !model->IsUncertain()) continue;
		
		thisClassID = thisLEList -> GetClassID();
		if(thisClassID == TYPE_OSSMLELIST || thisClassID == TYPE_SPRAYLELIST )
		{
			TOLEList *thisOLEList = (TOLEList*)thisLEList; // typecast
			foundLEList = true;
			break;
			// may want to require spill has not already been chemically dispersed? 
			// also check natural dispersion
			//if (thisOLEList->fDispersantData.bDisperseOil) bSomeSpillIsDispersed = true;
			//Seconds thisStartTime = thisOLEList->fSetSummary.startRelTime;
		}
	}
	if (!foundLEList) return;
	//r = DefineGrayRect(p, ZoomRectAction, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE);
	r = DefineGrayRect(p, ZoomRectAction, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE);
	
	if(currentScale <= (long)(1.01*mostZoomedInScale)) // JLM 12/19/97
	{ SysBeep(1); return;}
	
	if (RectWidth(r) > 2 && RectHeight(r) > 2) {
		check = ScreenToWorldRect(r, MapDrawingRect(), settings.currentView);
		if (WRectWidth(check) < MINWRECTDIST || WRectHeight(check) < MINWRECTDIST) { SysBeep(1); return ; }
		if (WRectWidth(check) > MAXWRECTDIST || WRectHeight(check) > MAXWRECTDIST) { SysBeep(1); return ; }
		//ChangeCurrentView(check, TRUE, TRUE);
	}
	else
		return;
	/*if (r.right > 1) {
	 where.h = r.left;
	 where.v = r.top;
	 MagnifyTool(where, -ZOOMPLUSTOOL);
	 }*/
	
	// bring up dialog to set api, maybe dispersion duration
	dispInfo = ((TOLEList*)thisLEList) -> GetDispersionInfo();
	// if already dispersing force to restart?
	dispInfo.bDisperseOil = 1;
	dispInfo.timeToDisperse = model->GetModelTime() - model->GetStartTime() + model->GetTimeStep();
	dispInfo.amountToDisperse = 1;
	dispInfo.duration = 0;
	dispInfo.areaToDisperse = check; 
	dispInfo.lassoSelectedLEsToDisperse = false;
	//set to entire map or could set area via the polygon - then area can't be a rect...
	//double api - undo from oil type, could set by hand, ignore for now since density already set;
	((TOLEList*)thisLEList) -> SetDispersionInfo(dispInfo); 
	((TOLEList*)thisLEList) -> bShowDispersantArea = true;
	//InvalidateMapImage();// invalidate the offscreen bitmap
	InvalMapDrawingRect();
	//model->NewDirtNotification(DIRTY_LIST);
	// make sure to redraw the screen
}

void TCompoundMap::DoLassoTool(Point p)
{
	// require there is a spill, not already chemically dispersed ?
	// if more than one spill ? use up/down arrows to change top spill
	// if in mid-run, expect user wants to select LEs and disperse immediately - bring up dialog
	// will need to set duration, api, anything else? - effectiveness, but this may be a property of the spill
	long i, j, n, numsegs, numLEs, count = 0;
	Point newPoint,startPoint;
	WorldPoint w;
	WORLDPOINTH wh=0;
	TLEList *thisLEList;
	LETYPE leType;
	ClassID thisClassID;
	Boolean foundLEList = 0;
	SEGMENTH poly = 0;
	double x, effectiveness = 100.;
	OSErr err = 0;
	
	// code goes here, lasso should apply to all LEs it captures, which may be multiple spills
	// code goes here, bring up a dialog to ask user for effectiveness, but won't let them change the value
	for (i = 0, n = model->LESetsList->GetItemCount() ; i < n && !err; i++) 
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !model->IsUncertain()) continue;
		
		thisClassID = thisLEList -> GetClassID();
		if(thisClassID == TYPE_OSSMLELIST || thisClassID == TYPE_SPRAYLELIST )
		{
			TOLEList *thisOLEList = (TOLEList*)thisLEList; // typecast
			foundLEList = true;
			break;
			// may want to require spill has not already been chemically dispersed? 
			// also check natural dispersion
			//if (thisOLEList->fDispersantData.bDisperseOil) bSomeSpillIsDispersed = true;
			//Seconds thisStartTime = thisOLEList->fSetSummary.startRelTime;
		}
	}
	if (!foundLEList) { printNote("Set a spill before using lasso tool to disperse oil"); return;}
	
	startPoint = p;
	MyMoveTo(p.h,p.v);
	w = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
	AppendToWORLDPOINTH(&wh,&w);	
	while(StillDown()) 
	{
		GetMouse(&newPoint);
		if((newPoint.h != p.h || newPoint.v != p.v))
		{
			MyLineTo(newPoint.h, newPoint.v);
			p = newPoint;
			w = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
			if(!AppendToWORLDPOINTH(&wh,&w))goto Err;
		}
	}
	w = ScreenToWorldPoint(startPoint, MapDrawingRect(), settings.currentView);
	AppendToWORLDPOINTH(&wh,&w);
	poly = WPointsToSegments(wh,_GetHandleSize((Handle)(wh))/sizeof(WorldPoint),&numsegs);
	if (!poly) goto Err;
	
	// loop over the LEs and check if they are in the polygon, if so mark to disperse
	// disperse immediately (at current time) and instantaneously, though maybe over an hour to test
	for (i = 0, n = model->LESetsList->GetItemCount() ; i < n && !err; i++) 
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !model->IsUncertain()) continue;
		
		thisClassID = thisLEList -> GetClassID();
		if(thisClassID == TYPE_OSSMLELIST || thisClassID == TYPE_SPRAYLELIST )
		{
			long numToDisperse=0;
			Boolean alreadyUsingLasso = false;
			WorldPoint w;
			DispersionRec dispInfo = ((TOLEList*)thisLEList) -> GetDispersionInfo();
			LERec theLE;
			if (poly != 0)
			{
				// bring up dialog to set api, maybe dispersion duration, effectiveness
				numLEs = thisLEList->GetLECount();
				for(j=0; j < numLEs; j++) // make this numLEs
				{
					thisLEList -> GetLE (j, &theLE);
					//code goes here, should total all LEs selected at different times to get amount
					// also if selecting LEs at earlier time than others, deselect the later ones?
					// will eventually need to be able to edit the lassoed regions
					// check if already dispersed
					if (theLE.dispersionStatus == HAVE_DISPERSED || theLE.dispersionStatus == HAVE_DISPERSED_NAT || theLE.dispersionStatus == HAVE_EVAPORATED) continue;
					w.pLong = theLE.p.pLong;
					w.pLat = theLE.p.pLat;
					if (PointInPolygon(w,poly,numsegs,true))	// true -> holes ??
					{
						count++;
						if (count==1) {GetScaleFactorFromUser("Input dispersant effectiveness as a decimal (0 to 1)",&effectiveness);
							if (effectiveness > 1) effectiveness = 1;}
						x = GetRandomFloat(0, 1.0);
						if (x <= effectiveness)
						{
							theLE.dispersionStatus = DISPERSE;	// but gets reset
							theLE.beachTime = model->GetModelTime();	// use for time to disperse
							numToDisperse++;
							thisLEList -> SetLE (j, &theLE);
						}
					}
				}
				if (dispInfo.lassoSelectedLEsToDisperse) alreadyUsingLasso = true;
				if (numToDisperse>0)
				{
					dispInfo.bDisperseOil = 1;
					//if (alreadyUsingLasso && dispInfo.timeToDisperse < model->GetModelTime() - model->GetStartTime())
					if (alreadyUsingLasso && dispInfo.timeToDisperse < model->GetModelTime() - ((TOLEList*)thisLEList) ->fSetSummary.startRelTime)
					{
					}
					else
						//dispInfo.timeToDisperse = model->GetModelTime() - model->GetStartTime();
						dispInfo.timeToDisperse = model->GetModelTime() - ((TOLEList*)thisLEList) ->fSetSummary.startRelTime;
					dispInfo.amountToDisperse = (float)numToDisperse/(float)numLEs;
					dispInfo.duration = 0;
					dispInfo.areaToDisperse = GetMapBounds(); 
					dispInfo.lassoSelectedLEsToDisperse = true;
					//set to entire map or could set area via the polygon - then area can't be a rect...
					//double api - undo from oil type, could set by hand, ignore for now since density already set;
					((TOLEList*)thisLEList) -> SetDispersionInfo(dispInfo); 
				}
			}
		}
	}
	
Err:
	if(wh) {DisposeHandle((Handle)wh); wh=0;}	// may want to save this to draw or whatever
	model->NewDirtNotification(DIRTY_LIST);
	return;
}
/////////////////////////////////////////////////
/*void TCompoundMap::SetSelectedBeach(LONGH *segh, LONGH selh)
 {
 long n,i,k,next_k;
 long numSelectedBoundaryPts;
 Boolean pointsAdded = false;
 OSErr err = 0;
 
 if (selh) n = _GetHandleSize((Handle)selh)/sizeof(**selh);
 if (n<1) return;
 // at this point only boundary tool can be used so all this is doing is copying one handle to another
 if((*selh)[n-1]==-1)	// boundary tool was used, all is well
 {
 for(i=0;i<n;i++)
 {
 k = (*selh)[i];
 AppendToLONGH(segh,k);
 }
 }
 else	// arrow tool was used (or both were used)
 {
 for(i=0;i<n-1;i++)
 {
 // make sure the selected points are boundary points 
 // and mark a segment or boundary switch with -1
 k = (*selh)[i];
 next_k = (*selh)[i+1];
 if (!IsBoundaryPoint(k)) 
 {
 err = -2;
 continue;
 }
 if (k==-1) continue;
 if (next_k==-1)
 {
 AppendToLONGH(segh,next_k);
 pointsAdded = false;
 continue;
 }
 if (ContiguousPoints(k,next_k))
 {
 if (!pointsAdded) 	// first point of segment
 AppendToLONGH(segh,k);
 AppendToLONGH(segh,next_k);
 pointsAdded=true;
 }
 else 
 {
 if(pointsAdded)
 {
 AppendToLONGH(segh,-1);
 pointsAdded=false;
 if (i==n-2 && err==0) err = -3;
 }
 else
 // otherwise skipping point
 if (err==0) err = -3;
 }
 }
 //numWaterBoundaryPts = GetNumLONGHItems(*segh);
 if (*segh) numSelectedBoundaryPts = _GetHandleSize((Handle)(*segh))/sizeof(***segh);
 if (numSelectedBoundaryPts>0 && (**segh)[numSelectedBoundaryPts-1]!=-1)
 AppendToLONGH(segh,-1);	// mark end of selected segment 
 if (n==1)
 printError("An isolated boundary point was selected. No water boundary will be set.");
 if (err == -2)
 printError("Non boundary points were selected and will be ignored");
 else if (err == -3)
 printError("Non contiguous boundary points were selected and will be ignored");
 }	
 }  
 
 void TCompoundMap::SetBeachSegmentFlag(LONGH *beachBoundaryH, long *numBeachBoundaries)
 {
 // rearranging points to parallel water boundaries to simplify drawing
 // code goes here, keep some sort of pointer to go back from new ordering to old ordering
 // that way can draw plots in order points were selected rather than in numerical order
 long i, startIndex, endIndex, curIndex=0, p, afterP;
 long segNo,lastPtOnSeg,firstPtOnSeg;
 long numBoundaryPts = GetNumBoundaryPts();
 
 for(i=0;i<numBoundaryPts;i++)
 {
 (**beachBoundaryH)[i] = 0;
 }
 
 while(MoreSegments(fSelectedBeachHdl,&startIndex, &endIndex, &curIndex))
 {
 if(endIndex <= startIndex)continue;
 
 segNo = PointOnWhichSeg((*fSelectedBeachHdl)[startIndex]);
 firstPtOnSeg = segNo == 0 ? 0: (*fBoundarySegmentsH)[segNo-1] + 1;
 lastPtOnSeg = (*fBoundarySegmentsH)[segNo];
 for(i=startIndex; i< endIndex -1; i++)
 {
 p = (*fSelectedBeachHdl)[i];
 afterP = (*fSelectedBeachHdl)[i+1];
 // segment endpoint indicates whether segment is selected
 if ((p<afterP && !(p==firstPtOnSeg && afterP==lastPtOnSeg)) || (afterP==firstPtOnSeg && p==lastPtOnSeg))
 {
 (**beachBoundaryH)[afterP]=1;
 *numBeachBoundaries += 1;
 }
 else if ((p>afterP && !(afterP==firstPtOnSeg && p==lastPtOnSeg)) || (p==firstPtOnSeg && afterP==lastPtOnSeg))
 {
 (**beachBoundaryH)[p]=1;
 *numBeachBoundaries += 1;
 }
 }
 }
 } 
 
 void TCompoundMap::ClearSelectedBeach()
 {
 MyDisposeHandle((Handle*)&fSegSelectedH);
 MyDisposeHandle((Handle*)&fSelectedBeachHdl);
 MyDisposeHandle((Handle*)&fSelectedBeachFlagHdl);
 }*/

/////////////////////////////////////////////////


/**************************************************************************************************/
/*typedef struct ConcTriNumPair
 {
 double conc;
 long triNum;
 } ConcTriNumPair, *ConcTriNumPairP, **ConcTriNumPairH;
 
 
 int ConcentrationCompare(void const *x1, void const *x2)
 {
 ConcTriNumPair *p1,*p2;	
 p1 = (ConcTriNumPair*)x1;
 p2 = (ConcTriNumPair*)x2;
 
 if (p1->conc < p2->conc) 
 return -1;  // first less than second
 else if (p1->conc > p2->conc)
 return 1;
 else return 0;// equivalent
 
 }*/
void TCompoundMap::TrackOutputDataFromMapIndex(long mapIndex)
{	// need all LELists
	TTriGridVel3D* triGrid = GetGrid3DFromMapIndex(mapIndex);
	long i, j, numOfLEs, numLESets, numTri;
	LERec LE;
	Boolean bShowContours, bTimeZero, bTimeToOutputData = false, bThereIsSubsurfaceOil = false;
	LONGH numLEsInTri = 0;
	DOUBLEH massInTriInGrams = 0;
	TopologyHdl topH = 0;
	TDagTree *dagTree = 0;
	DOUBLEH  dosageHdl = 0;
	ConcTriNumPairH concentrationH = 0;
	//TTriGridVel3D* triGrid = GetGrid3D(true);	
	Seconds modelTime = model->GetModelTime(),timeStep = model->GetTimeStep();
	Seconds startTime = model->GetStartTime();
	short oldIndex, nowIndex, massunits;
	double depthAtPt, density, LEmass, massInGrams, halfLife;
	TLEList *thisLEList = 0;
	
	if (!triGrid) return; // some error alert, no depth info to check
	
	Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
	
	dagTree = triGrid -> GetDagTree();
	if(!dagTree)	return;
	topH = dagTree->GetTopologyHdl();
	if(!topH)	return;
	numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
	numLEsInTri = (LONGH)_NewHandleClear(sizeof(long)*numTri);
	if (!numLEsInTri)
	{ TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
	massInTriInGrams = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
	if (!massInTriInGrams)
	{ TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
	//concentrationH = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
	concentrationH = (ConcTriNumPairH)_NewHandleClear(sizeof(ConcTriNumPair)*numTri);
	if (!concentrationH)
	{ TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
	
	//bTimeZero = (modelTime ==(*(TOLEList*)thisLEList).fDispersantData.timeToDisperse+startTime);
	nowIndex = (modelTime + timeStep - startTime) / model->LEDumpInterval;
	oldIndex = (modelTime - startTime) / model->LEDumpInterval;
	if(nowIndex > oldIndex ) bTimeToOutputData = true;
	
	numLESets = model->LESetsList->GetItemCount();	// code goes here, check bActive
	for (i = 0; i < numLESets; i++)
	{
		model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if (thisLEList->fLeType == UNCERTAINTY_LE)	
			continue;	// don't draw uncertainty for now...
		if (!thisLEList->IsActive()) continue;
		numOfLEs = thisLEList->numOfLEs;
		// density set from API
		//density =  GetPollutantDensity(thisLEList->GetOilType());	
		density = ((TOLEList*)thisLEList)->fSetSummary.density;	
		halfLife = (*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.halfLife;
		massunits = thisLEList->GetMassUnits();
		
		if (bTimeToOutputData)	// track budget at the same time
		{	// make budget table even if spill is not dispersed
			double amttotal,amtevap,amtbeached,amtoffmap,amtfloating,amtreleased,amtdispersed,amtremoved=0;
			Seconds timeAfterSpill;	
			// what to do for chemicals, amount dissolved?
			thisLEList->GetLEAmountStatistics(thisLEList->GetMassUnits(),&amttotal,&amtreleased,&amtevap,&amtdispersed,&amtbeached,&amtoffmap,&amtfloating,&amtremoved);
			BudgetTableData budgetTable; 
			// if chemical will need to get amount dissolved
			timeAfterSpill = nowIndex * model->LEDumpInterval;
			budgetTable.timeAfterSpill = timeAfterSpill;
			budgetTable.amountReleased = amtreleased;
			budgetTable.amountFloating = amtfloating;
			budgetTable.amountDispersed = amtdispersed;
			budgetTable.amountEvaporated = amtevap;
			budgetTable.amountBeached = amtbeached;
			budgetTable.amountOffMap = amtoffmap;
			budgetTable.amountRemoved = amtremoved;
			((TOLEList*)thisLEList)->AddToBudgetTableHdl(&budgetTable);
			// still track for each list, for output total everything
		}
		
		// time has not been updated at this point (in DrawContours time has been updated)
		bShowContours = (*(TOLEList*)thisLEList).fDispersantData.bDisperseOil && model->GetModelTime() - model->GetStartTime() >= (*(TOLEList*)thisLEList).fDispersantData.timeToDisperse;
		bShowContours = bShowContours || (*(TOLEList*)thisLEList).fAdiosDataH;
		bShowContours = bShowContours || (*(TOLEList*)thisLEList).fSetSummary.z > 0;	// for bottom spill
		if (bShowContours) bThereIsSubsurfaceOil = true;
		if (!bShowContours) continue;	// no need to track in this case
		// total LEs from all spills..., numLEsInTri as a permanent field, maybe should calculate during Step()
		for (j = 0 ; j < numOfLEs ; j++) {
			LongPoint lp;
			long triIndex;
			thisLEList -> GetLE (j, &LE);
			//if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
			//if (LE.statusCode == OILSTAT_EVAPORATED) continue;	// shouldn't happen, temporary for dissolved chemicals 
			if (!(LE.statusCode == OILSTAT_INWATER)) continue;	// Windows compiler requires extra parentheses
			lp.h = LE.p.pLong;
			lp.v = LE.p.pLat;
			LEmass = GetLEMass(LE,halfLife);	// will only vary for chemical with different release end time
			massInGrams = VolumeMassToGrams(LEmass, density, massunits);	// need to do this above too
			if (fContourDepth1==BOTTOMINDEX)
			{
				double depthAtLE = DepthAtPoint(LE.p);
				//if (LE.z > (depthAtLE-1.) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
				if (LE.z > (depthAtLE-fBottomRange) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
				{
					triIndex = dagTree -> WhatTriAmIIn(lp);
					if (triIndex>=0)  
					{
						(*numLEsInTri)[triIndex]++;
						(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
					}
				}
			}
			else if (LE.z>fContourDepth1 && LE.z<=fContourDepth2) 
			{
				triIndex = dagTree -> WhatTriAmIIn(lp);
				if (triIndex>=0) 
				{
					(*numLEsInTri)[triIndex]++;
					(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
				}
			}
		}
	}
	if (triGrid->bCalculateDosage)
	{
		dosageHdl = triGrid -> GetDosageHdl(true);
	}
	if (bThereIsSubsurfaceOil)
	{
		double triArea, triVol, oilDensityInWaterColumn, totalVol=0;
		long numLEsInTriangle,j,numLevels,totalLEs=0,totalMass=0,numDepths=0,maxTriNum=-1,numTrisWithOil = 0;
		double concInSelectedTriangles=0,maxConc=0,numTrisSelected=0,minLevel,maxLevel,range,avConcOverTriangles=0;
		if (!fContourLevelsH)
			if (!InitContourLevels()) return;
		numLevels = GetNumDoubleHdlItems(fContourLevelsH);
		if (fTriAreaArray)
		{delete [] fTriAreaArray; fTriAreaArray = 0;}
		fTriAreaArray = new double[numLevels];
		for (i=0;i<numLevels;i++)
			fTriAreaArray[i] = 0.;
		for (i=0;i<numTri;i++)
		{	// track ppm hours here
			double depthRange;
			depthAtPt=0;
			//WorldPoint centroid = {0,0};
			if (triSelected && !(*triSelected)[i]) continue;	// note this line keeps triareaarray from tracking for output..., dosage...
			//if (!triGrid->GetTriangleCentroidWC(i,&centroid))
			{
				//depthAtPt = DepthAtPoint(centroid);
				//depthAtPt = DepthAtCentroid(i);
				depthAtPt = DepthAtCentroidFromMapIndex(i,mapIndex);
			}
			triArea = (triGrid -> GetTriArea(i)) * 1000 * 1000;	// convert to meters
			if (!(fContourDepth1==BOTTOMINDEX))	
			{
				if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt > 0) depthRange = depthAtPt - fContourDepth1;
				else depthRange = fContourDepth2 - fContourDepth1;
			}
			else
			{
				//if (depthAtPt<1 && depthAtPt>0) depthRange = depthAtPt;	// should do a triangle volume 
				//else depthRange = 1.; // for bottom will always contour 1m
				if (depthAtPt<fBottomRange && depthAtPt>0) depthRange = depthAtPt;	// should do a triangle volume 
				else depthRange = fBottomRange; // for bottom will always contour 1m
			}
			//if (depthAtPt < depthRange && depthAtPt != 0) depthRange = depthAtPt;
			//if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt != 0) depthRange = depthAtPt - fContourDepth1;
			triVol = triArea * depthRange; // code goes here, check this depth range is ok at all vertices
			//triVol = triArea * (fContourDepth2 - fContourDepth1); // code goes here, check this depth range is ok at all vertices
			numLEsInTriangle = (*numLEsInTri)[i];
			if (numLEsInTriangle==0)
				continue;
			
			if (!(fContourDepth1==BOTTOMINDEX))		// need to decide what to do for bottom contour
				if (triGrid->CalculateDepthSliceVolume(&triVol,i,fContourDepth1,fContourDepth2)) goto done;
			oilDensityInWaterColumn = (*massInTriInGrams)[i] / triVol;	// units? milligrams/liter ?? for now gm/m^3
			
			//(*concentrationH)[numTrisWithOil] = oilDensityInWaterColumn;
			(*concentrationH)[numTrisWithOil].conc = oilDensityInWaterColumn;
			(*concentrationH)[numTrisWithOil].triNum = i;
			numTrisWithOil++;
			//for (j=0;j<numLevels-1;j++)
			for (j=0;j<numLevels;j++)
			{
				//if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
				if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
				{
					fTriAreaArray[j] = fTriAreaArray[j] + triArea/1000000;
					totalLEs += numLEsInTriangle;
					//totalMass += massInGrams;
					totalMass += (*massInTriInGrams)[i];
					totalVol += triVol;
					concInSelectedTriangles += oilDensityInWaterColumn;	// sum or track each one?
					if (oilDensityInWaterColumn > maxConc) 
					{
						maxConc = oilDensityInWaterColumn;
						numDepths = floor(depthAtPt)+1;	// split into 1m increments to track vertical slice
						maxTriNum = i;
					}
					numTrisSelected++;
				}
			}
			if (triGrid->bCalculateDosage && dosageHdl)
			{
				double dosage;
				(*dosageHdl)[i] += oilDensityInWaterColumn * timeStep / 3600;	// 
				dosage = (*dosageHdl)[i];
			}
		}
		//_SetHandleSize((Handle) concentrationH, (numTrisWithOil)*sizeof(double));
		if (numTrisWithOil>0)
		{
			_SetHandleSize((Handle) concentrationH, (numTrisWithOil)*sizeof(ConcTriNumPair));
			if (triGrid->fPercentileForMaxConcentration < 1)	
			{
				//qsort((*concentrationH),numTrisWithOil,sizeof(double),ConcentrationCompare);
				qsort((*concentrationH),numTrisWithOil,sizeof(ConcTriNumPair),ConcentrationCompare);
				j = (long)(triGrid->fPercentileForMaxConcentration*numTrisWithOil);	// round up or down?, for selected triangles?
				if (j>0) j--;
				//maxConc = (*concentrationH)[j];	// trouble with this percentile stuff if there are only a few values
				maxConc = (*concentrationH)[j].conc;	// trouble with this percentile stuff if there are only a few values
				maxTriNum = (*concentrationH)[j].triNum;	// trouble with this percentile stuff if there are only a few values
			}
		}
		//if (thisLEList->GetOilType() == CHEMICAL) 
		if (totalVol>0) avConcOverTriangles = totalMass / totalVol;
		//else
		//avConcOverTriangles = totalLEs * massInGrams / totalVol;
		// track concentrations over time, maybe define a new data type to hold everything...
		//if (triSelected) triGrid -> AddToOutputHdl(numTrisSelected>0 ? concInSelectedTriangles/numTrisSelected : 0,maxConc,model->GetModelTime());
		if (triSelected) triGrid -> AddToOutputHdl(numTrisSelected>0 ? avConcOverTriangles : 0,maxConc,model->GetModelTime());
		else triGrid -> AddToOutputHdl(numTrisSelected>0 ? avConcOverTriangles : 0, maxConc, model->GetModelTime());
		//if (triSelected) triGrid -> AddToOutputHdl(oilDensityInWaterColumn,model->GetModelTime());
		if ( bTimeToOutputData) // this will be messed up if there are triangles selected
			triGrid -> AddToTriAreaHdl(fTriAreaArray,numLevels);
		CreateDepthSlice(maxTriNum,&fDepthSliceArray);
		//CreateDepthSlice(maxTriNum,fDepthSliceArray);
		triGrid -> fMaxTri = maxTriNum; 
	}
	
done:
	if(numLEsInTri) {DisposeHandle((Handle)numLEsInTri); numLEsInTri=0;}
	if(massInTriInGrams) {DisposeHandle((Handle)massInTriInGrams); massInTriInGrams=0;}
	if(concentrationH) {DisposeHandle((Handle)concentrationH); concentrationH=0;}
	return;
}
//void TCompoundMap::TrackOutputData(TOLEList *thisLEList)
// might want to track each spill separately and combined
void TCompoundMap::TrackOutputData(void)
{	
	long i,n;
	TMap *map = 0;
	
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) 
	{
		mapList->GetListItem((Ptr)&map, i);
		//map -> TrackOutputData();
		if (map) TrackOutputDataFromMapIndex(i);
	}
	return;		
}
void TCompoundMap::TrackOutputDataInAllLayers(void)
{	// code goes here - for now just same as TrackOutputData, need to update
	long i,n;
	TMap *map = 0;
	
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) 
	{
		mapList->GetListItem((Ptr)&map, i);
		//map -> TrackOutputData();
		if (map) TrackOutputDataFromMapIndex(i);
	}
	return;		
}
/*void TCompoundMap::TrackOutputData(void)
 {	// need all LELists
 long i, j, numOfLEs, numLESets, numTri;
 LERec LE;
 Boolean bShowContours, bTimeZero, bTimeToOutputData = false, bThereIsSubsurfaceOil = false;
 LONGH numLEsInTri = 0;
 DOUBLEH massInTriInGrams = 0;
 TopologyHdl topH = 0;
 TDagTree *dagTree = 0;
 DOUBLEH  dosageHdl = 0;
 ConcTriNumPairH concentrationH = 0;
 TTriGridVel3D* triGrid = GetGrid3D(true);	
 Seconds modelTime = model->GetModelTime(),timeStep = model->GetTimeStep();
 Seconds startTime = model->GetStartTime();
 short oldIndex, nowIndex, massunits;
 double depthAtPt, density, LEmass, massInGrams;
 TLEList *thisLEList = 0;
 
 if (!triGrid) return; // some error alert, no depth info to check
 
 Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
 
 dagTree = triGrid -> GetDagTree();
 if(!dagTree)	return;
 topH = dagTree->GetTopologyHdl();
 if(!topH)	return;
 numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
 numLEsInTri = (LONGH)_NewHandleClear(sizeof(long)*numTri);
 if (!numLEsInTri)
 { TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
 massInTriInGrams = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
 if (!massInTriInGrams)
 { TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
 //concentrationH = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
 concentrationH = (ConcTriNumPairH)_NewHandleClear(sizeof(ConcTriNumPair)*numTri);
 if (!concentrationH)
 { TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
 
 //bTimeZero = (modelTime ==(*(TOLEList*)thisLEList).fDispersantData.timeToDisperse+startTime);
 nowIndex = (modelTime + timeStep - startTime) / model->LEDumpInterval;
 oldIndex = (modelTime - startTime) / model->LEDumpInterval;
 if(nowIndex > oldIndex ) bTimeToOutputData = true;
 
 numLESets = model->LESetsList->GetItemCount();	// code goes here, check bActive
 for (i = 0; i < numLESets; i++)
 {
 model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
 if (thisLEList->fLeType == UNCERTAINTY_LE)	
 continue;	// don't draw uncertainty for now...
 if (!thisLEList->IsActive()) continue;
 numOfLEs = thisLEList->numOfLEs;
 // density set from API
 //density =  GetPollutantDensity(thisLEList->GetOilType());	
 density = ((TOLEList*)thisLEList)->fSetSummary.density;	
 massunits = thisLEList->GetMassUnits();
 
 if (bTimeToOutputData)	// track budget at the same time
 {	// make budget table even if spill is not dispersed
 double amttotal,amtevap,amtbeached,amtoffmap,amtfloating,amtreleased,amtdispersed,amtremoved=0;
 Seconds timeAfterSpill;	
 // what to do for chemicals, amount dissolved?
 thisLEList->GetLEAmountStatistics(thisLEList->GetMassUnits(),&amttotal,&amtreleased,&amtevap,&amtdispersed,&amtbeached,&amtoffmap,&amtfloating,&amtremoved);
 BudgetTableData budgetTable; 
 // if chemical will need to get amount dissolved
 timeAfterSpill = nowIndex * model->LEDumpInterval;
 budgetTable.timeAfterSpill = timeAfterSpill;
 budgetTable.amountReleased = amtreleased;
 budgetTable.amountFloating = amtfloating;
 budgetTable.amountDispersed = amtdispersed;
 budgetTable.amountEvaporated = amtevap;
 budgetTable.amountBeached = amtbeached;
 budgetTable.amountOffMap = amtoffmap;
 budgetTable.amountRemoved = amtremoved;
 ((TOLEList*)thisLEList)->AddToBudgetTableHdl(&budgetTable);
 // still track for each list, for output total everything
 }
 
 // time has not been updated at this point (in DrawContours time has been updated)
 bShowContours = (*(TOLEList*)thisLEList).fDispersantData.bDisperseOil && model->GetModelTime() - model->GetStartTime() >= (*(TOLEList*)thisLEList).fDispersantData.timeToDisperse;
 bShowContours = bShowContours || (*(TOLEList*)thisLEList).fAdiosDataH;
 bShowContours = bShowContours || (*(TOLEList*)thisLEList).fSetSummary.z > 0;	// for bottom spill
 if (bShowContours) bThereIsSubsurfaceOil = true;
 if (!bShowContours) continue;	// no need to track in this case
 // total LEs from all spills..., numLEsInTri as a permanent field, maybe should calculate during Step()
 for (j = 0 ; j < numOfLEs ; j++) {
 LongPoint lp;
 long triIndex;
 thisLEList -> GetLE (j, &LE);
 //if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
 //if (LE.statusCode == OILSTAT_EVAPORATED) continue;	// shouldn't happen, temporary for dissolved chemicals 
 if (!(LE.statusCode == OILSTAT_INWATER)) continue;	// Windows compiler requires extra parentheses
 lp.h = LE.p.pLong;
 lp.v = LE.p.pLat;
 LEmass = GetLEMass(LE);	// will only vary for chemical with different release end time
 massInGrams = VolumeMassToGrams(LEmass, density, massunits);	// need to do this above too
 if (fContourDepth1==BOTTOMINDEX)
 {
 double depthAtLE = DepthAtPoint(LE.p);
 //if (LE.z > (depthAtLE-1.) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
 if (LE.z > (depthAtLE-fBottomRange) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
 {
 triIndex = dagTree -> WhatTriAmIIn(lp);
 if (triIndex>=0)  
 {
 (*numLEsInTri)[triIndex]++;
 (*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
 }
 }
 }
 else if (LE.z>fContourDepth1 && LE.z<=fContourDepth2) 
 {
 triIndex = dagTree -> WhatTriAmIIn(lp);
 if (triIndex>=0) 
 {
 (*numLEsInTri)[triIndex]++;
 (*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
 }
 }
 }
 }
 if (triGrid->bCalculateDosage)
 {
 dosageHdl = triGrid -> GetDosageHdl(true);
 }
 if (bThereIsSubsurfaceOil)
 {
 double triArea, triVol, oilDensityInWaterColumn, totalVol=0;
 long numLEsInTriangle,j,numLevels,totalLEs=0,totalMass=0,numDepths=0,maxTriNum=-1,numTrisWithOil = 0;
 double concInSelectedTriangles=0,maxConc=0,numTrisSelected=0,minLevel,maxLevel,range,avConcOverTriangles=0;
 if (!fContourLevelsH)
 if (!InitContourLevels()) return;
 numLevels = GetNumDoubleHdlItems(fContourLevelsH);
 if (fTriAreaArray)
 {delete [] fTriAreaArray; fTriAreaArray = 0;}
 fTriAreaArray = new double[numLevels];
 for (i=0;i<numLevels;i++)
 fTriAreaArray[i] = 0.;
 for (i=0;i<numTri;i++)
 {	// track ppm hours here
 double depthRange;
 depthAtPt=0;
 //WorldPoint centroid = {0,0};
 if (triSelected && !(*triSelected)[i]) continue;	// note this line keeps triareaarray from tracking for output..., dosage...
 //if (!triGrid->GetTriangleCentroidWC(i,&centroid))
 {
 //depthAtPt = DepthAtPoint(centroid);
 //depthAtPt = DepthAtCentroid(i);
 depthAtPt = DepthAtCentroidFromMapIndex(i,mapIndex);
 }
 triArea = (triGrid -> GetTriArea(i)) * 1000 * 1000;	// convert to meters
 if (!(fContourDepth1==BOTTOMINDEX))	
 {
 if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt > 0) depthRange = depthAtPt - fContourDepth1;
 else depthRange = fContourDepth2 - fContourDepth1;
 }
 else
 {
 //if (depthAtPt<1 && depthAtPt>0) depthRange = depthAtPt;	// should do a triangle volume 
 //else depthRange = 1.; // for bottom will always contour 1m
 if (depthAtPt<fBottomRange && depthAtPt>0) depthRange = depthAtPt;	// should do a triangle volume 
 else depthRange = fBottomRange; // for bottom will always contour 1m
 }
 //if (depthAtPt < depthRange && depthAtPt != 0) depthRange = depthAtPt;
 //if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt != 0) depthRange = depthAtPt - fContourDepth1;
 triVol = triArea * depthRange; // code goes here, check this depth range is ok at all vertices
 //triVol = triArea * (fContourDepth2 - fContourDepth1); // code goes here, check this depth range is ok at all vertices
 numLEsInTriangle = (*numLEsInTri)[i];
 if (numLEsInTriangle==0)
 continue;
 
 if (!(fContourDepth1==BOTTOMINDEX))		// need to decide what to do for bottom contour
 if (triGrid->CalculateDepthSliceVolume(&triVol,i,fContourDepth1,fContourDepth2)) goto done;
 oilDensityInWaterColumn = (*massInTriInGrams)[i] / triVol;	// units? milligrams/liter ?? for now gm/m^3
 
 //(*concentrationH)[numTrisWithOil] = oilDensityInWaterColumn;
 (*concentrationH)[numTrisWithOil].conc = oilDensityInWaterColumn;
 (*concentrationH)[numTrisWithOil].triNum = i;
 numTrisWithOil++;
 //for (j=0;j<numLevels-1;j++)
 for (j=0;j<numLevels;j++)
 {
 //if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
 if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
 {
 fTriAreaArray[j] = fTriAreaArray[j] + triArea/1000000;
 totalLEs += numLEsInTriangle;
 //totalMass += massInGrams;
 totalMass += (*massInTriInGrams)[i];
 totalVol += triVol;
 concInSelectedTriangles += oilDensityInWaterColumn;	// sum or track each one?
 if (oilDensityInWaterColumn > maxConc) 
 {
 maxConc = oilDensityInWaterColumn;
 numDepths = floor(depthAtPt)+1;	// split into 1m increments to track vertical slice
 maxTriNum = i;
 }
 numTrisSelected++;
 }
 }
 if (triGrid->bCalculateDosage && dosageHdl)
 {
 double dosage;
 (*dosageHdl)[i] += oilDensityInWaterColumn * timeStep / 3600;	// 
 dosage = (*dosageHdl)[i];
 }
 }
 //_SetHandleSize((Handle) concentrationH, (numTrisWithOil)*sizeof(double));
 if (numTrisWithOil>0)
 {
 _SetHandleSize((Handle) concentrationH, (numTrisWithOil)*sizeof(ConcTriNumPair));
 if (triGrid->fPercentileForMaxConcentration < 1)	
 {
 //qsort((*concentrationH),numTrisWithOil,sizeof(double),ConcentrationCompare);
 qsort((*concentrationH),numTrisWithOil,sizeof(ConcTriNumPair),ConcentrationCompare);
 j = (long)(triGrid->fPercentileForMaxConcentration*numTrisWithOil);	// round up or down?, for selected triangles?
 if (j>0) j--;
 //maxConc = (*concentrationH)[j];	// trouble with this percentile stuff if there are only a few values
 maxConc = (*concentrationH)[j].conc;	// trouble with this percentile stuff if there are only a few values
 maxTriNum = (*concentrationH)[j].triNum;	// trouble with this percentile stuff if there are only a few values
 }
 }
 //if (thisLEList->GetOilType() == CHEMICAL) 
 if (totalVol>0) avConcOverTriangles = totalMass / totalVol;
 //else
 //avConcOverTriangles = totalLEs * massInGrams / totalVol;
 // track concentrations over time, maybe define a new data type to hold everything...
 //if (triSelected) triGrid -> AddToOutputHdl(numTrisSelected>0 ? concInSelectedTriangles/numTrisSelected : 0,maxConc,model->GetModelTime());
 if (triSelected) triGrid -> AddToOutputHdl(numTrisSelected>0 ? avConcOverTriangles : 0,maxConc,model->GetModelTime());
 else triGrid -> AddToOutputHdl(numTrisSelected>0 ? avConcOverTriangles : 0, maxConc, model->GetModelTime());
 //if (triSelected) triGrid -> AddToOutputHdl(oilDensityInWaterColumn,model->GetModelTime());
 if ( bTimeToOutputData) // this will be messed up if there are triangles selected
 triGrid -> AddToTriAreaHdl(fTriAreaArray,numLevels);
 CreateDepthSlice(maxTriNum,&fDepthSliceArray);
 //CreateDepthSlice(maxTriNum,fDepthSliceArray);
 triGrid -> fMaxTri = maxTriNum; 
 }
 
 done:
 if(numLEsInTri) {DisposeHandle((Handle)numLEsInTri); numLEsInTri=0;}
 if(massInTriInGrams) {DisposeHandle((Handle)massInTriInGrams); massInTriInGrams=0;}
 if(concentrationH) {DisposeHandle((Handle)concentrationH); concentrationH=0;}
 return;
 }*/

double TCompoundMap::DepthAtPoint(WorldPoint wp)
{	// here need to check by priority
	float depth1,depth2,depth3;
	double depthAtPoint;	
	long mapIndex;
	InterpolationVal interpolationVal;
	FLOATH depthsHdl = 0;
	TTriGridVel3D* triGrid = 0;	
	//TTriGridVel3D* triGrid = GetGrid3D(false);	// don't use refined grid, depths aren't refined
	//NetCDFMover *mover = (NetCDFMover*)(model->GetMover(TYPE_NETCDFMOVER));
	//NetCDFMover *mover = (NetCDFMover*)(Get3DCurrentMover());
	
	mapIndex = WhichMapIsPtInWater(wp);
	if (mapIndex == -1) return -1.;	// off maps
	NetCDFMover *mover = dynamic_cast<NetCDFMover*>(Get3DCurrentMoverFromIndex(mapIndex));	// may not be netcdfmover?
	triGrid = GetGrid3DFromMapIndex(mapIndex);
	
	if (mover && mover->fVar.gridType==SIGMA_ROMS)
		return ((NetCDFMoverCurv*)mover)->GetTotalDepth(wp,-1);	// expand options here
	
	if (!triGrid) return -1; // some error alert, no depth info to check
	interpolationVal = triGrid->GetInterpolationValues(wp);
	depthsHdl = triGrid->GetDepths();
	if (!depthsHdl) return -1;	// some error alert, no depth info to check
	if (interpolationVal.ptIndex1<0)	
	{
		//printError("Couldn't find point in dagtree"); 
		return -1;
	}
	
	depth1 = (*depthsHdl)[interpolationVal.ptIndex1];
	depth2 = (*depthsHdl)[interpolationVal.ptIndex2];
	depth3 = (*depthsHdl)[interpolationVal.ptIndex3];
	depthAtPoint = interpolationVal.alpha1*depth1 + interpolationVal.alpha2*depth2 + interpolationVal.alpha3*depth3;
	
	return depthAtPoint;
}


long TCompoundMap::WhichMapIsPtIn(WorldPoint wp)
{
	long i,n;
	TMap *map = 0;
	
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) 
	{
		mapList->GetListItem((Ptr)&map, i);
		if (map -> InMap(wp)) return i;
	}
	return -1;	// means off all maps	
}

long TCompoundMap::WhichMapIsPtInWater(WorldPoint wp)
{
	long i,n;
	TMap *map = 0;
	
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) 
	{
		mapList->GetListItem((Ptr)&map, i);
		 if ((dynamic_cast<PtCurMap *>(map)) -> InWater(wp)) return i;
	}
	return -1;	// means off all maps	
}

Boolean TCompoundMap::ThereAreTrianglesSelected()
{
	TTriGridVel3D* triGrid = GetGrid3D(true);		// might we use this for 2D?
	if (!triGrid) return false;
	return triGrid->ThereAreTrianglesSelected();
	return false;
}

OSErr TCompoundMap::GetDepthAtMaxTri(long *maxTriIndex,double *depthAtPnt)	
{	// 
	long i,j,n,numOfLEs=0,numLESets,numDepths=0,numTri;
	TTriGridVel3D* triGrid = GetGrid3D(false);
	TDagTree *dagTree = 0;
	LONGH numLEsInTri = 0;
	DOUBLEH massInTriInGrams = 0;
	TopologyHdl topH = 0;
	LERec LE;
	OSErr err = 0;
	double triArea, triVol, oilDensityInWaterColumn, massInGrams, totalVol=0, depthAtPt = 0;
	long numLEsInTriangle,numLevels,totalLEs=0,maxTriNum=-1;
	double concInSelectedTriangles=0,maxConc=0;
	Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
	short massunits;
	double density, LEmass, halfLife;
	TLEList *thisLEList = 0;
	//short massunits = thisLEList->GetMassUnits();
	//double density =  thisLEList->fSetSummary.density;	// density set from API
	//double LEmass =  thisLEList->fSetSummary.totalMass / (double)(thisLEList->fSetSummary.numOfLEs);	
	
	dagTree = triGrid -> GetDagTree();
	if(!dagTree) return -1;
	topH = dagTree->GetTopologyHdl();
	if(!topH)	return -1;
	numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
	numLEsInTri = (LONGH)_NewHandleClear(sizeof(long)*numTri);
	if (!numLEsInTri)
	{ TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); err = -1; goto done; }
	massInTriInGrams = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
	if (!massInTriInGrams)
	{ TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); err = -1; goto done; }
	//numOfLEs = thisLEList->numOfLEs;
	//massInGrams = VolumeMassToGrams(LEmass, density, massunits);
	if (!fContourLevelsH)
		if (!InitContourLevels()) {err = -1; goto done;}
	numLevels = GetNumDoubleHdlItems(fContourLevelsH);
	
	numLESets = model->LESetsList->GetItemCount();
	for (i = 0; i < numLESets; i++)
	{
		model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if (thisLEList->fLeType == UNCERTAINTY_LE)	
			continue;	
		if (!((*(TOLEList*)thisLEList).fDispersantData.bDisperseOil && ((model->GetModelTime() - model->GetStartTime()) >= (*(TOLEList*)thisLEList).fDispersantData.timeToDisperse ) )
			&& !(*(TOLEList*)thisLEList).fAdiosDataH && !((*(TOLEList*)thisLEList).fSetSummary.z > 0)) 
			continue;
		numOfLEs = thisLEList->numOfLEs;
		// density set from API
		//density =  GetPollutantDensity(thisLEList->GetOilType());	
		density = ((TOLEList*)thisLEList)->fSetSummary.density;	
		halfLife = (*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.halfLife;
		massunits = thisLEList->GetMassUnits();
		
		for (j = 0 ; j < numOfLEs ; j++) 
		{
			LongPoint lp;
			long triIndex;
			thisLEList -> GetLE (j, &LE);
			//if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
			if (!(LE.statusCode == OILSTAT_INWATER)) continue;// Windows compiler requires extra parentheses
			lp.h = LE.p.pLong;
			lp.v = LE.p.pLat;
			LEmass = GetLEMass(LE,halfLife);	// will only vary for chemical with different release end time
			massInGrams = VolumeMassToGrams(LEmass, density, massunits);	// need to do this above too
			if (fContourDepth1==BOTTOMINDEX)
			{
				double depthAtLE = DepthAtPoint(LE.p);
				if (depthAtLE <= 0) continue;
				//if (LE.z > (depthAtLE-1.) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
				if (LE.z > (depthAtLE-fBottomRange) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
				{
					triIndex = dagTree -> WhatTriAmIIn(lp);
					//if (triIndex>=0) (*numLEsInTri)[triIndex]++;
					//if (triIndex>=0 && LE.pollutantType == CHEMICAL) (*massInTri)[triIndex]+=GetLEMass(LE);	// use weathering information
					if (triIndex>=0) 
					{
						(*numLEsInTri)[triIndex]++;
						(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
					}
				}
			}
			else if (LE.z>fContourDepth1 && LE.z<=fContourDepth2) 
			{
				triIndex = dagTree -> WhatTriAmIIn(lp);
				//if (triIndex>=0) (*numLEsInTri)[triIndex]++;
				//if (triIndex>=0 && LE.pollutantType == CHEMICAL) (*massInTri)[triIndex]+=GetLEMass(LE);	// use weathering information
				if (triIndex>=0) 
				{
					(*numLEsInTri)[triIndex]++;
					(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
				}
			}
		}
	}
	
	for (i=0;i<numTri;i++)
	{	
		depthAtPt=0;
		double depthRange;
		//WorldPoint centroid = {0,0};
		if (triSelected && !(*triSelected)[i]) continue;	
		//if (!triGrid->GetTriangleCentroidWC(i,&centroid))
		{
			//depthAtPt = DepthAtPoint(centroid);
			depthAtPt = DepthAtCentroid(i);
		}
		triArea = (triGrid -> GetTriArea(i)) * 1000 * 1000;	// convert to meters
		if (!(fContourDepth1==BOTTOMINDEX))
		{
			depthRange = fContourDepth2 - fContourDepth1;
		}
		else
		{
			//depthRange = 1.; // for bottom will always contour 1m 
			//if (depthAtPt<1 && depthAtPt>0) depthRange = depthAtPt;
			depthRange = fBottomRange; // for bottom will always contour 1m 
			if (depthAtPt<fBottomRange && depthAtPt>0) depthRange = depthAtPt;
		}
		//if (depthAtPt < depthRange && depthAtPt != 0) depthRange = depthAtPt;
		if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt > 0) depthRange = depthAtPt - fContourDepth1;
		triVol = triArea * depthRange; 
		//triVol = triArea * (fContourDepth2 - fContourDepth1); // code goes here, check this depth range is ok at all vertices
		numLEsInTriangle = (*numLEsInTri)[i];
		if (!(fContourDepth1==BOTTOMINDEX))		// need to decide what to do for bottom contour
			if (triGrid->CalculateDepthSliceVolume(&triVol,i,fContourDepth1,fContourDepth2)) goto done;
		/*if (thisLEList->GetOilType() == CHEMICAL) 
		 {
		 massInGrams = VolumeMassToGrams((*massInTri)[i], density, massunits);
		 oilDensityInWaterColumn = massInGrams / triVol;
		 }
		 else
		 oilDensityInWaterColumn = numLEsInTriangle * massInGrams / triVol; // units? milligrams/liter ?? for now gm/m^3
		 */
		if (numLEsInTriangle==0)
			continue;
		oilDensityInWaterColumn = (*massInTriInGrams)[i] / triVol;	// units? milligrams/liter ?? for now gm/m^3
		
		for (j=0;j<numLevels;j++)
		{
			if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
			{
				//fTriAreaArray[j] = fTriAreaArray[j] + triArea/1000000;
				//totalLEs += numLEsInTriangle;
				//totalVol += triVol;
				//concInSelectedTriangles += oilDensityInWaterColumn;	// sum or track each one?
				if (oilDensityInWaterColumn > maxConc) 
				{
					maxConc = oilDensityInWaterColumn;
					//numDepths = floor(depthAtPt)+1;	// split into 1m increments to track vertical slice
					maxTriNum = i;
				}
			}
		}
	}
	
	*depthAtPnt = depthAtPt;
	*maxTriIndex = maxTriNum;
done:
	if(numLEsInTri) {DisposeHandle((Handle)numLEsInTri); numLEsInTri=0;}
	if(massInTriInGrams) {DisposeHandle((Handle)massInTriInGrams); massInTriInGrams=0;}
	return err;
}

WorldPoint3D TCompoundMap::ReflectPoint(WorldPoint3D fromWPt,WorldPoint3D toWPt,WorldPoint3D wp)
{
	//WorldPoint3D movedPoint = model->TurnLEAlongShoreLine(fromWPt, wp, this);	// use length of fromWPt to beached point or to toWPt?
	WorldPoint3D movedPoint = TurnLEAlongShoreLine(fromWPt, wp, toWPt);	// use length of fromWPt to beached point or to toWPt?
	/*if (!InVerticalMap(movedPoint)) 
	 {
	 movedPoint.z = fromWPt.z;	// try not changing depth
	 if (!InVerticalMap(movedPoint))
	 movedPoint.p = fromWPt.p;	// use original point
	 }*/
	//movedPoint.z = toWPt.z; // attempt the z move
	// code goes here, check mixedLayerDepth?
	if (!InVerticalMap(movedPoint) || movedPoint.z == 0) // these points are supposed to be below the surface
	{
		double depthAtPt = DepthAtPoint(movedPoint.p);	// code goes here, a check on return value
		if (depthAtPt < 0) 
		{
			OSErr err = 0;
			return fromWPt;
		}
		if (depthAtPt==0)
			movedPoint.z = .1;
		if (movedPoint.z > depthAtPt) movedPoint.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
		//if (movedPoint.z > depthAtPt) movedPoint.z = GetRandomFloat(.7*depthAtPt,.99*depthAtPt);
		if (movedPoint.z <= 0) movedPoint.z = GetRandomFloat(.01*depthAtPt,.1*depthAtPt);
		//movedPoint.z = fromWPt.z;	// try not changing depth
		//if (!InVerticalMap(movedPoint))
		//movedPoint.p = fromWPt.p;	// use original point - code goes here, need to find a z in the map
	}
	return movedPoint;
}

