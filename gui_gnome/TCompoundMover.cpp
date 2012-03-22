#include "Cross.h"
#include "MapUtils.h"
#include "OUtils.h"
#include "EditWindsDialog.h"
#include "Uncertainty.h"
#include "GridCurMover.h"
#include "TideCurCycleMover.h"
#include "NetCDFMover.h"

#define REPLACE true

static VList sgObjects;
static CMyList *sMoverList=0;
static CMyList *sDialogList=0;
static CMyList *sDialogMapList=0;
static short NAME_COL;
static TCompoundMover	*sSharedDialogCompoundMover = 0;
static TCompoundMap	*sSavedCompoundMap = 0;
static CurrentUncertainyInfo sSharedCompoundUncertainyInfo; // used to hold the uncertainty dialog box info in case of a user cancel

CMyList *sMapList=0;
TCompoundMap	*sSharedDialogCompoundMap = 0;


TCompoundMap* CreateAndInitCompoundMap(char *path, WorldRect bounds);

TCompoundMover::TCompoundMover (TMap *owner, char *name) : TCurrentMover (owner, name)
{
	moverList = 0;
		
	bMoversOpen = TRUE;

	return;
}

void TCompoundMover::Dispose ()
{
	// code goes here, need to get rid of any compound map as well
	long i, n;
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
	TCurrentMover::Dispose ();
}



OSErr TCompoundMover::InitMover ()
{
	/*OSErr	err = noErr;
	 TMap * owner = this -> GetMoverMap();
	 err = TCurrentMover::InitMover ();
	 if(owner) refP = WorldRectCenter(owner->GetMapBounds());
	 return err;*/
	OSErr err = 0;
	moverList = new CMyList(sizeof(TMover *));
	if (!moverList)
	{ TechError("TCompoundMover::InitMover()", "new CMyList()", 0); return -1; }
	if (err = moverList->IList())
	{ TechError("TCompoundMover::InitMover()", "IList()", 0); return -1; }
	
	return 0;
}

OSErr TCompoundMover::DeleteItem(ListItem item)
{
	if (item.index == I_COMPOUNDNAME)
		return moverMap -> DropMover(dynamic_cast<TCompoundMover *>(this));
	
	return 0;
}


OSErr TCompoundMover::AddItem(ListItem item)	// is this being used?
{	// this should be used when Load is hit, or after ok
	Boolean	timeFileChanged = false;
	//short type, dItem;
	OSErr err = 0;
	
	if (item.index == I_MOVERS || item.index == I_UMOVERS || item.index == I_VMOVERS) 
	{
		//dItem = MyModalDialog (M21, mapWindow, (Ptr) &type, M21Init, M21Click);
		//if (dItem == M21LOAD)
		{
			//switch (type)
			{
				//case CURRENTS_MOVERTYPE:
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
								err = CATSSettingsDialog (dynamic_cast<TCATSMover *>(newMover), this->moverMap, &timeFileChanged);
								break;
							case TYPE_NETCDFMOVER:
							case TYPE_NETCDFMOVERCURV:
							case TYPE_NETCDFMOVERTRI:
							case TYPE_GRIDCURMOVER:
							case TYPE_PTCURMOVER:
							case TYPE_TRICURMOVER:
								err = newMover->SettingsDialog();
								break;
							default:
								printError("bad type in TMap::AddItem");
								break;
						}
						if(err)	{ newMover->Dispose(); delete newMover; newMover = 0;}
						
						if(newMover && !err)
						{	
							//if (!newMap) 
							{	// probably don't allow new map...
								//err = AddMoverToMap (this->moverMap, timeFileChanged, newMover);
								err = AddMover (newMover,0);
								if (!err && this->moverMap == model -> uMap)
								{
									//ChangeCurrentView(AddWRectBorders(((PtCurMover*)newMover)->fGrid->GetBounds(), 10), TRUE, TRUE);
									WorldRect newRect = newMover->GetGridBounds();
									ChangeCurrentView(AddWRectBorders(newRect, 10), TRUE, TRUE);	// so current loaded on the universal map can be found
								}
							}
							/*else
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
							 return -1; 
							 }
							 //if (newMover->IAm(TYPE_CATSMOVER3D) || newMover->IAm(TYPE_TRICURMOVER)) InitAnalysisMenu();
							 if (model->ThereIsA3DMover(&arrowDepth)) InitAnalysisMenu();	// want to have it come and go?
							 MySpinCursor();
							 }*/
						}	
					}
					//break;
				}
				
				
				//default: 
				//SysBeep (5); 
				//return -1;
			}
		}
	}
	
	return err;
}

Boolean TCompoundMover::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	switch (item.index) {
		case I_COMPOUNDNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					//if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
					}
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
	}
	
	
	if (buttonID == SETTINGSBUTTON) return TRUE;
	
	//	return TCurrentMover::FunctionEnabled(item, buttonID);
	return FALSE;
}

OSErr TCompoundMover::UpItem(ListItem item)
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
	
	return TMover::UpItem(item);
}

OSErr TCompoundMover::DownItem(ListItem item)
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
	
	return TMover::DownItem(item);
}

OSErr TCompoundMover::SettingsItem(ListItem item)
{
	TMap *newMap = 0;
	switch (item.index) {
		default:	// need new dialog here
			return CompoundMoverSettingsDialog(dynamic_cast<TCompoundMover *>(this), this -> moverMap, &newMap);	// if new map is added ? 
	}
	return 0;
}

Boolean TCompoundMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	TMap *newMap = 0;
	if (inBullet)
		switch (item.index) {
			case I_COMPOUNDNAME: bOpen = !bOpen; return TRUE;
			case I_COMPOUNDACTIVE:
			{
				bActive = !bActive;
				InvalMapDrawingRect();
				return TRUE;
			}
			case I_COMPOUNDCURRENT:
				bMoversOpen = !bMoversOpen; return TRUE;
				//return TRUE;
		}
	
	if (doubleClick)
	{
		CompoundMoverSettingsDialog(dynamic_cast<TCompoundMover *>(this), this -> moverMap, &newMap);	// if new map is added ? 
	}	
	
	return FALSE;
}

long TCompoundMover::GetListLength()
{
	long	i, n, listLength = 0;
	TMover *mover;
	
	listLength = 1;
	
	if (bOpen)
	{
		listLength += 1;		// add one for active box at top
		
		listLength += 1;		// one for title
		if (bMoversOpen)
		{
			n = moverList->GetItemCount() ;
			for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
				moverList->GetListItem((Ptr)&mover, i);
				listLength += mover->GetListLength();
			}
		}
	}
	
	return listLength;
}

ListItem TCompoundMover::GetNthListItem (long n, short indent, short *style, char *text)
{
	ListItem item = { dynamic_cast<TComponentMover *>(this), 0, indent, 0 };
	char *p, latS[20], longS[20], timeS[32],valStr[32],val2Str[32],unitStr[32];
	long i, m, count;
	TMover *mover;
	
	if (n == 0) {
		item.index = I_COMPOUNDNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Compound Current");
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}
	
	item.indent++;
	
	if (bOpen)
	{
		
		if (--n == 0) {
			item.index = I_COMPOUNDACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
		
		
		if (--n == 0) {
			item.index = I_COMPOUNDCURRENT;
			item.bullet = bMoversOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Current Movers");
			
			return item;
		}
		
		indent++;
		
		n = n - 1;
		if (bMoversOpen)
		{
			for (i = 0, m = moverList->GetItemCount() ; i < m ; i++) {
				moverList->GetListItem((Ptr)&mover, i);
				count = mover->GetListLength();
				if (count > n)
					return mover->GetNthListItem(n, indent + 1, style, text);
				
				n -= count;
			}
			
			
		}
		
	}
	
	return item;
}

Boolean TCompoundMover::DrawingDependsOnTime(void)
{
	/*Boolean depends = fVar.bShowArrows;
	 // if this is a constant current, we can say "no"
	 //if(this->GetNumTimesInFile()==1) depends = false;
	 if(this->GetNumTimesInFile()==1 && !(GetNumFiles()>1)) depends = false;
	 return depends;*/
	
	Boolean depends = false;
	long i, n;
	TMover *mover;
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
		moverList->GetListItem((Ptr)&mover, i);
		depends = mover->DrawingDependsOnTime();
		if (depends) break;
	}
	
	return depends;
}

void TCompoundMover::Draw(Rect r, WorldRect view)
{
	// loop through all the current movers and draw each
	// draw each of the movers
	long i, n;
	TMover *mover;
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
		moverList->GetListItem((Ptr)&mover, i);
		mover->Draw(r, view);
	}
	
}

OSErr TCompoundMover::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	char ourName[kMaxNameLen];
	long lVal;
	OSErr err = 0;
	
	// see if the message is of concern to us
	this->GetClassName(ourName);
	
	if(message->IsMessage(M_CREATEMOVER,ourName)) 
	{
		char moverName[kMaxNameLen];
		char typeName[64];
		char path[256];
		TCATSMover *mover = nil;
		Boolean unrecognizedType = false;
		Boolean needToInitMover = true;
		message->GetParameterString("NAME",moverName,kMaxNameLen);
		message->GetParameterString("TYPE",typeName,64);
		message->GetParameterString("PATH",path,256);
		ResolvePath(path);
		
		/*err = message->GetParameterAsLong("PATTERN",&lVal);
		 if(!err) 
		 { 
		 if(!strcmpnocase(typeName,"Cats")) 
		 {	// the cats mover needs a file and so is a special case
		 mover = CreateAndInitCatsCurrentsMover (this->moverMap,false,path,moverName);
		 }
		 ////////////// 
		 if(mover) 
		 {
		 mover -> refP = this -> refP;	// need to set mover's refP so it is drawn correctly, 3/28/00
		 switch(lVal)
		 {
		 case 1: this -> pattern1 = mover; break;
		 case 2: this -> pattern2 = mover; break;
		 default: printError("pattern value out of range in TCompoundMover"); break;
		 }
		 }
		 }*/
		model->NewDirtNotification();// tell model about dirt
	}
	else if(message->IsMessage(M_SETFIELD,ourName))
	{
		double val;
		char str[256];
		OSErr err = 0;
		
		////////////////
	}
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TCurrentMover::CheckAndPassOnMessage(message);
}

void TCompoundMover::SetShowDepthContours()
{
	long i,n;
	TMover *mover = 0;
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) 
	{	// movers should be listed in priority order
		moverList->GetListItem((Ptr)&mover, i);
		//should have separate arrowDepth for combined mover ?, otherwise which one to use?
		if (mover->IAm(TYPE_NETCDFMOVER)) /*OK*/(dynamic_cast<NetCDFMover*>(mover))->bShowDepthContours = !(dynamic_cast<NetCDFMover*>(mover))->bShowDepthContours;
	}
	return;
}

Boolean TCompoundMover::ShowDepthContourChecked()
{
	long i,n;
	TMover *mover = 0;
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) 
	{	// movers should be listed in priority order
		moverList->GetListItem((Ptr)&mover, i);
		//should have separate arrowDepth for combined mover ?, otherwise which one to use?
		if (mover->IAm(TYPE_NETCDFMOVER) && /*OK*/(dynamic_cast<NetCDFMover*>(mover))->bShowDepthContours == true) return true;
	}
	return false;
}


TCurrentMover* TCompoundMover::AddCurrent(OSErr *err,TCompoundMap **compoundMap)	
{	// this should be used when Load is hit, or after ok
	Boolean	timeFileChanged = false;
	//short type, dItem;
	//OSErr err = 0;
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
					*err = CATSSettingsDialog (dynamic_cast<TCATSMover *>(newMover), this->moverMap, &timeFileChanged);
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
			
			//if (newMap) newMap->Dispose(); delete newMap; newMap = 0; //  need to track all maps for boundaries in a list
			if(newMover && !(*err))
			{	
				if (!newMap) 
				{	
					//err = AddMoverToMap (this->moverMap, timeFileChanged, newMover);
					*err = /*CHECK*/dynamic_cast<TCompoundMover *>(this)->AddMover(newMover,0);
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
					//if (already is an embedded map) add map to the list, else create embedded map and add map to list
					//if (sSharedDialogCompoundMap)
					if (sSharedDialogCompoundMap)
					{
						WorldRect mapBounds = newMap->GetMapBounds(), boundsUnion, origBounds;
						origBounds = sSharedDialogCompoundMap->GetMapBounds();
						//sSharedDialogCompoundMap->AddMap(newMap,0);
						*err=sMapList->AppendItem((Ptr)&newMap);
						
						boundsUnion = UnionWRect(mapBounds,origBounds);
						sSharedDialogCompoundMap->SetMapBounds(boundsUnion);
						
					}
					else 
					{						
						WorldRect mapBounds = newMap->GetMapBounds();
						sSharedDialogCompoundMap = CreateAndInitCompoundMap("",mapBounds);
						//sSharedDialogCompoundMap->AddMap(newMap,0);
						if(!sMapList)
						{
							sMapList = new CMyList(sizeof(TMap *));
							if(!sMapList) *err = -1;
							if(sMapList->IList()) *err = -1;
						}
						*err=sMapList->AppendItem((Ptr)&newMap);
					}
					//*err = model -> AddMap(sSharedDialogCompoundMap, 0);
					//if (!*err) *err = AddMoverToMap(sSharedDialogCompoundMap, timeFileChanged, newMover);
					*err = AddMoverToMap (newMap, timeFileChanged, newMover);
					if (!*err) newMover->SetMoverMap(newMap);
					else
					{
						newMap->Dispose(); delete newMap; newMap = 0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return 0; 
					}
					newMover->bIAmPartOfACompoundMover = true;
					newMap->bIAmPartOfACompoundMap = true;
					//if (newMover->IAm(TYPE_CATSMOVER3D) || newMover->IAm(TYPE_TRICURMOVER)) InitAnalysisMenu();
					//if (model->ThereIsA3DMover(&arrowDepth)) InitAnalysisMenu();	// want to have it come and go?
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
}

/////////////////////////////////////////////////

void UpdateDisplayWithMoverNamesSet(DialogPtr dialog,char* moverName)
{	// this is display outside the box
	char namestr[256];
	//StringWithoutTrailingZeros(namestr,moverName,2);
	strcpy(namestr,moverName);
	mysetitext(dialog,COMPOUNDCURNAME,namestr);	
	
	//mysetitext(dialog,COMPOUNDDLGPAT2NAME,namestr);	
	
	/*Float2EditText(dialog,EPU,dvals.value.u,6);
	 Float2EditText(dialog,EPV,dvals.value.v,6);
	 Float2EditText(dialog,EPTEMP,dvals.temp,2);
	 Float2EditText(dialog,EPSAL,dvals.sal,2);*/
}


static void UpdateDisplayWithCurSelection(DialogPtr dialog)
{
	TMover* curMover;
	Point pos,mp;
	long curSelection;
	char nameStr[256];
	
	if(!VLAddRecordRowIsSelected(&sgObjects))
	{	// set the item text
		{
			VLGetSelect(&curSelection,&sgObjects);
			sMoverList->GetListItem((Ptr)&curMover,curSelection);
		}
		if (curMover) strcpy(nameStr,curMover->className);
		
		UpdateDisplayWithMoverNamesSet(dialog,nameStr);
	}
	
	//ShowHideAutoIncrement(dialog,curSelection); // JLM 9/17/98
}

static void SelectNthRow(DialogPtr dialog,long nrow)
{
	VLSetSelect(nrow, &sgObjects); 
	if(nrow > -1)
	{
		VLAutoScroll(&sgObjects);
	}
	VLUpdate(&sgObjects);
	UpdateDisplayWithCurSelection(dialog);	
}

static OSErr AddReplaceRecord(DialogPtr dialog,/*Boolean incrementDepth,*/Boolean replace,TMover* theMover)
{
	long n,itemnum,curSelection;
	OSErr err=0;
	
	if(!err)
	{
		// will need to define InsertSorted for the CDOG profiles data type, sort by depth
		//err=sMoverList->InsertSorted ((Ptr)&theMover,&itemnum,false);// false means don't allow duplicate times
		err=sMoverList->AppendItem((Ptr)&theMover);	// list of names vs list of movers
		
		if(!err) // new record
		{
			itemnum = sgObjects.numItems;
			VLAddItem(1,&sgObjects);
			VLSetSelect(itemnum, &sgObjects); 
			VLAutoScroll(&sgObjects);
			VLUpdate(&sgObjects);
			//if(incrementDepth)IncrementDepth(dialog,dvals.depth);
		}
		/*else if(err == -2) // found existing record. Replace if okay to replace
		 {	// not sure if need replace option, at least not based on matching name
		 if(replace)
		 {
		 sMoverList->DeleteItem(itemnum);
		 VLDeleteItem(itemnum,&sgObjects);
		 err = AddReplaceRecord(dialog,!INCREMENT_DEPTH,REPLACE,&theMover);
		 VLUpdate(&sgObjects);
		 //if(incrementDepth)IncrementDepth(dialog,dvals.depth);
		 err=0;
		 }
		 else
		 {
		 printError("A record with the specified depth already exists."
		 "If you want to edit the existing record, select it."
		 "If you want to add a new record, change the specified depth.");
		 VLUpdate(&sgObjects);
		 }
		 }*/
		else SysBeep(5);
	}
	return err;
}

void DisposeCOMPOUNDDLGStuff(void)
{
	if(sMoverList)
	{
		sMoverList->Dispose();// JLM 12/14/98
		delete sMoverList;
		sMoverList = 0;
	}
	
	if(sMapList)
	{
		sMapList->Dispose();// JLM 12/14/98
		delete sMapList;
		sMapList = 0;
	}
	
	//?? VLDispose(&sgObjects);// JLM 12/10/98, is this automatic on the mac ??
	memset(&sgObjects,0,sizeof(sgObjects));// JLM 12/10/98
}

void ShowHideTCompoundDialogItems(DialogPtr dialog)
{
	TCompoundMover	*mover = sSharedDialogCompoundMover;
	//short moverCode = GetPopSelection(dialog, COMPOUNDDLGMOVERTYPES);
	Boolean showPat2Items = FALSE, showScaleByItems = FALSE;
	
	return;
	
}

short CompoundDlgClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short 			item;
	OSErr 			err = 0;
	TCompoundMover	*mover = sSharedDialogCompoundMover;
	TMover			*currentMover=0;
	TMap			*currentMap = 0;
	long 			i,j,n,m,curSelection;
	Point 			pos;
	
	switch (itemNum) {
			
		case COMPOUNDDLGOK:
		{
			if(sMoverList)
			{
				//DepthValuesSetH dvalsh = sgDepthValuesH;
				n = sMoverList->GetItemCount();
				if(n == 0)
				{	// no items are entered, tell the user
					char msg[512],buttonName[64];
					GetWizButtonTitle_Cancel(buttonName);
					sprintf(msg,"You have not entered any data values.  Either enter data values and use the 'Load' button, or use the '%s' button to exit the dialog.",buttonName);
					printError(msg);
					break;
				}
				
				// check that all the values are in range - if there is some range
				// or may allow the user to change units
				for(i=0;i<n;i++)
				{
					char errStr[256] = "";
					err=sMoverList->GetListItem((Ptr)&mover,i);
					if(err) {SysBeep(5); break;}// this shouldn't ever happen
				}
				
				/////////////
				// point of no return
				//////////////
				//m = sDialogList->GetItemCount() ;
				/*if (n==0)
				 {
				 ClearList();
				 //{	// not sure if want movers or mover names here or maybe another list...
				 dvalsh = (DepthValuesSetH)_NewHandle(n*sizeof(DepthValuesSet));
				 if(!dvalsh)
				 {
				 TechError("EditProfilesClick:OKAY", "_NewHandle()", 0);
				 //return EPCANCEL;
				 break; // make them cancel so that code gets executed
				 }
				 sgDepthValuesH = dvalsh;
				 }
				 else
				 {
				 _SetHandleSize((Handle)dvalsh,n*sizeof(DepthValuesSet));
				 if(_MemError())
				 {
				 TechError("EditProfilesClick:OKAY", "_NewHandle()", 0);
				 //return EPCANCEL;
				 break; // make them cancel, so that code gets executed
				 }
				 }*/
				
				
				sDialogList->ClearList();
				for(i=0;i<n;i++)
				{
					if(err=sMoverList->GetListItem((Ptr)&currentMover,i))return COMPOUNDDLGOK;
					err=sDialogList->AppendItem((Ptr)&currentMover);
					if(err)return err;
				}
				if (sSharedDialogCompoundMap)
				{
					if(!sDialogMapList) 
					{
						//sSharedDialogCompoundMap = (TCompoundMap*)ownerMap;
						sDialogMapList = sSharedDialogCompoundMap->mapList;
						//sDialogMapList = new CMyList(sizeof(TMap *));
						//if(!sDialogMapList)return COMPOUNDDLGOK;
						//if(sDialogMapList->IList())return COMPOUNDDLGOK;
					}
					
					sDialogMapList->ClearList();
					m = sMapList->GetItemCount() ;
					for (i=0;i<n;i++)
					{
						if(err=sMoverList->GetListItem((Ptr)&currentMover,i))return COMPOUNDDLGOK;
						for (j=0;j<m;j++)// could have more maps? need to delete maps as we go...
						{
							if(err=sMapList->GetListItem((Ptr)&currentMap,j))return COMPOUNDDLGOK;
							if (currentMover->moverMap == currentMap)
							{
								err=sDialogMapList->AppendItem((Ptr)&currentMap);
								if (i==0) /*OK*/(dynamic_cast<PtCurMap *>(currentMap))->bDrawContours=true; else (dynamic_cast<PtCurMap *>(currentMap))->bDrawContours=false;
								break;
							}
						}
					}
				}
				
			}
			////////////////////////////////////////////////////////////
			DisposeCOMPOUNDDLGStuff();
			return COMPOUNDDLGOK;
		}
			break;
			
		case COMPOUNDDLGCANCEL:
			// delete any new patterns, restore original patterns
			//DeleteIfNotOriginalPattern(mover -> pattern1);
			//DeleteIfNotOriginalPattern(mover -> pattern2);
			//mover -> pattern1 = sSharedOriginalPattern1;
			//mover -> pattern2 = sSharedOriginalPattern2;
			
			DisposeCOMPOUNDDLGStuff();
			return COMPOUNDDLGCANCEL;
			break;
			
			
			/*case COMPOUNDDLGPAT1SELECT:
			 {	// this should be used when Load is hit, or after ok
			 Boolean	timeFileChanged = false;
			 OSErr err = 0;
			 
			 TMap *newMap = 0;
			 TCurrentMover *newMover = CreateAndInitCurrentsMover (sSharedDialogCompoundMover->moverMap,true,0,0,&newMap);
			 if (newMover)
			 {
			 switch (newMover->GetClassID()) 
			 {
			 case TYPE_CATSMOVER:
			 case TYPE_TIDECURCYCLEMOVER:
			 case TYPE_CATSMOVER3D:
			 err = CATSSettingsDialog ((TCATSMover*)newMover, sSharedDialogCompoundMover->moverMap, &timeFileChanged);
			 break;
			 case TYPE_NETCDFMOVER:
			 case TYPE_NETCDFMOVERCURV:
			 case TYPE_NETCDFMOVERTRI:
			 case TYPE_GRIDCURMOVER:
			 case TYPE_PTCURMOVER:
			 case TYPE_TRICURMOVER:
			 err = newMover->SettingsDialog();
			 break;
			 default:
			 printError("bad type in TCompoundMover::AddItem");
			 break;
			 }
			 if(err)	{ newMover->Dispose(); delete newMover; newMover = 0;}
			 
			 if(newMover && !err)
			 {	
			 err = sSharedDialogCompoundMover->AddMover (newMover,0);
			 if (!err && sSharedDialogCompoundMover->moverMap == model -> uMap)
			 {
			 //ChangeCurrentView(AddWRectBorders(((PtCurMover*)newMover)->fGrid->GetBounds(), 10), TRUE, TRUE);
			 WorldRect newRect = newMover->GetGridBounds();
			 ChangeCurrentView(AddWRectBorders(newRect, 10), TRUE, TRUE);	// so current loaded on the universal map can be found
			 }
			 newMover->bIAmPartOfACompoundMover = true;
			 }
			 }	
			 // put something on the dialog to list the mover
			 //SysBeep (5); 
			 }
			 //DeleteIfNotOriginalPattern(mover -> pattern1);
			 //mover -> pattern1 = CreateAndInitCatsCurrentsMover (mover -> moverMap,true,0,0);
			 ////
			 break;*/
			
		case COMPOUNDDLGDELETEALL:
			sMoverList->ClearList();
			sMapList->ClearList();	// need to delete further?
			VLReset(&sgObjects,1);
			UpdateDisplayWithCurSelection(dialog);
			break;
		case COMPOUNDDLGMOVEUP:
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				if (curSelection>0) 
				{
					sMoverList->SwapItems(curSelection,curSelection-1);
					if (sMapList) sMapList->SwapItems(curSelection,curSelection-1);	// need to check that maps correspond
					VLSetSelect(curSelection-1,&sgObjects);
					--curSelection;
				}
			}
			VLUpdate(&sgObjects);
			//VLReset(&sgObjects,1);
			UpdateDisplayWithCurSelection(dialog);
			break;
		case COMPOUNDDLGDELETEROWS_BTN:
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				sMoverList->DeleteItem(curSelection);
				if (sMapList) sMapList->DeleteItem(curSelection);	// check that the two go together
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
			
		case COMPOUNDDLGREPLACE:
			//err = RetrieveIncrementDepth(dialog);
		{
			TCompoundMap* newMap = 0;
			TCurrentMover* curMover = sSharedDialogCompoundMover->AddCurrent(&err,&newMap);
			//sMoverList = 
			if(err) break;
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				//TMover* thisCurMover=0;
				//err=GetDepthVals(dialog,&dvals);
				if(err) break;
				
				if(curSelection==sMoverList->GetItemCount())
				{
					// replacing blank record
					err = AddReplaceRecord(dialog,!REPLACE,(TMover*)curMover);
					//if (sMapList && newMap) err=sMapList->AppendItem((Ptr)&newMap);	
					if (!err) SelectNthRow(dialog, curSelection+1 ); 
				}
				else // replacing existing record
				{	// not allowed right now
					VLGetSelect(&curSelection,&sgObjects);
					sMoverList->DeleteItem(curSelection);
					//if (sMapList) sMapList->DeleteItem(curSelection);
					VLDeleteItem(curSelection,&sgObjects);		
					err = AddReplaceRecord(dialog,REPLACE,(TMover*)curMover);
					//if (sMapList && newMap) err=sMapList->AppendItem((Ptr)&newMap);	
				}
			}
		}
			break;
			
		case COMPOUNDDLGLIST:
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
				TMover* mover=0;
				sMoverList->GetListItem((Ptr)&mover,sMoverList->GetItemCount()-1);
				//err = RetrieveIncrementDepth(dialog);
				if(err) break;
				//IncrementDepth(dialog,dvals.depth);
			}
			UpdateDisplayWithCurSelection(dialog);
			break;
			
			
	}
	
	return 0;
}

///////////////////////////////////////////////////////////////////////////

/*static PopInfoRec compoundIntDlgPopTable[] = {
 { COMPOUNDDLG, nil, COMPOUNDDLGPAT1SPEEDUNITS, 0, pSPEEDUNITS,  0, 1, FALSE, nil },
 { COMPOUNDDLG, nil, COMPOUNDDLGPAT2SPEEDUNITS, 0, pSPEEDUNITS2, 0, 1, FALSE, nil },
 { COMPOUNDDLG, nil, COMPOUNDDLGLATDIR, 0, pNORTHSOUTH1, 0, 1, FALSE, nil },
 { COMPOUNDDLG, nil, COMPOUNDDLGLONGDIR, 0, pEASTWEST1, 0, 1, FALSE, nil },
 { COMPOUNDDLG, nil, COMPOUNDDLGMOVERTYPES, 0, pMOVERTYPES2, 0, 1, FALSE, nil },
 { COMPOUNDDLG, nil, COMPOUNDDLGSCALEBYTYPES, 0, pSCALEBYTYPES, 0, 1, FALSE, nil }
 };*/

void DrawMoverNameList(DialogPtr w, RECTPTR r, long n)
{
	char s[256]/*,nameStr[64]*/;
	//DepthValuesSet dvals;
	TMover* mover = 0;
	
	if(n == sgObjects.numItems-1)
	{
		strcpy(s,"****");
	 	MyMoveTo(NAME_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	//MyMoveTo(U_COL-stringwidth(s)/2,r->bottom);
		//drawstring(s);
	 	return; 
	}
	
	sMoverList->GetListItem((Ptr)&mover,n);
	if (mover) 
		strcpy(s, mover->className);
	else strcpy(s,"****");	// shouldn't happen
	//MyMoveTo(NAME_COL-stringwidth(s)/2,r->bottom);
	MyMoveTo(NAME_COL-stringwidth("Pattern Name")/2,r->bottom);
	//MyMoveTo(NAME_COL/*-stringwidth("Pattern Name")/2*/,r->bottom);
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

OSErr CompoundDlgInit(DialogPtr dialog, VOIDPTR data)
{
	char 			s[256];
	TCompoundMover	*compoundMover;
	Rect r = GetDialogItemBox(dialog, COMPOUNDDLGLIST);
	char blankStr[32];
	strcpy(blankStr,"");
	TMover *currentMover=0;
	TMap *currentMap = 0;
	long i,n,m;
	OSErr err = 0;
	short IBMoffset;
	
	compoundMover = sSharedDialogCompoundMover;
	if (!compoundMover) return -1;
	//sSharedOriginalPattern1 = mover -> pattern1;
	//sSharedOriginalPattern2 = mover -> pattern2;
	sSharedCompoundUncertainyInfo = compoundMover -> GetCurrentUncertaintyInfo();
	
	memset(&sgObjects,0,sizeof(sgObjects));// JLM 12/10/98
	
	{
		sMoverList = new CMyList(sizeof(TMover *));
		if(!sMoverList)return -1;
		if(sMoverList->IList())return -1;
		
		n = sDialogList->GetItemCount() ;
		if (n>0)
		{
			// copy list to temp list
			for(i=0;i<n;i++)
			{	// dvals is a list too in this case...
				//dvals=(*dvalsh)[i];
				sDialogList->GetListItem((Ptr)&currentMover, i);
				err=sMoverList->AppendItem((Ptr)&currentMover);
				if(err)return err;
			}
		}
		else  n=0;
		
		n++; // Always have blank row at bottom
		
		err = VLNew(dialog, COMPOUNDDLGLIST, &r,n, DrawMoverNameList, &sgObjects);
		if(err) return err;
	}
	if (sSharedDialogCompoundMap)
	{
		sMapList = new CMyList(sizeof(TMap *));
		if(!sMapList)return -1;
		if(sMapList->IList())return -1;
		
		m = sDialogMapList->GetItemCount() ;
		if (m>0)
		{
			// copy list to temp list
			for(i=0;i<m;i++)
			{	// dvals is a list too in this case...
				//dvals=(*dvalsh)[i];
				sDialogMapList->GetListItem((Ptr)&currentMap, i);
				err=sMapList->AppendItem((Ptr)&currentMap);
				if(err)return err;
			}
		}
		else  m=0;
	}
	//RegisterPopTable (compoundIntDlgPopTable, sizeof (compoundIntDlgPopTable) / sizeof (PopInfoRec));
	//RegisterPopUpDialog (COMPOUNDDLG, dialog);
	
	SetDialogItemHandle(dialog, COMPOUNDDLGHILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, COMPOUNDDLGLIST, (Handle)MoverNamesListUpdate);
	
#ifdef IBM
	IBMoffset = r.left;
#else 
	IBMoffset = 0;
#endif
	r = GetDialogItemBox(dialog, COMPOUNDNAME_LIST_LABEL);NAME_COL=(r.left+r.right)/2-IBMoffset;
	
	//ShowUnscaledComponentValues(dialog,nil,nil);
	ShowHideTCompoundDialogItems(dialog);
	UpdateDisplayWithCurSelection(dialog);
	
	return 0;
}

OSErr CompoundMoverSettingsDialog(TCompoundMover *theMover, TMap *ownerMap, TMap **newMap)
{
	short item;
	OSErr err = 0;
	
	if (!theMover)return -1;
	// code goes here, need to deal with deleting the map, canceling, etc.
	// delete sub maps as we go, but check that they are the corresponding mover map
	// check if we delete all movers that any map must be deleted
	// also need to handle changing map - calling routine needs to reset I think
	sSharedDialogCompoundMover = theMover;
	sSavedCompoundMap = nil;
	if (ownerMap->IAm(TYPE_COMPOUNDMAP))
	{
		sSharedDialogCompoundMap = dynamic_cast<TCompoundMap *>(ownerMap);
		// copy map to sSavedCompoundMap
		/*OK*/ sDialogMapList = (dynamic_cast<TCompoundMap *>(ownerMap))->mapList;
	}
	else
	{
		sSharedDialogCompoundMap = nil;
	}
	//if(sSharedDialogCompoundMap) 
	//err = sSharedDialogCompoundMap->MakeClone((TClassID**)&sSavedCompoundMap);
	// on cancel become clone, newMap should be nil in some cases
	
	sDialogList = theMover->moverList;
	item = MyModalDialog(COMPOUNDDLG, mapWindow, 0, CompoundDlgInit, CompoundDlgClick);
	sSharedDialogCompoundMover = 0;
	
	//if we clone we have to copy all the maps movers as well...
	//if(COMPOUNDDLGCANCEL == item)	// what if cancel after having added map
	//if (sSharedDialogCompoundMap) err = sSharedDialogCompoundMap->BecomeClone(sSavedCompoundMap);
	//else	// they did not cancel, so we don't need the saved clones
	//if(sSavedCompoundMap) {sSavedCompoundMap->Dispose(); delete sSavedCompoundMap; sSavedCompoundMap = nil;}
	*newMap = sSharedDialogCompoundMap;
	//sSharedDialogCompoundMap = 0;
	
	if(COMPOUNDDLGOK == item)	model->NewDirtNotification();// tell model about dirt
	return COMPOUNDDLGOK == item ? 0 : -1;
}

OSErr TCompoundMover::DropMover(TMover *theMover)
{
	long 	i;
	OSErr	err = noErr;
	
	if (moverList->IsItemInList((Ptr)&theMover, &i))
	{
		if (err = moverList->DeleteItem(i))
		{ TechError("TCompoundMover::DropMover()", "DeleteItem()", err); return err; }
	}
	SetDirty (true);
	
	return err;
}

#define TCompound_FileVersion 1
OSErr TCompoundMover::Write(BFPB *bfpb)
{
	long i,n,version = TCompound_FileVersion, numWinds;
	ClassID id = GetClassID ();
	OSErr err = 0;
	char c;
	TMover* mover = 0;
	
	if (err = TCurrentMover::Write(bfpb)) return err;
	
	// save class fields
	
	StartReadWriteSequence("TCompoundMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	if (err = WriteMacValue(bfpb,bMoversOpen)) return err;
	
	n = moverList->GetItemCount();
	if (err = WriteMacValue(bfpb,n)) return err;
	
	for (i = 0 ; i < n ; i++) {
		moverList->GetListItem((Ptr)&mover, i);
		id = mover->GetClassID();
		if (err = WriteMacValue(bfpb,id)) return err;
		if (err = mover->Write(bfpb)) return err;
	}
	
	
	return 0;
}

OSErr TCompoundMover::Read(BFPB *bfpb)
{	// code goes here, if on a CompoundMap movers need to connect to a Map
	long i, version, numMovers;
	OSErr err = 0;
	ClassID id;
	char c;
	TMover *mover;
	
	if (err = TCurrentMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("TCompoundMover::Write()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TCompoundMover::Read()", "id != GetClassID", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > TCompound_FileVersion) { printSaveFileVersionError(); return -1; }
	
	if (err = ReadMacValue(bfpb, &bMoversOpen)) return err;
	if (err = ReadMacValue(bfpb,&numMovers)) return err;
	
	// allocate and read each of the movers
	
	for (i = 0 ; i < numMovers ; i++) {
		if (err = ReadMacValue(bfpb,&id)) return err;
		mover = 0;
		switch (id) {
				//case TYPE_MOVER: mover = new TMover(this, ""); break;
				//case TYPE_RANDOMMOVER: mover = new TRandom(this, ""); break;
			case TYPE_CATSMOVER: mover = new TCATSMover(this->moverMap, ""); break;
				//case TYPE_WINDMOVER: mover = new TWindMover(this, ""); break;
			case TYPE_COMPONENTMOVER: mover = new TComponentMover(this->moverMap, ""); break;
				//case TYPE_CONSTANTMOVER: mover = new TConstantMover(this, ""); break;
			case TYPE_PTCURMOVER: mover = new PtCurMover(this->moverMap, ""); break;
			case TYPE_GRIDCURMOVER: mover = new GridCurMover(this->moverMap, ""); break;
			case TYPE_NETCDFMOVER: mover = new NetCDFMover(this->moverMap, ""); break;
			case TYPE_NETCDFMOVERCURV: mover = new NetCDFMoverCurv(this->moverMap, ""); break;
			case TYPE_NETCDFMOVERTRI: mover = new NetCDFMoverTri(this->moverMap, ""); break;
				//case TYPE_NETCDFWINDMOVER: mover = new NetCDFWindMover(this, ""); break;
				//case TYPE_NETCDFWINDMOVERCURV: mover = new NetCDFWindMoverCurv(this, ""); break;
				//case TYPE_GRIDWINDMOVER: mover = new GridWindMover(this, ""); break;
				//case TYPE_RANDOMMOVER3D: mover = new TRandom3D(this, ""); break;
			case TYPE_CATSMOVER3D: mover = new TCATSMover3D(this->moverMap, ""); break;
			case TYPE_TRICURMOVER: mover = new TriCurMover(this->moverMap, ""); break;
			case TYPE_TIDECURCYCLEMOVER: mover = new TideCurCycleMover(this->moverMap, ""); break;
			default: printError("Unrecognized mover type in TMap::Read()."); return -1;
		}
		if (!mover)
		{ TechError("TCompoundMover::Read()", "new TMover()", 0); return -1; };
		if (!err) err = mover->InitMover();
		
		if (!err) err = mover->Read(bfpb);
		if (!err) {
			err = AddMover(mover, 0);
			if(err)  
				TechError("TCompoundMover::Read()", "AddMover()",err);
		}
		
		if(err)
		{ delete mover; mover = 0; return err;}
		
	}
	
	return err;
}

CurrentUncertainyInfo TCompoundMover::GetCurrentUncertaintyInfo ()
{
	CurrentUncertainyInfo	info;
	
	memset(&info,0,sizeof(info));
	info.setEddyValues = FALSE;
	info.fUncertainStartTime	= this -> fUncertainStartTime;
	info.fDuration				= this -> fDuration;
	info.fEddyDiffusion			= 0;		
	info.fEddyV0				= 0;		
	info.fDownCurUncertainty	= this -> fDownCurUncertainty;	
	info.fUpCurUncertainty		= this -> fUpCurUncertainty;	
	info.fRightCurUncertainty	= this -> fRightCurUncertainty;	
	info.fLeftCurUncertainty	= this -> fLeftCurUncertainty;	
	
	return info;
}

void TCompoundMover::SetCurrentUncertaintyInfo (CurrentUncertainyInfo info)
{
	
	this -> fUncertainStartTime		= info.fUncertainStartTime;
	this -> fDuration 				= info.fDuration;
	this -> fDownCurUncertainty 	= info.fDownCurUncertainty;	
	this -> fUpCurUncertainty 		= info.fUpCurUncertainty;	
	this -> fRightCurUncertainty 	= info.fRightCurUncertainty;	
	this -> fLeftCurUncertainty 	= info.fLeftCurUncertainty;	
	
	return;
}

Boolean TCompoundMover::CurrentUncertaintySame (CurrentUncertainyInfo info)
{
	if (this -> fUncertainStartTime	== info.fUncertainStartTime 
		&&	this -> fDuration 				== info.fDuration
		&&	this -> fDownCurUncertainty 	== info.fDownCurUncertainty	
		&&	this -> fUpCurUncertainty 		== info.fUpCurUncertainty	
		&&	this -> fRightCurUncertainty 	== info.fRightCurUncertainty	
		&&	this -> fLeftCurUncertainty 	== info.fLeftCurUncertainty	)
		return true;
	else return false;
}

