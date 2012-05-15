
#include "TRandom3D.h"
#include "CROSS.H"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT TRANDOM3D
#endif
#endif

extern TRandom3D *sharedRMover3D;

TRandom3D::TRandom3D (TMap *owner, char *name) : TRandom (owner, name)
{
	//fDiffusionCoefficient = 100000; //  cm**2/sec 
	//memset(&fOptimize,0,sizeof(fOptimize));
	fVerticalDiffusionCoefficient = 5; //  cm**2/sec	
	//fVerticalBottomDiffusionCoefficient = .01; //  cm**2/sec, what to use as default?	
	fVerticalBottomDiffusionCoefficient = .11; //  cm**2/sec, Bushy suggested a larger default	
	fHorizontalDiffusionCoefficient = 126; //  cm**2/sec	
	bUseDepthDependentDiffusion = false;
	SetClassName (name);
	//fUncertaintyFactor = 2;		// default uncertainty mult-factor
}


long TRandom3D::GetListLength()
{
	long count = 1;
	
	if (bOpen) {
		count += 2;
		if(model->IsUncertain())count++;
		count++;	// vertical diffusion coefficient
	}
	
	return count;
}

ListItem TRandom3D::GetNthListItem(long n, short indent, short *style, char *text)
{
	ListItem item = { dynamic_cast<TClassID *>(this), 0, indent, 0 };
	char valStr[32],valStr2[32];
	
	if (n == 0) {
		item.index = I_RANDOMNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Random3D: \"%s\"", className);
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}
	
	n -= 1;
	item.indent++;
	
	if (bOpen) {
		if (n == 0) {
			item.index = I_RANDOMACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
		
		n -= 1;
		
		if (n == 0) {
			item.index = I_RANDOMAREA;
			StringWithoutTrailingZeros(valStr,fDiffusionCoefficient,0);
			sprintf(text, "%s cm**2/sec (surface)", valStr);
			
			return item;
		}
		
		n -= 1;
		
		if(model->IsUncertain())
		{
			if (n == 0) {
				item.index = I_RANDOMUFACTOR;
				StringWithoutTrailingZeros(valStr,fUncertaintyFactor,0);
				sprintf(text, "Uncertainty factor: %s", valStr);
				
				return item;
			}
			
			n -= 1;
		}
		
		if (n == 0) {
			item.index = I_RANDOMVERTAREA;
			StringWithoutTrailingZeros(valStr,fVerticalDiffusionCoefficient,0);
			StringWithoutTrailingZeros(valStr2,fHorizontalDiffusionCoefficient,0);
			if (bUseDepthDependentDiffusion)
				sprintf(text, "vert = f(z), %s cm**2/sec (horiz)", valStr2);
			else
				sprintf(text, "%s cm**2/sec (vert), %s cm**2/sec (horiz)", valStr, valStr2);		
			
			return item;
		}
		
		n -= 1;
		
	}
	
	item.owner = 0;
	
	return item;
}

Boolean TRandom3D::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_RANDOMNAME: bOpen = !bOpen; return TRUE;
			case I_RANDOMACTIVE: bActive = !bActive; 
				model->NewDirtNotification(); return TRUE;
		}
	
	if (doubleClick)
		if(item.index==I_RANDOMAREA)
			TRandom::ListClick(item,inBullet,doubleClick);
		else
			Random3DSettingsDialog(dynamic_cast<TRandom3D *>(this), this -> moverMap);
	
	// do other click operations...
	
	return FALSE;
}

/*Boolean TRandom3D::FunctionEnabled(ListItem item, short buttonID)
 {
 long i;
 switch (item.index) {
 case I_RANDOMNAME:
 switch (buttonID) {
 case ADDBUTTON: return FALSE;
 case SETTINGSBUTTON: return TRUE;
 case DELETEBUTTON: return TRUE;
 case UPBUTTON:
 case DOWNBUTTON:
 if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
 switch (buttonID) {
 case UPBUTTON: return i > 0;
 case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
 }
 }
 break;
 default:
 switch (buttonID) {
 case ADDBUTTON: return FALSE;
 case SETTINGSBUTTON: return TRUE;
 case DELETEBUTTON: return FALSE;
 }
 break;
 }
 
 return TMover::FunctionEnabled(item, buttonID);
 }*/

OSErr TRandom3D::SettingsItem(ListItem item)
{
	switch (item.index) {
		default:
			return Random3DSettingsDialog(dynamic_cast<TRandom3D *>(this), this -> moverMap);
	}
	
	return 0;
}

OSErr TRandom3D::DeleteItem(ListItem item)
{
	if (item.index == I_RANDOMNAME)
		return moverMap -> DropMover (dynamic_cast<TRandom3D *>(this));
	
	return 0;
}

OSErr TRandom3D::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	/*char ourName[kMaxNameLen];
	 
	 // see if the message is of concern to us
	 this->GetClassName(ourName);
	 if(message->IsMessage(M_SETFIELD,ourName))
	 {
	 double val = 0;
	 OSErr  err;
	 err = message->GetParameterAsDouble("coverage",&val); // old style
	 if(err) err = message->GetParameterAsDouble("Coefficient",&val);
	 if(!err)
	 {	
	 if(val >= 0)// do we have any other  max or min limits ?
	 {
	 this->fDiffusionCoefficient = val;
	 model->NewDirtNotification();// tell model about dirt
	 }
	 }
	 ///
	 err = message->GetParameterAsDouble("Uncertaintyfactor",&val);
	 if(!err)
	 {	
	 if(val >= 1.0)// do we have any other  max or min limits ?
	 {
	 this->fUncertaintyFactor = val;
	 model->NewDirtNotification();// tell model about dirt
	 }
	 }
	 ///
	 
	 }*/
	/////////////////////////////////////////////////
	// we have no sub-guys that that need us to pass this message 
	/////////////////////////////////////////////////
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TRandom::CheckAndPassOnMessage(message);
}



#define TRandom3D_FileVersion 2
//#define TRandom3D_FileVersion 1
OSErr TRandom3D::Write(BFPB *bfpb)
{
	long version = TRandom3D_FileVersion;
	ClassID id = GetClassID ();
	OSErr err = 0;
	
	if (err = TMover::Write(bfpb)) return err;
	
	StartReadWriteSequence("TRandom3D::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	if (err = WriteMacValue(bfpb, fDiffusionCoefficient)) return err;
	if (err = WriteMacValue(bfpb, fUncertaintyFactor)) return err;
	
	if (err = WriteMacValue(bfpb, fVerticalDiffusionCoefficient)) return err;
	if (err = WriteMacValue(bfpb, fHorizontalDiffusionCoefficient)) return err;
	
	//if (err = WriteMacValue(bfpb, fVerticalBottomDiffusionCoefficient)) return err;
	if (err = WriteMacValue(bfpb, bUseDepthDependentDiffusion)) return err;
	
	return 0;
}

OSErr TRandom3D::Read(BFPB *bfpb) 
{
	long version;
	ClassID id;
	OSErr err = 0;
	
	if (err = TMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("TRandom3D::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TRandom3D::Read()", "id != GetClassID", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > TRandom3D_FileVersion) { printSaveFileVersionError(); return -1; }
	
	if (err = ReadMacValue(bfpb, &fDiffusionCoefficient)) return err;
	if (err = ReadMacValue(bfpb, &fUncertaintyFactor)) return err;
	
	if (gMearnsVersion || version > 1)
	{
		if (err = ReadMacValue(bfpb, &fVerticalDiffusionCoefficient)) return err;
		if (err = ReadMacValue(bfpb, &fHorizontalDiffusionCoefficient)) return err;
		//if (err = ReadMacValue(bfpb, &fVerticalDiffusionCoefficient)) return err;
	}
	if (version>1)
		if (err = ReadMacValue(bfpb, &bUseDepthDependentDiffusion)) return err;
	
	return 0;
}

static PopInfoRec RandomPopTable[] = {
	{ M28b, nil, M28bINPUTTYPE, 0, pRANDOMINPUTTYPE, 0, 1, FALSE, nil }
};

void ShowHideRandomDialogItems(DialogPtr dialog)
{
	Boolean showHorizItems, showVertItems, showCurrentWindItems;
	short typeOfInfoSpecified = GetPopSelection(dialog, M28bINPUTTYPE);
	
	Boolean depthDep  = GetButton (dialog, M28bDEPTHDEPENDENT); 
	
	if (depthDep)
	{
		ShowHideDialogItem(dialog, M28bINPUTTYPE, false);
		typeOfInfoSpecified = 2;
	}
	else
		ShowHideDialogItem(dialog, M28bINPUTTYPE, true); 
	
	
	switch (typeOfInfoSpecified)
	{
			//default:
			//case Input eddy diffusion values:
		case 1:
			showHorizItems=TRUE;
			showVertItems=TRUE;
			showCurrentWindItems=FALSE;
			break;
			//case Input current and wind speed:
		case 2:
			showCurrentWindItems=FALSE;
			showHorizItems=TRUE;
			showVertItems=FALSE;
			break;
		case 3:
			showCurrentWindItems=TRUE;
			showHorizItems=FALSE;
			showVertItems=FALSE;
			break;
	}
	ShowHideDialogItem(dialog, M28bDIFFUSION, showHorizItems ); 
	ShowHideDialogItem(dialog, M28bUFACTOR, showHorizItems); 
	ShowHideDialogItem(dialog, M28bFROST1, showHorizItems); 
	ShowHideDialogItem(dialog, M28bDIFFUSIONLABEL1, showHorizItems); 
	ShowHideDialogItem(dialog, M28bDIFFUSIONUNITS1, showHorizItems); 
	ShowHideDialogItem(dialog, M28bUNCERTAINTYLABEL1, showHorizItems); 
	ShowHideDialogItem(dialog, M28bHORIZONTALLABEL, showHorizItems); 
	
	ShowHideDialogItem(dialog, M28bVERTDIFFUSION, showVertItems); 
	ShowHideDialogItem(dialog, M28bVERTUFACTOR, showVertItems); 
	ShowHideDialogItem(dialog, M28bFROST2, showVertItems); 
	ShowHideDialogItem(dialog, M28bDIFFUSIONLABEL2, showVertItems); 
	ShowHideDialogItem(dialog, M28bDIFFUSIONUNITS2, showVertItems); 
	ShowHideDialogItem(dialog, M28bUNCERTAINTYLABEL2, showVertItems); 
	ShowHideDialogItem(dialog, M28bVERTICALLABEL, showVertItems); 
	
	ShowHideDialogItem(dialog, M28bWINDSPEEDLABEL, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bWINDSPEED, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bCURRENTSPEEDLABEL, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bCURRENTSPEED, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bCURRENTSPEEDUNITS, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bWINDSPEEDUNITS, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bFROST3, showCurrentWindItems); 
	
	//ShowHideDialogItem(dialog, M28bBOTKZLABEL, depthDep); 
	//ShowHideDialogItem(dialog, M28bBOTKZ, depthDep); 
	//ShowHideDialogItem(dialog, M28bBOTKZUNITS, depthDep); 
}

OSErr M28bInit (DialogPtr dialog, VOIDPTR data)
// new random diffusion dialog init
{
	SetDialogItemHandle(dialog, M28bHILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M28bFROST1, (Handle)FrameEmbossed);
	SetDialogItemHandle(dialog, M28bFROST2, (Handle)FrameEmbossed);
	SetDialogItemHandle(dialog, M28bFROST3, (Handle)FrameEmbossed);
	
	RegisterPopTable (RandomPopTable, sizeof (RandomPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (M28b, dialog);
	
	SetPopSelection (dialog, M28bINPUTTYPE, 1);
	
	mysetitext(dialog, M28bNAME, sharedRMover3D -> className);
	//MySelectDialogItemText (dialog, M28bNAME, 0, 100);
	
	SetButton(dialog, M28bACTIVE, sharedRMover3D -> bActive);
	SetButton(dialog, M28bDEPTHDEPENDENT, sharedRMover3D -> bUseDepthDependentDiffusion);
	
	//Float2EditText(dialog, M28bDIFFUSION, sharedRMover3D->fDiffusionCoefficient, 0);
	Float2EditText(dialog, M28bDIFFUSION, sharedRMover3D->fHorizontalDiffusionCoefficient, 0);
	
	Float2EditText(dialog, M28bUFACTOR, sharedRMover3D->fUncertaintyFactor, 0);
	
	Float2EditText(dialog, M28bVERTDIFFUSION, sharedRMover3D->fVerticalDiffusionCoefficient, 0);
	
	Float2EditText(dialog, M28bBOTKZ, sharedRMover3D->fVerticalBottomDiffusionCoefficient, 0);
	
	//Float2EditText(dialog, M28bUFACTOR, sharedRMover3D->fUncertaintyFactor, 0);
	Float2EditText(dialog, M28bVERTUFACTOR, sharedRMover3D->fUncertaintyFactor, 0);
	
	ShowHideRandomDialogItems(dialog);
	MySelectDialogItemText(dialog, M28bDIFFUSION,0,255);
	
	return 0;
}

short M28bClick (DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
// old random diffusion dialog
{
	char	name [kMaxNameLen];
	double uncertaintyFactor,vertStep; 
	double U=0,W=0,botKZ=0;
	short typeOfInput;
	long menuID_menuItem;
	
	switch (itemNum) {
		case M28bOK:
		{
			PtCurMap *map = GetPtCurMap();
			//PtCurMap *map = (PtCurMap*) (sharedRMover3D -> GetMoverMap());
			// uncertaintyFactor enforce >= 1.0
			uncertaintyFactor = EditText2Float(dialog, M28bUFACTOR);
			if(uncertaintyFactor <1.0)
			{
				printError("The uncertainty factor must be >= 1.0");
				MySelectDialogItemText (dialog, M28bUFACTOR, 0, 255);
				break;
			}
			mygetitext(dialog, M28bNAME, name, kMaxNameLen - 1);		// get the mover's nameStr
			sharedRMover3D -> SetClassName (name);
			sharedRMover3D -> SetActive (GetButton(dialog, M28bACTIVE));
			sharedRMover3D -> bUseDepthDependentDiffusion = GetButton(dialog, M28bDEPTHDEPENDENT);
			
			typeOfInput = GetPopSelection(dialog, M28bINPUTTYPE);
			if (sharedRMover3D -> bUseDepthDependentDiffusion) 
			{
				typeOfInput = 2;	// why??
				/*botKZ = EditText2Float(dialog,M28bBOTKZ);
				 if (botKZ == 0)
				 {
				 printError("You must enter a value for the vertical diffusion coefficient on the bottom");
				 MySelectDialogItemText(dialog, M28bBOTKZ,0,255);
				 break;
				 }*/
				//sharedRMover3D -> fVerticalBottomDiffusionCoefficient = EditText2Float(dialog, M28bBOTKZ);
			}
			sharedRMover3D -> fVerticalBottomDiffusionCoefficient = EditText2Float(dialog, M28bBOTKZ);
			
			if (typeOfInput==1)
			{
				//sharedRMover3D -> fDiffusionCoefficient = EditText2Float(dialog, M28DIFFUSION);
				sharedRMover3D -> fHorizontalDiffusionCoefficient = EditText2Float(dialog, M28bDIFFUSION);
				sharedRMover3D -> fUncertaintyFactor = uncertaintyFactor;
				sharedRMover3D -> fVerticalDiffusionCoefficient = EditText2Float(dialog, M28bVERTDIFFUSION);
				//sharedRMover3D -> fVerticalUncertaintyFactor = uncertaintyFactor;
			}
			else if (typeOfInput==2)
			{
				//sharedRMover3D -> fDiffusionCoefficient = EditText2Float(dialog, M28DIFFUSION);
				sharedRMover3D -> fHorizontalDiffusionCoefficient = EditText2Float(dialog, M28bDIFFUSION);
				sharedRMover3D -> fUncertaintyFactor = uncertaintyFactor;
				sharedRMover3D -> fVerticalDiffusionCoefficient = sharedRMover3D -> fHorizontalDiffusionCoefficient/6.88;
				//sharedRMover3D -> fVerticalUncertaintyFactor = uncertaintyFactor;
			}
			else if (typeOfInput==3)
			{
				U = EditText2Float(dialog,M28bCURRENTSPEED);
				if (U == 0)
				{
					printError("You must enter a value for the current velocity");
					MySelectDialogItemText(dialog, M28bCURRENTSPEED,0,255);
					break;
				}
				W = EditText2Float(dialog,M28bWINDSPEED);
				if (W == 0)
				{
					printError("You must enter a value for the wind velocity");
					MySelectDialogItemText(dialog, M28bWINDSPEED,0,255);
					break;
				}
				sharedRMover3D -> fHorizontalDiffusionCoefficient = (272.8*U + 21.1*W); //cm^2/s - Note the conversion from m^2/s is done by leaving out a 10^-4 factor
				sharedRMover3D -> fVerticalDiffusionCoefficient = (39.7*U + 3.1*W);	//cm^2/s
			}
			vertStep = sqrt(6*(sharedRMover3D -> fVerticalDiffusionCoefficient/10000)*model->GetTimeStep()); // in meters
			// compare to mixed layer depth and warn if within a certain percentage - 
			if (map && vertStep > map->fMixedLayerDepth)
				printNote("The combination of large vertical diffusion coefficient and choice of timestep will likely result in particles moving vertically on the order of the size of the mixed layer depth. They will be randomly placed in the mixed layer if reflection fails.");
			
			
			return M28bOK;
		}
			
		case M28bCANCEL: return M28bCANCEL;
			
		case M28bACTIVE:
			ToggleButton(dialog, M28bACTIVE);
			break;
			
		case M28bDEPTHDEPENDENT:
			ToggleButton(dialog, M28bDEPTHDEPENDENT);
			sharedRMover3D -> bUseDepthDependentDiffusion = GetButton(dialog,M28bDEPTHDEPENDENT);
			ShowHideRandomDialogItems(dialog);
			break;
			
		case M28bDIFFUSION:
			CheckNumberTextItem(dialog, itemNum, false); //  don't allow decimals
			break;
			
		case M28bWINDSPEED:
		case M28bCURRENTSPEED:
		case M28bBOTKZ:
			CheckNumberTextItem(dialog, itemNum, true); //   allow decimals
			break;
			
		case M28bUFACTOR:
			CheckNumberTextItem(dialog, itemNum, true); //   allow decimals
			break;
			
		case M28bVERTDIFFUSION:
			CheckNumberTextItem(dialog, itemNum, true); //  allow decimals
			break;
			
		case M28bVERTUFACTOR:
			CheckNumberTextItem(dialog, itemNum, true); //   allow decimals
			break;
			
		case M28bINPUTTYPE:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowHideRandomDialogItems(dialog);
			if (GetPopSelection(dialog, M28bINPUTTYPE)==3)
				MySelectDialogItemText(dialog, M28bCURRENTSPEED,0,255);
			else
				MySelectDialogItemText(dialog, M28bDIFFUSION,0,255);
			break;
			
	}
	
	return 0;
}

OSErr Random3DSettingsDialog(TRandom3D *mover, TMap *owner)
{
	short item;
	TRandom3D *newMover = 0;
	OSErr err = 0;
	
	if (!mover) {
		newMover = new TRandom3D(owner, "3D Diffusion");
		if (!newMover)
		{ TechError("RandomSettingsDialog()", "new TRandom3D()", 0); return -1; }
		
		if (err = newMover->InitMover()) { delete newMover; return err; }
		
		sharedRMover3D = newMover;
	}
	else
		sharedRMover3D = mover;
	
	item = MyModalDialog(M28b, mapWindow, 0, M28bInit, M28bClick);
	
	if (item == M28bOK) model->NewDirtNotification();
	
	if (newMover) {
		if (item == M28bOK) {
			if (err = owner->AddMover(newMover, 0))
			{ newMover->Dispose(); delete newMover; return -1; }
		}
		else {
			newMover->Dispose();
			delete newMover;
		}
	}
	
	return 0;
}





