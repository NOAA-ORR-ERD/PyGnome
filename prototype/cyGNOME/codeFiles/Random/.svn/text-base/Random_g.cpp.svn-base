/*
 *  Random_g.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Random_g.h"
#include "CROSS.H"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT RANDOM_G
#endif
#endif

TRandom *sharedRMover;

long Random_g::GetListLength()
{
	long count = 1;
	
	if (bOpen) {
		count += 2;
		if(model->IsUncertain())count++;
	}
	
	return count;
}

ListItem Random_g::GetNthListItem(long n, short indent, short *style, char *text)
{
	ListItem item = { dynamic_cast<TRandom *>(this), 0, indent, 0 };
	char valStr[32];
	
	if (n == 0) {
		item.index = I_RANDOMNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Random: \"%s\"", className);
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
			sprintf(text, "%s cm**2/sec", valStr);
			
			return item;
		}
		
		n -= 1;
		
		if(model->IsUncertain())
		{
			if (n == 0) {
				item.index = I_RANDOMUFACTOR;
				StringWithoutTrailingZeros(valStr, fUncertaintyFactor,0);
				sprintf(text, "Uncertainty factor: %s", valStr);
				
				return item;
			}
			
			n -= 1;
		}
		
	}
	
	item.owner = 0;
	
	return item;
}

Boolean Random_g::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_RANDOMNAME: bOpen = !bOpen; return TRUE;
			case I_RANDOMACTIVE: bActive = !bActive; 
				model->NewDirtNotification(); return TRUE;
		}
	
	if (doubleClick)
		RandomSettingsDialog(dynamic_cast<TRandom *>(this), this -> moverMap);
	
	// do other click operations...
	
	return FALSE;
}

Boolean Random_g::FunctionEnabled(ListItem item, short buttonID)
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
	
	return Mover_g::FunctionEnabled(item, buttonID);
}

OSErr Random_g::SettingsItem(ListItem item)
{
	switch (item.index) {
		default:
			return RandomSettingsDialog(dynamic_cast<TRandom *>(this), this -> moverMap);
	}
	
	return 0;
}

OSErr Random_g::DeleteItem(ListItem item)
{
	if (item.index == I_RANDOMNAME)
		return moverMap -> DropMover (dynamic_cast<TRandom *>(this));
	
	return 0;
}

OSErr Random_g::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	char ourName[kMaxNameLen];
	Boolean useDepthDependent;
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
			if(val >= 1.0)// do we have any other max or min limits ?
			{
				this->fUncertaintyFactor = val;
				model->NewDirtNotification();// tell model about dirt
			}
		}
		err = message->GetParameterAsBoolean("DepthDependent",&useDepthDependent);
		if(!err)
		{	
			this->bUseDepthDependent = useDepthDependent;
			//model->NewDirtNotification();// tell model about dirt
		}
		///
		
	}
	/////////////////////////////////////////////////
	// we have no sub-guys that that need us to pass this message 
	/////////////////////////////////////////////////
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return Mover_g::CheckAndPassOnMessage(message);
}


//#define TRandom_FileVersion 1
#define TRandom_FileVersion 2
OSErr Random_g::Write(BFPB *bfpb)
{
	long version = TRandom_FileVersion;
	ClassID id = GetClassID ();
	OSErr err = 0;
	
	if (err = Mover_g::Write(bfpb)) return err;
	
	StartReadWriteSequence("TRandom::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	if (err = WriteMacValue(bfpb, fDiffusionCoefficient)) return err;
	if (err = WriteMacValue(bfpb, fUncertaintyFactor)) return err;
	
	if (err = WriteMacValue(bfpb, bUseDepthDependent)) return err;
	
	return 0;
}

OSErr Random_g::Read(BFPB *bfpb) 
{
	long version;
	ClassID id;
	OSErr err = 0;
	
	if (err = Mover_g::Read(bfpb)) return err;
	
	StartReadWriteSequence("TRandom::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TRandom::Read()", "id != GetClassID", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > TRandom_FileVersion || version < 1) { printSaveFileVersionError(); return -1; }
	
	if (err = ReadMacValue(bfpb, &fDiffusionCoefficient)) return err;
	if (err = ReadMacValue(bfpb, &fUncertaintyFactor)) return err;
	if (version>1)
		if (err = ReadMacValue(bfpb, &bUseDepthDependent)) return err;
	
	return 0;
}

OSErr M28Init (DialogPtr dialog, VOIDPTR data)
// new random diffusion dialog init
{
	SetDialogItemHandle(dialog, M28HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M28FROST1, (Handle)FrameEmbossed);
	
	mysetitext(dialog, M28NAME, sharedRMover -> className);
	MySelectDialogItemText (dialog, M28NAME, 0, 100);
	
	SetButton(dialog, M28ACTIVE, sharedRMover -> bActive);
	
	Float2EditText(dialog, M28DIFFUSION, sharedRMover->fDiffusionCoefficient, 0);
	
	Float2EditText(dialog, M28UFACTOR, sharedRMover->fUncertaintyFactor, 0);
	
	return 0;
}

short M28Click (DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
// old random diffusion dialog
{
	char	name [kMaxNameLen];
	double uncertaintyFactor; 
	
	switch (itemNum) {
		case M28OK:
			
			// uncertaintyFactor enforce >= 1.0
			uncertaintyFactor = EditText2Float(dialog, M28UFACTOR);
			if(uncertaintyFactor <1.0)
			{
				printError("The uncertainty factor must be >= 1.0");
				MySelectDialogItemText (dialog, M28UFACTOR, 0, 100);
				break;
			}
			mygetitext(dialog, M28NAME, name, kMaxNameLen - 1);		// get the mover's nameStr
			sharedRMover -> SetClassName (name);
			sharedRMover -> SetActive (GetButton(dialog, M28ACTIVE));
			sharedRMover -> fDiffusionCoefficient = EditText2Float(dialog, M28DIFFUSION);
			sharedRMover -> fUncertaintyFactor = uncertaintyFactor;
			
			return M28OK;
			
		case M28CANCEL: return M28CANCEL;
			
		case M28ACTIVE:
			ToggleButton(dialog, M28ACTIVE);
			break;
			
		case M28DIFFUSION:
			CheckNumberTextItem(dialog, itemNum, false); //  don't allow decimals
			break;
			
		case M28UFACTOR:
			CheckNumberTextItem(dialog, itemNum, true); //   allow decimals
			break;
			
	}
	
	return 0;
}

OSErr RandomSettingsDialog(TRandom *mover, TMap *owner)
{
	short item;
	TRandom *newMover = 0;
	OSErr err = 0;
	
	if (!mover) {
		newMover = new TRandom(owner, "Diffusion");
		if (!newMover)
		{ TechError("RandomSettingsDialog()", "new TRandom()", 0); return -1; }
		
		if (err = newMover->InitMover()) { delete newMover; return err; }
		
		sharedRMover = newMover;
	}
	else
		sharedRMover = mover;
	
	item = MyModalDialog(M28, mapWindow, 0, M28Init, M28Click);
	
	if (item == M28OK) model->NewDirtNotification();
	
	if (newMover) {
		if (item == M28OK) {
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

