/*
 *  Mover_g.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Mover_g.h"
#include "CROSS.H"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT MOVER_G
#endif
#endif

OSErr Mover_g::UpItem(ListItem item)
{
	long i;
	OSErr err = 0;
	
	if (item.index == I_MAPNAME)
		if (moverMap -> moverList -> IsItemInList((Ptr)&item.owner, &i))
			if (i > 0) {
				if (err = moverMap -> moverList -> SwapItems(i, i - 1))
				{ TechError("TMover::UpItem()", "moverMap -> moverList -> SwapItems()", err); return err; }
				SelectListItem(item);
				InvalListLength();// why ? JLM
			}
	
	return 0;
}

OSErr Mover_g::DownItem(ListItem item)
{
	long i;
	OSErr err = 0;
	
	if (item.index == I_MAPNAME)
		if (moverMap -> moverList -> IsItemInList((Ptr)&item.owner, &i))
			if (i < (moverMap -> moverList -> GetItemCount() - 1)) {
				if (err = moverMap -> moverList -> SwapItems(i, i + 1))
				{ TechError("TMover::UpItem()", "moverMap -> moverList -> SwapItems()", err); return err; }
				SelectListItem(item);
				InvalListLength();// why JLM
			}
	
	return 0;
}

OSErr Mover_g::MakeClone(TClassID **clonePtrPtr)
{
	// clone should be the address of a  ClassID ptr
	//ClassID *clone;
	OSErr err = 0;
	if(!clonePtrPtr) return -1; // we are supposed to fill
	if(*clonePtrPtr == nil) return -1; //we don't create objects of this type
	
	if(*clonePtrPtr)
	{	// copy the fields
		if((*clonePtrPtr)->GetClassID() == this->GetClassID()) // safety check
		{
			TMover * cloneP = dynamic_cast<TMover *>(*clonePtrPtr);// typecast 
			err =  ClassID_g::MakeClone(clonePtrPtr);//  pass clone to base class
			if(!err) 
			{
				cloneP->moverMap = this->moverMap;
				cloneP->fUncertainStartTime = this->fUncertainStartTime;
				cloneP->fDuration = this->fDuration;
				cloneP->fTimeUncertaintyWasSet = this->fTimeUncertaintyWasSet;
			}
		}
	}
	//done:
	if(err && *clonePtrPtr) 
	{
		(*clonePtrPtr)->Dispose();
	}
	return err;
}


OSErr Mover_g::BecomeClone(TClassID *clone)
{
	OSErr err = 0;
	
	if(clone)
	{
		if(clone->GetClassID() == this->GetClassID()) // safety check
		{
			TMover * cloneP = dynamic_cast<TMover *>(clone);// typecast
			
			dynamic_cast<TMover *>(this)->Dispose(); // get rid of any memory we currently are using
			////////////////////
			// do the memory stuff first, in case it fails
			////////
			
			err =  ClassID_g::BecomeClone(clone);//  pass clone to base class
			if(err) goto done;
			
			this->moverMap = cloneP->moverMap;
			this->fUncertainStartTime = cloneP->fUncertainStartTime;
			this->fDuration = cloneP->fDuration;
			this->fTimeUncertaintyWasSet = cloneP->fTimeUncertaintyWasSet;
		}
	}
done:
	if(err) dynamic_cast<TMover *>(this)->Dispose(); // don't leave ourselves in a weird state
	return err;
}

#define TMoverREADWRITEVERSION 2

OSErr Mover_g::Write(BFPB *bfpb)
{
	long version = TMoverREADWRITEVERSION;
	ClassID id = GetClassID ();	// base class id, version and header
	OSErr err = 0;
	char colorStr[64];
	
	StartReadWriteSequence("TMover::Write()");
	if (err = WriteMacValue(bfpb,id)) return err;
	if (err = WriteMacValue(bfpb,version)) return err;
	if (err = WriteMacValue(bfpb, className, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb,bDirty)) return err;
	if (err = WriteMacValue(bfpb,bActive)) return err;
	if (err = WriteMacValue(bfpb,bOpen)) return err;
	
	if (err = WriteMacValue(bfpb,fUncertainStartTime)) return err;
	if (err = WriteMacValue(bfpb,fDuration)) return err;
	
	RGBColorToString(fColor, colorStr);
	if (err = WriteMacValue(bfpb,colorStr,64)) return err;
	
	SetDirty(false);
	
	return 0;
}

OSErr Mover_g::Read(BFPB *bfpb)
{
	long version;
	ClassID id;
	OSErr err = 0;
	char colorStr[64];
	
	StartReadWriteSequence("TMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TMover::Read()", "id != TYPE_MOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	//if (version != 1) { printSaveFileVersionError(); return -1; }
	if (version > TMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	if (err = ReadMacValue(bfpb, className, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb,&bDirty)) return err;
	if (err = ReadMacValue(bfpb,&bActive)) return err;
	if (err = ReadMacValue(bfpb,&bOpen)) return err;
	
	if (err = ReadMacValue(bfpb,&fUncertainStartTime)) return err;
	if (err = ReadMacValue(bfpb,&fDuration)) return err;
	
	if (version>1)
	{
		if (err = ReadMacValue(bfpb,colorStr,64)) return err;
		fColor = StringToRGBColor(colorStr);
	}
	
	return 0;
}

///////////////////////////////////////////////////////////////////////////

OSErr Mover_g::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	char ourName[kMaxNameLen];
	
	// see if the message is of concern to us
	this->GetClassName(ourName);
	
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		double val;
		OSErr err = 0;
		////////////////
		err = message->GetParameterAsDouble("UncertaintyDuration",&val);
		if(!err && val >= 0) this->fDuration = val*3600; 
		////////////////
		err = message->GetParameterAsDouble("UncertaintyStartTime",&val);
		if(!err && val >= 0) this->fUncertainStartTime = val*3600; 
		////////////////
		model->NewDirtNotification();// tell model about dirt
	}
	
	/////////////////////////////////////////////////
	// we have no sub-guys that that need us to pass this message 
	/////////////////////////////////////////////////
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return ClassID_g::CheckAndPassOnMessage(message);
}

///////////////////////////////////////////////////////////////////////////

static PopInfoRec moverTypesPopTable[] = {
	{ M21, nil, M21TYPESITEM, 0, pMOVERTYPES, 0, 1, FALSE, nil }
};

OSErr M21Init(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	RegisterPopTable(moverTypesPopTable, sizeof(moverTypesPopTable) / sizeof(PopInfoRec));
	RegisterPopUpDialog(M21, dialog);
	SetPopSelection (dialog, M21TYPESITEM, 1); // default currents mover
	// code goes here, might want to get rid of create default option for currents - need 2 buttons show/hide on PC ?
	//if (!gNoaaVersion) MyEnableControl(dialog, M21CREATE, false);	 // disable Create button for currents
	//else ShowHideDialogItem(dialog,M21HILITEDEFAULT,FALSE); 
	MyEnableControl(dialog, M21CREATE, false);	 // disable Create button for currents - this is not working for carbon
	//MyEnableControl(dialog, M21LOAD,   true);
	//could do a showhide for Carbon, but need to reshow when switch mover type
	//ShowHideDialogItem(dialog,M21HILITEDEFAULT,FALSE);
	//ShowHideDialogItem(dialog,M21CREATE,FALSE);
	
	return 0;
}

short M21Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short	theType, *moverType = (short*) data;
	long	menuID_menuItem;
	
	switch (itemNum) {
		case M21CANCEL: return M21CANCEL;
			
		case M21CREATE:
		case M21LOAD:
			*moverType = GetPopSelection (dialog, M21TYPESITEM);
			return itemNum;
			
		case M21TYPESITEM:
		{
			PopClick(dialog, itemNum, &menuID_menuItem);
			theType = GetPopSelection (dialog, M21TYPESITEM);
			switch (theType)
			{
				case CURRENTS_MOVERTYPE:
					//if (!gNoaaVersion) MyEnableControl(dialog, M21CREATE, false);
					//else MyEnableControl(dialog, M21CREATE, true);
					MyEnableControl(dialog, M21CREATE, false);
					MyEnableControl(dialog, M21LOAD,   true);
					break;			
				case WIND_MOVERTYPE: 
					MyEnableControl(dialog, M21CREATE, true);
					MyEnableControl(dialog, M21LOAD,   true);
					break;
				case RANDOM_MOVERTYPE: 
					MyEnableControl(dialog, M21CREATE, true);
					MyEnableControl(dialog, M21LOAD,   false);
					break;
				case CONSTANT_MOVERTYPE: 
					MyEnableControl(dialog, M21CREATE, true);
					MyEnableControl(dialog, M21LOAD,   false);
					break;
				case COMPONENT_MOVERTYPE: 
					MyEnableControl(dialog, M21CREATE, true);
					MyEnableControl(dialog, M21LOAD,   false);
					break;
				case COMPOUND_MOVERTYPE: 
					MyEnableControl(dialog, M21CREATE, true);
					MyEnableControl(dialog, M21LOAD,   false);
					break;
			}
		}
			break;
	}
	
	return 0;
}

Boolean Mover_g::FunctionEnabled(ListItem item, short buttonID)
{
	//long i;
	
	//	if (item.index == 0) {
	//		if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
	//		switch (buttonID) {
	//			case UPBUTTON: return i > 0;
	//			case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
	//		}
	//	}
	
	return FALSE;
}


