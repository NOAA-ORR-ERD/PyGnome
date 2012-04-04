/*
 *  ClassID_g.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "ClassID_g.h"
#include "CROSS.H"

long gObjectCounter = 1;

ClassID_g::ClassID_g()
{
	unsigned long ticks = MyTickCount();
	SetClassName ("Untitled");
	bOpen = true;
	bActive = true;
	bDirty = false;
	
	fUniqueID.ticksAtCreation = ticks;
	fUniqueID.counter  = gObjectCounter++;
	
	return;
}

Boolean ClassID_g::GetSelectedListItem(ListItem *item)
{	// returns TRUE if the selected item belongs to this object
	ListItem localItem;
	Boolean itemSelected = SelectedListItem(&localItem);
	
	if (!itemSelected) 
		return FALSE;
	
	if (localItem.owner == this) {
		*item = localItem;
		return TRUE;
	}
	
	return FALSE;
}

Boolean ClassID_g::SelectedListItemIsMine(void)
{	// returns TRUE if the selected item belongs to this object
	ListItem localItem;
	Boolean itemSelected = SelectedListItem(&localItem);
	
	if (!itemSelected) 
		return FALSE;
	
	return (localItem.owner == this);
}

Boolean ClassID_g::IAmEditableInMapDrawingRect(void)
{	
	return FALSE;
}

Boolean ClassID_g::IAmCurrentlyEditableInMapDrawingRect(void)
{	
	return FALSE;
}

Boolean ClassID_g::UserIsEditingMeInMapDrawingRect(void)
{	
	return FALSE;
}

void ClassID_g::StartEditingInMapDrawingRect(void)
{	
	// ignore them
}

OSErr ClassID_g::StopEditingInMapDrawingRect(Boolean *deleteMe)
{	
	*deleteMe = FALSE;
	return 0;
}

Boolean ClassID_g::MatchesUniqueID(UNIQUEID uid)
{
	return EqualUniqueIDs(uid,this->fUniqueID);
}

OSErr ClassID_g::MakeClone(TClassID **clonePtrPtr)
{
	// clone should be the address of a  ClassID ptr
	//ClassID *clone;
	OSErr err = 0;
	if(!clonePtrPtr) return -1; // we are supposed to fill
	if(*clonePtrPtr == nil)
	{	// in other classes this would be an indication we were supposed to
		// create and return an object.
		
		// create 
		// *clonePtrPtr = new 
		//if(!*clonePtrPtr)/	
		// MemError
		
		// BUT, this base class does not create objects because
		// there are no real TClassID objects
		
	}
	if(*clonePtrPtr)
	{	// copy the fields
		if((*clonePtrPtr)->GetClassID() == this->GetClassID()) // safety check
		{
			(*clonePtrPtr)->bDirty = this->bDirty;
			(*clonePtrPtr)->bOpen = this->bOpen;
			(*clonePtrPtr)->bActive = this->bActive;
			strcpy((*clonePtrPtr)->className ,this->className);
			(*clonePtrPtr)->fUniqueID = this->fUniqueID;
		}
	}
	return err;
}


OSErr ClassID_g::BecomeClone(TClassID *clone)
{
	OSErr err = 0;
	
	if(clone)
	{
		if(clone->GetClassID() == this->GetClassID()) // safety check
		{
			this->bDirty = clone->bDirty;
			this->bOpen = clone->bOpen;
			this->bActive = clone->bActive;
			strcpy(this->className ,clone->className);
			this->fUniqueID = clone->fUniqueID;
		}
	}
	// no base class to pass clone to
	return err;
}



#define TClassID_FileVersion 1
OSErr ClassID_g::Read(BFPB *bfpb)
{
	long 	version;
	ClassID id;
	OSErr	err = noErr;
	
	StartReadWriteSequence("TClassID::::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TClassID::Read()", "id != GetClassID", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version != TClassID_FileVersion) { printSaveFileVersionError(); return -1; }
	
	bDirty = false; // we are reading
	if (err = ReadMacValue(bfpb, &bOpen)) return err;
	if (err = ReadMacValue(bfpb, &bActive)) return err;
	if (err = ReadMacValue(bfpb, className, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb, &fUniqueID.ticksAtCreation)) return err;
	if (err = ReadMacValue(bfpb, &fUniqueID.counter)) return err;
	
	return err;
}

OSErr ClassID_g::Write(BFPB *bfpb)
{
	long 	version = TClassID_FileVersion;
	ClassID id = GetClassID ();
	OSErr	err = noErr;
	
	StartReadWriteSequence("TClassID::::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	//bDirty -- no need to write this to the file
	bDirty = false; // we are writing
	if (err = WriteMacValue(bfpb, bOpen)) return err;
	if (err = WriteMacValue(bfpb, bActive)) return err;
	if (err = WriteMacValue(bfpb, className, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fUniqueID.ticksAtCreation)) return err;
	if (err = WriteMacValue(bfpb, fUniqueID.counter)) return err;
	
	return err;
}


void ClassID_g::SetClassName (char *newName)
{
	if (strlen (newName) > kMaxNameLen)
		newName [kMaxNameLen - 1] = 0;
	
	strnzcpy (className, newName, kMaxNameLen - 1);
}


OSErr ClassID_g::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	char ourName[kMaxNameLen];
	
	// see if the message is of concern to us
	this->GetClassName(ourName);
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		Boolean val;
		OSErr err = 0;
		
		err = message->GetParameterAsBoolean("bActive",&val);
		if(!err)
		{	
			this->bActive = val; 
			model->NewDirtNotification();// tell model about dirt
		}
	}
	/////////////////////////////////////////////////
	// we have no sub-guys that that need us to pass this message 
	/////////////////////////////////////////////////
	
	/////////////////////////////////////////////////
	//  no further base class to pass this onto
	/////////////////////////////////////////////////
	return noErr;
}

long ClassID_g::GetListLength()
{
	long count = 1;
	
	if (bOpen) {
		count += 1;
	}
	
	return count;
}

ListItem ClassID_g::GetNthListItem(long n, short indent, short *style, char *text)
{
	ListItem item = { dynamic_cast<TClassID *>(this), 0, indent, 0 };
	
	if (n == 0) {
		item.index = I_CLASSNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Class: \"%s\"", className);
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}
	
	n -= 1;
	item.indent++;
	
	if (bOpen) {
		if (n == 0) {
			item.index = I_CLASSACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
	}
	
	item.owner = 0;
	
	return item;
}

Boolean ClassID_g::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	//	if (item.index % 10 == 0)
	//		bOpen = !bOpen; return TRUE;
	
	if (inBullet)
		switch (item.index) {
			case I_CLASSNAME: bOpen = !bOpen; return TRUE;
			case I_CLASSACTIVE: bActive = !bActive; return TRUE;
		}
	
	return FALSE;
}

