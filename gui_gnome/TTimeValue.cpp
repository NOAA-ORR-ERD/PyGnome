/*
 *  TTimeValue.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 1/3/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "TTimeValue.h"

#include "CROSS.H"

OSErr TTimeValue::InitTimeFunc()
{
	return 0;
}

OSErr TTimeValue::Write(BFPB *bfpb)
{
	long version = 1;
	ClassID id = GetClassID ();	// base class id, version and header
	OSErr err = 0;
	
	StartReadWriteSequence("TTimeValue::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	return 0;
}

OSErr TTimeValue::Read(BFPB *bfpb)
{
	long version;
	ClassID id;
	OSErr err = 0;
	
	StartReadWriteSequence("TTimeValue::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TTimeValue::Read()", "id != TYPE_TIMEVALUES", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version != 1) { printSaveFileVersionError(); return -1; }
	
	return 0;
}

OSErr TTimeValue::MakeClone(TTimeValue **clonePtrPtr)
{
	// clone should be the address of a  ClassID ptr
	ClassID *clone;
	OSErr err = 0;
	if(!clonePtrPtr) return -1; 
	if(*clonePtrPtr == nil) return -1; // this class should not create a clone
	if(*clonePtrPtr)
	{	// copy the fields
		if((*clonePtrPtr)->GetClassID() == this->GetClassID()) // safety check
		{
			TTimeValue * cloneP = dynamic_cast<TTimeValue *>(*clonePtrPtr);// typecast 
			TClassID *tObj = dynamic_cast<TClassID *>(*clonePtrPtr);
			err =  TClassID::MakeClone(&tObj);//  pass clone to base class
			if(!err) 
			{
				cloneP->owner=this->owner;
			}
		}
	}
	return err;
}


OSErr TTimeValue::BecomeClone(TTimeValue *clone)
{
	OSErr err = 0;
	
	if(clone)
	{
		if(clone->GetClassID() == this->GetClassID()) // safety check
		{
			TTimeValue * cloneP = dynamic_cast<TTimeValue *>(clone);// typecast
			
			/*OK*/ dynamic_cast<TTimeValue *>(this)->TTimeValue::Dispose(); // get rid of any memory we currently are using
			
			err =  TClassID::BecomeClone(clone);//  pass clone to base class
			if(err) goto done;
			
			this->owner=cloneP->owner;
			
		}
	}
done:
	if(err) /*OK*/ dynamic_cast<TTimeValue *>(this)->TTimeValue::Dispose(); // don't leave ourselves in a weird state
	return err;
}