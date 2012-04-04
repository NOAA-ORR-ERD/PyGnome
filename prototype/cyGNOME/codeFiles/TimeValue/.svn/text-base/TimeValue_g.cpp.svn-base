/*
 *  TimeValue_g.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "TimeValue_g.h"
#include "CROSS.H"

OSErr TimeValue_g::InitTimeFunc()
{
	return 0;
}

OSErr TimeValue_g::Write(BFPB *bfpb)
{
	long version = 1;
	ClassID id = GetClassID ();	// base class id, version and header
	OSErr err = 0;
	
	StartReadWriteSequence("TTimeValue::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	return 0;
}

OSErr TimeValue_g::Read(BFPB *bfpb)
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

OSErr TimeValue_g::MakeClone(TClassID **clonePtrPtr)
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
			err =  ClassID_g::MakeClone(clonePtrPtr);//  pass clone to base class
			if(!err) 
			{
				cloneP->owner=this->owner;
			}
		}
	}
	return err;
}


OSErr TimeValue_g::BecomeClone(TClassID *clone)
{
	OSErr err = 0;
	
	if(clone)
	{
		if(clone->GetClassID() == this->GetClassID()) // safety check
		{
			TTimeValue * cloneP = dynamic_cast<TTimeValue *>(clone);// typecast
			
			dynamic_cast<TTimeValue *>(this)->Dispose(); // get rid of any memory we currently are using
			
			err =  ClassID_g::BecomeClone(clone);//  pass clone to base class
			if(err) goto done;
			
			this->owner=cloneP->owner;
			
		}
	}
done:
	if(err) dynamic_cast<TTimeValue *>(this)->Dispose(); // don't leave ourselves in a weird state
	return err;
}