/*
 *  TOSSMTimeValue.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "TOSSMTimeValue.h"
#include "MemUtils.h"
#include "CROSS.H"

Boolean IsLongWindFile(char* path,short *selectedUnitsP,Boolean *dataInGMTP);
Boolean IsHydrologyFile(char* path);
Boolean IsOSSMTideFile(char* path,short *selectedUnitsP);

TOSSMTimeValue::TOSSMTimeValue(TMover *theOwner,TimeValuePairH tvals,short userUnits) : TTimeValue(theOwner) 
{ 
	fileName[0]=0;
	timeValues = tvals;
	fUserUnits = userUnits;
	fFileType = OSSMTIMEFILE;
	fScaleFactor = 0.;
	fStationName[0] = 0;
	fStationPosition.pLat = 0;
	fStationPosition.pLong = 0;
	bOSSMStyle = true;
	fTransport = 0;
	fVelAtRefPt = 0;
}


TOSSMTimeValue::TOSSMTimeValue(TMover *theOwner) : TTimeValue(theOwner) 
{ 
	fileName[0]=0;
	timeValues = 0;
	fUserUnits = kUndefined; 
	fFileType = OSSMTIMEFILE;
	fScaleFactor = 0.;
	fStationName[0] = 0;
	fStationPosition.pLat = 0;
	fStationPosition.pLong = 0;
	bOSSMStyle = true;
	fTransport = 0;
	fVelAtRefPt = 0;
}

#define TOSSMMAXNUMDATALINESINLIST 201

long TOSSMTimeValue::GetListLength() 
{//JLM
	long listLength;
	 listLength =  this->GetNumValues();
	if(listLength > TOSSMMAXNUMDATALINESINLIST)
		listLength = TOSSMMAXNUMDATALINESINLIST; // JLM 7/21/00 , don't show the user too many lines in the case of a huge wind record
	return listLength;
}

ListItem TOSSMTimeValue::GetNthListItem(long n, short indent, short *style, char *text)
{//JLM
	ListItem item = { 0, 0, indent, 0 };
	text[0] = 0; 
	if( 0 <= n && n< TOSSMTimeValue::GetListLength())
	{
		DateTimeRec time;
		TimeValuePair pair;
		double valueInUserUnits, conversionFactor = 1.0;
		char *p,timeS[30];
		char unitsStr[32],valStr[32];
		
		if(n >=(TOSSMMAXNUMDATALINESINLIST-1))
		{	// JLM 7/21/00 ,this is the last line we will show, indicate that there are more lines but that we aren't going to show them 
			strcpy(text,"...  (there are too many lines to show here)");
			*style = normal;
			item.owner = dynamic_cast<TOSSMTimeValue *>(this);
			return item;
		}
		
		pair = INDEXH(this -> timeValues, n);
		SecondsToDate (pair.time, &time);
		Date2String(&time, timeS);
		if (p = strrchr(timeS, ':')) p[0] = 0; // remove seconds
		if (fFileType == HYDROLOGYFILE)
		{
			//ConvertToUnits (2, unitsStr);	// no units input, everything done in m/s
			valueInUserUnits = pair.value.u;
			/*switch(this->GetUserUnits())
			 {
			 case 1: strcpy(unitsStr,"CMS"); break;
			 case 2: strcpy(unitsStr,"KCMS"); break;
			 case 3: strcpy(unitsStr,"CFS"); break;
			 case 4: strcpy(unitsStr,"KCFS"); break;
			 }*/
			ConvertToTransportUnits(fUserUnits,unitsStr);
		}
		else
		{
			switch(this->GetUserUnits())
			{
				case kKnots: conversionFactor = KNOTSTOMETERSPERSEC; break;
				case kMilesPerHour: conversionFactor = MILESTOMETERSPERSEC; break;
				case kMetersPerSec: conversionFactor = 1.0; break;
					//default: err = -1; goto done;
			}
			valueInUserUnits = pair.value.u/conversionFactor; //JLM
			ConvertToUnits (this->GetUserUnits(), unitsStr);
		}
		
		StringWithoutTrailingZeros(valStr,valueInUserUnits,6); //JLM
		sprintf(text, "%s -> %s %s", timeS, valStr, unitsStr);///JLM
		*style = normal;
		item.owner = dynamic_cast<TOSSMTimeValue *>(this);
		//item.bullet = BULLET_DASH;
	}
	return item;
}

Boolean TOSSMTimeValue::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	
	/*if (inBullet) {
	 if (item.index == I_SHIOHIGHLOWS()) 
	 {fHighLowValuesOpen = !fHighLowValuesOpen; return TRUE;}
	 if (item.index == I_SHIOEBBFLOODS())
	 {fEbbFloodValuesOpen = !fEbbFloodValuesOpen; return TRUE;}
	 return TRUE;
	 }*/
	
	if (doubleClick && !inBullet)
	{
		TCATSMover *theOwner = dynamic_cast<TCATSMover*>(this->owner);
		Boolean timeFileChanged = false;
		if(theOwner)
			CATSSettingsDialog (theOwner, theOwner -> moverMap, &timeFileChanged);
		return TRUE;
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean TOSSMTimeValue::FunctionEnabled(ListItem item, short buttonID)
{
	if (buttonID == SETTINGSBUTTON) return TRUE;
	return FALSE;
}


OSErr TOSSMTimeValue::Write(BFPB *bfpb)
{
	long i, n = 0, version = /*1*/2;	// changed hydrology dialog 2/22/02
	ClassID id = GetClassID ();
	TimeValuePair pair;
	OSErr err = 0;
	
	if (err = TTimeValue::Write(bfpb)) return err;
	
	StartReadWriteSequence("TOSSMTimeValue::Write()");
	
	if (err = WriteMacValue(bfpb, fUserUnits)) return err;
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	if (err = WriteMacValue(bfpb, fileName, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fFileType)) return err;
	if (err = WriteMacValue(bfpb, fScaleFactor)) return err;
	//if (err = WriteMacValue(bfpb, fStationName, kMaxNameLen)) return err;
	//if (err = WriteMacValue(bfpb, fStationPosition.pLat)) return err;
	//if (err = WriteMacValue(bfpb, fStationPosition.pLong)) return err;
	if (err = WriteMacValue(bfpb, bOSSMStyle)) return err;
	if (err = WriteMacValue(bfpb, fTransport)) return err;
	if (err = WriteMacValue(bfpb, fVelAtRefPt)) return err;
	if (timeValues) n = this->GetNumValues();
	if (err = WriteMacValue(bfpb, n)) return err;
	
	if (timeValues)
		for (i = 0 ; i < n ; i++) {
			pair = INDEXH(timeValues, i);
			if (err = WriteMacValue(bfpb, pair.time)) return err;
			if (err = WriteMacValue(bfpb, pair.value.u)) return err;
			if (err = WriteMacValue(bfpb, pair.value.v)) return err;
		}
	
	return 0;
}

OSErr TOSSMTimeValue::Read(BFPB *bfpb)
{
	long i, n, version;
	ClassID id;
	TimeValuePair pair;
	OSErr err = 0;
	
	if (err = TTimeValue::Read(bfpb)) return err;
	
	StartReadWriteSequence("TOSSMTimeValue::Read()");
	
	if (err = ReadMacValue(bfpb, &fUserUnits)) return err;
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TOSSMTimeValue::Read()", "id != TYPE_OSSMTIMEVALUES", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	//if (version != 1) { printSaveFileVersionError(); return -1; }
	if (version > 2) { printSaveFileVersionError(); return -1; }
	if (err = ReadMacValue(bfpb, fileName, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb, &fFileType)) return err;
	if (err = ReadMacValue(bfpb, &fScaleFactor)) return err;
	if (version>1)
	{
		//if (err = ReadMacValue(bfpb, fStationName, kMaxNameLen)) return err;
		//if (err = ReadMacValue(bfpb, &fStationPosition.pLat)) return err;	// could get this from CATSMover refP
		//if (err = ReadMacValue(bfpb, &fStationPosition.pLong)) return err;
		if (err = ReadMacValue(bfpb, &bOSSMStyle)) return err;
		if (err = ReadMacValue(bfpb, &fTransport)) return err;	
		if (err = ReadMacValue(bfpb, &fVelAtRefPt)) return err;
	}
	if (err = ReadMacValue(bfpb, &n)) return err;
	
	if(n>0)
	{	// JLM: note: n = 0 means timeValues was originally nil
		// so only allocate if n> 0
		timeValues = (TimeValuePairH)_NewHandle(n * sizeof(TimeValuePair));
		if (!timeValues)
		{ TechError("TOSSMTimeValue::Read()", "_NewHandle()", 0); return -1; }
		
		if (timeValues)
			for (i = 0 ; i < n ; i++) {
				if (err = ReadMacValue(bfpb, &pair.time)) return err;
				if (err = ReadMacValue(bfpb, &pair.value.u)) return err;
				if (err = ReadMacValue(bfpb, &pair.value.v)) return err;
				INDEXH(timeValues, i) = pair;
			}
	}
	
	return err;
}




OSErr TOSSMTimeValue::CheckAndPassOnMessage(TModelMessage *message)
{	
	//char ourName[kMaxNameLen];
	
	// see if the message is of concern to us
	//this->GetClassName(ourName);
	
	/////////////////////////////////////////////////
	// we have no sub-guys that that need us to pass this message 
	/////////////////////////////////////////////////
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TTimeValue::CheckAndPassOnMessage(message);
}

OSErr TOSSMTimeValue::MakeClone(TOSSMTimeValue **clonePtrPtr)
{
	// clone should be the address of a  ClassID ptr
	ClassID *clone;
	OSErr err = 0;
	Boolean weCreatedIt = false;
	if(!clonePtrPtr) return -1; 
	if(*clonePtrPtr == nil)
	{	// create and return a cloned object.
		*clonePtrPtr = new TOSSMTimeValue(this->owner);
		weCreatedIt = true;
		if(!*clonePtrPtr) { TechError("MakeClone()", "new TConstantMover()", 0); return memFullErr;}	
	}
	if(*clonePtrPtr)
	{	// copy the fields
		if((*clonePtrPtr)->GetClassID() == this->GetClassID()) // safety check
		{
			TOSSMTimeValue * cloneP = dynamic_cast<TOSSMTimeValue *>(*clonePtrPtr);// typecast
			TTimeValue *tObj = dynamic_cast<TTimeValue *>(*clonePtrPtr);
			err =  TTimeValue::MakeClone(&tObj);//  pass clone to base class
			if(!err) 
			{
				if(this->timeValues)
				{
					cloneP->timeValues = this->timeValues;
					err = _HandToHand((Handle *)&cloneP->timeValues);
					if(err) 
					{
						cloneP->timeValues = nil;
						goto done;
					}
				}
				
				strcpy(cloneP->fileName,this->fileName);
				cloneP->fUserUnits = this->fUserUnits;
				cloneP->fFileType = this->fFileType;
				cloneP->fScaleFactor = this->fScaleFactor;
				strcpy(cloneP->fStationName,this->fStationName);
				cloneP->fStationPosition = this->fStationPosition;
				cloneP->bOSSMStyle = this->bOSSMStyle;
				cloneP->fTransport = this->fTransport;
				cloneP->fVelAtRefPt = this->fVelAtRefPt;
				
			}
		}
	}
done:
	if(err && *clonePtrPtr) 
	{
		(*clonePtrPtr)->Dispose();
		if(weCreatedIt)
		{
			delete *clonePtrPtr;
			*clonePtrPtr = nil;
		}
	}
	return err;
}


OSErr TOSSMTimeValue::BecomeClone(TOSSMTimeValue *clone)
{
	OSErr err = 0;
	
	if(clone)
	{
		if(clone->GetClassID() == this->GetClassID()) // safety check
		{
			TOSSMTimeValue * cloneP = dynamic_cast<TOSSMTimeValue *>(clone);// typecast
			
			/*OK*/ dynamic_cast<TOSSMTimeValue *>(this)->TOSSMTimeValue::Dispose(); // get rid of any memory we currently are using
			////////////////////
			// do the memory stuff first, in case it fails
			////////
			if(cloneP->timeValues)
			{
				this->timeValues = cloneP->timeValues;
				err = _HandToHand((Handle *)&this->timeValues);
				if(err) 
				{
					this->timeValues = nil;
					goto done;
				}
			}
			
			err =  TTimeValue::BecomeClone(clone);//  pass clone to base class
			if(err) goto done;
			
			strcpy(this->fileName,cloneP->fileName);
			this->fUserUnits = cloneP->fUserUnits;
			this->fFileType = cloneP->fFileType;
			this->fScaleFactor = cloneP->fScaleFactor;
			strcpy(this->fStationName,cloneP->fStationName);
			this->fStationPosition = cloneP->fStationPosition;
			this->bOSSMStyle = cloneP->bOSSMStyle;
			this->fTransport = cloneP->fTransport;
			this->fVelAtRefPt = cloneP->fVelAtRefPt;
			
		}
	}
done:
	if(err) /*OK*/ dynamic_cast<TOSSMTimeValue *>(this)->TOSSMTimeValue::Dispose(); // don't leave ourselves in a weird state
	return err;
}