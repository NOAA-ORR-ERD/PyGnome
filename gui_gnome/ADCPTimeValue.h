/*
 *  ADCPTimeValue.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADCPTimeValue__
#define __ADCPTimeValue__

#include "Earl.h"
#include "TypeDefs.h"
#include "ADCPTimeValue_c.h"
#include "TTimeValue.h"

class ADCPTimeValue : virtual public ADCPTimeValue_c,  public TTimeValue
{
	
public:
	ADCPTimeValue (TMover *theOwner);
	ADCPTimeValue (TMover *theOwner,TimeValuePairH3D tvals,short userUnits);
	virtual				   ~ADCPTimeValue () { Dispose (); }
	virtual void			Dispose ();
	virtual OSErr 			MakeClone(ADCPTimeValue **clonePtrPtr);
	virtual OSErr 			BecomeClone(ADCPTimeValue *clone);
	
	virtual OSErr			InitTimeFunc ();
	//virtual OSErr			ReadTimeValues_old (char *path, short format, short unitsIfKnownInAdvance);
	virtual OSErr			ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance);
	OSErr					ReadMetaDataFile (char *path);
	//OSErr 			ReadHydrologyHeader (char *path);
	virtual ClassID 		GetClassID () { return TYPE_ADCPTIMEVALUES; }
	virtual Boolean			IAm(ClassID id) { if(id==TYPE_ADCPTIMEVALUES) return TRUE; return TTimeValue::IAm(id); }
	// I/O methods
	virtual OSErr 			Read  (BFPB *bfpb);  // read from current position
	virtual OSErr 			Write (BFPB *bfpb);  // write to  current position
	
	virtual long 			GetListLength ();
	virtual ListItem 		GetNthListItem 	(long n, short indent, short *style, char *text);
	virtual Boolean 		ListClick 	  	(ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 		FunctionEnabled (ListItem item, short buttonID);
	
	virtual OSErr 			CheckAndPassOnMessage(TModelMessage *message);
	
	
};


#endif
