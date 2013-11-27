/*
 *  TOSSMTimeValue.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TOSSMTimeValue__
#define __TOSSMTimeValue__

#include "Earl.h"
#include "TypeDefs.h"
#include "OSSMTimeValue_c.h"
#include "TTimeValue.h"
#include "OUTILS.H"

class TOSSMTimeValue : virtual public OSSMTimeValue_c,  public TTimeValue
{

public:
	TOSSMTimeValue (TMover *theOwner);
	TOSSMTimeValue (TMover *theOwner,TimeValuePairH tvals,short userUnits);
	virtual				   ~TOSSMTimeValue () { Dispose (); }

	virtual ClassID 		GetClassID () { return TYPE_OSSMTIMEVALUES; }
	virtual Boolean			IAm(ClassID id) { if(id==TYPE_OSSMTIMEVALUES) return TRUE; return TTimeValue::IAm(id); }
	
	virtual OSErr 			MakeClone(TOSSMTimeValue **clonePtrPtr);
	virtual OSErr			BecomeClone(TOSSMTimeValue *clone);
	

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
