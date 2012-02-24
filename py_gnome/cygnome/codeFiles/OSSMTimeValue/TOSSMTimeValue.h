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
#include "TimeValue/TTimeValue.h"
#include "OUTILS.H"

class TOSSMTimeValue : virtual public OSSMTimeValue_c, public TTimeValue
{

public:
	using OSSMTimeValue_c::Dispose;
	TOSSMTimeValue (TMover *theOwner);
	TOSSMTimeValue (TMover *theOwner,TimeValuePairH tvals,short userUnits);

	virtual				   ~TOSSMTimeValue () { Dispose (); }
	virtual OSErr 			MakeClone(TClassID **clonePtrPtr);
	virtual OSErr 			BecomeClone(TClassID *clone);
	
	virtual OSErr			ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance);
	OSErr 			ReadHydrologyHeader (char *path);

	virtual void 			GetTimeFileName (char *theName) { strcpy (theName, fileName); }
	
	
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
