/*
 *  TModelMessage.h
 *  c_gnome
 *
 *  Created by Generic Programmer on 2/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TModelMessage__
#define __TModelMessage__

#include "Earl.h"
#include "TypeDefs.h"

class TModelMessage
{
public:
	// constructors
	TModelMessage(long messageCode,char* targetName,char *dataStr);
	TModelMessage(long messageCode,char* targetName,CHARH dataHdl);
	TModelMessage(long messageCode,UNIQUEID targetUniqueID,char *dataStr);
	TModelMessage(long messageCode,UNIQUEID targetUniqueID,CHARH dataHdl);
	// destructor
	~TModelMessage(); // does this need to be virtual?
	
	// Message utilities
	Boolean IsMessage(long messageCode);
	Boolean IsMessage(char* targetName);
	Boolean IsMessage(long messageCode,char* targetName);
	Boolean IsMessage(long messageCode,UNIQUEID targetUniqueID);
	long GetMessageCode(void) {return fMessageCode;}
	void GetParameterString(char * key,char * answerStr,long maxNumChars);
	OSErr GetParameterAsDouble(char * key,double * val);
	OSErr GetParameterAsLong(char * key,long * val);
	OSErr GetParameterAsBoolean(char * key,Boolean * val);
	OSErr GetParameterAsWorldPoint(char * key,WorldPoint * val,Boolean checkForLLInputWithoutDirection);
	OSErr GetParameterAsSeconds(char * key,Seconds * val);
	
	
private:
	Boolean StringsMatch(char* str1,char* str2);
	// instance variables
	long fMessageCode;
	UNIQUEID fTargetUniqueID;
	char* fTargetName;
	char *fDataStr;
	CHARH fDataHdl;
};


#endif