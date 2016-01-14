/*
 *  TimeValuesIOPD.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/27/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */


TOSSMTimeValue* LoadTOSSMTimeValue(TMover *theOwner,short unitsIfKnownInAdvance);
TOSSMTimeValue* CreateTOSSMTimeValue(TMover *theOwner,char* path, char* shortFileName, short unitsIfKnownInAdvance); //JLM
OSErr GetScaleFactorFromUser(char *msg, double *scaleFactor);