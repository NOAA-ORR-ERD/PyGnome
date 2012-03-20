/*
 *  LocaleWizard_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/19/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "LocaleWizard_c.h"
#include "StringFunctions.h"

#define STARTSTRING "RESNUM "

LocaleWizard_c::LocaleWizard_c()
{
	strcpy(this->className,"Location Wizard");
}

Boolean PathIsWizardResourceHelper(char* path,long* resNum)
{
	if(path && path[0])
	{
		if(!strncmpnocase(path,STARTSTRING,strlen(STARTSTRING)))
		{
			// get the file contents from a TEXT resource
			*resNum = atol(path+strlen(STARTSTRING));
			if(*resNum > 0)
			{
				return true;
			}
		}
	}
	return false;
}

Boolean LocaleWizard_c::PathIsWizardResource(char* path)
{
	long resNum;
	return PathIsWizardResourceHelper(path,&resNum);
}

