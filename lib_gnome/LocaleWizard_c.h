/*
 *  LocaleWizard_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/19/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __LocaleWizard_c__
#define __LocaleWizard_c__


#include "Basics.h"
#include "TypeDefs.h"
#include "ClassID_c.h"

Boolean PathIsWizardResourceHelper(char* path,long* resNum);

class LocaleWizard_c : virtual public ClassID_c {


public:
	LocaleWizard_c();

	Boolean PathIsWizardResource(char* path);

};



#endif