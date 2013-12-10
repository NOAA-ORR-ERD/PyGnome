/*
 *  Weatherer_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Weatherer_c__
#define __Weatherer_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "ClassID_c.h"

class Weatherer_c : virtual public ClassID_c {

public:
	Weatherer_c (char *name);
	Weatherer_c () {}
	
	//		void				SetWeathererName (char *newName)    { SetClassName (newName); }
	//		void				GetWeathererName (char *returnName) { GetClassName (returnName); }
	
	//virtual ClassID 	GetClassID () { return TYPE_WEATHERER; }
	virtual void		WeatherLE (LERec *theLE) {}
	
	
};

#endif
