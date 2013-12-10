/*
 *  OSSMWeatherer_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __OSSMWeatherer_c__
#define __OSSMWeatherer_c__


#include "Basics.h"
#include "TypeDefs.h"
#include "Weatherer_c.h"
#include "CMYLIST.H"

class OSSMWeatherer_c : virtual public Weatherer_c {

public:
	CMyList				*componentsList;
	
	OSSMWeatherer_c (char *name);
	OSSMWeatherer_c () {}
	//virtual ClassID		GetClassID () { return TYPE_OSSMWEATHERER; }
	virtual void		WeatherLE (LERec *theLE);

};

#endif