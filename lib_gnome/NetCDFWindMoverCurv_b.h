/*
 *  NetCDFWindMoverCurv_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFWindMoverCurv_b__
#define __NetCDFWindMoverCurv_b__

#include "Basics.h"
#include "TypeDefs.h"
#include "NetCDFWindMover_b.h"

class NetCDFWindMoverCurv_b : virtual public NetCDFWindMover_b {

public:
	LONGH fVerdatToNetCDFH;	// for curvilinear
	WORLDPOINTFH fVertexPtsH;		// for curvilinear, all vertex points from file
	
};

#endif
