/*
 *  NetCDFMoverCurv_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFMoverCurv_b__
#define __NetCDFMoverCurv_b__


#include "Earl.h"
#include "TypeDefs.h"
#include "NetCDFMover/NetCDFMover_b.h"

class NetCDFMoverCurv_b : virtual public NetCDFMover_b {

public:
	LONGH fVerdatToNetCDFH;	// for curvilinear
	WORLDPOINTFH fVertexPtsH;		// for curvilinear, all vertex points from file
	LONGH fVerdatToNetCDFH_2;	// for curvilinear
	


};

#endif
