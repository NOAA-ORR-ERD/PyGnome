/*
 *  NetCDFMoverTri_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/2/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFMoverTri_b__
#define __NetCDFMoverTri_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "NetCDFMoverCurv_b.h"

class NetCDFMoverTri_b : virtual public NetCDFMoverCurv_b {

public:
	//LONGH fVerdatToNetCDFH;	
	//WORLDPOINTFH fVertexPtsH;	// may not need this if set pts in dagtree	
	long fNumNodes;
	long fNumEles;	//for now, number of triangles
	Boolean bVelocitiesOnTriangles;
	
	
};

#endif
