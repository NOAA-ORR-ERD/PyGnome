/*
 *  TGridVel.h
 *  c_gnome
 *
 *  Created by Generic Programmer on 2/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TGridVel__
#define __TGridVel__

#include "Earl.h"
#include "TypeDefs.h"
#include "GridVel_c.h"

class TGridVel : virtual public GridVel_c {
	
public:
	
	TGridVel();
	virtual	~TGridVel() { Dispose (); }
	
	virtual OSErr TextWrite(char *path){return noErr;}
	virtual OSErr TextRead (char *path){return noErr;}
	virtual OSErr Write(BFPB *bfpb)=0;
	virtual OSErr Read (BFPB *bfpb)=0;

	virtual void Draw(Rect r, WorldRect view,WorldPoint refP,double refScale,
					  double arrowScale,Boolean bDrawArrows, Boolean bDrawGrid)=0;

};


#endif