/*
 *  TTriGridVel.h
 *  c_gnome
 *
 *  Created by Alex Hadjilambris on 2/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TTriGridVel__
#define __TTriGridVel__

#include "Earl.h"
#include "TypeDefs.h"
#include "TriGridVel_c.h"
#include "GridVel/TGridVel.h"

class TTriGridVel : virtual public TriGridVel_c, public TGridVel
{
	
public:
	
	TTriGridVel(){fDagTree = 0; fBathymetryH=0;}
	virtual	~TTriGridVel() { Dispose (); }
	

	OSErr TextRead(char *path);
	OSErr Read(BFPB *bfpb);
	OSErr Write(BFPB *bfpb);
	virtual void Draw (Rect r, WorldRect view,WorldPoint refP,double refScale,
					   double arrowScale,Boolean bDrawArrows, Boolean bDrawGrid);
	void DrawBitMapTriangles (Rect r);
	void DrawCurvGridPts(Rect r, WorldRect view);
	
	//private:
	void DrawTriangle(Rect *r,long triNum,Boolean fillTriangle);
};

#endif