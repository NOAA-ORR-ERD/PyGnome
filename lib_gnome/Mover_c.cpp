/*
 *  Mover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Mover_c.h"

//#ifdef pyGNOME
//#define TMap Map_c
//#endif

#ifndef pyGNOME
Mover_c::Mover_c(TMap *owner, char *name)
{
	SetMoverName(name);
	SetMoverMap(owner);
	
	bActive = true;
	//bOpen = true;
	bOpen = false; //JLM, I prefer them to be initally closed, otherwise they clutter the list too much
	fUncertainStartTime = 0;
	fDuration = 0; // JLM 9/18/98
	fTimeUncertaintyWasSet = 0;// JLM 9/18/98
	//fColor = colors[PURPLE];	// default to draw arrows in purple
}
#endif

Mover_c::Mover_c()
{	
	bActive = true;
	bOpen = false; //won't need this for pyGNOME
	fUncertainStartTime = 0;
	fDuration = 0; // JLM 9/18/98
	fTimeUncertaintyWasSet = 0;// JLM 9/18/98
}



Mover_c::~Mover_c()
{
	Dispose ();
}
OSErr Mover_c::UpdateUncertainty(void)
{
	return 0;	
}


WorldPoint3D Mover_c::GetMove (const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType) 
{
	//WorldPoint3D theLE3D [] = {(*theLE).p.pLat,(*theLE).p.pLong,(*theLE).z}; 
	WorldPoint3D theLE3D; 
	theLE3D.p.pLat = (*theLE).p.pLat;
	theLE3D.p.pLong = (*theLE).p.pLong;
	theLE3D.z = (*theLE).z; 
	return theLE3D;
}

//#undef TMap