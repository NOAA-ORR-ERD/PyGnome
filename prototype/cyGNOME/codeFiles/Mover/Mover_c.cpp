/*
 *  Mover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Mover_c.h"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT MOVER_C
#endif
#endif


OSErr Mover_c::InitMover()
{
	return noErr;
}

OSErr Mover_c::UpdateUncertainty(void)
{
	return 0;	
}


WorldPoint3D Mover_c::GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType) 
{
	//WorldPoint3D theLE3D [] = {(*theLE).p.pLat,(*theLE).p.pLong,(*theLE).z}; 
	WorldPoint3D theLE3D; 
	theLE3D.p.pLat = (*theLE).p.pLat;
	theLE3D.p.pLong = (*theLE).p.pLong;
	theLE3D.z = (*theLE).z; 
	return theLE3D;
}
