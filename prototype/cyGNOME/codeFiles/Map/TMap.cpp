/*
 *  TMap.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "TMap.h"
#include "Mover/TMover.h"
#include "CROSS.H"

TMap::TMap(char *name, WorldRect bounds) 
{
	SetMapName(name);
	fMapBounds = bounds;
	
	moverList = 0;
	
	SetDirty(FALSE);
	
	bOpen = TRUE;
	bMoversOpen = TRUE;
	
	fRefloatHalfLifeInHrs = 1.0;
	
	bIAmPartOfACompoundMap = false;
}

void TMap::Dispose()
{
	long i, n;
	TMover *mover;
	
	if (moverList != nil)
	{
		for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
			moverList->GetListItem((Ptr)&mover, i);
			mover->Dispose();
			delete mover;
		}
		
		moverList->Dispose();
		delete moverList;
		moverList = nil;
	}
}

OSErr TMap::InitMap()
{
	OSErr err = 0;
	moverList = new CMyList(sizeof(TMover *));
	if (!moverList)
	{ TechError("TMap::InitMap()", "new CMyList()", 0); return -1; }
	if (err = moverList->IList())
	{ TechError("TMap::InitMap()", "IList()", 0); return -1; }
	
	return 0;
}