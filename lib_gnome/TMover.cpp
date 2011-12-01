

#include "TMover.h"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT TMOVER
#endif
#endif


TMover::TMover(TMap *owner, char *name)
{
	SetMoverName(name);
	SetMoverMap(owner);
	
	bActive = true;
	//bOpen = true;
	bOpen = false; //JLM, I prefer them to be initally closed, otherwise they clutter the list too much
	fUncertainStartTime = 0;
	fDuration = 0; // JLM 9/18/98
	fTimeUncertaintyWasSet = 0;// JLM 9/18/98
	
	fColor = colors[PURPLE];	// default to draw arrows in purple
}


void TMover::Dispose()
{
	// nothing to dispose of	
}


