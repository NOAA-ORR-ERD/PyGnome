
#include "Cross.h"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT TRANDOM
#endif
#endif


TRandom::TRandom (TMap *owner, char *name) : TMover (owner, name)
{
	fDiffusionCoefficient = 100000; //  cm**2/sec 
	memset(&fOptimize,0,sizeof(fOptimize));
	SetClassName (name);
	fUncertaintyFactor = 2;		// default uncertainty mult-factor
	bUseDepthDependent = false;
}

