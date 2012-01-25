
#include "TWindMover.h"
#include "OUtils.h"
#include "Cross.h"
#include "EditWindsDialog.h"

#ifdef MAC
#ifdef MPW
#include <QDOffscreen.h>
#pragma SEGMENT TWINDMOVER
#endif
#endif

TWindMover::TWindMover(TMap *owner,char* name) : TMover(owner, name)
{
	if(!name || !name[0]) this->SetClassName("Variable Wind"); // JLM , a default useful in the wizard
	timeDep = nil;
	
	fUncertainStartTime = 0;
	fDuration = 3*3600; // 3 hours
	
	fWindUncertaintyList = 0;
	fLESetSizes = 0;
	
	fSpeedScale = 2;
	fAngleScale = .4;
	fMaxSpeed = 30; //mps
	fMaxAngle = 60; //degrees
	fSigma2 =0;
	fSigmaTheta =  0; 
	//conversion = 1.0;// JLM , I think this field should be removed
	bTimeFileOpen = FALSE;
	bUncertaintyPointOpen=false;
	bSubsurfaceActive = false;
	fGamma = 1.;
	
	fIsConstantWind = false;
	fConstantValue.u = fConstantValue.v = 0.0;
	
	memset(&fWindBarbRect,0,sizeof(fWindBarbRect)); 
	bShowWindBarb = true;
}



void TWindMover::Dispose()
{
	DeleteTimeDep ();
	
	this->DisposeUncertainty();
	
	TMover::Dispose ();
}



