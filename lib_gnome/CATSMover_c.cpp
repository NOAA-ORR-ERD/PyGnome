/*
 *  CATSMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Basics.h"
#include "CATSMover_c.h"
#include "CompFunctions.h"
#include "MemUtils.h"
#include "DagTreeIO.h"
#include "StringFunctions.h"
#include "OUTILS.H"

#ifndef pyGNOME
#include "CROSS.H"
#include "TCATSMover.h"
#include "TModel.h"
#include "TMap.h"
#include "GridVel.h"
extern TModel *model;
#else
#include "Replacements.h"
extern Model_c *model;
#endif

using std::fstream;
using std::ios;
using std::cout;

CATSMover_c::CATSMover_c () { 
	
	fDuration=48*3600; //48 hrs as seconds 
	fTimeUncertaintyWasSet =0;
	fLESetSizesH = 0;
	fUncertaintyListH = 0;
	fGrid = 0;
	SetTimeDep (nil);
	bTimeFileActive = false;
	fEddyDiffusion=0; // JLM 5/20/991e6; // cm^2/sec
	fEddyV0 = 0.1; // JLM 5/20/99

	bApplyLogProfile = false;

	memset(&fOptimize,0,sizeof(fOptimize));

}

CATSMover_c::CATSMover_c (TMap *owner, char *name) : CurrentMover_c(owner, name)
{
	fDuration=48*3600; //48 hrs as seconds 
	fTimeUncertaintyWasSet =0;
	
	fGrid = 0;
	SetTimeDep (nil);
	bTimeFileActive = false;
	fEddyDiffusion=0; // JLM 5/20/991e6; // cm^2/sec
	fEddyV0 = 0.1; // JLM 5/20/99
	
	bApplyLogProfile = false;

	memset(&fOptimize,0,sizeof(fOptimize));
	SetClassName (name);
}

#ifdef pyGNOME

OSErr CATSMover_c::ComputeVelocityScale(const Seconds& model_time) {	// AH 08/08/2012
	// this function computes and sets this->refScale
	// returns Error when the refScale is not defined
	// or in allowable range.  
	// Note it also sets the refScale to 0 if there is an error
#define MAXREFSCALE  1.0e6  // 1 million times is too much
#define MIN_UNSCALED_REF_LENGTH 1.0E-5 // it's way too small
	long i, j, m, n;
	double length, theirLengthSq, myLengthSq, dotProduct;
	VelocityRec theirVelocity,myVelocity;
	WorldPoint3D refPt3D = {0,0,0.};
	TMap *map;
	TCATSMover *mover;
	
	if (this->timeDep && this->timeDep->fFileType==HYDROLOGYFILE)
	{
		this->refScale = this->timeDep->fScaleFactor;
		return noErr;
	}
	
	refPt3D.p = refP; refPt3D.z = 0.;
	
	switch (scaleType) {
		case SCALE_NONE: this->refScale = 1; return noErr;
		case SCALE_CONSTANT:
			myVelocity = GetPatValue(refPt3D);
			length = sqrt(myVelocity.u * myVelocity.u + myVelocity.v * myVelocity.v);
			/// check for too small lengths
			if(fabs(scaleValue) > length*MAXREFSCALE
			   || length < MIN_UNSCALED_REF_LENGTH)
			{ this->refScale = 0;return -1;} // unable to compute refScale
			this->refScale = scaleValue / length; 
			return noErr;
	}
	
	this->refScale = 0;
	return -1;
	
}

#else

OSErr CATSMover_c::ComputeVelocityScale(const Seconds& model_time)
{	// this function computes and sets this->refScale
	// returns Error when the refScale is not defined
	// or in allowable range.  
	// Note it also sets the refScale to 0 if there is an error
#define MAXREFSCALE  1.0e6  // 1 million times is too much
#define MIN_UNSCALED_REF_LENGTH 1.0E-5 // it's way too small
	long i, j, m, n;
	double length, theirLengthSq, myLengthSq, dotProduct;
	VelocityRec theirVelocity,myVelocity;
	WorldPoint3D refPt3D = {0,0,0.};
	TMap *map;
	TCATSMover *mover;
	
	if (this->timeDep && this->timeDep->fFileType==HYDROLOGYFILE)
	{
		this->refScale = this->timeDep->fScaleFactor;
		return noErr;
	}
	
	refPt3D.p = refP; refPt3D.z = 0.;
	
	switch (scaleType) {
		case SCALE_NONE: this->refScale = 1; return noErr;
		case SCALE_CONSTANT:
			myVelocity = GetPatValue(refPt3D);
			length = sqrt(myVelocity.u * myVelocity.u + myVelocity.v * myVelocity.v);
			/// check for too small lengths
			if(fabs(scaleValue) > length*MAXREFSCALE
			   || length < MIN_UNSCALED_REF_LENGTH)
			{ this->refScale = 0;return -1;} // unable to compute refScale
			this->refScale = scaleValue / length; 
			return noErr;
		case SCALE_OTHERGRID:
			for (j = 0, m = model -> mapList -> GetItemCount() ; j < m ; j++) {
				model -> mapList -> GetListItem((Ptr)&map, j);
				
				for (i = 0, n = map -> moverList -> GetItemCount() ; i < n ; i++) {
					map -> moverList -> GetListItem((Ptr)&mover, i);
					if (mover -> GetClassID() != TYPE_CATSMOVER) continue;
					if (!strcmp(mover -> className, scaleOtherFile)) {
						// JLM, note: we are implicitly matching by file name above
						
						// JLM: This code left out the possibility of a time file
						//velocity = mover -> GetPatValue(refPt3D);
						//velocity.u *= mover -> refScale;
						//velocity.v *= mover -> refScale;
						// so use GetScaledPatValue() instead
						theirVelocity = mover -> GetScaledPatValue(model_time, refPt3D,nil);	// AH 07/10/2012
						
						theirLengthSq = (theirVelocity.u * theirVelocity.u + theirVelocity.v * theirVelocity.v);
						// JLM, we need to adjust the movers pattern 
						myVelocity = GetPatValue(refPt3D);
						myLengthSq = (myVelocity.u * myVelocity.u + myVelocity.v * myVelocity.v);
						// next problem is that the scale 
						// can be negative, we would have to look at the angle between 
						// these guys
						
						///////////////////////
						// JLM wonders if we should use a refScale 
						// that will give us the projection of their 
						// vector onto our vector instead of 
						// matching lengths.
						// Bushy etc may have a reason for the present method
						//
						// code goes here
						// ask about the proper method
						///////////////////////
						
						// JLM,  check for really small lengths
						if(theirLengthSq > myLengthSq*MAXREFSCALE*MAXREFSCALE
						   || myLengthSq <  MIN_UNSCALED_REF_LENGTH*MIN_UNSCALED_REF_LENGTH)
						{ this->refScale = 0;return -1;} // unable to compute refScale
						
						dotProduct = myVelocity.u * theirVelocity.u + myVelocity.v * theirVelocity.v;
						this->refScale = sqrt(theirLengthSq / myLengthSq);
						if(dotProduct < 0) this->refScale = -(this->refScale);
						return noErr;
					}
				}
			}
			break;
	}
	
	this->refScale = 0;
	return -1;
}

#endif

OSErr CATSMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty)
{
	/// 5/12/99 only add the eddy uncertainty when told to
	
	double u,v,lengthS,alpha,beta,v0,gammaScale;
	LEUncertainRec unrec;
	float rand1,rand2;
	OSErr err = 0;
	
	err = this -> UpdateUncertainty();
	if(err) return err;
	
	if(!fUncertaintyListH || !fLESetSizesH) return 0; // this is our clue to not add uncertainty
	
	
	if(useEddyUncertainty)
	{
		if(this -> fOptimize.isFirstStep)
		{
			GetRandomVectorInUnitCircle(&rand1,&rand2);
		}
		else
		{
			rand1 = GetRandomFloat(-1.0, 1.0);
			rand2 = GetRandomFloat(-1.0, 1.0);
		}
	}
	else
	{	// no need to calculate these when useEddyUncertainty is false
		rand1 = 0;
		rand2 = 0;
	}
	
	
	if(fUncertaintyListH && fLESetSizesH)
	{
		unrec=(*fUncertaintyListH)[(*fLESetSizesH)[setIndex]+leIndex];
		lengthS = sqrt(patVelocity->u*patVelocity->u + patVelocity->v * patVelocity->v);
		
		
		u = patVelocity->u;
		v = patVelocity->v;
		
		if(!this -> fOptimize.isOptimizedForStep)  this -> fOptimize.value = sqrt(6*(fEddyDiffusion/10000)/timeStep); // in m/s, note: DIVIDED by timestep because this is later multiplied by the timestep
		
		v0 = this -> fEddyV0;		 //meters /second
		
		if(lengthS>1e-6) // so we don't divide by zero
		{	
			if(useEddyUncertainty) gammaScale = this -> fOptimize.value * v0 /(lengthS * (v0+lengthS));
			else  gammaScale = 0.0;
			
			alpha = unrec.downStream + gammaScale * rand1;
			beta = unrec.crossStream + gammaScale * rand2;
			
			patVelocity->u = u*(1+alpha)+v*beta;
			patVelocity->v = v*(1+alpha)-u*beta;	
		}
		else
		{	// when lengthS is too small, ignore the downstream and cross stream and only use diffusion uncertainty	
			if(useEddyUncertainty) { // provided we are supposed to
				patVelocity->u = this -> fOptimize.value * rand1;
				patVelocity->v = this -> fOptimize.value * rand2;
			}
		}
	}
	else 
	{
		TechError("TCATSMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		patVelocity->u=patVelocity->v=0;
	}
	return err;
}

OSErr CATSMover_c::PrepareForModelRun()
{
	this -> fOptimize.isFirstStep = true;
	return noErr;
}

OSErr CATSMover_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain)
{
	OSErr err =0;
	if (err = CurrentMover_c::PrepareForModelStep(model_time, time_step, uncertain)) 
		return err; // note: this calls UpdateUncertainty()
	
	err = this -> ComputeVelocityScale(model_time);	// AH 07/10/2012
	
	this -> fOptimize.isOptimizedForStep = true;
	this -> fOptimize.value = sqrt(6*(fEddyDiffusion/10000)/time_step); // in m/s, note: DIVIDED by timestep because this is later multiplied by the timestep
	//this -> fOptimize.isFirstStep = (model_time == start_time);
	
	if (err) 
		printError("An error occurred in TCATSMover::PrepareForModelStep");
	return err;
}

void CATSMover_c::ModelStepIsDone()
{
	this -> fOptimize.isFirstStep = false;
	memset(&fOptimize,0,sizeof(fOptimize));
}

OSErr CATSMover_c::get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID) {	

	if(!delta || !ref) {
		//cout << "worldpoints array not provided! returning.\n";
		return 1;
	}
	
	// For LEType spillType, check to make sure it is within the valid values
	if( spillType < FORECAST_LE || spillType > UNCERTAINTY_LE)
	{
		// cout << "Invalid spillType.\n";
		return 2;
	}
	
	LERec* prec;
	LERec rec;
	prec = &rec;
	
	WorldPoint3D zero_delta ={0,0,0.};
	
	for (int i = 0; i < n; i++) {
		
		// only operate on LE if the status is in water
		if( LE_status[i] != OILSTAT_INWATER)
		{
			delta[i] = zero_delta;
			continue;
		}
		rec.p = ref[i].p;
		rec.z = ref[i].z;

		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1000000;	
		rec.p.pLong*= 1000000;
		
		delta[i] = GetMove(model_time, step_len, spill_ID, i, prec, spillType);
		
		delta[i].p.pLat /= 1000000;
		delta[i].p.pLong /= 1000000;
	}
	
	return noErr;
}

WorldPoint3D CATSMover_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	Boolean useEddyUncertainty = false;	
	double 		dLong, dLat;
	WorldPoint3D	deltaPoint={0,0,0.};
	WorldPoint3D refPoint3D = {0,0,0.};
	//WorldPoint refPoint = (*theLE).p;	
	VelocityRec scaledPatVelocity;

	refPoint3D.p = (*theLE).p;
	refPoint3D.z = (*theLE).z;
	
	scaledPatVelocity = this->GetScaledPatValue(model_time, refPoint3D,&useEddyUncertainty); // AH 07/10/2012
	
	if(leType == UNCERTAINTY_LE)
	{
		AddUncertainty(setIndex,leIndex,&scaledPatVelocity,timeStep,useEddyUncertainty);
	}
	dLong = ((scaledPatVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint3D.p.pLat);
	dLat  =  (scaledPatVelocity.v / METERSPERDEGREELAT) * timeStep;
	
	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}

VelocityRec CATSMover_c::GetScaledPatValue(const Seconds& model_time, WorldPoint3D p,Boolean * useEddyUncertainty)
{
	/// 5/12/99 JLM, we only add the eddy uncertainty when the vectors are big enough when the timeValue is 1 
	// This is in response to the Prince William sound problem where 5 patterns are being added together
	VelocityRec	patVelocity, timeValue = {1, 1};
	float lengthSquaredBeforeTimeFactor;
	OSErr err = 0;
	
	if(!this -> fOptimize.isOptimizedForStep && this->scaleType == SCALE_OTHERGRID) 
	{	// we need to update refScale
		this->ComputeVelocityScale(model_time);	// AH 07/10/2012
	}
	
	// get and apply our time file scale factor
	if (timeDep && bTimeFileActive)
	{
		// VelocityRec errVelocity={1,1};
		// JLM 11/22/99, if there are no time file values, use zero not 1
		VelocityRec errVelocity={0,1}; 
		err = timeDep -> GetTimeValue (model_time, &timeValue); // AH 07/10/2012
		if(err) timeValue = errVelocity;
	}
	
	
	patVelocity = GetPatValue (p);
	//	patVelocity = GetSmoothVelocity (p);
	
	patVelocity.u *= refScale; 
	patVelocity.v *= refScale; 

	if(useEddyUncertainty)
	{ // if they gave us a pointer to a boolean fill it in, otherwise don't
		lengthSquaredBeforeTimeFactor = patVelocity.u*patVelocity.u + patVelocity.v*patVelocity.v;
		if(lengthSquaredBeforeTimeFactor < (this -> fEddyV0 * this -> fEddyV0)) *useEddyUncertainty = false; 
		else *useEddyUncertainty = true;
	}
	
	patVelocity.u *= timeValue.u; // magnitude contained in u field only
	patVelocity.v *= timeValue.u; // magnitude contained in u field only

	return patVelocity;
}


VelocityRec CATSMover_c::GetPatValue(WorldPoint3D p)
{
	//Boolean bApplyLogProfile = false;
	double depthAtPoint = 0., scaleFactor = 1.;
	VelocityRec patVal = {0.,0.};

	if (p.z > 0 && !bApplyLogProfile) 
	{	
		if (!IAm(TYPE_CATSMOVER3D))
		return patVal;	
	}
	//if (IAm(TYPE_CATSMOVER3D)) return fGrid->GetPatValue(p.p);	// need to either store the fBathymetry or get the depths from the grid
	//if (gNoaaVersion) bApplyLogProfile = true;	// this will be a checkbox on the dialog
	if (p.z > 1 && bApplyLogProfile)	// start the profile after the first meter
	{
		//depthAtPoint = ((TTriGridVel*)fGrid)->GetDepthAtPoint(p.p);
		depthAtPoint = fGrid->GetDepthAtPoint(p.p);
		if (p.z >= depthAtPoint) scaleFactor = 0.;
		else if (depthAtPoint > 0) scaleFactor = 1. - log(p.z)/log(depthAtPoint);
	}
	patVal = fGrid->GetPatValue(p.p);
	patVal.u = patVal.u * scaleFactor;
	patVal.v = patVal.v * scaleFactor;

	return patVal;
	//return fGrid->GetPatValue(p.p);
}

VelocityRec CATSMover_c::GetSmoothVelocity (WorldPoint p)
{
	return fGrid->GetSmoothVelocity(p);
}

Boolean CATSMover_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	
	wp.z = arrowDepth;
	velocity = this->GetPatValue(wp);
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->refScale * lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
			this->className, uStr, sStr);
	return true;
	
}

void CATSMover_c::DeleteTimeDep () 
{
	if (timeDep)
	{
		timeDep -> Dispose ();
		delete timeDep;
		timeDep = nil;
	}
	
	return;
}

OSErr CATSMover_c::ReadTopology(char* path, TMap **newMap)
{
	// import PtCur triangle info so don't have to regenerate
	char s[1024], errmsg[256];
	long i, numPoints, numTopoPoints, line = 0, numPts;
	CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH depths=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds = voidWorldRect;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	//long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	//LONGH boundarySegs=0, waterBoundaries=0;
	
	errmsg[0]=0;
	
	if (!path || !path[0]) return 0;
	
	//	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
	//		TechError("TCATSMover::ReadTopology()", "ReadFileContents()", err);
	//		goto done;
	//	}

	try {
		char c;
		int x = i = 0;
		int j = 0;
		fstream *_ifstream = new fstream(path, ios::in);
		for(; _ifstream->get(c); x++);
		delete _ifstream;
		if(!(x > 0))
			throw("empty file.\n");
		_ifstream = new fstream(path, ios::in);
		for(; j < 7; j++) _ifstream->get(c);
		do {
			_ifstream->get(c);
			j++;
		} while((int)c == LINEFEED || (int)c == RETURN);
		f = _NewHandle(x-j+1);
		DEREFH(f)[i] = c;
		for(++i; i < x-j+1 && _ifstream->get(c); i++)
			DEREFH(f)[i] = c;

	} catch(...) {
	    cout << "Unable to read from the topology file. Exiting.\n";
		err = true;
		goto done;
	}
	_HLock((Handle)f); // JLM 8/4/99
	
	MySpinCursor(); // JLM 8/4/99
	if(err = ReadTVertices(f,&line,&pts,&depths,errmsg)) goto done;
	
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		numPts = _GetHandleSize((Handle)pts)/sizeof(LongPoint);
		if(numPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
		}
	}
	MySpinCursor();
	
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	/*if(IsBoundarySegmentHeaderLine(s,&numBoundarySegs)) // Boundary data from CATs
	 {
	 MySpinCursor();
	 if (numBoundarySegs>0)
	 err = ReadBoundarySegs(f,&line,&boundarySegs,numBoundarySegs,errmsg);
	 if(err) goto done;
	 NthLineInTextOptimized(*f, (line)++, s, 1024); 
	 }
	 else
	 {
	 //err = -1;
	 //strcpy(errmsg,"Error in Boundary segment header line");
	 //goto done;
	 // not needed for 2D files, but we require for now
	 }
	 MySpinCursor(); // JLM 8/4/99
	 
	 if(IsWaterBoundaryHeaderLine(s,&numWaterBoundaries,&numBoundaryPts)) // Boundary types from CATs
	 {
	 MySpinCursor();
	 err = ReadWaterBoundaries(f,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
	 if(err) goto done;
	 NthLineInTextOptimized(*f, (line)++, s, 1024); 
	 }
	 else
	 {
	 //err = -1;
	 //strcpy(errmsg,"Error in Water boundaries header line");
	 //goto done;
	 // not needed for 2D files, but we require for now
	 }
	 MySpinCursor(); // JLM 8/4/99
	 //NthLineInTextOptimized(*f, (line)++, s, 1024); 
	 */
	if(IsTTopologyHeaderLine(s,&numTopoPoints)) // Topology from CATs
	{
		MySpinCursor();
		err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numTopoPoints,true);	//AH 03/20/2012
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		err = -1; // for now we require TTopology
		strcpy(errmsg,"Error in topology header line");
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATs
	{
		MySpinCursor();
		err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
		if(err) goto done;
	}
	else
	{
		err = -1; // for now we require TIndexedDagTree
		strcpy(errmsg,"Error in dag tree header line");
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	/////////////////////////////////////////////////
	// if the boundary information is in the file we'll need to create a bathymetry map (required for 3D)
	
	/*if (waterBoundaries && (this -> moverMap == model -> uMap))
	 {
	 //PtCurMap *map = CreateAndInitPtCurMap(fVar.userName,bounds); // the map bounds are the same as the grid bounds
	 PtCurMap *map = CreateAndInitPtCurMap("Extended Topology",bounds); // the map bounds are the same as the grid bounds
	 if (!map) {strcpy(errmsg,"Error creating ptcur map"); goto done;}
	 // maybe move up and have the map read in the boundary information
	 map->SetBoundarySegs(boundarySegs);	
	 map->SetWaterBoundaries(waterBoundaries);
	 
	 *newMap = map;
	 }
	 
	 //if (!(this -> moverMap == model -> uMap))	// maybe assume rectangle grids will have map?
	 else	// maybe assume rectangle grids will have map?
	 {
	 if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
	 if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
	 }*/
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TCATSMover3D::ReadTopology()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(bounds); 
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		printError("Unable to read Extended Topology file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(depths);
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	//depths = 0;
	
done:
	
	if(depths) {_DisposeHandle((Handle)depths); depths=0;}
	if(f) 
	{
		_HUnlock((Handle)f); 
		_DisposeHandle((Handle)f); 
		f = 0;
	}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TCATSMover3D::ReadTopology");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		if(fGrid)
		{
			//fGrid ->Dispose(); AH
			delete fGrid;
			fGrid = 0;
		}
		/*if (*newMap) 
		 {
		 (*newMap)->Dispose();
		 delete *newMap;
		 *newMap=0;
		 }*/
		//if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		//if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
	}
	return err;
}

void CATSMover_c::SetTimeDep(TOSSMTimeValue *newTimeDep)
{
	timeDep = newTimeDep;
}