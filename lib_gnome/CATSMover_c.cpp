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
//#include "OUTILS.H"

#ifndef pyGNOME
#include "CROSS.H"
#include "TCATSMover.h"
#include "TModel.h"
#include "TMap.h"
#include "GridVel.h"
extern TModel *model;
#else
#include "Replacements.h"
#endif

using namespace std;

CATSMover_c::CATSMover_c () : CurrentMover_c() { 
	
	fDuration = 48 * 3600; //48 hrs as seconds
	fTimeUncertaintyWasSet = 0;
	fLESetSizesH = 0;
	fUncertaintyListH = 0;
	fGrid = 0;
	SetTimeDep(0);
	bTimeFileActive = false;
	fEddyDiffusion = 0; // JLM 5/20/991e6; // cm^2/sec
	fEddyV0 = 0.1; // JLM 5/20/99

	bApplyLogProfile = false;

	memset(&fOptimize, 0, sizeof(fOptimize));
}


#ifndef pyGNOME
CATSMover_c::CATSMover_c(TMap *owner, char *name) : CurrentMover_c(owner, name)
{
	fDuration = 48 * 3600; //48 hrs as seconds
	fTimeUncertaintyWasSet = 0;

	fGrid = 0;
	SetTimeDep (0);
	bTimeFileActive = false;
	fEddyDiffusion = 0; // JLM 5/20/991e6; // cm^2/sec
	fEddyV0 = 0.1; // JLM 5/20/99

	bApplyLogProfile = false;

	memset(&fOptimize, 0, sizeof(fOptimize));

	SetClassName(name);
}
#endif


void CATSMover_c::Dispose()
{
	if (fGrid) {
		fGrid->Dispose();
		delete fGrid;
		fGrid = 0;
	}

	//For pyGnome, let python/cython manage memory for this object.	
#ifndef pyGNOME
	DeleteTimeDep ();
#endif

	CurrentMover_c::Dispose ();
}



// this function computes and sets this->refScale
// returns Error when the refScale is not defined
// or in allowable range.
// Note it also sets the refScale to 0 if there is an error
#define MAXREFSCALE  1.0e6  // 1 million times is too much
#define MIN_UNSCALED_REF_LENGTH 1.0E-5 // it's way too small
OSErr CATSMover_c::ComputeVelocityScale(const Seconds& model_time)
{
	double length;
	VelocityRec myVelocity;
	WorldPoint3D refPt3D = { {0, 0}, 0.};

	if (this->timeDep && this->timeDep->fFileType == HYDROLOGYFILE) {
		this->refScale = this->timeDep->fScaleFactor;
		return noErr;
	}

	refPt3D.p = refP;
	refPt3D.z = 0.;

	switch (scaleType) {
		case SCALE_NONE:
			this->refScale = 1;
			return noErr;
		case SCALE_CONSTANT:
			myVelocity = GetPatValue(refPt3D);
			length = sqrt(myVelocity.u * myVelocity.u + myVelocity.v * myVelocity.v);

			/// check for too small lengths
			if (fabs(scaleValue) > length * MAXREFSCALE ||
				length < MIN_UNSCALED_REF_LENGTH)
			{
				// unable to compute refScale
				this->refScale = 0;
				return -1;
			}

			this->refScale = scaleValue / length; 
			return noErr;
#ifndef pyGNOME
		case SCALE_OTHERGRID:
			long i, j, m, n;
			double theirLengthSq, myLengthSq, dotProduct;
			VelocityRec theirVelocity;
			TMap *map;
			TCATSMover *mover;


			for (j = 0, m = model->mapList->GetItemCount(); j < m; j++) {
				model->mapList->GetListItem((Ptr)&map, j);

				for (i = 0, n = map->moverList->GetItemCount(); i < n; i++) {
					map->moverList->GetListItem((Ptr)&mover, i);
					if (mover->GetClassID() != TYPE_CATSMOVER)
						continue;

					if (!strcmp(mover->className, scaleOtherFile)) {
						// JLM, note: we are implicitly matching by file name above
						// JLM: This code left out the possibility of a time file
						// so use GetScaledPatValue() instead
						theirVelocity = mover->GetScaledPatValue(model_time, refPt3D, 0);

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
						if (theirLengthSq > myLengthSq * MAXREFSCALE * MAXREFSCALE ||
							myLengthSq <  MIN_UNSCALED_REF_LENGTH * MIN_UNSCALED_REF_LENGTH)
						{
							// unable to compute refScale
							this->refScale = 0;
							return -1;
						}
						
						dotProduct = myVelocity.u * theirVelocity.u + myVelocity.v * theirVelocity.v;
						this->refScale = sqrt(theirLengthSq / myLengthSq);

						if (dotProduct < 0)
							this->refScale = -(this->refScale);

						return noErr;
					}
				}
			}
			break;
#endif
	}

	this->refScale = 0;
	return -1;
}


/// 5/12/99 only add the eddy uncertainty when told to
OSErr CATSMover_c::AddUncertainty(long setIndex, long leIndex,
								  VelocityRec *patVelocity, double timeStep,
								  Boolean useEddyUncertainty)
{
	OSErr err = 0;

	float rand1, rand2;
	double u, v, lengthS, alpha, beta, v0, gammaScale;
	LEUncertainRec unrec;

	if (!fUncertaintyListH || !fLESetSizesH)
		return 0; // this is our clue to not add uncertainty

	if (useEddyUncertainty) {
		if (this->fOptimize.isFirstStep) {
			GetRandomVectorInUnitCircle(&rand1, &rand2);
		}
		else {
			rand1 = GetRandomFloat(-1.0, 1.0);
			rand2 = GetRandomFloat(-1.0, 1.0);
		}
	}
	else {
		// no need to calculate these when useEddyUncertainty is false
		rand1 = 0;
		rand2 = 0;
	}

	if (fUncertaintyListH && fLESetSizesH) {
		unrec = (*fUncertaintyListH)[(*fLESetSizesH)[setIndex] + leIndex];
		lengthS = sqrt((patVelocity->u * patVelocity->u) +
					   (patVelocity->v * patVelocity->v));

		u = patVelocity->u;
		v = patVelocity->v;

		if (!this->fOptimize.isOptimizedForStep) {
			// - in m/s
			// - note: DIVIDED by timestep because this is later multiplied by the timestep
			this->fOptimize.value = sqrt(6 * (fEddyDiffusion / 10000) / timeStep);
		}
		
		v0 = this->fEddyV0; //meters /second

		if (lengthS > 1e-6) {
			if (useEddyUncertainty)
				gammaScale = this->fOptimize.value * v0 / (lengthS * (v0 + lengthS));
			else  gammaScale = 0.0;
			
			alpha = unrec.downStream + gammaScale * rand1;
			beta = unrec.crossStream + gammaScale * rand2;
			
			patVelocity->u = u * (1 + alpha) + v * beta;
			patVelocity->v = v * (1 + alpha) - u * beta;
		}
		else {
			// when lengthS is too small, ignore the downstream and cross stream and only use diffusion uncertainty
			if (useEddyUncertainty) {
				// provided we are supposed to
				patVelocity->u = this->fOptimize.value * rand1;
				patVelocity->v = this->fOptimize.value * rand2;
			}
		}
	}
	else {
		TechError("TCATSMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		patVelocity->u = patVelocity->v = 0;
	}

	return err;
}


OSErr CATSMover_c::PrepareForModelRun()
{
	this->fOptimize.isFirstStep = true;
	return CurrentMover_c::PrepareForModelRun();
}


OSErr CATSMover_c::PrepareForModelStep(const Seconds &model_time, const Seconds &time_step,
									   bool uncertain, int numLESets, int *LESetsSizesList)
{
	OSErr err = 0;

	err = CurrentMover_c::PrepareForModelStep(model_time, time_step, uncertain, numLESets, LESetsSizesList);
	if (err)
		return err; // note: this calls UpdateUncertainty()

	if (!bActive)
		return noErr;

	err = this->ComputeVelocityScale(model_time);
	if (err) 
		printError("An error occurred in TCATSMover::PrepareForModelStep");

	// in m/s, note: DIVIDED by timestep because this is later multiplied by the timestep
	this->fOptimize.isOptimizedForStep = true;
	this->fOptimize.value = sqrt(6 * (fEddyDiffusion / 10000) / time_step);

	return err;
}


void CATSMover_c::ModelStepIsDone()
{
	this->fOptimize.isFirstStep = false;
	memset(&fOptimize, 0, sizeof(fOptimize));
	bIsFirstStep = false;
}


OSErr CATSMover_c::get_move(int n, Seconds model_time, Seconds step_len,
							WorldPoint3D *ref, WorldPoint3D *delta, short *LE_status,
							LEType spillType, long spill_ID)
{
	if(!delta || !ref) {
		return 1;
	}

	// For LEType spillType, check to make sure it is within the valid values
	if (spillType < FORECAST_LE ||
		spillType > UNCERTAINTY_LE)
	{
		return 2;
	}

	LERec* prec;
	LERec rec;
	prec = &rec;

	WorldPoint3D zero_delta = { {0, 0}, 0.};

	for (int i = 0; i < n; i++) {
		if ( LE_status[i] != OILSTAT_INWATER) {
			delta[i] = zero_delta;
			continue;
		}

		rec.p = ref[i].p;
		rec.z = ref[i].z;

		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1e6;
		rec.p.pLong *= 1e6;

		delta[i] = GetMove(model_time, step_len, spill_ID, i, prec, spillType);

		delta[i].p.pLat /= 1e6;
		delta[i].p.pLong /= 1e6;
	}

	return noErr;
}


WorldPoint3D CATSMover_c::GetMove(const Seconds &model_time, Seconds timeStep,
								  long setIndex, long leIndex, LERec *theLE, LETYPE leType)
{
	Boolean useEddyUncertainty = false;	
	double dLong, dLat;

	WorldPoint3D deltaPoint = { {0, 0}, 0.};
	WorldPoint3D refPoint3D = { {0, 0}, 0.};
	VelocityRec scaledPatVelocity;

	refPoint3D.p = (*theLE).p;
	refPoint3D.z = (*theLE).z;

	scaledPatVelocity = this->GetScaledPatValue(model_time, refPoint3D, &useEddyUncertainty);

	if (leType == UNCERTAINTY_LE) {
		AddUncertainty(setIndex, leIndex, &scaledPatVelocity, timeStep, useEddyUncertainty);
	}

	dLong = ((scaledPatVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3(refPoint3D.p.pLat);
	dLat  =  (scaledPatVelocity.v / METERSPERDEGREELAT) * timeStep;

	deltaPoint.p.pLong = dLong * 1e6;
	deltaPoint.p.pLat  = dLat  * 1e6;

	return deltaPoint;
}


/// 5/12/99 JLM, we only add the eddy uncertainty when the vectors are big enough when the timeValue is 1
// This is in response to the Prince William sound problem where 5 patterns are being added together
VelocityRec CATSMover_c::GetScaledPatValue(const Seconds &model_time,
										   WorldPoint3D p, Boolean *useEddyUncertainty)
{
	VelocityRec	patVelocity, timeValue = {1, 1};
	float lengthSquaredBeforeTimeFactor;
	OSErr err = 0;

	if (!this->fOptimize.isOptimizedForStep && this->scaleType == SCALE_OTHERGRID) {
		// we need to update refScale
		this->ComputeVelocityScale(model_time);
	}
	
	// get and apply our time file scale factor
	if (timeDep && bTimeFileActive) {
		// VelocityRec errVelocity={1,1};
		// JLM 11/22/99, if there are no time file values, use zero not 1
		VelocityRec errVelocity = {0, 1};

		err = timeDep->GetTimeValue(model_time, &timeValue); // AH 07/10/2012
		if (err)
			timeValue = errVelocity;
	}

	patVelocity = GetPatValue(p);
	patVelocity.u *= refScale; 
	patVelocity.v *= refScale; 

	if (useEddyUncertainty) {
		// if they gave us a pointer to a boolean fill it in, otherwise don't
		lengthSquaredBeforeTimeFactor = patVelocity.u * patVelocity.u + patVelocity.v * patVelocity.v;
		if (lengthSquaredBeforeTimeFactor < (this -> fEddyV0 * this -> fEddyV0))
			*useEddyUncertainty = false;
		else
			*useEddyUncertainty = true;
	}

	patVelocity.u *= timeValue.u; // magnitude contained in u field only
	patVelocity.v *= timeValue.u; // magnitude contained in u field only

	return patVelocity;
}


VelocityRec CATSMover_c::GetPatValue(WorldPoint3D p)
{
	double depthAtPoint = 0., scaleFactor = 1.;
	VelocityRec patVal = {0., 0.};

	if (p.z > 0 && !bApplyLogProfile) {
#ifndef pyGNOME
		//if (!IAm(TYPE_CATSMOVER3D))
			if (!(dynamic_cast<TCATSMover*>(this)->IAm(TYPE_CATSMOVER3D)))
			return patVal;
#else
		return patVal;
#endif
	}

	if (p.z > 1 && bApplyLogProfile) {
		// start the profile after the first meter
		depthAtPoint = fGrid->GetDepthAtPoint(p.p);

		if (p.z >= depthAtPoint)
			scaleFactor = 0.;
		else if (depthAtPoint > 0)
			scaleFactor = 1. - log(p.z) / log(depthAtPoint);
	}

	patVal = fGrid->GetPatValue(p.p);
	patVal.u *= scaleFactor;
	patVal.v *= scaleFactor;

	return patVal;
}


VelocityRec CATSMover_c::GetSmoothVelocity (WorldPoint p)
{
	return fGrid->GetSmoothVelocity(p);
}


Boolean CATSMover_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32], sStr[32];
	double lengthU, lengthS;
	VelocityRec velocity = {0., 0.};

	wp.z = arrowDepth;
	velocity = this->GetPatValue(wp);

	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->refScale * lengthU;

	StringWithoutTrailingZeros(uStr, lengthU, 4);
	StringWithoutTrailingZeros(sStr, lengthS, 4);

	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
			this->className, uStr, sStr);

	return true;
}


void CATSMover_c::DeleteTimeDep () 
{
	if (timeDep) {
		timeDep -> Dispose ();
		delete timeDep;
		timeDep = 0;
	}

	return;
}


// import PtCur triangle info so don't have to regenerate
OSErr CATSMover_c::TextRead(vector<string> &linesInFile)
{
	OSErr err = 0;
	char errmsg[256];
	string currentLine;
	long line = 1;

	long numPoints, numTopoPoints, numPts;

	TopologyHdl topo = 0;
	LongPointHdl pts = 0;
	FLOATH depths = 0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds = voidWorldRect;

	TTriGridVel *triGrid = 0;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;

	errmsg[0] = 0;

	MySpinCursor();

	err = ReadTVertices(linesInFile, &line, &pts, &depths, errmsg);
	if (err) {
		cerr << "failed to read Vertices..." << endl;
		goto done;
	}

	if (pts) {
		LongPoint thisLPoint;

		numPts = _GetHandleSize((Handle)pts) / sizeof(LongPoint);
		if (numPts > 0) {
			WorldPoint wp;
			for (long i = 0; i < numPts; i++) {
				thisLPoint = INDEXH(pts, i);

				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;

				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
		}
	}

	MySpinCursor();

	currentLine = linesInFile[(line)++];

	if (IsTTopologyHeaderLine(currentLine, numTopoPoints)) {
		// Topology from CATs
		MySpinCursor();

		err = ReadTTopologyBody(linesInFile, &line, &topo, &velH, errmsg, numTopoPoints, true);
		if (err)
			goto done;

		currentLine = linesInFile[(line)++];
	}
	else {
		err = -1; // for now we require TTopology
		strcpy(errmsg, "Error in topology header line");
		if (err)
			goto done;
	}

	MySpinCursor(); // JLM 8/4/99

	if (IsTIndexedDagTreeHeaderLine(currentLine, numPoints)) {
		// DagTree from CATs
		MySpinCursor();

		err = ReadTIndexedDagTreeBody(linesInFile, &line, &tree, errmsg, numPoints);
		if (err)
			goto done;
	}
	else {
		err = -1; // for now we require TIndexedDagTree
		strcpy(errmsg, "Error in dag tree header line");
		if (err)
			goto done;
	}

	MySpinCursor(); // JLM 8/4/99

	triGrid = new TTriGridVel;
	if (!triGrid) {
		err = true;
		TechError("Error in TCATSMover3D::ReadTopology()", "new TTriGridVel", err);
		goto done;
	}

	fGrid = (TTriGridVel*)triGrid;

	triGrid->SetBounds(bounds);
	dagTree = new TDagTree(pts, topo, tree.treeHdl, velH, tree.numBranches);
	if (!dagTree) {
		printError("Unable to read Extended Topology file.");
		goto done;
	}

	triGrid->SetDagTree(dagTree);

	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it

done:

	if (depths) {
		_DisposeHandle((Handle)depths);
		depths = 0;
	}

	if (err) {
		if (!errmsg[0])
			strcpy(errmsg, "An error occurred in TCATSMover3D::ReadTopology");
		printError(errmsg);
		if (pts) {
			DisposeHandle((Handle)pts);
			pts = 0;
		}
		if (topo) {
			DisposeHandle((Handle)topo);
			topo = 0;
		}
		if (velH) {
			DisposeHandle((Handle)velH);
			velH = 0;
		}
		if (tree.treeHdl) {
			DisposeHandle((Handle)tree.treeHdl);
			tree.treeHdl = 0;
		}
		if (depths) {
			DisposeHandle((Handle)depths);
			depths = 0;
		}
		if (fGrid) {
			delete fGrid;
			fGrid = 0;
		}
	}

	return err;
}


// import PtCur triangle info so don't have to regenerate
OSErr CATSMover_c::TextRead(char *path)
{
	string strPath = path;
	if (strPath.size() == 0)
		return 0;

	vector<string> linesInFile;
	if (ReadLinesInFile(strPath, linesInFile)) {
		return TextRead(linesInFile);
	}
	else {
		return -1;
	}
}


void CATSMover_c::SetTimeDep(TOSSMTimeValue *newTimeDep)
{
	timeDep = newTimeDep;
}
