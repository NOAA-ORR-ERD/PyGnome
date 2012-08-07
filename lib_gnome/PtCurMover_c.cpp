/*
 *  PtCurMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 1/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "PtCurMover_c.h"
#include "MemUtils.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

PtCurMover_c::PtCurMover_c (TMap *owner, char *name) : CurrentMover_c(owner, name), Mover_c(owner, name)
{
	memset(&fVar,0,sizeof(fVar));
	fVar.arrowScale = 5;
	fVar.arrowDepth = 0;
	fVar.alongCurUncertainty = .5;
	fVar.crossCurUncertainty = .25;
	//fVar.uncertMinimumInMPS = .05;
	fVar.uncertMinimumInMPS = 0.0;
	fVar.curScale = 1.0;
	fVar.startTimeInHrs = 0.0;
	fVar.durationInHrs = 24.0;
	fVar.numLandPts = 0; // default that boundary velocities are given
	fVar.maxNumDepths = 1; // 2D default
	fVar.gridType = TWO_D; // 2D default
	fVar.bLayerThickness = 0.; // FREESLIP default
	//
	// Override TCurrentMover defaults
	fDownCurUncertainty = -fVar.alongCurUncertainty; 
	fUpCurUncertainty = fVar.alongCurUncertainty; 	
	fRightCurUncertainty = fVar.crossCurUncertainty;  
	fLeftCurUncertainty = -fVar.crossCurUncertainty; 
	fDuration=fVar.durationInHrs*3600.; //24 hrs as seconds 
	fUncertainStartTime = (long) (fVar.startTimeInHrs*3600.);
	//
	fGrid = 0;
	fTimeDataHdl = 0;
	fIsOptimizedForStep = false;
	fOverLap = false;		// for multiple files case
	fOverLapStartTime = 0;
	
	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	fDepthsH = 0;
	fDepthDataInfo = 0;
	fInputFilesHdl = 0;	// for multiple files case
	
	SetClassName (name); // short file name
}

OSErr PtCurMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
{
	LEUncertainRec unrec;
	double u,v,lengthS,alpha,beta,v0;
	OSErr err = 0;
	
	err = this -> UpdateUncertainty();
	if(err) return err;
	
	if(!fUncertaintyListH || !fLESetSizesH) return 0; // this is our clue to not add uncertainty
	
	
	if(fUncertaintyListH && fLESetSizesH)
	{
		unrec=(*fUncertaintyListH)[(*fLESetSizesH)[setIndex]+leIndex];
		lengthS = sqrt(velocity->u*velocity->u + velocity->v * velocity->v);
		
		
		u = velocity->u;
		v = velocity->v;
		
		if(lengthS < fVar.uncertMinimumInMPS)
		{
			// use a diffusion  ??
			printError("nonzero UNCERTMIN is unimplemented");
			//err = -1;
		}
		else
		{	// normal case, just use cross and down stuff
			alpha = unrec.downStream;
			beta = unrec.crossStream;
			
			velocity->u = u*(1+alpha)+v*beta;
			velocity->v = v*(1+alpha)-u*beta;	
		}
	}
	else 
	{
		TechError("PtCurMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		err = -1;
		velocity->u=velocity->v=0;
	}
	return err;
}


void PtCurMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
}

void PtCurMover_c::DisposeLoadedData(LoadedData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}

void PtCurMover_c::ClearLoadedData(LoadedData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}


long PtCurMover_c::GetNumTimesInFile()
{
	long numTimes = 0;
	
	if (fTimeDataHdl) numTimes = _GetHandleSize((Handle)fTimeDataHdl)/sizeof(**fTimeDataHdl);
	return numTimes;     
}

long PtCurMover_c::GetNumFiles()
{
	long numFiles = 0;
	
	if (fInputFilesHdl) numFiles = _GetHandleSize((Handle)fInputFilesHdl)/sizeof(**fInputFilesHdl);
	return numFiles;     
}

long PtCurMover_c::GetNumDepths(void)
{
	long numDepths = 0;
	if (fDepthsH) numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
	
	return numDepths;
}

LongPointHdl PtCurMover_c::GetPointsHdl()
{
	TTriGridVel* triGrid = dynamic_cast<TTriGridVel*>(fGrid); // don't think need 3D here
	return triGrid -> GetPointsHdl();
}

TopologyHdl PtCurMover_c::GetTopologyHdl()
{
	TTriGridVel* triGrid = dynamic_cast<TTriGridVel*>(fGrid); // don't think need 3D here
	return triGrid -> GetTopologyHdl();
}

long PtCurMover_c::WhatTriAmIIn(WorldPoint wp)
{
	LongPoint lp;
	TTriGridVel* triGrid = dynamic_cast<TTriGridVel*>(fGrid);	// don't think need 3D here
	TDagTree *dagTree = triGrid->GetDagTree();
	lp.h = wp.pLong;
	lp.v = wp.pLat;
	return dagTree -> WhatTriAmIIn(lp);
}



VelocityRec PtCurMover_c::GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty)
{
	VelocityRec v = {0,0};
	printError("PtCurMover::GetScaledPatValue is unimplemented");
	return v;
}


VelocityRec PtCurMover_c::GetPatValue(WorldPoint p)
{
	VelocityRec v = {0,0};
	printError("PtCurMover::GetPatValue is unimplemented");
	return v;
}


void PtCurMover_c::GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2)
{
	long indexToDepthData = (*fDepthDataInfo)[ptIndex].indexToDepthData;
	long numDepths = (*fDepthDataInfo)[ptIndex].numDepths;
	float totalDepth = (*fDepthDataInfo)[ptIndex].totalDepth;
	
	
	switch(fVar.gridType) 
	{
		case TWO_D:	// no depth data
			*depthIndex1 = indexToDepthData;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
		case BAROTROPIC:	// values same throughout column, but limit on total depth
			if (depthAtPoint <= totalDepth)
			{
				*depthIndex1 = indexToDepthData;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			else
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			break;
		case MULTILAYER: //
			//break;
		case SIGMA: // 
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{
				long j;
				for(j=0;j<numDepths-1;j++)
				{
					if(INDEXH(fDepthsH,indexToDepthData+j)<depthAtPoint &&
					   depthAtPoint<=INDEXH(fDepthsH,indexToDepthData+j+1))
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = indexToDepthData+j+1;
					}
					else if(INDEXH(fDepthsH,indexToDepthData+j)==depthAtPoint)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = UNASSIGNEDINDEX;
					}
				}
				if(INDEXH(fDepthsH,indexToDepthData)==depthAtPoint)	// handles single depth case
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX;
				}
				else if(INDEXH(fDepthsH,indexToDepthData+numDepths-1)<depthAtPoint)
				{
					*depthIndex1 = indexToDepthData+numDepths-1;
					*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
				}
				else if(INDEXH(fDepthsH,indexToDepthData)>depthAtPoint)
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX; //TOP, for now just extrapolate highest depth value
				}
			}
			else // no data at this point
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			break;
		default:
			*depthIndex1 = UNASSIGNEDINDEX;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
	}
}

/**************************************************************************************************/
OSErr ScanVelocity (char *startChar, VelocityRec *VelocityPtr, long *scanLength)
{	// expects a number of the form 
	// <number><comma><number>
	//e.g.  "-120.2345,40.345"
	// JLM, 2/26/99 extended to handle
	// <number><whiteSpace><number>
	long	j, k, pairIndex;
	char	num [64];
	OSErr	err = 0;
	char delimiterChar = ',';
	Boolean scientificNotation = false;
	
	j = 0;	/* index into supplied string */
	
	for (pairIndex = 1; pairIndex <= 2 && !err; ++pairIndex)
	{
		/* scan u, then v */
		Boolean keepGoing = true;
		for (k = 0 ; keepGoing; j++)
		{	   			
			switch(startChar[j])
			{
				case ',': // delimiter
				case 0: // end of string
					keepGoing = false;
					break;
				case '.':
					num[k++] = startChar[j];
					break;
				case '+': // number
					if (!scientificNotation) 
					{
						err=-1; 
						return err;
					}
					// else number
				case '-': // number
				case '0': case '1': case '2': case '3': case '4':
				case '5': case '6': case '7': case '8': case '9':
					num[k++] = startChar[j];
					if(k>=32) // no space or comma found to signal end of number
					{
						err = -1;
						return err;
					}
					break;
				case 'e': 
					num[k++] = startChar[j];
					if(k>=32) // no space or comma found to signal end of number
					{
						err = -1;
						return err;
					}
					scientificNotation = true;
					break;
				case ' ':
				case '\t': // white space
					if(k == 0) continue; // ignore leading white space
					while(startChar[j+1] == ' ' || startChar[j+1] == '\t') j++; // move past any additional whitespace chars
					if(startChar[j+1] == ',') j++; // it was <whitespace><comma>, use the comma as the delimiter
					// we have either found a comma or will use the white space as a delimiter
					// in either case we stop this loop
					keepGoing = false;
					break;
				default:
					err = -1;
					return err;
					break;
			}
		}
		
		if(err) break; // so we break out of the main loop, shouldn't happen
		
		num[k++] = 0;									/* terminate the number-string */
		
		if (pairIndex == 1)
		{
			if (!scientificNotation) VelocityPtr -> u = atof(num);
			
			if (startChar[j] == ',')					/* increment j past the comma to next coordinate */
			{
				++j;
				delimiterChar = ','; // JLM reset the delimiter char
			}
		}
		else
		{
			if (!scientificNotation) VelocityPtr -> v = atof(num);
			*scanLength = j; // amount of input string that was read
		}
	}
	///////////////
	
	if (scientificNotation) return -2;
	return err;

}
