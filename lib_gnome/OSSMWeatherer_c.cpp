/*
 *  OSSMWeatherer_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "OSSMWeatherer_c.h"
#include "MYRANDOM.H"

#ifndef pyGNOME
	#include "CROSS.H"
#else
	#include "Replacements.h"
#endif

OSSMWeatherer_c::OSSMWeatherer_c (char *name): Weatherer_c(name)
{
	return;
}

void OSSMWeatherer_c::WeatherLE (LERec *theLE)
// returns true if LE properties are modified
{
	Boolean			bDecay = false;
	OilComponent	component;
	
	double		tHours, p [2][3], prNum = 0.0, prDen = 0.0, XINT = 0.0, xProb, rNum;
	short		i, ipr = 0;
	
	/// JOLM 1/11/99
	// for some reason, non-weathering oil sometimes weathers one LE
	// Conservative LEs are not supposed to weather , so check that here
	// code goes here, if dispersing naturally, don't want to weather here, check if AdiosDataH exists - outside?
	
	if(theLE -> pollutantType == OIL_CONSERVATIVE || theLE -> dispersionStatus == HAVE_DISPERSED) return;// no weathering
	// what about chemicals with half life?
	
	componentsList -> GetListItem ((Ptr) &component, theLE -> pollutantType - 1);
	//componentsList -> GetListItem ((Ptr) &component, theLE -> pollutantType );
	//	printf ("component.halfLife [0] = %f, [1] = %f, [2] = %f\n", component.halfLife [0], component.halfLife [1], component.halfLife [2]);
	//	printf ("component.percent  [0] = %f, [1] = %f, [2] = %f\n", component.percent [0], component.percent [1], component.percent [2]);
	
	///////////////////////////////
	// JLM 3/11/99 tHours should be the age of the oil in hours 
	// and not the RunDuration  !!  (Gasoline evaporates based on how old it is !!)
	//
	// old code
	//	tHours = (double) model -> GetRunDuration () / 3600.0;		// duration converted from seconds to hours
	//	//	tHours = fmod (tHours, 1.0);
	//
	tHours = (double) ( model -> GetModelTime () - theLE -> releaseTime)/ 3600.0  + theLE -> ageInHrsWhenReleased;
	///////////////////////////////
	
	// temporary fudge, probably won't weather chemicals, just have mass decrease, maybe need a new category for dissolved LEs
	if(theLE -> pollutantType == CHEMICAL)
	{
		double fracLeft = 0.;
		for(i = 0;i<3;i++)
		{
			if(component.percent[i] > 0.0)
			{
				fracLeft +=  (component.percent[i])*pow(0.5,tHours/(component.halfLife[i]));
			}
		}
		fracLeft = _max (0.0,fracLeft);
		fracLeft = _min (1.0,fracLeft);
		//if (fracLeft < 0.001)
		if (fracLeft == 0.0)
			theLE -> statusCode = OILSTAT_EVAPORATED; 
		return;
	}
	
	XINT = (double) model -> GetTimeStep () / 3600.0;			// time step converted from seconds to hours
	
	for (i = 0; i <= 2; ++i)
	{
		if (-component.XK [i] * (tHours + XINT) < -100)
		{
     		p [0][i] = 0.0;
      		p [1][i] = 0.0;
		}
		else
		{
			//			printf ("Exponent computed\n");
     		p [0][i] = exp (-component.XK [i] * tHours);
     		p [1][i] = exp (-component.XK [i] * (tHours + XINT));
		}
		
		//		printf ("p [0][i] = %f, p[1][i] = %f\n", p[0][i], p[1][i]);;
      	prNum += component.percent [i] * (p[0][i] - p[1][i]);
      	prDen += component.percent [i] *  p[0][i];
	}
	
	xProb = prNum / prDen;
	rNum = (double) MyRandom ();
	if (rNum <= xProb && theLE -> mass != 0)
	{
		theLE -> mass = 0;
		theLE -> statusCode = OILSTAT_EVAPORATED; //JLM,10/20/98 
		//		printf ("LE [%d] Evaporated\n", theLE -> leKey);
	}
	else
	{
		xProb = 0;
		ipr = 0;
	}
	
	//	printf ("tHours = %f, prNum = %f, prDen = %f, xProb = %f\n", tHours, prNum, prDen, xProb);

//	while (Button ());

	return;
}
