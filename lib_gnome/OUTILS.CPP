
#include "Basics.h"
#include "TypeDefs.h"
#include "MemUtils.h"
#include "StringFunctions.h"
#include "OUtils.h"

void ConvertToUV (double magnitude, double angle, double *u, double *v)
{
	angle += 180; 			// from system -> toward system
	angle *= -1; 			// 0-North system
	angle += 90; 			// -> 0-East system
	angle *= (PI / 180);	// degrees -> radians
	(*u) = cos(angle) * magnitude;
	(*v) = sin(angle) * magnitude;
}

double ConvertToDegrees (char *direction)
{
	StrToUpper(direction);
	RemoveTrailingSpaces(direction);

	// JLM 11/12/99 allow the string to contain a value in digits
	//if(isdigit(direction[0]))
	if (direction[0]>='0' && direction[0]<='9')	// for code warrior
	{
		float val;
		long count = sscanf(direction, "%f", &val);
		if (count == 1) {
			// make sure it is in the range 0-360
			double completeCircle = 360.0;
			long numCompleteCircles = val/completeCircle; // truncate
			val -= numCompleteCircles*completeCircle;
			while (val < 0.0) { val += completeCircle;}
			while (val > completeCircle) { val -= completeCircle;}
			return val;
		}
	}
	else
	{
		if (!strcmp(direction, "N")) return 0;
		if (!strcmp(direction, "E")) return 90;
		if (!strcmp(direction, "S")) return 180;
		if (!strcmp(direction, "W")) return 270;
		
		if (!strcmp(direction, "NE")) return 45;
		if (!strcmp(direction, "SE")) return 135;
		if (!strcmp(direction, "SW")) return 225;
		if (!strcmp(direction, "NW")) return 315;
		
		if (!strcmp(direction, "NNE")) return 22.5;
		if (!strcmp(direction, "ENE")) return 67.5;
		if (!strcmp(direction, "ESE")) return 112.5;
		if (!strcmp(direction, "SSE")) return 157.5;
		if (!strcmp(direction, "SSW")) return 202.5;
		if (!strcmp(direction, "WSW")) return 247.5;
		if (!strcmp(direction, "WNW")) return 292.5;
		if (!strcmp(direction, "NNW")) return 337.5;
	}	
	
	return 0;
}

double ConvertToDegrees (short directionCode)
{
	switch (directionCode)
	{
		case DIR_N: return 0;
		case DIR_E: return 90;
		case DIR_S: return 180;
		case DIR_W: return 270;

		case DIR_NE: return 45;
		case DIR_SE: return 135;
		case DIR_SW: return 225;
		case DIR_NW: return 315;

		case DIR_NNE: return 22.5;
		case DIR_ENE: return 67.5;
		case DIR_ESE: return 112.5;
		case DIR_SSE: return 157.5;
		case DIR_SSW: return 202.5;
		case DIR_WSW: return 247.5;
		case DIR_WNW: return 292.5;
		case DIR_NNW: return 337.5;
	}

	return -1;		// error, no match
}

void ConvertToDirection (double degrees, char *directionStr)
{
	double	minDiff;
	long	dirIndex, bestDir;
	
	DirectionRec directionTable [] =
	{
		  0.0, 	"N",
	 	 90.0, 	"E",
	 	180.0, 	"S",
	 	270.0, 	"W",

	 	 45.0, 	"NE",
	 	135.0, 	"SE",
	 	225.0, 	"SW",
		315.0, 	"NW",

	 	 22.5,	"NNE",
	 	 67.5,	"ENE",
	 	112.5,	"ESE",
	 	157.5,	"SSE",
	 	202.5,	"SSW",
	 	247.5,	"WSW",
	 	292.5,	"WNW",
	 	337.5,	"NNW",
	};
	
	strcpy (directionStr, "NNW");		// default if no better match is found
	bestDir = 15;
	minDiff = directionTable [15].direction;

	for (dirIndex = 0; dirIndex < kNumDirCodes; ++dirIndex)
	{
		if (fabs (directionTable [dirIndex].direction - degrees) < minDiff)
		{
			bestDir = dirIndex;
			minDiff = fabs (directionTable [dirIndex].direction - degrees);
		}
	}
	
	strcpy (directionStr, directionTable [bestDir].dirText);

	return;
}

void ConvertToUnits (long unitCode, char *unitStr)
{
	unitStr [0] = '\0';		// default is blank
	
	switch (unitCode)
	{
		case kKnots:
			strcpy (unitStr, "knots");
			break;
		
		case kMetersPerSec:
			strcpy (unitStr, "meters / sec");
			break;
		case kMilesPerHour:
			strcpy(unitStr,"miles / hour");
			break;
	}

	return;
}

void ConvertToUnitsShort (long unitCode, char *unitStr)
{
	unitStr [0] = '\0';		// default is blank
	
	switch (unitCode)
	{
		case kKnots:
			strcpy (unitStr, "knots");
			break;
		
		case kMetersPerSec:
			strcpy (unitStr, "m/s");
			break;
		case kMilesPerHour:
			strcpy(unitStr,"mph");
			break;
	}

	return;
}

void ConvertToTransportUnits (long unitCode, char *unitStr)
{
	unitStr [0] = '\0';		// default is blank
	
	switch (unitCode)
	{
		case kCMS:
			//strcpy (unitStr,"CMS");
			strcpy (unitStr,"m3/s");
			break;
		case kKCMS:
			//strcpy (unitStr,"KCMS");
			strcpy (unitStr,"k(m3/s)");
			break;
		case kCFS:
			strcpy(unitStr,"CFS");
			break;
		case kKCFS:
			strcpy(unitStr,"KCFS");
			break;
	}

	return;
}


double speedconversion(long speedUnits)
{
	switch(speedUnits)
	{
		case kKnots:
			return KNOTSTOMETERSPERSEC;
		case kMetersPerSec://JLM
			return 1;
		case kMilesPerHour://JLM
			return MILESTOMETERSPERSEC;
		default:
			return -1;
	}
}
