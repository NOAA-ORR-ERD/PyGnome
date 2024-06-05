#include "CROSS.H"
#include "MapUtils.h"
#include "Units.h"
/**************************************************************************************************/
OSErr GetScaleAndOffsets (Rect *SourceRectPtr, Rect *DestRectPtr, ScaleRecPtr ScaleInfoPtr)
/* given a source rectangle* and destination rectangle*, this subroutine calculates and stores the
	horizontal and vertical scale and offsets needed to map the sourceRect into the destRect	  */
{
	long	SrcRectDX, SrcRectDY, DestRectDX, DestRectDY;
	
	/* calculate the height and width of both source and destination rectangles */
	SrcRectDX =  ((long) SourceRectPtr -> right  - (long) SourceRectPtr -> left);
	SrcRectDY =  ((long) SourceRectPtr -> bottom - (long) SourceRectPtr -> top);
	DestRectDX =  ((long) DestRectPtr   -> right  - (long) DestRectPtr   -> left);
	DestRectDY =  ((long) DestRectPtr   -> bottom - (long) DestRectPtr   -> top);

	if ((SrcRectDX == 0) || (SrcRectDY == 0))
		return (1);

	ScaleInfoPtr -> XScale = ((double) DestRectDX / (double) SrcRectDX);
	ScaleInfoPtr -> YScale = ((double) DestRectDY / (double) SrcRectDY);

	ScaleInfoPtr -> XOffset = DestRectPtr -> left - SourceRectPtr -> left * ScaleInfoPtr -> XScale;
	ScaleInfoPtr -> YOffset = DestRectPtr -> top  - SourceRectPtr -> top  * ScaleInfoPtr -> YScale;
	
	return (0);
}

/////////////////////////////////////////////////

// JLM 9/7/00 changed from short to a long
//short SameDifference(long A, long B, long V, long a, long b)
long SameDifference(long A, long B, long V, long a, long b)
{
	// float ratio = ((float)V - (float)A) / ((float)B - (float)A);

	// return a + ratio * (b - a);

	long D = B - A, M, d = b - a, m, shift = 0;

	while (V < A) { A -= D; a -= d; }
	while (V > B) { B += D; b += d; }

	if (a < 0) { shift = -a; a = 0; b += shift; }

	for ( ; ; ) {
		M = (A + B) >> 1;
		m = (a + b) >> 1;
		//if (a == m) return _max(-32000, _min(a - shift, 32000));
		//if (b == m) return _max(-32000, _min(b - shift, 32000));
		if (a == m) return a - shift;
		if (b == m) return b - shift;
		if (V > M) { A = M; a = m; }
		else { B = M; b = m; }
	}
}

/////////////////////////////////////////////////
Point GetQuickDrawPt(long pLong, long pLat, Rect *r, Boolean *offPlane)
{
	Point p;
	long x,y;
	//short endOfTheWorld = 32767;  
	short endOfTheWorld = 30000; // we seem to need to use a smaller number on the MAC
	// maybe the system offsets the polygon some in the window ??
	// maybe a global coordinate system
	
	*offPlane = FALSE;
	
	x = SameDifferenceX(pLong);
	y = (r->bottom + r->top) - SameDifferenceY(pLat);
	
	if( x < -endOfTheWorld) 	
	{
		x = -endOfTheWorld;
		if(offPlane) *offPlane = TRUE;
	}
	
	if( x > endOfTheWorld) 	
	{
		x = endOfTheWorld;
		if(offPlane) *offPlane = TRUE;
	}
	
	if( y < -endOfTheWorld) 	
	{
		y = -endOfTheWorld;
		if(offPlane) *offPlane = TRUE;
	}
	
	if( y > endOfTheWorld) 	
	{
		y = endOfTheWorld;
		if(offPlane) *offPlane = TRUE;
	}
	
	p.h = x;
	p.v = y;

	return p;

}
/**************************************************************************************************/
void GetLPtsDist (LongPoint *LPt1, LongPoint *LPt2, long unitCode, double *distance)
{
	double exLat1, exLat2, exLong1, exLong2;

	// convert coordinates to double decimal degrees
	exLat1  = (double) LPt1 -> v / 1000000.0;
	exLat2  = (double) LPt2 -> v / 1000000.0;
	exLong1 = (double) LPt1 -> h / 1000000.0;
	exLong2 = (double) LPt2 -> h / 1000000.0;

	// now convert coordinates to radians
	exLat1 =  exLat1  * PI / 180.0;
	exLat2 =  exLat2  * PI / 180.0;
	exLong1 = exLong1 * PI / 180.0;
	exLong2 = exLong2 * PI / 180.0;

	// compute the distance in matrix coordinates
	*distance = sqrt ((exLat2  - exLat1)  * (exLat2  - exLat1) + cos ((exLat2 + exLat1) / 2) *
				      (exLong2 - exLong1) * (exLong2 - exLong1));

	*distance *= 180.0 / PI;			// convert back to decimal degrees
	
	if (unitCode == kMatrixCode)
		*distance *= 1000000;			// conver to matrix units

	if (unitCode == kMilesCode)
	{
		*distance *= 60.0;				// convert from degrees to nautical miles
		*distance *= 1.15;				// convert from nautical miles to miles
	}
	
	return;
}
/**************************************************************************************************/
long GetScrPtDist (Point scrPoint1, Point scrPoint2)
{
	long	theDistance;
	
	theDistance = sqrt ((double)((scrPoint1.v - scrPoint2.v) * (scrPoint1.v - scrPoint2.v) +
				  	    (scrPoint1.h - scrPoint2.h) * (scrPoint1.h - scrPoint2.h)));

	return theDistance;
}
/**************************************************************************************************/
void SetLPoint (LongPoint *LPtPtr, long h, long v)
{
	LPtPtr -> h = h;
	LPtPtr -> v = v;
	
	return;
}
/**************************************************************************************************/
Boolean LinesCross (LongPointPtr LongPtPtr1, LongPointPtr LongPtPtr2,
					LongPointPtr LongPtPtr3, LongPointPtr LongPtPtr4)
/* subroutine to check if line (x1, y1) to (x2, y2) crosses line (x3, y3) to (x4, y4) */
/* if lines cross, true is returned for function result */
{
	long		vx1, vy1, vx2, vy2;
	double	chk, chk1, chk2;
	long 		x1, y1, x2, y2, x3, y3, x4, y4;
	
	/* convert point type data to individual long type variables */
	x1 = LongPtPtr1 -> h;
	y1 = LongPtPtr1 -> v;
	x2 = LongPtPtr2 -> h;
	y2 = LongPtPtr2 -> v;
	x3 = LongPtPtr3 -> h;
	y3 = LongPtPtr3 -> v;
	x4 = LongPtPtr4 -> h;
	y4 = LongPtPtr4 -> v;
	
	vx1 = x2 - x1;
	vy1 = y2 - y1;
	vx2 = x3 - x1;
	vy2 = y3 - y1;
	chk1 = ((double) vx1 * (double) vy2 - (double) vx2 * (double) vy1);
	vx2 = x4 - x1;
	vy2 = y4 - y1;
	chk2 = ((double) vx1 * (double) vy2 - (double) vx2 * (double) vy1);
	chk = chk1 * chk2;
	
	if (chk >= 0.01)
		return (false);	
		
	vx1 = x4 - x3;
	vy1 = y4 - y3;
	vx2 = x1 - x3;
	vy2 = y1 - y3;
	chk1 = ((double) vx1 * (double) vy2 - (double) vx2 * (double) vy1);
	vx2 = x2 - x3;
	vy2 = y2 - y3;
	chk2 = ((double) vx1 * (double) vy2 - (double) vx2 * (double) vy1);
	chk = chk1 * chk2;
	
	if (chk >= 0.01)
		return (false);

	return (true);	
}
/**************************************************************************************************/
OSErr SelectZoomPoint (Point LocalPt, Rect *WindowRectPtr, Rect* ScrSelRectPtr)
/* subroutine to select the point into which the user wants to zoom in to a map on the screen and */
/* calculate a corresponding rectangle for the zoomed map.										  */
{
	Point		MousePoint;
	Rect		SelRect;
	OSErr		ErrCode = 0;
		
	MousePoint = LocalPt;
	MousePoint.h += 1;
	MousePoint.v += 1;
	
	if (MyPtInRect (MousePoint, WindowRectPtr))
	{
		SelRect.top    = MousePoint.v - (WindowRectPtr -> bottom - WindowRectPtr -> top)  / 4;
		SelRect.bottom = MousePoint.v + (WindowRectPtr -> bottom - WindowRectPtr -> top)  / 4;
		SelRect.left   = MousePoint.h - (WindowRectPtr -> right -  WindowRectPtr -> left) / 4;
		SelRect.right  = MousePoint.h + (WindowRectPtr -> right -  WindowRectPtr -> left) / 4;
		
		/* make sure calculated rect lies within screen rect coordinates */
		NudgeRect (&SelRect, WindowRectPtr);
	
		*ScrSelRectPtr = SelRect;
	}
	else
		ErrCode = 1;		/* bad selection rect */
	
	return (ErrCode);
}
/**************************************************************************************************/
void GetLLString (LongPoint *LPointPtr, long UnitCode, char *LatStr, char *LongStr)
{
	long		NegLatFlag, NegLongFlag;
	double		ExLatDeg, ExLongDeg, ExLatMin, ExLongMin, ExLatSec, ExLongSec;
	long		LatDeg, LongDeg, LatMin, LongMin, LatSec, LongSec;
	LongPoint	LLPoint;
	
	LLPoint = *LPointPtr;

	/* check for negative signs and set flags, then manually convert to
		absolute value (lib fuction was having problems) */
	if (LLPoint.v < 0)
	{
		NegLatFlag = -1;
		LLPoint.v = -LLPoint.v;
	}
	else if (LLPoint.v > 0)
		NegLatFlag = 1;
	else
		NegLatFlag = 0;

	if (LLPoint.h < 0)
	{
		NegLongFlag = -1;
		LLPoint.h = -LLPoint.h;
	}
	else if (LLPoint.h > 0)
		NegLongFlag = 1;
	else
		NegLongFlag = 0;

	/* convert matrix point to decimal degrees */
	ExLatDeg  = LLPoint.v / 1000000.0;
	ExLongDeg = LLPoint.h / 1000000.0;

	/* extract the degrees portion of decimal degrees */
	LatDeg  = ExLatDeg;
	LongDeg = ExLongDeg;

	/* now get decimal minutes by subtracting the int-degrees from decimal degrees */
	ExLatMin  = ExLatDeg  - (long) ExLatDeg;
	ExLongMin = ExLongDeg - (long) ExLongDeg;

	/* convert decimal degrees into decimal minutes */
	ExLatMin  *= 59.9999999;
	ExLongMin *= 59.9999999;

	LatMin  = ExLatMin;
	LongMin = ExLongMin;

	{
		/* now get the decimal secons using minutes */
		ExLatSec  = ExLatMin  - (long) ExLatMin;
		ExLongSec = ExLongMin - (long) ExLongMin;
		ExLatSec  *= 100.0;		/* each second is one/hundredth of a degree for better res */
		ExLongSec *= 100.0;		/* each second is one/hundredth of a degree for better res */
		
		/* convert to integer seconds */
		LatSec  = ExLatSec;
		LongSec = ExLongSec;
	}

//	Debug ("LatDeg = %2.2d, LatMin = %2.2d, LatSec = %2.2d\nExLatMin = %f\n", LatDeg, LatMin, LatSec, ExLatMin);

	if (UnitCode == kDMUnitCode)					/* degrees and decimal minutes requested */
	{
		char	latFormat [64], longFormat [64];
		
		char degChar = GetDegreeChar(GetTextFont()); //JLM
 
		strcpy (latFormat,  "%2.2d%c ");//JLM
		strcpy (longFormat, "%2.2d%c ");
		
		{
			if ((long) (ExLatMin * 3600) != 0)
			{
				if (LatSec == 99 || LatSec == 0)
					strcat (latFormat, "%2.0f' ");
				else
					strcat (latFormat, "%5.2f' ");
			}
			
			if ((long) (ExLongMin * 3600) != 0)
			{
				if (LongSec == 99 || LongSec == 0)
					strcat (longFormat, "%2.0f' ");
				else
					strcat (longFormat, "%5.2f' ");
			}
		}
		
		{		
			if (NegLatFlag < 0)
				strcat (latFormat, "S ");
			else if (NegLatFlag > 0)
				strcat (latFormat, "N ");
			else
				strcat (latFormat, "  ");
			
			if (NegLongFlag < 0)
				strcat (longFormat, "W ");
			else if (NegLongFlag > 0)
				strcat (longFormat, "E ");
			else
				strcat (longFormat, "  ");
		}
		
		sprintf (LatStr,  latFormat,  LatDeg,  degChar, ExLatMin); //JLM
		sprintf (LongStr, longFormat, LongDeg, degChar, ExLongMin);
	}
	else if (UnitCode == kDecDegUnitCode)		/* decimal degrees requested */
	{
		/* we need decimal degrees */
		ExLatDeg = LLPoint.v / 1000000.0;
		ExLongDeg = LLPoint.h / 1000000.0;

		if (NegLatFlag < 0)
			sprintf (LatStr,  "%05.3f' S ", ExLatDeg);
		else if (NegLatFlag > 0)
			sprintf (LatStr,  "%05.3f' N ", ExLatDeg);
		else
			sprintf (LatStr,  "%05.3f'   ", ExLatDeg);
		
		if (NegLongFlag < 0)
			sprintf (LongStr, "%05.3f' W ", ExLongDeg);
		else if (NegLongFlag > 0)
			sprintf (LongStr, "%05.3f' E ", ExLongDeg);
		else
			sprintf (LongStr, "%05.3f'   ", ExLongDeg);
	}
	else if (UnitCode == kDMSUnitCode)					/* degrees, minutes and seconds requested */
	{
	}

	return;
}
/**************************************************************************************************/
void GetLLStrings (LongPoint *LPointPtr, long UnitCode, char *theStrings)
// appends lat-string and long-string together and returns as single string
{
	char	latStr [256], longStr [256];
	
	GetLLString (LPointPtr, UnitCode, latStr, longStr);
	strcpy (theStrings, latStr);
	strcat (theStrings, longStr);
	
	return;
}
/**************************************************************************************************/
void DegreesToDM (double degrees, long *DegreesPtr, double *MinPtr)
{
	
	*DegreesPtr = (long) degrees;
	*MinPtr = (degrees - *DegreesPtr) * 59.9999999;

	return;
}
/**************************************************************************************************/
void RoundDMS (long MatrixDeg, long *LongDeg, double *ExMins)
/* given a matrix coordinate in MatrixDeg, this subroutine breaks it down into integer degrees and
	decimal minutes.  It rounds the minutes to down to two decimal places.  Note the returned
	integer-degrees and minute portions are either both positive or both negative.
*/
{
	double	ExDegrees, remainder, ExDivResult; 
	#ifdef MPW //JLM
		double_t ExOne;
	#else
		double ExOne;
	#endif
		
	long		divResult;
	Boolean		NegFlag = false;
	
	if (MatrixDeg < 0)
	{
		NegFlag = true;
		MatrixDeg = -MatrixDeg;
	}

	ExDegrees = (double) MatrixDeg / 1000000;
	*LongDeg = (long) ExDegrees;
	*ExMins = ExDegrees - (long) ExDegrees;
	*ExMins *= 59.999999999;				/* convert to decimal minutes */
	
	ExOne = 1.00;
	remainder = modf ((double) *ExMins, &ExOne);//JLM
	remainder = 1.0 - remainder;
	divResult = (*ExMins) / 1;
	
	/* extract the decimal portion of the number */
	ExDivResult = (*ExMins) / 1 - divResult;
	
	if (ExDivResult > .50 && ExDivResult < 1.00)
		*ExMins += remainder;
	else if (ExDivResult > 1.00 && ExDivResult < 1.50)
		*ExMins -= remainder;	

	if (*ExMins > 59.99)
	{
		*ExMins = 0.0;
		++(*LongDeg);
	}
	
	if (NegFlag)
	{
		*LongDeg = -(*LongDeg);
		*ExMins = -(*ExMins);
	}

	return;
}
/**************************************************************************************************/
void GetDegMin (long MatrixLong, long *degPtr, double *minPtr)
/* given a matrix value, this subroutine returns integer degrees in *degPtr, and decimal minutes
	in *minPtr */
{
	double	storEx;
	
	storEx = (double) labs (MatrixLong) / 1000000.0;
	*degPtr = (long) storEx;

	storEx  = storEx - (long) storEx;
	*minPtr = storEx * 60;

	return;
}
/**************************************************************************************************/
void PlotVector (long scrX, long scrY, double *vectorUvel, double *vectorVvel, long MilePixels)
{
	Point		HeadPt;
	double		Angle1, Angle2, cs, sn, result, Angle, CAngle, SAngle;
	double		Uvel, Vvel;

	/* move pen to vertex of vector */
	MyMoveTo (scrX, scrY);

	Uvel =   *vectorUvel;
	Vvel = -(*vectorVvel);			/* compensate for top-to-bottom screen pixel order */
	
	if (Uvel == 0)
	{
		if (Vvel == 0)
			Angle = 0;
		else if (Vvel > 0)
			Angle = 90 * PI / 180;
		else if (Vvel < 0)
			Angle = 270 * PI / 180;
	}
	else
	{
		Angle = atan (Vvel / Uvel);
		if (Angle < 0)
			Angle = Angle + 360 * PI / 180;
		if (Uvel < 0)
			Angle = Angle + 180 * PI / 180;
	}

	/* angle has been computed */
//	if (Angle != 0)
	{
		/* compute the resultant of the vector */
		result = (float) MilePixels * sqrt (Uvel * Uvel + Vvel * Vvel);
	
		CAngle = cos (Angle);				/* calculate the horizontal component */
		SAngle = sin (Angle);				/* calculate the vertical 	component */
		
		/* use h & v components to calculate arrow-head location */
		HeadPt.h = CAngle * result + (double) scrX;
		HeadPt.v = SAngle * result + (double) scrY;
		MyLineTo (HeadPt.h, HeadPt.v);		/* line-to the head of the arrow */
		
		/* now draw the arrow head wings */
		Angle1 = Angle - kArrowHang;				
		cs = cos (Angle1);					/* calculate the horizontal component */
		sn = sin (Angle1);					/* calculate the vertical 	component */
		
		scrX = cs * (result * kHeadRatio) + HeadPt.h;
		scrY = sn * (result * kHeadRatio) + HeadPt.v;

		MyMoveTo (HeadPt.h, HeadPt.v);		/* move pen back to head */
		MyLineTo (scrX, scrY);			/* line to the end of wing #1 */
		
		Angle2 = Angle + kArrowHang;
		cs = cos (Angle2);					/* calculate the horizontal component */
		sn = sin (Angle2);					/* calculate the vertical 	component */
		
		scrX = cs * (result * kHeadRatio) + HeadPt.h;
		scrY = sn * (result * kHeadRatio) + HeadPt.v;

		/* move pen back to head origin */
		MyMoveTo (HeadPt.h, HeadPt.v);		/* move pen back to head */
		MyLineTo (scrX, scrY);			/* line to the end of wing #2 */
	}
//	else
//		LineTo (scrX, scrY);

	return;
}
/**************************************************************************************************/
/*OSErr ScanMatrixPt (char *startChar, LongPoint *MatrixLPtPtr)
{	// expects a number of the form 
	// <number><comma><number>
	//e.g.  "-120.2345,40.345"
	// JLM, 2/26/99 extended to handle
	// <number><whiteSpace><number>
	long	deciPlaces, j, k, pairIndex;
	char	num [64];
	OSErr	ErrCode = 0;
	char errStr[256]="";
	char delimiterChar = ',';
	
	j = 0;	// index into supplied string //

	for (pairIndex = 1; pairIndex <= 2 && !ErrCode; ++pairIndex)
	{
	   // first convert the longitude //
	   Boolean keepGoing = true;
	   for (deciPlaces = -1, k = 0 ; keepGoing; j++)
	   {	   			
			switch(startChar[j])
			{
				case ',': // delimiter
				case 0: // end of string
					keepGoing = false;
					break;
				case '.':
					if(deciPlaces != -1)
					{
						strcpy(errStr,"Improper format err: two decimal place chars in same number");
						ErrCode = -4; // to many numbers before the decimal place
						keepGoing = false;
					}
					deciPlaces = 0;		// decimal point encountered 
					break;
				case '-':// number
				case '0': case '1': case '2': case '3': case '4':
				case '5': case '6': case '7': case '8': case '9':
					if (deciPlaces != -1)// if decimal has been found, keep track of places 
						++deciPlaces;
					if (deciPlaces <= kNumDeciPlaces)// don't copy more than 6 decimal place characters 
					{
						num[k++] = startChar[j];
						if(k>=20) 
						{
							if(deciPlaces == -1) 
							{	// still have not found the decimal place
								strcpy(errStr,"Improper format err: too many decimals before decimal place");
								ErrCode = -3; // to many numbers before the decimal place
							}
							keepGoing = false; // stop at 20
						}
					}
					break;
				case ' ':
				case '\t': // white space
					if(k == 0) continue;// ignore leading white space
					while(startChar[j+1] == ' ' || startChar[j+1] == '\t') j++;// movepass any addional whitespace chars
					if(startChar[j+1] == ',') j++; // it was <whitespace><comma>, use the comma as the delimiter
					// we have either found a comma or will use the white space as a delimiter
					// in either case we stop this loop
					keepGoing = false;
					break;
				default:
					strcpy(errStr,"Improper format err: unexpected char");
					ErrCode = -1;
					keepGoing = false;
					break;
			}
		}
		
		if(ErrCode) break; // so we break out of the main loop

		if (deciPlaces < kNumDeciPlaces)
		{
			if (deciPlaces == -1)						// if decimal point was not encountered //
				deciPlaces = 0;

			do
			{
				num[k++] = '0';
				++deciPlaces;
			}
			while (deciPlaces < kNumDeciPlaces);
		}
		
		num[k++] = 0;									// terminate the number-string //
		
		if (pairIndex == 1)
		{
			MatrixLPtPtr -> h = atol(num);
			
			if (startChar[j] == ',')					// increment j past the comma to next coordinate //
			{
				++j;
				delimiterChar = ','; // JLM reset the dilimiter char
			}
		}
		else
			MatrixLPtPtr -> v = atol(num);
	}
	///////////////
	
	if(ErrCode)
	{	//JLM
		char tempStr[68];
		strcat(errStr,NEWLINESTRING);
		strcat(errStr,"Expected  <number><comma or white space><number>");
		strcat(errStr,NEWLINESTRING);
		strcat(errStr,"Offending line:");
		strcat(errStr,NEWLINESTRING);
		// append the first part of the string to the error message
		strncpy(tempStr,startChar,60);
		tempStr[60] = 0;
		if(strlen(tempStr) > 50)
		{
			tempStr[50] = 0;
			strcat(tempStr,"...");
		}
		strcat(errStr,tempStr);
		printError(errStr);
	}

	return (ErrCode);

}*/
/**************************************************************************************************/
void GetLRectViewRatio (LongRect *theGeoRect, long mapProjCode, double *viewRatio)
/* this subroutine calculates both the MapMatrixRatio and MapViewRatio fields in ViewStatusRec.
   Before calling, make sure the Map-Matrix-Rect field has been initialized. */
{
	double	exLat1, exLong1, exLat2, exLong2, exMapWidth, exMapHeight, matrixRatio;

	if (mapProjCode == kLatLongProjCode)
	{
		/* first calculate the XToY ratio of the map matrix bounds rectangle (MapMatrixRatio field) */
		matrixRatio = (double) (labs (theGeoRect -> right  - theGeoRect -> left)) /
					  (double) (labs (theGeoRect -> bottom - theGeoRect -> top));
		
		/* now calculate of the map view rect (MapViewRatio field) using distances */
		exLat1  = (double) theGeoRect -> bottom / 1000000;
		exLat2  = (double) theGeoRect -> top    / 1000000;
		exLong1 = (double) theGeoRect -> left   / 1000000;
		exLong2 = (double) theGeoRect -> right  / 1000000;
		
		/* now convert coordinates to radians */
		exLat1 =  exLat1  * PI / 180;
		exLat2 =  exLat2  * PI / 180;
		exLong1 = exLong1 * PI / 180;
		exLong2 = exLong2 * PI / 180;
		
		/* now calculate the vertical distance of the current map at mid-map longitude */
		exMapHeight = exLat2 - exLat1;
		exMapWidth  = sqrt (cos ((exLat2 + exLat1) / 2.0) * (exLong2 - exLong1) * (exLong2 - exLong1));
	
		*viewRatio = exMapWidth / exMapHeight;
	}
	else
	{
	}

	return;
}
/**************************************************************************************************/
void GetScrMidPoint (Point scrPoint1, Point scrPoint2, Point *theMidPoint)
{
	theMidPoint -> h = (scrPoint1.h + scrPoint2.h) / 2;
	theMidPoint -> v = (scrPoint1.v + scrPoint2.v) / 2;
	
	return;
}

/**************************************************************************************************/
////////////////////////////////////////////////////////////////////////////////

#define MAXLATS 6 // maximum number of latitudes to be drawn on map
#define MAXLONGS 5 // maximum number of longitudes to be drawn on map
#define NUMINCREMENTS 24 // number of lat/long increment values
#define DEGREE 1000000
#define MINUTE (DEGREE / (double)60.0)
#define SECOND (MINUTE / (double)60.0)
#define HUNDREDTH (DEGREE / (double)100.0)
#define THOUSANDTH (HUNDREDTH / (double)10.0)

void DrawLong(long longVal, Rect r, WorldRect view, short precision, Boolean label)
{
	short /*x, y1, y2, */ w;
	Point x_y1,x_y2;
	WorldPoint wp;
	Boolean offQuickDrawPlane;
	char roundLat, roundLong, longString[20], latString[20];
	
	//if (longVal > 180000010 || longVal < -180000000) return;
	//if (longVal > 360000010 || longVal < -180000000) return;
	if (longVal > 360000010 || longVal < -360000000) return;
	//x = SameDifferenceX(longVal);
	view.loLat = _min(view.loLat, 90000000);
	view.loLat = _max(view.loLat, -90000000);
	//y1 = (r.bottom + r.top) - SameDifferenceY(view.loLat);
	x_y1 = GetQuickDrawPt(longVal,view.loLat,&r,&offQuickDrawPlane);
	view.hiLat = _min(view.hiLat, 90000000);
	view.hiLat = _max(view.hiLat, -90000000);
	//y2 = (r.bottom + r.top) - SameDifferenceY(view.hiLat);
	x_y2 = GetQuickDrawPt(longVal,view.hiLat,&r,&offQuickDrawPlane);
	
//	RGBForeColor(&colors[label ? BLACK : LIGHTGRAY]);
	RGBForeColor(&colors[LIGHTGRAY]);
// 	SetPenPat(label ? BLACK : GRAY);
	
//	MyMoveTo(x, y1 - 20);
	//MyMoveTo(x, y1);		// bottom edge of view
	//MyLineTo(x, y2 - 1);	// top edge of view
	MyMoveTo(x_y1.h, x_y1.v);		// bottom edge of view
	MyLineTo(x_y2.h, x_y2.v - 1);	// top edge of view
	
	if (label) {
		RGBForeColor(&colors[BLACK]);
		wp.pLong = longVal;
		wp.pLat = 0;
		WorldPointToStrings2(wp, latString, &roundLat, longString, &roundLong);	
		SimplifyLLString(longString, precision, roundLong);
		w = stringwidth(longString);
		//MyMoveTo(x - (w / 2), y1 - 1);
		MyMoveTo(x_y1.h - (w / 2), x_y1.v - 1);
		drawstring(longString);
	}
}

void DrawLat(long latVal, Rect r, WorldRect view, short precision, Boolean label)
{
	short /*x1, x2, y, */ w;
	Point x_y1, x_y2;
	WorldPoint wp;
	Boolean offQuickDrawPlane;
	char roundLat, roundLong, longString[20], latString[20];
	
	if (latVal > 90000010 || latVal < -90000000) return;
	//view.loLong = _min(view.loLong, 180000000);
	view.loLong = _min(view.loLong, 360000000);
	//view.loLong = _max(view.loLong, -180000000);
	view.loLong = _max(view.loLong, -360000000);
	//x1 = SameDifferenceX(view.loLong);
	x_y1 = GetQuickDrawPt(view.loLong,latVal,&r,&offQuickDrawPlane);
	//view.hiLong = _min(view.hiLong, 180000000);
	view.hiLong = _min(view.hiLong, 360000000);
	//view.hiLong = _max(view.hiLong, -180000000);
	view.hiLong = _max(view.hiLong, -360000000);
	//x2 = SameDifferenceX(view.hiLong);
	x_y2 = GetQuickDrawPt(view.hiLong,latVal,&r,&offQuickDrawPlane);
	//y = (r.bottom + r.top) - SameDifferenceY(latVal);
	
	//if (y > (r.bottom - 30)) return; // don't let low lat line write over long values
	if (x_y1.v > (r.bottom - 30)) return; // don't let low lat line write over long values
	
	wp.pLong = 0;
	wp.pLat = latVal;
	WorldPointToStrings2(wp, latString, &roundLat, longString, &roundLong);	
	SimplifyLLString(latString, precision, roundLat);
	
	w = label ? stringwidth(latString) : 0;
	
	RGBForeColor(&colors[LIGHTGRAY]);
//	SetPenPat(label ? BLACK : GRAY);
	
	//MyMoveTo(x1, y);
	MyMoveTo(x_y1.h, x_y1.v);
//	MyMoveTo(x1 + w + 10, y);
	//MyLineTo(x2, y);
	MyLineTo(x_y2.h, x_y2.v);
	
	if (label) {
		RGBForeColor(&colors[BLACK]);
//		MyMoveTo(x1 + 5, y + 3);
		//MyMoveTo(x1 + 1, y);
		MyMoveTo(x_y1.h + 1, x_y1.v);
		drawstring(latString);
	}
}

void DrawLatLongLines(Rect r, WorldRect view)
{
	Boolean latDivFound = FALSE, longDivFound = FALSE;
	short *precision, *subSteps,
		  latPrecision, longPrecision, latSubSteps = 0, longSubSteps = 0;
	long i, j, latStep, longStep, latRange, longRange, units, space;
	double x, y, properLatDiv, properLongDiv, firstLat, firstLong, *incVals;
	RGBColor saveColor;
	static double incVals1[NUMINCREMENTS] = { // for DEGREES format
					THOUSANDTH / 10, THOUSANDTH / 5, THOUSANDTH / 2, THOUSANDTH, THOUSANDTH * 2, THOUSANDTH * 5,
					HUNDREDTH, HUNDREDTH * 2, HUNDREDTH * 2.5, HUNDREDTH * 5, HUNDREDTH * 10, HUNDREDTH * 25, HUNDREDTH * 50,
					DEGREE, DEGREE * 2, DEGREE * 3, DEGREE * 5, DEGREE * 10, DEGREE * 20, DEGREE * 30,
					DEGREE * 60, DEGREE * 90, DEGREE * 120, DEGREE * 180
				};
	static short precision1[NUMINCREMENTS] = {
					6, 6, 6, 6, 3, 3,
					3, 3, 3, 3, 3, 3, 3,
					0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0
				};
	static short subSteps1[NUMINCREMENTS] = {
					5, 5, 4, 5, 4, 5,
					5, 4, 4, 5, 5, 4, 4,
					5, 4, 6, 5, 5, 4, 6,
					6, 4, 4, 6
				};
	static double incVals2[NUMINCREMENTS] = { // for DEGMIN format
					THOUSANDTH / 10, THOUSANDTH / 5, THOUSANDTH / 2, THOUSANDTH, THOUSANDTH * 2, THOUSANDTH * 5,
					MINUTE, MINUTE * 2, MINUTE * 3, MINUTE * 5, MINUTE * 10, MINUTE * 20, MINUTE * 30,
					DEGREE, DEGREE * 2, DEGREE * 3, DEGREE * 5, DEGREE * 10, DEGREE * 20, DEGREE * 30,
					DEGREE * 60, DEGREE * 90, DEGREE * 120, DEGREE * 180
				};
	static short precision2[NUMINCREMENTS] = {
					6, 6, 6, 6, 6, 6,
					6, 3, 3, 3, 3, 3, 3,
					0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0
				};
	static short subSteps2[NUMINCREMENTS] = {
					5, 5, 4, 5, 4, 5,
					6, 4, 6, 5, 5, 4, 6,
					5, 4, 6, 5, 5, 4, 6,
					6, 4, 4, 6
				};
	static double incVals3[NUMINCREMENTS] = { // for DMS format
					SECOND * 2, SECOND * 2, SECOND * 2, SECOND * 5, SECOND * 15, SECOND * 30,
					MINUTE, MINUTE * 2, MINUTE * 3, MINUTE * 5, MINUTE * 10, MINUTE * 20, MINUTE * 30,
					DEGREE, DEGREE * 2, DEGREE * 3, DEGREE * 5, DEGREE * 10, DEGREE * 20, DEGREE * 30,
					DEGREE * 60, DEGREE * 90, DEGREE * 120, DEGREE * 180
				};
	static short precision3[NUMINCREMENTS] = {
					6, 6, 6, 6, 6, 6,
					3, 3, 3, 3, 3, 3, 3,
					0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0
				};
	static short subSteps3[NUMINCREMENTS] = {
					4, 4, 4, 5, 5, 6,
					6, 4, 6, 5, 5, 4, 6,
					5, 4, 6, 5, 5, 4, 6,
					6, 4, 4, 6
				};
	
	TextFontSize(kFontIDMonaco,9);
	
	GetForeColor(&saveColor);
	PenNormal();
	
	if (settings.customGrid) {
		longDivFound = TRUE;
		properLongDiv = settings.longLabelSpace;
		switch (settings.longLabelUnits) {
			case DEGREES: properLongDiv *= DEGREE; break;
			case MINUTES: properLongDiv *= MINUTE; break;
			case SECONDS: properLongDiv *= SECOND; break;
		}
		space = settings.longLineSpace;
		switch (settings.longLineUnits) {
			case DEGREES: space *= DEGREE; break;
			case MINUTES: space *= MINUTE; break;
			case SECONDS: space *= SECOND; break;
		}
		longSubSteps = properLongDiv / space;
		if (longSubSteps)
			longStep = properLongDiv / longSubSteps;
		switch (settings.longLineUnits) {
			case DEGREES: longPrecision = 0; break;
			case MINUTES: longPrecision = 3; break;
			case SECONDS: longPrecision = 6; break;
		}
		
		latDivFound = TRUE;
		properLatDiv = settings.latLabelSpace;
		switch (settings.latLabelUnits) {
			case DEGREES: properLatDiv *= DEGREE; break;
			case MINUTES: properLatDiv *= MINUTE; break;
			case SECONDS: properLatDiv *= SECOND; break;
		}
		space = settings.latLineSpace;
		switch (settings.latLineUnits) {
			case DEGREES: space *= DEGREE; break;
			case MINUTES: space *= MINUTE; break;
			case SECONDS: space *= SECOND; break;
		}
		latSubSteps = properLatDiv / space;
		if (latSubSteps)
			latStep = properLatDiv / latSubSteps;
		switch (settings.latLineUnits) {
			case DEGREES: latPrecision = 0; break;
			case MINUTES: latPrecision = 3; break;
			case SECONDS: latPrecision = 6; break;
		}
	}
	else {
		switch (settings.latLongFormat) {
			case DEGREES:
				incVals = incVals1;
				precision = precision1;
				subSteps = subSteps1;
				break;
			case DEGMIN:
				incVals = incVals2;
				precision = precision2;
				subSteps = subSteps2;
				break;
			case DMS:
				incVals = incVals3;
				precision = precision3;
				subSteps = subSteps3;
				break;
		}
		
		longRange = WRectWidth(view);
		latRange = WRectHeight(view);
		
		for (i = 0 ; i < NUMINCREMENTS ; i++) {
			if (!longDivFound && (longRange / MAXLONGS) < incVals[i]) {
				longDivFound = TRUE;
				properLongDiv = incVals[i];
				if (settings.showIntermediateLines) {
					longSubSteps = subSteps[i];
					longStep = properLongDiv / longSubSteps;
				}
				longPrecision = precision[i];
			}
			if (!latDivFound && (latRange / MAXLATS) < incVals[i]) {
				latDivFound = TRUE;
				properLatDiv = incVals[i];
				if (settings.showIntermediateLines) {
					latSubSteps = subSteps[i];
					latStep = properLatDiv / latSubSteps;
				}
				latPrecision = precision[i];
			}
			if (latDivFound && longDivFound) break;
		}
	}
	
	if (longDivFound) {
		units = view.loLong / properLongDiv - 1; // - 1 to handle western longitudes
		firstLong = units * properLongDiv;
		// modVal = _max(properLongDiv, MINUTE);
		// firstLong = view.loLong - (view.loLong % (long)modVal);
		for (x = firstLong ; x < view.hiLong ; x += properLongDiv) {
			DrawLong(x, r, view, longPrecision, TRUE);
			for (j = 1 ; j < longSubSteps ; j++)
				DrawLong(x + j * longStep, r, view, 0, FALSE);
		}
	}
	
	if (latDivFound) {
		units = view.loLat / properLatDiv - 1; // - 1 to handle southern longitudes
		firstLat = units * properLatDiv;
		// modVal = _max(properLatDiv, MINUTE);
		// firstLat = view.loLat - (view.loLat % (long)modVal);
		for (y = firstLat ; y < view.hiLat ; y += properLatDiv) {
			DrawLat(y, r, view, latPrecision, TRUE);
			for (j = 1 ; j < latSubSteps ; j++)
				DrawLat(y + j * latStep, r, view, 0, FALSE);
		}
	}
	
	RGBForeColor(&saveColor);
//	TextMode(patOr);
//	SetPenPat(BLACK);
}
/**************************************************************************************************/
