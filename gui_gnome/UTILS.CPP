#include <assert.h>
#include "CROSS.H"


void MySpinCursor(void)
{
#ifdef NO_GUI
	return;
#endif
	static unsigned long sTickCountAtLastSpin = 0;
	static short ballNumber = 0;
	unsigned long nowTicks = TickCount();
	const short kSpinDelayInTicks = 20;
	if(sTickCountAtLastSpin <= nowTicks  && nowTicks <= sTickCountAtLastSpin + kSpinDelayInTicks)
	{
		// skip the spin
	}
	else	
	{
		ballNumber = (ballNumber+1)%4;
#ifdef MAC
		{
			static CursHandle ball[4];
			if(!ball[ballNumber]) ball[ballNumber] = GetCursor(11000+ballNumber);
			if(!ball[ballNumber]) ball[ballNumber] = GetCursor(watchCursor); // couldn't find the ball cursor, use watch
			if(ball[ballNumber]) SetCursor(*(ball[ballNumber]));
		}
#else //IBM
		{
			HCURSOR  cursHdl =NULL;
			switch(ballNumber)
			{
				case 1: cursHdl = LoadCursor(hInst,beachb1CURSOR);break;
				case 2: cursHdl = LoadCursor(hInst,beachb2CURSOR);break;
				case 3: cursHdl = LoadCursor(hInst,beachb3CURSOR);break;
				default: cursHdl = LoadCursor(hInst,beachb4CURSOR); break;
			}
			if(cursHdl) SetCursor(cursHdl);
		}
#endif
		sTickCountAtLastSpin = nowTicks;
	}
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

#ifdef MAC
RGBColor backColors[] = {
	//red		green		blue
	{ 0,		0,			0 },		// dummy
	{ 65535, 	65535, 		65535 },	// white
	{ 0, 		0, 			0 },		// black
	{ 42831, 	43263, 		43263 },	// grey
	{ 14680,	54114,		56831 },	// light blue
	{ 11487,	44543,		8268 }		// green
};

#else

RGBColor backColors[] = {
	//red		green		blue
	RGB(0,		0,			0),		// dummy
	RGB(255,	255,		255),	// white
	RGB(0,		0,			0),		// black
	RGB(192,	192,		192),	// grey
	RGB(0,		255,		255),	// light blue
	RGB(1,		255,		1)		// green
};
#endif



void SetPenPat(short pat)
{
#ifdef MAC
	Pattern p;
	
	GetIndPattern(&p, PATTERNS, pat);
	PenPat(&p);
#else
	FillPat(pat);
	PenStyle(pat, 1);
#endif
}


void SimplifyLLString(char *s, short precision, char rounder)
{
	char dc, mc, sc, dir, holder[] = "000000";
	long deg, min, sec;
#pragma unused(rounder)
	
	while (s[0] == ' ') s++;
	
	switch (settings.latLongFormat) {
		case DEGREES:
			sscanf(s, "%ld.%ld%c%c", &deg, &min, &dc, &dir);
			switch (precision) {
				case 0:
					// X23.456789dN -> X23dN
					if (min >= 500000) deg++;
					sprintf(s, "%ld%c%c", deg, dc, dir);
					break;
				case 3:
					// X23.456789dN -> X23.457dN
					if ((min % 1000) >= 500) {
						min += 1000;
						if (min >= 1000000) {
							deg++;
							min = 0;
						}
					}
					if (min >= 100000) holder[0] = 0;
					else if (min >= 10000) holder[1] = 0;
					else if (min >= 1000) holder[2] = 0;
					else holder[0] = 0;
					min /= 1000;
					sprintf(s, "%ld.%s%ld%c%c", deg, holder, min, dc, dir);
					break;
				case 6: // even though it says 6, we'll round to 4 places,
					// since that's the most needed by MARPLOT
					// X23.456789dN -> X23.4568dN
					if ((min % 100) >= 50) {
						min += 100;
						if (min >= 1000000) {
							deg++;
							min = 0;
						}
					}
					if (min >= 100000) holder[0] = 0;
					else if (min >= 10000) holder[1] = 0;
					else if (min >= 1000) holder[2] = 0;
					else holder[0] = 0;
					min /= 100;
					sprintf(s, "%ld.%s%ld%c%c", deg, holder, min, dc, dir);
					break;
			}
			break;
		case DEGMIN:
			sscanf(s, "%ld%c%ld.%ld%c%c", &deg, &dc, &min, &sec, &mc, &dir);
			switch (precision) {
				case 0:
					// X23d45.67mN -> X24dN
					if (min >= 30) deg++;
					sprintf(s, "%ld%c%c", deg, dc, dir);
					break;
				case 3:
					// X23d45.67mN -> X23d46mN
					if (sec >= 50) {
						min++;
						if (min >= 60) {
							deg++;
							min = 0;
						}
					}
					sprintf(s, "%ld%c%ld%c%c", deg, dc, min, mc, dir);
					break;
				case 6:
					/*
					 // X23d45.67mN -> X23d45.7mN
					 if ((sec % 10) >= 5) {
					 sec += 10;
					 if (sec >= 100) {
					 sec = 0;
					 min++;
					 if (min >= 60) {
					 deg++;
					 min = 0;
					 }
					 }
					 }
					 sec /= 10;
					 sprintf(s, "%ld%c%ld.%ld%c%c", deg, dc, min, sec, mc, dir);
					 */
					break;
			}
			break;
		case DMS:
			sscanf(s, "%ld%c%ld%c%ld%c%c", &deg, &dc, &min, &mc, &sec, &sc, &dir);
			switch (precision) {
				case 0:
					// X23d45m67sN -> X24dN
					if (min >= 30) deg++;
					sprintf(s, "%ld%c%c", deg, dc, dir);
					break;
				case 3:
					// X23d45m67sN -> X23d46mN
					if (sec >= 30) {
						min++;
						if (min >= 60) {
							deg++;
							min = 0;
						}
					}
					sprintf(s, "%ld%c%ld%c%c", deg, dc, min, mc, dir);
					break;
				case 6:
					if (sec == 59) { // && rounder == 9 round up to correct precision errors
						min ++;
						if (min >= 60) {
							deg++;
							min = 0;
						}
						sec = 0;
					}
					sprintf(s, "%ld%c%ld%c%ld%c%c", deg, dc, min, mc, sec, sc, dir);
					break;
			}
			break;
	}
}


short RectLeft(Rect r)
{
	return r.left;
}
/**************************************************************************************************/
unsigned char Pack2(char value1, char value2)
{
	return (value1 << 4) + value2;
}

void Unpack2(unsigned char packed, char *value1, char *value2)
{
	*value1 = packed >> 4;
	*value2 = packed & 0xF;
}

short CountBitsInShort(short s)
{
	short count = 0;
	
	s &= 0x7FFF; // don't count bit 15, the "invisible" flag
	
	for ( ; s ; s = s << 1)
		if (s & 0x8000)
			count++;
	
	return count;
}

static char sharedLatRound, sharedLongRound;

WorldPoint RoundWorldPoint(WorldPoint wp) // JLM 12/9/98
{ // this rounds the world point to the precision shown by the mousemeter
	float latVal, longVal, minF1, minF2, sec1=0.0, sec2 = 0.0;
	double val;
	long deg1, deg2, min1, min2;
	char dLat, dLong;

	switch (settings.latLongFormat) 
	{
		case DEGREES:
			//WorldPointToDoubles(wp, &latVal, &longVal, &dLat, &dLong,6);
			// nothing to do
			return wp;
		case DEGMIN:
			// 
			WorldPointToDegMin(wp, &deg1, &minF1, &deg2, &minF2, &dLat, &dLong,2);
			val = 1000000*(deg1 + minF1/60.0);
			wp.pLat = (wp.pLat < 0) ? -val: val;
			val = 1000000*(deg2 + minF2/60.0);
			wp.pLong = (wp.pLong < 0) ? -val: val;
			break;
		case DMS:
			WorldPointToDMS(wp, &deg1, &min1, &sec1, &deg2, &min2, &sec2, &dLat, &dLong,0);
			val = 1000000*(deg1 + min1/60.0 + sec1/3600.0);
			wp.pLat = (wp.pLat < 0) ? -val: val;
			val = 1000000*(deg2 + min2/60.0 + sec2/3600.0);
			wp.pLong = (wp.pLong < 0) ? -val: val;
			break;
	}
	return wp;
}



void WorldPointToStrings(WorldPoint wp, char *latString, char *longString)
{
#ifdef MAC
	static char format1lat[]  = "%9.6lf%c%c", 
				format1long[] = "%10.6lf%c%c",
				format2lat[]  = "%2ld%c%05.2f'%c",
				format2long[] = "%3ld%c%05.2f'%c",
				format3lat[]  = "%2ld%c%02ld'%02.0f%c%c",
				format3long[] = "%3ld%c%02ld'%02.0f%c%c";
#else 
	static char format1lat[]  = "%9.6f%c%c", 
				format1long[] = "%10.6f%c%c",
				format2lat[]  = "%2ld%c%05.2f\x92%c", // x92 = smart close single quote
				format2long[] = "%3ld%c%05.2f\x92%c",
				format3lat[]  = "%2ld%c%02ld\x92%02.0f%c%c", // x94 = smart close double quote depends on the font
				format3long[] = "%3ld%c%02ld\x92%02.0f%c%c";
#endif
	float latVal, longVal, minF1, minF2, sec1, sec2; // JLM 8/31/99
	long deg1, deg2, min1, min2;
	char dLat, dLong;
	char degChar = GetDegreeChar(GetTextFont());
	char dblQuoteChar = GetSmartDoubleQuoteChar(GetTextFont());
	
	switch (settings.latLongFormat) {
		case DEGREES:
		{
			double dLatVal,dLongVal;
			char latStr[32],longStr[32];
			WorldPointToDoubles(wp, &dLatVal, &dLongVal, &dLat, &dLong,6);
			sprintf(latString, lfFix(format1lat), dLatVal, degChar, dLat);
			sprintf(longString, lfFix(format1long), dLongVal, degChar, dLong);
			break;
		}
		case DEGMIN:
			WorldPointToDegMin(wp, &deg1, &minF1, &deg2, &minF2, &dLat, &dLong,2);
			sprintf(latString, format2lat, deg1, degChar, minF1, dLat);
			sprintf(longString, format2long, deg2, degChar, minF2, dLong);
			break;
		case DMS:
			WorldPointToDMS(wp, &deg1, &min1, &sec1, &deg2, &min2, &sec2, &dLat, &dLong,0);
			//if (sec1 > 59) {
			//	if (sec1 == 59.99) sharedLatRound = 9;
			//	sec1 = 59;
			//}
			//if (sec2 > 59) {
			//	if (sec2 == 59.99) sharedLongRound = 9;
			//	sec2 = 59;
			//}
			sprintf(latString, format3lat, deg1, degChar, min1, sec1, dblQuoteChar, dLat);
			sprintf(longString, format3long, deg2, degChar, min2, sec2, dblQuoteChar, dLong);
			break;
	}
}

void WorldPointToStrings2(WorldPoint wp, char *latString, char *latRound,
										 char *longString, char *longRound)
{
	sharedLatRound = sharedLongRound = 0;
	WorldPointToStrings(wp, latString, longString);
	*latRound = sharedLatRound;
	*longRound = sharedLongRound;
}

long ScreenToWorldDistance(short pixels)
{
	return CurrentScaleDenominator() * PixelsPerMilDeg() * pixels;
}

long ScaleToRectHeight(long scale, WorldRect wr)
{
	return WRectHeight(wr) / (PixelsPerMilDeg() * scale);
}

long ScaleDenominator(WorldRect wr, Rect r)
{
	return WRectHeight(wr) / (PixelsPerMilDeg() * RectHeight(r));
}

char *GetScaleString(char *s, short scaleFormat, long scale)
{
	char sLabel[20], sValue1[20], sValue2[20], sUnits[20];
	short saveFormat = 0; // settings.scaleFormat;
	
	// settings.scaleFormat = scaleFormat;
	switch (scaleFormat) {
		case RATIO: sprintf(s, "1:%ld", scale); break;
		case EQUALITY:
			GetEquality(scale, sLabel, sValue1, sUnits);
			sprintf(s, "%s %s %s", sLabel, sValue1, sUnits);
			break;
		case WINDOWWIDTH:
			GetWindowSize(scale, sValue1, sValue2, sUnits);
			sprintf(s, "%s x %s %s", sValue1, sValue2, sUnits);
			break;
	}
	// settings.scaleFormat = saveFormat;
	
	return s;
}

void GetUnitString(short format, char *s)
{
	s[0] = 0;
	
	if (format == MASSONLY || format == MASSPERAREA)
		switch (settings.massUnits) {
			case KILOGRAMS: strcat(s, "kg"); break;
			case METRICTONS: strcat(s, "m ton"); break;
			case SHORTTONS: strcat(s, "s ton"); break;
			case GALLONS: strcat(s, "gal"); break;
			case BARRELS: strcat(s, "bbl"); break;
			case CUBICMETERS: strcat(s, "m**3"); break;
		}
	
	if (format == MASSPERAREA) strcat(s, "/");
	
	if (format == AREAONLY || format == MASSPERAREA)
		switch (settings.areaUnits) {
			case SQUAREMILES: strcat(s, "mi**2"); break;
			case SQUARENAUTICALMILES: strcat(s, "nm**2"); break;
			case SQUAREKILOMETERS: strcat(s, "km**2"); break;
			case ACRES: strcat(s, "acre"); break;
			case SQUAREMETERS: strcat(s, "m**2"); break;
		}
}



float ConvertValue(short format, float mass, float area, float density)
{
	float value, rho;
	
	// mass is assumed to be in kg, area is assumed to be in km2
	
	rho = 141.5 / (density + 131.5);
	
	// LATER: should check that rhoO < 1
	
	rho *= 1000.0;
	
	switch (settings.massUnits) {
		case KILOGRAMS: break;
		case METRICTONS: mass *= 0.001; break;
		case SHORTTONS: mass *= 0.001 * 1.10; break;
		case GALLONS:
		case BARRELS:
			mass /= rho; // kg -> cubic meters
			mass /= 0.003785; // cubic meters -> gallons
			if (settings.massUnits == BARRELS)
				mass /= 42; // gallons -> barrels
			break;
		case CUBICMETERS: mass /= rho; break; // kg -> cubic meters
	}
	
	switch (settings.areaUnits) {
		case SQUAREMILES: area *= 0.386101; break;
		case SQUARENAUTICALMILES: area *= 0.291180; break;
		case SQUAREKILOMETERS: break;
		case ACRES: area *= 247.104343; break;
		case SQUAREMETERS: area *= 1000000; break;
	}
	
	switch (format) {
		case MASSONLY: value = mass; break;
		case AREAONLY: value = area; break;
		case MASSPERAREA: value = mass / area; break;
	}
	
	if (value <= 0) value = 0; // avoid "-0" problems
	
	return value;
}

void GetConvertedValueString(short format, float mass, float area, float density, char *s)
{
	char formatS[20];
	short d = 6;
	float value;
	
	// mass is assumed to be in kg, area is assumed to be in km2
	
	s[0] = 0;
	
	if (settings.areaUnits == SQUAREMETERS && format == AREAONLY) d = 0;
	
	value = ConvertValue(format, mass, area, density);
	
	// JLM use six significant digits
	if(value > 1e6) d = 0;
	else if (value >1e5) d=1;
	else if (value >1e4) d=2;
	else if (value >1e3) d=3;
	else if (value >1e2) d=4;
	else if (value >1e1) d=5;
	
	sprintf(formatS, "%%.%hdf", d);
	sprintf(s, formatS, value);
	ChopEndZeros(s);//JLM 6/12/95
	strcat(s, " ");
	GetUnitString(format, &s[strlen(s)]);
}


float NormalizeValue(short format, float value, float density)
{
	float rho;
	
	// convert from the current display units to kg and km2
	
	rho = 141.5 / (density + 131.5);
	
	// LATER: should check that rhoO < 1
	
	rho *= 1000.0;
	
	if (format == MASSONLY || format == MASSPERAREA)
		switch (settings.massUnits) {
			case KILOGRAMS: break;
			case METRICTONS: value /= 0.001; break;
			case SHORTTONS: value /= 0.001 * 0.5 * 2.20; break;
			case GALLONS:
			case BARRELS:
				if (settings.massUnits == BARRELS)
					value *= 42; // barrels -> gallons
				value *= 0.003785; // gallons -> cubic meters
				value *= rho; // cubic meters -> kg
				break;
			case CUBICMETERS: value *= rho; break; // cubic meters -> kg
		}
	
	if (format == AREAONLY)
		switch (settings.areaUnits) {
			case SQUAREMILES: value /= 0.386101; break;
			case SQUARENAUTICALMILES: value /= 0.291180; break;
			case SQUAREKILOMETERS: break;
			case ACRES: value /= 247.104343; break;
			case SQUAREMETERS: value /= 1000000; break;
		}
	
	if (format == MASSPERAREA)
		switch (settings.areaUnits) {
			case SQUAREMILES: value *= 0.386101; break;
			case SQUARENAUTICALMILES: value *= 0.291180; break;
			case SQUAREKILOMETERS: break;
			case ACRES: value *= 247.104343; break;
			case SQUAREMETERS: value *= 1000000; break;
		}
	
	return value;
}

void SetUndoNothing()
{
	settings.undoCode = UNDO_NOTHING;
}

void MarchBox(Rect antBox, short pNum, Rect insetBox)
{
#ifdef MAC
	short width = 2;
	Pattern	pat;
#else
	short clippedLeft, clippedTop, clippedRight, clippedBottom, width = 2;
	Rect r;
#endif
	
	// this is because rects SURROUND their pixels
	antBox.right--;
	antBox.bottom--;
	insetBox.right--;
	insetBox.bottom--;
	
	if (RectWidth(antBox) <= 4 && RectHeight(antBox) <= 4) {
#ifdef MAC
		PenSize(width, width);
		GetIndPattern(&pat, HORIZANTS, pNum);
		PenPat(&pat);
		MyMoveTo(insetBox.left, (antBox.top + antBox.bottom) / 2);
		MyLineTo(insetBox.right, (antBox.top + antBox.bottom) / 2);
		GetIndPattern(&pat, VERTANTS, pNum);
		PenPat(&pat);
		MyMoveTo((antBox.left + antBox.right) / 2, insetBox.top);
		MyLineTo((antBox.left + antBox.right) / 2, insetBox.bottom);
#else
		MySetRect(&r, insetBox.left, (antBox.top + antBox.bottom) / 2 - width,
					  insetBox.right, (antBox.top + antBox.bottom) / 2 + width);
		MyFillRect(&r, ANTSH1 + (pNum - 1));
		MySetRect(&r, (antBox.left + antBox.right) / 2 - width, insetBox.top + width,
					  (antBox.left + antBox.right) / 2 + width, insetBox.bottom + width);
		MyFillRect(&r, ANTSV1 + (pNum - 1));
#endif
	}
	else {
#ifdef MAC
		PenSize(width, width);
		GetIndPattern(&pat, HORIZANTS, pNum);
		PenPat(&pat);
		MyMoveTo(antBox.left, antBox.top);
		MyLineTo(antBox.right, antBox.top);
		MyMoveTo(antBox.left, antBox.bottom);
		MyLineTo(antBox.right, antBox.bottom);
		GetIndPattern(&pat, VERTANTS, pNum);
		PenPat(&pat);
		MyMoveTo(antBox.left, antBox.top);
		MyLineTo(antBox.left, antBox.bottom);
		MyMoveTo(antBox.right, antBox.top);
		MyLineTo(antBox.right, antBox.bottom);
#else
		clippedLeft = _max(antBox.left, insetBox.left);
		clippedTop = _min(antBox.top, insetBox.bottom + 1);
		clippedRight = _max(antBox.right, insetBox.left);
		clippedBottom = _min(antBox.bottom, insetBox.bottom + 1);
		if (antBox.top <= insetBox.bottom + 1) {
			MySetRect(&r, clippedLeft, clippedTop - width, clippedRight, clippedTop + width);
			MyFillRect(&r, ANTSH1 + (pNum - 1));
		}
		if (antBox.bottom <= insetBox.bottom + 1) {
			MySetRect(&r, clippedLeft, clippedBottom - width, clippedRight, clippedBottom + width);
			MyFillRect(&r, ANTSH1 + (pNum - 1));
		}
		if (antBox.left >= insetBox.left) {
			MySetRect(&r, clippedLeft - width, clippedTop, clippedLeft + width, clippedBottom);
			MyFillRect(&r, ANTSV1 + (pNum - 1));
		}
		if (antBox.right >= insetBox.left) {
			MySetRect(&r, clippedRight - width, clippedTop, clippedRight + width, clippedBottom);
			MyFillRect(&r, ANTSV1 + (pNum - 1));
		}
#endif
	}
	PenNormal();
}

void ActivateAllMenus(Boolean enable)
{
#ifdef MAC
	MyEnableMenuItem(GetMenuHandle(APPLEMENU), 0, enable);
#endif
	MyEnableMenuItem(GetMenuHandle(FILEMENU), 0, enable);
	MyEnableMenuItem(GetMenuHandle(EDITMENU), 0, enable);
//	MyEnableMenuItem(GetMenuHandle(SHARINGMENU), 0, enable);	// IAC removed for cossm
	MyEnableMenuItem(GetMenuHandle(HELPMENU), 0, enable);
}

///////////////////////////////////////////////////////////////////////////////////////

void DoublesToWorldPoint(double latVal, double longVal, char dirLat, char dirLong, WorldPoint *p)
{
	// JLM 8/31/99 change to doubles to increase accuracy
	if (dirLat == 'S') latVal *= -1;
	if (dirLong == 'W') longVal *= -1;
	latVal = latVal * 1000000;
	p->pLat = _max(-90000000, _min(latVal, 90000000));
	longVal = longVal * 1000000;
	//p->pLong = _max(-180000000, _min(longVal,  180000000));
	//p->pLong = _max(-180000000, _min(longVal,  360000000));
	p->pLong = _max(-360000000, _min(longVal,  360000000));
}

void WorldPointToDoubles(WorldPoint wp, double *latVal, double *longVal,
						char *dirLat, char *dirLong,short numDecimalPlaces)
{
	// JLM 8/31/99 change to doubles to increase accuracy
	double  dLatVal = _max(-90000000, _min(wp.pLat,  90000000)) / (double)1000000;
	//double dLongVal = _max(-180000000, _min(wp.pLong, 180000000)) / (double)1000000;
	//double dLongVal = _max(-180000000, _min(wp.pLong, 360000000)) / (double)1000000;
	double dLongVal = _max(-360000000, _min(wp.pLong, 360000000)) / (double)1000000;
	*dirLat = (dLatVal < 0) ? 'S' : 'N';
	*dirLong = (dLongVal <= 0) ? 'W' : 'E';
	dLatVal  = fabs(dLatVal);
	dLongVal = fabs(dLongVal);
	//new code JLM 11/30/98, round to number of decimal places
	if(numDecimalPlaces>= 0)
	{
		dLatVal = round(((double)dLatVal)*pow(10.,numDecimalPlaces))/pow(10.,numDecimalPlaces);
		dLongVal = round(((double)dLongVal)*pow(10.,numDecimalPlaces))/pow(10.,numDecimalPlaces);
	}
	*latVal = dLatVal;
	*longVal = dLongVal;
}

void WorldPointToDegMin(WorldPoint wp, long *deg1, float *min1,
									   long *deg2, float *min2,
									   char *dirLat, char *dirLong,short numDecimalPlaces)
{
	double latVal, longVal, remainder;
	
	WorldPointToDoubles(wp, &latVal, &longVal, dirLat, dirLong, -1);
	
	*deg1 = (long)latVal;
	remainder = latVal - *deg1;
	//*min1 = _min(59.99, 60.0 * remainder);
	*min1 = 60.0 * remainder; // JLM 
	
	*deg2 = (long)longVal;
	remainder = longVal - *deg2;
	//*min2 = _min(59.99, 60.0 * remainder);
	*min2 = 60.0 * remainder; // JLM 
	
	//new code JLM , round to number of decimal places
	if(numDecimalPlaces>= 0)
	{
		*min1 = round(((double)*min1)*pow(10.,numDecimalPlaces))/pow(10.,numDecimalPlaces);
		if(*min1 >= 60) {*min1 = 0;*deg1 += 1;} 
		*min2 = round(((double)*min2)*pow(10.,numDecimalPlaces))/pow(10.,numDecimalPlaces);
		if(*min2 >= 60) {*min2 = 0;*deg2 += 1;} 
	}

}

void WorldPointToDMS(WorldPoint wp, long *deg1, long *min1, float *sec1,
									long *deg2, long *min2, float *sec2,
									char *dirLat, char *dirLong,short numDecimalPlaces)
{
	double latVal, longVal, remainder, newvalue;
	
	WorldPointToDoubles(wp, &latVal, &longVal, dirLat, dirLong,-1);
	
	*deg1 = (long)latVal;
	remainder = latVal - *deg1;
	newvalue = 60.0 * remainder;
	*min1 = (long)newvalue;
	remainder = newvalue - *min1;
	//*sec1 = _min(59.99, 60.0 * remainder);
	*sec1 = 60.0 * remainder; // JLM 10/15/98
	
	*deg2 = (long)longVal;
	remainder = longVal - *deg2;
	newvalue = 60.0 * remainder;
	*min2 = (long)newvalue;
	remainder = newvalue - *min2;
	//*sec2 = _min(59.99, 60.0 * remainder);
	*sec2 = 60.0 * remainder; // JLM 10/15/98

	//new code JLM 10/15/98, round to number of decimal places
	if(numDecimalPlaces>= 0)
	{
		*sec1 = round(((double)*sec1)*pow(10.,numDecimalPlaces))/pow(10.,numDecimalPlaces);
		if(*sec1 >= 60) {*sec1 = 0;*min1 +=1;} 
		if(*min1 >= 60) {*min1 = 0;*deg1 +=1;} 
		*sec2 = round(((double)*sec2)*pow(10.,numDecimalPlaces))/pow(10.,numDecimalPlaces);
		if(*sec2 >= 60) {*sec2 = 0;*min2 +=1;} 
		if(*min2 >= 60) {*min2 = 0;*deg2 +=1;} 
	}
}

WorldPoint RoundWpToNearestSecond(WorldPoint wp)
{
	// this fuction helps ensure the user gets consistent answers when using the mouse 
	// the values are stored as longs = val*1000000
	wp.pLat = round(1000000.0*(round(wp.pLat*3600.0/1000000.0))/(3600.0));
	wp.pLong = round(1000000.0*(round(wp.pLong*3600.0/1000000.0))/(3600.0));
	return wp;
}


Point WorldToScreenPointRound(WorldPoint wp, WorldRect wr, Rect r)
{
	float xRatio = (float)(wp.pLong - wr.loLong) / (float)WRectWidth(wr),
		  yRatio = (float)(wp.pLat - wr.loLat) / (float)WRectHeight(wr);
	Point p;
	
	SetPt(&p, ScaleToShort(r.left + round(RectWidth(r) * xRatio)),
			  ScaleToShort(r.bottom - round(RectHeight(r) * yRatio)));
	
	return p;
}

Point WorldToScreenPoint(WorldPoint wp, WorldRect wr, Rect r)
{
	float xRatio = (float)(wp.pLong - wr.loLong) / (float)WRectWidth(wr),
		  yRatio = (float)(wp.pLat - wr.loLat) / (float)WRectHeight(wr);
	Point p;
	
	//SetPt(&p, ScaleToShort(r.left + round(RectWidth(r) * xRatio)),
			  //ScaleToShort(r.bottom - round(RectHeight(r) * yRatio)));
	SetPt(&p, ScaleToShort(r.left + RectWidth(r) * xRatio),
			 ScaleToShort(r.bottom - RectHeight(r) * yRatio));
	
	return p;
}

Rect WorldToScreenRect(WorldRect wr, WorldRect wr2, Rect r)
{
	Rect r2;
	float xRatio1 = (float)(wr.loLong - wr2.loLong) / (float)WRectWidth(wr2),
		  yRatio1 = (float)(wr.hiLat - wr2.loLat) / (float)WRectHeight(wr2),
		  xRatio2 = (float)(wr.hiLong - wr2.loLong) / (float)WRectWidth(wr2),
		  yRatio2 = (float)(wr.loLat - wr2.loLat) / (float)WRectHeight(wr2);
	
	//MySetRect(&r2, ScaleToShort(r.left + round(RectWidth(r) * xRatio1)),
				 //ScaleToShort(r.bottom - round(RectHeight(r) * yRatio1)),
				 //ScaleToShort(r.left + round(RectWidth(r) * xRatio2)),
				 //ScaleToShort(r.bottom - round(RectHeight(r) * yRatio2)));
	
	MySetRect(&r2, ScaleToShort(r.left + RectWidth(r) * xRatio1),
				 ScaleToShort(r.bottom - RectHeight(r) * yRatio1),
				 ScaleToShort(r.left + RectWidth(r) * xRatio2),
				 ScaleToShort(r.bottom - RectHeight(r) * yRatio2));
	
	return r2;
}

WorldPoint ScreenToWorldPointRound(Point p, Rect r, WorldRect wr)
{
	float xRatio = (float)(p.h - r.left) / (float)RectWidth(r),
		  yRatio = (float)(r.bottom - p.v) / (float)RectHeight(r);
	WorldPoint wp;
	
	wp.pLong = wr.loLong + round(WRectWidth(wr) * xRatio);
	wp.pLat = wr.loLat + round(WRectHeight(wr) * yRatio);
	
	//wp.pLong = _max(min(wp.pLong, 179999999), -179999999);
	//wp.pLong = _max(min(wp.pLong, 359999999), -179999999);
	wp.pLong = _max(_min(wp.pLong, 359999999), -359999999);
	wp.pLat = _max(_min(wp.pLat, 89999999), -89999999);
	
	return wp;
}

WorldPoint ScreenToWorldPoint(Point p, Rect r, WorldRect wr)
{
	float xRatio = (float)(p.h - r.left) / (float)RectWidth(r),
		  yRatio = (float)(r.bottom - p.v) / (float)RectHeight(r);
	WorldPoint wp;
	
	//wp.pLong = wr.loLong + round(WRectWidth(wr) * xRatio);
	//wp.pLat = wr.loLat + round(WRectHeight(wr) * yRatio);
	wp.pLong = wr.loLong + WRectWidth(wr) * xRatio;
	wp.pLat = wr.loLat + WRectHeight(wr) * yRatio;
	
	//wp.pLong = _max(min(wp.pLong, 179999999), -179999999);
	//wp.pLong = _max(min(wp.pLong, 359999999), -179999999);
	wp.pLong = _max(_min(wp.pLong, 359999999), -359999999);
	wp.pLat = _max(_min(wp.pLat, 89999999), -89999999);
	
	return wp;
}

WorldRect ScreenToWorldRect(Rect r, Rect r2, WorldRect wr)
{
	Point p1, p2;
	WorldPoint wp1, wp2;
	WorldRect wr2;
	
	SetPt(&p1, r.left, r.top);
	SetPt(&p2, r.right, r.bottom);
	wp1 = ScreenToWorldPoint(p1, r2, wr);
	wp2 = ScreenToWorldPoint(p2, r2, wr);
	wr2.loLat = wp2.pLat;
	wr2.loLong = wp1.pLong;
	wr2.hiLat = wp1.pLat;
	wr2.hiLong = wp2.pLong;
	
	return wr2;
}

WORLDPOINTH ScreenToWorldPoints(POINTH points, Rect r, WorldRect wr, long dLong, long dLat)
{
	WORLDPOINTH wPoints;
	long n = _GetHandleSize((Handle)points) / sizeof(Point), i;
	
	wPoints = (WORLDPOINTH)_NewHandle(n * sizeof(WorldPoint));
	if (!wPoints) { TechError("ScreenToWorldPoints()", "_NewHandle()", 0); return 0; }
	
	for (i = 0 ; i < n ; i++) {
		INDEXH(wPoints, i) = ScreenToWorldPoint(INDEXH(points, i), r, wr);
		INDEXH(wPoints, i).pLong += dLong;
		INDEXH(wPoints, i).pLat += dLat;
	}
	
	return wPoints;
}

Rect ShrinkRectToMatchRect(Rect r, Rect rModel)
{
	float dummy, rRatio, mRatio;
	
	// modify r so that it is no bigger than rModel in either dimension
	if (RectWidth(r) > RectWidth(rModel))
		r.right = r.left + RectWidth(rModel);
	if (RectHeight(r) > RectHeight(rModel))
		r.bottom = r.top + RectHeight(rModel);
	
	// change r so that it has the same aspect ratio as rModel
	rRatio = (float)RectWidth(r) / (float)RectHeight(r),
	mRatio = (float)RectWidth(rModel) / (float)RectHeight(rModel);
	
	dummy = rRatio; // to avoid compiler bug
	dummy = mRatio; // to avoid compiler bug
	
	if (mRatio > rRatio) // rModel flatter than r
		r.bottom = r.top + round(RectWidth(r) / mRatio);
	else // rModel taller than r
		r.left = r.right - round(RectHeight(r) * mRatio);
	
	return r;
}

Rect SameRectRatio(WorldRect wr1, WorldRect wr2, Rect r1)
{
	Rect r2;
	float xRatio1 = (float)(wr2.loLong - wr1.loLong) / (float)WRectWidth(wr1),
		  yRatio1 = (float)(wr2.hiLat - wr1.loLat) / (float)WRectHeight(wr1),
		  xRatio2 = (float)(wr2.hiLong - wr1.loLong) / (float)WRectWidth(wr1),
		  yRatio2 = (float)(wr2.loLat - wr1.loLat) / (float)WRectHeight(wr1);
	
	//MySetRect(&r2, ScaleToShort(r1.left + round(RectWidth(r1) * xRatio1)),
				// ScaleToShort(r1.bottom - round(RectHeight(r1) * yRatio1)),
				// ScaleToShort(r1.left + round(RectWidth(r1) * xRatio2)),
				 //ScaleToShort(r1.bottom - round(RectHeight(r1) * yRatio2)));
	
	MySetRect(&r2, ScaleToShort(r1.left + RectWidth(r1) * xRatio1),
				 ScaleToShort(r1.bottom - RectHeight(r1) * yRatio1),
				 ScaleToShort(r1.left + RectWidth(r1) * xRatio2),
				 ScaleToShort(r1.bottom - RectHeight(r1) * yRatio2));
	
	return r2;
}

Rect CorrespondingScreenRect(WorldRect wr, Rect r)
{
	long width = round(WRectWidth(wr) * (float)LongToLatRatio2(&wr)),
		 height = WRectHeight(wr);
	float dummy,
		  rRatio = (float)RectWidth(r) / (float)RectHeight(r),
		  vRatio = (float)width / (float)height;
	
	dummy = rRatio; // to avoid compiler bug
	dummy = vRatio; // to avoid compiler bug
	
	if (vRatio > rRatio) // wr flatter than r
		r.bottom = r.top + round(RectWidth(r) / vRatio);
	else // wr taller than r
		r.left = r.right - round(RectHeight(r) * vRatio);
	
	return r;
}

WorldRect ExtendWRToFitAspect(WorldRect wr, WorldRect aspect)
{
	long width = round(WRectWidth(wr) * (float)LongToLatRatio2(&wr)),
		 height = WRectHeight(wr),
		 widthA = round(WRectWidth(aspect) * (float)LongToLatRatio2(&aspect)),
		 heightA = WRectHeight(aspect),
		 difference;
	float dummy,
		  aRatio = (float)widthA / (float)heightA,
		  wRatio = (float)width / (float)height;
	
	dummy = aRatio; // to avoid compiler bug
	dummy = wRatio; // to avoid compiler bug
	
	if (wRatio > aRatio) { // wr flatter than given aspect
		difference = round(width / aRatio) - height;
		wr.loLat -= difference / 2;
		wr.hiLat += difference / 2;
	}
	else { // wr taller than given aspect
		difference = round(height * aRatio) - width;
		wr.loLong -= difference / 2;
		wr.hiLong += difference / 2;
	}
	
	return wr;
}

void CenterOnFocusPoint()
{
	long centerLongCV, centerLatCV, dLong, dLat;
	WorldRect newView;
	
	centerLongCV = (settings.currentView.loLong + settings.currentView.hiLong) / 2;
	centerLatCV = (settings.currentView.loLat + settings.currentView.hiLat) / 2;
	dLong = focusPoint.pLong - centerLongCV;
	dLat = focusPoint.pLat - centerLatCV;
	newView = settings.currentView;
	OffsetWRect(&newView, dLong, dLat);
	ChangeCurrentView(newView, FALSE, TRUE);
}

void MoveFocusPoint(WorldPoint newFocus)
{
	focusPoint = newFocus;
	
	if (WPointInWRect(focusPoint.pLong, focusPoint.pLat, &settings.currentView))
		InvalMapDrawingRect();
	else
		CenterOnFocusPoint();
}

static long currentScaleValue = -1;

long CurrentScaleDenominator()
{
	long height = WRectHeight(settings.currentView);
	Rect r = MapDrawingRect();
	
	if (currentScaleValue != -1)
		return currentScaleValue;
	
	if (height < 0) return 1000000;
	
	return round(height / (PixelsPerMilDeg() * RectHeight(r)));
}

void SetCurrentScaleDenominator(long scale)
{
	long width, height;
	double ratio;
	WorldRect newView;
	
	scale = _max(scale, MostZoomedInScale());
	scale = _min(scale, MostZoomedOutScale());
	
	width = WRectWidth(settings.currentView) ;
	height = WRectHeight(settings.currentView);
	ratio = (double)scale / (double)CurrentScaleDenominator();
	
	width = round(width * ratio);
	height = round(height * ratio);
	
	newView.loLat =  round(focusPoint.pLat - height/2);
	newView.loLong = round(focusPoint.pLong - width/2),
	newView.hiLat =  newView.loLat+ height;
	newView.hiLong  = newView.loLong + width;
	ChangeCurrentView(newView, FALSE, TRUE);
	
}

static Rect prevMDR = { 0, 0, 0, 0 };

void ForceViewAspect2(WorldRect *view)
{
	Rect r = MapDrawingRect();
	long width = WRectWidth(*view), height = WRectHeight(*view);
	float xRatio = (float)RectWidth(r) / (float)RectWidth(prevMDR),
		  yRatio = (float)RectHeight(r) / (float)RectHeight(prevMDR);
	
	width *= xRatio;
	height *= yRatio;
	
	view->hiLong = view->loLong + width;
	view->loLat = view->hiLat - height;
}

void ForceViewAspect(WorldRect *view, short option)
{
	long width = round(WRectWidth(*view) * (float)LongToLatRatio2(view)),
		 height = WRectHeight(*view),
		 difference;
	Rect r = MapDrawingRect();
	float dummy,
		  rRatio = (float)RectWidth(r) / (float)RectHeight(r),
		  vRatio = (float)width / (float)height;
	
	dummy = rRatio; // to avoid compiler bug
	dummy = vRatio; // to avoid compiler bug
	
	if (vRatio > rRatio) { // view flatter than rect
		difference = round(width / rRatio) - height;
		if (option == FV_UPPERLEFT)
			view->loLat -= difference;
		else {
			view->loLat -= difference / 2;
			view->hiLat += difference / 2;
		}
	}
	else { // view taller than rect
		difference = round((height * rRatio - width) / (float)LongToLatRatio2(view));
		if (option == FV_UPPERLEFT)
			view->hiLong += difference;
		else {
			view->loLong -= difference / 2;
			view->hiLong += difference / 2;
		}
	}
}

void ChangeCurrentView(WorldRect newView, Boolean check, Boolean forceAspect)
{
	long width,height, n, scale;
	long mostZoomedInScale, mostZoomedOutScale;
	Rect r = MapDrawingRect(), empty = { 0, 0, 0, 0 };
	WorldRect boundingRectOfAllMaps = model->GetMapBounds();
	
	mostZoomedInScale = MostZoomedInScale();
	mostZoomedOutScale = MostZoomedOutScale();

	width = round(WRectWidth(newView) * (double)LongToLatRatio2(&newView));
	height = WRectHeight(newView);
	

	scale = height / (PixelsPerMilDeg() * RectHeight(r));
	if (scale < mostZoomedInScale) {
		n = mostZoomedInScale * (PixelsPerMilDeg() * RectWidth(r));// I moved this inside this loop 12/19/97 JLM
		OutsetWRect(&newView, (n - WRectWidth(newView)) / 2, 0);// I moved this inside this loop 12/19/97 JLM
		n = mostZoomedInScale; 
		// I think this is a bug
		//OutsetWRect(&newView, (n - height) / 2, 0);
		// I think it should be 
		OutsetWRect(&newView, 0,(n - height) / 2);

		//  alternative JLM code
		/////////////////
		//	long newHt = mostZoomedInScale* (PixelsPerMilDeg() * RectHeight(r)); 
		//	long newWidth = (newHt *RectWidth(r))/(RectHeight(r)*(double)LongToLatRatio2(&newView));
		//	OutsetWRect(&newView, (newWidth - WRectWidth(newView)) / 2, 0);// I moved this inside this loop 12/19/97 JLM
		//	OutsetWRect(&newView, 0,(newHt - height) / 2);
	}
	
	/////////////////////
	// JLM 12/22/97
	/// force the map to stay on 1/3 the screen
	height = WRectHeight(boundingRectOfAllMaps);
	if(height > 0)
	{
		long oneThirdPt,offset;
		////////
		width = WRectWidth(newView);
		oneThirdPt = newView.loLong + width*OVERLAPFRACTION;
		if(boundingRectOfAllMaps.hiLong < oneThirdPt)
		{
			offset =  oneThirdPt - boundingRectOfAllMaps.hiLong;
			OffsetWRect(&newView,-offset,0);
		}
		oneThirdPt = newView.hiLong - width*OVERLAPFRACTION;
		if(boundingRectOfAllMaps.loLong > oneThirdPt)
		{
			offset =  boundingRectOfAllMaps.loLong - oneThirdPt;
			OffsetWRect(&newView,offset,0);
		}
		/////
		height = WRectHeight(newView);
		oneThirdPt = newView.loLat + height*OVERLAPFRACTION;
		if(boundingRectOfAllMaps.hiLat < oneThirdPt)
		{
			offset = oneThirdPt -  boundingRectOfAllMaps.hiLat;
			OffsetWRect(&newView,0,-offset);
		}
		oneThirdPt = newView.hiLat - height*OVERLAPFRACTION;
		if(boundingRectOfAllMaps.loLat > oneThirdPt)
		{
			offset =  boundingRectOfAllMaps.loLat - oneThirdPt;
			OffsetWRect(&newView,0,offset);
		}
		//////////
	}
	
	
	previousView = settings.currentView;
	if (forceAspect == FV_SAMESCALE && !EqualRects(prevMDR, empty))
		ForceViewAspect2(&newView);
	if (forceAspect)
		ForceViewAspect(&newView, forceAspect);
	settings.currentView = newView;
	prevMDR = r;
	
	//if (!WPointInWRect(focusPoint.pLong, focusPoint.pLat, &newView)) {
	// JLM 1/9/98 focus point will always be the center of the current view
		focusPoint.pLat = (settings.currentView.loLat + settings.currentView.hiLat) / 2;
		focusPoint.pLong = (settings.currentView.loLong + settings.currentView.hiLong) / 2;	
	//}
	
	if (check) {
		if (CurrentScaleDenominator() < mostZoomedInScale)
			SetCurrentScaleDenominator(mostZoomedInScale);
		if (CurrentScaleDenominator() > mostZoomedOutScale)
			SetCurrentScaleDenominator(mostZoomedOutScale);
	}
	
	
	InvalMapDrawingRect();
	InvalidateMapImage();
}

void AddRectToCurrentView(WorldRect wr)
{
	WorldRect newView = UnionWRect(settings.currentView, wr);
	
	if (!EqualWRects(newView, settings.currentView))
		ChangeCurrentView(newView, TRUE, TRUE);
	else
		InvalMapDrawingRect();
}

void RescaleTwoPoints(WorldPoint p1, WorldPoint p2)
{
	WorldRect newView;
	
	newView.loLong = _min(p1.pLong, p2.pLong);
	newView.hiLong = _max(p1.pLong, p2.pLong);
	newView.loLat = _min(p1.pLat, p2.pLat);
	newView.hiLat = _max(p1.pLat, p2.pLat);
	ChangeCurrentView(newView, TRUE, TRUE);
	settings.currentView = AddWRectBorders(settings.currentView, 10);
	ForceViewAspect(&settings.currentView, TRUE);
}

long ComputeTextWidth(char *text, short *lineCount)
{
	long index;
	long width = 0, lineWidth = 0;
	
	*lineCount = 1;
	
	for (index = 0 ; text[index] ; index++) {
		if (text[index] == RETURN) {
			width = _max(width, lineWidth);
			(*lineCount)++;
			lineWidth = 0;
		}
		else
			lineWidth += CharWidth(text[index]);
	}
	width = _max(width, lineWidth);
	
	return width;
}


void AbbreviateDString(DialogPtr dialog, short leftBar, short rightBar, CHARPTR text)
{
	short width = RectLeft(GetDialogItemBox(dialog, rightBar)) -
				  RectLeft(GetDialogItemBox(dialog, leftBar));
	
	AbbreviateString(text, width - stringwidth("WW"));
}

///////////////////////////////////////////////////////////////////////////////////////

float ConvertUnits(float n, short oldUnits, short newUnits)
{
	float conv1, conv2;
	
	switch (oldUnits) {
		case FEET:			conv1 = MILESTOFEET; break;
		case YARDS:			conv1 = MILESTOYARDS; break;
		case METERS:		conv1 = MILESTOMETERS; break;
		case KILOMETERS:	conv1 = MILESTOKILO; break;
		case MILES:			conv1 = 1.0; break;
		case NAUTICALMILES:	conv1 = MILESTONAUTS; break;
	}
	
	switch (newUnits) {
		case FEET:			conv2 = MILESTOFEET; break;
		case YARDS:			conv2 = MILESTOYARDS; break;
		case METERS:		conv2 = MILESTOMETERS; break;
		case KILOMETERS:	conv2 = MILESTOKILO; break;
		case MILES:			conv2 = 1.0; break;
		case NAUTICALMILES:	conv2 = MILESTONAUTS; break;
	}
	
	return (n / conv1) * conv2;
}

float INCMtoUnits(short units)
{
	switch (units) {
		case FEET:			return INCHESTOFEET;
		case YARDS:			return INCHESTOYARDS;
		case METERS:		return CMTOMETERS;
		case KILOMETERS:	return CMTOKILOMETERS;
		case MILES:			return INCHESTOMILES;
		case NAUTICALMILES:	return INCHESTONAUTS;
	}
	
	return INCHESTOMILES;
}

char *AppendUnits(char *s, short u)
{
	getindstring(&s[strlen(s)], UNITSTRINGS, u);
	
	return s;
}

Boolean MetricUnits(short units)
{
	return units == KILOMETERS || units == METERS;
}

float PixelConv(short units)
{
	return MetricUnits(units) ? PixelsPerCentimeter() : PixelsPerInchScreen();
}

char *ScaleToDisplayString(long scale, char *s, Boolean allowWindowWidth)
{
	short scaleFormat = 0; // settings.scaleFormat;
	char format[] = "%.2f";
	float conversion;
	Rect r = MapDrawingRect();
	
	if (scaleFormat == WINDOWWIDTH && !allowWindowWidth)
		scaleFormat = EQUALITY;
	if (settings.distanceUnits == FEET || settings.distanceUnits == YARDS || settings.distanceUnits == METERS)
		format[2] = '0';
	
	switch (scaleFormat) {
		case RATIO:
			sprintf(s, "%ld", scale);
			break;
		case EQUALITY:
			sprintf(s, format, (float)scale * INCMtoUnits(settings.distanceUnits));
			break;
		case WINDOWWIDTH:
			conversion = INCMtoUnits(settings.distanceUnits) * (float)RectWidth(r) / PixelConv(settings.distanceUnits);
			sprintf(s, format, scale * conversion);
			break;
	}
	if (s[0] == '-') s[0] = 0;
	
	return s;
}

long DisplayNumToScale(float n, Boolean allowWindowWidth)
{
	short scaleFormat = 0; // settings.scaleFormat;
	long scale;
	float conversion;
	Rect r = MapDrawingRect();
	
	if (scaleFormat == WINDOWWIDTH && !allowWindowWidth) scaleFormat = EQUALITY;
	
	switch (scaleFormat) {
		case RATIO:
			scale = (long)round(n);
			break;
		case EQUALITY:
			conversion = INCMtoUnits(settings.distanceUnits);
			scale = (long)round(n / conversion);
			break;
		case WINDOWWIDTH:
			conversion = INCMtoUnits(settings.distanceUnits) * (float)RectWidth(r) / PixelConv(settings.distanceUnits);
			scale = (long)round(n / conversion);
			break;
	}
	
	return scale;
}

void GetEquality(long scale, char *sLabel, char *sValue, char *sUnits)
{
	char *p;
	
	if (settings.distanceUnits == METERS || settings.distanceUnits == KILOMETERS) strcpy(sLabel, "1 cm =");
	else strcpy(sLabel, "1 in =");
	
	ScaleToDisplayString(scale, sValue, FALSE);
	if (strlen(sValue) >= 7 && (p = strchr(sValue, '.')))
		p[0] = 0; // no decimals when value >= 1000
	
	sUnits[0] = 0;
	AppendUnits(sUnits, settings.distanceUnits);
}

void GetWindowSize(long scale, char *sWidth, char *sHeight, char *sUnits)
{
	float conversion;
	char format[20], *p;
	Rect r = MapDrawingRect();
	
	if (settings.distanceUnits == FEET || settings.distanceUnits == YARDS || settings.distanceUnits == METERS)
		strcpy(format, "%.0f");
	else
		strcpy(format, "%.2f");
	
	conversion = INCMtoUnits(settings.distanceUnits) * (float)RectWidth(r) / PixelConv(settings.distanceUnits);
	sprintf(sWidth, format, scale * conversion);
	if (strlen(sWidth) >= 7 && (p = strchr(sWidth, '.')))
		p[0] = 0; // no decimals when value >= 1000
	
	conversion = INCMtoUnits(settings.distanceUnits) * (float)RectHeight(r) / PixelConv(settings.distanceUnits);
	sprintf(sHeight, format, scale * conversion);
	if (strlen(sHeight) >= 7 && (p = strchr(sHeight, '.')))
		p[0] = 0; // no decimals when value >= 1000
	
	sUnits[0] = 0;
	AppendUnits(sUnits, settings.distanceUnits);
}

///////////////////////////////////////////////////////////////////////////////////////

static short percentValue = 0;

void ShowPercentProgress(short n)
{
	GrafPtr oldPort;
	Rect r;
	
	if (n == percentValue) return;
	percentValue = n;
	
	SetWatchCursor();
	
	GetPortGrafPtr(&oldPort);
	SetPortDialogPort(progressDialog);
	r = GetDialogItemBox(progressDialog, M29BAR);
	MyFrameRect(&r);
	r.right = r.left + round((float)(r.right - r.left) * (float)n / 100) + 1;
#ifdef IBM
	r.bottom++;
#endif
	PaintRect(&r);
	SetPortGrafPort(oldPort);
}

#ifdef IBM
static Boolean sharedProgressBreak;
#endif

Boolean ShowModalProgress(short alertID, short n)
{
#ifdef MAC
	DialogPtr dummyDialog;
	short itemHit;
#endif
	EventRecord e;
	Boolean broke = FALSE;

	if (ItemHidden(progressDialog, M29STOP)) {
		ShowDialogItem(progressDialog, M29STOP);
		DrawDialog(progressDialog);
	}
	MyEnableControl(progressDialog, M29STOP, TRUE);
	
	ShowPercentProgress(n);
	
#ifdef MAC	
	if (GetNextEvent(mDownMask | keyDownMask | autoKeyMask | osMask, &e)) {
		if (e.what == MultFindEvt) DealWithEvent(e, TRUE);
		if (CmdPeriodEvent(&e))
			broke = TRUE;
		else if (IsDialogEvent(&e)) {
			DialogSelect(&e, &dummyDialog, &itemHit);
			if (itemHit == M29STOP)
				broke = TRUE;
		}
	}
#else
	sharedProgressBreak = FALSE;
	while (PeekMessage(&e, progressDialog, 0, 0, PM_REMOVE))
		if (!IsDialogMessage(progressDialog, &e)) {
		}
	if (PeekMessage(&e, mapWindow, WM_LBUTTONDOWN, WM_LBUTTONDOWN, PM_REMOVE)) SysBeep(1);
	while (PeekMessage(&e, mapWindow, WM_LBUTTONDOWN, WM_MOUSELAST, PM_REMOVE)) ;
	if (PeekMessage(&e, mapWindow, WM_PAINT, WM_PAINT, PM_REMOVE)) {
		TranslateMessage(&e);
		DispatchMessage(&e);
	}
	if (GetActiveWindow() != progressDialog)
		SetActiveWindow(progressDialog);
	broke = sharedProgressBreak;
#endif
	
	if (broke) {
#ifdef IBM
		SetActiveWindow(mapWindow);
		ShowWindow(progressDialog, SW_HIDE);
#endif
		SysBeep(1);
		if (CHOICEALERT(alertID, 0, FALSE))
			broke = FALSE;
#ifdef IBM
		ShowWindow(progressDialog, SW_SHOW);
		SetActiveWindow(progressDialog);
		DrawDialog(progressDialog);
#endif
	}
	
	return broke;
}

static char message1[256], message2[256];

// num == 0 -> change progress window title
// num == 1 -> change main message and reset progress bar
// num == 2 -> change secondary message and do not reset progress bar
// num == 3 -> change secondary message and reset progress bar

void ProgressMessage(char *msg, short num)
{
	Rect r;
	GrafPtr oldPort;
	
	if (num == 0) { setwtitle(GetDialogWindow(progressDialog), msg); return; }
	
	if (num == 1) {
		strcpy(message1, msg);
		strcpy(message2, "");
	}
	else
		strcpy(message2, msg);
#ifdef MAC
	paramtext(message1, message2, "", "");
#else
	mysetitext(progressDialog, 1, message1);
	mysetitext(progressDialog, 4, message2);
#endif
	
	GetPortGrafPtr(&oldPort);
	SetPortDialogPort(progressDialog);
	if (num == 1 || num == 3)
		r = GetDialogPortRect(progressDialog);
	else
		r = GetDialogItemBox(progressDialog, M29MESSAGES);
	
	if (num == 1 || num == 3) percentValue = 0;
	
#ifdef IBM
	if (num == 1 || num == 3)
#endif
		{ EraseRect(&r); DrawDialog(progressDialog); }
	// HideDialogItem(progressDialog, M29STOP);
	SetPortGrafPort(oldPort);
}

void ProgressMessage2(char *mssg)
{
	ProgressMessage(mssg, 2);
}

void ProgressMessage3(char *mssg)
{
	ProgressMessage(mssg, 3);
}

#ifdef IBM
BOOL CALLBACK ProgressProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	PAINTSTRUCT ps;
	short pv;
	
	switch (message) {
		case WM_INITDIALOG:
			CenterDialog(hWnd, 0);
			SetFocus(GetDlgItem(hWnd, M29STOP));
			return FALSE;
		case WM_DESTROY: return TRUE;
		case WM_PAINT:
			BeginPaint(hWnd, &ps);
			pv = percentValue;
			ShowPercentProgress(0);
			ShowPercentProgress(pv);
			EndPaint(hWnd, &ps);
			return TRUE;
		case WM_COMMAND:
			if (HIWORD(wParam) == BN_CLICKED && LOWORD(wParam) == M29STOP)
				sharedProgressBreak = TRUE;
			return TRUE;
		case WM_SETCURSOR:
			return TRUE;
	}
	
	return FALSE;
}
#endif

static GrafPtr oldProgressPort;

void OpenProgressWindow(char *title, WindowPtr parentWindow)
{
#ifdef MAC
	ActivateAllMenus(FALSE);
 	HiliteWindow(parentWindow, FALSE);
	MyShowWindow(GetDialogWindow(progressDialog));
	SelectWindow(GetDialogWindow(progressDialog));
	ProgressMessage(title, 0);
#else
	GetPortGrafPtr(&oldProgressPort);
	progressDialog = CreateDialog(hInst, MAKEINTRESOURCE(M29), parentWindow, (DLGPROC)ProgressProc);
	SetPortDialogPort(progressDialog);
	ProgressMessage(title, 0);
#endif
}

void CloseProgressWindow(WindowPtr parentWindow)
{
#ifdef MAC
	ActivateAllMenus(TRUE);
	ShowDialogItem(progressDialog, M29STOP);
	HideWindow(GetDialogWindow(progressDialog));
	HiliteWindow(parentWindow, TRUE);
	DrawMenuBar(); // need to rehilite menus since progress dialog is not disposed
#else
	if (progressDialog) {
		SetPortGrafPort(oldProgressPort);
		SetActiveWindow(parentWindow);
		DestroyWindow(progressDialog);
		progressDialog = 0;
		percentValue = 0;
	}
#endif
}

void printSomething(char *msg, long dialogID)
{
	//  support writing to the error log rather than putting up an error message
	OSErr err = WriteErrorToCommandFileErrorLog(msg,dialogID);
	if(!err)
		return; // we wrote to the error log file
	// if err, then fall through and let the process be stopped with the alert message so that the user sees it

#ifdef MAC
{
	NMRec N;
	EventRecord e;
	
	InitCursor();
	
	if (settings.inBackground) {
		N.qType = 8;
		N.nmMark = 1;
		N.nmIcon = GetResource('SICN', 128);
		N.nmSound = (Handle)-1;
		N.nmStr = nil;
		N.nmResp = nil;
		NMInstall(&N);
		while (settings.inBackground) {
			GetNextEvent(everyEvent, &e);
			DealWithEvent(e, FALSE);
		}
		NMRemove(&N);
	}
	
	if (!settings.doNotPrintError) {
		InitCursor();
		paramtext(message1, message2, msg, "");
		ActivateParentDialog(FALSE);//JLM 12/24/98
		Alert(dialogID, 0);
		ActivateParentDialog(TRUE); // JLM 12/24/98
		settings.doNotPrintError = mapWindow ? TRUE : FALSE; // to avoid infinite error loops
	}
	else
		DisplayMessage(msg);
}
#else
{
	long icon;
	
	if (!settings.doNotPrintError) {
		switch (dialogID) {
			case 8000: icon = MB_ICONSTOP; break;
			case 8002: icon = MB_ICONEXCLAMATION; break;
			default: icon = MB_ICONINFORMATION; break;
		}
		settings.doNotPrintError = TRUE; // to avoid infinite error loops
		MessageBox(FrontWindow() ? FrontWindow() : mapWindow, msg, ABBREVIATED_WINDOW_TITLE_STR, icon); // mapWindow
	}
	else
		DisplayMessage(msg);
}
#endif
}

void printSomethingAlways(char *msg, long dialogID)
{
	settings.doNotPrintError = FALSE;
	printSomething(msg, dialogID);
	settings.doNotPrintError = FALSE;
}

void printError(char *msg)
{
	printSomething(msg, 8000);
}

void printErrorAlways(char *msg)
{
	printSomethingAlways(msg, 8000);
}

void printNote(char *msg)
{
	printSomething(msg, 8001);
}

void printNoteAlways(char *msg)
{
	printSomethingAlways(msg, 8001);
}

void printWarning(char *msg)
{
	printSomething(msg, 8002);
}

void printWarningAlways(char *msg)
{
	printSomethingAlways(msg, 8002);
}



void printError2(char *format, long n)
{
	SetResLoad(TRUE);
	sprintf(mssg, format, n);
	printError(mssg);
}

void TechError(char *where, char *what, long n)
{
	char format[200];
	
	if (n == memFullErr || !strcmp(what, "_NewHandle()") || !strcmp(what, "_SetHandleSize()")) {
		if (rainyDayReserve) { DisposeHandle(rainyDayReserve); rainyDayReserve = 0; }
		printError(GS1(153, 0)); // Not enough memory to complete this operation.
								 // You may want to quit other running programs.
								 // (Macintosh users: You may also want
								 // to quit and use 'Get Info' in the Finder to give it
								 // more memory.)
	}
	else {
		getindstring(format, 1000, 3); // "Technical error: %s returned %ld in %s."
		sprintf(mssg, format, what, n, where);
		printError(mssg);
	}
	
	// reset these since they might have been messed up by a failed disk read or write
	sixteen = 16; eight = 8; five = 5; four = 4; three = 3; two = 2; one = 1;
	zero[0] = zero[1] = zero[2] = zero[3] = 0;
}

void OutOffMemoryAlert(void)
{
	TechError("","",memFullErr);
}

///////////////////////////////////////////////////////////////////////////////////////

static Boolean sharedPICTcancel;

/*#ifdef MAC

#pragma segment Main

typedef pascal_ifMac void (*LineProcPtr)(Point newPt);
typedef pascal_ifMac void (*PolyProcPtr)(GrafVerb verb, PolyHandle poly);

QDProcs *saveProcs;
CQDProcs *saveCProcs;

#ifdef VERSIONPPC

pascal_ifMac void MyPolyProc(GrafVerb verb, PolyHandle poly)
{
	PenSize(1, 1);
	
	if (settings.colorQDAvailable) {
		if (saveCProcs)
			(*(PolyProcPtr)(*saveCProcs).polyProc)(verb, poly);
		else
			StdPoly(verb, poly);
	}
	else {
		if (saveProcs)
			(*(PolyProcPtr)(*saveProcs).polyProc)(verb, poly);
		else
			StdPoly(verb, poly);
	}
}

pascal_ifMac void MyLineProc(Point newPt)
{
	PenSize(1, 1);
	
	if (settings.colorQDAvailable) {
		if (saveCProcs)
			(*(LineProcPtr)(*saveCProcs).lineProc)(newPt);
		else
			StdLine(newPt);
	}
	else {
		if (saveProcs)
			(*(LineProcPtr)(*saveProcs).lineProc)(newPt);
		else
			StdLine(newPt);
	}
}

pascal_ifMac void MyGetPicProc(Ptr p, short size)
{
	long i, end;
	
	if (size > 0 && CmdPeriod()) { // terminate the picture
		sharedPICTcancel = TRUE;
		FlushEvents(keyDownMask, autoKeyMask);
		end = size - 2;
		for (i = 0 ; i < end ; i++)
			p[i] = 0x00; // NO-0P
		p[size-1] = 0xFF; // terminate
	}
	else
		StdGetPic(p, size);
}

void DrawPictureWithCancel(WindowPtr w, PicHandle hPICT, Rect r)
{
	QDProcs theQDProcs;
	CQDProcs theCQDProcs;
	PictInfo info;
	OSErr err = 0;
	
	// there is a problem mixing the ESC-testing with pictures that
	// contain bitmaps (I think the problem is that it doesn't deal
	// well when the picture is cancelled in the middle of getting
	// the bitmap bytes from the picture) so we check for the presence
	// of bitmaps and do a non-interuptible draw if one is present
	
	if (err = GetPictInfo(hPICT, &info, 0, 0, 0, 0))
		{ SysBeep(1); return; }
	
	// if (info.bitMapCount || info.pixMapCount)
	// 	{ DrawPicture(hPICT, &r); return; }
	
	if (settings.colorQDAvailable) {
		saveCProcs = (CQDProcs *)w->grafProcs;
		if (saveCProcs)
			theCQDProcs = *saveCProcs;
		else
			SetStdCProcs(&theCQDProcs);
		
		if (!saveCProcs && !info.bitMapCount && !info.pixMapCount)
			theCQDProcs.getPicProc = (QDGetPicUPP)MakeUPP((ProcPtr)MyGetPicProc, uppQDGetPicProcInfo); 
		theCQDProcs.lineProc = (QDLineUPP)MakeUPP((ProcPtr)MyLineProc, uppQDLineProcInfo);
		theCQDProcs.polyProc = (QDPolyUPP)MakeUPP((ProcPtr)MyPolyProc, uppQDPolyProcInfo);
		w->grafProcs = (QDProcs *)&theCQDProcs;
		
		DrawPicture(hPICT, &r);
		
		w->grafProcs = (QDProcs *)saveCProcs;
	}
	else {
		saveProcs = w->grafProcs;
		if (saveProcs)
			theQDProcs = *saveProcs;
		else
			SetStdProcs(&theQDProcs);
		
		if (!saveProcs && !info.bitMapCount && !info.pixMapCount)
			theQDProcs.getPicProc = (QDGetPicUPP)MakeUPP((ProcPtr)MyGetPicProc, uppQDGetPicProcInfo); 
		theQDProcs.lineProc = (QDLineUPP)MakeUPP((ProcPtr)MyLineProc, uppQDLineProcInfo);
		theQDProcs.polyProc = (QDPolyUPP)MakeUPP((ProcPtr)MyPolyProc, uppQDPolyProcInfo);
		w->grafProcs = &theQDProcs;
		
		DrawPicture(hPICT, &r);
		
		w->grafProcs = saveProcs;
	}
}

#else

pascal_ifMac void MyPolyProc(GrafVerb verb, PolyHandle poly)
{
	PenSize(1, 1);
	
	if (settings.colorQDAvailable) {
		if (saveCProcs)
			(*(PolyProcPtr)(*saveCProcs).polyProc)(verb, poly);
		else
			StdPoly(verb, poly);
	}
	else {
		if (saveProcs)
			(*(PolyProcPtr)(*saveProcs).polyProc)(verb, poly);
		else
			StdPoly(verb, poly);
	}
}

pascal_ifMac void MyLineProc(Point newPt)
{
	PenSize(1, 1);
	
	if (settings.colorQDAvailable) {
		if (saveCProcs)
			(*(LineProcPtr)(*saveCProcs).lineProc)(newPt);
		else
			StdLine(newPt);
	}
	else {
		if (saveProcs)
			(*(LineProcPtr)(*saveProcs).lineProc)(newPt);
		else
			StdLine(newPt);
	}
}

pascal_ifMac void MyGetPicProc(Ptr p, short size)
{
	long i, end;
	
	if (size > 0 && CmdPeriod()) { // terminate the picture
		sharedPICTcancel = TRUE;
		FlushEvents(keyDownMask, autoKeyMask);
		end = size - 2;
		for (i = 0 ; i < end ; i++)
			p[i] = 0x00; // NO-0P
		p[size-1] = 0xFF; // terminate
	}
	else
		StdGetPic(p, size);
}

void DrawPictureWithCancel(WindowPtr w, PicHandle hPICT, Rect r)
{
	QDProcs theQDProcs;
	CQDProcs theCQDProcs;
	PictInfo info;
	OSErr err = 0;
	
	// there is a problem mixing the ESC-testing with pictures that
	// contain bitmaps (I think the problem is that it doesn't deal
	// well when the picture is cancelled in the middle of getting
	// the bitmap bytes from the picture) so we check for the presence
	// of bitmaps and do a non-interuptible draw if one is present
	
	if (err = GetPictInfo(hPICT, &info, 0, 0, 0, 0))
		{ SysBeep(1); return; }
	
	// if (info.bitMapCount || info.pixMapCount)
	// 	{ DrawPicture(hPICT, &r); return; }
	
	if (settings.colorQDAvailable) {
		saveCProcs = (CQDProcs *)w->grafProcs;
		if (saveCProcs)
			theCQDProcs = *saveCProcs;
		else
			SetStdCProcs(&theCQDProcs);
		
		if (!saveCProcs && !info.bitMapCount && !info.pixMapCount)
			theCQDProcs.getPicProc = (QDGetPicUPP)MyGetPicProc; 
		theCQDProcs.lineProc = (QDLineUPP)MyLineProc;
		theCQDProcs.polyProc = (QDPolyUPP)MyPolyProc;
		w->grafProcs = (QDProcs *)&theCQDProcs;
		
		DrawPicture(hPICT, &r);
		
		w->grafProcs = (QDProcs *)saveCProcs;
	}
	else {
		saveProcs = w->grafProcs;
		if (saveProcs)
			theQDProcs = *saveProcs;
		else
			SetStdProcs(&theQDProcs);
		
		if (!saveProcs && !info.bitMapCount && !info.pixMapCount)
			theQDProcs.getPicProc = (QDGetPicUPP)MyGetPicProc; 
		theQDProcs.lineProc = (QDLineUPP)MyLineProc;
		theQDProcs.polyProc = (QDPolyUPP)MyPolyProc;
		w->grafProcs = &theQDProcs;
		
		DrawPicture(hPICT, &r);
		
		w->grafProcs = saveProcs;
	}
}

#endif

#else

void DrawPictureWithCancel(WindowPtr w, PicHandle hPICT, Rect r)
{
#pragma unused(w)
	
	DrawPicture(hPICT, &r);
}

#endif
*/
#ifdef MAC

PicHandle ReadPNTG(short vRefNum, long dirID, char *fileName)
{
	BitMap		bitMap;
	//GrafPort	drawPort;
	GrafPtr		savePort;
	PicHandle	hPICT;
	Ptr			src, dst, saveDst;
	Rect		r;
	short		i;
	Handle		h;
	OSErr 		err = 0;
	#if TARGET_API_MAC_CARBON
		CGrafPtr drawPort = 0;
	#else
		GrafPort	drawPortSpace;
		GrafPtr		drawPort = &drawPortSpace;
	#endif
	
	GetPortGrafPtr(&savePort);

	if (err = ReadFileContents(NONTERMINATED,vRefNum, dirID, fileName, 0, 0, &h)) {
		SysBeep(1);
		DisplayMessage(GS1(154, 0)); // Can't read PICT file.
		return 0;
	}
	
	if (!(dst = _NewPtrClear(MAX_MACPAINT_SIZE))) {
		DisposeHandle(h);
		SysBeep(1);
		DisplayMessage(GS1(155, 0)); // Not enough memory to read PICT file.
		return 0;
	}
	
	src = (*h) + 512;
	saveDst = dst;
	
	_HLock(h);
	for (i = 1 ; i <= MAX_MACPAINT_LINES ; i++)
		UnpackBits(&src, &dst, MACPAINT_BYTE_UNPACK);
	_HUnlock(h);
	
	DisposeHandle(h);
	
	bitMap.baseAddr = saveDst;
	bitMap.rowBytes = MACPAINT_BYTE_UNPACK;
	SetRect(&bitMap.bounds, 0, 0, 576, 720);
	
	//GetPort(&savePort);
	//OpenPort(&drawPort);
	//SetPort(&drawPort);
	//ClipRect(&bitMap.bounds);
	#if TARGET_API_MAC_CARBON
		drawPort = CreateNewPort();
		if(!drawPort) goto done;
	#else
		OpenPort(drawPort);
	#endif
	SetPortGrafPort(drawPort);
	
	hPICT = OpenPicture(&bitMap.bounds);
		r = bitMap.bounds;
		r.bottom = r.top + 40;
		
		for (i = 0 ; i < 18 ; i++) {
			//CopyBits(&bitMap, &(drawPort.portBits), &r, &r, srcCopy, nil);
			CopyBits(&bitMap, GetPortBitMapForCopyBits(drawPort), &r, &r, srcCopy, nil);
			OffsetRect(&r, 0, 40);
		}
	ClosePicture();
	
	SetPortGrafPort(savePort);
	#if TARGET_API_MAC_CARBON
		DisposePort(drawPort);
	#else
		ClosePort(drawPort);
	#endif
	
done:
	if(h) { DisposeHandle(h); h = 0;}
	if(saveDst) {_DisposePtr(saveDst); saveDst = 0;}
	OffsetRect(&(*hPICT)->picFrame, -((*hPICT)->picFrame.left), -((*hPICT)->picFrame.top));
	
	return hPICT;
}

#else

PicHandle ReadPNTG(short vRefNum, long dirID, char *fileName)
{
#pragma unused(vRefNum)
#pragma unused(dirID)
#pragma unused(fileName)
	
	return 0;
}

#endif

PicHandle GetPICTFromFile(short vRefNum, long dirID, char *fileName)
{
#ifdef MAC
	PicHandle hPICT;
	short f;
	long startPos, endPos, length;
	FInfo fInfo;
	OSErr err = 0;
	
	if (!FileExists(vRefNum, dirID, fileName)) {
		SysBeep(1);
		DisplayMessage(GS1(156, 0)); // Can't access PICT file.
		return 0;
	}
	
	if (err = hgetfinfo(vRefNum, dirID, fileName, &fInfo)) {
		SysBeep(1);
		DisplayMessage(GS1(156, 0)); // Can't access PICT file.
		return 0;
	}
	
	if (fInfo.fdType == 'PNTG') {
		hPICT = ReadPNTG(vRefNum, dirID, fileName);
		if (!hPICT) return 0;
	}
	else {
		if (err = hopendf(vRefNum, dirID, fileName, fsCurPerm, &f)) {
			DisplayMessage(GS1(156, 0)); // Can't access PICT file.
			return 0;
		}
		SetFPos(f, fsFromStart, 512); // PICT file has 512 byte header
		GetFPos(f, &startPos);
		GetEOF(f, &endPos);
		length = endPos - startPos;
		if (!(hPICT = (PicHandle)_NewHandleClear(length))) {
			FSClose(f);
			SysBeep(1);
			DisplayMessage(GS1(155, 0)); // Not enough memory to read PICT file.
			return 0;
		}
		_HLock((Handle)hPICT);
		err = FSRead(f, &length, (Ptr)*hPICT);
		_HUnlock((Handle)hPICT);
		FSClose(f);
		if (err) {
			SysBeep(1);
			DisplayMessage(GS1(157, 0)); // Error reading PICT file.
			DisposeHandle((Handle)hPICT);
			return 0;
		}
	}
	
	return hPICT;
#else
	return 0;
#endif
}

Rect ShrinkPICTRect(Rect r, Rect rMatch)
{
	short newWidth, newHeight, wCenter, hCenter;
	float ratio;
	
	if (RectWidth(rMatch) > RectHeight(rMatch)) {
		newWidth = RectWidth(r);
		ratio = (float)RectHeight(rMatch) / (float)RectWidth(rMatch);
		newHeight = round(RectHeight(r) * ratio);
	}
	else {
		newHeight = RectHeight(r);
		ratio = (float)RectWidth(rMatch) / (float)RectHeight(rMatch);
		newWidth = round(RectWidth(r) * ratio);
	}
	wCenter = (r.left + r.right) / 2;
	hCenter = (r.top + r.bottom) / 2;
	MySetRect(&r, wCenter - newWidth / 2, hCenter - newHeight / 2,
				  wCenter + newWidth / 2, hCenter + newHeight / 2);
	
	return r;
}


#ifdef MAC

typedef struct
{
	short format;
	short version;
	char paramStrings[]; // the strings are variable length pascal
	// char radius[?], baseLat[?];				
	// char latTop[?], longLeft[?], latBottom[?], longRight[?];				
	// char pixTop[?], pixLeft[?], pixBottom[?], pixRight[?];
} MercParamsResRec;

#define GUESSATSIZE (8 * (5 + 6)) + sizeof(MercParamsResRec) + 32

OSErr AddpPrmResource(short vRefNum, long dirID, char *fileName, WorldRect wr)
{
	MercParamsResRec **params;
	long resRefNum;
	char *strings;
	float f;
	
	OSErr err = 0;
	
	hcreateresfile(vRefNum, dirID, fileName);
	resRefNum = hopenresfile(vRefNum, dirID, fileName, fsCurPerm);
	if (resRefNum == -1) { TechError("AddpPrmResource()", "hopenresfile()", -1); return -1; }
	
	if (params = (MercParamsResRec **)Get1Resource('pPrm', 0)) {
		if (!(err = ResError())) {
			RemoveResource((Handle)params); // remove the old resource
			if (!(err = ResError())) {
				UpdateResFile(resRefNum); // actually remove
				err = ResError();
			}
			DisposeHandle((Handle)params);
			if (!err) err = _MemError();
		}
	}
	
	if (!err) { // overallocate for safety
		params = (MercParamsResRec **)_NewHandle(GUESSATSIZE);
		err = _MemError();
	}
		
	if (!err) {
		_HLock((Handle)params);
		
		(**params).version = (**params).format = 1;
		strings = (**params).paramStrings;
		strcpy(strings, (char *)"\p6378.21\0");	// radius of Earth
		strings += strings[0] + 1;
		
		f = wr.hiLat + wr.loLat;
		f /= 2;
		f = f / 1000000.0; // base lat
		sprintf(strings, "%f", f);
		my_c2pstr(strings);
		strings += strings[0] + 1;
		
		f = (float)wr.hiLat / 1000000.0; // global top
		sprintf(strings, "%f", f);
		my_c2pstr(strings);
		strings += strings[0] + 1;
		
		f = -(float)wr.hiLong / 1000000.0; // global left // backwards
		sprintf(strings, "%f", f);
		my_c2pstr(strings);
		strings += strings[0] + 1;
		
		f = (float)wr.loLat / 1000000.0; // global bottom
		sprintf(strings, "%f", f);
		my_c2pstr(strings);
		strings += strings[0] + 1;
		
		f = -(float)wr.loLong / 1000000.0; // global right // backwards
		sprintf(strings, "%f", f);
		my_c2pstr(strings);
		strings += strings[0] + 1;
		
		f = (float)wr.hiLat / 1000000.0; // local top ???
		sprintf(strings, "%f", f);
		my_c2pstr(strings);
		strings += strings[0] + 1;
		
		f = -(float)wr.hiLong / 1000000.0; // local left ??? // backwards
		sprintf(strings, "%f", f);
		my_c2pstr(strings);
		strings += strings[0] + 1;
		
		f = (float)wr.loLat / 1000000.0; // local bottom ???
		sprintf(strings, "%f", f);
		my_c2pstr(strings);
		strings += strings[0] + 1;
		
		f = -(float)wr.loLong / 1000000.0; // local right ??? // backwards
		sprintf(strings, "%f", f);
		my_c2pstr(strings);
		strings += strings[0] + 1;

		_SetHandleSize((Handle)params, (long)strings - (long)&(*params)->format);
		if (!(err = _MemError())) {
			AddResource((Handle)params, 'pPrm', 0, "\psMerc");
			if (!(err = ResError())) {
				UpdateResFile(resRefNum);
				err = ResError();
			}
		}
		
		_HUnlock((Handle)params);
	}
	CloseResFile(resRefNum);
	
	if (err) printError(GS1(150, 0)); // Technical error while adding 'pPrm' resource to PICT file.
	
	return err;
}

OSErr GetpPrmResource(short vRefNum, long dirID, char *fileName, WorldRect *wr)
{
	short rRefNum;
	float f;
	Ptr p;
	Handle pprm = 0;
	
	OSErr err = 0;
	
	if ((rRefNum = hopenresfile(vRefNum, dirID, fileName, fsCurPerm)) != -1) {
		pprm = (Handle)getnamedresource('pPrm', "sMerc");
		if (pprm) {
			_HLock(pprm);
			p = *pprm + 4;	// skip format, version
			p += *p + 1;	// skip radius string
			p += *p + 1;	// skip baseLat string
			my_p2cstr((unsigned char *)p); sscanf(p, "%f", &f); wr->loLat = f * 1000000; p += strlen(p) + 1;
			my_p2cstr((unsigned char *)p); sscanf(p, "%f", &f); wr->loLong = -f * 1000000; p += strlen(p) + 1;
			my_p2cstr((unsigned char *)p); sscanf(p, "%f", &f); wr->hiLat = f * 1000000; p += strlen(p) + 1;
			my_p2cstr((unsigned char *)p); sscanf(p, "%f", &f); wr->hiLong = -f * 1000000;
			_HUnlock(pprm);
			ReleaseResource(pprm);
			if (wr->loLong > wr->hiLong) SwitchLongs(&wr->loLong, &wr->hiLong);
			if (wr->loLat > wr->hiLat) SwitchLongs(&wr->loLat, &wr->hiLat);
		}
		else
			err = -1;
		CloseResFile(rRefNum);
	}
	else
		err = -1;
	
	return err;
}
///////////////////////////////////////////////////////////////////////////////////////

OSErr MyWriteResource (OSType thisResType, long thisResID, Handle thisResHdl, short resFRefNum)
// this subroutine attempts to preserve restores the currently active resource file once it is
// done.  Note: it doesn't close the resource file after writing the resource.
{
	short	SaveResFile;
	char	ResName [256];
	OSErr	ErrCode = 0;
	Handle	dummyResHdl;
		
	/* create nil name for resource */
	ResName [0] = '\0';
	
	SaveResFile = CurResFile ();
	
	if (resFRefNum)
	{
		UseResFile (resFRefNum);
		
		/* remove existing resource if any */
		dummyResHdl = Get1Resource (thisResType, thisResID);
		if (dummyResHdl)
		{
			RemoveResource (dummyResHdl);
			DisposeHandle (dummyResHdl);
		}
		
		AddResource (thisResHdl, thisResType, thisResID, (StringPtr) ResName);
		ErrCode = ResError ();
		if (!ErrCode)
		{
			WriteResource (thisResHdl);
			ErrCode = ResError ();
			if (!ErrCode)
			{
				ReleaseResource (thisResHdl);
				ErrCode = ResError ();
			}
		}
	}
	else
		ErrCode = 1;

	UseResFile (SaveResFile);
	
	return (ErrCode);
}
#endif

///////////////////////////////////////////////////////////////////////////////////////

#ifdef MAC

#pragma segment Main

#if TARGET_API_MAC_CARBON
short MenuBarEquivalentID(short menuID) // this compensates for the Carbonized MAC version having two different menus, one for the popup and one for the menu bar
{
	// Note: The IBM and pre-CARBON use the same value for phCOLORS and phCOLORS_POPUP
	// and use a macro to define MenuBarEquivalentID(x) to be x
	/*switch(menuID) {
		case phCOLORS_POPUP:  return phCOLORS;
		case phLINEWIDTH_POPUP:  return phLINEWIDTH;
		case phLINEPATTERN_POPUP:  return phLINEPATTERN;
		case phFILLPATTERN_POPUP:  return phFILLPATTERN;
		case phSYMBOL_POPUP:  return phSYMBOL; 
	}*/
	
	return menuID;
}
#endif

void NewSAMenu_SetTextToEmptyForAllMenuItems(short menuID)
{
	// the resource has descriptive text which is not supposed to show when the user sees the menu item
	MenuRef menu = GetMenuHandle(menuID);
	short i, numItems =  CountMenuItems(menu);
	
	for(i = 1; i <= numItems; i++) {
		SetMenuItemText(menu, i, (StringPtr)"");
	}
	
}

/////////////////////////////////////////////////

short NewSAMenu_ExtraMenuWidth(short menuID)
{
	menuID = MenuBarEquivalentID(menuID); // this compensates for the Carbonized MAC version having two different menus, one for the popup and one for the menu bar
	
	switch(menuID) {
		//case phCOLORS:
		case pBACKGROUNDCOLORS:
			return 29;
			
		/*case phFILLPATTERN:
		case phLINEPATTERN:
		case phLINEWIDTH:
		case phDOTSIZE:
			return 41;
			
		case phSYMBOL:
			return 22;*/
			
	}
	
	return 0;
}
	
/////////////////////////////////////////////////

void NewSAMenu_DrawExtra(short menuID, short item, Rect extraRect)
{
	short w, y;
	//short widths[] = { 1, 2, 4, 6, 8, 10 }; // FATLINES
	RGBColor saveColor;
	MyGetForeColor(&saveColor);
	///
	
	menuID = MenuBarEquivalentID(menuID); // this compensates for the Carbonized MAC version having two different menus, one for the popup and one for the menu bar
	
	
	switch(menuID) {
		//case phCOLORS:
		case pBACKGROUNDCOLORS:
			MyInsetRect(&extraRect,3,3);
			//if(menuID == phCOLORS) MyRGBForeColor(&colors[item]);
			/*else*/ MyRGBForeColor(&backColors[item]);
			MyPaintRect(&extraRect);
			MyForeColor(blackColor);
			FrameRect(&extraRect);
			break;

	//	case phFILLPATTERN:
	/*	case phLINEPATTERN:
			MyInsetRect(&extraRect,3,3);
			// let's always  fill the inside with white
			MyForeColor(whiteColor); MyPaintRect(&extraRect);
			// let's alway draw the pattern in black
			MyForeColor(blackColor); 
			if (menuID == phLINEPATTERN  && item == 10) { // railroad
				short centerV = extraRect.top + RectHeight(extraRect) / 2;
				short i;
				MyMoveTo(extraRect.left, centerV);
				MyLineTo(extraRect.right - 1, centerV);
				for (i = extraRect.left ; i < extraRect.right ; i += 5) {
					MyMoveTo(i, centerV);
					MyLine(0, -2); MyLine(0, 4); MyLine(0, -2);
				}
			}
			else {
				Pattern	pat;
				GetIndPattern(&pat, PATTERNS, item);
				MyFillRect(&extraRect, pat);
			}
			MyFrameRect(&extraRect);
			break;

		case phLINEWIDTH:
		case phDOTSIZE:
		{
			MyInsetRect(&extraRect,3,3);
			// let's alway draw the line in black
			MyForeColor(blackColor); 
			//w = widths[item - 1];
			w = FatLineWidthAdjustment(item); // note: item is 1-relative and so is the same as the "assigned" lineWidth
			MyPenSize(w, w);
			y = extraRect.top + (extraRect.bottom - extraRect.top - w) / 2;
			if(menuID == phLINEWIDTH) {
				MyMoveTo(extraRect.left, y);
				MyLineTo(extraRect.right - w, y);
			} else {
				MyMoveTo((extraRect.left + extraRect.right - w) / 2, y);
				MyLineTo((extraRect.left + extraRect.right - w) / 2, y);
			}
			break;
		}
		
		case phSYMBOL:
		{
			short fontNum,w,h,v;
			char ch = (char)item;
			getfnum(currentAlternateFontName[0] ?
						currentAlternateFontName : "MARPLOT Font", &fontNum);
			TextFont(fontNum);
			TextSize(18); // 12
			w = CharWidth(ch);
			h = (extraRect.left+extraRect.right -w)/2;
			v = extraRect.bottom - 6;
			MyMoveTo(h,v);
			DrawChar(item);
			break;
		}*/

		
		
	}
	///
	MyRGBForeColor(&saveColor);
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
void DrawColorItem(MenuHandle theMenu, short item, Rect *r, char bPopBox,
				   /*const SysEnvRec *theWorld,*/ const RGBColor *DefaultFore,
				   const RGBColor *DefaultBack)
{
	Rect colorRect;
	char name[256];
	short markChar;
#pragma unused(theMenu, DefaultBack)
	
	// draw the item name
	MyMoveTo((*r).left + 40, ((*r).bottom + (*r).top)/2 + 5);
	GetMenuItemText(theMenu, item, (unsigned char *)name);
	my_p2cstr((unsigned char *)name);
	drawstring(name);
	
	// draw the color block
	colorRect.top =    (*r).top + 3;
	colorRect.bottom = (*r).bottom - 3;
	colorRect.left =   (*r).left + 14;
	colorRect.right =  colorRect.left + 23;
	
	//if (theWorld->hasColorQD)
	if (ColorQDAvailable())
		RGBForeColor(&colors[item]);
	else
		ForeColor(item == 2 ? whiteColor : blackColor);
	PaintRect(&colorRect);
	
	//if (theWorld->hasColorQD)
	if (ColorQDAvailable())
		RGBForeColor(&colors[1]);
	else
		ForeColor(blackColor);
	FrameRect(&colorRect);
	
	
	//if (theWorld->hasColorQD) RGBForeColor(DefaultFore);
	if (ColorQDAvailable()) RGBForeColor(DefaultFore);

	// draw a ">" if this item is hierarchical
	
	GetItemMark(theMenu, item, &markChar);
	if (markChar != kCheckMark && markChar != noMark) { // assume any other mark is
														// a hierarchical menu number
		MyMoveTo((*r).right - kCharWidth, ((*r).bottom + (*r).top)/2 + 5);
		DrawChar('>');
	}
	
	// check checked items, but not in the pop box
	
	if (!bPopBox)
		if (markChar == kCheckMark)
			DrawCheck(r);
}

void DrawBackColorItem(MenuHandle theMenu, short item, Rect *r, char bPopBox,
					   /*const SysEnvRec *theWorld,*/ const RGBColor *DefaultFore,
					   const RGBColor *DefaultBack)
{
	Rect colorRect;
	char name[256];
	short markChar;
#pragma unused(theMenu, DefaultBack)
	
	// draw the item name
	MyMoveTo((*r).left + 40, ((*r).bottom + (*r).top)/2 + 5);
	GetMenuItemText(theMenu, item, (unsigned char *)name);
	DrawString((unsigned char *)name);
	
	// draw the color block
	colorRect.top =    (*r).top + 3;
	colorRect.bottom = (*r).bottom - 3;
	colorRect.left =   (*r).left + 14;
	colorRect.right =  colorRect.left + 23;
	
	if (ColorQDAvailable())
	//if (theWorld->hasColorQD)
		RGBForeColor(&backColors[item]);
	else
		ForeColor(item == 2 ? whiteColor : blackColor);
	PaintRect(&colorRect);
	
	if (ColorQDAvailable())
	//if (theWorld->hasColorQD)
		RGBForeColor(&colors[1]);
	else
		ForeColor(blackColor);
	FrameRect(&colorRect);
	
	if (ColorQDAvailable()) RGBForeColor(DefaultFore);
	//if (theWorld->hasColorQD) RGBForeColor(DefaultFore);

	// draw a ">" if this item is hierarchical
	
	GetItemMark(theMenu, item, &markChar);
	if (markChar != kCheckMark && markChar != noMark) { // assume any other mark is
														// a hierarchical menu number
		MyMoveTo((*r).right - kCharWidth, ((*r).bottom + (*r).top)/2 + 5);
		DrawChar('>');
	}
	
	// check checked items, but not in the pop box
	
	if (!bPopBox)
		if (markChar == kCheckMark)
			DrawCheck(r);
}

void InvertColorItem(MenuHandle theMenu, short item, Rect *r, char bOn,
					 /*const SysEnvRec *theWorld,*/ const RGBColor *DefaultFore,
					 const RGBColor *DefaultBack)
{
	char name[256];
	short markChar;
	RGBColor saveHilite;
#pragma unused(DefaultBack)
	
	r->left += 40;
	EraseRect(r);
	MyMoveTo((*r).left, ((*r).bottom + (*r).top)/2 + 5);
	GetMenuItemText(theMenu, item, (unsigned char *)name);
	DrawString((unsigned char *)name);
	
	GetItemMark(theMenu, item, &markChar);
	if (markChar != kCheckMark && markChar != noMark) { // assume any other mark is
														// a hierarchical menu number
		MyMoveTo((*r).right - kCharWidth, ((*r).bottom + (*r).top)/2 + 5);
		DrawChar('>');
	}

	if (bOn) {		// if we're on a Mac Plus, use (non-hiliting) InvertRect()
		//if (!theWorld->hasColorQD) {
		if (!(ColorQDAvailable())) {
			InvertRect( r );
			return;
		}
	
		// save the hilite color
		LMGetHiliteRGB(&saveHilite);
		
		HiliteColor(DefaultFore);
		LMSetHiliteMode(LMGetHiliteMode() & 0x7F); // clear 0th bit = pHiliteBit
		MyInvertRect( r );
		
		// restore the hilite color
		HiliteColor(&saveHilite);
	}
}

#else

void DrawColorItem(MenuHandle theMenu, short item, Rect *r, char bPopBox,
				   /*const SysEnvRec *theWorld,*/ const RGBColor *DefaultFore,
				   const RGBColor *DefaultBack) {}
void DrawBackColorItem(MenuHandle theMenu, short item, Rect *r, char bPopBox,
				   /*const SysEnvRec *theWorld,*/ const RGBColor *DefaultFore,
				   const RGBColor *DefaultBack) {}
void InvertColorItem(MenuHandle theMenu, short item, Rect *r, char bOn,
					 /*const SysEnvRec *theWorld,*/ const RGBColor *DefaultFore,
					 const RGBColor *DefaultBack) {}

#endif
