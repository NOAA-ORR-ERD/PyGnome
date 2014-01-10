/*
 *  CompFunctions.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Basics.h"
#include "TypeDefs.h"
#include "CompFunctions.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif


double UorV(VelocityRec vector, short index)
{
	return (index == 0) ? vector.u : vector.v;
}

double UorV(VelocityRec3D vector, short index)
{
	return (index == 0) ? vector.u : vector.v;
}

double Hermite(double v1, double s1, double t1,
			   double v2, double s2, double t2,
			   double time)
{
	double h, dt, x;
	
	// hermite fit function
	// v1 and v2 are the two values
	// s1 and s2 are the two slopes
	// t1 and t2 are the two times
	// time is the time for the interpolation
	// return value is interpolated value at time
	
	dt = t2 - t1;
	if (dt < 0.0) dt = -dt;
	
	if (dt < TIMEVALUE_TOLERANCE) return v2;
	
	x = (time - t1) / (t2 - t1);
	s1 = s1* (t2 - t1);
	s2 = s2* (t2 - t1);
	
	h = (   2.0 * v1 - 2.0 * v2 +       s1 + s2) * x * x * x
	+ ( - 3.0 * v1 + 3.0 * v2 - 2.0 * s1 - s2) * x * x
	+ (                       +       s1     ) * x
	+ (         v1                           ) * 1.0;
	
	return h;
}

// ************************************************************
void Hermite(double v1,     // value at t1
             double s1,     // slope at t1
			 double t1,     // time t1
			 double v2,     // value at t2
			 double s2,     // slope at t2
			 double t2,     // time t2
			 double theTime,   // time for interpolation
			 double *vTime) // returns value at time
{	
	// Hermite interpolation function taken from OSSM
	double x,dt;
	/**************************************************************/
	//
	// check to make sure t1 and t2 are not the same
	//
	dt = t2 - t1;
	if( dt<0.000001){
		*vTime = v2;
		return;
	}
	//
	dt = t2-t1;
	x = (theTime-t1) / dt;
	s1 = s1 * dt;
	s2 = s2 * dt;
	
	// OK here is the 4 by 4
	
	*vTime = ( 2.0 * v1 - 2.0*v2 +    s1 + s2) * x*x*x
	+(-3.0 * v1 + 3.0*v2 -2.0*s1 - s2) * x*x
	+(                        s1     ) * x
	+(       v1                      );
	return;
	
	// note in the OSSM code Gary sets back the values of s1 and s2
	// we don't do that in C because those suckers are passed as values
}

//++ The following have been moved from GenUtil,


float max4(float f1, float f2, float f3, float f4)
{
	float x = _max(f1, f2), y = _max(f3, f4);
	
	return _max(x, y);
}

float min4(float f1, float f2, float f3, float f4)
{
	float x = _min(f1, f2), y = _min(f3, f4);
	
	return _min(x, y);
}

double logB(double b, double x)
{
	return log10(x) / log10(b);
}

float hypotenuse(float a, float b)
{
	return sqrt(a*a + b*b);
}

double myfabs(double x)
{
	return (x < 0) ? -x : x;
}


void SetSign(FLOATPTR n, short code)
{
	if (*n == -0.0)
		*n = 0.0;
	else {
		*n = fabs(*n);
		if (code == 2) *n = -*n;
	}
}

short ScaleToShort(long n)
{
	if (n < -32000) return -32000;
	if (n > 32000) return 32000;
	return n;
}

long GetRandom(long low, long high)
{
	float scale, n;
	
	scale = (float)(high - low) / (float)RAND_MAX;
	
	n = low + rand() * scale;
	
	return (long)n;
}

float GetRandomFloat(float low, float high)
{
	float scale, n;
	
	scale = (float)(high - low) / (float)RAND_MAX;
	
	n = low + rand() * scale;
	
	return n;
}

void GetRandomVectorInUnitCircle(float *u,float *v)
{	// JLM 9/11/98
	do
	{
		*u = GetRandomFloat(-1.0,1.0);
		*v = GetRandomFloat(-1.0,1.0);
	} while ( (*u)*(*u) +  (*v)*(*v) > 1.0);
}


char *SwapN(char *s, short n)
{
	char c;
	short i;
	
	for (i = 0 ; i < n / 2 ; i++) {
		c = s[i];
		s[i] = s[n - (i + 1)];
		s[n - (i + 1)] = c;
	}
	
	return s;
}

// Given an integer key, a list of pairs of integers, and the
// number of pairs in the list, return the integer paired with the key,
// or 0 if the key is not in the list.
long Assoc(long key, LONGPTR table, short n)
{
	short i;
	
	for (i = 0 ; i < n ; i++)
		if (table[i * 2] == key)
			return table[i * 2 + 1];
	
	return 0;
}

void SwitchShorts(SHORTPTR a, SHORTPTR b)
{
	short temp = *a;
	
	*a = *b;
	*b = temp;
}

void SwitchLongs(LONGPTR a, LONGPTR b)
{
	long temp = *a;
	
	*a = *b;
	*b = temp;
}

void SwitchStrings(CHARPTR a, CHARPTR b)
{
	char temp[256];
	
	strcpy(temp, a);
	strcpy(a, b);
	strcpy(b, temp);
}

short NumDecimals(CHARPTR str)
{
	char		str2[256];
	short		i, c = 0;
	
	strcpy(str2, str);
	i = strlen(str2) - 1;
	while (str2[i] == '0')
		str2[i--] = 0;
	
	while (i >= 0) {
		if (str2[i--] == '.')
			return c;
		c++;
	}
	
	return 0;
}

Boolean EarlierThan(Seconds time1, Seconds time2)
{
	return ((unsigned long)time1) < ((unsigned long)time2);
}

Boolean LaterThan(Seconds time1, Seconds time2)
{
	return ((unsigned long)time1) > ((unsigned long)time2);
}

// Densities in gm/cm^3
double GetPollutantDensity(OilType num)
{
	double density=1;
	switch(num)
	{
		case OIL_GAS:density=.75; break;
		case OIL_JETFUELS: density = .81; break;
		case OIL_DIESEL: density=.87; break;
		case OIL_4: density=.90; break;
		case OIL_CRUDE: density = .90;break;
		case OIL_6: density = .99; break;
		case OIL_CONSERVATIVE: density=1;break;
		case CHEMICAL: 	density=1; break;	// will need to override calls with input value, or calculated value
		default:printError("Pollutant type not found");
	}
	return density;
}



static void AssertionCheckMassUnits(short massUnits)
{
	Boolean b = (massUnits == KILOGRAMS || massUnits == METRICTONS ||massUnits == SHORTTONS );
	char str[256];
	if(!b) 
	{
		sprintf(str,"AssertionCheckMassUnits failed: massUnits = %d",massUnits);
		printError(str);
	}
}

static void AssertionCheckVolUnits(short volUnits)
{
	Boolean b = (volUnits == GALLONS || volUnits == BARRELS ||volUnits == CUBICMETERS );
	char str[256];
	if(!b) 
	{
		sprintf(str,"AssertionCheckVolUnits failed: volUnits = %d",volUnits);
		printError(str);
	}
}

static void AssertionCheckMassVolUnits(short massVolUnits)
{
	Boolean b = (massVolUnits == GALLONS || massVolUnits == BARRELS ||massVolUnits == CUBICMETERS
				 || massVolUnits == KILOGRAMS||massVolUnits == METRICTONS||massVolUnits == SHORTTONS);
	char str[256];
	if(!b) 
	{
		sprintf(str,"AssertionCheckMassVolUnits failed: massVolUnits = %d",massVolUnits);
		printError(str);
	}
}

double CM3ToVolumeMass(double val, double density,short massVolUnits)
{
	double retval = -1;
	
	AssertionCheckMassVolUnits(massVolUnits);
	
	switch(massVolUnits)
	{
		case KILOGRAMS:
		case METRICTONS:
		case SHORTTONS:
			retval = ConvertGramsToMass(ConvertCM3ToGrams(val,density),massVolUnits);
			break;
		case GALLONS:
		case BARRELS:
		case CUBICMETERS:
			retval = ConvertCM3ToVol(val,massVolUnits);
	}
	return retval;
}

double VolumeMassToCM3(double val, double density , short massVolUnits)
{
	double retval = -1;
	
	AssertionCheckMassVolUnits(massVolUnits);
	
	switch(massVolUnits)
	{
		case KILOGRAMS:
		case METRICTONS:
		case SHORTTONS:
			retval = ConvertGramsToCM3(ConvertMassToGrams(val,massVolUnits),density);
			break;
		case GALLONS:
		case BARRELS:
		case CUBICMETERS:
			retval = ConvertVolToCM3(val,massVolUnits);
			break;
	}
	return retval;
}

double VolumeMassToKilograms(double val, double density , short massVolUnits)
{
	double retval = -1;
	
	AssertionCheckMassVolUnits(massVolUnits);
	
	switch(massVolUnits)
	{
		case KILOGRAMS:
		case METRICTONS:
		case SHORTTONS:
			retval = ConvertMassToGrams(val,massVolUnits)/1000.0;
			break;
		case GALLONS:
		case BARRELS:
		case CUBICMETERS:
			retval = ConvertCM3ToGrams(ConvertVolToCM3(val,massVolUnits),density)/1000.0;
			break;
	}
	return retval;
}

double VolumeMassToGrams(double val, double density , short massVolUnits)
{
	double retval = -1;
	
	AssertionCheckMassVolUnits(massVolUnits);
	
	switch(massVolUnits)
	{
		case KILOGRAMS:
		case METRICTONS:
		case SHORTTONS:
			retval = ConvertMassToGrams(val,massVolUnits);
			break;
		case GALLONS:
		case BARRELS:
		case CUBICMETERS:
			retval = ConvertCM3ToGrams(ConvertVolToCM3(val,massVolUnits),density);
			break;
	}
	return retval;
}

double VolumeMassToVolumeMass(double val, double density , short massVolUnits, short desiredMassVolUnits)
{
	double retval = -1;
	
	AssertionCheckMassVolUnits(massVolUnits);
	AssertionCheckMassVolUnits(desiredMassVolUnits);
	
	if(massVolUnits == desiredMassVolUnits) return val;
	
	retval = CM3ToVolumeMass(VolumeMassToCM3(val,density,massVolUnits),density,desiredMassVolUnits);
	return retval;
}


double ConvertMassToGrams(double val, short massUnits)
{
	double retval=-1;
	
	AssertionCheckMassUnits(massUnits);
	
	switch(massUnits)
	{
		case KILOGRAMS:retval = val * 1000;break;
		case METRICTONS:retval = val * 1000000;break;
		case SHORTTONS:retval = val * 907185;break;
	}
	return retval;
}


double ConvertGramsToMass(double val, short massUnits)
{
	double retval=-1;
	
	AssertionCheckMassUnits(massUnits);
	
	switch(massUnits)
	{
		case KILOGRAMS:retval = val /1000;break;
		case METRICTONS:retval = val / 1000000;break;
		case SHORTTONS:retval = val / 907185;break;
	}
	return retval;
}

double ConvertVolToCM3(double val, short VolUnits)
{
	double retval=-1;
	
	AssertionCheckVolUnits(VolUnits);
	
	switch(VolUnits)
	{
		case GALLONS:retval = val * 3785.41;break;
		case BARRELS:retval = val * 158987;break;
		case CUBICMETERS:retval = val* 1000000;break;
	}
	return retval;
}

double ConvertCM3ToVol(double val,short VolUnits)
{
	double retval=-1;
	
	AssertionCheckVolUnits(VolUnits);
	
	switch(VolUnits)
	{
		case GALLONS:retval = val / 3785.41;break;
		case BARRELS:retval = val / 158987;break;
		case CUBICMETERS:retval = val/ 1000000;break;
	}
	return retval;
}

//density is assumed to be in grams/cm^3
double ConvertGramsToCM3(double val,double density)
{
	return val /density;
}

double ConvertCM3ToGrams(double val,double density)
{
	return val*density;
}


double GetLEMass(LERec theLE, double halfLife)	// AH 06/20/2012
{
#ifndef pyGNOME

	long i;
	double tHours, fracLeft = 0.;
	TWeatherer	*thisWeatherer;
	OilComponent	component;
	CMyList	*weatherList = model->GetWeatherList();
	
	if (theLE.pollutantType == CHEMICAL)
	{
		//weatherList->GetListItem((Ptr)&thisWeatherer, 0);	// assume there is only one
		//((dynamic_cast<TOSSMWeatherer*>(thisWeatherer)))->componentsList -> GetListItem ((Ptr) &component, theLE.pollutantType - 1);

		tHours = (double) ( model -> GetModelTime () - theLE.releaseTime)/ 3600.0  + theLE.ageInHrsWhenReleased;

		// if LE has not been released yet return 0?
		//if (theLE.releaseTime > model->GetModelTime()) return theLE.mass;
		if (theLE.releaseTime > model->GetModelTime()) return 0;
		
		/*for(i = 0;i<3;i++)	// at this point only using 1 half life component
		{
			if(component.percent[i] > 0.0)
			{
				fracLeft +=  (component.percent[i])*pow(0.5,tHours/(component.halfLife[i]));
			}
		}*/
		fracLeft = pow(.5,tHours/halfLife);
		
		fracLeft = _max (0.0,fracLeft);
		fracLeft = _min (1.0,fracLeft);
		return fracLeft*theLE.mass;
	}
	else
		return theLE.mass;
#else
	return theLE.mass;
#endif
}

/**************************************************************************************************/
OSErr ScanDepth (char *startChar, double *DepthPtr)
{	// expects a single depth value 
	long	j, k;
	char	num [64];
	OSErr	err = 0;
	
	j = 0;	/* index into supplied string */
	
	Boolean keepGoing = true;
	for (k = 0 ; keepGoing; j++)
	{	   			
		switch(startChar[j])
		{
			case 0: // end of string
				keepGoing = false;
				break;
			case '.':
				num[k++] = startChar[j];
				break;
			case '-': // depths can't be negative but -1 is end of data flag
			case '0': case '1': case '2': case '3': case '4':
			case '5': case '6': case '7': case '8': case '9':
				num[k++] = startChar[j];
				if(k>=32) // end of number not found
				{
					err = -1;
					return err;
				}
				break;
			case ' ':
			case '\t': // white space
				if(k == 0) continue; // ignore leading white space
				while(startChar[j+1] == ' ' || startChar[j+1] == '\t') j++; // move past any additional whitespace chars
				keepGoing = false;
				break;
			default:
				err = -1;
				return err;
				break;
		}
	}
	
	num[k++] = 0;									/* terminate the number-string */
	
	*DepthPtr = atof(num);
	///////////////
	
	return err;
	
}

#ifndef pyGNOME
// - expects a number of the form
//     <number><comma><number>
// - parses the number values and scales them
//   up six decimal places.
// - Populates the LongPoint h and v values
// example:
//      Incoming string: "-120.2345,40.345"
//      LongPoint out: (-120234500, 40345000)
// JLM, 2/26/99 extended to handle
// <number><whiteSpace><number>
OSErr ScanMatrixPt (char *startChar, LongPoint *MatrixLPtPtr)
{
	long	deciPlaces, j, k, pairIndex;
	char	num [64];
	OSErr	ErrCode = 0;
	char errStr[256]="";
	char delimiterChar = ',';
	
	j = 0;	/* index into supplied string */
	
	for (pairIndex = 1; pairIndex <= 2 && !ErrCode; ++pairIndex)
	{
		/* first convert the longitude */
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
			if (deciPlaces == -1)						/* if decimal point was not encountered */
				deciPlaces = 0;
			
			do
			{
				num[k++] = '0';
				++deciPlaces;
			}
			while (deciPlaces < kNumDeciPlaces);
		}
		
		num[k++] = 0;									/* terminate the number-string */
		
		if (pairIndex == 1)
		{
			MatrixLPtPtr -> h = atol(num);
			
			if (startChar[j] == ',')					/* increment j past the comma to next coordinate */
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
#endif

void CheckYear(short *year)
{
	if (*year < 1900) {
		// two digit date, so fix it
		if (*year >= 40 && *year <= 99)
			*year += 1900;
		else
			*year += 2000; // correct for year 2000 (00 to 40)
	}
}

