/*
 *  CompFunctions.h
 *  gnome
 *
 *  Created by Generic Programmer on 1/11/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef __CompFunctions__
#define __CompFunctions__

#include "Basics.h"
#include "TypeDefs.h"

#ifdef pyGNOME
#define PtCurMap PtCurMap_c
#endif

class PtCurMap;

double UorV(VelocityRec vector, short index);
double UorV(VelocityRec3D vector, short index);
double Hermite(double v1, double s1, double t1,
			   double v2, double s2, double t2,
			   double time);
void Hermite(double v1,     // value at t1
             double s1,     // slope at t1
			 double t1,     // time t1
			 double v2,     // value at t2
			 double s2,     // slope at t2
			 double t2,     // time t2
			 double theTime,   // time for interpolation
			 double *vTime); // returns value at time
float max4(float f1, float f2, float f3, float f4);
float min4(float f1, float f2, float f3, float f4);
double logB(double b, double x);
float hypotenuse(float a, float b);
double myfabs(double x);
void SetSign(FLOATPTR n, short code);
short ScaleToShort(long n);
long GetRandom(long low, long high);
float GetRandomFloat(float low, float high);
void GetRandomVectorInUnitCircle(float *u,float *v);
char *SwapN(char *s, short n);
long Assoc(long key, LONGPTR table, short n);
void SwitchShorts(SHORTPTR a, SHORTPTR b);
void SwitchLongs(LONGPTR a, LONGPTR b);
void SwitchStrings(CHARPTR a, CHARPTR b);
short NumDecimals(CHARPTR str);
Boolean EarlierThan(Seconds time1, Seconds time2);
Boolean LaterThan(Seconds time1, Seconds time2);
double 			GetPollutantDensity(OilType num);
double VolumeMassToVolumeMass(double val, double density , short massVolUnits, short desiredMassVolUnits);

double ConvertCM3ToVol(double val,short VolUnits);
double ConvertVolToCM3(double val, short volUnits);
double ConvertGramsToMass(double val, short massUnits);
double ConvertMassToGrams(double val, short massUnits);
double ConvertGramsToCM3(double val,double density);
double ConvertCM3ToGrams(double val,double density);
double VolumeMassToCM3(double val, double density , short massVolUnits);
double VolumeMassToKilograms(double val, double density , short massVolUnits);
double VolumeMassToGrams(double val, double density , short massVolUnits);
double CM3ToVolumeMass(double val, double density,short massVolUnits);
double GetLEMass(LERec theLE, double halfLife);
//Boolean EqualUniqueIDs(UNIQUEID uid,UNIQUEID uid2);
//OSErr ScanMatrixPt (char *startChar, LongPoint *MatrixLPtPtr);
OSErr ScanVelocity (char *startChar, VelocityRec *VelocityPtr, long *scanLength);
OSErr ScanDepth (char *startChar, double *DepthPtr);
void CheckYear(short *year);
PtCurMap* CreateAndInitPtCurMap(char *path, WorldRect bounds);



#undef PtCurMap
#endif
