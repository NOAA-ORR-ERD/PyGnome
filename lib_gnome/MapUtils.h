
#ifndef __MYMAPUTILS__
#define __MYMAPUTILS__

#include "Basics.h"
#include "TypeDefs.h"
#include "RectUtils.h"

#define				kWorldTop				89999999
#define				kWorldBottom 		   -89999999
//#define				kWorldLeft			   -179999999
//#define				kWorldRight				179999999
#define				kWorldLeft			   -359999999
#define				kWorldRight				359999999

#define				kMaxMapInt				32760
#define				kMinMapInt			   -32760
#define				kMaxLongInt				2000000000
#define			 	kMinLongInt			   -2000000000

#define				kOutOfRangeErr		   -100					/* map coordinate out of range error */

#define				kArrowHang				150 * PI / 180
#define				kHeadRatio				0.25
#define				kNumDeciPlaces			6					/* num-of-decimal places in map coordinates */

#define				kDMUnitCode				0		/* degrees and decimal minutes code */
#define				kDMSUnitCode			1		/* degrees, mins, and secs code */
#define				kDecDegUnitCode			2		/* decimal degrees code */

#define				kMercatorProjCode		1		/* code for mercator projection */
#define				kLatLongProjCode		2		/* code for lat/long projection */



#define SameDifferenceX(V) (SameDifference(AX, BX, V, aX, bX) + DX)
#define SameDifferenceY(V) (SameDifference(AY, BY, V, aY, bY) + DY)

OSErr	GetScaleAndOffsets (Rect *SourceRectPtr, Rect *DestRectPtr, ScaleRecPtr ScaleInfoPtr);
void	GetScrMidPoint (Point scrPoint1, Point scrPoint2, Point *theMidPoint);
//OSErr 	ScanMatrixPt (char *startChar, LongPoint *MatrixLPtPtr);
void 	PlotVector (long scrX, long scrY, double *vectorUvel, double *vectorVvel, long MilePixels);
void 	GetDegMin (long MatrixLong, long *degPtr, double *minPtr);
void 	RoundDMS (long MatrixDeg, long *LongDeg, double *ExMins);
void 	DegreesToDM (double degrees, long *DegreesPtr, double *MinPtr);
void 	GetLLStrings (LongPoint *LPointPtr, long UnitCode, char *theStrings);
void 	GetLLString (LongPoint *LPointPtr, long UnitCode, char *LatStr, char *LongStr);
OSErr 	SelectZoomPoint (Point LocalPt, Rect *WindowRectPtr, Rect* ScrSelRectPtr);
Boolean LinesCross (LongPointPtr LongPtPtr1, LongPointPtr LongPtPtr2,
					LongPointPtr LongPtPtr3, LongPointPtr LongPtPtr4);
void 	SetLPoint (LongPoint *LPtPtr, long h, long v);
long 	GetScrPtDist (Point scrPoint1, Point scrPoint2);
void 	GetLPtsDist (LongPoint *LPt1, LongPoint *LPt2, long unitCode, double *distance);

// JLM 9/7/00 changed from short to a long
//short	SameDifference (long A, long B, long a, long b, long V);
long	SameDifference (long A, long B, long a, long b, long V);
Point 	GetQuickDrawPt(long pLong, long pLat, Rect *r, Boolean *offPlane);

void 	DrawLatLongLines(Rect r, WorldRect view);
void 	DrawLat(long latVal, Rect r, WorldRect view, short precision, Boolean label);
void 	DrawLong(long longVal, Rect r, WorldRect view, short precision, Boolean label);

#endif
