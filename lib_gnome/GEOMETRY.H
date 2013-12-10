
#ifndef __GEOMETRY__
#define __GEOMETRY__

//#define MAC_WORLDRECT_SIZE 16

#define SetVector(v, xVal, yVal) v.x = xVal; v.y = yVal;

////////////////////////////////////////////////////////////////// CONSTANTS

//#define EARTHRADIUSINMETERS 	6371007
#define EARTHRADIUSINMILES 		3958.7602

#define FEETPERDEGREELAT 		364566.92993
#define YARDSPERDEGREELAT 		121522.30998
#define METERSPERDEGREELAT 		111120.00024
#define KILOMETERSPERDEGREELAT 111.12000024
#define MILESPERDEGREELAT	 	69.046767032
#define NAUTSPERDEGREELAT		60.0

#define INCHESTOFEET			.083333333333
#define INCHESTOYARDS			.027777777777
#define CMTOMETERS				.01
#define CMTOKILOMETERS			.00001
#define INCHESTOMILES			.000015782765
#define INCHESTONAUTS			.000013714843

#define MILESTOKILO				1.609344
#define	MILESTONAUTS			0.86897624
#define	MILESTOFEET				5280.0
#define	MILESTOYARDS			1760.0
#define	MILESTOMETERS			1609.344

#define  CENTPERINCH			2.54
#define  INPERMILE				63360
//#define  CMPERMILE				160934.4

enum { FEET = 1, YARDS, METERS, KILOMETERS, MILES, NAUTICALMILES }; // units
//enum { INTERSECTPOLYGONS = 1, UNIONPOLYGONS, DIFFERENCEPOLYGONS };

////////////////////////////////////////////////////////////////// MACROS

#define WRectWidth(wr) ((wr).hiLong - (wr).loLong)
#define WRectHeight(wr) ((wr).hiLat - (wr).loLat)

////////////////////////////////////////////////////////////////// VARIABLES

extern WorldRect theWorld, voidWorldRect, emptyWorldRect;

////////////////////////////////////////////////////////////////// FUNCTIONS

float LongToLatRatio(float baseLat);
//float MilesPerDegreeLong(float baseLat);
float MilesPerDegreeLat();
//float DegreesLongPerMile(float baseLat);
//float DegreesLatPerMile();
float LongToLatRatio2(WorldRect *wr);
float LongToLatRatio3(long baseLat);
//Boolean EqualSegments(Segment s1, Segment s2);
//Boolean EqualSegments2(Segment s1, Segment s2);

float DistanceBetweenWorldPoints(WorldPoint p1, WorldPoint p2);
float EarthRadius();

#ifndef pyGNOME
typedef struct GeoPoint
{ 	// right handed coordinate system
	long				h; 			// Latitude  (dec_deg*10e6)
	long				v;  		// Longitude (dec_deg*10e6)
}GeoPoint,  *GeoPointPtr,  **GeoPointHdl;

typedef struct {
	long fromLong;
	long fromLat;
	long toLong;
	long toLat;
} Segment, *SEGMENTP, **SEGMENTH, **SegmentsHdl;

typedef struct { long x; long y; } Vector;

Boolean SameSegmentEndPoints(Segment s1, Segment s2);
//Point ForcePointOnLineIntoRect(short x, short y, float slope, Rect r);

float UnitsPerDegreeLat();
float LongToDistance(long dLong, WorldPoint center);
float LatToDistance(long dLat);
long DistanceToLong(float distance, WorldPoint center);
long DistanceToLat(float distance);
//float ComputePolygonArea(long numPoints, PointH points, WorldRect bounds);
//float ComputeWorldRectArea(WorldRect bounds);

Boolean WPointNearWPoint(WorldPoint p1, WorldPoint p2, float d);
/*Boolean WPointNearWPoint2(long long1, long lat1, long long2, long lat2,
						  long dLong, long dLat, float d);
void CirclePieces(WorldRect circle, WorldPoint *center, float *radius);
Boolean PointInCircle(WorldPoint wp, WorldRect circle,
					   long dLong, long dLat, float d);
Boolean CircleINCircle(WorldRect circle1, WorldRect circle2,
					   long dLong, long dLat, float d);
Boolean CircleInCircle(WorldRect circle1, WorldRect circle2,
					   long dLong, long dLat, float d);*/
Boolean WPointInWRect(long longVal, long latVal, WorldRect *w);
//Boolean WPointInWRectE(long longVal, long latVal, WorldRect *w,
					   //long dLong, long dLat, float d);
Boolean WRectTouchesWRect(WorldRect *wr1, WorldRect *wr2);
#endif
/*Boolean WRectINWRect(WorldRect *wr1, WorldRect *wr2,
					 long dLong, long dLat, float d);
Boolean WRectInWRect(WorldRect *wr1, WorldRect *wr2,
					 long dLong, long dLat, float d);
Boolean WRectINCircle(WorldRect wr, WorldRect circle,
					  long dLong, long dLat, float d);
Boolean WRectInCircle(WorldRect wr, WorldRect circle,
					  long dLong, long dLat, float d);
Boolean CircleINWRect(WorldRect circle, WorldRect wr,
					  long dLong, long dLat, float d);
Boolean CircleInWRect(WorldRect circle, WorldRect wr,
					  long dLong, long dLat, float d);

Boolean WPointNearSegment(long pLong, long pLat,
						  long long1, long lat1, long long2, long lat2,
						  long dLong, long dLat, float d);
Boolean SegmentInCircle(long x1, long y1, long x2, long y2, WorldRect circle,
						long dLong, long dLat, float d);
Boolean SegmentInWRectE(long x1, long y1, long x2, long y2, WorldRect wr,
						long dLong, long dLat, float d);
Boolean SegmentInSegment(long x1, long y1, long x2, long y2,
						 long X1, long Y1, long X2, long Y2,
						 long dLong, long dLat, float d);
Boolean CircleInSegment(WorldRect circle, long x1, long y1, long x2, long y2,
						long dLong, long dLat, float d);
Boolean WRectInSegment(WorldRect wr, long x1, long y1, long x2, long y2,
					   long dLong, long dLat, float d);

Boolean PointInPolygonE(WorldPoint p, SEGMENTH segments, long numSegs, WorldRect bounds,
						long dLong, long dLat, float d, Boolean holes);
Boolean PointInPolyline(WorldPoint p, SEGMENTH segments, long numSegs, WorldRect bounds,
					 	long dLong, long dLat, float d);
Boolean SegmentInPolyline(long x1, long y1, long x2, long y2,
						  SEGMENTH segments, long numSegs, WorldRect bounds,
						  long dLong, long dLat, float d);
Boolean SegmentInPolygon(long x1, long y1, long x2, long y2,
						 SEGMENTH segments, long numSegs, WorldRect bounds,
						 long dLong, long dLat, float d, Boolean holes);
Boolean PolyInPolygon(SEGMENTH segments1, long numSegs1, WorldRect bounds1,
					  SEGMENTH segments2, long numSegs2, WorldRect bounds2,
					  long dLong, long dLat, float d, Boolean holes);
Boolean PolyInPolyline(SEGMENTH segments1, long numSegs1, WorldRect bounds1,
					   SEGMENTH segments2, long numSegs2, WorldRect bounds2,
					   long dLong, long dLat, float d);
Boolean PolyInWRect(SEGMENTH segments, long numSegs, WorldRect bounds, WorldRect wr,
					long dLong, long dLat, float d);
Boolean WRectInPolygon(WorldRect wr, SEGMENTH segments, long numSegs, WorldRect bounds,
					   long dLong, long dLat, float d, Boolean holes);
Boolean WRectInPolyline(WorldRect wr, SEGMENTH segments, long numSegs, WorldRect bounds,
					   long dLong, long dLat, float d);
Boolean CircleInPolygon(WorldRect circle, SEGMENTH segments, long numSegs, WorldRect bounds,
						long dLong, long dLat, long d, Boolean holes);
Boolean CircleInPolyline(WorldRect circle, SEGMENTH segments, long numSegs, WorldRect bounds,
						 long dLong, long dLat, long d);
Boolean PolyInCircle(SEGMENTH segments, long numSegs, WorldRect bounds, WorldRect circle,
					 long dLong, long dLat, float d);*/

WorldRect *AddWPointToWRect(long pLat, long pLong, WorldRect *w);

//void SetWPoint(WorldPoint *p, long pLong, long pLat);
//void SetSegment(Segment *s, long fromLong, long fromLat, long toLong, long toLat);
void SetWorldRect(WorldRect *w, long loLat, long loLong, long hiLat, long hiLong);
//void MakeWorldPointValid(long *pLong, long *pLat);
//void MakeWorldRectValid(WorldRect *wr);
Boolean EqualWRects(WorldRect wr1, WorldRect wr2);
#ifndef pyGNOME
Boolean EqualWPoints(WorldPoint p1, WorldPoint p2);
void OffsetWRect(WorldRect *w, long dh, long dv);
void InsetWRect(WorldRect *w, long dh, long dv);
void OutsetWRect(WorldRect *w, long dh, long dv);
WorldPoint WorldRectCenter(WorldRect w);
WorldRect UnionWRect(WorldRect w1, WorldRect w2);
//WorldRect SectWRect(WorldRect *w1, WorldRect *w2);
WorldRect AddWRectBorders(WorldRect w, short fraction);
WorldPoint Midpoint(Segment s);

float DotProduct(Vector v, Vector w);
Vector Perp(Vector v);
Vector SubVectors(Vector v, Vector w);
WorldPoint PointOfIntersection(Segment s1, Segment s2);
Boolean SegmentCrossesVerticalRay(long x1, long y1, long x2, long y2, long xp, long yp);
Boolean PointInPolygon(WorldPoint p, SEGMENTH segments, long numSegs, Boolean holes);
Boolean OppositeSides(long x1, long y1, long x2, long y2,
					  long X1, long Y1, long X2, long Y2);
Boolean SegmentTouchesSegment(Segment s1, Segment s2);
//Boolean SegmentInWRect(long x1, long y1, long x2, long y2, WorldRect wr);
//Boolean ClipSegmentToWRect(long *x1, long *y1, long *x2, long *y2, WorldRect wr);

//OSErr Solve3x3Matrix(double m[3][3], double v[3], double *a, double *b, double *c);
//double Compute3x3Determinant(double m[3][3]);
//double Compute2x2Determinant(double m[2][2]);

//WORLDPOINTH SegmentsToWPoints(SEGMENTH segments, long numSegs, WorldRect *wr);
SEGMENTH WPointsToSegments(WORLDPOINTH wPoints, long numPoints, long *numSegs);
//PointH SegmentsToPointTypes(SEGMENTH segments, long numSegs, long *numPoints);
//SEGMENTH PointTypesToSegments(PointH wPoints, long firstPoint, long numPoints, long *numSegs);

void SortSegments(SEGMENTH S, long numSegs);
//OSErr CompletePolygonIslands(SEGMENTH *P, long *numSegs);
//OSErr AppendPolygons(SEGMENTH *a, SEGMENTH *b);
//SEGMENTH NextPolygonPiece(SEGMENTH segments, long numSegs,
						  //long start, long *end, long *count, WorldRect *bounds);
SEGMENTH IntersectPolygons(SEGMENTH *P, SEGMENTH *Q);
//SEGMENTH UnionPolygons(SEGMENTH *P, SEGMENTH *Q, Boolean *appended);
//SEGMENTH DifferencePolygons(SEGMENTH *P, SEGMENTH *Q);

//SEGMENTH ComputeEnvelope(SEGMENTH segments, float radius);
void SetPt(POINTPTR p, short h, short v);
#endif

#endif
