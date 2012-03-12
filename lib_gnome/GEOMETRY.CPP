#include "Basics.h"
#include "TypeDefs.h"
#include "GEOMETRY.H"
#include "RectUtils.h"
#include "CompFunctions.h"

#ifdef MAC
	#pragma segment GEOMETRY
	short GEOMETRYDummy() { return 0; }
#endif


#ifdef LLSHIFT
WorldRect theWorld = { 0, 0, 360000000, 180000000 },
		  voidWorldRect = { 360000000, 180000000, 0, 0 },
		  emptyWorldRect = { 0, 0, 0, 0 };
#else
//WorldRect theWorld = { -180000000, -90000000, 180000000, 90000000 },
		 // voidWorldRect = { 180000000, 90000000, -180000000, -90000000 },
		 // emptyWorldRect = { 0, 0, 0, 0 };
//WorldRect theWorld = { -180000000, -90000000, 360000000, 90000000 },
		  //voidWorldRect = { 360000000, 90000000, -180000000, -90000000 },
		 // emptyWorldRect = { 0, 0, 0, 0 };
WorldRect theWorld = { -360000000, -90000000, 360000000, 90000000 },
		  voidWorldRect = { 360000000, 90000000, -360000000, -90000000 },
		  emptyWorldRect = { 0, 0, 0, 0 };
#endif

////////////////////////////////////////////////////////////// MISC

float LongToLatRatio(float baseLat)
#ifdef LLSHIFT
{ return cos((baseLat - 90) * 3.14159/180); }
#else
{ return cos(baseLat * 3.14159/180); }
#endif

float MilesPerDegreeLong(float baseLat)
{ return 69 * LongToLatRatio(baseLat); }

float MilesPerDegreeLat()
{ return 69; }

float DegreesLongPerMile(float baseLat)
{ return 1 / MilesPerDegreeLong(baseLat); }

float DegreesLatPerMile()
{ return 1.0 / 69.0; }


float LongToLatRatio2(WorldRect *wr)
#ifdef LLSHIFT
{
	long lo = _max(wr->loLat, 0), hi = _min(wr->hiLat, 180000000), m = (lo + hi) / 2;
	
	return cos(((float)(m - 90000000) / 1000000.0) * 3.14159 / 180);
}
#else
{
	long lo = _max(wr->loLat, -90000000), hi = _min(wr->hiLat, 90000000), m = (lo + hi) / 2;
	
	return cos(((float)m / 1000000.0) * 3.14159 / 180);
}
#endif

float LongToLatRatio3(long baseLat)
#ifdef LLSHIFT
{ return cos(((float)(baseLat - 90000000) / 1000000.0) * 3.14159 / 180); }
#else
{ return cos(((float)baseLat / 1000000.0) * 3.14159 / 180); }
#endif

Boolean SameSegmentEndPoints(Segment s1, Segment s2)
{
	return (s1.fromLong == s2.fromLong && s1.fromLat == s2.fromLat) ||
		   (s1.toLong == s2.fromLong && s1.toLat == s2.fromLat) ||
		   (s1.fromLong == s2.toLong && s1.fromLat == s2.toLat) ||
		   (s1.toLong == s2.toLong && s1.toLat == s2.toLat);
}

Point ForcePointOnLineIntoRect(short x, short y, float slope, Rect r)
{
	short x2, y2;
	Point p;
	
	SetPt(&p, x, y);
	
	if (MyPtInRect(p, &r)) return p;
	
	if (x < r.left) {
		y2 = y + (r.left - x) * slope;
		if (y2 >= r.top && y2 <= r.bottom) { SetPt(&p, r.left, y2); return p; }
	}
	if (x > r.right) {
		y2 = y - (x - r.right) * slope;
		if (y2 >= r.top && y2 <= r.bottom) { SetPt(&p, r.right, y2); return p; }
	}
	if (y < r.top) {
		x2 = x + (r.top - y) / slope;
		if (x2 >= r.left && x2 <= r.right) { SetPt(&p, x2, r.top); return p; }
	}
	if (y > r.bottom) {
		x2 = x - (y - r.bottom) / slope;
		if (x2 >= r.left && x2 <= r.right) { SetPt(&p, x2, r.bottom); return p; }
	}
	
	return p;
}

////////////////////////////////////////////////////////////// DISTANCES AND AREA

float UnitsPerDegreeLat()
{
#ifdef STANDARDUNITS
	switch (STANDARDUNITS) {
#else
	switch (KILOMETERS) {
#endif
		case FEET:			return FEETPERDEGREELAT;
		case YARDS:			return YARDSPERDEGREELAT;
		case METERS:		return METERSPERDEGREELAT;
		case KILOMETERS:	return KILOMETERSPERDEGREELAT;
		case MILES:			return MILESPERDEGREELAT;
		case NAUTICALMILES:	return NAUTSPERDEGREELAT;
	}
	
	return MILESPERDEGREELAT;
}

float EarthRadius()
{
	float radius = EARTHRADIUSINMILES;
	
#ifdef STANDARDUNITS
	switch (STANDARDUNITS) {
#else
	switch (KILOMETERS) {
#endif
		case FEET:			radius *= MILESTOFEET; break;
		case YARDS:			radius *= MILESTOYARDS; break;
		case METERS:		radius *= MILESTOMETERS; break;
		case KILOMETERS:	radius *= MILESTOKILO; break;
		case NAUTICALMILES:	radius *= MILESTONAUTS; break;
	}
	
	return radius;
}

float LongToDistance(long dLong, WorldPoint center)
{
	float distance, baseLat = center.pLat / 1000000.0;
	
	distance = (dLong / 1000000.0) * UnitsPerDegreeLat() * LongToLatRatio(baseLat);
	
	return distance;
}

float LatToDistance(long dLat)
{
	float distance = (dLat / 1000000.0) * UnitsPerDegreeLat();
	
	return distance;
}

long DistanceToLong(float distance, WorldPoint center)
{
	float baseLat = center.pLat / 1000000.0, dLong, ratio;
	
	ratio = LongToLatRatio(baseLat);
	dLong = distance / (UnitsPerDegreeLat() * ratio);
	
	return dLong * 1000000;
}

long DistanceToLat(float distance)
{
	float dLat;
	
	dLat = distance / UnitsPerDegreeLat();
	
	return dLat * 1000000;
}

/*
float DistanceBetweenWorldPoints(WorldPoint p1, WorldPoint p2)
{
	float d, distanceX, distanceY;
	WorldPoint center;
	
	center.pLong = (p1.pLong + p2.pLong) / 2;
	center.pLat = (p1.pLat + p2.pLat) / 2;
	
	distanceX = LongToDistance(abs(p1.pLong - p2.pLong), center);
	distanceY = LatToDistance(abs(p1.pLat - p2.pLat));
	
	d = sqrt(distanceX * distanceX + distanceY * distanceY);
	
	return d;
}
*/

float DistanceBetweenWorldPoints(WorldPoint p1, WorldPoint p2)
{
#ifdef LLSHIFT
	double d, c, x1, y1, z1, x2, y2, z2, w, R = EarthRadius(),
		   radLong1 = ((p1.pLong - 180000000) / 1000000.0) * (PI / 180),
		   radLat1 = ((p1.pLat - 90000000) / 1000000.0) * (PI / 180),
		   radLong2 = ((p2.pLong - 180000000) / 1000000.0) * (PI / 180),
		   radLat2 = ((p2.pLat - 90000000) / 1000000.0) * (PI / 180);
#else
	double d, c, x1, y1, z1, x2, y2, z2, w, R = EarthRadius(),
		   radLong1 = (p1.pLong / 1000000.0) * (PI / 180),
		   radLat1 = (p1.pLat / 1000000.0) * (PI / 180),
		   radLong2 = (p2.pLong / 1000000.0) * (PI / 180),
		   radLat2 = (p2.pLat / 1000000.0) * (PI / 180);
#endif
	
	x1 = cos(radLat1) * cos(radLong1);	x2 = cos(radLat2) * cos(radLong2);
	y1 = cos(radLat1) * sin(radLong1);	y2 = cos(radLat2) * sin(radLong2);
	z1 = sin(radLat1);					z2 = sin(radLat2);
	
	c = x1 * x2 + y1 * y2 + z1 * z2;
	
	// Due to what seems like a bug in the Microsoft C compiler, if you assign
	// a float (or double) value and then immediately do a comparison on it,
	// the value gets assigned improperly.  So here we do an assignment with
	// c before doing the comparison to make it keep the correct value.
   	
   	x1 = c;
	
	if (c > 1) c = 1; // SysBeep(1);
	
	w = acos(c);
	
	d = R * w;
	
	return d;
}

float ComputePolygonArea(long numPoints, PointH points, WorldRect bounds)
{
	float x1, x2, y1, y2, area = 0, subArea = 0;
	WorldPoint p, p0 = INDEXH(points, 0).p,
			   center = WorldRectCenter(bounds);
	long i;
	
	for (i = 2 ; i < numPoints ; i++) {
		p = INDEXH(points, i - 1).p;
		x1 = LongToDistance(p.pLong - p0.pLong, center);
		y1 = LatToDistance(p.pLat - p0.pLat);
		p = INDEXH(points, i).p;
		x2 = LongToDistance(p.pLong - p0.pLong, center);
		y2 = LatToDistance(p.pLat - p0.pLat);
		subArea += (x1 * y2 - y1 * x2);
		if (i == (numPoints - 1) || INDEXH(points, i + 1).newPiece) {
			// if (subArea < 0) subArea *= -1;
			area += subArea;
			subArea = 0;
		}
	}
	
	return area / 2;
}

float ComputeWorldRectArea(WorldRect bounds)
{
	PointType p[5], *q;
	
	p[0].p.pLong = bounds.loLong; p[0].p.pLat = bounds.loLat; p[0].newPiece = TRUE;
	p[1].p.pLong = bounds.hiLong; p[1].p.pLat = bounds.loLat; p[1].newPiece = FALSE;
	p[2].p.pLong = bounds.hiLong; p[2].p.pLat = bounds.hiLat; p[2].newPiece = FALSE;
	p[3].p.pLong = bounds.loLong; p[3].p.pLat = bounds.hiLat; p[3].newPiece = FALSE;
	p[4].p.pLong = bounds.loLong; p[4].p.pLat = bounds.loLat; p[4].newPiece = FALSE;
	
	q = &p[0];
	
	return ComputePolygonArea(5, (PointH)&q, bounds);
}

////////////////////////////////////////////////////////////// INTERSECTION: POINTS

// INTERSECTION TERMINOLOGY
//			"IN" means completely within
//			"In" means partially or completely within
//			"Touches" means shape A is "In" shape B or shape B is "In" shape A

////////////////////////////////////////////////////////////// INTERSECTION: POINTS

Boolean WPointNearWPoint(WorldPoint p1, WorldPoint p2, float d)
{
	return DistanceBetweenWorldPoints(p1, p2) <= d;
}

Boolean WPointNearWPoint2(long long1, long lat1, long long2, long lat2,
						 long dLong, long dLat, float d)
{
	WorldRect circle;
	WorldPoint wp;
	
	if (long1 == long2 && lat1 == lat2) return TRUE;
	
	wp.pLong = long1;
	wp.pLat = lat1;
	
	circle.loLong = circle.hiLong = long2;
	circle.loLat = circle.hiLat = lat2;
	
	return PointInCircle(wp, circle, dLong, dLat, d);
}

////////////////////////////////////////////////////////////// INTERSECTION: CIRCLES

void CirclePieces(WorldRect circle, WorldPoint *center, float *radius)
{
	WorldPoint top;
	
	center->pLong = (circle.loLong + circle.hiLong) / 2;
	center->pLat = (circle.loLat + circle.hiLat) / 2;
	
	top.pLong = (circle.loLong + circle.hiLong) / 2;
	top.pLat = circle.hiLat;
	
	*radius = DistanceBetweenWorldPoints(*center, top);
}

Boolean PointInCircle(WorldPoint wp, WorldRect circle,
					  long dLong, long dLat, float d)
{
	WorldPoint center;
	WorldRect r;
	float dummy, distance, radius, leg;
#pragma unused(d)
	
	center.pLong = (circle.loLong + circle.hiLong) / 2;
	center.pLat = (circle.loLat + circle.hiLat) / 2;
	
	OutsetWRect(&circle, dLong, dLat);
	
	// if the point is in the square inscribed in the circle, it is in the circle
	// we use the latitudinal radius, since _min(long rad, lat rad) = lat rad
	
	leg = ((circle.hiLat - circle.loLat) / 2) / ROOT2;
	r.loLong = center.pLong - leg;
	r.hiLong = center.pLong + leg;
	r.loLat = center.pLat - leg;
	r.hiLat = center.pLat + leg;
	if (WPointInWRect(wp.pLong, wp.pLat, &r)) return TRUE;
	
	// if the point is not in the square superscribed about the circle,
	// it is not in the circle
	// we use the longitudinal radius, since long rad >= lat rad
	
	leg = (circle.hiLong - circle.loLong) / 2;
	r.loLong = center.pLong - leg;
	r.hiLong = center.pLong + leg;
	r.loLat = center.pLat - leg;
	r.hiLat = center.pLat + leg;
	if (!WPointInWRect(wp.pLong, wp.pLat, &r)) return FALSE;
	
	// otherwise, we do the full check
	
	CirclePieces(circle, &center, &radius);
	distance = DistanceBetweenWorldPoints(center, wp);
	
	dummy = radius; // to avoid compiler bug
	dummy = distance; // to avoid compiler bug
	
	return distance <= radius;
}

Boolean CircleINCircle(WorldRect circle1, WorldRect circle2,
					   long dLong, long dLat, float d)
{
	WorldPoint center1, center2;
	float radius1, radius2, distance, dummy;
#pragma unused(d)
	
	OutsetWRect(&circle2, dLong, dLat);
	
	CirclePieces(circle1, &center1, &radius1);
	CirclePieces(circle2, &center2, &radius2);
	
	distance = DistanceBetweenWorldPoints(center1, center2);
	
	dummy = distance; // to avoid compiler bug
	
	return (distance + radius1) <= radius2;
}

Boolean CircleInCircle(WorldRect circle1, WorldRect circle2,
					   long dLong, long dLat, float d)
{
	WorldPoint center1, center2;
	float radius1, radius2, distance, dummy;
#pragma unused(d)
	
	OutsetWRect(&circle2, dLong, dLat);
	
	CirclePieces(circle1, &center1, &radius1);
	CirclePieces(circle2, &center2, &radius2);
	
	distance = DistanceBetweenWorldPoints(center1, center2);
	
	dummy = distance; // to avoid compiler bug
	
	if (distance > (radius1 + radius2)) return FALSE; // circles don't intersect
	if (radius1 > (distance + radius2)) return FALSE; // circle2 IN circle1
	
	return TRUE;
}

////////////////////////////////////////////////////////////// INTERSECTION: RECTS

Boolean WPointInWRect(long longVal, long latVal, WorldRect *w)
{
	return latVal >= w->loLat && latVal <= w->hiLat &&
		   longVal >= w->loLong && longVal <= w->hiLong;
}

Boolean WPointInWRectE(long longVal, long latVal, WorldRect *w,
					   long dLong, long dLat, float d)
{
	WorldRect w2;
	
	// if the point is in the rect, we're done
	
	if (WPointInWRect(longVal, latVal, w)) return TRUE;
	
	// if the point is not in the hyper-rect, it is not in the hyper-rounded-rect
	
	w2 = *w;
	OutsetWRect(&w2, dLong, dLat);
	
	if (!WPointInWRect(longVal, latVal, &w2)) return FALSE;
	
	// otherwise, we need to check if its near any of the rect's segments
	// (which includes being near its corners)
	
	if (WPointNearSegment(longVal, latVal,
						  w->loLong, w->loLat, w->loLong, w->hiLat, dLong, dLat, d) ||
		WPointNearSegment(longVal, latVal,
						  w->hiLong, w->loLat, w->hiLong, w->hiLat, dLong, dLat, d) ||
		WPointNearSegment(longVal, latVal,
						  w->loLong, w->loLat, w->hiLong, w->loLat, dLong, dLat, d) ||
		WPointNearSegment(longVal, latVal,
						  w->loLong, w->hiLat, w->hiLong, w->hiLat, dLong, dLat, d))
		return TRUE;
	
	return FALSE;
}

Boolean WRectTouchesWRect(WorldRect *wr1, WorldRect *wr2)
{
	if (wr1->loLat > wr2->hiLat) return FALSE;
	if (wr2->loLat > wr1->hiLat) return FALSE;
	if (wr1->loLong > wr2->hiLong) return FALSE;
	if (wr2->loLong > wr1->hiLong) return FALSE;
	
	return TRUE;
}

Boolean WRectINWRect(WorldRect *wr1, WorldRect *wr2,
					 long dLong, long dLat, float d)
{
	return WPointInWRectE(wr1->loLong, wr1->loLat, wr2, dLong, dLat, d) &&
		   WPointInWRectE(wr1->loLong, wr1->hiLat, wr2, dLong, dLat, d) &&
		   WPointInWRectE(wr1->hiLong, wr1->loLat, wr2, dLong, dLat, d) &&
		   WPointInWRectE(wr1->hiLong, wr1->hiLat, wr2, dLong, dLat, d);
}

Boolean WRectInWRect(WorldRect *wr1, WorldRect *wr2,
					 long dLong, long dLat, float d)
{
	WorldRect wB = *wr2;
	
	OutsetWRect(&wB, dLong, dLat);
	
	if (!WRectTouchesWRect(wr1, &wB)) return FALSE;
	if (WRectINWRect(wr1, wr2, 0, 0, 0)) return TRUE;
	if (WRectINWRect(&wB, wr1, 0, 0, 0)) return FALSE;
	
	return SegmentInWRectE(wr2->loLong, wr2->loLat, wr2->loLong, wr2->hiLat, *wr1, dLong, dLat, d) ||
		   SegmentInWRectE(wr2->hiLong, wr2->loLat, wr2->hiLong, wr2->hiLat, *wr1, dLong, dLat, d) ||
		   SegmentInWRectE(wr2->loLong, wr2->loLat, wr2->hiLong, wr2->loLat, *wr1, dLong, dLat, d) ||
		   SegmentInWRectE(wr2->loLong, wr2->hiLat, wr2->hiLong, wr2->hiLat, *wr1, dLong, dLat, d);
}

Boolean WRectINCircle(WorldRect wr, WorldRect circle,
					  long dLong, long dLat, float d)
{
	WorldPoint corner1, corner2, corner3, corner4;
	
	SetWPoint(&corner1, wr.loLong, wr.loLat);
	SetWPoint(&corner2, wr.loLong, wr.hiLat);
	SetWPoint(&corner3, wr.hiLong, wr.loLat);
	SetWPoint(&corner4, wr.hiLong, wr.hiLat);
	
	return PointInCircle(corner1, circle, dLong, dLat, d) &&
		   PointInCircle(corner2, circle, dLong, dLat, d) &&
		   PointInCircle(corner3, circle, dLong, dLat, d) &&
		   PointInCircle(corner4, circle, dLong, dLat, d);
}

Boolean WRectInCircle(WorldRect wr, WorldRect circle,
					  long dLong, long dLat, float d)
{
	return SegmentInCircle(wr.loLong, wr.loLat, wr.loLong, wr.hiLat, circle, dLong, dLat, d) ||
		   SegmentInCircle(wr.hiLong, wr.loLat, wr.hiLong, wr.hiLat, circle, dLong, dLat, d) ||
		   SegmentInCircle(wr.loLong, wr.loLat, wr.hiLong, wr.loLat, circle, dLong, dLat, d) ||
		   SegmentInCircle(wr.loLong, wr.hiLat, wr.hiLong, wr.hiLat, circle, dLong, dLat, d);
}

Boolean CircleINWRect(WorldRect circle, WorldRect wr,
					  long dLong, long dLat, float d)
{
	return WRectInWRect(&circle, &wr, dLong, dLat, d);
}

Boolean CircleInWRect(WorldRect circle, WorldRect wr,
					  long dLong, long dLat, float d)
{
	WorldRect wB = wr;
	
	if (CircleINWRect(circle, wr, 0, 0, 0)) return TRUE;
	
	OutsetWRect(&wB, dLong, dLat);
	
	if (!WRectTouchesWRect(&circle, &wB)) return FALSE;
	// if (WRectINWRect(&circle, wr1, 0, 0)) return FALSE;
	
	return SegmentInCircle(wr.loLong, wr.loLat, wr.loLong, wr.hiLat, circle, dLong, dLat, d) ||
		   SegmentInCircle(wr.hiLong, wr.loLat, wr.hiLong, wr.hiLat, circle, dLong, dLat, d) ||
		   SegmentInCircle(wr.loLong, wr.loLat, wr.hiLong, wr.loLat, circle, dLong, dLat, d) ||
		   SegmentInCircle(wr.loLong, wr.hiLat, wr.hiLong, wr.hiLat, circle, dLong, dLat, d);
}

////////////////////////////////////////////////////////////// INTERSECTION: SEGMENTS

Boolean WPointNearSegment(long pLong, long pLat,
						  long long1, long lat1, long long2, long lat2,
						  long dLong, long dLat, float d)
{
	float a, b, x, y, h, dist, dist2, numer, dummy;
	WorldPoint p, p2;
	
	if (long1 < long2) { if (pLong < (long1 - dLong) ||
							 pLong > (long2 + dLong)) return FALSE; }
	else			   { if (pLong < (long2 - dLong) ||
							 pLong > (long1 + dLong)) return FALSE; }
	
	if (lat1 < lat2) { if (pLat < (lat1 - dLat) ||
						   pLat > (lat2 + dLat)) return FALSE; }
	else			 { if (pLat < (lat2 - dLat) ||
						   pLat > (lat1 + dLat)) return FALSE; }
	
	p.pLong = pLong;
	p.pLat = pLat;
	
	if (WPointNearWPoint2(pLong, pLat, long1, lat1, dLong, dLat, d)) return TRUE;
	if (WPointNearWPoint2(pLong, pLat, long2, lat2, dLong, dLat, d)) return TRUE;
	
	// translate origin to start of segment
	
	a = LongToDistance(long2 - long1, p);
	b = LatToDistance(lat2 - lat1);
	x = LongToDistance(pLong - long1, p);
	y = LatToDistance(pLat - lat1);
	h = sqrt(a * a + b * b);
	
	// distance from point to segment
	numer = abs(a * y - b * x);
	dist = numer / h;
	dummy = dist; // see comment below
	
	if (dist > d) return FALSE;
	
	// the rest of this code checks if the point is beyond the ends of the
	// segment (and beyond the radii of the circles at the endpoints)
	
	// length of projection of point onto segment
	numer = a * x + b * y;
	dist = numer / h;
	dummy = dist; // see comment below
	
	if (dist < 0) return FALSE;
	
	p.pLong = long1;
	p.pLat = lat1;
	p2.pLong = long2;
	p2.pLat = lat2;
	dist2 = DistanceBetweenWorldPoints(p, p2);
	
	// Due to what seems like a bug in the Microsoft C compiler, if you assign
	// a float (or double) value and do nohting but comparisons with it,
	// the value gets lost after the assignment on 386 machines.  So here we do
	// an assignment with dist2 and dist before doing the comparison to make they
	// keep their values.  (The problem may be limited to doing a comparison
	// right after an assignment.)
	
	dummy = dist2;
	dummy = dist;
	
	if (dist > dist2) return FALSE;
	
	return TRUE;
}

Boolean SegmentInCircle(long x1, long y1, long x2, long y2, WorldRect circle,
						long dLong, long dLat, float dist)
{
	float a, c, d, e, r, radius, dummy;
	WorldPoint center, y, z, p;
	
	OutsetWRect(&circle, dLong, dLat);
	
	if (!SegmentInWRectE(x1, y1, x2, y2, circle, dLong, dLat, dist)) return FALSE;
	
	p.pLong = x1;
	p.pLat = y1;
	if (PointInCircle(p, circle, 0, 0, 0)) return TRUE;
	p.pLong = x2;
	p.pLat = y2;
	if (PointInCircle(p, circle, 0, 0, 0)) return TRUE;
	
	y.pLong = x1;
	y.pLat = y1;
	z.pLong = x2;
	z.pLat = y2;
	CirclePieces(circle, &center, &radius);
	
	c = DistanceBetweenWorldPoints(center, y);
	d = DistanceBetweenWorldPoints(center, z);
	e = DistanceBetweenWorldPoints(y, z);
	
	a = (c * c - d * d + e * e) / (2 * e);
	
	if (a < 0 || a > e) return FALSE;
	
	r = sqrt(c * c - a * a);
	
	dummy = r; // to avoid compiler bug
	
	return r <= radius;
}

Boolean SegmentInWRectE(long x1, long y1, long x2, long y2, WorldRect wr,
						long dLong, long dLat, float d)
{
	WorldRect wr2 = wr;
	
	if (SegmentInWRect(x1, y1, x2, y2, wr)) return TRUE;
	
	wr2 = wr;
	OutsetWRect(&wr2, dLong, dLat);
	if (!SegmentInWRect(x1, y1, x2, y2, wr2)) return FALSE;
	
	return SegmentInSegment(x1, y1, x2, y2,
							wr.loLong, wr.loLat, wr.loLong, wr.hiLat, dLong, dLat, d) ||
		   SegmentInSegment(x1, y1, x2, y2,
							wr.hiLong, wr.loLat, wr.hiLong, wr.hiLat, dLong, dLat, d) ||
		   SegmentInSegment(x1, y1, x2, y2,
							wr.loLong, wr.loLat, wr.hiLong, wr.loLat, dLong, dLat, d) ||
		   SegmentInSegment(x1, y1, x2, y2,
							wr.loLong, wr.hiLat, wr.hiLong, wr.hiLat, dLong, dLat, d);
}

Boolean SegmentInSegment(long x1, long y1, long x2, long y2,
						 long X1, long Y1, long X2, long Y2,
						 long dLong, long dLat, float d)
{
	Segment s1, s2;
	
	SetSegment(&s1, x1, y1, x2, y2);
	SetSegment(&s2, X1, Y1, X2, Y2);
	
	if (SegmentTouchesSegment(s1, s2)) return TRUE;
	
	if (WPointNearSegment(x1, y1, X1, Y1, X2, Y2, dLong, dLat, d)) return TRUE;
	if (WPointNearSegment(x2, y2, X1, Y1, X2, Y2, dLong, dLat, d)) return TRUE;
	
	if (WPointNearSegment(X1, Y1, x1, y1, x2, y2, dLong, dLat, d)) return TRUE;
	if (WPointNearSegment(X2, Y2, x1, y1, x2, y2, dLong, dLat, d)) return TRUE;
	
	return FALSE;
}

Boolean CircleInSegment(WorldRect circle, long x1, long y1, long x2, long y2,
						long dLong, long dLat, float d)
{
	WorldPoint p1, p2;
	
	p1.pLong = x1;
	p1.pLat = y1;
	p2.pLong = x2;
	p2.pLat = y2;
	
	return SegmentInCircle(x1, y1, x2, y2, circle, dLong, dLat, d) &&
		   (!PointInCircle(p1, circle, dLong, dLat, d) ||
		   	!PointInCircle(p2, circle, dLong, dLat, d));
}

Boolean WRectInSegment(WorldRect wr, long x1, long y1, long x2, long y2,
					   long dLong, long dLat, float d)
{
	WorldPoint p1, p2;
	
	p1.pLong = x1;
	p1.pLat = y1;
	p2.pLong = x2;
	p2.pLat = y2;
	
	return SegmentInWRectE(x1, y1, x2, y2, wr, dLong, dLat, d) &&
		   (!PointInCircle(p1, wr, dLong, dLat, d) ||
		   	!PointInCircle(p2, wr, dLong, dLat, d));
}

////////////////////////////////////////////////////////////// INTERSECTION: POLYGONS

Boolean PointInPolygonE(WorldPoint p, SEGMENTH segments, long numSegs, WorldRect bounds,
						long dLong, long dLat, float d, Boolean holes)
{
	long i;
	
	if (!WPointInWRectE(p.pLong, p.pLat, &bounds, dLong, dLat, d)) return FALSE;
	if (PointInPolygon(p, segments, numSegs, holes)) return TRUE;
	
	for (i = 0 ; i < numSegs ; i++)
		if (WPointNearSegment(p.pLong, p.pLat,
							  INDEXH(segments, i).fromLong, INDEXH(segments, i).fromLat,
							  INDEXH(segments, i).toLong, INDEXH(segments, i).toLat,
							  dLong, dLat, d))
			return TRUE;
	
	return FALSE;
}

Boolean PointInPolyline(WorldPoint p, SEGMENTH segments, long numSegs, WorldRect bounds,
					 	long dLong, long dLat, float d)
{
	long i;
	
	if (!WPointInWRectE(p.pLong, p.pLat, &bounds, dLong, dLat, d)) return FALSE;
	
	for (i = 0 ; i < numSegs ; i++)
		if (WPointNearSegment(p.pLong, p.pLat,
							  INDEXH(segments, i).fromLong, INDEXH(segments, i).fromLat,
							  INDEXH(segments, i).toLong, INDEXH(segments, i).toLat,
							  dLong, dLat, d))
			return TRUE;
	
	return FALSE;
}

Boolean SegmentInPolyline(long x1, long y1, long x2, long y2,
						  SEGMENTH segments, long numSegs, WorldRect bounds,
						  long dLong, long dLat, float d)
{
	long i;
	
	if (!SegmentInWRectE(x1, y1, x2, y2, bounds, dLong, dLat, d)) return FALSE;
	
	for (i = 0 ; i < numSegs ; i++)
		if (SegmentInSegment(x1, y1, x2, y2,
							 INDEXH(segments, i).fromLong, INDEXH(segments, i).fromLat,
							 INDEXH(segments, i).toLong, INDEXH(segments, i).toLat,
							 dLong, dLat, d))
			return TRUE;
	
	return FALSE;
}

Boolean SegmentInPolygon(long x1, long y1, long x2, long y2,
						 SEGMENTH segments, long numSegs, WorldRect bounds,
						 long dLong, long dLat, float d, Boolean holes)
{
	long i;
	Segment s;
	WorldPoint p;
	
	if (!SegmentInWRectE(x1, y1, x2, y2, bounds, dLong, dLat, d)) return FALSE;
	
	p.pLong = x1;
	p.pLat = y1;
	if (PointInPolygon(p, segments, numSegs, holes)) return TRUE;
	p.pLong = x2;
	p.pLat = y2;
	if (PointInPolygon(p, segments, numSegs, holes)) return TRUE;
	
	s.fromLong = x1;
	s.fromLat = y1;
	s.toLong = x2;
	s.toLat = y2;
	
	for (i = 0 ; i < numSegs ; i++)
		if (SegmentInSegment(x1, y1, x2, y2,
							 INDEXH(segments, i).fromLong, INDEXH(segments, i).fromLat,
							 INDEXH(segments, i).toLong, INDEXH(segments, i).toLat,
							 dLong, dLat, d))
			return TRUE;
	
	return FALSE;
}

Boolean PolyInPolygon(SEGMENTH segments1, long numSegs1, WorldRect bounds1,
					  SEGMENTH segments2, long numSegs2, WorldRect bounds2,
					  long dLong, long dLat, float d, Boolean holes)
{
	long i;
	
	if (!WRectInWRect(&bounds1, &bounds2, dLong, dLat, d)) return FALSE;
	
	for (i = 0 ; i < numSegs1 ; i++)
		if (SegmentInPolygon(INDEXH(segments1, i).fromLong, INDEXH(segments1, i).fromLat,
							 INDEXH(segments1, i).toLong, INDEXH(segments1, i).toLat,
							 segments2, numSegs2, bounds2, dLong, dLat, d, holes))
			return TRUE;
	
	return FALSE;
}

Boolean PolyInPolyline(SEGMENTH segments1, long numSegs1, WorldRect bounds1,
					   SEGMENTH segments2, long numSegs2, WorldRect bounds2,
					   long dLong, long dLat, float d)
{
	long i;
	
	if (!WRectInWRect(&bounds1, &bounds2, dLong, dLat, d)) return FALSE;
	
	for (i = 0 ; i < numSegs1 ; i++)
		if (SegmentInPolyline(INDEXH(segments1, i).fromLong, INDEXH(segments1, i).fromLat,
							  INDEXH(segments1, i).toLong, INDEXH(segments1, i).toLat,
							  segments2, numSegs2, bounds2, dLong, dLat, d))
			return TRUE;
	
	return FALSE;
}

Boolean PolyInWRect(SEGMENTH segments, long numSegs, WorldRect bounds, WorldRect wr,
					long dLong, long dLat, float d)
{
	long i;
	
	if (!WRectInWRect(&bounds, &wr, dLong, dLat, d)) return FALSE;
	
	for (i = 0 ; i < numSegs ; i++)
		if (SegmentInWRectE(INDEXH(segments, i).fromLong, INDEXH(segments, i).fromLat,
							INDEXH(segments, i).toLong, INDEXH(segments, i).toLat,
							wr, dLong, dLat, d))
			return TRUE;
	
	return FALSE;
}

Boolean WRectInPolygon(WorldRect wr, SEGMENTH segments, long numSegs, WorldRect bounds,
					   long dLong, long dLat, float d, Boolean holes)
{
	if (!WRectInWRect(&wr, &bounds, dLong, dLat, d)) return FALSE;
	
	return SegmentInPolygon(wr.loLong, wr.loLat, wr.loLong, wr.hiLat, segments, numSegs, bounds, dLong, dLat, d, holes) ||
		   SegmentInPolygon(wr.hiLong, wr.loLat, wr.hiLong, wr.hiLat, segments, numSegs, bounds, dLong, dLat, d, holes) ||
		   SegmentInPolygon(wr.loLong, wr.loLat, wr.hiLong, wr.loLat, segments, numSegs, bounds, dLong, dLat, d, holes) ||
		   SegmentInPolygon(wr.loLong, wr.hiLat, wr.hiLong, wr.hiLat, segments, numSegs, bounds, dLong, dLat, d, holes);
}

Boolean WRectInPolyline(WorldRect wr, SEGMENTH segments, long numSegs, WorldRect bounds,
					   long dLong, long dLat, float d)
{
	if (!WRectInWRect(&wr, &bounds, dLong, dLat, d)) return FALSE;
	
	return SegmentInPolyline(wr.loLong, wr.loLat, wr.loLong, wr.hiLat, segments, numSegs, bounds, dLong, dLat, d) ||
		   SegmentInPolyline(wr.hiLong, wr.loLat, wr.hiLong, wr.hiLat, segments, numSegs, bounds, dLong, dLat, d) ||
		   SegmentInPolyline(wr.loLong, wr.loLat, wr.hiLong, wr.loLat, segments, numSegs, bounds, dLong, dLat, d) ||
		   SegmentInPolyline(wr.loLong, wr.hiLat, wr.hiLong, wr.hiLat, segments, numSegs, bounds, dLong, dLat, d);
}

Boolean CircleInPolygon(WorldRect circle, SEGMENTH segments, long numSegs, WorldRect bounds,
						long dLong, long dLat, long d, Boolean holes)
{
	WorldPoint center;
	float radius;
	
	if (!WRectInWRect(&circle, &bounds, dLong, dLat, d)) return FALSE;
	
	if (PolyInCircle(segments, numSegs, bounds, circle, dLong, dLat, d))
		return CircleInPolyline(circle, segments, numSegs, bounds, dLong, dLat, d);
	
	CirclePieces(circle, &center, &radius);
	
	return PointInPolygonE(center, segments, numSegs, bounds, dLong, dLat, d, holes);
}

Boolean CircleInPolyline(WorldRect circle, SEGMENTH segments, long numSegs, WorldRect bounds,
						 long dLong, long dLat, long d)
{
	long i;
	
	if (!WRectInWRect(&circle, &bounds, dLong, dLat, d)) return FALSE;
	
	for (i = 0 ; i < numSegs ; i++)
		if (CircleInSegment(circle,
							INDEXH(segments, i).fromLong, INDEXH(segments, i).fromLat,
							INDEXH(segments, i).toLong, INDEXH(segments, i).toLat,
							dLong, dLat, d))
			return TRUE;
	
	return FALSE;
}

Boolean PolyInCircle(SEGMENTH segments, long numSegs, WorldRect bounds, WorldRect circle,
					 long dLong, long dLat, float d)
{
	long i;
	
	if (!WRectInWRect(&bounds, &circle, dLong, dLat, d)) return FALSE;
	
	for (i = 0 ; i < numSegs ; i++)
		if (SegmentInCircle(INDEXH(segments, i).fromLong, INDEXH(segments, i).fromLat,
							INDEXH(segments, i).toLong, INDEXH(segments, i).toLat,
							circle, dLong, dLat, d))
			return TRUE;
	
	return FALSE;
}

////////////////////////////////////////////////////////////// WORLDPOINT AND WORLDRECT

void SetWPoint(WorldPoint *p, long pLong, long pLat)
{
	p->pLong = pLong; p->pLat = pLat;
}

void SetSegment(Segment *s, long fromLong, long fromLat, long toLong, long toLat)
{
	s->fromLong = fromLong;
	s->fromLat = fromLat;
	s->toLong = toLong;
	s->toLat = toLat;
}

void SetWorldRect(WorldRect *w, long loLat, long loLong, long hiLat, long hiLong)
{
	w->loLat = loLat; w->loLong = loLong; w->hiLat = hiLat; w->hiLong = hiLong;
}

void MakeWorldPointValid(long *pLong, long *pLat)
{
	if ((*pLong) > theWorld.hiLong) (*pLong) = theWorld.hiLong;
	if ((*pLong) < theWorld.loLong) (*pLong) = theWorld.loLong;
	if ((*pLat) > theWorld.hiLat) (*pLat) = theWorld.hiLat;
	if ((*pLat) < theWorld.loLat) (*pLat) = theWorld.loLat;
}

void MakeWorldRectValid(WorldRect *wr)
{
	wr->loLong = _min(_max(wr->loLong, theWorld.loLong), theWorld.hiLong);
	wr->hiLong = _min(_max(wr->hiLong, theWorld.loLong), theWorld.hiLong);
	wr->loLat = _min(_max(wr->loLat, theWorld.loLat), theWorld.hiLat);
	wr->hiLat = _min(_max(wr->hiLat, theWorld.loLat), theWorld.hiLat);
	
	if (wr->loLong > wr->hiLong) SwitchLongs(&wr->loLong, &wr->hiLong);
	if (wr->loLat > wr->hiLat) SwitchLongs(&wr->loLat, &wr->hiLat);
}

Boolean EqualWPoints(WorldPoint p1, WorldPoint p2)
{
	return p1.pLong == p2.pLong && p1.pLat == p2.pLat;
}

Boolean EqualWRects(WorldRect wr1, WorldRect wr2)
{
	return wr1.loLong == wr2.loLong && wr1.loLat == wr2.loLat &&
		   wr1.hiLong == wr2.hiLong && wr1.hiLat == wr2.hiLat;
}

void OffsetWRect(WorldRect *w, long dh, long dv)
{
	w->loLat += dv; w->loLong += dh; w->hiLat += dv; w->hiLong += dh;
}

void InsetWRect(WorldRect *w, long dh, long dv)
{
	w->loLat += dv; w->loLong += dh; w->hiLat -= dv; w->hiLong -= dh;
}

void OutsetWRect(WorldRect *w, long dh, long dv)
{
	w->loLat -= dv; w->loLong -= dh; w->hiLat += dv; w->hiLong += dh;
}

WorldPoint WorldRectCenter(WorldRect w)
{
	WorldPoint p;
	
	p.pLong = (w.loLong + w.hiLong) / 2;
	p.pLat = (w.loLat + w.hiLat) / 2;
	
	return p;
}

WorldRect UnionWRect(WorldRect w1, WorldRect w2)
{
	WorldRect w;
	
	SetWorldRect(&w, _min(w1.loLat, w2.loLat),
					 _min(w1.loLong, w2.loLong),
					 _max(w1.hiLat, w2.hiLat),
					 _max(w1.hiLong, w2.hiLong));
	
	return w;
}

WorldRect SectWRect(WorldRect *w1, WorldRect *w2)
{
	WorldRect w;
	
	SetWorldRect(&w, _max(w1->loLat, w2->loLat),
					 _max(w1->loLong, w2->loLong),
					 _min(w1->hiLat, w2->hiLat),
					 _min(w1->hiLong, w2->hiLong));
	
	if (w.loLat > w.hiLat || w.loLong > w.hiLong) w = voidWorldRect;
	
	return w;
}

WorldRect AddWRectBorders(WorldRect w, short fraction)
{
	long d = 0;
	
	if(fraction > 0) d = _max(WRectWidth(w), WRectHeight(w)) / fraction;
	InsetWRect(&w, -d, -d);
	
	return w;
}

WorldRect *AddWPointToWRect(long pLat, long pLong, WorldRect *w)
{
	SetWorldRect(w, _min(w->loLat, pLat),
					_min(w->loLong, pLong),
					_max(w->hiLat, pLat),
					_max(w->hiLong, pLong));
	
	return w;
}

WorldPoint Midpoint(Segment s)
{
	WorldPoint w;
	
	w.pLong = (s.fromLong + s.toLong) / 2;
	w.pLat = (s.fromLat + s.toLat) / 2;
	
	return w;
}

////////// POINT OF INTERSECTION

float DotProduct(Vector v, Vector w)
{
	return (float)v.x * (float)w.x + (float)v.y * (float)w.y;
}

Vector Perp(Vector v)
{
	Vector w;
	
	w.x = -v.y;
	w.y = v.x;
	
	return w;
}

Vector SubVectors(Vector v, Vector w)
{
	Vector a;
	
	a.x = v.x - w.x;
	a.y = v.y - w.y;
	
	return a;
}

WorldPoint PointOfIntersection(Segment s1, Segment s2)
{
	// This routine is based on pages 250-251 of "Turtle Geometry".
	// This routine assumes that s1 and s2 intersect, and thus does less computation
	// than the complete version, which has to compute two fractions to see if the
	// segments intersect.
	
	Vector S1, // vector from (0, 0) to start of s1
		   S2, // vector from (0, 0) to start of s2
		   E1, // vector from (0, 0) to end of s1
		   E2, // vector from (0, 0) to end of s2
		   V1, // vector from start of s1 to end of s1
		   V2; // vector from start of s2 to end of s2
	float f; // between 0 and 1; the fraction of the way along V1 the point of intersection is
	WorldPoint p;
	
	SetVector(S1, s1.fromLong, s1.fromLat);
	SetVector(S2, s2.fromLong, s2.fromLat);
	SetVector(E1, s1.toLong, s1.toLat);
	SetVector(E2, s2.toLong, s2.toLat);
	
	V1 = SubVectors(E1, S1);
	V2 = SubVectors(E2, S2);
	
	f = DotProduct(Perp(V2), SubVectors(S2, S1)) / DotProduct(Perp(V2), V1);
	
	p.pLong = S1.x + V1.x * f;
	p.pLat = S1.y + V1.y * f;
	
	return p;
}

////////// POINT IN POLYGON

Boolean SegmentCrossesVerticalRay(long x1, long y1, long x2, long y2, long xp, long yp)
{
	long dy, DY, dx, DX;
	
	if (y1 <= yp && y2 <= yp) return FALSE;
	if (x1 <= xp && x2 <= xp) return FALSE;
	if (x1 >= xp && x2 >= xp) return FALSE;
	
	if (y1 > yp && y2 > yp) return TRUE;
	
	DX = x2 - x1;
	DY = y2 - y1;
	dx = xp - x1;
	dy = ((float)dx * (float)DY) / (float)DX;
	
	return (y1 + dy) > yp;
}

Boolean PointInPolygon(WorldPoint p, SEGMENTH segments, long numSegs, Boolean holes)
{
	Boolean newPiece;
	long xS, yS, x1, y1, x2, y2, i, pLong = p.pLong, pLat = p.pLat, crosses = 0;
	
	for (i = 0 ; i < numSegs ; i++) {
		if (!holes) {
			if (i == 0 ||
				INDEXH(segments, i).fromLong != INDEXH(segments, i-1).toLong ||
				INDEXH(segments, i).fromLat != INDEXH(segments, i-1).toLat)
				newPiece = TRUE;
			if (newPiece) {
				if ((crosses & 1) == 1) return TRUE; // inside any sub-polygon -> inside
				crosses = 0; // start count again for new sub-polygon				
				newPiece = FALSE;
				xS = INDEXH(segments, i).fromLong;
				yS = INDEXH(segments, i).fromLat;
			}
		}
 		// avoid the problem of counting vertex crosses twice
		x1 = INDEXH(segments, i).fromLong;
		y1 = INDEXH(segments, i).fromLat;
		x2 = INDEXH(segments, i).toLong;
		y2 = INDEXH(segments, i).toLat;
		if (x1 == pLong) x1++;
		if (y1 == pLat) y1++;
		if (x2 == pLong) x2++;
		if (y2 == pLat) y2++;
		if (SegmentCrossesVerticalRay(x1, y1, x2, y2, pLong, pLat))
			crosses++;
		if (!holes)
			if (i < (numSegs - 1))
				if (INDEXH(segments, i).toLong == xS &&
					INDEXH(segments, i).toLat == yS)
					newPiece = TRUE;
	}
	
	return (crosses & 1) == 1;
}

////////// SEGMENT TOUCHES SEGMENT

