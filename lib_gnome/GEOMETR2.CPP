#include "Earl.h"
#include "TypeDefs.h"
#include "GEOMETRY.H"
#include "RectUtils.h"
#include "CompFunctions.h"
#include "MemUtils.h"

#define TechError(a, b, c) printf(a)

////////////////////////////////////////////////////////////// MISC
Boolean OppositeSides(long x1, long y1, long x2, long y2,
					  long X1, long Y1, long X2, long Y2)
{
	long dx, dy, dX, dY, dx1, dy1, dx2, dy2, dX1, dY1, dX2, dY2;
	float same, SAME, dummy;
	
	// see ALGORITHMS (Sedgewick) page 313
	
	if (x1 < X1 && x2 < X1 && x1 < X2 && x2 < X2) return FALSE;
	if (x1 > X1 && x2 > X1 && x1 > X2 && x2 > X2) return FALSE;
	if (y1 < Y1 && y2 < Y1 && y1 < Y2 && y2 < Y2) return FALSE;
	if (y1 > Y1 && y2 > Y1 && y1 > Y2 && y2 > Y2) return FALSE;
	
	dx  = x2 - x1, dy  = y2 - y1, dX  = X2 - X1, dY  = Y2 - Y1,
	dx1 = x1 - X1, dy1 = y1 - Y1, dx2 = x2 - X1, dy2 = y2 - Y1,
	dX1 = X1 - x1, dY1 = Y1 - y1, dX2 = X2 - x1, dY2 = Y2 - y1;
	
	same = ((float)dX * (float)dy1 - (float)dY * (float)dx1) *
		   ((float)dX * (float)dy2 - (float)dY * (float)dx2);
	
	SAME = ((float)dx * (float)dY1 - (float)dy * (float)dX1) *
		   ((float)dx * (float)dY2 - (float)dy * (float)dX2);
	
	dummy = same; // to avoid compiler bug
	dummy = SAME; // to avoid compiler bug
	
	return same <= 0 && SAME <= 0;
}

Boolean SegmentTouchesSegment(Segment s1, Segment s2)
{
	return (OppositeSides(s1.fromLong, s1.fromLat, s1.toLong, s1.toLat,
						  s2.fromLong, s2.fromLat, s2.toLong, s2.toLat));
}

#define SquareIndex(val, lo, hi) ((val) < (lo) ? 0 : ((val) > (hi) ? 2 : 1))

Boolean SegmentInWRect(long x1, long y1, long x2, long y2, WorldRect wr)
{
	//		6  |  7  |  8			// 0 = can't be touching ;
	//		-------------			// 1 = is touching ;
	//		3  |  4  |  5			// 2 = don't know (more computation necessary)
	//		-------------
	//		0  |  1  |  2			// square 4 represents wr
	
							// pt 1:  0  1  2  3  4  5  6  7  8		// pt 2:
	static char crossings[9][9] = { { 0, 0, 0, 0, 1, 2, 0, 2, 2 },	// 0
									{ 0, 0, 0, 2, 1, 2, 2, 1, 2 },	// 1
									{ 0, 0, 0, 2, 1, 0, 2, 2, 0 },	// 2
									{ 0, 2, 2, 0, 1, 1, 0, 2, 2 },	// 3
									{ 1, 1, 1, 1, 1, 1, 1, 1, 1 },	// 4
									{ 2, 2, 0, 1, 1, 0, 2, 2, 0 },	// 5
									{ 0, 2, 2, 0, 1, 2, 0, 0, 0 },	// 6
									{ 2, 1, 2, 2, 1, 2, 0, 0, 0 },	// 7
									{ 2, 2, 0, 2, 1, 0, 0, 0, 0 },	// 8
								  };
	char fromSquare = SquareIndex(y1, wr.loLat, wr.hiLat) * 3 +
				 	  SquareIndex(x1, wr.loLong, wr.hiLong),
		 toSquare = SquareIndex(y2, wr.loLat, wr.hiLat) * 3 +
				 	SquareIndex(x2, wr.loLong, wr.hiLong),
		 crossing = crossings[fromSquare][toSquare];
	
	if (crossing == 0) return FALSE;
	if (crossing == 1) return TRUE;
	
	return OppositeSides(x1, y1, x2, y2, wr.loLong, wr.loLat, wr.loLong, wr.hiLat) ||
		   OppositeSides(x1, y1, x2, y2, wr.loLong, wr.loLat, wr.hiLong, wr.loLat) ||
		   OppositeSides(x1, y1, x2, y2, wr.loLong, wr.hiLat, wr.hiLong, wr.hiLat) ||
		   OppositeSides(x1, y1, x2, y2, wr.hiLong, wr.loLat, wr.hiLong, wr.hiLat);
}

void ClipSegmentToSegment(long *x1, long *y1, long *x2, long *y2,
						  long X1, long Y1, long X2, long Y2,
						  Boolean modifyFirstPoint, Boolean *touches)
{
	WorldPoint p;
	Segment s, S;
	
	SetSegment(&s, *x1, *y1, *x2, *y2);
	SetSegment(&S, X1, Y1, X2, Y2);
	
	if (!SegmentTouchesSegment(s, S)) return;
	
	(*touches) = TRUE;
	
	p = PointOfIntersection(s, S);
	if (modifyFirstPoint)
		{ *x1 = p.pLong; *y1 = p.pLat; }
	else
		{ *x2 = p.pLong; *y2 = p.pLat; }
}

Boolean ClipSegmentToWRect(long *x1, long *y1, long *x2, long *y2, WorldRect wr)
{
	Boolean touches = FALSE;
	
	ClipSegmentToSegment(x1, y1, x2, y2,
						 wr.loLong, wr.hiLat, wr.hiLong, wr.hiLat, // top
						 (*y1) > (*y2), &touches);
	
	ClipSegmentToSegment(x1, y1, x2, y2,
						 wr.loLong, wr.loLat, wr.hiLong, wr.loLat, // bottom
						 (*y1) < (*y2), &touches);
	
	ClipSegmentToSegment(x1, y1, x2, y2,
						 wr.loLong, wr.loLat, wr.loLong, wr.hiLat, // left
						 (*x1) < (*x2), &touches);
	
	ClipSegmentToSegment(x1, y1, x2, y2,
						 wr.hiLong, wr.loLat, wr.hiLong, wr.hiLat, // right
						 (*x1) > (*x2), &touches);
	
	return touches;
}

////////////////////////////////////////////////////////////// MATRIX OPERATIONS

OSErr Solve3x3Matrix(double m[3][3], double v[3], double *a, double *b, double *c)
{
	short i, j;
	double col[3], dets[3], determinant, dummy;
	
	// d(x, y) = ax + by + c;
	// | x1  y1   1 | | a | = | v1 |
	// | x2  y2   1 | | b | = | v2 |
	// | x3  y3   1 | | c | = | v3 |
	// 
	// use Cramer's rule:	a = detA / det
	//						b = detB / det
	//						c = detC / det
	
	// solve overall determinant
	determinant = Compute3x3Determinant(m);
	dummy = determinant; // avoid compiler bug
	if (determinant == 0) return -1; // matrix has no solution
	
	for (i = 0 ; i < 3 ; i++) {
		// save column i in col and replace column i with v
		for (j = 0 ; j < 3 ; j++) {
			col[j] = m[j][i];
			m[j][i] = v[j];
		}
		
		dets[i] = Compute3x3Determinant(m);
		
		// replace column i with original data
		for (j = 0 ; j < 3 ; j++)
			m[j][i] = col[j];
	}
	
	(*a) = dets[0] / determinant;
	(*b) = dets[1] / determinant;
	(*c) = dets[2] / determinant;
	
	return 0;
}

double Compute3x3Determinant(double m[3][3])
{
	double a, b, c, n[2][2]; // [row][col]
	
	//      0   1   2
    // 0  [ a0, b0, c0 ]
    // 1  [ a1, b1, c1 ]
    // 2  [ a2, b2, c2 ]
	
	// a = a0 * det(b1, c2)
	n[0][0] = m[1][1]; n[0][1] = m[1][2];
	n[1][0] = m[2][1]; n[1][1] = m[2][2];
	a = m[0][0] * Compute2x2Determinant(n);
	
	// b = b0 * det(a1, c2)
	n[0][0] = m[1][0]; n[0][1] = m[1][2];
	n[1][0] = m[2][0]; n[1][1] = m[2][2];
	b = m[0][1] * Compute2x2Determinant(n);
	
	// c = c0 * det(a1, b2)
	n[0][0] = m[1][0]; n[0][1] = m[1][1];
	n[1][0] = m[2][0]; n[1][1] = m[2][1];
	c = m[0][2] * Compute2x2Determinant(n);
	
	return (a - b) + c;
}

double Compute2x2Determinant(double m[2][2])
{
	return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}
////////////////////////////////////////////////////////////// SEGMENTS <-> WPOINTS

WORLDPOINTH SegmentsToWPoints(SEGMENTH segments, long numSegs, WorldRect *wr)
{
	Boolean completedLoop = FALSE;
	long i, numPoints, pieceStartLong = -1, pieceStartLat = -1;
	WORLDPOINTH wPoints;
	
	// overallocate, assuming each segment has its own FROM point
	wPoints = (WORLDPOINTH)_NewHandle((numSegs  * 2) * sizeof(WorldPoint));
	if (_MemError()) { TechError("SegmentsToWPoints()", "_NewHandle()", 0); return 0; }
	
	numPoints = 0;
	
	for (i = 0 ; i < numSegs ; i++) {
		if (i == 0 || completedLoop ||
			(INDEXH(segments, i).fromLong != INDEXH(wPoints, numPoints - 1).pLong) ||
			(INDEXH(segments, i).fromLat != INDEXH(wPoints, numPoints - 1).pLat)) {
			completedLoop = FALSE;
			pieceStartLong = INDEXH(segments, i).fromLong;
			pieceStartLat = INDEXH(segments, i).fromLat;
			// add FROM point
			INDEXH(wPoints, numPoints).pLong = pieceStartLong | 0x80000000;
			INDEXH(wPoints, numPoints).pLat = pieceStartLat;
			numPoints++;
		}
		// add TO point
		INDEXH(wPoints, numPoints).pLong = INDEXH(segments, i).toLong;
		INDEXH(wPoints, numPoints).pLat = INDEXH(segments, i).toLat;
		numPoints++;
		if (INDEXH(segments, i).toLong == pieceStartLong &&
			INDEXH(segments, i).toLat == pieceStartLat)
			completedLoop = TRUE;
	}
	
	// downsize for actual number of points
	_SetHandleSize((Handle)wPoints, numPoints * sizeof(WorldPoint));
	if (_MemError()) {
		TechError("SegmentsToWPoints()", "_SetHandleSize()", 0);
		DisposeHandle((Handle)wPoints);		
		return 0;
	}
	
	(*wr) = voidWorldRect;
	for (i = 0 ; i < numPoints ; i++) {
		wr->loLong = _min(wr->loLong, INDEXH(wPoints, i).pLong & 0x7FFFFFFF);
		wr->hiLong = _max(wr->hiLong, INDEXH(wPoints, i).pLong & 0x7FFFFFFF);
		wr->loLat = _min(wr->loLat, INDEXH(wPoints, i).pLat);
		wr->hiLat = _max(wr->hiLat, INDEXH(wPoints, i).pLat);
	}
	
	return wPoints;
}

SEGMENTH WPointsToSegments(WORLDPOINTH wPoints, long numPoints, long *numSegs)
{
	long i, j;
	SEGMENTH segments;
	
	if (EqualWPoints(INDEXH(wPoints, 0), INDEXH(wPoints, numPoints - 1)))
		(*numSegs) = numPoints - 1;
	else
		(*numSegs) = numPoints;
	
	segments = (SEGMENTH)_NewHandle((*numSegs) * sizeof(Segment));
	if (_MemError()) { TechError("WPointsToSegments()", "_NewHandle()", 0); return 0; }
	
	for (i = 0 ; i < (*numSegs) ; i++) {
		INDEXH(segments, i).fromLong = INDEXH(wPoints, i).pLong;
		INDEXH(segments, i).fromLat = INDEXH(wPoints, i).pLat;
		if ((*numSegs) == numPoints && i == numPoints - 1)
			j = 0;
		else
			j = i + 1;
		INDEXH(segments, i).toLong = INDEXH(wPoints, j).pLong;
		INDEXH(segments, i).toLat = INDEXH(wPoints, j).pLat;
	}
	
	return segments;
}

PointH SegmentsToPointTypes(SEGMENTH segments, long numSegs, long *numPoints)
{
	Boolean completedLoop = FALSE;
	long i, n, pieceStartLong = -1, pieceStartLat = -1;
	PointH points;
	
	(*numPoints) = 0;
	
	// overallocate, assuming each segment has its own FROM point
	points = (PointH)_NewHandle((numSegs  * 2) * sizeof(PointType));
	if (!points) return 0;
	
	for (i = 0, n = 0 ; i < numSegs ; i++) {
		if (i == 0 || completedLoop ||
			(INDEXH(segments, i).fromLong != INDEXH(points, n - 1).p.pLong) ||
			(INDEXH(segments, i).fromLat != INDEXH(points, n - 1).p.pLat)) {
			completedLoop = FALSE;
			pieceStartLong = INDEXH(segments, i).fromLong;
			pieceStartLat = INDEXH(segments, i).fromLat;
			// add FROM point
			INDEXH(points, n).newPiece = TRUE;
			INDEXH(points, n).p.pLong = pieceStartLong;
			INDEXH(points, n).p.pLat = pieceStartLat;
			n++;
		}
		// add TO point
		INDEXH(points, n).newPiece = FALSE;
		INDEXH(points, n).p.pLong = INDEXH(segments, i).toLong;
		INDEXH(points, n).p.pLat = INDEXH(segments, i).toLat;
		n++;
		if (INDEXH(segments, i).toLong == pieceStartLong &&
			INDEXH(segments, i).toLat == pieceStartLat)
			completedLoop = TRUE;
	}
	
	// downsize for actual number of points
	_SetHandleSize((Handle)points, n * sizeof(PointType));
	if (_MemError()) { DisposeHandle((Handle)points); return 0; }
	
	(*numPoints) = n;
	
	return points;
}

SEGMENTH PointTypesToSegments(PointH wPoints, long firstPoint, long numPoints, long *numSegs)
{
	long i, j, n = firstPoint + numPoints;
	SEGMENTH segments;
	
	if (EqualWPoints(INDEXH(wPoints, 0).p, INDEXH(wPoints, n - 1).p))
		(*numSegs) = numPoints - 1;
	else
		(*numSegs) = numPoints;
	
	segments = (SEGMENTH)_NewHandle((*numSegs) * sizeof(Segment));
	if (_MemError()) { TechError("WPointsToSegments()", "_NewHandle()", 0); return 0; }
	
	for (i = 0 ; i < (*numSegs) ; i++) {
		INDEXH(segments, i).fromLong = INDEXH(wPoints, firstPoint + i).p.pLong;
		INDEXH(segments, i).fromLat = INDEXH(wPoints, firstPoint + i).p.pLat;
		if ((*numSegs) == numPoints && i == numPoints - 1)
			j = 0;
		else
			j = i + 1;
		INDEXH(segments, i).toLong = INDEXH(wPoints, firstPoint + j).p.pLong;
		INDEXH(segments, i).toLat = INDEXH(wPoints, firstPoint + j).p.pLat;
	}
	
	return segments;
}

////////////////////////////////////////////////////////////// POLYGON UNION/INTERSECTION

Boolean CoincidentSegments(Segment s1, Segment s2)
{
	return (s1.fromLat == s2.fromLat && s1.fromLong == s2.fromLong) ||
		   (s1.toLat == s2.fromLat && s1.toLong == s2.fromLong) ||
		   (s1.fromLat == s2.toLat && s1.fromLong == s2.toLong) ||
		   (s1.toLat == s2.toLat && s1.toLong == s2.toLong);
}

Boolean HaveCoincidentSegments(SEGMENTH a, SEGMENTH b)
{
	long numSegsA, numSegsB, i, j;
	
	numSegsA = _GetHandleSize((Handle)a) / sizeof(Segment);
	numSegsB = _GetHandleSize((Handle)b) / sizeof(Segment);
	
	for (i = 0 ; i < numSegsA ; i++)
		for (j = 0 ; j < numSegsB ; j++)
			if (CoincidentSegments(INDEXH(a, i), INDEXH(b, j)))
				return TRUE;
	
	return FALSE;
}

void ExchangeSegs(SEGMENTH S, long a, long b)
{
	long i;
	Segment m;
	
	/* old way
	m = INDEXH(S, a);
	INDEXH(S, a) = INDEXH(S, b);
	INDEXH(S, b) = m;
	*/
	
	// new way
	
	m = INDEXH(S, b);
	for (i = b ; i > a ; i--)
		INDEXH(S, i) = INDEXH(S, i - 1);
	INDEXH(S, a) = m;
}

void SortSegments(SEGMENTH S, long numSegs)
{
	long i, startPiece, nextPos = 0;
	
	while (nextPos < numSegs) {
		// find a segment whose FROM doesn't match the FROM or TO of any other segment
		for (startPiece = nextPos ; startPiece < numSegs ; startPiece++) {
			// check next and previous first as a special case
			if (startPiece > 0)
				if ( (INDEXH(S, startPiece).fromLat == INDEXH(S, startPiece - 1).fromLat &&
					  INDEXH(S, startPiece).fromLong == INDEXH(S, startPiece - 1).fromLong) ||
					 (INDEXH(S, startPiece).fromLat == INDEXH(S, startPiece - 1).toLat &&
					  INDEXH(S, startPiece).fromLong == INDEXH(S, startPiece - 1).toLong) )
					continue;
			if (startPiece < (numSegs - 1))
				if ( (INDEXH(S, startPiece).fromLat == INDEXH(S, startPiece + 1).fromLat &&
					  INDEXH(S, startPiece).fromLong == INDEXH(S, startPiece + 1).fromLong) ||
					 (INDEXH(S, startPiece).fromLat == INDEXH(S, startPiece + 1).toLat &&
					  INDEXH(S, startPiece).fromLong == INDEXH(S, startPiece + 1).toLong) )
					continue;
			// check all segments
			for (i = nextPos ; i < numSegs ; i++) {
				if (i == startPiece) continue;
				if ( (INDEXH(S, startPiece).fromLat == INDEXH(S, i).fromLat &&
					  INDEXH(S, startPiece).fromLong == INDEXH(S, i).fromLong) ||
					 (INDEXH(S, startPiece).fromLat == INDEXH(S, i).toLat &&
					  INDEXH(S, startPiece).fromLong == INDEXH(S, i).toLong) )
					break;
			}
			if (i == numSegs) break;
		}
		
		if (startPiece == numSegs) {
			// couldn't find one,
			// so try to find a segment whose TO doesn't match the FROM or TO of any other segment 
			for (startPiece = nextPos ; startPiece < numSegs ; startPiece++) {
				// check next and previous first as a special case
				if (startPiece > 0)
					if ( (INDEXH(S, startPiece).toLat == INDEXH(S, startPiece - 1).fromLat &&
						  INDEXH(S, startPiece).toLong == INDEXH(S, startPiece - 1).fromLong) ||
						 (INDEXH(S, startPiece).toLat == INDEXH(S, startPiece - 1).toLat &&
						  INDEXH(S, startPiece).fromLong == INDEXH(S, startPiece - 1).toLong) )
						continue;
				if (startPiece < (numSegs - 1))
					if ( (INDEXH(S, startPiece).toLat == INDEXH(S, startPiece + 1).fromLat &&
						  INDEXH(S, startPiece).toLong == INDEXH(S, startPiece + 1).fromLong) ||
						 (INDEXH(S, startPiece).toLat == INDEXH(S, startPiece + 1).toLat &&
						  INDEXH(S, startPiece).toLong == INDEXH(S, startPiece + 1).toLong) )
						continue;
				for (i = nextPos ; i < numSegs ; i++) {
					if (i == startPiece) continue;
					if ( (INDEXH(S, startPiece).toLat == INDEXH(S, i).fromLat &&
						  INDEXH(S, startPiece).toLong == INDEXH(S, i).fromLong) ||
						 (INDEXH(S, startPiece).toLat == INDEXH(S, i).toLat &&
						  INDEXH(S, startPiece).toLong == INDEXH(S, i).toLong) )
						break;
				}
				if (i == numSegs) {
					// reverse the segment, since we want to follow its TO chain
					SwitchLongs(&INDEXH(S, startPiece).toLat, &INDEXH(S, startPiece).fromLat);
					SwitchLongs(&INDEXH(S, startPiece).toLong, &INDEXH(S, startPiece).fromLong);
					break;
				}
			}
			if (startPiece == numSegs)
				// at this point we've checked all the remaining segments and all of their
				// FROM and TO points are inter-connected,
				// so we have a ring and can choose the first segment arbitrarily as the start
				startPiece = nextPos;
		}
		
		// string the TO chain starting from it until a segment without a TO is reached
		if (startPiece != nextPos) ExchangeSegs(S, nextPos, startPiece);
		nextPos++;
		while (nextPos < numSegs) {
			for (i = nextPos ; i < numSegs ; i++) {
				if ( (INDEXH(S, i).fromLat == INDEXH(S, nextPos - 1).toLat &&
					  INDEXH(S, i).fromLong == INDEXH(S, nextPos - 1).toLong) ||
					 (INDEXH(S, i).toLat == INDEXH(S, nextPos - 1).toLat &&
					  INDEXH(S, i).toLong == INDEXH(S, nextPos - 1).toLong) )
					break;
			}
			if (i == numSegs) break;
			if (i != nextPos) ExchangeSegs(S, nextPos, i);
			if (INDEXH(S, nextPos).toLat == INDEXH(S, nextPos - 1).toLat &&
				INDEXH(S, nextPos).toLong == INDEXH(S, nextPos - 1).toLong) {
				// reverse
				SwitchLongs(&INDEXH(S, nextPos).toLat, &INDEXH(S, nextPos).fromLat);
				SwitchLongs(&INDEXH(S, nextPos).toLong, &INDEXH(S, nextPos).fromLong);
			}
			nextPos++;
		}
	}
}

OSErr DeleteSegment(SEGMENTH *segments, long *numSegs, long i)
{
	for ( ; i < ((*numSegs) - 1) ; i++)
		INDEXH(*segments, i) = INDEXH(*segments, i + 1);
	
	_SetHandleSize((Handle)*segments, ((*numSegs) - 1) * sizeof(Segment));
	if (_MemError()) { TechError("DeleteSegment()", "_SetHandleSize()", 0); return -1; }
	
	(*numSegs)--;
	
	return 0;
}

OSErr InsertSegment(SEGMENTH *segments, long *numSegs, long index, WorldPoint m)
{
	long i;
	
	if ((m.pLong == INDEXH(*segments, index).toLong && m.pLat == INDEXH(*segments, index).toLat) ||
		(m.pLong == INDEXH(*segments, index).fromLong && m.pLat == INDEXH(*segments, index).fromLat))
		return 0;
	
	_SetHandleSize((Handle)*segments, ((*numSegs) + 1) * sizeof(Segment));
	if (_MemError()) { TechError("AddSegment()", "_SetHandleSize()", 0); return -1; }
	
	/* old way: insert at end
	INDEXH(*segments, *numSegs) = INDEXH(*segments, index);
	INDEXH(*segments, index).toLong = m.pLong;
	INDEXH(*segments, index).toLat = m.pLat;
	INDEXH(*segments, *numSegs).fromLong = m.pLong;
	INDEXH(*segments, *numSegs).fromLat = m.pLat;
	*/
	
	(*numSegs)++;
	
	// new way: insert as next segment
	
	for (i = (*numSegs) - 1 ; i > index ; i--)
		INDEXH(*segments, i) = INDEXH(*segments, i - 1);
	INDEXH(*segments, index).toLong = m.pLong;
	INDEXH(*segments, index).toLat = m.pLat;
	INDEXH(*segments, index + 1).fromLong = m.pLong;
	INDEXH(*segments, index + 1).fromLat = m.pLat;
	
	return 0;
}

OSErr AddSegment(SEGMENTH *segments, long *numSegs, Segment s)
{
	_SetHandleSize((Handle)*segments, ((*numSegs) + 1) * sizeof(Segment));
	if (_MemError()) { TechError("AddSegment()", "_SetHandleSize()", 0); return -1; }
	
	INDEXH(*segments, *numSegs) = s;
	(*numSegs)++;
	
	return 0;
}

OSErr AddSegmentI(SEGMENTH *segments, long *numSegs, Segment s, long i)
{
	long j;
	
	_SetHandleSize((Handle)*segments, ((*numSegs) + 1) * sizeof(Segment));
	if (_MemError()) { TechError("AddSegment()", "_SetHandleSize()", 0); return -1; }
	
	for (j = *numSegs ; j > i ; j--)
		INDEXH(*segments, j) = INDEXH(*segments, j - 1);
	
	INDEXH(*segments, i) = s;
	(*numSegs)++;
	
	return 0;
}

/*
OSErr RemoveCommonSegments(SEGMENTH *P, long *numSegsP, SEGMENTH *Q, long *numSegsQ)
{
	long i, j, err;
	
	for (i = 0 ; i < (*numSegsP) ; i++)
		for (j = 0 ; j < (*numSegsQ) ; j++)
			if (EqualSegments2(INDEXH(*P, i), INDEXH(*Q, j))) {
				if (err = DeleteSegment(P, numSegsP, i)) return err;
				if (err = DeleteSegment(Q, numSegsQ, j)) return err;
				i--;
				break;
			}
	
	return 0;
}
*/

OSErr CompletePolygonIslands(SEGMENTH *P, long *numSegs)
{
	Segment s;
	long i, err, islandStart = 0;
	
	for (i = 1 ; i <= *numSegs ; i++)
		if (i == *numSegs || (INDEXH(*P, i).fromLong != INDEXH(*P, i - 1).toLong ||
							  INDEXH(*P, i).fromLat != INDEXH(*P, i - 1).toLat)) {
			if (INDEXH(*P, i - 1).toLong != INDEXH(*P, islandStart).fromLong ||
				INDEXH(*P, i - 1).toLat != INDEXH(*P, islandStart).fromLat) {
				s.fromLong = INDEXH(*P, i - 1).toLong;
				s.fromLat = INDEXH(*P, i - 1).toLat;
				s.toLong = INDEXH(*P, islandStart).fromLong;
				s.toLat = INDEXH(*P, islandStart).fromLat;
				if (err = AddSegmentI(P, numSegs, s, i)) return err;
				i++;
			}
			islandStart = i;
		}
	
	return 0;
}

OSErr AppendPolygons(SEGMENTH *a, SEGMENTH *b)
{
	long numSegsA, numSegsB, i;
	
	numSegsA = _GetHandleSize((Handle)*a) / sizeof(Segment);
	numSegsB = _GetHandleSize((Handle)*b) / sizeof(Segment);
	
	_SetHandleSize((Handle)*a, (numSegsA + numSegsB) * sizeof(Segment));
	if (_MemError())
		{ TechError("AppendPolygons()", "_SetHandleSize()", 0); return -1; }
	
	for (i = 0 ; i < numSegsB ; i++)
		INDEXH(*a, numSegsA + i) = INDEXH(*b, i);
	
	return 0;
}

SEGMENTH NextPolygonPiece(SEGMENTH segments, long numSegs,
						  long start, long *end, long *count, WorldRect *bounds)
{
	long i, j;
	SEGMENTH S;
	
	if (start >= numSegs) return 0;
	
	if (bounds) (*bounds) = voidWorldRect;
	
	for (i = start ; i < numSegs ; i++) {
		if (i > start)
			if (INDEXH(segments, i).fromLat != INDEXH(segments, i - 1).toLat ||
				INDEXH(segments, i).fromLong != INDEXH(segments, i - 1).toLong)
				break;
		if (bounds) {
			AddWPointToWRect(INDEXH(segments, i).fromLat, INDEXH(segments, i).fromLong, bounds);
			AddWPointToWRect(INDEXH(segments, i).toLat, INDEXH(segments, i).toLong, bounds);
		}
		if (INDEXH(segments, i).toLat == INDEXH(segments, start).fromLat &&
			INDEXH(segments, i).toLong == INDEXH(segments, start).fromLong)
			{ i++; break; }
	}
	
	(*end) = i;
	(*count) = i - start;
	
	S = (SEGMENTH)_NewHandle((*count) * sizeof(Segment));
	if (_MemError()) { TechError("NextPolygonPiece()", "_NewHandle()", 0); return 0; }
	
	for (i = start, j = 0 ; i < (*end) ; i++, j++)
		INDEXH(S, j) = INDEXH(segments, i);
	
	return S;
}

SEGMENTH CombinePolygonPieces(SEGMENTH *A1, SEGMENTH *B1,
							  Boolean keepAinB, Boolean keepAnotinB,
							  Boolean keepBinA, Boolean keepBnotinA)
{
	long i, j, err, numSegsA, numSegsB, numSegsC = 0;
	WorldPoint m;
	SEGMENTH C = 0, A2 = 0, B2 = 0, *A = 0, *B = 0;
	
	err = 0;
	
	numSegsA = _GetHandleSize((Handle)*A1) / sizeof(Segment);
	numSegsB = _GetHandleSize((Handle)*B1) / sizeof(Segment);
	
	A2 = (SEGMENTH)_NewHandle(numSegsA * sizeof(Segment));
	if (_MemError()) { TechError("CombinePolygonPieces()", "_NewHandle()", 0); goto done; }
	for (i = 0 ; i < numSegsA ; i++)
		INDEXH(A2, i) = INDEXH(*A1, i);
	A = &A2;
	
	B2 = (SEGMENTH)_NewHandle(numSegsB * sizeof(Segment));
	if (_MemError()) { TechError("CombinePolygonPieces()", "_NewHandle()", 0); goto done; }
	for (i = 0 ; i < numSegsB ; i++)
		INDEXH(B2, i) = INDEXH(*B1, i);
	B = &B2;
	
	for (i = 0 ; i < numSegsA ; i++)
		for (j = 0 ; j < numSegsB ; j++)
			if (SegmentTouchesSegment(INDEXH(*A, i), INDEXH(*B, j)) &&
				!SameSegmentEndPoints(INDEXH(*A, i), INDEXH(*B, j))) {
				m = PointOfIntersection(INDEXH(*A, i), INDEXH(*B, j));
				if (err = InsertSegment(A, &numSegsA, i, m)) goto done;
				if (err = InsertSegment(B, &numSegsB, j, m)) goto done;
			}
	
	C = (SEGMENTH)_NewHandle(0);
	if (_MemError()) { TechError("CombinePolygonPieces()", "_NewHandle()", 0); goto done; }
	
	for (i = 0 ; i < numSegsA ; i++) {
		m = Midpoint(INDEXH(*A, i));
		if ((keepAinB && PointInPolygon(m, *B, numSegsB, TRUE)) ||
			(keepAnotinB && !PointInPolygon(m, *B, numSegsB, TRUE)))
			if (err = AddSegment(&C, &numSegsC, INDEXH(*A, i))) goto done;
	}
	for (j = 0 ; j < numSegsB ; j++) {
		m = Midpoint(INDEXH(*B, j));
		if ((keepBinA && PointInPolygon(m, *A, numSegsA, TRUE)) ||
			(keepBnotinA && !PointInPolygon(m, *A, numSegsA, TRUE)))
			if (err = AddSegment(&C, &numSegsC, INDEXH(*B, j))) goto done;
	}
	
	SortSegments(C, numSegsC);
	
	done:
		if (A2) DisposeHandle((Handle)A2);
		if (B2) DisposeHandle((Handle)B2);
		if (err && C) DisposeHandle((Handle)C);
		
		return err ? 0 : C;
}

SEGMENTH UnionPolygons(SEGMENTH *P, SEGMENTH *Q, Boolean *appended)
{
	SEGMENTH total;
	OSErr err = 0;
	
	(*appended) = FALSE;
	
	if (HaveCoincidentSegments(*P, *Q)) {
		(*appended) = TRUE;
		
		total = (SEGMENTH)_NewHandle(0);
		if (_MemError()) { TechError("UnionPolygons()", "_NewHandle()", 0); return 0; }
		
		if (err = AppendPolygons(&total, P)) return total;
		if (err = AppendPolygons(&total, Q)) return total;
		
		return total;
	}
	
	return CombinePolygonPieces(P, Q, FALSE, TRUE, FALSE, TRUE);
}

SEGMENTH IntersectPolygons(SEGMENTH *P, SEGMENTH *Q)
{
	return CombinePolygonPieces(P, Q, TRUE, FALSE, TRUE, FALSE);
}

SEGMENTH DifferencePolygons(SEGMENTH *P, SEGMENTH *Q)
{
	return CombinePolygonPieces(P, Q, FALSE, TRUE, TRUE, FALSE);
}

/*

// NOTE: This function modifies P and Q (only by completing their islands).

SEGMENTH ComputePolygons(SEGMENTH *P, SEGMENTH *Q, short function,
						 Boolean keepAinB, Boolean keepAnotinB,
						 Boolean keepBinA, Boolean keepBnotinA);
{
	long i, err, numSegsP, numSegsQ,
		 countA, countB, endP, endQ, startP = 0, startQ = 0;
	WorldRect boundsA, boundsB;
	SEGMENTH A = 0, B = 0, C = 0, total = 0;
	
	numSegsP = _GetHandleSize((Handle)*P) / sizeof(Segment);
	numSegsQ = _GetHandleSize((Handle)*Q) / sizeof(Segment);
	
	if (err = CompletePolygonIslands(P, &numSegsP)) goto done;
	if (err = CompletePolygonIslands(Q, &numSegsQ)) goto done;
	
	switch (function) {
		case UNIONPOLYGONS:
			total = (SEGMENTH)MyNewHandleTemp(numSegsQ * sizeof(Segment));
			if (_MemError()) { TechError("UnionPolygons()", "_NewHandle()", 0); return 0; }
			for (i = 0 ; i < numSegsQ ; i++)
				INDEXH(total, i) = INDEXH(*Q, i);
			break;
		case INTERSECTPOLYGONS:
			total = (SEGMENTH)MyNewHandleTemp(0);
			if (_MemError()) { TechError("IntersectPolygons()", "_NewHandle()", 0); return 0; }
			break;
		case DIFFERENCEPOLYGONS:
			break;
	}
	
	// if (err = RemoveCommonSegments(P, &numSegsP, Q, &numSegsQ)) goto done;
	
	while (A = NextPolygonPiece(*P, numSegsP, startP, &endP, &countA, &boundsA)) {
 		while (B = NextPolygonPiece(*Q, numSegsQ, startQ, &endQ, &countB, &boundsB)) {
			if (WRectTouchesWRect(&boundsB, &boundsA)) {
				C = CombinePolygonPieces(&A, countA, &B, countB,
										 keepAinB, keepAnotinB,
										 keepBinA, keepBnotinA);
				if (!C) { err = -1; goto done; }
				switch (function) {
					case UNIONPOLYGONS:
						DisposeHandle((Handle)A);
						A = C;
						C = 0;
						break;
					case INTERSECTPOLYGONS:
							if (err = AppendPolygons(&total, &C)) goto done;
							DisposeHandle((Handle)C); C = 0;
						break;
					case DIFFERENCEPOLYGONS: break;
				}
			}
			DisposeHandle((Handle)B); B = 0;
			startQ = endQ;
		}
		switch (function) {
			case UNIONPOLYGONS:
				if (err = AppendPolygons(&total, &A)) goto done;
				break;
			case INTERSECTPOLYGONS:
				DisposeHandle((Handle)A);
				A = 0;
				break;
			case DIFFERENCEPOLYGONS: break;
		}
		DisposeHandle((Handle)A); A = 0;
		startP = endP;
		startQ = 0;
	}
	
	done:
		if (A) DisposeHandle((Handle)A);
		if (B) DisposeHandle((Handle)B);
		if (C) DisposeHandle((Handle)C);
		if (err && total) DisposeHandle((Handle)total);
		
		return err ? 0 : total;
}

SEGMENTH UnionPolygons(SEGMENTH *P, SEGMENTH *Q)
{
	return ComputePolygon(P, Q. UNIONPOLYGONS, FALSE, TRUE, FALSE, TRUE);
}

SEGMENTH IntersectPolygons(SEGMENTH *P, SEGMENTH *Q)
{
	return ComputePolygon(P, Q. INTERSECTPOLYGONS, TRUE, FALSE, TRUE, FALSE);
}

SEGMENTH DifferencePolygons(SEGMENTH *P, SEGMENTH *Q)
{
	return ComputePolygon(P, Q. DIFFERENCEPOLYGONS, FALSE, TRUE, TRUE, FALSE);
}

SEGMENTH UnionPolygons(SEGMENTH *P, SEGMENTH *Q)
{
	long err, numSegsP, numSegsQ;
	SEGMENTH total = 0;
	
	numSegsP = _GetHandleSize((Handle)*P) / sizeof(Segment);
	numSegsQ = _GetHandleSize((Handle)*Q) / sizeof(Segment);
	
	if (err = CompletePolygonIslands(P, &numSegsP)) goto done;
	if (err = CompletePolygonIslands(Q, &numSegsQ)) goto done;
	
	total = (SEGMENTH)MyNewHandleTemp(0);
	if (_MemError()) { TechError("UnionPolygons()", "_NewHandle()", 0); return 0; }
	
	if (err = AppendPolygons(&total, P)) goto done;
	if (err = AppendPolygons(&total, Q)) goto done;
	
	done:
		if (err && total) DisposeHandle((Handle)total);
		
		return err ? 0 : total;
}
*/

////////////////////////////////////////////////////////////// ENVELOPES

Boolean PointOutsideEnvelopePolygons(WorldPoint points[], long i,
									 SEGMENTH prevBox, SEGMENTH prevCircle,
									 SEGMENTH nextBox, SEGMENTH nextCircle,
									 long nPB, long nPC, long nNB, long nNC)
{
	if (prevBox && PointInPolygon(points[i], prevBox, nPB, FALSE)) return FALSE;
	if (prevCircle && PointInPolygon(points[i], prevCircle, nPC, FALSE)) return FALSE;
	if (nextBox && PointInPolygon(points[i], nextBox, nNB, FALSE)) return FALSE;
	if (nextCircle && PointInPolygon(points[i], nextCircle, nNC, FALSE)) return FALSE;
	
	return TRUE;
}

SEGMENTH PointToEnvelopePolyCircle(float radius, long pLong, long pLat,
								   SEGMENTH prevBox, SEGMENTH prevCircle,
								   SEGMENTH nextBox, SEGMENTH nextCircle,
								   float angleOffset)
{
	Boolean centerAdded = FALSE;
	long dLong, dLat, x, y, nPB, nPC, nNB, nNC, i, j, n = 0;
	float a;
	WorldPoint center, points[32], points2[33];
	SEGMENTH circle;
	
	center.pLong = pLong; // + 1; // imperceptible offset to make sure circles and rectangles
	center.pLat = pLat; // + 1; // never share points exactly (which can lead to drawing problems)
	
	dLong = DistanceToLong(radius, center);
	dLat = DistanceToLat(radius);
	
	for (i = 0 ; i < 32 ; i++) {
		a = 2 * PI * (i / 32.0) + angleOffset;
		x = cos(a) * dLong;
		y = sin(a) * dLat;
		points[i].pLong = center.pLong + x;
		points[i].pLat = center.pLat + y;
	}
	
	if (prevBox) nPB = _GetHandleSize((Handle)prevBox) / sizeof(Segment);
	if (prevCircle) nPC = _GetHandleSize((Handle)prevCircle) / sizeof(Segment);
	if (nextBox) nNB = _GetHandleSize((Handle)nextBox) / sizeof(Segment);
	if (nextCircle) nNC = _GetHandleSize((Handle)nextCircle) / sizeof(Segment);
	
	// we want to make sure the string of kept points starts with an "inside" point,
	// of which there will always be at least one
	
	for (i = 0 ; i < 32 ; i++)
		if (!PointOutsideEnvelopePolygons(points, i,
										  prevBox, prevCircle, nextBox, nextCircle,
										  nPB, nPC, nNB, nNC))
			break;
	
	if (i == 32) i = 0; // should never happen
	
	for (j = 0 ; j < 32 ; j++, i = (i + 1) % 32) {
		if (PointOutsideEnvelopePolygons(points, (i == 0) ? 31 : i - 1,
										 prevBox, prevCircle, nextBox, nextCircle,
										 nPB, nPC, nNB, nNC) ||
			PointOutsideEnvelopePolygons(points, i,
										 prevBox, prevCircle, nextBox, nextCircle,
										 nPB, nPC, nNB, nNC) ||
			PointOutsideEnvelopePolygons(points, (i == 31) ? 0 : i + 1,
										 prevBox, prevCircle, nextBox, nextCircle,
										 nPB, nPC, nNB, nNC)) {
			points2[n] = points[i];
			n++;
		}
		else
			if (!centerAdded) {
				points2[n] = center;
				n++;
				centerAdded = TRUE;
			}
	}
	
	if (n < 3) {
		if (angleOffset < ((2 * PI) / 32))
			return PointToEnvelopePolyCircle(radius, pLong, pLat,
											 prevBox, prevCircle, nextBox, nextCircle,
											 angleOffset + (2 * PI) / 320);
		else
			n = 0;
	}
	
	circle = (SEGMENTH)_NewHandle(n * sizeof(Segment));
	if (_MemError()) { TechError("PointToEnvelopePolyCircle()", "_NewHandle()", 0); return 0; }
	
	for (i = 0 ; i < n ; i++) {
		INDEXH(circle, i).fromLong = points2[i].pLong;
		INDEXH(circle, i).fromLat = points2[i].pLat;
		if (i == 0) {
			INDEXH(circle, n - 1).toLong = points2[i].pLong;
			INDEXH(circle, n - 1).toLat = points2[i].pLat;
		}
		else {
			INDEXH(circle, i - 1).toLong = points2[i].pLong;
			INDEXH(circle, i - 1).toLat = points2[i].pLat;
		}
	}
	
	return circle;
}

SEGMENTH SegToEnvelopeRectangle(float radius, Segment s)
{
	SEGMENTH rectangle;
	WorldPoint center;
	WorldRect wr;
	long dLong, dLat, xPlus, yPlus, xMinus, yMinus;
	float a, ratio, rise, run;
	
	wr.loLat = _min(s.fromLat, s.toLat);
	wr.hiLat = _max(s.fromLat, s.toLat);
	ratio = LongToLatRatio2(&wr);
	
	rise = abs(s.toLat - s.fromLat);
	run = abs(s.toLong - s.fromLong) / ratio;
	
	a = asin(rise / sqrt(rise * rise + run * run));
	
	if ((s.fromLat < s.toLat) != (s.fromLong < s.toLong)) a *= -1;
	
	center.pLong = (s.fromLong + s.toLong) / 2;
	center.pLat = (s.fromLat + s.toLat) / 2;
	
	dLong = DistanceToLong(radius, center);
	dLat = DistanceToLat(radius);
	
	rectangle = (SEGMENTH)_NewHandle(4 * sizeof(Segment));
	if (_MemError()) { TechError("SegToEnvelopeRectangle()", "_NewHandle()", 0); return 0; }
	
	xPlus = cos((PI / 2) + a) * dLong;
	yPlus = sin((PI / 2) + a) * dLat;
	xMinus = cos((-PI / 2) + a) * dLong;
	yMinus = sin((-PI / 2) + a) * dLat;
	
	INDEXH(rectangle, 3).toLong = INDEXH(rectangle, 0).fromLong = s.fromLong + xPlus;
	INDEXH(rectangle, 3).toLat = INDEXH(rectangle, 0).fromLat = s.fromLat + yPlus;
	
	INDEXH(rectangle, 0).toLong = INDEXH(rectangle, 1).fromLong = s.toLong + xPlus;
	INDEXH(rectangle, 0).toLat = INDEXH(rectangle, 1).fromLat = s.toLat + yPlus;
	
	INDEXH(rectangle, 1).toLong = INDEXH(rectangle, 2).fromLong = s.toLong + xMinus;
	INDEXH(rectangle, 1).toLat = INDEXH(rectangle, 2).fromLat = s.toLat + yMinus;
	
	INDEXH(rectangle, 2).toLong = INDEXH(rectangle, 3).fromLong = s.fromLong + xMinus;
	INDEXH(rectangle, 2).toLat = INDEXH(rectangle, 3).fromLat = s.fromLat + yMinus;
	
	return rectangle;
}

SEGMENTH ComputeEnvelope(SEGMENTH segments, float radius)
{
	Boolean newIsland, islandEnd;
	long i, err, numSegs = _GetHandleSize((Handle)segments) / sizeof(Segment);
	SEGMENTH total = 0, piece = 0, boxA = 0, boxB = 0, circleA = 0, circleB = 0;
	
	total = (SEGMENTH)_NewHandle(0);
	if (_MemError()) { TechError("ComputeEnvelope()", "_NewHandle()", 0); err = -1; goto done; }
	
	for (i = 0 ; i < numSegs ; i++) {
		newIsland = (i == 0 ||
					 INDEXH(segments, i).fromLong != INDEXH(segments, i - 1).toLong ||
					 INDEXH(segments, i).fromLat != INDEXH(segments, i - 1).toLat);
		islandEnd = (i == (numSegs - 1) ||
					 INDEXH(segments, i).toLong != INDEXH(segments, i + 1).fromLong ||
					 INDEXH(segments, i).toLat != INDEXH(segments, i + 1).fromLat);
		if (newIsland) {
			boxB = SegToEnvelopeRectangle(radius, INDEXH(segments, i));
			if (!boxB) goto done;
			circleB = PointToEnvelopePolyCircle(radius,
												INDEXH(segments, i).toLong,
												INDEXH(segments, i).toLat,
												0, 0, 0, 0, 0);
			if (!circleB) goto done;
			// add the circle about the "from" vertex
			piece = PointToEnvelopePolyCircle(radius,
											  INDEXH(segments, i).fromLong,
											  INDEXH(segments, i).fromLat,
											  0, 0, boxB, circleB, 0);
			DisposeHandle((Handle)circleB);
			circleB = 0;
			if (!piece) goto done;
			if (err = AppendPolygons(&total, &piece)) goto done;
			DisposeHandle((Handle)piece);
			piece = 0;
			
			circleA = PointToEnvelopePolyCircle(radius,
												INDEXH(segments, i).fromLong,
												INDEXH(segments, i).fromLat,
												0, 0, 0, 0, 0);
			if (!circleA) goto done;
			boxA = boxB;
			boxB = 0;
		}
		
		// add the rectangle about the segment
		if (err = AppendPolygons(&total, &boxA)) goto done;
		
		// compute following rectangle and circle
		if (!islandEnd) {
			boxB = SegToEnvelopeRectangle(radius, INDEXH(segments, i + 1));
			if (!boxB) goto done;
			circleB = PointToEnvelopePolyCircle(radius,
												INDEXH(segments, i + 1).toLong,
												INDEXH(segments, i + 1).toLat,
												0, 0, 0, 0, 0);
			if (!circleB) goto done;
		}
		
		// add the circle about the "to" vertex
		piece = PointToEnvelopePolyCircle(radius,
										  INDEXH(segments, i).toLong,
										  INDEXH(segments, i).toLat,
										  boxA, circleA, boxB, circleB, 0);
		if (circleB) DisposeHandle((Handle)circleB);
		circleB = 0;
		if (!piece) { err = -1; goto done; }
		if (err = AppendPolygons(&total, &piece)) goto done;

		DisposeHandle((Handle)piece);
		piece = 0;
		
		// shift new to old
		if (boxA) DisposeHandle((Handle)boxA);
		boxA = boxB;
		boxB = 0;
		if (circleA) DisposeHandle((Handle)circleA);
		circleA = PointToEnvelopePolyCircle(radius,
											INDEXH(segments, i).toLong,
											INDEXH(segments, i).toLat,
											0, 0, 0, 0, 0);
		if (!circleA) goto done;
	}
	
	done:
		
		if (piece) DisposeHandle((Handle)piece);
		if (boxA) DisposeHandle((Handle)boxA);
		if (boxB) DisposeHandle((Handle)boxB);
		if (circleA) DisposeHandle((Handle)circleA);
		if (circleB) DisposeHandle((Handle)circleB);
		
		return total;
}
