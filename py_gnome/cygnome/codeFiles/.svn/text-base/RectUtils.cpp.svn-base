
#include "Cross.h"
#include	"RectUtils.h"
#include	"GenDefs.h"
/**************************************************************************************************/
Boolean operator != (Rect r1, Rect r2)
{
	if (r1.top    == r2.top &&
		r1.left   == r2.left &&
		r1.bottom == r2.bottom &&
		r1.right  == r2.right)
		return (false);
	else
		return (true);
}
/**************************************************************************************************/
Boolean operator == (Rect r1, Rect r2)
{
	if (r1.top    == r2.top &&
		r1.left   == r2.left &&
		r1.bottom == r2.bottom &&
		r1.right  == r2.right)
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
Boolean operator == (LongPoint p1, LongPoint p2)
{
	if (p1.h == p2.h && p1.v == p2.v)
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
Boolean operator != (LongPoint p1, LongPoint p2)
{
	if (p1.h == p2.h && p1.v == p2.v)
		return (false);
	else
		return (true);
}
/**************************************************************************************************/
Boolean operator == (Point p1, Point p2)
{
	if (p1.h == p2.h && p1.v == p2.v)
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
Boolean operator != (Point p1, Point p2)
{
	if (p1.h == p2.h && p1.v == p2.v)
		return (false);
	else
		return (true);
}
/**************************************************************************************************/
Boolean operator != (LongRect r1, LongRect r2)
{
	if (r1.top    == r2.top &&
		r1.left   == r2.left &&
		r1.bottom == r2.bottom &&
		r1.right  == r2.right)
		return (false);
	else
		return (true);
}
/**************************************************************************************************/
Boolean operator == (LongRect r1, LongRect r2)
{
	if (r1.top    == r2.top &&
		r1.left   == r2.left &&
		r1.bottom == r2.bottom &&
		r1.right  == r2.right)
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
/**************************************************************************************************/
/* given a rectangle in SmallRect, this subroutine retuns the smallest rectangle that can be drawn
	around SmallRect with the ratio supplied in XToYRatio.  The result is put in *SmallestRectPtr */
void GetSmallestRect (Rect *SmallRectPtr, double XToYRatio, Rect *SmallestRectPtr)
{
	short	SmallRectDX, SmallRectDY, ScaledDX, ScaledDY;
	
	SmallRectDX = SmallRectPtr -> right - SmallRectPtr -> left;
	SmallRectDY = SmallRectPtr -> bottom - SmallRectPtr -> top;
	
	/* use delta X to see if height will come out too big */
	ScaledDX = SmallRectDX;
	ScaledDY = SmallRectDX / XToYRatio;
	if (ScaledDY < SmallRectDY)
	{
		ScaledDY = SmallRectDY;
		ScaledDX = SmallRectDY * XToYRatio;
	}
	
	*SmallestRectPtr = *SmallRectPtr;
	SmallestRectPtr -> right = SmallestRectPtr -> left + ScaledDX;
	SmallestRectPtr -> bottom = SmallestRectPtr -> top + ScaledDY;
	
	return;
}
/**************************************************************************************************/
/* given a rectangle in SmallRect, this subroutine retuns the smallest rectangle that can be drawn
	around SmallRect with the ratio supplied in XToYRatio.  The result is put in *SmallestRectPtr */
void GetSmallestLRect (LongRect *SmallRectPtr, double XToYRatio, LongRect *SmallestRectPtr)
{
	long	SmallRectDX, SmallRectDY, ScaledDX, ScaledDY;
	Boolean	TopGTBottom;
	
	SmallRectDX = SmallRectPtr -> right - SmallRectPtr -> left;
	
	if (SmallRectPtr -> top > SmallRectPtr -> bottom)
	{
		SmallRectDY = SmallRectPtr -> top - SmallRectPtr -> bottom;
		TopGTBottom = true;
	}
	else
	{
		SmallRectDY = SmallRectPtr -> bottom - SmallRectPtr -> top;
		TopGTBottom = false;
	}
	
	/* use delta X to see if height will come out too big */
	ScaledDX = SmallRectDX;
	ScaledDY = SmallRectDX / XToYRatio;
	if (ScaledDY < SmallRectDY)
	{
		ScaledDY = SmallRectDY;
		ScaledDX = SmallRectDY * XToYRatio;
	}
	
	*SmallestRectPtr = *SmallRectPtr;
	SmallestRectPtr -> right = SmallestRectPtr -> left + ScaledDX;
	
	if (TopGTBottom)
		SmallestRectPtr -> bottom = SmallestRectPtr -> top - ScaledDY;
	else
		SmallestRectPtr -> bottom = SmallestRectPtr -> top + ScaledDY;
	
	return;
}
/**************************************************************************************************/
void SetLRect (LongRectPtr LRectPtr, long left, long top, long right, long bottom)
{
	LRectPtr -> left   = left;
	LRectPtr -> top    = top;
	LRectPtr -> right  = right;
	LRectPtr -> bottom = bottom;
	
	return;
}
/**************************************************************************************************/
void SetExRect (ExRect *ExRectPtr, double left, double top, double right, double bottom)
{
	ExRectPtr -> left   = left;
	ExRectPtr -> top    = top;
	ExRectPtr -> right  = right;
	ExRectPtr -> bottom = bottom;
	
	return;
}
/**************************************************************************************************/
/* given two long rectangles, this subroutine calculates the smallest rectangle that encompasses
   a areas covered by both rectangles. The resulting rect is stored in *MaxRectPtr				  */
/* warning: this subroutine assumes the top is greater than the bottom for both rects */

void GetMaxLRect (LongRect *Rect1Ptr, LongRect *Rect2Ptr, LongRect *MaxRectPtr)
{
	if (Rect1Ptr -> top > Rect2Ptr -> top)
		MaxRectPtr -> top = Rect1Ptr -> top;
	else
		MaxRectPtr -> top = Rect2Ptr -> top;
		
	if (Rect1Ptr -> left < Rect2Ptr -> left)
		MaxRectPtr -> left = Rect1Ptr -> left;
	else
		MaxRectPtr -> left = Rect2Ptr -> left;
	
	if (Rect1Ptr -> right > Rect2Ptr -> right)
		MaxRectPtr -> right = Rect1Ptr -> right;
	else
		MaxRectPtr -> right = Rect2Ptr -> right;

	if (Rect1Ptr -> bottom < Rect2Ptr -> bottom)
		MaxRectPtr -> bottom = Rect1Ptr -> bottom;
	else
		MaxRectPtr -> bottom = Rect2Ptr -> bottom;

	return;
}
/**************************************************************************************************/
/* given two rectangles, this subroutine calculates the smallest rectangle that encompasses
   a areas covered by both rectangles. The resulting rect is stored in *MaxRectPtr				  */
   
void GetMaxRect (Rect *Rect1Ptr, Rect *Rect2Ptr, Rect *MaxRectPtr)
{
	if (Rect1Ptr -> top < Rect2Ptr -> top)
		MaxRectPtr -> top = Rect1Ptr -> top;
	else
		MaxRectPtr -> top = Rect2Ptr -> top;
		
	if (Rect1Ptr -> left < Rect2Ptr -> left)
		MaxRectPtr -> left = Rect1Ptr -> left;
	else
		MaxRectPtr -> left = Rect2Ptr -> left;
	
	if (Rect1Ptr -> right > Rect2Ptr -> right)
		MaxRectPtr -> right = Rect1Ptr -> right;
	else
		MaxRectPtr -> right = Rect2Ptr -> right;

	if (Rect1Ptr -> bottom > Rect2Ptr -> bottom)
		MaxRectPtr -> bottom = Rect1Ptr -> bottom;
	else
		MaxRectPtr -> bottom = Rect2Ptr -> bottom;

	return;
}
/**************************************************************************************************/
void NudgeRect (Rect *RectPtr, Rect *LimitRectPtr)
{
	if (RectPtr -> left < LimitRectPtr -> left)
	{
		RectPtr -> right = RectPtr -> right + (LimitRectPtr -> left - RectPtr -> left);
		RectPtr -> left = LimitRectPtr -> left;
	}
	
	if (RectPtr -> top < LimitRectPtr -> top)
	{
		RectPtr -> bottom = RectPtr -> bottom + (LimitRectPtr -> top - RectPtr -> top);
		RectPtr -> top = LimitRectPtr -> top;
	}

	if (RectPtr -> right > LimitRectPtr -> right)
	{
		RectPtr -> left = RectPtr -> left - (RectPtr -> right - LimitRectPtr -> right);
		RectPtr -> right = LimitRectPtr -> right;
	}

	if (RectPtr -> bottom > LimitRectPtr -> bottom)
	{
		RectPtr -> top = RectPtr -> top - (RectPtr -> bottom - LimitRectPtr -> bottom);
		RectPtr -> bottom = LimitRectPtr -> bottom;
	}

	return;
}
/**************************************************************************************************/
Boolean IntersectLRect (LongRectPtr Rect1Ptr, LongRectPtr Rect2Ptr, LongRectPtr CommonLRectPtr)
/* given pointers to two rectangles, this subroutine returns a flag indicating whether there is
	any overlap between the two rects */
{
	Boolean 	XIntersects, YIntersects;
	
	XIntersects = false;
	YIntersects = false;
	
	/* check the x-coordinates */
	if ((Rect2Ptr -> left >= Rect1Ptr -> left) && (Rect2Ptr -> left <= Rect1Ptr -> right))
		XIntersects = true;
	if ((Rect2Ptr -> right >= Rect1Ptr -> left) && (Rect2Ptr -> right <= Rect1Ptr -> right))
		XIntersects = true;		
	if ((Rect1Ptr -> left >= Rect2Ptr -> left) && (Rect1Ptr -> left <= Rect2Ptr -> right))
		XIntersects = true;
	if ((Rect1Ptr -> right >= Rect2Ptr -> left) && (Rect1Ptr -> right <= Rect2Ptr -> right))
		XIntersects = true;
	
	if (XIntersects)	/* if x-coordinates intersect, check the y-coordinates */
	{
		if ((Rect2Ptr -> top <= Rect1Ptr -> top) && (Rect2Ptr -> top >= Rect1Ptr -> bottom))
			YIntersects = true;	
		if ((Rect2Ptr -> bottom <= Rect1Ptr -> top) && (Rect2Ptr -> bottom >= Rect1Ptr -> bottom))
			YIntersects = true;
		if ((Rect1Ptr -> top <= Rect2Ptr -> top) && (Rect1Ptr -> top >= Rect2Ptr -> bottom))
			YIntersects = true;	
		if ((Rect1Ptr -> bottom <= Rect2Ptr -> top) && (Rect1Ptr -> bottom >= Rect2Ptr -> bottom))
			YIntersects = true;
	}
	
	if (XIntersects && YIntersects)
	{
		if (CommonLRectPtr != nil)		/* calculate the intersection rect if requested */
		{
			CommonLRectPtr -> top    = _min (Rect1Ptr -> top,    Rect2Ptr -> top);
			CommonLRectPtr -> bottom = _max (Rect1Ptr -> bottom, Rect2Ptr -> bottom);
			CommonLRectPtr -> left   = _max (Rect1Ptr -> left,   Rect2Ptr -> left);
			CommonLRectPtr -> right  = _min (Rect1Ptr -> right,  Rect2Ptr -> right);
		}
		
		return (true);
	}
	else
		return (false);
}
/**************************************************************************************************/
void InsetLRect (LongRectPtr LRectPtr, long ReductionDX, long ReductionDY)
/* warning: assumes that top is greater than bottom for LongRect */
{
	LRectPtr -> top -= ReductionDY;
	LRectPtr -> left += ReductionDX;
	LRectPtr -> bottom += ReductionDY;
	LRectPtr -> right -= ReductionDX;
	
	return;
}
/**************************************************************************************************/
Boolean PtInLRect (LongPoint *LPointPtr, LongRectPtr LRectPtr)
{
	LongRect	NormLRect;
	
	NormLRect = *LRectPtr;
	NormalizeLRect (&NormLRect);
	
	if ((LPointPtr -> h >= NormLRect.left) && (LPointPtr -> h <= NormLRect.right) &&
		(LPointPtr -> v >= NormLRect.top ) && (LPointPtr -> v <= NormLRect.bottom))
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
void OffsetLRect (LongRectPtr LRectPtr, long DX, long DY)
{
	LRectPtr -> left += DX;
	LRectPtr -> right += DX;
	LRectPtr -> top += DY;
	LRectPtr -> bottom += DY;

	return;
}
/**************************************************************************************************/
void NormalizeLRect (LongRect *RectPtr)
/* subroutine to ensure that the top-left of the given rectangle is above and to the left of its
	bottom-right and that the rect is not empty */
{
	long temp;
	
	temp = RectPtr -> top;
	RectPtr -> top = _min (RectPtr -> top, RectPtr -> bottom);
	RectPtr -> bottom = _max (temp, RectPtr -> bottom);
	
	temp = RectPtr -> left;
	RectPtr -> left = _min (RectPtr -> left, RectPtr -> right);
	RectPtr -> right = _max (RectPtr -> right, temp);
	
/*	if (RectPtr -> top == RectPtr -> bottom) 
		++RectPtr -> bottom;
		
	if (RectPtr -> left == RectPtr -> right)
		++RectPtr -> right;*/
		
	return;
}
/*************************************************************************************************/
Boolean NudgeShiftLRect (LongRect *NudgeRectPtr, LongRect *LimitRectPtr)
/* given a rect in NudgeRect, this subroutine shifts it to fit inside limit rect.  If rect is
	too large to fit within LimitRect, it is made equal to LimitRect and function returns true.
   Warning: this subroutine assumes the rect tops to be greater than their bottoms */
{
	short	NudgeCount;
	
	NudgeCount = 0;
	
	/* check to see if top needs to be shifted */
	if (NudgeRectPtr -> top > LimitRectPtr -> top)
	{
		NudgeRectPtr -> bottom -= NudgeRectPtr -> top - LimitRectPtr -> top;
		NudgeRectPtr -> top = LimitRectPtr -> top;
		
		/* now check for overrun at the bottom */
		if (NudgeRectPtr -> bottom < LimitRectPtr -> bottom)
		{
			NudgeRectPtr -> bottom = LimitRectPtr -> bottom;
			++NudgeCount;
		}
	}
	
	/* check to see if bottom needs to be shifted */
	if (NudgeRectPtr -> bottom < LimitRectPtr -> bottom)
	{
		NudgeRectPtr -> top += LimitRectPtr -> bottom - NudgeRectPtr -> bottom;
		NudgeRectPtr -> bottom = LimitRectPtr -> bottom;
		
		/* now check for overrun at top */
		if (NudgeRectPtr -> top > LimitRectPtr -> top)
		{
			NudgeRectPtr -> top = LimitRectPtr -> top;
			++NudgeCount;
		}
	}
	
	/* check to see if left needs to be shifted */
	if (NudgeRectPtr -> left < LimitRectPtr -> left)
	{
		NudgeRectPtr -> right += LimitRectPtr -> left - NudgeRectPtr -> left;
		NudgeRectPtr -> left = LimitRectPtr -> left;
		
		/* now check for overrun at right */
		if (NudgeRectPtr -> right > LimitRectPtr -> right)
		{
			NudgeRectPtr -> right = LimitRectPtr -> right;
			++NudgeCount;
		}
	}
	
	/* check to see if right needs to be shifted */
	if (NudgeRectPtr -> right > LimitRectPtr -> right)
	{
		NudgeRectPtr -> left -= NudgeRectPtr -> right - LimitRectPtr -> right;
		NudgeRectPtr -> right = LimitRectPtr -> right;
		
		/* now check for overrun at left */
		if (NudgeRectPtr -> left < LimitRectPtr -> left)
		{
			NudgeRectPtr -> left = LimitRectPtr -> left;
			++NudgeCount;
		}
	}
	
	if (NudgeCount > 1)
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
Boolean LRectInLRect (LongRectPtr InsideLRectPtr, LongRectPtr OutsideLRectPtr)
{
	if (InsideLRectPtr -> top <= OutsideLRectPtr -> top &&
	    InsideLRectPtr -> left >= OutsideLRectPtr -> left &&
	    InsideLRectPtr -> right <= OutsideLRectPtr -> right &&
	    InsideLRectPtr -> bottom >= OutsideLRectPtr -> bottom)
	
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
/* given a rectangle in LargeRect, this subroutine retuns the largest rectangle that can be drawn
	inside LargeRect with the ratio supplied in XToYRatio.  The result is put in *LargestRectPtr */
void GetLargestLRect (LongRect *LargeRectPtr, double XToYRatio, LongRect *LargestRectPtr)
{
	long	LargeRectDX, LargeRectDY, ScaledDX, ScaledDY;
	Boolean	TopGTBottom;
	
	LargeRectDX = LargeRectPtr -> right - LargeRectPtr -> left;
	
	if (LargeRectPtr -> top > LargeRectPtr -> bottom)
	{
		LargeRectDY = LargeRectPtr -> top - LargeRectPtr -> bottom;
		TopGTBottom = true;
	}
	else
	{
		LargeRectDY = LargeRectPtr -> bottom - LargeRectPtr -> top;
		TopGTBottom = false;
	}
	
	/* use delta X to see if height will come out too big */
	ScaledDX = LargeRectDX;
	ScaledDY = LargeRectDX / XToYRatio;
	if (ScaledDY > LargeRectDY)
	{
		ScaledDY = LargeRectDY;
		ScaledDX = LargeRectDY * XToYRatio;
	}
	
	*LargestRectPtr = *LargeRectPtr;
	LargestRectPtr -> right = LargestRectPtr -> left + ScaledDX;
	
	if (TopGTBottom)
		LargestRectPtr -> bottom = LargestRectPtr -> top - ScaledDY;
	else
		LargestRectPtr -> bottom = LargestRectPtr -> top + ScaledDY;
	
	return;
}
/**************************************************************************************************/
void GetLargestRect (Rect *RectPtr, double XToYRatio, Rect *LargestRectPtr)
{
	short	CurrRectDX, CurrRectDY, ScaledDX, ScaledDY;
	
	CurrRectDX = RectPtr -> right  - RectPtr -> left;
	CurrRectDY = RectPtr -> bottom - RectPtr -> top;
	
	ScaledDX = CurrRectDX;
	ScaledDY = CurrRectDX / XToYRatio;
	
	if (ScaledDY > CurrRectDY)
	{
		ScaledDY = CurrRectDY;
		ScaledDX = CurrRectDY * XToYRatio;
	}
	
	*LargestRectPtr = *RectPtr;
	LargestRectPtr -> right = LargestRectPtr -> left + ScaledDX;
	LargestRectPtr -> bottom = LargestRectPtr -> top + ScaledDY;
	
	return;
}
/**************************************************************************************************/
void TrimLRect (LongRect *MatrixRectPtr, LongRect* LimitRectPtr)
/* this subroutine trims the MatrixRect if its boundaries exceed those of the LimitRect.
	Warning: this subroutine assumes the rect tops to be greater than their bottoms */
{
	if (MatrixRectPtr -> top > LimitRectPtr -> top)
		MatrixRectPtr -> top = LimitRectPtr -> top;
		
	if (MatrixRectPtr -> left < LimitRectPtr -> left)
		MatrixRectPtr -> left = LimitRectPtr -> left;
		
	if (MatrixRectPtr -> right > LimitRectPtr -> right)
		MatrixRectPtr -> right = LimitRectPtr -> right;
	
	if (MatrixRectPtr -> bottom < LimitRectPtr -> bottom)
		MatrixRectPtr -> bottom = LimitRectPtr -> bottom;
		
	return;
}
/**************************************************************************************************/
void TrimRect (Rect *theRectPtr, Rect* LimitRectPtr)
/* this subroutine trims the given rect if its boundaries exceed those of the LimitRect */
{
	if (theRectPtr -> top < LimitRectPtr -> top)
		theRectPtr -> top = LimitRectPtr -> top;
		
	if (theRectPtr -> left < LimitRectPtr -> left)
		theRectPtr -> left = LimitRectPtr -> left;
		
	if (theRectPtr -> right > LimitRectPtr -> right)
		theRectPtr -> right = LimitRectPtr -> right;
	
	if (theRectPtr -> bottom > LimitRectPtr -> bottom)
		theRectPtr -> bottom = LimitRectPtr -> bottom;
		
	return;
}
/**************************************************************************************************/
void CenterRectInRect (Rect *smallRect, Rect *bigRect)
{
	long	smallRectWidth, smallRectHeight, bigRectWidth, bigRectHeight;
	
	smallRectWidth  = RectWidth  (*smallRect);
	smallRectHeight = RectHeight (*smallRect);
	bigRectWidth	= RectWidth  (*bigRect);
	bigRectHeight   = RectHeight (*bigRect);

	smallRect -> left   = bigRect   -> left + (bigRectWidth  - smallRectWidth) / 2;
	smallRect -> top    = bigRect   -> top  + (bigRectHeight - smallRectHeight) / 2;
	smallRect -> right  = smallRect -> left + smallRectWidth;
	smallRect -> bottom = smallRect -> top  + smallRectHeight;

	return;
}
/**************************************************************************************************/
void CenterLRectinLRect (LongRect *smallLRect, LongRect *bigLRect)
/* given a smallRect that fits in Bigrect, this subroutine changes the coordinates of SmallRect to
	be centered inside big rect.  Warning: this subroutine assumes the rect tops to be larger than
	bottoms */
{
	long	smallRectWidth, smallRectHeight, bigRectWidth, bigRectHeight;

	smallRectWidth  = LRectWidth  (*smallLRect);
	smallRectHeight = LRectHeight (*smallLRect);
	bigRectWidth	= LRectWidth  (*bigLRect);
	bigRectHeight   = LRectHeight (*bigLRect);

	smallLRect -> left   = bigLRect   -> left    + (bigRectWidth  - smallRectWidth) / 2;
	smallLRect -> bottom = bigLRect   -> bottom  + (bigRectHeight - smallRectHeight) / 2;
	smallLRect -> right  = smallLRect -> left    + smallRectWidth;
	smallLRect -> top    = smallLRect -> bottom  + smallRectHeight;

	return;
}
/**************************************************************************************************/
void GetScaledRect (Rect *CurrRectPtr, double XToYRatio, Rect *LimitRectPtr, long XMargin,
					long YMargin)
/* Scaled rectangle replaces CurrRect */
/* Note: current sizes do not reflect the margins to be configured by subroutine */
/* Note: the top and left coordinates of the LimitRect are used only to calculate the maximum Delta
		  X and Y, and have to absolute relevence */
{
	short	DeltaX, DeltaY, MaxDeltaX, MaxDeltaY, ScaledDX, ScaledDY;
	Rect	ScaledRect;

	DeltaX = CurrRectPtr -> right  - CurrRectPtr -> left;
	DeltaY = CurrRectPtr -> bottom - CurrRectPtr -> top;

	ScaledDX = DeltaX * XToYRatio;
	ScaledDY = DeltaY / XToYRatio;
	
	MaxDeltaX = LimitRectPtr -> right  - LimitRectPtr -> left - XMargin;
	MaxDeltaY = LimitRectPtr -> bottom - LimitRectPtr -> top  - YMargin;
	
/*	printf ("   DeltaX = %hd    DeltaY = %hd\n", DeltaX, DeltaY);
	printf ("MaxDeltaX = %hd MaxDeltaY = %hd\n", MaxDeltaX, MaxDeltaY);*/

	/* if DeltaX is greater than DeltaY use DeltaX to calculate scaled DeltaY */
	if (myabs(ScaledDX) > myabs(ScaledDY))
	{
		ScaledRect = *CurrRectPtr;
		ScaledRect.bottom = CurrRectPtr -> top + DeltaX / XToYRatio;
	}
	else
	{
		ScaledRect = *CurrRectPtr;
		ScaledRect.right = CurrRectPtr -> left + DeltaY * XToYRatio;
	}
	
/*	printf ("ScaledRect before limit check = ");
	printfRect (&ScaledRect);*/

	/* now check to see if we are going beyond allowd limits set by LimitRect */
	if ((ScaledRect.right - ScaledRect.left > MaxDeltaX) || (ScaledRect.bottom - ScaledRect.top > MaxDeltaY))
	{
		ScaledRect.right = ScaledRect.left + MaxDeltaX;
		ScaledRect.bottom = ScaledRect.top + MaxDeltaX / XToYRatio;
		if (ScaledRect.bottom - ScaledRect.top > MaxDeltaY)
		{
			ScaledRect.bottom = ScaledRect.top + MaxDeltaY;
			ScaledRect.right = ScaledRect.left + MaxDeltaY * XToYRatio;
		}
	}

/*	printfRect (&ScaledRect);*/
	*CurrRectPtr = ScaledRect;

	return;
}
/**************************************************************************************************/
void GetScaledLRect (LongRect *CurrRectPtr, double XToYRatio, LongRect *LimitRectPtr,
				long XMargin, long YMargin)
/* this subroutine is the same as GetScaledRect except for LongRects */
/* Warning: this subroutine assumes a top greater than bottom */
/* Scaled rectangle replaces CurrRect */
/* Note: current sizes do not reflect the margins to be configured by subroutine */
/* Note: the top and left coordinates of the LimitRect are used only to calculate the maximum Delta
		  X and Y, and have to absolute relevence */
{
	long		DeltaX, DeltaY, MaxDeltaX, MaxDeltaY, ScaledDX, ScaledDY;
	LongRect	ScaledRect;
	
	DeltaX = CurrRectPtr -> right  - CurrRectPtr -> left;
	DeltaY = CurrRectPtr -> top - CurrRectPtr -> bottom;
	
	ScaledDX = DeltaX * XToYRatio;
	ScaledDY = DeltaY / XToYRatio;
	
	MaxDeltaX = LimitRectPtr -> right  - LimitRectPtr -> left - XMargin;
	MaxDeltaY = LimitRectPtr -> top - LimitRectPtr -> bottom  - YMargin;
	
/*	printf ("   DeltaX = %hd    DeltaY = %hd\n", DeltaX, DeltaY);
	printf ("MaxDeltaX = %hd MaxDeltaY = %hd\n", MaxDeltaX, MaxDeltaY);*/

	/* if DeltaX is greater than DeltaY use DeltaX to calculate scaled DeltaY */
//	if (sabs (ScaledDX) > sabs (ScaledDY))
	if (myabs(ScaledDX) > myabs(ScaledDY))
	{
		ScaledRect = *CurrRectPtr;
		ScaledRect.top = CurrRectPtr -> bottom + DeltaX / XToYRatio;
	}
	else
	{
		ScaledRect = *CurrRectPtr;
		ScaledRect.right = CurrRectPtr -> left + DeltaY * XToYRatio;
	}
	
/*	printf ("ScaledRect before limit check = ");
	printfRect (&ScaledRect);*/

	/* now check to see if we are going beyond allowd limits set by LimitRect */
	if ((ScaledRect.right - ScaledRect.left > MaxDeltaX) || (ScaledRect.top - ScaledRect.bottom > MaxDeltaY))
	{
		ScaledRect.right = ScaledRect.left + MaxDeltaX;
		ScaledRect.top = ScaledRect.bottom + MaxDeltaX / XToYRatio;
		if (ScaledRect.top - ScaledRect.bottom > MaxDeltaY)
		{
			ScaledRect.top = ScaledRect.bottom + MaxDeltaY;
			ScaledRect.right = ScaledRect.left + MaxDeltaY * XToYRatio;
		}
	}

/*	printfRect (&ScaledRect);*/
	*CurrRectPtr = ScaledRect;

	return;
}
/**************************************************************************************************/
void UnionLRect (LongRect *LRect1Ptr, LongRect *LRect2Ptr, LongRect *UnionLRectPtr)
{
	UnionLRectPtr -> top    = _max (LRect1Ptr -> top,    LRect2Ptr -> top);
	UnionLRectPtr -> right  = _max (LRect1Ptr -> right,  LRect2Ptr -> right);
	UnionLRectPtr -> bottom = _min (LRect1Ptr -> bottom, LRect2Ptr -> bottom);
	UnionLRectPtr -> left   = _min (LRect1Ptr -> left,   LRect2Ptr -> left);
	
	return;
}
/**************************************************************************************************/
Boolean EmptyLRect (LongRect *TestLRectPtr)
{
	Boolean emptyRect = false;
	
	if (TestLRectPtr -> top <= TestLRectPtr -> bottom)
		emptyRect = true;
	
	if (TestLRectPtr -> right <= TestLRectPtr -> left)
		emptyRect = true;
	
	return (emptyRect);
}
/**************************************************************************************************/
void GetLRectCenter (LongRect *thisLRectPtr, LongPoint *CenterLPtPtr)
{
	CenterLPtPtr -> h = thisLRectPtr -> left   + ((thisLRectPtr -> right - thisLRectPtr -> left)   / 2);
	CenterLPtPtr -> v = thisLRectPtr -> bottom + ((thisLRectPtr -> top   - thisLRectPtr -> bottom) / 2);

	return;
}
/**************************************************************************************************/
void GetRectCenter (Rect *thisRectPtr, Point *CenterPtPtr)
{
	CenterPtPtr -> h = thisRectPtr -> left + ((thisRectPtr -> right  - thisRectPtr -> left) / 2);
	CenterPtPtr -> v = thisRectPtr -> top  + ((thisRectPtr -> bottom - thisRectPtr -> top)  / 2);

	return;
}
/**************************************************************************************************/
Boolean IntersectRect (Rect *Rect1Ptr, Rect *Rect2Ptr, Rect *CommonRectPtr)
/* given pointers to two rectangles, this subroutine returns a flag indicating whether there is
	any overlap between the two rects */
{
	Boolean 	XIntersects, YIntersects;
		
	XIntersects = false;
	YIntersects = false;
	
	/* check the x-coordinates */
	if ((Rect2Ptr -> left >= Rect1Ptr -> left) && (Rect2Ptr -> left <= Rect1Ptr -> right))
		XIntersects = true;
	if ((Rect2Ptr -> right >= Rect1Ptr -> left) && (Rect2Ptr -> right <= Rect1Ptr -> right))
		XIntersects = true;		
	if ((Rect1Ptr -> left >= Rect2Ptr -> left) && (Rect1Ptr -> left <= Rect2Ptr -> right))
		XIntersects = true;
	if ((Rect1Ptr -> right >= Rect2Ptr -> left) && (Rect1Ptr -> right <= Rect2Ptr -> right))
		XIntersects = true;
	
	/* check the y-coordinates */
	if ((Rect2Ptr -> top >= Rect1Ptr -> top) && (Rect2Ptr -> top <= Rect1Ptr -> bottom))
		YIntersects = true;	
	if ((Rect2Ptr -> bottom >= Rect1Ptr -> top) && (Rect2Ptr -> bottom <= Rect1Ptr -> bottom))
		YIntersects = true;
	if ((Rect1Ptr -> top >= Rect2Ptr -> top) && (Rect1Ptr -> top <= Rect2Ptr -> bottom))
		YIntersects = true;	
	if ((Rect1Ptr -> bottom >= Rect2Ptr -> top) && (Rect1Ptr -> bottom <= Rect2Ptr -> bottom))
		YIntersects = true;
	
	if (XIntersects && YIntersects)
	{
		if (CommonRectPtr != nil)
		{
			/* calculate intersection rect */
			CommonRectPtr -> top    = _max (Rect1Ptr -> top,    Rect2Ptr -> top);
			CommonRectPtr -> bottom = _min (Rect1Ptr -> bottom, Rect2Ptr -> bottom);
			CommonRectPtr -> left   = _max (Rect1Ptr -> left,   Rect2Ptr -> left);
			CommonRectPtr -> right  = _min (Rect1Ptr -> right,  Rect2Ptr -> right);
		}
		
		return (true);
	}
	else
	{
		// JLM 12/23/98 /////////////{
		if (CommonRectPtr != nil)
		{	// set rect to 0,0,0,0 in this case like the MAC SectRect does	
			CommonRectPtr -> top    = 0;
			CommonRectPtr -> bottom = 0;
			CommonRectPtr -> left   = 0;
			CommonRectPtr -> right  = 0;
		}
		/////////////////////////////}
	
		return (false);
	}
}
/**************************************************************************************************/
void NormalizeRect (Rect *RectPtr)
/* subroutine to ensure that the top-left of the given rectangle is above and to the left of its
	bottom-right and that the rect is not empty */
{
	short temp;
	
	temp = RectPtr -> top;
	RectPtr -> top = _min (RectPtr -> top, RectPtr -> bottom);
	RectPtr -> bottom = _max (temp, RectPtr -> bottom);
	
	temp = RectPtr -> left;
	RectPtr -> left = _min (RectPtr -> left, RectPtr -> right);
	RectPtr -> right = _max (RectPtr -> right, temp);
	
/*	if (RectPtr -> top == RectPtr -> bottom) 
		++RectPtr -> bottom;
		
	if (RectPtr -> left == RectPtr -> right)
		++RectPtr -> right;*/
		
	return;
}
/**************************************************************************************************/
Boolean RectInRect (Rect *InsideRectPtr, Rect *OutsideRectPtr)
{
	if (InsideRectPtr -> top > OutsideRectPtr -> top &&
	    InsideRectPtr -> left > OutsideRectPtr -> left &&
	    InsideRectPtr -> right < OutsideRectPtr -> right &&
	    InsideRectPtr -> bottom < OutsideRectPtr -> bottom)
	
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
Boolean RectInRect2 (Rect *InsideRectPtr, Rect *OutsideRectPtr)
{	// allow inside rect to touch edge of outside rect
	if (InsideRectPtr -> top >= OutsideRectPtr -> top &&
	    InsideRectPtr -> left >= OutsideRectPtr -> left &&
	    InsideRectPtr -> right <= OutsideRectPtr -> right &&
	    InsideRectPtr -> bottom <= OutsideRectPtr -> bottom)
	
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
Boolean MyPtInRect (Point p, Rect *r)
#ifdef MAC
{
	return PtInRect(p, r);
}
#else
{
	return p.h >= r->left && p.h < r->right && p.v >= r->top && p.v < r->bottom;
}
#endif
/**************************************************************************************************/
/*Boolean MyPtInRect (Point checkPt, Rect *checkRect)
{
	if ((checkPt.h >= checkRect -> left) && (checkPt.h <= checkRect -> right) &&
		(checkPt.v >= checkRect -> top ) && (checkPt.v <= checkRect -> bottom))
		return (true);
	else
		return (false);
}*/
/**************************************************************************************************/
void AlignRect (Rect *rectToAlign, Rect *frameRect, short justCode)
/* this subroutine modifies the coordinates of rect-To-Align to justify it horizontally around the
	frame rect. The just-Code parameter can contain teJustCenter, teJustLeft, or teJustRight. */
{
	short	DeltaX, RectWidth;
	
	if (justCode == teJustCenter)
	{
		RectWidth = rectToAlign -> right - rectToAlign -> left;
		DeltaX = rectToAlign -> left  - frameRect -> left +
				 frameRect -> right - rectToAlign -> right;
		rectToAlign -> left  = frameRect -> left + (DeltaX / 2);
		rectToAlign -> right = rectToAlign -> left + RectWidth;
	}
	else if (justCode == teJustLeft)
	{
		DeltaX = rectToAlign -> left - frameRect -> left;
		rectToAlign -> left -= DeltaX;
		rectToAlign -> right -= DeltaX;
	}
	else if (justCode == teJustRight)
	{
		DeltaX = frameRect -> right - rectToAlign -> right;
		rectToAlign -> left += DeltaX;
		rectToAlign -> right += DeltaX;
	}
//	else
//		Debug ("justCode = %hd\n", justCode);

	return;
}
/**************************************************************************************************/
void InsetExRect (ExRect *exRectPtr, double reductionDX, double reductionDY)
// Note: a normal rect is expected
{
	exRectPtr -> left   += reductionDX;
	exRectPtr -> top    += reductionDY;
	exRectPtr -> right  -= reductionDX;
	exRectPtr -> bottom -= reductionDY;
	
	return;
}
/**************************************************************************************************/
/*
void PinToRect (Rect theRect, Point *thePoint)
{
	short	LargestIndex, SortArray [4], i, SmallestDiff;
	
	// init rect array to be sorted by index number for the smallest distance from given point
	SortArray [0] = PointPtr -> h - RectPtr -> left;
	SortArray [1] = PointPtr -> h - RectPtr -> right;
	SortArray [2] = PointPtr -> v - RectPtr -> top;
	SortArray [3] = PointPtr -> v - RectPtr -> bottom;
	
	// sort array for largest absolute difference from point to one of rectangle sides
	for (i = 0, SmallestDiff = MaxShort; i <= 3; i++)
	{
		if (abs (SortArray [i]) < SmallestDiff) // flag the element of rect closest to the point
		{
			LargestIndex = i;					// save the index number
			SmallestDiff = abs (SortArray [i]);	// and also save the difference itself
		}
	}
	if (SortArray [LargestIndex] == 0)	// if point is already touching one of the sides
		return;
	else if ((LargestIndex == 0) || (LargestIndex == 1)) // left or right side of rect is closest
		PinnedPtPtr -> h = PointPtr -> h - SortArray [LargestIndex];
	else												 // top or bottom side of rect is closest
		PinnedPtPtr -> v = PointPtr -> v - SortArray [LargestIndex];
		
	// added 1/24/91
	if (PointPtr -> h > RectPtr -> right)
		PinnedPtPtr -> h = RectPtr -> right;

 	if (PointPtr -> v > RectPtr -> bottom)
		PinnedPtPtr -> v = RectPtr -> bottom;

	return;
}
*/
/**************************************************************************************************/
void EnsurePtInLRect (LongPoint *thePoint, LongRect *limitRect)
{
	if (thePoint -> h < limitRect -> left)
		thePoint -> h = limitRect -> left;
	
	if (thePoint -> h > limitRect -> right)
		thePoint -> h = limitRect -> right;
	
	if (thePoint -> v < limitRect -> bottom)
		thePoint -> v = limitRect -> bottom;

	if (thePoint -> v > limitRect -> top)
		thePoint -> v = limitRect -> top;

	return;
}
/**************************************************************************************************/
void printfRect (Rect *RectPtr)
/* debugging utility / tool to print each field of supplied rectangle */
{
	printf ("R.lft = %hd, R.tp = %hd, R.rgt = %hd, R.btm = %hd\n", 
			RectPtr -> left, RectPtr -> top, RectPtr -> right, RectPtr -> bottom);
}
/**************************************************************************************************/
void printfLRect (LongRectPtr RectPtr)
/* debugging utility / tool to print each field of supplied rectangle */
{
	printf ("left  = %d, top    = %d\n right = %d, bottom = %d\n", RectPtr -> left, RectPtr -> top,
															   	  RectPtr -> right, RectPtr -> bottom);
}
/////////////////////////////////////////////////

short ForceOntoQuickDrawPlane(long n)
{
	return _max(-32767, _min(32767,n));
}
/////////////////////////////////////////////////

Boolean IntersectToQuickDrawPlane(LongRect r,Rect* qdr)
{
	Boolean onQuickDrawPlane = true;
	short maxPix= 32767, minPix = -32767;
	
	if(r.top < minPix) {r.top = minPix; onQuickDrawPlane = false; }
	if(r.top > maxPix) {r.top = maxPix; onQuickDrawPlane = false; }

	if(r.left < minPix) {r.left = minPix; onQuickDrawPlane = false; }
	if(r.left > maxPix) {r.left = maxPix; onQuickDrawPlane = false; }

	if(r.bottom < minPix) {r.bottom = minPix; onQuickDrawPlane = false; }
	if(r.bottom > maxPix) {r.bottom = maxPix; onQuickDrawPlane = false; }

	if(r.right < minPix) {r.right = minPix; onQuickDrawPlane = false; }
	if(r.right > maxPix) {r.right = maxPix; onQuickDrawPlane = false; }

	qdr->top = (short)r.top;
	qdr->left = (short)r.left;
	qdr->bottom = (short)r.bottom;
	qdr->right = (short)r.right;
	
	return onQuickDrawPlane;
}

/////////////////////////////////////////////////
