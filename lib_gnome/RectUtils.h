
#ifndef __MYRECTUTILS__
#define __MYRECTUTILS__

#include "Basics.h"
#include "TypeDefs.h"

typedef struct ExPoint
					{
    						double	 				h;
    						double	 				v;
					} ExPoint, *ExPointPtr;

typedef struct LongPoint
					{
    						long 					h;
    						long 					v;
					} LongPoint, *LongPointPtr, **LongPointHdl;

typedef struct LongRect
					{
    						long 					top;
    						long 					left;
    						long 					bottom;
    						long 					right;
					} LongRect, *LongRectPtr;

typedef struct ExRect
					{
    						double	 				top;
    						double	 				left;
    						double	 				bottom;
    						double	 				right;
					} ExRect, *ExRectPtr;

/**************************************************************************************************/
/*										      Macros   						    				  */
#define RectWidth(r) ((r).right - (r).left)
#define RectHeight(r) ((r).bottom - (r).top)
#define LRectWidth(r) ((r).right - (r).left)
#define LRectHeight(r) ((r).top - (r).bottom)
/**************************************************************************************************/
/*										Operator Overloading				    				  */
Boolean operator != (Point p1, Point p2);
Boolean operator == (Point p1, Point p2);
Boolean operator != (Rect r1, Rect r2);
Boolean operator == (Rect r1, Rect r2);
Boolean operator != (LongRect r1, LongRect r2);
Boolean operator == (LongRect r1, LongRect r2);
Boolean operator == (LongPoint p1, LongPoint p2);
Boolean operator != (LongPoint p1, LongPoint p2);
/**************************************************************************************************/

void printfRect (Rect *RectPtr);
void NormalizeRect (Rect *RectPtr);
void printfLRect (LongRectPtr RectPtr);
Boolean EmptyLRect (LongRect *TestLRectPtr);
Boolean MyPtInRect (Point checkPt, Rect *checkRect);
Boolean RectInRect (Rect *InsideRectPtr, Rect *OutsideRectPtr);
Boolean RectInRect2 (Rect *InsideRectPtr, Rect *OutsideRectPtr);
void GetSmallestRect (Rect *SmallRectPtr, double XToYRatio, Rect *SmallestRectPtr);
void GetSmallestLRect (LongRect *SmallRectPtr, double XToYRatio, LongRect *SmallestRectPtr);
void SetLRect (LongRectPtr LRectPtr, long left, long top, long right, long bottom);
void GetMaxLRect (LongRect *Rect1Ptr, LongRect *Rect2Ptr, LongRect *MaxRectPtr);
void GetMaxRect (Rect *Rect1Ptr, Rect *Rect2Ptr, Rect *MaxRectPtr);
void GetRectCenter (Rect *thisRectPtr, Point *CenterPtPtr);
void GetLRectCenter (LongRect *thisLRectPtr, LongPoint *CenterLPtPtr);
void NudgeRect (Rect *RectPtr, Rect *LimitRectPtr);
Boolean NudgeShiftLRect (LongRect *NudgeRectPtr, LongRect *LimitRectPtr);
void EnsurePtInLRect (LongPoint *thePoint, LongRect *limitRect);
Boolean IntersectRect (Rect *Rect1Ptr, Rect *Rect2Ptr, Rect *CommonRectPtr);
Boolean IntersectLRect (LongRectPtr Rect1Ptr, LongRectPtr Rect2Ptr, LongRectPtr CommonLRectPtr);
void InsetLRect (LongRectPtr LRectPtr, long ReductionDX, long ReductionDY);
Boolean PtInLRect (LongPoint *LPointPtr, LongRectPtr LRectPtr);
void OffsetLRect (LongRectPtr LRectPtr, long DX, long DY);
void NormalizeLRect (LongRect *RectPtr);
Boolean LRectInLRect (LongRectPtr InsideLRectPtr, LongRectPtr OutsideLRectPtr);
void GetLargestLRect (LongRect *LargeRectPtr, double XToYRatio, LongRect *LargestRectPtr);
void GetLargestRect (Rect *RectPtr, double XToYRatio, Rect *LargestRectPtr);
void TrimRect (Rect *theRectPtr, Rect* LimitRectPtr);
void TrimLRect (LongRect *MatrixRectPtr, LongRect* LimitRectPtr);
void CenterRect (Rect *RectPtr);
void CenterRectInRect (Rect *SmallRectPtr, Rect *BigRectPtr);
void CenterLRectinLRect (LongRect *SmallLRectPtr, LongRect *BigLRectPtr);
void AlignRect (Rect *rectToAlign, Rect *frameRect, long justCode);
void SetExRect (ExRect *ExRectPtr, double left, double top, double right, double bottom);
void InsetExRect (ExRect *exRectPtr, double reductionDX, double reductionDY);
void UnionLRect (LongRect *LRect1Ptr, LongRect *LRect2Ptr, LongRect *UnionLRectPtr);
void GetScaledRect (Rect *CurrRectPtr, double XToYRatio, Rect *LimitRectPtr,
				long XMargin, long YMargin);
void GetScaledLRect (LongRect *CurrRectPtr, double XToYRatio, LongRect *LimitRectPtr,
				long XMargin, long YMargin);


short ForceOntoQuickDrawPlane(long n);
Boolean IntersectToQuickDrawPlane(LongRect r,Rect* qdr);
OSErr	GetLScaleAndOffsets (LongRectPtr SourceRectPtr, LongRectPtr DestRectPtr, ScaleRecPtr ScaleInfoPtr);

void MySetRect(RECTPTR r, short left, short top, short right, short bottom);
Point RectCenter(Rect r);

#endif

