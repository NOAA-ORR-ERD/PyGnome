
#ifndef __CMAP__
#define __CMAP__

//#include	"CMyApp.h"
#include	"CMYLIST.H"  //JLM

#include	"MapUtils.h"

#define				kDefMaxLats				2		// maximum number of latitudes to be drawn on map
#define				kDefMaxLongs			3		// maximum number of longitudes to be drawn on map
#define				kMaxLLSlots				40		// maximum number of lat/long increment values
#define				GridColorInd			79		// index number for lat/long grid color changed from 248

#define				kZoomCursID				160		// id for zoom marquee cursor
#define				kZoomInSound	   		139		// zoom-in sound effect
#define				kZoomOutSound   		140		// zoom-out sound effect

#define				kZoomToller				2500	// tollerence for zooming into map (LRect matrix difference)

#define SavedMapData \
					Boolean					bVFirstField;		/* this field must be the first field */\
					Boolean					bLatLongVisible;	/* true if lat-long lines are visible */\
					Boolean					bLinesInBack;		/* true if lat-long lines should be drawn in the back */\
					short					prefLats;\
					short					prefLongs;\
					LongRect				matrixViewRect;		/* window's content rect in matrix coords */\
					Boolean					bZoomedOut;			/* true if map has been zoomed out completely */\
					short					mapProjCode;		/* true if map has been zoomed out completely */\
					LongRect				mapBoundsRect;		/* actual matrix map bounds */\
					Boolean					bColorMap;			/* true if map is to be displayed in color */

// TODO: when compiling 64-bit executables, the OS X version of GCC ignores requests for mac68k alignment.
//#ifndef IBM
//#pragma options align=mac68k
//#endif
// Why do we align this structure, is it for compatibility with a binary input file?
// - if it is not important to align this structure, the pragma should go away.
// - if it is important (which we will assume for now) then we change it to #pragma pack(2),
//   which is equivalent to mac68k.
#ifndef IBM
	#pragma pack(2)
#endif
typedef struct SavedMDataRec									// struct used to determine saved-data size
			{
					SavedMapData
			} SavedMDataRec;
#ifndef IBM
	#pragma options align=reset
#endif

typedef struct ViewStatusRec
				{
					SavedMapData									// data saved in save files
					Boolean					bModified;				// dirty flag
					LongRect				mapMatrixRect;			// scaled version of current map matrix rect
					LongRect				mapWindowRect;			// scaled version of matrix view rect (screen coordinates)
					Rect					mapWDrawRect;			// window's port rect minus grow area
					ScaleRec				matrixToScrScaleRec;
					ScaleRec				scrToMatrixScaleRec;
					double					mapMatrixRatio;
					double					mapViewRatio;
				} ViewStatusRec, *ViewStatusRecPtr;

class CMap
			{
				protected:
					ViewStatusRec			MapViewStatus;
					ViewStatusRecPtr		ViewStatusPtr;
					WindowPtr				wPtr;

				public:
					OSErr					IMap (WindowPtr ViewWPtr, Boolean LLVisFlag, Boolean LinesInBackFlag,
												  long MapProjCode, Boolean bInvalidate);
												  
					//					   ~CMap ();
					virtual				   ~CMap ()	{ Dispose(); }
					
					virtual void			Dispose() {return;}
										   
					virtual void			SetWDrawRect ();
					virtual void			ViewAllMap (Boolean bInvalidate);

					void					GetMapDrawRect (Rect *WDrawRectPtr);
					void					InvalRect (Rect *r);
					void					InvalWDrawRect ();
					void					ClipWDrawRect ();
					void					GetMapLBounds (LongRect *MapBoundsLRectPtr);
					void					SetMapLBounds (LongRect *MapBoundsLRectPtr);
					void					GetMapViewLRect (LongRect *ViewLRectPtr);
					void					GetMapMatrixRect (LongRect *MapMatrixRectPtr);

					OSErr					ZoomMapIn (Point LocalPt);
					OSErr					ZoomMapOut (Point LocalPt);
					void					SetZoomedOut (Boolean bZoomedOut);
					Boolean					IsZoomedOut ();
					void					CenterToLPoint (LongPoint centerLPoint, Boolean bInvalidate);
					OSErr					DrawMapInLRect (LongRect *MapLRectPtr, Boolean bInvalidate);
					
					Boolean					GetLLBehind ();
					void 					SetLLBehind (Boolean Behind);
					void					SetLatLongVis (Boolean Visible);
					void					GetPrefLines (long *prefLats, long *prefLongs);
					void					SetPrefLines (long newPrefLats, long newPrefLongs);
					void 					DrawLatLongs (Boolean DrawLines, Boolean DrawLabels, long Mode);

					WindowPtr				GetCMapWindow () {return wPtr;}
					
					void					LMoveTo (long MatrixX, long MatrixY);
					void					LLineTo (long MatrixX, long MatrixY);

					void					CalcMatrixViewRect ();
					void					GetMatrixPoint (Point *ScrPointPtr, LongPointPtr LPointPtr);
					void					GetMatrixLPoint (LongPoint *ScrLPointPtr, LongPointPtr LPointPtr);
					OSErr 					GetMatrixLRect (Rect *ScreenRectPtr, LongRectPtr MatrixRectPtr);
					void					GetScrPoint (LongPointPtr LPointPtr, Point *ScrPointPtr);
					void					GetScrRect (LongRectPtr LRectPtr, Rect *ScrRectPtr);
					void					GetScrLRect (LongRectPtr LRectPtr, LongRect *ScrLRectPtr);
					void					GetScrLPoint (LongPointPtr MatrixLPointPtr, LongPoint *ScrLPointPtr);
					long					MatrixToPixels (long MatrixDist);

					void					TrackMouse ();
					long					GetMapScale ();
					void					ShowMapScale ();
					void 					ShowCursPos ();
					void 					GetMapViewWidth (long UnitCode, double *ViewWidthPtr);
					void					DoScaleTool (Point LocalPt);
					void					ShowPerimeter (CMyList *DistPtsList, LongPoint TrackingPt);
					OSErr					DoGrowRect (Rect *StartRect, Point LocalClickPt, Rect *WindowRectPtr);
					void					ShowArea (CMyList *DistPtsList);
					OSErr					DoHandTool (Point ClickPt);
					OSErr					GetDragLine (Point ClickPt, Point *TopLeftPt, Point *BottomRightPt);
					OSErr					GetDragRect (Point ClickPt, Boolean bChangeCurs, long DragCursID, Rect* ScrSelRectPtr);

					Boolean					DoMenu (long menuResult, char *ChoiceStr, Boolean IsFrontMap);
					void					SetMenuStates (Boolean IsFrontMap);
					OSErr					SaveMapData (short FRefNum);
					OSErr					ReadMapData (short FRefNum, Boolean bInvalidate);

					long					GetMapProjCode ();
					void					SetMapProjCode (long ProjCode);

					Boolean					IsColorMap ();
					void					SetMapColorFlag (Boolean bColorFlag, Boolean bInvalidate);

					Boolean					IsMapModified ();
					void					SetMapModified (Boolean bModFlag);

				protected:
					OSErr					SetMapScales ();
					void					CalcMapRatios ();
					void					CalcMapWindowRect ();
					void					InitViewMercator (LongRect *ViewLRectPtr);
					void					DrawLat (long MatrixLat, Boolean ColorFlag);
					void					DrawLong (long MatrixLong, Boolean ColorFlag);
					void					ShowSegLength (LongPoint LPt1, LongPoint LPt2);
					void					GetMaxDeltaLRect (LongRectPtr MaxDeltaLRectPtr);
					void					LabelLat (long *MatrixLatPtr, Boolean DrawLabelFlag);
					void					LabelLong (long *MatrixLongPtr, Boolean DrawLabelFlag);
					OSErr					SelectZoomRect (Point LocalPt, Rect* ScrSelRectPtr);
			};

#endif
