
#ifndef __OBJECTUTILS__
#define __OBJECTUTILS__


#include	"CMap2.h"
#include	"GEOMETRY.H"
#include	"RectUtils.h"


///////////////////////////////
// from MYTEXTUTILS.h -- JLM
#define				kFontNameLen		32				/* length of font name used for labels */
#define				kMaxFSize			300

#define				kReduceSizeCode		-2
#define				kIncreaseSizeCode	-1

typedef struct TextSpecRec
			{
					char				textFont [kFontNameLen];
					long				textSize;
					long				textStyle;
					long				textJust;
					long				textColor;
			} TextSpecRec, *TextSpecRecPtr;
//////////////////////////

#define				kTextType		'TEXT'		/* type string for text files								  */

#define				kSymbolCursID		555				// cursor used to plot symbols

#define				kSelectedMask		14				// zero-based bit offset
#define				kMarkedMask			13
#define				kVisibleMask		12
#define				kLabelVisMask		11
#define				kSymbolVisMask		10
#define				kGroupedMask		 9
#define				kHotLinkedMask		 8
#define				kUndoMask			 7

#define				kDefObjFlags		0x04			// visible mask bit set by default

#define				kObjectNameLen		32
#define				kLayerNameLen		32

#define				kPolyType			"POLY"	//AH 03/20/2012
#define				kSymbolType			'SYMB'
#define				kPolyLineType		'POLN'
#define				kLEType				'OSLE'
#define				kRectType			'RECT'
#define				kLineType			'LINE'
#define				kBeachedLEType		'BCLE'

#define				kNoFillCode			0				// paint codes passed into draw-no-sect poly
#define				kPatFillCode		1
#define				kPaintFillCode		2

#define				kNoPart				-1
#define				kInObjectFrame		0
#define				kTopLeftCorner		1
#define				kTopRightCorner		2
#define				kBottomRightCorner	3
#define				kBottomLeftCorner	4

#define				kNoFrameCode		0				// paint codes passed into draw-no-sect poly
#define				kPatFrameCode		1
#define				kPaintFrameCode		2

#define				kUndoDragCode		1				// undo codes for supported operations
#define				kUndoGrowCode		2
#define				kUndoDeleteCode		3

typedef struct DrawSpecRec
			{
					long				mode;
					Boolean				bColor;
					Boolean				bErase;
					Boolean				bClosed;
					
					long				frameCode;	// [kNoFrameCode, kPatFrameCode, kPaintFrameCode]						
					long				fillCode;	// [kNoFillCode, kPatFillCode, kPaintFillCode]
					
					long				offsetDx;
					long				offsetDy;
					long				backColorInd;
					long				foreColorInd;
					Pattern				backPattern;
					Pattern				forePattern;

					Boolean				bDrawOLabels;
			} DrawSpecRec, *DrawSpecRecPtr;

typedef struct ObjectIDRec
			{
					long				hi;
					long				lo;
			} ObjectIDRec, *ObjectIDRecPtr;

Boolean operator == (ObjectIDRec Obj1ID, ObjectIDRec Obj2ID);

#define ObjectInfoData \
					OSType				objectType;\
					ObjectIDRec			objectID;\
					LongPoint			objectLPoint;					/* additional point data */\
					LongPoint			objectLabelLPoint;				/* center point of object label */\
					LongRect			objectLRect;\
					long				objectSymbol;\
					long				objectFlags;\
					long				objectColorInd;					/* also the frame color */\
					long				objectBColorInd;				/* also the fill  color */\
					Pattern				objectFillPat;					/* white by default */\
					Rect				objectLabelRect;				/* btm-rgt used only if Zoom-Label-Mask is set */\
					long				objectLabelJust;				/* label justification code-rel. to label LPoint */\
					long				objectHotLinkData;\
					char				objectLabel [kObjectNameLen];\
					long				objectCustData;					/* custom object data (optional) */\
					Handle				objectDataHdl;					/* additional object data (optional) */\
					long				objectESICode;\
					long				futureLong1;\
					long				futureLong2;\
					long				futureLong3;\
					long				futureLong4;\
	 			    ObjectInfoRec	  **groupToObjectHdl;

typedef struct ObjectInfoRec
			{
					ObjectInfoData
			} ObjectInfoRec, **ObjectRecHdl;

#define LayerInfoData \
					Boolean				bLayerFirstField;			/* must be first field */\
					Boolean				bLayerVisible;\
					long				layerMinVisScale;\
					long				layerMaxVisScale;\
\
					char				layerName [kLayerNameLen];\
					LongRect			layerScopeLRect;\
\
					Boolean				bLayerOLabelVisible;\
					long				oLabelMinVisScale;\
					long				oLabelMaxVisScale;\
\
					long				lastTextFont;\
					long				lastTextStyle;\
					long				lastTextSize;\
					long				lastTextJust;\
\
					char				layerLabelFont	[kFontNameLen];\
					long				layerLabelFontSize;\
					long				layerLabelFontStyle;\
					long				layerLabelColorInd;

typedef struct LayerInfoRec
			{
					LayerInfoData
			} LayerInfoRec, **LayerInfoHdl;

typedef struct LinkDataRec
			{
					ObjectRecHdl		objectHandle;
					CMyList				*objectList;		/* nil if object index if valid */
			} LinkDataRec;

typedef struct PolyObjectRec
			{
					ObjectInfoData
					long				pointCount;
					Boolean				bClosedPoly;				/* first point connects to last */
					Boolean				bWaterPoly;
					long				polyLineWidth;
			} PolyObjectRec, *PolyObjectRecPtr, **PolyObjectHdl;

typedef struct TextObjectRec
			{
					ObjectInfoData
					TextSpecRec			textSpec;
					char				textStr [256];
			} TextObjectRec, *TextObjectRecPtr, **TextObjectRecHdl;

typedef struct LineObjectRec
			{
					ObjectInfoData
					LongPoint			lineStartLPoint;
					LongPoint			lineEndLPoint;
			} LineObjectRec, *LineObjectRecPtr, **LineObjectRecHdl;

typedef struct UndoObjectRec
			{
					ObjectInfoData
					long				pointCount;
					Boolean				bClosedPoly;
					Boolean				bWaterPoly;
					long				polyLineWidth;
					LongRect			growFromLRect;
					LongRect			growToLRect;
			} UndoObjectRec, *UndoObjectRecPtr, **UndoObjectRecHdl;
			
extern class CToolSet dummyToolSet;

class CMapLayer_c  {
	
protected:
	LayerInfoData
	
	CMap				*Map;
	CMyList				*layerObjectList;
	CMyList				*layerGroupList;					/* sizeof (CMyList*) */
	CMyList				*layerSelectList;					/* sizeof (SelectDataRec)  */
	
	Boolean				bLayerModified;
	
public:
	
	//ClassID 			GetClassID () { return TYPE_CMAPLAYER; }//JLM
	
	
	long				GetLayerObjectCount ();
	CMyList			   *GetLayerObjectList ();
	CMyList			   *GetLayerGroupList ();
	
	/***************	Layer status getting routines */
	
	long				GetLayerPolyPtsCount ();
	void				GetLayerName (char *LayerNamePtr);
	void				GetLayerScope (LongRect *LayerLBoundsPtr, Boolean bRecalc);
	Boolean				IsLayerModified ();

};



#endif

