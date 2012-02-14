
#ifndef __OBJECTUTILS__
#define __OBJECTUTILS__

#include	"CMap2.h"
#include	"Geometry.h"
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

#define				kPolyType			'POLY'
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

class CMapLayer
			{
				private:
					LayerInfoData
					CMap				*Map;
					WindowPtr			WPtr;

					CMyList				*layerObjectList;
					CMyList				*layerGroupList;					/* sizeof (CMyList*) */
					CMyList				*layerSelectList;					/* sizeof (SelectDataRec)  */

					Boolean				bLayerModified;
					UndoObjectRecHdl	layerUndoObjectHdl;
					long				layerUndoCode;
					
					long				layerTextItemNum;					/* index number of edit-text object */
					Point				layerTextStartPt;					/* start click point for edit-text */

				public:

					virtual OSErr 		Read (BFPB *bfpb); //JLM
					virtual OSErr 		Write (BFPB *bfpb); //JLM
					ClassID 				GetClassID () { return TYPE_CMAPLAYER; }//JLM

					OSErr				IMapLayer (WindowPtr mapWPtr, CMap *Map, Boolean bLayerVisible, Boolean bLayerOLabelVisble);
					virtual			   ~CMapLayer() { Dispose(); }
					virtual void		Dispose ();
					
					Boolean				PerformUndo ();
					void				UndoLayerObjects ();
					long				GetLayerObjectCount ();
					CMyList			   *GetLayerObjectList ();
					CMyList			   *GetLayerGroupList ();

					void				DoIdle ();
					Boolean				DoMenu (long menuResult, char *ChoiceStr, Boolean *modified);
					Boolean				DoTextMenu (long menuResult, char *ChoiceStr);
					Boolean				DoKey (WindowPtr WPtr, long message);
					ObjectRecHdl		AddNewObject (OSType ObjectType, LongRect *thisObjectLRectPtr, Boolean bPutAtTop);
//					Boolean				DoClick (CToolSet *mapToolSet, Point LocalPt, Boolean *bModified);
					Boolean				DoClick (char *currToolName, Point LocalPt, Boolean *bModified);
					ObjectRecHdl		FindClickedObject (Point LocalClickPt);
					Boolean				DoGrowSelection (Point LocalClickPt);
					Boolean				DoDragSelection (Point LocalClickPt);
					Boolean				DoObjectDblClick (ObjectRecHdl ClickedObjectHdl);
					void				DeleteObject (ObjectRecHdl theObjectHdl, Boolean bInvalidate);
					void				DeleteSelection ();
					void				SetMenuStates ();

					Boolean				DrawLayer (LongRect *UpdateLRectPtr, long mode);
					long				GetSelObjectCount ();
					void				InvertSelection (Boolean bSetFlag, Boolean bTheFlag);
					void				ClearSelection (Boolean bInvert, Boolean bSetFlag);
					Boolean				GroupSelection   (Boolean bInvert);
					Boolean				UngroupSelection (Boolean bInvert);
					
					/***************	Layer Text Edit routines */
					
					OSErr				StartNewText (Point LocalPt, TextObjectRecHdl textObjectHdl);
					Boolean				SaveActiveText ();

					/***************	Layer status setting routines */

					void				SetLayerName (char *LayerNamePtr);
					void				SetLayerScope (LongRect *thisLayerLRectPtr);
					void				SetLayerModified (Boolean theModFlag);

					void				SetLayerVisible (Boolean Visible);
					void				SetLayerMinVisScale (long theScale);
					void				SetLayerMaxVisScale (long theScale);

					void				SetLayerOLabelVisible (Boolean OLabelVisible);
					void				SetLayerOLMinScale (long theScale);
					void				SetLayerOLMaxScale (long theScale);

					void				SetLayerOLabelFont (char *LayerLabelFont);
					void				SetLayerOLabelFSize (long ObjectFSize);
					void				SetLayerOLabelFStyle (long ObjectFStyle);				
					void	 			SetLayerOLabelColor (long OLabelColorInd);

					void				SetLayerObjectList (CMyList *ObjectList);
					void				SetLayerGroupList (CMyList *theGroupList);
					void				SetLayerSelectList (CMyList *theSelectList);

					/***************	Layer status getting routines */

					long				GetLayerPolyPtsCount ();
					void				GetLayerName (char *LayerNamePtr);
					void				GetLayerScope (LongRect *LayerLBoundsPtr, Boolean bRecalc);
					Boolean				IsLayerModified ();

					Boolean				IsLayerVisible ();
					long				GetLayerMinVisScale ();
					long				GetLayerMaxVisScale ();
					Boolean				IsLayerOLabelVisible ();
					
					/***************	Layer reading and writing routines */
					
					OSErr				WriteLayer (short FRefNum);
					OSErr				ReadLayer  (short FRefNum);

					/***************	Misc public Layer routines */

					OSErr				ClipLayerToScreen ();

					//// JLM --  I need GetDrawSettings() to be public
					void				GetDrawSettings (DrawSpecRecPtr drawSettings, ObjectRecHdl theObjectHdl, long theMode);
				private:
										/* routines that deal with groups */
					CMyList			   *AddNewGroup ();								/* adds new group header */
					void				DeleteGroup (CMyList *thisGroupOList);
					long				GetLayerGroupCount ();						/* number of groups in layer */
					CMyList			   *GetGroupObjectList (long GroupNum);
										
										/* more routines that deal with groups */
					OSErr				AddObjectToGroup (CMyList *theObjectList, ObjectRecHdl theObjectHdl);
					OSErr				AddOListToGroup (CMyList *mainObjectList, CMyList *groupObjectList);

										/* more routines that deal with groups */
					void				InvertGroup (CMyList *theObjectList);
					void				SelectGroup (CMyList *theObjectList, Boolean bInvert, Boolean bSetFlag, Boolean bSelected);
					void				GetGroupLRect (CMyList *thisGroupOList, LongRect *theGroupLRectPtr);
					Boolean				IsObjectGrouped (ObjectRecHdl theObjectHdl, CMyList **theGroupListPtr, long *theGroupNumPtr);
					
										/* routines to read and write layer group information */
					OSErr				WriteLayerGroups (short FRefNum);
					OSErr				ReadLayerGroups (short FRefNum);
					OSErr				WriteLinkedList (short FRefNum, CMyList *theObjectList);
					OSErr				ReadLinkedList (short FRefNum, CMyList *theObjectList);

										/* routines that deal with the current selection list */
					Boolean				IsGroupSelected (CMyList *theObjectList);
					void				SelObjectsInGroup (long groupNum, Boolean bInvert, Boolean bSelect);
					OSErr				AddObjectToSelList  (ObjectRecHdl theObjectHdl, Boolean bInvert, Boolean bSetFlags);
					OSErr				AddObjectsToSelList (CMyList *theObjectList);
					void				RemObjectFromSelList (ObjectRecHdl theObjectHdl, Boolean bInvert, Boolean bSetFlags);

					OSErr				AddGroupToSelList (CMyList *theGroupOList);
					void				RemGroupFromSelList (CMyList *theGroupOList);
			};

/*************** recursive routines that take object-data-rec lists as parameters */

void			GetGroupListLRect (CMyList *groupObjectList, LongRect *theLRectPtr);
void			SetGroupSelected (CMyList *groupObjectList, Boolean bSelected);
Boolean			IsListInList (CMyList *keyList, CMyList *mainList);
Boolean			IsObjectInList (ObjectRecHdl theObjectHdl, CMyList *groupObjectList);
void			RemoveObjectFromList (CMyList *theObjectList, ObjectRecHdl theObjectHdl);
void 			MarkAllObjects (CMyList *theLayerList, Boolean MarkFlag);
void			MarkObjectsInGroup (CMyList *groupObjectList);
void			PurgeObjectList (CMyList *theOjbectList);
void			DrawObjectsInList (CMyList *thisObjectList, CMap *Map, LongRect *UpdateLRectPtr, DrawSpecRecPtr drawSettings);
void 			OffsetObjectsInList (CMyList *thisObjectList, CMap *Map, Boolean bInvalidate, long LDx, long LDy);
void			UndoObjectsInList (CMyList *thisObjectList, Boolean bUndoFlag);
void 			ScaleObjectsInList (CMyList *thisObjectList, CMap *Map, Boolean bInvalidate, ScaleRec *ScaleInfo);
Boolean 		ListContainsList (CMyList *theOjbectList, CMyList **NextSubList);
Boolean			IsObjectTypeInList (CMyList *theObjectList, OSType theObjectType,
									ObjectRecHdl *theObjectHdl);
void			TextSpecObjectsInList (CMyList *thisObjectList, CMap *Map, long SpecCode, long NewSpec,
									   Boolean bUpdate);

/**************	routines that use ID's */

void			GetUniqueID (ObjectIDRecPtr idPtr, unsigned short idCounter);
Boolean			IsIdZero (ObjectIDRecPtr idPtr);
ObjectRecHdl	GetObjectWithID (CMyList *theLayerList, ObjectIDRecPtr objectIDPtr);

/**************	Object flag get and set utilities */

Boolean			IsObjectSelected (ObjectRecHdl ObjectHdl);
Boolean			IsObjectMarked (ObjectRecHdl ObjectHdl);
Boolean			IsObjectVisible (ObjectRecHdl ObjectHdl);
Boolean			IsOLabelVisible (ObjectRecHdl ObjectHdl);
Boolean			IsOSymbolVisible (ObjectRecHdl thisObjectHdl);
Boolean			IsOLabelZoom (ObjectRecHdl thisObjectHdl);
Boolean			IsObjectFGrouped (ObjectRecHdl thisObjectHdl);
Boolean			IsObjectHotLinked (ObjectRecHdl thisObjectHdl);
Boolean			IsObjectUndone (ObjectRecHdl thisObjectHdl);

void			SetObjectSelected (ObjectRecHdl ObjectHdl, Boolean bSelected);
void			SetObjectMarked (ObjectRecHdl ObjectHdl, Boolean bMarked);
void			SetObjectVisible (ObjectRecHdl ObjectHdl, Boolean bVisible);
void			SetOLabelVisible (ObjectRecHdl ObjectHdl, Boolean bLabelVisible);
void			SetOSymbolVisible (ObjectRecHdl ObjectHdl, Boolean bSymVisible);
void			SetOLabelZoom (ObjectRecHdl ObjectHdl, Boolean bLabelZoomFlag);
void			SetObjectGrouped (ObjectRecHdl ObjectHdl, Boolean bGrouped);
void			SetOHotLinked (ObjectRecHdl ObjectHdl, Boolean bHotLinked);
void			SetObjectUndone (ObjectRecHdl ObjectHdl, Boolean bUndoFlag);

/**************	Object info data get and set utilities */

void			GetObjectType (ObjectRecHdl ObjectHdl, OSType *ObjectTypePtr);
void			GetObjectID (ObjectRecHdl ObjectHdl, ObjectIDRecPtr objectIDPtr);
long			GetObjectSymbol (ObjectRecHdl thisObjectHdl);
void			GetObjectLRect (ObjectRecHdl ObjectHdl, LongRect *ObjectLRectPtr);
void			GetObjectLPoint (ObjectRecHdl ObjectHdl, LongPoint *ObjectLPointPtr);
long			GetObjectColor (ObjectRecHdl ObjectHdl);
long			GetObjectBColor (ObjectRecHdl ObjectHdl);
void			GetOLabelLPoint (ObjectRecHdl thisObjectHdl, LongPoint *LabelLPointPtr);
void			GetObjectLabelRect (ObjectRecHdl ObjectHdl, Rect *LabelRectPtr);
long			GetObjectHotLinkData (ObjectRecHdl ObjectHdl);
void			GetObjectLabel (ObjectRecHdl ObjectHdl, char *ObjectName);
long			GetOLabelJust (ObjectRecHdl ObjectHdl);
long			GetObjectCustData (ObjectRecHdl thisObjectHdl);
Handle			GetObjectDataHdl (ObjectRecHdl thisObjectHdl);
void			GetOLabelScrRect (ObjectRecHdl thisObjectHdl, Rect *theScrRect);
void			GetObjectESICode (ObjectRecHdl ObjectHdl, long *theESICode);
void			GetObjectFillPat (ObjectRecHdl theObjectHdl, Pattern *theFillPat);

void			SetObjectType (ObjectRecHdl ObjectHdl, OSType theObjectType);
void			SetObjectID (ObjectRecHdl ObjectHdl, ObjectIDRecPtr objectIDPtr);
void			SetObjectSymbol (ObjectRecHdl thisObjectHdl, long ObjectSymbol);
void			SetObjectLRect (ObjectRecHdl ObjectHdl, LongRect *ObjectLRectPtr);
void			SetObjectLPoint (ObjectRecHdl ObjectHdl, LongPoint *ObjectLPointPtr);
void			SetObjectColor (ObjectRecHdl ObjectHdl, long ObjectColorInd);
void			SetObjectBColor (ObjectRecHdl ObjectHdl, long ObjectColorInd);
void			SetOLabelLPoint (ObjectRecHdl thisObjectHdl, LongPoint *LabelLPointPtr);
void			SetObjectLabelRect (ObjectRecHdl ObjectHdl, Rect *LabelRectPtr);
void			SetObjectHotLinkData (ObjectRecHdl ObjectHdl, long HotLinkData);
void			SetObjectLabel (ObjectRecHdl ObjectHdl, char *ObjectName);
void			SetOLabelJust (ObjectRecHdl ObjectHdl, long justCode);
void			SetObjectCustData (ObjectRecHdl thisObjectHdl, long theCustData);
void 			SetObjectDataHdl (ObjectRecHdl thisObjectHdl, Handle theDataHdl);
void			SetObjectESICode (ObjectRecHdl ObjectHdl, long theESICode);
void			SetObjectFillPat (ObjectRecHdl theObjectHdl, Pattern *theFillPat);

/**************	Other object utilities */

				/* index-from-handle and handle-from-index utilities */
long			GetObjectIndex  (ObjectRecHdl theObjectHdl, CMyList *theObjectList);
ObjectRecHdl	GetObjectHandle (CMyList *theObjectList, long ObjectIndex);
void			GetObjectScrRect (CMap *Map, ObjectRecHdl theObjectHdl, Rect *theScrRect);
void			CalcSetOLabelRect (CMap *Map, ObjectRecHdl theObjectHdl, Rect *newLabelRect);

				/* object writing and reading routines */
OSErr			WriteObjectInfo (short FRefNum, ObjectRecHdl thisObjectHdl);
OSErr			ReadObjectInfo (short FRefNum, ObjectRecHdl *thisObjectHdlPtr);

				/* general object routines */
Boolean			PtInObjectLRect (ObjectRecHdl theObjectHdl, LongPoint *MatrixLPtPtr);
Boolean			PointOnObject (CMap *Map, ObjectRecHdl theObjectHdl, Point LocalPt, long *clickedPart);
Boolean			ObjectSectLRect (ObjectRecHdl theObjectHdl, LongRect *theLRectPtr);
void			InvertObject (CMap *Map, ObjectRecHdl thisObjectHdl);
void			SelectObject (CMap *Map, ObjectRecHdl thisObjectHdl, Boolean Invert, Boolean bSetFlag, Boolean bTheFlag);
void			KillObject (ObjectRecHdl ObjectHdl);
void			SortObjectsBySize (CMyList *thisObjectList);
double			GetObjectArea (ObjectRecHdl thisObjectHdl);
void			GetObjectCenter (ObjectRecHdl thisObjectHdl, LongPoint *CenterLPtPtr);
void 			OffsetObject (ObjectRecHdl ObjectHdl, CMap *Map, long LDx, long LDy, Boolean bInvalidate);
void			ScaleObject  (ObjectRecHdl ObjectHdl, CMap *Map, ScaleRecPtr ScaleInfoPtr, Boolean bInvalidate);
void			MarkObjectsInList (CMyList *thisObjectList, Boolean bMark);
void 			DrawObject (ObjectRecHdl theObjectHdl, CMap *Map, LongRect *UpdateLRectPtr, DrawSpecRecPtr drawSettings);
long			GetObjectPart (CMap *Map, ObjectRecHdl theObjectHdl, LongPoint *MatrixLPtPtr);
void			InvalObjectRect (CMap *Map, ObjectRecHdl theObjectHdl);
Boolean			DoBringToFront (CMap *Map, CMyList *theObjectList);
Boolean			DoSendToBack (CMap *Map, CMyList *theObjectList);

void			ClipAboveObject (CMap *Map, CMyList *theObjectList, ObjectRecHdl theObjectHdl, Boolean ExpandClip);
void			InvalAboveObject (CMap *Map, CMyList *theObjectList, ObjectRecHdl theObjectHdl);
void			InvalBelowObject (CMap *Map, CMyList *theObjectList, ObjectRecHdl theObjectHdl);

/**************	routines that deal with layer lists */

void			DisposeLayerList (CMyList *theLayerList);
void			GetMapLBounds (CMyList *theLayerList, LongRect *MapLRectPtr);
CMapLayer	   *AddNewLayer (CMap *Map, CMyList *theLayerList, Boolean bLayerVisFlag, Boolean bLOLabelVisFlag);

/**************	Public Layer Handling routines */

CMyList		   *GetNewLayerList ();

/**************	Private Layer Handling routines */

CMapLayer	   *GetMapLayer (CMyList *theLayerList, long LayerNum);

/**************	Polygon handling routines */

OSErr			WritePolyPoints (short FRefNum, PolyObjectHdl thePolyHdl);
void			CalcSetPolyLRect (PolyObjectHdl thePolyHdl);
CMyList 	   *GetSectPolyList (PolyObjectHdl MainPolyHdl, LongRect sectLRect, OSErr &err);
SegmentsHdl		RgnToSegment (PolyObjectHdl MapPolyHdl);	// converts rgnPoly to segments handle
OSErr		 	SegmentToRgn (SegmentsHdl theSegment, PolyObjectHdl newPoly);
SegmentsHdl		MakeSectPoly (LongRect *sectLRect, PolyObjectHdl MapPolyHdl);
Boolean			PointInPoly (LongPoint *theLPoint, PolyObjectHdl mapPolyHdl);
OSErr 			IntersectPoly (PolyObjectHdl MainPolyHdl, LongRect *sectLRect, CMyList *theObjectList);
OSErr 			AddPointToPoly (PolyObjectHdl thePoly, LongPoint *newLPoint);

LongPoint**		GetPolyPointsHdl (PolyObjectHdl thePolyHdl);
long			GetPolyPointCount (PolyObjectHdl thePolyHdl);
void			SetPolyClosed (PolyObjectHdl thePolyHdl, Boolean bClosed);
void			SetPolyWaterFlag (PolyObjectHdl thePolyHdl, Boolean bWaterPoly);

Handle 			CreateNewPtsHdl (PolyObjectHdl thePolyHdl, long PointCount);
void			SetPolyPointsHdl (PolyObjectHdl thePolyHdl, LongPoint **thePointsHdl);
void			SetPolyPointCount (PolyObjectHdl thePolyHdl, long PointCount);
Boolean			IsPolyClosed (PolyObjectHdl thePolyHdl);
Boolean			IsWaterPoly (PolyObjectHdl thePolyHdl);

				/* polygon drawing routines */
void			DrawMapPoly (CMap *Map, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings);
//#ifdef IBM
void			DrawMapBoundsPoly (CMap *Map, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings, Boolean erasePolygon);
//#endif
void			DrawNoSectPoly (CMap *Map, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings);
OSErr			DrawSectPoly (CMap *Map, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings);
void 			DrawBeachLEs (CMap *theMap, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings);

/**************	Zoom-Text handling routines */

#define			kNewFontCode		1
#define			kNewSizeCode		2
#define			kNewStyleCode		3
#define			kNewJustCode		4


void 			SetObjectText (TextObjectRecHdl theObjectHdl, char *TextStr);
void			GetObjectText (TextObjectRecHdl theObjectHdl, char *TextStr);
void			SetObjectText  (TextObjectRecHdl theTextObjectHdl, char *theObjectStr);
long			GetObjectFont (TextObjectRecHdl theObjectHdl);
void			SetObjectTextSpec (TextObjectRecHdl TextObjectHdl, CMap *Map, long SpecCode, long NewSpec,
								   Boolean bUpdate);
Boolean			DrawTextObject (TextObjectRecHdl theObjectHdl, CMap *Map, LongRect *UpdateLRectPtr,
								DrawSpecRecPtr drawSettings);

/**************	Misc. independent routines */

#ifdef IBM
	Boolean BitTst(void *bytePtr, long bitNum); //JLM
	void BitSet(void *bytePtr, long bitNum);
	void BitClr(void *bytePtr, long bitNum);
#endif

extern "C" { int ObjSizeComp (const void *Object1HdlPtr, const void *Object2HdlPtr); }

#endif

