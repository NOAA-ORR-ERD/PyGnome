
#ifndef __GRIDVEL__
#define __GRIDVEL__

#include "Earl.h"
#include "TypeDefs.h"
#include "DagTree.h"
#include "GridVel_c.h"
#include "TriGridVel_c.h"
#include "TriGridVel3D_c.h"
#include "MemUtils.h"
#include "TClassID.h"

#ifdef __cplusplus
extern "C" {
#endif
int ConcentrationCompare(void const *x1, void const *x2);
int WorldPoint3DCompare(void const *x1, void const *x2);
#ifdef __cplusplus
}
#endif

class TGridVel : virtual public GridVel_c
{
	public:

		TGridVel();
		virtual	~TGridVel() { Dispose (); }
		virtual ClassID 	GetClassID 	() { return TYPE_GRIDVEL; }	
		virtual OSErr TextWrite(char *path){return noErr;}
		virtual OSErr TextRead (char *path){return noErr;}
		virtual OSErr Write(BFPB *bfpb)=0;
		virtual OSErr Read (BFPB *bfpb)=0;
		virtual void Draw(Rect r, WorldRect view,WorldPoint refP,double refScale,
						  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor)=0;
};

#include "RectGridVeL_c.h"

class TRectGridVel : virtual public RectGridVel_c, public TGridVel
{

		
	public:

		TRectGridVel();
		virtual	~TRectGridVel() { Dispose (); }
		 
		virtual ClassID GetClassID 	() { return TYPE_RECTGRIDVEL; }

		OSErr 			TextRead(char *path);
		OSErr 			ReadOssmCurFile(char *path);
		OSErr 			ReadOilMapFile(char *path);
		OSErr 			ReadGridCurFile(char *path);
		 
		OSErr 			Write(BFPB *bfpb);
		OSErr 			Read(BFPB *bfpb);
		
		void 			Draw (Rect r, WorldRect view,WorldPoint refP,double refScale,
		 					double arrowScale,double arrowDepth,Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor);
};


class TTriGridVel : virtual public TriGridVel_c, public TGridVel
{
	public:
		DOUBLEH fDepthContoursH;
		Rect	fLegendRect;
		
	public:		
		TTriGridVel(){fDagTree = 0; fBathymetryH=0;	fDepthContoursH=0;	memset(&fLegendRect,0,sizeof(fLegendRect));}
		virtual	~TTriGridVel() { if(fDepthContoursH) {DisposeHandle((Handle)fDepthContoursH); fDepthContoursH=0;} Dispose (); }

		virtual ClassID 	GetClassID 	() { return TYPE_TRIGRIDVEL; }
	
		OSErr TextRead(char *path);
		OSErr Read(BFPB *bfpb);
		OSErr Write(BFPB *bfpb);
		virtual void Draw (Rect r, WorldRect view,WorldPoint refP,double refScale,
				   double arrowScale,double arrowDepth,Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor);
		void DrawBitMapTriangles (Rect r);
		void DrawCurvGridPts(Rect r, WorldRect view);
		
		void DrawDepthContours(Rect r, WorldRect view, Boolean showLabels);
		void DrawContourScale(Rect r, WorldRect view/*, Rect *legendRect*/);
		void DrawContourLine(short *ix, short *iy, double** contourValue,Boolean showvals,double level);
		void DrawContourLines(Boolean printing,DOUBLEH dataVals, Boolean showvals,DOUBLEH contourLevels, short *sxi,short *syi);

		OSErr DepthContourDialog();
	//private:
		void DrawTriangle(Rect *r,long triNum,Boolean fillTriangle);
};


class TTriGridVel3D : virtual public TriGridVel3D_c, public TTriGridVel
{

	public:
		
		TTriGridVel3D();
		virtual	~TTriGridVel3D() { Dispose (); }

		virtual ClassID 	GetClassID 	() { return TYPE_TRIGRIDVEL3D; }
	
		Boolean **GetPtsSelection(Boolean initHdl);
		Boolean ThereAreTrianglesSelected() {if (fTriSelected) return true; else return false;}
		Boolean ThereAreTrianglesSelected2(void);
		Boolean SelectTriInPolygon(WORLDPOINTH wh, Boolean *needToRefresh);

		void 	ClearTriSelection();
		void 	ClearPtsSelection();

		OSErr 	ExportOilConcHdl(char* path);
		OSErr 	ExportTriAreaHdl(char* path,long numLevels);
		OSErr 	ExportAllDataAtSetTimes(char* path);	//maybe move to TModel since also handles budget table
		//OSErr TextRead(char *path);
		OSErr Read(BFPB *bfpb);
		OSErr Write(BFPB *bfpb);
		void DeselectAll(void);
		void DeselectAllPoints(void);
		void ToggleTriSelection(long i);
		void TogglePointSelection(long i);
		Boolean 	PointsSelected();

		long FindTriNearClick(Point where);
		//virtual InterpolationVal GetInterpolationValues(WorldPoint refPoint);
		DOUBLEH  GetDepthContours(){return fDepthContoursH;}
		//void SetDepthContours(DOUBLEH depthContoursH){if(fDepthContoursH!=depthContoursH) (fDepthContoursH=depthContoursH;}
		long 	GetNumDepthContours(void);
		virtual void Draw (Rect r, WorldRect view,WorldPoint refP,double refScale,
				   double arrowScale,double arrowDepth,Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor);
		void 	DrawPointAt(Rect *r,long verIndex,short selectMode );
		void DrawTriangleStr(Rect *r,long triNum,double value);
		void DrawDepthContours(Rect r, WorldRect view, Boolean showLabels);

		
	//private:
		void DrawTriangle3D(Rect *r,long triNum,Boolean fillTriangle,Boolean selected);
};

Boolean IsTriGridFile (char *path);
Boolean IsRectGridFile (char *path);
short ConcentrationTable(outputData **oilConcHdl,float *depthSlice,short tableType/*,double *triAreaArray,long numLevels*/);	// send oilconchdl

#endif
