
#ifndef __GRIDVEL__
#define __GRIDVEL__

#include "Earl.h"
#include "TypeDefs.h"
#include "DagTree/DagTree.h"

#ifndef pyGNOME
#ifdef __cplusplus
extern "C" {
#endif
int ConcentrationCompare(void const *x1, void const *x2);
int WorldPoint3DCompare(void const *x1, void const *x2);
#ifdef __cplusplus
}
#endif
#endif


//++ TGridVel

#include "GridVel/TGridVel.h"

//--

//++ TTriGridVel

#include "TriGridVel/TTriGridVel.h"

//--

class TRectGridVel : public TGridVel
{
	protected:
		VelocityH 	fGridHdl;
		long 		fNumRows;
		long 		fNumCols;
		
	public:
		virtual ClassID GetClassID 	() { return TYPE_RECTGRIDVEL; }

		TRectGridVel();
		virtual	~TRectGridVel() { Dispose (); }
		virtual void 	Dispose ();
		 
		virtual void 	SetBounds(WorldRect bounds);	

		OSErr 			TextRead(char *path);
		OSErr 			ReadOssmCurFile(char *path);
		OSErr 			ReadOilMapFile(char *path);
		OSErr 			ReadGridCurFile(char *path);
		 
		OSErr 			Write(BFPB *bfpb);
		OSErr 			Read(BFPB *bfpb);
		
		long 			NumVelsInGridHdl(void);
		VelocityRec 	GetPatValue(WorldPoint p);
		VelocityRec 	GetSmoothVelocity(WorldPoint p);
		
		void 			Draw (Rect r, WorldRect view,WorldPoint refP,double refScale,
		 					double arrowScale,Boolean bDrawArrows, Boolean bDrawGrid);
};



typedef struct {
	double avOilConcOverSelectedTri;
	double maxOilConcOverSelectedTri;
	Seconds time;
} outputData,*outputDataP,**outputDataHdl;


typedef struct {
	long maxLayer;
	long maxTri;
	Seconds time;
} maxLayerData,*maxLayerDataP,**maxLayerDataHdl;

class TTriGridVel3D : public TTriGridVel
{
	protected:
		FLOATH fDepthsH;
		DOUBLEH fDepthContoursH;
		Boolean	**fTriSelected;
		Boolean	**fPtsSelected;
		// maybe define a new class to handle all the output...
		outputDataHdl fOilConcHdl; 
		maxLayerDataHdl fMaxLayerDataHdl;
		double **fTriAreaHdl;
		DOUBLEH fDosageHdl;
		//WORLDPOINTDH gCoord;	// is this used??, maybe will need later if do refining on the fly
	public:
		Rect	fLegendRect;
		Boolean	bShowSelectedTriangles;
		float	fPercentileForMaxConcentration;
		Boolean bCalculateDosage;
		Boolean bShowDosage;
		float fDosageThreshold;	// this may need to be an array
		long fMaxTri;	// or selected tri
		Boolean bShowMaxTri;
	public:
		virtual ClassID 	GetClassID 	() { return TYPE_TRIGRIDVEL3D; }
		
		TTriGridVel3D();
		virtual	~TTriGridVel3D() { Dispose (); }
		virtual void 		Dispose ();

		void SetDepths(FLOATH depthsH){fDepthsH=depthsH;}
		FLOATH  GetDepths(){return fDepthsH;}
		void 	ScaleDepths(double scaleFactor);
		//void SetDepthContours(DOUBLEH depthContoursH){if(fDepthContoursH!=depthContoursH) (fDepthContoursH=depthContoursH;}
		DOUBLEH  GetDepthContours(){return fDepthContoursH;}
		OSErr DepthContourDialog();
		//WORLDPOINTDH  GetCoords(){return gCoord;}
		long 	GetNumDepths(void);
		//long	GetNumTriangles(void);
		long GetNumPoints(void);
		long 	GetNumDepthContours(void);
		long 	GetNumOutputDataValues(void);
		Boolean **GetTriSelection(Boolean initHdl);
		Boolean **GetPtsSelection(Boolean initHdl);
		Boolean ThereAreTrianglesSelected() {if (fTriSelected) return true; else return false;}
		Boolean ThereAreTrianglesSelected2(void);
		Boolean SelectTriInPolygon(WORLDPOINTH wh, Boolean *needToRefresh);
		void 	GetTriangleVerticesWP(long i, WorldPoint *w);
		OSErr 	GetTriangleVerticesWP3D(long i, WorldPoint3D *w);
		OSErr 	GetTriangleVertices3D(long i, long *x, long *y, long *z);
		outputDataHdl GetOilConcHdl(){return fOilConcHdl;}
		double **GetTriAreaHdl(){return fTriAreaHdl;}
		double **GetDosageHdl(Boolean initHdl);
		void 	ClearTriSelection();
		void 	ClearPtsSelection();
		OSErr 	GetTriangleVertices(long i, long *x, long *y);
		OSErr 	GetTriangleDepths(long i, float *z);
		OSErr 	GetMaxDepthForTriangle(long triNum, double *maxDepth);
		OSErr 	GetTriangleCentroidWC(long trinum, WorldPoint *p);
		double	GetTriArea(long triNum);
		OSErr 	CalculateDepthSliceVolume(double *triVol, long triNum,float upperDepth, float lowerDepth);
		double	GetMaxAtPreviousTimeStep(Seconds time);
		void	AddToOutputHdl(double avConcOverSelectedTriangles,double maxConcOverSelectedTriangles,Seconds time);
		void 	AddToTriAreaHdl(double *triAreaArray, long numValues);
		void	ClearOutputHandles();
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
		Boolean FloatPointInTriangle(double x1, double y1,double x2, double y2,
			double x3, double y3,double xp, double yp);
		long FindTriNearClick(Point where);
		//virtual InterpolationVal GetInterpolationValues(WorldPoint refPoint);
		virtual void Draw (Rect r, WorldRect view,WorldPoint refP,double refScale,
				   double arrowScale,Boolean bDrawArrows, Boolean bDrawGrid);
		void 	DrawPointAt(Rect *r,long verIndex,short selectMode );
		void DrawTriangleStr(Rect *r,long triNum,double value);
		//void DrawBitMapTriangles (Rect r);
		void DrawDepthContours(Rect r, WorldRect view, Boolean showLabels);
		void DrawContourScale(Rect r, WorldRect view/*, Rect *legendRect*/);
		void DrawContourLine(short *ix, short *iy, double** contourValue,Boolean showvals,double level);
		void DrawContourLines(Boolean printing,DOUBLEH dataVals, Boolean showvals,DOUBLEH contourLevels, short *sxi,short *syi);

		
	//private:
		void DrawTriangle3D(Rect *r,long triNum,Boolean fillTriangle,Boolean selected);
};

Boolean IsTriGridFile (char *path);
Boolean IsRectGridFile (char *path);
Boolean IsNetCDFFile (char *path, short *gridType);
Boolean IsNetCDFPathsFile (char *path, Boolean *isNetCDFPathsFile, char *fileNamesPath, short *gridType);
short ConcentrationTable(outputData **oilConcHdl,float *depthSlice,short tableType/*,double *triAreaArray,long numLevels*/);	// send oilconchdl

#endif
