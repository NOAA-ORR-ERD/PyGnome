#ifndef __PLOTH__
#define __PLOTH__
#include "Cross.h"

//// fudge Error menu items
#define kDepthSlice	13701		//  change to depth slice
#define kTriArea		13702		//  change to area 
#define kAvConc		13703
#define kMaxConc		13704
#define kComboConc	13705


////// for oiled shoreline plot
#define kShorelineGals	1
#define kShorelineMiles	2

//////// graphing
#define kPlainStyle					0
#define kNoScroll					false
#define kDefaultSetting				nil
#define kManualSetValue				1

#define kBorderDistance				20	// not used
#define kVertical					true
#define kHorizontal					false
#define kGetBorderError				-2	// not used

/* who set the min/max values */
#define kSetManually				1
#define kSetByDefault			2

/* Axis Locations */
#define kTopAxis					0
#define kLeftAxis					1
#define kBottomAxis				2
#define kRightAxis				3

/* Border Types */
#define kProgrammerDefinedBorder	-1
#define kPlainBorder				1

/* Grid Types */
#define kProgrammerDefinedGrid		-1
#define kStandardGrid				1

/* Tick Type */
#define kPrimaryTicks				true
#define kSecondaryTicks				false

#define kNoLabels					false
#define kUseLabels					true

#define kProgrammerDefinedTicks		-1
#define kNoTicks					0
#define kPlainTicks					1
#define kSmallPlainTicks			2

#define kNoGrid						false
#define kDrawGrid					true

#define kNoIcon						0

// data types
#define kProgrammerDefinedDataType	-1
#define kShort						1
#define kLong						2
#define kDouble						3
#define kFloat						4

// special data actions
#define kCalcMinMax					1
#define kGetElementValues			2

// Label Item Messages
#define kInitMsg					1
#define kDisposeMsg					2
#define kDrawMsg					3

// text draw justification
#define kLeftJust		0
#define kCenterJust		1
#define kRightJust		2

#define kNumSpecies	7
#define kNumConcernLevels	3

enum { LOWCONCERN = 1, MEDIUMCONCERN = 2, HIGHCONCERN = 3 };
enum { ADULTFISH = 1, CRUSTACEANS = 2, SENSLIFESTAGES = 3, ADULTCORAL, STRESSEDCORAL, CORALEGGS, SEAGRASS };

typedef OSErr (*MyPlotProc)(double *xPtr, double *yPtr, long pntnum);
typedef OSErr (*MyOffPlotProc)(double *pxPtr, double *pyPtr, double *xPtr, double *yPtr, Boolean moveOnly, long pntnum);
typedef void (*MyDrawFuncProc)(short xStart, short yStart, short xEnd, short yEnd);
typedef void (*MyGridFuncProc)(short xStart, short yStart, short xEnd, short yEnd);
typedef void (*MyTickFuncProc)(short borderElem, Boolean useLabels, short startH, short startV, double *tickVal, short numDecPos);

typedef struct PenInfo
{
	short	size;
	long	color;
	short	patInfo;
} PenInfo, *PenInfoPtr, **PenInfoHandle;

typedef struct TextInfo
{
	short			theFont;
	short			theSize;
	short			theJust;			// text justification
} TextInfo, *TextInfoPtr, **TextInfoHandle;

typedef struct TickInfo
{
	short			type;				// the type of tick mark to use
										//  -2  - use the icon resource specified in iconNum
										//  -1  - use the drawing function passed to draw the ticks
										//   0  - not in use
	Boolean			useLabels;			// is this type of tick to be labeled
	Boolean			drawGrid;			// should grid lines be drawn on this tick mark
	MyTickFuncProc	drawingFunc;		// a ptr to a specialized drawing routine
	double			spreadVal;			// the spread in value between tick marks of this type
	short			iconToUse;			// the icon resource to use.
	TextInfo		labelTextInfo;		// type of text to be used in labeling
} TickInfo, *TickInfoPtr, **TickInfoHandle;

typedef struct AxisInfo
{
	short			type;				// type of label
	MyDrawFuncProc	drawingFunc;		// a ptr to a specialized drawing routine
	char			titleStr[256];		// characters to used as the label description
	TextInfo		titleTextInfo;		// text info for the title string
	short			titleRotation;		// orientation of the title
	TickInfo		primaryTicks;		// primary tick mark info
	TickInfo		secondTicks;		// secondary tick info
} AxisInfo, *AxisInfoPtr, **AxisInfoHandle;

typedef struct OverLayInfo
{
	Boolean showOverLay;
	char	labelStr[64];		// characters to use as the label description
	double xOverLay[3];
	double yOverLay[3];
}OverLayInfo, *OverLayInfoPtr, **OverLayInfoHandle;

typedef struct OverLayDataBase
{
	//Boolean showOverLay;
	char	speciesTypeStr[64];		// characters to use as the label description
	//OverLayInfo concernLevels[3];
	OverLayInfo lowConcern;
	OverLayInfo mediumConcern;
	OverLayInfo highConcern;
}OverLayDataBase, *OverLayDataBasePtr, **OverLayDataBaseH;

typedef struct GraphValues
{
	char			graphTitle[256];	// the title of the graph
	char			graphSubTitle[256];	// the title of the graph
	TextInfo		graphTitleInfo;		// title text style, size and font info
	Rect			theArea;			// the dimensions of the graph
	Rect			gridrect;
	short			width;				// total graph width
	short			height;				// total graph height
	Boolean		hasVertScroll;		// is a vertical scroll bar available
	Boolean		hasHorizScroll;		// is a horizontal scroll bar available
	Handle		vertScrollHdl;		// the vertical scroll control
	Handle		horizScrollHdl;		// the horizontal scroll control
	short			borderDistTop;		// distances from the edge of the graph area to the
	short			borderDistLeft;		//   drawn edges of the graph
	short			borderDistBot;		//   < 0 indicate the graph code calcs a default
	short			borderDistRight;
	short			gridType;			// type of grid to use
	short			gridPat;			// pattern to use in drawing the grid
	MyGridFuncProc	gridDrawFunc;		// an option ptr to a specialized grid line 
										//   drawing function
	Ptr			xData;				// a ptr to the x direction data coordinates
										//   or a link list header
	Ptr			yData;				// a ptr to the y direction data coordinates

	double		minXValue;			// the minimum value of the X axis
	double		maxXValue;			// the maximum value of the X axis
	short			mmXSetBy;			// who set the min/max value for X
										//		0 -		not set
										//		1 -		set manually
										//		2 -		set by the default
	double		minXLabVal;			// the beginning value of the x axis
	double		maxXLabVal;			// the ending value of the x axis
	double		minYValue;			// the minimum value of the Y axis
	double		maxYValue;			// the maximum value of the Y axis
	short			mmYSetBy;			// who set the min/max value for Y
										//		0 -		not set
										//		1 -		set manually
										//		2 -		set by the default
	double		minYLabVal;			// the beginning value of the y axis
	double		maxYLabVal;			// the ending value of the y axis
	short			ndcmlplcs;			// number of decimal places to use (aps 1/17/95)
	short			islogplot;			// is this a log y axis plot (aps 1/17/95)
	AxisInfo		theLabels[4];		// the graph labels
	long			numDataElems;		// the number of data points in the arrays below (or)
										//	-2		'xData' has a pointer to a function to plot for
										//	-1		A linked list is being used in place of an array
										//			  'xData' holds a ptr/handle to the first item
										//			  'yData' holds a ptr to a function that reads the data
	short			dataType;			// what type of data is in the arrays
										//		0 -		not set
										//		1 -		short
										//		2 -		long
										//		3 - 	double
										//		4 -		float
										//   or a pointer to a function that performs certain
										//   actions on a linked list
	//Ptr xOverLay;
	//Ptr yOverLay;
	Boolean		multipleCurves;
	Boolean		ShowToxicityThresholds;
	//short			LevelOfConcern;	// low, medium, high
	char			LegendStr[64];	// level of concern or species type
	OverLayInfo		ToxicityLevels[3];	// fish, crustaceans, sensitive life stages, store all levels for each??
	MyPlotProc		dataPlotFunc;		// an option ptr to a specialized data plotting function
	MyOffPlotProc	dataOffPlotFunc;	// an option ptr to a specialized data plotting function
} GrafVal, *GrafValPtr, **GrafValHdl;


OSErr SetOverLayDataBase();
void CalcGraphMinMax(GrafValHdl theGraph);
OSErr Coord2GraphPt(GrafValHdl theGraph, double* xValPtr, double* yValPtr, short* xDistPtr, short* yDistPtr);
void DrawAllBorders(GrafValHdl theGraph);
void DrawBorder(GrafValHdl theGraph, short borderElem);
void DrawBorderTitle(GrafValHdl theGraph, short borderElem);
void DrawGraph(GrafValHdl theGraphHdl);
void DrawGrid(GrafValHdl theGraph);
void DrawLine(GrafValHdl theGraph, double* startXPtr, double* startYPtr, double* endXPtr, double* endYPtr,
				MyDrawFuncProc	drawLineFunc);
void DrawTicks(GrafValHdl theGraph, short	borderElem, Boolean isPrimary);
void DrawTitle(GrafValHdl theGraphHdl);
OSErr Ext2PixelNum(double*	value, GrafValHdl theGraph, Boolean isVertical, short *pixelDist);
short GetBorderDistance(GrafValHdl theGraph, short fromWhichSide);
OSErr GetGraphDataElem(GrafValHdl theGraph, double *xValPtr, double *yValPtr, long elemNum);
TextInfo GetTextType(void);
GrafVal	**InitGraph(Boolean hasVScroll, Boolean hasHScroll);
void InitLabel(AxisInfo* theAxis);
void InitTextInfo(TextInfo* textInfo);
void InitTickInfo(TickInfo* tickInfo);
void NumDraw(Point textPt, double* tickValPtr, short numDecimals, short just);

void PlotData(GrafValHdl theGraph);
void PlotOverLay(GrafValHdl theGraph);
void SetAxisFontData(GrafValHdl theGraph,short	labelElem, short	theFont, short	theSize, short	theJustification);
void SetAxisTypeData(GrafValHdl theGraph, short borderNum, short axisType,MyDrawFuncProc drawingFunc);
void SetBorderDistance(GrafValHdl theGraph, short fromTop, short fromLeft, short fromBottom, short fromRight);
void SetBorderTitle(GrafValHdl theGraph, short borderElem, char* theTitleStr, short theFont, short theSize, 
						short	theJustification, short	rotation);
void SetData(GrafValHdl theGraph, Ptr xDataPtr,Ptr yDataPtr, long numDataElems, short dataType, Boolean calcDefaultMM);
void SetDataPlotType(GrafValHdl theGraphHdl, MyPlotProc dataPlotFunc, MyOffPlotProc dataOffPlotFunc);

void SetGraphArea(GrafValHdl theGraphHdl, Rect* theDimensions);
void SetGraphTitle(GrafValHdl	theGraphHdl, char* titleStr, char* subTitleStr, short theFont, short	theSize);
void SetGrid(GrafValHdl theGraph, short gridType, MyGridFuncProc drawingFunc, short gridPat);
void SetTextType(TextInfo* theTextType);
void SetTickInfo(GrafValHdl theGraph, short	labelNum, Boolean isPrimary, Boolean useLabels, Boolean drawGrid, 
					short type, double spreadVal, short	iconToUse, TextInfo* labelText);
void SetToxicityThresholds(GrafValHdl theGraph, OverLayInfo toxicityInfo, short overLayType);
OverLayInfo GetToxicityThresholds(GrafValHdl theGraph, short overLayType);
void SetXMinMax(GrafValHdl theGraph, double* min, double* max);
void SetXMinMaxLong(GrafValHdl theGraph, long min, long max);
void SetYMinMax(GrafValHdl theGraph, double* min, double* max);
void SetYMinMaxLong(GrafValHdl theGraph, long min, long max);

double GetRemainder(double X, double Y);
void IntDraw(Point textPt, double *val, short just);
void DblDraw(short ndcmlplcs, Point textPt, double *val, short just);
void DblExpDraw(short ndcmlplcs, Point textPt, double *val, short just);

void GetGridRect(GrafValHdl grafHdl, Rect *theRect);

//short DataPlot1DepthBlue(GrafValHdl	theGraphHdl, double *xValPtr, double *yValPtr, long pntnumber);
//short DataOffPlot1DepthBlue(GrafValHdl theGraphHdl, double *prevXValPtr, double *prevYValPtr, 
		//double *xValPtr, double *yValPtr, Boolean moveOnly, long pntnumber);

void DrawDepth1Line(short xStart, short yStart, short xEnd, short yEnd);

void DrawSmallTick(short borderElem, Boolean useLabels, short startH, short startV, double *tickVal, short numDecPos);
void DrawPlainTick(short borderElem, Boolean useLabels, short startH, short startV, double *tickVal, short numDecPos);

OSErr StandardPlot(double *xPtr, double *yPtr, long pntnum);
OSErr StandardPlot2(double *xPtr, double *yPtr, long pntnum);
OSErr StandardOffPlot(double *pxPtr, double *pyPtr, double *xPtr, double *yPtr, Boolean moveOnly, long pntnum);
OSErr StandardOffPlot2(double *pxPtr, double *pyPtr, double *xPtr, double *yPtr, Boolean moveOnly, long pntnum);
OSErr StandardPlot3(double *xPtr, double *yPtr, long pntnum);
OSErr StandardOffPlot3(double *pxPtr, double *pyPtr, double *xPtr, double *yPtr, Boolean moveOnly, long pntnum);
OSErr StandardPlot4(double *xPtr, double *yPtr, long pntnum);
OSErr StandardOffPlot4(double *pxPtr, double *pyPtr, double *xPtr, double *yPtr, Boolean moveOnly, long pntnum);
void SplitDouble(double *value, double *frac_value, short *power);
void Raise2Pow(double *inval, double *outval, short power);
void ArrayMinMaxDouble(double *array, long n, double *minv, double *maxv);
void ArrayMinMaxFloat(float *array, long n, double *minv, double *maxv);
void ArrayMinMaxLong(long *array, long n, double *minv, double *maxv);
void ArrayMinMaxShort(short *array, long n, double *minv, double *maxv);
short GetNumDecPlaces(double* theValue);
void TextDraw(char *str, short just);

OSErr ToxicityOverLayDialog();
OSErr SetAxesDialog();
OSErr SetAxesDialog2();
#endif	// __PLOTH__