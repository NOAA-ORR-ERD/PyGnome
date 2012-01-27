
#ifndef __NetCDFMover__
#define __NetCDFMover__

#include "NetCDFMover_c.h"

#include "GridVel.h"
#include "PtCurMover.h"
#include "PtCurMap.h"

Seconds RoundDateSeconds(Seconds timeInSeconds);
PtCurMap* GetPtCurMap(void);


class NetCDFMover : virtual public NetCDFMover_c,  public TCurrentMover
{
	public:
							NetCDFMover (TMap *owner, char *name);
						   ~NetCDFMover () { Dispose (); }
		virtual void		Dispose ();
	virtual OSErr		InitMover (); //  use TCATSMover version which sets grid ?
	virtual ClassID 	GetClassID () { return TYPE_NETCDFMOVER; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFMOVER) return TRUE; return TCurrentMover::IAm(id); }
	void 					DisposeLoadedData(LoadedData * dataPtr);	
	void 					ClearLoadedData(LoadedData * dataPtr);
	void 					DisposeAllLoadedData();
	virtual	OSErr 	ReplaceMover();
	
	long 					GetNumTimesInFile();
	long 					GetNumFiles();
	
	virtual long 		GetNumDepthLevelsInFile();	// eventually get rid of this
	//virtual DepthValuesSetH 	GetDepthProfileAtPoint(WorldPoint refPoint) {return nil;}
	virtual OSErr 	GetDepthProfileAtPoint(WorldPoint refPoint, long timeIndex, DepthValuesSetH *profilesH) {*profilesH=nil; return 0;}
	virtual OSErr 		CheckAndScanFile(char *errmsg);
	virtual Boolean 	CheckInterval(long &timeDataInterval);
	virtual OSErr	 	SetInterval(char *errmsg);
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	//virtual OSErr		TextRead(char *path,TMap **newMap);
	virtual OSErr		TextRead(char *path,TMap **newMap,char *topFilePath);
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	OSErr 				ScanFileForTimes(char *path,Seconds ***timeH,Boolean setStartTime);
	//OSErr 				ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH);
	OSErr 				ReadInputFileNames(char *fileNamesPath);
	
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
	virtual void		DrawContourScale(Rect r, WorldRect view);
	virtual void		Draw (Rect r, WorldRect view);
	virtual Boolean	DrawingDependsOnTime(void);
	
	virtual long		GetListLength ();
	virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//virtual OSErr 		AddItem (ListItem item);
	virtual OSErr 		SettingsItem (ListItem item);
	virtual OSErr 		DeleteItem (ListItem item);
	
	virtual OSErr 		SettingsDialog();
	
	OSErr					SaveAsNetCDF(char *path,double *lats, double *lons);	// for testing -  may use in CATS eventually
	//OSErr					SaveAsVis5d(char *path,double *lats, double *lons);	// for testing 
	//OSErr					SaveAsVis5d(double endLat, double startLon, double dLat, double dLon);	// for testing 
	
	//OSErr 				SetTimesForVis5d(int *timestamp, int *datestamp);

};


#endif //  __NETCDFMOVER__
