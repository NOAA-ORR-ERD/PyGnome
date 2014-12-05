

#ifndef		__OVERLAYS__
#define		__OVERLAYS__

Boolean IsOverlayFile(char* path);
OSErr AddOverlayFromFile(char *path);
OSErr AddOverlayDialog(void);


/////////////////////////////////////////////////
/////////////////////////////////////////////////


// a base class for Overlay objects
class TOverlay : public TClassID  
{
	public:
		Boolean 			bShowOverlay;
		char				fFilePath[kMaxNameLen]; // path to the file the overlay was read from
		RGBColor			fColor;
		WorldRect			fBounds;

	public:
						TOverlay ();
		virtual			~TOverlay () { Dispose (); }
		virtual void	Dispose ();
		virtual OSErr	CheckAndPassOnMessage(TModelMessage *message);



		virtual ClassID 	GetClassID () { return TYPE_OVERLAY; }
		virtual Boolean		IAm(ClassID id) { return(id==TYPE_OVERLAY); }

		virtual void GetFileName(char *name); 		
		virtual void SetClassNameToFileName();  // for command files

		// I/O methods
		virtual OSErr ReadFromFile(char *path);
		// we are not currently going to save overlay objects to the save file

		// list display methods: base class functionality
		virtual void Draw (Rect r, WorldRect view);
		virtual long		GetListLength ();
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
		virtual OSErr		UpItem(ListItem item);
		virtual OSErr		DownItem(ListItem item);
		virtual OSErr 		SettingsItem (ListItem item);
		virtual OSErr 		DeleteItem (ListItem item);
};

/////////////////////////
// a NESDIS Overlay objects
/////////////////////////


typedef struct {
	double lat;
	double lng;
	int polyNum;
	//
	int flag; // the flag from the file, not sure what 0 and 1 mean (does 1 mean has interior rings?)
	//
	// sometimes the NESDIS file out of ARC has a line that says "InteriorRing"
	int interiorRingFlag; // set to zero for the outer rings, and increasing positive numbers for interior rings

} NesdisPoint;

typedef struct {
	long numAllocated;
	long numFilledIn;
	NesdisPoint *pts;
} NesdisPointsInfo;

class TNesdisOverlay : public TOverlay  
{
	public:
		NesdisPointsInfo fNesdisPoints; // note this is allocated with malloc and so needs to  use free

#ifdef IBM
		HDIB			fBitmap;
#else
		BitMap			fBitmap; 
#endif		

	public:
						TNesdisOverlay ();
		virtual			~TNesdisOverlay () { Dispose (); }
		virtual void	Dispose ();


		virtual ClassID 	GetClassID () { return TYPE_NESDIS_OVERLAY; }
		virtual Boolean		IAm(ClassID id) { return(id==TYPE_NESDIS_OVERLAY); }
		
		// override base class functionality
		virtual OSErr	ReadFromFile(char *path);
		virtual OSErr	ReadFromBNAFile(char *path);
		virtual OSErr	ReadFromShapeFile(char *path);
		virtual void	Draw (Rect r, WorldRect view);

		//
		void		DisposeNesdisPoints(void);
		OSErr		AllocateNesdisPoints(long numToAllocate);
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		
		
		virtual OSErr SettingsItem (ListItem item);	 // an easy way to get this implemented	
		OSErr MakeBitmap(void);
		void DrawBitmap(Rect r, WorldRect view);
		void AddSprayedLEsToLEList(void);
};

//////////////////////////
// a BUOY Overlay objects
//////////////////////////
typedef struct {
	double lat;
	double lng;
	long buoyNum;
} BuoyPoint;

typedef struct {
	long numAllocated;
	long numFilledIn;
	BuoyPoint *pts;
} BuoyPointsInfo;

class TBuoyOverlay : public TOverlay  
{
	public:
		BuoyPointsInfo fBuoyPoints; // note this is allocated with malloc and so needs to  use free

		WorldRect fBounds;

	public:
						TBuoyOverlay ();
		virtual			~TBuoyOverlay () { Dispose (); }
		virtual void	Dispose ();


		virtual ClassID 	GetClassID () { return TYPE_BUOY_OVERLAY; }
		virtual Boolean		IAm(ClassID id) { return(id==TYPE_BUOY_OVERLAY); }
		
		// override base class functionality
		virtual OSErr	ReadFromFile(char *path);
		virtual void	Draw (Rect r, WorldRect view);

		virtual	long	NumBuoys ();
		//
		virtual void	DisposeBuoyPoints(void);
		virtual OSErr	AllocateBuoyPoints(long numToAllocate);

		virtual long		GetListLength ();
		virtual ListItem	GetNthListItem (long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
		virtual OSErr 		SettingsItem (ListItem item);

};

//////////////////////////

class TBpBuoyOverlay : public TBuoyOverlay  
{
	public:

	public:
						TBpBuoyOverlay ();
		virtual			~TBpBuoyOverlay () { Dispose (); }


		virtual ClassID 	GetClassID () { return TYPE_BP_BUOY_OVERLAY; }
		virtual Boolean		IAm(ClassID id) { return(id==TYPE_BP_BUOY_OVERLAY); }
		
		// override base class functionality
		virtual OSErr	ReadFromFile(char *path);

};

class TSLDMBBuoyOverlay : public TBuoyOverlay  
{
	public:

	public:
						TSLDMBBuoyOverlay ();
		virtual			~TSLDMBBuoyOverlay () { Dispose (); }


		virtual ClassID 	GetClassID () { return TYPE_SLDMB_BUOY_OVERLAY; }
		virtual Boolean		IAm(ClassID id) { return(id==TYPE_SLDMB_BUOY_OVERLAY); }
		
		// override base class functionality
		virtual OSErr	ReadFromFile(char *path);

};


////////////////
// OVERFLIGHTS
////////////////


typedef struct {
	double lat;
	double lng;
	int trackNum; 
} TrackPoint;


typedef struct {
	double lat;
	double lng;
	int wayPtNum; 
} WayPoint;

typedef struct {
	long numAllocated;
	long numFilledIn;
	TrackPoint *pts;
} TrackPointsInfo;

class TOverflightOverlay : public TOverlay  
{
	public:
		TrackPointsInfo fTrackPoints; // note this is allocated with malloc and so needs to  use free

	public:
						TOverflightOverlay ();
		virtual			~TOverflightOverlay () { Dispose (); }
		virtual void	Dispose ();


		virtual ClassID 	GetClassID () { return TYPE_OVERFLIGHT_OVERLAY; }
		virtual Boolean		IAm(ClassID id) { return(id==TYPE_OVERFLIGHT_OVERLAY); }
		
		// override base class functionality
		virtual OSErr	ReadFromFile(char *path);
		virtual void	Draw (Rect r, WorldRect view);

		//
		void		DisposeTrackPoints(void);
		OSErr		AllocateTrackPoints(long numToAllocate);
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		
};



#endif
