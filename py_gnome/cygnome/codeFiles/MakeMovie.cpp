#include <QuicktimeComponents.h>
#include <Movies.h>
#ifdef IBM
	#include "Windows.h"
	#include "Gestalt.h"
	#include "QTML.h"
	#include <stdio.h>
	#include "Quickdraw.h"
	#include "Files.h"
	////
	#define GetPortPixMap(x) ((x)->portPixMap)
	void printError(char* str);
	//void c2pstr(char *string);// latest quicktime sdk 7.3 has conflict with c2pstr
	void MySpinCursor(void);

#endif

#include "MakeMovie.h"


Boolean 	gMoviesAllowed = FALSE;
Boolean  gHaveCalledInitMovies = FALSE;
void myc2pstr(char *string)// latest quicktime sdk 7.3 has conflict with c2pstr
{
	short i, len;
	
	len = strlen(string);
	if (len > 255) len = 255;
	for (i = len ; i > 0 ; i--)
		string[i] = string[i - 1];
	string[0] = (unsigned char)len;
}

void CleanupMovieStuff(void)
{
	long QTVersion;
	if(gHaveCalledInitMovies && Gestalt(gestaltQuickTime, &QTVersion) == noErr)
	{
		ExitMovies();
#ifdef IBM
		TerminateQTML();
#endif
	}

}

#ifdef MAC
Boolean CanMakeMovie(void)
#else
BOOL CanMakeMovie(void)
#endif
{
	if(!gHaveCalledInitMovies)
	{
		OSErr err = InitMovies();	
	}
	return gMoviesAllowed;
}

long InitMovies (void)
{
	OSErr		err = noErr;
	long		QTVersion;
	Rect		MovieWRect;
	
	gHaveCalledInitMovies = TRUE;
	
	#ifdef IBM
		InitializeQTML(0);  
	#endif 

	err = Gestalt (gestaltQuickTime, &QTVersion);
	if (!err) {
		gMoviesAllowed = TRUE;
		err = EnterMovies ();
		if (err)
		{
			printError("Error initializing QuickTime.  Movie cannot be generated.");
			gMoviesAllowed = FALSE;
		}
	}
	return err;
}


/*static short GetDepthFromFile(char *bitmapPath,long frameNum, Rect *naturalBoundsPtr)
{	// might want to do this if setting the depth to zero and letting the routines decide doesn't work well
   Rect                    myRect;
   OSErr                    err;
   FSSpec                    fspec;
   Str255                    fileName;
   GraphicsImportComponent fileImporter;
   short depth = -1;

   sprintf((char *)fileName,bitmapPath,frameNum);
   c2pstr((char*)fileName);

   err=FSMakeFSSpec(0,0,fileName,&fspec);
   if(err)
   {
       return -1;
   }
   err = GetGraphicsImporterForFile(&fspec, &fileImporter);
   if(err != noErr)
   {
       printError("Could not convert bitmaps to movie.");
       return -1;
   }

   err = GraphicsImportGetNaturalBounds( fileImporter, naturalBoundsPtr );


   ImageDescriptionHandle descHdl = 0;

   err = GraphicsImportGetImageDescription (fileImporter,&descHdl);


   if (descHdl) {
       depth = (**descHdl).depth;
       DisposeHandle((Handle)descHdl);
   }


   CloseComponent(fileImporter);

   return depth;


}*/


static OSErr GetAndDrawBitMapFromFile(char *bitmapPath,long frameNum,GWorldPtr gw)
{
	Rect					myRect, naturalBounds;
	OSErr					err;
	FSSpec					fspec;
	Str255					fileName;
	GraphicsImportComponent fileImporter;

	sprintf((char *)fileName,bitmapPath,frameNum);
	myc2pstr((char*)fileName);// latest quicktime sdk 7.3 has conflict with c2pstr

	err=FSMakeFSSpec(0,0,fileName,&fspec);
	if(err)
	{
		return -1;
	}
	err = GetGraphicsImporterForFile(&fspec, &fileImporter);
	if(err != noErr)
	{
		printError("Could not convert bitmaps to movie.");
		return -1;
	}

	GraphicsImportSetGWorld(fileImporter, (CGrafPtr)gw, NULL);
	GraphicsImportDraw(fileImporter);
	CloseComponent(fileImporter);

	return noErr;
}

/////////////////////////////////////////////////
long PICStoMovie(	char *moviePath,
					char *frameformatStr,short startIndex,short endIndex,
					short frameTop,short frameLeft, short frameBottom, short frameRight)
{
		ComponentInstance	ci = nil;
		PicHandle			thePict = nil;
		ImageDescription	**idh = nil;
		Handle				compressedData = nil;
		Movie				dstMovie = nil;
		Media				dstMedia = nil;
		Track				dstTrack = nil;
		
		GWorldPtr			pictGWorld = nil;
		GDHandle			saveGDevice = nil;

		OSErr				result = noErr;
		SCParams			p;
		Point				where;
		Rect				pictRect, r;
		
		GrafPtr				savePort = 0;
		long				n, intFrameRate, compressedFrameSize;
		ImageSequence		srcSeqID = 0;
		ImageSequence		dstSeqID = 0;
		TimeScale			dstTimeScale;
		
		long				frameIndex;
		short				hstate, dstMovieRefNum = 0;
		//StandardFileReply	reply;
		char	 			myMoviePrompt[255];
		char	 			myMovieFileName[255];
		OSErr				myErr = userCanceledErr;
		FSSpec				movie_fsspec;
		short				vRefNum;
		long				dirID;
		Str255				movieFile;
		char				bitmapPath[256];
		
		strcpy((char*)movieFile,moviePath);
		strcpy(bitmapPath,frameformatStr);
		pictRect.top = frameTop;
		pictRect.left = frameLeft;
		pictRect.bottom = frameBottom;
		pictRect.right = frameRight;

		ci = OpenDefaultComponent(StandardCompressionType, StandardCompressionSubType);
		BailOnNil(ci);
	
		p.flags = scShowMotionSettings;
		p.theCodecType = 'rle ';
		p.theCodec = anyCodec;
		p.spatialQuality = codecNormalQuality;
		p.temporalQuality = codecNormalQuality;
		//p.depth = 8;
		p.depth = 0;	// kMgrChoose, lets routines figure it out 12/17/07
		p.frameRate = (long)4 << 16;
		p.keyFrameRate = 0; // JLM per Larry 7/16/99, was =1;

	
		//SCSetTestImagePictHandle(ci,thePict,nil,0);
		//	Get compression settings from user.  Center dialog on best screen
		//SetPt (&where, -2, -2);
		//result = SCGetCompression(ci,&p,where);

		saveGDevice = GetGDevice();	
#ifdef MAC
		GetPortGrafPtr (&savePort);
#else
		GetPort (&savePort);	// for some reason Windows cannot deal with the GetPortGrafPtr function. Will get back to it later...
#endif	

		result = NewGWorld(&pictGWorld,p.depth,&pictRect,nil,nil,0);
		BailOnError(result);

		//LockPixels(pictGWorld->portPixMap);
		LockPixels(GetPortPixMap(pictGWorld));
		SetGWorld(pictGWorld,nil);
		EraseRect(&pictRect);
		
		myc2pstr((char *)movieFile);	// latest quicktime sdk 7.3 has conflict with c2pstr
		result = FSMakeFSSpec(0,0,movieFile,&movie_fsspec);
		if(result != noErr && result != fnfErr)
		{
			BailOnError(result);
		}

		result = CreateMovieFile(
								&movie_fsspec,
								'TVOD',0,
								createMovieFileDeleteCurFile | createMovieFileDontCreateResFile,
								&dstMovieRefNum,
								&dstMovie);
		BailOnError(result);		
	
		
		dstTrack = NewMovieTrack(dstMovie,(long)(pictRect.right - pictRect.left) << 16,
										  (long)(pictRect.bottom - pictRect.top) << 16,0);
		
	
		intFrameRate = p.frameRate;
		intFrameRate = intFrameRate >> 16;		/* non-fractional portion of fixed data type */
		if (intFrameRate <= 0)					/* to prevent zero-divide */
			intFrameRate = 1;

		n = (p.frameRate + 0x00008000) >> 16;
		dstTimeScale = 60;
		while (n > dstTimeScale)
			dstTimeScale *= 10;
		dstMedia = NewTrackMedia(dstTrack,VideoMediaType,dstTimeScale,0,0);
		result = BeginMediaEdits(dstMedia);
		BailOnError(result);

		idh = (ImageDescription**)_NewHandle(sizeof(ImageDescription));
		
		//result = GetMaxCompressionSize(pictGWorld->portPixMap,&pictRect,p.depth,p.spatialQuality,
					//p.theCodecType,p.theCodec,&compressedFrameSize);
		result = GetMaxCompressionSize(GetPortPixMap(pictGWorld),&pictRect,p.depth,p.spatialQuality,
				p.theCodecType,p.theCodec,&compressedFrameSize);
		BailOnError(result);
		compressedData = _NewHandle(compressedFrameSize);
		BailOnNil(compressedData);
		_HLock(compressedData);
		
		
		//result = CompressSequenceBegin(&srcSeqID,pictGWorld->portPixMap,nil,&pictRect,nil,p.depth,
					//p.theCodecType,p.theCodec,p.spatialQuality,p.temporalQuality,
					//p.keyFrameRate,nil,codecFlagUpdatePrevious,idh);
		result = CompressSequenceBegin(&srcSeqID,GetPortPixMap(pictGWorld),nil,&pictRect,nil,p.depth,
				p.theCodecType,p.theCodec,p.spatialQuality,p.temporalQuality,
				p.keyFrameRate,nil,codecFlagUpdatePrevious,idh);

		BailOnError(result);


		for (frameIndex = startIndex; frameIndex <= endIndex; frameIndex++)
		{
			unsigned char	similarity;
			Boolean			syncFlag;
			TimeValue		duration;
			long			flags;
			
			MySpinCursor();
			SetGWorld(pictGWorld,nil);
			{
				EraseRect (&pictRect);
				GetAndDrawBitMapFromFile(bitmapPath,frameIndex,pictGWorld);

			}
			SetGWorld((CGrafPtr)savePort, saveGDevice);

			flags = codecFlagUpdatePrevious + codecFlagUpdatePreviousComp;
			//result = CompressSequenceFrame(srcSeqID,pictGWorld->portPixMap,&pictRect,flags,
						//StripAddress(*compressedData),&compressedFrameSize,&similarity,nil);
			result = CompressSequenceFrame(srcSeqID,GetPortPixMap(pictGWorld),&pictRect,flags,
					//StripAddress(*compressedData),&compressedFrameSize,&similarity,nil);
					*compressedData,&compressedFrameSize,&similarity,nil);
			BailOnError(result);


			syncFlag = (similarity ? mediaSampleNotSync : 0);
			duration = dstTimeScale / intFrameRate;
			result = AddMediaSample(dstMedia,compressedData,0,compressedFrameSize,duration,
						(SampleDescriptionHandle)idh,1,syncFlag,nil);
			BailOnError(result);
		}
		
		_HUnlock((Handle)compressedData);


		{
	
		result = EndMediaEdits(dstMedia);
		BailOnError(result);
		InsertMediaIntoTrack(dstTrack,0,0,GetMediaDuration(dstMedia),1L<<16);
		result = GetMoviesError();
		BailOnError(result);

		// add movie to data fork 
		// also need to enable createMovieFileDontCreateResFile flag in CreateMovie call
		short resID = movieInDataForkResID; 
		result = AddMovieResource(dstMovie,dstMovieRefNum,&resID,NULL);
		BailOnError(result);
		dstMedia = nil;
		}

	error:
		/***************************************
		 *
		 *	Deallocate everything that was created.
		 *
		 ***************************************/

		if (dstMovieRefNum) {
			CloseMovieFile(dstMovieRefNum);
			dstMovieRefNum = 0;
		}

		if (dstMovie) {
			DisposeMovie(dstMovie);
			dstMovie = nil;
		}

		if (srcSeqID) {
			CDSequenceEnd(srcSeqID);
			srcSeqID = 0;
		}
		
		if (dstSeqID) {
			CDSequenceEnd(dstSeqID);
			dstSeqID = 0;
		}	

		if (compressedData) {
			DisposeHandle(compressedData);
			compressedData = nil;
		}
		
		if (idh) {
			DisposeHandle((Handle)idh);
			idh = nil;
		}
		
		if (pictGWorld) {
			//UnlockPixels(pictGWorld->portPixMap);
			UnlockPixels(GetPortPixMap(pictGWorld));
			DisposeGWorld(pictGWorld);
			pictGWorld = nil;
		}
		
		if (ci)
			CloseComponent(ci);
			
#ifdef MAC
		if (savePort) SetPort(savePort);	// for some reason the IBM doesn't like this
#endif		
		return (result);

}


/////////////////////////////////////////////////
