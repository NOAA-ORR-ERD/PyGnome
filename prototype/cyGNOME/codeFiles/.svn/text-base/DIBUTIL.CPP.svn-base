
#include "CROSS.h"

// #define DRAWTODESKTOP

#ifdef MAC

//////////////////////////////// MAC CODE //////////////////////////////////////

#ifdef MPW
//#include <QDOffscreen.h>
#pragma SEGMENT UTILS3
#endif
#ifdef CODEWARRIOR
#include <QDOffscreen.h>
#endif

PixMapHandle PixMapCreate (WindowPtr CWPtr, Rect *theBounds, Handle *PixMapHdlPtr, OSErr *ErrCodePtr)
{
	PixMapHandle	thePixels = nil;
	CTabHandle		theCTable = nil;
	short			targetDepth, i;
	OSErr			ErrCode = 0;

	targetDepth = GetScrDepth(); // get current screen's bit depth

	*PixMapHdlPtr = nil;

	if ((theBounds == NULL) || (targetDepth <= 0))
		ErrCode = 1;
	
	thePixels = NewPixMap ();
	if (thePixels == NULL)
		ErrCode = memFullErr;
	
	if (!ErrCode)
	{
		/* allocate memory for pixel map */
		(**thePixels).bounds = *theBounds;
		(**thePixels).pixelSize = targetDepth;
		*PixMapHdlPtr = AllocPixMapMem (thePixels);
		if (*PixMapHdlPtr == nil)
			ErrCode = memFullErr;
	}
		
	if (!ErrCode)
	{
		/* get a new copy of the current GDevice's CTable */
		/* give the new seed a non-GD-type CTFlag and re-index */
		if (targetDepth > 8)
		{
			theCTable = (CTabHandle) _NewHandleClear (8L);
			if (theCTable != nil)
				(**theCTable).ctSeed = targetDepth;
			else
				ErrCode = memFullErr;
		}
		else
		{
//#if MACB4CARBON
//#ifdef MPW
//			theCTable = (**(CWPtr -> portPixMap)).pmTable;
//#else
			theCTable = (**GetPortPixMap(GetWindowPort(CWPtr))).pmTable;
//#endif
			ErrCode = _HandToHand((Handle *) &theCTable);
			if (ErrCode)
				theCTable = nil;

			if (!ErrCode)
			{
				/* perform fixup for non-GDevice based Colortables */
				(**theCTable).ctFlags = 0x0000;
				for (i = 0; i <= (**theCTable).ctSize; i++)
					(**theCTable).ctTable [i].value = i;
			
				/* dispose CTable stub left by NewPixMap () and install new CTable */
				DisposeHandle ((Handle) (**thePixels).pmTable);
				(**thePixels).pmTable = theCTable;

				if (targetDepth <= 8)
				{
					(**thePixels).pixelType = 0;
					(**thePixels).cmpCount = 1;
					(**thePixels).cmpSize = targetDepth;
				}
				else
				{
					(**thePixels).pixelType = 16;
					(**thePixels).cmpCount = 3;
					(**thePixels).cmpSize = ((targetDepth == 32) ? 8 : 5);
				}
				
				(**thePixels).packType = 0;
				(**thePixels).packSize = 0;
			#if MACB4CARBON
				(**thePixels).planeBytes = 0L;
			#endif
				
				/* at least until a new version comes out */
				(**thePixels).pmVersion = 0;
			}
		}
	}
	
	if (ErrCode)		/* clean up in case of an error */
	{
		if (thePixels != nil)
		{
			/* dispose of storage allocated by NewPixMap () */
			DisposePixMap (thePixels);
			thePixels = nil;
		}
		
		if (*PixMapHdlPtr != nil)
		{
			_HUnlock (*PixMapHdlPtr);		/* was locked by Alloc-Pix-Map-Mem */
			DisposeHandle (*PixMapHdlPtr);
			*PixMapHdlPtr = nil;
		}
		
		if (theCTable != nil)
			DisposeHandle ((Handle) theCTable);
	}
		
	*ErrCodePtr = ErrCode;
	
	return (thePixels);
}

Handle AllocPixMapMem (PixMapHandle thePixels)
/* this subroutine allocates a handle for the memory needed for the pixel map.  If the allocation is
	successful, it locks the handle and assigns (*theMemHdl) to the pixel map's base address field */
{
	Rect	theBounds;
	long	imageWidth, imageHeight;
	short	theRowBytes;
	Handle	PixMapMemHdl = nil;
	Size	ImageSize;
	
	if (thePixels != NULL)
	{
		theBounds = (**thePixels).bounds;
		
		imageWidth = theBounds.right - theBounds.left + 1;
		imageHeight = theBounds.bottom - theBounds.top + 1;
		
		theRowBytes = ((**thePixels).pixelSize * ((imageWidth + 15) / 16)) * 2;
		ImageSize = theRowBytes * imageHeight;
		
		PixMapMemHdl = MyNewHandleTemp (ImageSize);
		if (PixMapMemHdl != nil)
		{
			/* lock handle and get pointer to it */
			HLockHi (PixMapMemHdl);

			/* set pixel map's base address and rowbytes fields */
			(**thePixels).baseAddr = *PixMapMemHdl;
			(**thePixels).rowBytes = theRowBytes | 0x8000;
		}
		else
		{
			(**thePixels).baseAddr = NULL;
			(**thePixels).rowBytes = 0;
		}
	}
	
	return (PixMapMemHdl);
}

short GetScrDepth()
{
	GDHandle		gdh;
	short			Depth;
	PixMapHandle	pmh;
	
	gdh = GetMainDevice();
	
	pmh = (**gdh).gdPMap;
	Depth = (**pmh).pixelSize;

	return (Depth);
}

PixMapHandle GetPortPix()
/* returns the pix-map of the current c-port */
{
	CGrafPtr	currPort;
	
	GetPortGrafPtr ((GrafPtr*) &currPort);
	
//#ifndef CODEWARRIOR 
	return GetPortPixMap(currPort);
//#else
	//return (currPort -> portPixMap);
//#endif
}

CGrafPtr MakeNewGWorld (short worldDepth, Rect *theRect)
{
	OSErr		err = noErr;
	short		worldFlags = 0;
	CGrafPtr 	theGWorld = nil;
	CTabHandle	theCTable = nil;

	if (worldDepth == 0)
		worldDepth = GetScrDepth ();	// gets depth of main g-device

	err = NewGWorld (&theGWorld, worldDepth, theRect, theCTable, nil, worldFlags);
	if (err)
		theGWorld = nil;
		
	return theGWorld;
}

void SaveGWorld (MyGWorldRecPtr gWorldRecPtr)
{
	gWorldRecPtr -> savePortPalHdl = nil;

	GetPort(&gWorldRecPtr -> savePort);
	
	#ifdef MAC
		// save the source window's palette handle
		gWorldRecPtr -> savePortPalHdl = GetPalette (GetWindowFromPort(gWorldRecPtr -> savePort));
	#endif
	
	GetGWorld ((CGrafPtr*) &(gWorldRecPtr -> savePort), &(gWorldRecPtr -> saveGDevice));

	return;
}

void BeginGWorldDraw (CGrafPtr theGWorld, Boolean bCopyScreen, Boolean bErase)
// prepares the given GWorld for drawing into, and returns its port
{
	PaletteHandle	thePaletteHandle = nil;
	Rect			theRect;
	CGrafPtr		window;
	
	GetPortGrafPtr ((GrafPtr*) &window);
	#ifdef MAC
		thePaletteHandle = GetPalette (GetWindowFromPort(window));
	#endif

	if (theGWorld != nil)
	{
		//LockPixels (theGWorld -> portPixMap);
		LockPixels (GetPortPixMap(theGWorld));
		SetGWorld (theGWorld, nil);
		//theRect = theGWorld -> portRect;
		theRect = GetWindowPortRect(GetWindowFromPort(theGWorld));

		// copy the window's palette to be used for new gworld
		if (thePaletteHandle != nil)
			SetPalette (GetWindowFromPort(theGWorld), thePaletteHandle, false);

		// copy the screen image into the new gworld if requested */
		if (bCopyScreen)
		{
			//_HLock ((Handle) window -> portPixMap);
			_HLock ((Handle) GetPortPixMap(window));

			//CopyBits ((BitMap *) *(window -> portPixMap), (BitMap *) *(theGWorld -> portPixMap),
					  // &theRect, &theRect, srcCopy, nil);
			CopyBits ((BitMap *) *(GetPortPixMap(window)), (BitMap *) *(GetPortPixMap(theGWorld)),
					   &theRect, &theRect, srcCopy, nil);

			//_HUnlock ((Handle) window -> portPixMap);
			_HUnlock ((Handle) GetPortPixMap(window));
		}
		else if (bErase)
			EraseRect (&theRect);
	}

	return;
}

void EndGWorldDraw (CGrafPtr theGWorld, MyGWorldRecPtr gWorldRecPtr)
// used to set the port BACK to original after drawing into private GWorld
{
	if (theGWorld)
	{
		//UnlockPixels (theGWorld -> portPixMap);
		UnlockPixels (GetPortPixMap(theGWorld));

		//SetGWorld ((CGrafPtr) gWorldRecPtr -> savePort, gWorldRecPtr -> saveGDevice);
		SetGWorld ((CGrafPtr)(gWorldRecPtr -> savePort), gWorldRecPtr -> saveGDevice);
		
		if (gWorldRecPtr -> savePortPalHdl)
			SetPalette (GetWindowFromPort(gWorldRecPtr -> savePort), gWorldRecPtr -> savePortPalHdl, false);
	}
	
	return;
}

void KillGWorld (CGrafPtr theGWorld)
{
	if (theGWorld != nil)
	{
		//UnlockPixels (theGWorld -> portPixMap);
		UnlockPixels (GetPortPixMap(theGWorld));
		DisposeGWorld (theGWorld);
	}
	
	return;
}

void CopyWorldToScreen (CGrafPtr theGWorld, Rect theRect, short transferMode)
// prepares the given GWorld for drawing into, and returns its port
{
	if (theGWorld != nil)
	{
		//WindowPtr	dumpWPtr;
		GrafPtr	dumpWPtr;
		CGrafPtr	cWPtr;
		
		GetPortGrafPtr (&dumpWPtr);		
		cWPtr = (CGrafPtr) dumpWPtr;

		//_HLock ((Handle) cWPtr -> portPixMap);
		_HLock ((Handle) GetPortPixMap(cWPtr));
		
		//CopyBits ((BitMap *) *(theGWorld -> portPixMap),
				  //(BitMap *) *(cWPtr -> portPixMap),
				  // &theRect, &theRect, transferMode, nil);
		CopyBits ((BitMap *) *(GetPortPixMap(theGWorld)),
				  (BitMap *) *(GetPortPixMap(cWPtr)),
				   &theRect, &theRect, transferMode, nil);

		//_HUnlock ((Handle) cWPtr -> portPixMap);
		_HUnlock ((Handle) GetPortPixMap(cWPtr));
	}

	return;
}

OSErr MyNewGWorld(WindowPtr SourceWPtr, short WDepth, Rect *WorldRectPtr,
				    MyGWorldRecPtr GWorldRecPtr)
{
	OSErr		ErrCode = 0;
	CTabHandle	theCTable = nil;
	short		worldFlags = 0;

	/* save the source window's palette handle if available */
	GWorldRecPtr -> sourcePort = (CGrafPtr) SourceWPtr;
	GWorldRecPtr -> savePortPalHdl = GetPalette (SourceWPtr);

	/* save the supplied gworld rectangle */
	GWorldRecPtr -> theGWorldRect = *WorldRectPtr;

	if (WDepth == 0)
		WDepth = GetScrDepth ();	/* gets depth of main g-device */

	ErrCode = NewGWorld (&(GWorldRecPtr -> theGWorld), WDepth, WorldRectPtr, theCTable, nil, worldFlags);
	if (ErrCode)
		GWorldRecPtr -> theGWorld = nil;

	return (ErrCode);
}

GrafPtr MySetGWorld(MyGWorldRecPtr GWorldRecPtr, Boolean bCopyScreen, Boolean bErase)
// prepares the given GWorld for drawing into, and returns its port
{
	GrafPtr	newPort;

	GetPortGrafPtr (&newPort);		/* in case of error, the current port will be returned instead */

	if (GWorldRecPtr -> theGWorld != nil)
	{
		/* save the current gdevice AND the current port within our GWorld record */
		GetPortGrafPtr (&(GWorldRecPtr -> savePort));
		GWorldRecPtr -> saveGDevice = GetGDevice ();

		//LockPixels (GWorldRecPtr -> theGWorld -> portPixMap);
		LockPixels (GetPortPixMap(GWorldRecPtr -> theGWorld) );
		SetGWorld (GWorldRecPtr -> theGWorld, nil);
		
		/* reset the new gworld's palette to be the same as the current port */
		if (GWorldRecPtr -> savePortPalHdl != nil)
			SetPalette( GetWindowFromPort(GWorldRecPtr -> theGWorld), GWorldRecPtr -> savePortPalHdl, false);
		
		if (bErase)
			EraseRect (&(GWorldRecPtr -> theGWorldRect));
		
		/* copy the screen image into the new gworld if requested */
		if (bCopyScreen)
		{
			CGrafPtr	SourceWPtr;
			
			SourceWPtr = GWorldRecPtr -> sourcePort;

			//_HLock ((Handle) SourceWPtr -> portPixMap);
			_HLock ((Handle) GetPortPixMap(SourceWPtr));
			
			//CopyBits ((BitMap *) *(SourceWPtr -> portPixMap),
					//  (BitMap *) *(GWorldRecPtr -> theGWorld -> portPixMap),
					 //  &GWorldRecPtr -> theGWorldRect, &GWorldRecPtr -> theGWorldRect,
					//   srcCopy, nil);
			CopyBits ((BitMap *) *(GetPortPixMap(SourceWPtr)),
					  (BitMap *) *(GetPortPixMap(GWorldRecPtr -> theGWorld)),
					   &GWorldRecPtr -> theGWorldRect, &GWorldRecPtr -> theGWorldRect,
					   srcCopy, nil);
		
			//_HUnlock ((Handle) SourceWPtr -> portPixMap);
			_HUnlock ((Handle) GetPortPixMap(SourceWPtr));
		}
		
		GetPortGrafPtr (&newPort);				/* obtain current port to be returned by this function */
	}

	return (newPort);
}

void MyRestoreGWorld (MyGWorldRecPtr GWorldRecPtr)
/* restores original GWorld / GDevice, after having drawn into the new one */
{
	if (GWorldRecPtr -> theGWorld != nil)
	{
		SetGWorld ((CGrafPtr) GWorldRecPtr -> savePort, GWorldRecPtr -> saveGDevice);
		//UnlockPixels (GWorldRecPtr -> theGWorld -> portPixMap);
		UnlockPixels (GetPortPixMap(GWorldRecPtr -> theGWorld));
	}
	
	return;
}

void MyBlitGWorld(MyGWorldRecPtr GWorldRecPtr, WindowPtr DumpWPtr, short transferMode)
/* bit-copies the contents of the given GWorld into the given window */
{
	if (GWorldRecPtr -> theGWorld != nil)
	{
		if (DumpWPtr != nil)
		{
			CGrafPtr	CWPtr;
			
			CWPtr = GetWindowPort(DumpWPtr);

			//_HLock ((Handle) CWPtr -> portPixMap);
			_HLock ((Handle) GetPortPixMap(CWPtr));
			
			//CopyBits ((BitMap *) *(GWorldRecPtr -> theGWorld -> portPixMap),
					 // (BitMap *) *(CWPtr -> portPixMap),
					 //  &GWorldRecPtr -> theGWorldRect, &GWorldRecPtr -> theGWorldRect,
					 //  transferMode, nil);
			CopyBits ((BitMap *) *(GetPortPixMap(GWorldRecPtr -> theGWorld)),
					  (BitMap *) *(GetPortPixMap(CWPtr)),
					   &GWorldRecPtr -> theGWorldRect, &GWorldRecPtr -> theGWorldRect,
					   transferMode, nil);

			//_HUnlock ((Handle) CWPtr -> portPixMap);
			_HUnlock ((Handle) GetPortPixMap(CWPtr));
		}
	}

	return;
}

void MyKillGWorld(MyGWorldRecPtr GWorldRecPtr)
{
	if (GWorldRecPtr -> theGWorld != nil)
	{
		//UnlockPixels (GWorldRecPtr -> theGWorld -> portPixMap);	/* locked when it was created */
		UnlockPixels (GetPortPixMap(GWorldRecPtr -> theGWorld));	/* locked when it was created */
		DisposeGWorld (GWorldRecPtr -> theGWorld);
		GWorldRecPtr -> theGWorld = nil;
	}
	
	return;
}

void BitMapDispose(PixMapHandle bits, WindowPtr w)
{
	// _DisposePtr((**bits).baseAddr);
	//DisposeHandle((Handle)(**bits).baseAddr); //JLM
	if((**bits).baseAddr) //JLM 7/15/98, this contains a locked handle allocated by in AllocPixMapMem()
	{
		Handle h = RecoverHandle((**bits).baseAddr);
		if(h)
		{
			_HUnlock(h);
			DisposeHandle(h); h = 0;
			(**bits).baseAddr = nil; 
		}
	}
	if (WindowHasColor(w)) DisposeHandle((Handle)(**bits).pmTable);
	DisposeHandle((Handle)bits);
}

void CopyBitsFromWindow(WindowPtr w, PixMapHandle bits, Rect srcRect, Rect dstRect)
{
	//Ptr saveHandle = (**bits).baseAddr;
	
//	Debug ("Inside  CopyBitsFromWindow\n");

//	_HLock((Handle)saveHandle);
//	(**bits).baseAddr = *(Handle)(**bits).baseAddr;
//	CopyBits(&(w->portBits), (BitMap *)*bits, &srcRect, &dstRect, srcCopy, nil);
//	(**bits).baseAddr = saveHandle;
//	_HUnlock((Handle)saveHandle);

//	Debug ("window = %d, bits = %ld, srcRect = ", w, bits);
//	printfRect (&srcRect);
//	printfRect (&dstRect);

			//_HLock ((Handle) ((CGrafPtr) w) -> portPixMap);
			_HLock ((Handle) (GetPortPixMap(GetWindowPort(w))));
			//CopyBits ((BitMap *) *(((CGrafPtr) w) -> portPixMap),
					 // (BitMap *) *bits,
					  // &srcRect, &dstRect,
					  // srcCopy, nil);
			CopyBits ((BitMap *) *(GetPortPixMap(GetWindowPort(w))),
					  (BitMap *) *bits,
					   &srcRect, &dstRect,
					   srcCopy, nil);

			//_HUnlock ((Handle) ((CGrafPtr) w) -> portPixMap);
			_HUnlock ((Handle) (GetPortPixMap(GetWindowPort(w))));

/*			EraseRect (&srcRect);
					  
			CopyBits ((BitMap *) *bits,
					  (BitMap *) *(((CGrafPtr) w) -> portPixMap),
					   &dstRect, &srcRect,
					   srcCopy, nil);

	SysBeep (5);
*/
//	Debug ("Exiting CopyBitsFromWindow\n");
}

void CopyBitsFromGWorld(CGrafPtr sourceGWorld, PixMapHandle bits, Rect srcRect, Rect dstRect)
{
	/*_HLock ((Handle) sourceGWorld -> portPixMap);
	CopyBits ((BitMap *) *((sourceGWorld) -> portPixMap),
			  (BitMap *) *bits,
			   &srcRect, &dstRect,
			   srcCopy, nil);

	_HUnlock ((Handle) sourceGWorld -> portPixMap);*/
	_HLock ((Handle) GetPortPixMap(sourceGWorld));
	CopyBits ((BitMap *) *(GetPortPixMap(sourceGWorld)),
			  (BitMap *) *bits,
			   &srcRect, &dstRect,
			   srcCopy, nil);

	_HUnlock ((Handle) GetPortPixMap(sourceGWorld));
}

void CopyBitsToWindow(PixMapHandle bits, WindowPtr w, Rect srcRect, Rect dstRect)
{
	//Ptr saveHandle = (**bits).baseAddr;
	
//	Debug ("Inside  CopyBitsToWindow\ndestRect = ");

//	printfRect (&dstRect);
//	Debug ("window = %ld\n", w);
//	Debug ("bits = %d\n", bits);

//	_HLock((Handle)saveHandle);
//	(**bits).baseAddr = *(Handle)(**bits).baseAddr;
//	CopyBits((BitMap *)*bits, &(w->portBits), &srcRect, &dstRect, srcCopy, nil);
//	(**bits).baseAddr = saveHandle;
//	_HUnlock((Handle)saveHandle);

			/*_HLock ((Handle) ((CGrafPtr) w) -> portPixMap);
			CopyBits ((BitMap *) *bits,
					  (BitMap *) *(((CGrafPtr) w) -> portPixMap),
					   &srcRect, &dstRect,
					   srcCopy, nil);
			_HUnlock ((Handle) ((CGrafPtr) w) -> portPixMap);*/
			_HLock ((Handle) (GetPortPixMap(GetWindowPort(w))));
			CopyBits ((BitMap *) *bits,
					  (BitMap *) *(GetPortPixMap(GetWindowPort(w))),
					   &srcRect, &dstRect,
					   srcCopy, nil);
			_HUnlock ((Handle) (GetPortPixMap(GetWindowPort(w))));

//	Debug ("Exiting CopyBitsToWindow\n");
}

void CopyBitsToGWorld(PixMapHandle bits, CGrafPtr destGWorld, Rect srcRect, Rect dstRect)
{
	//Ptr saveHandle = (**bits).baseAddr;
	
//	Debug ("Inside  CopyBitsToWindow\ndestRect = ");

//	printfRect (&dstRect);
//	Debug ("window = %ld\n", w);
//	Debug ("bits = %d\n", bits);

//	_HLock((Handle)saveHandle);
//	(**bits).baseAddr = *(Handle)(**bits).baseAddr;
//	CopyBits((BitMap *)*bits, &(w->portBits), &srcRect, &dstRect, srcCopy, nil);
//	(**bits).baseAddr = saveHandle;
//	_HUnlock((Handle)saveHandle);

			/*_HLock ((Handle) destGWorld -> portPixMap);
			CopyBits ((BitMap *) *bits,
					  (BitMap *) *((destGWorld) -> portPixMap),
					   &srcRect, &dstRect,
					   srcCopy, nil);
			_HUnlock ((Handle) destGWorld -> portPixMap);*/

			_HLock ((Handle) GetPortPixMap(destGWorld));
			CopyBits ((BitMap *) *bits,
					  (BitMap *) *(GetPortPixMap(destGWorld)),
					   &srcRect, &dstRect,
					   srcCopy, nil);
			_HUnlock ((Handle) GetPortPixMap(destGWorld));

//	Debug ("Exiting CopyBitsToWindow\n");
}

///////////////////////////////////////////////////////////////////////

OSErr OSNewPortPix(PortPixRecordP pixRec, WindowPtr w, Rect r)
{
	OSErr err;
	Handle pixelsHandle;
	GrafPtr savePort;
	
	#if TARGET_API_MAC_CARBON
		pixRec->newPort = CreateNewPort();
	#else
		pixRec->newPort = (CGrafPtr)_NewPtr(sizeof(CGrafPort));
		if (!pixRec->newPort) return -1;
		OpenCPort(pixRec->newPort);
	#endif
	
	//RectRgn(pixRec->newPort->visRgn, &r);
	
	pixRec->newPixMap = PixMapCreate(w, &r, &pixelsHandle, &err);
	
	if (err) 
	{ 
		#if TARGET_API_MAC_CARBON
			if(pixRec->newPort)  {DisposePort(pixRec->newPort);pixRec->newPort = 0;}
		#else
			if(pixRec->newPort) {CloseCPort(pixRec->newPort); _DisposePtr((Ptr)pixRec->newPort); pixRec->newPort = 0;}
		#endif
		return -1; 
	}
	
	// _HUnlock(pixelsHandle); // locked by AllocPixMapMem()
	pixRec->pixelsHandle = pixelsHandle;
	pixRec->r = r;
	
	GetPortGrafPtr(&savePort);
	SetPortGrafPort((GrafPtr)pixRec->newPort);
	pixRec->oldPixMap = GetPortPix();
	SetPortPix(pixRec->newPixMap);
	ClipRect(&r);
	SetPortGrafPort(savePort);
	
	return 0;
}

// I don't think this is used
/*PicHandle GetScreenPicture (WindowPtr wPtr, Rect *picRect)
{
	GrafPtr			SavePort, WMgrPort;
	CWindowPtr		cWPtr;
	PicHandle		thePicture = nil;
	Rect			WindowDrawRect, GblWDrawRect;
	RgnHandle		SaveClip = nil;
	
	cWPtr = (CWindowPtr) wPtr;			// convert window pointer to color window type
	
	GetPortGrafPtr (&SavePort);
	SetPortWindowPort (wPtr);

	SaveClip = NewRgn ();
	GetClip (SaveClip);				// preserve current clip region

	// if map window's content rect is ouside the screen bounds, clip it to the screen 
	WindowDrawRect = *picRect;
	GblWDrawRect = WindowDrawRect;
	LocalToGlobalRect (&GblWDrawRect);		// convert draw rect to global coordinates
	GetWMgrPort (&WMgrPort);
	if (!RectInRect (&GblWDrawRect, &(WMgrPort -> portRect)))
	{
		// find the intersection rect between map window and screen
		SectRect (&GblWDrawRect, &(WMgrPort -> portRect), &WindowDrawRect);
		GlobalToLocalRect (&WindowDrawRect);
	}
		
	ClipRect (&WindowDrawRect);	
	thePicture = OpenPicture (&WindowDrawRect);

	if (thePicture != nil)
	{
		//_HLock ((Handle) cWPtr -> portPixMap);

		//CopyBits ((BitMap *) *(cWPtr -> portPixMap), (BitMap *) *(cWPtr -> portPixMap),
				  // &WindowDrawRect, &WindowDrawRect, srcCopy, nil);
	
		//_HUnlock ((Handle) cWPtr -> portPixMap);
		_HLock ((Handle) GetPortPixMap((CGrafPtr)cWPtr));

		CopyBits ((BitMap *) *(GetPortPixMap((CGrafPtr)cWPtr)), (BitMap *) *(GetPortPixMap((CGrafPtr)cWPtr)),
				   &WindowDrawRect, &WindowDrawRect, srcCopy, nil);
	
		_HUnlock ((Handle) GetPortPixMap((CGrafPtr)cWPtr));

		ClosePicture ();
	}
	
	SetClip (SaveClip);				// restore original clip region
	DisposeRgn (SaveClip);			// get rid of saved clip region
	SetPortGrafPort (SavePort);

	return (thePicture);
}*/

void OSSetPort(PortPixRecordP pixRec)
{
	// GetPort(&pixRec->w);
	// _HLock(pixRec->pixelsHandle);
#ifdef DRAWTODESKTOP
	(**pixRec->newPixMap).baseAddr = qd.screenBits.baseAddr;
#else
	(**pixRec->newPixMap).baseAddr = *pixRec->pixelsHandle;
#endif
	SetPortGrafPort((GrafPtr)pixRec->newPort);
}

void OSResetPort(PortPixRecordP pixRec)
{
#pragma unused(pixRec)
	
	// SetPort(pixRec->w);
	// _HUnlock(pixRec->pixelsHandle);
}

void OSCopyToWindow(PortPixRecordP pixRec, WindowPtr w, Rect rSrc, Rect rDst, short mode)
{
	GrafPtr savePort;
	
	GetPortGrafPtr(&savePort);
	SetPortWindowPort(w); // (GrafPtr)pixRec->newPort
	
	_HLock((Handle)pixRec->newPixMap);
	// _HLock(pixRec->pixelsHandle);
#ifdef DRAWTODESKTOP
	(**pixRec->newPixMap).baseAddr = qd.screenBits.baseAddr;
#else
	(**pixRec->newPixMap).baseAddr = *pixRec->pixelsHandle;
#endif
	//CopyBits((BitMap *)*pixRec->newPixMap, &w->portBits, &rSrc, &rDst, mode, nil);
	CopyBits((BitMap *)*pixRec->newPixMap, GetPortBitMapForCopyBits(GetWindowPort(w)), &rSrc, &rDst, mode, nil);
	// _HUnlock(pixRec->pixelsHandle);
	_HUnlock((Handle)pixRec->newPixMap);
	
	SetPortGrafPort(savePort);
}

void OSDisposePortPix(PortPixRecordP pixRec)
{
	GrafPtr savePort;
	
	GetPortGrafPtr(&savePort);
	SetPortGrafPort((GrafPtr)pixRec->newPort);
	SetPortPix(pixRec->oldPixMap);
	SetPortGrafPort(savePort);
	
	#if TARGET_API_MAC_CARBON
		if(pixRec->newPort)  {DisposePort(pixRec->newPort);pixRec->newPort = 0;}
	#else
		if(pixRec->newPort) {CloseCPort(pixRec->newPort); _DisposePtr((Ptr)pixRec->newPort); pixRec->newPort = 0;}
	#endif
	//CloseCPort(pixRec->newPort);
	//_DisposePtr((Ptr)pixRec->newPort);
	DisposePixMap(pixRec->newPixMap);
	_HUnlock(pixRec->pixelsHandle);
	DisposeHandle(pixRec->pixelsHandle);
}

#if TARGET_API_MAC_CARBON
BitMap OpenBlackAndWhiteBitMap(Rect theBounds, MyGWorldRec *saveGWorld)
{
	BitMap			paintBits;
	memset(&paintBits,0,sizeof(paintBits));
	
	return paintBits;
}

void CloseBlackAndWhiteBitMap (MyGWorldRec *saveGWorld)
{
#pragma unused(saveGWorld)
	return;
}

void DrawBlackAndWhiteBitMap(BitMap *bitmap, Rect m)
{	// m holds the map rectangle in local coordinates
	// use current port rather than pass it in to make making bitmaps with bitmaps easier
	CGrafPtr port;
	GetPort(&port);
	if(port && bitmap && bitmap->baseAddr)
	{
		Rect 			scrRect = bitmap->bounds;
		Rect 			r,dstRect ;
		Boolean 		intersect;
		
		GetPortBounds(port,&r);
		intersect = IntersectRect(&r,&m,&dstRect);
		
		if(!intersect) return; 
		scrRect = dstRect; // find the corresponding source rect
		MapRect(&scrRect,&m,&bitmap->bounds);

		_HLock ((Handle) (GetPortPixMap(port)));
		CopyBits(bitmap, 
			(BitMap *) *(GetPortPixMap(port)),
			&scrRect, &dstRect,srcOr, nil); //
		_HUnlock ((Handle) (GetPortPixMap(port)));
	}
}

void DisposeBlackAndWhiteBitMap (BitMap *theBitMap)
{
	if(!theBitMap) return;
	if(theBitMap -> baseAddr) // JLM
	{
		_DisposePtr(theBitMap -> baseAddr);
		theBitMap -> baseAddr = nil;
	}
	memset(theBitMap,0,sizeof(*theBitMap)); //JLM
}
#else
BitMap OpenBlackAndWhiteBitMap(Rect theBounds, MyGWorldRec *saveGWorld)
{
	BitMap			paintBits;
	RgnHandle		newVis = 0;

	memset(&paintBits,0,sizeof(paintBits));
	
	saveGWorld -> saveVisRgn = nil;
	
	GetPort(&(saveGWorld -> savePort));
	OpenPort(&(saveGWorld -> newPort));// we can not count on anything as far as the current port
	SetPort(&(saveGWorld -> newPort));// so we need our own

	ForeColor(blackColor);			
	BackColor(whiteColor);
	
	// set the dimensions of the offsceen bit map here
	paintBits.bounds = theBounds;
  
	paintBits.rowBytes = (((paintBits.bounds.right - paintBits.bounds.left) >> 4) + 1) << 1;
	paintBits.baseAddr = _NewPtrClear((paintBits.bounds.bottom - paintBits.bounds.top)
												* paintBits.rowBytes);	
	if(paintBits.baseAddr)
	{
		
		// Get new clip region 
		newVis  = NewRgn();
		
		saveGWorld -> savePortBits = saveGWorld -> newPort.portBits;
		saveGWorld -> saveVisRgn   = saveGWorld -> newPort.visRgn;

		SetPortBits(&paintBits);
		ClipRect(&(saveGWorld -> newPort.portBits.bounds));
		RectRgn(newVis, &(saveGWorld -> newPort.portBits.bounds));
		saveGWorld -> newPort.visRgn = newVis;
	}
	
	return paintBits;
}

void CloseBlackAndWhiteBitMap (MyGWorldRec *saveGWorld)
{
	SetPortBits(&(saveGWorld -> savePortBits));
	DisposeRgn (saveGWorld -> newPort.visRgn);
	saveGWorld -> newPort.visRgn = saveGWorld -> saveVisRgn;
	SetPort(saveGWorld -> savePort);
	ClosePort(&(saveGWorld -> newPort));  // deallocates ports visRgn and clipRgn

	return;
}

void DrawBlackAndWhiteBitMap(BitMap *bitmap, Rect m)
{	// m holds the map rectangle in local coordinates
	WindowPtr w = 0 ; // use current port rather than pass it in to make making bitmaps with bitmaps easier
	GetPort(&w);
	if(w && bitmap && bitmap->baseAddr)
	{
		// draw bitmap for diagnostic
		// copy in OR mode
		Rect 			scrRect = bitmap->bounds;
//		Rect 			r = MapDrawingRect();
	// JLM,  why not the MapDrawingRect ???
		Rect 			r = w -> portRect;		// window rect
		Rect 			dstRect ;
		Boolean 		intersect = IntersectRect(&r,&m,&dstRect);
		
		if(!intersect) return; // JLM 12/23/98
		scrRect = dstRect; // find the corresponding source rect
		MapRect(&scrRect,&m,&bitmap->bounds);

		_HLock ((Handle) ((CGrafPtr) w) -> portPixMap);
		CopyBits(bitmap, 
			(BitMap *) *(((CGrafPtr) w) -> portPixMap),
			&scrRect, &dstRect,srcOr, nil); //
		_HUnlock ((Handle) ((CGrafPtr) w) -> portPixMap);
	}
}

void DisposeBlackAndWhiteBitMap (BitMap *theBitMap)
{
	if(!theBitMap) return;
	if(theBitMap -> baseAddr) // JLM
	{
		_DisposePtr(theBitMap -> baseAddr);
		theBitMap -> baseAddr = nil;
	}
	memset(theBitMap,0,sizeof(*theBitMap)); //JLM
}

#endif
#else

//////////////////////////////// WINDOWS CODE //////////////////////////////////

OSErr OSNewPortPix(PortPixRecordP pixRec, WindowPtr w, Rect r)
{
	HDC hDC, hScrDC;
	HBITMAP hDDB;
	PixMapHandle hDIB = 0;
	
#ifdef DRAWTODESKTOP
	pixRec->hMemDC = GetDC(GetDesktopWindow());
#else
	hDC = GetDC(w);
	hDDB = CreateCompatibleBitmap(hDC, RectWidth(r) + r.left,
									   RectHeight(r) + r.top);
	ReleaseDC(w, hDC);
	if (!hDDB) return -1;
	
	hScrDC = CreateDC("DISPLAY", NULL, NULL, NULL);
	pixRec->hMemDC = CreateCompatibleDC(hScrDC);
	DeleteDC(hScrDC);
	
	pixRec->hOldBitmap = (HBITMAP)SelectObject(pixRec->hMemDC, hDDB);
#endif
	
	IntersectClipRect(pixRec->hMemDC, r.left, r.top, r.right, r.bottom);
	
	pixRec->r = r;
	
	return 0;
}

void OSCopyToWindow(PortPixRecordP pixRec, WindowPtr w, Rect rSrc, Rect rDst, short mode)
{
	Boolean isWindow = MyIsWindow(w);
	HDC hDC = isWindow ? GetDC(w) : (HDC)w;
	
	if (EqualRects(rSrc, rDst))
		BitBlt(hDC, rDst.left, rDst.top, RectWidth(rDst), RectHeight(rDst),
			   pixRec->hMemDC, rSrc.left, rSrc.top, SRCCOPY); // mode
	else
		StretchBlt(hDC, rDst.left, rDst.top, RectWidth(rDst), RectHeight(rDst),
				   pixRec->hMemDC, rSrc.left, rSrc.top, RectWidth(rSrc), RectHeight(rSrc),
				   SRCCOPY);
	
	if (isWindow) ReleaseDC(w, hDC);
}

void OSDisposePortPix(PortPixRecordP pixRec)
{
	HBITMAP hDDB;
	
#ifdef DRAWTODESKTOP
	ReleaseDC(GetDesktopWindow(), pixRec->hMemDC);
#else
	hDDB = (HBITMAP)SelectObject(pixRec->hMemDC, pixRec->hOldBitmap);
	DeleteObject(hDDB);
	DeleteDC(pixRec->hMemDC);
#endif
}

///////////////////////////////////////////////////////////////////////

//void BitMapDispose(PixMapHandle bits, WindowPtr w)	// not used on IBM (codewarrior mod)
//{
//#pragma unused(w)
//
//	Debug ("Inside  BitMapDispose\n");
//
//	DisposPixMap (bits);
//	bits = nil;
//
//	Debug ("Exiting BitMapDispose\n");
//	DestroyDIB(bits);
//}

void BitMapDisposeB(PixMapHandle bits, WindowPtr w)
{
#pragma unused(w)
	
	DeleteObject((HBITMAP)bits);
}

void CopyBitsFromWindow(WindowPtr w, PixMapHandle *bits, Rect srcRect, Rect dstRect)
{
	RECT R;
	
	MakeWindowsRect(&srcRect, &R);
	MapWindowPoints(w, GetDesktopWindow(), (POINT *)&R, 2);
	if (!EqualRects(srcRect, dstRect)) { // go into special stretch mode for making bitmaps
		gDIBDestRectWidth = RectWidth(dstRect);
		gDIBDestRectHeight = RectHeight(dstRect);
	}
	(*bits) = CopyScreenToDIB(&R);
	gDIBDestRectWidth = gDIBDestRectHeight = -1; // exit special stretch mode
}

void CopyBitsFromWindowB(WindowPtr w, PixMapHandle *bits, Rect srcRect, Rect dstRect)
{
	RECT R;
	
	MakeWindowsRect(&srcRect, &R);
	MapWindowPoints(w, GetDesktopWindow(), (POINT *)&R, 2);
	if (!EqualRects(srcRect, dstRect)) { // go into special stretch mode for making bitmaps
		gDIBDestRectWidth = RectWidth(dstRect);
		gDIBDestRectHeight = RectHeight(dstRect);
	}
	(*bits) = (PixMapHandle)CopyScreenToBitmap(&R);
	gDIBDestRectWidth = gDIBDestRectHeight = -1; // exit special stretch mode
}

void CopyBitsToWindow(PixMapHandle bits, WindowPtr w, Rect srcRect, Rect dstRect)
{
	HDC hDC;
	RECT Rsrc, Rdst;
	HPALETTE hPal;
	LPSTR lpDIB;
	
	if (EqualRects(srcRect, bigRect)) {
		lpDIB = (LPSTR)GlobalLock(bits);
		MySetRect(&srcRect, 0, 0, DIBWidth(lpDIB), DIBHeight(lpDIB));
		GlobalUnlock(bits);
	}
	else
		MyOffsetRect(&srcRect, -srcRect.left, -srcRect.top);
	MakeWindowsRect(&srcRect, &Rsrc);
	MakeWindowsRect(&dstRect, &Rdst);
	
	// hPal = CreateDIBPalette(bits);
	hPal = GetSystemPalette();
	hDC = MyIsWindow(w) ? GetDC(w) : (HDC)w;
	PaintDIB(hDC, &Rdst, bits, &Rsrc, hPal);
	if (MyIsWindow(w)) ReleaseDC(w, hDC);
	DeleteObject(hPal);
}

void CopyBitsToWindowB(PixMapHandle bits, WindowPtr w, Rect srcRect, Rect dstRect)
{
	HDC hDC;
	RECT Rsrc, Rdst;
	BITMAP bm;
	
	if (EqualRects(srcRect, bigRect)) {
		if (!GetObject((HBITMAP)bits, sizeof(bm), (LPSTR)&bm))
			{ SysBeep(1); return; }
		MySetRect(&srcRect, 0, 0, bm.bmWidth, bm.bmHeight);
	}
	else
		MyOffsetRect(&srcRect, -srcRect.left, -srcRect.top);
	MakeWindowsRect(&srcRect, &Rsrc);
	MakeWindowsRect(&dstRect, &Rdst);
	
	hDC = MyIsWindow(w) ? GetDC(w) : (HDC)w;
	PaintBitmap(hDC, &Rdst, (HBITMAP)bits, &Rsrc, 0);
	if (MyIsWindow(w)) ReleaseDC(w, hDC);
}

///////////////////////////////////////////////////////////////////////

HBITMAP ResizeBitmap(HBITMAP hDDB, short width, short height)
{
	HDC hDC, hScrDC, hMemDC1, hMemDC2;
	HBITMAP oldBitmap1, oldBitmap2, bitmap1 = hDDB, bitmap2 = 0;
	BITMAP bm;
	
   	if (!GetObject(bitmap1, sizeof(bm), (LPSTR)&bm)) return 0;
	
	hDC = GetDC(GetDesktopWindow());
	bitmap2 = CreateCompatibleBitmap(hDC, width, height);
	ReleaseDC(GetDesktopWindow(), hDC);
	
	if (!bitmap2) return 0;
	
	hScrDC = CreateDC("DISPLAY", NULL, NULL, NULL);
	hMemDC1 = CreateCompatibleDC(hScrDC);
	hMemDC2 = CreateCompatibleDC(hScrDC);
	DeleteDC(hScrDC);
	
	oldBitmap1 = (HBITMAP)SelectObject(hMemDC1, bitmap1);
	oldBitmap2 = (HBITMAP)SelectObject(hMemDC2, bitmap2);
	
	SetStretchBltMode(hMemDC2, COLORONCOLOR);
	StretchBlt(hMemDC2, 0, 0, width, height,
			   hMemDC1, 0, 0, bm.bmWidth, bm.bmHeight, SRCCOPY);
	
	bitmap1 = (HBITMAP)SelectObject(hMemDC1, oldBitmap1);
	bitmap2 = (HBITMAP)SelectObject(hMemDC2, oldBitmap2);
	
	DeleteDC(hMemDC1);
	DeleteDC(hMemDC2);
	
	return bitmap2;
}

HDIB ResizeDIB(HDIB hDIB, short width, short height)
{
	HBITMAP bitmap1 = 0, bitmap2 = 0;
	HDIB newDIB = 0;
	
	bitmap1 = DIBToBitmap(hDIB, 0);
	if (bitmap1)
		bitmap2 = ResizeBitmap(bitmap1, width, height);
	if (bitmap2)
		newDIB = BitmapToDIB(bitmap2, 0);
	
	if (bitmap1) DeleteObject(bitmap1);
	if (bitmap2) DeleteObject(bitmap2);
	
	return newDIB;
}

///////////////////////////////////////////////////////////////////////

#define IDC_PRINTTEXT1              401
#define IDC_PRINTTEXT2              402
#define IDC_PERCENTAGE              403

#define PALVERSION   0x300
#define IS_WIN30_DIB(lpbi)  ((*(LPDWORD)(lpbi)) == sizeof(BITMAPINFOHEADER))
#define RECTWIDTH(lpRect)     ((lpRect)->right - (lpRect)->left)
#define RECTHEIGHT(lpRect)    ((lpRect)->bottom - (lpRect)->top)

// WIDTHBYTES performs DWORD-aligning of DIB scanlines.  The "bits"
// parameter is the bit count for the scanline (biWidth * biBitCount),
// and this macro returns the number of DWORD-aligned bytes needed 
// to hold those bits.

#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)

/* Error constants */
enum {
      ERR_MIN = 0,                     // All error #s >= this value
      ERR_NOT_DIB = 0,                 // Tried to load a file, NOT a DIB!
      ERR_MEMORY,                      // Not enough memory!
      ERR_READ,                        // Error reading file!
      ERR_LOCK,                        // Error on a GlobalLock()!
      ERR_OPEN,                        // Error opening a file!
      ERR_CREATEPAL,                   // Error creating palette.
      ERR_GETDC,                       // Couldn't get a DC.
      ERR_CREATEDDB,                   // Error create a DDB.
      ERR_STRETCHBLT,                  // StretchBlt() returned failure.
      ERR_STRETCHDIBITS,               // StretchDIBits() returned failure.
      ERR_SETDIBITSTODEVICE,           // SetDIBitsToDevice() failed.
      ERR_STARTDOC,                    // Error calling StartDoc().
      ERR_NOGDIMODULE,                 // Couldn't find GDI module in memory.
      ERR_SETABORTPROC,                // Error calling SetAbortProc().
      ERR_STARTPAGE,                   // Error calling StartPage().
      ERR_NEWFRAME,                    // Error calling NEWFRAME escape.
      ERR_ENDPAGE,                     // Error calling EndPage().
      ERR_ENDDOC,                      // Error calling EndDoc().
      ERR_SETDIBITS,                   // Error calling SetDIBits().
      ERR_FILENOTFOUND,                // Error opening file in GetDib()
      ERR_INVALIDHANDLE,               // Invalid Handle
      ERR_DIBFUNCTION,                 // Error on call to DIB function
      ERR_MAX                          // All error #s < this value
     };

/*************************************************************************
 *
 * CopyWindowToDIB()
 *
 * Parameters:
 *
 * HWND hWnd        - specifies the window
 *
 * WORD fPrintArea  - specifies the window area to copy into the device-
 *                    independent bitmap
 *
 * Return Value:
 *
 * HDIB             - identifies the device-independent bitmap
 *
 * Description:
 *
 * This function copies the specified part(s) of the window to a device-
 * independent bitmap.
 *
 * History:   Date      Author              Reason
 *            9/15/91   Patrick Schreiber   Created
 *            9/25/91   Patrick Schreiber   Added header and comments
 *
 ************************************************************************/


HDIB FAR CopyWindowToDIB(HWND hWnd, WORD fPrintArea)
{
   HDIB hDIB = NULL;  // handle to DIB

   /* check for a valid window handle */

   if (!hWnd) return NULL;
   switch (fPrintArea) {
   case PW_WINDOW: // copy entire window
   {
      RECT rectWnd;

      /* get the window rectangle */

      GetWindowRect(hWnd, &rectWnd);

      /*  get the DIB of the window by calling
       *  CopyScreenToDIB and passing it the window rect
       */
      hDIB = CopyScreenToDIB(&rectWnd);
   }
      break;

   case PW_CLIENT: // copy client area
   {
      RECT rectClient;
      POINT pt1, pt2;

      /* get the client area dimensions */

      GetClientRect(hWnd, &rectClient);

      /* convert client coords to screen coords */
      pt1.x = rectClient.left;
      pt1.y = rectClient.top;
      pt2.x = rectClient.right;
      pt2.y = rectClient.bottom;
      ClientToScreen(hWnd, &pt1);
      ClientToScreen(hWnd, &pt2);
      rectClient.left = pt1.x;
      rectClient.top = pt1.y;
      rectClient.right = pt2.x;
      rectClient.bottom = pt2.y;

      /*  get the DIB of the client area by calling
       *  CopyScreenToDIB and passing it the client rect
       */
      hDIB = CopyScreenToDIB(&rectClient);
   }
      break;

   default:    // invalid print area
      return NULL;
      }

   /* return the handle to the DIB */
   return hDIB;
}


/*************************************************************************
 *
 * CopyScreenToDIB()
 *
 * Parameter:
 *
 * LPRECT lpRect    - specifies the window
 *
 * Return Value:
 *
 * HDIB             - identifies the device-independent bitmap
 *
 * Description:
 *
 * This function copies the specified part of the screen to a device-
 * independent bitmap.
 *
 * History:   Date      Author             Reason
 *            9/15/91   Patrick Schreiber  Created
 *            9/25/91   Patrick Schreiber  Added header and comments
 *            12/10/91  Patrick Schreiber  Released palette
 *
 ************************************************************************/


HDIB FAR CopyScreenToDIB(LPRECT lpRect)
{
   HBITMAP hBitmap;    // handle to device-dependent bitmap
   HPALETTE hPalette;  // handle to palette
   HDIB hDIB = NULL;   // handle to DIB

   /*  get the device-dependent bitmap in lpRect by calling
    *  CopyScreenToBitmap and passing it the rectangle to grab
    */

   hBitmap = CopyScreenToBitmap(lpRect);

   /* check for a valid bitmap handle */
   if (!hBitmap) return NULL;

   /* get the current palette */
   hPalette = GetSystemPalette();

   /* convert the bitmap to a DIB */
   hDIB = BitmapToDIB(hBitmap, hPalette);

   /* clean up */
   DeleteObject(hPalette);
   DeleteObject(hBitmap);

   /* return handle to the packed-DIB */
   return hDIB;
}


/*************************************************************************
 *
 * CopyWindowToBitmap()
 *
 * Parameters:
 *
 * HWND hWnd        - specifies the window
 *
 * WORD fPrintArea  - specifies the window area to copy into the device-
 *                    dependent bitmap
 *
 * Return Value:
 *
 * HDIB         - identifies the device-dependent bitmap
 *
 * Description:
 *
 * This function copies the specified part(s) of the window to a device-
 * dependent bitmap.
 *
 * History:   Date      Author              Reason
 *            9/15/91   Patrick Schreiber   Created
 *            9/25/91   Patrick Schreiber   Added header and comments
 *
 ************************************************************************/


HBITMAP FAR CopyWindowToBitmap(HWND hWnd, WORD fPrintArea)
{
   HBITMAP hBitmap = NULL;  // handle to device-dependent bitmap

   /* check for a valid window handle */

   if (!hWnd)
      return NULL;
   switch (fPrintArea)
      {
   case PW_WINDOW: // copy entire window
   {
      RECT rectWnd;

      /* get the window rectangle */

      GetWindowRect(hWnd, &rectWnd);

      /*  get the bitmap of that window by calling
       *  CopyScreenToBitmap and passing it the window rect
       */
      hBitmap = CopyScreenToBitmap(&rectWnd);
   }
   break;

   case PW_CLIENT: // copy client area
   {
      RECT rectClient;
      POINT pt1, pt2;

      /* get client dimensions */

      GetClientRect(hWnd, &rectClient);

      /* convert client coords to screen coords */
      pt1.x = rectClient.left;
      pt1.y = rectClient.top;
      pt2.x = rectClient.right;
      pt2.y = rectClient.bottom;
      ClientToScreen(hWnd, &pt1);
      ClientToScreen(hWnd, &pt2);
      rectClient.left = pt1.x;
      rectClient.top = pt1.y;
      rectClient.right = pt2.x;
      rectClient.bottom = pt2.y;

      /*  get the bitmap of the client area by calling
       *  CopyScreenToBitmap and passing it the client rect
       */
      hBitmap = CopyScreenToBitmap(&rectClient);
   }
   break;

   default:    // invalid print area
      return NULL;
      }

   /* return handle to the bitmap */
   return hBitmap;
}


/*************************************************************************
 *
 * CopyScreenToBitmap()
 *
 * Parameter:
 *
 * LPRECT lpRect    - specifies the window
 *
 * Return Value:
 *
 * HDIB             - identifies the device-dependent bitmap
 *
 * Description:
 *
 * This function copies the specified part of the screen to a device-
 * dependent bitmap.
 *
 * History:   Date      Author             Reason
 *            9/15/91   Patrick Schreiber  Created
 *            9/25/91   Patrick Schreiber  Added header and comments
 *
 ************************************************************************/

// allow applications to copy screen images into variable-size bitmaps
short gDIBDestRectWidth = -1, gDIBDestRectHeight = -1;

HBITMAP FAR CopyScreenToBitmap(LPRECT lpRect)
{
   HDC hScrDC, hMemDC;           // screen DC and memory DC
   HBITMAP hBitmap, hOldBitmap;  // handles to deice-dependent bitmaps
   int nX, nY, nX2, nY2;         // coordinates of rectangle to grab
   int nWidth, nHeight;          // DIB width and height
   int xScrn, yScrn;             // screen resolution

   /* check for an empty rectangle */

   if (IsRectEmpty(lpRect))
      return NULL;

   /*  create a DC for the screen and create
    *  a memory DC compatible to screen DC
    */
   hScrDC = CreateDC("DISPLAY", NULL, NULL, NULL);
   hMemDC = CreateCompatibleDC(hScrDC);

   /* get points of rectangle to grab */
   nX = lpRect->left;
   nY = lpRect->top;
   nX2 = lpRect->right;
   nY2 = lpRect->bottom;

   /* get screen resolution */
   xScrn = GetDeviceCaps(hScrDC, HORZRES);
   yScrn = GetDeviceCaps(hScrDC, VERTRES);

   /* make sure bitmap rectangle is visible */
   if (nX < 0)
      nX = 0;
   if (nY < 0)
      nY = 0;
   if (nX2 > xScrn)
      nX2 = xScrn;
   if (nY2 > yScrn)
      nY2 = yScrn;
   if (gDIBDestRectWidth == -1) { // regular mode
      nWidth = nX2 - nX;
      nHeight = nY2 - nY;
   }
   else { // special stretch mode
      nWidth = gDIBDestRectWidth;
	  nHeight = gDIBDestRectHeight;
   }

   /* create a bitmap compatible with the screen DC */
   hBitmap = CreateCompatibleBitmap(hScrDC, nWidth, nHeight);

   /* select new bitmap into memory DC */
   hOldBitmap = (HBITMAP)SelectObject(hMemDC, hBitmap);

   if (gDIBDestRectWidth == -1) // regular mode: BitBlt() screen DC to memory DC
      BitBlt(hMemDC, 0, 0, nWidth, nHeight, hScrDC, nX, nY, SRCCOPY);
   else // special stretch mode: use StretchBlt()
      StretchBlt(hMemDC, 0, 0, nWidth, nHeight, hScrDC, 
				 nX, nY, nX2 - nX, nY2 - nY, SRCCOPY);

   /*  select old bitmap back into memory DC and get handle to
    *  bitmap of the screen
    */
   hBitmap = (HBITMAP)SelectObject(hMemDC, hOldBitmap);

   /* clean up */
   DeleteDC(hScrDC);
   DeleteDC(hMemDC);

   /* return handle to the bitmap */
   return hBitmap;
}


/*************************************************************************
 *
 * PaintDIB()
 *
 * Parameters:
 *
 * HDC hDC          - DC to do output to
 *
 * LPRECT lpDCRect  - rectangle on DC to do output to
 *
 * HDIB hDIB        - handle to global memory with a DIB spec
 *                    in it followed by the DIB bits
 *
 * LPRECT lpDIBRect - rectangle of DIB to output into lpDCRect
 *
 * Return Value:
 *
 * BOOL             - TRUE if DIB was drawn, FALSE otherwise
 *
 * Description:
 *   Painting routine for a DIB.  Calls StretchDIBits() or
 *   SetDIBitsToDevice() to paint the DIB.  The DIB is
 *   output to the specified DC, at the coordinates given
 *   in lpDCRect.  The area of the DIB to be output is
 *   given by lpDIBRect.
 *
 * NOTE: This function always selects the palette as background. Before
 * calling this function, be sure your palette is selected to desired
 * priority (foreground or background).
 *
 * History:   
 *            
 *    Date      Author               Reason        
 *    6/1/91    Garrett McAuliffe    Created        
 *    12/12/91  Patrick Schreiber    Removed palette param, added get        
 *                                   palette stuff, return value, header
 *                                   and some comments
 *    6/8/92    Patrick Schreiber    Put palette param back in, select as
 *                                   background palette always. Added NOTE
 *                                   above.
 *
 ************************************************************************/

BOOL FAR PaintDIB(HDC      hDC,
                  LPRECT   lpDCRect,
                  HDIB     hDIB,
                  LPRECT   lpDIBRect,
                  HPALETTE hPal)
{
   LPSTR    lpDIBHdr;            // Pointer to BITMAPINFOHEADER
   LPSTR    lpDIBBits;           // Pointer to DIB bits
   BOOL     bSuccess=FALSE;      // Success/fail flag
   HPALETTE hOldPal=NULL;        // Previous palette

   /* Check for valid DIB handle */
   if (!hDIB)
      return FALSE;

   /* Lock down the DIB, and get a pointer to the beginning of the bit
    *  buffer
    */
   lpDIBHdr  = (LPSTR)GlobalLock(hDIB);
   lpDIBBits = FindDIBBits(lpDIBHdr);

   /* Select and realize our palette as background */
   if (hPal)
   {
      hOldPal = SelectPalette(hDC, hPal, TRUE);
      RealizePalette(hDC);
   }

   /* Make sure to use the stretching mode best for color pictures */
   SetStretchBltMode(hDC, COLORONCOLOR);

   /* Determine whether to call StretchDIBits() or SetDIBitsToDevice() */
   if ((RECTWIDTH(lpDCRect)  == RECTWIDTH(lpDIBRect)) &&
       (RECTHEIGHT(lpDCRect) == RECTHEIGHT(lpDIBRect)))
      bSuccess = SetDIBitsToDevice(hDC,                        // hDC
                                   lpDCRect->left,             // DestX
                                   lpDCRect->top,              // DestY
                                   RECTWIDTH(lpDCRect),        // nDestWidth
                                   RECTHEIGHT(lpDCRect),       // nDestHeight
                                   lpDIBRect->left,            // SrcX
                                   (int)DIBHeight(lpDIBHdr) -
                                      lpDIBRect->top - 
                                      RECTHEIGHT(lpDIBRect),   // SrcY
                                   0,                          // nStartScan
                                   (WORD)DIBHeight(lpDIBHdr),  // nNumScans
                                   lpDIBBits,                  // lpBits
                                   (LPBITMAPINFO)lpDIBHdr,     // lpBitsInfo
                                   DIB_RGB_COLORS);            // wUsage
   else 
      bSuccess = StretchDIBits(hDC,                            // hDC
                               lpDCRect->left,                 // DestX
                               lpDCRect->top,                  // DestY
                               RECTWIDTH(lpDCRect),            // nDestWidth
                               RECTHEIGHT(lpDCRect),           // nDestHeight
                               lpDIBRect->left,                // SrcX
                               lpDIBRect->top,                 // SrcY
                               RECTWIDTH(lpDIBRect),           // wSrcWidth
                               RECTHEIGHT(lpDIBRect),          // wSrcHeight
                               lpDIBBits,                      // lpBits
                               (LPBITMAPINFO)lpDIBHdr,         // lpBitsInfo
                               DIB_RGB_COLORS,                 // wUsage
                               SRCCOPY);                       // dwROP

   /* Unlock the memory block */
   GlobalUnlock(hDIB);

   /* Reselect old palette */
   if (hOldPal)
     SelectPalette(hDC, hOldPal, FALSE);

   /* Return with success/fail flag */
   return bSuccess;
}


/*************************************************************************
 *
 * PaintBitmap()
 *
 * Parameters:
 *
 * HDC hDC          - DC to do output to
 *
 * LPRECT lpDCRect  - rectangle on DC to do output to
 *
 * HBITMAP hDDB     - handle to device-dependent bitmap (DDB)
 *
 * LPRECT lpDDBRect - rectangle of DDB to output into lpDCRect
 *
 * HPALETTE hPalette - handle to the palette to use with hDDB
 *
 * Return Value:
 *
 * BOOL             - TRUE if bitmap was drawn, FLASE otherwise
 *
 * Description:
 *
 * Painting routine for a DDB.  Calls BitBlt() or
 * StretchBlt() to paint the DDB.  The DDB is
 * output to the specified DC, at the coordinates given
 * in lpDCRect.  The area of the DDB to be output is
 * given by lpDDBRect.  The specified palette is used.
 *
 * NOTE: This function always selects the palette as background. Before
 * calling this function, be sure your palette is selected to desired
 * priority (foreground or background).
 *
 * History:   
 *            
 *   Date      Author               Reason         
 *   6/1/91    Garrett McAuliffe    Created         
 *   12/12/91  Patrick Schreiber    Added return value, realizepalette,
 *                                       header and some comments
 *   6/8/92    Patrick Schreiber    Select palette as background always, added
 *                                  NOTE above.
 *
 ************************************************************************/

BOOL FAR PaintBitmap(HDC      hDC,
                     LPRECT   lpDCRect, 
                     HBITMAP  hDDB, 
                     LPRECT   lpDDBRect, 
                     HPALETTE hPal)
{
   HDC      hMemDC;            // Handle to memory DC
   HBITMAP  hOldBitmap;        // Handle to previous bitmap
   HPALETTE hOldPal1 = NULL;   // Handle to previous palette
   HPALETTE hOldPal2 = NULL;   // Handle to previous palette
   BOOL     bSuccess = FALSE;  // Success/fail flag

   /* Create a memory DC */
   hMemDC = CreateCompatibleDC (hDC);

   /* If this failed, return FALSE */
   if (!hMemDC)
      return FALSE;

   /* If we have a palette, select and realize it */
   if (hPal)
   {
      hOldPal1 = SelectPalette(hMemDC, hPal, TRUE);
      hOldPal2 = SelectPalette(hDC, hPal, TRUE);
      RealizePalette(hDC);
   }

   /* Select bitmap into the memory DC */
   hOldBitmap = (HBITMAP)SelectObject (hMemDC, hDDB);

   /* Make sure to use the stretching mode best for color pictures */   
   SetStretchBltMode (hDC, COLORONCOLOR);

   /* Determine whether to call StretchBlt() or BitBlt() */ 
   if ((RECTWIDTH(lpDCRect)  == RECTWIDTH(lpDDBRect)) &&
       (RECTHEIGHT(lpDCRect) == RECTHEIGHT(lpDDBRect)))
      bSuccess = BitBlt(hDC,
                        lpDCRect->left,
                        lpDCRect->top,
                        lpDCRect->right - lpDCRect->left,
                        lpDCRect->bottom - lpDCRect->top,
                        hMemDC,
                        lpDDBRect->left,
                        lpDDBRect->top,
                        SRCCOPY);
   else
      bSuccess = StretchBlt(hDC,
                            lpDCRect->left, 
                            lpDCRect->top, 
                            lpDCRect->right - lpDCRect->left,
                            lpDCRect->bottom - lpDCRect->top,
                            hMemDC,
                            lpDDBRect->left, 
                            lpDDBRect->top, 
                            lpDDBRect->right - lpDDBRect->left,
                            lpDDBRect->bottom - lpDDBRect->top,
                            SRCCOPY);

   /* Clean up */
   SelectObject(hMemDC, hOldBitmap);

   if (hOldPal1)
      SelectPalette (hMemDC, hOldPal1, FALSE);

   if (hOldPal2)
      SelectPalette (hDC, hOldPal2, FALSE);

   DeleteDC (hMemDC);

   /* Return with success/fail flag */
   return bSuccess;
}

//////////////////////////////////////////////////////////////////////////////////

/*************************************************************************
 *
 * CreateDIB()
 *
 * Parameters:
 *
 * DWORD dwWidth    - Width for new bitmap, in pixels
 * DWORD dwHeight   - Height for new bitmap 
 * WORD  wBitCount  - Bit Count for new DIB (1, 4, 8, or 24)
 *
 * Return Value:
 *
 * HDIB             - Handle to new DIB
 *
 * Description:
 *
 * This function allocates memory for and initializes a new DIB by
 * filling in the BITMAPINFOHEADER, allocating memory for the color
 * table, and allocating memory for the bitmap bits.  As with all
 * HDIBs, the header, colortable and bits are all in one contiguous
 * memory block.  This function is similar to the CreateBitmap() 
 * Windows API.
 *
 * The colortable and bitmap bits are left uninitialized (zeroed) in the
 * returned HDIB.
 *
 *
 * History:   Date      Author              Reason
 *            3/20/92   Mark Bader          Created
 *
 ************************************************************************/

HDIB FAR CreateDIB(DWORD dwWidth, DWORD dwHeight, WORD wBitCount)
{
   BITMAPINFOHEADER bi;         // bitmap header
   LPBITMAPINFOHEADER lpbi;     // pointer to BITMAPINFOHEADER
   DWORD dwLen;                 // size of memory block
   HDIB hDIB;
   DWORD dwBytesPerLine;        // Number of bytes per scanline


   // Make sure bits per pixel is valid
   if (wBitCount <= 1)
      wBitCount = 1;
   else if (wBitCount <= 4)
      wBitCount = 4;
   else if (wBitCount <= 8)
      wBitCount = 8;
   else if (wBitCount <= 24)
      wBitCount = 24;
   else
      wBitCount = 4;  // set default value to 4 if parameter is bogus

   // initialize BITMAPINFOHEADER
   bi.biSize = sizeof(BITMAPINFOHEADER);
   bi.biWidth = dwWidth;         // fill in width from parameter
   bi.biHeight = dwHeight;       // fill in height from parameter
   bi.biPlanes = 1;              // must be 1
   bi.biBitCount = wBitCount;    // from parameter
   bi.biCompression = BI_RGB;    
   bi.biSizeImage = 0;           // 0's here mean "default"
   bi.biXPelsPerMeter = 0;
   bi.biYPelsPerMeter = 0;
   bi.biClrUsed = 0;
   bi.biClrImportant = 0;

   // calculate size of memory block required to store the DIB.  This
   // block should be big enough to hold the BITMAPINFOHEADER, the color
   // table, and the bits

   dwBytesPerLine = WIDTHBYTES(wBitCount * dwWidth);
   dwLen = bi.biSize + PaletteSize((LPSTR)&bi) + (dwBytesPerLine * dwHeight);

   // alloc memory block to store our bitmap
   hDIB = GlobalAlloc(GHND, dwLen);

   // major bummer if we couldn't get memory block
   if (!hDIB)
   {
      return NULL;
   }

   // lock memory and get pointer to it
   lpbi = (LPBITMAPINFOHEADER)GlobalLock(hDIB);

   // use our bitmap info structure to fill in first part of
   // our DIB with the BITMAPINFOHEADER
   *lpbi = bi;

   // Since we don't know what the colortable and bits should contain,
   // just leave these blank.  Unlock the DIB and return the HDIB.

   GlobalUnlock(hDIB);

   /* return handle to the DIB */
   return hDIB;
}

/*************************************************************************
 *
 * DestroyDIB ()
 *
 * Purpose:  Frees memory associated with a DIB
 *
 * Returns:  Nothing
 *
 * History:   Date      Author       Reason
 *            9/15/91   Mark Bader   Created
 *
 *************************************************************************/


WORD FAR DestroyDIB(HDIB hDib)
{
   GlobalFree(hDib);
   return 0;
}


/*************************************************************************
 *
 * FindDIBBits()
 *
 * Parameter:
 *
 * LPSTR lpDIB      - pointer to packed-DIB memory block
 *
 * Return Value:
 *
 * LPSTR            - pointer to the DIB bits
 *
 * Description:
 *
 * This function calculates the address of the DIB's bits and returns a
 * pointer to the DIB bits.
 *
 * History:   Date      Author              Reason
 *            6/01/91   Garrett McAuliffe   Created
 *            9/15/91   Patrick Schreiber   Added header and comments
 *
 ************************************************************************/


LPSTR FAR FindDIBBits(LPSTR lpDIB)
{
   return (lpDIB + *(LPDWORD)lpDIB + PaletteSize(lpDIB));
}


/*************************************************************************
 *
 * DIBWidth()
 *
 * Parameter:
 *
 * LPSTR lpDIB      - pointer to packed-DIB memory block
 *
 * Return Value:
 *
 * DWORD            - width of the DIB
 *
 * Description:
 *
 * This function gets the width of the DIB from the BITMAPINFOHEADER
 * width field if it is a Windows 3.0-style DIB or from the BITMAPCOREHEADER
 * width field if it is an OS/2-style DIB.
 *
 * History:   Date      Author               Reason
 *            6/01/91   Garrett McAuliffe    Created
 *            9/15/91   Patrick Schreiber    Added header and comments
 *
 ************************************************************************/


DWORD FAR DIBWidth(LPSTR lpDIB)
{
   LPBITMAPINFOHEADER lpbmi;  // pointer to a Win 3.0-style DIB
   LPBITMAPCOREHEADER lpbmc;  // pointer to an OS/2-style DIB

   /* point to the header (whether Win 3.0 and OS/2) */

   lpbmi = (LPBITMAPINFOHEADER)lpDIB;
   lpbmc = (LPBITMAPCOREHEADER)lpDIB;

   /* return the DIB width if it is a Win 3.0 DIB */
   if (lpbmi->biSize == sizeof(BITMAPINFOHEADER))
      return lpbmi->biWidth;
   else  /* it is an OS/2 DIB, so return its width */
      return (DWORD)lpbmc->bcWidth;
}


/*************************************************************************
 *
 * DIBHeight()
 *
 * Parameter:
 *
 * LPSTR lpDIB      - pointer to packed-DIB memory block
 *
 * Return Value:
 *
 * DWORD            - height of the DIB
 *
 * Description:
 *
 * This function gets the height of the DIB from the BITMAPINFOHEADER
 * height field if it is a Windows 3.0-style DIB or from the BITMAPCOREHEADER
 * height field if it is an OS/2-style DIB.
 *
 * History:   Date      Author               Reason
 *            6/01/91   Garrett McAuliffe    Created
 *            9/15/91   Patrick Schreiber    Added header and comments
 *
 ************************************************************************/


DWORD FAR DIBHeight(LPSTR lpDIB)
{
   LPBITMAPINFOHEADER lpbmi;  // pointer to a Win 3.0-style DIB
   LPBITMAPCOREHEADER lpbmc;  // pointer to an OS/2-style DIB

   /* point to the header (whether OS/2 or Win 3.0 */

   lpbmi = (LPBITMAPINFOHEADER)lpDIB;
   lpbmc = (LPBITMAPCOREHEADER)lpDIB;

   /* return the DIB height if it is a Win 3.0 DIB */
   if (lpbmi->biSize == sizeof(BITMAPINFOHEADER))
      return lpbmi->biHeight;
   else  /* it is an OS/2 DIB, so return its height */
      return (DWORD)lpbmc->bcHeight;
}


/*************************************************************************
 *
 * PaletteSize()
 *
 * Parameter:
 *
 * LPSTR lpDIB      - pointer to packed-DIB memory block
 *
 * Return Value:
 *
 * WORD             - size of the color palette of the DIB
 *
 * Description:
 *
 * This function gets the size required to store the DIB's palette by
 * multiplying the number of colors by the size of an RGBQUAD (for a
 * Windows 3.0-style DIB) or by the size of an RGBTRIPLE (for an OS/2-
 * style DIB).
 *
 * History:   Date      Author             Reason
 *            6/01/91   Garrett McAuliffe  Created
 *            9/15/91   Patrick Schreiber  Added header and comments
 *
 ************************************************************************/


WORD FAR PaletteSize(LPSTR lpDIB)
{
   /* calculate the size required by the palette */
   if (IS_WIN30_DIB (lpDIB))
      return (DIBNumColors(lpDIB) * sizeof(RGBQUAD));
   else
      return (DIBNumColors(lpDIB) * sizeof(RGBTRIPLE));
}


/*************************************************************************
 *
 * DIBNumColors()
 *
 * Parameter:
 *
 * LPSTR lpDIB      - pointer to packed-DIB memory block
 *
 * Return Value:
 *
 * WORD             - number of colors in the color table
 *
 * Description:
 *
 * This function calculates the number of colors in the DIB's color table
 * by finding the bits per pixel for the DIB (whether Win3.0 or OS/2-style
 * DIB). If bits per pixel is 1: colors=2, if 4: colors=16, if 8: colors=256,
 * if 24, no colors in color table.
 *
 * History:   Date      Author               Reason
 *            6/01/91   Garrett McAuliffe    Created
 *            9/15/91   Patrick Schreiber    Added header and comments
 *
 ************************************************************************/


WORD FAR DIBNumColors(LPSTR lpDIB)
{
   WORD wBitCount;  // DIB bit count

   /*  If this is a Windows-style DIB, the number of colors in the
    *  color table can be less than the number of bits per pixel
    *  allows for (i.e. lpbi->biClrUsed can be set to some value).
    *  If this is the case, return the appropriate value.
    */

   if (IS_WIN30_DIB(lpDIB))
   {
      DWORD dwClrUsed;

      dwClrUsed = ((LPBITMAPINFOHEADER)lpDIB)->biClrUsed;
      if (dwClrUsed)
     return (WORD)dwClrUsed;
   }

   /*  Calculate the number of colors in the color table based on
    *  the number of bits per pixel for the DIB.
    */
   if (IS_WIN30_DIB(lpDIB))
      wBitCount = ((LPBITMAPINFOHEADER)lpDIB)->biBitCount;
   else
      wBitCount = ((LPBITMAPCOREHEADER)lpDIB)->bcBitCount;

   /* return number of colors based on bits per pixel */
   switch (wBitCount)
      {
   case 1:
      return 2;

   case 4:
      return 16;

   case 8:
      return 256;

   default:
      return 0;
      }
}


/*************************************************************************
 *
 * CreateDIBPalette()
 *
 * Parameter:
 *
 * HDIB hDIB        - specifies the DIB
 *
 * Return Value:
 *
 * HPALETTE         - specifies the palette
 *
 * Description:
 *
 * This function creates a palette from a DIB by allocating memory for the
 * logical palette, reading and storing the colors from the DIB's color table
 * into the logical palette, creating a palette from this logical palette,
 * and then returning the palette's handle. This allows the DIB to be
 * displayed using the best possible colors (important for DIBs with 256 or
 * more colors).
 *
 * History:   Date      Author               Reason
 *            6/01/91   Garrett McAuliffe    Created
 *            9/15/91   Patrick Schreiber    Added header and comments
 *
 ************************************************************************/


HPALETTE FAR CreateDIBPalette(HDIB hDIB)
{
   LPLOGPALETTE lpPal;      // pointer to a logical palette
   HANDLE hLogPal;          // handle to a logical palette
   HPALETTE hPal = NULL;    // handle to a palette
   int i, wNumColors;       // loop index, number of colors in color table
   LPSTR lpbi;              // pointer to packed-DIB
   LPBITMAPINFO lpbmi;      // pointer to BITMAPINFO structure (Win3.0)
   LPBITMAPCOREINFO lpbmc;  // pointer to BITMAPCOREINFO structure (OS/2)
   BOOL bWinStyleDIB;       // flag which signifies whether this is a Win3.0 DIB

   /* if handle to DIB is invalid, return NULL */

   if (!hDIB)
      return NULL;

   /* lock DIB memory block and get a pointer to it */
   lpbi = (LPSTR)GlobalLock(hDIB);

   /* get pointer to BITMAPINFO (Win 3.0) */
   lpbmi = (LPBITMAPINFO)lpbi;

   /* get pointer to BITMAPCOREINFO (OS/2 1.x) */
   lpbmc = (LPBITMAPCOREINFO)lpbi;

   /* get the number of colors in the DIB */
   wNumColors = DIBNumColors(lpbi);

   /* is this a Win 3.0 DIB? */
   bWinStyleDIB = IS_WIN30_DIB(lpbi);
   if (wNumColors)
   {
      /* allocate memory block for logical palette */
      hLogPal = GlobalAlloc(GHND, sizeof(LOGPALETTE) + sizeof(PALETTEENTRY) *
                wNumColors);

      /* if not enough memory, clean up and return NULL */
      if (!hLogPal)
      {
     GlobalUnlock(hDIB);
     return NULL;
      }

      /* lock memory block and get pointer to it */
      lpPal = (LPLOGPALETTE)GlobalLock(hLogPal);

      /* set version and number of palette entries */
      lpPal->palVersion = PALVERSION;
      lpPal->palNumEntries = wNumColors;

      /*  store RGB triples (if Win 3.0 DIB) or RGB quads (if OS/2 DIB)
       *  into palette
       */
      for (i = 0; i < wNumColors; i++)
      {
     if (bWinStyleDIB)
     {
        lpPal->palPalEntry[i].peRed = lpbmi->bmiColors[i].rgbRed;
        lpPal->palPalEntry[i].peGreen = lpbmi->bmiColors[i].rgbGreen;
        lpPal->palPalEntry[i].peBlue = lpbmi->bmiColors[i].rgbBlue;
        lpPal->palPalEntry[i].peFlags = 0;
     }
     else
     {
        lpPal->palPalEntry[i].peRed = lpbmc->bmciColors[i].rgbtRed;
        lpPal->palPalEntry[i].peGreen = lpbmc->bmciColors[i].rgbtGreen;
        lpPal->palPalEntry[i].peBlue = lpbmc->bmciColors[i].rgbtBlue;
        lpPal->palPalEntry[i].peFlags = 0;
     }
      }

      /* create the palette and get handle to it */
      hPal = CreatePalette(lpPal);

      /* if error getting handle to palette, clean up and return NULL */
      if (!hPal)
      {
     GlobalUnlock(hLogPal);
     GlobalFree(hLogPal);
     return NULL;
      }
   }

   /* clean up */
   GlobalUnlock(hLogPal);
   GlobalFree(hLogPal);
   GlobalUnlock(hDIB);

   /* return handle to DIB's palette */
   return hPal;
}


/*************************************************************************
 *
 * DIBToBitmap()
 *
 * Parameters:
 *
 * HDIB hDIB        - specifies the DIB to convert
 *
 * HPALETTE hPal    - specifies the palette to use with the bitmap
 *
 * Return Value:
 *
 * HBITMAP          - identifies the device-dependent bitmap
 *
 * Description:
 *
 * This function creates a bitmap from a DIB using the specified palette.
 * If no palette is specified, default is used.
 *
 * NOTE:
 *
 * The bitmap returned from this funciton is always a bitmap compatible
 * with the screen (e.g. same bits/pixel and color planes) rather than
 * a bitmap with the same attributes as the DIB.  This behavior is by
 * design, and occurs because this function calls CreateDIBitmap to
 * do its work, and CreateDIBitmap always creates a bitmap compatible
 * with the hDC parameter passed in (because it in turn calls
 * CreateCompatibleBitmap).
 *
 * So for instance, if your DIB is a monochrome DIB and you call this
 * function, you will not get back a monochrome HBITMAP -- you will
 * get an HBITMAP compatible with the screen DC, but with only 2
 * colors used in the bitmap.
 *
 * If your application requires a monochrome HBITMAP returned for a
 * monochrome DIB, use the function SetDIBits().
 *
 * Also, the DIBpassed in to the function is not destroyed on exit. This
 * must be done later, once it is no longer needed.
 *
 * History:   Date      Author               Reason
 *            6/01/91   Garrett McAuliffe    Created
 *            9/15/91   Patrick Schreiber    Added header and comments
 *            3/27/92   Mark Bader           Added comments about resulting
 *                                           bitmap format
 *
 ************************************************************************/


HBITMAP FAR DIBToBitmap(HDIB hDIB, HPALETTE hPal)
{
   LPSTR lpDIBHdr, lpDIBBits;  // pointer to DIB header, pointer to DIB bits
   HBITMAP hBitmap;            // handle to device-dependent bitmap
   HDC hDC;                    // handle to DC
   HPALETTE hOldPal = NULL;    // handle to a palette

   /* if invalid handle, return NULL */

   if (!hDIB)
      return NULL;

   /* lock memory block and get a pointer to it */
   lpDIBHdr = (LPSTR)GlobalLock(hDIB);

   /* get a pointer to the DIB bits */
   lpDIBBits = FindDIBBits(lpDIBHdr);

   /* get a DC */
   hDC = GetDC(NULL);
   if (!hDC)
   {
      /* clean up and return NULL */
      GlobalUnlock(hDIB);
      return NULL;
   }

   /* select and realize palette */
   if (hPal)
      hOldPal = SelectPalette(hDC, hPal, FALSE);
   RealizePalette(hDC);

   /* create bitmap from DIB info. and bits */
   hBitmap = CreateDIBitmap(hDC, (LPBITMAPINFOHEADER)lpDIBHdr, CBM_INIT,
                lpDIBBits, (LPBITMAPINFO)lpDIBHdr, DIB_RGB_COLORS);

   /* restore previous palette */
   if (hOldPal)
      SelectPalette(hDC, hOldPal, FALSE);

   /* clean up */
   ReleaseDC(NULL, hDC);
   GlobalUnlock(hDIB);

   /* return handle to the bitmap */
   return hBitmap;
}


/*************************************************************************
 *
 * BitmapToDIB()
 *
 * Parameters:
 *
 * HBITMAP hBitmap  - specifies the bitmap to convert
 *
 * HPALETTE hPal    - specifies the palette to use with the bitmap
 *
 * Return Value:
 *
 * HDIB             - identifies the device-dependent bitmap
 *
 * Description:
 *
 * This function creates a DIB from a bitmap using the specified palette.
 *
 * History:   Date      Author               Reason
 *            6/01/91   Garrett McAuliffe    Created
 *            9/15/91   Patrick Schreiber    Added header and comments
 *            12/10/91  Patrick Schreiber    Added bits per pixel validation
 *                                           and check GetObject return value
 *
 ************************************************************************/


HDIB FAR BitmapToDIB(HBITMAP hBitmap, HPALETTE hPal)
{
   BITMAP bm;                   // bitmap structure
   BITMAPINFOHEADER bi;         // bitmap header
   BITMAPINFOHEADER FAR *lpbi;  // pointer to BITMAPINFOHEADER
   DWORD dwLen;                 // size of memory block
   HANDLE hDIB, h;              // handle to DIB, temp handle
   HDC hDC;                     // handle to DC
   WORD biBits;                 // bits per pixel

   /* check if bitmap handle is valid */

   if (!hBitmap)
      return NULL;

   /* fill in BITMAP structure, return NULL if it didn't work */
   if (!GetObject(hBitmap, sizeof(bm), (LPSTR)&bm))
      return NULL;

   /* if no palette is specified, use default palette */
   if (hPal == NULL)
      hPal = (HPALETTE)GetStockObject(DEFAULT_PALETTE);

   /* calculate bits per pixel */
   biBits = bm.bmPlanes * bm.bmBitsPixel;

   /* make sure bits per pixel is valid */
   if (biBits <= 1)
      biBits = 1;
   else if (biBits <= 4)
      biBits = 4;
   else if (biBits <= 8)
      biBits = 8;
   else /* if greater than 8-bit, force to 24-bit */
      biBits = 24;

   /* initialize BITMAPINFOHEADER */
   bi.biSize = sizeof(BITMAPINFOHEADER);
   bi.biWidth = bm.bmWidth;
   bi.biHeight = bm.bmHeight;
   bi.biPlanes = 1;
   bi.biBitCount = biBits;
   bi.biCompression = BI_RGB;
   bi.biSizeImage = 0;
   bi.biXPelsPerMeter = 0;
   bi.biYPelsPerMeter = 0;
   bi.biClrUsed = 0;
   bi.biClrImportant = 0;

   /* calculate size of memory block required to store BITMAPINFO */
   dwLen = bi.biSize + PaletteSize((LPSTR)&bi);

   /* get a DC */
   hDC = GetDC(NULL);

   /* select and realize our palette */
   hPal = SelectPalette(hDC, hPal, FALSE);
   RealizePalette(hDC);

   /* alloc memory block to store our bitmap */
   hDIB = GlobalAlloc(GHND, dwLen);

   /* if we couldn't get memory block */
   if (!hDIB)
   {
      /* clean up and return NULL */
      SelectPalette(hDC, hPal, TRUE);
      RealizePalette(hDC);
      ReleaseDC(NULL, hDC);
      return NULL;
   }

   /* lock memory and get pointer to it */
   lpbi = (BITMAPINFOHEADER FAR *)GlobalLock(hDIB);

   /* use our bitmap info. to fill BITMAPINFOHEADER */
   *lpbi = bi;

   /*  call GetDIBits with a NULL lpBits param, so it will calculate the
    *  biSizeImage field for us
    */
   GetDIBits(hDC, hBitmap, 0, (WORD)bi.biHeight, NULL, (LPBITMAPINFO)lpbi,
         DIB_RGB_COLORS);

   /* get the info. returned by GetDIBits and unlock memory block */
   bi = *lpbi;
   GlobalUnlock(hDIB);

   /* if the driver did not fill in the biSizeImage field, make one up */
   if (bi.biSizeImage == 0)
      bi.biSizeImage = WIDTHBYTES((DWORD)bm.bmWidth * biBits) * bm.bmHeight;

   /* realloc the buffer big enough to hold all the bits */
   dwLen = bi.biSize + PaletteSize((LPSTR)&bi) + bi.biSizeImage;
   if (h = GlobalReAlloc(hDIB, dwLen, 0))
      hDIB = h;
   else
   {
      /* clean up and return NULL */
      GlobalFree(hDIB);
      hDIB = NULL;
      SelectPalette(hDC, hPal, TRUE);
      RealizePalette(hDC);
      ReleaseDC(NULL, hDC);
      return NULL;
   }

   /* lock memory block and get pointer to it */
   lpbi = (BITMAPINFOHEADER FAR *)GlobalLock(hDIB);

   /*  call GetDIBits with a NON-NULL lpBits param, and actualy get the
    *  bits this time
    */
   if (GetDIBits(hDC, hBitmap, 0, (WORD)bi.biHeight, (LPSTR)lpbi + (WORD)lpbi
         ->biSize + PaletteSize((LPSTR)lpbi), (LPBITMAPINFO)lpbi,
         DIB_RGB_COLORS) == 0)
   {
      /* clean up and return NULL */
      GlobalUnlock(hDIB);
      GlobalFree(hDIB);//JLM 1/21/99
      hDIB = NULL;
      SelectPalette(hDC, hPal, TRUE);
      RealizePalette(hDC);
      ReleaseDC(NULL, hDC);
      return NULL;
   }
   bi = *lpbi;

   /* clean up */
   GlobalUnlock(hDIB);
   SelectPalette(hDC, hPal, TRUE);
   RealizePalette(hDC);
   ReleaseDC(NULL, hDC);

   /* return handle to the DIB */
   return hDIB;
}


/*************************************************************************
 *
 * PalEntriesOnDevice()
 *
 * Parameter:
 *
 * HDC hDC          - device context
 *
 * Return Value:
 *
 * int              - number of palette entries on device
 *
 * Description:
 *
 * This function gets the number of palette entries on the specified device
 *
 * History:   Date      Author               Reason
 *            6/01/91   Garrett McAuliffe    Created
 *            9/15/91   Patrick Schreiber    Added header and comments
 *
 ************************************************************************/


int FAR PalEntriesOnDevice(HDC hDC)
{
   int nColors;  // number of colors

   /*  Find out the number of palette entries on this
    *  device.
    */

   nColors = GetDeviceCaps(hDC, SIZEPALETTE);

   /*  For non-palette devices, we'll use the # of system
    *  colors for our palette size.
    */
   if (!nColors)
      nColors = GetDeviceCaps(hDC, NUMCOLORS);
   // assert(nColors);
   if (!nColors) SysBeep(1);
   return nColors;
}


/*************************************************************************
 *
 * GetSystemPalette()
 *
 * Parameters:
 *
 * None
 *
 * Return Value:
 *
 * HPALETTE         - handle to a copy of the current system palette
 *
 * Description:
 *
 * This function returns a handle to a palette which represents the system
 * palette.  The system RGB values are copied into our logical palette using
 * the GetSystemPaletteEntries function.  
 *
 * History:   
 *            
 *    Date      Author               Reason        
 *    6/01/91   Garrett McAuliffe    Created        
 *    9/15/91   Patrick Schreiber    Added header and comments
 *    12/20/91  Mark Bader           Added GetSystemPaletteEntries call
 *
 ************************************************************************/


HPALETTE FAR GetSystemPalette(void)
{
   HDC hDC;                // handle to a DC
   static HPALETTE hPal = NULL;   // handle to a palette
   HANDLE hLogPal;         // handle to a logical palette
   LPLOGPALETTE lpLogPal;  // pointer to a logical palette
   int nColors;            // number of colors

	return (HPALETTE)GetStockObject(DEFAULT_PALETTE); // MAK
	
   /* Find out how many palette entries we want. */

   hDC = GetDC(NULL);
   if (!hDC)
      return NULL;
   nColors = PalEntriesOnDevice(hDC);   // Number of palette entries

   /* Allocate room for the palette and lock it. */
   hLogPal = GlobalAlloc(GHND, sizeof(LOGPALETTE) + nColors * sizeof(
             PALETTEENTRY));

   /* if we didn't get a logical palette, return NULL */
   if (!hLogPal)
      return NULL;

   /* get a pointer to the logical palette */
   lpLogPal = (LPLOGPALETTE)GlobalLock(hLogPal);

   /* set some important fields */
   lpLogPal->palVersion = PALVERSION;
   lpLogPal->palNumEntries = nColors;

   /* Copy the current system palette into our logical palette */

   GetSystemPaletteEntries(hDC, 0, nColors, 
                           (LPPALETTEENTRY)(lpLogPal->palPalEntry));

   /*  Go ahead and create the palette.  Once it's created,
    *  we no longer need the LOGPALETTE, so free it.
    */

   hPal = CreatePalette(lpLogPal);

   /* clean up */
   GlobalUnlock(hLogPal);
   GlobalFree(hLogPal);
   ReleaseDC(NULL, hDC);

   return hPal;
}


/*************************************************************************
 *
 * AllocRoomForDIB()
 *
 * Parameters:
 *
 * BITMAPINFOHEADER - bitmap info header stucture
 *
 * HBITMAP          - handle to the bitmap
 *
 * Return Value:
 *
 * HDIB             - handle to memory block
 *
 * Description:
 *
 *  This routine takes a BITMAPINOHEADER, and returns a handle to global
 *  memory which can contain a DIB with that header.  It also initializes
 *  the header portion of the global memory.  GetDIBits() is used to determine
 *  the amount of room for the DIB's bits.  The total amount of memory
 *  needed = sizeof(BITMAPINFOHEADER) + size of color table + size of bits.
 *
 * History:   Date      Author               Reason
 *            6/01/91   Garrett McAuliffe    Created
 *            12/11/91  Patrick Schreiber    Added header and some comments
 *
 ************************************************************************/

HANDLE AllocRoomForDIB(BITMAPINFOHEADER bi, HBITMAP hBitmap)
{
   DWORD              dwLen;
   HANDLE             hDIB;
   HDC                hDC;
   LPBITMAPINFOHEADER lpbi;
   HANDLE             hTemp;

   /* Figure out the size needed to hold the BITMAPINFO structure
    * (which includes the BITMAPINFOHEADER and the color table).
    */

   dwLen = bi.biSize + PaletteSize((LPSTR) &bi);
   hDIB  = GlobalAlloc(GHND,dwLen);

   /* Check that DIB handle is valid */
   if (!hDIB)
      return NULL;

   /* Set up the BITMAPINFOHEADER in the newly allocated global memory,
    * then call GetDIBits() with lpBits = NULL to have it fill in the
    * biSizeImage field for us.
    */
   lpbi  = (BITMAPINFOHEADER FAR *)GlobalLock(hDIB);
   *lpbi = bi;

   hDC   = GetDC(NULL);
   GetDIBits(hDC, hBitmap, 0, (WORD) bi.biHeight,
          NULL, (LPBITMAPINFO) lpbi, DIB_RGB_COLORS);
   ReleaseDC(NULL, hDC);

   /* If the driver did not fill in the biSizeImage field,
    * fill it in -- NOTE: this is a bug in the driver!
    */
   if (lpbi->biSizeImage == 0)
      lpbi->biSizeImage = WIDTHBYTES((DWORD)lpbi->biWidth * lpbi->biBitCount) *
              lpbi->biHeight;

   /* Get the size of the memory block we need */
   dwLen = lpbi->biSize + PaletteSize((LPSTR) &bi) + lpbi->biSizeImage;

   /* Unlock the memory block */
   GlobalUnlock(hDIB);

   /* ReAlloc the buffer big enough to hold all the bits */
   if (hTemp = GlobalReAlloc(hDIB,dwLen,0))
      return hTemp;
   else
      {
      /* Else free memory block and return failure */
      GlobalFree(hDIB);
      return NULL;
      }
}


/*************************************************************************
 *
 * ChangeDIBFormat()
 *
 * Parameter:
 *
 * HDIB             - handle to packed-DIB in memory
 *
 * WORD             - desired bits per pixel
 *
 * DWORD            - desired compression format
 *
 * Return Value:
 *
 * HDIB             - handle to the new DIB if successful, else NULL
 *
 * Description:
 *
 * This function will convert the bits per pixel and/or the compression
 * format of the specified DIB. Note: If the conversion was unsuccessful,
 * we return NULL. The original DIB is left alone. Don't use code like the
 * following:
 *
 *    hMyDIB = ChangeDIBFormat(hMyDIB, 8, BI_RLE4);
 *
 * The conversion will fail, but hMyDIB will now be NULL and the original
 * DIB will now hang around in memory. We could have returned the old
 * DIB, but we wanted to allow the programmer to check whether this
 * conversion succeeded or failed.
 *
 * History:   
 *            
 *   Date      Author             Reason         
 *   6/01/91   Garrett McAuliffe  Created         
 *   12/10/91  Patrick Schreiber  Modified from converting RGB to RLE8        
 *                                  to converting RGB/RLE to RGB/RLE.         
 *                                  Added wBitCount and dwCompression         
 *                                  parameters. Also added header and         
 *                                  comments.         
 *
 ************************************************************************/

HDIB FAR ChangeDIBFormat(HDIB hDIB, WORD wBitCount, DWORD dwCompression)
{
   HDC                hDC;             // Handle to DC
   HBITMAP            hBitmap;         // Handle to bitmap
   BITMAP             Bitmap;          // BITMAP data structure
   BITMAPINFOHEADER   bi;              // Bitmap info header
   LPBITMAPINFOHEADER lpbi;            // Pointer to bitmap info
   HDIB               hNewDIB = NULL;  // Handle to new DIB
   HPALETTE           hPal, hOldPal;   // Handle to palette, prev pal
   WORD               DIBBPP, NewBPP;  // DIB bits per pixel, new bpp
   DWORD              DIBComp, NewComp;// DIB compression, new compression

   /* Check for a valid DIB handle */
   if (!hDIB)
      return NULL;

   /* Get the old DIB's bits per pixel and compression format */
   lpbi = (LPBITMAPINFOHEADER)GlobalLock(hDIB);
   DIBBPP = ((LPBITMAPINFOHEADER)lpbi)->biBitCount;
   DIBComp = ((LPBITMAPINFOHEADER)lpbi)->biCompression;
   GlobalUnlock(hDIB);

   /* Validate wBitCount and dwCompression
    * They must match correctly (i.e., BI_RLE4 and 4 BPP or
    * BI_RLE8 and 8BPP, etc.) or we return failure */
   if (wBitCount == 0)
      {
      NewBPP = DIBBPP;
      if ((dwCompression == BI_RLE4 && NewBPP == 4) ||
      (dwCompression == BI_RLE8 && NewBPP == 8) ||
      (dwCompression == BI_RGB))
     NewComp = dwCompression;
      else
     return NULL;
      }
   else if (wBitCount == 1 && dwCompression == BI_RGB)
      {
      NewBPP = wBitCount;
      NewComp = BI_RGB;
      }
   else if (wBitCount == 4)
      {
      NewBPP = wBitCount;
      if (dwCompression == BI_RGB || dwCompression == BI_RLE4)
     NewComp = dwCompression;
      else
     return NULL;
      }
   else if (wBitCount == 8)
      {
      NewBPP = wBitCount;
      if (dwCompression == BI_RGB || dwCompression == BI_RLE8)
     NewComp = dwCompression;
      else
     return NULL;
      }
   else if (wBitCount == 24 && dwCompression == BI_RGB)
      {
      NewBPP = wBitCount;
      NewComp = BI_RGB;
      }
   else
      return NULL;

   /* Save the old DIB's palette */
   hPal = CreateDIBPalette(hDIB);
   if (!hPal)
      return NULL;

   /* Convert old DIB to a bitmap */
   hBitmap = DIBToBitmap(hDIB, hPal);
   if (!hBitmap)
      {
      DeleteObject(hPal);
      return NULL;
      }

   /* Get info about the bitmap */
   GetObject(hBitmap, sizeof(BITMAP), (LPSTR)&Bitmap);

   /* Fill in the BITMAPINFOHEADER appropriately */
   bi.biSize               = sizeof(BITMAPINFOHEADER);
   bi.biWidth              = Bitmap.bmWidth;
   bi.biHeight             = Bitmap.bmHeight;
   bi.biPlanes             = 1;
   bi.biBitCount           = NewBPP;
   bi.biCompression        = NewComp;
   bi.biSizeImage          = 0;
   bi.biXPelsPerMeter      = 0;
   bi.biYPelsPerMeter      = 0;
   bi.biClrUsed            = 0;
   bi.biClrImportant       = 0;

   /* Go allocate room for the new DIB */
   hNewDIB = AllocRoomForDIB(bi, hBitmap);
   if (!hNewDIB)
      return NULL;

   /* Get a pointer to the new DIB */
   lpbi = (BITMAPINFOHEADER FAR *)GlobalLock(hNewDIB);

   /* Get a DC and select/realize our palette in it */
   hDC  = GetDC(NULL);
   hOldPal = SelectPalette(hDC, hPal, FALSE);
   RealizePalette(hDC);

   /* Call GetDIBits and get the new DIB bits */
   if (!GetDIBits(hDC, hBitmap, 0, (WORD) lpbi->biHeight,
       (LPSTR)lpbi + (WORD)lpbi->biSize + PaletteSize((LPSTR)lpbi),
       (LPBITMAPINFO)lpbi, DIB_RGB_COLORS))
      {
      GlobalUnlock(hNewDIB);
      GlobalFree(hNewDIB);
      hNewDIB = NULL;
      }

   /* Clean up and return */
   SelectPalette(hDC, hOldPal, TRUE);
   RealizePalette(hDC);
   ReleaseDC(NULL, hDC);

   if (hNewDIB)
      /* Unlock the new DIB's memory block */
      GlobalUnlock(hNewDIB);

   DeleteObject(hBitmap);
   DeleteObject(hPal);

   return hNewDIB;
}


/*************************************************************************
 *
 * ChangeBitmapFormat()
 *
 * Parameter:
 *
 * HBITMAP          - handle to a bitmap
 *
 * WORD             - desired bits per pixel
 *
 * DWORD            - desired compression format
 *
 * HPALETTE         - handle to palette
 *
 * Return Value:
 *
 * HDIB             - handle to the new DIB if successful, else NULL
 *
 * Description:
 *
 * This function will convert a bitmap to the specified bits per pixel
 * and compression format. The bitmap and it's palette will remain
 * after calling this function.
 *
 * History:   
 *            
 *   Date      Author             Reason         
 *   6/01/91   Garrett McAuliffe  Created         
 *   12/10/91  Patrick Schreiber  Modified from converting RGB to RLE8         
 *                                 to converting RGB/RLE to RGB/RLE.         
 *                                 Added wBitCount and dwCompression         
 *                                 parameters. Also added header and         
 *                                 comments.         
 *   12/11/91  Patrick Schreiber  Destroy old DIB if conversion was         
 *                                 successful.         
 *   12/16/91  Patrick Schreiber  Modified from converting DIB to new         
 *                                 DIB to bitmap to new DIB. Added palette
 *                                 parameter.
 *
 ************************************************************************/

HDIB FAR ChangeBitmapFormat(HBITMAP  hBitmap,
            WORD     wBitCount,
            DWORD    dwCompression,
            HPALETTE hPal)
{
   HDC                hDC;          // Screen DC
   HDIB               hNewDIB=NULL; // Handle to new DIB
   BITMAP             Bitmap;       // BITMAP data structure
   BITMAPINFOHEADER   bi;           // Bitmap info. header
   LPBITMAPINFOHEADER lpbi;         // Pointer to bitmap header
   HPALETTE           hOldPal=NULL; // Handle to palette
   WORD               NewBPP;       // New bits per pixel
   DWORD              NewComp;      // New compression format

   /* Check for a valid bitmap handle */
   if (!hBitmap)
      return NULL;

   /* Validate wBitCount and dwCompression
    * They must match correctly (i.e., BI_RLE4 and 4 BPP or
    * BI_RLE8 and 8BPP, etc.) or we return failure
    */
   if (wBitCount == 0)
      {
      NewComp = dwCompression;
      if (NewComp == BI_RLE4)
     NewBPP = 4;
      else if (NewComp == BI_RLE8)
     NewBPP = 8;
      else /* Not enough info */
     return NULL;
      }
   else if (wBitCount == 1 && dwCompression == BI_RGB)
      {
      NewBPP = wBitCount;
      NewComp = BI_RGB;
      }
   else if (wBitCount == 4)
      {
      NewBPP = wBitCount;
      if (dwCompression == BI_RGB || dwCompression == BI_RLE4)
         NewComp = dwCompression;
      else
         return NULL;
      }
   else if (wBitCount == 8)
      {
      NewBPP = wBitCount;
      if (dwCompression == BI_RGB || dwCompression == BI_RLE8)
         NewComp = dwCompression;
      else
         return NULL;
      }
   else if (wBitCount == 24 && dwCompression == BI_RGB)
      {
      NewBPP = wBitCount;
      NewComp = BI_RGB;
      }
   else
      return NULL;

   /* Get info about the bitmap */
   GetObject(hBitmap, sizeof(BITMAP), (LPSTR)&Bitmap);

   /* Fill in the BITMAPINFOHEADER appropriately */
   bi.biSize               = sizeof(BITMAPINFOHEADER);
   bi.biWidth              = Bitmap.bmWidth;
   bi.biHeight             = Bitmap.bmHeight;
   bi.biPlanes             = 1;
   bi.biBitCount           = NewBPP;
   bi.biCompression        = NewComp;
   bi.biSizeImage          = 0;
   bi.biXPelsPerMeter      = 0;
   bi.biYPelsPerMeter      = 0;
   bi.biClrUsed            = 0;
   bi.biClrImportant       = 0;

   /* Go allocate room for the new DIB */
   hNewDIB = AllocRoomForDIB(bi, hBitmap);
   if (!hNewDIB)
      return NULL;

   /* Get a pointer to the new DIB */
   lpbi = (BITMAPINFOHEADER FAR *)GlobalLock(hNewDIB);

   /* If we have a palette, get a DC and select/realize it */
   if (hPal)
   {
      hDC  = GetDC(NULL);
      hOldPal = SelectPalette(hDC, hPal, FALSE);
      RealizePalette(hDC);
   }

   /* Call GetDIBits and get the new DIB bits */
   if (!GetDIBits(hDC, hBitmap, 0, (WORD) lpbi->biHeight,
       (LPSTR)lpbi + (WORD)lpbi->biSize + PaletteSize((LPSTR)lpbi),
       (LPBITMAPINFO)lpbi, DIB_RGB_COLORS))
      {
      GlobalUnlock(hNewDIB);
      GlobalFree(hNewDIB);
      hNewDIB = NULL;
      }

   /* Clean up and return */
   if (hOldPal)
   {
      SelectPalette(hDC, hOldPal, TRUE);
      RealizePalette(hDC);
      ReleaseDC(NULL, hDC);
   }

   if (hNewDIB)
      {
      /* Unlock the new DIB's memory block */
      GlobalUnlock(hNewDIB);
      }

   return hNewDIB;
}

//////////////////////////////////////////////////////////////////////////////////

/*
 * Dib Header Marker - used in writing DIBs to files
 */
#define DIB_HEADER_MARKER   ((WORD) ('M' << 8) | 'B')


/*********************************************************************
 *
 * Local Function Prototypes
 *
 *********************************************************************/


HANDLE ReadDIBFile(int);
BOOL MyRead(int, LPSTR, DWORD);
BOOL SaveDIBFile(void);
BOOL WriteDIB(LPSTR, HANDLE);
DWORD PASCAL MyWrite(int, VOID FAR *, DWORD);

/*************************************************************************
 *
 * LoadDIB()
 *
 * Loads the specified DIB from a file, allocates memory for it,
 * and reads the disk file into the memory.
 *
 *
 * Parameters:
 *
 * LPSTR lpFileName - specifies the file to load a DIB from
 *
 * Returns: A handle to a DIB, or NULL if unsuccessful.
 *
 * NOTE: The DIB API were not written to handle OS/2 DIBs; This
 * function will reject any file that is not a Windows DIB.
 *
 * History:   Date      Author       Reason
 *            9/15/91   Mark Bader   Based on DIBVIEW
 *
 *************************************************************************/


HDIB FAR LoadDIB(LPSTR lpFileName)
{
   HDIB hDIB;
   int hFile;
   OFSTRUCT ofs;

   /*
    * Set the cursor to a hourglass, in case the loading operation
    * takes more than a sec, the user will know what's going on.
    */

   SetCursor(LoadCursor(NULL, IDC_WAIT));
   if ((hFile = OpenFile(lpFileName, &ofs, OF_READ)) != -1)
   {
      hDIB = ReadDIBFile(hFile);
      _lclose(hFile);
      SetCursor(LoadCursor(NULL, IDC_ARROW));
      return hDIB;
   }
   else
   {
      DIBError(ERR_FILENOTFOUND);
      SetCursor(LoadCursor(NULL, IDC_ARROW));
      return NULL;
   }
}


/*************************************************************************
 *
 * SaveDIB()
 *
 * Saves the specified DIB into the specified file name on disk.  No
 * error checking is done, so if the file already exists, it will be
 * written over.
 *
 * Parameters:
 *
 * HDIB hDib - Handle to the dib to save
 *
 * LPSTR lpFileName - pointer to full pathname to save DIB under
 *
 * Return value: 0 if successful, or one of:
 *        ERR_INVALIDHANDLE
 *        ERR_OPEN
 *        ERR_LOCK
 *
 * History:
 *
 * NOTE: The DIB API were not written to handle OS/2 DIBs, so this
 * function will not save a file if it is not a Windows DIB.
 *
 * History:   Date      Author       Reason
 *            9/15/91   Mark Bader   Taken from DIBVIEW (which was taken
 *                                      from SHOWDIB)
 *            1/30/92   Mark Bader   Fixed problem of writing too many 
 *                                      bytes to the file
 *            6/24/92   Mark Bader   Added check for OS/2 DIB
 *
 *************************************************************************/


WORD FAR SaveDIB(HDIB hDib, LPSTR lpFileName)
{
   BITMAPFILEHEADER bmfHdr; // Header for Bitmap file
   LPBITMAPINFOHEADER lpBI;   // Pointer to DIB info structure
   int fh;     // file handle for opened file
   OFSTRUCT of;     // OpenFile structure
   DWORD dwDIBSize;
   DWORD dwError;   // Error return from MyWrite

   if (!hDib)
      return ERR_INVALIDHANDLE;
   fh = OpenFile(lpFileName, &of, OF_CREATE | OF_READWRITE);
   if (fh == -1)
      return ERR_OPEN;

   /*
    * Get a pointer to the DIB memory, the first of which contains
    * a BITMAPINFO structure
    */
   lpBI = (LPBITMAPINFOHEADER)GlobalLock(hDib);
   if (!lpBI)
      return ERR_LOCK;

   // Check to see if we're dealing with an OS/2 DIB.  If so, don't
   // save it because our functions aren't written to deal with these
   // DIBs.

   if (lpBI->biSize != sizeof(BITMAPINFOHEADER))
   {
     GlobalUnlock(hDib);
     return ERR_NOT_DIB;
   }

   /*
    * Fill in the fields of the file header
    */

   /* Fill in file type (first 2 bytes must be "BM" for a bitmap) */
   bmfHdr.bfType = DIB_HEADER_MARKER;  // "BM"

   // Calculating the size of the DIB is a bit tricky (if we want to
   // do it right).  The easiest way to do this is to call GlobalSize()
   // on our global handle, but since the size of our global memory may have
   // been padded a few bytes, we may end up writing out a few too
   // many bytes to the file (which may cause problems with some apps,
   // like HC 3.0).
   //
   // So, instead let's calculate the size manually.
   //
   // To do this, find size of header plus size of color table.  Since the
   // first DWORD in both BITMAPINFOHEADER and BITMAPCOREHEADER conains
   // the size of the structure, let's use this.

   dwDIBSize = *(LPDWORD)lpBI + PaletteSize((LPSTR)lpBI);  // Partial Calculation

   // Now calculate the size of the image

   if ((lpBI->biCompression == BI_RLE8) || (lpBI->biCompression == BI_RLE4)) {

      // It's an RLE bitmap, we can't calculate size, so trust the
      // biSizeImage field

      dwDIBSize += lpBI->biSizeImage;
      }
   else {
      DWORD dwBmBitsSize;  // Size of Bitmap Bits only

      // It's not RLE, so size is Width (DWORD aligned) * Height

      dwBmBitsSize = WIDTHBYTES((lpBI->biWidth)*((DWORD)lpBI->biBitCount)) * lpBI->biHeight;

      dwDIBSize += dwBmBitsSize;
      
      // Now, since we have calculated the correct size, why don't we
      // fill in the biSizeImage field (this will fix any .BMP files which 
      // have this field incorrect).

      lpBI->biSizeImage = dwBmBitsSize;
      }


   // Calculate the file size by adding the DIB size to sizeof(BITMAPFILEHEADER)
                   
   bmfHdr.bfSize = dwDIBSize + sizeof(BITMAPFILEHEADER);
   bmfHdr.bfReserved1 = 0;
   bmfHdr.bfReserved2 = 0;

   /*
    * Now, calculate the offset the actual bitmap bits will be in
    * the file -- It's the Bitmap file header plus the DIB header,
    * plus the size of the color table.
    */
   bmfHdr.bfOffBits = (DWORD)sizeof(BITMAPFILEHEADER) + lpBI->biSize +
                      PaletteSize((LPSTR)lpBI);

   /* Write the file header */
   _lwrite(fh, (LPSTR)&bmfHdr, sizeof(BITMAPFILEHEADER));

   /*
    * Write the DIB header and the bits -- use local version of
    * MyWrite, so we can write more than 32767 bytes of data
    */
   dwError = MyWrite(fh, (LPSTR)lpBI, dwDIBSize);
   GlobalUnlock(hDib);
   _lclose(fh);

   if (dwError == 0)
     return ERR_OPEN; // oops, something happened in the write
   else
     return 0; // Success code
}


//************************************************************************
//
// Auxiliary Functions which the above procedures use
//
//************************************************************************


/*************************************************************************
 *
 * Function:  ReadDIBFile (int)
 *
 *  Purpose:  Reads in the specified DIB file into a global chunk of
 *            memory.
 *
 *  Returns:  A handle to a dib (hDIB) if successful.
 *            NULL if an error occurs.
 *
 * Comments:  BITMAPFILEHEADER is stripped off of the DIB.  Everything
 *            from the end of the BITMAPFILEHEADER structure on is
 *            returned in the global memory handle.
 *
 *
 * NOTE: The DIB API were not written to handle OS/2 DIBs, so this
 * function will reject any file that is not a Windows DIB.
 *
 * History:   Date      Author       Reason
 *            9/15/91   Mark Bader   Based on DIBVIEW
 *            6/25/92   Mark Bader   Added check for OS/2 DIB
 *            7/21/92   Mark Bader   Added code to deal with bfOffBits
 *                                     field in BITMAPFILEHEADER      
 *            9/11/92   Mark Bader   Fixed Realloc Code to free original mem
 *
 *************************************************************************/

HANDLE ReadDIBFile(int hFile)
{
   BITMAPFILEHEADER bmfHeader;
//   DWORD dwBitsSize;
   UINT nNumColors;   // Number of colors in table
   HANDLE hDIB;        
   HANDLE hDIBtmp;    // Used for GlobalRealloc() //MPB
   LPBITMAPINFOHEADER lpbi;
   DWORD offBits;

   /*
    * get length of DIB in bytes for use when reading
    */

//   dwBitsSize = filelength(hFile);	// not used after assign, codewarrior mod

   // Allocate memory for header & color table.	We'll enlarge this
   // memory as needed.

   hDIB = GlobalAlloc(GMEM_MOVEABLE,
       (DWORD)(sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD)));
   
   if (!hDIB) return NULL;

   lpbi = (LPBITMAPINFOHEADER)GlobalLock(hDIB);
   if (!lpbi) 
   {
     GlobalFree(hDIB);
     return NULL;
   }

   // read the BITMAPFILEHEADER from our file

   if (sizeof (BITMAPFILEHEADER) != _lread (hFile, (LPSTR)&bmfHeader, sizeof (BITMAPFILEHEADER)))
     goto ErrExit;

   if (bmfHeader.bfType != 0x4d42)	/* 'BM' */
     goto ErrExit;

   // read the BITMAPINFOHEADER

   if (sizeof(BITMAPINFOHEADER) != _lread (hFile, (LPSTR)lpbi, sizeof(BITMAPINFOHEADER)))
     goto ErrExit;

   // Check to see that it's a Windows DIB -- an OS/2 DIB would cause
   // strange problems with the rest of the DIB API since the fields
   // in the header are different and the color table entries are
   // smaller.
   //
   // If it's not a Windows DIB (e.g. if biSize is wrong), return NULL.

   if (lpbi->biSize == sizeof(BITMAPCOREHEADER))
     goto ErrExit;

   // Now determine the size of the color table and read it.  Since the
   // bitmap bits are offset in the file by bfOffBits, we need to do some
   // special processing here to make sure the bits directly follow
   // the color table (because that's the format we are susposed to pass
   // back)

   if (!(nNumColors = (UINT)lpbi->biClrUsed))
    {
      // no color table for 24-bit, default size otherwise
      if (lpbi->biBitCount != 24)
        nNumColors = 1 << lpbi->biBitCount; /* standard size table */
    }

   // fill in some default values if they are zero
   if (lpbi->biClrUsed == 0)
     lpbi->biClrUsed = nNumColors;

   if (lpbi->biSizeImage == 0)
   {
     lpbi->biSizeImage = ((((lpbi->biWidth * (DWORD)lpbi->biBitCount) + 31) & ~31) >> 3)
			 * lpbi->biHeight;
   }

   // get a proper-sized buffer for header, color table and bits
   GlobalUnlock(hDIB);
   hDIBtmp = GlobalReAlloc(hDIB, lpbi->biSize +
                        nNumColors * sizeof(RGBQUAD) +
                        lpbi->biSizeImage, 0);

   if (!hDIBtmp) // can't resize buffer for loading
     goto ErrExitNoUnlock; //MPB
   else
     hDIB = hDIBtmp;

   lpbi = (LPBITMAPINFOHEADER)GlobalLock(hDIB);

   // read the color table
   _lread (hFile, (LPSTR)(lpbi) + lpbi->biSize, nNumColors * sizeof(RGBQUAD));

   // offset to the bits from start of DIB header
   offBits = lpbi->biSize + nNumColors * sizeof(RGBQUAD);

   // If the bfOffBits field is non-zero, then the bits might *not* be
   // directly following the color table in the file.  Use the value in
   // bfOffBits to seek the bits.

   if (bmfHeader.bfOffBits != 0L)
      _llseek(hFile, bmfHeader.bfOffBits, SEEK_SET);
   
   if (MyRead(hFile, (LPSTR)lpbi + offBits, lpbi->biSizeImage))
     goto OKExit;


ErrExit:
    GlobalUnlock(hDIB);    
ErrExitNoUnlock:    
    GlobalFree(hDIB);
    return NULL;

OKExit:
    GlobalUnlock(hDIB);
    return hDIB;
}

/*************************************************************************

  Function:  MyRead (int, LPSTR, DWORD)

   Purpose:  Routine to read files greater than 64K in size.

   Returns:  TRUE if successful.
             FALSE if an error occurs.

  
  History:   Date      Author       Reason
             9/15/91   Mark Bader   Based on DIBVIEW
 
*************************************************************************/


BOOL MyRead(int hFile, LPSTR lpBuffer, DWORD dwSize)
{
   char *lpInBuf = (char *)lpBuffer;
   int nBytes;

   /*
    * Read in the data in 32767 byte chunks (or a smaller amount if it's
    * the last chunk of data read)
    */

   while (dwSize)
   {
      nBytes = (int)(dwSize > (DWORD)32767 ? 32767 : LOWORD (dwSize));
      if (_lread(hFile, (LPSTR)lpInBuf, nBytes) != (WORD)nBytes)
         return FALSE;
      dwSize -= nBytes;
      lpInBuf += nBytes;
   }
   return TRUE;
}


/****************************************************************************

 FUNCTION   : MyWrite(int fh, VOID FAR *pv, DWORD ul)

 PURPOSE    : Writes data in steps of 32k till all the data is written.
              Normal _lwrite uses a WORD as 3rd parameter, so it is
              limited to 32767 bytes, but this procedure is not.

 RETURNS    : 0 - If write did not proceed correctly.
              number of bytes written otherwise.
 
  History:   Date      Author       Reason
             9/15/91   Mark Bader   Based on DIBVIEW

 ****************************************************************************/


DWORD PASCAL MyWrite(int iFileHandle, VOID FAR *lpBuffer, DWORD dwBytes)
{
   DWORD dwBytesTmp = dwBytes;       // Save # of bytes for return value
   BYTE *hpBuffer = (BYTE *)lpBuffer;   // make a pointer to the data

   /*
    * Write out the data in 32767 byte chunks.
    */

   while (dwBytes > 32767)
   {
      if (_lwrite(iFileHandle, (LPSTR)hpBuffer, (WORD)32767) != 32767)
         return 0;
      dwBytes -= 32767;
      hpBuffer += 32767;
   }

   /* Write out the last chunk (which is < 32767 bytes) */
   if (_lwrite(iFileHandle, (LPSTR)hpBuffer, (WORD)dwBytes) != (WORD)dwBytes)
      return 0;
   return dwBytesTmp;
}

//////////////////////////////////////////////////////////////////////////////////

typedef struct
{
   BOOL bGraphics;
   BOOL bText;
   RECT GraphicsRect;
} BANDINFOSTRUCT;


/****************************************************************
 * Variables
 ***************************************************************/

HWND hDlgAbort;                    // Handle to Abort Dialog
char szPrintDlg[] = "PRINTCANCEL";    // Name of Print dialog from .RC
BOOL bAbort = FALSE;               // Abort a print operation?
char gszDevice[50];                // Keeps track out device (e.g. "HP LaserJet")
char gszOutput[50];                // Output device (e.g. "LPT1:")

/***************************************************************
 * Function prototypes for functions local to this module
 **************************************************************/


BOOL FAR PASCAL PrintAbortProc(HDC, short);
int FAR PASCAL PrintAbortDlg(HWND, unsigned, WORD, LONG);
WORD PrintBand(HDC, LPRECT, LPRECT, BOOL, BOOL, LPBITMAPINFOHEADER, LPSTR);
HDC GetDIBPrinterDC(void);
void CalculatePrintRect(HDC, LPRECT, WORD, DWORD, DWORD);


/**********************************************************************
 *
 * PrintWindow()
 *
 *
 * Description:
 *
 * This function prints the specified window on the default
 * printer.
 *
 * Parameters:
 *
 * HWND hWnd       - Specifies the window to print.  The window must
 *                   not be iconic and must be topmost on the display.
 *
 * WORD fPrintArea - Specifies the area of the window to print.  Must be
 *                   one of PW_ALL, PW_CLIENT, PW_CAPTION,  or PW_MENUBAR
 *
 * WORD fPrintOpt  - Print options (one of PW_BESTFIT, PW_STRETCHTOPAGE, or
 *                   PW_SCALE)
 *
 * WORD wXScale, wYScale - X and Y scaling factors if PW_SCALE is specified
 *
 * LPSTR szJobName - Name that you would like to give to this print job (this
 *                   name shows up in the Print Manager as well as the
 *                   "Now Printing..." dialog box).
 * Return Value:
 *      ERR_DIBFUNCTION or any return value from PrintDIB
 *
 * History:   
 * 
 *     Date      Author       Reason         
 *     9/15/91   Mark Bader   Created         
 *
 **********************************************************************/


WORD FAR PrintWindow(HWND hWnd,         // Window to be printed
                 WORD fPrintArea,   // Area of window to be printed
                 WORD fPrintOpt,    // Print options
                 WORD wXScale,      // X Scaling factor if PW_SCALE is used
                 WORD wYScale,      // Y Scaling factor if PW_SCALE is used
                 LPSTR szJobName)   // Name of print job
{
   HDIB hDib;          // Handle to the DIB
   WORD wReturn;       // our return value

   /*
    * Parameter validation
    */
   if (!hWnd)
      return (ERR_INVALIDHANDLE);  // Invalid Window

   /*
    * Copy the Window to a DIB and print it.
    */
   hDib = CopyWindowToDIB(hWnd, fPrintArea);
   if (!hDib)
      return (ERR_DIBFUNCTION); // CopyWindowToDIB failed!
   wReturn = PrintDIB(hDib, fPrintOpt, wXScale, wYScale, szJobName);

   /*
    * Call DestroyDIB to free the memory the dib takes up.
    */
   DestroyDIB(hDib);
   return wReturn;   // return the value from PrintDIB
}



/**********************************************************************
 *
 * PrintScreen()
 *
 *
 * Description:
 *
 * This function prints the specified portion of the display screen on the
 * default printer using the print options specified.  The print
 * options are listed in dibapi.h.
 *
 * Parameters:
 *
 * LPRECT rRegion  - Specifies the region of the screen (in screen
 *                   coordinates) to print
 *
 * WORD fPrintOpt  - Print options  (PW_BESTFIT, PW_STRETCHTOPAGE, or PW_SCALE)
 *
 * WORD wXScale, wYScale - X and Y scaling factors if PW_SCALE is specified
 *
 * LPSTR szJobName - Name that you would like to give to this print job (this
 *                   name shows up in the Print Manager as well as the
 *                   "Now Printing..." dialog box).
 *
 * Return Value:
 *      ERR_DIBFUNCTION or any return value from PrintDIB
 *
 * History:   
 * 
 *     Date      Author       Reason         
 *     9/15/91   Mark Bader   Created         
 *
 **********************************************************************/


WORD FAR PrintScreen(LPRECT rRegion,    // Region to print (in screen coords)
                 WORD fPrintOpt,    // print options
                 WORD wXScale,      // X scaling (used if PW_SCALE specified)
                 WORD wYScale,      // Y scaling (used if PW_SCALE specified)
                 LPSTR szJobName)   // Name of print job
{
   HDIB hDib;          // A Handle to our DIB
   WORD wReturn;       // Return value

   /*
    * Copy the screen contained in the specified rectangle to a DIB
    */
   hDib = CopyScreenToDIB(rRegion);
   if (!hDib)
      return (ERR_DIBFUNCTION);   // CopyScreenToDIB failed!
   wReturn = PrintDIB(hDib, fPrintOpt, wXScale, wYScale, szJobName);
   DestroyDIB(hDib);
   return wReturn; // Return the value that PrintDIB returned
}






/**********************************************************************
 *
 * PrintDIB()
 *
 * Description:
 *
 * This routine prints the specified DIB.  The actual printing is done
 * in the PrintBand() routine (see below), this procedure drives the
 * printing operation.  PrintDIB() has the code to handle both banding
 * and non-banding printers.  A banding printer can be distinguished by
 * the GetDeviceCaps() API (see the code below).  On banding devices,
 * must repeatedly call the NEXTBAND escape to get the next banding
 * rectangle to print into.  If the device supports the BANDINFO escape,
 * it should be used to determine whether the band "wants" text or
 * graphics (or both).  On non-banding devices, we can ignore all this
 * and call PrintBand() on the entire page.
 *
 * Parameters:
 *
 * HDIB hDib       - Handle to dib to be printed
 *
 * WORD fPrintOpt  - tells which print option to use (PW_BESTFIT,
 *                   PW_STRETCHTOPAGE, OR PW_SCALE)
 *
 * WORD wXScale, wYScale - X and Y scaling factors (integers) for
 *                   printed output if the PW_SCALE option is used.
 *
 * LPSTR szJobName - Name that you would like to give to this print job (this
 *                   name shows up in the Print Manager as well as the
 *                   "Now Printing..." dialog box).
 *
 * Return Value:  (see errors.h for description)
 *
 * One of: ERR_INVALIDHANDLE
 *         ERR_LOCK
 *         ERR_SETABORTPROC
 *         ERR_STARTDOC
 *         ERR_NEWFRAME
 *         ERR_ENDDOC
 *         ERR_GETDC
 *         ERR_STRETCHDIBITS
 *
 * History:   
 * 
 *     Date      Author       Reason         
 *     9/15/91   Mark Bader   Created, based on DIBVIEW's DibPrint()
 *
 ********************************************************************/


WORD FAR PrintDIB(HDIB hDib,    // Handle to the DIB
              WORD fPrintOpt,   // Print Options
              WORD wXScale,     // X Scaling factor
              WORD wYScale,     // Y Scaling factor
              LPSTR szJobName)  // Name of print job
{
   HDC hPrnDC;                  // DC to the printer
   RECT rect;                   // Rect structure used for banding
   DOCINFO docInfo;
   // BANDINFOSTRUCT biBandInfo;   // Used for banding
   int id;               // Temp number used to check banding capability
   LPSTR lpBits;                // pointer to the DIB bits
   LPBITMAPINFOHEADER lpDIBHdr; // Pointer to DIB header
   // int nBandCount = 0;          // used for print dialog box to count bands
   WORD wErrorCode = 0;         // Error code to return
   RECT rPrintRect;             // Rect which specifies the area on the printer
                                // (in printer coordinates) which we
                                // want the DIB to go to
   char szBuffer[70];           // Buffer to hold message for "Printing" dlg box
   char szJobNameTrunc[35];     // szJobName truncated to 31 characters for STARTDOC
   DWORD e;

   if (!hDib) return (ERR_INVALIDHANDLE);

   lpDIBHdr = (LPBITMAPINFOHEADER)GlobalLock(hDib);
   if (!lpDIBHdr) // Check that we have a valid pointer
      return (ERR_LOCK);
   lpBits = FindDIBBits((LPSTR)lpDIBHdr); // Find pointer to DIB bits
   if (hPrnDC = GetDIBPrinterDC()) {
      lstrcpy(szJobNameTrunc, szJobName);
      szJobNameTrunc[31] = '\0';           // Truncate string to 31 chars
      
      docInfo.cbSize = sizeof(DOCINFO);
	  docInfo.lpszDocName = &szJobNameTrunc[0];
	  docInfo.lpszOutput = 0;
	  docInfo.lpszDatatype = 0;
	  docInfo.fwType = 0;

		// if (Escape(hPrnDC, STARTDOC, lstrlen((LPSTR)szJobNameTrunc), (LPSTR)szJobNameTrunc, NULL) < 0)
		if ((id = StartDoc(hPrnDC, &docInfo)) <= 0) {
			e = GetLastError();
			goto error;
		}
      
      
      SetStretchBltMode(hPrnDC, COLORONCOLOR);

      /*
       * Determine rPrintRect (printer area to print to) from the
       * fPrintOpt.  Fill in rPrintRect.left and .top from wXScale and
       * wYScale just in case we use PW_SCALE (see the function
       * CalculatePrintRect).
       */
      rPrintRect.left = wXScale;
      rPrintRect.top = wYScale;
      CalculatePrintRect(hPrnDC, &rPrintRect, fPrintOpt, lpDIBHdr->biWidth,
                         lpDIBHdr->biHeight);

      /*
       * Initialize the abort procedure.
       */
      hDlgAbort = CreateDialog(hInst, szPrintDlg, GetFocus(), (DLGPROC)PrintAbortDlg);

      /*
       * Set the text inside the dialog to the name of our print job
       */
      wsprintf(szBuffer, "Printing %s...", (LPSTR)szJobNameTrunc);
      SetDlgItemText(hDlgAbort, IDC_PRINTTEXT1, (LPSTR)szBuffer);

      /*
       * Set global variable bAbort to FALSE.  This will get set to TRUE
       * in our PrintAbortDlg() procedure if the user selects the
       * CANCEL button in our dialog box
       */
      bAbort = FALSE;

      // if (Escape(hPrnDC, SETABORTPROC, 0, (LPSTR)PrintAbortProc, NULL) < 0)
      if (SetAbortProc(hPrnDC, (ABORTPROC)PrintAbortProc) < 0) goto error;

      /*
       * Call Escape() with STARTDOC -- starts print job
       */
      // if (Escape(hPrnDC, STARTDOC, lstrlen((LPSTR)szJobNameTrunc), (LPSTR)szJobNameTrunc, NULL) < 0)
		
		if (StartPage(hPrnDC) <= 0) goto error;
		
		// Print the whole page -- non-banding device.
		rect = rPrintRect;
		SetDlgItemText(hDlgAbort, IDC_PERCENTAGE, (LPSTR)
		            "Sending bitmap to printer...");
		wErrorCode = PrintBand(hPrnDC, &rPrintRect, &rect, TRUE, TRUE,
		                    lpDIBHdr, lpBits);
		
		EndPage(hPrnDC);
		EndDoc(hPrnDC);
		DestroyWindow(hDlgAbort);
		DeleteDC(hPrnDC);
   }
   else
      wErrorCode = ERR_GETDC;   // Couldn't get Printer DC!
   GlobalUnlock(hDib);
   
   return 0;

	error:
         // Oops, something happened, let's clean up here and return
         DestroyWindow(hDlgAbort);   // Remove abort dialog box
         DeleteDC(hPrnDC);
         GlobalUnlock(hDib);

		 return -1;
}




// *******************************************************************
// Auxiliary Functions
//     -- Local to this module only
// *******************************************************************


/*********************************************************************
 *
 * CalculatePrintRect()
 *
 * Given fPrintOpt and a size of the DIB, return the area on the
 * printer where the image should go (in printer coordinates).  If
 * fPrintOpt is PW_SCALE, then lpPrintRect.left and .top should
 * contain WORDs which specify the scaling factor for the X and
 * Y directions, respecively.
 *
 * History:   
 * 
 *     Date      Author       Reason         
 *     9/15/91   Mark Bader   Created based on DIBVIEW
 *
 ********************************************************************/


void CalculatePrintRect(HDC hDC,             // HDC to printer DC
                        LPRECT lpPrintRect,  // Returned PrintRect
                        WORD fPrintOpt,      // Options
                        DWORD cxDIB,         // Size of DIB - x
                        DWORD cyDIB)         // Size of DIB - y
{
   int cxPage, cyPage, cxInch, cyInch;

   if (!hDC)
      return;

   /*
    * Get some info from printer driver
    */
   cxPage = GetDeviceCaps(hDC, HORZRES);     // Width of printr page - pixels
   cyPage = GetDeviceCaps(hDC, VERTRES);     // Height of printr page - pixels
   cxInch = GetDeviceCaps(hDC, LOGPIXELSX);  // Printer pixels per inch - X
   cyInch = GetDeviceCaps(hDC, LOGPIXELSY);  // Printer pixels per inch - Y
   switch (fPrintOpt)
      {

   /*
    * Best Fit case -- create a rectangle which preserves
    * the DIB's aspect ratio, and fills the page horizontally.
    *
    * The formula in the "->bottom" field below calculates the Y
    * position of the printed bitmap, based on the size of the
    * bitmap, the width of the page, and the relative size of
    * a printed pixel (cyInch / cxInch).
    */
   case PW_BESTFIT:
      lpPrintRect->top = 0;
      lpPrintRect->left = 0;
      lpPrintRect->bottom = (int)(((double)cyDIB * cxPage * cyInch) / ((double
                            )cxDIB * cxInch));
      lpPrintRect->right = cxPage;
      break;

   /*
    * Scaling option -- lpPrintRect's top/left contain
    * multipliers to multiply the DIB's height/width by.
    */

   case PW_SCALE:
   {
      int cxMult, cyMult;

      cxMult = lpPrintRect->left;
      cyMult = lpPrintRect->top;
      lpPrintRect->top = 0;
      lpPrintRect->left = 0;
      lpPrintRect->bottom = (int)(cyDIB * cyMult);
      lpPrintRect->right = (int)(cxDIB * cxMult);
   }
      break;

   /*
    * Stretch To Page case -- create a rectangle
    * which covers the entire printing page (note that this
    * is also the default).
    */
   case PW_STRETCHTOPAGE:
   default:
      lpPrintRect->top = 0;
      lpPrintRect->left = 0;
      lpPrintRect->bottom = cyPage;
      lpPrintRect->right = cxPage;
      break;
      }

  lpPrintRect->top += cyInch / 2; // leave a 1/2 inch margin MAK
  lpPrintRect->left += cxInch / 2;
  lpPrintRect->bottom -= cyInch / 2;
  lpPrintRect->right -= cxInch / 2;
}



/*********************************************************************
 *
 * PrintBand()
 *
 * This routine does ALL output to the printer.  It is called from
 * the PrintDIB() routine.  It is called for both banding and non-
 * banding printing devices.  lpRectClip contains the rectangular
 * area we should do our output into (i.e. we should clip our output
 * to this area).  The flags fDoText and fDoGraphics should be set
 * appropriately (if we want any text output to the rectangle, set
 * fDoText to true).  Normally these flags are returned on banding
 * devices which support the BANDINFO escape.
 *
 * History:   
 * 
 *     Date      Author       Reason         
 *     9/15/91   Mark Bader   Created based on DIBVIEW
 *
 ********************************************************************/


WORD PrintBand(HDC hDC,           // Handle to the Printer DC
               LPRECT lpRectOut,  // Rect where entire DIB is to go
               LPRECT lpRectClip, // Clippping rect where this portion goes
               BOOL fDoText,      // TRUE if this band is for text
               BOOL fDoGraphics,  // TRUE if this band is for graphics
               LPBITMAPINFOHEADER lpDIBHdr,   // Pointer to DIB header
               LPSTR lpDIBBits)   // Pointer to DIB bits
{
   RECT rect;                   // Temporary rectangle
   double dblXScaling,          // X and Y scaling factors
          dblYScaling;
   WORD wReturn = 0;            // Return code

   if (fDoGraphics)
   {
      dblXScaling = ((double)lpRectOut->right - lpRectOut->left) / (double)
                    lpDIBHdr->biWidth;
      dblYScaling = ((double)lpRectOut->bottom - lpRectOut->top) / (double)
                    lpDIBHdr->biHeight;
      /*
       * Now we set up a temporary rectangle -- this rectangle
       *  holds the coordinates on the paper where our bitmap
       *  WILL be output.  We can intersect this rectangle with
       *  the lpClipRect to see what we NEED to output to this
       *  band.  Then, we determine the coordinates in the DIB
       *  to which this rectangle corresponds (using dbl?Scaling).
       */
      IntersectRect(&rect, lpRectOut, lpRectClip);
      if (!IsRectEmpty(&rect))
      {
         RECT rectIn;

         rectIn.left = (int)((rect.left - lpRectOut->left) / dblXScaling + 0.5
                       );
         rectIn.top = (int)((rect.top - lpRectOut->top) / dblYScaling + 0.5);
         rectIn.right = (int)(rectIn.left + (rect.right - rect.left) /
                        dblXScaling + 0.5);
         rectIn.bottom = (int)(rectIn.top + (rect.bottom - rect.top) /
                         dblYScaling + 0.5);
         if (!StretchDIBits(hDC,                              // DestDC
                            rect.left,                        // DestX
                            rect.top,                         // DestY
                            rect.right - rect.left,           // DestWidth
                            rect.bottom - rect.top,           // DestHeight
                            rectIn.left,                      // SrcX
                            (int)(lpDIBHdr->biHeight) -       // SrcY
                            rectIn.top - (rectIn.bottom - rectIn.top),
                            rectIn.right - rectIn.left,       // SrcWidth
                            rectIn.bottom - rectIn.top,       // SrcHeight
                            lpDIBBits,                        // lpBits
                            (LPBITMAPINFO)lpDIBHdr,           // lpBitInfo
                            DIB_RGB_COLORS,                   // wUsage
                            SRCCOPY))                         // dwROP
            wReturn = ERR_STRETCHDIBITS; // StretchDIBits() failed!
      }
   }
   return wReturn;
}


/***********************************************************************
 *
 * GetDIBPrinterDC()
 *
 * Return a DC to the currently selected printer.
 * Returns NULL on error.
 *
 * History:   
 * 
 *     Date      Author       Reason         
 *     9/15/91   Mark Bader   Created based on DIBVIEW
 *
 ***********************************************************************/


HDC GetDIBPrinterDC(void)
{
	Boolean result;
	PRINTDLG pd;
	
	memset((void *)&pd, 0, sizeof(PRINTDLG));
	pd.lStructSize = sizeof(PRINTDLG);
	pd.hwndOwner = FrontWindow();
	pd.Flags = PD_RETURNDC | PD_NOSELECTION | PD_USEDEVMODECOPIES;
	// pd.nFromPage = pd.nToPage = pd.nCopies = 1;
	
	result = PrintDlg(&pd);
	// if (pd.hDevNames) GlobalFree(pd.hDevNames);
	// if (pd.hDevMode) GlobalFree(pd.hDevMode);
	
	if (!result) return 0;

	return pd.hDC;
	
	/*
	static char szPrinter[64];
	char *szDevice, *szDriver, *szOutput;
	
	GetProfileString("windows", "device", "", szPrinter, 64);
	if ((szDevice = strtok(szPrinter, ",")) && (szDriver = strtok(NULL, ", "))
	   && (szOutput = strtok(NULL, ", ")))
	{
	  lstrcpy((LPSTR)gszDevice, (LPSTR)szDevice);    // Copy to global variables
	  lstrcpy((LPSTR)gszOutput, (LPSTR)szOutput);
	  return CreateDC(szDriver, szDevice, szOutput, NULL);
	}
	return NULL;
	*/
}


/**********************************************************************
 * PrintAbortProc()
 *
 * Abort procedure - contains the message loop while printing is
 * in progress.  By using a PeekMessage() loop, multitasking
 * can occur during printing.
 *
 * History:   
 * 
 *     Date      Author       Reason         
 *     9/15/91   Mark Bader   Created         
 *
 **********************************************************************/


BOOL FAR PASCAL PrintAbortProc(HDC hDC, short code)
{
   MSG msg;

   while (!bAbort && PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
      if (!IsDialogMessage(hDlgAbort, &msg))
      {
         TranslateMessage(&msg);
         DispatchMessage(&msg);
      }
   return (!bAbort);
}

/***********************************************************************
 *
 * PrintAbortDlg()
 *
 *
 * This is the Dialog Procedure which will handle the "Now Printing"
 * dialog box.  When the user presses the "Cancel" button, the
 * global variable bAbort is set to TRUE, which causes the
 * PrintAbortProc to exit, which in turn causes the printing
 * operation to terminate.
 *
 * History:   
 * 
 *     Date      Author       Reason         
 *     9/15/91   Mark Bader   Created         
 *
 ***********************************************************************/


int FAR PASCAL PrintAbortDlg(HWND hWnd,   /* Handle to dialog box */ unsigned
                             msg, /* Message */ WORD wParam, LONG lParam)
{
   switch (msg)
      {
   case WM_INITDIALOG:
   {
      char szBuffer[100];

      /*
       * Fill in the text which specifies where this bitmap
       * is going ("on HP LaserJet on LPT1", for example)
       */

      wsprintf(szBuffer, "on %s on %s", (LPSTR)gszDevice, (LPSTR)gszOutput);
      SetDlgItemText(hWnd, IDC_PRINTTEXT2, (LPSTR)szBuffer);
      SetFocus(GetDlgItem(hWnd, IDCANCEL));
   }
      return TRUE;     // Return TRUE because we called SetFocus()

   case WM_COMMAND:
      bAbort = TRUE;
      DestroyWindow(hWnd);
      return TRUE;
      break;
      }
   return FALSE;
}

//////////////////////////////////////////////////////////////////////////////////

static char *szErrors[] =
{
   "Not a Windows DIB file.",
   "Couldn't allocate memory.",
   "Error reading file.",
   "Error locking memory.",
   "Error opening file.",
   "Error creating palette.",
   "Error getting a DC.",
   "Error creating Device Dependent Bitmap.",
   "StretchBlt() failed.",
   "StretchDIBits() failed.",
   "SetDIBitsToDevice() failed.",
   "Printer: StartDoc failed.",
   "Printing: GetModuleHandle() couldn't find GDI.",
   "Printer: SetAbortProc failed.",
   "Printer: StartPage failed.",
   "Printer: NEWFRAME failed.",
   "Printer: EndPage failed.",
   "Printer: EndDoc failed.",
   "SetDIBits() failed.",
   "File Not Found.",
   "Invalid Handle.",
   "General Error on call to DIB function."
};


void FAR DIBError(int ErrNo)
{
	
	if ((ErrNo < ERR_MIN) || (ErrNo >= ERR_MAX))
		TechError("DIBError", "Unknown DIB error", ErrNo);
	else
		TechError("DIBError", szErrors[ErrNo], ErrNo);
}


///////////////////


Rect GetDIBFrame(HDIB hDib)
{
	Rect r = {0,0,0,0};
	if(hDib) {
		LPBITMAPINFOHEADER lpDIBHdr  = (LPBITMAPINFOHEADER)GlobalLock(hDib);
		if (lpDIBHdr)
		{
			r.right = lpDIBHdr->biWidth;
			r.bottom = lpDIBHdr->biHeight;
		}
		GlobalUnlock(hDib);
	}
	return r;
}



#endif
