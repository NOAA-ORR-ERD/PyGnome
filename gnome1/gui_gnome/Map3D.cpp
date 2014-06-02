
#include "Cross.h"
#include "MapUtils.h"
#include "GridMapUtils.h"
#include "GenDefs.h"
#include "GridVel.h"
#include "NetCDFMover.h"
#include "NetCDFMoverCurv.h"
#include "NetCDFWindMover.h"
#include "NetCDFMoverTri.h"
#include "NetCDFWindMoverCurv.h"
#include "TideCurCycleMover.h"
#include "Contdlg.h"
#include "ObjectUtilsPD.h"
#include "Map3D.h"
#include "DagTreePD.h"
#include "DagTreeIO.h"
#include "netcdf.h"

/**************************************************************************************************/
OSErr AddMap3D(char *path, WorldRect bounds)
{
	char 		nameStr[256];
	OSErr		err = noErr;
	Map3D 	*map = nil;
	char fileName[256],s[256];
	strcpy(s,path);
	SplitPathFile (s, fileName);

	strcpy (nameStr, "BathymetryMap: ");
	strcat (nameStr, fileName);

	map = new Map3D(nameStr, bounds);
	if (!map)
		{ TechError("AddMap3D()", "new Map3D()", 0); return -1; }

	if (err = map->InitMap()) { delete map; return err; }

	if (err = model->AddMap(map, 0))
		{ map->Dispose(); delete map; map=0; return -1; }
	else {
		model->NewDirtNotification();
	}

	return err;
}


///////////////////////////////////////////////////////////////////////////

Map3D::Map3D(char* name, WorldRect bounds) : TMap(name, bounds)
{
	fGrid = 0;

	fBoundarySegmentsH = 0;
	fBoundaryTypeH = 0;
	fBoundaryPointsH = 0;
#ifdef MAC
	memset(&fWaterBitmap,0,sizeof(fWaterBitmap));
	memset(&fLandBitmap,0,sizeof(fLandBitmap));
#else
	fWaterBitmap = 0;
	fLandBitmap = 0;
#endif
	bDrawLandBitMap = false;	// combined option for now
	bDrawWaterBitMap = false;

	//bShowSurfaceLEs = true;
	//bShowLegend = true;
	//bShowGrid = false;
	bShowDepthContours = false;
	
	//bDrawContours = true;

	//fDropletSizesH = 0;
	
	//memset(&fLegendRect,0,sizeof(fLegendRect)); 
	
	//fWaterDensity = 1020;
	//fMixedLayerDepth = 10.;	//meters
	//fBreakingWaveHeight = 1.;	// meters
	//fDiagnosticStrType = 0;
	
	fMinDistOffshore = 0.;	//km - use bounds to set default
	//bUseLineCrossAlgorithm = false;
	//bUseSmoothing = false;

	//fWaveHtInput = 0;	// default compute from wind speed

	fVerticalGridType = TWO_D;
	fGridType = CURVILINEAR;
}

void Map3D::Dispose()
{
	if (fBoundarySegmentsH) {
		DisposeHandle((Handle)fBoundarySegmentsH);
		fBoundarySegmentsH = 0;
	}
	
	if (fBoundaryTypeH) {
		DisposeHandle((Handle)fBoundaryTypeH);
		fBoundaryTypeH = 0;
	}
	
	if (fBoundaryPointsH) {
		DisposeHandle((Handle)fBoundaryPointsH);
		fBoundaryPointsH = 0;
	}
	
	/*if (fDropletSizesH) {
		DisposeHandle((Handle)fDropletSizesH);
		fDropletSizesH = 0;
	}*/

#ifdef MAC
	DisposeBlackAndWhiteBitMap (&fWaterBitmap);
	DisposeBlackAndWhiteBitMap (&fLandBitmap);
#else
	if(fWaterBitmap) DestroyDIB(fWaterBitmap);
	fWaterBitmap = 0;
	if(fLandBitmap) DestroyDIB(fLandBitmap);
	fLandBitmap = 0;
#endif
	
	if (fGrid)
	{
		fGrid -> Dispose();
		delete fGrid;
		fGrid = nil;
	}
	
	TMap::Dispose();
}



void DrawFilledWaterTriangles2(void * object,WorldRect wRect,Rect r)
{
	Map3D* map3D = (Map3D*)object; // typecast
	TTriGridVel* triGrid = 0;	
	
	triGrid = map3D->GetGrid();
	if(triGrid) {
		// draw triangles as filled polygons
		triGrid->DrawBitMapTriangles(r);
	}
	return;
}

static Boolean drawingLandBitMap;
void DrawWideLandSegments2(void * object,WorldRect wRect,Rect r)
{
	Map3D* map3D = (Map3D*)object; // typecast
	
		// draw land boundaries as wide lines
		drawingLandBitMap = TRUE;
		map3D -> DrawBoundaries(r);
		drawingLandBitMap = FALSE;
}
Boolean Map3D::InMap(WorldPoint p)
{
	WorldRect ourBounds = this -> GetMapBounds(); 
	//code goes here, option to check based on grid in addition to bitmap
	//TTriGridVel* triGrid = GetGrid();	// don't think need 3D here
	//TDagTree *dagTree = triGrid->GetDagTree();
	
	/*LongPoint lp;
	 lp.h = p.pLong;
	 lp.v = p.pLat;
	 if (dagTree -> WhatTriAmIIn(lp) >= 0) return true;*/
	
	if (!WPointInWRect(p.pLong, p.pLat, &ourBounds))
		return false;
	Boolean onLand = IsBlackPixel(p,ourBounds,fLandBitmap);
	Boolean inWater = IsBlackPixel(p,ourBounds,fWaterBitmap);
	if (onLand || inWater) 
		return true;
	else
		return false;
}

long Map3D::GetLandType(WorldPoint p)
{
	// This isn't used at the moment
	WorldRect ourBounds = this -> GetMapBounds(); 
	Boolean onLand = IsBlackPixel(p,ourBounds,fLandBitmap);
	Boolean inWater = IsBlackPixel(p,ourBounds,fWaterBitmap);
	if (onLand) 
		return LT_LAND;
	else if (inWater)
		return LT_WATER;
	else
		return LT_UNDEFINED;
	
}

Boolean Map3D::InWater(WorldPoint p)
{
	WorldRect ourBounds = this -> GetMapBounds(); 
	Boolean inWater = false;
	TTriGridVel* triGrid = GetGrid();	
	TDagTree *dagTree = triGrid->GetDagTree();
	
	if (!WPointInWRect(p.pLong, p.pLat, &ourBounds)) return false; // off map is not in water
	
	inWater = IsBlackPixel(p,ourBounds,fLandBitmap);
	LongPoint lp;
	lp.h = p.pLong;
	lp.v = p.pLat;
	if (dagTree -> WhatTriAmIIn(lp) >= 0) inWater = true;
	
	return inWater;
}


Boolean Map3D::OnLand(WorldPoint p)
{
	WorldRect ourBounds = this -> GetMapBounds(); 
	Boolean onLand = false;
	TTriGridVel* triGrid = GetGrid();	
	TDagTree *dagTree = triGrid->GetDagTree();
	
	if (!WPointInWRect(p.pLong, p.pLat, &ourBounds)) return false; // off map is not on land
	
	onLand = IsBlackPixel(p,ourBounds,fLandBitmap);
	
	if (bIAmPartOfACompoundMap) return onLand;	
	
	if (gDispersedOilVersion) return onLand;	
	// code goes here, for narrow channels bitmap is too large and beaches too many LEs but this allows some LEs to cross boundary...
	// maybe let user set parameter instead
	
	/*LongPoint lp;
	 lp.h = p.pLong;
	 lp.v = p.pLat;
	 if (dagTree -> WhatTriAmIIn(lp) >= 0) onLand = false;*/
	
	return onLand;
}


WorldPoint3D	Map3D::MovementCheck2 (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed)
{
	// check every pixel along the line it makes on the water bitmap
	// for any non-water point check the land bitmap as well and if it crosses a land boundary
	// force the point to the closest point in the bounds
#ifdef MAC
	BitMap bm = fWaterBitmap;
#else
	HDIB bm = fWaterBitmap;
#endif
	
	// this code is similar to IsBlackPixel
	Rect bounds;
	char* baseAddr= 0;
	long rowBytes;
	long rowByte,bitNum,byteNumber,offset;
	Point fromPt,toPt;
	Boolean isBlack = false;
	
#ifdef MAC
	bounds = bm.bounds;
	rowBytes = bm.rowBytes;
	baseAddr = bm.baseAddr;
#else //IBM
	if(bm)
	{
		LPBITMAPINFOHEADER lpDIBHdr  = (LPBITMAPINFOHEADER)GlobalLock(bm);
		baseAddr = (char*) FindDIBBits((LPSTR)lpDIBHdr);
#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)
		rowBytes = WIDTHBYTES(lpDIBHdr->biBitCount * lpDIBHdr->biWidth);
		MySetRect(&bounds,0,0,lpDIBHdr->biWidth,lpDIBHdr->biHeight);
	}
#endif
	
	Boolean LEsOnSurface = (fromWPt.z == 0 && toWPt.z == 0);
	if (toWPt.z == 0 && !isDispersed) LEsOnSurface = true;
	//Boolean LEsOnSurface = true;
	if (!gDispersedOilVersion) LEsOnSurface = true;	// something went wrong
	//if (bUseLineCrossAlgorithm) return SubsurfaceMovementCheck(fromWPt, toWPt, status);	// dispersed oil GNOME had some diagnostic options
	if(baseAddr)
	{
		// find the point in question in the bitmap
		// determine the pixel in the bitmap we need to look at
		// think of the bitmap as an array of pixels 
		long maxChange;
		WorldPoint3D wp = {0,0,0.};
		WorldRect mapBounds = this->GetMapBounds();
		
		fromPt = WorldToScreenPoint(fromWPt.p,mapBounds,bounds);
		toPt = WorldToScreenPoint(toWPt.p,mapBounds,bounds);
		
		// check the bitmap for each pixel when in range
		// so find the number of pixels change hori and vertically
		maxChange = _max(abs(toPt.h - fromPt.h),abs(toPt.v - fromPt.v));
		
		if(maxChange == 0) {
			// it's the same pixel, there is nothing to do
		}
		else { // maxChange >= 1
			long i;
			double fraction;
			Point pt, prevPt = fromPt;
			WorldPoint3D prevWPt = fromWPt;
			
			// note: there is no need to check the first or final pixel, so i only has to go to maxChange-1 
			//for(i = 0; i < maxChange; i++) 
			for(i = 0; i < maxChange+1; i++) 
			{
				//fraction = (i+1)/(double)(maxChange); 
				fraction = (i)/(double)(maxChange); 
				wp.p.pLat = (1-fraction)*fromWPt.p.pLat + fraction*toWPt.p.pLat;
				wp.p.pLong = (1-fraction)*fromWPt.p.pLong + fraction*toWPt.p.pLong;
				wp.z = (1-fraction)*fromWPt.z + fraction*toWPt.z;
				
				pt = WorldToScreenPoint(wp.p,mapBounds,bounds);
				
				// only check this pixel if it is in range
				// otherwise it is not on our map, hence not our problem
				// so assume it is water and OK
				
				if(bounds.left <= pt.h && pt.h < bounds.right
				   && bounds.top <= pt.v && pt.v < bounds.bottom)
				{
					
#ifdef IBM
					/// on the IBM, the rows of pixels are "upsidedown"
					offset = rowBytes*(long)(bounds.bottom - 1 - pt.v);
					/// on the IBM, for a mono map, 1 is background color,
					isBlack = !BitTst(baseAddr + offset, pt.h);
#else
					offset = (rowBytes*(long)pt.v);
					isBlack = BitTst(baseAddr + offset, pt.h);
#endif
					
					// don't beach LEs that are below the surface, reflect in some way
					if (!isBlack) // checking water bitmap, so not water
					{  // either a land point or outside a water boundary, calling code will check which is the case
						if (LEsOnSurface)
							return wp; 
						else
							// reflect and check z and return, but if not inmap return as is (or return towpt?)
						{	// here return the point and then see if it's on another map, else use toWPt
							if (!InMap(wp.p))
							{
								if(!InMap(toWPt.p))
									return toWPt;
								else
									return wp;
							}
							if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
								goto done;
							return ReflectPoint(fromWPt,toWPt,wp);
						}
					}
					else
					{	// also check if point is on both bitmaps and if so beach it
						Boolean onLand = OnLand(wp.p);	// on the boundary
						if (onLand) 
						{
							if (LEsOnSurface)	
								return wp;
							else
							{
								if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
									goto done;
								return ReflectPoint(fromWPt,toWPt,wp);
							}
						}
					}
					if (abs(pt.h - prevPt.h) == 1 && abs(pt.v - prevPt.v) == 1)
					{	// figure out which pixel was crossed
						
						float xRatio = (float)(wp.p.pLong - mapBounds.loLong) / (float)WRectWidth(mapBounds),
						yRatio = (float)(wp.p.pLat - mapBounds.loLat) / (float)WRectHeight(mapBounds);
						float ptL = bounds.left + RectWidth(bounds) * xRatio;
						float ptB = bounds.bottom - RectHeight(bounds) * yRatio;
						xRatio = (float)(prevWPt.p.pLong - mapBounds.loLong) / (float)WRectWidth(mapBounds);
						yRatio = (float)(prevWPt.p.pLat - mapBounds.loLat) / (float)WRectHeight(mapBounds);
						float prevPtL = bounds.left + RectWidth(bounds) * xRatio;
						float prevPtB = bounds.bottom - RectHeight(bounds) * yRatio;
						float dir = (ptB - prevPtB)/(ptL - prevPtL);
						float testv; 
						
						testv = dir*(_max(prevPt.h,pt.h) - prevPtL) + prevPtB;
						
						if (prevPt.v < pt.v)
						{
							if (ceil(testv) == pt.v)
								prevPt.h = pt.h;
							else if (floor(testv) == pt.v)
								prevPt.v = pt.v;
						}
						else if (prevPt.v > pt.v)
						{
							if (ceil(testv) == prevPt.v)
								prevPt.v = pt.v;
							else if (floor(testv) == prevPt.v)
								prevPt.h = pt.h;
						}
						
						if(bounds.left <= prevPt.h && prevPt.h < bounds.right
						   && bounds.top <= prevPt.v && prevPt.v < bounds.bottom)
						{
							
#ifdef IBM
							/// on the IBM, the rows of pixels are "upsidedown"
							offset = rowBytes*(long)(bounds.bottom - 1 - prevPt.v);
							/// on the IBM, for a mono map, 1 is background color,
							isBlack = !BitTst(baseAddr + offset, prevPt.h);
#else
							offset = (rowBytes*(long)prevPt.v);
							isBlack = BitTst(baseAddr + offset, prevPt.h);
#endif
							
							if (!isBlack) 
							{  // either a land point or outside a water boundary, calling code will check which is the case
								wp.p = ScreenToWorldPoint(prevPt, bounds, mapBounds);		
								if (LEsOnSurface)
									return wp; 
								else
									// reflect and check z and return, but if not inmap return as is (or return towpt?)
								{
									if (!InMap(wp.p))
									{
										if(!InMap(toWPt.p))
											return toWPt;
										else
											return wp;
									}
									if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
										goto done;
									return ReflectPoint(fromWPt,toWPt,wp);
								}
							}
							else
							{	// also check if point is on both bitmaps and if so beach it
								Boolean onLand = OnLand(ScreenToWorldPoint(prevPt, bounds, mapBounds));	// on the boundary
								if (onLand) 
								{
									wp.p = ScreenToWorldPoint(prevPt, bounds, mapBounds);	// update wp.z too
									if (LEsOnSurface)	
										return wp;
									else
									{
										if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
											goto done;
										return ReflectPoint(fromWPt,toWPt,wp);
									}
								}
							}
						}
					}
				}
				prevPt = pt;
				prevWPt = wp;
			}
		}
	}
	
done:
	
#ifdef IBM
	if(bm) GlobalUnlock(bm);
#endif
	
	if (!LEsOnSurface && InMap(toWPt.p)) // if off map let it go
	{	
		//if (toWPt.z < 0)
		//toWPt.z = -toWPt.z;
		//toWPt.z = 0.;
		if (!InVerticalMap(toWPt) || toWPt.z == 0)	// check z is ok, else use original z, or entire fromWPt
		{
			double depthAtPt = DepthAtPoint(toWPt.p);	// check depthAtPt return value
			if (depthAtPt <= 0)
			{
				OSErr err = 0;
				return fromWPt;	// something is wrong, can't force new point into vertical map
			}
			//	if (toWPt.z > depthAtPt) toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
			if (toWPt.z > depthAtPt) 
			{
				/*if (bUseSmoothing)	// just testing some ideas, probably don't want to do this
				{
					// get depth at previous point, add a kick of horizontal diffusion based on the difference in depth
					// this will flatten out the blips but also takes longer to pass through the area
					double dLong, dLat, horizontalDiffusionCoefficient = 0;
					float rand1,rand2,r,w;
					double horizontalScale = 1, depthAtPrevPt = DepthAtPoint(fromWPt.p);
					WorldPoint3D deltaPoint ={0,0,0.};
					TRandom3D* diffusionMover = model->Get3DDiffusionMover();
					
					if (diffusionMover) horizontalDiffusionCoefficient = diffusionMover->fHorizontalDiffusionCoefficient;
					if (depthAtPrevPt > depthAtPt) horizontalScale = 1 + sqrt(depthAtPrevPt - depthAtPt); // or toWPt.z ?
					//if (depthAtPrevPt > depthAtPt) horizontalScale = sqrt(depthAtPrevPt - depthAtPt); // or toWPt.z ?
					// then recheck if in vertical map and force up
					
					//horizontalDiffusionCoefficient = sqrt(2.*(fHorizontalDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT;
					horizontalDiffusionCoefficient = sqrt(2.*(horizontalDiffusionCoefficient/10000.)*model->GetTimeStep())/METERSPERDEGREELAT;
					if (depthAtPrevPt > depthAtPt) horizontalDiffusionCoefficient *= horizontalScale*horizontalScale;
					//if (depthAtPrevPt > depthAtPt) horizontalDiffusionCoefficient *= horizontalScale;
					GetRandomVectorInUnitCircle(&rand1,&rand2);
					r = sqrt(rand1*rand1+rand2*rand2);
					w = sqrt(-2*log(r)/r);
					//dLong = (rand1 * w * horizontalDiffusionCoefficient )/ LongToLatRatio3 (refPoint.pLat);
					dLong = (rand1 * w * horizontalDiffusionCoefficient )/ LongToLatRatio3 (fromWPt.p.pLat);
					dLat  = rand2 * w * horizontalDiffusionCoefficient;
					
					deltaPoint.p.pLong = dLong * 1000000;
					deltaPoint.p.pLat  = dLat  * 1000000;
					toWPt.p.pLong += deltaPoint.p.pLong;
					toWPt.p.pLat += deltaPoint.p.pLat;
					
					if (!InVerticalMap(toWPt))	// check z is ok, else use original z, or entire fromWPt
					{
						toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
					}	
				}
				else*/
					toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
			}
			if (toWPt.z <= 0) 
			{
				toWPt.z = GetRandomFloat(.01*depthAtPt,.1*depthAtPt);
			}
			//toWPt.z = fromWPt.z;
			//if (!InVerticalMap(toWPt))	
			//toWPt.p = fromWPt.p;
			//toWPt = fromWPt;
		}
	}
	
	return toWPt;
}

WorldPoint3D	Map3D::MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed)
{
	// check every pixel along the line it makes on the water bitmap
	// for any non-water point check the land bitmap as well and if it crosses a land boundary
	// force the point to the closest point in the bounds
#ifdef MAC
	BitMap bm = fWaterBitmap;
#else
	HDIB bm = fWaterBitmap;
#endif
	
	// this code is similar to IsBlackPixel
	Rect bounds;
	char* baseAddr= 0;
	long rowBytes;
	long rowByte,bitNum,byteNumber,offset;
	Point fromPt,toPt;
	Boolean isBlack = false;
	
#ifdef MAC
	bounds = bm.bounds;
	rowBytes = bm.rowBytes;
	baseAddr = bm.baseAddr;
#else //IBM
	if(bm)
	{
		LPBITMAPINFOHEADER lpDIBHdr  = (LPBITMAPINFOHEADER)GlobalLock(bm);
		baseAddr = (char*) FindDIBBits((LPSTR)lpDIBHdr);
#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)
		rowBytes = WIDTHBYTES(lpDIBHdr->biBitCount * lpDIBHdr->biWidth);
		MySetRect(&bounds,0,0,lpDIBHdr->biWidth,lpDIBHdr->biHeight);
	}
#endif
	
	Boolean LEsOnSurface = (fromWPt.z == 0 && toWPt.z == 0);
	if (toWPt.z == 0 && !isDispersed) LEsOnSurface = true;
	//Boolean LEsOnSurface = true;
	if (!gDispersedOilVersion) LEsOnSurface = true;	// something went wrong
	//if (bUseLineCrossAlgorithm) return SubsurfaceMovementCheck(fromWPt, toWPt, status);	// dispersed oil GNOME had some diagnostic options
	if(baseAddr)
	{
		// find the point in question in the bitmap
		// determine the pixel in the bitmap we need to look at
		// think of the bitmap as an array of pixels 
		long maxChange;
		WorldPoint3D wp = {0,0,0.};
		WorldRect mapBounds = this->GetMapBounds();
		
		fromPt = WorldToScreenPoint(fromWPt.p,mapBounds,bounds);
		toPt = WorldToScreenPoint(toWPt.p,mapBounds,bounds);
		
		// check the bitmap for each pixel when in range
		// so find the number of pixels change hori and vertically
		maxChange = _max(abs(toPt.h - fromPt.h),abs(toPt.v - fromPt.v));
		
		if(maxChange == 0) {
			// it's the same pixel, there is nothing to do
		}
		else { // maxChange >= 1
			long i;
			double fraction;
			Point pt, prevPt = fromPt;
			WorldPoint3D prevWPt = fromWPt;
			
			// note: there is no need to check the first or final pixel, so i only has to go to maxChange-1 
			for(i = 0; i < maxChange; i++) 
			{
				fraction = (i+1)/(double)(maxChange); 
				wp.p.pLat = (1-fraction)*fromWPt.p.pLat + fraction*toWPt.p.pLat;
				wp.p.pLong = (1-fraction)*fromWPt.p.pLong + fraction*toWPt.p.pLong;
				wp.z = (1-fraction)*fromWPt.z + fraction*toWPt.z;
				
				pt = WorldToScreenPoint(wp.p,mapBounds,bounds);
				
				// only check this pixel if it is in range
				// otherwise it is not on our map, hence not our problem
				// so assume it is water and OK
				
				if(bounds.left <= pt.h && pt.h < bounds.right
				   && bounds.top <= pt.v && pt.v < bounds.bottom)
				{
					
#ifdef IBM
					/// on the IBM, the rows of pixels are "upsidedown"
					offset = rowBytes*(long)(bounds.bottom - 1 - pt.v);
					/// on the IBM, for a mono map, 1 is background color,
					isBlack = !BitTst(baseAddr + offset, pt.h);
#else
					offset = (rowBytes*(long)pt.v);
					isBlack = BitTst(baseAddr + offset, pt.h);
#endif
					
					// don't beach LEs that are below the surface, reflect in some way
					if (!isBlack) // checking water bitmap, so not water
					{  // either a land point or outside a water boundary, calling code will check which is the case
						if (LEsOnSurface)
							return wp; 
						else
							// reflect and check z and return, but if not inmap return as is (or return towpt?)
						{
							if (!InMap(wp.p))
							{
								if(!InMap(toWPt.p))
									return toWPt;
								else
									return wp;
							}
							if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
								goto done;
							return ReflectPoint(fromWPt,toWPt,wp);
						}
					}
					else
					{	// also check if point is on both bitmaps and if so beach it
						Boolean onLand = OnLand(wp.p);	// on the boundary
						if (onLand) 
						{
							if (LEsOnSurface)	
								return wp;
							else
							{
								if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
									goto done;
								return ReflectPoint(fromWPt,toWPt,wp);
							}
						}
					}
					if (abs(pt.h - prevPt.h) == 1 && abs(pt.v - prevPt.v) == 1)
					{	// figure out which pixel was crossed
						
						float xRatio = (float)(wp.p.pLong - mapBounds.loLong) / (float)WRectWidth(mapBounds),
						yRatio = (float)(wp.p.pLat - mapBounds.loLat) / (float)WRectHeight(mapBounds);
						float ptL = bounds.left + RectWidth(bounds) * xRatio;
						float ptB = bounds.bottom - RectHeight(bounds) * yRatio;
						xRatio = (float)(prevWPt.p.pLong - mapBounds.loLong) / (float)WRectWidth(mapBounds);
						yRatio = (float)(prevWPt.p.pLat - mapBounds.loLat) / (float)WRectHeight(mapBounds);
						float prevPtL = bounds.left + RectWidth(bounds) * xRatio;
						float prevPtB = bounds.bottom - RectHeight(bounds) * yRatio;
						float dir = (ptB - prevPtB)/(ptL - prevPtL);
						float testv; 
						
						testv = dir*(_max(prevPt.h,pt.h) - prevPtL) + prevPtB;
						
						if (prevPt.v < pt.v)
						{
							if (ceil(testv) == pt.v)
								prevPt.h = pt.h;
							else if (floor(testv) == pt.v)
								prevPt.v = pt.v;
						}
						else if (prevPt.v > pt.v)
						{
							if (ceil(testv) == prevPt.v)
								prevPt.v = pt.v;
							else if (floor(testv) == prevPt.v)
								prevPt.h = pt.h;
						}
						
						if(bounds.left <= prevPt.h && prevPt.h < bounds.right
						   && bounds.top <= prevPt.v && prevPt.v < bounds.bottom)
						{
							
#ifdef IBM
							/// on the IBM, the rows of pixels are "upsidedown"
							offset = rowBytes*(long)(bounds.bottom - 1 - prevPt.v);
							/// on the IBM, for a mono map, 1 is background color,
							isBlack = !BitTst(baseAddr + offset, prevPt.h);
#else
							offset = (rowBytes*(long)prevPt.v);
							isBlack = BitTst(baseAddr + offset, prevPt.h);
#endif
							
							if (!isBlack) 
							{  // either a land point or outside a water boundary, calling code will check which is the case
								wp.p = ScreenToWorldPoint(prevPt, bounds, mapBounds);		
								if (LEsOnSurface)
									return wp; 
								else
									// reflect and check z and return, but if not inmap return as is (or return towpt?)
								{
									if (!InMap(wp.p))
									{
										if(!InMap(toWPt.p))
											return toWPt;
										else
											return wp;
									}
									if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
										goto done;
									return ReflectPoint(fromWPt,toWPt,wp);
								}
							}
							else
							{	// also check if point is on both bitmaps and if so beach it
								Boolean onLand = OnLand(ScreenToWorldPoint(prevPt, bounds, mapBounds));	// on the boundary
								if (onLand) 
								{
									wp.p = ScreenToWorldPoint(prevPt, bounds, mapBounds);	// update wp.z too
									if (LEsOnSurface)	
										return wp;
									else
									{
										if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
											goto done;
										return ReflectPoint(fromWPt,toWPt,wp);
									}
								}
							}
						}
					}
				}
				prevPt = pt;
				prevWPt = wp;
			}
		}
	}
	
done:
	
#ifdef IBM
	if(bm) GlobalUnlock(bm);
#endif
	
	if (!LEsOnSurface && InMap(toWPt.p)) // if off map let it go
	{	
		//if (toWPt.z < 0)
		//toWPt.z = -toWPt.z;
		//toWPt.z = 0.;
		if (!InVerticalMap(toWPt) || toWPt.z == 0)	// check z is ok, else use original z, or entire fromWPt
		{
			double depthAtPt = DepthAtPoint(toWPt.p);	// check depthAtPt return value
			if (depthAtPt <= 0)
			{
				OSErr err = 0;
				return fromWPt;	// something is wrong, can't force new point into vertical map
			}
			//	if (toWPt.z > depthAtPt) toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
			if (toWPt.z > depthAtPt) 
			{
				/*if (bUseSmoothing)	// just testing some ideas, probably don't want to do this
				{
					// get depth at previous point, add a kick of horizontal diffusion based on the difference in depth
					// this will flatten out the blips but also takes longer to pass through the area
					double dLong, dLat, horizontalDiffusionCoefficient = 0;
					float rand1,rand2,r,w;
					double horizontalScale = 1, depthAtPrevPt = DepthAtPoint(fromWPt.p);
					WorldPoint3D deltaPoint ={0,0,0.};
					TRandom3D* diffusionMover = model->Get3DDiffusionMover();
					
					if (diffusionMover) horizontalDiffusionCoefficient = diffusionMover->fHorizontalDiffusionCoefficient;
					if (depthAtPrevPt > depthAtPt) horizontalScale = 1 + sqrt(depthAtPrevPt - depthAtPt); // or toWPt.z ?
					//if (depthAtPrevPt > depthAtPt) horizontalScale = sqrt(depthAtPrevPt - depthAtPt); // or toWPt.z ?
					// then recheck if in vertical map and force up
					
					//horizontalDiffusionCoefficient = sqrt(2.*(fHorizontalDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT;
					horizontalDiffusionCoefficient = sqrt(2.*(horizontalDiffusionCoefficient/10000.)*model->GetTimeStep())/METERSPERDEGREELAT;
					if (depthAtPrevPt > depthAtPt) horizontalDiffusionCoefficient *= horizontalScale*horizontalScale;
					//if (depthAtPrevPt > depthAtPt) horizontalDiffusionCoefficient *= horizontalScale;
					GetRandomVectorInUnitCircle(&rand1,&rand2);
					r = sqrt(rand1*rand1+rand2*rand2);
					w = sqrt(-2*log(r)/r);
					//dLong = (rand1 * w * horizontalDiffusionCoefficient )/ LongToLatRatio3 (refPoint.pLat);
					dLong = (rand1 * w * horizontalDiffusionCoefficient )/ LongToLatRatio3 (fromWPt.p.pLat);
					dLat  = rand2 * w * horizontalDiffusionCoefficient;
					
					deltaPoint.p.pLong = dLong * 1000000;
					deltaPoint.p.pLat  = dLat  * 1000000;
					toWPt.p.pLong += deltaPoint.p.pLong;
					toWPt.p.pLat += deltaPoint.p.pLat;
					
					if (!InVerticalMap(toWPt))	// check z is ok, else use original z, or entire fromWPt
					{
						toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
					}	
				}
				else*/
					toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
			}
			if (toWPt.z <= 0) 
			{
				toWPt.z = GetRandomFloat(.01*depthAtPt,.1*depthAtPt);
			}
			//toWPt.z = fromWPt.z;
			//if (!InVerticalMap(toWPt))	
			//toWPt.p = fromWPt.p;
			//toWPt = fromWPt;
		}
	}
	
	return toWPt;
}


OSErr Map3D::MakeBitmaps()
{
	OSErr err = 0;
		
	{ // make the bitmaps etc
		Rect bitMapRect;
		long bmWidth, bmHeight;
		WorldRect wRect = this -> GetMapBounds();
		if (err) goto done;
		err = LandBitMapWidthHeight(wRect,&bmWidth,&bmHeight);
		if (err) goto done;
		MySetRect (&bitMapRect, 0, 0, bmWidth, bmHeight);
		fWaterBitmap = GetBlackAndWhiteBitmap(DrawFilledWaterTriangles2,this,wRect,bitMapRect,&err);
		if(err) goto done;
		fLandBitmap = GetBlackAndWhiteBitmap(DrawWideLandSegments2,this,wRect,bitMapRect,&err); 
		if(err) goto done;	
	}
done:	
	if(err)
	{
#ifdef MAC
		DisposeBlackAndWhiteBitMap (&fWaterBitmap);
		DisposeBlackAndWhiteBitMap (&fLandBitmap);
#else
		if(fWaterBitmap) DestroyDIB(fWaterBitmap);
		fWaterBitmap = 0;
		if(fLandBitmap) DestroyDIB(fLandBitmap);
		fLandBitmap = 0;
#endif
	}
	return err;
}

OSErr Map3D::AddMover(TMover *theMover, short where)
{
	OSErr err = 0;
	
	err = TMap::AddMover(theMover,where);
	return err;
	
}

OSErr Map3D::DropMover(TMover *theMover)
{
	long 	i, numMovers;
	OSErr	err = noErr;
	TCurrentMover *mover = 0;
	TMover *thisMover = 0;
	
	if (moverList->IsItemInList((Ptr)&theMover, &i))
	{
		if (err = moverList->DeleteItem(i))
			{ TechError("TMap::DropMover()", "DeleteItem()", err); return err; }
	}
	numMovers = moverList->GetItemCount();
	//mover = Get3DCurrentMover();
	//if (numMovers==0) err = model->DropMap(this);

	/*if (!mover)
	{
		for (i = 0; i < numMovers; i++)
		{
			this -> moverList -> GetListItem ((Ptr) &thisMover, 0); // will always want the first item in the list
			if (err = this->DropMover(thisMover)) return err; // gets rid of first mover, moves rest up
		}
		err = model->DropMap(this);
	}*/
	SetDirty (true);
	
	return err;
}

OSErr Map3D::ReplaceMap()	// code goes here, maybe not for NetCDF?
{
	char 		path[256], nameStr [256];
	short 		item, gridType;
	OSErr		err = noErr;
	Point 		where = CenteredDialogUpLeft(M38b);
	Map3D 	*map = nil;
	OSType 	typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 	reply;
/*
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, M38b, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);
#else
	sfpgetfile(&where, "",
			   (FileFilterUPP)0,
			   -1, typeList,
			   (DlgHookUPP)0,
			   &reply, M38b,
			   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	if (!reply.good) return USERCANCEL;

	my_p2cstr(reply.fName);
	#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
	#else
		strcpy(path, reply.fName);
	#endif
#endif
	if (IsPtCurFile (path))
	{
		TMap *newMap = 0;
		TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,"ptcurfile",&newMap);	// already have path
		
		if (newMover)
		{
			PtCurMover *ptCurMover = dynamic_cast<PtCurMover*>(newMover);
			err = ptCurMover -> SettingsDialog();
			if(err)	
			{ 
				newMover->Dispose(); delete newMover; newMover = 0;
				if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} 
			}
	
			if(newMover && !err)
			{
				Boolean timeFileChanged = false;
				if (!newMap) 
				{
					err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
				}
				else
				{
					err = model -> AddMap(newMap, 0);
					if (err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					err = AddMoverToMap(newMap, timeFileChanged, newMover);
					if(err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					newMover->SetMoverMap(newMap);
				}
			}
		}
		map = dynamic_cast<Map3D *>(newMap);
	}
	else if (IsNetCDFFile (path, &gridType))
	{
		TMap *newMap = 0;
		char s[256],fileName[256];
		strcpy(s,path);
		SplitPathFile (s, fileName);
		strcat (nameStr, fileName);
		TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,fileName,&newMap);	// already have path
		
		if (newMover && newMap)
		{
			NetCDFMover *netCDFMover = dynamic_cast<NetCDFMover*>(newMover);
			err = netCDFMover -> SettingsDialog();
			if(err)	
			{ 
				newMover->Dispose(); delete newMover; newMover = 0;
				if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} 
			}
	
			if(newMover && !err)
			{
				Boolean timeFileChanged = false;
				if (!newMap) 
				{
					err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
				}
				else
				{
					err = model -> AddMap(newMap, 0);
					if (err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					err = AddMoverToMap(newMap, timeFileChanged, newMover);
					if(err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					newMover->SetMoverMap(newMap);
				}
			}
		}
		else
		{
			printError("NetCDF file must include a map.");
			if (newMover) {newMover->Dispose(); delete newMover; newMover = 0;}
			if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} // shouldn't happen
			return USERCANCEL;
		}
		map = dynamic_cast<PtCurMap *>(newMap);
	}
	else 
	{
		printError("New map must be of the same type.");
		return USERCANCEL;	// to return to the dialog
	}
	{
		// put movers on the new map and activate
		TMover *thisMover = nil;
		Boolean	timeFileChanged = false;
		long k, d = this -> moverList -> GetItemCount ();
		for (k = 0; k < d; k++)
		{
			this -> moverList -> GetListItem ((Ptr) &thisMover, 0); // will always want the first item in the list
			if (!thisMover->IAm(TYPE_PTCURMOVER) && !thisMover->IAm(TYPE_NETCDFMOVERCURV) && !IAm(TYPE_NETCDFMOVERTRI) )
			{
				if (err = AddMoverToMap(map, timeFileChanged, thisMover)) return err; 
				thisMover->SetMoverMap(map);
			}
			if (err = this->DropMover(thisMover)) return err; // gets rid of first mover, moves rest up
		}
		if (err = model->DropMap(this)) return err;
		model->NewDirtNotification();
	}*/

	return err;
	
}

void Map3D::FindNearestBoundary(WorldPoint wp, long *verNum, long *segNo)
{
	long startVer = 0,i,jseg;
	//WorldPoint wp = ScreenToWorldPoint(where, MapDrawingRect(), settings.currentView);
	WorldPoint wp2;
	LongPoint lp;
	long lastVer = GetNumBoundaryPts();
	//long nbounds = GetNumBoundaries();
	long nSegs = GetNumBoundarySegs();	
	float wdist = LatToDistance(ScreenToWorldDistance(4));
	LongPointHdl ptsHdl = GetPointsHdl();
	if(!ptsHdl) return;
	*verNum= -1;
	*segNo =-1;
	for(i = 0; i < lastVer; i++)
	{
		//wp2 = (*gVertices)[i];
		lp = (*ptsHdl)[i];
		wp2.pLat = lp.v;
		wp2.pLong = lp.h;
		
		if(WPointNearWPoint(wp,wp2 ,wdist))
		{
			//for(jseg = 0; jseg < nbounds; jseg++)
			for(jseg = 0; jseg < nSegs; jseg++)
			{
				if(i <= (*fBoundarySegmentsH)[jseg])
				{
					*verNum  = i;
					*segNo = jseg;
					break;
				}
			}
		}
	} 
}

void Map3D::FindNearestBoundary(Point where, long *verNum, long *segNo)
{
	long startVer = 0,i,jseg;
	WorldPoint wp = ScreenToWorldPoint(where, MapDrawingRect(), settings.currentView);
	WorldPoint wp2;
	LongPoint lp;
	long lastVer = GetNumBoundaryPts();
	//long nbounds = GetNumBoundaries();
	long nSegs = GetNumBoundarySegs();	
	float wdist = LatToDistance(ScreenToWorldDistance(4));
	LongPointHdl ptsHdl = GetPointsHdl();	
	if(!ptsHdl) return;
	*verNum= -1;
	*segNo =-1;
	for(i = 0; i < lastVer; i++)
	{
		//wp2 = (*gVertices)[i];
		lp = (*ptsHdl)[i];
		wp2.pLat = lp.v;
		wp2.pLong = lp.h;
		
		if(WPointNearWPoint(wp,wp2 ,wdist))
		{
			//for(jseg = 0; jseg < nbounds; jseg++)
			for(jseg = 0; jseg < nSegs; jseg++)
			{
				if(i <= (*fBoundarySegmentsH)[jseg])
				{
					*verNum  = i;
					*segNo = jseg;
					break;
				}
			}
		}
	} 
}

#define Map3DReadWriteVersion 1	
OSErr Map3D::Write(BFPB *bfpb)
{
	long i,val;
	//long version = 1;
	long version = Map3DReadWriteVersion;
	ClassID id = this -> GetClassID ();
	OSErr	err = noErr;
	long numBoundarySegs = this -> GetNumBoundarySegs();
	long numBoundaryPts = this -> GetNumBoundaryPts();
	long numContourLevels = 0, numDepths = 0, numDropletSizes = 0, numSegSelected = 0;
	double val2;
	//double dropsize, probability;
	
	if (err = TMap::Write(bfpb)) return err;
		
	StartReadWriteSequence("Map3D::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	////
	id = fGrid -> GetClassID (); //JLM
	if (err = WriteMacValue(bfpb, id)) return err; //JLM
	if (err = fGrid -> Write (bfpb)) return err;
	
	/////
	if (fBoundarySegmentsH)
	{
		if (err = WriteMacValue(bfpb, numBoundarySegs)) return err;
		for (i = 0 ; i < numBoundarySegs ; i++) {
			val = INDEXH(fBoundarySegmentsH, i);
			if (err = WriteMacValue(bfpb, val)) return err;
		}
	}
	else
	{
		numBoundarySegs = 0;
		if (err = WriteMacValue(bfpb, numBoundarySegs)) return err;
	}
	/////
	if (fBoundaryTypeH)
	{
		if (err = WriteMacValue(bfpb, numBoundaryPts)) return err;
		for (i = 0 ; i < numBoundaryPts ; i++) {
			val = INDEXH(fBoundaryTypeH, i);
			if (err = WriteMacValue(bfpb, val)) return err;
		}
	}
	else
	{
		numBoundaryPts = 0;
		if (err = WriteMacValue(bfpb, numBoundaryPts)) return err;
	}
	
	/////
	if (fBoundaryPointsH)
	{
		if (err = WriteMacValue(bfpb, numBoundaryPts)) return err;
		for (i = 0 ; i < numBoundaryPts ; i++) {
			val = INDEXH(fBoundaryPointsH, i);
			if (err = WriteMacValue(bfpb, val)) return err;
		}
	}
	else
	{	// only curvilinear netcdf algorithm uses the full set of boundary pts
		numBoundaryPts = 0;
		if (err = WriteMacValue(bfpb, numBoundaryPts)) return err;
	}
	
	if (err = WriteMacValue(bfpb,bDrawLandBitMap)) return err;
	if (err = WriteMacValue(bfpb,bDrawWaterBitMap)) return err;
	
	//if (err = WriteMacValue(bfpb,fLegendRect)) return err;
	//if (err = WriteMacValue(bfpb,bShowLegend)) return err;
	if (err = WriteMacValue(bfpb,bShowDepthContours)) return err;
	
	//if (err = WriteMacValue(bfpb,fWaterDensity)) return err;
	//if (err = WriteMacValue(bfpb,fMixedLayerDepth)) return err;
	//if (err = WriteMacValue(bfpb,fBreakingWaveHeight)) return err;

	//if (err = WriteMacValue(bfpb,fDiagnosticStrType)) return err;

	//if (err = WriteMacValue(bfpb,fWaveHtInput)) return err;

	//if (fDropletSizesH) numDropletSizes = _GetHandleSize((Handle)fDropletSizesH)/sizeof(**fDropletSizesH);
	
	//if (err = WriteMacValue(bfpb, numDropletSizes)) return err;
	/*for (i = 0 ; i < numDropletSizes ; i++) {
		dropsize = INDEXH(fDropletSizesH, i).dropletSize;
		probability = INDEXH(fDropletSizesH, i).probability;
		if (err = WriteMacValue(bfpb, dropsize)) return err;
		if (err = WriteMacValue(bfpb, probability)) return err;
	}*/
	
	return 0;
}

OSErr Map3D::Read(BFPB *bfpb)
{
	long i,version,val;
	ClassID id;
	OSErr err = 0;
	long 	numBoundarySegs,numBoundaryPts,numContourLevels,numDepths,numDropletSizes,numSegSelected;
	float depthVal;
	double val2;
	double dropsize, probability;
	
	if (err = TMap::Read(bfpb)) return err;

	StartReadWriteSequence("Map3D::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("Map3D::Read()", "id == TYPE_MAP3D", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version < 1 || version > Map3DReadWriteVersion) { printSaveFileVersionError(); return -1; }
	
	// read the type of grid used for the map (should always be trigrid...)
	if (err = ReadMacValue(bfpb,&id)) return err;
	switch(id)
	{
		case TYPE_RECTGRIDVEL: fGrid = new TRectGridVel;break;
		case TYPE_TRIGRIDVEL: fGrid = new TTriGridVel;break;
		case TYPE_TRIGRIDVEL3D: fGrid = new TTriGridVel3D;break;
		default: printError("Unrecognized Grid type in Map3D::Read()."); return -1;
	}
	
	if (err = fGrid -> Read (bfpb)) return err;

	if (err = ReadMacValue(bfpb, &numBoundarySegs)) return err;	
	if (!err && numBoundarySegs>0)
	{
		fBoundarySegmentsH = (LONGH)_NewHandleClear(sizeof(long)*numBoundarySegs);
		if (!fBoundarySegmentsH)
			{ TechError("Map3D::Read()", "_NewHandleClear()", 0); return -1; }
		
		for (i = 0 ; i < numBoundarySegs ; i++) {
			if (err = ReadMacValue(bfpb, &val)) { printSaveFileVersionError(); return -1; }
			INDEXH(fBoundarySegmentsH, i) = val;
		}
	}
	if (err = ReadMacValue(bfpb, &numBoundaryPts)) return err;	
	if (!err && numBoundaryPts>0)
	{
		fBoundaryTypeH = (LONGH)_NewHandleClear(sizeof(long)*numBoundaryPts);
		if (!fBoundaryTypeH)
			{ TechError("Map3D::Read()", "_NewHandleClear()", 0); return -1; }
		
		for (i = 0 ; i < numBoundaryPts ; i++) {
			if (err = ReadMacValue(bfpb, &val)) { printSaveFileVersionError(); return -1; }
			INDEXH(fBoundaryTypeH, i) = val;
		}
	}
	
	//if (version>1 && !gMearnsVersion)	// 1/9/04 redid algorithm, now store all boundary points
	{
		if (err = ReadMacValue(bfpb, &numBoundaryPts)) return err;	
		if (!err && numBoundaryPts>0)
		{
			fBoundaryPointsH = (LONGH)_NewHandleClear(sizeof(long)*numBoundaryPts);
			if (!fBoundaryPointsH)
				{ TechError("Map3D::Read()", "_NewHandleClear()", 0); return -1; }
			
			for (i = 0 ; i < numBoundaryPts ; i++) {
				if (err = ReadMacValue(bfpb, &val)) { printSaveFileVersionError(); return -1; }
				INDEXH(fBoundaryPointsH, i) = val;
			}
		}
	}
	

	if (err = ReadMacValue(bfpb, &bDrawLandBitMap)) return err;
	if (err = ReadMacValue(bfpb, &bDrawWaterBitMap)) return err;
	
	//if (err = ReadMacValue(bfpb, &fLegendRect)) return err;
	//if (err = ReadMacValue(bfpb, &bShowLegend)) return err;
	if (err = ReadMacValue(bfpb, &bShowDepthContours)) return err;

	//if (err = ReadMacValue(bfpb, &fWaterDensity)) return err;
	//if (err = ReadMacValue(bfpb, &fMixedLayerDepth)) return err;
	//if (err = ReadMacValue(bfpb, &fBreakingWaveHeight)) return err;
	//if (err = ReadMacValue(bfpb, &fDiagnosticStrType)) return err;
	
	//if (err = ReadMacValue(bfpb, &fWaveHtInput)) return err;
	/*if ((gMearnsVersion && version>1) || version>2)
	{
		if (err = ReadMacValue(bfpb, &numDropletSizes)) return err;
		fDropletSizesH = (DropletInfoRecH)_NewHandleClear(sizeof(DropletInfoRec)*numDropletSizes);
		if(!fDropletSizesH)
			{ err = -1; TechError("Map3D::Read()", "_NewHandle()", 0); return err; }
		for (i = 0 ; i < numDropletSizes ; i++) {
			if (err = ReadMacValue(bfpb, &dropsize)) return err;
			if (err = ReadMacValue(bfpb, &probability)) return err;
			INDEXH(fDropletSizesH, i).dropletSize = dropsize;
			INDEXH(fDropletSizesH, i).probability = probability;
		}
	}*/
	//////////////////
	// now reconstruct the offscreen Land and Water bitmaps
	///////////////////
	//if (gMearnsVersion) SetMinDistOffshore(GetMapBounds());	// I don't think this does anything...
	
	if (!(this->IAm(TYPE_COMPOUNDMAP)))
	{
		Rect bitMapRect;
		long bmWidth, bmHeight;
		WorldRect wRect = this -> GetMapBounds(); // bounds have been read in by the base class
		err = LandBitMapWidthHeight(wRect,&bmWidth,&bmHeight);
		if (err) {printError("Unable to recreate bitmap in Map3D::Read"); return err;}
		MySetRect (&bitMapRect, 0, 0, bmWidth, bmHeight);

		fLandBitmap = GetBlackAndWhiteBitmap(DrawWideLandSegments2,this,wRect,bitMapRect,&err);

		if(!err)
			fWaterBitmap = GetBlackAndWhiteBitmap(DrawFilledWaterTriangles2,this,wRect,bitMapRect,&err);
 		
		switch(err) 
		{
			case noErr: break;
			case memFullErr: printError("Out of memory in Map::Read"); break;
			default: TechError("Map3D::Read", "GetBlackAndWhiteBitmap", err); break;
		}
	}
	return 0;
}

/**************************************************************************************************/
long Map3D::GetListLength()
{
	long i, n, count = 1;
	TMover *mover;

		//if (bIAmPartOfACompoundMap) {if (bOpen) count++; return count;}

	if (bOpen) {
		count++;// name

 		count++; // REFLOATHALFLIFE

		count++; // bitmap-visible-box

		//count++; // show grid
		count++; // show depth contours
		//if(this->ThereIsADispersedSpill()) count++; // draw contours
		//if(this->ThereIsADispersedSpill()) count++; // set contours
		//if(this->ThereIsADispersedSpill()) count++; // draw legend
		//if(this->ThereIsADispersedSpill()) count++; // concentration table

		//if(this->ThereIsADispersedSpill()) count++; // surface LEs
		//if(this->ThereIsADispersedSpill()) count++; // water density
		//if(this->ThereIsADispersedSpill()) count++; // mixed layer depth
		//if(this->ThereIsADispersedSpill()) count++; // breaking wave height

		//if(this->ThereIsADispersedSpill()) count++; // diagnostic string info
		//if(this->ThereIsADispersedSpill()) count++; // droplet data

		if (bMoversOpen)
			for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
				moverList->GetListItem((Ptr)&mover, i);
				count += mover->GetListLength();
			}

	}

	return count;
}
/**************************************************************************************************/
ListItem Map3D::GetNthListItem(long n, short indent, short *style, char *text)
{
	long i, m, count;
	TMover *mover;
	ListItem item = { this, 0, indent, 0 };
		
	if (n == 0) {
		item.index = I_3DMAPNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		strcpy(text, className);
		
		return item;
	}
	n -= 1;

	/*if (bIAmPartOfACompoundMap && bOpen) {
	if (n == 0) {
		item.indent++;
		item.index = I_3DDRAWCONTOURSFORMAP;
		item.bullet = bDrawContours ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
		strcpy(text, "Draw Contours");
		
		return item;
	}
	n -= 1;
	}*/

	if (bIAmPartOfACompoundMap) { item.owner = 0;return item;}
	
	if (n == 0) {
		item.index = I_3DREFLOATHALFLIFE; // override the index
		item.indent = indent; // override the indent
			sprintf(text, "Refloat half life: %g hr",fRefloatHalfLifeInHrs);
		return item;
	}
	n -= 1;
	
	if (n == 0) {
		item.indent++;
		item.index = I_3DDRAWLANDWATERBITMAP;
		item.bullet = (bDrawLandBitMap && bDrawWaterBitMap) ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
		strcpy(text, "Show Land / Water Map");
		
		return item;
	}
	n -= 1;
	
	/*if (n == 0) {
		item.index = I_3DSHOWGRID;
		item.bullet = bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
		sprintf(text, "Show Grid");
		item.indent++;
		return item;
	}
	n -= 1;*/
		
	if (n == 0) {
		item.index = I_3DSHOWDEPTHCONTOURS;
		item.bullet = bShowDepthContours ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
		sprintf(text, "Show Depth Contours");
		item.indent++;
		return item;
	}
	n -= 1;
		
	/*if(this ->ThereIsADispersedSpill())
	{
		if (n == 0) {
			//item.indent++;
			item.index = I_3DWATERDENSITY;
			sprintf(text, "Water Density : %ld (kg/m^3)",fWaterDensity);
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			//item.indent++;
			item.index = I_3DMIXEDLAYERDEPTH;
			sprintf(text, "Mixed Layer Depth : %g m",fMixedLayerDepth);
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			//item.indent++;
			item.index = I_3DBREAKINGWAVEHT;
			//sprintf(text, "Breaking Wave Height : %g m",fBreakingWaveHeight);
			sprintf(text, "Breaking Wave Height : %g m",GetBreakingWaveHeight());
			//if (fWaveHtInput==0) 	// user input value by hand, also show wind speed?				
			
			return item;
		}
		n -= 1;
	}*/
	
	
	if (bOpen) {
		indent++;
		if (n == 0) {
			item.index = I_3DMOVERS;
			item.indent = indent;
			item.bullet = bMoversOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Movers");
			
			return item;
		}
		
		n -= 1;
		
		if (bMoversOpen)
			for (i = 0, m = moverList->GetItemCount() ; i < m ; i++) {
				moverList->GetListItem((Ptr)&mover, i);
				count = mover->GetListLength();
				if (count > n)
				{
					item =  mover->GetNthListItem(n, indent + 1, style, text);
					if (mover->bActive) *style = italic;
					return item;
					//return mover->GetNthListItem(n, indent + 1, style, text);
				}
				
				n -= count;
			}
	}
	
	item.owner = 0;
	
	return item;
}

/**************************************************************************************************/
Boolean Map3D::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet) {
		switch (item.index) {
			case I_3DMAPNAME: bOpen = !bOpen; return TRUE;
			case I_3DDRAWLANDWATERBITMAP:
				bDrawLandBitMap = !bDrawLandBitMap;
				bDrawWaterBitMap = !bDrawWaterBitMap;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			//case I_3DSHOWLEGEND:
				//bShowLegend = !bShowLegend;
				///model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			//case I_3DSHOWGRID:
				//bShowGrid = !bShowGrid;
				//model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_3DSHOWDEPTHCONTOURS:
				bShowDepthContours = !bShowDepthContours;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_3DMOVERS: bMoversOpen = !bMoversOpen; return TRUE;
			//case I_3DDRAWCONTOURSFORMAP: bDrawContours = !bDrawContours; return TRUE;
		}
	}
	
	if (doubleClick) 
	{
		if (this -> FunctionEnabled(item, SETTINGSBUTTON)) 
		{
			//if (item.index == I_3DDRAWCONTOURSFORMAP) 
				//return TRUE;
			if (item.index == I_3DSHOWDEPTHCONTOURS)		
				//if (err = triGrid->DepthContourDialog()) break;
			{
				((TTriGridVel*)fGrid)->DepthContourDialog(); return TRUE;
			}
			
			item.index = I_3DMAPNAME;
			this -> SettingsItem(item);
			return TRUE;
		}

		if (item.index == I_3DMOVERS)
		{
			item.owner -> AddItem (item);
			return TRUE;
		}
		
	}

	return false;
}
/**************************************************************************************************/
Boolean Map3D::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	
	switch (item.index) {
		case I_3DMAPNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE; 
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (!model->mapList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (model->mapList->GetItemCount() - 1);
					}
			}
			break;
		case I_3DMOVERS:
			switch (buttonID) {
				case ADDBUTTON: return TRUE;
				case SETTINGSBUTTON: return FALSE;
				case DELETEBUTTON: return FALSE;
			}
			break;
		case I_3DDRAWLANDWATERBITMAP:
		//case I_3DSHOWGRID:
		case I_3DSHOWDEPTHCONTOURS:
		//case I_PDRAWCONTOURS:
		//case I_PSHOWLEGEND:
		//case I_PSHOWSURFACELES:
		//case I_PSETCONTOURS:
		//case I_PCONCTABLE:
		case I_3DWATERDENSITY:
		case I_3DMIXEDLAYERDEPTH:
		case I_3DBREAKINGWAVEHT:
		//case I_PDIAGNOSTICSTRING:
		//case I_PDROPLETINFO:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
		case I_3DREFLOATHALFLIFE:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
			break;
	}
	
	return FALSE;
}

/**************************************************************************************************/

void Map3D::DrawDepthContourScale(Rect r, WorldRect view)
{
	Point		p;
	short		h,v,x,y,dY,widestNum=0;
	RGBColor	rgb;
	Rect		rgbrect;
	char 		numstr[30],numstr2[30],text[30];
	long 		i,numLevels;
	double	minLevel, maxLevel;
	double 	value;
	TTriGridVel* triGrid = GetGrid();	
	
	//triGrid->DrawContourScale(r,view);

	return;
}

void Map3D::Draw(Rect r, WorldRect view)
{
	/////////////////////////////////////////////////
	// JLM 6/10/99 maps must erase their rectangles in case a lower priority map drew in our rectangle
	// CMO 11/16/00 maps must erase their polygons in case a lower priority map drew in our polygon
	LongRect	mapLongRect;
	Rect m;
	Boolean  onQuickDrawPlane, changedLineWidth = false;
	WorldRect bounds = this -> GetMapBounds();
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	RgnHandle saveClip=0, newClip=0;
	
	mapLongRect.left = SameDifferenceX(bounds.loLong);
	mapLongRect.top = (r.bottom + r.top) - SameDifferenceY(bounds.hiLat);
	mapLongRect.right = SameDifferenceX(bounds.hiLong);
	mapLongRect.bottom = (r.bottom + r.top) - SameDifferenceY(bounds.loLat);
	onQuickDrawPlane = IntersectToQuickDrawPlane(mapLongRect,&m);

	if (AreUsingThinLines())
	{
		StopThinLines();
		changedLineWidth = true;
	}
	//EraseRect(&m); 
	//EraseReg instead
	if (fBoundaryTypeH)
	{
#ifdef MAC
		saveClip = NewRgn(); //
		if(saveClip) {
			GetClip(saveClip);///
			newClip = NewRgn();
			if(newClip) {
				OpenRgn();
				DrawBoundaries(r);
				CloseRgn(newClip);
				EraseRgn(newClip);		
				DisposeRgn(newClip);
			}
			SetClip(saveClip);//
			DisposeRgn(saveClip);
		}
#else
		EraseRegion(r);
#endif
	}

	/////////////////////////////////////////////////

	if (fBoundaryTypeH) DrawBoundaries(r);

	if (this -> bDrawWaterBitMap && onQuickDrawPlane)
		DrawDIBImage(LIGHTBLUE,&fWaterBitmap,m);
		
	if (this -> bDrawLandBitMap && onQuickDrawPlane)
		DrawDIBImage(DARKGREEN,&fLandBitmap,m);
		
	//////
	
	if (changedLineWidth)
	{
		StartThinLines();
	}
	TMap::Draw(r, view);

	//if(fGrid && bShowGrid)
		//fGrid->Draw(r,view,refPt,refScale,arrowScale,arrowDepth,bShowArrows,bShowGrid,fColor);
	//if (bShowDepthContours) ((TTriGridVel3D*)fGrid)->DrawDepthContours(r,view,bShowDepthContourLabels);
	if (bShowDepthContours) ((TTriGridVel*)fGrid)->DrawDepthContours(r,view,false);
	//if (bShowGrid) fGrid->Draw(r,view,wayOffMapPt,1.0,1.0,0.,false,bShowGrid,colors[BLACK]);

}

void Map3D::DrawBoundaries(Rect r)
{
	long nSegs = GetNumBoundarySegs();	
	long theSeg,startver,endver,j;
	long x,y;
	Point pt;
	Boolean offQuickDrawPlane = false;

	long penWidth = 3;
	long halfPenWidth = penWidth/2;

	PenNormal();
	RGBColor sc;
	GetForeColor(&sc);
	
	// to support new curvilinear algorithm
	if (fBoundaryPointsH)
	{
		DrawBoundaries2(r);
		return;
	}

	LongPointHdl ptsHdl = GetPointsHdl();	
	if(!ptsHdl) return;

#ifdef MAC
	PenSize(penWidth,penWidth);
#else
	PenStyle(BLACK,penWidth);
#endif

	// have each seg be a polygon with a fill option - land only, maybe fill with a pattern?
	for(theSeg = 0; theSeg < nSegs; theSeg++)
	{
		startver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
		endver = (*fBoundarySegmentsH)[theSeg]+1;
	
		pt = GetQuickDrawPt((*ptsHdl)[startver].h,(*ptsHdl)[startver].v,&r,&offQuickDrawPlane);
		MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		for(j = startver + 1; j < endver; j++)
		{
			if ((*fBoundaryTypeH)[j]==2)	// a water boundary
				RGBForeColor(&colors[BLUE]);
			else// add option to change color, light or dark depending on which is easier to see , see premerge GNOME_beta
			{
				RGBForeColor(&colors[BROWN]);	// land
			}
			pt = GetQuickDrawPt((*ptsHdl)[j].h,(*ptsHdl)[j].v,&r,&offQuickDrawPlane);
			if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[j]==1))
			{
				MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
			}
			else
				MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		}
		if ((*fBoundaryTypeH)[startver]==2)	// a water boundary
			RGBForeColor(&colors[BLUE]);
		else
		{
			RGBForeColor(&colors[BROWN]);	// land
		}
		pt = GetQuickDrawPt((*ptsHdl)[startver].h,(*ptsHdl)[startver].v,&r,&offQuickDrawPlane);
		if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[startver]==1))
		{
			MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		}
		else
			MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
	}

#ifdef MAC
	PenSize(1,1);
#else
	PenStyle(BLACK,1);
#endif
	RGBForeColor(&sc);
}

void Map3D::DrawBoundaries2(Rect r)
{
	// should combine into original DrawBoundaries, just check for fBoundaryPointsH
	PenNormal();
	RGBColor sc;
	GetForeColor(&sc);
	
	//TMover *mover=0;

	long nSegs = GetNumBoundarySegs();	
	long theSeg,startver,endver,j;
	long x,y,index1,index;
	Point pt;
	Boolean offQuickDrawPlane = false;

	long penWidth = 3;
	long halfPenWidth = penWidth/2;

	LongPointHdl ptsHdl = GetPointsHdl();	
	if(!ptsHdl) return;
	
#ifdef MAC
	PenSize(penWidth,penWidth);
#else
	PenStyle(BLACK,penWidth);
#endif


	for(theSeg = 0; theSeg < nSegs; theSeg++)
	{
		startver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
		endver = (*fBoundarySegmentsH)[theSeg]+1;
		index1 = (*fBoundaryPointsH)[startver];
		pt = GetQuickDrawPt((*ptsHdl)[index1].h,(*ptsHdl)[index1].v,&r,&offQuickDrawPlane);
		MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		for(j = startver + 1; j < endver; j++)
		{
			index = (*fBoundaryPointsH)[j];
     		if ((*fBoundaryTypeH)[j]==2)	// a water boundary
				RGBForeColor(&colors[BLUE]);
			else
				RGBForeColor(&colors[BROWN]);	// land
			pt = GetQuickDrawPt((*ptsHdl)[index].h,(*ptsHdl)[index].v,&r,&offQuickDrawPlane);
			if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[j]==1))
			{
				MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
			}
			else
				MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		}
		if ((*fBoundaryTypeH)[startver]==2)	// a water boundary
			RGBForeColor(&colors[BLUE]);
		else
			RGBForeColor(&colors[BROWN]);	// land
		pt = GetQuickDrawPt((*ptsHdl)[index1].h,(*ptsHdl)[index1].v,&r,&offQuickDrawPlane);
		if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[startver]==1))
		{
			MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		}
		else
			MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
	}

#ifdef MAC
	PenSize(1,1);
#else
	PenStyle(BLACK,1);
#endif
	RGBForeColor(&sc);
}

/**************************************************************************************************/
#ifdef IBM
void Map3D::EraseRegion(Rect r)
{
	long nSegs = GetNumBoundarySegs();	
	long theSeg,startver,endver,j,index;
	Point pt;
	Boolean offQuickDrawPlane = false;

	LongPointHdl ptsHdl = GetPointsHdl(); 
	if(!ptsHdl) return;

	for(theSeg = 0; theSeg< nSegs; theSeg++)
	{
		startver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
		endver = (*fBoundarySegmentsH)[theSeg]+1;
		long numPts = endver - startver;
		POINT *pointsPtr = (POINT*)_NewPtr(numPts *sizeof(POINT));
		RgnHandle newClip=0;
		HBRUSH whiteBrush;
	
		for(j = startver; j < endver; j++)
		{
			if (fBoundaryPointsH)	// the reordered curvilinear grid
				index = (*fBoundaryPointsH)[j];
			else index = j;
			pt = GetQuickDrawPt((*ptsHdl)[index].h,(*ptsHdl)[index].v,&r,&offQuickDrawPlane);
			(pointsPtr)[j-startver] = MakePOINT(pt.h,pt.v);
		}

		newClip = CreatePolygonRgn((const POINT*)pointsPtr,numPts,ALTERNATE);
		whiteBrush = (HBRUSH)GetStockObject(WHITE_BRUSH);
		//err = SelectClipRgn(currentHDC,savedClip);
		FillRgn(currentHDC, newClip, whiteBrush);
		DisposeRgn(newClip);
		//DeleteObject(newClip);
		//SelectClipRgn(currentHDC,0);
		if(pointsPtr) {_DisposePtr((Ptr)pointsPtr); pointsPtr = 0;}
	}

}
#endif
/**************************************************************************************************/

/////////////////////////////////////////////////
Rect Map3D::DoArrowTool(long triNum)	
{	// show depth concentration profile at selected triangle
	long n,listIndex,numDepths=0;
	Rect r = MapDrawingRect();
	return r;
}
/////////////////////////////////////////////////////////////////
OSErr Map3D::ReadCATSMap(char *path) 
{
	char s[1024], errmsg[256];
	long i, numPoints, line = 0;
	CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH depths=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0;
	
	errmsg[0]=0;
	
	
	if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("Map3D::ReadCATSMap()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); 
	
	MySpinCursor(); 
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTVerticesHeaderLine(s, &numPoints))
	{
		MySpinCursor();
		err = ReadTVerticesBody(f,&line,&pts,&depths,errmsg,numPoints,true);
		if(err) goto done;
	}
	else
	{
		err = -1; 
		goto done;
	}
	MySpinCursor();
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsBoundarySegmentHeaderLine(s,&numBoundarySegs)) // Boundary data from CATS
	{
		MySpinCursor();
		err = ReadBoundarySegs(f,&line,&boundarySegs,numBoundarySegs,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		err = -1; 
		goto done;
	}
	MySpinCursor(); 
	
	if(IsWaterBoundaryHeaderLine(s,&numWaterBoundaries,&numBoundaryPts)) // Boundary types from CATS
	{
		MySpinCursor();
		err = ReadWaterBoundaries(f,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		err = -1; 
		goto done;
	}
	
	if(IsTTopologyHeaderLine(s,&numPoints)) // Topology from CATS
	{
		MySpinCursor();
		err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numPoints,TRUE);
		if(err) goto done;
	}
	else
	{
		//if (!haveBoundaryData) {err=-1; strcpy(errmsg,"File must have boundary data to create topology"); goto done;}
		//DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage("Making Triangles");
		if (err = maketriangles(&topo,pts,numPoints,boundarySegs,numBoundarySegs))  // use maketriangles.cpp
			err = -1; // for now we require TTopology
		// code goes here, support Galt style ??
		DisplayMessage(0);
		velH = (VelocityFH)_NewHandleClear(sizeof(**velH)*numPoints);
		if(!velH)
		{
			strcpy(errmsg,"Not enough memory.");
			goto done;
		}
		for (i=0;i<numPoints;i++)
		{
			INDEXH(velH,i).u = 0.;
			INDEXH(velH,i).v = 0.;
		}
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATS
	{
		MySpinCursor();
		err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
		if(err) goto done;
	}
	else
	{
		//DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage("Making Dag Tree");
		tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); // use CATSDagTree.cpp and my_build_list.h
		DisplayMessage(0);
		if (errmsg[0])	
			err = -1; // for now we require TIndexedDagTree
		// code goes here, support Galt style ??
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	// figure out the bounds
	bounds = voidWorldRect;
	long numPts;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		numPts = _GetHandleSize((Handle)pts)/sizeof(LongPoint);
		if(numPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
		}
	}
	
	
	/////////////////////////////////////////////////
	// create the bathymetry map 
	
	if (waterBoundaries)
	{
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(boundarySegs);	
		this->SetWaterBoundaries(waterBoundaries);
		this->SetMapBounds(bounds);
	}
	else
	{
		err = -1;
		goto done;
	}
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in Map3D::ReadCATSMap()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(bounds); //need to set map bounds too
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		printError("Unable to read Triangle Velocity file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	triGrid -> SetBathymetry(depths);
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	depths = 0; // because fGrid is now responsible for it
	
done:
	
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in Map3D::ReadCATSMap");
		printError(errmsg); 
		if(pts)DisposeHandle((Handle)pts);
		if(topo)DisposeHandle((Handle)topo);
		if(velH)DisposeHandle((Handle)velH);
		if(tree.treeHdl)DisposeHandle((Handle)tree.treeHdl);
		if(depths)DisposeHandle((Handle)depths);
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(boundarySegs)DisposeHandle((Handle)boundarySegs);
		if(waterBoundaries)DisposeHandle((Handle)waterBoundaries);
	}
	return err;
}
OSErr Map3D::GetPointsAndMask(char *path,DOUBLEH *maskH,WORLDPOINTFH *vertexPointsH, FLOATH *depthPointsH, long *numRows, long *numCols)	
{
	// this code is for curvilinear grids
	OSErr err = 0;
	long i,j,k, numScanned, indexOfStart = 0;
	int status, ncid, latIndexid, lonIndexid, latid, lonid, recid, sigmaid, sigmavarid, sigmavarid2, hcvarid, depthid, depthdimid, depthvarid, mask_id, numdims;
	size_t latLength, lonLength, t_len2, sigmaLength=0;
	float startLat,startLon,endLat,endLon,hc_param=0.;
	char dimname[NC_MAX_NAME], s[256], topPath[256], outPath[256];
	WORLDPOINTFH vertexPtsH=0;
	FLOATH totalDepthsH=0, sigmaLevelsH=0;
	double *lat_vals=0,*lon_vals=0,timeVal;
	float *depth_vals=0,*sigma_vals=0,*sigma_vals2=0;
	static size_t latIndex=0,lonIndex=0,timeIndex,ptIndex[2]={0,0},sigmaIndex=0;
	static size_t pt_count[2], sigma_count;
	char errmsg[256] = "";
	char fileName[64],*modelTypeStr=0;
	Point where;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	Boolean bTopFile = false, isLandMask = true, fIsNavy = false;
	VelocityFH velocityH = 0;
	static size_t mask_index[] = {0,0};
	static size_t mask_count[2];
	double *landmask = 0; 
	DOUBLEH mylandmaskH=0;
	//long numTimesInFile = 0;
	long fNumRows, fNumCols;
	short fGridType;
	
	if (!path || !path[0]) return 0;
	//strcpy(fVar.pathName,path);
	
	strcpy(s,path);
	SplitPathFile (s, fileName);
	//strcpy(fVar.userName, fileName); // maybe use a name from the file
	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	// check number of dimensions - 2D or 3D
	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_inq_attlen(ncid,NC_GLOBAL,"generating_model",&t_len2);
	if (status != NC_NOERR) {status = nc_inq_attlen(ncid,NC_GLOBAL,"generator",&t_len2); if (status != NC_NOERR) {fIsNavy = false; /*goto done;*/}}	// will need to split for Navy vs LAS
	else 
	{
		fIsNavy = true;
		// may only need to see keyword is there, since already checked grid type
		modelTypeStr = new char[t_len2+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "generating_model", modelTypeStr);
		if (status != NC_NOERR) {status = nc_inq_attlen(ncid,NC_GLOBAL,"generator",&t_len2); if (status != NC_NOERR) {fIsNavy = false; goto done;}}	// will need to split for regridded or non-Navy cases 
		modelTypeStr[t_len2] = '\0';
		
		//strcpy(fVar.userName, modelTypeStr); // maybe use a name from the file
		/*
		 if (!strncmp (modelTypeStr, "SWAFS", 5))
		 fIsNavy = true;
		 else if (!strncmp (modelTypeStr, "fictitious test data", strlen("fictitious test data")))
		 fIsNavy = true;
		 else
		 fIsNavy = false;*/
	}		
	
	if (fIsNavy)
	{
		status = nc_inq_dimid(ncid, "gridy", &latIndexid); //Navy
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimlen(ncid, latIndexid, &latLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimid(ncid, "gridx", &lonIndexid);	//Navy
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimlen(ncid, lonIndexid, &lonLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		// option to use index values?
		status = nc_inq_varid(ncid, "grid_lat", &latid);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_varid(ncid, "grid_lon", &lonid);
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	else
	{
		for (i=0;i<numdims;i++)
		{
			//if (i == recid) continue;
			status = nc_inq_dimname(ncid,i,dimname);
			if (status != NC_NOERR) {err = -1; goto done;}
			//if (!strncmpnocase(dimname,"X",1) || !strncmpnocase(dimname,"LON",3))
			if (!strncmpnocase(dimname,"X",1) || !strncmpnocase(dimname,"LON",3) || !strncmpnocase(dimname,"NX",2))
			{
				lonIndexid = i;
			}
			//if (!strncmpnocase(dimname,"Y",1) || !strncmpnocase(dimname,"LAT",3))
			if (!strncmpnocase(dimname,"Y",1) || !strncmpnocase(dimname,"LAT",3) || !strncmpnocase(dimname,"NY",2))
			{
				latIndexid = i;
			}
		}
		
		status = nc_inq_dimlen(ncid, latIndexid, &latLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		
		status = nc_inq_dimlen(ncid, lonIndexid, &lonLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		
		status = nc_inq_varid(ncid, "LATITUDE", &latid);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "lat", &latid);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		status = nc_inq_varid(ncid, "LONGITUDE", &lonid);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "lon", &lonid);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
	}
	
	pt_count[0] = latLength;
	pt_count[1] = lonLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(latLength*lonLength*sizeof(WorldPointF));
	if (!vertexPtsH) {err = memFullErr; goto done;}
	lat_vals = new double[latLength*lonLength]; 
	lon_vals = new double[latLength*lonLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	status = nc_get_vara_double(ncid, latid, ptIndex, pt_count, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_double(ncid, lonid, ptIndex, pt_count, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<latLength;i++)
	{
		for (j=0;j<lonLength;j++)
		{
			// grid ordering does matter for creating ptcurmap, assume increases fastest in x/lon, then in y/lat
			INDEXH(vertexPtsH,i*lonLength+j).pLat = lat_vals[(latLength-i-1)*lonLength+j];	
			INDEXH(vertexPtsH,i*lonLength+j).pLong = lon_vals[(latLength-i-1)*lonLength+j];
		}
	}
	*vertexPointsH	= vertexPtsH;// get first and last, lat/lon values, then last-first/total-1 = dlat/dlon
	
	
	status = nc_inq_dimid(ncid, "sigma", &sigmaid); 	
	//if (status != NC_NOERR || fIsNavy) {fVar.gridType = TWO_D; /*err = -1; goto done;*/}	// check for zgrid option here
	if (status != NC_NOERR)
	{
		status = nc_inq_dimid(ncid, "depth", &depthdimid); 
		if (status != NC_NOERR || fIsNavy) 
		{
			fGridType = TWO_D; /*err = -1; goto done;*/
		}	
		else
		{// check for zgrid option here
			fGridType = MULTILAYER; /*err = -1; goto done;*/
			status = nc_inq_varid(ncid, "depth", &sigmavarid); //Navy
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, depthdimid, &sigmaLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			//fVar.maxNumDepths = sigmaLength;
			sigma_vals = new float[sigmaLength];
			if (!sigma_vals) {err = memFullErr; goto done;}
			sigma_count = sigmaLength;
			status = nc_get_vara_float(ncid, sigmavarid, &sigmaIndex, &sigma_count, sigma_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
	}
	else
	{
		status = nc_inq_varid(ncid, "sigma", &sigmavarid); //Navy
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "sc_r", &sigmavarid);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_varid(ncid, "Cs_r", &sigmavarid2);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			fGridType = SIGMA_ROMS;
			//fVar.maxNumDepths = sigmaLength;
			sigma_vals = new float[sigmaLength];
			if (!sigma_vals) {err = memFullErr; goto done;}
			sigma_vals2 = new float[sigmaLength];
			if (!sigma_vals2) {err = memFullErr; goto done;}
			sigma_count = sigmaLength;
			status = nc_get_vara_float(ncid, sigmavarid, &sigmaIndex, &sigma_count, sigma_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_get_vara_float(ncid, sigmavarid2, &sigmaIndex, &sigma_count, sigma_vals2);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_varid(ncid, "hc", &hcvarid);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_get_var1_float(ncid, hcvarid, &sigmaIndex, &hc_param);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		else
		{
			// code goes here, for SIGMA_ROMS the variable isn't sigma but sc_r and Cs_r, with parameter hc
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			// check if sigmaLength > 1
			fGridType = SIGMA;
			//fVar.maxNumDepths = sigmaLength;
			sigma_vals = new float[sigmaLength];
			if (!sigma_vals) {err = memFullErr; goto done;}
			sigma_count = sigmaLength;
			status = nc_get_vara_float(ncid, sigmavarid, &sigmaIndex, &sigma_count, sigma_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		// once depth is read in 
	}
	
	status = nc_inq_varid(ncid, "depth", &depthid);	// this is required for sigma or multilevel grids
	if (status != NC_NOERR || fIsNavy) {fGridType = TWO_D;}
	else
	{	
		if (fGridType==MULTILAYER)
		{
			// for now
			totalDepthsH = (FLOATH)_NewHandleClear(latLength*lonLength*sizeof(float));
			if (!totalDepthsH) {err = memFullErr; goto done;}
			depth_vals = new float[latLength*lonLength];
			if (!depth_vals) {err = memFullErr; goto done;}
			for (i=0;i<latLength*lonLength;i++)
			{
				INDEXH(totalDepthsH,i)=sigma_vals[sigmaLength-1];
				depth_vals[i] = INDEXH(totalDepthsH,i);
			}
		
		}
		else
		{
			totalDepthsH = (FLOATH)_NewHandleClear(latLength*lonLength*sizeof(float));
			if (!totalDepthsH) {err = memFullErr; goto done;}
			depth_vals = new float[latLength*lonLength];
			if (!depth_vals) {err = memFullErr; goto done;}
			status = nc_get_vara_float(ncid, depthid, ptIndex,pt_count, depth_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
		
			//status = nc_get_att_double(ncid, depthid, "scale_factor", &scale_factor);
			//if (status != NC_NOERR) {}	// don't require scale factor
		}
	}
		
	*numRows = latLength;
	*numCols = lonLength;
	fNumRows = latLength;
	fNumCols = lonLength;
	
	mask_count[0] = latLength;
	mask_count[1] = lonLength;
	
	status = nc_inq_varid(ncid, "mask", &mask_id);
	if (status != NC_NOERR)	{isLandMask = false; err=-1; goto done; }
	if (isLandMask)
	{
		landmask = new double[latLength*lonLength]; 
		if(!landmask) {TechError("Map3D::GetPointsAndMask()", "new[]", 0); err = memFullErr; goto done;}
		//mylandmask = new double[latlength*lonlength]; 
		//if(!mylandmask) {TechError("NetCDFMoverCurv::ReoderPointsNoMask()", "new[]", 0); err = memFullErr; goto done;}
		mylandmaskH = (double**)_NewHandleClear(latLength*lonLength*sizeof(double));
		if(!mylandmaskH) {TechError("Map3D::GetPointsAndMask()", "_NewHandleClear()", 0); err = memFullErr; goto done;}
		status = nc_get_vara_double(ncid, mask_id, mask_index, mask_count, landmask);
		if (status != NC_NOERR) {err = -1; goto done;}
	}
		
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	for (i=0;i<latLength;i++)
	{
		for (j=0;j<lonLength;j++)
		{
			INDEXH(mylandmaskH,i*lonLength+j) = landmask[(latLength-i-1)*lonLength+j];
		}
	}
	*maskH = mylandmaskH;
	//err = this -> SetInterval(errmsg);
	//if(err) goto done;
	
	// look for topology in the file
	// for now ask for an ascii file, output from Topology save option
	// need dialog to ask for file
	//{if (topFilePath[0]) {strcpy(fTopFilePath,topFilePath); err = ReadTopology(fTopFilePath,newMap); goto done;}}
	//{if (topFilePath[0]) {err = ReadTopology(topFilePath,newMap); goto depths;}}
	//if (isLandMask/*fIsNavy*//*true*/)	// allow for the LAS files too ?
	/*if (true)	// allow for the LAS files too ?
	{
		short buttonSelected;
		buttonSelected  = MULTICHOICEALERT(1688,"Do you have an extended topology file to load?",FALSE);
		switch(buttonSelected){
			case 1: // there is an extended top file
				bTopFile = true;
				break;  
			case 3: // no extended top file
				bTopFile = false;
				break;
			case 4: // cancel
				err=-1;// stay at this dialog
				goto done;
		}
	}*/
	/*if(bTopFile)
	{
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
					 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		//if (!reply.good) return USERCANCEL;
		if (!reply.good) 
		{
			if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;
			err = dynamic_cast<NetCDFMoverCurv *>(this)->ReorderPoints(velocityH,newMap,errmsg);	
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto done;
		}
		else
			strcpy(topPath, reply.fullPath);
		
#else
		where = CenteredDialogUpLeft(M38c);
		sfpgetfile(&where, "",
				   (FileFilterUPP)0,
				   -1, typeList,
				   (DlgHookUPP)0,
				   &reply, M38c,
				   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		if (!reply.good) 
		{
			//numTimesInFile = this -> GetNumTimesInFile();	// use recs?
			//if (numTimesInFile>0)
			if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;
			err = dynamic_cast<NetCDFMoverCurv *>(this)->ReorderPoints(velocityH,newMap,errmsg);	
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto done;
			//return 0;
		}
		
		my_p2cstr(reply.fName);
		
#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, topPath);
#else
		strcpy(topPath, reply.fName);
#endif
#endif		
		strcpy (s, topPath);
		err = ReadTopology(topPath,newMap);	// newMap here
		goto depths;
		//SplitPathFile (s, fileName);
	}*/
	
	//numTimesInFile = this -> GetNumTimesInFile();
	//if (numTimesInFile>0)
	//if (recs>0)
		//err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
	//else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
	if(err) goto done;
	//if (isLandMask) err = ReorderPoints(velocityH,newMap,errmsg);
	//else err = ReorderPointsNoMask(velocityH,newMap,errmsg);
	//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	
depths:
	if (err) goto done;
	// also translate to fDepthDataInfo and fDepthsH here, using sigma or zgrid info
	
	if (totalDepthsH)
	{
		/*//fDepthsH = (FLOATH)_NewHandle(sizeof(float)*fNumRows*fNumCols);
		//if(!fDepthsH){TechError("NetCDFMoverCurv::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
		fBathymetryH = (FLOATH)_NewHandle(sizeof(float)*fNumRows*fNumCols);
		if(!fBathymetryH){TechError("NetCDFMoverCurv::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
		for (i=0;i<latLength;i++)
		{
			for (j=0;j<lonLength;j++)
			{
				//if (lat_vals[(latLength-i-1)*lonLength+j]==fill_value)	// this would be an error
				//lat_vals[(latLength-i-1)*lonLength+j]=0.;
				//if (lon_vals[(latLength-i-1)*lonLength+j]==fill_value)
				//lon_vals[(latLength-i-1)*lonLength+j]=0.;
				INDEXH(totalDepthsH,i*lonLength+j) = depth_vals[(latLength-i-1)*lonLength+j] * scale_factor;	
				//INDEXH(fDepthsH,i*lonLength+j) = depth_vals[(latLength-i-1)*lonLength+j] * scale_factor;	
				INDEXH(fBathymetryH,i*lonLength+j) = depth_vals[(latLength-i-1)*lonLength+j] * scale_factor;	
			}
		}
		//((TTriGridVel3D*)fGrid)->SetDepths(totalDepthsH);*/
		for (i=0;i<latLength;i++)
		{
			for (j=0;j<lonLength;j++)
			{
				// grid ordering does matter for creating ptcurmap, assume increases fastest in x/lon, then in y/lat
				//INDEXH(totalDepthsH,i*lonLength+j) = depth_vals[(latLength-i-1)*lonLength+j];	
				INDEXH(totalDepthsH,i*lonLength+j) = fabs(depth_vals[(latLength-i-1)*lonLength+j]);	
			}
		}
		*depthPointsH = totalDepthsH;
	
	}
	
	//fNumDepthLevels = sigmaLength;
	if (sigmaLength>1)
	{
		/*float sigma = 0;
		fDepthLevelsHdl = (FLOATH)_NewHandleClear(sigmaLength * sizeof(float));
		if (!fDepthLevelsHdl) {err = memFullErr; goto done;}
		for (i=0;i<sigmaLength;i++)
		{	// decide what to do here, may be upside down for ROMS
			sigma = sigma_vals[i];
			if (sigma_vals[0]==1) 
				INDEXH(fDepthLevelsHdl,i) = (1-sigma);	// in this case velocities will be upside down too...
			else
			{
				if (fVar.gridType == SIGMA_ROMS)
					INDEXH(fDepthLevelsHdl,i) = sigma;
				else
					INDEXH(fDepthLevelsHdl,i) = abs(sigma);
			}
			
		}
		if (fVar.gridType == SIGMA_ROMS)
		{
			fDepthLevelsHdl2 = (FLOATH)_NewHandleClear(sigmaLength * sizeof(float));
			if (!fDepthLevelsHdl2) {err = memFullErr; goto done;}
			for (i=0;i<sigmaLength;i++)
			{
				sigma = sigma_vals2[i];
				//if (sigma_vals[0]==1) 
				//INDEXH(fDepthLevelsHdl,i) = (1-sigma);	// in this case velocities will be upside down too...
				//else
				INDEXH(fDepthLevelsHdl2,i) = sigma;
			}
			hc = hc_param;
		}*/
	}
	
	/*if (totalDepthsH)
	{	// may need to extend the depth grid along with lat/lon grid - not sure what to use for the values though...
		// not sure what map will expect in terms of depths order
		long n,ptIndex,iIndex,jIndex;
		long numPoints = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(**fVerdatToNetCDFH);
		//_SetHandleSize((Handle)totalDepthsH,(fNumRows+1)*(fNumCols+1)*sizeof(float));
		_SetHandleSize((Handle)totalDepthsH,numPoints*sizeof(float));
		//for (i=0; i<fNumRows*fNumCols; i++)
		//for (i=0; i<(fNumRows+1)*(fNumCols+1); i++)
		
		for (i=0; i<numPoints; i++)
		{	// works okay for simple grid except for far right column (need to extend depths similar to lat/lon)
			// if land use zero, if water use point next to it?
			ptIndex = INDEXH(fVerdatToNetCDFH,i);
			iIndex = ptIndex/(fNumCols+1);
			jIndex = ptIndex%(fNumCols+1);
			if (iIndex>0 && jIndex<fNumCols)
				ptIndex = (iIndex-1)*(fNumCols)+jIndex;
			else
				ptIndex = -1;
			
			//n = INDEXH(fVerdatToNetCDFH,i);
			//if (n<0 || n>= fNumRows*fNumCols) {printError("indices messed up"); err=-1; goto done;}
			//INDEXH(totalDepthsH,i) = depth_vals[n];
			if (ptIndex<0 || ptIndex>= fNumRows*fNumCols) 
			{
				//printError("indices messed up"); 
				//err=-1; goto done;
				INDEXH(totalDepthsH,i) = 0;	// need to figure out what to do here...
				continue;
			}
			//INDEXH(totalDepthsH,i) = depth_vals[ptIndex];
			INDEXH(totalDepthsH,i) = INDEXH(fDepthsH,ptIndex);
		}
		((TTriGridVel*)fGrid)->SetDepths(totalDepthsH);
	}*/
	
done:
	if (err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"Error opening NetCDF file");
		printNote(errmsg);
		//printNote("Error opening NetCDF file");
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;}
		if(sigmaLevelsH) {DisposeHandle((Handle)sigmaLevelsH); sigmaLevelsH = 0;}
		//if (fDepthLevelsHdl) {DisposeHandle((Handle)fDepthLevelsHdl); fDepthLevelsHdl=0;}
		//if (fDepthLevelsHdl2) {DisposeHandle((Handle)fDepthLevelsHdl2); fDepthLevelsHdl2=0;}
	}
	
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (depth_vals) delete [] depth_vals;
	if (sigma_vals) delete [] sigma_vals;
	if (modelTypeStr) delete [] modelTypeStr;
	//if (velocityH) {DisposeHandle((Handle)velocityH); velocityH = 0;}
	return err;
}

OSErr Map3D::GetPointsAndBoundary(char *path,WORLDPOINTFH *vertexPointsH, FLOATH *depthPtsH, long *numNodes, LONGPTR *boundary_indices, LONGPTR *boundary_nums, LONGPTR *boundary_type, long *numBoundaryPts, LONGPTR *triangle_verts, LONGPTR *triangle_neighbors, long *ntri)
{
	// needs to be updated once triangle grid format is set
	
	OSErr err = 0;
	long i, numScanned;
	int status, ncid, nodeid, nbndid, bndid, neleid, latid, lonid, recid, timeid, sigmaid, sigmavarid, depthid, nv_varid, nbe_varid;
	int curr_ucmp_id, uv_dimid[3], uv_ndims;
	size_t nodeLength, nbndLength, neleLength, recs, t_len, sigmaLength=0;
	float timeVal;
	//char recname[NC_MAX_NAME], *timeUnits=0;	
	WORLDPOINTFH vertexPtsH=0;
	FLOATH totalDepthsH=0;
	//, sigmaLevelsH=0;
	float *lat_vals=0,*lon_vals=0,*depth_vals=0;
	//, *sigma_vals=0;
	long *bndry_indices=0, *bndry_nums=0, *bndry_type=0, *top_verts=0, *top_neighbors=0;
	static size_t latIndex=0,lonIndex=0,timeIndex,ptIndex=0,bndIndex[2]={0,0};
	static size_t pt_count, bnd_count[2], sigma_count,topIndex[2]={0,0}, top_count[2];
	//Seconds startTime, startTime2;
	double timeConversion = 1., scale_factor = 1.;
	char errmsg[256] = "";
	char fileName[64],s[256],topPath[256], outPath[256];
	
	char *modelTypeStr=0;
	Point where;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	Boolean bTopFile = false, bTopInfoInFile = false, bVelocitiesOnTriangles = false;
	
	if (!path || !path[0]) return 0;
	//strcpy(fVar.pathName,path);
	
	//strcpy(s,path);
	//SplitPathFile (s, fileName);
	//strcpy(fVar.userName, fileName); // maybe use a name from the file
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	
	status = nc_inq_dimid(ncid, "node", &nodeid); 
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimlen(ncid, nodeid, &nodeLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimid(ncid, "nbnd", &nbndid);	
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "bnd", &bndid);	
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimlen(ncid, nbndid, &nbndLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	bnd_count[0] = nbndLength;
	bnd_count[1] = 1;
	//bndry_indices = new short[nbndLength]; 
	bndry_indices = new long[nbndLength]; 
	//bndry_nums = new short[nbndLength]; 
	//bndry_type = new short[nbndLength]; 
	bndry_nums = new long[nbndLength]; 
	bndry_type = new long[nbndLength]; 
	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}
	//bndIndex[1] = 0;
	bndIndex[1] = 1;	// take second point of boundary segments instead, so that water boundaries work out
	//status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_indices);
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_indices);
	if (status != NC_NOERR) {err = -1; goto done;}
	bndIndex[1] = 2;
	//status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_nums);
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_nums);
	if (status != NC_NOERR) {err = -1; goto done;}
	bndIndex[1] = 3;
	//status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_type);
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_type);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// option to use index values?
	status = nc_inq_varid(ncid, "lat", &latid);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "lon", &lonid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	pt_count = nodeLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(nodeLength*sizeof(WorldPointF));
	if (!vertexPtsH) {err = memFullErr; goto done;}
	lat_vals = new float[nodeLength]; 
	lon_vals = new float[nodeLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	status = nc_get_vara_float(ncid, latid, &ptIndex, &pt_count, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, lonid, &ptIndex, &pt_count, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_inq_varid(ncid, "depth", &depthid);	// this is required for a map
	//if (status != NC_NOERR) {/*fVar.gridType = TWO_D;*/err = -1; goto done;}
	//if (status != NC_NOERR) {/*err = -1; goto done;*/}
	//else
	{	
		totalDepthsH = (FLOATH)_NewHandleClear(nodeLength*sizeof(float));
		if (!totalDepthsH) {err = memFullErr; goto done;}
		if (status != NC_NOERR)
		{
		}
		else
		{
		depth_vals = new float[nodeLength];
		if (!depth_vals) {err = memFullErr; goto done;}
		status = nc_get_vara_float(ncid, depthid, &ptIndex, &pt_count, depth_vals);
		if (status != NC_NOERR) {err = -1; goto done;}
		
		status = nc_get_att_double(ncid, depthid, "scale_factor", &scale_factor);
		if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require scale factor
		}
	}
	
	for (i=0;i<nodeLength;i++)
	{
		INDEXH(vertexPtsH,i).pLat = lat_vals[i];	
		INDEXH(vertexPtsH,i).pLong = lon_vals[i];
		if (depth_vals) INDEXH(totalDepthsH,i) = depth_vals[i];
		else INDEXH(totalDepthsH,i) = INFINITE_DEPTH;	// let map have infinite depth
	}
	*vertexPointsH	= vertexPtsH;// get first and last, lat/lon values, then last-first/total-1 = dlat/dlon
	*depthPtsH = totalDepthsH;
	*numBoundaryPts = nbndLength;
	*numNodes = nodeLength;
	//status = nc_inq_dim(ncid, recid, recname, &recs);
	//if (status != NC_NOERR) {err = -1; goto done;}
	
	
	// check if file has topology in it
	{
		status = nc_inq_varid(ncid, "nv", &nv_varid); //Navy
		if (status != NC_NOERR) {/*err = -1; goto done;*/}
		else
		{
			status = nc_inq_varid(ncid, "nbe", &nbe_varid); //Navy
			if (status != NC_NOERR) {/*err = -1; goto done;*/}
			else bTopInfoInFile = true;
		}
		if (bTopInfoInFile)
		{
			status = nc_inq_dimid(ncid, "nele", &neleid);	
			if (status != NC_NOERR) {err = -1; goto done;}	
			status = nc_inq_dimlen(ncid, neleid, &neleLength);
			if (status != NC_NOERR) {err = -1; goto done;}	
			//fNumEles = neleLength;
			top_verts = new long[neleLength*3]; 
			if (!top_verts ) {err = memFullErr; goto done;}
			top_neighbors = new long[neleLength*3]; 
			if (!top_neighbors ) {err = memFullErr; goto done;}
			top_count[0] = 3;
			top_count[1] = neleLength;
			status = nc_get_vara_long(ncid, nv_varid, topIndex, top_count, top_verts);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_get_vara_long(ncid, nbe_varid, topIndex, top_count, top_neighbors);
			if (status != NC_NOERR) {err = -1; goto done;}
			
			//determine if velocities are on triangles
			status = nc_inq_varid(ncid, "u", &curr_ucmp_id);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_varndims(ncid, curr_ucmp_id, &uv_ndims);
			if (status != NC_NOERR) {err = -1; goto done;}
			
			status = nc_inq_vardimid (ncid, curr_ucmp_id, uv_dimid);	// see if dimid(1) or (2) == nele or node, depends on uv_ndims
			if (status==NC_NOERR) 
			{
				if (uv_ndims == 3 && uv_dimid[2] == neleid)
				{bVelocitiesOnTriangles = true;}
				else if (uv_ndims == 2 && uv_dimid[1] == neleid)
				{bVelocitiesOnTriangles = true;}
			}
			
		}
	}
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	//err = this -> SetInterval(errmsg);
	//if(err) goto done;
	
	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}
	
	 *boundary_indices = bndry_indices;
	 *boundary_nums = bndry_nums;
	 *boundary_type = bndry_type;

	 if (bVelocitiesOnTriangles)
	 {
		if (!top_verts || !top_neighbors) {err = memFullErr; goto done;}
		*ntri = neleLength;
		*triangle_verts = top_verts;
		*triangle_neighbors = top_neighbors;
	 }
	//{if (topFilePath[0]) {err = ReadTopology(topFilePath,newMap); goto depths;}}
	// look for topology in the file
	// for now ask for an ascii file, output from Topology save option
	// need dialog to ask for file
	/*if (!bTopFile)
	{
		short buttonSelected;
		buttonSelected  = MULTICHOICEALERT(1688,"Do you have an extended topology file to load?",FALSE);
		switch(buttonSelected){
			case 1: // there is an extended top file
				bTopFile = true;
				break;  
			case 3: // no extended top file
				bTopFile = false;
				break;
			case 4: // cancel
				err=-1;// stay at this dialog
				goto done;
		}
	}
	if(bTopFile)
	{
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
					 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		//if (!reply.good) return USERCANCEL;
		if (!reply.good) 
		{
			if (bVelocitiesOnTriangles)
			{
				err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength,top_verts,top_neighbors,neleLength);	 
				//err = ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength,top_verts,top_neighbors,neleLength);	 
				if (err) goto done;
				goto depths;
			}
			else
			{
				err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
				//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
				if (err) goto done;
	 			goto depths;
			}
		}
		else
			strcpy(topPath, reply.fullPath);
		
#else
		where = CenteredDialogUpLeft(M38c);
		sfpgetfile(&where, "",
				   (FileFilterUPP)0,
				   -1, typeList,
				   (DlgHookUPP)0,
				   &reply, M38c,
				   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		if (!reply.good) 
		{
			if (bVelocitiesOnTriangles)
			{
				err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength,top_verts,top_neighbors,neleLength);	 
				//err = ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength,top_verts,top_neighbors,neleLength);		 
				if (err) goto done;
				goto depths;
			}
			else
			{
				err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
				//goto done;
				if (err) goto done;	
				goto depths;
			}	
			//return 0;
		}
		
		my_p2cstr(reply.fName);
		
#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, topPath);
#else
		strcpy(topPath, reply.fName);
#endif
#endif		
		strcpy (s, topPath);
		err = ReadTopology(topPath,newMap);	// newMap here
		if (err) goto done;
		goto depths;
		//goto done;
		//SplitPathFile (s, fileName);
	}*/
	
	/*if (bVelocitiesOnTriangles)
		err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength,top_verts,top_neighbors,neleLength);	 
	else
		err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
	*/
depths:
	if (err) goto done;
		
	/*if (totalDepthsH)	
	{
		for (i=0; i<nodeLength; i++)
		{
			long n;
			
			n = INDEXH(fVerdatToNetCDFH,i);
			if (n<0 || n>= fNumNodes) {printError("indices messed up"); err=-1; goto done;}
			INDEXH(totalDepthsH,i) = depth_vals[n] * scale_factor;
		}
		((TTriGridVel3D*)fGrid)->SetDepths(totalDepthsH);
	}*/
	
done:
	if (err)
	{
		if (!errmsg[0]) 
			strcpy(errmsg,"Error opening NetCDF file");
		printNote(errmsg);
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;}
	if (bndry_indices) delete [] bndry_indices;
	if (bndry_nums) delete [] bndry_nums;
	if (bndry_type) delete [] bndry_type;
	}
	
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (depth_vals) delete [] depth_vals;
	//if (bndry_indices) delete [] bndry_indices;
	//if (bndry_nums) delete [] bndry_nums;
	//if (bndry_type) delete [] bndry_type;
	
	return err;
}

OSErr Map3D_c::SetUpCurvilinearGrid(DOUBLEH landMaskH, long numRows, long numCols, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, char* errmsg)
{
	long i, j, n, ntri, numVerdatPts=0; 
	long numRows_ext = numRows+1, numCols_ext = numCols+1;
	long nv = numRows * numCols, nv_ext = numRows_ext*numCols_ext;
	long currentIsland=0, islandNum, nBoundaryPts=0, nEndPts=0, waterStartPoint;
	long nSegs, segNum = 0, numIslands, rectIndex; 
	long iIndex,jIndex,index,currentIndex,startIndex; 
	long triIndex1,triIndex2,waterCellNum=0;
	long ptIndex = 0,cellNum = 0,diag = 1;
	Boolean foundPt = false, isOdd;
	OSErr err = 0;
	
	LONGH landWaterInfo = (LONGH)_NewHandleClear(numRows * numCols * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));
	
	LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv_ext * sizeof(**ptIndexHdl));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**verdatPtsH));
	GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nv * sizeof(**gridCellInfo));
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	FLOATH depths=0;
	
	LONGH boundaryPtsH = 0;
	LONGH boundaryEndPtsH = 0;
	LONGH waterBoundaryPtsH = 0;
	Boolean** segUsed = 0;
	SegInfoHdl segList = 0;
	LONGH flagH = 0;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	
	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}
	
	for (i=0;i<numRows;i++)
	{
		for (j=0;j<numCols;j++)
		{
			if (INDEXH(landMaskH,i*numCols+j)==0)	// land point
			{
				INDEXH(landWaterInfo,i*numCols+j) = -1;	// may want to mark each separate island with a unique number
			}
			else
			{
				INDEXH(landWaterInfo,i*numCols+j) = 1;
				INDEXH(ptIndexHdl,i*numCols_ext+j) = -2;	// water box
				INDEXH(ptIndexHdl,i*numCols_ext+j+1) = -2;
				INDEXH(ptIndexHdl,(i+1)*numCols_ext+j) = -2;
				INDEXH(ptIndexHdl,(i+1)*numCols_ext+j+1) = -2;
			}
		}
	}
	
	for (i=0;i<numRows_ext;i++)
	{
		for (j=0;j<numCols_ext;j++)
		{
			if (INDEXH(ptIndexHdl,i*numCols_ext+j) == -2)
			{
				INDEXH(ptIndexHdl,i*numCols_ext+j) = ptIndex;	// count up grid points
				ptIndex++;
			}
			else
				INDEXH(ptIndexHdl,i*numCols_ext+j) = -1;
		}
	}
	
	for (i=0;i<numRows;i++)
	{
		for (j=0;j<numCols;j++)
		{
			if (INDEXH(landWaterInfo,i*numCols+j)>0)
			{
				INDEXH(gridCellInfo,i*numCols+j).cellNum = cellNum;
				cellNum++;
				INDEXH(gridCellInfo,i*numCols+j).topLeft = INDEXH(ptIndexHdl,i*numCols_ext+j);
				INDEXH(gridCellInfo,i*numCols+j).topRight = INDEXH(ptIndexHdl,i*numCols_ext+j+1);
				INDEXH(gridCellInfo,i*numCols+j).bottomLeft = INDEXH(ptIndexHdl,(i+1)*numCols_ext+j);
				INDEXH(gridCellInfo,i*numCols+j).bottomRight = INDEXH(ptIndexHdl,(i+1)*numCols_ext+j+1);
			}
			else INDEXH(gridCellInfo,i*numCols+j).cellNum = -1;
		}
	}
	ntri = cellNum*2;	// each water cell is split into two triangles
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology)))){err = memFullErr; goto done;}	
	for (i=0;i<nv_ext;i++)
	{
		if (INDEXH(ptIndexHdl,i) != -1)
		{
			INDEXH(verdatPtsH,numVerdatPts) = i;
			//INDEXH(verdatPtsH,INDEXH(ptIndexHdl,i)) = i;
			numVerdatPts++;
		}
	}
	_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(**verdatPtsH));
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	depths = (FLOATH)_NewHandle(sizeof(float)*(numVerdatPts));
	if(pts == nil || depths == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	
	for (i=0; i<=numVerdatPts; i++)	// make a list of grid points that will be used for triangles
	{
		float fLong, fLat, fDepth, dLon, dLat, dLon1, dLon2, dLat1, dLat2;
		double val, u=0., v=0.;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	// since velocities are defined at the lower left corner of each grid cell
			// need to add an extra row/col at the top/right of the grid
			// set lat/lon based on distance between previous two points 
			// these are just for boundary/drawing purposes, velocities are set to zero
			index = i+1;
			n = INDEXH(verdatPtsH,i);
			iIndex = n/numCols_ext;
			jIndex = n%numCols_ext;
			if (iIndex==0)
			{
				if (jIndex<numCols)
				{
					dLat = INDEXH(vertexPtsH,numCols+jIndex).pLat - INDEXH(vertexPtsH,jIndex).pLat;
					fLat = INDEXH(vertexPtsH,jIndex).pLat - dLat;
					dLon = INDEXH(vertexPtsH,numCols+jIndex).pLong - INDEXH(vertexPtsH,jIndex).pLong;
					fLong = INDEXH(vertexPtsH,jIndex).pLong - dLon;
					fDepth = INDEXH(depthPtsH,jIndex);
				}
				else
				{
					dLat1 = (INDEXH(vertexPtsH,jIndex-1).pLat - INDEXH(vertexPtsH,jIndex-2).pLat);
					dLat2 = INDEXH(vertexPtsH,numCols+jIndex-1).pLat - INDEXH(vertexPtsH,numCols+jIndex-2).pLat;
					fLat = 2*(INDEXH(vertexPtsH,jIndex-1).pLat + dLat1)-(INDEXH(vertexPtsH,numCols+jIndex-1).pLat+dLat2);
					dLon1 = INDEXH(vertexPtsH,numCols+jIndex-1).pLong - INDEXH(vertexPtsH,jIndex-1).pLong;
					dLon2 = (INDEXH(vertexPtsH,numCols+jIndex-2).pLong - INDEXH(vertexPtsH,jIndex-2).pLong);
					fLong = 2*(INDEXH(vertexPtsH,jIndex-1).pLong-dLon1)-(INDEXH(vertexPtsH,jIndex-2).pLong-dLon2);
					fDepth = INDEXH(depthPtsH,jIndex-1);
				}
			}
			else 
			{
				if (jIndex<numCols)
				{
					fLat = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex).pLat;
					fLong = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex).pLong;
					fDepth = INDEXH(depthPtsH,(iIndex-1)*numCols+jIndex);
				}
				else
				{
					dLat = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-1).pLat - INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-2).pLat;
					fLat = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-1).pLat + dLat;
					dLon = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-1).pLong - INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-2).pLong;
					fLong = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-1).pLong + dLon;
					fDepth = INDEXH(depthPtsH,(iIndex-1)*numCols+jIndex-1);
				}
			}
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			INDEXH(pts,i) = vertex;
			INDEXH(depths,i) = fDepth;
		}
		else { // for outputting a verdat the last line should be all zeros
			//index = 0;
			//fLong = fLat = fDepth = 0.0;
		}
		/////////////////////////////////////////////////
		
	}
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	
	/////////////////////////////////////////////////
	for (i=0;i<numRows;i++)
	{
		for (j=0;j<numCols;j++)
		{
			if (INDEXH(landWaterInfo,i*numCols+j)==-1)
				continue;
			waterCellNum = INDEXH(gridCellInfo,i*numCols+j).cellNum;	// split each cell into 2 triangles
			triIndex1 = 2*waterCellNum;
			triIndex2 = 2*waterCellNum+1;
			// top/left tri in rect
			(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*numCols+j).topRight;
			(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*numCols+j).topLeft;
			(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*numCols+j).bottomLeft;
			if (j==0 || INDEXH(gridCellInfo,i*numCols+j-1).cellNum == -1)
				(*topo)[triIndex1].adjTri1 = -1;
			else
			{
				(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*numCols+j-1).cellNum * 2 + 1;
			}
			(*topo)[triIndex1].adjTri2 = triIndex2;
			if (i==0 || INDEXH(gridCellInfo,(i-1)*numCols+j).cellNum==-1)
				(*topo)[triIndex1].adjTri3 = -1;
			else
			{
				(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*numCols+j).cellNum * 2 + 1;
			}
			// bottom/right tri in rect
			(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*numCols+j).bottomLeft;
			(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*numCols+j).bottomRight;
			(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*numCols+j).topRight;
			if (j==numCols-1 || INDEXH(gridCellInfo,i*numCols+j+1).cellNum == -1)
				(*topo)[triIndex2].adjTri1 = -1;
			else
			{
				(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*numCols+j+1).cellNum * 2;
			}
			(*topo)[triIndex2].adjTri2 = triIndex1;
			if (i==numRows-1 || INDEXH(gridCellInfo,(i+1)*numCols+j).cellNum == -1)
				(*topo)[triIndex2].adjTri3 = -1;
			else
			{
				(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*numCols+j).cellNum * 2;
			}
		}
	}

	
	/////////////////////////////////////////////////
	// go through topo look for -1, and list corresponding boundary sides
	// then reorder as contiguous boundary segments - need to group boundary rects by islands
	// will need a new field for list of boundary points since there can be duplicates, can't just order and list segment endpoints
	
	nSegs = 2*ntri; //number of -1's in topo
	boundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryPtsH));
	boundaryEndPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryEndPtsH));
	waterBoundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**waterBoundaryPtsH));
	flagH = (LONGH)_NewHandleClear(nv_ext * sizeof(**flagH));
	segUsed = (Boolean**)_NewHandleClear(nSegs * sizeof(Boolean));
	segList = (SegInfoHdl)_NewHandleClear(nSegs * sizeof(**segList));
	// first go through rectangles and group by island
	// do this before making dagtree, 
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Numbering Islands");
	MySpinCursor(); // JLM 8/4/99
	err = NumberIslands(&maskH2, landMaskH, landWaterInfo, numRows, numCols, &numIslands);	// numbers start at 3 (outer boundary)
	MySpinCursor(); // JLM 8/4/99
	if (err) goto done;
	for (i=0;i<ntri;i++)
	{
		if ((i+1)%2==0) isOdd = 0; else isOdd = 1;
		// the middle neighbor triangle is always the other half of the rectangle so can't be land or outside the map
		// odd - left/top, even - bottom/right the 1-2 segment is top/bot, the 2-3 segment is right/left
		if ((*topo)[i].adjTri1 == -1)
		{
			// add segment pt 2 - pt 3 to list, need points, triNum and whether it's L/W boundary (boundary num)
			(*segList)[segNum].pt1 = (*topo)[i].vertex2;
			(*segList)[segNum].pt2 = (*topo)[i].vertex3;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check left rectangle for L/W border 
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/numCols_ext;
				jIndex = rectIndex%numCols_ext;
				if (jIndex>0 && INDEXH(maskH2,iIndex*numCols_ext + jIndex-1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*numCols_ext + jIndex-1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			else 
			{	
				// check right rectangle for L/W border convert back to row/col
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/numCols_ext;
				jIndex = rectIndex%numCols_ext;
				if (jIndex<numCols && INDEXH(maskH2,iIndex*numCols_ext + jIndex+1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*numCols_ext + jIndex+1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			segNum++;
		}
		
		if ((*topo)[i].adjTri3 == -1)
		{
			// add segment pt 1 - pt 2 to list
			// odd top, even bottom
			(*segList)[segNum].pt1 = (*topo)[i].vertex1;
			(*segList)[segNum].pt2 = (*topo)[i].vertex2;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check top rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/numCols_ext;
				jIndex = rectIndex%numCols_ext;
				if (iIndex>0 && INDEXH(maskH2,(iIndex-1)*numCols_ext + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex-1)*numCols_ext + jIndex);
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;
				}
			}
			else 
			{
				// check bottom rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/numCols_ext;
				jIndex = rectIndex%numCols_ext;
				if (iIndex<numRows && INDEXH(maskH2,(iIndex+1)*numCols_ext + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex+1)*numCols_ext + jIndex);		// this should be the neighbor's value
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;		
				}
			}
			segNum++;
		}
	}
	nSegs = segNum;
	_SetHandleSize((Handle)segList,nSegs*sizeof(**segList));
	_SetHandleSize((Handle)segUsed,nSegs*sizeof(**segUsed));
	// go through list of segments, and make list of boundary segments
	// as segment is taken mark so only use each once
	// get a starting point, add the first and second to the list
	islandNum = 3;
findnewstartpoint:
	if (islandNum > numIslands) 
	{
		//err = -1; goto done;
		_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
		_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
		_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
		goto setFields;	// off by 2 - 0,1,2 are water cells, 3 and up are land
	}
	foundPt = false;
	for (i=0;i<nSegs;i++)
	{
		if ((*segUsed)[i]) continue;
		waterStartPoint = nBoundaryPts;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt1;
		(*flagH)[(*segList)[i].pt1] = 1;
		(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt2;
		(*flagH)[(*segList)[i].pt2] = 1;
		currentIndex = (*segList)[i].pt2;
		startIndex = (*segList)[i].pt1;
		currentIsland = (*segList)[i].islandNumber;	
		foundPt = true;
		(*segUsed)[i] = true;
		break;
	}
	if (!foundPt)
	{
		printNote("Lost trying to set boundaries");
		err = -1; goto done;
		// clean up handles and set grid without a map
		/*if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
		goto setFields;*/
	}
	
findnextpoint:
	for (i=0;i<nSegs;i++)
	{
		// look for second point of the previous selected segment, add the second to point list
		if ((*segUsed)[i]) continue;
		if ((*segList)[i].islandNumber > 3 && (*segList)[i].islandNumber != currentIsland) continue;
		if ((*segList)[i].islandNumber > 3 && currentIsland <= 3) continue;
		index = (*segList)[i].pt1;
		if (index == currentIndex)	// found next point
		{
			currentIndex = (*segList)[i].pt2;
			(*segUsed)[i] = true;
			if (currentIndex == startIndex) // completed a segment
			{
				islandNum++;
				(*boundaryEndPtsH)[nEndPts++] = nBoundaryPts-1;
				(*waterBoundaryPtsH)[waterStartPoint] = (*segList)[i].isWater+1;	// need to deal with this
				goto findnewstartpoint;
			}
			else
			{
				(*boundaryPtsH)[nBoundaryPts] = (*segList)[i].pt2;
				(*flagH)[(*segList)[i].pt2] = 1;
				(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
				nBoundaryPts++;
				goto findnextpoint;
			}
		}
	}
	// shouldn't get here unless there's a problem...
	_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
	_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
	_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
	
setFields:	
	
	/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in Map3D::SetUpCurvilinearGrid()","new TTriGridVel",err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(triBounds); 
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
	{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement
	triGrid -> SetDepths(depths);	// used by PtCurMap to check vertical movement
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	depths = 0;
	
	if (waterBoundaryPtsH)	// maybe assume rectangle grids will have map?
	{
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(boundaryEndPtsH);	
		this->SetWaterBoundaries(waterBoundaryPtsH);
		this->SetBoundaryPoints(boundaryPtsH);
		this->SetMapBounds(triBounds);		
	}
	else
	{
		err = -1;
		goto done;
	}
	
	/////////////////////////////////////////////////
done:
	if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
	if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
	if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
	if (segUsed) {DisposeHandle((Handle)segUsed); segUsed = 0;}
	if (segList) {DisposeHandle((Handle)segList); segList = 0;}
	if (flagH) {DisposeHandle((Handle)flagH); flagH = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in Map3D_c::SetUpCurvilinearGrid");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
		
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
	}
	return err;
}

OSErr Map3D_c::SetUpCurvilinearGrid2(DOUBLEH landMaskH, long numRows, long numCols, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, char* errmsg)
{	// this is for the points on nodes case (old coops_mask)
	OSErr err = 0;
	long i,j,k;
	char *velUnits=0; 
	long latlength = numRows, numtri = 0;
	long lonlength = numCols;
	Boolean isLandMask = true;
	float fDepth1, fLat1, fLong1;
	long index1=0;
	
	errmsg[0]=0;
	
	long n, ntri, numVerdatPts=0;
	long numRows_minus1 = numRows-1, numCols_minus1 = numCols-1;
	long nv = numRows * numCols;
	long nCells = numRows_minus1 * numCols_minus1;
	long iIndex, jIndex, index; 
	long triIndex1, triIndex2, waterCellNum=0;
	long ptIndex = 0, cellNum = 0;
	
	long currentIsland=0, islandNum, nBoundaryPts=0, nEndPts=0, waterStartPoint;
	long nSegs, segNum = 0, numIslands, rectIndex; 
	long currentIndex,startIndex; 
	long diag = 1;
	Boolean foundPt = false, isOdd;
	
	LONGH landWaterInfo = (LONGH)_NewHandleClear(nCells * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv * sizeof(long));
	
	LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv * sizeof(**ptIndexHdl));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatPtsH));
	GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nCells * sizeof(**gridCellInfo));
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	FLOATH depths=0;
	
	LONGH boundaryPtsH = 0;
	LONGH boundaryEndPtsH = 0;
	LONGH waterBoundaryPtsH = 0;
	Boolean** segUsed = 0;
	SegInfoHdl segList = 0;
	LONGH flagH = 0;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	/////////////////////////////////////////////////
	
	if (!landMaskH) return -1;
	
	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}
	
	index1 = 0;
	for (i=0;i<numRows-1;i++)
	{
		for (j=0;j<numCols-1;j++)
		{
			if (INDEXH(landMaskH,i*numCols+j)==0)	// land point
			{
				INDEXH(landWaterInfo,i*numCols_minus1+j) = -1;	// may want to mark each separate island with a unique number
			}
			else
			{
				if (INDEXH(landMaskH,(i+1)*numCols+j)==0 || INDEXH(landMaskH,i*numCols+j+1)==0 || INDEXH(landMaskH,(i+1)*numCols+j+1)==0)
				{
					INDEXH(landWaterInfo,i*numCols_minus1+j) = -1;	// may want to mark each separate island with a unique number
				}
				else
				{
					INDEXH(landWaterInfo,i*numCols_minus1+j) = 1;
					INDEXH(ptIndexHdl,i*numCols+j) = -2;	// water box
					INDEXH(ptIndexHdl,i*numCols+j+1) = -2;
					INDEXH(ptIndexHdl,(i+1)*numCols+j) = -2;
					INDEXH(ptIndexHdl,(i+1)*numCols+j+1) = -2;
				}
			}
		}
	}
	
	for (i=0;i<numRows;i++)
	{
		for (j=0;j<numCols;j++)
		{
			if (INDEXH(ptIndexHdl,i*numCols+j) == -2)
			{
				INDEXH(ptIndexHdl,i*numCols+j) = ptIndex;	// count up grid points
				ptIndex++;
			}
			else
				INDEXH(ptIndexHdl,i*numCols+j) = -1;
		}
	}
	
	for (i=0;i<numRows-1;i++)
	{
		for (j=0;j<numCols-1;j++)
		{
			if (INDEXH(landWaterInfo,i*numCols_minus1+j)>0)
			{
				INDEXH(gridCellInfo,i*numCols_minus1+j).cellNum = cellNum;
				cellNum++;
				INDEXH(gridCellInfo,i*numCols_minus1+j).topLeft = INDEXH(ptIndexHdl,i*numCols+j);
				INDEXH(gridCellInfo,i*numCols_minus1+j).topRight = INDEXH(ptIndexHdl,i*numCols+j+1);
				INDEXH(gridCellInfo,i*numCols_minus1+j).bottomLeft = INDEXH(ptIndexHdl,(i+1)*numCols+j);
				INDEXH(gridCellInfo,i*numCols_minus1+j).bottomRight = INDEXH(ptIndexHdl,(i+1)*numCols+j+1);
			}
			else INDEXH(gridCellInfo,i*numCols_minus1+j).cellNum = -1;
		}
	}
	ntri = cellNum*2;	// each water cell is split into two triangles
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology)))){err = memFullErr; goto done;}	
	for (i=0;i<nv;i++)
	{
		if (INDEXH(ptIndexHdl,i) != -1)
		{
			INDEXH(verdatPtsH,numVerdatPts) = i;
			numVerdatPts++;
		}
	}
	_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(**verdatPtsH));
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	if(pts == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	/////////////////////////////////////////////////
	//index = 0;
	for (i=0; i<=numVerdatPts; i++)	// make a list of grid points that will be used for triangles
	{
		float fLong, fLat, fDepth, dLon, dLat, dLon1, dLon2, dLat1, dLat2;
		double val, u=0., v=0.;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	// since velocities are defined at the lower left corner of each grid cell
			// need to add an extra row/col at the top/right of the grid
			// set lat/lon based on distance between previous two points 
			// these are just for boundary/drawing purposes, velocities are set to zero
			index = i+1;
			n = INDEXH(verdatPtsH,i);
			iIndex = n/numCols;
			jIndex = n%numCols;
			//fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
			//fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
			fLat = INDEXH(vertexPtsH,(iIndex)*numCols+jIndex).pLat;
			fLong = INDEXH(vertexPtsH,(iIndex)*numCols+jIndex).pLong;
			fDepth = INDEXH(depthPtsH,(iIndex)*numCols+jIndex);
			
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			//fDepth = 1.;
			INDEXH(pts,i) = vertex;
		}
		else { // for outputting a verdat the last line should be all zeros
			//index = 0;
			//fLong = fLat = fDepth = 0.0;
		}
		/////////////////////////////////////////////////
		
	}
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	
	/////////////////////////////////////////////////
	for (i=0;i<numRows_minus1;i++)
	{
		for (j=0;j<numCols_minus1;j++)
		{
			if (INDEXH(landWaterInfo,i*numCols_minus1+j)==-1)
				continue;
			waterCellNum = INDEXH(gridCellInfo,i*numCols_minus1+j).cellNum;	// split each cell into 2 triangles
			triIndex1 = 2*waterCellNum;
			triIndex2 = 2*waterCellNum+1;
			// top/left tri in rect
			(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*numCols_minus1+j).topRight;
			(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*numCols_minus1+j).topLeft;
			(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*numCols_minus1+j).bottomLeft;
			if (j==0 || INDEXH(gridCellInfo,i*numCols_minus1+j-1).cellNum == -1)
				(*topo)[triIndex1].adjTri1 = -1;
			else
			{
				(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*numCols_minus1+j-1).cellNum * 2 + 1;
			}
			(*topo)[triIndex1].adjTri2 = triIndex2;
			if (i==0 || INDEXH(gridCellInfo,(i-1)*numCols_minus1+j).cellNum==-1)
				(*topo)[triIndex1].adjTri3 = -1;
			else
			{
				(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*numCols_minus1+j).cellNum * 2 + 1;
			}
			// bottom/right tri in rect
			(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*numCols_minus1+j).bottomLeft;
			(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*numCols_minus1+j).bottomRight;
			(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*numCols_minus1+j).topRight;
			if (j==numCols-2 || INDEXH(gridCellInfo,i*numCols_minus1+j+1).cellNum == -1)
				(*topo)[triIndex2].adjTri1 = -1;
			else
			{
				(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*numCols_minus1+j+1).cellNum * 2;
			}
			(*topo)[triIndex2].adjTri2 = triIndex1;
			if (i==numRows-2 || INDEXH(gridCellInfo,(i+1)*numCols_minus1+j).cellNum == -1)
				(*topo)[triIndex2].adjTri3 = -1;
			else
			{
				(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*numCols_minus1+j).cellNum * 2;
			}
		}
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
	{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
	/////////////////////////////////////////////////
	
	/////////////////////////////////////////////////
	// go through topo look for -1, and list corresponding boundary sides
	// then reorder as contiguous boundary segments - need to group boundary rects by islands
	// will need a new field for list of boundary points since there can be duplicates, can't just order and list segment endpoints
	
	nSegs = 2*ntri; //number of -1's in topo
	 boundaryPtsH = (LONGH)_NewHandleClear(nv * sizeof(**boundaryPtsH));
	 boundaryEndPtsH = (LONGH)_NewHandleClear(nv * sizeof(**boundaryEndPtsH));
	 waterBoundaryPtsH = (LONGH)_NewHandleClear(nv * sizeof(**waterBoundaryPtsH));
	 flagH = (LONGH)_NewHandleClear(nv * sizeof(**flagH));
	 segUsed = (Boolean**)_NewHandleClear(nSegs * sizeof(Boolean));
	 segList = (SegInfoHdl)_NewHandleClear(nSegs * sizeof(**segList));
	 // first go through rectangles and group by island
	 // do this before making dagtree, 
	 DisplayMessage("NEXTMESSAGETEMP");
	 DisplayMessage("Numbering Islands");
	 MySpinCursor(); // JLM 8/4/99
	 //err = NumberIslands(&maskH2, velocityH, landWaterInfo, fNumRows_minus1, fNumCols_minus1, &numIslands);	// numbers start at 3 (outer boundary)
	 err = NumberIslands(&maskH2, landMaskH, landWaterInfo, numRows_minus1, numCols_minus1, &numIslands);	// numbers start at 3 (outer boundary)
	 //numIslands++;	// this is a special case for CBOFS, right now the only coops_mask example
	 MySpinCursor(); // JLM 8/4/99
	 if (err) goto done;
	 for (i=0;i<ntri;i++)
	 {
	 if ((i+1)%2==0) isOdd = 0; else isOdd = 1;
	 // the middle neighbor triangle is always the other half of the rectangle so can't be land or outside the map
	 // odd - left/top, even - bottom/right the 1-2 segment is top/bot, the 2-3 segment is right/left
	 if ((*topo)[i].adjTri1 == -1)
	 {
	 // add segment pt 2 - pt 3 to list, need points, triNum and whether it's L/W boundary (boundary num)
	 (*segList)[segNum].pt1 = (*topo)[i].vertex2;
	 (*segList)[segNum].pt2 = (*topo)[i].vertex3;
	 // check which land block this segment borders and mark the island
	 if (isOdd) 
	 {
	 // check left rectangle for L/W border 
	 rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
	 iIndex = rectIndex/numCols;
	 jIndex = rectIndex%numCols;
	 if (jIndex>0 && INDEXH(maskH2,iIndex*numCols + jIndex-1)>=3)
	 {
	 (*segList)[segNum].isWater = 0;
	 (*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*numCols + jIndex-1);	
	 }
	 else
	 {
	 (*segList)[segNum].isWater = 1;
	 (*segList)[segNum].islandNumber = 1;	
	 }
	 }
	 else 
	 {	
	 // check right rectangle for L/W border convert back to row/col
	 rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
	 iIndex = rectIndex/numCols;
	 jIndex = rectIndex%numCols;
	 //if (jIndex<fNumCols && INDEXH(maskH2,iIndex*fNumCols + jIndex+1)>=3)
	 if (jIndex<numCols_minus1 && INDEXH(maskH2,iIndex*numCols + jIndex+1)>=3)
	 {
	 (*segList)[segNum].isWater = 0;
	 //(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*fNumCols + jIndex+1);	
	 (*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*numCols + jIndex+1);	
	 }
	 else
	 {
	 (*segList)[segNum].isWater = 1;
	 (*segList)[segNum].islandNumber = 1;	
	 }
	 }
	 segNum++;
	 }
	 
	 if ((*topo)[i].adjTri3 == -1)
	 {
		 // add segment pt 1 - pt 2 to list
		 // odd top, even bottom
		 (*segList)[segNum].pt1 = (*topo)[i].vertex1;
		 (*segList)[segNum].pt2 = (*topo)[i].vertex2;
		 // check which land block this segment borders and mark the island
		 if (isOdd) 
		 {
			 // check top rectangle for L/W border
			 rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
			 iIndex = rectIndex/numCols;
			 jIndex = rectIndex%numCols;
			 if (iIndex>0 && INDEXH(maskH2,(iIndex-1)*numCols + jIndex)>=3)
			 {
				 (*segList)[segNum].isWater = 0;
				 (*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex-1)*numCols + jIndex);
			 }
			 else
			 {
				 (*segList)[segNum].isWater = 1;
				 (*segList)[segNum].islandNumber = 1;
			 }
		 }
		 else 
		 {
			 // check bottom rectangle for L/W border
			 rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
			 iIndex = rectIndex/numCols;
			 jIndex = rectIndex%numCols;
			 //if (iIndex<fNumRows && INDEXH(maskH2,(iIndex+1)*fNumCols + jIndex)>=3)
			 if (iIndex<numRows_minus1 && INDEXH(maskH2,(iIndex+1)*numCols + jIndex)>=3)
			 {
				 (*segList)[segNum].isWater = 0;
				 //(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex+1)*fNumCols + jIndex);		// this should be the neighbor's value
				 (*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex+1)*numCols + jIndex);		// this should be the neighbor's value
			 }
			 else
			 {
				 (*segList)[segNum].isWater = 1;
				 (*segList)[segNum].islandNumber = 1;		
			 }
		 }
		 segNum++;
		 }
	 }
	 nSegs = segNum;
	 _SetHandleSize((Handle)segList,nSegs*sizeof(**segList));
	 _SetHandleSize((Handle)segUsed,nSegs*sizeof(**segUsed));
	 // go through list of segments, and make list of boundary segments
	 // as segment is taken mark so only use each once
	 // get a starting point, add the first and second to the list
	 islandNum = 3;
	 findnewstartpoint:
	 if (islandNum > numIslands) 
	 {
		 //err = -1; goto done
		 _SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
		 _SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
		 _SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
		 goto setFields;	// off by 2 - 0,1,2 are water cells, 3 and up are land
	 }
	 foundPt = false;
	 for (i=0;i<nSegs;i++)
	 {
		 if ((*segUsed)[i]) continue;
		 waterStartPoint = nBoundaryPts;
		 (*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt1;
		 (*flagH)[(*segList)[i].pt1] = 1;
		 (*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
		 (*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt2;
		 (*flagH)[(*segList)[i].pt2] = 1;
		 currentIndex = (*segList)[i].pt2;
		 startIndex = (*segList)[i].pt1;
		 currentIsland = (*segList)[i].islandNumber;	
		 foundPt = true;
		 (*segUsed)[i] = true;
		 break;
	 }
	 if (!foundPt)
	 {
		 printNote("Lost trying to set boundaries");
		 err = -1; goto done;
		 // clean up handles and set grid without a map
		 //if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		 //if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		 //if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
		 //goto setFields;
	 }
	 
	 findnextpoint:
	 for (i=0;i<nSegs;i++)
	 {
		 // look for second point of the previous selected segment, add the second to point list
		 if ((*segUsed)[i]) continue;
		 if ((*segList)[i].islandNumber > 3 && (*segList)[i].islandNumber != currentIsland) continue;
		 if ((*segList)[i].islandNumber > 3 && currentIsland <= 3) continue;
		 index = (*segList)[i].pt1;
		 if (index == currentIndex)	// found next point
		 {
			 currentIndex = (*segList)[i].pt2;
			 (*segUsed)[i] = true;
			 if (currentIndex == startIndex) // completed a segment
			 {
				 islandNum++;
				 (*boundaryEndPtsH)[nEndPts++] = nBoundaryPts-1;
				 (*waterBoundaryPtsH)[waterStartPoint] = (*segList)[i].isWater+1;	// need to deal with this
				 goto findnewstartpoint;
			 }
			 else
			 {
				 (*boundaryPtsH)[nBoundaryPts] = (*segList)[i].pt2;
				 (*flagH)[(*segList)[i].pt2] = 1;
				 (*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
				 nBoundaryPts++;
				 goto findnextpoint;
			}
		 }
	 }
	 // shouldn't get here unless there's a problem...
	 _SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
	 _SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
	 _SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
	 	
setFields:	
	
	/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TimeGridVelCurv_c::ReorderPointsCOOPSMask()","new TTriGridVel",err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(triBounds); 

	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	
	if (waterBoundaryPtsH)	// maybe assume rectangle grids will have map?
	{
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(boundaryEndPtsH);	
		this->SetWaterBoundaries(waterBoundaryPtsH);
		this->SetBoundaryPoints(boundaryPtsH);
		this->SetMapBounds(triBounds);		
	}
	else
	{
		err = -1;
		goto done;
	}
	
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	
	/////////////////////////////////////////////////
done:
	if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
	if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
	if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
	if (segUsed) {DisposeHandle((Handle)segUsed); segUsed = 0;}
	if (segList) {DisposeHandle((Handle)segList); segList = 0;}
	if (flagH) {DisposeHandle((Handle)flagH); flagH = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in Map3D_c::SetUpCurvilinearGrid2");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
		
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
	}
		
	return err;	
}

OSErr Map3D_c::SetUpTriangleGrid2(long numNodes, long ntri, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts, long *tri_verts, long *tri_neighbors) 
{
	OSErr err = 0;
	char errmsg[256];
	long i, n, nv = numNodes;
	long currentBoundary;
	long numVerdatPts = 0, numVerdatBreakPts = 0;
	
	LONGH vertFlagsH = (LONGH)_NewHandleClear(nv * sizeof(**vertFlagsH));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatPtsH));
	LONGH verdatBreakPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatBreakPtsH));
	
	TopologyHdl topo=0;
	DAGTreeStruct tree;
	
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	FLOATH depths = 0;
	WorldRect triBounds;
	LONGH waterBoundariesH=0;
	LONGH boundaryPtsH = 0;
	
	TTriGridVel *triGrid = nil;
	
	Boolean addOne = false;	// for debugging
	
	if (!vertFlagsH || !verdatPtsH || !verdatBreakPtsH) {err = memFullErr; goto done;}
	
	// put boundary points into verdat list
	
	// code goes here, double check that the water boundary info is also reordered
	currentBoundary=1;
	if (bndry_nums[0]==0) addOne = true;	// for debugging
	for (i = 0; i < numBoundaryPts; i++)
	{	
		//short islandNum, index;
		long islandNum, index;
		index = bndry_indices[i];
		islandNum = bndry_nums[i];
		if (addOne) islandNum++;	// for debugging
		INDEXH(vertFlagsH,index-1) = 1;	// note that point has been used
		INDEXH(verdatPtsH,numVerdatPts++) = index-1;	// add to verdat list
		if (islandNum>currentBoundary)
		{
			// for verdat file indices are really point numbers, subtract one for actual index
			INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = i;	// passed a break point
			currentBoundary++;
		}
		//INDEXH(boundaryPtsH,i) = bndry_indices[i]-1;
	}
	INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = numBoundaryPts;
	
	// add the rest of the points to the verdat list (these points are the interior points)
	for(i = 0; i < nv; i++) {
		if(INDEXH(vertFlagsH,i) == 0)	
		{
			INDEXH(verdatPtsH,numVerdatPts++) = i;
			INDEXH(vertFlagsH,i) = 0; // mark as used
		}
	}
	if (numVerdatPts!=nv) 
	{
		printNote("Not all vertex points were used");
		// it seems this should be an error...
		err = -1;
		goto done;
		// shrink handle
		//_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(long));
	}
	
	numVerdatPts = nv;	//for now, may reorder later
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	depths = (FLOATH)_NewHandle(sizeof(float)*(numVerdatPts));
	if(pts == nil || depths == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	//numVerdatPts = nv;	//for now, may reorder later
	for (i=0; i<=numVerdatPts; i++)
	{
		//long index;
		float fLong, fLat, fDepth;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	
			//index = i+1;
			//n = INDEXH(verdatPtsH,i);
			n = i;	// for now, not sure if need to reorder
			fLat = INDEXH(vertexPtsH,n).pLat;	// don't need to store fVertexPtsH, just pass in and use here
			fLong = INDEXH(vertexPtsH,n).pLong;
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = INDEXH(depthPtsH,n);	// this will be set from bathymetry, just a fudge here for outputting a verdat
			INDEXH(pts,i) = vertex;
			INDEXH(depths,i) = fDepth;
		}
		else { // the last line should be all zeros
			//index = 0;
			//fLong = fLat = fDepth = 0.0;
		}
		/////////////////////////////////////////////////
	}
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	// shrink handle
	_SetHandleSize((Handle)verdatBreakPtsH,numVerdatBreakPts*sizeof(long));
	for(i = 0; i < numVerdatBreakPts; i++ )
	{
		INDEXH(verdatBreakPtsH,i)--;
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	// use new maketriangles to force algorithm to avoid 3 points in the same row or column
	MySpinCursor(); // JLM 8/4/99
	//if (err = maketriangles(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts))
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology))))goto done;	
	
	// point and triangle indices should start with zero
	for(i = 0; i < 3*ntri; i ++)
	{
		//if (tri_neighbors[i]==0)
		 //tri_neighbors[i]=-1;
		 //else 
		tri_neighbors[i] = tri_neighbors[i] - 1;
		tri_verts[i] = tri_verts[i] - 1;
	}
	for(i = 0; i < ntri; i ++)
	{	// topology data needs to be CCW
		(*topo)[i].vertex1 = tri_verts[i];
		//(*topo)[i].vertex2 = tri_verts[i+ntri];
		(*topo)[i].vertex3 = tri_verts[i+ntri];
		//(*topo)[i].vertex3 = tri_verts[i+2*ntri];
		(*topo)[i].vertex2 = tri_verts[i+2*ntri];
		(*topo)[i].adjTri1 = tri_neighbors[i];
		//(*topo)[i].adjTri2 = tri_neighbors[i+ntri];
		(*topo)[i].adjTri3 = tri_neighbors[i+ntri];
		//(*topo)[i].adjTri3 = tri_neighbors[i+2*ntri];
		(*topo)[i].adjTri2 = tri_neighbors[i+2*ntri];
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
	{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
	/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in Map3D_c::SetUpTriangleGrid2()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(triBounds); 
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement
	triGrid -> SetDepths(depths);
	//if (topo) fNumEles = _GetHandleSize((Handle)topo)/sizeof(**topo);	// should be set in TextRead
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	depths = 0; // because fGrid is now responsible for it
	//totalDepthH = 0; // because fGrid is now responsible for it
	
	/////////////////////////////////////////////////
	numBoundaryPts = INDEXH(verdatBreakPtsH,numVerdatBreakPts-1)+1;
	waterBoundariesH = (LONGH)_NewHandle(sizeof(long)*numBoundaryPts);
	if (!waterBoundariesH) {err = memFullErr; goto done;}
	boundaryPtsH = (LONGH)_NewHandleClear(numBoundaryPts * sizeof(**boundaryPtsH));
	if (!boundaryPtsH) {err = memFullErr; goto done;}
	
	for (i=0;i<numBoundaryPts;i++)
	{
		INDEXH(waterBoundariesH,i)=1;	// default is land
		if (bndry_type[i]==1)	
			INDEXH(waterBoundariesH,i)=2;	// water boundary, this marks start point rather than end point...
		INDEXH(boundaryPtsH,i) = bndry_indices[i]-1;
	}
	
	if (waterBoundariesH)	// maybe assume rectangle grids will have map?
	{
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(verdatBreakPtsH);	
		this->SetWaterBoundaries(waterBoundariesH);
		this->SetBoundaryPoints(boundaryPtsH);
		this->SetMapBounds(triBounds);
	}
	else
	{
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH=0;}
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
	}
	
	/////////////////////////////////////////////////
	
done:
	if (err) printError("Error reordering gridpoints into verdat format");
	if (vertFlagsH) {DisposeHandle((Handle)vertFlagsH); vertFlagsH = 0;}
	if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in Map3D_c::SetUpTriangleGrid2");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
	}
	return err;
}

OSErr Map3D_c::SetUpTriangleGrid(long numNodes, long numTri, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts) 
{
	OSErr err = 0;
	char errmsg[256];
	long i, n, nv = numNodes;
	long currentBoundary;
	long numVerdatPts = 0, numVerdatBreakPts = 0;
	
	LONGH vertFlagsH = (LONGH)_NewHandleClear(nv * sizeof(**vertFlagsH));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatPtsH));
	LONGH verdatBreakPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatBreakPtsH));
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	LONGH waterBoundariesH=0;
	FLOATH depths=0;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	Boolean addOne = false;	// for debugging
	
	if (!vertFlagsH || !verdatPtsH || !verdatBreakPtsH) {err = memFullErr; goto done;}
	
	// put boundary points into verdat list
	
	// code goes here, double check that the water boundary info is also reordered
	currentBoundary=1;
	if (bndry_nums[0]==0) addOne = true;	// for debugging
	for (i = 0; i < numBoundaryPts; i++)
	{	
		//short islandNum, index;
		long islandNum, index;
		index = bndry_indices[i];
		islandNum = bndry_nums[i];
		if (addOne) islandNum++;	// for debugging
		INDEXH(vertFlagsH,index-1) = 1;	// note that point has been used
		INDEXH(verdatPtsH,numVerdatPts++) = index-1;	// add to verdat list
		if (islandNum>currentBoundary)
		{
			// for verdat file indices are really point numbers, subtract one for actual index
			INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = i;	// passed a break point
			currentBoundary++;
		}
	}
	INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = numBoundaryPts;
	
	// add the rest of the points to the verdat list (these points are the interior points)
	for(i = 0; i < nv; i++) {
		if(INDEXH(vertFlagsH,i) == 0)	
		{
			INDEXH(verdatPtsH,numVerdatPts++) = i;
			INDEXH(vertFlagsH,i) = 0; // mark as used
		}
	}
	if (numVerdatPts!=nv) 
	{
		printNote("Not all vertex points were used");
		// it seems this should be an error...
		err = -1;
		goto done;
		// shrink handle
		//_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(long));
	}
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	depths = (FLOATH)_NewHandle(sizeof(float)*(numVerdatPts));
	if(pts == nil || depths == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	for (i=0; i<=numVerdatPts; i++)
	{
		long index;
		float fLong, fLat, fDepth;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	
			index = i+1;
			n = INDEXH(verdatPtsH,i);
			fLat = INDEXH(vertexPtsH,n).pLat;	
			fLong = INDEXH(vertexPtsH,n).pLong;
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = INDEXH(depthPtsH,n);	
			INDEXH(pts,i) = vertex;
			INDEXH(depths,i) = fDepth;
		}
		else { // the last line should be all zeros
			index = 0;
			fLong = fLat = fDepth = 0.0;
		}
		/////////////////////////////////////////////////
	}
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	// shrink handle
	_SetHandleSize((Handle)verdatBreakPtsH,numVerdatBreakPts*sizeof(long));
	for(i = 0; i < numVerdatBreakPts; i++ )
	{
		INDEXH(verdatBreakPtsH,i)--;
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	// use new maketriangles to force algorithm to avoid 3 points in the same row or column
	MySpinCursor(); // JLM 8/4/99
	if (err = maketriangles(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts))
		//if (err = maketriangles2(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts,verdatPtsH,fNumCols_ext))
		goto done;
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
	{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
	/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in Map3D_c::ReorderPoints()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	//fGrid = (TTriGridVel3D*)triGrid;
	
	triGrid -> SetBounds(triBounds); 
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement
	triGrid -> SetDepths(depths);	// used by PtCurMap to check vertical movement
	//if (topo) fNumEles = _GetHandleSize((Handle)topo)/sizeof(**topo);
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	depths = 0; // because fGrid is now responsible for it
	//totalDepthH = 0; // because fGrid is now responsible for it
	
	/////////////////////////////////////////////////
	numBoundaryPts = INDEXH(verdatBreakPtsH,numVerdatBreakPts-1)+1;
	waterBoundariesH = (LONGH)_NewHandle(sizeof(long)*numBoundaryPts);
	if (!waterBoundariesH) {err = memFullErr; goto done;}
	
	for (i=0;i<numBoundaryPts;i++)
	{
		INDEXH(waterBoundariesH,i)=1;	// default is land
		if (bndry_type[i]==1)	
			INDEXH(waterBoundariesH,i)=2;	// water boundary, this marks start point rather than end point...
	}
	
	if (waterBoundariesH)	// maybe assume rectangle grids will have map?
	{
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(verdatBreakPtsH);	
		this->SetWaterBoundaries(waterBoundariesH);
		//this->SetBoundaryPoints(boundaryPtsH);
		this->SetMapBounds(triBounds);
	}
	else
	{
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH=0;}
	}
	
	/////////////////////////////////////////////////
	
done:
	if (err) printError("Error reordering gridpoints into verdat format");
	if (vertFlagsH) {DisposeHandle((Handle)vertFlagsH); vertFlagsH = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in Map3D_c::SetUpTriangleGrid");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
	}
	return err;
}

OSErr Map3D::ReadTopology(char* path)
{
	// import NetCDF triangle info so don't have to regenerate
	// this is same as curvilinear mover so may want to combine later
	char s[1024], errmsg[256];
	long i, numPoints, numTopoPoints, line = 0, numPts;
	CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH depths=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds = voidWorldRect;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries=0, numBoundaryPts=0, numBoundarySegs=0;
	LONGH boundarySegs=0, waterBoundaries=0, boundaryPts=0;
	
	errmsg[0]=0;
	
	if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("Map3D::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	
	// No header
	// start with transformation array and vertices
	MySpinCursor(); // JLM 8/4/99
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	if(IsTransposeArrayHeaderLine(s,&numPts)) // 
	{
		LONGH verdatToNetCDFH = 0;
		if (err = ReadTransposeArray(f,&line,&verdatToNetCDFH,numPts,errmsg)) 
		{strcpy(errmsg,"Error in ReadTransposeArray"); goto done;}
		if (verdatToNetCDFH) {DisposeHandle((Handle)verdatToNetCDFH); verdatToNetCDFH=0;}
	}
	else 
	//{err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}
	{
		//if (!bVelocitiesOnTriangles) {err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}
		//else line--;
		line--;
	}
	if(err = ReadTVertices(f,&line,&pts,&depths,errmsg)) goto done;
	
	if(pts) 
	{
		LongPoint	thisLPoint;
		Boolean needDepths = false;

		numPts = _GetHandleSize((Handle)pts)/sizeof(LongPoint);
		if (!depths) 
		{
			depths = (FLOATH)_NewHandle(sizeof(FLOATH)*(numPts));
			if(depths == nil)
			{
				strcpy(errmsg,"Not enough memory to read topology file.");
				goto done;
			}
			needDepths = true;
		}
		if(numPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
				if (needDepths) INDEXH(depths,i) = INFINITE_DEPTH;
				
			}
		}
	}
	MySpinCursor();
	
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	if(IsBoundarySegmentHeaderLine(s,&numBoundarySegs)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundarySegs>0)
			err = ReadBoundarySegs(f,&line,&boundarySegs,numBoundarySegs,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary segment header line");
		//goto done;
		// not needed for 2D files, but we require for now
	}
	MySpinCursor(); // JLM 8/4/99
	
	if(IsWaterBoundaryHeaderLine(s,&numWaterBoundaries,&numBoundaryPts)) // Boundary types from CATs
	{
		MySpinCursor();
		err = ReadWaterBoundaries(f,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Water boundaries header line");
		//goto done;
		// not needed for 2D files, but we require for now
	}
	MySpinCursor(); // JLM 8/4/99
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsBoundaryPointsHeaderLine(s,&numBoundaryPts)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundaryPts>0)
			err = ReadBoundaryPts(f,&line,&boundaryPts,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary points header line");
		//goto done;
		// not always needed ? probably always needed for curvilinear
	}
	MySpinCursor(); // JLM 8/4/99
	
	if(IsTTopologyHeaderLine(s,&numTopoPoints)) // Topology from CATs
	{
		MySpinCursor();
		err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numTopoPoints,FALSE);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		err = -1; // for now we require TTopology
		strcpy(errmsg,"Error in topology header line");
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATs
	{
		MySpinCursor();
		err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
		if(err) goto done;
	}
	else
	{
		err = -1; // for now we require TIndexedDagTree
		strcpy(errmsg,"Error in dag tree header line");
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	/////////////////////////////////////////////////
	// if the boundary information is in the file we'll need to create a bathymetry map (required for 3D)
	
	// check if bVelocitiesOnTriangles and boundaryPts
	if (waterBoundaries && boundarySegs)
	{
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(boundarySegs);	
		this->SetWaterBoundaries(waterBoundaries);
		if (boundaryPts) this->SetBoundaryPoints(boundaryPts);	
		this->SetMapBounds(bounds);		
	}
	else	// maybe assume rectangle grids will have map?
	{
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts = 0;}
	}
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in Map3D::ReadTopology()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(bounds); 
	triGrid -> SetDepths(depths);
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		printError("Unable to read Extended Topology file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(depths);
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	//depths = 0;
	
done:
	
	//if(depths) {DisposeHandle((Handle)depths); depths=0;}
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in Map3D::ReadTopology");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts = 0;}
	}
	return err;
}

OSErr Map3D::ExportTopology(char* path)
{
	// export NetCDF curvilinear info so don't have to regenerate each time
	// move to NetCDFMover so Tri can use it too
	OSErr err = 0;
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0, nBoundaryPts;
	long i, n, v1,v2,v3,n1,n2,n3;
	double x,y,z;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	FLOATH depthsH=0;
	DAGHdl		treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0, boundaryPointsH = 0;
	BFPB bfpb;
	
	triGrid = (TTriGridVel*)(this->fGrid);
	if (!triGrid) {printError("There is no topology to export"); return -1;}
	dagTree = triGrid->GetDagTree();
	if (dagTree) 
	{
		ptsH = dagTree->GetPointsHdl();
		topH = dagTree->GetTopologyHdl();
		treeH = dagTree->GetDagTreeHdl();
	}
	else 
	{
		printError("There is no topology to export");
		return -1;
	}
	depthsH = triGrid->GetDepths();
	if(!ptsH || !topH || !treeH || !depthsH) 
	{
		printError("There is no topology to export");
		return -1;
	}
	boundaryTypeH = GetWaterBoundaries();
	boundarySegmentsH = GetBoundarySegs();
	boundaryPointsH = GetBoundaryPoints();	// if no boundaryPointsH just a special case
	if (!boundaryTypeH || !boundarySegmentsH /*|| !boundaryPointsH*/) 
		{printError("No map info to export"); err=-1; goto done;}
	else
	{
		// any issue with trying to write out non-existent fields?
	}
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	
	// Write out values
	/*if (fVerdatToNetCDFH) n = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(long);
	else {printError("There is no transpose array"); err = -1; goto done;}
	sprintf(hdrStr,"TransposeArray\t%ld\n",n);	
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i=0;i<n;i++)
	{	
		sprintf(topoStr,"%ld\n",(*fVerdatToNetCDFH)[i]);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}*/
	
	nver = _GetHandleSize((Handle)ptsH)/sizeof(**ptsH);
	//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
	sprintf(hdrStr,"Vertices\t%ld\n",nver);	// total vertices
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	sprintf(hdrStr,"%ld\t%ld\n",nver,nver);	// junk line
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i=0;i<nver;i++)
	{	
		x = (*ptsH)[i].h/1000000.0;
		y =(*ptsH)[i].v/1000000.0;
		z = (*depthsH)[i];
		//sprintf(topoStr,"%ld\t%lf\t%lf\t%lf\n",i+1,x,y,(*gDepths)[i]);
		//sprintf(topoStr,"%ld\t%lf\t%lf\n",i+1,x,y);
		//sprintf(topoStr,"%lf\t%lf\n",x,y);
		sprintf(topoStr,"%lf\t%lf\t%lf\n",x,y,z);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	//boundary points - an optional handle, only for curvilinear case
	
	if (boundarySegmentsH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundarySegmentsH)/sizeof(long);
		//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
		sprintf(hdrStr,"BoundarySegments\t%ld\n",nBoundarySegs);	// total vertices
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundarySegs;i++)
		{	
			//sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]);
			sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]+1);	// when reading in subtracts 1
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}
	nBoundarySegs = 0;
	if (boundaryTypeH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundaryTypeH)/sizeof(long);	// should be same size as previous handle
		//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
		for(i=0;i<nBoundarySegs;i++)
		{	
			if ((*boundaryTypeH)[i]==2) nWaterBoundaries++;
		}
		sprintf(hdrStr,"WaterBoundaries\t%ld\t%ld\n",nWaterBoundaries,nBoundarySegs);	
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundarySegs;i++)
		{	
			if ((*boundaryTypeH)[i]==2)
				//sprintf(topoStr,"%ld\n",(*boundaryTypeH)[i]);
			{
				sprintf(topoStr,"%ld\n",i);
				strcpy(buffer,topoStr);
				if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			}
		}
	}
	
	nBoundaryPts = 0;
	if (boundaryPointsH) 
	{
		nBoundaryPts = _GetHandleSize((Handle)boundaryPointsH)/sizeof(long);	// should be same size as previous handle
		sprintf(hdrStr,"BoundaryPoints\t%ld\n",nBoundaryPts);	// total boundary points
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundaryPts;i++)
		{	
			sprintf(topoStr,"%ld\n",(*boundaryPointsH)[i]);	// when reading in subtracts 1
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}
	numTriangles = _GetHandleSize((Handle)topH)/sizeof(**topH);
	sprintf(hdrStr,"Topology\t%ld\n",numTriangles);
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i = 0; i< numTriangles;i++)
	{
		v1 = (*topH)[i].vertex1;
		v2 = (*topH)[i].vertex2;
		v3 = (*topH)[i].vertex3;
		n1 = (*topH)[i].adjTri1;
		n2 = (*topH)[i].adjTri2;
		n3 = (*topH)[i].adjTri3;
		sprintf(topoStr, "%ld\t%ld\t%ld\t%ld\t%ld\t%ld\n",
				v1, v2, v3, n1, n2, n3);
		
		/////
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
	numBranches = _GetHandleSize((Handle)treeH)/sizeof(**treeH);
	sprintf(hdrStr,"DAGTree\t%ld\n",dagTree->fNumBranches);
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	
	for(i = 0; i<dagTree->fNumBranches; i++)
	{
		sprintf(topoStr,"%ld\t%ld\t%ld\n",(*treeH)[i].topoIndex,(*treeH)[i].branchLeft,(*treeH)[i].branchRight);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		printError("Error writing topology");
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}