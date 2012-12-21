/*
 *  Replacements.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef __Replacements__
#define __Replacements__

#include "StringFunctions.h"
#include "OSSMTimeValue_c.h"
#include "PtCurMover_c.h"
#include "OLEList_c.h"
#include "LEList_c.h"
#include "Model_c.h"
#include "Mover_c.h"
#include "Map_c.h"
#include "GridVel_c.h"
#include "CurrentMover_c.h"
#include "CATSMover_c.h"
#include "WindMover_c.h"
//#include "CATSMover3D_c.h"
#include "TriCurMover_c.h"
#include "TriGridVel_c.h"
#include "RectGridVel_c.h"
#include "TideCurCycleMover_c.h"
#include "Weatherer_c.h"
#include "OSSMWeatherer_c.h"
#include "Random_c.h"
#include "CompoundMover_c.h"
#include "CompoundMap_c.h"
#include "ShioTimeValue_c.h"
#include "ObjectUtils.h"
#define TShioTimeValue ShioTimeValue_c
#define TWindMover WindMover_c
#define TRandom Random_c
#define TRandom3D Random3D_c
#define TCompoundMover CompoundMover_c
#define TCompoundMap CompoundMap_c
#define TCurrentMover CurrentMover_c
#define TideCurCycleMover TideCurCycleMover_c
#define TOSSMTimeValue OSSMTimeValue_c
#define NetCDFMover NetCDFMover_c
#define NetCDFMoverTri NetCDFMoverTri_c
#define NetCDFMoverCurv NetCDFMoverCurv_c
#define NetCDFWindMover NetCDFWindMover_c
#define TCATSMover3D CATSMover3D_c
#define TCATSMover CATSMover_c
#define TTriGridVel TriGridVel_c
#define TTriGridVel3D TriGridVel3D_c
#define TriCurMover TriCurMover_c
#define PtCurMap PtCurMap_c
#define PtCurMover PtCurMover_c
#define TWeatherer Weatherer_c
#define TOSSMWeatherer OSSMWeatherer_c
#define TGridVel GridVel_c
#define TRectGridVel RectGridVel_c
#define CMapLayer CMapLayer_c
#define TLEList LEList_c
#define TOLEList OLEList_c
#define TMover Mover_c
#define TModel Model_c
#define TMap Map_c
#define TimeGridVel TimeGridVel_c
#define TimeGridVelRect TimeGridVelRect_c
#define TimeGridVelCurv TimeGridVelCurv_c
#define TimeGridVelTri TimeGridVelTri_c

#define TechError(a, b, c) printf(a)
#define printError(msg) printf(msg)
#define printNote(msg) printf(msg)
#define DisplayMessage(msg) printf(msg)
#ifndef ibmpyGNOME
#define _isnan isnan
#endif

//PtCurMap_c *GetPtCurMap(void);
void MySpinCursor(void);
void SysBeep(short);
//Boolean OSPlotDialog(OiledShorelineData** oiledShorelineHdl);
Boolean FileExists(short vRefNum, long dirID, CHARPTR filename);
OSErr MyGetFileSize(short vRefNum, long dirID, CHARPTR pathName, LONGPTR size);
OSErr ReadSectionOfFile(short vRefNum, long dirID, CHARPTR name,
						long offset, long length, VOIDPTR ptr, CHARHP handle);
OSErr ReadFileContents(short terminationFlag, short vRefNum, long dirID, CHARPTR name,
					   VOIDPTR ptr, long length, CHARHP handle);
void paramtext(char* p0,char* p1,char* p2,char* p3);
OSErr AskUserForUnits(short* selectedUnits,Boolean *userCancel);
Boolean CmdPeriod(void);
void PenNormal(void);
long			ScreenToWorldDistance(short pixels);

void AddDelimiterAtEndIfNeeded(char* str);
Boolean IsPartialPath(char* relativePath);
void ResolvePartialPathFromThisFolderPath(char* relativePath,char * thisFolderPath);
void ResolvePathFromInputFile(char *pathOfTheInputFile, char* pathToResolve); 
#endif