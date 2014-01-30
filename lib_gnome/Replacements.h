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

//#include <vector>
#include "StringFunctions.h"
#include "OSSMTimeValue_c.h"
#include "Mover_c.h"
#include "GridVel_c.h"
#include "CurrentMover_c.h"
#include "CATSMover_c.h"
#include "WindMover_c.h"
#include "TriGridVel_c.h"
#include "RectGridVeL_c.h"
//#include "TideCurCycleMover_c.h"	// this will be part of gridcurrent mover
//#include "Weatherer_c.h"
//#include "OSSMWeatherer_c.h"	// may want to add this eventually
#include "Random_c.h"
//#include "CompoundMover_c.h"
#include "ShioTimeValue_c.h"
//#include "ObjectUtils.h"
#include "OUTILS.H"	// for the units


#define TShioTimeValue ShioTimeValue_c
#define TWindMover WindMover_c
#define TRandom Random_c
//#define TCompoundMover CompoundMover_c
#define TCurrentMover CurrentMover_c
//#define TideCurCycleMover TideCurCycleMover_c
#define TOSSMTimeValue OSSMTimeValue_c
#define TCATSMover CATSMover_c
#define TTriGridVel TriGridVel_c
//#define TWeatherer Weatherer_c
//#define TOSSMWeatherer OSSMWeatherer_c
#define TGridVel GridVel_c
#define TRectGridVel RectGridVel_c
//#define CMapLayer CMapLayer_c
#define TMover Mover_c
#define TimeGridVel TimeGridVel_c
#define TimeGridVelRect TimeGridVelRect_c
#define TimeGridVelCurv TimeGridVelCurv_c
#define TimeGridVelTri TimeGridVelTri_c
#define TimeGridCurTri TimeGridCurTri_c
#define TimeGridCurRect TimeGridCurRect_c
#define TimeGridWindRect TimeGridWindRect_c
#define TimeGridWindCurv TimeGridWindCurv_c

#define TechError(a, b, c) printf("%s", a)
#define printError(msg) printf("%s", msg)
#define printNote(msg) printf("%s", msg)

void DisplayMessage(const char *msg);
void MySpinCursor(void);
void SysBeep(short);

Boolean FileExists(short vRefNum, long dirID, const char *filename);



OSErr MyGetFileSize(short vRefNum, long dirID, CHARPTR pathName, LONGPTR size);
OSErr ReadSectionOfFile(short vRefNum, long dirID, CHARPTR name,
						long offset, long length, VOIDPTR ptr, CHARHP handle);
OSErr ReadFileContents(short terminationFlag, short vRefNum, long dirID, CHARPTR name,
					   VOIDPTR ptr, long length, CHARHP handle);

OSErr AskUserForUnits(short* selectedUnits,Boolean *userCancel);
Boolean CmdPeriod(void);

void AddDelimiterAtEndIfNeeded(char* str);
Boolean IsPartialPath(char* relativePath);
void ResolvePartialPathFromThisFolderPath(char* relativePath,char * thisFolderPath);

void ResolvePathFromInputFile(char *pathOfTheInputFile, char* pathToResolve); 

bool IsPtCurFile(std::vector<std::string> &linesInFile);
Boolean IsPtCurFile (char *path);

bool IsGridCurTimeFile (std::vector<std::string> &linesInFile, short *selectedUnitsOut);
Boolean IsGridCurTimeFile (char *path, short *selectedUnits);

bool IsGridWindFile (std::vector<std::string> &linesInFile, short *selectedUnitsOut);
Boolean IsGridWindFile (char *path, short *selectedUnits);

bool IsShioFile (std::vector<std::string> &linesInFile);
Boolean IsShioFile (char *path);

Boolean IsCATS3DFile (char *path);

#endif
