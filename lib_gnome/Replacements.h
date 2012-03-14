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

#include "LEList_c.h"
#include "OLEList_c.h"
#include "PtCurMap_c.h"
#include "Model_c.h"
#include "Mover_c.h"
#include "Map_c.h"
#include "OSSMTimeValue_c.h"
#include "GridVel_c.h"
#include "TriGridVel_c.h"

#define TOSSMTimeValue OSSMTimeValue_c
#define TLEList LEList_c
#define TOLEList OLEList_c
#define TGridVel GridVel_c
#define TTriGridVel TriGridVel_c
#define PtCurMap PtCurMap_c
#define TCATSMover CATSMover_c
#define TMover Mover_c
#define TModel Model_c
#define TMap Map_c

#define memFullErr -108


#define TechError(a, b, c) printf(a)
#define printError(msg) printf(msg)
#define printNote(msg) printf(msg)


extern Model_c *model;
extern Settings settings;

PtCurMap_c *GetPtCurMap(void);
void MySpinCursor(void);
void SysBeep(short x);

#endif