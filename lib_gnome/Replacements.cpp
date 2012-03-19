/*
 *  Replacements.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "Replacements.h"

Model_c *model = 0;
Settings settings;

PtCurMap_c *GetPtCurMap(void) {
	return NULL;
}

void MySpinCursor(void) { return; }

void SysBeep(short x) { return; }

Boolean OSPlotDialog(OiledShorelineData** oiledShorelineHdl) { return 0; }