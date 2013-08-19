/*
 *  TimeValuesIO.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/27/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <vector>
#include <algorithm>

#include "Basics.h"
#include "TypeDefs.h"
#include "StringFunctions.h"
#include "CompFunctions.h"
#include "OUTILS.H"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

bool IsTimeFile(vector<string> &linesInFile);
Boolean IsTimeFile(char *path);

bool IsOSSMHeightFile(vector<string> &linesInFile, short *selectedUnitsP);
Boolean IsOSSMHeightFile(char *path, short *selectedUnitsP);

bool IsOSSMTimeFile(vector<string> &linesInFile, short *selectedUnitsOut);
Boolean IsOSSMTimeFile(char *path, short *selectedUnitsP);

bool IsHydrologyFile(vector<string> &linesInFile);
Boolean IsHydrologyFile(char *path);

bool IsNDBCWindFile(vector<string> &linesInFile, long *numHeaderLines);
Boolean IsNDBCWindFile(char *path, long *numHeaderLines);

bool IsNCDCWindFile(vector<string> &linesInFile);
Boolean IsNCDCWindFile(char *path);

bool IsLongWindFile(vector<string> &linesInFile, short *selectedUnitsOut, bool *dataInGMTOut);
Boolean IsLongWindFile(char *path, short *selectedUnitsOut, Boolean *dataInGMTOut);
