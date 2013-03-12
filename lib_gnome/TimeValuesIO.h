/*
 *  TimeValuesIO.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/27/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

Boolean IsTimeFile(char* path);
Boolean IsOSSMHeightFile(char* path,short *selectedUnitsP);
Boolean IsOSSMTimeFile(char* path,short *selectedUnitsP);
Boolean IsHydrologyFile(char* path);
Boolean IsNDBCWindFile(char* path);
Boolean IsLongWindFile(char* path,short *selectedUnitsP,Boolean *dataInGMTP);