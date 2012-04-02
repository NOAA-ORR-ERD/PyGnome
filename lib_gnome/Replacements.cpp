/*
 *  Replacements.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "Replacements.h"
#include <fstream>
#include <ios>

using std::fstream;
using std::ios;

Model_c *model = new Model_c();	// poo poo global.
Settings settings;

PtCurMap_c *GetPtCurMap(void) {
	return NULL;
}

void MySpinCursor(void) { return; }

void SysBeep(short x) { return; }

Boolean OSPlotDialog(OiledShorelineData** oiledShorelineHdl) { return 0; }

Boolean CmdPeriod(void) { return false; }

void PenNormal(void) { return; }

long ScreenToWorldDistance(short pixels) { return 0; } // temporary, obviously.\

OSErr ReadSectionOfFile(short vRefNum, long dirID, CHARPTR name,
						long offset, long length, VOIDPTR ptr, CHARHP handle) {
	char c;
	int x = 0, i = 0;

	try {
		fstream *_ifstream = new fstream(name, ios::in);
		for(; _ifstream->get(c); x++);
		delete _ifstream;
		if(!(x > 0))
			throw("empty file.\n");
		_ifstream = new fstream(name, ios::in);
		for(int k = 0; k < offset; k++) _ifstream->get(c); 
		if(x > offset+length && length != 0)
		    x = offset+length;
		if(handle) {
			*handle = _NewHandle(x-offset);
			for(; i < x-offset && _ifstream->get(c); i++)
				DEREFH(*handle)[i] = c;
		} 
		else {
			for(; i < x-offset && _ifstream->get(c); i++)
				((char *)ptr)[i] = c;
		}
	}
	catch(...) {
		printError("We are unable to open or read from the file. \nBreaking from ReadSectionOfFile().\n");
		return true;
	}
	return false;
}

OSErr ReadFileContents(short terminationFlag, short vRefNum, long dirID, CHARPTR name,
					   VOIDPTR ptr, long length, CHARHP handle) {
	char c;
	int x = 0, i = 0;

	try {
		fstream *_ifstream = new fstream(name, ios::in);
		for(; _ifstream->get(c); x++);
		delete _ifstream;
		if(!(x > 0))
			throw("empty file.\n");
		_ifstream = new fstream(name, ios::in);
		if(x > length && length != 0)
		    x = length;
		if(handle) {
			*handle = _NewHandle(x);
			for(; i < x && _ifstream->get(c); i++)
				DEREFH(*handle)[i] = c;
		} 
		else {
			for(; i < x && _ifstream->get(c); i++)
				((char *)ptr)[i] = c;
		}
	}
	catch(...) {
		printError("We are unable to open or read from the file. \nBreaking from ReadSectionOfFile().\n");
		return true;
	}
	return false;
}

void paramtext(char* p0,char* p1,char* p2,char* p3) { return; }

OSErr MyGetFileSize(short vRefNum, long dirID, CHARPTR pathName, LONGPTR size) {
	
	char c;
	long x = 0;

	try {
		fstream *_ifstream = new fstream(pathName, ios::in);
		for(; _ifstream->get(c); x++);
		delete _ifstream;
		*size = x;
		}
    catch(...) {
        printError("We are unable to open or read from the file. \nBreaking from MyGetFileSize().\n");
        return true;
    }
    return false;
}