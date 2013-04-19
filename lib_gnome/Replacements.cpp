/*
 *  Replacements.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef pyGNOME
#include "Replacements.h"
#include <fstream>
#include <ios>
#include <sys/stat.h>

using std::fstream;
using std::ios;

void DisplayMessage(char *msg) {return;}

void MySpinCursor(void) { return; }

void SysBeep(short x) { return; }

Boolean CmdPeriod(void) { return false; }

//void PenNormal(void) { return; }

//long ScreenToWorldDistance(short pixels) { return 0; } // temporary, obviously.\

Boolean FileExists(short vRefNum, long dirID, CHARPTR filename)
{
	struct stat buffer;
 	return (stat(filename,&buffer)==0);
}

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
		delete _ifstream;
	}
	catch(...) {
		printError("We are unable to open or read from the file. \nBreaking from ReadSectionOfFile().\n");
		return true;
	}
	return false;
}

// Make sure terminationFlag can be typed to a Boolean
OSErr ReadFileContents(short terminationFlag, short vRefNum, long dirID, CHARPTR name,
					   VOIDPTR ptr, long length, CHARHP handle) {
	char c;
	int x = 0, i = 0;

	Boolean terminate;
	if(handle) *handle = 0; 
	switch(terminationFlag)
	{
		case TERMINATED:
			terminate  = true; break;
		case NONTERMINATED:
			terminate  = false; break;
		default:
			printError("Bad flag in ReadFileContents");return -1;
	}	

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
			*handle = _NewHandle(x + (terminate ? 1 : 0));
			for(; i < x && _ifstream->get(c); i++)
				DEREFH(*handle)[i] = c;
		} 
		else {
			for(; i < x && _ifstream->get(c); i++)
				((char *)ptr)[i] = c;
		}
		delete _ifstream;
	}
	catch(...) {
		printError("We are unable to open or read from the file. \nBreaking from ReadSectionOfFile().\n");
		return true;
	}
	return false;
}

//void paramtext(char* p0,char* p1,char* p2,char* p3) { return; }

OSErr AskUserForUnits(short* selectedUnits,Boolean *userCancel) { return -1; }

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

void AddDelimiterAtEndIfNeeded(char* str)
{	// add delimiter at end if the last char is not a delimiter
	long len = strlen(str);
	if(str[len-1] != DIRDELIMITER) 
	{
		str[len] = DIRDELIMITER;
		str[len+1] = 0;
	}
}

Boolean IsPartialPath(char* relativePath)
{
	// To simplify scripting, we allow the relative path to use either 
	// platform delimiter.
	char macDelimiter = ':';
	char ibmDelimiter = '\\';
	char unixDelimiter = '/';	// handle ./, ../, etc
	char unixDirectoryUp = '.';	// handle ./, ../, etc
	char delimiter = NEWDIRDELIMITER;
	char otherDelimiter;
	Boolean isRelativePath;
	
	if (IsFullPath(relativePath)) return false;
	
	if(delimiter == macDelimiter)
		otherDelimiter = ibmDelimiter;
	else
		otherDelimiter = macDelimiter;
	
	
	isRelativePath = 	(relativePath[0] == macDelimiter || relativePath[0] == ibmDelimiter || relativePath[0] == unixDirectoryUp);
	
	return true;	// try if not full path then relative to make sure we get pathname only case
	//return(isRelativePath);
}

void ResolvePartialPathFromThisFolderPath(char* relativePath,char * thisFolderPath)
{
	// To simplify scripting, we allow the relative path to use either 
	// platform delimiter.
	char classicDelimiter = ':';
	char ibmDelimiter = '\\';
	char macOrUnixDelimiter = '/';
	char unixDirectoryUp = '.';	// handle ./, ../, etc
	char delimiter = NEWDIRDELIMITER;
	char otherDelimiter;
	long numChops = 0;
	long len,i;
	char* p;
	Boolean isRelativePath;
	char fullFolderPath[256];
	
	
	if(!IsPartialPath(relativePath))
		return; 
		
	strcpy(fullFolderPath,thisFolderPath);

	if(delimiter == macOrUnixDelimiter)
		otherDelimiter = ibmDelimiter;
	else
		otherDelimiter = macOrUnixDelimiter;

	// substitute to the appropriate delimiter
	// be careful of the IBM  "C:\" type stings
	
	len = strlen(relativePath);
	for(i = 0; i < len  && relativePath[i]; i++)
	{
		if(relativePath[i] == otherDelimiter && relativePath[i+1] != delimiter)
			relativePath[i] = delimiter;
	}
	
	if (relativePath[0]==unixDirectoryUp)
	{
		// count the number of directories to chop (# delimiters - 1)
		for(i = 1; i < len  && relativePath[i] == unixDirectoryUp; i++)
		{
			numChops++;
		}
	}
	else
	{
		// count the number of directories to chop (# delimiters - 1), old style
		for(i = 1; i < len  && relativePath[i] == delimiter; i++)
		{
			numChops++;
		}
	}
	
	// to be nice, we will be flexible about whether or not fullFolderPath ends in a DIRDELIMITER.
	// so chop the delimiter if there is one
	len = strlen(fullFolderPath);
	if(len > 0 && fullFolderPath[len-1] == NEWDIRDELIMITER)
		fullFolderPath[len-1] = 0;// chop this delimiter
	
	for(i = 0; i < numChops; i++)
	{
		// chop the support files directory, i.e. go up one directory
		p = strrchr(fullFolderPath,NEWDIRDELIMITER);
		if(p) *p = 0;
	}
	// add the relative part 
	if (relativePath[0]==unixDirectoryUp)
		strcat(fullFolderPath,relativePath + numChops + 1);
	else
	{
		if (relativePath[0]!=delimiter)	// allow for filenames
			AddDelimiterAtEndIfNeeded(fullFolderPath);
		strcat(fullFolderPath,relativePath + numChops);
	}
	
	// finally copy the path back into the input variable 
	strcpy(relativePath,fullFolderPath);
}

void ResolvePathFromInputFile(char *pathOfTheInputFile, char* pathToResolve) // JLM 6/8/10
{
	// Chris has asked that the input files can use a relative path from the input file.
	// Previously he was forced to use absolute paths. 
	// So now, sometimes the path is saved in an input file will just be a file name, 
	// and sometimes it will have been an absolute path, but the absolute path may have been broken when the file is moved.
	// Often the referenced files are in a folder with the input file and it is just that the folder has been moved.
	// This function helps look for these referenced files and changes the input parameter pathToResolveFromInputFile
	// if it can find the file.
	// Otherwise pathToResolveFromInputFile will be unchanged.
	char pathToTry[2*kMaxNameLen] = "",pathToTest[kMaxNameLen], unixPath[kMaxNameLen];
	char pathToTearApart[kMaxNameLen] = "";
	char delimiter = NEWDIRDELIMITER;
	char directoryOfSaveFile[kMaxNameLen];
	char *p,*q;
	int i,numDelimiters;

	if(!pathOfTheInputFile)
	//if(pathOfTheInputFile == NULL)
		return;

	//if(pathOfTheInputFile[0] == NULL)
	if(pathOfTheInputFile[0] == 0)
		return;

	if(!pathToResolve)
	//if(pathToResolve == NULL)
		return;

	//if(pathToResolve[0] == NULL)
	if(pathToResolve[0] == 0)
		return;

	RemoveLeadingAndTrailingWhiteSpace(pathToResolve);

	if (ConvertIfClassicPath(pathToResolve, unixPath)) strcpy(pathToResolve,unixPath);

	if (IsFullPath(pathToResolve))
	{
		if(FileExists(0,0,pathToResolve)) {
			// no problem, the file exists at the path given
			//strcpy(pathToResolve,pathToTest);
			return;
		}
	}
	
	// otherwise we have to try to find it
	//////////////////////////////

	///////////////

	// get the directory of the save file
	strcpy(directoryOfSaveFile,pathOfTheInputFile);
	p = strrchr(directoryOfSaveFile,NEWDIRDELIMITER);
	if(p) *(p+1) = 0; // chop off the file name, leave the delimiter

	// First try to resolve relative path to the SaveFile (or whatever has been designated)
	strcpy(pathToTest,pathToResolve);
	if(IsPartialPath(pathToTest)) {
		ResolvePartialPathFromThisFolderPath(pathToTest,directoryOfSaveFile);
	}

	if(FileExists(0,0,pathToTest)) {
		// no problem, the file exists at the path given
		strcpy(pathToResolve,pathToTest);
		return;
	}

	// typically the files are either in directoryOfSaveFile or down one level, but we will try any number of levels
	q = pathToResolve;
	for(;;) { // forever
		// find the next delimiter from left to right
		// and append that path onto the directoryOfSaveFile
		strcpy(pathToTry,directoryOfSaveFile);
		strcat(pathToTry,q);
		if(strlen(pathToTry) < kMaxNameLen) { // don't try paths that we know are too long for Windows and Mac pascal strings
			if(FileExists(0,0,pathToTry)) {
				// we found the file
				strcpy(pathToResolve,pathToTry);
				return;
			}
		}
		// find the next part of the path to try
		p = strchr(q,NEWDIRDELIMITER);
		if(p == 0){
			break;// no more delimiters
		}
		//
		q = p+1; // the char after the delimiter
	}	
	return;	// file not found - may want to return a Boolean or error
}
#endif	
