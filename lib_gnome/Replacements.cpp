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
#include <vector>

#include <numeric>

using namespace std;
using std::cout;

void DisplayMessage(const char *msg)
{
	return;
}

void MySpinCursor(void)
{
	return;
}

void SysBeep(short x)
{
	return;
}

Boolean CmdPeriod(void)
{
	return false;
}

Boolean FileExists(short vRefNum, long dirID, const char *filename)
{
	struct stat buffer;
 	return (stat(filename, &buffer) == 0);
}

OSErr ReadSectionOfFile(short vRefNum, long dirID, CHARPTR name,
						long offset, long length,
						VOIDPTR ptr, CHARHP handle)
{
	// TODO: right now this is a hack.
	//       we really need to change the API for this function,
	//       but a lot of stuff is dependant on it being as it is.
	ios::pos_type localOffset = (ios::pos_type)offset;
	ios::pos_type localLength = (ios::pos_type)length;

	try {
		fstream inputFile(name, ios::in|ios::binary);

		if (inputFile.is_open()) {
			ios::pos_type fileSize = FileSize(inputFile);
			if(fileSize <= 0)
				throw("Empty file.\n");

			if (localOffset > fileSize)
				throw("Trying to read beyond the file size.\n");

			// figure out how many bytes to read
			if ((localOffset + localLength) > fileSize)
				localLength = fileSize - localOffset;

			inputFile.seekg(localOffset, ios::beg);

			if(handle) {
				*handle = _NewHandle((long)localLength);
				inputFile.read(DEREFH(*handle), localLength);
			}
			else {
				inputFile.read((char *)ptr, localLength);
			}
			inputFile.close();
		}
		else {
			throw("Unable to open file");
		}
	}
	catch(...) {
		printError("We are unable to open or read from the file. \nBreaking from ReadSectionOfFile().\n");
		return true;
	}
	return false;
}

// Make sure terminationFlag can be typed to a Boolean
OSErr ReadFileContents(short terminationFlag, short vRefNum, long dirID,
					   CHARPTR name, VOIDPTR ptr, long length, CHARHP handle)
{
	// TODO: right now this is a hack.
	//       we really need to change the API for this function,
	//       but a lot of stuff is dependant on it being as it is.
	ios::pos_type localLength = (ios::pos_type)length;

	bool terminate;

	if (handle) {
		*handle = 0;
	}

	switch (terminationFlag) {
		case TERMINATED:
			terminate  = true;
			break;
		case NONTERMINATED:
			terminate  = false;
			break;
		default:
			printError("Bad flag in ReadFileContents");
			return -1;
	}

	try {
		fstream inputFile(name, ios::in|ios::binary);

		if (inputFile.is_open()) {
			ios::pos_type fileSize = FileSize(inputFile);

			if (fileSize <=  0)
				throw("Empty file.\n");

			if ((localLength == (ios::pos_type)0) ||
				(localLength > fileSize))
			{
				// either we didn't specify a length, or
				// we specified a length that was bigger than the file.
				localLength = fileSize;
			}

			inputFile.seekg(0, ios::beg);

			if(handle) {
				*handle = _NewHandle((long)localLength + (terminate ? 1 : 0));

				inputFile.read(DEREFH(*handle), localLength);
			}
			else {
				inputFile.read((char *)ptr, localLength);
			}
			inputFile.close();
		}
		else {
			throw("Unable to open file");
		}
	}
	catch(...) {
		printError("We are unable to open or read from the file. \nBreaking from ReadSectionOfFile().\n");
		return true;
	}

	return false;
}


OSErr AskUserForUnits(short *selectedUnits, Boolean *userCancel)
{
	return -1;
}

OSErr MyGetFileSize(short vRefNum, long dirID, CHARPTR pathName, LONGPTR size)
{
	try {
		fstream inputFile(pathName, ios::in|ios::binary|ios::ate);

		if (inputFile.is_open()) {
			*size = (long)inputFile.tellg();
			inputFile.close();
		}
		else {
			throw("Unable to open file");
		}
	}
    catch(...) {
        printError("We are unable to open or read from the file. \nBreaking from MyGetFileSize().\n");
        return true;
    }
    return false;
}

// add delimiter at end if the last char is not a delimiter
void AddDelimiterAtEndIfNeeded(char *str)
{
	long len = strlen(str);

	if (str[len - 1] != DIRDELIMITER) {
		str[len] = DIRDELIMITER;
		str[len + 1] = 0;
	}
}

Boolean IsPartialPath(char *relativePath)
{
	// To simplify scripting, we allow the relative path to use either 
	// platform delimiter.
	char delimiter = NEWDIRDELIMITER;
	char macDelimiter = ':';
	char ibmDelimiter = '\\';
	char unixDelimiter = '/';	// handle ./, ../, etc
	char otherDelimiter;

	char unixDirectoryUp = '.';	// handle ./, ../, etc
	Boolean isRelativePath;
	
	if (IsFullPath(relativePath))
		return false;
	
	if (delimiter == macDelimiter)
		otherDelimiter = ibmDelimiter;
	else
		otherDelimiter = macDelimiter;
	
	
	isRelativePath = (relativePath[0] == macDelimiter ||
					  relativePath[0] == ibmDelimiter ||
					  relativePath[0] == unixDirectoryUp);
	
	return true;
	// try if not full path then relative to make sure we get pathname only case
	//return(isRelativePath);
}

void ResolvePartialPathFromThisFolderPath(char *relativePath, char *thisFolderPath)
{
	// To simplify scripting, we allow the relative path to use either 
	// platform delimiter.
	char delimiter = NEWDIRDELIMITER;
	char ibmDelimiter = '\\';
	char macOrUnixDelimiter = '/';
	char classicDelimiter = ':';
	char otherDelimiter;

	char unixDirectoryUp = '.';	// handle ./, ../, etc
	long numChops = 0;
	long len;
	char *p;
	char fullFolderPath[256];

	if(!IsPartialPath(relativePath))
		return; 
		
	strcpy(fullFolderPath, thisFolderPath);

	if (delimiter == macOrUnixDelimiter)
		otherDelimiter = ibmDelimiter;
	else
		otherDelimiter = macOrUnixDelimiter;

	// substitute to the appropriate delimiter
	// be careful of the IBM  "C:\" type stings
	len = strlen(relativePath);
	for (long i = 0; i < len  && relativePath[i]; i++) {
		if (relativePath[i] == otherDelimiter && relativePath[i + 1] != delimiter)
			relativePath[i] = delimiter;
	}
	
	if (relativePath[0] == unixDirectoryUp) {
		// count the number of directories to chop (# delimiters - 1)
		for (long i = 1; i < len  && relativePath[i] == unixDirectoryUp; i++) {
			numChops++;
		}
	}
	else {
		// count the number of directories to chop (# delimiters - 1), old style
		for (long i = 1; i < len  && relativePath[i] == delimiter; i++) {
			numChops++;
		}
	}
	
	// to be nice, we will be flexible about whether or not fullFolderPath ends in a DIRDELIMITER.
	// so chop the delimiter if there is one
	len = strlen(fullFolderPath);
	if (len > 0 && fullFolderPath[len - 1] == NEWDIRDELIMITER)
		fullFolderPath[len - 1] = 0; // chop this delimiter
	
	for (long i = 0; i < numChops; i++) {
		// chop the support files directory, i.e. go up one directory
		p = strrchr(fullFolderPath, NEWDIRDELIMITER);
		if (p)
			*p = 0;
	}

	// add the relative part 
	if (relativePath[0] == unixDirectoryUp)
		strcat(fullFolderPath, relativePath + numChops + 1);
	else {
		if (relativePath[0] != delimiter) // allow for filenames
			AddDelimiterAtEndIfNeeded(fullFolderPath);

		strcat(fullFolderPath, relativePath + numChops);
	}
	
	// finally copy the path back into the input variable 
	strcpy(relativePath, fullFolderPath);
}


// This function tries to determine whether a path is a valid absolute path,
// or a path that is relative to a containing directory.
// If the path is successfully resolved, the resolved form of the path is
// written to the pathToResolve argument.
//
// typically the files are either in directoryOfSaveFile or down one level,
// but we will try any number of levels
// example search execution:
//     inputDir = "."
//     pathToResolve = "dir1/dir2/file1.txt"
//
//     iteration 0:
//         pathToTry = "./dir1/dir2/file1.txt"
//     iteration 1:
//         pathToTry = "./dir2/file1.txt"
//     iteration 2:
//         pathToTry = "./file1.txt"

// Chris has asked that the input files can use a relative path from the input file.
// Previously he was forced to use absolute paths.
// So now, sometimes the path is saved in an input file will just be a file name,
// and sometimes it will have been an absolute path, but the absolute path
// may have been broken when the file is moved.
// Often the referenced files are in a folder with the input file and it is just that
// the folder has been moved.
// This function helps look for these referenced files and changes the input parameter
// pathToResolveFromInputFile if it can find the file.
// Otherwise pathToResolveFromInputFile will be unchanged.
void ResolvePathFromInputFile(char *pathOfTheInputFile, char *pathToResolve) // JLM 6/8/10
{
	string inputPath = pathOfTheInputFile;
	string linkedFile = pathToResolve;

	string dir, file;
	SplitPathIntoDirAndFile(inputPath, dir, file);

	if (ResolvePath(dir, linkedFile)) {
		strcpy(pathToResolve, linkedFile.c_str());
	}

	return;
}


bool IsGridCurTimeFile (vector<string> &linesInFile, short *selectedUnitsOut)
{
	long lineIdx = 0;
	string currentLine;

	short selectedUnits = kUndefined;
	string value1S, value2S;

	// First line, must start with '[GRIDCURTIME] <units>'
	// <units> == the designation of units for the file.
	currentLine = trim(linesInFile[lineIdx++]);

	istringstream lineStream(currentLine);

	lineStream >> value1S >> value2S;
	if (lineStream.fail())
		return false;

	if (value1S != "[GRIDCURTIME]")
		return false;

	selectedUnits = StrToSpeedUnits((char *)value2S.c_str());
	if (selectedUnits == kUndefined)
		return false;

	*selectedUnitsOut = selectedUnits;
	
	return true;
}


Boolean IsGridCurTimeFile(char *path, short *selectedUnitsOut)
{
	vector<string> linesInFile;

	if (ReadLinesInFile(path, linesInFile, 10)) {
		return IsGridCurTimeFile(linesInFile, selectedUnitsOut);
	}
	else {
		return false;
	}
}

bool IsGridWindFile (vector<string> &linesInFile, short *selectedUnitsOut)
{
	long lineIdx = 0;
	string currentLine;

	short selectedUnits = kUndefined;
	string value1S, value2S;

	// First line, must start with '[GRIDCURTIME] <units>'
	// <units> == the designation of units for the file.
	currentLine = trim(linesInFile[lineIdx++]);

	istringstream lineStream(currentLine);

	lineStream >> value1S >> value2S;
	if (lineStream.fail())
		return false;

	if (value1S != "[GRIDWIND]" && value1S != "[GRIDWINDTIME]")
		return false;

	selectedUnits = StrToSpeedUnits((char *)value2S.c_str());
	if (selectedUnits == kUndefined)
		return false;

	*selectedUnitsOut = selectedUnits;
	
	return true;
}


Boolean IsGridWindFile(char *path, short *selectedUnitsOut)
{
	vector<string> linesInFile;

	if (ReadLinesInFile(path, linesInFile, 10)) {
		return IsGridWindFile(linesInFile, selectedUnitsOut);
	}
	else {
		return false;
	}
}


bool IsPtCurFile(vector<string> &linesInFile)
{
	string key;

	// First line, must start with [FILETYPE] PTCUR
	if (ParseKeyedLine(linesInFile[0], "[FILETYPE]", key) && key == "PTCUR")
		return true;
	else
		return false;
}

Boolean IsPtCurFile(char *path)
{
	vector<string> linesInFile;

	if (ReadLinesInFile(path, linesInFile, 1))
		return IsPtCurFile(linesInFile);
	else
		return false;
}


bool IsCATS3DFile(vector<string> &linesInFile)
{
	string key;

	// must start with CATS3D
	if (ParseLine(linesInFile[0], key) && key == "CATS3D")
		return true;
	else
		return false;
}

Boolean IsCATS3DFile(char *path)
{
	string strPath = path;

	if (strPath.size() == 0)
		return false;

	vector<string> linesInFile;
	if (ReadLinesInFile(strPath, linesInFile, 1))
		return IsCATS3DFile(linesInFile);

	else
		return false;
}

bool IsShioFile(vector<string> &linesInFile)
{
	long line = 0;
	string value;
	
	// the first line of the file needs to be "[StationInfo]"
	if (ParseKeyedLine(linesInFile[line++], "[StationInfo]", value))
		return true;
	else
		return false;
}

Boolean IsShioFile(char* path)
{
	vector<string> linesInFile;
	
	if (ReadLinesInFile(path, linesInFile))
		return IsShioFile(linesInFile);
	else
		return false;
}

#endif
