/*
 *  StringFunctions.h
 *  gnome
 *
 *  Created by Generic Programmer on 1/11/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef __StringFunctions__
#define __StringFunctions__
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>
#include <vector>
#include <string>
#include <numeric>

using namespace std;

#include "Basics.h"
#include "TypeDefs.h"
#include "MemUtils.h"
#include "DagTree.h"
#include "ExportSymbols.h"

char* lfFix(char* str);
OSErr StringToDouble(char* str,double* val);
void StringWithoutTrailingZeros(char* str,double val,short maxNumDecimalPlaces); //JLM
void ChopEndZeros(CHARPTR cString); //JLM


std::string &ltrim(std::string &s);
std::string &rtrim(std::string &s);
std::string &trim(std::string &s);
std::vector<std::string> &rtrim_empty_lines(std::vector<std::string> &lines);
std::vector<std::string> &split(const std::string &strIn, const std::string &delims,
								std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, const std::string &delims);
std::vector<std::string> SplitPath(std::string &path);
std::string join(const std::vector<std::string> &v, const std::string &delim);


bool ParseKeyedLine(const std::string &strIn, const std::string &key,
					std::string &out);
bool ParseKeyedLine(const std::string &strIn, const std::string &key,
					short &out1);
bool ParseKeyedLine(const std::string &strIn, const std::string &key,
					long &out1);
bool ParseKeyedLine(const std::string &strIn, const std::string &key,
					double &out1);
bool ParseKeyedLine(const std::string &strIn, const std::string &key,
					long &out1, long &out2);
bool ParseKeyedLine(const std::string &strIn, const std::string &key,
					DateTimeRec &out1);

bool ParseLine(const std::string &strIn,
			   long &out1);
bool ParseLine(const std::string &strIn,
			   std::string &out1);
bool ParseLine(const std::string &strIn,
			   long &out1, double &out2, double &out3);
bool ParseLine(const std::string &strIn,
			   double &out1, double &out2);
bool ParseLine(const std::string &strIn,
			   double &out1, double &out2, double &out3);
bool ParseLine(const std::string &strIn,
		   	   long &out1, long &out2, long &out3);
bool ParseLine(const std::string &strIn,
			   DAG &out1);
bool ParseLine(const std::string &strIn,
			   Topology &out1);
bool ParseLine(const std::string &strIn,
			   Topology &out1, VelocityFRec &out2);
bool ParseLine(const std::string &strIn,
					DateTimeRec &out1,
					VelocityRec &out2);
bool ParseLine(const std::string &strIn,
			   DateTimeRec &out1,
			   std::string &out2,
			   std::string &out3);
bool ParseLine(std::istringstream &lineStream,
			   VelocityRec &velOut);

CHARPTR RemoveTrailingSpaces(CHARPTR s);
void RemoveLeadingTrailingQuotes(CHARPTR s);
void RemoveLeadingTrailingWhiteSpaceHelper(CHARPTR s,Boolean removeLeading,Boolean removeTrailing);
void RemoveLeadingWhiteSpace(CHARPTR s);
void RemoveTrailingWhiteSpace(CHARPTR s);
void RemoveLeadingAndTrailingWhiteSpace(CHARPTR s);
long CountSetInString(CHARPTR s, CHARPTR set, CHARPTR stop);
long CountSetInString(CHARPTR s, CHARPTR set);
void RemoveSetFromString(CHARPTR strin, CHARPTR set, CHARPTR strout);
CHARPTR AfterSetInString(CHARPTR s, CHARPTR set, long count);
CHARPTR StringSubstitute(CHARPTR s, char old, char newC);
CHARPTR strnzcpy(CHARPTR to, CHARPTR from, short n);
CHARPTR strnztrimcpy(CHARPTR to, CHARPTR from, short n);
long antol(CHARPTR s, short n);
CHARPTR strtrimcpy(CHARPTR to, CHARPTR from);
char mytoupper(char c);
CHARPTR StrToUpper(CHARPTR s);
char mytolower(char c);
CHARPTR StrToLower(CHARPTR s);
char *strstrnocase(const char *s1, const char *s2);
short strcmpnocase(const char *s1, const char *s2);
short strncmpnocase(const char *s1, const char *s2, short n);
short strnblankcmp(CHARPTR s1, CHARPTR s2, short n);
Boolean strcmptoreturn(CHARPTR s1, CHARPTR s2);
Boolean strcmptoreturnnocase(CHARPTR s1, CHARPTR s2);
CHARPTR mypstrcat(CHARPTR dest, CHARPTR src);
CHARPTR mypstrcpy(CHARPTR dest, CHARPTR src);
void mypstrcatJM(void* dest_asVoidPtr, void* src_asVoidPtr);
void mypstrcpyJM(void* dest_asVoidPtr, void* src_asVoidPtr);
long NumLinesInText(CHARPTR text);
CHARPTR NthLineInTextHelper(CHARPTR text, long n, CHARPTR line, Boolean optimize, long maxLen);
CHARPTR NthLineInTextOptimized(CHARPTR text, long n, CHARPTR line, long maxLen);
CHARPTR NthLineInTextNonOptimized(CHARPTR text, long n, CHARPTR line, long maxLen);
CHARPTR LineInTextHelper(CHARPTR text, CHARPTR line, Boolean caseSensitive);
CHARPTR LineInText(CHARPTR text, CHARPTR line);
CHARPTR LineInTextNoCase(CHARPTR text, CHARPTR line);
CHARPTR AddLineToText(CHARPTR text, CHARPTR line);
CHARPTR IntersectLinesInText(CHARPTR text1, CHARPTR text2, CHARPTR intersection);
CHARPTR strcpyToDelimeter(CHARPTR target, CHARPTR source, char delimiter);
CHARPTR strcpyWithDelimeter(CHARPTR target, CHARPTR source, char delimiter);
Boolean IsDirectionChar(char ch);
Boolean ForceStringNumberHelper(CHARPTR s,Boolean allowNegative,Boolean allowDecimal,Boolean allowDirectionChars);
Boolean ForceStringNumber(CHARPTR s);
Boolean ForceStringNumberAllowingNegative(CHARPTR s);
Boolean DecForceStringNumber(CHARPTR s);
Boolean DecForceStringNumberAllowingNegative(CHARPTR s);
Boolean DecForceStringDirection(CHARPTR s);

// ..

void Secs2DateStrings(unsigned long seconds, CHARPTR dateLong, CHARPTR dateShort, CHARPTR time24, CHARPTR time12);
void Secs2DateString(unsigned long seconds, CHARPTR s);
void Secs2DateString2(unsigned long seconds, CHARPTR s);
void Secs2DateStringNetCDF(unsigned long seconds, CHARPTR s);
unsigned long DateString2Secs(CHARPTR s);
char *Date2String(DateTimeRec *time, char *s);
char *Date2KmlString(DateTimeRec *time, char *s);
void SplitPathFile(CHARPTR fullPath, CHARPTR fileName);
void SplitPathFileName(CHARPTR fullPath, CHARPTR fileName);

void my_p2cstr(void *string);
void my_c2pstr(void *string);

Seconds RoundDateSeconds(Seconds timeInSeconds);

#ifndef MAC
void DLL_API DateToSeconds(DateTimeRec *date, Seconds *seconds);
void GetDateTime(Seconds *seconds);
void DLL_API SecondsToDate(Seconds seconds, DateTimeRec *date);
#endif

Boolean IsWindowsPath(char* path);
Boolean IsUnixPath(char* path);
Boolean IsClassicPath(char* path);
Boolean IsUnixAbsolutePath(char* path);
Boolean IsWindowsAbsolutePath(char* path);
Boolean IsClassicAbsolutePath(char* path);
Boolean IsFullPath(char* path);
Boolean ConvertIfClassicPath(char* path, char* unixPath);

std::ios::pos_type FileSize(std::fstream &file);
std::istream& safeGetLine(std::istream& is, std::string& t);
bool ReadLinesInFile(const string &name, vector<std::string> &stringList, size_t linesToRead = 0);
bool ReadLinesInFile(const char *name, std::vector<string> &stringList, size_t linesToRead = 0);
bool ReadLinesInBuffer(CHARH fileBufH, vector<string> &stringList, size_t linesToRead = 0);
void ConvertDriveLetterToUnixStyle(string &pathPart);
void SplitPathIntoDirAndFile(string &path, string &dir, string &file);
void ConvertPathToCurrentPlatform(string inputPath);
bool ResolvePath(string &containingDir, string &pathToResolve);

#endif
