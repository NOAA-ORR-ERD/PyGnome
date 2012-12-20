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

#include "Basics.h"
#include "TypeDefs.h"
#include "MemUtils.h"
#include "ExportSymbols.h"

char* lfFix(char* str);
OSErr StringToDouble(char* str,double* val);
void StringWithoutTrailingZeros(char* str,double val,short maxNumDecimalPlaces); //JLM
void ChopEndZeros(CHARPTR cString); //JLM
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
char *strstrnocase(CHARPTR s1, CHARPTR s2);
short strcmpnocase(CHARPTR s1, CHARPTR s2);
short strncmpnocase(CHARPTR s1, CHARPTR s2, short n);
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
void SplitPathFile(CHARPTR fullPath, CHARPTR fileName);
void SplitPathFileName(CHARPTR fullPath, CHARPTR fileName);

void my_p2cstr(void *string);
void my_c2pstr(void *string);

Seconds RoundDateSeconds(Seconds timeInSeconds);

#ifndef MAC
void GNOMEDLL_API DateToSeconds(DateTimeRec *date, unsigned long *seconds);
void GetDateTime(unsigned long *seconds);
void GNOMEDLL_API SecondsToDate(unsigned long seconds, DateTimeRec *date);
#endif

Boolean IsWindowsPath(char* path);
Boolean IsUnixPath(char* path);
Boolean IsClassicPath(char* path);
Boolean IsUnixAbsolutePath(char* path);
Boolean IsWindowsAbsolutePath(char* path);
Boolean IsClassicAbsolutePath(char* path);
Boolean IsFullPath(char* path);
Boolean ConvertIfClassicPath(char* path, char* unixPath);

#endif
