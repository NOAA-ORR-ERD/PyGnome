/*
 *  StringFunctions.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 1/11/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "Basics.h"
#include "TypeDefs.h"
#include "MemUtils.h"
#include "StringFunctions.h"
#include <iostream>
#include <time.h>

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

Boolean gUseColonIn24HrTime = true; // JLM likes the colon

char* lfFix(char* str)
{
	// Make the compiler happy by using either lf or Lf
	// MPW wants lf
	// CodeWarrior wants Lf - latest wants lf
	// IBM uses Lf
	short i, len = strlen(str);
	//#ifdef MPW
	// latest IBM seems to want lf
	//#ifdef MAC
	char desiredChar = 'l', badChar = 'L';
	//#else
	//char desiredChar = 'L', badChar = 'l';
	//#endif
	for(i = 0; i < len-2; i++)
	{
		if(str[i] == '%' && str[i+1] == badChar && str[i+2] == 'f')
			str[i+1] = desiredChar; 
	}	
	return str;
}


OSErr StringToDouble(char* str,double* val)
{	// returns error if function fails
	// i.e string does not represent a number
	//
	// use of this function is MPW compiler compatible
	double localVal;
	char localStr[64];
	*val = 0;// Note: always return value of 0 if there is an error
	if(strlen(str) > 63) return -1;
	strcpy(localStr,str);
	RemoveTrailingWhiteSpace(localStr);
	// code goes here, check str is a valid number
	if(!str[0]) return -1; // it was just white space
	
	//#ifdef MPW
#ifdef MAC
	long numScanned = sscanf(localStr,"%lf",&localVal);
#else
#ifndef pyGNOME
	long numScanned = sscanf(localStr,"%Lf",&localVal);
#else	//AH 03/20/2012 (this must be temporary)
	long numScanned = sscanf(localStr,"%lf",&localVal);
#endif
#endif
	if(numScanned == 1) 
	{
		*val = localVal;
		return noErr;
	}
	return -1;
}


void StringWithoutTrailingZeros(char* str,double val,short maxNumDecimalPlaces) //JLM
{
	//char* format = "%.*Lf";
	char format[32];
	//strcpy(format,"%.*Lf");
	strcpy(format,"%.*lf");
	//#ifdef MPW
	//#ifdef MAC	// latest code warrior likes small 'L'
	//format[3] = 'l'; // MPW likes a small 'L", code warrior likes a big 'L", IBM doesn't care
	//#endif
	if(maxNumDecimalPlaces > 9 )  maxNumDecimalPlaces = 9;
	if(maxNumDecimalPlaces < 0 )  maxNumDecimalPlaces = 0;
	format[2] = '0'+maxNumDecimalPlaces;
	sprintf(str,format,val);
	ChopEndZeros(str);
}



void ChopEndZeros(CHARPTR cString) //JLM
{
	///////////////////////////////////////////////////////////////////////////
	//	The function ChopEndZeros takes in cString and chops off the trailing
	//	zeroes after the decimal point. It removes the decimal point if approproate.
	//	Modified to handle exponential notation.
	////////////////////////////////////////////////////////////////////////
	int		i,index, length;
	long		decPtIndex = -99;	// Use -99 as a flag for no decimal point.
	Boolean	hasExponent= false;	/* initialize to false */
	char		exponentStr[56];
	
	length = strlen(cString);
	
	/* Find the index of cString[i] that holds the decimal point. */
	for (i=0; i<length; i++)
	{
		if (cString[i] == '.')
		{
			/* We have a decimal point. */
			decPtIndex = i;	
			break;
			// Consider the unlikely event that we have more than 1 decimal point. 
		}
		else
		{
			/* We have a number; continue with loop. */
		}
	}
	if (decPtIndex == -99)
	{
		/* There is no decimal point; return cString as it is. */
		return;	
	}
	
	
	/********************************************/	
	/* next we take off the exponent (if any).	*/
	/********************************************/	
	for (i=decPtIndex; i<length && hasExponent== false; i++)
	{
		if (cString[i] == 'e' ||cString[i] == 'E')
		{
			/* we have found the exponent string */
			hasExponent= true;
			/* copy the string from this point on into  exponentStr */
			strcpy(exponentStr,&cString[i]);
			/* then cut the exponent from cString */
			cString[i] = 0;	/* null terminate */
		}
	}
	
	/* Starting with the last (length-th) decimal place in the string, check if */
	/* that end character is a '0' or '.'.   */
	length = strlen(cString);	/* reset the string length */
	for (index = length - 1; index >= decPtIndex; index = index-1)
	{
		if (cString[index] == '0' || cString[index] == '.')
		{
			if (cString[index] == '.')
			{
				cString[index] = 0;	/* Change to null & break out of loop. */
				break;		
			}	
			/* Else it's equal to zero.	*/		
			cString[index] = 0;		/* Change to null & keep going in loop. */
		}
		else
		{
			/* We have a number (i.e., a non-0 and non-decPt). */
			break;
		}		
	}
	if (strlen(cString) == 0)
	{ 
		/* We got rid of the whole string (e.g., input was ".000"). */
		cString[0] = '0';
		cString[1] = 0; 	
	}
	else if (hasExponent)
	{
		/***********************************************/	
		/* then we need to replace the exponent string */
		/***********************************************/	
		strcat(cString,exponentStr);
	}
	
	return;
	
}

CHARPTR RemoveTrailingSpaces(CHARPTR s)
{
	char *p;
	
	if (!s[0]) return s;
	
	p = s + strlen(s) - 1;
	
	while (p >= s && (*p == ' ' || *p == LINEFEED || *p == RETURN)) *(p--) = 0;
	
	return s;
}

void RemoveLeadingTrailingQuotes(CHARPTR s)
{
	char *p = s;
	long len;
	if (!s[0]) return;
	if (s[0]=='"')
	{
		len = strlen(s);
		if (s[len-1]=='"')
		{
			p++;
			while (*p) {*s=*p; s++;p++;}
			*(s-1) = 0;
		}
		else return;	// if no trailing quote it's not quoted
	}
	else return;	//	if no leading quote it's not quoted
}
/////////////////////////////////////////////////
void RemoveLeadingTrailingWhiteSpaceHelper(CHARPTR s,Boolean removeLeading,Boolean removeTrailing)
{
	Boolean foundLeadingWhiteSpace = false;
	char *originalS = s;
	char *p = s;
	if (!s[0]) return ;
	if(removeLeading)
	{
		// move past leading white space
		while (*p == ' ' || *p == '\t' || *p == LINEFEED || *p == RETURN) {foundLeadingWhiteSpace = true;p++;}
		///
		if(foundLeadingWhiteSpace) 
		{	//// do shift left/copy code	
			while (*p) { *s = *p; s++;p++;} 
			*s = 0;// terminate the C-String
			// at this point s points to the null terminator
		}
	}
	////////////
	if(removeTrailing)
	{
		if(!foundLeadingWhiteSpace){
			// we did not do the shift left/copy above
			// so we need to find the end of our string
			// to sync up before going to trailing code
			// i.e, make s point to the null terminator
			s = originalS + strlen(originalS);
		}
		//
		// we need to backup from the char before s until we hit non-white Space
		if(!originalS[0]) return;// this catches the case where s was nothing but white space
		p = s-1;
		while (p >= originalS && (*p == ' ' || *p == '\t' || *p == LINEFEED || *p == RETURN)) *(p--) = 0;
	}
}
/////////////////////////////////////////////////
void RemoveLeadingWhiteSpace(CHARPTR s)
{
	RemoveLeadingTrailingWhiteSpaceHelper(s,true,false);
}
//
void RemoveTrailingWhiteSpace(CHARPTR s)
{
	RemoveLeadingTrailingWhiteSpaceHelper(s,false,true);
}
//
void RemoveLeadingAndTrailingWhiteSpace(CHARPTR s)
{
	RemoveLeadingTrailingWhiteSpaceHelper(s,true,true);
}
/////////////////////////////////////////////////

long CountSetInString(CHARPTR s, CHARPTR set, CHARPTR stop)
{
	long count = 0;
	
	for ( ; *s && (!stop || s <= stop) ; s++)
		if (strchr(set, *s)) count++;
	
	return count;
}

long CountSetInString(CHARPTR s, CHARPTR set)
{
	long count = 0, setLen = 0, i;
	
	setLen = strlen (s);
	
	for (i = 0; *s && i < setLen ; s++, i++)
		if (strchr(set, *s)) count++;
	
	return count;
}

void RemoveSetFromString(CHARPTR strin, CHARPTR set, CHARPTR strout)
{
	long i, j = 0;
	
	for (i = 0 ; strin[i] ; i++)
		if (!strchr(set, strin[i]))
			strout[j++] = strin[i];
	
	strout[j] = 0;
}

// return pointer to character in s occurring after count occurrences of characters from set
CHARPTR AfterSetInString(CHARPTR s, CHARPTR set, long count)
{
	long found = 0;
	
	for ( ; *s && (found < count) ; s++)
		if (strchr(set, *s)) found++;
	
	return s;
}

CHARPTR StringSubstitute(CHARPTR s, char old, char newC)
{
	char *p = s;
	
	for ( ; *p ; p++)
		if (p[0] == old)
			p[0] = newC;
	
	return s;
}

CHARPTR strnzcpy(CHARPTR to, CHARPTR from, short n)
{
	strncpy((char *) to, (const char *) from, (size_t) n);
	to[n] = 0; // terminates the (n+1)'st char
	
	return to;
}

CHARPTR strnztrimcpy(CHARPTR to, CHARPTR from, short n)
{
	while (*from == ' ' && n) { from++; n--; }
	strnzcpy(to, from, n);
	RemoveTrailingSpaces(to);
	
	return to;
}

long antol(CHARPTR s, short n)
{
	char c[256];
	
	strnztrimcpy(c, s, n);
	
	return atol(c);
}

CHARPTR strtrimcpy(CHARPTR to, CHARPTR from)
{
	while (*from == ' ') from++;
	strcpy(to, from);
	RemoveTrailingSpaces(to);
	
	return to;
}

char mytoupper(char c)
{
#ifndef MAC
	static char table[][2] = { { -118, -128 }, { -116, -127 }, { -115, -126 },
		{ -114, -125 }, { -106, -124 }, { -102, -123 },
		{  -97, -122 }, { -120,  -53 }, { -117,  -52 },
		{ -101,  -51 }, {  -49,  -50 }, {  -66,  -82 },
		{  -65,  -81 }, {    0,    0 } };
	short i;
#endif
	
	if (c >= ' ' && c <= 'Z') return c; // leave upper case
	if (c >= 'a' && c <= 'z') return c - 32; // convert to upper case
	if (c == 0) return 0;
	
#ifdef MAC
	//UpperText(&c, 1);
	UppercaseText(&c, 1,smSystemScript);	//smCurrentScript
#else
	for (i = 0; table[i][0] ; i++) // international characters
		if (table[i][0] == c)
			return table[i][1];
#endif
	
	return c;
}

CHARPTR StrToUpper(CHARPTR s)
{
	char *p = s;
	
	for ( ; *p ; p++)
		*p = mytoupper(*p);
	
	return s;
}

char mytolower(char c)
{
#ifndef MAC
	static char table[][2] = { { -118, -128 }, { -116, -127 }, { -115, -126 },
		{ -114, -125 }, { -106, -124 }, { -102, -123 },
		{  -97, -122 }, { -120,  -53 }, { -117,  -52 },
		{ -101,  -51 }, {  -49,  -50 }, {  -66,  -82 },
		{  -65,  -81 }, {    0,    0 } };
	short i;
#endif
	
	if (c >= ' ' && c < 'A') return c; // leave alone
	if (c >= 'A' && c <= 'Z') return c + 32; // convert to lower case
	if (c >= 'a' && c <= 'z') return c ; // leave alone
	if (c == 0) return 0;
	
#ifdef MAC
	//LowerText(&c, 1);
	LowercaseText(&c, 1,smSystemScript);	//smCurrentScript
#else
	for (i = 0; table[i][0] ; i++) // international characters
		if (table[i][1] == c)
			return table[i][0];
#endif
	
	return c;
}

CHARPTR StrToLower(CHARPTR s)
{
	char *p = s;
	
	for ( ; *p ; p++)
		*p = mytolower(*p);
	
	return s;
}

char *strstrnocase(CHARPTR s1, CHARPTR s2)
{
	char *p, *retP, s1up[256], s2up[256];
	
	strcpy(s1up, s1);
	strcpy(s2up, s2);
	
	StrToUpper(s1up);
	StrToUpper(s2up);
	
	p = strstr(s1up, s2up);
	
	if (!p) return 0;
	
	//return (s1 + (p - s1up));
	retP = (s1 + (p - s1up));
	return retP;
}

short strcmpnocase(CHARPTR s1, CHARPTR s2)
{
	/*
	 char s1up[256], s2up[256];
	 
	 strcpy(s1up, s1);
	 strcpy(s2up, s2);
	 
	 StrToUpper(s1up);
	 StrToUpper(s2up);
	 
	 return strcmp(s1up, s2up);
	 */
	short i;
	unsigned char c1, c2;
	
	for (i = 0 ; ; i++) {
		if (!s1[i] && !s2[i]) return 0;
		c1 = (unsigned char)mytoupper(s1[i]);
		c2 = (unsigned char)mytoupper(s2[i]);
		if (c1 > c2) return 1;
		if (c2 > c1) return -1;
	}
	
	return 0;
}

short strncmpnocase(CHARPTR s1, CHARPTR s2, short n)
{
	short i;
	unsigned char c1, c2;
	
	for (i = 0 ; i < n ; i++) {
		if (!s1[i] && !s2[i]) return 0;
		c1 = (unsigned char)mytoupper(s1[i]);
		c2 = (unsigned char)mytoupper(s2[i]);
		if (c1 > c2) return 1;
		if (c2 > c1) return -1;
	}
	
	return 0;
}

short strnblankcmp(CHARPTR s1, CHARPTR s2, short n)
{
	short i;
	
	for (i = 0 ; i < n ; i++) if (s1[i] != ' ') break;
	if (i == n) return 0;
	for (i = 0 ; i < n ; i++) if (s2[i] != ' ') break;
	if (i == n) return 0;
	return strncmp(s1, s2, n);
}

Boolean strcmptoreturn(CHARPTR s1, CHARPTR s2)
{
	for ( ; ; ) {
		if (*s1 == RETURN || *s1 == 0) return (*s2 == RETURN || *s2 == 0);
		if (*s2 == RETURN || *s2 == 0) return (*s1 == RETURN || *s1 == 0);
		if (*s1 != *s2) return false;
		s1++;
		s2++;
	}
}

Boolean strcmptoreturnnocase(CHARPTR s1, CHARPTR s2)
{
	for ( ; ; ) {
		if (*s1 == RETURN || *s1 == 0) return (*s2 == RETURN || *s2 == 0);
		if (*s2 == RETURN || *s2 == 0) return (*s1 == RETURN || *s1 == 0);
		if (toupper(*s1) != toupper(*s2)) return FALSE;
		s1++;
		s2++;
	}
}

CHARPTR mypstrcat(CHARPTR dest, CHARPTR src)
{
	long length = _min(((unsigned char)*src), 250 - ((unsigned char)*dest));
	
	_BlockMove(src + 1, dest + *dest + 1, length);
	*dest += length;
	
	return dest;
}

CHARPTR mypstrcpy(CHARPTR dest, CHARPTR src)
{
	_BlockMove(src, dest, ((unsigned char)*src) + 1);
	
	return dest;
}

void mypstrcatJM(void* dest_asVoidPtr, void* src_asVoidPtr)
{
	unsigned char* dest = (unsigned char*)dest_asVoidPtr;
	unsigned char* src = (unsigned char*)src_asVoidPtr;
	long length = _min((*src), 255 - (*dest));
	
	_MyBlockMove(src + 1, dest + (*dest) + 1, length);
	*dest += length;
}

void mypstrcpyJM(void* dest_asVoidPtr, void* src_asVoidPtr)
{
	unsigned char* dest = (unsigned char*)dest_asVoidPtr;
	unsigned char* src = (unsigned char*)src_asVoidPtr;
	_MyBlockMove(src, dest, *src + 1);
}

long NumLinesInText(CHARPTR text)
{
	long count = 1, i = 0;
	
	while (text[i]) {
		// count LINEFEEDs and RETURNs that are not part of a RETURN-LINEFEED sequence
		// don't count terminating newline sequence as a separate line
		if (text[i] == LINEFEED && text[i + 1] != 0)
			count++;
		if (text[i] == RETURN && text[i + 1] != 0 && text[i + 1] != LINEFEED)
			count++;
		i++;
	}
	
	return count;
}

CHARPTR NthLineInTextHelper(CHARPTR text, long n, CHARPTR line, Boolean optimize, long maxLen)
{	// this function now handles MAC, IBM and UNIX style lines. -- JLM 12/1/00
	Boolean lineFeed = FALSE;
	char *s, *q;
	long count = 1, i = 0;
	static long linesRead;
	static CHARPTR t, p;
	long numCharCopied = 0;
	long lineLengthInFile = 0;
	
	if(maxLen < 0) maxLen = 0;
	
	if (optimize && n == 0) {
		t = text;
		linesRead = 0;
	}
	
	p = optimize ? t : text;
	
	if (optimize) n -= linesRead;
	
	if(maxLen > 0)
		line[0] = 0;
	
	while (p[i]) 
	{
		if ((count - 1) == n) 
		{
			s = &p[i];
			q = line;
			while (s[0] && s[0] != RETURN && s[0] != LINEFEED)
			{ 	// this copies to the variable
				lineLengthInFile++; // keep track of the chars even though we may not put them in the string
				if(numCharCopied < maxLen) {
					q[0] = s[0]; 
					numCharCopied++;
				}
				q++; 
				s++; 
			}
			if (s[0] == RETURN && s[1] == LINEFEED)
				lineFeed = TRUE;
			*q = 0;	// this copies to the variable
			line[maxLen-1] = 0; // always set the last char to 0
			break;
		}
		if (p[i] == LINEFEED && p[i + 1] != 0)
			count++;
		if (p[i] == RETURN && p[i + 1] != 0 && p[i + 1] != LINEFEED)
			count++;
		i++;
	}
	
	if (optimize) 
	{
		t += i + strlen(line) + 1 + (lineFeed ? 1 : 0);
		linesRead++;
	}
	
	return line;
}

CHARPTR NthLineInTextOptimized(CHARPTR text, long n, CHARPTR line, long maxLen)
{
	return NthLineInTextHelper(text, n, line, TRUE, maxLen);
}

CHARPTR NthLineInTextNonOptimized(CHARPTR text, long n, CHARPTR line, long maxLen)
{
	return NthLineInTextHelper(text, n, line, FALSE, maxLen);
}

// is null-terminated line one of the lines in return-delimited, null-terminated text?

CHARPTR LineInTextHelper(CHARPTR text, CHARPTR line, Boolean caseSensitive)
{
	short lineLength = strlen(line);
	long textLength = strlen(text);
	
	while (textLength >= lineLength) {
		if (caseSensitive)
		{ if (strcmptoreturn(text, line)) return text; }
		else
		{ if (strcmptoreturnnocase(text, line)) return text; }
		while (*text && *text != RETURN) { text++; textLength--; }
		if (*text == RETURN) { text++; textLength--; }
		if (*text == LINEFEED) { text++; textLength--; }
	}
	
	return nil;
}

CHARPTR LineInText(CHARPTR text, CHARPTR line)
{
	return LineInTextHelper(text, line, TRUE);
}

CHARPTR LineInTextNoCase(CHARPTR text, CHARPTR line)
{
	return LineInTextHelper(text, line, FALSE);
}

CHARPTR AddLineToText(CHARPTR text, CHARPTR line)
{
	char r[2] = "r";
	static CHARPTR t;
	
	if (!text[0]) t = text;
	
	strcpy(t, line);
	r[0] = RETURN;
	strcat(t, r);
#ifdef IBM
	strcat(t, "\n");
#endif
	t = &t[strlen(t)];
	
	return text;
}

CHARPTR IntersectLinesInText(CHARPTR text1, CHARPTR text2, CHARPTR intersection)
{
	long i, count = NumLinesInText(text1);
	char line[256];
	
	intersection[0] = 0;
	
	for (i = 0 ; i < count ; i++) {
		NthLineInTextOptimized(text1, i, line, 256);
		if (LineInText(text2, line))
			AddLineToText(intersection, line);
	}
	
	return intersection;
}

CHARPTR strcpyToDelimeter(CHARPTR target, CHARPTR source, char delimiter)
{
	char *save = target;
	
	while (*source && *source != delimiter) {
		*target = *source;
		target++;
		source++;
	}
	*target = 0;
	
	return save;
}

CHARPTR strcpyWithDelimeter(CHARPTR target, CHARPTR source, char delimiter)
{
	char *save = target;
	
	while (*source != delimiter) {
		*target = *source;
		target++;
		source++;
	}
	*target = *source;
	
	return save;
}

Boolean IsDirectionChar(char ch)
{	// note this is not international friendly
	switch(ch) 
	{
		case 'n': case 'N':
		case 'e': case 'E':
		case 's': case 'S':
		case 'w': case 'W':
			return true;
	}
	return false;
}

// modify string if it does not match the desired format
Boolean ForceStringNumberHelper(CHARPTR s,Boolean allowNegative,Boolean allowDecimal,Boolean allowDirectionChars)
{	// JLM
	// Returns TRUE if the string has been changed
	Boolean error = FALSE;
	short i, j, n = strlen(s), numDec = 0,numNeg = 0;
	Boolean allowNumbers = true;
	
	if(allowDirectionChars)
	{
		// "NNE" is allowed
		// BUT !! if the first char is a letter then only letters are allowed 
		// if the first char is a number only numbers are allowed
		//
		// find first valid char
		for (i = 0 ; i < n ; i++)
		{
			if (	('0' <= s[i] && s[i] <= '9') 
				|| (allowDecimal && s[i] == '.') )
			{
				allowDirectionChars = false;
				break;
			}
			if(IsDirectionChar(s[i])) 
			{
				allowNumbers = false;
				break;
			}
		}
	}
	
	if (n > 11) { error = TRUE; n = 11; }
	
	for (i = 0, j = 0 ; i < n ; i++) 
	{
		if(allowDirectionChars && IsDirectionChar(s[i])) s[j++] = s[i];
		else if(allowNumbers && '0' <= s[i] && s[i] <= '9') s[j++] = s[i];
		else if(allowNumbers && allowDecimal && s[i] == '.')
		{
			if (++numDec > 1) error = true;
			else s[j++] = s[i];
		}
		else if(allowNumbers && allowNegative && s[i] == '-')
		{
			if (++numNeg > 1) error = true;
			else s[j++] = s[i];
		}
		else error = true;
	}
	s[j] = 0;
	if (j == 0) {
		if(allowDecimal) strcpy(s, "0.0");
		else strcpy(s, "0");
		error = TRUE;
	}
	
	if (error) { SysBeep(1); return TRUE; } // we changed the string
	
	return FALSE;
}

// beep and modify string if it contains non-digits
Boolean ForceStringNumber(CHARPTR s)
{	//JLM
	Boolean allowNegativeFlag = false;
	Boolean allowDecimalFlag = false;
	Boolean allowDirectionChars = false;
	return ForceStringNumberHelper(s,allowNegativeFlag,allowDecimalFlag,allowDirectionChars);
}

Boolean ForceStringNumberAllowingNegative(CHARPTR s)
{	//JLM
	Boolean allowNegativeFlag = true;
	Boolean allowDecimalFlag = false;
	Boolean allowDirectionChars = false;
	return ForceStringNumberHelper(s,allowNegativeFlag,allowDecimalFlag,allowDirectionChars);
}

// same as ForceStringNumber() but allow a decimal
Boolean DecForceStringNumber(CHARPTR s)
{	//JLM
	Boolean allowNegativeFlag = false;
	Boolean allowDecimalFlag = true;
	Boolean allowDirectionChars = false;
	return ForceStringNumberHelper(s,allowNegativeFlag,allowDecimalFlag,allowDirectionChars);
}

Boolean DecForceStringNumberAllowingNegative(CHARPTR s)
{	//JLM
	Boolean allowNegativeFlag = true;
	Boolean allowDecimalFlag = true;
	Boolean allowDirectionChars = false;
	return ForceStringNumberHelper(s,allowNegativeFlag,allowDecimalFlag,allowDirectionChars);
}

Boolean DecForceStringDirection(CHARPTR s)
{	//JLM
	Boolean allowNegativeFlag = false; // 12/7/98 they don't want negative directions
	Boolean allowDecimalFlag = false;
	Boolean allowDirectionChars = true;
	return ForceStringNumberHelper(s,allowNegativeFlag,allowDecimalFlag,allowDirectionChars);
}

void Secs2DateStrings(unsigned long seconds,
					  CHARPTR dateLong, CHARPTR dateShort,
					  CHARPTR time24, CHARPTR time12)
#ifdef MAC
{
	Intl0Hndl iZero;
	Intl1Hndl iOne;
	Intl0Rec saveZero;
	Intl1Rec saveOne;
	
	iZero = (Intl0Hndl) GetIntlResource(0);
	iOne  = (Intl1Hndl) GetIntlResource(1);
	
	if (!iZero || !iOne) { SysBeep(1); return; }
	
	saveZero = (**iZero);
	saveOne = (**iOne);
	
	(**iOne).suppressDay = 255; // suppress day
	//if (dateLong) IUDatePString(seconds, longDate, (StringPtr)dateLong, (Handle)iOne);
	if (dateLong) DateString(seconds, longDate, (StringPtr)dateLong, (Handle)iOne);
	
	(**iZero).shrtDateFmt |= dayLdingZ; // use leading zeros
	(**iZero).shrtDateFmt &= ~century; // suppress century
	//if (dateShort) IUDatePString(seconds, shortDate, (StringPtr)dateShort, (Handle)iZero);
	if (dateShort) DateString(seconds, shortDate, (StringPtr)dateShort, (Handle)iZero);
	
	//if (time12) IUTimeString(seconds, FALSE, (StringPtr)time12);
	if (time12) TimeString(seconds, FALSE, (StringPtr)time12, (Handle)iZero);
	
	(**iZero).timeCycle = 0; // 24 hour
	(**iZero).timeFmt |= hrLeadingZ | minLeadingZ; // use leading zeros
	
	if(gUseColonIn24HrTime) (**iZero).timeSep = ':';
	else  (**iZero).timeSep = 0; // no ':'
	
	
	//if (time24) IUTimePString(seconds, FALSE, (StringPtr)time24, (Handle)iZero);
	if (time24) TimeString(seconds, FALSE, (StringPtr)time24, (Handle)iZero);
	
	(**iZero) = saveZero;
	(**iOne) = saveOne;
	
	if (dateLong) my_p2cstr((StringPtr)dateLong);
	if (dateShort) my_p2cstr((StringPtr)dateShort);
	if (time24) my_p2cstr((StringPtr)time24);
	if (time12) my_p2cstr((StringPtr)time12);
}
#else
{
	struct tm *time;
#ifndef pyGNOME
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
#endif
	time = localtime((long *)&seconds); // gmtime(&seconds);
	if (!time) { SysBeep(1); return; }
	
	if (time->tm_isdst == 1)
	{ 	// we want to show standard time so we need to fake out the daylight savings time
		if(time->tm_hour > 0)
			time->tm_hour--;
		else
		{	
			seconds -= 3600;
			time = localtime((long *)&seconds);
			if(time->tm_isdst == 1) {
				// life is good, both times were daylight savings time
			}
			else
				printError("Programmer error in Secs2DateStrings");
				}
	}
	
	if (dateLong) strftime(dateLong, 32, "%B %d, %Y", time);
		if (dateShort) strftime(dateShort, 32, "%m/%d/%y", time);
			if (time24) 
			{
				if(gUseColonIn24HrTime) strftime(time24, 32, "%H:%M", time);
					else strftime(time24, 32, "%H%M", time);
						}
	if (time12) strftime(time12, 32, "%#I:%M %p", time);
		}
#endif

void Secs2DateString(unsigned long seconds, CHARPTR s)
#ifdef MAC
{
	DateTimeRec DTR;
	
	SecondsToDate (seconds, &DTR);
	DTR.year = DTR.year %100;// year 2000 fix , JLM 1/25/99
	sprintf(s, "%02hd/%02hd/%02hd", DTR.month, DTR.day, DTR.year);
}
#else
{
	struct tm *newTime;
#ifndef pyGNOME
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
#endif
	newTime = localtime((long *)&seconds); // gmtime(&seconds)
	newTime->tm_year = newTime->tm_year %100;// year 2000 fix , JLM 1/25/99
	if (newTime)
	{
		if (newTime->tm_isdst == 1)
		{ 	// we want to show standard time so we need to fake out the daylight savings time
			if(newTime->tm_hour > 0)
				newTime->tm_hour--;
			else
			{	
				seconds -= 3600;
				newTime = localtime((long *)&seconds);
				if(newTime->tm_isdst == 1) {
					newTime->tm_year = newTime->tm_year %100;// year 2000 fix , JLM 1/25/99
					// life is good, both times were daylight savings time
				}
				else
					printError("Programmer error in Secs2DateString");
					}
		}
		sprintf(s, "%02ld/%02ld/%02ld", (long)newTime->tm_mon + 1, (long)newTime->tm_mday, (long)newTime->tm_year);
	}
	else
	{ strcpy(s, "???"); SysBeep(1); }
}
#endif

void Secs2DateString2(unsigned long seconds, CHARPTR s)
{ // returns in format: January 2, 1998 
	short day ; //1-31
	short month; // 1-12
	short year4; // 4 digit year
	short hour; // 0-23
	short minute;// 0-59 //
	char str[255];
#ifdef MAC
	DateTimeRec DTR;
	SecondsToDate (seconds, &DTR);
	month = DTR.month;
	day = DTR.day;
	year4 = DTR.year;
	hour = DTR.hour;
	minute = DTR.minute;
#else
	// IBM
	struct tm *newTime;
#ifndef pyGNOME
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
#endif
	newTime = localtime((long *)&seconds); // gmtime(&seconds)
	if (newTime)
	{
		if (newTime->tm_isdst == 1)
		{	// we want to show standard time so we need to fake out the daylight savings time
			if(newTime->tm_hour > 0)
				newTime->tm_hour--;
			else
			{	
				seconds -= 3600;
				newTime = localtime((long *)&seconds);
				if(newTime->tm_isdst == 1) {
					// life is good, both times were daylight savings time
				}
				else
					printError("Programmer error in Secs2DateStrings2");
			}
		}
		
		month = newTime->tm_mon + 1;
		day = newTime->tm_mday;
		year4 = newTime->tm_year+1900; // year 2000 OK
		hour = newTime->tm_hour;
		minute = newTime->tm_min;
	}
	else
	{ strcpy(s, "???"); return; }
	
#endif
	switch (month)
	{
		case  1: strcpy (s, "January "); break;
		case  2: strcpy (s, "February "); break;
		case  3: strcpy (s, "March "); break;
		case  4: strcpy (s, "April "); break;
		case  5: strcpy (s, "May "); break;
		case  6: strcpy (s, "June "); break;
		case  7: strcpy (s, "July "); break;
		case  8: strcpy (s, "August "); break;
		case  9: strcpy (s, "September "); break;
		case 10: strcpy (s, "October "); break;
		case 11: strcpy (s, "November "); break;
		case 12: strcpy (s, "December "); break;
	}
	
	sprintf(str, "%02hd, %hd ", day, year4);
	strcat (s, str);
	sprintf (str, "%2.2d:%2.2d", hour, minute);
	strcat (s, str);
}

void Secs2DateStringNetCDF(unsigned long seconds, CHARPTR s)
{ // returns in format: 2012-08-23 08:31:00 
	short day ; //1-31
	short month; // 1-12
	short year4; // 4 digit year
	short hour; // 0-23
	short minute;// 0-59 //
	short second = 0;
	char str[255];
#ifdef MAC
	DateTimeRec DTR;
	SecondsToDate (seconds, &DTR);
	month = DTR.month;
	day = DTR.day;
	year4 = DTR.year;
	hour = DTR.hour;
	minute = DTR.minute;
#else
	// IBM
	struct tm *newTime;
#ifndef pyGNOME
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
#endif
	newTime = localtime((long *)&seconds); // gmtime(&seconds)
	if (newTime)
	{
		if (newTime->tm_isdst == 1)
		{	// we want to show standard time so we need to fake out the daylight savings time
			if(newTime->tm_hour > 0)
				newTime->tm_hour--;
			else
			{	
				seconds -= 3600;
				newTime = localtime((long *)&seconds);
				if(newTime->tm_isdst == 1) {
					// life is good, both times were daylight savings time
				}
				else
					printError("Programmer error in Secs2DateStringNetCDF");
			}
		}
		
		month = newTime->tm_mon + 1;
		day = newTime->tm_mday;
		year4 = newTime->tm_year+1900; // year 2000 OK
		hour = newTime->tm_hour;
		minute = newTime->tm_min;
	}
	else
	{ strcpy(s, "???"); return; }
	
#endif
	
	sprintf(str, "%hd-%02hd-%02hd ", year4, month, day);
	strcpy (s, str);
	sprintf (str, "%2.2d:%2.2d:%2.2d", hour, minute, second);
	strcat (s, str);
}

unsigned long DateString2Secs(CHARPTR s)
#ifdef MAC
{
	DateTimeRec DTR;
	unsigned long seconds;
	
	sscanf(s, "%hd/%hd/%hd", &DTR.month, &DTR.day, &DTR.year);
	if(DTR.year < 40) DTR.year += 2000;// year 2000 solution, JLM 1/25/99
	if (DTR.year < 200) DTR.year += 1900;
	DTR.hour = DTR.minute = DTR.second = 0;
	
	DateToSeconds (&DTR, &seconds);
	
	return seconds;
}
#else
{
	short month, day, year;
	unsigned long seconds;
	long long_secs;
	struct tm newTime;
	
	sscanf(s, "%hd/%hd/%hd", &month, &day, &year);
	newTime.tm_sec = newTime.tm_min = newTime.tm_hour = 0;
	newTime.tm_mday = day;
	newTime.tm_mon = month - 1;
	if(year < 40) year+= 100;// year 2000 solution, JLM 1/25/99
		newTime.tm_year = (year < 200) ? year : year - 1900;
		//newTime.tm_isdst = -1; // let mktime() determine if it's daylight savings time
		newTime.tm_isdst = 0; // use standard time to avoid duplicates at changeover 4/12/01
		
		long_secs = mktime(&newTime);
		//seconds = mktime(&newTime);
		//if (seconds == -1) SysBeep(1);
		if (long_secs==-1) 
		{
			long_secs=0; 
			printNote("The Windows function localtime() does not accept dates earlier than January 1,  00:00 GMT");
			SysBeep(1);
		}	// here check if year < 1970 and convert
	seconds = long_secs;
#ifndef pyGNOME
	seconds += 2082816000L; // convert from seconds since 1970 to seconds since 1904
#endif
	return seconds;
}
#endif

char *Date2String(DateTimeRec *time, char *s)
{
	sprintf(s, "%02hd/%02hd/%02hd %02hd:%02hd:%02hd",
			time->month, time->day, time->year,
			time->hour, time->minute, time->second);
	
	return s;
}

void SplitPathFile(CHARPTR fullPath, CHARPTR fileName)
{
	char *p;
	
	fileName[0] = 0;
	
	if (p = strrchr(fullPath, DIRDELIMITER)) {
		if (p[1] == 0) { // treat final directory as file name to be retrieved
			*p = 0;
			SplitPathFile(fullPath, fileName);
			return;
		}
		p++;
		strcpy(fileName, p);
		*p = 0;
	}
	else {
		strcpy(fileName, fullPath);
		fullPath[0] = 0;
	}
}



void my_p2cstr(void *string)
{
	short i, len;
	char *s = (char*) string;
	
	if(!s) return;
	
	len = (unsigned char)s[0];
	for (i = 0 ; i < len ; i++)
		s[i] = s[i + 1];
	s[len] = 0;
	
}

void my_c2pstr(void *string)
{
	short i, len;
	char *s = (char*) string;
	
	if(!s) return;
	
	len = strlen(s);
	if (len > 255) len = 255;
	for (i = len ; i > 0 ; i--)
		s[i] = s[i - 1];
	s[0] = (unsigned char)len;
}



#ifndef MAC

void GetDateTime(unsigned long *seconds)
{
	unsigned long secs;
	
	time((long *)&secs); // time() wants a near pointer; returns secs since 1/1/70
#ifndef pyGNOME
	secs += 2082816000L; // convert to seconds since 1904
#endif
	// would be secs += 126230400 if time returned time since 1900, as it says in the book
	*seconds = secs;
}

void SecondsToDate (unsigned long seconds, DateTimeRec *date)
{
	struct tm *newTime;
#ifndef pyGNOME
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
#endif
	// note time_t is now 64 bit by default on Windows, preprocessor definition for 32 bit (good til 2038)
	newTime = localtime((long *)&seconds); // gmtime(&seconds)
	if (newTime) {
		if (newTime->tm_isdst == 1)
		{	// we want to show standard time so we need to fake out the daylight savings time
			if(newTime->tm_hour > 0)
				newTime->tm_hour--;
			else
			{	
				seconds -= 3600;
				newTime = localtime((long *)&seconds);
				if(newTime->tm_isdst == 1) {
					newTime->tm_year = newTime->tm_year %100;// year 2000 fix , JLM 1/25/99
					// life is good, both times were daylight savings time
				}
				else
					printError("Programmer error in Secs2DateStrings");
			}
		}
		
		date->year = newTime->tm_year;
		// this mimics the mac function which has a 4 digit value in the time field
		if(date->year<40)date->year += 2000;
		else if(date->year<200) date->year += 1900;
		// code ges here JLM , does localtime have a year 2000 issue ?
		date->month = newTime->tm_mon + 1;
		date->day = newTime->tm_mday;
		date->hour = newTime->tm_hour;
		date->minute = newTime->tm_min;
		date->second = newTime->tm_sec;
        // JS: Not sure 1 is added to day of the week.
        //     This doesn't seem to effect the date / time value - leave it as is.
        // NOTE:The C++ time struct 0=Sunday and 6=Sat
        //      For Python time struct 0=Monday and 6=Sunday
		date->dayOfWeek = newTime->tm_wday + 1;
	}
	else {
		SysBeep(1);
		date->year = 0;
		date->month = 0;
		date->day = 0;
		date->hour = 0;
		date->minute = 0;
		date->second = 0;
		date->dayOfWeek = 0;
	}
}



void DateToSeconds (DateTimeRec *date, unsigned long *seconds)
{
	char s[100];
	unsigned long secs;
	
	sprintf(s, "%02hd/%02hd/%02hd", date->month, date->day, date->year);
	
	secs = DateString2Secs(s);
	
	secs += date->hour * 3600 + date->minute * 60 + date->second;
	
	(*seconds) = secs;
}
#endif

