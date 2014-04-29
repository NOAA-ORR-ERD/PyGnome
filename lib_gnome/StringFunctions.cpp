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

// JLM, April 2013
// Apparently this function "fixes" the format string
// that is passed into sscanf().  I do not believe
// that this is needed at this time since most
// development environments (Mac OSX, Linux, VS2010)
// are compatible with GNU development standards.
// Basically I believe this function should not be used.
char *lfFix(char *str)
{
	// Make the compiler happy by using either lf or Lf
	// MPW wants lf
	// CodeWarrior wants Lf - latest wants lf
	// IBM uses Lf
	short len = strlen(str);
	char desiredChar = 'l', badChar = 'L';

	for (int i = 0; i < len - 2; i++) {
		if(str[i] == '%' && str[i + 1] == badChar && str[i + 2] == 'f')
			str[i + 1] = desiredChar;
	}

	return str;
}


// returns error if function fails
// i.e string does not represent a number
//
// use of this function is MPW compiler compatible
OSErr StringToDouble(char* str,double* val)
{
	double localVal;
	char localStr[64];

	memset(localStr, 0, 64);

	*val = 0; // Note: always return value of 0 if there is an error

	if (strlen(str) > 63)
		return -1;

	strncpy(localStr, str, 64);
	RemoveTrailingWhiteSpace(localStr);

	// code goes here, check str is a valid number
	if (!str[0])
		return -1; // it was just white space
	
	//#ifdef MPW
#ifdef MAC
	long numScanned = sscanf(localStr, "%lf", &localVal);
#else
#  ifndef pyGNOME
	long numScanned = sscanf(localStr, "%Lf", &localVal);
#  else	//AH 03/20/2012 (this must be temporary)
	long numScanned = sscanf(localStr, "%lf", &localVal);
#  endif
#endif

	if (numScanned == 1) {
		*val = localVal;
		return noErr;
	}

	return -1;
}


void StringWithoutTrailingZeros(char* str,double val,short maxNumDecimalPlaces) //JLM
{
	char format[32];
	strcpy(format, "%.*lf");

	if (maxNumDecimalPlaces > 9 )
		maxNumDecimalPlaces = 9;

	if (maxNumDecimalPlaces < 0 )
		maxNumDecimalPlaces = 0;

	format[2] = '0'+ maxNumDecimalPlaces;
	sprintf(str,format,val);
	ChopEndZeros(str);
}



///////////////////////////////////////////////////////////////////////////
//	The function ChopEndZeros takes in cString and chops off the trailing
//	zeroes after the decimal point. It removes the decimal point if approproate.
//	Modified to handle exponential notation.
////////////////////////////////////////////////////////////////////////
void ChopEndZeros(CHARPTR cString) //JLM
{
	int index, length;
	long decPtIndex = -99; // Use -99 as a flag for no decimal point.
	bool hasExponent = false; /* initialize to false */
	char exponentStr[56];
	
	length = strlen(cString);
	
	/* Find the index of cString[i] that holds the decimal point. */
	for (int i = 0; i < length; i++) {
		if (cString[i] == '.') {
			/* We have a decimal point. */
			decPtIndex = i;
			break;
			// Consider the unlikely event that we have more than 1 decimal point. 
		}
		else {
			/* We have a number; continue with loop. */
		}
	}

	if (decPtIndex == -99) {
		/* There is no decimal point; return cString as it is. */
		return;	
	}

	/********************************************/	
	/* next we take off the exponent (if any).	*/
	/********************************************/	
	for (int i = decPtIndex; i < length && hasExponent == false; i++) {
		if (cString[i] == 'e' || cString[i] == 'E') {
			/* we have found the exponent string */
			hasExponent = true;

			/* copy the string from this point on into  exponentStr */
			strcpy(exponentStr, &cString[i]);

			/* then cut the exponent from cString */
			cString[i] = 0;	/* null terminate */
		}
	}
	
	/* Starting with the last (length-th) decimal place in the string, check if */
	/* that end character is a '0' or '.'.   */
	length = strlen(cString);
	for (index = length - 1; index >= decPtIndex; index = index - 1) {
		if (cString[index] == '0' || cString[index] == '.') {
			if (cString[index] == '.') {
				cString[index] = 0;	/* Change to null & break out of loop. */
				break;		
			}	
			/* Else it's equal to zero.	*/		
			cString[index] = 0; /* Change to null & keep going in loop. */
		}
		else {
			/* We have a number (i.e., a non-0 and non-decPt). */
			break;
		}		
	}

	if (strlen(cString) == 0) {
		/* We got rid of the whole string (e.g., input was ".000"). */
		cString[0] = '0';
		cString[1] = 0; 	
	}
	else if (hasExponent) {
		/***********************************************/	
		/* then we need to replace the exponent string */
		/***********************************************/	
		strcat(cString,exponentStr);
	}
	
	return;
}


////
////  Begin JLM new string handling functions.
////

// trim from start
std::string &ltrim(std::string &s)
{
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}

// trim from end
std::string &rtrim(std::string &s)
{
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

// trim from both ends
std::string &trim(std::string &s)
{
        return ltrim(rtrim(s));
}

std::vector<std::string> &rtrim_empty_lines(std::vector<std::string> &lines)
{
	std::vector<std::string>::iterator i = lines.end() - 1;

	while(i != lines.begin()) {
		if (i->size() == 0)
			//lines.erase(i);	// this just makes the line empty but doesn't remove it
			i = lines.erase(i);
		else
			break;

		i--;
	}

	return lines;
}

vector<string> &split(const string &strIn, const string &delims, vector<string> &elems) {
    std::size_t prev = 0, pos;
    while ((pos = strIn.find_first_of(delims, prev)) != std::string::npos) {
        if (pos == 0)
        	elems.push_back("");
        if (pos > prev)
        	elems.push_back(strIn.substr(prev, pos-prev));
        prev = pos+1;
    }
    if (prev < strIn.length())
    	elems.push_back(strIn.substr(prev, std::string::npos));

    return elems;
}

vector<string> split(const string &s, const string &delims) {
    std::vector<std::string> elems;
    split(s, delims, elems);
    return elems;
}

vector<string> SplitPath(string &path)
{
	return split(path, "\\/");
}

string join(const vector<string> &v, const string &delim) {
    ostringstream s;

    for (std::vector<string>::const_iterator it = v.begin(); it != v.end(); ++it) {
        s << *it;
        if (it + 1 != v.end())
            s << delim;
    }

    return s.str();
}

////
////  End JLM new string handling functions.
////


////
////  Begin JLM new textfile line parsing functions.
////

bool ParseKeyedLine(const string &strIn, const string &key,
					string &out)
{
	string tempKey, value;
	istringstream lineStream(strIn);

	lineStream >> tempKey >> value;
	if (lineStream.fail())
		return false;

	if (tempKey != key)
		return false;

	out = value;
	return true;
}

bool ParseKeyedLine(const string &strIn, const string &key,
					short &out1)
{
	string tempKey;
	short value;
	
	istringstream lineStream(strIn);
	
	lineStream >> tempKey >> value;
	if (lineStream.fail())
		return false;
	
	if (tempKey != key)
		return false;
	
	out1 = value;
	return true;
}

bool ParseKeyedLine(const string &strIn, const string &key,
					long &out1)
{
	string tempKey;
	long value1;

	istringstream lineStream(strIn);
	lineStream >> tempKey >> value1;
	if (lineStream.fail()) {
		cerr << "Parsing string (" << strIn
			 << ") failed." << endl;
		return false;
	}

	if (tempKey != key) {
		//cerr << "string (" << tempKey
			 //<< ") does not match (" << key << ")." << endl;
		return false;
	}

	out1 = value1;
	return true;
}

bool ParseKeyedLine(const string &strIn, const string &key,
					double &out1)
{
	string tempKey;
	double value1;

	istringstream lineStream(strIn);
	lineStream >> tempKey >> value1;
	if (lineStream.fail())
		return false;

	if (tempKey != key)
		return false;

	out1 = value1;
	return true;
}

bool ParseKeyedLine(const string &strIn, const string &key,
					long &out1, long &out2)
{
	string tempKey;
	long value1, value2;

	istringstream lineStream(strIn);
	lineStream >> tempKey >> value1 >> value2;
	if (lineStream.fail())
		return false;

	if (tempKey != key)
		return false;

	out1 = value1;
	out2 = value2;
	return true;
}

bool ParseKeyedLine(const string &strIn, const string &key,
					DateTimeRec &out1)
{
	string tempKey;
	DateTimeRec value1;

	istringstream lineStream(strIn);
	lineStream >> tempKey
			   >> value1.day >> value1.month >> value1.year
			   >> value1.hour >> value1.minute;
	if (lineStream.fail())
		return false;

	if (tempKey != key)
		return false;

	out1.day = value1.day;
	out1.month = value1.month;
	out1.year = value1.year;
	out1.hour = value1.hour;
	out1.minute = value1.minute;

	return true;
}

bool ParseLine(const string &strIn,
			   long &out1)
{
	long value1;

	istringstream lineStream(strIn);
	lineStream >> value1;
	if (lineStream.fail())
		return false;

	out1 = value1;
	return true;
}

bool ParseLine(const string &strIn,
			   string &out1)
{
	string value1;

	istringstream lineStream(strIn);
	lineStream >> value1;
	if (lineStream.fail())
		return false;

	out1 = value1;
	return true;
}

bool ParseLine(const string &strIn,
			   long &out1, double &out2, double &out3)
{
	long value1;
	double value2, value3;

	istringstream lineStream(strIn);
	lineStream >> value1 >> value2 >> value3;
	if (lineStream.fail())
		return false;

	out1 = value1;
	out2 = value2;
	out3 = value3;
	return true;
}

bool ParseLine(const string &strIn,
			   double &out1, double &out2)
{
	double v1, v2;

	istringstream lineStream(strIn);
	lineStream >> v1 >> v2;
	if (lineStream.fail())
		return false;

	out1 = v1;
	out2 = v2;

	return true;
}

bool ParseLine(const string &strIn,
			   double &out1, double &out2, double &out3)
{
	double v1, v2, v3;

	istringstream lineStream(strIn);
	lineStream >> v1 >> v2 >> v3;
	if (lineStream.fail())
		return false;

	out1 = v1;
	out2 = v2;
	out3 = v3;

	return true;
}

bool ParseLine(const string &strIn,
		   	   long &out1, long &out2, long &out3)
{
	long v1, v2, v3;

	istringstream lineStream(strIn);
	lineStream >> v1 >> v2 >> v3;
	if (lineStream.fail())
		return false;

	out1 = v1;
	out2 = v2;
	out3 = v3;

	return true;
}

bool ParseLine(const string &strIn,
			   DAG &out1)
{
	DAG v1;

	istringstream lineStream(strIn);
	lineStream >> v1.topoIndex >> v1.branchLeft >> v1.branchRight;
	if (lineStream.fail())
		return false;

	out1.topoIndex = v1.topoIndex;
	out1.branchLeft = v1.branchLeft;
	out1.branchRight = v1.branchRight;

	return true;
}

bool ParseLine(const string &strIn,
			   Topology &out1)
{
	Topology v1;

	istringstream lineStream(strIn);
	lineStream >> v1.vertex1 >> v1.vertex2 >> v1.vertex3
			   >> v1.adjTri1 >> v1.adjTri2 >> v1.adjTri3;
	if (lineStream.fail())
		return false;

	out1.vertex1 = v1.vertex1;
	out1.vertex2 = v1.vertex2;
	out1.vertex3 = v1.vertex3;

	out1.adjTri1 = v1.adjTri1;
	out1.adjTri2 = v1.adjTri2;
	out1.adjTri3 = v1.adjTri3;

	return true;
}

bool ParseLine(const string &strIn,
			   Topology &out1, VelocityFRec &out2)
{
	Topology v1;
	VelocityFRec v2;

	istringstream lineStream(strIn);
	lineStream >> v1.vertex1 >> v1.vertex2 >> v1.vertex3
			   >> v1.adjTri1 >> v1.adjTri2 >> v1.adjTri3
			   >> v2.u >> v2.v;
	if (lineStream.fail())
		return false;

	out1.vertex1 = v1.vertex1;
	out1.vertex2 = v1.vertex2;
	out1.vertex3 = v1.vertex3;

	out1.adjTri1 = v1.adjTri1;
	out1.adjTri2 = v1.adjTri2;
	out1.adjTri3 = v1.adjTri3;

	out2.u = v2.u;
	out2.v = v2.v;

	return true;
}

bool ParseLine(const string &strIn,
					DateTimeRec &out1,
					VelocityRec &out2)
{
	DateTimeRec value1;
	VelocityRec value2;

	istringstream lineStream(strIn);
	lineStream >> value1.day >> value1.month >> value1.year
			   >> value1.hour >> value1.minute
			   >> value2.u >> value2.v;
	if (lineStream.fail())
		return false;

	out1.day = value1.day;
	out1.month = value1.month;
	out1.year = value1.year;
	out1.hour = value1.hour;
	out1.minute = value1.minute;

	out2.u = value2.u;
	out2.v = value2.v;

	return true;
}

bool ParseLine(const string &strIn,
			   DateTimeRec &out1,
			   string &out2,
			   string &out3)
{
	DateTimeRec value1;
	string value2, value3;
	
	istringstream lineStream(strIn);
	lineStream >> value1.day >> value1.month >> value1.year
	>> value1.hour >> value1.minute
	>> value2 >> value3;
	if (lineStream.fail())
		return false;
	
	out1.day = value1.day;
	out1.month = value1.month;
	out1.year = value1.year;
	out1.hour = value1.hour;
	out1.minute = value1.minute;
	
	out2 = value2;
	out3 = value3;
	
	return true;
}

// This one is slightly different than the others in that it takes
// a stream.
// This is so we can parse multiple velocities from
// a single string while keeping our place within the string.
bool ParseLine(istringstream &lineStream,
			   VelocityRec &velOut)
{
	VelocityRec v1;

	lineStream >> v1.u >> v1.v;
	if (lineStream.fail())
		return false;

	velOut.u = v1.u;
	velOut.v = v1.v;

	return true;
}



////
////  End JLM new textfile line parsing functions.
////






CHARPTR RemoveTrailingSpaces(CHARPTR s)
{
	char *p;

	if (!s[0])
		return s;

	p = s + strlen(s) - 1;

	while (p >= s && (*p == ' ' || *p == LINEFEED || *p == RETURN))
		*(p--) = 0;

	return s;
}

void RemoveLeadingTrailingQuotes(CHARPTR s)
{
	char *p = s;
	long len;

	if (!s[0])
		return;

	if (s[0] == '"') {
		len = strlen(s);
		if (s[len - 1] == '"') {
			p++;
			while (*p) {
				*s = *p;
				s++;
				p++;
			}
			*(s - 1) = 0;
		}
		else return;	// if no trailing quote it's not quoted
	}
	else return;	//	if no leading quote it's not quoted
}


void RemoveLeadingTrailingWhiteSpaceHelper(CHARPTR stringIn, Boolean removeLeading, Boolean removeTrailing)
{
	bool foundLeadingWhiteSpace = false;
	char *originalS = stringIn;
	char *p = stringIn;

	if (!stringIn[0])
		return;

	if (removeLeading) {
		// move past leading white space
		while (*p == ' ' || *p == '\t' || *p == LINEFEED || *p == RETURN) {
			foundLeadingWhiteSpace = true;
			p++;
		}

		if (foundLeadingWhiteSpace) {
			//// do shift left/copy code
			while (*p) {
				*stringIn = *p;
				stringIn++;
				p++;
			}
			*stringIn = 0; // terminate the C-String
			// at this point s points to the null terminator
		}
	}

	if (removeTrailing) {
		if (!foundLeadingWhiteSpace) {
			// we did not do the shift left/copy above
			// so we need to find the end of our string
			// to sync up before going to trailing code
			// i.e, make s point to the null terminator
			stringIn = originalS + strlen(originalS);
		}

		// we need to backup from the char before s until we hit non-white Space
		if (!originalS[0])
			return;// this catches the case where s was nothing but white space

		p = stringIn - 1;
		while (p >= originalS && (*p == ' ' || *p == '\t' || *p == LINEFEED || *p == RETURN))
			*(p--) = 0;
	}
}


void RemoveLeadingWhiteSpace(CHARPTR s)
{
	RemoveLeadingTrailingWhiteSpaceHelper(s, true, false);
}


void RemoveTrailingWhiteSpace(CHARPTR s)
{
	RemoveLeadingTrailingWhiteSpaceHelper(s, false, true);
}


void RemoveLeadingAndTrailingWhiteSpace(CHARPTR s)
{
	RemoveLeadingTrailingWhiteSpaceHelper(s, true, true);
}


long CountSetInString(CHARPTR s, CHARPTR set, CHARPTR stop)
{
	long count = 0;

	for ( ; *s && (!stop || s <= stop) ; s++)
		if (strchr(set, *s))
			count++;

	return count;
}


long CountSetInString(CHARPTR s, CHARPTR set)
{
	long count = 0, setLen = 0;

	setLen = strlen(s);

	for (long i = 0; *s && i < setLen ; s++, i++)
		if (strchr(set, *s))
			count++;

	return count;
}


void RemoveSetFromString(CHARPTR strin, CHARPTR set, CHARPTR strout)
{
	long j = 0;

	for (long i = 0 ; strin[i]; i++)
		if (!strchr(set, strin[i]))
			strout[j++] = strin[i];

	strout[j] = 0;
}


// return pointer to character in s occurring after count occurrences of characters from set
CHARPTR AfterSetInString(CHARPTR s, CHARPTR set, long count)
{
	long found = 0;

	for ( ; *s && (found < count); s++)
		if (strchr(set, *s))
			found++;

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
	strncpy((char *)to, (const char *)from, (size_t)n);
	to[n] = 0;

	return to;
}


CHARPTR strnztrimcpy(CHARPTR to, CHARPTR from, short n)
{
	while (*from == ' ' && n) {
		from++;
		n--;
	}

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
	while (*from == ' ')
		from++;

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
#endif

	if (c >= ' ' && c <= 'Z')
		return c; // leave upper case
	if (c >= 'a' && c <= 'z')
		return c - 32; // convert to upper case
	if (c == 0)
		return 0;

#ifdef MAC
	//UpperText(&c, 1);
	UppercaseText(&c, 1,smSystemScript);	//smCurrentScript
#else
	for (int i = 0; table[i][0] ; i++) // international characters
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
#endif
	
	if (c >= ' ' && c < 'A') return c; // leave alone
	if (c >= 'A' && c <= 'Z') return c + 32; // convert to lower case
	if (c >= 'a' && c <= 'z') return c ; // leave alone
	if (c == 0) return 0;
	
#ifdef MAC
	//LowerText(&c, 1);
	LowercaseText(&c, 1,smSystemScript);	//smCurrentScript
#else
	for (int i = 0; table[i][0]; i++) // international characters
		if (table[i][1] == c)
			return table[i][0];
#endif
	
	return c;
}


CHARPTR StrToLower(CHARPTR s)
{
	char *p = s;
	
	for ( ; *p; p++)
		*p = mytolower(*p);
	
	return s;
}


char *strstrnocase(const char *s1, const char *s2)
{
	char *p, *retP, s1up[256], s2up[256];

	strcpy(s1up, s1);
	strcpy(s2up, s2);

	StrToUpper(s1up);
	StrToUpper(s2up);

	p = strstr(s1up, s2up);

	if (!p)
		return 0;

	//return (s1 + (p - s1up));
	retP = ((char *)s1 + (p - s1up));

	return retP;
}

short strcmpnocase(const char *s1, const char *s2)
{
	unsigned char c1, c2;

	for (int i = 0 ; ; i++) {
		if (!s1[i] && !s2[i])
			return 0;

		c1 = (unsigned char)mytoupper(s1[i]);
		c2 = (unsigned char)mytoupper(s2[i]);

		if (c1 > c2)
			return 1;

		if (c2 > c1)
			return -1;
	}

	return 0;
}


short strncmpnocase(const char *s1, const char *s2, short n)
{
	unsigned char c1, c2;

	for (int i = 0; i < n; i++) {
		if (!s1[i] && !s2[i])
			return 0;

		c1 = (unsigned char)mytoupper(s1[i]);
		c2 = (unsigned char)mytoupper(s2[i]);

		if (c1 > c2)
			return 1;

		if (c2 > c1)
			return -1;
	}

	return 0;
}


short strnblankcmp(CHARPTR s1, CHARPTR s2, short n)
{
	short i;

	for (i = 0; i < n; i++)
		if (s1[i] != ' ')
			break;

	if (i == n)
		return 0;

	for (i = 0; i < n; i++)
		if (s2[i] != ' ')
			break;

	if (i == n)
		return 0;

	return strncmp(s1, s2, n);
}

Boolean strcmptoreturn(CHARPTR s1, CHARPTR s2)
{
	for ( ; ; ) {
		if (*s1 == RETURN || *s1 == 0)
			return (*s2 == RETURN || *s2 == 0);

		if (*s2 == RETURN || *s2 == 0)
			return (*s1 == RETURN || *s1 == 0);

		if (*s1 != *s2)
			return false;

		s1++;
		s2++;
	}

	return true;
}

Boolean strcmptoreturnnocase(CHARPTR s1, CHARPTR s2)
{
	for ( ; ; ) {
		if (*s1 == RETURN || *s1 == 0)
			return (*s2 == RETURN || *s2 == 0);

		if (*s2 == RETURN || *s2 == 0)
			return (*s1 == RETURN || *s1 == 0);

		if (toupper(*s1) != toupper(*s2))
			return false;

		s1++;
		s2++;
	}

	return true;
}


// ok this looks a bit tricky, kinda like the
// memutils code.  Apparently, the first character
// designates the length of the strings referenced
// by our character pointers.
CHARPTR mypstrcat(CHARPTR dest, CHARPTR src)
{
	unsigned char length = _min(((unsigned char)*src),
					   250 - ((unsigned char)*dest));
	
	_BlockMove(src + 1, dest + *dest + 1, length);
	*dest += length;
	
	return dest;
}


CHARPTR mypstrcpy(CHARPTR dest, CHARPTR src)
{
	_BlockMove(src, dest, ((unsigned char)*src) + 1);
	
	return dest;
}


void mypstrcatJM(void *dest_asVoidPtr, void *src_asVoidPtr)
{
	unsigned char *dest = (unsigned char *)dest_asVoidPtr;
	unsigned char *src = (unsigned char *)src_asVoidPtr;
	unsigned char length = _min((*src), 255 - (*dest));
	
	_MyBlockMove(src + 1, dest + (*dest) + 1, length);
	*dest += length;
}


void mypstrcpyJM(void *dest_asVoidPtr, void *src_asVoidPtr)
{
	unsigned char *dest = (unsigned char *)dest_asVoidPtr;
	unsigned char *src = (unsigned char *)src_asVoidPtr;
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

// this function now handles MAC, IBM and UNIX style lines. -- JLM 12/1/00
// TODO: this could be drastically improved.
CHARPTR NthLineInTextHelper(CHARPTR text, long n, CHARPTR line, Boolean optimize, long maxLen)
{
	bool lineFeed = false;
	char *s, *q;
	long count = 1, i = 0;
	static long linesRead;
	static CHARPTR t, p;
	long numCharCopied = 0;
	long lineLengthInFile = 0;

	if (maxLen < 0)
		maxLen = 0;

	if (optimize && n == 0) {
		t = text;
		linesRead = 0;
	}

	p = optimize ? t : text;

	if (optimize)
		n -= linesRead;

	if(maxLen > 0)
		line[0] = 0;

	while (p[i]) {

		if ((count - 1) == n) {
			s = &p[i];
			q = line;

			while (   s[0] != 0
			       && s[0] != RETURN
			       && s[0] != LINEFEED)
			{
				// this copies to the variable
				lineLengthInFile++; // keep track of the chars even though we may not put them in the string

				if (numCharCopied < maxLen) {
					q[0] = s[0]; 
					numCharCopied++;
				}

				q++; 
				s++; 
			}

			if (s[0] == RETURN && s[1] == LINEFEED)
				lineFeed = true;

			*q = 0;	// this copies to the variable
			line[maxLen - 1] = 0; // always set the last char to 0
			break;
		}

		if (p[i] == LINEFEED && p[i + 1] != 0)
			count++;

		if (p[i] == RETURN && p[i + 1] != 0 && p[i + 1] != LINEFEED)
			count++;
		i++;
	}

	if (optimize) {
		t += i + strlen(line) + 1 + (lineFeed ? 1 : 0);
		linesRead++;
	}

	return line;
}

CHARPTR NthLineInTextOptimized(CHARPTR text, long n, CHARPTR line, long maxLen)
{
	return NthLineInTextHelper(text, n, line, true, maxLen);
}


CHARPTR NthLineInTextNonOptimized(CHARPTR text, long n, CHARPTR line, long maxLen)
{
	return NthLineInTextHelper(text, n, line, false, maxLen);
}


// is null-terminated line one of the lines in return-delimited, null-terminated text?
CHARPTR LineInTextHelper(CHARPTR text, CHARPTR line, Boolean caseSensitive)
{
	short lineLength = strlen(line);
	long textLength = strlen(text);

	while (textLength >= lineLength) {
		if (caseSensitive) {
			if (strcmptoreturn(text, line))
				return text;
		}
		else {
			if (strcmptoreturnnocase(text, line))
				return text;
		}

		while (*text && *text != RETURN) {
			text++;
			textLength--;
		}

		if (*text == RETURN) {
			text++;
			textLength--;
		}

		if (*text == LINEFEED) {
			text++;
			textLength--;
		}
	}

	return nil;
}


CHARPTR LineInText(CHARPTR text, CHARPTR line)
{
	return LineInTextHelper(text, line, true);
}


CHARPTR LineInTextNoCase(CHARPTR text, CHARPTR line)
{
	return LineInTextHelper(text, line, false);
}


CHARPTR AddLineToText(CHARPTR text, CHARPTR line)
{
	static CHARPTR t;

	if (!text[0])
		t = text;

	strcpy(t, line);
	strcat(t, "\n");

	t = &t[strlen(t)];

	return text;
}


CHARPTR IntersectLinesInText(CHARPTR text1, CHARPTR text2, CHARPTR intersection)
{
	long count = NumLinesInText(text1);
	char line[256];
	
	memset(line, 0, 256);
	intersection[0] = 0;
	
	for (long i = 0; i < count; i++) {
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

// note this is not international friendly
Boolean IsDirectionChar(char ch)
{
	switch (ch) {
		case 'n': case 'N':
		case 'e': case 'E':
		case 's': case 'S':
		case 'w': case 'W':
			return true;
	}

	return false;
}


// modify string if it does not match the desired format
// Returns TRUE if the string has been changed
Boolean ForceStringNumberHelper(CHARPTR s,
		Boolean allowNegative, Boolean allowDecimal, Boolean allowDirectionChars)
{
	bool error = false;
	short i, j;
	short n = strlen(s), numDec = 0, numNeg = 0;
	bool allowNumbers = true;

	if (allowDirectionChars) {
		// "NNE" is allowed
		// BUT !! if the first char is a letter then only letters are allowed 
		// if the first char is a number only numbers are allowed
		//
		// find first valid char
		for (i = 0; i < n; i++) {
			if (('0' <= s[i] && s[i] <= '9') ||
				(allowDecimal && s[i] == '.'))
			{
				allowDirectionChars = false;
				break;
			}

			if (IsDirectionChar(s[i])) {
				allowNumbers = false;
				break;
			}
		}
	}

	if (n > 11) {
		error = true;
		n = 11;
	}

	for (i = 0, j = 0; i < n; i++) {
		if (allowDirectionChars && IsDirectionChar(s[i]))
			s[j++] = s[i];
		else if (allowNumbers && '0' <= s[i] && s[i] <= '9')
			s[j++] = s[i];
		else if (allowNumbers && allowDecimal && s[i] == '.') {
			if (++numDec > 1)
				error = true;
			else
				s[j++] = s[i];
		}
		else if(allowNumbers && allowNegative && s[i] == '-') {
			if (++numNeg > 1)
				error = true;
			else
				s[j++] = s[i];
		}
		else
			error = true;
	}

	s[j] = 0;

	if (j == 0) {
		if (allowDecimal)
			strcpy(s, "0.0");
		else
			strcpy(s, "0");

		error = true;
	}

	if (error) {
		// we changed the string
		SysBeep(1);
		return true;
	}

	return false;
}


// beep and modify string if it contains non-digits
Boolean ForceStringNumber(CHARPTR s)
{	//JLM
	Boolean allowNegativeFlag = false;
	Boolean allowDecimalFlag = false;
	Boolean allowDirectionChars = false;
	return ForceStringNumberHelper(s, allowNegativeFlag, allowDecimalFlag, allowDirectionChars);
}


Boolean ForceStringNumberAllowingNegative(CHARPTR s)
{	//JLM
	Boolean allowNegativeFlag = true;
	Boolean allowDecimalFlag = false;
	Boolean allowDirectionChars = false;
	return ForceStringNumberHelper(s, allowNegativeFlag, allowDecimalFlag, allowDirectionChars);
}


// same as ForceStringNumber() but allow a decimal
Boolean DecForceStringNumber(CHARPTR s)
{	//JLM
	Boolean allowNegativeFlag = false;
	Boolean allowDecimalFlag = true;
	Boolean allowDirectionChars = false;
	return ForceStringNumberHelper(s, allowNegativeFlag, allowDecimalFlag, allowDirectionChars);
}


Boolean DecForceStringNumberAllowingNegative(CHARPTR s)
{	//JLM
	Boolean allowNegativeFlag = true;
	Boolean allowDecimalFlag = true;
	Boolean allowDirectionChars = false;
	return ForceStringNumberHelper(s, allowNegativeFlag, allowDecimalFlag, allowDirectionChars);
}


Boolean DecForceStringDirection(CHARPTR s)
{	//JLM
	Boolean allowNegativeFlag = false; // 12/7/98 they don't want negative directions
	Boolean allowDecimalFlag = false;
	Boolean allowDirectionChars = true;
	return ForceStringNumberHelper(s, allowNegativeFlag, allowDecimalFlag, allowDirectionChars);
}


void Secs2DateStrings(Seconds seconds,
					  CHARPTR dateLong, CHARPTR dateShort,
					  CHARPTR time24, CHARPTR time12)
{
#ifndef pyGNOME
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
#endif

	struct tm *time;
	time_t converted_seconds = (time_t)seconds;

	time = localtime(&converted_seconds); // gmtime(&seconds);

	if (!time) {
		SysBeep(1);
		return;
	}
	
	if (time->tm_isdst == 1) {
		// we want to show standard time so we need to fake out the daylight savings time
		if(time->tm_hour > 0)
			time->tm_hour--;
		else {
			seconds -= 3600;
			time = localtime(&converted_seconds);
			if (time->tm_isdst == 1) {
				// life is good, both times were daylight savings time
			}
			else
				printError("Programmer error in Secs2DateStrings");
		}
	}

	if (dateLong)
		strftime(dateLong, 32, "%B %d, %Y", time);

	if (dateShort)
		strftime(dateShort, 32, "%m/%d/%y", time);

	if (time24) {
		if (gUseColonIn24HrTime)
			strftime(time24, 32, "%H:%M", time);
		else
			strftime(time24, 32, "%H%M", time);
	}

	if (time12)
		strftime(time12, 32, "%#I:%M %p", time);
}


void Secs2DateString(Seconds seconds, CHARPTR s)
{
#ifndef pyGNOME
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
#endif

	struct tm *newTime;
	time_t converted_seconds = (time_t)seconds;

	newTime = localtime(&converted_seconds); // gmtime(&seconds)
	newTime->tm_year = newTime->tm_year % 100; // year 2000 fix , JLM 1/25/99
	if (newTime) {
		if (newTime->tm_isdst == 1) {
			// we want to show standard time so we need to fake out the daylight savings time
			if(newTime->tm_hour > 0)
				newTime->tm_hour--;
			else {
				seconds -= 3600;
				newTime = localtime(&converted_seconds);
				if (newTime->tm_isdst == 1) {
					newTime->tm_year = newTime->tm_year % 100; // year 2000 fix , JLM 1/25/99
					// life is good, both times were daylight savings time
				}
				else
					printError("Programmer error in Secs2DateString");
			}
		}
		sprintf(s, "%02ld/%02ld/%02ld", (long)newTime->tm_mon + 1, (long)newTime->tm_mday, (long)newTime->tm_year);
	}
	else {
		strcpy(s, "???");
		SysBeep(1);
	}
}

void Secs2DateString2(Seconds seconds, CHARPTR s)
{
#ifndef pyGNOME
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
#endif

	// returns in format: January 2, 1998
	short day; //1-31
	short month; // 1-12
	short year4; // 4 digit year
	short hour; // 0-23
	short minute;// 0-59 //
	char str[255];

	struct tm *newTime;
	time_t converted_seconds = (time_t)seconds;

	newTime = localtime(&converted_seconds); // gmtime(&seconds)
	if (newTime) {
		if (newTime->tm_isdst == 1) {
			// we want to show standard time so we need to fake out the daylight savings time
			if (newTime->tm_hour > 0)
				newTime->tm_hour--;
			else {
				seconds -= 3600;
				newTime = localtime(&converted_seconds);

				if (newTime->tm_isdst == 1) {
					// life is good, both times were daylight savings time
				}
				else
					printError("Programmer error in Secs2DateStrings2");
			}
		}

		month = newTime->tm_mon + 1;
		day = newTime->tm_mday;
		year4 = newTime->tm_year + 1900; // year 2000 OK
		hour = newTime->tm_hour;
		minute = newTime->tm_min;
	}
	else {
		strcpy(s, "???");
		return;
	}

	switch (month)
	{
		case  1:
			strcpy (s, "January ");
			break;
		case  2:
			strcpy (s, "February ");
			break;
		case  3:
			strcpy (s, "March ");
			break;
		case  4:
			strcpy (s, "April ");
			break;
		case  5:
			strcpy (s, "May ");
			break;
		case  6:
			strcpy (s, "June ");
			break;
		case  7:
			strcpy (s, "July ");
			break;
		case  8:
			strcpy (s, "August ");
			break;
		case  9:
			strcpy (s, "September ");
			break;
		case 10:
			strcpy (s, "October ");
			break;
		case 11:
			strcpy (s, "November ");
			break;
		case 12:
			strcpy (s, "December ");
			break;
	}
	
	sprintf(str, "%02hd, %hd ", day, year4);
	strcat (s, str);
	sprintf (str, "%2.2d:%2.2d", hour, minute);
	strcat (s, str);
}

void Secs2DateStringNetCDF(Seconds seconds, CHARPTR s)
{
#ifndef pyGNOME
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
#endif

	// returns in format: 2012-08-23 08:31:00
	short day ; //1-31
	short month; // 1-12
	short year4; // 4 digit year
	short hour; // 0-23
	short minute;// 0-59 //
	short second = 0;
	char str[255];

	// IBM
	struct tm *newTime;
	time_t converted_seconds = (time_t)seconds;

	newTime = localtime(&converted_seconds); // gmtime(&seconds)

	if (newTime) {
		if (newTime->tm_isdst == 1) {
			// we want to show standard time so we need to fake out the daylight savings time
			if(newTime->tm_hour > 0)
				newTime->tm_hour--;
			else {
				seconds -= 3600;
				newTime = localtime(&converted_seconds);
				if(newTime->tm_isdst == 1) {
					// life is good, both times were daylight savings time
				}
				else
					printError("Programmer error in Secs2DateStringNetCDF");
			}
		}
		
		month = newTime->tm_mon + 1;
		day = newTime->tm_mday;
		year4 = newTime->tm_year + 1900; // year 2000 OK
		hour = newTime->tm_hour;
		minute = newTime->tm_min;
	}
	else {
		strcpy(s, "???");
		return;
	}
	

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
	time_t seconds;
	struct tm newTime;
	
	sscanf(s, "%hd/%hd/%hd", &month, &day, &year);
	newTime.tm_sec = newTime.tm_min = newTime.tm_hour = 0;
	newTime.tm_mday = day;
	newTime.tm_mon = month - 1;

	if (year < 40)
		year+= 100; // year 2000 solution, JLM 1/25/99

	newTime.tm_year = (year < 200) ? year : year - 1900;
	//newTime.tm_isdst = -1; // let mktime() determine if it's daylight savings time
	newTime.tm_isdst = 0; // use standard time to avoid duplicates at changeover 4/12/01

	seconds = mktime(&newTime);

	if (seconds == -1) {
		seconds = 0;
		printNote("The Windows function localtime() does not accept dates earlier than January 1, 1970 GMT");
		SysBeep(1);
	}	// here check if year < 1970 and convert

#ifndef pyGNOME
	seconds += 2082816000L; // convert from seconds since 1970 to seconds since 1904
#endif

	return (unsigned long)seconds;
}
#endif

char *Date2String(DateTimeRec *time, char *s)
{
	sprintf(s, "%02hd/%02hd/%02hd %02hd:%02hd:%02hd",
			time->month, time->day, time->year,
			time->hour, time->minute, time->second);
	
	return s;
}

char *Date2KmlString(DateTimeRec *time, char *s)
{
	short year = time->year;
	if (time->year < 100)
		year = year+2000; 
	sprintf(s, "%04hd-%02hd-%02hdT%02hd:%02hd:%02hdZ",
			year, time->month, time->day,
			time->hour, time->minute, time->second);
	
	return s;
}

void SplitPathFile(CHARPTR fullPath, CHARPTR fileName)
{
	char *p;

	fileName[0] = 0;

	if ((p = strrchr(fullPath, DIRDELIMITER)) != NULL) {
		if (p[1] == 0) {
			// treat final directory as file name to be retrieved
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

void SplitPathFileName(CHARPTR fullPath, CHARPTR fileName)
{
	char *p;
	
	fileName[0] = 0;
	
	if ((p = strrchr(fullPath, NEWDIRDELIMITER)) != 0) {
		if (p[1] == 0) {
			// treat final directory as file name to be retrieved
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
	short len;
	char *s = (char *)string;

	if (!s)
		return;

	len = (unsigned char)s[0];

	for (short i = 0; i < len; i++)
		s[i] = s[i + 1];

	s[len] = 0;
}


void my_c2pstr(void *string)
{
	short len;
	char *s = (char*) string;

	if(!s)
		return;

	len = strlen(s);

	if (len > 255)
		len = 255;

	for (short i = len; i > 0; i--)
		s[i] = s[i - 1];

	s[0] = (unsigned char)len;
}


Seconds RoundDateSeconds(Seconds timeInSeconds)
{
	double	DaysInMonth[13] = {0.0, 31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0};
	DateTimeRec date;
	Seconds roundedTimeInSeconds;

	// get rid of the seconds since they get garbled in the dialogs
	SecondsToDate(timeInSeconds, &date);

	if (date.second == 0)
		return timeInSeconds;

	if (date.second > 30) {
		if (date.minute < 59)
			date.minute++;
		else {
			date.minute = 0;
			if (date.hour < 23)
				date.hour++;
			else {
				if ( (date.year % 4 == 0 && date.year % 100 != 0) || date.year % 400 == 0)
					DaysInMonth[2] = 29.0;

				date.hour = 0;

				if (date.day < DaysInMonth[date.month])
					date.day++;
				else {
					date.day = 1;

					if (date.month < 12)
						date.month++;
					else {
						date.month = 1;

						if (date.year > 2019) {
							printError("Time outside of model range");
							/*err=-1; goto done;*/
						}
						else
							date.year++;

						date.year++;
					}
				}
			}
		}
	}

	date.second = 0;
	DateToSeconds(&date, &roundedTimeInSeconds);
	return roundedTimeInSeconds;
}


#ifndef MAC

void GetDateTime(Seconds *seconds)
{
	time_t secs;
	
	time(&secs); // time() wants a near pointer; returns secs since 1/1/70

#ifndef pyGNOME
	secs += 2082816000L; // convert to seconds since 1904
#endif

	// would be secs += 126230400 if time returned time since 1900, as it says in the book
	*seconds = (Seconds)secs;
}


void SecondsToDate (Seconds seconds, DateTimeRec *date)
{
#ifndef pyGNOME
	seconds -= 2082816000L; // convert from seconds since 1904 to seconds since 1970
#endif

	struct tm *newTime;
	time_t converted_seconds = (time_t)seconds;

	// note time_t is now 64 bit by default on Windows, preprocessor definition for 32 bit (good til 2038)
	newTime = localtime(&converted_seconds); // gmtime(&seconds)

	if (newTime)
	{
		if (newTime->tm_isdst == 1) {
			// we want to show standard time so we need to fake out the daylight savings time
			if (newTime->tm_hour > 0)
				newTime->tm_hour--;
			else {
				seconds -= 3600;
				newTime = localtime(&converted_seconds);

				if (newTime->tm_isdst == 1) {
					newTime->tm_year = newTime->tm_year % 100;// year 2000 fix , JLM 1/25/99
					// life is good, both times were daylight savings time
				}
				else
					printError("Programmer error in Secs2DateStrings");
			}
		}
		
		date->year = newTime->tm_year;

		// this mimics the mac function which has a 4 digit value in the time field
		if (date->year < 40)
			date->year += 2000;
		else if (date->year < 200)
			date->year += 1900;

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


void DateToSeconds (DateTimeRec *date, Seconds *seconds)
{
	char s[100];
	unsigned long secs;
	
	sprintf(s, "%02hd/%02hd/%02hd", date->month, date->day, date->year);
	
	secs = DateString2Secs(s);
	secs += date->hour * 3600 + date->minute * 60 + date->second;
	
	(*seconds) = secs;
}
#endif

// Path functions - eventually replace with a library
Boolean IsClassicAbsolutePath(char *path)
{
	// classic paths use ':' delimiters and full paths start with drive name
	if (IsClassicPath(path) && path[0] != ':' && path[0] != '.')
		return true;

	return false;
}


Boolean IsUnixAbsolutePath(char *path)
{
	if (path[0] == '/')
		return true;

	return false;
}


Boolean IsWindowsAbsolutePath(char *path)
{
	// check for mapped drive 
	if (path[1] == ':' && path[2] == '\\')
		return true;

	// check for unmapped drive
	if (path[1] == '\\' && path[2] == '\\')
		return true;

	// at some point switch the leading \ to be full path rather than partial, have to figure out the drive
	return false;
}


Boolean IsWindowsPath(char *path)
{
	long len;

	//If leads with a {drive letter}:\ it's a full path, this is covered below
	// unmapped drive \\, also covered below
	if (IsWindowsAbsolutePath(path))
		return true;

	// if has '\' anywhere in path it's Windows (though Mac allows '\' in filenames, for now assuming it's a delimiter)
	len = strlen(path);
	for (long i = 0; i < len  && path[i]; i++) {
		if (path[i] == '\\')
			return true;
	}

	return false;	// is filename only true or false...
}


Boolean IsUnixPath(char* path)
{
	long len;

	//If leads with a '/' it's a full path, this is covered below
	//if (IsUnixAbsolutePath(path) return true;

	// if has '/' anywhere in path it's unix (though Mac allows '/' in filenames, for now assuming it's a delimiter)
	len = strlen(path);
	for (long i = 0; i < len  && path[i]; i++) {
		if (path[i] == '/')
			return true;
	}

	return false;	// filename only should be true...
}


Boolean IsClassicPath(char *path)
{
	long len;

	if (IsWindowsAbsolutePath(path))
		return false;

	// if has ':' anywhere in path it's classic (Windows and Mac don't allow ':' in filenames)
	len = strlen(path);
	for (long i = 0; i < len  && path[i]; i++) {
		if (path[i] == ':')
			return true;
	}

	return false;	// is filename only true or false...
}

Boolean IsFullPath(char *path)
{
	if (IsWindowsAbsolutePath(path))
		return true;
	if (IsClassicAbsolutePath(path))
		return true;
	if (IsUnixAbsolutePath(path))
		return true;
	return false;
}

Boolean ConvertIfClassicPath(char *path, char *unixPath)
{
	// do we need to support old filelists?
	if (IsWindowsPath(path))
		return false;

	if (IsUnixPath(path))
		return false;
	
#ifdef MAC
	if (IsClassicAbsolutePath(path)) 
	{
		OSErr err = 0;
		err = ConvertTraditionalPathToUnixPath((const char *)path, unixPath, kMaxNameLen); 
		return true;
	}
#endif	

	if (IsClassicPath(path))
	{
		// partial path
		// leading ':' colons can mean directory up (first is current directory)

		bool foundLeadingColons = false;
		short numLeadingColons = 0;
		char *p = path;

		StringSubstitute(path, ':', '/');

		while (*p == '/') {
			foundLeadingColons = true;
			p++;
			numLeadingColons++;
		}

		if(foundLeadingColons) {
			// do shift left/copy code
			while (*p) {
				*path = *p;
				path++;
				p++;
			}

			*path = 0;

			// at this point s points to the null terminator
			if (numLeadingColons == 1) {
				strcpy(unixPath, path);
			}
			else {
				for (short i = 0; i < numLeadingColons - 1; i++) {
					strcat(unixPath, ".");
				}

				strcat(unixPath, "/");
				strcat(unixPath, path);
			}
		}

		return true;
	}

	// assume if doesn't have any file delimiters it's a filename - leave as is
	strcpy(unixPath, path);

	return true;
}



//
//  begin JLM new file handling functions.
//

// get the size of our open file
// - we do not explicitly handle any exceptions, that
//   is up to the caller
// - we do not alter the current position of
//   our file stream.
ios::pos_type FileSize(fstream &file)
{
	ios::pos_type currentPosition = file.tellg();

	file.seekg(0, ios::end);
	ios::pos_type fileSize = file.tellg();

	file.seekg(currentPosition, ios::beg);

	return fileSize;
}


// A safe getline function that handles all three line endings.
//     ("\r", "\n" and "\r\n")
// The characters in the stream are read one-by-one using a std::streambuf.
// That is faster than reading them one-by-one using the std::istream.
// Code that uses streambuf this way must be guarded by a sentry object.
// The sentry object performs various tasks,
// such as thread synchronization and updating the stream state.
istream &safeGetLine(istream &is, string &t)
{
    t.clear();

    istream::sentry se(is, true);
    streambuf* sb = is.rdbuf();

    for(;;) {
        int c = sb->sbumpc();
        switch (c) {
        case '\n':
            return is;
        case '\r':
            if(sb->sgetc() == '\n')
                sb->sbumpc();
            return is;
        case EOF:
            // Also handle the case when the last line has no line ending
            if(t.empty())
                is.setstate(ios::eofbit);
            return is;
        default:
            t += (char)c;
        }
    }

    // We shouldn't actually get here, but it is good
    // practice to return something in all cases
    return is;
}

// Reads the lines contained in a text file in as safely a manner
// as we can.
// Returns: true if we were successful, otherwise returns false
// Modifies: populates the incoming string vector with the strings
//           contained in the file.
bool ReadLinesInFile(const char *name, vector<string> &stringList, size_t linesToRead)
{
	try {
		fstream inputFile(name, ios::in);
		if (inputFile.is_open()) {
			stringList.clear();

			if (linesToRead == 0)
				linesToRead = (size_t) -1; // max value possible

			while (inputFile.good() && (linesToRead-- != 0)) {
				string line;

				safeGetLine(inputFile, line);
				stringList.push_back(line);
			}

			inputFile.close();
		}
		else {
			throw("Unable to open file");
		}
	}
	catch(...) {
		printError("We are unable to open or read from the file. \nBreaking from ReadLinesInFile().\n");
		return false;
	}
	return true;
}

bool ReadLinesInFile(const string &name, vector<string> &stringList, size_t linesToRead)
{
	return ReadLinesInFile(name.c_str(), stringList, linesToRead);
}

// Reads the lines contained in a text buffer in as safely a manner
// as we can.
// Returns: true if we were successful, otherwise returns false
// Modifies: populates the incoming string vector with the strings
//           contained in the file.
bool ReadLinesInBuffer(CHARH fileBufH, vector<string> &stringList, size_t linesToRead)
{
	try {
		stringList.clear();

		string buffer = *fileBufH;
		istringstream lineStream(buffer);

		if (linesToRead == 0)
			linesToRead = (size_t) -1; // max value possible

		for (string each;
			 safeGetLine(lineStream, each) && (linesToRead-- != 0);
			 stringList.push_back(each));
	}
	catch(...) {
		printError("We are unable to open or read from the buffer. \nBreaking from ReadLinesInBuffer().\n");
		return false;
	}
	return true;
}

void ConvertDriveLetterToUnixStyle(string &pathPart)
{
#ifndef _WIN32
	// we only do this if we are running on a non Windows platform
	// and we need to process a DOS-style path that contains a
	// drive letter prefix.  In this case, we will convert it
	// into an absolute path.
	if (pathPart.size() != 2)
		return;

	if (std::isalpha(pathPart[0]) &&
		pathPart[1] == ':')
	{
		// convert from '<driveletter>:' to '/driveletter'
		pathPart = "/";
		pathPart += pathPart[0];
	}
#endif

	return;
}

void SplitPathIntoDirAndFile(string &path, string &dir, string &file)
{
	size_t slashPosition = path.find_last_of("\\/");
	if (slashPosition == string::npos) {
		dir = "";
		file = path;
	}
	else {
		dir = path.substr(0, slashPosition);
		file = path.substr(slashPosition);
	}
}

// Convert the input path to a usable form for the
// platform we are running on.
// Example 1:
//   If:
//     - We are on OSX
//     - We are processing the path 'C:\dir1\dir2\file.txt'
//   Then:
//     - We will convert the path to '/C/dir1/dir2/file.txt'
// Example 2:
//   If:
//     - We are on OSX
//     - We are processing a mixed path 'C:\dir1\dir2/file.txt'
//   Then:
//     - We will convert the path to '/C/dir1/dir2/file.txt'
// Example 3:
//   If:
//     - We are on Windows
//     - We are processing a mixed path 'C:\dir1\dir2/file.txt'
//   Then:
//     - We will convert the path to 'C:\dir1\dir2\file.txt'
// Example 4:
//   If:
//     - We are on Windows
//     - We are processing a unix path '/dir1/dir2/file.txt'
//   Then:
//     - We will convert the path to '\dir1\dir2\file.txt'
//     - No assumption will be made as to a drive letter.
//       Thus, the current active drive will be implicitly referenced.
void ConvertPathToCurrentPlatform(string inputPath)
{
	std::vector<std::string> pathParts = SplitPath(inputPath);
	ConvertDriveLetterToUnixStyle(pathParts[0]);
	//cerr << "ConvertPathToCurrentPlatform(): " << pathParts.size() << " path components." << endl;
	std::string delim;
	delim += NEWDIRDELIMITER;
	inputPath = std::accumulate( pathParts.begin(), pathParts.end(), delim);
}

bool ResolvePath(string &containingDir, string &pathToResolve)
{
	// test if we got good inputs
	if (pathToResolve.size() == 0) {
		cerr << "ResolvePath(): path to resolve is empty" << endl;
		return false;
	}
	
	if (containingDir.size() == 0) {
		// we assume the current working directory
		//cerr << "ResolvePath(): containing directory is empty.  Setting to current working directory" << endl;
		containingDir = '.';
	}
	
	string inputPath = containingDir;
	ConvertPathToCurrentPlatform(trim(inputPath));
	
	string resolvedPath = pathToResolve;
	ConvertPathToCurrentPlatform(trim(resolvedPath));
	
	
	//if (FileExists(0, 0, (char *)resolvedPath.c_str())) {
	if (FileExists(0, 0, resolvedPath.c_str())) {
		// no problem, the file exists at the path given
		pathToResolve = resolvedPath;
		return true;
	}
	
	// otherwise we have to try to find it
	//if (!FileExists(0, 0, (char *)inputPath.c_str())) {
	if (!FileExists(0, 0, inputPath.c_str())) {
		// If the containing directory is not valid, no point going further.
		cerr << "ResolvePath(): Containing directory is not valid" << endl;
		return false;
	}
	//cerr << "ResolvePath(): Our base path: " << inputPath << endl;
	//cerr << "ResolvePath(): Our path to resolve: " << resolvedPath << endl;
	
	string pathToTry;
	vector<string> pathComponentsToReference = SplitPath(inputPath);
	vector<string> pathComponentsToResolve = SplitPath(resolvedPath);
	
	while (pathComponentsToResolve.size() > 0) {
		vector<string> concatPath = pathComponentsToReference;
		concatPath.insert(concatPath.end(),
						  pathComponentsToResolve.begin(),
						  pathComponentsToResolve.end());
		
		string delim;
		delim += NEWDIRDELIMITER;
		//cerr << "ResolvePath(): Our delimiter is '" << delim << "'"<< endl;
		pathToTry = join(concatPath, delim);
		//cerr << "ResolvePath(): Trying path " << pathToTry << endl;
		//if (FileExists(0, 0, (char *) pathToTry.c_str())) {
		if (FileExists(0, 0, pathToTry.c_str())) {
			pathToResolve = pathToTry;
			return true;
		}
		
		pathComponentsToResolve.erase(pathComponentsToResolve.begin());
	}
	
	cerr << "ResolvePath(): Could not resolve path" << endl;
	return false;
}


//
//  end JLM new file handling functions.
//

