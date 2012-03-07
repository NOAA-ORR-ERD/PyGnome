/*
 *  StringFunctions.cpp
 *  c_gnome
 *
 *  Created by Generic Programmer on 2/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "Earl.h"
#include "TypeDefs.h"


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
void RemoveTrailingWhiteSpace(CHARPTR s)
{
	RemoveLeadingTrailingWhiteSpaceHelper(s,false,true);
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