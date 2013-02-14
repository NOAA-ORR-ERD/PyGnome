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

/////////////////////////////////////////////////
/////////////////////////////////////////////////

/////////////////////////////////////////////////
// OSSM supports what Bushy calls a "Long Wind File"
// A Long Wind file is a wind file with a 4 line 
// header. Here is the info from Bushy
/////////////////////////////////////////////////
// 
//INCHON    
//37,30,126,38
//knots
//LTIME
//0,0,0,0,0,0,0,0
//

//Station name
//Station latitude degree, latitude minute, longitude degree, longitude minute
//units
//time units
//bounding box
//
//
//Notes:
//The latitudes and longitudes are always positive.  OSSM figures out hemisphere from the maps.
//
//Allowable speed units ... not case sensitive
//knots
//meters per second
//centimeters per second
//miles per hour
//
//Time zone label is used to correct time offset if the data is not in local time 
//and the user is running in local time.  Often times wind data comes in GMT or local standard time.  
//In TAP we convert it all to local standard time and never bother with daylight savings time.
//
//Bounding box is an interpolation option for dealing with multiple wind files.  
//There are 4 wind interpolation options in OSSM and bounding boxes that can overlap is one.  
//If you chose another scheme, the bounding box data is ignored, that's why you see all the zeros.
//Upper left latitude, Lower right latitude, Upper left longitude, Lower Right Longitude
//

double UorV(VelocityRec vector, short index);
double UorV(VelocityRec3D vector, short index);

Boolean IsLongWindFile(char* path,short *selectedUnitsP,Boolean *dataInGMTP)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	short selectedUnits = kUndefined;
	Boolean dataInGMT = FALSE;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	// code goes here, if lines are long may run out of space in array
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		DateTimeRec time;
		char value1S[256], value2S[256];
		long numScanned;
		// we check 
		// that the 3rd line contains the units 
		// that the 4th line is either LTIME or GMTTIME
		// that the 6th line can scan 7 values
		line = 0;
		bIsValid = true;
		/////////////////////////////////////////////////
		
		// first line , station name
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		// for now we ignore station name
		/////////////////////////////////////////////////
		
		// second line, station position
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);
		// for now we ignore station position
		/////////////////////////////////////////////////
		
		// third line, units
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		selectedUnits = StrToSpeedUnits(strLine);// note we are not supporting cm/sec in gnome
		if(selectedUnits == kUndefined)
			bIsValid = false; 
		
		/////////////////////////////////////////////////
		
		// fourth line, local or GMT time  
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		if(!strcmpnocase(strLine,"LTIME"))
			dataInGMT = FALSE;
		if(!strncmpnocase(strLine,"GMT",strlen("GMT"))) 
			dataInGMT = TRUE;
		else
		{
			dataInGMT = FALSE; // Bushy says the flags can be things like PST, but they all boil down to local time
			// check if this is a valid data line, then it is probably a valid tide file
			// tide files with header have same first 3 lines as long wind files, followed by data
			StringSubstitute(strLine, ',', ' ');
			numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
						  &time.day, &time.month, &time.year,
						  &time.hour, &time.minute, value1S, value2S);
			if (numScanned == 7)	
				bIsValid = false;
		}	
		/////////////////////////////////////////////////
		
		// fifth line, grid
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		// for now we ignore the grid
		/////////////////////////////////////////////////

		// sixth line, first line of data
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		StringSubstitute(strLine, ',', ' ');
		numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute, value1S, value2S);
		if (numScanned != 7)	
			bIsValid = false;
		/////////////////////////////////////////////////
	}
	
	if(bIsValid)
	{
		*selectedUnitsP = selectedUnits;
		*dataInGMTP = dataInGMT;
	}
	return bIsValid;
}

Boolean IsHydrologyFile(char* path)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		DateTimeRec time;
		char value1S[256], value2S[256];
		long numScanned;
		// we check 
		// that the 3rd line contains the units 
		// that the 4th line can scan 7 values
		line = 0;
		bIsValid = true;
		/////////////////////////////////////////////////
		
		// first line , station name
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		// for now we ignore station name
		/////////////////////////////////////////////////
		
		// second line, station position
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);
		// for now we ignore station position
		/////////////////////////////////////////////////
		
		// third line, units, only cubic feet per second for now
		// added cubic meters per second, and the k versions which should cover all cases 5/18/01
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		if (!strcmpnocase(strLine,"CFS") || !strcmpnocase(strLine,"KCFS") 
			|| !strcmpnocase(strLine,"CMS") || !strcmpnocase(strLine,"KCMS")) 
			bIsValid = true;
		else 
		{
			bIsValid = false;
			return bIsValid;
		}
		
		/////////////////////////////////////////////////
				
		// fourth line, first line of data
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		StringSubstitute(strLine, ',', ' ');
		numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute, value1S, value2S);
		if (numScanned != 7)	
			bIsValid = false;
		/////////////////////////////////////////////////
	}
	
	return bIsValid;
}

Boolean IsOSSMTimeFile(char* path,short *selectedUnitsP)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	short selectedUnits = kUndefined;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		DateTimeRec time;
		char value1S[256], value2S[256];
		long numScanned;
		// we check 
		// that the 3rd line contains the units 
		// that the 4th line can scan 7 values
		line = 0;
		bIsValid = true;
		/////////////////////////////////////////////////
		
		// first line , station name
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		// for now we ignore station name
		/////////////////////////////////////////////////
		
		// second line, station position
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);
		// for now we ignore station position
		/////////////////////////////////////////////////
		
		// third line, units
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		selectedUnits = StrToSpeedUnits(strLine);// note we are not supporting cm/sec in gnome
		if(selectedUnits == kUndefined)
			bIsValid = false; 
		
		/////////////////////////////////////////////////
				
		// fourth line, first line of data
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		StringSubstitute(strLine, ',', ' ');
		numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute, value1S, value2S);
		if (numScanned != 7)	
			bIsValid = false;
		/////////////////////////////////////////////////
	}
	if(bIsValid)
	{
		*selectedUnitsP = selectedUnits;
	}
	return bIsValid;
}

Boolean IsOSSMHeightFile(char* path,short *selectedUnitsP)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	short selectedUnits = kUndefined;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		DateTimeRec time;
		char value1S[256], value2S[256];
		long numScanned;
		// we check 
		// that the 3rd line contains the units 
		// that the 4th line can scan 7 values
		line = 0;
		bIsValid = true;
		/////////////////////////////////////////////////
		
		// first line , station name
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		// for now we ignore station name
		/////////////////////////////////////////////////
		
		// second line, station position
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);
		// for now we ignore station position
		/////////////////////////////////////////////////
		
		// third line, units
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);	// code goes here, decide what units to support - m, ft,...
		selectedUnits = StrToSpeedUnits(strLine);// note we are not supporting cm/sec in gnome
		if(selectedUnits == kUndefined)
			bIsValid = false; 
		
		/////////////////////////////////////////////////
				
		// fourth line, first line of data
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		StringSubstitute(strLine, ',', ' ');
		numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute, value1S, value2S);
		if (numScanned != 7)	
			bIsValid = false;
		/////////////////////////////////////////////////
	}
	if(bIsValid)
	{
		*selectedUnitsP = selectedUnits;
	}
	return bIsValid;
}


Boolean IsTimeFile(char* path)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		DateTimeRec time;
		char value1S[256], value2S[256];
		long numScanned;
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		StringSubstitute(strLine, ',', ' ');
		numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute, value1S, value2S);
		if (numScanned == 7)	
			bIsValid = true;
	}
	return bIsValid;
}
