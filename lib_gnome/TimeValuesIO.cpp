#include "TimeValuesIO.h"


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

//double UorV(VelocityRec vector, short index);
//double UorV(VelocityRec3D vector, short index);



// we check
// that the 3rd line contains the units
// that the 4th line is either LTIME, GMTTIME, or a valid date entry.
// that the 6th line can scan 7 values
bool IsLongWindFile(vector<string> &linesInFile, short *selectedUnitsOut, bool *dataInGMTOut)
{
	long line = 0;
	string currentLine, val1Str, val2Str;

	DateTimeRec time;
	VelocityRec velocity;

	short selectedUnits = kUndefined;
	bool dataInGMT = false;

	// size check
	if (linesInFile.size() < 5)
		return false;

	// First line , station name. For now we ignore this.
	line++;

	// Second line, station position. For now we ignore this.
	line++;

	// Third line, units
	// Note: We are not supporting cm/sec in gnome
	currentLine = trim(linesInFile[line++]);
	selectedUnits = StrToSpeedUnits((char *)currentLine.c_str());
	if (selectedUnits == kUndefined)
		return false;

	// fourth line, local or GMT time
	currentLine = trim(linesInFile[line++]);
	std::transform(currentLine.begin(),
				   currentLine.end(),
				   currentLine.begin(),
				   ::tolower);
	if (currentLine == "ltime")
		dataInGMT = false;
	else if (currentLine == "gmt")
		dataInGMT = true;
	else {
		dataInGMT = false;
		
		// If we don't choose the ltime or gmtime, I guess we need to have
		// a specific date/time entry on this line.  I would like to have good
		// examples of these entries, but it looks like it's probably in the
		// format of 'DD,MM,YYYY,HH,MM,value1,value2'
		// Bushy says the flags can be things like PST, but they all boil down to local time
		// check if this is a valid data line, then it is probably a valid tide file
		// tide files with header have same first 3 lines as long wind files, followed by data
		
		// Not sure what is going on here - this is not an optional line
		//std::replace(currentLine.begin(), currentLine.end(), ',', ' ');

		//if (!ParseLine(currentLine, time, val1Str, val2Str))
			//return false;

	}

	// fifth line, grid.  For now we ignore this.
	line++;

	// sixth line, first line of data
	currentLine = trim(linesInFile[line++]);

	std::replace(currentLine.begin(), currentLine.end(), ',', ' ');
	//if (!ParseLine(currentLine, time, velocity))
	if (!ParseLine(currentLine, time, val1Str, val2Str))
		return false;

	*selectedUnitsOut = selectedUnits;
	*dataInGMTOut = dataInGMT;

	return true;
}


Boolean IsLongWindFile(char *path, short *selectedUnitsOut, Boolean *dataInGMTOut)
{
	vector<string> linesInFile;

	if (ReadLinesInFile(path, linesInFile))
		return IsLongWindFile(linesInFile, selectedUnitsOut, (bool *)dataInGMTOut);
	else
		return false;
}


bool IsNDBCWindFile(vector<string> &linesInFile, long *numHeaderLines)
{
	bool bIsValid = false;
	string currentLine;

	if (linesInFile.size() == 0)
		return false;

	*numHeaderLines = 0;	// if we return false don't set this value

	currentLine = trim(linesInFile[0]);

	if (currentLine == "#YY") {
		*numHeaderLines = 2;
		return true;
	}
	else if (currentLine == "YYYY" || currentLine == "YY") {
		*numHeaderLines = 1;
		return true;
	}

	return false;
}


// Returns: false = error
//          true = success
Boolean IsNDBCWindFile(char* path, long *numHeaderLines)
{
	vector<string> linesInFile;

	if (ReadLinesInFile(path, linesInFile))
		return IsNDBCWindFile(linesInFile, numHeaderLines);
	else
		return false;
}


bool IsNCDCWindFile(vector<string> &linesInFile)
{
	long line = 0;
	string value;

	// first line , header - USAF  WBAN YR--MODAHRMN DIR SPD GUS CLG SKC L M H  VSB MW MW MW MW AW AW AW AW W TEMP DEWP    SLP   ALT    STP MAX MIN PCP01 PCP06 PCP24 PCPXX SD
	if (ParseKeyedLine(linesInFile[line++], "USAF", value))
		return true;
	else
		return false;
}

Boolean IsNCDCWindFile(char *path)
{
	vector<string> linesInFile;

	if (ReadLinesInFile(path, linesInFile))
		return IsNCDCWindFile(linesInFile);
	else
		return false;
}


// we check
// that the 3rd line contains the units
// that the 4th line can scan 7 values
bool IsHydrologyFile(vector<string> &linesInFile)
{
	long line = 0;
	string currentLine;

	DateTimeRec time;
	VelocityRec velocity;

	// First line , station name. For now we ignore this.
	line++;

	// Second line, station position. For now we ignore this.
	line++;

	// Third line, units, only cubic feet per second for now
	// added cubic meters per second, and the k versions
	// which should cover all cases 5/18/01
	currentLine = trim(linesInFile[line++]);
	std::transform(currentLine.begin(),
				   currentLine.end(),
				   currentLine.begin(),
				   ::tolower);
	if (!(currentLine == "cfs" ||
		currentLine == "kcfs" ||
		currentLine == "cms" ||
		currentLine == "kcms"))
	{
		return false;
	}

	// fourth line, first line of data
	currentLine = trim(linesInFile[line++]);
	std::replace(currentLine.begin(), currentLine.end(), ',', ' ');

	if (!ParseLine(currentLine, time, velocity))
		return false;

	return true;
}


Boolean IsHydrologyFile(char *path)
{
	vector<string> linesInFile;

	if (ReadLinesInFile(path, linesInFile))
		return IsHydrologyFile(linesInFile);
	else
		return false;
}


bool IsOSSMTimeFile(vector<string> &linesInFile, short *selectedUnitsOut)
{
	long line = 0;
	string currentLine, val1Str, val2Str;

	short selectedUnits = kUndefined;
	DateTimeRec time;
	VelocityRec velocity;

	// size check
	if (linesInFile.size() < 3)
		return false;

	// First line , station name. For now we ignore this.
	line++;

	// Second line, station position. For now we ignore this.
	line++;

	// Third line, units
	// Note: We are not supporting cm/sec in gnome
	currentLine = trim(linesInFile[line++]);
	selectedUnits = StrToSpeedUnits((char *)currentLine.c_str());
	if (selectedUnits == kUndefined)
		return false;

	// fourth line, first line of data
	currentLine = trim(linesInFile[line++]);
	std::replace(currentLine.begin(), currentLine.end(), ',', ' ');

	if (!ParseLine(currentLine, time, val1Str, val2Str))
		return false;

	*selectedUnitsOut = selectedUnits;

	return true;
}

Boolean IsOSSMTimeFile(char *path, short *selectedUnitsOut)
{
	vector<string> linesInFile;

	if (ReadLinesInFile(path, linesInFile))
		return IsOSSMTimeFile(linesInFile, selectedUnitsOut);
	else
		return false;
}


// we check
// that the 3rd line contains the units
// that the 4th line can scan 7 values
bool IsOSSMHeightFile(vector<string> &linesInFile, short *selectedUnitsP)
{
	Boolean	bIsValid = true;
	string currentLine;
	long line = 0;

	short selectedUnits = kUndefined;

	DateTimeRec time;
	VelocityRec velocity;

	// first line , station name (for now we ignore this)
	line++;

	// second line, station position (for now we ignore this)
	line++;

	// third line, units
	// note we are not supporting cm/sec in gnome
	currentLine = trim(linesInFile[line++]);

	selectedUnits = StrToSpeedUnits((char *)currentLine.c_str());
	if(selectedUnits == kUndefined)
		bIsValid = false;

	// fourth line, first line of data
	currentLine = trim(linesInFile[line++]);

	std::replace(currentLine.begin(), currentLine.end(), ',', ' ');
	if (!ParseLine(currentLine, time, velocity))
		bIsValid = false;

	// finally, populate our out argument
	if (bIsValid) {
		*selectedUnitsP = selectedUnits;
	}

	return bIsValid;
}


Boolean IsOSSMHeightFile(char *path, short *selectedUnitsP)
{
	vector<string> linesInFile;

	if (ReadLinesInFile(path, linesInFile))
		return IsOSSMHeightFile(linesInFile, selectedUnitsP);
	else
		return false;
}


bool IsTimeFile(vector<string> &linesInFile)
{
	bool bIsValid = true;
	string currentLine;
	long line = 0;

	DateTimeRec time;
	VelocityRec velocity;

	currentLine = linesInFile[line++];

	std::replace(currentLine.begin(), currentLine.end(), ',', ' ');
	if (!ParseLine(currentLine, time, velocity))
		bIsValid = false;

	return bIsValid;
}


Boolean IsTimeFile(char *path)
{
	vector<string> linesInFile;

	if (ReadLinesInFile(path, linesInFile))
		return IsTimeFile(linesInFile);
	else
		return false;
}

