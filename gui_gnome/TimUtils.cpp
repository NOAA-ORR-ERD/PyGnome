
#include "Cross.h"

Seconds RetrieveTime(DialogPtr dialog, short monthItem)
{
	DateTimeRec time;
	Seconds seconds;
	
	time.month = EditText2Long(dialog, monthItem);
	time.day = EditText2Long(dialog, monthItem + 1);
	time.year = EditText2Long(dialog, monthItem + 2);
	time.hour = EditText2Long(dialog, monthItem + 3);
	time.minute = EditText2Long(dialog, monthItem + 4);
	time.second = 0;
	if (time.year < 1900)						// two digit date, so fix it
	{
		if (time.year >= 40 && time.year <= 99)	// JLM
			time.year += 1900;
		else
			time.year += 2000;					// correct for year 2000 (00 to 40)
	}

	if (time.month && time.day && time.year)
		DateToSeconds (&time, &seconds);
	else
		seconds = 0;

	return seconds;
}

Seconds RetrievePopTime(DialogPtr dialog, short monthItem,OSErr * err)
{
	DateTimeRec time;
	Seconds seconds = 0;
	short minDay = 1, maxDay;
	char msg[256] = "";
	
	*err = 0;
	
	time.month = GetPopSelection (dialog, monthItem);
	time.year = (FirstYearInPopup()  - 1) + GetPopSelection(dialog, monthItem + 2);

	// retrieve the day
	time.day = EditText2Long(dialog, monthItem + 1);
	switch(time.month)
	{
		case 1: case 3: case 5: case 7: case 8: case 10: case 12: maxDay = 31; break;
		case 2: //feb 
			if( (time.year % 4 == 0 && time.year % 100 != 0) || time.year % 400 == 0)  maxDay=29;
			else maxDay = 28;
			break;
		default: maxDay = 30; break;
	}
	if(time.day < minDay) strcpy(msg,"Your day value must be greater than 0.");
	else if (time.day > maxDay )
	{
		char *format = "Your day value cannot exceed %d."; 
		sprintf(msg,format,maxDay);
	}
	if(msg[0])
	{
		*err = 1;
		printError(msg);
		MySelectDialogItemText(dialog, monthItem + 1, 0, 100);
		return 0;
	}
	
	
	time.hour = EditText2Long(dialog, monthItem + 3);
	if(time.hour >= 24)
	{
		*err = 1;
		printError("Your hour value must be less than 24.");
		MySelectDialogItemText(dialog, monthItem + 3, 0, 100);
		return 0;
	}
	
	time.minute = EditText2Long(dialog, monthItem + 4);
	if(time.hour >= 60)
	{
		*err = 1;
		printError("Your minute value must be less than 60.");
		MySelectDialogItemText(dialog, monthItem + 4, 0, 100);
		return 0;
	}

	time.second = 0;

	if (time.month && time.day && time.year)
		DateToSeconds (&time, &seconds);
	else
		seconds = 0;

	return seconds;
}

void DisplayTime(DialogPtr dialog, short monthItem, Seconds seconds)
{
	char num[20];
	DateTimeRec time;
	
	SecondsToDate (seconds, &time);
	Float2EditText(dialog, monthItem, time.month, 0);
	Float2EditText(dialog, monthItem + 1, time.day, 0);
	Float2EditText(dialog, monthItem + 2, time.year, 0);
	Float2EditText(dialog, monthItem + 3, time.hour, 0);
	sprintf(num, "%02hd", time.minute);
	mysetitext(dialog, monthItem + 4, num);
}
