

#ifndef		__TSHIOTIMEVALUE__
#define		__TSHIOTIMEVALUE__

#include "Basics.h"
#include "TypeDefs.h"
#include "Shio.h"
#include "ShioTimeValue_c.h"

class TShioTimeValue : virtual public ShioTimeValue_c, public TOSSMTimeValue
{

	public:
								TShioTimeValue (TMover *theOwner);
								TShioTimeValue (TMover *theOwner,TimeValuePairH tvals);
		virtual					   ~TShioTimeValue () { this->Dispose (); }
		virtual OSErr 			MakeClone(TShioTimeValue **clonePtrPtr);
		virtual OSErr 			BecomeClone(TShioTimeValue *clone);
		//virtual void			Dispose ();


		// I/O methods
		virtual OSErr 			Read  (BFPB *bfpb);  // read from current position
		virtual OSErr 			Write (BFPB *bfpb);  // write to  current position

		virtual long 			GetListLength (); 
		virtual ListItem 		GetNthListItem 	(long n, short indent, short *style, char *text);
		virtual Boolean 		ListClick	(ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 		FunctionEnabled(ListItem item, short buttonID);
		virtual OSErr			ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance);
		virtual OSErr 			CheckAndPassOnMessage(TModelMessage *message);
};

Boolean IsShioFile(char* path);
char* GetKeyedLine(CHARH f, char*key, long lineNum, char *strLine);

#endif