

#ifndef		__TSHIOTIMEVALUE__
#define		__TSHIOTIMEVALUE__

#include "Shio.h"

//enum { WIZ_POPUP = 1, WIZ_UNITS , WIZ_EDIT, WIZ_BMP, WIZ_HELPBUTTON };

#define MAXNUMSHIOYEARS  20
typedef struct
{
	short year;// 1998, etc
	YEARDATAHDL yearDataHdl;
} ShioYearInfo;


YEARDATAHDL GetYearData(short year);

class TShioTimeValue : public TOSSMTimeValue
{

	public:
								TShioTimeValue (TMover *theOwner);
								TShioTimeValue (TMover *theOwner,TimeValuePairH tvals);
							   ~TShioTimeValue () { this->Dispose (); }
		virtual OSErr 			MakeClone(TClassID **clonePtrPtr);
		virtual OSErr 			BecomeClone(TClassID *clone);
		virtual OSErr			InitTimeFunc ();

		virtual void			Dispose ();


		// I/O methods
		virtual OSErr 			Read  (BFPB *bfpb);  // read from current position
		virtual OSErr 			Write (BFPB *bfpb);  // write to  current position

		virtual long 			GetListLength (); 
		virtual ListItem 		GetNthListItem 	(long n, short indent, short *style, char *text);
		virtual Boolean 		ListClick	(ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 		FunctionEnabled(ListItem item, short buttonID);

		virtual OSErr 			CheckAndPassOnMessage(TModelMessage *message);

		void 					InitInstanceVariables(void);

		long 					I_SHIOHIGHLOWS(void);
		long 					I_SHIOEBBFLOODS(void);


};

Boolean IsShioFile(char* path);

#endif