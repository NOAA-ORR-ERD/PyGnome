/*
 *  TClassID.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TClassID__
#define __TClassID__

#include "Earl.h"
#include "TypeDefs.h"
#include "GuiTypeDefs.h"
#include "ClassID_c.h"


/*#ifndef pyGNOME
typedef long ClassID;

const ClassID TYPE_UNDENTIFIED	 	= 0;
const ClassID TYPE_MODEL	 		= 100;
const ClassID TYPE_LELISTLIST	 	= 200;
const ClassID TYPE_MAPLIST 			= 201;
const ClassID TYPE_MOVERLIST	 	= 202;
const ClassID TYPE_LELIST 			= 300;
const ClassID TYPE_OSSMLELIST		= 301;
const ClassID TYPE_SPRAYLELIST		= 302;
const ClassID TYPE_CDOGLELIST		= 303;
const ClassID TYPE_MAP 				= 400;
const ClassID TYPE_OSSMMAP			= 401;
const ClassID TYPE_VECTORMAP		= 402;
const ClassID TYPE_PTCURMAP			= 403;
const ClassID TYPE_COMPOUNDMAP		= 404;
const ClassID TYPE_MAP3D			= 405;
const ClassID TYPE_GRIDMAP			= 406;

const ClassID TYPE_MOVER 			= 500;
const ClassID TYPE_RANDOMMOVER		= 501;
const ClassID TYPE_CATSMOVER		= 502;
const ClassID TYPE_WINDMOVER		= 503;
//const ClassID TYPE_CONSTANTMOVER	= 504; // no longer supported, replaced by an enhanced TYPE_WINDMOVER, JLM 2/18/00
const ClassID TYPE_COMPONENTMOVER	= 505;
const ClassID TYPE_PTCURMOVER		= 506;
const ClassID TYPE_CURRENTMOVER		= 507;
const ClassID TYPE_RANDOMMOVER3D	= 508;
const ClassID TYPE_CATSMOVER3D		= 509;
const ClassID TYPE_GRIDCURMOVER		= 510;
const ClassID TYPE_NETCDFMOVER		= 511;
const ClassID TYPE_NETCDFMOVERCURV	= 512;
const ClassID TYPE_NETCDFMOVERTRI	= 513;
const ClassID TYPE_NETCDFWINDMOVER	= 514;
const ClassID TYPE_GRIDWNDMOVER	= 515;
const ClassID TYPE_NETCDFWINDMOVERCURV	= 516;
const ClassID TYPE_TRICURMOVER	= 517;
const ClassID TYPE_TIDECURCYCLEMOVER	= 518;
const ClassID TYPE_COMPOUNDMOVER	= 519;
const ClassID TYPE_ADCPMOVER		= 520;
const ClassID TYPE_GRIDCURRENTMOVER	= 521;
const ClassID TYPE_GRIDWINDMOVER	= 522;

const ClassID TYPE_TIMEVALUES		= 600;
const ClassID TYPE_OSSMTIMEVALUES	= 601;
const ClassID TYPE_SHIOTIMEVALUES	= 602;
const ClassID TYPE_ADCPTIMEVALUES	= 602;
const ClassID TYPE_WEATHERER		= 700;
const ClassID TYPE_OSSMWEATHERER	= 701;
const ClassID TYPE_GRIDVEL			= 800;
const ClassID TYPE_RECTGRIDVEL		= 801;
const ClassID TYPE_TRIGRIDVEL		= 802;
const ClassID TYPE_TRIGRIDVEL3D		= 803;
const ClassID TYPE_TIMEGRIDVEL		= 810;
const ClassID TYPE_TIMEGRIDVELRECT		= 811;
const ClassID TYPE_TIMEGRIDVELCURV		= 812;
const ClassID TYPE_TIMEGRIDVELTRI		= 813;
const ClassID TYPE_TIMEGRIDWINDRECT		= 814;
const ClassID TYPE_TIMEGRIDWINDCURV		= 815;
const ClassID TYPE_TIMEGRIDCURRECT		= 816;
const ClassID TYPE_TIMEGRIDCURTRI		= 817;
const ClassID TYPE_CMAPLAYER 		= 901; //JLM

const ClassID TYPE_OVERLAY	= 910; //JLM
const ClassID TYPE_NESDIS_OVERLAY	= 920; //JLM
const ClassID TYPE_BUOY_OVERLAY	= 930; //JLM
const ClassID TYPE_BP_BUOY_OVERLAY	= 931; //JLM
const ClassID TYPE_SLDMB_BUOY_OVERLAY	= 932; //JLM
const ClassID TYPE_OVERFLIGHT_OVERLAY	= 940; //JLM
#endif*/

class TModelMessage;

class TClassID : virtual public ClassID_c
{
	
public:
	UNIQUEID			fUniqueID;

	TClassID ();
	virtual			   ~TClassID () { Dispose (); }
	
	virtual ClassID 	GetClassID 	() { return TYPE_UNDENTIFIED; }
	virtual Boolean		IAm(ClassID id) { return FALSE; }
	Boolean 			GetSelectedListItem(ListItem *item);
	Boolean 			SelectedListItemIsMine(void);
	virtual Boolean 	IAmEditableInMapDrawingRect(void);
	virtual Boolean 	IAmCurrentlyEditableInMapDrawingRect(void);
	virtual Boolean 	UserIsEditingMeInMapDrawingRect(void);
	virtual void	 	StartEditingInMapDrawingRect(void);
	virtual OSErr 		StopEditingInMapDrawingRect(Boolean *deleteMe);
	
	virtual OSErr 		MakeClone(TClassID **clonePtrPtr);
	virtual OSErr 		BecomeClone(TClassID *clone);
	
	UNIQUEID			GetUniqueID () { return fUniqueID; }
	Boolean 			MatchesUniqueID(UNIQUEID uid);	
	
	virtual OSErr 		Read  (BFPB *bfpb);  			
	virtual OSErr 		Write (BFPB *bfpb); 			
	
	virtual long 		GetListLength 	();				 
	virtual Boolean 	ListClick 	  	(ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID) { return FALSE; }
	virtual ListItem 	GetNthListItem 	(long n, short indent, short *style, char *text);
	virtual OSErr 		UpItem 			(ListItem item) { return 0; }
	virtual OSErr 		DownItem 		(ListItem item) { return 0; }
	virtual OSErr 		AddItem 		(ListItem item) { return 0; }
	virtual OSErr 		SettingsItem 	(ListItem item) { return 0; }
	virtual OSErr 		DeleteItem 		(ListItem item) { return 0; }
	
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage * model);
	
};

#endif
