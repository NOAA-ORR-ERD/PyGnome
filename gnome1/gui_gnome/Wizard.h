

#ifndef		__WIZARD__
#define		__WIZARD__

#define WIZARDSAVEFILETYPE '.LFS'
#ifdef MAC
	// code goes here, it might be noce to change this to be '.LOC' on mac as well
	#define  WIZARDFILETYPE '.WIZ' 
#else
	#define  WIZARDFILETYPE '.LOC'
#endif
Boolean IsWizardFile(char* fullPathName);

void GetOurAppPathName(char* path);
Boolean ConvertIfUnixPath(char* path, char* classicPath);
Boolean IsPartialPath(char* relativePath);
void ResolvePathFromApplication(char* relativePath);
void ResolvePathFromInputFile(char *pathOfTheInputFile, char* pathToResolve); // JLM 6/8/10
OSErr ResolvePathFromCommandFile(char* pathToResolve); 
void ResolvePath(char* pathToResolve);
void ResolvePartialPathFromThisFolderPath(char* relativePath,char * thisFolderPath);

void AddDelimiterAtEndIfNeeded(char* str);

extern Boolean gSuppressDrawing;
extern Boolean gCommandFileRun;
#define STDCOMMANDFILENAME  "COMMAND.TXT"
extern char gCommandFileErrorLogPath[];
OSErr WriteErrorToCommandFileErrorLog(char *msg, long dialogID);
Boolean IsCommandFile(char *path);
OSErr DoCommandFile(char *path);

Boolean GetLineFromHdl(long lineNum1Relative,CHARH paramListHdl, char*answerStr,long maxNumChars);
Boolean GetLineFromStr(long lineNum1Relative,char* str, char*answerStr,long maxNumChars);


void GetWizButtonTitle_Next(char *str);
void GetWizButtonTitle_Previous(char *str);
void GetWizButtonTitle_Done(char *str);
void GetWizButtonTitle_Cancel(char *str);

#define kMAXNUMWIZARDDIALOGS  16
#define kMAXNUMWIZARDDIALOGITEMS  30
#define VALSTRMAX 512
typedef struct
{
	Boolean hidden;
	short type;
	CHARH hdl;
	short popupNum; // for WIZ_POPUP and WIZ_UNITS (used for bmp number for WIZ_BMP)
	short editID; // for WIZ_UNITS
	char str[256];
	char valStr[VALSTRMAX];
} WizDialogItemInfo;


enum { WIZ_POPUP = 1, WIZ_UNITS , WIZ_EDIT, WIZ_BMP, WIZ_HELPBUTTON, WIZ_WINDPOPUP };
#define WINDFLAG  -1000 // just a flag, not a resource number
#define MODELFLAG  -2000
#define ALMOSTDONEFLAG -3000
#define WNDTYPEFLAG -4000
#define WELCOMEFLAG -5000

#define WIZ_OK 1
#define WIZ_CANCEL 3
///
#define FIRSTDIALOG 1
#define LASTDIALOG 2
#define ONLYDIALOG 4 
#define CHANGEBUTTONS 8 
#define FIRSTTIMETHROUGHFILE 16 
// these are bit flags, i.e. power of 2 

#define MAX_IF_LEVEL 6

#define WIZ_VARIABLE_NAME_LENGTH 32
#define WIZ_VARIABLE_VALUE_LENGTH 64
#define MAX_NUM_WIZ_VARIABLES 50
typedef struct
{
	char name[WIZ_VARIABLE_NAME_LENGTH];
	char value[WIZ_VARIABLE_VALUE_LENGTH];
} CommandVariable;

// left hand side answers
#define LHSANSWER_TEXT_LENGTH 64
#define MAX_NUM_LHS_ANSWERS 10
typedef struct
{
	char text[LHSANSWER_TEXT_LENGTH];
	short settingsDialogResNum; // == 0 means use entire wizard sequence of dialogs
	Boolean lhs; // include on left hand side 
	Boolean print;// include on printout
} LHSAnswer;

typedef long EvaluationType;
// these are bit flags, i.e. power of 2 
#define WEAK_EVALUATION 0  
#define STRICT_EVALUATION 1
#define FORCE_EVALUATION 2
// get both by using bitwise OR operator
// STRICT_EVALUATION | FORCE_EVALUATION
/////////////////////////////////////////////////
/////////////////////////////////////////////////
class WizardFile 
{
	public:
	
		OSErr GoThroughDialogs(Boolean * userCancel,Boolean firstTimeThroughFile);
		OSErr DoCommandBlock(char* blockName);
		OSErr DoWizardCommand(char* line,long maxExpansionLen);
		void DoWizardWmsgCommands(short flags);
		OSErr CheckExpirationDate();

		OSErr SetPathName(char *pathName);
		void GetPathName(char *pathName);
		OSErr OpenFile(void); // opens current file
		OSErr OpenFile(char *pathName); // sets file then opens the file
		void CloseFile(void);
		long NumWizardDialogs(void);
		#ifdef MAC
			void DrawingRoutine(DialogPtr theDialog, short itemNum);
		#else 
			void WM_Paint(HWND theDialog);
		#endif
		Boolean IsVariableWind(void) {return this->fUseVariableWind;}

		void WizardHelpFilePath(char* helpPath);
		void DoWizardHelp(char* title);

	// constuctor	
	WizardFile(void);
	WizardFile(char *pathName);
	// destructor
	~WizardFile(void);

		void Dispose(void);
		CHARH GetResource(OSType type,short resNum,Boolean terminate);
		Boolean ResourceExists(OSType type,short resNum);
		Boolean ResourceExists(OSType type,short resNum,long* size);
		OSErr DoWizardDialogResNum(long resNum, long dialogFlags, Boolean *userCancel, CHARH prevAnswers,CHARH *userAnswers, CHARH *messages);
		OSErr CallWindDialog(long dialogFlags, Boolean *userCancel, CHARH *userAnswers, CHARH *messages);
		OSErr InitWizardDialog(short dialogResNum,DialogPtr theDialog,long dialogFlags,CHARH prevAnswers);
		Boolean	DoWizardItemHit(DialogPtr theDialog, short itemHit,CHARH *userAnswers, CHARH *messages);

		// parameter utilites
		void GetPhraseFromLine(long phraseNum1Relative,char* line, char*answerStr,long maxNumChars);
		void GetParameterString(char* key,char* line, char*answerStr,long maxNumChars);
		void GetParameterString(char* key,long lineNum1Relative,char* str, char*answerStr,long maxNumChars);
		void GetParameterString(char* key,long lineNum1Relative,CHARH paramListHdl,char*answerStr,long maxNumChars);

		OSErr WriteWizardSaveFile(BFPB *bfpb,char * path);
		OSErr SetWizardFileStuff(char* line,char* answerStr);
		OSErr SaveFileGoThoughDialogs(CHARH f);
		OSErr SaveFileAddSpills(CHARH f);

		void 	ClearLeftHandSideAnswers(void);
		short	NumLeftHandSideAnswersInList(void);
		short	NumLeftHandSideAnswersOnPrintout(void);
		Boolean	LeftHandSideAnswerSettingsItemHit(short i);
		void	LeftHandSideTextForList(short desired0RelativeLineNum,char* text);
		void	LeftHandSideTextForPrintout(short desired0RelativeLineNum,char* text);

		Boolean GetMinKilometersPerScreenInch(float *minKilometers);

	private:
		void	LeftHandSideTextForListOrPrintoutHelper(short desired1RelativeLineNum,char* text,Boolean forPrintout);
		OSErr GetFreeIndexLHSAnswers(short* index);

		short	ComboBoxIDToStaticTextID(short itemHit);
		short	StaticTextIDToComboBoxID(short itemHit);
		Boolean ModelDialogIncludedInFile(void);
		Boolean DialogExists(long dialogNum);
		//void GetStringResource(char* str,short resNum);
		//void GetStringResource(char* str,short resNum,long strNum);
		OSErr DoWizardDialog(short dialogNum, long dialogFlags, Boolean *userCancel, CHARH prevAnswers,CHARH *userAnswers, CHARH *messages);
		OSErr CallModelDialog(long dialogFlags, Boolean *userCancel, CHARH *userAnswers, CHARH *messages);

		TModelDialogVariables DefaultModelSettingsForWizard(void);
		CHARH GetResourceHelper(OSType type,short resNum,Boolean returnHandle,Boolean terminate);
		CHARH GetWMSG(short dialogResNum);
		long DialogResNum(long dialogNum,long*specialFlag);

		OSErr EvaluateBasicBlock(char* str,long maxStrLen,EvaluationType evaluationType);
		OSErr EvaluateBooleanString(char* evaluationString,long maxStrLen,Boolean * bVal);
		OSErr EvaluateNumberString(char* evaluationString,long maxStrLen,double * dVal);
		OSErr EvaluateAndTrimString(char* str,long maxStrLen,EvaluationType evaluationType);
		OSErr SubstituteDollarVariable(CHARH msgH,Boolean *hdlChanged,DialogPtr theDialog);
		OSErr RetrieveValueStrings(DialogPtr theDialog);
		CHARH RetrieveUserAnswer(DialogPtr theDialog);
		void SetPopupItemSelected(DialogPtr theDialog, short itemNum, short itemSelected);
		short GetPopupItemSelected(DialogPtr theDialog, short itemNum);
		void DoPopupShowHide(DialogPtr theDialog);
		void	InitWizardFile(void);
		void	Clear_Dialog(void);
		void	Dispose_Dialog(void);
		OSErr SaveAnswers(void);
		void RestoreAnswers(void);
		void Dispose_SavedAnswers(void);
		void Dispose_Answers(void);
		
		void ClearCommandVariables(void);
		
		
	//// instance variables
		short 	fIfLevel;	
		Boolean 	fIfValue[MAX_IF_LEVEL+1];
		CommandVariable fCommandVariable[MAX_NUM_WIZ_VARIABLES];
		//
		LHSAnswer fLHSAnswer[MAX_NUM_LHS_ANSWERS];
		//
		
		
		// version identifiers from the 
		char  fLocationFileIdStr[64];
		short fLocationFileFormat;
		short fLocationFileVersion;
		long  fLocationFileExpirationDate;
		
		float fMinKilometersPerInch; // zero or negative means not limited (by the location file) 
		// note: fMinKilometersPerInch need not be saved in the save file since it will be re-read from the location file
		
		CHARH 	fText10000Hdl;
		char		fPathName[256];
		Boolean	fIsOpen;
		#ifdef IBM
			HINSTANCE fInstDLL;
		#else
			long fResRefNum;
		#endif
		
		//TModelDialogVariables fModelDialogVariables;
		//TModelDialogVariables fModelSavedDialogVariables;
		///
		Boolean fUseVariableWind; // use variable wind dialog vs. constant wind dialog
		Boolean fUseVariableWindSaved; 
		///
		//TConstantMoverVariables fConstantWindVariables;
		///
		//TimeValuePairH fVariableWindAnswer; // answers to variable wind dialog
		//////////////////////
		CHARH fAnswer[kMAXNUMWIZARDDIALOGS];
		CHARH fSavedAnswer[kMAXNUMWIZARDDIALOGS]; // used to hold answers in case user cancels

		CHARH fMessage[kMAXNUMWIZARDDIALOGS];
		CHARH fSavedMessage[kMAXNUMWIZARDDIALOGS];
		///////////////////////
		
	// instance vaiables used when doing a dialog
		short fDialogResNum;
		WizDialogItemInfo  fItem[kMAXNUMWIZARDDIALOGITEMS];
		///

};

#include "LocaleWizard_c.h"
class LocaleWizard : virtual public LocaleWizard_c, public TClassID
{
	public:
		//OSErr CheckAndPassOnMessage(TModelMessage * model);

		void OpenMenuHit(void);
		void OpenWizFile(char* path);
		long CloseMenuHit(void);
		Boolean OKToChangeWizardMode(long oldMode,long newMode,Boolean * closeFile);
		OSErr SaveAsMenuHit(void);
		Boolean QuitMenuHit(void);
		Boolean StartUp(void);
		void InvokeWizardMenuHit(void);
		Boolean	HaveOpenWizardFile(void) { return (this->fCurrentFile != nil);  }
		void GetLocationFileFullPathName(char *pathName);
		void GetLocationFileName(char* name, Boolean chopExtension);
		void WizardHelpFilePath(char* helpPath);
		Boolean GetMinKilometersPerScreenInch(float *minKilometers);

		OSErr 		SettingsItem 	(ListItem item) ;
		long 		GetListLength 	();
		Boolean 	ListClick 	  	(ListItem item, Boolean inBullet, Boolean doubleClick);
		Boolean 	FunctionEnabled (ListItem item, short buttonID);
		ListItem 	GetNthListItem 	(long n, short indent, short *style, char *text);

		// constuctor	
		LocaleWizard();
		// destructor
		~LocaleWizard();
	
		// just the default stuff for a TClassID
		ClassID 	GetClassID 	() { return TYPE_UNDENTIFIED; }
		void		Dispose 	();

		Boolean		IsDirty  	() { return bDirty;  }
		Boolean		IsOpen   	() { return bOpen;   }
		Boolean		IsActive 	() { return bActive; }
		void		SetDirty  (Boolean bNewDirty)  { bDirty  = bNewDirty; }
		void		SetOpen   (Boolean bNewOpen)   { bOpen   = bNewOpen;  }
		void		SetActive (Boolean bNewActive) { bActive = bNewActive;}
				
		// list display methods: base class functionality
		OSErr 		UpItem 			(ListItem item) { return 0; }
		OSErr 		DownItem 		(ListItem item) { return 0; }
		OSErr 		AddItem 		(ListItem item) { return 0; }
		OSErr 		DeleteItem 		(ListItem item) { return 0; }

		OSErr ReadFileContentsFromResource(char* path,CHARHP handle,Boolean terminate);
		OSErr ReadSectionOfFileFromResource(char* path,char* ptr,long maxLength,long offset);
		OSErr MyGetFileSizeFromResource(CHARPTR pathName, LONGPTR size);

		// parameter utilites
		void GetPhraseFromLine(long phraseNum1Relative,char* line, char*answerStr,long maxNumChars);
		void GetParameterString(char* key,char* line, char*answerStr,long maxNumChars);
		void GetParameterString(char* key,long lineNum1Relative,char* str, char*answerStr,long maxNumChars);
		void GetParameterString(char* key,long lineNum1Relative,CHARH paramListHdl,char*answerStr,long maxNumChars);

		short	NumLeftHandSideAnswersOnPrintout(void);
		void	LeftHandSideTextForPrintout(short desired0RelativeLineNum,char* text);
		
	private:

		ListItem GetNthListItemOrListLength(long n, short indent, short *style, char *text,long *listLength);
		void DisposeCurrentFile(void);
		void OffToSeeTheWizard(Boolean ask,char* providedPath);

		WizardFile *fCurrentFile;

};

 
// Broadcast MESSAGES 
// note: add code to WizStringToMessageCode() when adding a new message type
// wizard messages are in the 1's
// model messages are in the 1000's
enum{M_SETFIELD = 1,M_CREATEMOVER,M_CREATEMAP,
	// messages to support TAP/ command file
	M_CLOSE, M_CLEARSPILLS, M_CLEARWINDS, M_RESET, M_OPEN, M_RUNSPILL, M_QUIT,
	// messages to support testing/development
	M_STARTTIMER, M_STOPTIMER, M_SAVE,
	// messages to split up M_RUNSPILL
	M_RUN,M_CREATESPILL
};

extern Boolean gInWizard;


#endif
