
#include "CROSS.h"
#include "Wizard.h"
#include <sys/stat.h>
#include <fstream>
#include <ios>

using std::fstream;
using std::ios;

typedef struct {
		short drive;
		short vRefNum;
		char name[32];
		Boolean cd;
		Boolean wasUsed;
	} VolumeInfo;

static VolumeInfo volumes[26];

Boolean FileExists(short vRefNum, long dirID, CHARPTR filename)
{
	OSErr err = 0;
	char filePath[256], tempFilePath[256];
	////////
	if(model && model->fWizard && model->fWizard->PathIsWizardResource(filename))
	{	// wizard support, JLM 7/25/00
		return TRUE;
	}
	/////////////////
	if (vRefNum!=0 || dirID!=0) 
	{
		sprintf(tempFilePath, "%s%s", TATFolder, filename);
	}
	else
		strcpy(tempFilePath, filename);
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) tempFilePath, filePath, kMaxNameLen) ;
#else
		strcpy(filePath,tempFilePath);
#endif

	struct stat buffer;
 	return (stat(filePath,&buffer)==0);
}

/*Boolean FileExists(short vRefNum, long dirID, CHARPTR fileName)
{
	FInfo fInfo;
	long err;
	
	////////
	if(model && model->fWizard && model->fWizard->PathIsWizardResource(fileName))
	{	// wizard support, JLM 7/25/00
		return TRUE;
	}
	/////////////////
	if (err = hgetfinfo(vRefNum, dirID, fileName, &fInfo))
		return FALSE;
	
	return TRUE;
}*/

Boolean FolderExists(short vRefNum, long dirID, CHARPTR folderName)
#ifdef MAC
{
	long dummy, err;
	
	if (err = dircreate(vRefNum, dirID, folderName, &dummy))
	{
		if (err == -48 )// Duplicate filename and version so already exists
			return TRUE;
		else return FALSE; // some other error trying to create the folder
	}
	
	hdelete(vRefNum, dirID, folderName);
	
	return FALSE;
}
#else
{
	char copy[256], *p;
	
	strcpy(copy, folderName);
	p = strrchr(copy, DIRDELIMITER);
	if (p && p[1] == 0)
		p[0] = 0; // remove terminating backslash to treat folder name as file name
	
	if (NameIsIBMRoot(copy)) return TRUE; // a root directory is a folder
	
	return FileExists(vRefNum, dirID, copy);
}
#endif

OSErr AddFolderIfMissing(long vRefNum, long parentDirID, char *name, long *dirID)
{ // note: we do not report error to user
	OSErr err = noErr;
	if (FolderExists(vRefNum, parentDirID, name)) {
		(*dirID) = GetFolderID(vRefNum, parentDirID, name);
		if ((*dirID) == -1) err = -1;
	}
	else
		err = dircreate(vRefNum, parentDirID, name, dirID);
	
	return err;
}



OSErr hgetvol(CHARPTR volName, SHORTPTR vRefNum, LONGPTR dirID)
#ifdef MAC
{
   OSErr result;
   
   result = HGetVol((StringPtr)volName, vRefNum, dirID);
   my_p2cstr((StringPtr)volName);
   return result;
}
#else
{
	char path[256]; // fileName[32], *commandLine[] = GetCommandLine();
	
	if (!GetCurrentDirectory(256, path)) return -1;
	
	if (path[strlen(path) - 1] != '\\') strcat(path, "\\");
	
	// strcpy(path, commandLine);
	// SplitPathFile(path, fileName);
	
	// get or create the dirID for the given path
	if (!GetIDFromPath(dirID, path, TRUE))
		return -1;
	
	// get then drive letter (first character) and assign it a number
	// A = 0, B = 1, etc.

	// network code added  6/25/98 JLM
	*vRefNum = VRefNumFromIBMName(path);  
	
	if (volName)
		strcpy(volName, path);
	
	return 0;
}
#endif

OSErr hsetvol(CHARPTR volName, short vRefNum, long dirID)
#ifdef MAC
{
   OSErr result;
   char localStr[256];
   
   if (volName) {
   		strcpy(localStr,volName);
		my_c2pstr(localStr);
   		result = HSetVol((StringPtr)localStr, vRefNum, dirID);
	}
	else
   		result = HSetVol(0, vRefNum, dirID);
   
   
   //if (volName) c2pstr(volName);
  // result = HSetVol((StringPtr)volName, vRefNum, dirID);
  // if (volName) p2cstr((StringPtr)volName);
   
   return result;
}
#else
{
   char pathName[256];

   if (!GetDirectoryFromID(dirID, pathName)) return -1;
   
   return SetCurrentDirectory(pathName) ? 0 : -1;
}
#endif

OSErr hopen(short vRefNum, long dirID, CHARPTR fileName, char permission, SHORTPTR refNum)
#ifdef MAC
{
   OSErr result;
   char localStr[256];
   
   strcpy(localStr,fileName);
   my_c2pstr(localStr);
   result = HOpen(vRefNum, dirID, (StringPtr)localStr, permission, refNum);
  // c2pstr(fileName);
  // result = HOpen(vRefNum, dirID, (StringPtr)fileName, permission, refNum);
  // p2cstr((StringPtr)fileName);
   
   return result;
}
#else
{
   DWORD perm;
   char pathName[256];

   if (!ResolveFileDirectory(vRefNum, dirID, fileName, pathName)) return -1;
   if ((permission == fsRdPerm) || 
       ((permission == fsCurPerm) & (GetFileAttributes(pathName) & FILE_ATTRIBUTE_READONLY)))
      perm = GENERIC_READ;
   else
      perm = GENERIC_READ | GENERIC_WRITE;
   *refNum = (short)CreateFile (pathName, perm, FILE_SHARE_READ, NULL, OPEN_EXISTING, 
                                FILE_ATTRIBUTE_NORMAL, NULL);
   if ((*refNum) == (long)INVALID_HANDLE_VALUE) return -1;
   SetFilePointer((HANDLE)(*refNum), 0, NULL, FILE_BEGIN);
   
   return 0;
}
#endif

OSErr hopendf(short vRefNum, long dirID, CHARPTR fileName, char permission, SHORTPTR refNum)
#ifdef MAC
{
   OSErr result;   
   char localStr[256];
   
   strcpy(localStr,fileName);
   my_c2pstr(localStr);
  // my_c2pstr(fileName);
  // result = HOpenDF(vRefNum, dirID, (StringPtr)fileName, permission, refNum);
   result = HOpenDF(vRefNum, dirID, (StringPtr)localStr, permission, refNum);
  // my_p2cstr((StringPtr)fileName);
   if(result != noErr) *refNum = 0;
   
   return result;
}
#else
{
   DWORD perm, error;
   char pathName[256];

   if (!ResolveFileDirectory(vRefNum, dirID, fileName, pathName)) return -1;
   if ((permission == fsRdPerm) || 
       ((permission == fsCurPerm) & (GetFileAttributes(pathName) & FILE_ATTRIBUTE_READONLY)))
      perm = GENERIC_READ;
   else
      perm = GENERIC_READ | GENERIC_WRITE;
   *refNum = (short)CreateFile (pathName, perm, FILE_SHARE_READ, NULL, OPEN_EXISTING, 
                                FILE_ATTRIBUTE_NORMAL, NULL);
   
   if (*refNum == (short)INVALID_HANDLE_VALUE) {
      error = GetLastError();
      return -1;
   }
   SetFilePointer((HANDLE)(*refNum), 0, NULL, FILE_BEGIN);
   
   return 0;
}
#endif

OSErr hopenrf(short vRefNum, long dirID, CHARPTR fileName, char permission, SHORTPTR refNum)
#ifdef MAC
{
   OSErr result;
   char localStr[256];
   
   strcpy(localStr,fileName);
   my_c2pstr(localStr);
   
   result = HOpenRF(vRefNum, dirID, (StringPtr)localStr, permission, refNum);

  // c2pstr(fileName);
  // result = HOpenRF(vRefNum, dirID, (StringPtr)fileName, permission, refNum);
  // p2cstr((StringPtr)fileName);
   
   return result;
}
#else
{
	return 0;
}
#endif

short hopenresfile(short vRefNum, long dirID, CHARPTR fileName, char permission)
#ifdef MAC
{
   short result;
   char localStr[256];
   
   strcpy(localStr,fileName);
   my_c2pstr(localStr);
  
   result = HOpenResFile(vRefNum, dirID, (StringPtr)localStr, permission);
  // c2pstr(fileName);
  // result = HOpenResFile(vRefNum, dirID, (StringPtr)fileName, permission);
  // p2cstr((StringPtr)fileName);
   
   return result;
}
#else
{
	return 0;
}
#endif

OSErr hcreate(short vRefNum, long dirID, CHARPTR fileName, OSType creator, OSType fileType)
#ifdef MAC
{
   OSErr result;
   char localStr[256];
   
   strcpy(localStr,fileName);
   my_c2pstr(localStr);
   
   result = HCreate(vRefNum, dirID, (StringPtr)localStr, creator, fileType);
  // c2pstr(fileName);
  // result = HCreate(vRefNum, dirID, (StringPtr)fileName, creator, fileType);
 //  p2cstr((StringPtr)fileName);
   
   return result;
}
#else
{
   char pathName[256];
   HANDLE hFile;

   if (!ResolveFileDirectory(vRefNum, dirID, fileName, pathName)) return -1;
   hFile = CreateFile (pathName, GENERIC_READ | GENERIC_WRITE,
                       FILE_SHARE_READ, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
   if (hFile == INVALID_HANDLE_VALUE) return -1;
   CloseHandle(hFile);
   return 0;
}
#endif

void hcreateresfile(short vRefNum, long dirID, CHARPTR fileName)
#ifdef MAC
{
   char localStr[256];
   
   strcpy(localStr,fileName);
   my_c2pstr(localStr);

   HCreateResFile(vRefNum, dirID, (StringPtr)localStr);
  // c2pstr(fileName);
 //  HCreateResFile(vRefNum, dirID, (StringPtr)fileName);
  // p2cstr((StringPtr)fileName);
}
#else
{
}
#endif

OSErr dircreate(short vRefNum, long parentDirID, CHARPTR directoryName, LONGPTR createdDirID)
#ifdef MAC
{
	OSErr result;
    char localStr[256];
   	
	if (parentDirID) {
		if (directoryName && strlen(directoryName) > 31) {
			directoryName[31] = 0;
		}
		ZapGremlins(directoryName);
	}
	
    strcpy(localStr,directoryName);
    my_c2pstr(localStr);
	result = DirCreate(vRefNum, parentDirID, (StringPtr)localStr, createdDirID);
	//c2pstr(directoryName);
	//result = DirCreate(vRefNum, parentDirID, (StringPtr)directoryName, createdDirID);
	//p2cstr((StringPtr)directoryName);
	
	return result;
}
#else
{
	OSErr result = 0;
	char pathName[256];
	
	if (!parentDirID) {
		(*createdDirID) = 0;
		
		return (_mkdir(directoryName));
	}
	
	if (directoryName && strlen(directoryName) > 8) {
		directoryName[8] = 0;
	}
	ZapGremlins(directoryName);
	
	if (!ResolveFileDirectory(vRefNum, parentDirID, directoryName, pathName)) return -1;
	if (_mkdir(pathName)) result = -1;
	strcat(pathName, "\\");
	
	if (!GetIDFromPath(createdDirID, pathName, TRUE)) return -1;
	
	return result;
}
#endif

OSErr hdelete(short vRefNum, long dirID, CHARPTR fileName)
#ifdef MAC
{
   OSErr result;
   char localStr[256];
   
   strcpy(localStr,fileName);
   my_c2pstr(localStr);
   result = HDelete(vRefNum, dirID, (StringPtr)localStr);
   
   //c2pstr(fileName);
  // result = HDelete(vRefNum, dirID, (StringPtr)fileName);
  // p2cstr((StringPtr)fileName);
   
   return result;
}
#else
{
	char pathName[256];
	DWORD error;
	
	if (!ResolveFileDirectory(vRefNum, dirID, fileName, pathName)) return -1;
	if (!DeleteFile(pathName) && !RemoveDirectory(pathName)) {
		error = GetLastError();
		return -1;
	}
	
	return 0;
}
#endif

OSErr hgetfinfo(short vRefNum, long dirID, CHARPTR fileName, FINFOPTR fndrInfo)
#ifdef MAC
{
   OSErr result;
   char localStr[256];
   
   strcpy(localStr,fileName);
   my_c2pstr(localStr);
   
   result = HGetFInfo(vRefNum, dirID, (StringPtr)localStr, fndrInfo);
  // c2pstr(fileName);
  // result = HGetFInfo(vRefNum, dirID, (StringPtr)fileName, fndrInfo);
 //  p2cstr((StringPtr)fileName);
   
   return result;
}
#else
{
	char pathName[256];
	WIN32_FIND_DATA fileData;
	HANDLE findHandle;
	
	fndrInfo->fdType = 0;
	fndrInfo->fdCreator = 0;
	fndrInfo->fdFlags = 0;
	SetPt(&fndrInfo->fdLocation, 0, 0);
	fndrInfo->fdFldr = 0;
	
	if (!ResolveFileDirectory(vRefNum, dirID, fileName, pathName)) return -1;
	
	if ((findHandle = FindFirstFile(pathName, &fileData)) != INVALID_HANDLE_VALUE) {
		FindClose(findHandle);
		return 0;
	}
	
	return -1;
}
#endif

OSErr hsetfinfo(short vRefNum, long dirID, CHARPTR fileName, FINFOPTR fndrInfo)
#ifdef MAC
{
   OSErr result;
   char localStr[256];
   
   strcpy(localStr,fileName);
   my_c2pstr(localStr);
   
   result = HSetFInfo(vRefNum, dirID, (StringPtr)localStr, fndrInfo);
  // c2pstr(fileName);
 //  result = HSetFInfo(vRefNum, dirID, (StringPtr)fileName, fndrInfo);
  // p2cstr((StringPtr)fileName);
   
   return result;
}
#else
{
	return 0;
}
#endif

OSErr hsetflock(short vRefNum, long dirID, CHARPTR fileName)
#ifdef MAC
{
   OSErr result;
   char localStr[256];
   
   strcpy(localStr,fileName);
   my_c2pstr(localStr);
   result = HSetFLock(vRefNum, dirID, (StringPtr)localStr);
   
  // c2pstr(fileName);
  // result = HSetFLock(vRefNum, dirID, (StringPtr)fileName);
  // p2cstr((StringPtr)fileName);
   
   return result;
}
#else
{
	return 0;
}
#endif

OSErr hrstflock(short vRefNum, long dirID, CHARPTR fileName)
#ifdef MAC
{
   OSErr result;
   char localStr[256];
   
   strcpy(localStr,fileName);
   my_c2pstr(localStr);
   result = HRstFLock(vRefNum, dirID, (StringPtr)localStr);
   
  // c2pstr(fileName);
  // result = HRstFLock(vRefNum, dirID, (StringPtr)fileName);
	//p2cstr((StringPtr)fileName);
   
   return result;
}
#else
{
	return 0;
}
#endif

OSErr hrename(short vRefNum, long dirID, CHARPTR oldName, CHARPTR newName)
#ifdef MAC
{
   OSErr result;
   char localNewNameStr[256];
   char localOldNameStr[256];
   
   strcpy(localNewNameStr,newName);
   strcpy(localOldNameStr,oldName);
   my_c2pstr(localNewNameStr);
   my_c2pstr(localOldNameStr);
	
   result = HRename(vRefNum, dirID, (StringPtr)localOldNameStr, (StringPtr)localNewNameStr);
  // c2pstr(oldName);
 //  c2pstr(newName);
  // result = HRename(vRefNum, dirID, (StringPtr)oldName, (StringPtr)newName);
 //  p2cstr((StringPtr)oldName);
  // p2cstr((StringPtr)newName);
   
   return result;
}
#else
{
	char oldPathName[256], newPathName[256];
	OSErr result;
#pragma unused(vRefNum)
	
	if (!ResolveFileDirectory(vRefNum, dirID, oldName, oldPathName)) return -1;
	if (!ResolveFileDirectory(vRefNum, dirID, newName, newPathName)) return -1;
	
	result = MoveFile(oldPathName, newPathName) ? 0 : -1;
	
	if (!result) {
		strcat(oldPathName, "\\");
		strcat(newPathName, "\\");
		RenameDirectoryInTable(oldPathName, newPathName);
	}
	
	return result;
}
#endif

OSErr catmove(short vRefNum, long dirID, CHARPTR oldName, long newDirID, CHARPTR newName)
#ifdef MAC
{
   OSErr result;
   char localNewNameStr[256];
   char localOldNameStr[256];
   
   strcpy(localNewNameStr,newName);
   strcpy(localOldNameStr,oldName);
   my_c2pstr(localNewNameStr);
   my_c2pstr(localOldNameStr);
	
   result = CatMove(vRefNum, dirID, (StringPtr)localOldNameStr, newDirID, (StringPtr)localNewNameStr);
  // c2pstr(oldName);
  // c2pstr(newName);
  // result = CatMove(vRefNum, dirID, (StringPtr)oldName, newDirID, (StringPtr)newName);
  // p2cstr((StringPtr)oldName);
 //  p2cstr((StringPtr)newName);
   
   return result;
}
#else
{
   // (MIKE) (not yet used) if such a thing is possible, this function moves the
   // directory or file specified by oldName to the path specified by newName

   return 0;
}
#endif

// I don't think this is used, would have to update GetVInfo to use PBHGetVInfoSync for carbon
Boolean ScanVolumes(Boolean initialize, Boolean cdOnly, short *changes)
#ifdef MAC
{
#if TARGET_API_MAC_CARBON
	return false;
#else
	Boolean locked, ejectable, cd, change = initialize;
	char vName[256], flags;
	short vRefNum, i;
	OSErr err = 0;
	double freeBytes;
	DrvQElPtr drives;
	VolumeInfo oldVolumes[26];
	
	if (!initialize)
		for (i = 0 ; i < 26 ; i++)
			oldVolumes[i] = volumes[i];
	
	for (i = 0 ; i < 26 ; i++) {
		volumes[i].drive = 0;
		volumes[i].vRefNum = 0;
		volumes[i].name[0] = 0;
		if (initialize) {
			volumes[i].cd = FALSE;
			volumes[i].wasUsed = FALSE;
		}
	}
	
	drives = (DrvQElPtr)(GetDrvQHdr()->qHead);
	for (i = 0 ; drives && i < 26 ; drives = (DrvQElPtr)drives->qLink, i++) {
		// I think we can't tell if it's actually a CD on the Mac, so we'll
		// check if it's /* locked and */ ejectable (see New IM page 2-85).
		if (initialize) {
			flags = *(char *)(((long)drives) - 4);
			locked = (flags & 0x40) != 0;
			flags = *(char *)(((long)drives) - 3);
			ejectable = flags == 0 || flags == 1 || flags == 2;
			cd = /* locked && */ ejectable;
			volumes[i].cd = cd;
		}
		if (cdOnly && !volumes[i].cd) continue;
		// we use all CD drives, even if they do not contain a volume
		if (volumes[i].cd /* && cdOnly */)
			volumes[i].wasUsed = TRUE;
		volumes[i].drive = drives->dQDrive;
		if (err = MyGetVInfo(drives->dQDrive, (StringPtr)vName, &vRefNum, &freeBytes)) continue;
		volumes[i].vRefNum = vRefNum;
		my_p2cstr((StringPtr)vName);
		strcpy(volumes[i].name, vName);
		volumes[i].wasUsed = TRUE;
	}
	
	if (!initialize)
		for (i = 0 ; i < 26 ; i++)
			if (volumes[i].drive != oldVolumes[i].drive ||
				strcmp(volumes[i].name, oldVolumes[i].name) ||
				volumes[i].cd != oldVolumes[i].cd) {
				if (changes) changes[i] = volumes[i].vRefNum;
				change = TRUE;
			}
	
	return change;
#endif
}
#else
{	
	Boolean driveAvailable, cd, change = initialize;
	char vName[32], p[4] = "C:\\";
	short i;
	DWORD maxLength, flags, drives;
	VolumeInfo oldVolumes[26];
	
	if (!initialize)
		for (i = 0 ; i < 26 ; i++)
			oldVolumes[i] = volumes[i];
	
	drives = GetLogicalDrives();
	for (i = 0 ; i < 26 ; i++) {
		volumes[i].drive = 0;
		volumes[i].vRefNum = 0;
		volumes[i].name[0] = 0;
		if (initialize) {
			volumes[i].cd = FALSE;
			volumes[i].wasUsed = FALSE;
		}
		if (((drives >> i) & 1) == 0 || i < 2) // avoid spinning A and B
			continue;
		p[0] = 'A' + i;
		if (initialize) {
			WITHOUTERRORS(cd = GetDriveType(p) == DRIVE_CDROM;)
			volumes[i].cd = cd;
		}
		if (cdOnly && !volumes[i].cd) continue;
		// we use all CD drives, even if they do not contain a volume
		if (/* cdOnly && */ volumes[i].cd)
			volumes[i].wasUsed = TRUE;
		WITHOUTERRORS(driveAvailable = GetVolumeInformation(p, vName, 31, 0, &maxLength, &flags, 0, 0);)
		if (driveAvailable) {
			volumes[i].drive = 'A' + i;
			volumes[i].vRefNum = i;
			strnzcpy(volumes[i].name, vName, 31);
			volumes[i].wasUsed = TRUE;
		}
	}
	
	if (!initialize)
		for (i = 0 ; i < 26 ; i++)
			if (volumes[i].drive != oldVolumes[i].drive ||
				strcmp(volumes[i].name, oldVolumes[i].name) ||
				volumes[i].cd != oldVolumes[i].cd) {
				if (changes) changes[i] = volumes[i].vRefNum;
				change = TRUE;
			}
	
	return change;
}
#endif

Boolean VolumesChanged(Boolean cdOnly)
#ifdef MAC
{
	char vName[256];
	short i, vRefNum;
	double freeBytes; 
	OSErr err;
	
	for (i = 0 ; i < 26 ; i++) {
		if (!volumes[i].wasUsed && !volumes[i].name[0]) continue;
		if (cdOnly)
			if (!volumes[i].cd) continue;
		vName[0] = 0;
		err = MyGetVInfo(volumes[i].drive, (StringPtr)vName, &vRefNum, &freeBytes);
		my_p2cstr((StringPtr)vName);
		if (strcmp(volumes[i].name, vName)) return TRUE;
	}
	
	return FALSE;
}
#else
{
	Boolean driveAvailable;
	char vName[32], p[4] = "C:\\";
	short i;
	DWORD maxLength, flags;
	
	for (i = 0 ; i < 26 ; i++) {
		if (!volumes[i].wasUsed && !volumes[i].name[0]) continue;
		if (cdOnly)
			if (!volumes[i].cd) continue;
		p[0] = 'A' + i;
		vName[0] = 0;
		WITHOUTERRORS(driveAvailable = GetVolumeInformation(p, vName, 31, 0, &maxLength, &flags, 0, 0);)
		if (// !driveAvailable ||
		    strcmp(volumes[i].name, vName)) return TRUE;
	}
	
	return FALSE;
}
#endif

Boolean VolumeOnLine(char *volumeName, char *drive, short *vRefNum, Boolean cdOnly)
{
	short i;
	
	for (i = 0 ; i < 26 ; i++) {
		if (cdOnly && !volumes[i].cd) continue;
		if (!volumes[i].name[0]) continue;
		if (!strcmpnocase(volumes[i].name, volumeName)) {
#ifdef MAC
			char vName[32];
			double freeBytes;
			
			if (drive) strcpy(drive, volumes[i].name);
			if (vRefNum)
				MyGetVInfo(volumes[i].drive, (StringPtr)vName, vRefNum, &freeBytes);
#else
	// JLM 7/8/98 this function is not used on the IBM in COSSM
			strcpy(drive, "A:");
			drive[0] = 'A' + i;
			if (vRefNum) (*vRefNum) = i;
#endif
			
			return TRUE;
		}
	}
	
	return FALSE;
}

CHARPTR PathNameFromDirID(long dirID, short vRefNum, CHARPTR pathName)
#ifdef MAC
{
	CInfoPBRec block;
	char directoryName[255];
	OSErr err;
	
	pathName[0] = 0;
	block.dirInfo.ioNamePtr = (StringPtr)directoryName;
	block.dirInfo.ioDrParID = dirID;
	do {
		block.dirInfo.ioVRefNum = vRefNum;
		block.dirInfo.ioFDirIndex = -1;
		block.dirInfo.ioDrDirID = block.dirInfo.ioDrParID;
		if (err = PBGetCatInfo(&block, false)) return pathName;
		mypstrcatJM(directoryName, (char *)"\p:");
		mypstrcatJM(directoryName, pathName);
		mypstrcpyJM(pathName, directoryName);
	} while (block.dirInfo.ioDrDirID != fsRtDirID);
	
	return pathName;
}
#else
{
	GetDirectoryFromID(dirID, pathName);
	my_c2pstr(pathName);
	
	return pathName;
}
#endif

#ifdef MAC
#if MACB4CARBON
CHARPTR PathNameFromWD(long vRefNum, CHARPTR pathName)
{
	WDPBRec myBlock;
	
	myBlock.ioNamePtr = nil;
	myBlock.ioVRefNum = vRefNum;
	myBlock.ioWDIndex = 0;
	myBlock.ioWDProcID = 0;
	PBGetWDInfo(&myBlock, false);
	
	return PathNameFromDirID(myBlock.ioWDDirID, myBlock.ioWDVRefNum, pathName);
}
#endif
#endif

#ifdef MAC
CHARPTR GetFullPath(short vRefNum, long dirID, CHARPTR fileName, CHARPTR path)
// this gets the path as a p-string
/*#ifdef IBM
{
	long len;
	path[0] = 0;
	PathNameFromDirID(dirID,vRefNum, path);
	// add delimiter if necessary
	len = strlen(path);
	if(len > 0 && path[len-1] != DIRDELIMITER) 
	{
		path[len] = DIRDELIMITER;
		path[len+1] = 0;
	}
	// would need to convert back to c-string here
	strcat(path,fileName);
	// then the entire string back to p-string again
	// for now just make this a mac only function as it is in CATS
	return path;
}
#else*/
{
#if MACB4CARBON
	char s[64];
	long size, err;
	Handle h = 0;
	FSSpec spec;
	AliasHandle alias = 0;
	AliasInfoType i;
	
	strcpy(s, fileName);
	my_c2pstr(s);
	FSMakeFSSpec(vRefNum, dirID, (StringPtr)s, &spec);
	
	s[0] = path[0] = 0;
	if (err = NewAlias(0, &spec, &alias))
		return path;
	
	if (!(h = _NewHandle(0)))
		{ DisposeHandle((Handle)alias); return path; }
	
	i = asiAliasName;
	if (!GetAliasInfo(alias, i, (StringPtr)s)) {
		while (strlen(s) > 0 && !err) { // s == "" when done traversing the path
			size = s[0] + 1;       
			s[0] = ':'; // use the size byte to store the colon
			size = Munger(h, 0, nil, 0, s, size);
			i = i + asiParentName; // set the index to the next parent
			err = GetAliasInfo(alias, i, (StringPtr)s); // get the parentName
		}
		
		if (!err) {
			err = GetAliasInfo (alias, asiVolumeName, (StringPtr)s);
			size = s[0];
			size = Munger (h, 0, nil, 0, &s[1], size);
		} 
	}
	
	if (alias)
		DisposeHandle((Handle)alias);
	
	if (h) {
		size = _GetHandleSize(h);
		_HLock(h);
		_BlockMove(*h, path, size);
		path[size] = 0;
		_HUnlock(h);
		DisposeHandle(h);
	}
	
	return path;
#else
		// JLM 12/09/06, removed the "alias" code and changed to just get the directory path and then cat the filename
	if(dirID == 0 && vRefNum == 0) 
	{
		// then fileName must already be the full path, just return it
		strcpy(path,fileName);
	} 
	else 
	{
		PathNameFromDirID(dirID,vRefNum,path); // P-str with delimiter
		my_p2cstr((StringPtr)path);
		strcat(path,fileName);
	}
	
	return path;
#endif

}
#endif



void AddFileSuffix(char *fileName, char *suffix, short vRefNum)
{
	char *p;
#ifdef MAC
	short end = 27;
#pragma unused(vRefNum)
#else
	short end = LongFileNames(vRefNum) ? 27 : 8;
#endif
	
	//if (p = strchr(fileName, '.')) *p = 0;
	// JLM 1/22/99 changed to strrchr
	if (p = strrchr(fileName, '.')) *p = 0;
	if (end < strlen(fileName))
		fileName[end] = 0;
	strcat(fileName, suffix);
}

/////////////////////////////////////////////////
#ifdef MAC
Boolean IsBundledApplication(char *path)
{
	// On the Mac , bundled applications are in a folder that 
	// ends with .app
	// and contains a "Contents" subfolder
	
	if(strstr(path,".app:Contents:"))
		return true;

	return false;
}
#endif

pascal_ifMac Boolean STDFilter(DialogPtr theDialog, EventRecord *theEvent, short *itemHit)
#ifdef MAC
{
	char buttonTitle[256], help[256], windowTitle[256], cancel[256], close[256], done[256];
	char wizCancel[256],wizPrevious[256];
	//char buttonTitle[32], help[32], windowTitle[64], cancel[32], close[32], done[32];
	//char wizCancel[32],wizPrevious[32];
	short i;
	GrafPtr oldPort;
	ControlHandle whichControl;
	Point p;
	static DialogPtr currentDialog = nil;
	static Boolean justDragged = FALSE;
	
	// if (GetAppRefCon(globalSTPtr)->inBackground) RESUME(globalSTPtr, nil);
	
#if TARGET_API_MAC_CARBON
	// for OS X we are using movable dialog and are supposed to call StdFilterProc
	// for apple events etc..
	// it also says we are supposed to handle update events for our windows but it does not appear to be necessary
	Boolean handledTheEvent = StdFilterProc (theDialog,theEvent,itemHit);
	if(handledTheEvent) 
		return handledTheEvent; 
#endif
	
	GetPortGrafPtr(&oldPort);
	SetPortDialogPort(theDialog);
	
#if MACB4CARBON
	// we do not want to override the cursor behavior  in TARGET_API_MAC_CARBON, we want the sytem to use an I-beam when the mouse is over an edit box
	if (resetCursor) InitCursor();
	// if (currentDialog != theDialog) InitCursor();
	currentDialog = theDialog;
#endif
	
	settings.doNotPrintError = false;//JLM 6/19/98, allows dialogs to come up more than once
	
	lastEvent = *theEvent;
	
	if (justDragged || theEvent->what != updateEvt)
		DealWithEvent(*theEvent, TRUE);
	
	if (theEvent->what != updateEvt && theEvent->what != nullEvent)
		justDragged = FALSE;
	
	if (KeyEvent(theEvent, RETURN, ENTER)) {
		//if ((**(ControlHandle)GetDialogItemHandle(theDialog, 1)).contrlHilite == 0) {
		//if (GetControlHilite((ControlRef)GetDialogItemHandle(theDialog, 1)) == 0) {
		if (ControlEnabled(theDialog, 1)) {	// try this
			*itemHit = 1;
			FlashItem(theDialog, 1);
		}
		else
			*itemHit = 0;
		SetPortGrafPort(oldPort);
		return TRUE;
	}
	
	if (KeyEvent(theEvent, ESC, ESC) || CommandKeyEvent(theEvent, '.', '.')) 
	{
		getindstring(cancel, DLG_STRINGS, CANCEL_STRING); // "Cancel"
		getindstring(close, DLG_STRINGS, CLOSE_STRING); // "Close"
		GetWizButtonTitle_Previous(wizPrevious);
		GetWizButtonTitle_Cancel(wizCancel);

		for (i = 1 ; i <= 3 ; i++)
		{
			if((GetDialogItemType(theDialog, i) & 0x7) == ctrlItem)
			{
				MyGetControlTitle(theDialog, i, buttonTitle);
				if (!strcmpnocase(cancel, buttonTitle) || !strcmpnocase(close, buttonTitle)
					|| !strcmpnocase(wizPrevious, buttonTitle)  //JLM support for wizard
					|| !strcmpnocase(wizCancel, buttonTitle)  //JLM support for wizard
					) {
					FlashItem(theDialog, i);
					*itemHit = i;
					SetPortGrafPort(oldPort);
					return TRUE;
				}
			}
		}
		
		// if there was no cancel button ,look to see if there is a done button
		// that is not the first button
		getindstring(done, DLG_STRINGS, DONE_STRING); // "Done"
		for (i = 2 ; i <= 3 ; i++)
		{
			if((GetDialogItemType(theDialog, i) & 0x7) == ctrlItem)
			{
				MyGetControlTitle(theDialog, i, buttonTitle);
				if (!strcmpnocase(done, buttonTitle)) 
				{
					FlashItem(theDialog, i);
					*itemHit = i;
					SetPortGrafPort(oldPort);
					return TRUE;
				}
			}
		}
	}
	
	p = theEvent->where;
	GlobalToLocal(&p);
	if (theEvent->what == mouseDown && FindControl(p, GetDialogWindow(theDialog), &whichControl)) {
		Rect r = GetDialogPortRect(theDialog); // GetDialogItemBox(theDialog, 1)
		
		if (p.v > ((r.top + r.bottom) / 2)) {
			getindstring(help, DLG_STRINGS, HELP_STRING); // "Help..."
			GetControlTitle(whichControl, (StringPtr)buttonTitle);
			my_p2cstr((StringPtr)buttonTitle);
			if (
				!strcmp(help, buttonTitle)
				|| !strcmp("Help", buttonTitle)
				|| !strcmp("Help...", buttonTitle)
			) {
				if (TrackControl(whichControl, p, nil)) {
					getwtitle(GetDialogWindow(theDialog), windowTitle);
					if (GetHelp(windowTitle, TRUE) == HELP_TOPICS)
						GetHelpTopics(windowTitle, TRUE);
					SetPortGrafPort(oldPort);
					return FALSE; // application might want to react to help in some way
				}
			}
		}
	}
	
	if (DragDuringDialog(theDialog, theEvent)) {
		justDragged = TRUE;
		*itemHit = 0;
		SetPortGrafPort(oldPort);
		return TRUE;
	}
	
	SetPortGrafPort(oldPort);
	return FALSE;
}
#else
{
	settings.doNotPrintError = false;//JLM 6/19/98, allows dialogs to come up more than once
	return FALSE;
}
#endif

OSErr FindFileFromPath(char *path, char *suffix, char *name, short dialogID, Boolean exeFiles)
{	// fix for carbon - this file not used anyway
	char copy[256], fileName[256], fileType[32] = "*.";
	short numTypes = exeFiles ? 4 : -1;
	Point where;
#ifdef MAC
	OSType typeList[] = { 'APPL', 'APPL', 'APPL', 'APPL' };
#else
	OSType typeList[] = { 'XXXX', 'XXXX', 'XXXX', 'EXE.' };
#endif
	MySFReply reply;
	
	if (!path || !path[0])
		return 0;
	
	if (!FileExists(0, 0, path) ) {
		strcpy(copy, path);
		SplitPathFile(copy, fileName);
		paramtext(fileName, name, "", "");
		strcat(fileType, suffix);
		//sfpgetfile(&where, fileType, (FileFilterUPP)0, numTypes, typeList,
				   //(DlgHookUPP)0, &reply, dialogID, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
#if !TARGET_API_MAC_CARBON
		sfpgetfile(&where, fileType, (FileFilterUPP)0, numTypes, typeList,
				   (DlgHookUPP)0, &reply, dialogID,  (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		if (!reply.good) return USERCANCEL;
		my_p2cstr(reply.fName);
#else
		mysfpgetfile(&where, fileType, numTypes, typeList,
				   (MyDlgHookUPP)0, &reply, dialogID,MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
#endif
#ifdef MAC
#if TARGET_API_MAC_CARBON
	strcpy(path,reply.fullPath);
	return 0;
#else
		PathNameFromWD(reply.vRefNum, path);
		my_p2cstr((StringPtr)path);
		strcat(path, (char *)reply.fName);
#endif
#else
		strcpy(path, reply.fName);
#endif
	}
	
	return 0;
}

CHARPTR MyGetFolderName(short vRefNum, long dirID, Boolean fullPath, CHARPTR folderName)
#ifdef MAC
{
	CInfoPBRec block;
	
	if (fullPath)
		return GetFullPath(vRefNum, dirID, "", folderName);
	
	if (folderName) folderName[0] = 0;
	block.dirInfo.ioFDirIndex = -1;
	block.dirInfo.ioNamePtr = (StringPtr)folderName;
	block.dirInfo.ioDrDirID = dirID;
	block.dirInfo.ioVRefNum = vRefNum;
	PBGetCatInfo(&block, false); // don't beep because it might be an alias
	my_p2cstr((StringPtr)folderName);
	
	return folderName;                                  
}
#else
{
	char temp[256], *p;
	
	GetDirectoryFromID(dirID, folderName);
	if (!fullPath) {
		strcpy(temp, folderName);
		if (temp[strlen(temp) - 1] == DIRDELIMITER) temp[strlen(temp) - 1] = 0;
		if (p = strrchr(temp, DIRDELIMITER))
			strcpy(folderName, p + 1);
	}
	
	return folderName;
}
#endif

#ifdef MAC
static long sharedFileCount;

OSErr AddFolderFile(CINFOPBRECPTR block)
{
#pragma unused(block)
	
	sharedFileCount++;
	
	return 0;
}

#endif

long GetFolderNumFiles(short vRefNum, long dirID, CHARPTR pathName)
#ifdef MAC
{
	CInfoPBRec block;
	long err;
	
	if (pathName && pathName[0]) {
		sharedFileCount = 0;
		if (err = ForEachItemInFolder(0, 0, pathName, AddFolderFile, 0))
			return 0;
		
		return sharedFileCount;
	}
	
	block.dirInfo.ioFDirIndex = -1;
	block.dirInfo.ioNamePtr = nil;
	block.dirInfo.ioDrDirID = dirID;
	block.dirInfo.ioVRefNum = vRefNum;
	PBGetCatInfo(&block, false); // don't beep because it might be an alias
	
	return block.dirInfo.ioDrNmFls;
}
#else
{
	HANDLE findHandle;
	char dirPath[256], filePath[256];
	WIN32_FIND_DATA fileData;
	long numFiles = 0;
			
	if (!pathName[0]) {
		if (!GetDirectoryFromID(dirID, dirPath)) return -1;
	}
	else
		strcpy(dirPath, pathName);

	strcpy(filePath, dirPath);
	strcat(filePath, "*");
	findHandle = FindFirstFile(filePath, &fileData);
	if (findHandle == INVALID_HANDLE_VALUE) return 0;
	numFiles++;
	for ( ; ; ) {
		if (!FindNextFile(findHandle, &fileData)) break;
		numFiles++;
	}
	FindClose(findHandle);

	return numFiles;
}
#endif

long ParentFolderID(short vRefNum, long dirID, CHARPTR pathName)
#ifdef MAC
{
   CInfoPBRec block;
#pragma unused(pathName)
   
   block.dirInfo.ioFDirIndex = -1;
   block.dirInfo.ioNamePtr = nil;
   block.dirInfo.ioDrDirID = dirID;
   block.dirInfo.ioVRefNum = vRefNum;
   PBGetCatInfo(&block, false); // don't beep because it might be an alias
   
   return block.dirInfo.ioDrParID;
}
#else
{
	char path[256], *end;
	long newDirID;
	
	if (pathName && NameStartsAtIBMRoot(pathName)) // full pathName given
		strcpy(path, pathName);
	else
		if (!GetDirectoryFromID(dirID, path)) return -1;
	
	end = strrchr(path, DIRDELIMITER);
	
	if (!end) return -1;
	
	// Remember, there may or may not be a '\' at the very end of this path.
	// ...if there is, skip that one and find the next
	if (end[1] == 0) {
		end--;
		while (end[0] != DIRDELIMITER)
			end--;
	}
	end[1] = 0;
	
	GetIDFromPath(&newDirID, path, TRUE);
	
	return newDirID; 
}
#endif

long GetFolderID(short vRefNum, long dirID, CHARPTR folder)
#ifdef MAC
{
	CInfoPBRec block;
	char path[256];
	
	PathNameFromDirID(dirID, vRefNum, path);
	my_p2cstr((StringPtr)path);
	strcat(path, folder);
	my_c2pstr(path);
	block.dirInfo.ioFDirIndex = 0;
	block.dirInfo.ioNamePtr = (StringPtr)path;
	block.dirInfo.ioDrDirID = 0;
	block.dirInfo.ioVRefNum = vRefNum;
	if (PBGetCatInfo(&block, false))
		return -1;
	
	return block.dirInfo.ioDrDirID;
}
#else
{
	FSSpec spec;
	//char name[33];	// this is too short and causes buffer overrun in release mode
	char name[64];	
	
	strcpy(name, folder);
	strcat(name, "\\");
	my_c2pstr(name);
	if (FSMakeFSSpec(FALSE, dirID, name, &spec))
		return -1;
	
	return spec.parID;
}
#endif
OSErr MyGetFileSize(short vRefNum, long dirID, CHARPTR pathName, LONGPTR size) {
	
	char c;
	long x = 0;

	OSErr err = 0;
	char filePath[256], tempFilePath[256];
	/////

	if(model && model->fWizard && model->fWizard->PathIsWizardResource(pathName))
	{	// wizard support, JLM 10/2/98
		err = model->fWizard->MyGetFileSizeFromResource(pathName,size);
		return err;
	}
	if (vRefNum!=0 || dirID!=0) 
	{
		sprintf(tempFilePath, "%s%s", TATFolder, pathName);
	}
	else
		strcpy(tempFilePath, pathName);
	////////
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) tempFilePath, filePath, kMaxNameLen) ;
#else
		strcpy(filePath,tempFilePath);
#endif	

	struct stat stat_buf;
    int rc = stat(filePath, &stat_buf);
	if (rc == 0) {*size = stat_buf.st_size; return noErr;}
	else {*size = 0; return -1;}
}

/*OSErr MyGetFileSize(short vRefNum, long dirID, CHARPTR pathName, LONGPTR size)
{
	OSErr err = 0;
	/////
	if(model && model->fWizard && model->fWizard->PathIsWizardResource(pathName))
	{	// wizard support, JLM 10/2/98
		err = model->fWizard->MyGetFileSizeFromResource(pathName,size);
		return err;
	}
	////////

#ifdef MAC
{
	HParamBlockRec block;
	OSErr result;
	char localStr[256];

	strcpy(localStr,pathName);
	my_c2pstr(localStr);
	//my_c2pstr(pathName);
	//block.ioParam.ioNamePtr = (StringPtr)pathName;
	block.ioParam.ioNamePtr = (StringPtr)localStr;
	block.ioParam.ioVRefNum = vRefNum;
	block.fileParam.ioDirID = dirID;
	block.fileParam.ioFVersNum = 0;
	block.fileParam.ioFDirIndex = 0;
	result = PBHGetFInfo(&block, FALSE);
	*size = block.fileParam.ioFlLgLen;
	//my_p2cstr((StringPtr)pathName);
	
	return result;
}
#else
{
   char path[256];
   HANDLE hFile;

   if (!ResolveFileDirectory(vRefNum, dirID, pathName, path)) return -1;

   // JLM 1/10/01 , this code fixes a NOVELL server bug
   //hFile = CreateFile (path, 0, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
   hFile = CreateFile (path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
   
   if (hFile == INVALID_HANDLE_VALUE)  return -1;

   *size = GetFileSize (hFile, NULL);
   if(*size < 0) {*size = 0; return -3;} // JLM 1/10/01
   CloseHandle (hFile);
   return 0;
}
#endif	
}*/

OSErr GetFileModDate(short vRefNum, long dirID, CHARPTR pathName, LONGPTR seconds)
#ifdef MAC
{
	HParamBlockRec block;
	long err;
	char p[256];
	
	block.ioParam.ioCompletion = 0;
	strcpy(p, pathName);
	my_c2pstr(p);
	block.ioParam.ioNamePtr = (StringPtr)p;
	block.ioParam.ioVRefNum = vRefNum;
	block.fileParam.ioFDirIndex = 0;
	block.fileParam.ioDirID = dirID;
	if (err = PBHGetFInfo(&block, FALSE)) return err;
	
	*seconds = block.fileParam.ioFlMdDat;
	return 0;
}
#else
{
   char path[256];
   HANDLE hFile;
   BOOL result;

   if (!ResolveFileDirectory(vRefNum, dirID, pathName, path)) return -1;

   hFile = CreateFile (path, 0, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
   if (hFile == INVALID_HANDLE_VALUE)  return -1;
   
   result = GetModDateLong((short)hFile, seconds);
   CloseHandle (hFile);
   
   return result ? 0 : -1;
}
#endif	

OSErr GetVolumeName(short vRefNum, char *volName)
#ifdef MAC
{
	long err;
	HParamBlockRec paramBlock;
	
	volName[0] = 0;
	paramBlock.ioParam.ioCompletion = 0;
	paramBlock.ioParam.ioVRefNum = vRefNum;
	paramBlock.ioParam.ioNamePtr = (StringPtr)volName;
	paramBlock.volumeParam.ioVolIndex = 0;
	if (err = PBHGetVInfo(&paramBlock, FALSE)) return err;
	my_p2cstr((StringPtr)volName);
	
	return 0;
}
#else
{
	// JLM 7/8/98 network code 
	if(0 <= vRefNum && vRefNum <= 25)
	{
		strcpy(volName, "A:");
		volName[0] = 'A' + vRefNum;
		return 0;
	}
	else
	{	// it must be network volume number, JLM 6/25/98
		 return NetworkGetVolumeName(vRefNum,volName);
	}
}
#endif	
	
//	JLM - 5/6/97 - changed to a double at the request of a CENSUS guy
// 	because they have a really big harddrive
//	OSErr FreeBytesOnVolume(short vRefNum, LONGPTR freeBytes, CHARPTR volName)
OSErr FreeBytesOnVolume(short vRefNum, double* dFreeBytes, CHARPTR volName)
#ifdef MAC
{
	OSErr err;
	double freeBytes=0;
	short dummyVRefNum;
	HParamBlockRec paramBlock;
#pragma unused(volName)
	
	*dFreeBytes = 0;
	
	paramBlock.ioParam.ioCompletion = 0;
	paramBlock.ioParam.ioNamePtr = 0;
	paramBlock.ioParam.ioVRefNum = vRefNum;
	paramBlock.volumeParam.ioVolIndex = 0;
	if (err = PBHGetVInfo(&paramBlock, FALSE)) return err;
	
	err =  MyGetVInfo(paramBlock.volumeParam.ioVDrvInfo, 0, &dummyVRefNum, &freeBytes);
	*dFreeBytes = (double)freeBytes;// change long to a double
	return err;
}
#else
{
	char volPath[4] = "a:\\";
	DWORD secPerClust, bytesPerSec, freeClust, clusters, error;
#pragma unused(vRefNum)
	
	*dFreeBytes = 0;
	if (volName && volName[0])
		strnzcpy(volPath, volName, 3);
	else
		volPath[0] = 'a' + vRefNum;
	if (!GetDiskFreeSpace(volPath, &secPerClust, &bytesPerSec, &freeClust, &clusters)) {
		error = GetLastError();
		return -1;
	}
	
	// *dFreeBytes = freeClust * secPerClust * bytesPerSec;
	*dFreeBytes = freeClust * secPerClust * (double)bytesPerSec;
	return 0;
}
#endif	*/

OSErr ForEachItemInFolder(short vRefNum, long dirID, CHARPTR pathName,
						  OSErr (*proc)(CINFOPBRECPTR), Boolean (*filter)(CINFOPBRECPTR))
#ifdef MAC
{
	short			index = 1;
	CInfoPBRec		block;
	char			nameStr[256];
	FSSpec			spec;
	OSErr			errCode;
	
	// if pathName is provided, it is assumed that pathName to be the full
	// path to an existing file in the desired directory
	
	if (pathName && pathName[0]) {
		strcpy(nameStr, pathName);
		my_c2pstr(nameStr);
		if (errCode = FSMakeFSSpec(0, 0, (StringPtr)nameStr, &spec)) return errCode;
		vRefNum = spec.vRefNum;
		dirID = spec.parID;
	}
	
	for ( ; ; ) {
		block.hFileInfo.ioCompletion = nil;
		block.hFileInfo.ioVRefNum = vRefNum;
		block.dirInfo.ioDrDirID = dirID;
		nameStr[0] = 0;
		block.hFileInfo.ioNamePtr = (StringPtr)nameStr;
		block.hFileInfo.ioFDirIndex = index;
		errCode = PBGetCatInfo(&block, FALSE);
		if (errCode) return (errCode == fnfErr) ? 0 : errCode;
		if (!filter || (*filter)(&block)) if (errCode = (*proc)(&block)) return errCode;
		index++;
	}
}
#else
{
	HANDLE hFile;
	char dirPath[256], filePath[256], *p;
	WIN32_FIND_DATA fileData;
	CInfoPBRec block;
	OSErr errCode;
	
	// if pathName is provided, it is assumed that pathName to be the full
	// path to an existing file in the desired directory, unless it ends
	// with a backslash, in which case it is taken as the desired directory
	
	if (!pathName || !pathName[0]) {
		if (!GetDirectoryFromID(dirID, dirPath)) return -1;
	}
	else
		strcpy(dirPath, pathName);
	
	if (dirPath[strlen(dirPath) - 1] != DIRDELIMITER) {
		if (dirPath[strlen(dirPath) - 1] == ':')
			strcat(dirPath, "\\");
		else
			if (p = strrchr(dirPath, DIRDELIMITER))
				p[1] = 0;
	}
	strcpy(filePath, dirPath);
	strcat(filePath, "*");

	hFile = FindFirstFile(filePath, &fileData);
	// JLM 7/8/98
	// when there is no file, FindFirstFile() will return INVALID_HANDLE_VALUE
	// when running on a network.  This is not really an error.
	//This is what resulted in the "View List not coming up when there were no views bug"
	//if (hFile == INVALID_HANDLE_VALUE) return -1;
	if (hFile == INVALID_HANDLE_VALUE) return 0;

	for ( ; ; ) {
		strcpy(filePath, dirPath);
		strcat(filePath, fileData.cFileName);
		if (fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			strcat(filePath, "\\");
		block.hFileInfo.ioNamePtr = filePath;
		if (fileData.cFileName[0] != '.' && (!filter || (*filter)(&block))) {
			if (fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
				if (!GetIDFromPath(&block.hFileInfo.ioFlParID, filePath, TRUE))
					{ FindClose(hFile); return -1; }
				if (!GetIDFromPath(&block.dirInfo.ioDrDirID, filePath, TRUE))
					{ FindClose(hFile); return -1; }
			}
			else {
				if (!GetIDFromPath(&block.hFileInfo.ioFlParID, dirPath, TRUE))
					{ FindClose(hFile); return -1; }
				if (!GetIDFromPath(&block.dirInfo.ioDrDirID, dirPath, TRUE))
					{ FindClose(hFile); return -1; }
				if (p = strrchr(filePath, DIRDELIMITER))
					block.hFileInfo.ioNamePtr = &p[1];
			} 
			block.hFileInfo.ioVRefNum = vRefNum;
			if (errCode = (*proc)(&block))
				{ FindClose(hFile); return errCode; }
		}
		if (!FindNextFile(hFile, &fileData)) break;
	}
	FindClose(hFile);
	
	return 0;
}
#endif

OSErr MyDeleteFile(CINFOPBRECPTR block)
#ifdef MAC
{
	char fileName[32];
	
	my_p2cstr(block->hFileInfo.ioNamePtr);
	strcpy(fileName, (char const *)block->hFileInfo.ioNamePtr);
	my_c2pstr((char *)block->hFileInfo.ioNamePtr);
	
	return hdelete(block->hFileInfo.ioVRefNum,
				   block->hFileInfo.ioFlParID,
				   fileName);
}
#else
{
	// (MIKE) delete the file specified by block
	return 0;
}
#endif

OSErr DeleteFilesInFolder(short vRefNum, long dirID, CHARPTR pathName)
#ifdef MAC
{
	short			index = 1;
	CInfoPBRec		block;
	char			nameStr[100];
	OSErr			errCode;
#pragma unused(pathName)
	
	for ( ; ; ) {
		block.hFileInfo.ioCompletion = nil;
		block.hFileInfo.ioVRefNum = vRefNum;
		block.dirInfo.ioDrDirID = dirID;
		nameStr[0] = 0;
		block.hFileInfo.ioNamePtr = (StringPtr)nameStr;
		block.hFileInfo.ioFDirIndex = index;
		errCode = PBGetCatInfo(&block, FALSE);
		if (errCode) return (errCode == fnfErr) ? 0 : errCode;
		if (errCode = MyDeleteFile(&block)) index++; // skip items that can't be deleted
	}
}
#else
{
	HANDLE hFile;
	char dirPath[256], filePath[256];
	WIN32_FIND_DATA fileData;
	OSErr err = 0;
			
	if (pathName && pathName[0])
		strcpy(dirPath, pathName);
	else
		if (!GetDirectoryFromID(dirID, dirPath)) return -1;
	
	for ( ; ; ) {  // JLM, 7/8/98 ,  this is a rather strange loop, but maybe Michael had his reasons
		strcpy(filePath, dirPath);
		strcat(filePath, "*");
		// skip "." and ".."
		hFile = FindFirstFile(filePath, &fileData); 
		if (hFile == INVALID_HANDLE_VALUE){
			err = GetLastError();
		}
		//FindNextFile(hFile, &fileData);
		////////////
		// when running on a network you don't get "." and ".."
		//////{
		for(;;)
		{
			if(err)
			{
				if(err == ERROR_NO_MORE_FILES) err = 0; // not an error
				FindClose(hFile);
				return 0; // this is the main way to exit DeleteFilesInFolder() 
			}
			if(!strcmp(fileData.cFileName,".") || !strcmp(fileData.cFileName,".."))
			{
				// it is "." or ".."
			 	// try again
				Boolean successful = FindNextFile(hFile, &fileData);
				if (!successful)
					err = GetLastError();
			}
			else 
			{
				break; // we found the first file
			}
		}
		FindClose(hFile);
		
		strcpy(filePath, dirPath);
		strcat(filePath, fileData.cFileName);
		// JLM 7/8/98 Note:on the MAC we keep going if some file can't be deleted,
		// but we can't do the same thing here because of the way the loop is structured
		if (!DeleteFile(filePath)) return -1;
	}
}
#endif

#ifdef MAC
OSErr ForEachMountedVolume(OSErr (*proc)(HParamBlockRec *))
{
   HParamBlockRec block;
   long i, err;
   char name[256];
   
   for (i = 1 ; ; i++) {
      block.ioParam.ioCompletion = 0;
      block.ioParam.ioNamePtr = (StringPtr)name;
      block.ioParam.ioVRefNum = 0;
      block.volumeParam.ioVolIndex = i;
      if (err = PBHGetVInfo(&block, FALSE)) return 0;
      if (block.volumeParam.ioVDrvInfo <= 0) continue; // volume off-line or ejected
      if (err = (*proc)(&block)) return err;
   }
}
#else
OSErr ForEachMountedVolume(OSErr (*proc)(char))
{
	Boolean driveAvailable;
	char vName[100], p[4];
	short d;
	OSErr err;
	DWORD maxLength, flags, drives;
	
	drives = GetLogicalDrives();
	strcpy(p, "C:\\");
	for (d = 2 ; d < 26 ; d++) { // start at 2 to avoid spinning A and B
		if (((drives >> d) & 1) == 0)
			continue;
		p[0] = 'A' + d;
		WITHOUTERRORS(driveAvailable = GetVolumeInformation(p, vName, 99, 0, &maxLength, &flags, 0, 0);)
		if (driveAvailable)
			if (err = (*proc)(p[0])) return err;
	}
	
	return 0;
}
#endif

#ifdef MAC

typedef struct {
		char path[128];
		short vRefNum;
		long dirID;
	} PathInfo, *PathInfoP, **PathInfoH;

void GrowTable(PathInfoH table,short * tableSize)
{	// JLM 11/10/98
	#define NUMRECORDSTOADD 100
	short newSize = NUMRECORDSTOADD + *tableSize;// try adding more records
	long oldNumChars,newNumChars,desiredNumChars,i;
	Handle h; 
	if(!table) return;
	oldNumChars = _GetHandleSize((Handle)table);
	desiredNumChars = oldNumChars + 100 * sizeof(**table);
	_SetHandleSize((Handle)table , desiredNumChars);
	newNumChars = _GetHandleSize((Handle)table);
	//if(newNumChars != desiredNumChars) return; // memory error
	// init the newly allocated memory to zero
	h = (Handle) table;
	for(i = oldNumChars; i < newNumChars;i++) INDEXH(h,i) = 0;
	// return the new size
	*tableSize = newNumChars/sizeof(**table);
}

OSErr ScanFolders(char *path, short vRefNum, long dirID, PathInfoH table,short * tableSize,  short *count, short level)
{
	char *q, p[128], name[32];
	short index = 1;
	long err = 0;
	CInfoPBRec block;
	
	if (level == 3) return 0;
	//if ((*count) >= 500) return 0; // JLM 11/10/98, this check replaced via GrowTable code
	if (strlen(path) > 96) return 0;
	
	block.hFileInfo.ioCompletion = nil;
	
	if (!dirID) {
		if (!strchr(path, ':'))
			dirID = fsRtDirID;
		else {
			strcpy(p, path);
			my_c2pstr(p);
			block.hFileInfo.ioVRefNum = vRefNum;
			block.dirInfo.ioDrDirID = 0;
			block.hFileInfo.ioNamePtr = (StringPtr)p;
			block.hFileInfo.ioFDirIndex = 0;
			err = PBGetCatInfo(&block, FALSE);
			if (err) return err;
			dirID = block.dirInfo.ioDrDirID;
		}
	}
	
	strcpy(p, path);
	q = &p[strlen(p)];
	
	block.hFileInfo.ioVRefNum = vRefNum;
	block.hFileInfo.ioNamePtr = (StringPtr)name;
	
	for (index = 1 ; ; index++) {
		name[0] = 0;
		block.dirInfo.ioDrDirID = dirID;
		block.hFileInfo.ioFDirIndex = index;
		err = PBGetCatInfo(&block, FALSE);
		if (err) return (err == fnfErr) ? 0 : err;
		if (!(block.hFileInfo.ioFlAttrib & 16)) continue; // not a directory
		my_p2cstr((StringPtr)name);
		strcat(name, ":");
		strcpy(q, name);
		///
		// JLM 11/10/98, don't overwrite the array
		if(*count >= *tableSize) GrowTable(table,tableSize);
		if(*count >= *tableSize) return memFullErr;
		//////
		(*table)[(*count)].vRefNum = vRefNum;
		(*table)[(*count)].dirID = block.dirInfo.ioDrDirID;
		strcpy((*table)[(*count)].path, p);
		(*count)++;
		if (err = ScanFolders(p, vRefNum, block.dirInfo.ioDrDirID, table,tableSize, count, level + 1)) return err;
	}
	
	return 0;
}


int ComparePaths(const void *rec1, const void *rec2)
{
	PathInfo *p1 = (PathInfo *)rec1, *p2 = (PathInfo *)rec2;
	
	return strcmpnocase(p1->path, p2->path);
}

Boolean FolderInTable(char *path, PathInfoH table, short count, short *vRefNum, long *dirID)
{
/*
	short i;
	
	for (i = 0 ; i < count ; i++)
		if (!strcmpnocase((*table)[i].path, path)) {
			(*vRefNum) = (*table)[i].vRefNum;
			(*dirID) = (*table)[i].dirID;
			
			return TRUE;
		}
*/
	PathInfo *found, p;
	
	strcpy(p.path, path);
	
	_HLock((Handle)table);
	found = (PathInfo *)bsearch((void *)&p, (void *)DEREFH(table), count,
								sizeof(PathInfo), ComparePaths);
	_HUnlock((Handle)table);
	
	if (found) {
		(*vRefNum) = found->vRefNum;
		(*dirID) = found->dirID;
		
		return TRUE;
	}
	
	return FALSE;
}
#endif

OSErr QuickSpec(char *path, short *vRefNum, long *dirID)
{
#ifdef MAC
	char *p, q, pathC[256];
	short vRef;
	long err;
	FSSpec spec;
	static short count;
	static PathInfoH table = 0;
	//// JLM , 11/10/98 bug with LV3_04 CD , more than 500 folders
	static short sTableSize = 0;
	//
	
	if (!path) { if (table) DisposeHandle((Handle)table); table = 0;sTableSize = 0; return 0; }
	
	if (!vRefNum) {
		strcpy(pathC, path);
		if (p = strchr(pathC, '\\')) (*p) = 0;
		if (!VolumeOnLine(pathC, 0, &vRef, FALSE)) return -1;
		count = 0;
		if (table) { DisposeHandle((Handle)table); table = 0;sTableSize = 0;}// 4/21/98 JLM, deallocate here too, just to be sure

		sTableSize = 500;
		table = (PathInfoH)_NewHandleClear(sTableSize * sizeof(PathInfo));
		if (!table) return -1;
		
		strcpy(pathC, path);
		StringSubstitute(pathC, '\\', ':');
		strcat(pathC, ":");
		if (err = ScanFolders(pathC, vRef, 0, table,&sTableSize, &count, 1))
			{ DisposeHandle((Handle)table); table = 0;sTableSize = 0; return err; }
		_HLock((Handle)table);
		qsort((void *)DEREFH(table), count, sizeof(PathInfo), ComparePaths);
		_HUnlock((Handle)table);
		
		return 0;
	}
	
	// path is of the form disk:folder1:...:foldern:file
	
	if (p = strrchr(path, DIRDELIMITER)) { q = p[1]; p[1] = 0; } // remove file name
	
	if (table && FolderInTable(path, table, count, vRefNum, dirID)) return 0;
	
	p[1] = q;
	my_c2pstr(path);
	if (err = FSMakeFSSpec(0, 0, (StringPtr)path, &spec)) {
		my_p2cstr((StringPtr)path);
		strcat(path, ";1");
		my_c2pstr(path);
		if (err = FSMakeFSSpec(0, 0, (StringPtr)path, &spec)) return err;
	}
	
	(*vRefNum) = spec.vRefNum;
	(*dirID) = spec.parID;
#else
	(*vRefNum) = VRefNumFromIBMName(path);
	if (!GetIDFromPath(dirID, path, FALSE)) return -1;
#endif
	
	return 0;
}

#ifdef IBM
///////////////////////////////////////////////////////// MACUTIL ///////////////
#endif


///////// LAUNCHING ///////////////////////////////////////////////////////////////

#ifdef MAC

void ChopExtensionAndHiddenPackageFolders(char *fullPath)
{ 	// ON the MAC, OS X carbon applications are actually folders ending with .app 
	// which have inner folders where the actual application is kept
	/////////////////////////////////////////////////
	#if TARGET_API_MAC_CARBON
	{
		// in MAC OS X, the application is inside a bundled folder
		// so look up the chain for a folder ending in .app
		// This will be the real "application" folder
		char tempPath[256];
		char *p;
		long len;
		strcpy(tempPath,fullPath);
		for(;;){ // forever
			p = strrchr(tempPath,DIRDELIMITER);
			if(!p) break;
			*p = 0; // chop it off
			len = strlen(tempPath);
			if(len < 5) break;
			if(strcmp(tempPath+len-4,".app") == 0 
				|| strcmp(tempPath+len-4,".APP") == 0	
				|| strcmp(tempPath+len-4,".App") == 0) {
				// we found the folder which is a packaged "application"
				// chop off the hidden package extension
				tempPath[len-4] = 0;
				strcpy(fullPath,tempPath);
				break;
			}
		}
	}
	#else 
		#pragma unused(fullPath) // Old style Mac, nothing to chop
	#endif
	/////////////////////////////////////////////////
}

OSType String2OSType(char * str)
{
	OSType type = 0;
	char * p = (char*)&type;
	short i;
	short len = strlen(str);
	
	// copy the first 4 chars of the string
	for(i = 0; i < 4 && i < len; i++)
		p[i] = str[i];
	
	// for Intel chips, we need to swap the order
	#ifdef SWAP_BINARY
		type = (OSType) SwapLong((long) type);
	#endif
	
	return type;
}

OSErr SendOpenDocument(char *signature, char *document)
{
	// given a ptr to a Full Path Name, this proc sends the core apple event
	// 'open documents' for the document named to the application whose 
	// creator type is 'signature'
	
	OSErr			err = noErr;
	FSSpec			docSpec;
	AliasHandle		alias;
	AppleEvent		ae, reply;
	AEAddressDesc	addr;
	AEDescList		paramList;
	char localStr[256];
	OSType applSig = String2OSType(signature);
	
	if (!document || document[0] == 0) return 0;
	
	strcpy(localStr,document);
	//my_c2pstr(document);
	//err = FSMakeFSSpec(0, 0, (unsigned char *)document, &docSpec);
	//p2cstr((unsigned char *)document);
	my_c2pstr(localStr);
	err = FSMakeFSSpec(0, 0, (unsigned char *)localStr, &docSpec);
	if (err) return err;
	
	err = NewAliasMinimal(&docSpec, &alias);
	if (err) return err;
	
	//AECreateDesc(typeApplSignature, (Ptr)signature, 4, &addr);
	AECreateDesc(typeApplSignature, (Ptr)&applSig, 4, &addr);
	AECreateAppleEvent(kCoreEventClass, kAEOpenDocuments, &addr, kAutoGenerateReturnID, kAnyTransactionID, &ae);
	AECreateList(0, 0, false, &paramList);
	_HLock((Handle)alias);
	AEPutPtr(&paramList, 0, typeAlias, (Ptr) *alias, _GetHandleSize((Handle) alias));
	_HUnlock((Handle)alias);
	AEPutParamDesc(&ae, keyDirectObject, &paramList);
	
	err = AESend(&ae, &reply, kAENoReply + kAECanInteract + kAECanSwitchLayer, kAENormalPriority, kNoTimeOut, nil, nil);
	
	AEDisposeDesc(&addr);
	AEDisposeDesc(&paramList);
	AEDisposeDesc(&ae);
	AEDisposeDesc(&reply);
	DisposeHandle((Handle)alias);
	
	return err;
}

OSErr TryToLaunch(char* appPath, Boolean bringToFront, ProcessSerialNumber *launchedProcessSN)
{
	char					app[256], doc[256];
	EventRecord				event;
	OSErr					err = 0, err2 = 0;
	FSSpec					appSpec, docSpec;
	AliasHandle				alias = 0;
	/*AEDesc					launchdesc,
							//dNull = { typeNull, (Handle)0 }; // null descriptor
							dNull = { typeNull, (AEDataStorageType*)0 };*/ // null descriptor
	AEDesc launchdesc = { typeNull, 0 };
	AEDesc dNull = { typeNull, 0 }; // null descriptor
	AEDescList				paramList;
	AppleEvent				ae;
	LaunchParamBlockRec		lpbr;
	
	memset(launchedProcessSN,0,sizeof(*launchedProcessSN));
	
	//if (AppIsRunning(friend.signature) && friend.running) return 0; // just to be sure we don't try to launch a running app
	
	
	//if (err = FindFileFromPath(friend.appPath, "", friend.name, 10000, TRUE))
	//	return 1;
	
	strcpy(app, appPath);
	my_c2pstr(app);
	if (err = FSMakeFSSpec(0, 0, (StringPtr)app, &appSpec))
		return -1;
	
	
	lpbr.launchBlockID = extendedBlock;
	lpbr.launchEPBLength = extendedBlockLen;
	lpbr.launchFileFlags = 0;
	lpbr.launchControlFlags = launchNoFileFlags + launchContinue;
	if (!bringToFront) lpbr.launchControlFlags += launchDontSwitch;
	lpbr.launchAppSpec = &appSpec;	
	lpbr.launchAppParameters = 0;
	
	err = LaunchApplication(&lpbr);
	
	if (err == memFullErr || err == memFragErr) {
		lpbr.launchControlFlags += launchUseMinimum;
		err = LaunchApplication(&lpbr);
	}
	
	if(err == noErr)
		*launchedProcessSN = lpbr.launchProcessSN;
	
	
	EventAvail(everyEvent, &event);
	InitCursor();
	
	return err;
}

#else

OSErr TryToLaunch(char* appPath, Boolean bringToFront)
{
	char commandLine[256], currentDirectory[256], fileName[256], path[256];
	long err, currentDrive;
	DWORD ticks;
	PROCESS_INFORMATION processInformation;
	STARTUPINFO suInfo;
	BOOL success;
	
	
	tryAgain:
	
	// seems to help to wait after calling AppIsRunning() before checking
	// the path; if we do it too soon after, the check sometimes fails falsely
	for (ticks = GetTickCount() ; GetTickCount() < (ticks + 250) ; ) ;
	
	
	strcpy(commandLine, appPath);
	
	memset((char *)&suInfo, 0, sizeof(STARTUPINFO));
	suInfo.cb = sizeof(STARTUPINFO);
	suInfo.dwFlags = STARTF_USESHOWWINDOW;
	suInfo.wShowWindow = SW_SHOWNORMAL;
	
	strcpy(path, appPath);
	SplitPathFile(path, fileName);
	_getcwd(currentDirectory, 255);
	currentDrive = _getdrive();
	if (_chdrive(1 + tolower(path[0]) - 'a')) { SysBeep(1); return -1; }
	if (_chdir(path)) { SysBeep(1); return -1; }
	// if (WinExec(commandLine, SW_SHOWNORMAL) <= 31)
	success = CreateProcess(0, commandLine, 0, 0, FALSE, 0, 0, 0, &suInfo, &processInformation);
	if (!success) {
		goto tryAgain;
	}
	else {
		// the Windows documentation says we should close these handles immediately after creating the process if they are not needed
		CloseHandle(processInformation.hProcess);
		CloseHandle(processInformation.hThread);
	}
	_chdrive(currentDrive);
	_chdir(currentDirectory);
	
	
	return 0;
}

#endif

/////////