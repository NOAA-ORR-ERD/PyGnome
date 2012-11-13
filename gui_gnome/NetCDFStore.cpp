#include "CROSS.H"
#include "NetCDFStore.h"
#include "netcdf.h"
#include "CLASSES.H"
#include <map>
#include <string>
#include <list>
#include <new>
#include <iostream>

using namespace std;

NetCDFStore::NetCDFStore() {

    this->time = NULL;
    this->pCount = NULL;
    this->lon = NULL;
    this->lat = NULL;
    this->depth = NULL;
    this->mass = NULL;
    this->age = NULL;
    this->flag = NULL;
    this->status = NULL;
    this->id = NULL;
}

bool NetCDFStore::Capture(TModel* model, bool uncertain, map<string, int> *ncVarIDs, map<string, int> *ncDimIDs) {
    
	if(!model->IsUncertain() && uncertain)
		return true;
	
    int c, i, j, n;
    bool threeMovement;
	TLEList* thisLEList;
    CMyList* LESetsList;
    list<LERecP> tLEsContainer;
    NetCDFStore* netStore;
	char errStr[256];
    
    netStore = new NetCDFStore();
    tLEsContainer = list<LERecP>();
    LESetsList = model->LESetsList;
	
 // Grab initial positions, and store them;

    for (i = 0, n = LESetsList->GetItemCount(); i < n; i++) {
	    LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if(!thisLEList->IsActive()) continue;
		LETYPE typy = thisLEList->GetLEType();
        if(uncertain && typy == UNCERTAINTY_LE && model->IsUncertain()) {
			for (j = 0, c = thisLEList->numOfLEs; j < c; j++) {
				if(INDEXH(thisLEList->LEHandle, j).statusCode == OILSTAT_NOTRELEASED);
				else {
					try {
						(INDEXH(thisLEList->LEHandle, j)).leUnits = thisLEList->GetMassUnits();
						tLEsContainer.push_back(&(INDEXH(thisLEList->LEHandle, j)));
					}
					catch(...) {
						strcpy(errStr,"We are unable to allocate the memory required to perform this task.\n");
						strcat(errStr,"The run will continue, notwithstanding.");
						printError(errStr);
						//cerr << "We are unable to allocate the memory required to perform this task.\n";
						//cerr << "The run will continue, notwithstanding.\n";
						//cerr << "(Timestep " << model->currentStep << ")\n";
						return false;
					}
				}
			}
		}
        else if(!uncertain && typy != UNCERTAINTY_LE) {
			for (j = 0, c = thisLEList->numOfLEs; j < c; j++) {
					if(INDEXH(thisLEList->LEHandle, j).statusCode == OILSTAT_NOTRELEASED);
					else {
						try {
							(INDEXH(thisLEList->LEHandle, j)).leUnits = thisLEList->GetMassUnits();
							tLEsContainer.push_back(&(INDEXH(thisLEList->LEHandle, j)));
						}
						catch(...) {
							strcpy(errStr,"We are unable to allocate the memory required to perform this task.\n");
							strcat(errStr,"The run will continue, notwithstanding.");
							printError(errStr);
							//cerr << "We are unable to allocate the memory required to perform this task.\n";
							//cerr << "The run will continue, notwithstanding.\n";
							//cerr << "(Timestep " << model->currentStep << ")\n";
							return false;
						}
					}
			}
		}
    }

    c = tLEsContainer.size();
    netStore->time = model->modelTime - model->GetStartTime();
    //netStore->time = model->modelTime;
	//netStore->time -= 2082816000L;
	
	float tFloat;
    netStore->pCount = c;
	
	try {	
		netStore->lon = new float[c];
		netStore->lat = new float[c];
		if(threeMovement = model->ThereIsA3DMover(&tFloat))
			netStore->depth = new float[c];
		netStore->mass = new float[c];
		netStore->age = new long[c];
		netStore->flag = new short[c];
		netStore->status = new long[c];
		netStore->id = new long[c];	
	}
	catch(std::bad_alloc) {
		strcpy(errStr,"We are unable to allocate the memory required to perform this task.\n");
		strcat(errStr,"The run will continue, notwithstanding.");
		printError(errStr);
		//cerr << "We are unable to allocate the memory required to perform this task.\n";
		//cerr << "The run will continue, notwithstanding.\n";
		//cerr << "(Timestep " << model->currentStep << ")\n";
		return false;
	}

    list<LERecP>::iterator tIter = tLEsContainer.begin();

/*		enum { KILOGRAMS = 1, METRICTONS, SHORTTONS,
			GALLONS, BARRELS, CUBICMETERS, LES };		*/

    for(j = 0; j < c; j++) {
        LERecP tLE = *tIter;
        netStore->lon[j] = tLE->p.pLong;
		netStore->lon[j] /= 1000000;
        netStore->lat[j] = tLE->p.pLat;
		netStore->lat[j] /= 1000000;
        if(threeMovement)
            netStore->depth[j] = tLE->z;
        netStore->mass[j] = GetLEMass(*tLE);
		float tMass = netStore->mass[j];
		try {
			long tUnits = (*tLE).leUnits;
			if(tUnits == BARRELS || tUnits == GALLONS || tUnits == CUBICMETERS)
				netStore->mass[j] = VolumeMassToGrams(tMass, (*tLE).density, tUnits);
			else if(tUnits == KILOGRAMS || tUnits == METRICTONS || tUnits == SHORTTONS)
				netStore->mass[j] = ConvertMassToGrams(tMass, tUnits);
			else
				netStore->mass[j] = -tMass;
		}
		catch (...) {
			netStore->mass[j] = -tMass;
		}
        netStore->age[j] = tLE->ageInHrsWhenReleased*(3600);
		netStore->age[j] += model->modelTime - tLE->releaseTime;
		netStore->flag[j] = 0;
		netStore->status[j] = OILSTAT_INWATER;
		if(threeMovement)
			if(tLE->z > 0)
				netStore->flag[j] += 16;

		switch(tLE->statusCode) {
			case OILSTAT_ONLAND:
				netStore->flag[j] += 2;
				netStore->status[j] = OILSTAT_ONLAND;
				break;
			case OILSTAT_OFFMAPS:
				netStore->flag[j] += 4;
				netStore->status[j] = OILSTAT_OFFMAPS;
				break;
			case OILSTAT_EVAPORATED:
				netStore->flag[j] += 8;
				netStore->status[j] = OILSTAT_EVAPORATED;
				break;
			default:
				break;
		}
		
        netStore->id[j] = tLE->leKey;
        tIter++;
		

    }

    tLEsContainer.clear();
	netStore->ncVarIDs = ncVarIDs;
	netStore->ncDimIDs = ncDimIDs;
    netStore->Write(model, threeMovement, uncertain);
	
	
	delete[] netStore->lat;
	delete[] netStore->lon;
	if(threeMovement)
		delete[] netStore->depth;
	delete[] netStore->mass;
	delete[] netStore->flag;
	delete[] netStore->status;
	delete[] netStore->age;
	delete[] netStore->id;
    delete netStore;           
    return true;

}

bool NetCDFStore::Create(char *path, bool overwrite, int* ncID) {

    int ncErr;

    if(overwrite)
        ncErr = nc_create(path, NC_CLOBBER, ncID);
    else
        ncErr = nc_create(path, NC_NOCLOBBER, ncID);

    return CheckNC(ncErr);
}

bool NetCDFStore::Define(TModel* model, bool uncertain, map<string, int> *ncVarIDs, map<string, int> *ncDimIDs) 
{
    
	Seconds seconds;
	char currentTimeStr[256], startTimeStr[256], timeStr[256];

	if(!model->IsUncertain() && uncertain)
		return true;
	
    bool threeMovement;

// Assuming that the output format will not change.

// ++ Dimensions:

#define VAR_DIMS    1
#define NUM_VARs    10
	
	*ncDimIDs = map<string, int>();	
	*ncVarIDs = map<string, int>();
	
    int ncErr, ncID, *varIDs, tD;
    int tTime[VAR_DIMS], tData[VAR_DIMS], tVariables[NUM_VARs];

	if(!uncertain)
		ncID = model->ncID;
	else
		ncID = model->ncID_C;
	
   // ncErr = nc_def_dim(ncID, "time", model->stepsCount, tTime);
    ncErr = nc_def_dim(ncID, "time", model->outputStepsCount, tTime);
    if(!CheckNC(ncErr)) return false; // handle error. 

	ncErr = nc_def_dim(ncID, "data", NC_UNLIMITED, tData);
    if(!CheckNC(ncErr)) return false; // handle error.
	
    (*ncDimIDs)["Time"] = *tTime;
    (*ncDimIDs)["Data"] = *tData;
    

// ++ Variables:

    varIDs = tVariables;
    ncErr = nc_def_var(ncID, "time", NC_DOUBLE, VAR_DIMS, tTime, varIDs);
    if(!CheckNC(ncErr)) return false; // handle error.
    
    (*ncVarIDs)["Time"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "particle_count", NC_INT, VAR_DIMS, tTime, varIDs);
    if(!CheckNC(ncErr)) return false; // handle error.

    (*ncVarIDs)["Particle_Count"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "longitude", NC_FLOAT, VAR_DIMS, tData, varIDs);
    if(!CheckNC(ncErr)) return false; // handle error.

    (*ncVarIDs)["Longitude"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "latitude", NC_FLOAT, VAR_DIMS, tData, varIDs);
    if(!CheckNC(ncErr)) return false; // handle error.
    
    (*ncVarIDs)["Latitude"] = *varIDs;
    ++varIDs;

	float tFloat;
    threeMovement = model->ThereIsA3DMover(&tFloat);
	
	if(threeMovement) {
		ncErr = nc_def_var(ncID, "depth", NC_FLOAT, threeMovement ? VAR_DIMS : 0, tData, varIDs);
		if(!CheckNC(ncErr)) return false; // handle error.

		(*ncVarIDs)["Depth"] = *varIDs;
		++varIDs;
	}

    ncErr = nc_def_var(ncID, "mass", NC_FLOAT, VAR_DIMS, tData, varIDs);
    if(!CheckNC(ncErr)) return false; // handle error.
    
    (*ncVarIDs)["Mass"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "age", NC_INT, VAR_DIMS, tData, varIDs);
    if(!CheckNC(ncErr)) return false; // handle error.
    
    (*ncVarIDs)["Age"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "flag", NC_BYTE, VAR_DIMS, tData, varIDs);
    if(!CheckNC(ncErr)) return false; // handle error.

    (*ncVarIDs)["Flag"] = *varIDs;
    ++varIDs;

     ncErr = nc_def_var(ncID, "status", NC_INT, VAR_DIMS, tData, varIDs);
    if(!CheckNC(ncErr)) return false; // handle error.

    (*ncVarIDs)["Status"] = *varIDs;
    ++varIDs;

   ncErr = nc_def_var(ncID, "id", NC_INT, VAR_DIMS, tData, varIDs);
    if(!CheckNC(ncErr)) return false; // handle error.

    (*ncVarIDs)["ID"] = *varIDs;

    varIDs = tVariables;
    char *tStr;

// ++ Global Attributes: 

	GetDateTime(&seconds);
	Secs2DateStringNetCDF(seconds, currentTimeStr);
	//Secs2Date(seconds, &daterec);
	//sprintf(trajectoryTime, "%02hd%02hd", daterec.hour, daterec.minute);
	//sprintf(trajectoryDate, "%hd/%hd/%02hd", daterec.month, daterec.day, daterec.year % 100);
	
	tStr = "Particle output from the NOAA GNOME model";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "comment", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;

	// code goes here - get current time 
	//tStr = "2011-14-09T11:18:40.0000";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "creation_date", strlen(currentTimeStr), currentTimeStr);
	if(!CheckNC(ncErr)) return false;
	
	tStr = "GNOME version 1.3.5";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "source", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;
	
	tStr = "http://response.restoration.noaa.gov/gnome";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "references", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;
	
	//tStr = "Sample data/file for particle trajectory format";
	//ncErr = nc_put_att_text(ncID, NC_GLOBAL, "title", strlen(tStr), tStr);
	//if(!CheckNC(ncErr)) return false;
	
	if(!uncertain)
		tStr = "particle_trajectories";
	else
		tStr = "uncertainty_particle_trajectories";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "feature_type", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;
	
	tStr = "NOAA Emergency Response Division";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "institution", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;
	
	tStr = "CF-1.6";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "conventions", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;

	
// ++ Local Attributes:
	
    // Time:
    //tStr = "seconds since 1970-01-01 0:00:00";
    //tStr = "seconds since ";
	//strcat(tStr,currentTimeStr);
    //ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "units", strlen(tStr), tStr);
	strcpy(timeStr, "seconds since ");
	seconds = model->GetStartTime();
	Secs2DateStringNetCDF(seconds, startTimeStr);
	strcat(timeStr, startTimeStr);
	//sprintf(currentTimeStr,"%s%s",timeStr,startTimeStr);

    //if(!CheckNC(ncErr)) return false; // handle error.

    // Time:
    //tStr = "seconds since 1970-01-01 0:00:00";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "units", strlen(timeStr), timeStr);
    if(!CheckNC(ncErr)) return false; // handle error.

    tStr = "time";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "long_name", strlen(tStr), tStr);
    if(!CheckNC(ncErr)) return false; // handle error.

    tStr = "time";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "standard_name", strlen(tStr), tStr);
    if(!CheckNC(ncErr)) return false; // handle error.
	
	tStr = "gregorian";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "calendar", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;

 	tStr = "unspecified time zone";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "comment", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;

   // Particle count
    tStr = "1";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Particle_Count"], "units", strlen(tStr), tStr);
    if(!CheckNC(ncErr)) return false; // handle error.
   
    tStr = "number of particles in a given timestep";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Particle_Count"], "long_name", strlen(tStr), tStr);
    if(!CheckNC(ncErr)) return false; // handle error.

    tStr = "particle count at nth timestep";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Particle_Count"], "ragged_row_count", strlen(tStr), tStr);
    if(!CheckNC(ncErr)) return false; // handle error.
    
    // Longitude

    tStr = "longitude of the particle";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Longitude"], "long_name", strlen(tStr), tStr);
    if(!CheckNC(ncErr)) return false; // handle error.

	tStr = "degrees_east";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Longitude"], "units", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;
	
    // Latitude

    tStr = "latitude of the particle";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Latitude"], "long_name", strlen(tStr), tStr);
    if(!CheckNC(ncErr)) return false;

	tStr = "degrees_north";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Latitude"], "units", strlen(tStr), tStr);
// Leaving definition mode.

	// Depth
	
	if(threeMovement) {
		tStr = "particle depth below sea surface";
		ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Depth"], "long_name", strlen(tStr), tStr);
		if(!CheckNC(ncErr)) return false;
		
		tStr = "meters";
		ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Depth"], "units", strlen(tStr), tStr);
		if(!CheckNC(ncErr)) return false;

		tStr = "z positive down";
		ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Depth"], "axis", strlen(tStr), tStr);
		if(!CheckNC(ncErr)) return false;

	}
	
	// Mass
	
	tStr = "grams";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Mass"], "units", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;
	
	// Age
	
	tStr = "from age at time of release";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Age"], "description", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;
	
	tStr = "seconds";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Age"], "units", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;
	
	
	// Flags
	
	tStr = "particle status flag";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Flag"], "long_name", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;
	
	{
		const int tRange[] = {0, 5};
		ncErr = nc_put_att_int(ncID, (*ncVarIDs)["Flag"], "valid_range", NC_BYTE, 2, tRange);
		if(!CheckNC(ncErr)) return false;

		const int tVals[] = { 1, 2, 3, 4 };
		ncErr = nc_put_att_int(ncID, (*ncVarIDs)["Flag"], "flag_values", NC_BYTE, 4, tVals);
		if(!CheckNC(ncErr)) return false;
		
		tStr = "on_land off_maps evaporated below_surface";
		ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Flag"], "flag_meanings", strlen(tStr), tStr);
		if(!CheckNC(ncErr)) return false;
		
			// we don't need masks.
	}
	
	// Status
	
	tStr = "particle status flag";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Status"], "long_name", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;
	
	{
		const int tRange[] = {0, 10};
		ncErr = nc_put_att_int(ncID, (*ncVarIDs)["Status"], "valid_range", NC_INT, 2, tRange);
		if(!CheckNC(ncErr)) return false;

		const int tVals[] = { 2, 3, 7, 10 };
		ncErr = nc_put_att_int(ncID, (*ncVarIDs)["Status"], "flag_values", NC_INT, 4, tVals);
		if(!CheckNC(ncErr)) return false;
		
		tStr = "2:in_water 3:on_land 7:off_maps 10:evaporated";
		ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Status"], "flag_meanings", strlen(tStr), tStr);
		if(!CheckNC(ncErr)) return false;
		
			// we don't need masks.
	}
	
	tStr = "particle ID";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["ID"], "description", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;
	
	tStr = "1";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["ID"], "units", strlen(tStr), tStr);
	if(!CheckNC(ncErr)) return false;	
	// Leaving definition mode.
	
	
    ncErr = nc_enddef(ncID);
    if(!CheckNC(ncErr)) return false; // handle error.

    return true;
}

bool NetCDFStore::Open(char* path, int* ncID) {
// Opens strictly for read access.

    int ncErr;
    ncErr = nc_open(path, NC_WRITE, ncID);
    return CheckNC(ncErr);
}

bool NetCDFStore::Read() {

    return true;
}

bool NetCDFStore::Write(TModel* model, bool threeMovement, bool uncertain) {

    
	short ncID, *cFlag, ncErr = 0;
    long j, *cID, tID, timeStep, *cAge, *cStatus;
    float *cLon, *cLat, *cDepth, *cMass;
    static map<string, int> lastCoord_M = map<string, int>();
	static map<string, int> lastCoord_C = map<string, int>();
	map<string, int> *lastCoord;
	
	if(!uncertain)
		lastCoord = &lastCoord_M;
	else
		lastCoord = &lastCoord_C;
    timeStep = model->currentStep;
	if(!uncertain)
		ncID = model->ncID;
	else
		ncID = model->ncID_C;

    // Check coordinates..
    if((*lastCoord).size() == 0 || model->ncSnapshot || model->modelTime == model->GetStartTime()) {
        (*lastCoord)["Time"] = 0;
        (*lastCoord)["Particle_Count"] = 0;
        (*lastCoord)["Longitude"] = 0;
        (*lastCoord)["Latitude"] = 0;
		(*lastCoord)["Depth"] = 0;
        (*lastCoord)["Mass"] = 0;
        (*lastCoord)["Age"] = 0;
        (*lastCoord)["Flag"] = 0;
        (*lastCoord)["Status"] = 0;
        (*lastCoord)["ID"] = 0;
    }	
             // Time: 
    tID = (*this->ncVarIDs)["Time"];
    {
        const size_t tCoord[] = {(*lastCoord)["Time"]};
        ncErr = nc_put_var1_double(ncID, tID, tCoord, &this->time);
        (*lastCoord)["Time"]+= 1;
    }
    if(!CheckNC(ncErr)) return false; // handle error.

             // Particle Count:
    tID = (*this->ncVarIDs)["Particle_Count"];
    {
        const size_t tCoord[] = {(*lastCoord)["Particle_Count"]};
        ncErr = nc_put_var1_long(ncID, tID, tCoord, &this->pCount);
        (*lastCoord)["Particle_Count"]+= 1;
    }
    if(!CheckNC(ncErr)) return false; // handle error.
    
            // Longitudes:
    tID = (*this->ncVarIDs)["Longitude"];
    for(cLon = this->lon, j = 0;  ncErr == NC_NOERR && j < this->pCount; j++, cLon++) {
        const size_t tCoord[] = {(*lastCoord)["Longitude"]+j};
        ncErr = nc_put_var1_float(ncID, tID, tCoord, cLon);
    }
    (*lastCoord)["Longitude"] += j;
    if(!CheckNC(ncErr)) return false; // handle error.

            // Latitudes:
    tID = (*this->ncVarIDs)["Latitude"];
    for(cLat = this->lat, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cLat++) {
        const size_t tCoord[] = {(*lastCoord)["Latitude"]+j};
        ncErr = nc_put_var1_float(ncID, tID, tCoord, cLat);
    }
    (*lastCoord)["Latitude"]+=j;
    if(!CheckNC(ncErr)) return false; // handle error.
    
    if(threeMovement) {
                // Depths:
        tID = (*this->ncVarIDs)["Depth"];
        for(cDepth = this->depth, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cDepth++) {
            const size_t tCoord[] = {(*lastCoord)["Depth"]+j};
            ncErr = nc_put_var1_float(ncID, tID, tCoord, cDepth);
        }
        (*lastCoord)["Depth"]+=j;
        if(!CheckNC(ncErr)) return false;
    }

            // Mass:
    tID = (*this->ncVarIDs)["Mass"];
    for(cMass = this->mass, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cMass++) {
        const size_t tCoord[] = {(*lastCoord)["Mass"]+j};
        ncErr = nc_put_var1_float(ncID, tID, tCoord, cMass);
    }
    (*lastCoord)["Mass"]+=j;
    if(!CheckNC(ncErr)) return false;

            // Age:
    tID = (*this->ncVarIDs)["Age"];
    for(cAge = this->age, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cAge++) {
        const size_t tCoord[] = {(*lastCoord)["Age"]+j};
        ncErr = nc_put_var1_long(ncID, tID, tCoord, cAge);
    }
    (*lastCoord)["Age"]+=j;
    if(!CheckNC(ncErr)) return false;
        
            // Flags:
    tID = (*this->ncVarIDs)["Flag"];
    for(cFlag = this->flag, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cFlag++) {
        const size_t tCoord[] = {(*lastCoord)["Flag"]+j};
		ncErr = nc_put_var1_short(ncID, tID, tCoord, cFlag);
    }
    (*lastCoord)["Flag"]+=j;
    if(!CheckNC(ncErr)) return false;

             // Status:
    tID = (*this->ncVarIDs)["Status"];
    for(cStatus = this->status, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cStatus++) {
        const size_t tCoord[] = {(*lastCoord)["Status"]+j};
		ncErr = nc_put_var1_long(ncID, tID, tCoord, cStatus);
    }
    (*lastCoord)["Status"]+=j;
    if(!CheckNC(ncErr)) return false;

           // ID:
    tID = (*this->ncVarIDs)["ID"];
    for(cID = this->id, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cID++) {
        const size_t tCoord[] = {(*lastCoord)["ID"]+j};
        ncErr = nc_put_var1_long(ncID, tID, tCoord, cID);
    }
    (*lastCoord)["ID"]+=j;
    if(!CheckNC(ncErr)) return false;

    return true;
}

bool NetCDFStore::fClose(int ncID) {

    int ncErr;

    ncErr = nc_close(ncID);
    return CheckNC(ncErr);
}


bool NetCDFStore::CheckNC(int ncErr) {
    
	char errStr[256];
    if(ncErr != NC_NOERR) {
        switch(ncErr) {

            case NC_ENOMEM:
					strcpy(errStr,"We are unable to allocate the memory required to perform this task.\n");
					strcat(errStr,"The run will continue, notwithstanding. (Exception caught in write mode.)");
					printError(errStr);
					//cerr << "We are unable to allocate the memory required to perform this task.\n";
					//cerr << "The run will continue, notwithstanding. (Exception caught in write mode.)\n";
					return false;
            case NC_EBADID:
					strcpy(errStr,"Error: NC_EBADID.\n");
					strcat(errStr,"We will attempt to continue the run.");
					printError(errStr);
					//cerr << "Error: NC_EBADID.\n";
					//cerr << "We will attempt to continue the run.\n";
                    return false;
            case NC_EINVALCOORDS:
					strcpy(errStr,"Error: NC_EINVALCOORDS.\n");
					strcat(errStr,"We will attempt to continue the run.");
					printError(errStr);
					//cerr << "Error: NC_EINVALCOORDS.\n";
					//cerr << "We will attempt to continue the run.\n";
                    return false;
            case NC_EINDEFINE:
					strcpy(errStr,"Error: NC_EINDEFINE.\n");
					strcat(errStr,"We will attempt to continue the run.");
					printError(errStr);
					//cerr << "Error: NC_EINDEFINE.\n";
					//cerr << "We will attempt to continue the run.\n";
                    return false;
            default:
                // ..
                    return false;
        }
    }

    else return true;
}