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

OSErr NetCDFStore::Capture(TModel* model, bool uncertain, map<string, int> *ncVarIDs, map<string, int> *ncDimIDs) {
    
	//if(!model->IsUncertain() && uncertain)	// this is checked on the outside...
		//return true;
	
    int c, i, j, n;
    //bool threeMovement;
	TLEList* thisLEList;
    CMyList* LESetsList;
    list<LERecP> tLEsContainer;
    NetCDFStore* netStore;
	char errStr[256];
	double halfLife;
	OSErr err = 0;
    
    netStore = new NetCDFStore(); // why do we need this?
    tLEsContainer = list<LERecP>();
    LESetsList = model->LESetsList;
	
 // Grab initial positions, and store them - may want to do this on the outside and pass in the tLEsContainer

    for (i = 0, n = LESetsList->GetItemCount(); i < n; i++) {
	    LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if(!thisLEList->IsActive()) continue;
		LETYPE type = thisLEList->GetLEType();
		halfLife = (*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.halfLife;
					
        if(uncertain && type == UNCERTAINTY_LE /*&& model->IsUncertain()*/) {
			for (j = 0, c = thisLEList->numOfLEs; j < c; j++) {
				if(INDEXH(thisLEList->LEHandle, j).statusCode == OILSTAT_NOTRELEASED);
				else {
					try {
						(INDEXH(thisLEList->LEHandle, j)).leUnits = thisLEList->GetMassUnits();
						tLEsContainer.push_back(&(INDEXH(thisLEList->LEHandle, j)));
					}
					catch(...) {
						strcpy(errStr,"We are unable to allocate the memory required to perform this task.\n");
						//strcat(errStr,"The run will continue, notwithstanding.");
						printError(errStr);
						//cerr << "We are unable to allocate the memory required to perform this task.\n";
						//cerr << "The run will continue, notwithstanding.\n";
						//cerr << "(Timestep " << model->currentStep << ")\n";
						err = -1;
						return err;
					}
				}
			}
		}
        else if(!uncertain && type != UNCERTAINTY_LE) {
			for (j = 0, c = thisLEList->numOfLEs; j < c; j++) {
					if(INDEXH(thisLEList->LEHandle, j).statusCode == OILSTAT_NOTRELEASED);
					else {
						try {
							(INDEXH(thisLEList->LEHandle, j)).leUnits = thisLEList->GetMassUnits();
							tLEsContainer.push_back(&(INDEXH(thisLEList->LEHandle, j)));
						}
						catch(...) {
							strcpy(errStr,"We are unable to allocate the memory required to perform this task.\n");
							//strcat(errStr,"The run will continue, notwithstanding.");
							printError(errStr);
							//cerr << "We are unable to allocate the memory required to perform this task.\n";
							//cerr << "The run will continue, notwithstanding.\n";
							//cerr << "(Timestep " << model->currentStep << ")\n";
							err = -1;
							return err;
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
		//if(threeMovement = model->ThereIsA3DMover(&tFloat))	// always output depth
			netStore->depth = new float[c];
		netStore->mass = new float[c];
		netStore->age = new long[c];
		netStore->flag = new short[c];
		netStore->status = new long[c];
		netStore->id = new long[c];	
	}
	catch(std::bad_alloc) {
		strcpy(errStr,"We are unable to allocate the memory required to perform this task.\n");
		//strcat(errStr,"The run will continue, notwithstanding.");
		printError(errStr);
		//cerr << "We are unable to allocate the memory required to perform this task.\n";
		//cerr << "The run will continue, notwithstanding.\n";
		//cerr << "(Timestep " << model->currentStep << ")\n";
		err = -1;
		return err;
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
        //if(threeMovement)
            netStore->depth[j] = tLE->z;
        netStore->mass[j] = GetLEMass(*tLE,halfLife);
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
		//if(threeMovement)
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
   // netStore->Write(model, threeMovement, uncertain);
	err = netStore->Write(model, uncertain);
	if (err) return err;
	
	
	delete[] netStore->lat;
	delete[] netStore->lon;
	//if(threeMovement)
		delete[] netStore->depth;
	delete[] netStore->mass;
	delete[] netStore->flag;
	delete[] netStore->status;
	delete[] netStore->age;
	delete[] netStore->id;
    delete netStore;           
    return err;

}

OSErr NetCDFStore::Create(char *path, bool overwrite, int* ncID) {

    int ncErr;

    if(overwrite)
        ncErr = nc_create(path, NC_CLOBBER, ncID);
    else
        ncErr = nc_create(path, NC_NOCLOBBER, ncID);

    return CheckNC(ncErr);
}

OSErr NetCDFStore::Define(TModel* model, bool uncertain, map<string, int> *ncVarIDs, map<string, int> *ncDimIDs) 
{
	Seconds seconds;
	OSErr err = 0;
	char currentTimeStr[256], startTimeStr[256], timeStr[256];

	//if(!model->IsUncertain() && uncertain)
		//return true;
	
    //bool threeMovement;
	// pass in model start time, outputStepsCount, and ncID - may not need uncertain

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
    err = CheckNC(ncErr); {if (err) return err;} // handle error. 

	ncErr = nc_def_dim(ncID, "data", NC_UNLIMITED, tData);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.
	
    (*ncDimIDs)["Time"] = *tTime;
    (*ncDimIDs)["Data"] = *tData;
    

// ++ Variables:

    varIDs = tVariables;
    ncErr = nc_def_var(ncID, "time", NC_DOUBLE, VAR_DIMS, tTime, varIDs);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.
    
    (*ncVarIDs)["Time"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "particle_count", NC_INT, VAR_DIMS, tTime, varIDs);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.

    (*ncVarIDs)["Particle_Count"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "longitude", NC_FLOAT, VAR_DIMS, tData, varIDs);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.

    (*ncVarIDs)["Longitude"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "latitude", NC_FLOAT, VAR_DIMS, tData, varIDs);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.
    
    (*ncVarIDs)["Latitude"] = *varIDs;
    ++varIDs;

	float tFloat;
    //threeMovement = model->ThereIsA3DMover(&tFloat);
	
	//if(threeMovement) {
		//ncErr = nc_def_var(ncID, "depth", NC_FLOAT, threeMovement ? VAR_DIMS : 0, tData, varIDs);
		ncErr = nc_def_var(ncID, "depth", NC_FLOAT, VAR_DIMS, tData, varIDs);
		err = CheckNC(ncErr); {if (err) return err;} // handle error.

		(*ncVarIDs)["Depth"] = *varIDs;
		++varIDs;
	//}

    ncErr = nc_def_var(ncID, "mass", NC_FLOAT, VAR_DIMS, tData, varIDs);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.
    
    (*ncVarIDs)["Mass"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "age", NC_INT, VAR_DIMS, tData, varIDs);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.
    
    (*ncVarIDs)["Age"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "flag", NC_BYTE, VAR_DIMS, tData, varIDs);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.

    (*ncVarIDs)["Flag"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "status", NC_INT, VAR_DIMS, tData, varIDs);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.

    (*ncVarIDs)["Status"] = *varIDs;
    ++varIDs;

    ncErr = nc_def_var(ncID, "id", NC_INT, VAR_DIMS, tData, varIDs);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.

    (*ncVarIDs)["ID"] = *varIDs;

    varIDs = tVariables;
    char *tStr;

// ++ Global Attributes: 

	GetDateTime(&seconds);
	Secs2DateStringNetCDF(seconds, currentTimeStr);

	tStr = "Particle output from the NOAA GNOME model";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "comment", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}

	// code goes here - get current time 
	//tStr = "2011-14-09T11:18:40.0000";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "creation_date", strlen(currentTimeStr), currentTimeStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
	tStr = "GNOME version 1.3.9";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "source", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
	tStr = "http://response.restoration.noaa.gov/gnome";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "references", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
	//tStr = "Sample data/file for particle trajectory format";
	//ncErr = nc_put_att_text(ncID, NC_GLOBAL, "title", strlen(tStr), tStr);
	//err = CheckNC(ncErr); {if (err) return err;}
	
	if(!uncertain)
		tStr = "particle_trajectories";
	else
		tStr = "uncertainty_particle_trajectories";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "feature_type", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
	tStr = "NOAA Emergency Response Division";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "institution", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
	tStr = "CF-1.6";
	ncErr = nc_put_att_text(ncID, NC_GLOBAL, "conventions", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}

	
// ++ Local Attributes:
	
    // Time:
    //tStr = "seconds since 1970-01-01 0:00:00";
    //ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "units", strlen(tStr), tStr);
	strcpy(timeStr, "seconds since ");
	seconds = model->GetStartTime();
	Secs2DateStringNetCDF(seconds, startTimeStr);
	strcat(timeStr, startTimeStr);
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "units", strlen(timeStr), timeStr);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.

    tStr = "time";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "long_name", strlen(tStr), tStr);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.

    tStr = "time";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "standard_name", strlen(tStr), tStr);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.
	
	tStr = "gregorian";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "calendar", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}

 	tStr = "unspecified time zone";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Time"], "comment", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}

    // Particle count
    tStr = "1";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Particle_Count"], "units", strlen(tStr), tStr);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.
   
    tStr = "number of particles in a given timestep";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Particle_Count"], "long_name", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;} // handle error.

    tStr = "particle count at nth timestep";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Particle_Count"], "ragged_row_count", strlen(tStr), tStr);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.
    
    // Longitude

    tStr = "longitude of the particle";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Longitude"], "long_name", strlen(tStr), tStr);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.

	tStr = "degrees_east";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Longitude"], "units", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
    // Latitude

    tStr = "latitude of the particle";
    ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Latitude"], "long_name", strlen(tStr), tStr);
    err = CheckNC(ncErr); {if (err) return err;}

	tStr = "degrees_north";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Latitude"], "units", strlen(tStr), tStr);
// Leaving definition mode.

	// Depth
	
	//if(threeMovement) {
		tStr = "particle depth below sea surface";
		ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Depth"], "long_name", strlen(tStr), tStr);
		err = CheckNC(ncErr); {if (err) return err;}
		
		tStr = "meters";
		ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Depth"], "units", strlen(tStr), tStr);
		err = CheckNC(ncErr); {if (err) return err;}

		tStr = "z positive down";
		ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Depth"], "axis", strlen(tStr), tStr);
		err = CheckNC(ncErr); {if (err) return err;}

	//}
	
	// Mass
	
	tStr = "grams";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Mass"], "units", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
	// Age
	
	tStr = "from age at time of release";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Age"], "description", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
	tStr = "seconds";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Age"], "units", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
	
	// Flags
	
	tStr = "particle status flag";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Flag"], "long_name", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
	{
		const int tRange[] = {0, 5};
		ncErr = nc_put_att_int(ncID, (*ncVarIDs)["Flag"], "valid_range", NC_BYTE, 2, tRange);
		err = CheckNC(ncErr); {if (err) return err;}

		const int tVals[] = { 1, 2, 3, 4 };
		ncErr = nc_put_att_int(ncID, (*ncVarIDs)["Flag"], "flag_values", NC_BYTE, 4, tVals);
		err = CheckNC(ncErr); {if (err) return err;}
		
		tStr = "on_land off_maps evaporated below_surface";
		ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Flag"], "flag_meanings", strlen(tStr), tStr);
		err = CheckNC(ncErr); {if (err) return err;}
		
			// we don't need masks.
	}
	
	// Status
	
	tStr = "particle status flag";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Status"], "long_name", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
	{
		const int tRange[] = {0, 10};
		ncErr = nc_put_att_int(ncID, (*ncVarIDs)["Status"], "valid_range", NC_INT, 2, tRange);
		err = CheckNC(ncErr); {if (err) return err;}

		const int tVals[] = { 2, 3, 7, 10 };
		ncErr = nc_put_att_int(ncID, (*ncVarIDs)["Status"], "flag_values", NC_INT, 4, tVals);
		err = CheckNC(ncErr); {if (err) return err;}
		
		tStr = "2:in_water 3:on_land 7:off_maps 10:evaporated";
		ncErr = nc_put_att_text(ncID, (*ncVarIDs)["Status"], "flag_meanings", strlen(tStr), tStr);
		err = CheckNC(ncErr); {if (err) return err;}
		
			// we don't need masks.
	}
	
	tStr = "particle ID";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["ID"], "description", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;}
	
	tStr = "1";
	ncErr = nc_put_att_text(ncID, (*ncVarIDs)["ID"], "units", strlen(tStr), tStr);
	err = CheckNC(ncErr); {if (err) return err;};	
	// Leaving definition mode.
	
	
    ncErr = nc_enddef(ncID);
    err = CheckNC(ncErr); {if (err) return err;} // handle error.

    return err;
}

OSErr NetCDFStore::Open(char* path, int* ncID) {
// Opens strictly for read access.

    int ncErr;
    ncErr = nc_open(path, NC_WRITE, ncID);
    return CheckNC(ncErr);
}

OSErr NetCDFStore::Read() {

	OSErr err = 0;
    return err;
}

//bool NetCDFStore::Write(TModel* model, bool threeMovement, bool uncertain) {
OSErr NetCDFStore::Write(TModel* model, bool uncertain) {

    short *cFlag;
	int ncID, ncErr = 0;
    long j, *cID, tID, timeStep, *cAge, *cStatus;
    float *cLon, *cLat, *cDepth, *cMass;
    static map<string, int> lastCoord_M = map<string, int>();
	static map<string, int> lastCoord_C = map<string, int>();
	map<string, int> *lastCoord;
	OSErr err = 0;
	
	if(!uncertain)
		lastCoord = &lastCoord_M;
	else
		lastCoord = &lastCoord_C;
    //timeStep = model->currentStep;
	if(!uncertain)
		ncID = model->ncID;
	else
		ncID = model->ncID_C;

    // Check coordinates.. - maybe pass in a isFirstStep to avoid the need for model
    if((*lastCoord).size() == 0 || model->ncSnapshot || (!model->bHindcast && model->modelTime == model->GetStartTime()) || (model->bHindcast && model->modelTime == model->GetEndTime())) {
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
    err = CheckNC(ncErr); {if (err) return err;} // handle error.

             // Particle Count:
    tID = (*this->ncVarIDs)["Particle_Count"];
    {
        const size_t tCoord[] = {(*lastCoord)["Particle_Count"]};
        ncErr = nc_put_var1_long(ncID, tID, tCoord, &this->pCount);
        (*lastCoord)["Particle_Count"]+= 1;
    }
    err = CheckNC(ncErr); {if (err) return err;} // handle error.
    
            // Longitudes:
    tID = (*this->ncVarIDs)["Longitude"];
    for(cLon = this->lon, j = 0;  ncErr == NC_NOERR && j < this->pCount; j++, cLon++) {
        const size_t tCoord[] = {(*lastCoord)["Longitude"]+j};
        ncErr = nc_put_var1_float(ncID, tID, tCoord, cLon);
    }
    (*lastCoord)["Longitude"] += j;
    err = CheckNC(ncErr); {if (err) return err;} // handle error.

            // Latitudes:
    tID = (*this->ncVarIDs)["Latitude"];
    for(cLat = this->lat, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cLat++) {
        const size_t tCoord[] = {(*lastCoord)["Latitude"]+j};
        ncErr = nc_put_var1_float(ncID, tID, tCoord, cLat);
    }
    (*lastCoord)["Latitude"]+=j;
    err = CheckNC(ncErr); {if (err) return err;} // handle error.
    
    //if(threeMovement) {
                // Depths:
        tID = (*this->ncVarIDs)["Depth"];
        for(cDepth = this->depth, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cDepth++) {
            const size_t tCoord[] = {(*lastCoord)["Depth"]+j};
            ncErr = nc_put_var1_float(ncID, tID, tCoord, cDepth);
        }
        (*lastCoord)["Depth"]+=j;
		err = CheckNC(ncErr); {if (err) return err;}
    //}

            // Mass:
    tID = (*this->ncVarIDs)["Mass"];
    for(cMass = this->mass, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cMass++) {
        const size_t tCoord[] = {(*lastCoord)["Mass"]+j};
        ncErr = nc_put_var1_float(ncID, tID, tCoord, cMass);
    }
    (*lastCoord)["Mass"]+=j;
    err = CheckNC(ncErr); {if (err) return err;}

            // Age:
    tID = (*this->ncVarIDs)["Age"];
    for(cAge = this->age, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cAge++) {
        const size_t tCoord[] = {(*lastCoord)["Age"]+j};
        ncErr = nc_put_var1_long(ncID, tID, tCoord, cAge);
    }
    (*lastCoord)["Age"]+=j;
    err = CheckNC(ncErr); {if (err) return err;}
        
            // Flags:
    tID = (*this->ncVarIDs)["Flag"];
    for(cFlag = this->flag, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cFlag++) {
        const size_t tCoord[] = {(*lastCoord)["Flag"]+j};
		ncErr = nc_put_var1_short(ncID, tID, tCoord, cFlag);
    }
    (*lastCoord)["Flag"]+=j;
    err = CheckNC(ncErr); {if (err) return err;}

             // Status:
    tID = (*this->ncVarIDs)["Status"];
    for(cStatus = this->status, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cStatus++) {
        const size_t tCoord[] = {(*lastCoord)["Status"]+j};
		ncErr = nc_put_var1_long(ncID, tID, tCoord, cStatus);
    }
    (*lastCoord)["Status"]+=j;
    err = CheckNC(ncErr); {if (err) return err;}

           // ID:
    tID = (*this->ncVarIDs)["ID"];
    for(cID = this->id, j = 0; ncErr == NC_NOERR && j < this->pCount; j++, cID++) {
        const size_t tCoord[] = {(*lastCoord)["ID"]+j};
        ncErr = nc_put_var1_long(ncID, tID, tCoord, cID);
    }
    (*lastCoord)["ID"]+=j;
    err = CheckNC(ncErr); {if (err) return err;}

    return err;
}

OSErr NetCDFStore::fClose(int ncID) {

    int ncErr;

    ncErr = nc_close(ncID);
    return CheckNC(ncErr);
}


OSErr NetCDFStore::CheckNC(int ncErr) {
    
	char errStr[256];
	OSErr err = 0;
	
    if(ncErr != NC_NOERR) {
		err = -1;
        switch(ncErr) {

            case NC_ENOMEM:
					strcpy(errStr,"We are unable to allocate the memory required to perform this task.\n");
					//strcat(errStr,"The run will continue, notwithstanding. (Exception caught in write mode.)");
					printError(errStr);
					//cerr << "We are unable to allocate the memory required to perform this task.\n";
					//cerr << "The run will continue, notwithstanding. (Exception caught in write mode.)\n";
					return err;
            case NC_EBADID:
					strcpy(errStr,"Error: NC_EBADID.\n");
					//strcat(errStr,"We will attempt to continue the run.");
					printError(errStr);
					//cerr << "Error: NC_EBADID.\n";
					//cerr << "We will attempt to continue the run.\n";
                    return err;
            case NC_EINVALCOORDS:
					strcpy(errStr,"Error: NC_EINVALCOORDS.\n");
					//strcat(errStr,"We will attempt to continue the run.");
					printError(errStr);
					//cerr << "Error: NC_EINVALCOORDS.\n";
					//cerr << "We will attempt to continue the run.\n";
                    return err;
            case NC_EINDEFINE:
					strcpy(errStr,"Error: NC_EINDEFINE.\n");
					//strcat(errStr,"We will attempt to continue the run.");
					printError(errStr);
					//cerr << "Error: NC_EINDEFINE.\n";
					//cerr << "We will attempt to continue the run.\n";
                    return err;
            default:
                // ..
				sprintf(errStr,"There was an error trying to write the NetCDF file - ncErr = %ld.\n",ncErr);
				//strcpy(errStr,"There was an error trying to write the NetCDF file.\n");
				//strcat(errStr,"We will attempt to continue the run.");
				printError(errStr);
				return err;
        }
    }

    else return err;
}