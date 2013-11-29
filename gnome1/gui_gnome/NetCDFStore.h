
#ifndef __NetCDFStore__
#define __NetCDFStore__

#include "netcdf.h"
#include <map>
#include <string>


class TModel;

class NetCDFStore // Houses stepwise run information for NetCDF output (and input)
{
    public:               
        short ncID;
        double time;
        long pCount;
        float *lon;
        float *lat;
        float *depth;
        float *mass;
        long *age;
        short *flag;
        long *status;
        long *id;
		map<string, int> *ncVarIDs, *ncDimIDs;
	
	
    public:
                    NetCDFStore();
        static OSErr Capture(TModel* model, bool uncertain, map<string, int> *ncVarIDs, map<string, int> *ncDimIDs);
        static OSErr Create(char* outFile, bool overwrite, int* ncID);
        static OSErr Define(TModel* model, bool uncertain, map<string, int> *ncVarIDs, map<string, int> *ncDimIDs);
        static OSErr fClose(int ncID);
        static OSErr Open(char* inFile, int* ncID);
        static OSErr CheckNC(int ncErr);
			   OSErr Write(TModel* model, bool uncertain);
               OSErr Read();
};

#endif