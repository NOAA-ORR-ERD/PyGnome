//#include "CROSS.H"
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
        long *id;
		map<string, int> *ncVarIDs, *ncDimIDs;
	
	
    public:
                    NetCDFStore();
        static bool Capture(TModel* model, bool uncertain, map<string, int> *ncVarIDs, map<string, int> *ncDimIDs);
        static bool Create(char* outFile, bool overwrite, int* ncID);
        static bool Define(TModel* model, bool uncertain, map<string, int> *ncVarIDs, map<string, int> *ncDimIDs);
        static bool fClose(int ncID);
        static bool Open(char* inFile, int* ncID);
        static bool CheckNC(int ncErr);
               bool Write(TModel* model, bool threeMovement, bool uncertain);
               bool Read();
};
