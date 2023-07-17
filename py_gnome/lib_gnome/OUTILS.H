
#ifndef __OSSMUTILS__
#define __OSSMUTILS__

#define	KNOTSTOMETERSPERSEC		.5144
#define MILESTOMETERSPERSEC		.44704
#define KMHRTOMETERSPERSEC		.277777

#define	kNumDirCodes	16

//#define kNumSpeedCodes	2		// used in wind-speed-units selection
#define	kFudgeFlag		-3	// don't think this is used
#define	kUndefined		-1
#define	kKnots			1
#define kMetersPerSec	2
#define kMilesPerHour	3
#define kKilometersPerHour	4

#define	kCMS			1		// Transport units for Hydrology time files
#define kKCMS			2
#define kCFS			3
#define kKCFS			4

typedef struct {
	double  direction;
	char	dirText [32];
} DirectionRec;

long StrToSpeedUnits(char* str);
void SpeedUnitsToStr(long unitCode, char *unitStr);

void	ConvertToUV (double magnitude, double angle, double *u, double *v);
double	ConvertToDegrees (char *direction);
double	ConvertToDegrees (short direction);
void 	ConvertToDirection (double degrees, char *directionStr);
void 	ConvertToUnits (long unitCode, char *unitStr);
void 	ConvertToUnitsShort (long unitCode, char *unitStr);
void 	ConvertToTransportUnits (long unitCode, char *unitStr);
double	UorV (VelocityRec vector, short index);
double	Hermite (double v1, double s1, double t1,
			     double v2, double s2, double t2,
			     double time);
double speedconversion(long speedUnits);

#endif
