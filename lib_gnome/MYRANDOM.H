
#ifndef __MYRANDOM__
#define __MYRANDOM__

#include "Basics.h"
#include "TypeDefs.h"
#include "ExportSymbols.h"

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

void ResetMyRandomSeed(void);
float GNOMEDLL_API MyRandom (void);

void GNOMEDLL_API ResetAllRandomSeeds(void);

#endif
