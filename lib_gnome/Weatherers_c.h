/*
 *  Weatherers_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Weatherers_c__
#define __Weatherers_c__

#include "Basics.h"
#include "TypeDefs.h"
//#include "Mover_c.h"
#include "ExportSymbols.h"

// emulsify is exposed to Cython/Python for PyGnome

//OSErr DLL_API emulsify(int n, unsigned long step_len, double *frac_water, double *le_interfacial_area, double *frac_evap, double *droplet_diameter, unsigned long *age, unsigned long *bulltime, double k_emul, unsigned long emul_time, double emul_C, double S_max, double Y_max, double drop_max);
OSErr DLL_API emulsify(int n, unsigned long step_len, double *frac_water, double *le_interfacial_area, double *frac_evap, double *droplet_diameter, int *age, double *bulltime, double k_emul, double emul_time, double emul_C, double S_max, double Y_max, double drop_max);

#endif
