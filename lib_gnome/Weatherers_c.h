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
#include "ExportSymbols.h"

// emulsify and disperse are exposed to Cython/Python for PyGnome

//OSErr DLL_API emulsify(int n, unsigned long step_len, double *frac_water, double *le_interfacial_area, double *frac_evap, double *droplet_diameter, unsigned long *age, unsigned long *bulltime, double k_emul, unsigned long emul_time, double emul_C, double S_max, double Y_max, double drop_max);
OSErr DLL_API emulsify(int n, unsigned long step_len, double *frac_water, double *le_interfacial_area, double *frac_evap, int32_t *age, double *bulltime, double k_emul, double emul_time, double emul_C, double S_max, double Y_max, double drop_max);
OSErr DLL_API disperse(int n, unsigned long step_len, double *frac_water, double *le_mass, double *le_viscosity, double *le_density, double *fay_area, double *d_disp, double *d_sed, double frac_breaking_waves, double disp_wave_energy, double wave_height, double visc_w, double rho_w, double C_sed, double V_entrain, double ka);

#endif
