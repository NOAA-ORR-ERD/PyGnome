/*
 *  Weatherers_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Weatherers_c.h"
// may not need anything else...
#include "CompFunctions.h"
#include "GEOMETRY.H"
#include "Units.h"
#include "Replacements.h"

using namespace std;


OSErr emulsify(int n, unsigned long step_len, double *frac_water, double *interfacial_area, double *frac_evap, double *droplet_diameter, long *age, double *bulltime, double k_emul, double emul_time, double emul_C, double S_max, double Y_max, double drop_max)
{
	OSErr err = 0;
	double Y, S;
	//Seconds start;
	double start, le_age;	// convert to double for calculations
	//char errmsg[256];
	
	for (int i=0; i < n; i++)
	{
		S = interfacial_area[i];
		le_age = age[i];
		//sprintf(errmsg,"for i = %ld, S = %lf, age = %lf, emul_time =%lf, frac_evap[i] = %lf\n",i,S,le_age,emul_time,frac_evap[i]);
		//printNote(errmsg);
		//sprintf(errmsg,"k_emul = %lf, emul_C = %lf, Y_max = %lf, S_max = %lf, drop_max = %lf\n",k_emul,emul_C,Y_max,S_max,drop_max);
		//printNote(errmsg);
		//if ((age[i] >= emul_time && emul_time >= 0.) || frac_evap[i] >= emul_C && emul_C > 0.)
		if ((le_age >= emul_time && emul_time >= 0.) || frac_evap[i] >= emul_C && emul_C > 0.)
		{
			if (emul_time > 0.)	// user has set value
				start = emul_time;
			else
			{
				if (bulltime[i] < 0.)
				{
					//start = age[i];
					//bulltime[i] = age[i];
					start = le_age;
					bulltime[i] = le_age;
				}
				else
					start = bulltime[i];
			}
			//S = S + k_emul * step_len * exp( (-k_emul / S_max) * (age[i] - start));
			S = S + k_emul * step_len * exp( (-k_emul / S_max) * (le_age - start));
			if (S > S_max)
				S = S_max;
		}
		else
		{
			S = 0.;
		}
		
		if (S < ((6.0 / drop_max) * (Y_max / (1.0 - Y_max))))
		{
			Y = S * drop_max / (6.0 + (S * drop_max));
			//droplet_diameter[i] = drop_max;
		//sprintf(errmsg,"Y = %lf, S = %lf, droplet_diameter = %lf\n",Y,S,droplet_diameter[i]);
		//printNote(errmsg);
		}
		else
		{
			Y = Y_max;
			//droplet_diameter[i] = (6.0 / S) * (Y_max / (1.0 - Y_max));
		//sprintf(errmsg,"Y = %lf, S = %lf, droplet_diameter = %lf\n",Y,S,droplet_diameter[i]);
		//printNote(errmsg);
		}
		
		if (Y < 0) { err = -1; return err;}
		
		frac_water[i] = Y;
		interfacial_area[i] = S;
	}
	
	return err;
}

