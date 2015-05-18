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


OSErr emulsify(int n, unsigned long step_len, double *frac_water, double *interfacial_area, double *frac_evap, int32_t *age, double *bulltime, double k_emul, double emul_time, double emul_C, double S_max, double Y_max, double drop_max)
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
		//sprintf(errmsg,"Y = %lf, S = %lf\n",Y,S);
		//printNote(errmsg);
		}
		else
		{
			Y = Y_max;
		//sprintf(errmsg,"Y = %lf, S = %lf\n",Y,S);
		//printNote(errmsg);
		}
		
		if (Y < 0) { err = -1; return err;}
		
		frac_water[i] = Y;
		interfacial_area[i] = S;
	}
	
	return err;
}

OSErr disperse(int n, unsigned long step_len, double *frac_water, double *le_mass, double *le_viscosity, double *le_density, double *fay_area, double *d_disp, double *d_sed, double frac_breaking_waves, double disp_wave_energy, double wave_height, double visc_w, double rho_w, double C_sed, double V_entrain, double ka)
{
	OSErr err = 0;
	
	double rho, mass, Y, C_Roy, C_disp, A, q_disp;

	double Vemul, thickness, droplet, speed;
	double V_refloat, q_refloat;
	double C_oil, Q_sed, visc, g = 9.80665;
	double De = disp_wave_energy, fbw = frac_breaking_waves, Hrms = wave_height;

	// would need to pass in Vol_0
	/*if (Vol_0 > 100. && visc > 40. * 1.e-6)
	{
		fbw *= pow(100. / Vol_0, .2);
	}*/
	V_entrain = 3.9e-8;
	C_disp = pow(De, 0.57) * fbw; // dispersion term at current time

	for (int i=0; i < n; i++)
	{
		rho = le_density[i];	// pure oil density
		mass = le_mass[i];	
		visc = le_viscosity[i]; // oil (or emulsion) viscosity

		Y = frac_water[i]; // water fraction
		if (Y>=1) {d_disp[i]=0; continue;}	// shouldn't happen

		C_Roy = 2400.0 * exp(-73.682 * sqrt(visc)); // Roy's constant

		Vemul = (le_mass[i] / le_density[i])  / (1.0 - Y); // emulsion volume (m3)
		A = fay_area[i];
		if (A>0) thickness = Vemul / A;

		q_disp = C_Roy * C_disp * V_entrain * (1.0 - Y) * A / rho; // (m3/sec)

		if (C_sed > 0.0 && thickness >= 1.0e-4) {
			// sediment load and > min thick
			droplet = 0.613 * thickness;
			rho_w = 1020;	// this is set to 997 in water module...
			speed = droplet * droplet * g * (1.0 - rho / rho_w) / (18.0 * visc_w); //vert drop vel
			V_refloat = 0.588 * (pow(thickness, 1.7) - 5.0e-8); //vol of refloat oil/wave p
			if (V_refloat < 0.0)
				V_refloat = 0.0;
	
			q_refloat = C_Roy * C_disp * V_refloat * A; //(kg/m2-sec) mass rate of emulsion
			C_oil = q_refloat * step_len / (speed * step_len + 1.5 * Hrms);
			Q_sed = 1.6 * ka * sqrt(Hrms * De * fbw / (rho_w * visc_w)) * C_oil * C_sed / rho; //vol rate		
		
		}
		else
			Q_sed=0.0;
	
		//d_disp[i] = (q_disp + (1.0 - Y) * Q_sed) * step_len; //total vol oil loss
		d_disp[i] = q_disp * step_len; //total vol oil loss due to dispersion
		d_sed[i] = (1.0 - Y) * Q_sed * step_len; //total vol oil loss due to sedimentation
		
		d_disp[i] = d_disp[i] * le_density[i]; 
		d_sed[i] = d_sed[i] * le_density[i]; 
	
		//if (d_disp[i] > le_mass[i])
			//d_disp[i] = le_mass[i];
		if (d_disp[i] + d_sed[i] > le_mass[i])
		{
			double ratio = d_disp[i] / (d_disp[i] + d_sed[i]);
			d_disp[i] = ratio * le_mass[i];
			d_sed[i] = le_mass[i] - d_disp[i];
		}
	}

	return err;
}