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


OSErr emulsify(int n, unsigned long step_len,
			   double *frac_water,
			   double *interfacial_area,
			   double *frac_evap,
			   int32_t *age,
			   double *bulltime,
			   double k_emul,
			   double emul_time,
			   double emul_C,
			   double S_max,
			   double Y_max,
			   double drop_max)
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


OSErr adios2_disperse(int n, unsigned long step_len,
                      double *frac_water,
                      double *le_mass,
                      double *le_viscosity,
                      double *le_density,
                      double *fay_area,
                      double *d_disp,
                      double *d_sed,
                      double frac_breaking_waves,
                      double disp_wave_energy,
                      double wave_height,
                      double visc_w,
                      double rho_w,
                      double C_sed,
                      double V_entrain,
                      double ka)
{
	OSErr err = 0;

	double g = 9.80665;
	double De = disp_wave_energy;
	double fbw = frac_breaking_waves;
	double Hrms = wave_height;

	double C_disp = pow(De, 0.57) * fbw; // dispersion term at current time

	for (int i=0; i < n; i++)
	{
		double rho = le_density[i];	// pure oil density
		double mass = le_mass[i];
		double visc = le_viscosity[i]; // oil (or emulsion) viscosity
		double Y = frac_water[i]; // water fraction
		double A = fay_area[i];

		double d_disp_out = 0.0;
		double d_sed_out = 0.0;

		if (Y >= 1) {
		    d_disp[i] = 0.0;
		    d_sed[i] = 0.0;
		    continue;
		}  // shouldn't happen

		double C_Roy = 2400.0 * exp(-73.682 * sqrt(visc)); // Roy's constant

        // surface oil slick thickness
		double thickness = 0.0;
		if (A > 0) {
            // emulsion volume (m3)
            double Vemul = (mass / rho)  / (1.0 - Y);

            thickness = Vemul / A;
		}

		// mass rate of oil driven into the first 1.5 wave height (m3/sec)
		double Q_disp = C_Roy * C_disp * V_entrain * (1.0 - Y) * A / rho;

		// Net mass loss rate due to sedimentation (kg/s)
		// (Note: why not in m^3/s???)
		double Q_sed = 0.0;
		if (C_sed > 0.0 && thickness >= 1.0e-4) {
			// average droplet size based on surface oil slick thickness
			double droplet = 0.613 * thickness;

			// droplet average rise velocity
			double speed = (droplet * droplet * g *
			                (1.0 - rho / rho_w) /
			                (18.0 * visc_w));

			// vol of refloat oil/wave p
			double V_refloat = 0.588 * (pow(thickness, 1.7) - 5.0e-8);
			if (V_refloat < 0.0)
				V_refloat = 0.0;

			// (kg/m2-sec) mass rate of emulsion
			double q_refloat = C_Roy * C_disp * V_refloat * A;

			double C_oil = (q_refloat * step_len /
			                (speed * step_len + 1.5 * Hrms));

			//vol rate
			Q_sed = (1.6 * ka *
			         sqrt(Hrms * De * fbw / (rho_w * visc_w)) *
			         C_oil * C_sed / rho);
		}

		//total vol oil loss due to dispersion
		d_disp_out = Q_disp * step_len;

		//total vol oil loss due to sedimentation
		d_sed_out = (1.0 - Y) * Q_sed * step_len;

		d_disp_out *= rho;
		d_sed_out *= rho;

		if (d_disp_out + d_sed_out > mass) {
			double ratio = d_disp_out / (d_disp_out + d_sed_out);

			d_disp_out = ratio * mass;
			d_sed_out = mass - d_disp_out;
		}

        // assign our final values to our output arrays
		d_disp[i] = d_disp_out;
        d_sed[i] = d_sed_out;

	}

	return err;
}
