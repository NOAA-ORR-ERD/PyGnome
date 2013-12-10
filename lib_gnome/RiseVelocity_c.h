/*
 *  RiseVelocity_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __RiseVelocity_c__
#define __RiseVelocity_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "Mover_c.h"
#include "ExportSymbols.h"

// functions for computing rise velocity from droplet size
// get_rise_velocity is exposed to Cython/Python for PyGnome
OSErr DLL_API get_rise_velocity(int n, double *rise_velocity, double *le_density, double *le_droplet_size, double water_viscosity, double water_density);
double GetRiseVelocity(double le_density, double le_droplet_size, double water_viscosity, double water_density);

class DLL_API RiseVelocity_c : virtual public Mover_c {

public:
	//double water_density;
	//double water_viscosity;

#ifndef pyGNOME
	RiseVelocity_c (TMap *owner, char *name);
#endif
	RiseVelocity_c();

	virtual OSErr PrepareForModelRun();
	virtual OSErr PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList);

	virtual void ModelStepIsDone();

	virtual WorldPoint3D GetMove(const Seconds& model_time, Seconds timeStep,
								 long setIndex, long leIndex, LERec *theLE, LETYPE leType);

	OSErr get_move(int n, unsigned long model_time, unsigned long step_len,
				   WorldPoint3D *ref, WorldPoint3D *delta,
				  // double *rise_velocity, double *density, double *droplet_size,
				   double *rise_velocity,
				   short *LE_status, LEType spillType, long spill_ID);

protected:

	void Init();

};

#endif
