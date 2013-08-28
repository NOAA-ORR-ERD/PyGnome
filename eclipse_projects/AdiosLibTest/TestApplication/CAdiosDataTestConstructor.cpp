/*
 * CAdiosDataTestConstructor.cpp
 *
 *  Created on: Mar 21, 2013
 *      Author: james.makela
 */

#include <iostream>
#include "cute.h"
#include "ide_listener.h"
#include "cute_runner.h"

#include "AdiosComp.h"
#include "CAdiosDataTestConstructor.h"

using namespace std;

CAdiosDataTestConstructor::CAdiosDataTestConstructor() {
	// Auto-generated constructor stub
	refcount++;
    adiosData.reset(new CAdiosData);
	cout << ">> ::CAdiosDataTestConstructor(), refcount = " << refcount << endl;
	setup();
}

CAdiosDataTestConstructor::CAdiosDataTestConstructor(CAdiosDataTestConstructor const& other) {
	// Auto-generated constructor stub
	refcount = other.refcount + 1;
	adiosData = other.adiosData;
	cout << ">> ::CAdiosDataTestConstructor(const&), refcount = " << refcount << endl;
	setup();
}

CAdiosDataTestConstructor::~CAdiosDataTestConstructor() {
	// Auto-generated destructor stub
	cout << ">> ::~CAdiosDataTestConstructor(), refcount = " << refcount << endl;
	teardown();
}

void CAdiosDataTestConstructor::operator()()
{
	// I would love some better documentation for CUTE unit testing.
	// but it seems that this is where CUTE takes the class and
	// runs the class tests.
	test();
}

void CAdiosDataTestConstructor::setup()
{
	// here is where you would setup any input conditions
}

void CAdiosDataTestConstructor::teardown()
{
	// here is where you would perform any post-test cleanup
}

void CAdiosDataTestConstructor::test()
{
	TestLocalDataMembers();
	TestContainedSource();
	TestContainedLeak();
	TestContainedTankDisplay();
	TestContainedEnvironment();
	TestContainedFay();
	TestContainedLE();
	TestContainedCleanup();
	TestContainedRelease();
	TestContainedInitialOil();
	TestContainedSlick();
	TestContainedSink();
	TestContainedMisc();
	TestContainedWarning();
	TestContainedUncertainty();
	TestContainedRobert();
	TestContainedResetFlag();

}


void CAdiosDataTestConstructor::TestLocalDataMembers()
{
	ASSERT_EQUALM("adiosData->Spillhour not initialized", (double)0, adiosData->Spillhour);
}

void CAdiosDataTestConstructor::TestContainedSource()
{
	for (int i = 0; i < MMAXHOUR; i++) {
		ASSERT_EQUALM("adiosData->Source not initialized", (double)0, adiosData->Source[i]);
	}
}

void CAdiosDataTestConstructor::TestContainedLeak()
{
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", false, adiosData->Leak.pipe);
	TestContainedLeakInstantData();
	TestContainedLeakContinuousData();
	TestContainedLeakTankData();
}

void CAdiosDataTestConstructor::TestContainedLeakInstantData()
{
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", false, adiosData->Leak.Instant.thereIsAnInstantRelease);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Instant.startHour);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Instant.volume);
}
void CAdiosDataTestConstructor::TestContainedLeakContinuousData()
{
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", false, adiosData->Leak.Continuous.thereIsAnInstantRelease);

	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Continuous.startHour);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Continuous.endHour);

	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Continuous.initialRate);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Continuous.finalRate);
}

void CAdiosDataTestConstructor::TestContainedLeakTankData()
{
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", false, adiosData->Leak.Tank.tankIsLeaking);

	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.startHour);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.endHour);

	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", false, adiosData->Leak.Tank.isVented);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", false, adiosData->Leak.Tank.isAground);

	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.Q);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.Dh);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.Ah);

	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)0.65, adiosData->Leak.Tank.Cd);

	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.Zwo);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.At);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.Zt);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.Zoi);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.Vt);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.Voil);
	ASSERT_EQUALM("adiosData->Leak.Instant not initialized", (double)-999, adiosData->Leak.Tank.Zh);
}



void CAdiosDataTestConstructor::TestContainedTankDisplay()
{
	ASSERT_EQUALM("adiosData->TankDisplay not initialized", (int)-999, adiosData->TankDisplay.numTimeSteps);
	ASSERT_EQUALM("adiosData->TankDisplay not initialized", (int)-999, adiosData->TankDisplay.tstop);
	for (int i = 0; i < MNTS; i++) {
		ASSERT_EQUALM("adiosData->TankDisplay not initialized", (double)-999, adiosData->TankDisplay.Zw[i]);
	}
	for (int i = 0; i < MNTS; i++) {
		ASSERT_EQUALM("adiosData->TankDisplay not initialized", (double)-999, adiosData->TankDisplay.Z[i]);
	}
	for (int i = 0; i < MNTS; i++) {
		ASSERT_EQUALM("adiosData->TankDisplay not initialized", (double)-999, adiosData->TankDisplay.Q[i]);
	}
	for (int i = 0; i < MNTS; i++) {
		ASSERT_EQUALM("adiosData->TankDisplay not initialized", (double)-999, adiosData->TankDisplay.time[i]);
	}
}


void CAdiosDataTestConstructor::TestContainedEnvironment()
{
	ASSERT_EQUALM("adiosData->Environ not initialized", (double)0, adiosData->Environ.fetchN);
	ASSERT_EQUALM("adiosData->Environ not initialized", (double)0, adiosData->Environ.fetchE);
	ASSERT_EQUALM("adiosData->Environ not initialized", (double)0, adiosData->Environ.fetchW);
	ASSERT_EQUALM("adiosData->Environ not initialized", (double)0, adiosData->Environ.fetchS);
	TestContainedEnvironmentWindData();
	TestContainedEnvironmentWaveData();
	TestContainedEnvironmentWaterData();
	TestContainedEnvironmentAirData();
}

void CAdiosDataTestConstructor::TestContainedEnvironmentWindData()
{
	for (int i = 0; i < MMAXHOUR; i++) {
		ASSERT_EQUALM("adiosData->Environ.Wind not initialized", (double)-999, adiosData->Environ.Wind[i].speed);
		ASSERT_EQUALM("adiosData->Environ.Wind not initialized", (double)-999, adiosData->Environ.Wind[i].angle);
		ASSERT_EQUALM("adiosData->Environ.Wind not initialized", (double)-999, adiosData->Environ.Wind[i].sine);
		ASSERT_EQUALM("adiosData->Environ.Wind not initialized", (double)-999, adiosData->Environ.Wind[i].cosine);
		ASSERT_EQUALM("adiosData->Environ.Wind not initialized", (double)-999, adiosData->Environ.Wind[i].fetch);
	}
}

void CAdiosDataTestConstructor::TestContainedEnvironmentWaveData()
{
	for (int i = 0; i < MMAXHOUR; i++) {
		ASSERT_EQUALM("adiosData->Environ.Wave not initialized", (double)-999, adiosData->Environ.Wave[i].H0);
		ASSERT_EQUALM("adiosData->Environ.Wave not initialized", (double)-999, adiosData->Environ.Wave[i].Hrms);
		ASSERT_EQUALM("adiosData->Environ.Wave not initialized", (double)-999, adiosData->Environ.Wave[i].HrmsH);
		ASSERT_EQUALM("adiosData->Environ.Wave not initialized", (double)-999, adiosData->Environ.Wave[i].HrmsL);

		ASSERT_EQUALM("adiosData->Environ.Wave not initialized", (double)-999, adiosData->Environ.Wave[i].f_bw);
		ASSERT_EQUALM("adiosData->Environ.Wave not initialized", (double)-999, adiosData->Environ.Wave[i].f_bwH);
		ASSERT_EQUALM("adiosData->Environ.Wave not initialized", (double)-999, adiosData->Environ.Wave[i].f_bwL);

		ASSERT_EQUALM("adiosData->Environ.Wave not initialized", (double)-999, adiosData->Environ.Wave[i].De);
		ASSERT_EQUALM("adiosData->Environ.Wave not initialized", (double)-999, adiosData->Environ.Wave[i].DeH);
		ASSERT_EQUALM("adiosData->Environ.Wave not initialized", (double)-999, adiosData->Environ.Wave[i].DeL);
	}
}

void CAdiosDataTestConstructor::TestContainedEnvironmentWaterData()
{
	ASSERT_EQUALM("adiosData->Environ.Water not initialized", (double)-999, adiosData->Environ.Water.temp);
	ASSERT_EQUALM("adiosData->Environ.Water not initialized", (double)-999, adiosData->Environ.Water.density);

	ASSERT_EQUALM("adiosData->Environ.Water not initialized", (double)10e-7, adiosData->Environ.Water.viscosity);

	ASSERT_EQUALM("adiosData->Environ.Water not initialized", (double)-999, adiosData->Environ.Water.C_sed);
	ASSERT_EQUALM("adiosData->Environ.Water not initialized", (double)-999, adiosData->Environ.Water.speed);
	ASSERT_EQUALM("adiosData->Environ.Water not initialized", (double)-999, adiosData->Environ.Water.angle);
	ASSERT_EQUALM("adiosData->Environ.Water not initialized", (double)-999, adiosData->Environ.Water.dxCur);
	ASSERT_EQUALM("adiosData->Environ.Water not initialized", (double)-999, adiosData->Environ.Water.dyCur);
}

void CAdiosDataTestConstructor::TestContainedEnvironmentAirData()
{
	ASSERT_EQUALM("adiosData->Environ.Water not initialized", (double)-999, adiosData->Environ.Air.temp);

	// TODO: I don't know if this is a good initial value for air pressure
	ASSERT_EQUALM("adiosData->Environ.Water not initialized", (double)101300, adiosData->Environ.Air.pressure);

	ASSERT_EQUALM("adiosData->Environ.Water not initialized", (double)1.2, adiosData->Environ.Air.density);
}



void CAdiosDataTestConstructor::TestContainedFay()
{
	ASSERT_EQUALM("adiosData->Fay not initialized", (double)1.53, adiosData->Fay.k1);
	ASSERT_EQUALM("adiosData->Fay not initialized", (double)1.21, adiosData->Fay.k2);
	ASSERT_EQUALM("adiosData->Fay not initialized", (double)-999, adiosData->Fay.delta);
	ASSERT_EQUALM("adiosData->Fay not initialized", (double)-999, adiosData->Fay.C_Fay);
}

void CAdiosDataTestConstructor::TestContainedLE()
{
	for (int i = 0; i < MMAXLE; i++) {
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].hour);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].x);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].y);

		for (int j; j < MMAXHOUR; j++) {
			ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].Area[j]);
		}

		for (int j; j < MMAXHOUR; j++) {
			ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].Area0[j]);
		}

		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].initialVolume);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].volume);

		for (int j; j < MMAXHOUR; j++) {
			ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].thick[j]);
		}

		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].Y);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].S);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].Bulltime);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].f_evap);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].visc);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].rho);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].vp);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].benz);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].mv);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].vol_disp);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].Spill);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].FayDif);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].alpha);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].evex);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].minArea);
		ASSERT_EQUALM("adiosData->LE not initialized", (double)-999, adiosData->LE[i].benzeneVol);
	}
}

void CAdiosDataTestConstructor::TestContainedCleanup()
{
	TestContainedCleanupDispersantData();
	TestContainedCleanupSkimmerData();
	TestContainedCleanupBurnData();
	TestContainedCleanupSmokeData();
}

void CAdiosDataTestConstructor::TestContainedCleanupDispersantData()
{
	for (int i = 0; i < 5; i++) {
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (int)-999, adiosData->Cleanup.dispersant.start[i]);
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (int)-999, adiosData->Cleanup.dispersant.duration[i]);
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-999, adiosData->Cleanup.dispersant.f_Spr[i]);
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-999, adiosData->Cleanup.dispersant.f_eff[i]);
	}
}

void CAdiosDataTestConstructor::TestContainedCleanupSkimmerData()
{
	for (int i = 0; i < 5; i++) {
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (int)-999, adiosData->Cleanup.skimmer.start[i]);
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (int)-999, adiosData->Cleanup.skimmer.duration[i]);
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-999, adiosData->Cleanup.skimmer.skimRate[i]);
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-999, adiosData->Cleanup.skimmer.skimVol[i]);
	}
}

void CAdiosDataTestConstructor::TestContainedCleanupBurnData()
{
	for (int i = 0; i < 5; i++) {
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (int)-999, adiosData->Cleanup.burn.start[i]);
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (int)-999, adiosData->Cleanup.burn.duration[i]);
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-999, adiosData->Cleanup.burn.thickness[i]);
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-999, adiosData->Cleanup.burn.area[i]);
		ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-1, adiosData->Cleanup.burn.rate[i]);
	}
	ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)4e-5, adiosData->Cleanup.burn.reg.vdefault);
	ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-999, adiosData->Cleanup.burn.reg.user);
}

void CAdiosDataTestConstructor::TestContainedCleanupSmokeData()
{
	ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-999, adiosData->Cleanup.smoke.fsmoke.vdefault);
	ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-999, adiosData->Cleanup.smoke.fsmoke.user);

	ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)1.5e-7, adiosData->Cleanup.smoke.LOC.vdefault);
	ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-999, adiosData->Cleanup.smoke.LOC.user);

	ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)5, adiosData->Cleanup.smoke.u10.vdefault);
	ASSERT_EQUALM("adiosData->Cleanup not initialized", (double)-999, adiosData->Cleanup.smoke.u10.user);

	ASSERT_EQUALM("adiosData->Cleanup not initialized", (int)100, adiosData->Cleanup.smoke.NumPts);
}


void CAdiosDataTestConstructor::TestContainedRelease()
{
	for (int i = 0; i < MMAXHOUR; i++) {
		ASSERT_EQUALM("adiosData->Release not initialized", (double)-999, adiosData->Release[i].vol);
		ASSERT_EQUALM("adiosData->Release not initialized", (double)-999, adiosData->Release[i].volsum);
		ASSERT_EQUALM("adiosData->Release not initialized", (double)-999, adiosData->Release[i].area);
		ASSERT_EQUALM("adiosData->Release not initialized", (double)-999, adiosData->Release[i].R0);
		ASSERT_EQUALM("adiosData->Release not initialized", (double)-999, adiosData->Release[i].FayDiff);
		ASSERT_EQUALM("adiosData->Release not initialized", (int)0, adiosData->Release[i].LErel);
		ASSERT_EQUALM("adiosData->Release not initialized", (int)0, adiosData->Release[i].LEsum);
		ASSERT_EQUALM("adiosData->Release not initialized", (double)-999, adiosData->Release[i].volperLE);
	}
}


void CAdiosDataTestConstructor::TestContainedInitialOil()
{
	std::string expected_oil_type("default");

	ASSERT_EQUALM("adiosData->InitialOil not initialized", 0, expected_oil_type.compare(adiosData->InitialOil.type));

	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.API);

	for (int i = 0; i < 6; i++) {
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.rhoRef[i].value);
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.rhoRef[i].refTemp);
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)0, adiosData->InitialOil.rhoRef[i].fractEvap);

		ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.viscRef[i].value);
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.viscRef[i].refTemp);
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)0, adiosData->InitialOil.viscRef[i].fractEvap);

		ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.dyViscRef[i].value);
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.dyViscRef[i].refTemp);
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)0, adiosData->InitialOil.dyViscRef[i].fractEvap);
	}

	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.rho0);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.visc0);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.PourPt);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Ymax);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Smax);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.k0Y);

	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)3.9e-8, adiosData->InitialOil.V_entrain);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)1e-5, adiosData->InitialOil.drop_max);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)10e-7, adiosData->InitialOil.drop_min);

	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.R_nu);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)0.0001, adiosData->InitialOil.ka);

	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Bullwinkle.InitialValue);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Bullwinkle.SecondValue);

	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Bulltime.InitialValue);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Bulltime.SecondValue);

	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.f_Asph);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Nickel);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Vanadium);

	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)0.18, adiosData->InitialOil.CF);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)0, adiosData->InitialOil.CFF);

	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Dist.distFlag);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Dist.NumDistCuts);

	for (int i = 0; i < MNDC; i++) {
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Dist.Temp[i]);
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.Dist.f_Dist[i]);
	}

	ASSERT_EQUALM("adiosData->InitialOil not initialized", (int)-999, adiosData->InitialOil.PsC.NumComp);
	for (int i = 0; i < MAXPC; i++) {
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (int)-999, adiosData->InitialOil.PsC.molVol[i]);
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (int)-999, adiosData->InitialOil.PsC.molWt[i]);
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (int)-999, adiosData->InitialOil.PsC.Pv[i]);
		ASSERT_EQUALM("adiosData->InitialOil not initialized", (int)-999, adiosData->InitialOil.PsC.fraction[i]);
	}

	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.T0);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.TG);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.f_benzMass);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.f_benz);
	ASSERT_EQUALM("adiosData->InitialOil not initialized", (double)-999, adiosData->InitialOil.minThick);
}


void CAdiosDataTestConstructor::TestContainedSlick()
{
	for (int i = 0; i < 480; i++) {
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_oil.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_oil.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_oil.average);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_spill.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_spill.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_spill.average);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_emul.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_emul.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_emul.average);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_evap.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_evap.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_evap.average);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_disp.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_disp.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_disp.average);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_chem.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_chem.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_chem.average);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_burn.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_burn.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_burn.average);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_mech.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_mech.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_mech.average);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_beach.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_beach.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].V_beach.average);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].benz.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].benz.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].benz.average);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].benz.avr2);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].rho.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].rho.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].rho.average);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].rho.avr2);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].visc.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].visc.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].visc.average);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].visc.avr2);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].Y.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].Y.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].Y.average);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].Y.avr2);

		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].Area.high);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].Area.low);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Slick[i].Area.average);
	}
}

void CAdiosDataTestConstructor::TestContainedSink()
{
	for (int i = 0; i < MMAXHOUR; i++) {
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Sink[i].Q_burn);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Sink[i].Q_mech);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Sink[i].fRate_chem);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Sink[i].fSpray);
		ASSERT_EQUALM("adiosData->Slick not initialized", (double)-999, adiosData->Sink[i].Q_beach);
	}
}

void CAdiosDataTestConstructor::TestContainedMisc()
{
	ASSERT_EQUALM("adiosData->Misc not initialized", (int)6, adiosData->Misc.maxReferenceNumber);
	ASSERT_EQUALM("adiosData->Misc not initialized", (double)1000000, adiosData->Misc.maxFetch);

	ASSERT_EQUALM("adiosData->Misc not initialized", (int)-999, adiosData->Misc.maxLECount);
	ASSERT_EQUALM("adiosData->Misc not initialized", (int)-999, adiosData->Misc.maxHour);
	ASSERT_EQUALM("adiosData->Misc not initialized", (int)-999, adiosData->Misc.maxTimeSteps);

	ASSERT_EQUALM("adiosData->Misc not initialized", (double)30, adiosData->Misc.maxWindSpeedInMetersPerSec);
	ASSERT_EQUALM("adiosData->Misc not initialized", (double)1, adiosData->Misc.minWindSpeedInMetersPerSec);
	ASSERT_EQUALM("adiosData->Misc not initialized", (double)0.011, adiosData->Misc.minBenezeneConcentrationInPpm);

	ASSERT_EQUALM("adiosData->Misc not initialized", (int)240, adiosData->Misc.userSelectedLENum);
	ASSERT_EQUALM("adiosData->Misc not initialized", (double)-999, adiosData->Misc.userSelectedMaxSpillArea);

	ASSERT_EQUALM("adiosData->Misc not initialized", (int)-999, adiosData->Misc.NumSpills);
	ASSERT_EQUALM("adiosData->Misc not initialized", (int)-999, adiosData->Misc.timeStepSizeInSecs);
	ASSERT_EQUALM("adiosData->Misc not initialized", (int)-999, adiosData->Misc.timeStepsPerHour);
	ASSERT_EQUALM("adiosData->Misc not initialized", (int)2, adiosData->Misc.LEsplit);
}

void CAdiosDataTestConstructor::TestContainedWarning()
{
	ASSERT_EQUALM("adiosData->Warning not initialized", (bool)false, adiosData->Warning.burn.wind);
	ASSERT_EQUALM("adiosData->Warning not initialized", (bool)false, adiosData->Warning.burn.wave);
	ASSERT_EQUALM("adiosData->Warning not initialized", (bool)false, adiosData->Warning.burn.emulsion);
}

void CAdiosDataTestConstructor::TestContainedUncertainty()
{
	ASSERT_EQUALM("adiosData->Uncertainty not initialized", (bool)false, adiosData->Uncertainty.wind.flag);
	ASSERT_EQUALM("adiosData->Uncertainty not initialized", (double)-999, adiosData->Uncertainty.wind.speed);

	ASSERT_EQUALM("adiosData->Uncertainty not initialized", (bool)false, adiosData->Uncertainty.Vol.flag);
	ASSERT_EQUALM("adiosData->Uncertainty not initialized", (double)-999, adiosData->Uncertainty.Vol.percent);

	ASSERT_EQUALM("adiosData->Uncertainty not initialized", (bool)false, adiosData->Uncertainty.Bullwinkle.flag);
	ASSERT_EQUALM("adiosData->Uncertainty not initialized", (double)-999, adiosData->Uncertainty.Bullwinkle.percent);
	ASSERT_EQUALM("adiosData->Uncertainty not initialized", (int)-999, adiosData->Uncertainty.Bullwinkle.hour);
}

void CAdiosDataTestConstructor::TestContainedRobert()
{
	ASSERT_EQUALM("adiosData->Robert not initialized", (double)8.3144, adiosData->Robert.R);
	ASSERT_EQUALM("adiosData->Robert not initialized", (double)1.987, adiosData->Robert.Rcal);
	ASSERT_EQUALM("adiosData->Robert not initialized", (double)-999, adiosData->Robert.Kref);
	ASSERT_EQUALM("adiosData->Robert not initialized", (int)4000, adiosData->Robert.MAXINDEX);

	for (int i = 0; i < MINDX; i++) {
		for (int j = 0; j < 6; j++) {
			ASSERT_EQUALM("adiosData->Robert not initialized", (double)-999, adiosData->Robert.Array[i][j]);
		}
		for (int j = 0; j < MAXPC; j++) {
			ASSERT_EQUALM("adiosData->Robert not initialized", (double)0, adiosData->Robert.MassFrac[j][i]);
		}
	}
}

void CAdiosDataTestConstructor::TestContainedResetFlag()
{
	ASSERT_EQUALM("adiosData->ResetFlag not initialized", (bool)true, adiosData->ResetFlag.Environ);
	ASSERT_EQUALM("adiosData->ResetFlag not initialized", (bool)true, adiosData->ResetFlag.Oil);
	ASSERT_EQUALM("adiosData->ResetFlag not initialized", (bool)true, adiosData->ResetFlag.Spill);
	ASSERT_EQUALM("adiosData->ResetFlag not initialized", (bool)true, adiosData->ResetFlag.Cleanup);
}
