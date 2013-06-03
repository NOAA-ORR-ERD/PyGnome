/*
 * CAdiosDataTest.h
 *
 *  Created on: Mar 21, 2013
 *      Author: james.makela
 */

#ifndef CADIOSDATATEST_H_
#define CADIOSDATATEST_H_

#include <tr1/memory>
#include "AdiosComp.h"

class CAdiosDataTestConstructor {
private:
	int refcount;
	std::tr1::shared_ptr<CAdiosData> adiosData;

	// whatever tests you need
	void setup();
	void teardown();

	void test();
	void TestLocalDataMembers();
	void TestContainedSource();

	void TestContainedLeak();
	void TestContainedLeakInstantData();
	void TestContainedLeakContinuousData();
	void TestContainedLeakTankData();

	void TestContainedTankDisplay();

	void TestContainedEnvironment();
	void TestContainedEnvironmentWindData();
	void TestContainedEnvironmentWaveData();
	void TestContainedEnvironmentWaterData();
	void TestContainedEnvironmentAirData();

	void TestContainedFay();
	void TestContainedLE();

	void TestContainedCleanup();
	void TestContainedCleanupDispersantData();
	void TestContainedCleanupSkimmerData();
	void TestContainedCleanupBurnData();
	void TestContainedCleanupSmokeData();

	void TestContainedRelease();
	void TestContainedInitialOil();
	void TestContainedSlick();
	void TestContainedSink();
	void TestContainedMisc();
	void TestContainedWarning();
	void TestContainedUncertainty();
	void TestContainedRobert();
	void TestContainedResetFlag();

public:
	CAdiosDataTestConstructor();
	CAdiosDataTestConstructor(CAdiosDataTestConstructor const& other);
	virtual ~CAdiosDataTestConstructor();

	// must define void operator() with no arguments
	void operator() ();

};

#endif /* CADIOSDATATEST_H_ */
