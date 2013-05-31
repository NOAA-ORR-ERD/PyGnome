/*
 * Test.cpp
 *
 *  Created on: Mar 21, 2013
 *      Author: james.makela
 *
 *  This is the top level unit testing application
 *  for the Adios core libraries.
 *  - We assume all the core library sources have been compiled
 *    into a single dynamic library named AdiosLib, but that
 *    is not a requirement.
 *  - We are compiling in Eclipse using the CUTE C++ unit
 *    testing framework.
 */
#include "cute.h"
#include "ide_listener.h"
#include "cute_runner.h"

#include "MyClassTest.h"
#include "CAdiosDataTestConstructor.h"

void thisIsATest()
{
	std::string s, s2, expected;
	s = "Hello";
	s2 = "World";
	expected = "Hello World";

	//ASSERTM("String does not match", s.append("\n" + s2).compare(expected) == 0);
	ASSERT_EQUAL(expected, s.append(" " + s2));
}

void runSuite()
{
	cute::suite s;

	s.push_back(CUTE(thisIsATest));
	s.push_back(MyClassTest());

	s.push_back(CAdiosDataTestConstructor());

	cute::ide_listener lis;
	cute::makeRunner(lis)(s, "The Suite");
}

int main()
{
    runSuite();
    return 0;
}
