#include "cute.h"
#include "ide_listener.h"
#include "cute_runner.h"

#include "MyClassTest.h"

MyClassTest::MyClassTest()
{
}

void MyClassTest::operator()()
{
	// it says in the docs that we need to define this, but to what???
	// I would love some better documentation for CUTE unit testing.
	setup();
	test();
	teardown();
}

void MyClassTest::setup()
{
	// here is where you would setup any input conditions
}

void MyClassTest::teardown()
{
	// here is where you would perform any post-test cleanup
}

void MyClassTest::test()
{
	// here is the actual test
	double cx = 0.1, ax = 0.3;
	ASSERT_EQUAL_DELTAM("custom message for failing test", cx, ax, 0.2);
}
