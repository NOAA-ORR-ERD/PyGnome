#ifndef __my_class_test__
#define __my_class_test__


class MyClassTest
{
public:
	MyClassTest();

	// must define void operator() with no arguments
	void operator() ();

private:
	// whatever tests you need
	void setup();
	void test();
	void teardown();
};

#endif __my_class_test__
