unit_tests/ directory is getting crowded.

Break out integration tests from unit_tests/ to integration_tests/ directory.

Also created tests/sample_data/ 
* contains sample data files used by multiple types of tests,
  like integration_tests/ and unit_tests/

In addition, each test directory has its own sample_data/ directory.
* unit_tests/sample_data/ contains files that are used only by tests in unit_tests directory
* integration_tests/sample_data/ contains files that are used only by tests in unit_tests directory
