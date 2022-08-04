Recipe for building a conda package locally.

This is a bit fragile -- ideally we would generate the build configuration
automatically, so that it can be kept in sync with new dependencies, etc.

Building the conda package should be as easy as::

  > conda install conda-build # make sure you have that!

  > cd conda_recipe  # this dir

  > conda build .


The trick is then how to install it. If you point conda install at teh packge directly, it will install, but not use the dependencies, which is not so useful. I had success on my machine doing::

  > conda install -c /Users/chris.barker/miniconda3/envs/gnome/conda-bld/osx-64/ py_gnome

But then we'll need to figure out that path on the CI or wherever.


NOTES:

The python and numpy versions are specified in the `conda_build_config.yaml` file.

This is because PyGNOME uses the C APIs of both python and numpy, so the pacakge will nly work with the versions it was built against.

This could be used to build multiple versions, but only one for now.

