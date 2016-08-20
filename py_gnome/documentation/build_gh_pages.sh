#!/bin/sh

# simple script to build and push to gh-pages
# designed to be run from master

# make the docs
make html

# make sure other copy of repo is in the right branch
pushd ../../../PyGnome.gh-pages/
git checkout gh-pages
popd

# copy to other repo (on the gh-pages branch)
cp -R _build/html/ ../../../PyGnome.gh-pages/

pushd ../../../PyGnome.gh-pages/
git add * # in case there are new files added
git commit -a -m "updating documentation"
# make sure we're in sync with other changes in repo
git pull -s ours
git push

