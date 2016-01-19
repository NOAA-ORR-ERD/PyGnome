#!/bin/sh

# simple script to build and push to gh-pages
# designed to be run from master

# make the docs
make html

# copy to other repo (on the gh-pages branch)
cp -R build/html/ ../../IntroToPython.gh-pages

cd ../../IntroToPython.gh-pages
git checkout gh-pages
git add * # in case there are new files added
git commit -a -m "updating presentation materials"
git pull -s ours
git push
