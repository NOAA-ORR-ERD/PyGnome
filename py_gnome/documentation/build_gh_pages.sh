#!/bin/sh

# script to build and push to gh-pages

# this version uses a separate copy of the repo in a parallel dir
# to keep the gh-pages branch nicely separate from the main branch.
# this makes it easier to have the index.html in the root dir
#  -- to be nicely served by gh-pages

# designed to be run from master -- or the latest release tag.

# note that you should have the lastest version of gnome build and importable -- run the tests first!
#      Maybe we should run the tests as part of this script??

# It's also a good idea to run "make html" first, to make sure the docs build cleanly

# You also need another copy of the repo next to this one called: ../../../PyGnome.gh-pages/

# Ideally, we'd have a script that setup a conda environment, build gnome, built the docs, pushed them to gh
#
# even better, it would be set with a hook to happen whenever a new release is pushed to gitHub..


GHPAGESDIR=../../../PyGnome.gh-pages/
GHPAGES_URL=http://noaa-orr-erd.github.io/PyGnome/

# make sure gh-pages dir is there -- exit if not
if [ ! -d $GHPAGESDIR ]; then
    echo "To build the gitHub pages, you must first create a parallel repo: $GHPAGESDIR"
    exit
fi

if [ ! -d $GHPAGESDIR/.git ]; then
    echo "To build the gitHub pages, you must first create a parallel repo: $GHPAGESDIR"
    echo "It must be a git repo -- do a new git clone"
    exit
fi

# **Maybe this should be there? -- but I'm wary of doing it too automatically
# make sure that the main branch is pushed, so that pages are in sync with master
# git commit -a -m "updating before rendering docs"
# git push

# make sure the gh pages repo is there and in the right branch
pushd $GHPAGESDIR
git checkout gh-pages
popd

# make the docs
make html
# copy to other repo (on the gh-pages branch)
cp -vfR _build/html/* $GHPAGESDIR

pushd $GHPAGESDIR
git add .  # in case there are new files added
# NOTE: This does not remove any files!
git commit -a -m "updating rendered materials"
git branch -u origin/gh-pages  # make sure we're tracking origin
git pull -s ours --no-edit  # gotta pull before push.. yet maintain local updates
git push
popd

echo "Now verify the render on github.io at the following link:"
echo $GHPAGES_URL


