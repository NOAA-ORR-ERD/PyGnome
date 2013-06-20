#!/bin/sh

# script to build the arc.c example from GD

GD_DIR=../static_libs/

gcc -arch i386 -o arc  -I$GD_DIR/include/ $GD_DIR/lib/libgd.a arc.c

