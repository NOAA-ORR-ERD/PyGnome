#!/bin/sh

# script to build the arc.c example from GD

GD_DIR=../static_libs/

gcc -arch i386 -o arc  -lz -I$GD_DIR/include/ $GD_DIR/lib/libgd.a $GD_DIR/lib/libpng.a arc.c

