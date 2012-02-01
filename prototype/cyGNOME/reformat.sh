#!/bin/bash

for FILE in `ls *.py && ls *.pyx && ls *.pxi && ls *.pxd` ; 
do 
	cp ${FILE} ${FILE}+~_!;

	sed s/"	"/"    "/g ${FILE}+~_! > ${FILE};

	rm ${FILE}+~_! 
done
