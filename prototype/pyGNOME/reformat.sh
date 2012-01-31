#!/bin/bash

for FILE in `ls *.py` ; 
do 
	cp ${FILE} ${FILE}+~_!;

	sed s/"	"/"    "/g ${FILE}+~_! > ${FILE};

	rm ${FILE}+~_! 
done
