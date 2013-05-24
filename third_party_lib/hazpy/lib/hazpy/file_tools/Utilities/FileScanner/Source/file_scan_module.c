#include "Python.h"


#include <numpy/arrayobject.h>

// NOTE: these buffer sizes were picked very arbitrarily, and have
// remarkably little impact on performance on my system.
#define BUFFERSIZE1 1024
#define BUFFERSIZE2 64


int filescan(FILE *infile, int NNums, double *array){

    double N;
    int i, j;
    int c;

    for (i=0; i<NNums; i++){
	while ( (j = fscanf(infile, "%lg", &N)) == 0 ){
	    c = fgetc(infile);
	}
	if (j == EOF) {
	    return(i);
	}
	array[i] = N;
    }
    // Go to the end of any whitespace:
    while ( isspace(c = fgetc(infile)) ){
	//printf("skipping a whitespace character: %i\n", c);
	//printf("I'm at position %i in the file\n",ftell(infile));
    }
     if (c > -1){
	 // not EOF, rewind the file one byte.
	 fseek(infile, -1, SEEK_CUR);
     }
    return(i);
}

static char doc_FileScanN[] =
"FileScanN(file, N)\n\n"
"Reads N values in the ascii file, and produces a Numeric vector of\n"
"length N full of Floats (C doubles).\n\n"
"Raises an exception if there are fewer than N  numbers in the file.\n\n"
"All text in the file that is not part of a floating point number is\n"
"skipped over.\n\n"
"After reading N numbers, the file is left before the next non-whitespace\n"
"character in the file. This will often leave the file at the start of\n"
"the next line, after scanning a line full of numbers.\n";

static PyObject * FileScanner_FileScanN(PyObject *self, PyObject *args)
{

    PyFileObject *File;
    PyArrayObject *Array;
    npy_intp length;
    
    double *Data;
    int i;

    //printf("Starting\n");

    if (!PyArg_ParseTuple(args, "O!i", &PyFile_Type, &File, &length) ) {
	return NULL;
    }  

    Data = calloc(length, sizeof(double) );

    if ((i = filescan(PyFile_AsFile( (PyObject*)File ), length, Data)) < length){
	    PyErr_SetString (PyExc_ValueError,
                     "End of File reached before all numbers found");
	    free(Data);
	    return NULL;
    }
    
    Array = (PyArrayObject *) PyArray_SimpleNew(1, &length, PyArray_DOUBLE);
  
    for (i = 0; i< length ; i++){
	*(double *)(Array->data + (i * Array->strides[0] ) ) = Data[i];
    }

    free(Data);

    return PyArray_Return(Array);
}

static char doc_FileScan[] =
"FileScan(file)\n\n"
"Reads all the values in rest of the open ascii file: file, and produces\n"
"a Numeric vector full of Floats (C doubles).\n\n"
"All text in the file that is not part of a floating point number is\n"
"skipped over.\n\n"
;


static PyObject * FileScanner_FileScan(PyObject *self, PyObject *args)
{

    FILE *infile;
    char *DataPtr;
    PyFileObject *File;
    PyArrayObject *Array;
    double *(*P_array);
    double *(*Old_P_array);
    int i,j,k;
    //int ScanCount = 0;
    npy_intp ScanCount = 0;
    int BufferSize = BUFFERSIZE2;
    int OldBufferSize = 0;
    int StartOfBuffer = 0;
    int NumBuffers = 0;

    if (!PyArg_ParseTuple(args, "O!", &PyFile_Type, &File) ) {
	return NULL;
    }  
    infile = PyFile_AsFile( (PyObject*)File );

    P_array = (double**) calloc(BufferSize, sizeof(void*) );
    while (1) {
	for (j=StartOfBuffer; j < BufferSize; j++){
	    P_array[j] = (double*) calloc(BUFFERSIZE1, sizeof(double));
	    NumBuffers++ ;
	    i = filescan(infile, BUFFERSIZE1, P_array[j]);
	    if (i) {
		ScanCount += i;
		//for (k=0; k<BUFFERSIZE1; k++){ 
		//    printf("%.14g\n", P_array[j][k]);
		//}
	    }
	    if (i == 0){
		break;
	    }
	}
	if (i == 0) {
	    break;
	}
	// Need more memory
	OldBufferSize = BufferSize;
	BufferSize += BUFFERSIZE2;
	StartOfBuffer += BUFFERSIZE2;
	Old_P_array = P_array;
	P_array = (double**) calloc(BufferSize, sizeof(void*) );
	
	for (j=0; j < OldBufferSize; j++){
	    P_array[j] = Old_P_array[j];
	}
	free(Old_P_array);
    }

    // copy all the data to a PyArray
    Array = (PyArrayObject *) PyArray_SimpleNew(1, &ScanCount, PyArray_DOUBLE);

    i = 0;
    DataPtr = Array->data;
    for (j=0; j<BufferSize; j++){
	for (k=0; k<BUFFERSIZE1; k++){
	    if (i >= ScanCount) {
		break;
	    }
	    *(double *)DataPtr = P_array[j][k];
	    DataPtr +=  Array->strides[0];
	    i++;
	}
    }

    //free all the memory
    for (j=0; j<NumBuffers; j++){
	free(P_array[j]);
    }
    free(P_array);

    return PyArray_Return(Array);
}


static PyMethodDef FileScannerMethods[] = {
  {"FileScanN", FileScanner_FileScanN, METH_VARARGS, doc_FileScanN},
  {"FileScan", FileScanner_FileScan, METH_VARARGS, doc_FileScan},
  //  {"byteswap", NumericExtras_byteswap, METH_VARARGS, doc_byteswap},
  //{"changetype", NumericExtras_changetype, METH_VARARGS, doc_changetype},
  {NULL, NULL} /* Sentinel */
};


void initfile_scanner(void){
  (void) Py_InitModule("file_scanner", FileScannerMethods);
  import_array()
}


