#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include"version.h"

#define SIZE_FILE_NAME 256
#define SUCCESS 0
#define FAILURE 1
#define NINPUTS 2

/* Structure to store the input options */
struct optionsList{
    char qFitsFile[SIZE_FILE_NAME];
    char uFitsFile[SIZE_FILE_NAME];
    char outPrefix[SIZE_FILE_NAME];
};

/* Function definitions */
int getInputs(char *parsetFileName, struct optionsList *inOptions);
