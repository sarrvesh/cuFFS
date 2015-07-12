#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include"version.h"
#include"libconfig.h"
#include"fitsio.h"

#define SUCCESS 0
#define FAILURE 1

#define DEFAULT_OUT_PREFIX "output_"
#define SCREEN_WIDTH       24

/* Structure to store the input options */
struct optionsList{
    char *qCubeName;
    char *uCubeName;
    char *freqFileName;
    char *outPrefix;
};

/* Function declarations */
void printOptions(struct optionsList inOptions);
