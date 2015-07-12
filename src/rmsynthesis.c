/******************************************************************************

A GPU Based implementation of RM Synthesis

Version: 0.1
Last edited: July 11, 2015

Version history:
================
v0.1    Assume FITS cubes have at least 3 frames with frequency being the 
         3rd axis. Also, implicitly assumes each freq channel has equal weight.
         

******************************************************************************/
#include"rmsynthesis.h"

/*************************************************************
*
* Parse the input file and extract the relevant keywords
*
*************************************************************/
struct optionsList parseInput(char *parsetFileName) {
    config_t cfg;
    struct optionsList inOptions;
    const char *str;
    
    /* Initialize configuration */
    config_init(&cfg);
    
    /* Read in the configuration file */
    if(!config_read_file(&cfg, parsetFileName)) {
        printf("\nError: Error reading parset file. Exiting with message: %s\n\n", 
               config_error_text(&cfg));
        config_destroy(&cfg);
        exit(FAILURE);
    }
    
    /* Get the names of fits files */
    if(config_lookup_string(&cfg, "qCubeName", &str)) {
        inOptions.qCubeName = malloc(strlen(str));
        strcpy(inOptions.qCubeName, str);
    }
    else {
        printf("\nError: 'qCubeName' undefined in parset");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    if(config_lookup_string(&cfg, "uCubeName", &str)) {
        inOptions.uCubeName = malloc(strlen(str));
        strcpy(inOptions.uCubeName, str);
    }
    else {
        printf("\nError: 'uCubeName' undefined in parset");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    
    /* Get the name of the frequency file */
    if(config_lookup_string(&cfg, "freqFileName", &str)) {
        inOptions.freqFileName = malloc(strlen(str));
        strcpy(inOptions.freqFileName, str);
    }
    else {
        printf("\nError: 'freqFileName' undefined in parset");
        config_destroy(&cfg);
        exit(FAILURE);
    }

    /* Get prefix for output files */
    if(config_lookup_string(&cfg, "outPrefix", &str)) {
        inOptions.outPrefix = malloc(strlen(str));
        strcpy(inOptions.outPrefix, str);
    }
    else {
        printf("\nINFO: 'outPrefix' is not defined. Defaulting to %s", DEFAULT_OUT_PREFIX);
        inOptions.outPrefix = malloc(strlen(DEFAULT_OUT_PREFIX));
        strcpy(inOptions.outPrefix, DEFAULT_OUT_PREFIX);
    }
    
    /* Get Faraday depth */
    if(! config_lookup_float(&cfg, "phiMin", &inOptions.phiMin)) {
        printf("\nError: 'phiMin' undefined in parset");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    if(! config_lookup_float(&cfg, "dPhi", &inOptions.dPhi)) {
        printf("\nError: 'dPhi' undefined in parset");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    if(! config_lookup_int(&cfg, "nPhi", &inOptions.nPhi)) {
        printf("\nError: 'nPhi' undefined in parset");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    
    config_destroy(&cfg);
    return(inOptions);
}

/*************************************************************
*
* Print parsed input to screen
*
*************************************************************/
void printOptions(struct optionsList inOptions) {
    int i;
    
    printf("\n");
    for(i=0; i<SCREEN_WIDTH; i++) { printf("#"); }
    printf("\n");
    printf("\nQ Cube: %s", inOptions.qCubeName);
    printf("\nU Cube: %s", inOptions.uCubeName);
    printf("\n");
    printf("\nphi min: %.2lf", inOptions.phiMin);
    printf("\n# of phi planes: %d", inOptions.nPhi);
    printf("\ndelta phi: %.2lf", inOptions.dPhi);
    printf("\n\n");
    for(i=0; i<SCREEN_WIDTH; i++) { printf("#"); }
    printf("\n");
}

/*************************************************************
*
* Read header information from the fits files
*
*************************************************************/
int getFitsHeader(struct optionsList *inOptions, struct parList *params) {
    int fitsStatus = SUCCESS;
    char fitsComment[FLEN_COMMENT];
    
    /* Get the image dimensions from the Q cube */
    fits_read_key(params->qFile, TINT, "NAXIS", &params->qAxisNum, fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS1", &params->qAxisLen1, fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS2", &params->qAxisLen2, fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS3", &params->qAxisLen3, fitsComment, &fitsStatus);
    /* Get the image dimensions from the Q cube */
    fits_read_key(params->uFile, TINT, "NAXIS", &params->uAxisNum, fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS1", &params->uAxisLen1, fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS2", &params->uAxisLen2, fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS3", &params->uAxisLen3, fitsComment, &fitsStatus);
    
    return(fitsStatus);
}

/*************************************************************
*
* Read the list of frequencies from the input freq file
*
*************************************************************/
int getFreqList(struct optionsList *inOptions, struct parList *params) {
    int i;
    double tempDouble;
    
    params->freqList = calloc(params->qAxisLen3, sizeof(params->freqList));
    if(params->freqList == NULL) {
        printf("Error: Mem alloc failed while reading in frequency list");
        return(FAILURE);
    }
    for(i=0; i<params->qAxisLen3; i++) {
        fscanf(params->freq, "%lf", &params->freqList[i]);
        if(feof(params->freq)) {
            printf("\nError: Less frequency values present than fits frames\n\n");
            return(FAILURE);
        }
    }
    fscanf(params->freq, "%lf", &tempDouble);
    if(! feof(params->freq)) {
        printf("\nError: More frequency values present than fits frames\n\n");
        return(FAILURE);
    }
    
    /* Compute \lambda^2 from the list of generated frequencies */
    params->lambda2  = calloc(params->qAxisLen3, sizeof(params->lambda2));
    if(params->lambda2 == NULL) {
        printf("Error: Mem alloc failed while reading in frequency list");
        return(FAILURE);
    }
    params->lambda20 = 0.0;
    for(i=0; i<params->qAxisLen3; i++)
        params->lambda2[i] = (LIGHTSPEED / params->freqList[i]) * 
                             (LIGHTSPEED / params->freqList[i]);
    
    return(SUCCESS);
}

/*************************************************************
*
* Read the list of frequencies from the input freq file
*
*************************************************************/
int generateRMSF(struct optionsList *inOptions, struct parList *params) {
    int i, j;
    double K;
    
    params->rmsf     = calloc(inOptions->nPhi, sizeof(params->rmsf));
    params->rmsfReal = calloc(inOptions->nPhi, sizeof(params->rmsfReal));
    params->rmsfImag = calloc(inOptions->nPhi, sizeof(params->rmsfImag));
    params->phiAxis  = calloc(inOptions->nPhi, sizeof(params->phiAxis));
    
    if(params->rmsf     == NULL || params->rmsfReal == NULL ||
       params->rmsfImag == NULL || params->phiAxis  == NULL)
        return(FAILURE);
    
    /* Get the normalization factor K */
    K = 1.0 / params->qAxisLen3;
    
    /* First generate the phi axis */
    for(i=0; i<inOptions->nPhi; i++) {
        params->phiAxis[i] = inOptions->phiMin + i * inOptions->dPhi;
        
        /* For each phi value, compute the corresponding RMSF */
        for(j=0; j<params->qAxisLen3; j++) {
            params->rmsfReal[i] += cos(2 * params->phiAxis[i] *
                                   (params->lambda2[j] - params->lambda20 ));
            params->rmsfImag[i] -= sin(2 * params->phiAxis[i] *
                                   (params->lambda2[j] - params->lambda20 ));
        }
        // Normalize with K
        params->rmsfReal[i] *= K;
        params->rmsfImag[i] *= K;
        params->rmsf[i] = sqrt( params->rmsfReal[i] * params->rmsfReal[i] +
                                params->rmsfImag[i] * params->rmsfImag[i] );
    }
    return(SUCCESS);
}

/*************************************************************
*
* Comparison function used by quick sort.
*
*************************************************************/
int compFunc(const void * a, const void * b) {
   return ( *(double*)a - *(double*)b );
}

/*************************************************************
*
* Find the median \lambda^2_0
*
*************************************************************/
void getMedianLambda20(struct parList *params) {
    double *tempArray;
    int i;
    
    tempArray = calloc(params->qAxisLen3, sizeof(tempArray));
    for(i=0; i<params->qAxisLen3; i++)
        tempArray[i] = params->lambda2[i];
    
    /* Sort the list of lambda2 freq */
    qsort(tempArray, params->qAxisLen3, sizeof(tempArray), compFunc);
    
    /* Find the median value of the sorted list */
    params->lambda20 = tempArray[params->qAxisLen3/2];
}

/*************************************************************
*
* Write RMSF to disk
*
*************************************************************/
int writeRMSF(struct optionsList inOptions, struct parList params) {
    FILE *rmsf;
    char filename[FILENAME_LEN];
    int i;
    
    /* Open a text file */
    sprintf(filename, "%srmsf.txt", inOptions.outPrefix);
    printf("\nINFO: Writing RMSF to %s", filename);
    rmsf = fopen(filename, FILE_READWRITE);
    if(rmsf == NULL)
        return(FAILURE);
    
    for(i=0; i<inOptions.nPhi; i++)
        fprintf(rmsf, "%lf\t%lf\t%lf\t%lf\n", params.phiAxis[i], params.rmsfReal[i],
                params.rmsfImag[i], params.rmsf[i]);
    
    fclose(rmsf);
    return(SUCCESS);
}

/*************************************************************
*
* Read in the stokes-Q and -U images 
*
*************************************************************/
int getImageData(struct optionsList *inOptions, struct parList *params) {
    long *fPixel;
    int i;
    LONGLONG nElements;
    int status = SUCCESS;
    
    /* Set the starting pixel to read from the FITS file */
    fPixel = calloc(params->qAxisNum, sizeof(fPixel));
    for(i=1; i<=params->qAxisNum; i++)
        fPixel[i-1] = 1;
    
    /* Allocate memory to store the Q and U image arrays */
    nElements = params->qAxisLen1 * params->qAxisLen2 * params->qAxisLen3;
    params->qImageArray = calloc(nElements, sizeof(params->qImageArray));
    params->uImageArray = calloc(nElements, sizeof(params->uImageArray));
    
    /* Read pixel values */
    fits_read_pix(params->qFile, TDOUBLE, fPixel, nElements, NULL, params->qImageArray, NULL, &status);
    fits_read_pix(params->uFile, TDOUBLE, fPixel, nElements, NULL, params->uImageArray, NULL, &status);
    if(status) {
        printf("\nError: Unable to read fits files. Terminating with message\n");
        fits_report_error(stdout, status);
        return(FAILURE);
    }
    return(SUCCESS);
}

/*************************************************************
*
* Main code
*
*************************************************************/
int main(int argc, char *argv[]) {
    /* Variable declaration */
    char *parsetFileName = argv[1];
    struct optionsList inOptions;
    struct parList params;
    int fitsStatus;
    int status;
    
    printf("\nRM Synthesis v%s", VERSION_STR);
    printf("\nWritten by Sarrvesh S. Sridhar\n");
    
    /* Verify command line input */
    if(argc!=2) {
        printf("\nERROR: Invalid command line input. Terminating Execution!");
        printf("\nUsage: %s <parset filename>\n\n", argv[0]);
        return(FAILURE);
    } 
    if(strcmp(parsetFileName, "-h") == 0) {
        /* Print help and exit */
        printf("\nUsage: %s <parset filename>\n\n", argv[0]);
        return(SUCCESS);
    }
    
    /* Parse the input file */
    printf("\nINFO: Parsing input file %s", parsetFileName);
    inOptions = parseInput(parsetFileName);
    
    /* Print parset input options to screen */
    printOptions(inOptions);
    
    /* Open the input files */
    printf("\nINFO: Accessing the input files");
    printf("\nWARN: Assuming the 3rd axis in the fits files is the frequency axis");
    fitsStatus = SUCCESS;
    fits_open_file(&params.qFile, inOptions.qCubeName, READONLY, &fitsStatus);
    fits_open_file(&params.uFile, inOptions.uCubeName, READONLY, &fitsStatus);
    if(fitsStatus != SUCCESS) {
        printf("\nError: Unable to read the fits files. ");
        printf("\n\nError msg:");
        fits_report_error(stdout, fitsStatus);
    }
    params.freq = fopen(inOptions.freqFileName, FILE_READONLY);
    if(params.freq == NULL) {
        printf("\nError: Unable to open the frequency file\n\n");
        return(FAILURE);
    }
    
    /* Gather information from fits header */
    status = getFitsHeader(&inOptions, &params);
    if(status) {
        printf("\n\nQuiting with error msg:\n");
        fits_report_error(stdout, status);
        return(FAILURE);
    }
    
    /* Read frequency list */
    if(getFreqList(&inOptions, &params))
        return(FAILURE);
    
    /* Find median lambda20 */
    getMedianLambda20(&params);
    
    /* Generate RMSF */
    printf("\nINFO: Computing RMSF");
    if(generateRMSF(&inOptions, &params)) {
        printf("\nError: Mem alloc failed while generating RMSF");
        return(FAILURE);
    }
    
    /* Write RMSF to disk */
    if(writeRMSF(inOptions, params)) {
        printf("\nError: Unable to write RMSF to disk\n\n");
        return(FAILURE);
    }
    
    /* Read image planes from the Q and U cubes */
    printf("\nINFO: Reading in FITS images");
    if(getImageData(&inOptions, &params))
        return(FAILURE);

    printf("\n\n");
    return(SUCCESS);
}
