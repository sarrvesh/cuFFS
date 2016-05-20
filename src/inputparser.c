#include "libconfig.h"
#include "structures.h"
#include<stdlib.h>
#include<string.h>

#include "inputparser.h"

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
        printf("\nError: Error reading parset file. %s\n\n", 
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

    /* Check if an image mask is defined */
    if(! config_lookup_string(&cfg, "imageMask", &str)) {
        printf("\nINFO: Image mask not specified");
        inOptions.isImageMaskDefined = FALSE;
    }
    else {
        inOptions.imageMask = malloc(strlen(str));
        strcpy(inOptions.imageMask, str);
        inOptions.isImageMaskDefined = TRUE;
    }

    /* Get prefix for output files */
    if(config_lookup_string(&cfg, "outPrefix", &str)) {
        inOptions.outPrefix = malloc(strlen(str));
        strcpy(inOptions.outPrefix, str);
    }
    else {
        printf("\nINFO: 'outPrefix' is not defined. Defaulting to %s", 
                DEFAULT_OUT_PREFIX);
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
    if(! config_lookup_bool(&cfg, "plotRMSF", &inOptions.plotRMSF)) {
        printf("\nINFO: 'plotRMSF' undefined in parset");
        inOptions.plotRMSF = FALSE;
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
    if(inOptions.isImageMaskDefined == TRUE) {
        printf("\nImage mask: %s\n", inOptions.imageMask);
        printf("\n");
    }
    printf("\nphi min: %.2f", inOptions.phiMin);
    printf("\n# of phi planes: %d", inOptions.nPhi);
    printf("\ndelta phi: %.2lf", inOptions.dPhi);
    printf("\n\n");
    for(i=0; i<SCREEN_WIDTH; i++) { printf("#"); }
    printf("\n");
}
