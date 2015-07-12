/******************************************************************************

A GPU Based implementation of RM Synthesis

Version: 0.1
Last edited: July 11, 2015
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
    
    config_destroy(&cfg);
    return inOptions;
}

/*************************************************************
*
* Print parsed input to screen
*
*************************************************************/
void printOptions(struct optionsList inOptions) {
    int i;
    
    printf("\n");
    for(i=0; i<SCREEN_WIDTH; i++) { printf("*"); }
    printf("\nQ Cube: %s", inOptions.qCubeName);
    printf("\nQ Cube: %s", inOptions.qCubeName);
    printf("\nOther Input Files");
    for(i=0; i<SCREEN_WIDTH; i++) { printf("*"); }
    printf("\n");
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
    
    printf("\nRM Synthesis v%s", VERSION_STR);
    printf("\nWritten by Sarrvesh S. Sridhar\n");
    
    /* Verify command line input */
    if(argc!=2) {
        printf("\nERROR: Invalid command line input. Terminating Execution!");
        printf("\nUsage: %s <parset filename>\n\n", argv[0]);
        exit(FAILURE);
    } 
    if(strcmp(parsetFileName, "-h") == 0) {
        /* Print help and exit */
        printf("\nUsage: %s <parset filename>\n\n", argv[0]);
        exit(SUCCESS);
    }
    
    /* Parse the input file */
    printf("\nINFO: Parsing input file %s", parsetFileName);
    inOptions = parseInput(parsetFileName);
    
    /* Print parset input options to screen */
    printOptions(inOptions);

    printf("\n");
    return(SUCCESS);
}
