/******************************************************************************
rmsynthesis.c : GPU Based implementation of RM Synthesis

Version: 0.1
Last edited: July 11, 2015
******************************************************************************/
#include"rmsynthesis.h"

int getInputs(char *parsetFileName, struct optionsList *inOptions) {
    FILE *fParset;
    
    if((fParset = fopen(parsetFileName, "r")) == NULL)
        return FAILURE;
    return SUCCESS;
}

int main(int argc, char *argv[])  {
    /* Variable declaration */
    char *parsetFileName = argv[1];
    struct optionsList inOptions;
    int status;
    
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
    if(getInputs(parsetFileName, &inOptions) == FAILURE) {
        printf("\nError: Unable to parse file %s\n\n", parsetFileName);
        exit(FAILURE);
    }
    
    printf("\n");
    return SUCCESS;
}
