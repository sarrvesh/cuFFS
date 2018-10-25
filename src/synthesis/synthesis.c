#include<stdio.h>
#include<time.h>
#include<string.h>

#include "constants.h"
#include "synth_fileaccess.h"

/*************************************************************
*
* Main code
*
*************************************************************/
int main(int argc, char *argv[]) {
   char *parsetFileName = argv[1];
   clock_t startTime;
   struct optionsList inOptions;
   
   /* Start the clock */
   startTime = clock();
   
   printf("\n");
   printf("Faraday synthesis v%s\n", VERSION_STR);
   printf("Written by Sarrvesh S. Sridhar\n\n");
   
   /* Verify the command line input */
   if(argc != NUM_INPUTS){
      printf("ERROR: Invalid command line input. Terminating execution!\n");
      printf("Usage: %s <parset filename>\n\n", argv[0]);
      return(FAILURE);
   }
   if(strcmp(parsetFileName, "-h") == 0) {
      /* Print help and exit */
      printf("Usage: %s <parset filename>\n\n", argv[0]);
      return(SUCCESS);
   }
   
   /* Parse the input file */
   printf("INFO: Parsing parset file %s\n", parsetFileName);
   inOptions = parseInput(parsetFileName);
   printf("INFO: Specified MS is %s\n", inOptions.msName);
   
   printf("\n");
   return(SUCCESS);
}
