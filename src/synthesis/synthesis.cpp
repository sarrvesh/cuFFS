#include<stdio.h>
#include<time.h>
#include<string.h>
#include<stdlib.h>

#include "constants.h"
#include "synth_fileaccess.h"
#include "ms_access.h"

/*************************************************************
*
* Main code
*
*************************************************************/
int main(int argc, char *argv[]) {
   char *parsetFileName = argv[1];
   clock_t startTime, endTime;
   struct optionsList inOptions;
   struct structHeader msHeader;
   unsigned int hours, mins, secs;
   unsigned int elapsedTime;
   
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
   
   /* Get header information from the measurement set */
   if(FAILURE == getMsHeader(inOptions, &msHeader)) { exit(FAILURE); }

   /* Find the minimum and maximum baselines */
   if(FAILURE == getUvRange(inOptions, &msHeader)) { exit(FAILURE); }   
   
   /* Print some useful information */ 
   printUserInfo(inOptions, msHeader);
   
   /* Report execution time */
   endTime = clock();
   elapsedTime = (unsigned int)(endTime - startTime)/CLOCKS_PER_SEC;
   hours = (unsigned int)elapsedTime/SEC_PER_HOUR;
   mins  = (unsigned int)(elapsedTime%SEC_PER_HOUR)/SEC_PER_MIN;
   secs  = (unsigned int)(elapsedTime%SEC_PER_HOUR)%SEC_PER_MIN;
   printf("INFO: Total execution time: %d:%d:%d\n", hours, mins, secs);
   
   printf("\n");
   return(SUCCESS);
}
