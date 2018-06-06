#include<stdio.h>
#include<time.h>
#include<string.h>

#include "constants.h"
#include "transpose.h"

int main(int argc, char *argv[]) {
   clock_t startTime, endTime, execTime;
   char *mode = argv[1];
   char *inName = argv[2];
   char *outName = argv[3];
   
   /* Start the clock */
   startTime = 0;
   startTime = clock();
   
   printf("\n");
   printf("RM synthesis v%s\n", VERSION_STR);
   printf("Written by Sarrvesh S. Sridhar\n");
   
   /* Parse the command line input */
   if((strcmp(mode, "-h") == 0) || (strcmp(mode, "--help") == 0)){
      /* Print help and exit */
      printf("Usage: %s <mode: rotate/derotate> <input> <output>\n\n", argv[0]);
      return(SUCCESS);
   }
   if(argc!=NUM_INPUTS) {
      printf("ERROR: Invalid command line input. Terminating execution!\n");
      printf("Usage: %s <mode: rotate/derotate> <input> <output>\n\n", argv[0]);
      return(FAILURE);
   }
   if((strcmp(mode, ROTATE) != 0) && (strcmp(mode, DEROTATE) != 0)) {
      printf("ERROR: Invalid mode specified. Terminating execution!\n");
      printf("Valid modes are 'rotate' and 'derotate'\n\n");
      return(FAILURE);
   }
   
   if(transpose(mode, inName, outName)) {
      return(FAILURE);
   }
   
   /* End the clock */
   endTime = 0;
   endTime = clock();
   
   /* Report execution time */
   execTime = (unsigned int)(endTime - startTime)/CLOCKS_PER_SEC;
   printf("INFO: Total execution time: %ld seconds\n\n", execTime);
   return(SUCCESS);
}
