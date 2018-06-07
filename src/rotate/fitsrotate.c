#include<stdio.h>
#include<time.h>
#include<string.h>

#include "constants.h"
#include "transpose.h"

int main(int argc, char *argv[]) {
   clock_t startTime, endTime;
   float execTime;
   char *inName = argv[1];
   char *outName = argv[2];
   
   /* Start the clock */
   startTime = 0;
   startTime = clock();
   
   printf("\n");
   printf("RM synthesis v%s\n", VERSION_STR);
   printf("Written by Sarrvesh S. Sridhar\n");
   
   /* Parse the command line input */
   if((strcmp(inName, "-h") == 0) || (strcmp(inName, "--help") == 0)){
      /* Print help and exit */
      printf("Usage: %s <input> <output>\n\n", argv[0]);
      return(SUCCESS);
   }
   if(argc!=NUM_INPUTS) {
      printf("ERROR: Invalid command line input. Terminating execution!\n");
      printf("Usage: %s <input> <output>\n\n", argv[0]);
      return(FAILURE);
   }
   
   if(transpose(inName, outName)) {
      return(FAILURE);
   }
   
   /* End the clock */
   endTime = 0;
   endTime = clock();
   
   /* Report execution time */
   execTime = (float)(endTime - startTime)/CLOCKS_PER_SEC;
   printf("INFO: Total execution time: %0.2f seconds\n\n", execTime);
   return(SUCCESS);
}
