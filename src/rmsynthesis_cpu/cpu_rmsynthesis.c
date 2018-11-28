/******************************************************************************
cpu_rmsynthesis.c: A CPU based implementation of RM Synthesis.
Copyright (C) 2016  {fullname}

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Correspondence concerning cuFFS should be addressed to: 
sarrvesh.ss@gmail.com

******************************************************************************/

#include<stdio.h>
#include<time.h>
#include<string.h>

#include "cpu_version.h"
#include "cpu_fileaccess.h"

#define NUM_INPUTS 2
#define SEC_PER_HOUR 3600
#define SEC_PER_MIN 60

/*************************************************************
*
* Main code
*
*************************************************************/
int main(int argc, char *argv[]) {
   char *parsetFileName = argv[1];
   struct optionsList inOptions;
   clock_t startTime, endTime;
   unsigned int cpuTime, hours, mins, secs;
   
   /* Start the clock */
   startTime = clock();
   
   printf("\n");
   printf("RM synthesis v%s (CPU version)\n", VERSION_STR);
   printf("Written by Sarrvesh S. Sridhar\n");

   /* Verify command line input */
   if(argc!=NUM_INPUTS) {
      printf("ERROR: Invalid command line input. Terminating Execution!\n");
      printf("Usage: %s <parset filename>\n\n", argv[0]);
      return 1;
   }
   if((strcmp(parsetFileName, "-h") == 0)) {
      /* Print help and exit */
      printf("Usage: %s <parset filename>\n\n", argv[0]);
      return 0;
   }
   
   /* Parse the input file */
   printf("INFO: Parsing input file %s\n", parsetFileName);
   inOptions = parseInput(parsetFileName);
   
   /* Check input files */
   
   /* Gather information about input fits header and setup output images */
   
   /* Print some useful information */
   
   /* Read in the frequency list */
   
   /* Find median lambda20 */
   
   /* Generate RMSF and write to disk */
   
   /* Start RM synthesis */
   
   /* Free up all allocated memory */
   
   /* Close all open files */
   
   /* Estimate the execution time */  
   endTime = clock();
   cpuTime = (unsigned int)(endTime - startTime)/CLOCKS_PER_SEC;
   hours = (unsigned int)cpuTime/SEC_PER_HOUR;
   mins  = (unsigned int)(cpuTime%SEC_PER_HOUR)/SEC_PER_MIN;
   secs  = (unsigned int)(cpuTime%SEC_PER_HOUR)%SEC_PER_MIN;

   printf("\n");
   return 0;
}
