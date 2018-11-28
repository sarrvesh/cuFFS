/******************************************************************************
inputparser.c
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

Correspondence concerning RMSynth_GPU should be addressed to: 
sarrvesh.ss@gmail.com

******************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include "libconfig.h"
#include "cpu_fileaccess.h"

#define DEFAULT_OUT_PREFIX "output_"

/*************************************************************
*
* Parse the input file and extract the relevant keywords
*
*************************************************************/
struct optionsList parseInput(char *parsetFileName) {
   config_t cfg;
   struct optionsList inOptions;
   const char *str;
   char *tempStr;
   
   /* Initialize configurations */
   config_init(&cfg);
   
   /* Read in the configuration file */
   if(!config_read_file(&cfg, parsetFileName)) {
      printf("ERROR: Error reading parset file %s\n\n", 
             config_error_text(&cfg));
      config_destroy(&cfg);
      exit(1);
   }
   
   /* Get the number of threads to use */
   if(! config_lookup_int(&cfg, "nThreads", &inOptions.nThreads)) {
        printf("INFO: 'nThreads' undefined in parset\n");
        printf("INFO: Continuing with nThreads=1.\n");
        inOptions.nThreads = 1;
    }
    if(inOptions.nThreads <= 0) {
       printf("Error: nThreads cannot be less than 0\n\n");
       config_destroy(&cfg);
       exit(0);
    }
    
    /* Get the number of nPhi */
    if(! config_lookup_int(&cfg, "nPhi", &inOptions.nPhi)) {
        printf("Error: 'nPhi' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    if(inOptions.nPhi <= 0) {
       printf("Error: nPhi cannot be less than 0\n\n");
       config_destroy(&cfg);
       exit(1);
    }
    
    /* Get Faraday depth */
    if(! config_lookup_float(&cfg, "phiMin", &inOptions.phiMin)) {
        printf("Error: 'phiMin' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    
    /* Get number of output phi planes */
    if(! config_lookup_float(&cfg, "dPhi", &inOptions.dPhi)) {
        printf("Error: 'dPhi' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    if(inOptions.dPhi <= 0) {
       printf("Error: dPhi cannot be less than 0\n\n");
       config_destroy(&cfg);
       exit(1);
    }
    
    /* Get prefix for output files */
    if(config_lookup_string(&cfg, "outPrefix", &str)) {
        inOptions.outPrefix = malloc(strlen(str)+1);
        strcpy(inOptions.outPrefix, str);
    }
    else {
        printf("INFO: 'outPrefix' is not defined. Defaulting to %s\n\n", 
                DEFAULT_OUT_PREFIX);
        inOptions.outPrefix = malloc(strlen(DEFAULT_OUT_PREFIX)+1);
        strcpy(inOptions.outPrefix, DEFAULT_OUT_PREFIX);
    }
    
    /* Get the name of the frequency file */
    if(config_lookup_string(&cfg, "freqFileName", &str)) {
        inOptions.freqFileName = malloc(strlen(str)+1);
        strcpy(inOptions.freqFileName, str);
    }
    else {
        printf("Error: 'freqFileName' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    
    /* Get the names of fits files */
    if(config_lookup_string(&cfg, "qCubeName", &str)) {
        inOptions.qCubeName = malloc(strlen(str)+1);
        strcpy(inOptions.qCubeName, str);
    }
    else {
        printf("Error: 'qCubeName' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    if(config_lookup_string(&cfg, "uCubeName", &str)) {
        inOptions.uCubeName = malloc(strlen(str)+1);
        strcpy(inOptions.uCubeName, str);
    }
    else {
        printf("Error: 'uCubeName' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    
    config_destroy(&cfg);
    return(inOptions);
    
}
