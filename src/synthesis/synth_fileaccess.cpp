#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdexcept>
#include<cassert>
#include<limits>
#include<libconfig.h>
#include<iostream>

#include "synth_fileaccess.h"
#include "constants.h"
#include "ms_access.h"

/*************************************************************
*
* Print useful input/output information 
*
*************************************************************/
void printUserInfo(struct optionsList inOptions, 
                   struct structHeader msHeader) {
   printf("\n*************************\n\n");
   printf("Processing %s\n", inOptions.msName);
   printf("SPW: %ld\n", msHeader.nSPW);
   printf("Channels: %ld\n", msHeader.chanPerSPW);
   printf("Channel width:%0.4f kHz\n", msHeader.chanWidth/kHz);
   std::cout << "Phase center: " << msHeader.coordStr << std::endl;
   printf("UV range: %0.2f - %0.2f m\n", msHeader.minUVWm, msHeader.maxUVWm);
   printf("Output image size: %d x %d\n", inOptions.imsize, inOptions.imsize);
   printf("Pixel size: %0.1f x %0.1f\n", inOptions.cellsize, 
                                         inOptions.cellsize);
   printf("Compute mode: ");
   if(inOptions.hardware == CPU)
      printf("CPU\n");
   else
      printf("GPU\n");
      
   printf("Invert mode: ");
   if(inOptions.invertmode == DFT)
      printf("DFT\n");
   else
      printf("FFT\n");
      
   printf("\n*************************\n\n");
}

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
   
   /* Read the config file */
   if(!config_read_file(&cfg, parsetFileName)) {
      printf("Error: Error reading parset file. %s\n\n", 
               config_error_text(&cfg));
      config_destroy(&cfg);
      exit(FAILURE);
   }
   
   /* Get the name of the measurement set */
   if(config_lookup_string(&cfg, "ms", &str)) {
      inOptions.msName = (char *)malloc(strlen(str)+1);
      strcpy(inOptions.msName, str);
   }
   else {
      config_destroy(&cfg);
      throw std::runtime_error("ERROR: Unable to find MS name in the parset\n");
   }
   
   /* Get uvmin and uvmax values */
   static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
   if(! config_lookup_float(&cfg, "uvmin", &inOptions.uvmin)) {
      inOptions.uvmin = -std::numeric_limits<float>::infinity();
   }
   if(! config_lookup_float(&cfg, "uvmax", &inOptions.uvmax)) {
      inOptions.uvmax = std::numeric_limits<float>::infinity();
   }
   
   /* Get the image and cell size */
   if(! config_lookup_int(&cfg, "imsize", &inOptions.imsize)) {
      config_destroy(&cfg);
      throw std::runtime_error("ERROR: imsize not found\n");
   }
   if(! config_lookup_float(&cfg, "cellsize", &inOptions.cellsize)) {
      if(! config_lookup_float(&cfg, "cellsize", &inOptions.cellsize)) {
         config_destroy(&cfg);
         throw std::runtime_error("ERROR: cellsize not found\n");
      }
   }
   
   /* Find out which hardware to use */
   if(config_lookup_string(&cfg, "hardware", &str)) {
      if(0 == strcasecmp(str, "GPU")) {
         inOptions.hardware = GPU;
         throw std::runtime_error("GPU support is yet to be implemented\n");
      }
      else 
         inOptions.hardware = CPU;
   }
   else { inOptions.hardware = CPU; }
   
   /* Find out which inversion mode to use */
   if(config_lookup_string(&cfg, "invertmode", &str)) {
      if(0 == strcasecmp(str, "fft")) {
         inOptions.invertmode = FFT;
         throw std::runtime_error("FFT support is yet to be implemented\n");
      }
      else
         inOptions.invertmode = DFT;
   }
   
   /* Destroy and return the structure */
   config_destroy(&cfg);
   return(inOptions);
}
