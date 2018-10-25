#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include "libconfig.h"
#include "synth_fileaccess.h"
#include "constants.h"

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
      inOptions.msName = malloc(strlen(str)+1);
      strcpy(inOptions.msName, str);
   }
   else {
      printf("ERROR: Unable to find MS name in the parset\n\n");
      config_destroy(&cfg);
      exit(FAILURE);
   }
   
   /* Destroy and return the structure */
   config_destroy(&cfg);
   return(inOptions);
}
