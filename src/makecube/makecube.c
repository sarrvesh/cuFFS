#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<errno.h>
#include<string.h>
#include<stdbool.h>

#include "constants.h"
#include "fitsio.h"

/*************************************************************
*
* For a given file pointer, return the number of lines in
*   the file. Return FILE_ERROR on failure.
*
*************************************************************/
int countLines(FILE *parset) {
   int nLines = 0;
   char ch;

   while(!feof(parset)) {
      ch = fgetc(parset);
      if(ferror(parset) != SUCCESS) {
         nLines = FILE_ERROR;
         break;
      }
      if(ch == '\n') {
         nLines += 1;
      }
   }

   return(nLines);
}

/*************************************************************
*
* Decode FITS status code and return FAILURE on error
*
*************************************************************/
int checkFitsError(int status) {
   if(status) {
      fits_report_error(stdout, status);
      printf("\n");
      return(FAILURE);
   }
   return(SUCCESS);
}

/*************************************************************
*
* Main function
*
*************************************************************/
int main(int argc, char *argv[]) {
   clock_t startTime, endTime;
   float execTime;
   char *inName = argv[1];
   char *outName = argv[2];
   FILE *parset;
   char *fitsFileName = NULL;
   char *tempFileName = NULL;
   size_t lenFileName;
   int nFitsFiles;
   fitsfile *in, *out;
   int fitsStatus=0;
   long inAxisLen[CUBE_DIM], axisLen[CUBE_DIM];
   long outAxisLen[CUBE_DIM_OUT];
   long fPixel[CUBE_DIM], oPixel[CUBE_DIM_OUT];
   long nElements;
   char fitsComment[FLEN_COMMENT];
   bool isFirstImage;
   float *imageData;
   int thisChannel = 0;
   int nHeaderKeys = 0;
   char hdrKeyName[FLEN_COMMENT];
   int i, leftover;

   /* Start the clock */
   startTime = 0;
   startTime = clock();

   printf("\n");
   printf("RM synthesis v%s\n", VERSION_STR);
   printf("Written by Sarrvesh S. Sridhar\n");

   /* Parse the command line input */
   if((strcmp(inName, "-h") == 0) || (strcmp(inName, "--help") == 0)){
      /* Print help and exit */
      printf("Usage: %s <parset file> <output>\n\n", argv[0]);
      return(FAILURE);
   }
   if(argc!=NUM_INPUTS) {
      printf("ERROR: Invalid command line input. Terminating execution!\n");
      printf("Usage: %s <parset file> <output>\n\n", argv[0]);
      return(FAILURE);
   }

   /* Open the parset file. */
   parset = fopen(inName, "r");
   if(parset == NULL){
      printf("\nError: Unable to read the input parset.\n");
      printf("Error: %s\n\n", strerror(errno));
      return(FAILURE);
   }
   /* Estimate the number of input fits files */
   nFitsFiles = countLines(parset);
   printf("INFO: Found %d files in input file %s\n", nFitsFiles, inName);
   /* Loop over each specified file */
   rewind(parset);
   isFirstImage = true;
   while(getline(&tempFileName, &lenFileName, parset) > 0) {
      fitsFileName = strtok(tempFileName, "\n");
      printf("\nProcessing file %s", fitsFileName);
      /* Open this fits file */
      fits_open_file(&in, fitsFileName, READONLY, &fitsStatus);
      if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
      /* Read the size of the input cube */
      inAxisLen[0] = inAxisLen[1] = inAxisLen[2] = inAxisLen[3] = 0;
      fits_read_key(in, TINT, "NAXIS1", &inAxisLen[0], fitsComment, &fitsStatus);
      fits_read_key(in, TINT, "NAXIS2", &inAxisLen[1], fitsComment, &fitsStatus);
      fits_read_key(in, TINT, "NAXIS3", &inAxisLen[2], fitsComment, &fitsStatus);
      fits_read_key(in, TINT, "NAXIS4", &inAxisLen[3], fitsComment, &fitsStatus);
      if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
      /* Check if all input images have the same size */
      if(isFirstImage == true) {
         axisLen[0] = inAxisLen[0];
         axisLen[1] = inAxisLen[1];
         axisLen[2] = inAxisLen[2];
         axisLen[3] = inAxisLen[3];
         nElements = axisLen[0] * axisLen[1] * axisLen[2] * axisLen[3];
         isFirstImage = false;

         /* While we are at it, create the output file */
         outAxisLen[0] = inAxisLen[0];
         outAxisLen[1] = inAxisLen[1];
         outAxisLen[2] = nFitsFiles;
         fits_create_file(&out, outName, &fitsStatus);
         fits_create_img(out, FLOAT_IMG, CUBE_DIM_OUT, outAxisLen, &fitsStatus);
         oPixel[0] = oPixel[1] = 1;
         /* Copy FITS keywords from input image to output cube */
         fits_get_hdrspace(in, &nHeaderKeys, NULL, &fitsStatus);
         for(i=0; i<nHeaderKeys; i++) {
            fits_read_record(in, i+1, hdrKeyName, &fitsStatus);
            /* Ignore this record if it contains the keys SIMPLE, BITPIX, NAXIS, 
               NAXIS1, NAXIS2, NAXIS3, and EXTEND */
            if(strstr(hdrKeyName, "SIMPLE")!=NULL ||
               strstr(hdrKeyName, "BITPIX")!=NULL ||
               strstr(hdrKeyName, "NAXIS")!=NULL ||
               strstr(hdrKeyName, "NAXIS1")!=NULL ||
               strstr(hdrKeyName, "NAXIS2")!=NULL ||
               strstr(hdrKeyName, "NAXIS3")!=NULL ||
               strstr(hdrKeyName, "EXTEND")!=NULL) {}
            else {
               fits_write_record(out, hdrKeyName, &fitsStatus);
            }
         }
         if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
      }
      else {
         if((axisLen[0] != inAxisLen[0]) || 
            (axisLen[1] != inAxisLen[1]) || 
            (axisLen[2] != inAxisLen[2]) || 
            (axisLen[3] != inAxisLen[3])) {
            printf("\nError: Input cube %s has different dimensions\n", 
                    fitsFileName);
         }
      }
      fPixel[0] = fPixel[1] = fPixel[2] = fPixel[3] = 1;
      imageData = calloc(nElements, sizeof(float));
      fits_read_pix(in, TFLOAT, fPixel, nElements, 
                    NULL, imageData, NULL, &fitsStatus);
      fits_close_file(in, &fitsStatus);
      /* Write the read image as a channel to the out cube */
      oPixel[2] = thisChannel + 1;
      fits_write_pix(out, TFLOAT, oPixel, nElements, imageData, &fitsStatus);
      if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
      thisChannel += 1;
   }
   free(fitsFileName);
   fclose(parset);
   fits_close_file(out, &fitsStatus);
   if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }

   /* End the clock */
   endTime = 0;
   endTime = clock();

   /* Report execution time */
   execTime = (float)(endTime - startTime)/CLOCKS_PER_SEC;
   printf("\nINFO: Total execution time: %0.2f seconds\n\n", execTime);
   return(SUCCESS);
}
