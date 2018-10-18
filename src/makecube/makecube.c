#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<errno.h>
#include<string.h>
#include<glob.h>

#include "constants.h"
#include "fitsio.h"

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
   const char *inPattern = argv[1];
   const char *outName = argv[2];
   const char *freqFile = argv[3];
   FILE *freqFileFd;
   char fitsFileName[FILE_NAME_LEN];
   int nFitsFiles;
   fitsfile *in, *out;
   int fitsStatus=0;
   long inAxisLen[CUBE_DIM], axisLen[CUBE_DIM];
   long outAxisLen[CUBE_DIM_OUT];
   long fPixel[CUBE_DIM], oPixel[CUBE_DIM_OUT];
   long nElements;
   char fitsComment[FLEN_COMMENT];
   float *imageData;
   float freqval;
   int thisChannel = 0;
   int nHeaderKeys = 0;
   char hdrKeyName[FLEN_COMMENT];
   int i;
   glob_t pglob;
   

   /* Start the clock */
   startTime = 0;
   startTime = clock();

   printf("\n");
   printf("RM synthesis v%s\n", VERSION_STR);
   printf("Written by Sarrvesh S. Sridhar\n");

   /* Parse the command line input */
   if((strcmp(inPattern, "-h") == 0) || (strcmp(inPattern, "--help") == 0)){
      /* Print help and exit */
      printf("Usage: %s <parset file> <output>\n\n", argv[0]);
      return(FAILURE);
   }
   if(argc!=NUM_INPUTS) {
      printf("ERROR: Invalid command line input. Terminating execution!\n");
      printf("Usage: %s <pattern for input> <output fits file> <output freq file>\n\n", argv[0]);
      return(FAILURE);
   }

   /* Find all files matching the input pattern */
   if( glob(inPattern, GLOB_NOSORT, NULL, &pglob) != 0 ) {
      printf("\nError: Unable to populate files matching the specified pattern\n\n");
      return(FAILURE);
   }
   else {
      nFitsFiles = pglob.gl_pathc;
   }
   printf("\nINFO: Found %d fits files", nFitsFiles);

   /* Loop over each specified file */
   for(thisChannel=0; thisChannel < nFitsFiles; thisChannel++) {
      strcpy(fitsFileName, pglob.gl_pathv[thisChannel]);
      //fitsFileName = pglob.gl_pathv[thisChannel];
      printf("\nProcessing file %s", fitsFileName);
      /* Open this fits file */
      fits_open_file(&in, fitsFileName, READONLY, &fitsStatus);
      if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
      /* Read the size of the input cube */
      inAxisLen[0] = inAxisLen[1] = inAxisLen[2] = inAxisLen[3] = 0;
      fits_read_key(in, TINT, "NAXIS1", &inAxisLen[0], fitsComment,&fitsStatus);
      fits_read_key(in, TINT, "NAXIS2", &inAxisLen[1], fitsComment,&fitsStatus);
      fits_read_key(in, TINT, "NAXIS3", &inAxisLen[2], fitsComment,&fitsStatus);
      fits_read_key(in, TINT, "NAXIS4", &inAxisLen[3], fitsComment,&fitsStatus);
      if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
      /* Check if all input images have the same size */
      if(thisChannel == 0) {
         axisLen[0] = inAxisLen[0];
         axisLen[1] = inAxisLen[1];
         axisLen[2] = inAxisLen[2];
         axisLen[3] = inAxisLen[3];
         nElements = axisLen[0] * axisLen[1] * axisLen[2] * axisLen[3];
         imageData = calloc(nElements, sizeof(float));

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
            /* Ignore this record if it contains the keys SIMPLE, 
               BITPIX, NAXIS, NAXIS1, NAXIS2, NAXIS3, and EXTEND */
            if(strstr(hdrKeyName, "SIMPLE")!=NULL ||
               strstr(hdrKeyName, "BITPIX")!=NULL ||
               strstr(hdrKeyName, "NAXIS") !=NULL ||
               strstr(hdrKeyName, "NAXIS1")!=NULL ||
               strstr(hdrKeyName, "NAXIS2")!=NULL ||
               strstr(hdrKeyName, "NAXIS3")!=NULL ||
               strstr(hdrKeyName, "EXTEND")!=NULL) {}
            else {
               fits_write_record(out, hdrKeyName, &fitsStatus);
            }
         }
         if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
         /* Open the freq files to write frequencies */
         freqFileFd = fopen(freqFile, "w+");
         if(freqFileFd == NULL) {
            printf("\nError: Unable to create the frequency file\n\n");
            return(FAILURE);
         }
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
      /* Read the frequency information from the fits header */
      fits_read_key(in, TFLOAT, "CRVAL3", &freqval, fitsComment, &fitsStatus);
      fprintf(freqFileFd, "%f\n", freqval);
      /* Read the image data */
      fPixel[0] = fPixel[1] = fPixel[2] = fPixel[3] = 1;
      fits_read_pix(in, TFLOAT, fPixel, nElements, 
                    NULL, imageData, NULL, &fitsStatus);
      fits_close_file(in, &fitsStatus);
      /* Write the read image as a channel to the out cube */
      oPixel[2] = thisChannel + 1;
      fits_write_pix(out, TFLOAT, oPixel, nElements, imageData, &fitsStatus);
      if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
   }
   free(imageData);
   fclose(freqFileFd);
   //globfree(&pglob);
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
