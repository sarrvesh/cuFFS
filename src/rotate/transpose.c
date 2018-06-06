#include<stdlib.h>
#include<time.h>

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
}

/*************************************************************
*
* Transpose the input FITS image
*
*************************************************************/
int transpose(char *mode, char *inName, char *outName) {
   fitsfile *in, *out;
   int fitsStatus = SUCCESS;
   char fitsComment[FLEN_COMMENT];
   int nAxis;
   long inAxisLen[CUBE_DIM], outAxisLen[CUBE_DIM];
   int nInElements;
   long fInPixel[CUBE_DIM], fOutPixel[CUBE_DIM], lOutPixel[CUBE_DIM];
   float *thisRow;
   //clock_t start=0, end=0, readTime=0, writeTime=0;
   
   /* Try opening the input FItS file */
   //start = clock();
   fits_open_file(&in, inName, READONLY, &fitsStatus);
   if(fitsStatus != SUCCESS) {
      return(checkFitsError(fitsStatus));
   }
   //end = clock();
   //readTime += (end - start);
   
   /* Create the output FITS cube */
   //start = clock();
   fits_create_file(&out, outName, &fitsStatus);
   if(fitsStatus != SUCCESS) {
      return(checkFitsError(fitsStatus));
   }
   //end = clock();
   //writeTime += (end - start);
   
   /* Read the size of the input cube */
   //start = clock();
   fits_read_key(in, TINT, "NAXIS", &nAxis, fitsComment, &fitsStatus);   
   fits_read_key(in, TINT, "NAXIS1", &inAxisLen[0], fitsComment, &fitsStatus);   
   fits_read_key(in, TINT, "NAXIS2", &inAxisLen[1], fitsComment, &fitsStatus);   
   fits_read_key(in, TINT, "NAXIS3", &inAxisLen[2], fitsComment, &fitsStatus);   
   if(fitsStatus != SUCCESS) {
      return(checkFitsError(fitsStatus));
   }
   /* Terminate if input cube is non-three dimensional */
   if(nAxis != CUBE_DIM) {
      printf("Error: Input cube is not 3-dimensional.\n");
      printf("Terminating execution\n\n");
      return(FAILURE);
   }   
   printf("Input cube: %ld x %ld x %ld\n", 
           inAxisLen[0], inAxisLen[1], inAxisLen[2]);
   //end = clock();
   //readTime += (end - start);
   
   /* Create the image HDU for the output cube */
   //start = clock();
   outAxisLen[0] = inAxisLen[2];
   outAxisLen[1] = inAxisLen[1];
   outAxisLen[2] = inAxisLen[0];
   fits_create_img(out, FLOAT_IMG, CUBE_DIM, outAxisLen, &fitsStatus);
   if(fitsStatus != SUCCESS) {
      return(checkFitsError(fitsStatus));
   }
   //end = clock();
   //writeTime += (end - start);
   
   /* Do the cube rotation */
   nInElements = inAxisLen[0];
   thisRow = (float *)calloc(nInElements, sizeof(*thisRow));
   fInPixel[0] = 1;
   fOutPixel[2] = 1;
   lOutPixel[2] = nInElements;
   for(int i=1; i<=inAxisLen[2]; i++) {
      for(int j=1; j<=inAxisLen[1]; j++) {
         /* Read a row from the input */
         fInPixel[1] = j;
         fInPixel[2] = i;
         //start = clock();
         fits_read_pix(in, TFLOAT, fInPixel, nInElements, 
                       NULL, thisRow, NULL, &fitsStatus);
         //end = clock();
         //readTime += (end - start);
         
         /* Write the read row as the third dimension */
         fOutPixel[0] = lOutPixel[0] = i;
         fOutPixel[1] = lOutPixel[1] = j;
         //start = clock();
         fits_write_subset(out, TFLOAT, fOutPixel, lOutPixel, 
                           thisRow, &fitsStatus);
         //end = clock();
         //writeTime += (end - start);
      }
   }
   if(fitsStatus != SUCCESS) {
      return(checkFitsError(fitsStatus));
   }


   
   /* Close the open FITS files */
   free(thisRow);
   //start = clock();
   fits_close_file(in, &fitsStatus);
   //end = clock();
   //readTime += (end - start);
   //start = clock();
   fits_close_file(out, &fitsStatus);
   //end = clock();
   //writeTime += (end - start);
   if(fitsStatus != SUCCESS) {
      return(checkFitsError(fitsStatus));
   }
   
   /* Report execution time */
   //readTime = readTime / CLOCKS_PER_SEC;
   //writeTime= writeTime/ CLOCKS_PER_SEC;
   //printf("INFO: Read time: %ld\n", readTime);
   //printf("INFO: Write time: %ld\n", writeTime);
   
   return(SUCCESS);
}
