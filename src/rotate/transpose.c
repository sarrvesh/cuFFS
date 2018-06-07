#include<stdio.h>
#include<stdlib.h>
#include<sys/mman.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>
#include<time.h>

#include "constants.h"
#include "fitsio.h"

#define MMAP_FILE "./MMAP_FILE"

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
* Transpose the input FITS image
*
*************************************************************/
int transpose(char *inName, char *outName) {
   fitsfile *in, *out;
   int fitsStatus = SUCCESS;
   char fitsComment[FLEN_COMMENT];
   int nAxis;
   long inAxisLen[CUBE_DIM], outAxisLen[CUBE_DIM];
   float *thisInFrame;
   long fPixel[CUBE_DIM];
   long nInElements, nOutElements;
   int inIdx, outIdx;
   clock_t start, end;
   float readTime=0, rotTime=0, writeTime=0, closeTime=0;
      
   /* Variables for the memory map */
   int fDescriptor;
   off_t seekStatus;
   ssize_t writeStatus;
   int status;
   float *mmappedData;
   size_t mmapLen;
   
   /* Try opening the input FItS file */
   fits_open_file(&in, inName, READONLY, &fitsStatus);
   if(fitsStatus != SUCCESS) {
      return(checkFitsError(fitsStatus));
   }
   
   /* Read the size of the input cube */
   inAxisLen[0] = inAxisLen[1] = inAxisLen[2] = 0;
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
   
   /* Create a new file for the output FITS cube */
   outAxisLen[0] = inAxisLen[2];
   outAxisLen[1] = inAxisLen[1];
   outAxisLen[2] = inAxisLen[0];
   fits_create_file(&out, outName, &fitsStatus);
   fits_create_img(out, FLOAT_IMG, CUBE_DIM, outAxisLen, &fitsStatus);
   if(fitsStatus != SUCCESS) {
      return(checkFitsError(fitsStatus));
   }
   
   /* Create a memory mapped file */
   /* Based on code found on this web page: */
   /* https://www.linuxquestions.org/questions/programming-9/mmap-tutorial-c-c-511265/ */
   fDescriptor = open(MMAP_FILE, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
   if(fDescriptor < 0) {
      printf("ERROR: Unable to create a temp file.\n");
      return(FAILURE);
   }
   
   mmapLen = inAxisLen[0] * inAxisLen[1] * inAxisLen[2] * sizeof(float);
   
   /* Stretch the file size to match the size of the mmap */
   /* NOTE: Code segfaults without stretch and writing the last byte */
   seekStatus = 0;
   seekStatus = lseek(fDescriptor, mmapLen-1, SEEK_SET);
   if(seekStatus < 0) {
      close(fDescriptor);
      printf("ERROR: Unable to stretch the temp file.\n");
      printf("Please contact Sarrvesh if you see this.\n");
      return(FAILURE);
   }
   
   /* Write a single character to the end of the stretched file */
   writeStatus = 0;
   writeStatus = write(fDescriptor, "", 1);
   if(writeStatus < 0) {
      close(fDescriptor);
      printf("ERROR: Unable to write to the temp file.\n");
      return(FAILURE);
   }
   
   /* Now, map the file to memory */
   mmappedData = mmap(NULL, mmapLen, PROT_READ | PROT_WRITE, 
                      MAP_SHARED, fDescriptor, 0);
   if(mmappedData == MAP_FAILED) {
      close(fDescriptor);
      printf("ERROR: Unable to create mmap.");
      return(FAILURE);
   }
   
   /* Read the FITS cube, rotate, and then write to memory map */
   nInElements = inAxisLen[0] * inAxisLen[1];
   thisInFrame = (float *)calloc(nInElements, sizeof(*thisInFrame));
   fPixel[0] = fPixel[1] = 1;
   for(int frame=0; frame<inAxisLen[2]; frame++) {
      fPixel[2] = frame+1;
      start = clock();
      fits_read_pix(in, TFLOAT, fPixel, nInElements, 
                    NULL, thisInFrame, NULL, &fitsStatus);
      if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
      end = clock();
      readTime += (float)(end - start)/CLOCKS_PER_SEC;
      
      /* Rotate the read frame and write to the mmap array */
      start = clock();
      for(int row=0; row<inAxisLen[1]; row++) {
         for(int col=0; col<inAxisLen[0]; col++) {
            outIdx = col*outAxisLen[0]*outAxisLen[1] + 
                     row*outAxisLen[0] + frame;
            inIdx  = row*inAxisLen[0] + col;
            mmappedData[outIdx] = thisInFrame[inIdx];
         }
      }
      end = clock();
      rotTime += (float)(end - start)/CLOCKS_PER_SEC;
   }
   printf("Read time: %0.2f seconds\n", readTime);
   printf("Rotation time: %0.2f seconds\n", rotTime);
   
   /* Write the mmappedData to disk */
   start = clock();
   fPixel[0] = fPixel[1] = fPixel[2] = 1;
   nOutElements = inAxisLen[0] * inAxisLen[1] * inAxisLen[2];
   fits_write_pix(out, TFLOAT, fPixel, nOutElements, 
                  mmappedData, &fitsStatus);
   if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
   end = clock();
   writeTime = (float)(end - start)/CLOCKS_PER_SEC;
   printf("Write time: %0.2f seconds\n", writeTime);
      
   /* Free the mapped memory */
   start = clock();
   status = 0;
   status = munmap(mmappedData, mmapLen);
   if(status < 0) {
      close(fDescriptor);
      printf("ERROR: Unable to free the memory map.\n");
      return(FAILURE);
   }
   
   /* Close all open files */
   close(fDescriptor);
   remove(MMAP_FILE);
   fits_close_file(in, &fitsStatus);
   fits_close_file(out, &fitsStatus);
   if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
   end = clock();
   closeTime = (float)(end - start)/CLOCKS_PER_SEC;
   printf("Close time: %0.2f seconds\n", closeTime);
   
   return(SUCCESS);
}
