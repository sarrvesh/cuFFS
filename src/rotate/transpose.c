#include<stdio.h>
#include<stdlib.h>
#include<sys/mman.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>
#include<time.h>
#include<string.h>
#include<errno.h>

#include "constants.h"
#include "fitsio.h"

#define MMAP_FILE_IN  "./MMAP_IN_FILE"
#define MMAP_FILE_OUT "./MMAP_OUT_FILE"

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
   long fPixel[CUBE_DIM];
   long nElements;
   int inIdx, outIdx;
   long frameSize, frameOffset, frameAndRowOffset, outRowOffset;
   clock_t start, end;
   float readTime, rotTime, writeTime, assignTime=0, idxTime = 0;
      
   /* Variables for the memory map */
   int fDescIn, fDescOut;
   char errStr[FLEN_COMMENT];
   off_t seekStatIn, seekStatOut;
   ssize_t writeStatIn, writeStatOut;
   int statIn, statOut;
   float *mmappedIn, *mmappedOut;
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
   fDescIn  = open(MMAP_FILE_IN,  O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
   fDescOut = open(MMAP_FILE_OUT, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
   if((fDescIn < 0) || (fDescOut < 0)) {
      printf("ERROR: Unable to create a temp file.\n");
      return(FAILURE);
   }
   
   mmapLen = inAxisLen[0] * inAxisLen[1] * inAxisLen[2] * sizeof(float);
   
   /* Stretch the file size to match the size of the mmap */
   /* NOTE: Code segfaults without stretch and writing the last byte */
   seekStatIn  = seekStatOut = 0;
   seekStatIn  = lseek(fDescIn,  mmapLen-1, SEEK_SET);
   seekStatOut = lseek(fDescOut, mmapLen-1, SEEK_SET);
   if((seekStatIn < 0) || (seekStatOut < 0)) {
      close(fDescIn); close(fDescOut);
      printf("ERROR: Unable to stretch the temp file.\n");
      printf("Please contact Sarrvesh if you see this.\n");
      return(FAILURE);
   }
   
   /* Write a single character to the end of the stretched file */
   writeStatIn = writeStatOut = 0;
   writeStatIn = write(fDescIn, "", 1);
   writeStatOut= write(fDescOut,"", 1);
   if((writeStatIn < 0) || (writeStatOut < 0)) {
      close(fDescIn); close(fDescOut);
      printf("ERROR: Unable to write to the temp file.\n");
      return(FAILURE);
   }
   
   /* Now, map the file to memory */
   mmappedIn = mmap(NULL, mmapLen, PROT_READ | PROT_WRITE, 
                      MAP_SHARED, fDescIn, 0);
   mmappedOut= mmap(NULL, mmapLen, PROT_READ | PROT_WRITE, 
                      MAP_SHARED, fDescOut,0);
   if((mmappedIn == MAP_FAILED) || (mmappedOut == MAP_FAILED)) {
      close(fDescIn); close(fDescOut);
      printf("ERROR: Unable to create mmap.");
      strerror_r(errno, errStr, FLEN_COMMENTS);
      puts(errStr);
      return(FAILURE);
   }
   
   /* Read the FITS cube, rotate, and then write to memory map */
   start = clock();
   fPixel[0] = fPixel[1] = fPixel[2] = 1;
   nElements = inAxisLen[0] * inAxisLen[1] * inAxisLen[2];
   fits_read_pix(in, TFLOAT, fPixel, nElements, 
                 NULL, mmappedIn, NULL, &fitsStatus);
   if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
   end = clock();
   readTime = (float)(end - start)/CLOCKS_PER_SEC;
   printf("INFO: Read time: %f seconds\n", readTime);
   
   /* Rotate the input array */
   start = clock();
   frameSize = outAxisLen[0] * outAxisLen[1];
   for(int frame=0; frame<inAxisLen[2]; frame++) {
      frameOffset = frame*inAxisLen[2]*inAxisLen[1];
      for(int row=0; row<inAxisLen[1]; row++) {
         frameAndRowOffset = frameOffset + row*inAxisLen[0];
         outRowOffset = row*outAxisLen[0] + frame;
         for(int col=0; col<inAxisLen[0]; col++) {
            //start = clock();
            inIdx = frameAndRowOffset + col;
            outIdx = col*frameSize + outRowOffset;
            //end = clock();
            //idxTime += (float)(end - start)/CLOCKS_PER_SEC;
            
            //start = clock();
            mmappedOut[outIdx] = mmappedIn[inIdx]; 
            //end = clock();
            //assignTime += (float)(end - start)/CLOCKS_PER_SEC;
         }      
      }
   }
   end = clock();
   rotTime = (float)(end - start)/CLOCKS_PER_SEC;
   printf("INFO: Rotation time: %f seconds\n", rotTime);
   //printf("INFO: Index time: %f seconds\n", idxTime);
   //printf("INFO: Assign time: %f seconds\n", assignTime);
   
   /* Write the rotated array to disk */
   start = clock();
   fPixel[0] = fPixel[1] = fPixel[2] = 1;
   fits_write_pix(out, TFLOAT, fPixel, nElements, 
                 mmappedOut, &fitsStatus);
   if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
   end = clock();
   writeTime = (float)(end - start)/CLOCKS_PER_SEC;
   printf("INFO: Write time: %f seconds\n", writeTime);
   
   /* Free the mapped memory */
   statIn = statOut = 0;
   statIn = munmap(mmappedIn,  mmapLen);
   statOut= munmap(mmappedOut, mmapLen);
   if((statIn < 0) || (statOut < 0)) {
      close(fDescIn); close(fDescOut);
      printf("ERROR: Unable to free the memory map.\n");
      return(FAILURE);
   }
   /* Close all open files */
   close(fDescIn); close(fDescOut);
   remove(MMAP_FILE_IN);
   remove(MMAP_FILE_OUT);
   fits_close_file(in, &fitsStatus);
   fits_close_file(out, &fitsStatus);
   if(fitsStatus != SUCCESS) { return(checkFitsError(fitsStatus)); }
   
   return(SUCCESS);
}
