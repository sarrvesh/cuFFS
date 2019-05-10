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
#include<string.h>
#include<sys/mman.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>
#include<omp.h>
#include<math.h>
#include<stdbool.h>

#include "cpu_version.h"
#include "cpu_fileaccess.h"
#include "cpu_rmsf.h"
#include "cpu_rmclean.h"

#define NUM_INPUTS 2
#define MMAP_Q "./.MMAP_Q"
#define MMAP_U "./.MMAP_Q"
#define MMAP_P "./.MMAP_Q"
#define Q_DIRTY "q_dirty_cube"
#define U_DIRTY "u_dirty_cube"
#define P_DIRTY "p_dirty_cube"

/*************************************************************
*
* Set all elements in an array to zero
*
*************************************************************/
void zeroInitialize(float array[], long nElements) {
   int i;
   for(i=0; i<nElements; i++) { array[i] = 0.; }
}

/*************************************************************
*
* Multiply all elements of an array by a constant
*
*************************************************************/
void multiplyByConstant(float array[], float K, 
                        long nOutElements, int nThreads) {

   int i;
   #pragma omp parallel for num_threads(nThreads)
   for(i=0; i<nOutElements; i++) { array[i] *= K; }
}

/*************************************************************
*
* Form P(\phi) using Q(\phi) and U(\phi)
*
*************************************************************/
void formPFromQU(float mmappedQ[], float mmappedU[], float mmappedP[], 
                 long nOutElements, int nThreads) {
   long k;
   #pragma omp parallel for num_threads(nThreads)
   for(k=0; k<nOutElements; k++) {
      mmappedP[k] = sqrt(mmappedQ[k]*mmappedQ[k] +
                         mmappedU[k]*mmappedU[k]);
   }
}

/*************************************************************
*
* Main code
*
*************************************************************/
int main(int argc, char *argv[]) {
   char *parsetFileName = argv[1];
   struct optionsList inOptions;
   struct parList params;
   int i, k;
   char filenamefull[256];
   
   /* Variables for memory map */
   int fDescQ, fDescU, fDescP;
   off_t seekStatQ, seekStatU, seekStatP;
   off_t writeStatQ, writeStatU, writeStatP;
   float *mmappedQ, *mmappedU, *mmappedP;
   size_t sizeOfOutCube;
   
   /* Variables for FITS input */
   int fitsStatus;
   float *qImageArray, *uImageArray;
   long nInElements, nOutElements;
   long fPixel[CUBE_DIM];
   
   /* Variables for RM synthesis */
   float *lambdaDiff2;
   
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
   printf("INFO: Checking input files\n");
   checkInputFiles(&inOptions, &params);
   
   /* Gather information about input fits header and setup output images */
   fitsStatus = getFitsHeader(&inOptions, &params);
   checkFitsError(fitsStatus);
   
   /* Estimate the size of the output cubes
   * If they are fit in main memory, hold them there
   * If not create, memory maps. */
   printf("INFO: Allocating memory for output cubes.\n");
   nOutElements = params.qAxisLen1 * params.qAxisLen2 * inOptions.nPhi;
   sizeOfOutCube = nOutElements * sizeof(float);
   /* Try to create memory maps for the output cubes */
   /* Based on code found on this web page: */
   /* https://www.linuxquestions.org/questions/programming-9/mmap-tutorial-c-c-511265/ */
   fDescQ = open(MMAP_Q,  O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
   fDescU = open(MMAP_U,  O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
   fDescP = open(MMAP_P,  O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
   if( (fDescQ<0) || (fDescU<0) || (fDescP<0) ) {
      printf("ERROR: Unable to create a temp file.\n\n");
      return 1;
   }
   // Stretch the file size to match the size of the mmap
   // Code will segfault without stretch and writing the last byte 
   seekStatQ = seekStatU = seekStatP = 0;
   seekStatQ = lseek(fDescQ, sizeOfOutCube-1, SEEK_SET);
   seekStatU = lseek(fDescU, sizeOfOutCube-1, SEEK_SET);
   seekStatP = lseek(fDescP, sizeOfOutCube-1, SEEK_SET);
   if( (seekStatQ<0) || (seekStatU<0) || (seekStatP<0) ) {
      close(fDescQ); close(fDescU); close(fDescP);
      printf("ERROR: Unable to stretch temp files.\n");
      printf("Please contact Sarrvesh if you see this.\n\n");
      return 1;
   }
   // Write a single character to the end of the stretched file 
   writeStatQ = writeStatU = writeStatP = 0;
   writeStatQ = write(fDescQ, "", 1);
   writeStatU = write(fDescU, "", 1);
   writeStatP = write(fDescP, "", 1);
   if( (writeStatQ<0) || (writeStatU<0) || (writeStatP<0) ) {
      close(fDescQ); close(fDescU); close(fDescP);
      printf("ERROR: Unable to write to the temp files.\n");
      printf("Please contact Sarrvesh if you see this.\n\n");
      return 1;
   }
   // Finally, map the file to memory
   mmappedQ = mmap(NULL, sizeOfOutCube, PROT_READ | PROT_WRITE, 
                     MAP_SHARED, fDescQ, 0);
   mmappedU = mmap(NULL, sizeOfOutCube, PROT_READ | PROT_WRITE, 
                     MAP_SHARED, fDescU, 0);
   mmappedP = mmap(NULL, sizeOfOutCube, PROT_READ | PROT_WRITE, 
                     MAP_SHARED, fDescP, 0);
   // Initialize mmappedQ, mmappedU, and to zero
   zeroInitialize(mmappedQ, nOutElements);
   zeroInitialize(mmappedU, nOutElements);
   
   /* Print some useful information */
   printOptions(inOptions, params);
   
   /* Read in the frequency list */
   if(getFreqList(&inOptions, &params)) { return 1; }
   
   /* Generate RMSF and write to disk */
   printf("INFO: Computing RMSF\n");
   if(generateRMSF(&inOptions, &params)) {
      printf("Error: Mem alloc failed while generating RMSF\n");
      return 1;
   }
   if(writeRMSF(inOptions, params)) {
      printf("Error: Unable to write RMSF to disk\n\n");
      return 1;
   }
   
   /* Do RM synthesis */
   // Allocate memory for input images
   nInElements = params.qAxisLen1 * params.qAxisLen2;
   qImageArray = calloc(nInElements, sizeof(float));
   uImageArray = calloc(nInElements, sizeof(float));
   lambdaDiff2 = calloc(params.qAxisLen3, sizeof(float));
   if((qImageArray == NULL) || 
      (uImageArray == NULL) ||
      (lambdaDiff2 == NULL)) {
      printf("ERROR: Unable to allocate memory for input images.\n\n");
      return 1;
   }
   // Compute lambdaDiff2
   for(i=0; i<params.qAxisLen1; i++) 
      lambdaDiff2[i] = 2.0*(params.lambda2[i]-params.lambda20);
   // Process each frame individually
   fPixel[0] = fPixel[1] = 1;
   for(k=1; k<=params.qAxisLen3; k++) {
      // Read this frequency plane
      fPixel[2] = k;
      fits_read_pix(params.qFile, TFLOAT, fPixel, nInElements, NULL,
                    qImageArray, NULL, &fitsStatus);
      fits_read_pix(params.uFile, TFLOAT, fPixel, nInElements, NULL,
                    uImageArray, NULL, &fitsStatus);
      checkFitsError(fitsStatus);
      // Spawn threads to compute output Q(\phi) and U(\phi)
      #pragma omp parallel for num_threads(inOptions.nThreads)
      for(int planeIdx = 0; planeIdx<inOptions.nPhi; planeIdx++) {
         int rowIdx, colIdx, outIdx, inIdx;
         float angle = params.phiAxis[planeIdx] * lambdaDiff2[k];
         float cos_angle = cos(angle);
         float sin_angle = sin(angle);
         for(rowIdx=0; rowIdx<params.qAxisLen1; rowIdx++) {
            for(colIdx=0; colIdx<params.qAxisLen2; colIdx++) {
               inIdx  = (rowIdx*params.qAxisLen1) + colIdx;
               outIdx = (planeIdx*params.qAxisLen1*params.qAxisLen2) + inIdx;
               mmappedQ[outIdx] += qImageArray[inIdx]*cos_angle +
                                      uImageArray[inIdx]*sin_angle;
               mmappedU[outIdx] += uImageArray[inIdx]*cos_angle -
                                      qImageArray[inIdx]*sin_angle;
            }
         }
      } // End of pragma
   }
   // Q(\phi) and U(\phi) still need to be scaled by K
   multiplyByConstant(mmappedQ, params.K, nOutElements, inOptions.nThreads);
   multiplyByConstant(mmappedU, params.K, nOutElements, inOptions.nThreads);
   
   // Combine Q(\phi) and U(\phi) to form P(\phi)
   formPFromQU(mmappedQ, mmappedU, mmappedP, nOutElements, inOptions.nThreads);
   
   /* If requested, do RM-CLEAN */
   if((inOptions.doRMClean == 1) && (inOptions.nIter > 0)) {
      #pragma omp parallel for num_threads(inOptions.nThreads)
      for(int rowIdx=0; rowIdx<params.qAxisLen1; rowIdx++) {
         float qResidual[inOptions.nPhi], uResidual[inOptions.nPhi], 
               pResidual[inOptions.nPhi];
         float qClean[inOptions.nPhi], uClean[inOptions.nPhi], 
               pClean[inOptions.nPhi];
         float maxP;
         int idxMaxP;
         /* Loop over each line of sight */
         for(int colIdx=0; colIdx<params.qAxisLen2; colIdx++) {
            /* Get the Q, U, and P values for this LOS */
            getLineOfSight(qResidual, mmappedQ, rowIdx, colIdx, 
                           inOptions.nPhi, params.qAxisLen1, params.qAxisLen2);
            getLineOfSight(uResidual, mmappedU, rowIdx, colIdx, 
                           inOptions.nPhi, params.qAxisLen1, params.qAxisLen2);
            getLineOfSight(pResidual, mmappedP, rowIdx, colIdx, 
                           inOptions.nPhi, params.qAxisLen1, params.qAxisLen2);
            
            /* Do deconvolution */
            while(true) {
               /* Initialize the clean arrays to 0 */
               zeroInitialize(qClean, inOptions.nPhi);
               zeroInitialize(uClean, inOptions.nPhi);
               zeroInitialize(pClean, inOptions.nPhi);
               
               /* Check if P in this LOS has values greater than threshold */
               findMaxP(pResidual, inOptions.nPhi, &maxP, &idxMaxP);
               if(maxP < inOptions.threshold) { break; }
            }
            
            /* Deconvolution has ended. */
            /* Write the output products for this LoS to cube */
         }
      }
   }
   
   /* Transfer the outputs from a memory map to fits cubes */
   printf("INFO: Writing dirty cubes to disk\n");
   sprintf(filenamefull, "%s%s.fits", inOptions.outPrefix, Q_DIRTY);
   writeOutputToDisk(&inOptions, &params, mmappedQ, nOutElements, filenamefull);
   munmap(mmappedQ, sizeOfOutCube); 
   close(fDescQ); remove(MMAP_Q);
   sprintf(filenamefull, "%s%s.fits", inOptions.outPrefix, U_DIRTY);
   writeOutputToDisk(&inOptions, &params, mmappedU, nOutElements, filenamefull);
   munmap(mmappedU, sizeOfOutCube); 
   close(fDescU); remove(MMAP_U);
   sprintf(filenamefull, "%s%s.fits", inOptions.outPrefix, P_DIRTY);
   writeOutputToDisk(&inOptions, &params, mmappedP, nOutElements, filenamefull);
   munmap(mmappedP, sizeOfOutCube); 
   close(fDescP); remove(MMAP_P);
   
   /* Free up all allocated memory */
   freeStructures(&inOptions, &params);
   free(qImageArray); free(uImageArray);
   free(lambdaDiff2);
   
   /* Close all open files */
   // Close the input and output fits files
   fits_close_file(params.qFile, &fitsStatus);
   fits_close_file(params.uFile, &fitsStatus);
   checkFitsError(fitsStatus);

   printf("\n");
   return 0;
}
