/******************************************************************************
fileaccess.c
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
#include "fitsio.h"
#include "structures.h"
#include "constants.h"
#include "fileaccess.h"
#include "hdf5.h"
#include "hdf5_hl.h"

#define BUNIT   "JY/BEAM"
#define RM      "PHI"

/*************************************************************
*
* Check Fitsio error and exit if required.
*
*************************************************************/
void checkFitsError(int status) {
    if(status) {
        printf("ERROR:");
        fits_report_error(stdout, status);
        printf("\n");
        exit(FAILURE);
    }
}

/*************************************************************
*
* Check of the input files are open-able
*
*************************************************************/
void checkInputFiles(struct optionsList *inOptions, struct parList *params) {
   int fitsStatus = SUCCESS;
   herr_t error;
   char buf[STRING_BUF_LEN];
   
   if(inOptions->fileFormat == FITS) {
      /* Check if all the input fits files are accessible */
      fits_open_file(&(params->qFile), inOptions->qCubeName, READONLY, &fitsStatus);
      fits_open_file(&(params->uFile), inOptions->uCubeName, READONLY, &fitsStatus);
      checkFitsError(fitsStatus);
   }
   else if(inOptions->fileFormat == HDF5) { 
      /* Open HDF5 files */
      params->qFileh5 = H5Fopen(inOptions->qCubeName, H5F_ACC_RDONLY, H5P_DEFAULT);
      params->uFileh5 = H5Fopen(inOptions->uCubeName, H5F_ACC_RDONLY, H5P_DEFAULT);
      if(params->qFileh5 < 0 || params->uFileh5 < 0) {
         printf("Error: Unable to open the input HDF5 files\n\n");
         exit(FAILURE);
      }
      /* Check if the hdf5 files are compatible with HDFITS format */
      error = H5LTget_attribute_string(params->qFileh5, "/", "CLASS", buf);
   }
   else {}
   
   /* Check if you can open the frequency file */
   params->freq = fopen(inOptions->freqFileName, FILE_READONLY);
   if(params->freq == NULL) {
      printf("Error: Unable to open the frequency file\n\n");
      exit(FAILURE);
   }
}

/*************************************************************
*
* Read header information from the fits files
*
*************************************************************/
int getFitsHeader(struct optionsList *inOptions, struct parList *params) {
    int fitsStatus = SUCCESS;
    char fitsComment[FLEN_COMMENT];
    
    /* Remember that the input fits images are rotated. */
    /* Frequency is the first axis */
    /* RA is the second */
    /* Dec is the third */
    
    /* Get the image dimensions from the Q cube */
    fits_read_key(params->qFile, TINT, "NAXIS", &params->qAxisNum, 
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS1", &params->qAxisLen3, 
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS2", &params->qAxisLen1, 
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS3", &params->qAxisLen2, 
      fitsComment, &fitsStatus);
    /* Get the image dimensions from the U cube */
    fits_read_key(params->uFile, TINT, "NAXIS", &params->uAxisNum, 
      fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS1", &params->uAxisLen3, 
      fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS2", &params->uAxisLen1, 
      fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS3", &params->uAxisLen2,
      fitsComment, &fitsStatus);
    /* Get WCS information */
    fits_read_key(params->qFile, TFLOAT, "CRVAL1", &params->crval3,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CRVAL2", &params->crval1,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CRVAL3", &params->crval2,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CRPIX1", &params->crpix3, 
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CRPIX2", &params->crpix1,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CRPIX3", &params->crpix2,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CDELT1", &params->cdelt3,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CDELT2", &params->cdelt1,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CDELT3", &params->cdelt2,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TSTRING, "CTYPE1", &params->ctype3,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TSTRING, "CTYPE2", &params->ctype1,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TSTRING, "CTYPE3", &params->ctype2,
      fitsComment, &fitsStatus);
    
    return(fitsStatus);
}

/*************************************************************
*
* Read header information from the HDF5 files
*
*************************************************************/
int getHDF5Header(struct optionsList *inOptions, struct parList *params) {
    hsize_t tempArr[N_DIMS];
    herr_t error;
    
    /* Remember that the input fits images are NOT rotated. */
    /* RA is the first axis */
    /* Dec is the second */
    /* Frequency is the third */
    
    /* Get the dimensionality of the input datasets */
    error = H5LTget_dataset_ndims(params->qFileh5, PRIMARYDATA, &(params->qAxisNum));
    error = H5LTget_dataset_ndims(params->uFileh5, PRIMARYDATA, &(params->uAxisNum));
    /* Get the sizes of each dimension */
    error = H5LTget_dataset_info(params->qFileh5, PRIMARYDATA, tempArr, NULL, NULL);
    params->qAxisLen1 = tempArr[1];
    params->qAxisLen2 = tempArr[2];
    params->qAxisLen3 = tempArr[0];
    error = H5LTget_dataset_info(params->uFileh5, PRIMARYDATA, tempArr, NULL, NULL);
    params->uAxisLen1 = tempArr[1];
    params->uAxisLen2 = tempArr[2];
    params->uAxisLen3 = tempArr[0];
    /* Get WCS information */
    H5LTget_attribute_float(params->qFileh5, PRIMARY, "CRVAL1", &params->crval1);
    H5LTget_attribute_float(params->qFileh5, PRIMARY, "CRVAL2", &params->crval2);
    H5LTget_attribute_float(params->qFileh5, PRIMARY, "CRVAL3", &params->crval3);
    H5LTget_attribute_float(params->qFileh5, PRIMARY, "CDELT1", &params->cdelt1);
    H5LTget_attribute_float(params->qFileh5, PRIMARY, "CDELT2", &params->cdelt2);
    H5LTget_attribute_float(params->qFileh5, PRIMARY, "CDELT3", &params->cdelt3);
    H5LTget_attribute_float(params->qFileh5, PRIMARY, "CRPIX1", &params->crpix1);
    H5LTget_attribute_float(params->qFileh5, PRIMARY, "CRPIX2", &params->crpix2);
    H5LTget_attribute_float(params->qFileh5, PRIMARY, "CRPIX3", &params->crpix3);
    H5LTget_attribute_string(params->qFileh5, PRIMARY, "CTYPE1", params->ctype1);
    H5LTget_attribute_string(params->qFileh5, PRIMARY, "CTYPE2", params->ctype2);
    H5LTget_attribute_string(params->qFileh5, PRIMARY, "CTYPE3", params->ctype3);
}

/*************************************************************
*
* Create output FITS images
*
*************************************************************/
void makeOutputFitsImages(struct optionsList *inOptions, struct parList *params) {
   int stat = SUCCESS;
   char filenamefull[FILENAME_LEN];
   long naxis[FITS_OUT_NAXIS];
   char fComment[FILENAME_LEN];
   float tempVar;

   /* Create the output Q, U, and P images */
   sprintf(filenamefull, "%s%s.fits", inOptions->outPrefix, Q_DIRTY);
   fits_create_file(&params->qDirty, filenamefull, &stat);
   sprintf(filenamefull, "%s%s.fits", inOptions->outPrefix, U_DIRTY);
   fits_create_file(&params->uDirty, filenamefull, &stat);
   sprintf(filenamefull, "%s%s.fits", inOptions->outPrefix, P_DIRTY);
   fits_create_file(&params->pDirty, filenamefull, &stat);
   checkFitsError(stat);
   
   /* What are the output cube sizes */
   naxis[0] = inOptions->nPhi;
   naxis[1] = params->qAxisLen1;
   naxis[2] = params->qAxisLen2;
   
   /* Create the header for each output image */
   fits_create_img(params->qDirty, FLOAT_IMG, FITS_OUT_NAXIS, naxis, &stat);
   fits_create_img(params->uDirty, FLOAT_IMG, FITS_OUT_NAXIS, naxis, &stat);
   fits_create_img(params->pDirty, FLOAT_IMG, FITS_OUT_NAXIS, naxis, &stat);
   checkFitsError(stat);
   
   /* Set the relevant keyvalues */
   fits_write_key(params->qDirty, TSTRING, "BUNIT", BUNIT, fComment, &stat);
   fits_write_key(params->uDirty, TSTRING, "BUNIT", BUNIT, fComment, &stat);
   fits_write_key(params->pDirty, TSTRING, "BUNIT", BUNIT, fComment, &stat);

   fits_write_key(params->qDirty, TDOUBLE, "CRVAL1", &inOptions->phiMin, fComment, &stat);
   fits_write_key(params->uDirty, TDOUBLE, "CRVAL1", &inOptions->phiMin, fComment, &stat);
   fits_write_key(params->pDirty, TDOUBLE, "CRVAL1", &inOptions->phiMin, fComment, &stat);

   fits_write_key(params->qDirty, TDOUBLE, "CDELT1", &inOptions->dPhi, fComment, &stat);
   fits_write_key(params->uDirty, TDOUBLE, "CDELT1", &inOptions->dPhi, fComment, &stat);
   fits_write_key(params->pDirty, TDOUBLE, "CDELT1", &inOptions->dPhi, fComment, &stat);

   tempVar = 1;
   fits_write_key(params->qDirty, TFLOAT, "CRPIX1", &tempVar, fComment, &stat);
   fits_write_key(params->uDirty, TFLOAT, "CRPIX1", &tempVar, fComment, &stat);
   fits_write_key(params->pDirty, TFLOAT, "CRPIX1", &tempVar, fComment, &stat);

   fits_write_key(params->qDirty, TSTRING, "CTYPE1", RM, fComment, &stat);
   fits_write_key(params->uDirty, TSTRING, "CTYPE1", RM, fComment, &stat);
   fits_write_key(params->pDirty, TSTRING, "CTYPE1", RM, fComment, &stat);

   fits_write_key(params->qDirty, TFLOAT, "CRVAL2", &params->crval1, fComment, &stat);
   fits_write_key(params->uDirty, TFLOAT, "CRVAL2", &params->crval1, fComment, &stat);
   fits_write_key(params->pDirty, TFLOAT, "CRVAL2", &params->crval1, fComment, &stat);

   fits_write_key(params->qDirty, TFLOAT, "CDELT2", &params->cdelt1, fComment, &stat);
   fits_write_key(params->uDirty, TFLOAT, "CDELT2", &params->cdelt1, fComment, &stat);
   fits_write_key(params->pDirty, TFLOAT, "CDELT2", &params->cdelt1, fComment, &stat);

   fits_write_key(params->qDirty, TFLOAT, "CRPIX2", &params->crpix1, fComment, &stat);
   fits_write_key(params->uDirty, TFLOAT, "CRPIX2", &params->crpix1, fComment, &stat);
   fits_write_key(params->pDirty, TFLOAT, "CRPIX2", &params->crpix1, fComment, &stat);

   fits_write_key(params->qDirty, TSTRING, "CTYPE2", params->ctype1, fComment, &stat);
   fits_write_key(params->uDirty, TSTRING, "CTYPE2", params->ctype1, fComment, &stat);
   fits_write_key(params->pDirty, TSTRING, "CTYPE2", params->ctype1, fComment, &stat);
   
   fits_write_key(params->qDirty, TFLOAT, "CRVAL3", &params->crval2, fComment, &stat);
   fits_write_key(params->uDirty, TFLOAT, "CRVAL3", &params->crval2, fComment, &stat);
   fits_write_key(params->pDirty, TFLOAT, "CRVAL3", &params->crval2, fComment, &stat);
   
   fits_write_key(params->qDirty, TFLOAT, "CDELT3", &params->cdelt2, fComment, &stat);
   fits_write_key(params->uDirty, TFLOAT, "CDELT3", &params->cdelt2, fComment, &stat);
   fits_write_key(params->pDirty, TFLOAT, "CDELT3", &params->cdelt2, fComment, &stat);
   
   fits_write_key(params->qDirty, TFLOAT, "CRPIX3", &params->crpix2, fComment, &stat);
   fits_write_key(params->uDirty, TFLOAT, "CRPIX3", &params->crpix2, fComment, &stat);
   fits_write_key(params->pDirty, TFLOAT, "CRPIX3", &params->crpix2, fComment, &stat);
   
   fits_write_key(params->qDirty, TSTRING, "CTYPE3", params->ctype2, fComment, &stat);
   fits_write_key(params->uDirty, TSTRING, "CTYPE3", params->ctype2, fComment, &stat);
   fits_write_key(params->pDirty, TSTRING, "CTYPE3", params->ctype2, fComment, &stat);
   
   checkFitsError(stat);
}

/*************************************************************
*
* Create output HDF5 cubes
*
*************************************************************/
void makeOutputHDF5Images(struct optionsList *inOptions, struct parList *params) {
   char filenamefull[FILENAME_LEN];
   float tempVar = 1;

   /* Create the output Q, U, and P images */
   sprintf(filenamefull, "%s%s.h5", inOptions->outPrefix, Q_DIRTY);
   params->qDirtyH5 = H5Fcreate(filenamefull, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
   sprintf(filenamefull, "%s%s.h5", inOptions->outPrefix, U_DIRTY);
   params->uDirtyH5 = H5Fcreate(filenamefull, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
   sprintf(filenamefull, "%s%s.h5", inOptions->outPrefix, P_DIRTY);
   params->pDirtyH5 = H5Fcreate(filenamefull, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);

   /* Create attributes for the / group */
   H5LTset_attribute_float(params->qDirtyH5, ROOT, "CRVAL1", &(params->crval1), sizeof(params->crval1));
   H5LTset_attribute_float(params->qDirtyH5, ROOT, "CRVAL2", &(params->crval2), sizeof(params->crval2));
   H5LTset_attribute_double(params->qDirtyH5, ROOT, "CRVAL3", &(inOptions->phiMin), sizeof(inOptions->phiMin));
   H5LTset_attribute_float(params->qDirtyH5, ROOT, "CRPIX1", &(params->crpix1), sizeof(params->crpix1));
   H5LTset_attribute_float(params->qDirtyH5, ROOT, "CRPIX2", &(params->crpix2), sizeof(params->crpix2));
   H5LTset_attribute_float(params->qDirtyH5, ROOT, "CRPIX3", &tempVar, sizeof(tempVar));
   H5LTset_attribute_float(params->qDirtyH5, ROOT, "CDELT1", &(params->cdelt1), sizeof(params->cdelt1));
   H5LTset_attribute_float(params->qDirtyH5, ROOT, "CDELT2", &(params->cdelt2), sizeof(params->cdelt2));
   H5LTset_attribute_double(params->qDirtyH5, ROOT, "CDELT3", &(inOptions->dPhi), sizeof(inOptions->dPhi));
   H5LTset_attribute_string(params->qDirtyH5, ROOT, "CTYPE1", params->ctype1);
   H5LTset_attribute_string(params->qDirtyH5, ROOT, "CTYPE2", params->ctype2);
   H5LTset_attribute_string(params->qDirtyH5, ROOT, "CTYPE3", RM);

   H5LTset_attribute_float(params->uDirtyH5, ROOT, "CRVAL1", &(params->crval1), sizeof(params->crval1));
   H5LTset_attribute_float(params->uDirtyH5, ROOT, "CRVAL2", &(params->crval2), sizeof(params->crval2));
   H5LTset_attribute_double(params->uDirtyH5, ROOT, "CRVAL3", &(inOptions->phiMin), sizeof(inOptions->phiMin));
   H5LTset_attribute_float(params->uDirtyH5, ROOT, "CRPIX1", &(params->crpix1), sizeof(params->crpix1));
   H5LTset_attribute_float(params->uDirtyH5, ROOT, "CRPIX2", &(params->crpix2), sizeof(params->crpix2));
   H5LTset_attribute_float(params->uDirtyH5, ROOT, "CRPIX3", &tempVar, sizeof(tempVar));
   H5LTset_attribute_float(params->uDirtyH5, ROOT, "CDELT1", &(params->cdelt1), sizeof(params->cdelt1));
   H5LTset_attribute_float(params->uDirtyH5, ROOT, "CDELT2", &(params->cdelt2), sizeof(params->cdelt2));
   H5LTset_attribute_double(params->uDirtyH5, ROOT, "CDELT3", &(inOptions->dPhi), sizeof(inOptions->dPhi));
   H5LTset_attribute_string(params->uDirtyH5, ROOT, "CTYPE1", params->ctype1);
   H5LTset_attribute_string(params->uDirtyH5, ROOT, "CTYPE2", params->ctype2);
   H5LTset_attribute_string(params->uDirtyH5, ROOT, "CTYPE3", RM);

   H5LTset_attribute_float(params->pDirtyH5, ROOT, "CRVAL1", &(params->crval1), sizeof(params->crval1));
   H5LTset_attribute_float(params->pDirtyH5, ROOT, "CRVAL2", &(params->crval2), sizeof(params->crval2));
   H5LTset_attribute_double(params->pDirtyH5, ROOT, "CRVAL3", &(inOptions->phiMin), sizeof(inOptions->phiMin));
   H5LTset_attribute_float(params->pDirtyH5, ROOT, "CRPIX1", &(params->crpix1), sizeof(params->crpix1));
   H5LTset_attribute_float(params->pDirtyH5, ROOT, "CRPIX2", &(params->crpix2), sizeof(params->crpix2));
   H5LTset_attribute_float(params->pDirtyH5, ROOT, "CRPIX3", &tempVar, sizeof(tempVar));
   H5LTset_attribute_float(params->pDirtyH5, ROOT, "CDELT1", &(params->cdelt1), sizeof(params->cdelt1));
   H5LTset_attribute_float(params->pDirtyH5, ROOT, "CDELT2", &(params->cdelt2), sizeof(params->cdelt2));
   H5LTset_attribute_double(params->pDirtyH5, ROOT, "CDELT3", &(inOptions->dPhi), sizeof(inOptions->dPhi));
   H5LTset_attribute_string(params->pDirtyH5, ROOT, "CTYPE1", params->ctype1);
   H5LTset_attribute_string(params->pDirtyH5, ROOT, "CTYPE2", params->ctype2);
   H5LTset_attribute_string(params->pDirtyH5, ROOT, "CTYPE3", RM);

   /* CLASS attribute of ROOT should be set to HDFITS */
   H5LTset_attribute_string(params->qDirtyH5, ROOT, "CLASS", HDFITS);
   H5LTset_attribute_string(params->uDirtyH5, ROOT, "CLASS", HDFITS);
   H5LTset_attribute_string(params->pDirtyH5, ROOT, "CLASS", HDFITS);

   /* Create the /PRIMARY/DATA dataset */ 
   
}

/*************************************************************
*
* Read the list of frequencies from the input freq file
*
*************************************************************/
int getFreqList(struct optionsList *inOptions, struct parList *params) {
    int i;
    float tempFloat;
    
    params->freqList = calloc(params->qAxisLen3, sizeof(params->freqList));
    if(params->freqList == NULL) {
        printf("Error: Mem alloc failed while reading in frequency list\n\n");
        return(FAILURE);
    }
    for(i=0; i<params->qAxisLen3; i++) {
        fscanf(params->freq, "%f", &params->freqList[i]);
        if(feof(params->freq)) {
            printf("Error: Frequency values and fits frames don't match\n");
            return(FAILURE);
        }
    }
    fscanf(params->freq, "%f", &tempFloat);
    if(! feof(params->freq)) {
        printf("Error: More frequency values present than fits frames\n\n");
        return(FAILURE);
    }
    
    /* Compute \lambda^2 from the list of generated frequencies */
    params->lambda2  = calloc(params->qAxisLen3, sizeof(params->lambda2));
    if(params->lambda2 == NULL) {
        printf("Error: Mem alloc failed while reading in frequency list\n\n");
        return(FAILURE);
    }
    params->lambda20 = 0.0;
    for(i=0; i<params->qAxisLen3; i++)
        params->lambda2[i] = (LIGHTSPEED / params->freqList[i]) * 
                             (LIGHTSPEED / params->freqList[i]);
    
    return(SUCCESS);
}
