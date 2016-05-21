#include "fitsio.h"
#include "structures.h"
#include "constants.h"
#include "fileaccess.h"

#define IM_TYPE FLOAT_IMG

/*************************************************************
*
* Check Fitsio error and exit if required.
*
*************************************************************/
void checkFitsError(int status) {
    if(status) {
        printf("\nERROR:");
        fits_report_error(stdout, status);
        printf("\n");
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
    
    /* Get the image dimensions from the Q cube */
    fits_read_key(params->qFile, TINT, "NAXIS", &params->qAxisNum, 
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS1", &params->qAxisLen1, 
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS2", &params->qAxisLen2, 
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS3", &params->qAxisLen3, 
      fitsComment, &fitsStatus);
    /* Get the image dimensions from the Q cube */
    fits_read_key(params->uFile, TINT, "NAXIS", &params->uAxisNum, 
      fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS1", &params->uAxisLen1, 
      fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS2", &params->uAxisLen2, 
      fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS3", &params->uAxisLen3,
      fitsComment, &fitsStatus);
    
    return(fitsStatus);
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
        printf("\nError: Mem alloc failed while reading in frequency list\n\n");
        return(FAILURE);
    }
    for(i=0; i<params->qAxisLen3; i++) {
        fscanf(params->freq, "%f", &params->freqList[i]);
        if(feof(params->freq)) {
            printf("\nError: Frequency values and fits frames don't match\n");
            return(FAILURE);
        }
    }
    fscanf(params->freq, "%f", &tempFloat);
    if(! feof(params->freq)) {
        printf("\nError: More frequency values present than fits frames\n\n");
        return(FAILURE);
    }
    
    /* Compute \lambda^2 from the list of generated frequencies */
    params->lambda2  = calloc(params->qAxisLen3, sizeof(params->lambda2));
    if(params->lambda2 == NULL) {
        printf("\nError: Mem alloc failed while reading in frequency list\n\n");
        return(FAILURE);
    }
    params->lambda20 = 0.0;
    for(i=0; i<params->qAxisLen3; i++)
        params->lambda2[i] = (LIGHTSPEED / params->freqList[i]) * 
                             (LIGHTSPEED / params->freqList[i]);
    
    return(SUCCESS);
}

/*************************************************************
*
* Read in the input mask image
*
*************************************************************/
int getImageMask(struct optionsList *inOptions, struct parList *params) {
    int fitsStatus = SUCCESS;
    char fitsComment[FLEN_COMMENT];

    fits_read_key(params->maskFile, TINT, "NAXIS1", &params->maskAxisLen1, 
                  fitsComment, &fitsStatus);
    fits_read_key(params->maskFile, TINT, "NAXIS2", &params->maskAxisLen2, 
                  fitsComment, &fitsStatus);
    return(SUCCESS);
}

/*************************************************************
*
* Write the output Q, U or P cubes to disk
*
*************************************************************/
int writePolCubeToDisk(float *fitsCube, char *fileName, 
                       struct optionsList *inOptions, struct parList *params) {
    fitsfile *ptr;
    int status = SUCCESS;
    long naxis[FITS_OUT_NAXIS];
    char fitsComment[FLEN_COMMENT];
    char card[FLEN_CARD];
    char filenamefull[FILENAME_LEN];
    
    /* Open the output fitsfile */
    sprintf(filenamefull, "%s%s", inOptions->outPrefix, fileName);
    fits_create_file(&ptr, filenamefull, &status);
    /* Set the axes lengths */
    naxis[RA_AXIS] = params->uAxisLen1;
    naxis[DEC_AXIS] = params->qAxisLen2;
    naxis[PHI_AXIS] = inOptions->nPhi;
    fits_create_img(ptr, IM_TYPE, FITS_OUT_NAXIS, naxis, &status);
    /* Write appropriate keywords to fits header */
    fits_write_key(ptr, TDOUBLE, "CRVAL3", &inOptions->phiMin, fitsComment, 
                   &status);
    fits_write_key(ptr, TDOUBLE, "CDELT3", &inOptions->dPhi, fitsComment,
                   &status);
    /* Close the created file */
    fits_close_file(ptr, &status);
    checkFitsError(status);
}
