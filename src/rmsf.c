#include "structures.h"
#include<math.h>

#include "rmsf.h"

/*************************************************************
*
* Generate Rotation Measure Spread Function
*
*************************************************************/
int generateRMSF(struct optionsList *inOptions, struct parList *params) {
    int i, j;
    double K;
    
    params->rmsf     = calloc(inOptions->nPhi, sizeof(params->rmsf));
    params->rmsfReal = calloc(inOptions->nPhi, sizeof(params->rmsfReal));
    params->rmsfImag = calloc(inOptions->nPhi, sizeof(params->rmsfImag));
    params->phiAxis  = calloc(inOptions->nPhi, sizeof(params->phiAxis));
    
    if(params->rmsf     == NULL || params->rmsfReal == NULL ||
       params->rmsfImag == NULL || params->phiAxis  == NULL)
        return(FAILURE);
    
    /* Get the normalization factor K */
    K = 1.0 / params->qAxisLen3;
    
    /* First generate the phi axis */
    for(i=0; i<inOptions->nPhi; i++) {
        params->phiAxis[i] = inOptions->phiMin + i * inOptions->dPhi;
        
        /* For each phi value, compute the corresponding RMSF */
        for(j=0; j<params->qAxisLen3; j++) {
            params->rmsfReal[i] += cos(2 * params->phiAxis[i] *
                                   (params->lambda2[j] - params->lambda20 ));
            params->rmsfImag[i] -= sin(2 * params->phiAxis[i] *
                                   (params->lambda2[j] - params->lambda20 ));
        }
        // Normalize with K
        params->rmsfReal[i] *= K;
        params->rmsfImag[i] *= K;
        params->rmsf[i] = sqrt( params->rmsfReal[i] * params->rmsfReal[i] +
                                params->rmsfImag[i] * params->rmsfImag[i] );
    }
    return(SUCCESS);
}

/*************************************************************
*
* Comparison function used by quick sort.
*
*************************************************************/
int compFunc(const void * a, const void * b) {
   return ( *(double*)a - *(double*)b );
}

/*************************************************************
*
* Find the median \lambda^2_0
*
*************************************************************/
void getMedianLambda20(struct parList *params) {
    double *tempArray;
    int i;
    
    tempArray = calloc(params->qAxisLen3, sizeof(tempArray));
    for(i=0; i<params->qAxisLen3; i++)
        tempArray[i] = params->lambda2[i];
    
    /* Sort the list of lambda2 freq */
    qsort(tempArray, params->qAxisLen3, sizeof(tempArray), compFunc);
    
    /* Find the median value of the sorted list */
    params->lambda20 = tempArray[params->qAxisLen3/2];
}

/*************************************************************
*
* Write RMSF to disk
*
*************************************************************/
int writeRMSF(struct optionsList inOptions, struct parList params) {
    FILE *rmsf;
    char filename[FILENAME_LEN];
    int i;
    
    /* Open a text file */
    sprintf(filename, "%srmsf.txt", inOptions.outPrefix);
    printf("\nINFO: Writing RMSF to %s", filename);
    rmsf = fopen(filename, FILE_READWRITE);
    if(rmsf == NULL)
        return(FAILURE);
    
    for(i=0; i<inOptions.nPhi; i++)
        fprintf(rmsf, "%f\t%f\t%f\t%f\n", params.phiAxis[i], params.rmsfReal[i],
                params.rmsfImag[i], params.rmsf[i]);
    
    fclose(rmsf);
    return(SUCCESS);
}

#ifdef GNUPLOT_ENABLE
/*************************************************************
*
* Plot RMSF
*
*************************************************************/
int plotRMSF(struct optionsList inOptions) {
    FILE *gnuplotPipe;
    char commands[STRING_BUF_LEN];
    
    gnuplotPipe = popen("gnuplot -persist", FILE_READWRITE);
    if(gnuplotPipe == NULL)
        return(FAILURE);
    
    /* Plot the RMSF using the file that was written in writeRMSF() */
    sprintf(commands, "set title \"Rotation Measure Spread Function\"\n");
    sprintf(commands, "%sset xlabel \"Faraday Depth\"\n", commands);
    sprintf(commands, "%sset autoscale\n", commands);
    sprintf(commands, "%splot \"%srmsf.txt\" using 1:2 title 'RMSF' with lines,", 
            commands, inOptions.outPrefix);
    sprintf(commands, "%s \"%srmsf.txt\" using 1:3 title 'Real' with lines,", 
            commands, inOptions.outPrefix);
    sprintf(commands, "%s \"%srmsf.txt\" using 1:4 title 'Imag' with lines\n", 
            commands, inOptions.outPrefix);
    fprintf(gnuplotPipe, "%s", commands);
    pclose(gnuplotPipe);
    
    return(SUCCESS);
}
#endif
