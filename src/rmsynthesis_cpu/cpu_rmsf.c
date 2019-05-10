/******************************************************************************
cpu_rmsf.c
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

#include "cpu_fileaccess.h"
#include<omp.h>
#include<math.h>

/*************************************************************
*
* Generate Rotation Measure Spread Function
*
*************************************************************/
int generateRMSF(struct optionsList *inOptions, struct parList *params) {
    int i, j;
    float doubleStartPhi;
    
    params->rmsf     = calloc(inOptions->nPhi, sizeof(params->rmsf));
    params->rmsfReal = calloc(inOptions->nPhi, sizeof(params->rmsfReal));
    params->rmsfImag = calloc(inOptions->nPhi, sizeof(params->rmsfImag));
    params->phiAxis  = calloc(inOptions->nPhi, sizeof(params->phiAxis));
    
    if(params->rmsf     == NULL || params->rmsfReal == NULL ||
       params->rmsfImag == NULL || params->phiAxis  == NULL)
        return 1;
    
    /* Get the normalization factor K */
    params->K = 1.0 / params->qAxisLen3;
    
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
        params->rmsfReal[i] *= params->K;
        params->rmsfImag[i] *= params->K;
        params->rmsf[i] = sqrt( params->rmsfReal[i] * params->rmsfReal[i] +
                                params->rmsfImag[i] * params->rmsfImag[i] );
    }
    
    if(inOptions->doRMClean) {
      /* TODO: Create rmsf in the double mode 
         Define new variables params->rmsfDouble, params->resfRealDouble, 
         params->rmsfImagDouble, and params->phiAxisDouble */
      params->phiAxisDouble = calloc(2*inOptions->nPhi, 
                                   sizeof(params->phiAxisDouble));
      params->rmsfRealDouble = calloc(2*inOptions->nPhi, 
                                   sizeof(params->rmsfRealDouble));
      params->rmsfImagDouble = calloc(2*inOptions->nPhi,
                                   sizeof(params->rmsfImagDouble));
      params->rmsfDouble     = calloc(2*inOptions->nPhi,
                                   sizeof(params->rmsfDouble));
      if((params->phiAxisDouble  == NULL) ||
         (params->rmsfRealDouble == NULL) ||
         (params->rmsfImagDouble == NULL) ||
         (params->rmsfDouble     == NULL)) { return 1; }
      doubleStartPhi = inOptions->phiMin - (inOptions->nPhi * inOptions->dPhi)/2;
      for(i=0; i<2*inOptions->nPhi; i++) {         
         params->phiAxisDouble[i] = doubleStartPhi + i * inOptions->dPhi;
         /* For each phi value, compute the corresponding RMSF */
         for(j=0; j<params->qAxisLen3; j++) {
            params->rmsfRealDouble[i] += cos(2*params->phiAxisDouble[i] * 
                                       (params->lambda2[j] - params->lambda20));
            params->rmsfImagDouble[i] -= sin(2*params->phiAxisDouble[i] *
                                       (params->lambda2[j] - params->lambda20));
         }
         // Normalize with K
         params->rmsfRealDouble[i] *= params->K;
         params->rmsfImagDouble[i] *= params->K;
         params->rmsfDouble[i] = sqrt(
                         params->rmsfRealDouble[i] * params->rmsfRealDouble[i] +
                         params->rmsfImagDouble[i] * params->rmsfImagDouble[i]);
      }
    
      /* TODO: Fit a Gaussian to rmsfReal, rmsfImage, and rmsf 
          Define new variables params->rmsfDoubleFit, 
          params->rmsfRealDoubleFit, and params->rmsfImagDoubleFit */
    
    }
    return 0;
}

/*************************************************************
*
* Write RMSF to disk
*
*************************************************************/
int writeRMSF(struct optionsList inOptions, struct parList params) {
    FILE *rmsf;
    char filename[256];
    int i;
    
    /* Open a text file */
    sprintf(filename, "%srmsf.txt", inOptions.outPrefix);
    printf("INFO: Writing RMSF to %s\n", filename);
    rmsf = fopen(filename, "w");
    if(rmsf == NULL)
        return 1;
    
    for(i=0; i<inOptions.nPhi; i++)
        fprintf(rmsf, "%f\t%f\t%f\t%f\n", params.phiAxis[i], params.rmsfReal[i],
                params.rmsfImag[i], params.rmsf[i]);
    
    fclose(rmsf);
    return 0;
}
