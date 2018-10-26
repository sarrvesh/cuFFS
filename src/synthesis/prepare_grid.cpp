#include "prepare_grid.h"
#include "ms_access.h"
#include<math.h>

#include<iostream>

/*************************************************************
*
* Compute the direction cosines corresponding to the coordinate
* grid of the output image.
*
*************************************************************/
void computeLMGrid(struct optionsList inOptions, 
                   struct structHeader msHeader, 
                   double lArray[], double mArray[]) {
   size_t nElements = inOptions.imsize * inOptions.imsize;
   double thisRa, thisDec;
   for(int i=0; i<nElements; ++i) {
      /* What are this cell's coordinates? */
      thisRa = msHeader.firstRa - msHeader.cdelt * (i%inOptions.imsize);
      thisDec= msHeader.firstDec - msHeader.cdelt * (i/inOptions.imsize);
      
      /* Compute l and n using the equation from Perley (1999) */
      lArray[i] = cos(thisDec) * sin(thisRa - msHeader.pointRa);
      mArray[i] = (sin(thisDec) * cos(msHeader.pointDec)) - 
                  (cos(thisDec) * sin(msHeader.pointDec) * 
                  cos(thisRa - msHeader.pointRa));
   }
}
