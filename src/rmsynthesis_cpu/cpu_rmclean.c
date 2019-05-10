/******************************************************************************
cpu_rmclean.h
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

/*************************************************************
*
* Copy all pixels in cube[] to los[] corresponding to the 
*   line of sight specified by rowIdx and colIdx
*
*************************************************************/
void getLineOfSight(float los[], float cube[], int rowIdx, 
                    int colIdx, int nPhi, int rowLen, int colLen) {
   int outIdx;
   int offset = rowIdx*rowLen + colIdx;
   for(int i=0; i<nPhi; i++) {
      outIdx = i*rowLen*colLen + offset;
      los[i] = cube[outIdx];
   }
}

/*************************************************************
*
* Find the maximum value and its index in a specified array
*
*************************************************************/
void findMaxP(float array[], int nPhi, float *maxP, int *idxMaxP) {
   for(int i=0; i<nPhi; i++) {
      if(i == 0) {
         *maxP = array[i];
         *idxMaxP = i;
      }
      else {
         if(array[i] > *maxP) {
            *maxP = array[i];
            *idxMaxP = i;
         }
      }
   }
}
