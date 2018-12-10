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
#ifndef CPU_RMCLEAN_H
#define CPU_RMCLEAN_H

void getLineOfSight(float los[], float cube[], int rowIdx, 
                    int colIdx, int nPhi, int rowLen, int colLen);
void findMaxP(float array[], int nPhi, float *maxP, int *idxMaxP);
#endif
