/******************************************************************************
rmsf.h
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
#ifndef RMSF_H
#define RMSF_H

#ifdef __cplusplus
extern "C"
#endif

int generateRMSF(struct optionsList *inOptions, struct parList *params);
int compFunc(const void * a, const void * b);
void getMedianLambda20(struct parList *params);
int writeRMSF(struct optionsList inOptions, struct parList params);
int plotRMSF(struct optionsList inOptions);

#endif
