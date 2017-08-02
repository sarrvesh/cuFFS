/******************************************************************************
constants.h
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

#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifdef __cplusplus
extern "C"
#endif

#define NO_DEVICE 0

#define N_DIMS 3
#define X_DIM  0
#define Y_DIM  1
#define Z_DIM  2

#define NUM_INPUTS 2

#define ZERO 0

#define SUCCESS 0
#define FAILURE 1

#define FILENAME_LEN        256
#define STRING_BUF_LEN      256
#define DEFAULT_OUT_PREFIX  "output_"
#define Q_DIRTY             "q.phi.dirty"
#define U_DIRTY             "u.phi.dirty"
#define P_DIRTY             "p.phi.dirty"
#define SCREEN_WIDTH        40
#define FILE_READONLY       "r"
#define FILE_READWRITE      "w"
#define CTYPE_LEN           10

#define FITS_OUT_NAXIS 3
#define RA_AXIS        0
#define DEC_AXIS       1
#define PHI_AXIS       2

#define LIGHTSPEED 299792458.
#define KILO       1000.
#define MEGA       1000000.
#define SEC_PER_MIN 60
#define SEC_PER_HOUR 3600

#define FITS 0
#define HDF5 1

#define ROOT "/"
#define CLASS "CLASS"
#define PRIMARY "/PRIMARY"
#define PRIMARYDATA "/PRIMARY/DATA"
#define HDFITS "HDFITS"

#endif
