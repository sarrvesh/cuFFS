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

#define SUCCESS 0
#define FAILURE 1

#define FILENAME_LEN        256
#define STRING_BUF_LEN      256
#define DEFAULT_OUT_PREFIX  "output_"
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
#define SEC_PER_MIN 60.

#endif
