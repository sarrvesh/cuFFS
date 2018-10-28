#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdexcept>
#include<limits>
#include<libconfig.h>
#include<iostream>
#include<fitsio.h>
#include<math.h>

#include "synth_fileaccess.h"
#include "constants.h"
#include "ms_access.h"

#define outName "./sample.fits"
#define NAXIS 3
#define FLEN_COMMENTS 256

void checkFitsError(int fitsStatus) {
   if(fitsStatus) {
      fits_report_error(stdout, fitsStatus);
      throw std::runtime_error("\n");
   }
}

/*************************************************************
*
* Write the specified output image to disk
*
*************************************************************/
void writeOutputImages(struct optionsList inOptions,
                       struct structHeader msHeader,
                       float imageData[]) {
   fitsfile *out;
   int fitsStatus = 0;
   size_t nElements = inOptions.imsize * inOptions.imsize;
   long axisLen[NAXIS];
   long oPixel[NAXIS];
   char fitsComment[FLEN_COMMENT];
   char ctype1[] = "RA--SIN";
   char ctype2[] = "DEC--SIN";
   char ctype3[] = "FREQ";
   char unitDeg[] = "deg";
   char unitHz[] = "Hz";
   char unitJy[] = "JY/BEAM";
   char bType[] = "Intensity";
   double equinox = 2000.0;
   double cdelt1, cdelt2, cdelt3;
   int crpix1, crpix2, crpix3;
   
   imageData[1] = 4.;
   
   /* Create the output file */
   fits_create_file(&out, outName, &fitsStatus);
   axisLen[0] = inOptions.imsize;
   axisLen[1] = inOptions.imsize;
   axisLen[2] = 1;
   fits_create_img(out, FLOAT_IMG, NAXIS, axisLen, &fitsStatus);
   oPixel[0] = oPixel[1] = oPixel[2] = 1;
   
   /* Write the output pixels */
   fits_write_pix(out, TFLOAT, oPixel, nElements, imageData, &fitsStatus);
   
   /* Set the appropriate header information */
   fits_write_key(out, TSTRING, "BUNIT", unitJy, "", &fitsStatus);
   fits_write_key(out, TSTRING, "BTYPE", bType, "", &fitsStatus);
   
   fits_write_key(out, TSTRING, "CTYPE1", ctype1, "", &fitsStatus);
   fits_write_key(out, TDOUBLE, "CRVAL1", &(msHeader.pointRaDeg), 
                  "", &fitsStatus);
   cdelt1 = -msHeader.cdelt*(180.0/M_PI);
   fits_write_key(out, TDOUBLE, "CDELT1", &cdelt1, "", &fitsStatus);
   crpix1 = (inOptions.imsize/2)+1;
   fits_write_key(out, TINT, "CRPIX1", &crpix1, "", &fitsStatus);
   fits_write_key(out, TSTRING, "CUNIT1", unitDeg, "", &fitsStatus);
   
   fits_write_key(out, TSTRING, "CTYPE2", ctype2, "", &fitsStatus);
   fits_write_key(out, TDOUBLE, "CRVAL2", &(msHeader.pointDecDeg), 
                  "", &fitsStatus);
   cdelt2 = msHeader.cdelt*(180.0/M_PI);
   fits_write_key(out, TDOUBLE, "CDELT2", &cdelt2, "", &fitsStatus);
   crpix2 = (inOptions.imsize/2)+1;
   fits_write_key(out, TINT, "CRPIX2", &crpix2, "", &fitsStatus);
   fits_write_key(out, TSTRING, "CUNIT2", unitDeg, "", &fitsStatus);
   
   fits_write_key(out, TSTRING, "CTYPE3", ctype3, "Central Frequency",
                  &fitsStatus);
   fits_write_key(out, TDOUBLE, "CRVAL3", &(msHeader.chanFreq[0]), 
                  "", &fitsStatus);
   cdelt3 = msHeader.chanFreq[1] - msHeader.chanFreq[0];
   fits_write_key(out, TDOUBLE, "CDELT3", &cdelt3, "", &fitsStatus);
   crpix3 = 1;
   fits_write_key(out, TINT, "CRPIX3", &crpix3, "", &fitsStatus);
   fits_write_key(out, TSTRING, "CUNIT3", unitHz, "", &fitsStatus);
   fits_write_key(out, TDOUBLE, "EQUINOX", &equinox, "J2000.0", &fitsStatus);
   fits_write_key(out, TSTRING, "ORIGIN", (char*) MY_NAME, 
                  "Faraday synthesis written by Sarrvesh Sridhar",
                  &fitsStatus);
                  
   // TODO: Write the observation date to fits header
   
   fits_close_file(out, &fitsStatus);
   checkFitsError(fitsStatus);
}

/*************************************************************
*
* Print useful input/output information 
*
*************************************************************/
void printUserInfo(struct optionsList inOptions, 
                   struct structHeader msHeader) {
   printf("\n*************************\n\n");
   printf("Processing %s\n", inOptions.msName);
   printf("SPW: %ld\n", msHeader.nSPW);
   printf("Channels: %ld\n", msHeader.chanPerSPW);
   printf("Channel width:%0.4f kHz\n", msHeader.chanWidth/kHz);
   std::cout << "Phase center: " << msHeader.coordStr << std::endl;
   printf("UV range: %0.2f - %0.2f m\n", msHeader.minUVWm, msHeader.maxUVWm);
   printf("U range: %0.2f - %0.2f m\n", msHeader.uMinM, msHeader.uMaxM);
   printf("V range: %0.2f - %0.2f m\n", msHeader.vMinM, msHeader.vMaxM);
   printf("Output image size: %d x %d\n", inOptions.imsize, inOptions.imsize);
   printf("Pixel size: %0.1f x %0.1f\n", inOptions.cellsize, 
                                         inOptions.cellsize);
   printf("Compute mode: ");
   if(inOptions.hardware == CPU)
      printf("CPU\n");
   else
      printf("GPU\n");
      
   printf("Invert mode: ");
   if(inOptions.invertmode == DFT)
      printf("DFT\n");
   else
      printf("FFT\n");
      
   printf("\n*************************\n\n");
}

/*************************************************************
*
* Parse the input file and extract the relevant keywords
*
*************************************************************/
struct optionsList parseInput(char *parsetFileName) {
   config_t cfg;
   struct optionsList inOptions;
   const char *str;
   
   /* Initialize configuration */
   config_init(&cfg);
   
   /* Read the config file */
   if(!config_read_file(&cfg, parsetFileName)) {
      printf("Error: Error reading parset file. %s\n\n", 
               config_error_text(&cfg));
      config_destroy(&cfg);
      exit(FAILURE);
   }
   
   /* Get the name of the measurement set */
   if(config_lookup_string(&cfg, "ms", &str)) {
      inOptions.msName = (char *)malloc(strlen(str)+1);
      strcpy(inOptions.msName, str);
   }
   else {
      config_destroy(&cfg);
      throw std::runtime_error("ERROR: Unable to find MS name in the parset\n");
   }
   
   /* Get uvmin and uvmax values */
   static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
   if(! config_lookup_float(&cfg, "uvmin", &inOptions.uvmin)) {
      inOptions.uvmin = -std::numeric_limits<float>::infinity();
   }
   if(! config_lookup_float(&cfg, "uvmax", &inOptions.uvmax)) {
      inOptions.uvmax = std::numeric_limits<float>::infinity();
   }
   
   /* Get the image and cell size */
   if(! config_lookup_int(&cfg, "imsize", &inOptions.imsize)) {
      config_destroy(&cfg);
      throw std::runtime_error("ERROR: imsize not found\n");
   }
   if(! config_lookup_float(&cfg, "cellsize", &inOptions.cellsize)) {
      if(! config_lookup_float(&cfg, "cellsize", &inOptions.cellsize)) {
         config_destroy(&cfg);
         throw std::runtime_error("ERROR: cellsize not found\n");
      }
   }
   
   /* Find out which hardware to use */
   if(config_lookup_string(&cfg, "hardware", &str)) {
      if(0 == strcasecmp(str, "GPU")) {
         inOptions.hardware = GPU;
         throw std::runtime_error("GPU support is yet to be implemented\n");
      }
      else 
         inOptions.hardware = CPU;
   }
   else { inOptions.hardware = CPU; }
   
   /* Find out which inversion mode to use */
   if(config_lookup_string(&cfg, "invertmode", &str)) {
      if(0 == strcasecmp(str, "fft")) {
         inOptions.invertmode = FFT;
         throw std::runtime_error("FFT support is yet to be implemented\n");
      }
      else
         inOptions.invertmode = DFT;
   }
   
   /* Destroy and return the structure */
   config_destroy(&cfg);
   return(inOptions);
}
