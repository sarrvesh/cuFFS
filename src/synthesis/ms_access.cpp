#include<casacore/ms/MeasurementSets/MeasurementSet.h>
#include<casacore/tables/Tables/ScalarColumn.h>
#include<casacore/tables/Tables/ArrayColumn.h>
#include<casacore/ms/MeasurementSets/MSFieldColumns.h>
#include<casacore/casa/Quanta/MeasValue.h>
#include<stdio.h>
#include<string>
#include<assert.h>

#include "ms_access.h"
#include "constants.h"

using namespace casacore;

int getUvRange(struct optionsList inOptions, struct structHeader *msHeader) {
   double uInM, vInM, wInM;
   
   MeasurementSet ms(inOptions.msName, Table::Old);
   /* Read the UVW column */
   ROArrayColumn<double> uvwColumn(ms, MS::columnName(MSMainEnums::UVW));
   
   /* Find the max and mins */
   msHeader->uMaxM   = 0; msHeader->uMinM   = 0;
   msHeader->vMaxM   = 0; msHeader->vMinM   = 0;
   msHeader->wMaxM   = 0; msHeader->wMinM   = 0;
   msHeader->minUVWm = 0; msHeader->maxUVWm = 0;
   for(size_t thisRow=0; thisRow!=msHeader->nVisRows; ++thisRow) {
      Vector<double> uvw = uvwColumn(thisRow);
      uInM = uvw(0); vInM = uvw(1); wInM = uvw(2);
      if(msHeader->uMaxM < uInM) {msHeader->uMaxM = uInM;}
      if(msHeader->uMinM > uInM) {msHeader->uMinM = uInM;}
      
      if(msHeader->vMaxM < vInM) {msHeader->vMaxM = vInM;}
      if(msHeader->vMinM > vInM) {msHeader->vMinM = vInM;}
      
      if(msHeader->wMaxM < wInM) {msHeader->wMaxM = wInM;}
      if(msHeader->wMinM > wInM) {msHeader->wMinM = wInM;}
      
      double uvwDist = sqrt(uInM*uInM + vInM*vInM + wInM*wInM);
      if(msHeader->minUVWm > uvwDist) { msHeader->minUVWm = uvwDist; }
      if(msHeader->maxUVWm < uvwDist) { msHeader->maxUVWm = uvwDist; }
   }
   /*printf("U = %f to %f\n", msHeader->uMinM, msHeader->uMaxM);
   printf("V = %f to %f\n", msHeader->vMinM, msHeader->vMaxM);
   printf("W = %f to %f\n", msHeader->wMinM, msHeader->wMaxM);*/
   return(SUCCESS);
}

int getMsHeader(struct optionsList inOptions, struct structHeader *msHeader) {
   try {
      MeasurementSet ms(inOptions.msName, Table::Old);
      msHeader->nVisRows = ms.nrow();
      if(msHeader->nVisRows == 0) {
         throw std::runtime_error("ERROR: MS has no data\n");
      }
      
      MSSpectralWindow spwTable = ms.spectralWindow();
      msHeader->nSPW = spwTable.nrow();
   
      /* Ensure that we are not working with more than one SPW */
      if(msHeader->nSPW != 1) {
         throw std::runtime_error("ERROR: Cannot handle MS with SPW>1\n");
      }
   
      /* Find channels per SPW */
      ROScalarColumn<int> numChanCol(spwTable, 
         MSSpectralWindow::columnName(MSSpectralWindowEnums::NUM_CHAN));
      msHeader->chanPerSPW = numChanCol.get(0);
      
      /* Find the frequency value for each SPW */
      ArrayColumn<double> chanFreqCol(spwTable, 
         MSSpectralWindow::columnName(MSSpectralWindowEnums::CHAN_FREQ));
      Array<double> chanFreqArr = chanFreqCol.getColumn();
      double *chanFreq = chanFreqArr.data();
      /* Copy chan freq to msHeader */
      msHeader->chanFreq = (double*)calloc(msHeader->chanPerSPW, sizeof(double));
      for(int i=0; i<msHeader->chanPerSPW; ++i) {
         msHeader->chanFreq[i] = chanFreq[i];
      }
      
      /* Find channel width */
      ArrayColumn<double> bwCol(spwTable, 
         MSSpectralWindow::columnName(MSSpectralWindowEnums::CHAN_WIDTH));
      Array<double> bwArr = bwCol.getColumn();
      msHeader->chanWidth = bwArr.data()[0];
      
      /* Find the coordinate of the phase center in J2000 */
      ROMSFieldColumns fieldCol(ms.field());
      if(fieldCol.nrow() != 1) {
         throw std::runtime_error("Error: MS has multiple fields\n");
      }
      MDirection refDir = fieldCol.phaseDirMeas(0);
      if(refDir.getRefString().compare("J2000") != 0) {
         /* Quit if not in J2000 */
         throw std::runtime_error("Error: MS not in J2000\n");
      }
      Quantum<Vector<double> > refCoord = refDir.getAngle();
      msHeader->pointRa = refCoord.getValue()[0];
      msHeader->pointDec= refCoord.getValue()[1];
      msHeader->coordStr = refDir.toString();
      
      /* Check what type of feeds were used */
      MSPolarization polCol = ms.polarization();
      ArrayColumn<int> polArr(polCol, "CORR_TYPE");
      Array<int> polInfo = polArr.getColumn();
      if((polInfo.data()[0] == 9) && (polInfo.data()[1] == 10) &&
         (polInfo.data()[2] == 11)&& (polInfo.data()[3] == 12)) {
         msHeader->corrType = LINEAR;      
      }
      else {
         throw std::runtime_error("ERROR: Unrecognized correlation type\n");
      }
   }
   catch(AipsError) {
      printf("ERROR: Unable to read %s\n\n", inOptions.msName);
      return(FAILURE);
   }
   return(SUCCESS);
}
