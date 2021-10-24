// linear program solver (simplex method), orthopteroid@gmail.com, BSD2 license

#ifndef _ANALYSIS_H_
#define _ANALYSIS_H_

#include <stdint.h>

#include "tinylp.h"

///////////////
// enter-leave-active-variable hash oscillation detector

struct ELAVHODInfo
{
  double rhstol;
  int iHistory;
  TLP_UINT *var_entropy; // use entropy instead of var index number
  uint64_t *history; // hash history of entering, leaving and active variables
  double *rhs; // rhsides
};

void elavhod_setup( struct MXInfo *pInfo, struct ELAVHODInfo *htlod, uint8_t history, double rhstol );
void elavhod_dump_history( struct MXInfo *pInfo, struct ELAVHODInfo *htlod );
TLP_RCCODE elavhod_check( struct MXInfo *pInfo, struct ELAVHODInfo *htlod );
void elavhod_fin( struct MXInfo *pInfo, struct ELAVHODInfo *htlod );

///////////////

int test_is_concave(double* vecX, double* mxCoef, TLP_UINT size);
void gauss(double* soln, double* mx, int n);

///////////////

double determinant( double* mx, TLP_UINT size );

#endif // _ANALYSIS_H_
