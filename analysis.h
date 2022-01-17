// linear program solver (simplex method), orthopteroid@gmail.com, BSD2 license

#ifndef _ANALYSIS_H_
#define _ANALYSIS_H_

#include <stdint.h>

#include "tinylp.h"

///////////////

int test_is_concave(double* vecX, double* mxCoef, TLP_UINT size);
TLP_RCCODE gauss(double* soln, double* mx, int rows);

///////////////

double determinant( double* mx, TLP_UINT size );

///////////////

TLP_RCCODE
qeq_setup_min(
  double** ppMX_new,
  const double* pOBJMX, TLP_UINT iVariables,
  const double* pEQMX, TLP_UINT iEQConstraints
);
TLP_RCCODE qeq_fin( double** ppMX );


#endif // _ANALYSIS_H_
