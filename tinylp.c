// linear program solver (simplex method), orthopteroid@gmail.com, BSD2 license
// based on "Operations Research", Hiller & Lieberman (1974)

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <limits.h>
#include <assert.h>

#include "tinylp.h"

#define TLP_EQUAL(a,b) (fabs((a)-(b)) < (pInfo->fZero))
#define TLP_ALMOSTZERO(a) (fabs(a) < (pInfo->fZero))

// read source mx
#define LEMX(_r, _c ) ( pLEMX[ (_r) * (iVariables +1) + (_c) ] )
#define EQMX(_r, _c ) ( pEQMX[ (_r) * (iVariables +1) + (_c) ] )
#define GEMX(_r, _c ) ( pGEMX[ (_r) * (iVariables +1) + (_c) ] )

// access simplex tableau
#define TBMX(_r, _c ) ( pInfo->pMatrix[ (_r) * pInfo->iCols + (_c) ] )

///////////////////////////////////////

#ifndef NDEBUG

TLP_RCCODE tlp_rc_encode(TLP_UINT e) { return (TLP_RCCODE)e<<(2*TLP_RCBITS_IDX); }
TLP_UINT tlp_rc_decode(TLP_RCCODE rc) { return (TLP_UINT)(rc>>(2*TLP_RCBITS_IDX)); }
TLP_RCCODE tlp_rc_encode_info(TLP_UINT e, TLP_UINT r, TLP_UINT c)
{
  return (TLP_RCCODE)e<<(2*TLP_RCBITS_IDX) | (TLP_RCCODE)r<<TLP_RCBITS_IDX | (TLP_RCCODE)c;
}
void tlp_rc_decode_info(TLP_RCCODE rc, TLP_UINT *e, TLP_UINT *r, TLP_UINT *c)
{
  *e = ((1<<TLP_RCBITS_STAT)-1) & (TLP_UINT)(rc>>(2*TLP_RCBITS_IDX));
  *r = TLP_BADINDEX & (TLP_UINT)(rc>>TLP_RCBITS_IDX);
  *c = TLP_BADINDEX & (TLP_UINT)(rc);
}

#define TLP_INLINE

#else

#define TLP_INLINE inline

#endif // DEBUG

///////////////////////////////////////

void tlp_dump_tableau( struct MXInfo* pInfo, TLP_UINT r1, TLP_UINT c1 )
{
  TLP_UINT r, c;
  for (r = 0; r < pInfo->iRows; r++) {
    for (c = 0; c < pInfo->iCols; c++) {
      char d1 =
        ((c1 == c) && (r1 == r)) ||                         /* hilight cell */
        ((c1 == c) && (r1 == TLP_BADINDEX)) ||              /* hilight column */
        ((r1 == r) && (c1 == TLP_BADINDEX) && (c == 0))     /* hilight 1st element of row */
        ? '[' : ' ';
      char d2 =
        ((c1 == c) && (r1 == r)) ||                         /* hilight cell */
        ((c1 == c) && (r1 == TLP_BADINDEX)) ||              /* hilight column */
        ((r1 == r) && (c1 == TLP_BADINDEX) && (c == (pInfo->iCols - 1)))     /* hilight last element of row */
        ? ']' : ' ';
      printf("%1c%+10.4f%1c", d1, pInfo->pMatrix[r * pInfo->iCols + c], d2);
    }
    putchar('\n');
  }
}

void tlp_dump_config( struct MXInfo* pInfo )
{
  printf(
    "%d, %d %d, %d %d %d\n",
    pInfo->bMaximize, pInfo->iCols, pInfo->iRows,
    pInfo->iConstraints, pInfo->iDefiningvars, pInfo->iSlackvars
  );
}

void tlp_dump_active_vars( struct MXInfo* pInfo )
{
  for(TLP_UINT v = 0; v < pInfo->iVars; v++ )
    printf("%d ", pInfo->pActiveVariables[ v ]);
  putchar('\n');
}

void tlp_dump_current_soln( struct MXInfo* pInfo )
{
  if( tlp_is_augmented( pInfo ) ) { printf("AUGMENTED\n"); return; }

  TLP_UINT rhscol = pInfo->iCols - 1;
  TLP_UINT  v;

  if( pInfo->bMaximize ) {
    for(v = 0; v < pInfo->iDefiningvars; v++ )
      printf("%+10.4f", TBMX( pInfo->pActiveVariables[ v ] +2, rhscol ) ); // +2 skips rows M and Z
  } else {
    for(v = 0; v < pInfo->iConstraints; v++ ) // transpose requires iConstraints
      printf("%+10.4f", TBMX( +1, rhscol -1 -pInfo->pActiveVariables[ v ] ) ); // +1 skips row Z, -1 skips col Z, -ve for RtL
  }
  printf(" = %+10.4f\n", TBMX(1, rhscol ));
}

void tlp_dump_active_soln( struct MXInfo* pInfo )
{
  TLP_UINT rhscol = pInfo->iCols - 1;
  TLP_UINT  v;

  if( pInfo->bMaximize ) {
    for(v = 0; v < pInfo->iVars; v++ )
      printf("%+10.4f", TBMX( pInfo->pActiveVariables[ v ] +2, rhscol ) ); // +2 skips rows M and Z
  } else {
    for(v = 0; v < pInfo->iVars; v++ ) // transpose
      printf("%+10.4f", TBMX( +1, rhscol -1 -pInfo->pActiveVariables[ v ] ) ); // +1 skips row Z, -1 skips col Z, -ve for RtL
  }
  printf(" = %+10.4f %s\n", TBMX(1, rhscol ), tlp_is_augmented( pInfo ) ? "(AUGMENTED)" : "");
}

///////////////////////////////////////

// divide row r1 by the coef of r1,c1
static TLP_INLINE TLP_RCCODE
tlp_rowdiv(
  struct MXInfo* pInfo, TLP_UINT r1, TLP_UINT c1
)
{
#ifndef NDEBUG
  if( r1 >= TLP_BADINDEX || c1 >= TLP_BADINDEX ) return tlp_rc_encode(TLP_ASSERT);
  if( c1 >= pInfo->iCols || r1 >= pInfo->iRows ) return tlp_rc_encode(TLP_GEOMETRY);
#endif

  // divide row r1 by the coef of r1,c1
  double v = pInfo->pMatrix[ r1 * pInfo->iCols + c1 ];
  if( (v < pInfo->fMin) && (v > pInfo->fMinNeg) ) return tlp_rc_encode(TLP_ZERO);

  TLP_UINT c;
  for( c = 0; c < pInfo->iCols; c++ )
  {
    double v1 = pInfo->pMatrix[ r1 * pInfo->iCols + c ];
    pInfo->pMatrix[ r1 * pInfo->iCols + c ] = v1 / v;
  }

  return tlp_rc_encode(TLP_OK);
}

// divide row r1 by the coef of r1,c1 and subtract r2 from r1
 TLP_RCCODE
tlp_rowdivsub(
  struct MXInfo* pInfo, TLP_UINT r1, TLP_UINT c1, TLP_UINT r2
)
{
  TLP_UINT c;

#ifndef NDEBUG
  if( r1 >= TLP_BADINDEX || r2 >= TLP_BADINDEX || c1 >= TLP_BADINDEX ) return tlp_rc_encode(TLP_ASSERT);
  if( c1 >= pInfo->iCols || r1 >= pInfo->iRows || r2 >= pInfo->iRows ) return tlp_rc_encode(TLP_GEOMETRY);
#endif

  // divide row r1 by the coef of r1,c1 and subtract r2 from r1
  double v = pInfo->pMatrix[ r1 * pInfo->iCols + c1 ];
  if( (v < pInfo->fMin) && (v > pInfo->fMinNeg) ) return tlp_rc_encode_info(TLP_ZERO, r1, c1);
  if( (v == DBL_MIN) || (v == DBL_MAX) ) return tlp_rc_encode_info(TLP_INFINITY, r1, c1);

  // pre-check for zero-division
  for( c = 0; c < pInfo->iCols; c++ )
  {
    double v2 = pInfo->pMatrix[ r2 * pInfo->iCols + c ];
    if( TLP_ALMOSTZERO( v - v2 ) ) return tlp_rc_encode_info(TLP_INFINITY, r2, c);
  }

  for( c = 0; c < pInfo->iCols; c++ )
  {
    double v1 = pInfo->pMatrix[ r1 * pInfo->iCols + c ];
    double v2 = pInfo->pMatrix[ r2 * pInfo->iCols + c ];
    pInfo->pMatrix[ r1 * pInfo->iCols + c ] = v1 / v - v2;
  }

  return tlp_rc_encode(TLP_OK);
}

// from row r1, subtract the product of row r2 and the coef of r1,c1
 TLP_RCCODE
tlp_rowsubmul(
  struct MXInfo* pInfo, TLP_UINT r1, TLP_UINT c1, TLP_UINT r2
)
{
#ifndef NDEBUG
  if( r1 >= TLP_BADINDEX || r2 >= TLP_BADINDEX || c1 >= TLP_BADINDEX ) return tlp_rc_encode(TLP_ASSERT);
  if( c1 >= pInfo->iCols || r1 >= pInfo->iRows || r2 >= pInfo->iRows ) return tlp_rc_encode(TLP_GEOMETRY);
#endif

  // from row r1, subtract the product of row r2 and the coef of r1,c1
  double v = pInfo->pMatrix[ r1 * pInfo->iCols + c1 ];
  if( (v == DBL_MIN) || (v == DBL_MAX) ) return tlp_rc_encode_info(TLP_INFINITY, r1, c1);

  TLP_UINT c;
  for( c = +0; c < pInfo->iCols; c++ )
    pInfo->pMatrix[ r1 * pInfo->iCols + c ] -= v * pInfo->pMatrix[ r2 * pInfo->iCols + c ];

  return tlp_rc_encode(TLP_OK);
}

// for all rows r except for pivot row r1, subtract the product of row r and the coef of r,c1
static TLP_INLINE TLP_RCCODE
tlp_mxsubmul(
  struct MXInfo* pInfo, TLP_UINT r1, TLP_UINT c1
)
{
  TLP_UINT r, c;

#ifndef NDEBUG
  if( r1 >= TLP_BADINDEX || c1 >= TLP_BADINDEX ) return tlp_rc_encode(TLP_ASSERT);
  if( c1 >= pInfo->iCols || r1 >= pInfo->iRows ) return tlp_rc_encode(TLP_GEOMETRY);
#endif

  for( r = +0; r < pInfo->iRows; r++ )
  {
    if( r == r1 ) continue; // skip pivot row
    for( c = +0; c < pInfo->iCols; c++ )
    {
      double v = pInfo->pMatrix[ r * pInfo->iCols + c ];
      if( (v == DBL_MIN) || (v == DBL_MAX) ) return tlp_rc_encode_info(TLP_INFINITY, r, c);
    }
  }

  for( r = +0; r < pInfo->iRows; r++ )
  {
    if( r == r1 ) continue; // skip pivot row
    double v = pInfo->pMatrix[ r * pInfo->iCols + c1 ];
    for( c = +0; c < pInfo->iCols; c++ )
      pInfo->pMatrix[ r * pInfo->iCols + c ] -= v * pInfo->pMatrix[ r1 * pInfo->iCols + c ];
  }

  return tlp_rc_encode(TLP_OK);
}

// starting at row r1, find the row r2 forwhich coef r2,c1 / coef r2,c2 is smallest and positive, -1 if there is none.
static TLP_INLINE TLP_RCCODE
tlp_colSmallestPosRatio(
    struct MXInfo* pInfo,
    TLP_UINT r1, TLP_UINT c1, TLP_UINT c2,
    TLP_UINT* pR2, double* pRatio
)
{
#ifndef NDEBUG
  if( !pR2 || !pRatio ) return tlp_rc_encode(TLP_ASSERT);
  if( r1 >= TLP_BADINDEX || c1 >= TLP_BADINDEX || c2 >= TLP_BADINDEX ) return tlp_rc_encode(TLP_ASSERT);
  if( c1 >= pInfo->iCols || c2 >= pInfo->iCols || r1 >= pInfo->iRows ) return tlp_rc_encode(TLP_GEOMETRY);
#endif

  // compare the ratios of the pivotcol and the RHS for all rows
  // the row with the smallest ratio is the leaving variable

  *pRatio = DBL_MAX;
  *pR2 = TLP_BADINDEX;

  TLP_UINT r;
  for( r = r1; r < pInfo->iRows; r++ )
  {
    double v1 = pInfo->pMatrix[ r * pInfo->iCols + c1 ];
    if( TLP_ALMOSTZERO( v1 ) ) continue; // pre-handle DIV errors
    double v2 = pInfo->pMatrix[ r * pInfo->iCols + c2 ];
    double ratio = v2 / v1;
    if( ratio < 0. ) continue; // skip -ve numbers
    if( ratio < *pRatio )
    {
      *pRatio = ratio;
      *pR2 = r;
    }
  }

  return tlp_rc_encode(((*pR2 == TLP_BADINDEX) ? TLP_INVALID : TLP_OK));
}

// check specified row for most -ve coef
static TLP_INLINE TLP_RCCODE
tlp_rowLargestNegCoef(
  struct MXInfo* pInfo,
  TLP_UINT c1, TLP_UINT c2, TLP_UINT r,
  TLP_UINT* pC, double* pV
)
{
#ifndef NDEBUG
  if( !pInfo ) return tlp_rc_encode(TLP_ASSERT);
  if( !pInfo->pMatrix ) return tlp_rc_encode(TLP_ASSERT);
  if( !pC || !pV ) return tlp_rc_encode(TLP_ASSERT);
  if( c1 >= TLP_BADINDEX || c2 >= TLP_BADINDEX ) return tlp_rc_encode(TLP_ASSERT);
  if( c1 >= pInfo->iCols || c2 >= pInfo->iCols || r >= pInfo->iRows ) return tlp_rc_encode(TLP_GEOMETRY);
#endif

  TLP_UINT c;
  for( c = c1; c < c2; c++ )
  {
    TLP_UINT var = c -1; // -1 converts from col to var

/*    // trivially exclude variables that are already active, review: broken!
    for(TLP_UINT i=0; i<pInfo->iVars; i++ )
      if( (pInfo->pActiveVariables[i] == var) )
        goto skip_variable_and_check_another; */ // already in active set

    double v = pInfo->pMatrix[ r * pInfo->iCols + c];
    if( (v >= *pV) ) goto skip_variable_and_check_another; // not smaller

    *pV = v;
    *pC = c;

skip_variable_and_check_another: ;
  }

  return tlp_rc_encode(TLP_OK);
}

// check specified row for most -ve coef
// per H&L's restricted entry rule p687 s13.7 7th ed. for QP
// very horrorific nested looping in here...
static TLP_INLINE TLP_RCCODE
tlp_rowLargestNegCoef_QPRule(
  struct MXInfo* pInfo,
  TLP_UINT c1, TLP_UINT c2, TLP_UINT r,
  TLP_UINT* pC, double* pV
)
{
#ifndef NDEBUG
  if( !pInfo ) return tlp_rc_encode(TLP_ASSERT);
  if( !pInfo->pMatrix ) return tlp_rc_encode(TLP_ASSERT);
  if( !pC || !pV ) return tlp_rc_encode(TLP_ASSERT);
  if( c1 >= TLP_BADINDEX || c2 >= TLP_BADINDEX ) return tlp_rc_encode(TLP_ASSERT);
  if( c1 >= pInfo->iCols || c2 >= pInfo->iCols || r >= pInfo->iRows ) return tlp_rc_encode(TLP_GEOMETRY);
#endif

//  printf("active variables: ");
//  for(TLP_UINT i = 0; i<pInfo->iVars; i++ ) printf("%d ", pInfo->pActiveVariables[i]);
//  putchar('\n');

  // due to square mx, pInfo->iConstraints count the number of complemntary variables
  TLP_UINT n = pInfo->iConstraints;

  TLP_UINT c;
  for( c = c1; c < c2; c++ )
  {
    TLP_UINT var = c -1; // -1 converts from col to var

/*    // trivially exclude variables that are already active
    for(TLP_UINT i = 0; i<pInfo->iVars; i++ )
      if( (pInfo->pActiveVariables[i] == var) )
        goto skip_variable_and_check_another; */ // already in active set

    double v = pInfo->pMatrix[ r * pInfo->iCols + c];
    if( (v >= *pV) ) goto skip_variable_and_check_another; // not smaller

    // variables are numbered into 3 groups: 0..group1..n..group2..m..group3..k, m=2n
    // group1 and group2 are complementary and have restricted entry
    // group3 are the aux vars from the obj function and are unrestricted
    // avoid consideration of complementary variables for comparison and selection

    // trivially, if var is group 3 we can track it
    if( var > 2 * n ) goto track_variable_and_check_another;

    // since var is group1 or group2, determine if it's complement is already active
    for(TLP_UINT i=0; i<n; i++ )
    {
      TLP_UINT a = var, b = pInfo->pActiveVariables[i];
      if( b > a )
      {
        if( b < 2 * n ) continue; // max not in group2, check another active variable
        if( (2 * a) == b ) goto skip_variable_and_check_another; // aha, min and max are complementary variables. exclude variable var.
      } else {
        if( a < 2 * n ) continue; // max not in group2, check another active variable
        if( (2 * b) == a ) goto skip_variable_and_check_another; // aha, min and max are complementary variables. exclude variable var.
      }
    }

track_variable_and_check_another:
    *pV = v;
    *pC = c;
    continue;
skip_variable_and_check_another: ;
//    printf("skip variable: %d\n", var);
  }
//  printf("selected variable: %d\n", *pC -1); // -1 converts col to var

  return tlp_rc_encode(TLP_OK);
}

TLP_RCCODE
tlp_pivot(
  struct MXInfo* pInfo
)
{
  TLP_RCCODE rc;

#ifndef NDEBUG
  if( !pInfo ) return tlp_rc_encode(TLP_ASSERT);
  if( !pInfo->pActiveVariables ) return tlp_rc_encode(TLP_ASSERT);
#endif

  TLP_UINT rhsCol = pInfo->iCols - 1;
  TLP_UINT mRow = 0, zRow = 1;

  pInfo->iIter++;

  // check both rows M and Z for the best pivot col (ie the smallest -ve value)
  TLP_UINT pivCol = TLP_BADINDEX;
  double pivVal = 0.0; // to look for -ve pivot values, start with an invalid value

  // +1 to skip col z
  if( pInfo->bQuadratic ) {
    // per H&L's restricted entry rule p687 s13.7 7th ed.
    if( (rc = tlp_rowLargestNegCoef_QPRule(pInfo, +1, rhsCol, mRow, &pivCol, &pivVal)) ) return rc;
    if( (rc = tlp_rowLargestNegCoef_QPRule(pInfo, +1, rhsCol, zRow, &pivCol, &pivVal)) ) return rc;
  } else {
    if( (rc = tlp_rowLargestNegCoef(pInfo, +1, rhsCol, mRow, &pivCol, &pivVal)) ) return rc;
    if( (rc = tlp_rowLargestNegCoef(pInfo, +1, rhsCol, zRow, &pivCol, &pivVal)) ) return rc;
  }

  // stopping condition is when obj fxn has no more -ve coefs
  if( pivCol == TLP_BADINDEX ) return tlp_rc_encode(TLP_OK);

  // find the pivot row
  TLP_UINT pivRow = TLP_BADINDEX;

  // +2 to skip rows M and Z
  if( (rc = tlp_colSmallestPosRatio(pInfo, +2, pivCol, rhsCol, &pivRow, &pivVal)) ) return rc;

  // no +ve coef in pivot column?
  if( pivRow == TLP_BADINDEX ) return tlp_rc_encode(TLP_UNBOUNDED);

  // record entering/leaving variables
  TLP_UINT rVarLeaves = pivRow -2; // -2 skips rows M and Z
  pInfo->iVarEnters = pivCol -1; // -1 skips col Z
  pInfo->iVarLeaves = pInfo->pActiveVariables[ rVarLeaves ];
  pInfo->pActiveVariables[ rVarLeaves ] = pInfo->iVarEnters;

  // normalize row r1 by the pivot r1,c1
  if( (rc = tlp_rowdiv( pInfo, pivRow, pivCol )) ) return rc;

  // reduce mx by v * v1 for all rows except the pivot row
  if( (rc = tlp_mxsubmul( pInfo, pivRow, pivCol )) ) return rc;

  // todo: write a routine that will calculate sum( fabs( infeasibilities ) )
  // so that the caller can decide to terminate early if value <= epsilon.

  return tlp_rc_encode(TLP_UNFINISHED);
}

///////////////////////

TLP_RCCODE
tlp_setup_max(
  struct MXInfo* pInfo,
  const double* pOBJMX, TLP_UINT iVariables,
  const double* pLEMX, TLP_UINT iLEConstraints,
  const double* pEQMX, TLP_UINT iEQConstraints,
  const char** szVars
)
{
  TLP_RCCODE rc;

  if( !pInfo ) return tlp_rc_encode(TLP_ASSERT);
  if( !pOBJMX || (iVariables<2) ) return tlp_rc_encode(TLP_ASSERT);
  if( iLEConstraints && !pLEMX ) return tlp_rc_encode(TLP_ASSERT);
  if( iEQConstraints && !pEQMX ) return tlp_rc_encode(TLP_ASSERT);

  pInfo->bMaximize = 1;
  pInfo->bQuadratic = 0;
  pInfo->iConstraints = iLEConstraints + iEQConstraints;
  pInfo->iDefiningvars = iVariables;
  pInfo->iSlackvars = iLEConstraints + iEQConstraints;

  pInfo->iRows = 1 + 1 + pInfo->iConstraints; // rows M, Z and constraints
  pInfo->iCols = 1 + pInfo->iDefiningvars + pInfo->iSlackvars + 1; // Z, vars, slacks, RHS
  pInfo->iVars = pInfo->iDefiningvars + pInfo->iSlackvars;

  pInfo->fMax = DBL_MAX;
  pInfo->fMaxNeg = -pInfo->fMax;
  pInfo->fMin = DBL_MIN;
  pInfo->fMinNeg = -pInfo->fMin;
  pInfo->fZero = 1.0e-10;
  pInfo->szVars = szVars;
  pInfo->iIter = -1;
  pInfo->iVarEnters = TLP_BADINDEX;
  pInfo->iVarLeaves = TLP_BADINDEX;

  unsigned long int iBytes;

  iBytes = sizeof(double) * pInfo->iRows * pInfo->iCols;
  double *pMXData = aligned_alloc(16, iBytes);
  memset(pMXData, 0, iBytes);
  pInfo->pMatrix = pMXData;

  iBytes = sizeof(TLP_UINT) * pInfo->iVars;
  pInfo->pActiveVariables = (TLP_UINT *) malloc(iBytes);
  memset(pInfo->pActiveVariables, 0, iBytes);

  TLP_UINT c, r, v;
  TLP_UINT rhsCol = pInfo->iCols - 1; // -1 for RHS col
  TLP_UINT zCol = 1, mRow = 0;

  // put obj fxn into the tableau (the Z row)
  TBMX(+1, 0) = 1.0; // +1 to skip row M
  for (v = 0; v < iVariables; v++) TBMX(+1, v +1) = -1 * pOBJMX[ v ]; // +1 to skip row M, +1 to skip col Z, -1 for ?

  TLP_UINT slackCol = iVariables + 1; // the defining variables will not be the initial basic variables, +1 to skip col Z
  TLP_UINT constrRow = +2; // +2 to skip rows M and Z

  // add LE rows
  for (r = 0; r < iLEConstraints; r++) {
    // coefs, slack var and rhs
    for (c = 0; c < iVariables; c++) TBMX(constrRow, zCol + c) = LEMX(r, c);
    TBMX(constrRow, slackCol) = 1.0;
    TBMX(constrRow, rhsCol) = LEMX(r, iVariables);
    pInfo->pActiveVariables[ constrRow -2 ] = slackCol -1; // -2 skips rows M and Z, -1 skips col Z

    slackCol++;
    constrRow++;
  }

  // add EQ rows
  for (r = 0; r < iEQConstraints; r++) {
    // coefs, slack var and rhs
    for (c = 0; c < iVariables; c++) TBMX(constrRow, zCol + c) = EQMX(r, c);
    TBMX(constrRow, slackCol) = 1.0;
    TBMX(constrRow, rhsCol) = EQMX(r, iVariables);
    pInfo->pActiveVariables[ constrRow -2 ] = slackCol -1; // -2 skips rows M and Z, -1 skips col Z

    // add an artifical variable with M = 1 for each of the EQ constraints to the objective function
    TBMX(mRow, slackCol) = 1.0;
    if( (rc = tlp_rowsubmul(pInfo, mRow, slackCol, constrRow)) ) return rc; // was break

    slackCol++;
    constrRow++;
  }

  return tlp_rc_encode(TLP_OK);
}

TLP_RCCODE
tlp_setup_min(
  struct MXInfo* pInfo,
  const double* pOBJMX, TLP_UINT iVariables,
  const double* pGEMX, TLP_UINT iGEConstraints,
  const double* pEQMX, TLP_UINT iEQConstraints,
  const char** szVars
)
{
  TLP_RCCODE rc;

  if( !pInfo ) return tlp_rc_encode(TLP_ASSERT);
  if( !pOBJMX || (iVariables<2) ) return tlp_rc_encode(TLP_ASSERT);
  if( iGEConstraints && !pGEMX ) return tlp_rc_encode(TLP_ASSERT);
  if( iEQConstraints && !pEQMX ) return tlp_rc_encode(TLP_ASSERT);

  // dual uses transpose, which switches constraints with variables and rhs with constraint consts

  pInfo->bMaximize = 0;
  pInfo->bQuadratic = 0;
  pInfo->iConstraints = iVariables;
  pInfo->iDefiningvars = iGEConstraints + iEQConstraints;
  pInfo->iSlackvars = iVariables;

  pInfo->iRows = 1 + 1 + pInfo->iConstraints; // rows M, Z and constraints
  pInfo->iCols = 1 + pInfo->iDefiningvars + pInfo->iSlackvars + 1; // Z, vars, slacks, RHS
  pInfo->iVars = pInfo->iConstraints;

  pInfo->fMax = DBL_MAX;
  pInfo->fMaxNeg = -pInfo->fMax;
  pInfo->fMin = DBL_MIN;
  pInfo->fMinNeg = -pInfo->fMin;
  pInfo->fZero = 1.0e-10;
  pInfo->szVars = szVars;
  pInfo->iIter = -1;
  pInfo->iVarEnters = TLP_BADINDEX;
  pInfo->iVarLeaves = TLP_BADINDEX;

  unsigned long int iBytes;

  iBytes = sizeof(double) * pInfo->iRows * pInfo->iCols;
  double *pMXData = aligned_alloc(16, iBytes);
  memset(pMXData, 0, iBytes);
  pInfo->pMatrix = pMXData;

  iBytes = sizeof(TLP_UINT) * pInfo->iVars;
  pInfo->pActiveVariables = (TLP_UINT *) malloc(iBytes);
  memset(pInfo->pActiveVariables, 0, iBytes);

  TLP_UINT c, r, i, j;
  TLP_UINT rhsCol = pInfo->iCols - 1; // -1 for RHS col
  TLP_UINT zCol = 1, mRow = 0;

  // put obj fxn into the tableau (the Z row)
  TBMX(+1, 0) = 1.0; // +1 to skip row M
  for (i = 0; i < iGEConstraints; i++) TBMX(+1, i +1) = -1 * GEMX(i, iVariables); // +1 to skip row M, +1 to skip col Z, -1 for minimize
  for (i = 0; i < iEQConstraints; i++) TBMX(+1, i +1) = -1 * EQMX(i, iVariables); // +1 to skip row M, +1 to skip col Z, -1 for minimize

  TLP_UINT slackCol = pInfo->iDefiningvars + 1; // the defining variables will not be the initial basic variables, +1 to skip col Z
  TLP_UINT constrRow = +2; // +2 to skip rows M and Z

  // add GE rows
  for (j = 0; j < iGEConstraints; j++)
  {
    c = zCol;
    // coefs, slack var and rhs
    for (i = 0; i < iGEConstraints; i++) TBMX(constrRow, c + i) = GEMX(i, j); // transposed read
    TBMX(constrRow, slackCol) = 1.0;
    TBMX(constrRow, rhsCol) = pOBJMX[ j ];
    pInfo->pActiveVariables[ constrRow -2 ] = slackCol -1; // -2 skips rows M and Z, -1 skips col Z

    slackCol++;
    constrRow++;
  }

  // add EQ rows
  for (j = 0; j < iEQConstraints; j++)
  {
    c = zCol + iGEConstraints; // start at next dest col
    // coefs, slack var and rhs
    for (i = 0; i < iEQConstraints; i++) TBMX(constrRow, c + i) = EQMX(i, j); // transposed read
    TBMX(constrRow, slackCol) = 1.0;
    TBMX(constrRow, rhsCol) = pOBJMX[ j ];
    pInfo->pActiveVariables[ constrRow -2 ] = slackCol -1; // -2 skips rows M and Z, -1 skips col Z

    // add an artifical variable with M = 1 for each of the EQ constraints to the objective function
    TBMX(mRow, slackCol) = 1.0;
    if( (rc = tlp_rowsubmul(pInfo, mRow, slackCol, constrRow)) ) return rc; // review: break instead?

    slackCol++;
    constrRow++;
  }

  return tlp_rc_encode(TLP_OK);
}

TLP_RCCODE
tlp_soln(
  struct MXInfo* pInfo,
  double* pSOLMX
)
{
  TLP_UINT v;

  if( !pInfo ) return tlp_rc_encode(TLP_ASSERT);
  if( !pInfo->pActiveVariables ) return tlp_rc_encode(TLP_ASSERT);

  TLP_UINT rhscol = pInfo->iCols - 1;

  if( pInfo->bMaximize ) {
    for(v = 0; v < pInfo->iDefiningvars; v++ )
      pSOLMX[ v ] = TBMX( pInfo->pActiveVariables[ v ] +2, rhscol ); // +2 skips rows M and Z
  } else {
    for(v = 0; v < pInfo->iConstraints; v++ ) // transpose requires iConstraints
      pSOLMX[ v ] = TBMX( +1, rhscol -1 -pInfo->pActiveVariables[ v ] ); // +1 skips row Z, -1 skips col Z, -ve for RtL
  }
  pSOLMX[ pInfo->iDefiningvars ] = TBMX(1, rhscol );

  if( tlp_is_augmented( pInfo ) ) return tlp_rc_encode(TLP_AUGMENTED);

  return tlp_rc_encode(TLP_OK);
}

TLP_RCCODE
tlp_fini(
  struct MXInfo* pInfo
)
{
  if( !pInfo ) return tlp_rc_encode(TLP_ASSERT);

  // cleanup tableau
  if( pInfo->pActiveVariables )
    free( pInfo->pActiveVariables ); // tokens
  if( pInfo->pMatrix )
    free( pInfo->pMatrix ); // doubles
  memset( pInfo, 0, sizeof(struct MXInfo) );

  return tlp_rc_encode(TLP_OK);
}

int tlp_is_augmented( struct MXInfo* pInfo )
{
  if( !pInfo ) return tlp_rc_encode(TLP_ASSERT);
  if( !pInfo->pActiveVariables ) return tlp_rc_encode(TLP_ASSERT);

  TLP_UINT n = pInfo->bMaximize ? pInfo->iDefiningvars : pInfo->iConstraints;

  // check if any of the active variables are slacks
  for(TLP_UINT i=0; i<pInfo->iVars; i++)
    if( pInfo->pActiveVariables[ i ] >= n ) return 1;

  return 0;
}

TLP_RCCODE
tlp_mxequal(
  const double* pA,
  const double* pB,
  const double fZero,
  TLP_UINT n
)
{
  int diff = 0;
  if( !pA || !pB ) return tlp_rc_encode(TLP_ASSERT);

  while( n-- ) {
    //printf("%+10.4f %+10.4f\n", pA[n], pB[n] );
    if( fabs( pA[n] - pB[n] ) > fZero ) diff = 1;
  }

  return diff ? tlp_rc_encode_info(TLP_INVALID, 0, 0) : tlp_rc_encode(TLP_OK);
}
