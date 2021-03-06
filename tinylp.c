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
#define TLP_NONZERO(a) (fabs(a) > (pInfo->fZero))

// read source mx
#define LEMX(_r, _c ) ( pLEMX[ (_r) * (iVariables +1) + (_c) ] )
#define EQMX(_r, _c ) ( pEQMX[ (_r) * (iVariables +1) + (_c) ] )
#define GEMX(_r, _c ) ( pGEMX[ (_r) * (iVariables +1) + (_c) ] )

// access simplex tableau
#define TBMX(_r, _c ) ( pInfo->pMatrix[ (_r) * pInfo->iCols + (_c) ] )

// quadratic terms of obj fnx
#define QMX(_r,_c) (pOBJMX[iVariables + ((_r) * iVariables + (_c))]) // Q starts after linear terms

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

///////////////
// oscillation detector hash alg

// https://cessu.blogspot.com/2008/11/hashing-with-sse2-revisited-or-my-hash.html
// kudos: Cessu
static inline uint64_t tlp__hash_64x32(uint64_t h, uint32_t v)
{
  h ^= 2857720171ULL * v;
  h ^= h >> 29;
  h += h << 16;
  h ^= h >> 21;
  h += h << 32;
  return h;
}

static inline uint32_t tlp__hash_32x16(uint32_t h, uint16_t v)
{
  h ^= 43691UL * v;
  h ^= h >> 13;
  h += h << 8;
  h ^= h >> 10;
  h += h << 16;
  return h;
}

///////////////////////////////////////

void tlp_dump_mx( double* pMX, TLP_UINT rr, TLP_UINT cc )
{
  for(int r = 0; r < rr; r++) {
    for(int c = 0; c < cc; c++) {
      printf(" %+10.4f", pMX[r * cc + c]);
    }
    putchar('\n');
  }
}

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

void tlp_dump_active_cols( struct MXInfo* pInfo )
{
  for(TLP_UINT v = 0; v < pInfo->iActivevars; v++ )
    printf("%d ", pInfo->pActive[ v ]);
  putchar('\n');
}

void tlp_dump_current_soln( struct MXInfo* pInfo )
{
  TLP_UINT rhscol = pInfo->iCols - 1;
  TLP_UINT v, r;

  if( pInfo->bMaximize ) {
    for(v = 0; v < pInfo->iDefiningvars; v++ ) {
      r = pInfo->pActive[ v ] +1; // +1 skips row M
      if(r < pInfo->iRows)
        printf("%+10.4f", TBMX( r, rhscol ) );
      else
        printf("%+10.4f", NAN );
    }
  } else {
    for(v = 0; v < pInfo->iConstraints; v++ ) {
      if(pInfo->pActive[ v ] < rhscol) // review
        printf("%+10.4f", TBMX( +1, rhscol - pInfo->pActive[ v ] ) ); // +1 skips row M, -ve for RtoL
      else
        printf("%+10.4f", NAN );
    }
  }
  printf(" = %+10.4f %s\n", TBMX(1, rhscol ), tlp_is_augmented( pInfo ) ? "(AUGMENTED)" : "");
}

void tlp_dump_active_soln( struct MXInfo* pInfo )
{
  TLP_UINT rhscol = pInfo->iCols - 1;
  TLP_UINT v, r;

  if( pInfo->bMaximize ) {
    for(v = 0; v <pInfo->iActivevars; v++ ) {
      r = pInfo->pActive[ v ] +1; // +1 skips row M
      if(r < pInfo->iRows)
        printf("%+10.4f", TBMX( r, rhscol ) );
      else
        printf("%+10.4f", NAN );
    }
  } else {
    for(v = 0; v <pInfo->iActivevars; v++ ) {
      if(pInfo->pActive[ v ] < rhscol) // review
        printf("%+10.4f", TBMX( +1, rhscol - pInfo->pActive[ v ] ) ); // +1 skips row M, -ve for RtoL
      else
        printf("%+10.4f", NAN );
    }
  }
  printf(" = %+10.4f %s\n", TBMX(1, rhscol ), tlp_is_augmented( pInfo ) ? "(AUGMENTED)" : "");
}

void tlp_dump_osc_history( struct MXInfo *pInfo )
{
  for(int i=0; i<TLP_OSC_HISTORY; i++)
  {
    int j = ( pInfo->iIter + (TLP_OSC_HISTORY -i) ) % TLP_OSC_HISTORY;
    printf(
      "%6d  %08lX  %+10.4f\n",
      pInfo->iIter - i, pInfo->osc.pivHashArr[ j ], pInfo->osc.rhsArr[ j ]
    );
  }
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

    // trivially exclude variables that are already active
    for(TLP_UINT i=0; i<pInfo->iActivevars; i++ )
      if( (pInfo->pActive[i] -1) == var ) // -1 converts col to var
        goto skip_variable_and_check_another; // already in active set

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
// very horrorific nested looping in here
// review: reimplement using multiscan buffer or LINQish-streaming technique?
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
//  for(TLP_UINT i = 0; i<pInfo->iActive; i++ ) printf("%d ", pInfo->pActiveVariables[i]);
//  putchar('\n');

  // due to square mx, pInfo->iConstraints count the number of complemntary variables
  TLP_UINT n = pInfo->iConstraints;

  TLP_UINT c;
  for( c = c1; c < c2; c++ )
  {
    TLP_UINT var = c -1; // -1 converts from col to var

    // trivially exclude variables that are already active
    for(TLP_UINT i = 0; i<pInfo->iActivevars; i++ )
      if( (pInfo->pActive[i] -1) == var ) // -1 converts col to var
        goto skip_variable_and_check_another; // already in active set

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
      TLP_UINT a = var, b = pInfo->pActive[i] -1; // -1 converts col to var
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
  if( !pInfo->pActive ) return tlp_rc_encode(TLP_ASSERT);
#endif

  //tlp_dump_active_cols(pInfo);

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
  pInfo->cEnters = pivCol;
  pInfo->cLeaves = pInfo->pActive[ pivRow -2 ]; // -2 skips M and Z
  pInfo->pActive[ pivRow -2 ] = pivCol; // -2 skips M and Z

  // normalize row r1 by the pivot r1,c1
  if( (rc = tlp_rowdiv( pInfo, pivRow, pivCol )) ) return rc;

  // reduce mx by v * v1 for all rows except the pivot row
  if( (rc = tlp_mxsubmul( pInfo, pivRow, pivCol )) ) return rc;

  // todo: write a routine that will calculate sum( fabs( infeasibilities ) )
  // so that the caller can decide to terminate early if value <= epsilon.

  // oscillation detection requires first requires hashing and storing the pivot-state
  int oscSlot = pInfo->iIter % TLP_OSC_HISTORY;
  uint64_t pivHash = 0;
  pivHash = tlp__hash_64x32(pivHash, (uint32_t)pInfo->cEnters);
  pivHash = tlp__hash_64x32(pivHash, (uint32_t)pInfo->cLeaves);
  for(int i=0; i<pInfo->iActivevars; i++ )
    pivHash = tlp__hash_64x32(pivHash, (uint32_t)pInfo->pActive[ i ]);
  pInfo->osc.pivHashArr[ oscSlot ] = pivHash;
  pInfo->osc.rhsArr[ oscSlot ] = TBMX(1, rhsCol );

  // oscillation detection then requires state-scanning
  for(int i=0; i<TLP_OSC_HISTORY; i++)
    if( i != oscSlot )
      if( pInfo->osc.pivHashArr[ oscSlot ] == pInfo->osc.pivHashArr[ i ] )
        if( TLP_EQUAL( pInfo->osc.rhsArr[ oscSlot ], pInfo->osc.rhsArr[ i ] ) )
          return tlp_rc_encode(TLP_OSCILLATION);

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

  pInfo->iActivevars = pInfo->iConstraints;
  pInfo->iRows = 1 + 1 + pInfo->iConstraints; // rows M, Z and constraints
  pInfo->iCols = 1 + pInfo->iDefiningvars + pInfo->iSlackvars + 1; // Z, vars, slacks, RHS

  pInfo->fMax = DBL_MAX;
  pInfo->fMaxNeg = -pInfo->fMax;
  pInfo->fMin = DBL_MIN;
  pInfo->fMinNeg = -pInfo->fMin;
  pInfo->fZero = 1.0e-10;
  pInfo->szVars = szVars;
  pInfo->iIter = -1;
  pInfo->cEnters = TLP_BADINDEX;
  pInfo->cLeaves = TLP_BADINDEX;

  size_t uBytes;

  uBytes = sizeof(uint64_t) * TLP_OSC_HISTORY;
  pInfo->osc.pivHashArr = (uint64_t*)aligned_alloc( 16, uBytes );
  pInfo->osc.rhsArr = (double*)aligned_alloc( 16, uBytes );
  memset(pInfo->osc.pivHashArr, 0, uBytes);
  memset(pInfo->osc.rhsArr, 0, uBytes);

  uBytes = sizeof(double) * pInfo->iRows * pInfo->iCols;
  double *pMXData = aligned_alloc(16, uBytes);
  memset(pMXData, 0, uBytes);
  pInfo->pMatrix = pMXData;

  uBytes = sizeof(TLP_UINT) * pInfo->iActivevars;
  pInfo->pActive = (TLP_UINT *) malloc(uBytes);
  memset(pInfo->pActive, 0, uBytes);

  TLP_UINT c, r, v;
  TLP_UINT rhsCol = pInfo->iCols - 1; // -1 for RHS col
  TLP_UINT zCol = 1, mRow = 0;

  // put obj fxn into the tableau (the Z row)
  TBMX(+1, 0) = 1.0; // +1 to skip row M
  for (v = 0; v < iVariables; v++) TBMX(+1, v +1) = -1 * pOBJMX[ v ]; // +1 to skip row M, +1 to skip col Z, -1 for ?

  TLP_UINT slackCol = iVariables + 1; // the defining variables will not be the initial basic variables, +1 to skip col Z
  TLP_UINT constrRow = +2; // +2 to skip rows M and Z

//tlp_dump_tableau( pInfo, TLP_BADINDEX, TLP_BADINDEX );
//tlp_dump_active_cols(pInfo);

  // add LE rows
  for (r = 0; r < iLEConstraints; r++) {
    // coefs, slack var and rhs
    for (c = 0; c < iVariables; c++) TBMX(constrRow, zCol + c) = LEMX(r, c);
    TBMX(constrRow, slackCol) = 1.0;
    TBMX(constrRow, rhsCol) = LEMX(r, iVariables);
    pInfo->pActive[ constrRow -2 ] = slackCol; // -2 skips M and Z offsets to make base-0 index

    slackCol++;
    constrRow++;
//tlp_dump_tableau( pInfo, TLP_BADINDEX, TLP_BADINDEX );
//tlp_dump_active_cols(pInfo);
  }

  // add EQ rows
  for (r = 0; r < iEQConstraints; r++) {
    // coefs, slack var and rhs
    for (c = 0; c < iVariables; c++) TBMX(constrRow, zCol + c) = EQMX(r, c);
    TBMX(constrRow, slackCol) = 1.0;
    TBMX(constrRow, rhsCol) = EQMX(r, iVariables);
    pInfo->pActive[ constrRow -2 ] = slackCol; // -2 skips M and Z offsets to make base-0 index

    // add an artifical variable with M = 1 for each of the EQ constraints to the objective function
    TBMX(mRow, slackCol) = 1.0;
    if( (rc = tlp_rowsubmul(pInfo, mRow, slackCol, constrRow)) ) return tlp_rc_encode(TLP_INVALID); // was break

    slackCol++;
    constrRow++;
//tlp_dump_tableau( pInfo, TLP_BADINDEX, TLP_BADINDEX );
//tlp_dump_active_cols(pInfo);
  }

  return tlp_rc_encode(TLP_OK);
}

TLP_RCCODE
tlp_setup_max_qlp(
  struct MXInfo* pInfo,
  const double* pOBJMX, TLP_UINT iVariables,
  const double* pLEMX, TLP_UINT iLEConstraints,
  const char** szVars
)
{
  // QLP maximization by maximizing the -F' KKT minimization problem [H&L ed7 p687]
  // max -F' == z1 + z2
  // st Q1 == -2*Q[1,1]*x1 -2*Q[1,2]*x2 + G[1]*u1 - y1 = 15
  // st Q2 == -2*Q[2,1]*x1 -2*Q[2,2]*x2 + G[2]*u1 - y2 = 30
  // st G' == 1*x1 + 2*x2 + v1 = 30
  // st variable selection ensures x1y1 + x2y2 + u1v1 = 0
  //
  // rows = M + Z + #variables + #constraints
  // columns = Z + #quadratics + #lagranges + #slacks + #artificals + rhs
  //  Z       x1      x2       u1   y1  y2  v1  z1  z2  rhs
  //  0       0       0         0    0   0  0   0   0     0, M
  // -1       0       0         0    0   0  0   1   1     0, Z, -1 constructs -F' problem
  //  0  -2*Q[1,1] -2*Q[1,2]  G[1]  -1   0  0   1   0   rhs, Q1
  //  0  -2*Q[2,1] -2*Q[2,2]  G[2]   0  -1  0   0   1   rhs, Q2
  //  0     G[1]     G[2]                   1           rhs, G`
  // where:
  // y1, y2 are (negated) slacks that convert Q1 & Q2 from KKT <= into LP =
  // z1, z2 are artificals to minimize F'
  // u1, v1 are created from constaint G1 (u is lagrange coefs and v is a slack)
  // p688 shows mx differences... two phase method? p144
  TLP_RCCODE rc;

  if( !pInfo ) return tlp_rc_encode(TLP_ASSERT);
  if( !pOBJMX || (iVariables<2) ) return tlp_rc_encode(TLP_ASSERT);
  if( iLEConstraints && !pLEMX ) return tlp_rc_encode(TLP_ASSERT);

  pInfo->bMaximize = 1;
  pInfo->bQuadratic = 1;

  pInfo->iDefiningvars = iVariables;
  pInfo->iConstraints = iVariables + iLEConstraints;
  pInfo->iSlackvars = 2 * iVariables + iLEConstraints; // y's and z's for variables and v's for constr
  pInfo->iActivevars = iLEConstraints + iVariables; // constraints-slacks + z-slacks: v1 z1 z2

  pInfo->iRows = 1 + 1 + pInfo->iConstraints; // rows M, Z and constraints
  pInfo->iCols = +1 /*Z*/ + 3*iVariables /*every variable needs x, y, z*/ + 2*iLEConstraints /*every constr needs u, v*/ +1 /*rhs*/ ;

  pInfo->fMax = DBL_MAX;
  pInfo->fMaxNeg = -pInfo->fMax;
  pInfo->fMin = DBL_MIN;
  pInfo->fMinNeg = -pInfo->fMin;
  pInfo->fZero = 1.0e-10;
  pInfo->szVars = szVars;
  pInfo->iIter = -1;
  pInfo->cEnters = TLP_BADINDEX;
  pInfo->cLeaves = TLP_BADINDEX;

  size_t uBytes;

  uBytes = sizeof(uint64_t) * TLP_OSC_HISTORY;
  pInfo->osc.pivHashArr = (uint64_t*)aligned_alloc( 16, uBytes );
  pInfo->osc.rhsArr = (double*)aligned_alloc( 16, uBytes );
  memset(pInfo->osc.pivHashArr, 0, uBytes);
  memset(pInfo->osc.rhsArr, 0, uBytes);

  uBytes = sizeof(double) * pInfo->iRows * pInfo->iCols;
  double *pMXData = aligned_alloc(16, uBytes);
  memset(pMXData, 0, uBytes);
  pInfo->pMatrix = pMXData;

  uBytes = sizeof(TLP_UINT) * pInfo->iActivevars;
  pInfo->pActive = (TLP_UINT *) malloc(uBytes);
  memset(pInfo->pActive, 0, uBytes);

  int iDefiningconst = pInfo->iConstraints -pInfo->iDefiningvars; // -iDefiningVars because QP problem treats Q rows as constr
  int rowZ = 1;
  int rowQ = 2;
  int rowG = +1 +1 +pInfo->iDefiningvars; // +1 for rowM, +1 rowZ, +iDefiningvars for Q rows
  int colQ = 1;
  int colU = +1 + pInfo->iDefiningvars; // +1 skips colZ
  int colY = colU + iDefiningconst;
  int colV = colY + pInfo->iDefiningvars;
  int colZ = colV + iDefiningconst;
  int rhs = colZ + pInfo->iDefiningvars;

  // construct -F' problem
  TBMX(1,0) = -1;

  // add objective dependencies
  for(int n=0; n<pInfo->iDefiningvars; n++)
  {
    for(int i=0; i<pInfo->iDefiningvars; i++) TBMX(rowQ+n, colQ+i) = -2. * QMX(n, i); // quadratic coefs
    TBMX(rowQ+n, rhs) = pOBJMX[n]; // linear term to rhs
    TBMX(rowQ+n, colY+n) = -1.; // negated slack for KKT
    TBMX(rowZ,   colZ+n) = 1.; // z artificial for rowZ
    TBMX(rowQ+n, colZ+n) = 1.; // z artificial for rowQn
  }

  // add constraint dependencies
  for(int n=0; n<iDefiningconst; n++)
  {
    for(int i=0; i<pInfo->iDefiningvars; i++)
    {
      TBMX(rowQ+i, colU+n) = pLEMX[i]; // colUn constraint terms
      TBMX(rowG+n, colQ+i) = pLEMX[i]; // rowGn constraint term
    }
    TBMX(rowG+n, colV+n) = 1.; // v slack for Gn'
    TBMX(rowG+n, rhs) = pLEMX[pInfo->iDefiningvars]; // constraint rhs
  }

  //tlp_dump_tableau( pInfo, 0, 0 );

  // factor the z slacks
  for(int n=0; n<pInfo->iDefiningvars; n++)
  {
    rc = tlp_rowsubmul(pInfo, rowZ, colZ+n, rowQ+n); // factor zn from Qn
    //printf("%d %s\n",__LINE__,tlp_messages[tlp_rc_decode(rc)]);
  }

  // set slacks as first active variables
  for(int i=0; i<pInfo->iActivevars; i++)
  {
    pInfo->pActive[i] = colV+i; // slacks are all colV and all colZ
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

  pInfo->iActivevars = pInfo->iConstraints;
  pInfo->iRows = 1 + 1 + pInfo->iConstraints; // rows M, Z and constraints
  pInfo->iCols = 1 + pInfo->iDefiningvars + pInfo->iSlackvars + 1; // Z, vars, slacks, RHS

  pInfo->fMax = DBL_MAX;
  pInfo->fMaxNeg = -pInfo->fMax;
  pInfo->fMin = DBL_MIN;
  pInfo->fMinNeg = -pInfo->fMin;
  pInfo->fZero = 1.0e-10;
  pInfo->szVars = szVars;
  pInfo->iIter = -1;
  pInfo->cEnters = TLP_BADINDEX;
  pInfo->cLeaves = TLP_BADINDEX;

  size_t uBytes;

  uBytes = sizeof(uint64_t) * TLP_OSC_HISTORY;
  pInfo->osc.pivHashArr = (uint64_t*)aligned_alloc( 16, uBytes );
  pInfo->osc.rhsArr = (double*)aligned_alloc( 16, uBytes );
  memset(pInfo->osc.pivHashArr, 0, uBytes);
  memset(pInfo->osc.rhsArr, 0, uBytes);

  uBytes = sizeof(double) * pInfo->iRows * pInfo->iCols;
  double *pMXData = aligned_alloc(16, uBytes);
  memset(pMXData, 0, uBytes);
  pInfo->pMatrix = pMXData;

  uBytes = sizeof(TLP_UINT) * pInfo->iActivevars;
  pInfo->pActive = (TLP_UINT *) malloc(uBytes);
  memset(pInfo->pActive, 0, uBytes);

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
    pInfo->pActive[ constrRow -2 ] = slackCol; // -2 skips M and Z offsets to make base-0 index

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
    pInfo->pActive[ constrRow -2 ] = slackCol; // -2 skips M and Z offsets to make base-0 index

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
  if( !pInfo ) return tlp_rc_encode(TLP_ASSERT);
  if( !pInfo->pActive ) return tlp_rc_encode(TLP_ASSERT);

  TLP_UINT rhscol = pInfo->iCols - 1;
  TLP_UINT v;

  if( pInfo->bMaximize ) {
    for(v = 0; v < pInfo->iDefiningvars; v++ )
      pSOLMX[ v ] = TBMX( pInfo->pActive[ v ] +1, rhscol ); // +1 skips row M
  } else {
    for(v = 0; v < pInfo->iConstraints; v++ )
      pSOLMX[ v ] = TBMX( +1, rhscol - pInfo->pActive[ v ] ); // +1 skips row M, -ve for RtL
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
  if( pInfo->pActive )
    free( pInfo->pActive ); // tokens
  if( pInfo->pMatrix )
    free( pInfo->pMatrix ); // doubles
  if( pInfo->osc.pivHashArr )
    free( pInfo->osc.pivHashArr );
  if( pInfo->osc.rhsArr )
    free( pInfo->osc.rhsArr );

  memset( pInfo, 0, sizeof(struct MXInfo) );

  return tlp_rc_encode(TLP_OK);
}

int tlp_is_augmented( struct MXInfo* pInfo )
{
  if( !pInfo ) return tlp_rc_encode(TLP_ASSERT);
  if( !pInfo->pActive ) return tlp_rc_encode(TLP_ASSERT);

  TLP_UINT rhscol = pInfo->iCols - 1;

//printf("[ ");
//for(int i=0; i<pInfo->iActivevars; i++) printf("AC %d ", pInfo->pActive[i]);
//printf("]");

  // review
  // check if any of the active variables are non-zero slacks
  for(TLP_UINT i=0; i<pInfo->iActivevars; i++) {
    if(pInfo->bMaximize) {
      if( pInfo->pActive[ i ] -1 < pInfo->iDefiningvars ) continue; // -1 convs from col to idx
      if( pInfo->pActive[ i ] +1 >= pInfo->iRows ) return 1; // +1 skips row M
      if( TLP_NONZERO( TBMX( pInfo->pActive[ i ] +1, rhscol ) ) ) return 1; // +1 skips row M
    } else {
      if( pInfo->pActive[ i ] -1 < pInfo->iConstraints ) continue; // -1 convs from col to idx
      if( pInfo->pActive[ i ] >= pInfo->iCols ) return 1;
      if( TLP_NONZERO( TBMX( +1, rhscol - pInfo->pActive[ i ] ) ) ) return 1; // +1 skips row M, -ve for RtL
    }
  }

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
