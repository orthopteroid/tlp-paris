// linear program solver (simplex method), orthopteroid@gmail.com, BSD2 license

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <assert.h>

#include "analysis.h"

#define EQUAL(a,b,d) (fabs((a)-(b)) < (d))
#define ALMOSTZERO(a,d) (fabs(a) < (d))

#define TBMX(_r, _c ) ( pInfo->pMatrix[ (_r) * pInfo->iCols + (_c) ] )

///////////////
// elavh oscillation detector

// https://cessu.blogspot.com/2008/11/hashing-with-sse2-revisited-or-my-hash.html
// kudos: Cessu
static inline uint64_t elavhod__hash_64x32(uint64_t h, uint32_t v)
{
  h ^= 2857720171ULL * v;
  h ^= h >> 29;
  h += h << 16;
  h ^= h >> 21;
  h += h << 32;
  return h;
}

static inline uint32_t elavhod__hash_32x16(uint32_t h, uint16_t v)
{
  h ^= 43691UL * v;
  h ^= h >> 13;
  h += h << 8;
  h ^= h >> 10;
  h += h << 16;
  return h;
}

void elavhod_setup( struct MXInfo *pInfo, struct ELAVHODInfo *htlod, uint8_t history, double rhstol )
{
  memset(htlod, 0, sizeof(struct ELAVHODInfo));

  htlod->rhstol = rhstol;
  htlod->iHistory = history;

  htlod->history = (uint64_t*)malloc( sizeof(uint64_t) * htlod->iHistory );
  memset(htlod->history, 0, sizeof(uint64_t) * htlod->iHistory);

  htlod->rhs = (double*)malloc( sizeof(double) * htlod->iHistory );
  memset(htlod->rhs, 0, sizeof(double) * htlod->iHistory);

  // https://www.cyrill-gremaud.ch/howto-generate-secure-random-number-on-nix/
  // kudos: Bennet Yee
  int fd;

  if( (fd = open("/dev/urandom", O_RDONLY)) == -1 )
    perror("Error: impossible to read randomness source\n");

  size_t uVarBytes = sizeof(TLP_UINT) * pInfo->iCols;
  htlod->var_entropy = (TLP_UINT*)malloc( uVarBytes );
  if( read(fd, htlod->var_entropy, uVarBytes) != uVarBytes )
    perror("read() failed\n");

  close(fd);
}

void elavhod_dump_history( struct MXInfo *pInfo, struct ELAVHODInfo *htlod )
{
  for(int i=0; i<htlod->iHistory; i++)
  {
    int j = ( pInfo->iIter + (htlod->iHistory -i) ) % htlod->iHistory;
    printf(
      "%6d  %08lX  %+10.4f\n",
      pInfo->iIter - i, htlod->history[ j ], htlod->rhs[ j ]
    );
  }
}

static void elavhod_update( struct MXInfo *pInfo, struct ELAVHODInfo *htlod )
{
  uint64_t elavh = 0;
  elavh = elavhod__hash_64x32(elavh, (uint32_t)pInfo->cEnters);
  elavh = elavhod__hash_64x32(elavh, (uint32_t)pInfo->cLeaves);

  for(int i=0; i<pInfo->iActivevars; i++ )
    elavh = elavhod__hash_64x32(elavh, (uint32_t)pInfo->pActive[ i ]);

  TLP_UINT rhscol = pInfo->iCols - 1;
  double rhs = TBMX(1, rhscol );

  int j = pInfo->iIter % htlod->iHistory;
  htlod->history[ j ] = elavh;
  htlod->rhs[ j ] = rhs;
}

static int elavhod_detect( struct MXInfo *pInfo, struct ELAVHODInfo *htlod )
{
  int j = pInfo->iIter % htlod->iHistory;
  for(int i=0; i<htlod->iHistory; i++)
    if( i != j )
      if( htlod->history[ j ] == htlod->history[ i ] )
        if( EQUAL( htlod->rhs[ j ], htlod->rhs[ i ], htlod->rhstol ) )
          return 1;

  return 0;
}

TLP_RCCODE elavhod_check( struct MXInfo *pInfo, struct ELAVHODInfo *htlod )
{
  elavhod_update(pInfo, htlod);
  if( elavhod_detect(pInfo, htlod) ) return tlp_rc_encode(TLP_OSCILLATION);
  return tlp_rc_encode(TLP_OK);
}

void elavhod_fin( struct MXInfo *pInfo, struct ELAVHODInfo *htlod  )
{
  free(htlod->var_entropy);
  free(htlod->history);
  free(htlod->rhs);
}

///////////////

int test_is_concave(double* vecX, double* mxCoef, TLP_UINT size)
{
  TLP_UINT i, j;
  double sum = 0;
  for(i=0; i<size; i++)
    for(j=0; j<size; j++)
      sum = sum + mxCoef[ i, j ] * vecX[ i ] * vecX[ j ];

  return sum >= 0 ? 1 : 0;
}

#define GMX(r,c) mx[(r) * (rows +1) + (c)]

TLP_RCCODE gauss(double* soln, double* mx, int rows)
{
  TLP_RCCODE rc = tlp_rc_encode(TLP_OK);
  int * pivotrows = (int*)malloc(rows * sizeof(int));
  int * solnrows = (int*)malloc(rows * sizeof(int));
  for(int i=0; i<rows; i++) pivotrows[i] = i;
  for(int i=0; i<rows; i++) solnrows[i] = -1;
  int pivots = rows;

  for(int i=0; i<rows; i++)
  {
    // for column i, find row with biggest pivot and remove it from pivot row collection
    int t, p_pivot = 0, r_pivot = pivotrows[ p_pivot ];
    double biggest_pivot = fabs(GMX( pivotrows[ p_pivot ], i ));
    for(int p=0; p<pivots; p++)
      if(fabs(GMX( pivotrows[ p ], i )) > fabs(biggest_pivot))
        biggest_pivot = fabs(GMX( r_pivot = pivotrows[ p_pivot = p ], i ));
    pivotrows[ p_pivot ] = pivotrows[ pivots-- -1 ];

    // add this pivot row to the solution-decoder
    solnrows[i] = r_pivot;

    if(fabs(biggest_pivot) < 1e-10) { rc = tlp_rc_encode(TLP_ZERO); goto fail; }

    double divider = GMX( r_pivot, i );
    if(fabs(divider) < 1e-10) { rc = tlp_rc_encode(TLP_INFINITY); goto fail; }

    // reduce non-pivot rows
    for(int r=0; r<r_pivot; r++)
      for(int c=rows; c>=i; c--) // count down RTL from rhs
        GMX( r, c ) -= GMX( r_pivot, c ) * GMX( r, i ) / divider;
    for(int r=r_pivot +1; r<rows; r++)
      for(int c=rows; c>=i; c--) // count down RTL from rhs
        GMX( r, c ) -= GMX( r_pivot, c ) * GMX( r, i ) / divider;

    // factor pivot row
    for(int c=rows; c>=i; c--) // count down RTL from rhs
      GMX( r_pivot, c ) /= GMX( r_pivot, i );
  }

  // extract solution using indirect lookup through solution-decoder
  for(int j=0; j<rows; j++) soln[ j ] = GMX( solnrows[ j ], rows);
/*
  for(int r=0; r<rows; r++)
    for(int c=0; c<rows +1; c++)
      printf("%15.8f%c", GMX(r,c), (c<n) ? ' ' : '\n' );
  putchar('\n');
  for(int j=0; j<rows; j++) printf("%10.4f ", soln[j]);
  putchar('\n');
*/

fail:
  free(pivotrows);
  free(solnrows);

  return rc;
}

///////////////

#define DMX(r,c) mx1[(r) * size + (c)]

// https://en.wikipedia.org/wiki/Bareiss_algorithm
// https://www.ams.org/journals/mcom/1968-22-103/S0025-5718-1968-0226829-0/S0025-5718-1968-0226829-0.pdf
// https://github.com/adolfos94/Bareiss-Algorithm/blob/master/index.js
double determinant( double* mx, TLP_UINT size )
{
  size_t bytes = size * size * sizeof(double);
  double* mx1 = (double*)malloc(bytes);
  memcpy( mx1, mx, bytes );

  for(int i=0; i<size; i++ )
  {
    double pivot = DMX( i, i );
    double divider = i == 0 ? 1 : DMX( i - 1, i - 1 );
    for(int row=0; row<size; row++ )
    {
      if(i == row) continue;
      for(int col=i+1; col<size; col++ )
        DMX( row, col )= ( pivot * DMX( row, col ) - DMX( row, i ) * DMX( i, col ) ) / divider;
    }
  }

  double det = DMX( size - 1, size - 1 );
  free(mx1);
  return det;
}

///////////////

TLP_RCCODE
setup_min_qeq(
  double** ppMX_new,
  const double* pOBJMX, TLP_UINT iVariables,
  const double* pEQMX, TLP_UINT iEQConstraints
)
{
  // https://www.youtube.com/watch?v=gCs4YKiHIhg
  // obj = 1 * x1 ^ 2 + 3 * x2 + 2 * x3 ^ 2

  // mx[ # variables + # constraints , # constraints + # variables + rhs ]
  //   L1        L2       x1    x2    x3     rhs
  // -dG1/dx1  -dG2/dx1  dO/x1   0     0     rhs
  // -dG1/dx2  -dG2/dx2    0    dO/x2  0     rhs
  // -dG1/dx3  -dG2/dx3    0     0    dO/x3  rhs
  //                      C11   C12   C13    rhs
  //                      C21   C22   C23    rhs
  // where L1, L2 are lagrangian lambdas
  // solve for x1, x2, x3
  TLP_RCCODE rc;

  if( !ppMX_new ) return tlp_rc_encode_info(TLP_ASSERT, TLP_BADINDEX, __LINE__);
  if( *ppMX_new ) return tlp_rc_encode_info(TLP_ASSERT, TLP_BADINDEX, __LINE__);
  if( !pOBJMX || (iVariables<2) ) return tlp_rc_encode_info(TLP_ASSERT, TLP_BADINDEX, __LINE__);
  if( (!pEQMX && iEQConstraints) || (iEQConstraints<2) ) return tlp_rc_encode_info(TLP_ASSERT, TLP_BADINDEX, __LINE__);

  int iRows = iVariables + iEQConstraints;
  int iCols = iVariables + iEQConstraints +1; // +1 for rhs

  size_t iBytes = sizeof(double) * iRows * iCols;
  double *pMX = *ppMX_new = (double*) aligned_alloc(16, iBytes);
  memset(pMX, 0, iBytes);

  double *pQOBJMX = (double*) &pOBJMX[ iVariables ];

  // add lagragian coef columns
  for(int c=0; c<iEQConstraints; ++c)
  {
    for(int v=0; v<iVariables; ++v)
    {
      pMX[v * iCols + c] = pEQMX[ (iVariables +1) * c + v ] * -1; // +1 for rhs, -1 because KKT
    }
  }

  // add dO/xn diagonal
  for(int v=0; v<iVariables; ++v)
  {
    int col = iEQConstraints +v;
    int row = +v;
    pMX[row * iCols + col] = pQOBJMX[ iVariables * v + v ] *2; // *2 since we're differentiating quadratic terms
  }

  // add linear terms to rhs
  for(int v=0; v<iVariables; ++v)
  {
    int col = iVariables + iEQConstraints; // rhs col
    int row = +v;
    pMX[row * iCols + col] = pOBJMX[ v ] *-1; // *-1 because KKT
  }

  // add EQ constraints
  for(int c=0; c<iEQConstraints; ++c)
  {
    for(int v=0; v<iVariables +1; ++v) // +1 to include rhs
    {
      int col = iEQConstraints +v;
      int row = iVariables +c;
      pMX[row * iCols + col] = pEQMX[ (iVariables +1) * c + v ]; // +1 for rhs
    }
  }

  return tlp_rc_encode(TLP_OK);
}
























