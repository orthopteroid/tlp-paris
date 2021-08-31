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

static inline void elavhod__avh_hash(uint32_t* h, TLP_UINT v)
{
  *h ^= (*h >> 1) + (uint32_t)v;
  *h += (uint32_t)v << 15;
}

void elavhod_setup( struct MXInfo *pInfo, struct ELAVHODInfo *htlod, uint8_t history, double rhstol )
{
  memset(htlod, 0, sizeof(struct ELAVHODInfo));

  htlod->rhstol = rhstol;
  htlod->iHistory = history;
  htlod->iActiveVars = pInfo->iDefiningvars + pInfo->iSlackvars;

  htlod->var_entropy = (TLP_UINT*)malloc( sizeof(TLP_UINT) * htlod->iActiveVars );
  memset(htlod->var_entropy, 0, sizeof(TLP_UINT) * htlod->iActiveVars);

  htlod->elv = (uint32_t*)malloc( sizeof(uint32_t) * htlod->iHistory );
  memset(htlod->elv, 0, sizeof(uint32_t) * htlod->iHistory);

  htlod->avh = (uint32_t*)malloc( sizeof(uint32_t) * htlod->iHistory );
  memset(htlod->avh, 0, sizeof(uint32_t) * htlod->iHistory);

  htlod->rhs = (double*)malloc( sizeof(double) * htlod->iHistory );
  memset(htlod->rhs, 0, sizeof(double) * htlod->iHistory);

  // https://www.cyrill-gremaud.ch/howto-generate-secure-random-number-on-nix/
  // kudos: Bennet Yee
  int fd;

  if( (fd = open("/dev/urandom", O_RDONLY)) == -1 )
    perror("Error: impossible to read randomness source\n");

  if( read(fd, htlod->var_entropy, sizeof(TLP_UINT) * htlod->iActiveVars) != sizeof(TLP_UINT) * htlod->iActiveVars )
    perror("read() failed\n");

  close(fd);
}

void elavhod_dump_history( struct MXInfo *pInfo, struct ELAVHODInfo *htlod )
{
  int i;

  for(i=0; i<htlod->iHistory; i++)
  {
    int j = ( pInfo->iIter + (htlod->iHistory -i) ) % htlod->iHistory;
    printf(
      "%6d  %08X  %08X  %+10.4f\n",
      pInfo->iIter - i, htlod->elv[ j ], htlod->avh[ j ], htlod->rhs[ j ]
    );
  }
}

void elavhod_update( struct MXInfo *pInfo, struct ELAVHODInfo *htlod )
{
  int i;

  uint32_t elv =
    ((uint32_t)htlod->var_entropy[ pInfo->iVarEnters ] << 16) |
    (uint32_t)htlod->var_entropy[ pInfo->iVarLeaves ];

  uint32_t avh = 0;
  for(i=0; i<htlod->iActiveVars; i++ )
    elavhod__avh_hash(&avh, htlod->var_entropy[ pInfo->pActiveVariables[ i ] ]);

  TLP_UINT rhscol = pInfo->iCols - 1;
  double rhs = TBMX(1, rhscol );

  i = pInfo->iIter % htlod->iHistory;
  htlod->elv[ i ] = elv;
  htlod->avh[ i ] = avh;
  htlod->rhs[ i ] = rhs;
}

int elavhod_detect( struct MXInfo *pInfo, struct ELAVHODInfo *htlod )
{
  int i, j;
  j = pInfo->iIter % htlod->iHistory;
  for(i=0; i<htlod->iHistory; i++)
    if( i != j )
      if( htlod->elv[ j ] == htlod->elv[ i ] )
        if( htlod->avh[ j ] == htlod->avh[ i ] )
          if( EQUAL( htlod->rhs[ j ], htlod->rhs[ i ], htlod->rhstol ) )
            return 1;

  return 0;
}

void elavhod_fin( struct MXInfo *pInfo, struct ELAVHODInfo *htlod  )
{
  free(htlod->var_entropy);
  free(htlod->elv);
  free(htlod->avh);
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

#define GMX(r,c) mx[(r) * (n +1) + (c)]

void gauss(double* soln, double* mx, int n)
{
  int * pivotrows = (int*)malloc(n * sizeof(int));
  int * solnrows = (int*)malloc(n * sizeof(int));
  for(int i=0; i<n; i++) pivotrows[i] = i;
  for(int i=0; i<n; i++) solnrows[i] = -1;
  int pivots = n;

  for(int i=0; i<n; i++)
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

    //if(biggest_pivot < 1e-10) continue;

    // reduce non-pivot rows
    for(int r=0; r<r_pivot; r++)
      for(int c=n; c>=i; c--) // count down RTL from rhs
        GMX( r, c ) -= GMX( r_pivot, c ) * GMX( r, i ) / GMX( r_pivot, i );
    for(int r=r_pivot +1; r<n; r++)
      for(int c=n; c>=i; c--) // count down RTL from rhs
        GMX( r, c ) -= GMX( r_pivot, c ) * GMX( r, i ) / GMX( r_pivot, i );

    // factor pivot row
    for(int c=n; c>=i; c--) // count down RTL from rhs
      GMX( r_pivot, c ) /= GMX( r_pivot, i );
  }

  // extract solution using indirect lookup through solution-decoder
  for(int j=0; j<n; j++) soln[ j ] = GMX( solnrows[ j ], n);
/*
  for(int r=0; r<n; r++)
    for(int c=0; c<n +1; c++)
      printf("%15.8f%c", GMX(r,c), (c<n) ? ' ' : '\n' );
  putchar('\n');
  for(int j=0; j<n; j++) printf("%10.4f ", soln[j]);
  putchar('\n');
*/
  free(pivotrows);
  free(solnrows);
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



























