// linear program solver (simplex method), orthopteroid@gmail.com, BSD2 license

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>

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
