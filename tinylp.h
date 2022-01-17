// linear program solver (simplex method), orthopteroid@gmail.com, BSD2 license
// based on "Operations Research", Hiller & Lieberman (2nd ed. 1974, 7th ed. 2001)

#ifndef _TINYLP_H_
#define _TINYLP_H_

#include <stdint.h>
#include <stdalign.h>

#define TLP_UINT uint16_t
#define TLP_RCCODE uint32_t

// error codes

enum
{
  TLP_OK=0,
  TLP_UNFINISHED=1,
  TLP_ASSERT=2,
  TLP_GEOMETRY=3,
  TLP_INVALID=4,
  TLP_ZERO=5,
  TLP_INFINITY=6,
  TLP_UNBOUNDED=7,
  TLP_AUGMENTED=8,
  TLP_OSCILLATION=9,
};

const static char* tlp_messages[] =
{
  "TLP_OK",
  "TLP_UNFINISHED",
  "TLP_ASSERT",
  "TLP_GEOMETRY",
  "TLP_INVALID",
  "TLP_ZERO",
  "TLP_INFINITY",
  "TLP_UNBOUNDED",
  "TLP_AUGMENTED",
  "TLP_OSCILLATION",
};

// for some errors, knowing the relevant row/col that caused the problem
// is desired. so, we pack the status, row and col into a single 32bit
// value: 13 bits for each of row and col and 6 bits for error code.
// This makes 16382 rows/cols (16383 represents bad-index) and 64 error/status codes.

enum
{
  TLP_OSC_HISTORY=20,
  TLP_RCBITS_IDX=13,
  TLP_RCBITS_STAT=6,
  TLP_BADINDEX=((1<<(TLP_RCBITS_IDX+1))-1),
};

#ifdef NDEBUG

inline TLP_RCCODE tlp_rc_encode(TLP_UINT e) { return (TLP_RCCODE)e<<(2*TLP_RCBITS_IDX); }
inline TLP_UINT tlp_rc_decode(TLP_RCCODE rc) { return (TLP_UINT)(rc>>(2*TLP_RCBITS_IDX)); }
inline TLP_RCCODE tlp_rc_encode_info(TLP_UINT e, TLP_UINT r, TLP_UINT c)
{
  return (TLP_RCCODE)e<<(2*TLP_RCBITS_IDX) | (TLP_RCCODE)r<<TLP_RCBITS_IDX | (TLP_RCCODE)c;
}
inline void tlp_rc_decode_info(TLP_RCCODE rc, TLP_UINT *e, TLP_UINT *r, TLP_UINT *c)
{
  *e = ((1<<TLP_RCBITS_STAT)-1) & (TLP_UINT)(rc>>(2*TLP_RCBITS_IDX));
  *r = TLP_BADINDEX & (TLP_UINT)(rc>>TLP_RCBITS_IDX);
  *c = TLP_BADINDEX & (TLP_UINT)(rc);
}

#else

TLP_RCCODE tlp_rc_encode(TLP_UINT e);
TLP_UINT tlp_rc_decode(TLP_RCCODE rc);
TLP_RCCODE tlp_rc_encode_info(TLP_UINT e, TLP_UINT r, TLP_UINT c);
void tlp_rc_decode_info(TLP_RCCODE rc, TLP_UINT *e, TLP_UINT *r, TLP_UINT *c);

#endif // DEBUG

/***************************************/

struct MXInfo
{
  TLP_UINT bMaximize : 1, bQuadratic : 1;
  // matrix structure
  TLP_UINT iCols, iRows;
  double* pMatrix;
  double fMin, fMinNeg;
  double fMax, fMaxNeg;
  double fZero;
  // active set: indexed by row (ie constraint) and returns column (variable)
  TLP_UINT iActivevars;
  TLP_UINT* pActive;
  // matrix contents
  TLP_UINT iConstraints, iDefiningvars, iSlackvars;
  const char** szVars;
  // status
  TLP_UINT iIter;
  TLP_UINT cEnters, cLeaves;
  struct
  {
    uint64_t *pivHashArr;
    double *rhsArr;
  } osc;
};

void tlp_dump_mx( double* pMX, TLP_UINT rr, TLP_UINT cc );
void tlp_dump_tableau( struct MXInfo* pInfo, TLP_UINT r1, TLP_UINT c1 );
void tlp_dump_active_cols( struct MXInfo* pInfo );
void tlp_dump_active_soln( struct MXInfo* pInfo );
void tlp_dump_current_soln( struct MXInfo* pInfo );
void tlp_dump_osc_history( struct MXInfo *pInfo );

TLP_RCCODE
tlp_setup_max(
  struct MXInfo* pInfo,
  const double* pOBJMX, TLP_UINT iVariables,
  const double* pLEMX, TLP_UINT iLEConstraints,
  const double* pEQMX, TLP_UINT iEQConstraints,
  const char** szVars
);

TLP_RCCODE
tlp_setup_max_qlp(
  struct MXInfo* pInfo,
  const double* pOBJMX, TLP_UINT iVariables,
  const double* pLEMX, TLP_UINT iLEConstraints,
  const char** szVars
);

TLP_RCCODE
tlp_setup_min(
  struct MXInfo* pInfo,
  const double* pOBJMX, TLP_UINT iVariables,
  const double* pGEMX, TLP_UINT iGEConstraints,
  const double* pEQMX, TLP_UINT iEQConstraints,
  const char** szVars
);

TLP_RCCODE tlp_pivot( struct MXInfo* pInfo );
TLP_RCCODE tlp_soln( struct MXInfo* pInfo, double* pSOLMX );
TLP_RCCODE tlp_mxequal( const double* pA, const double* pB, const double fZero, TLP_UINT n );
TLP_RCCODE tlp_fini( struct MXInfo* pInfo );

int tlp_is_augmented( struct MXInfo* pInfo );

 TLP_RCCODE
tlp_rowsubmul(
  struct MXInfo* pInfo, TLP_UINT r1, TLP_UINT c1, TLP_UINT r2
);

 TLP_RCCODE
tlp_rowdivsub(
  struct MXInfo* pInfo, TLP_UINT r1, TLP_UINT c1, TLP_UINT r2
);


#endif /*_TINYLP_H_*/
