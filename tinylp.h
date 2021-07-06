// linear program solver (simplex method), orthopteroid@gmail.com, BSD2 license
// based on "Operations Research", Hiller & Lieberman (1974)

#ifndef _TINYLP_H_
#define _TINYLP_H_

#include <stdint.h>
#include <stdalign.h>

#define TLP_UINT uint16_t
#define TLP_RCCODE uint32_t

// for some errors, knowing the relevant row/col that caused the problem
// is desired. so, we pack the status, row and col into a single 32bit
// value: 14 bits for each of row and col and 4 bits for error code.
// This makes 16384 rows/cols (-1 for bad index) and 16 error/status codes.

#define TLP_RCBITS_IDX 14
#define TLP_RCBITS_STAT 4

enum
{
  TLP_OK=0,
  TLP_UNFINISHED,
  TLP_ASSERT,
  TLP_GEOMETRY,
  TLP_INVALID,
  TLP_ZERO,
  TLP_INFINITY,
  TLP_UNBOUNDED,
};

#define TLP_BADINDEX ((1<<(TLP_RCBITS_IDX+1))-1)
#define TLP_RCUNFINISHED (TLP_UNFINISHED<<(2*TLP_RCBITS_IDX))

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
  // matrix structure
  TLP_UINT bMaxProblem;
  TLP_UINT iCols, iRows;
  TLP_UINT* pActiveVariables;
  double* pMatrix;
  double fMin, fMinNeg;
  double fMax, fMaxNeg;
  double fZero;
  // matrix contents
  TLP_UINT iConstraints, iDefiningvars, iSlackvars;
  const char** szVars;
  // status
  TLP_UINT iIter;
  TLP_UINT iVarEnters, iVarLeaves;
};

void tlp_dump_tableau( struct MXInfo* pInfo, TLP_UINT r1, TLP_UINT c1 );
void tlp_dump_active_vars( struct MXInfo* pInfo );
void tlp_dump_current_soln( struct MXInfo* pInfo );

TLP_RCCODE
tlp_setup_max(
  struct MXInfo* pInfo,
  const double* pOBJMX, TLP_UINT iVariables,
  const double* pLEMX, TLP_UINT iLEConstraints,
  const double* pEQMX, TLP_UINT iEQConstraints,
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

TLP_RCCODE
tlp_pivot(
  struct MXInfo* pInfo
);

TLP_RCCODE
tlp_soln(
  struct MXInfo* pInfo,
  double* pSOLMX
);

TLP_RCCODE
tlp_mxequal(
  const double* pA,
  const double* pB,
  const double fZero,
  TLP_UINT n
);

TLP_RCCODE
tlp_fini(
  struct MXInfo* pInfo
);

#endif /*_TINYLP_H_*/
