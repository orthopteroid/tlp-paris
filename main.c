// linear program solver (simplex method), orthopteroid@gmail.com, BSD2 license
// based on "Operations Research", Hiller & Lieberman (1974)

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "tinylp.h"
#include "analysis.h"

#define CHECK \
  do { \
    if( rc ) { \
      tlp_rc_decode_info(rc, &e, &r, &c); \
      printf("e %d r %d c %d\n", e, r, c); \
      assert( !rc ); \
    } \
  } while(0) \

int main()
{
  struct ELAVHODInfo htlod;
  struct MXInfo mx2;
  TLP_RCCODE rc;
  TLP_UINT e, r, c;

  printf("----- new problem\n");
  {
    // https://www.youtube.com/watch?v=gCs4YKiHIhg
    // obj = 1 * x1 ^ 2 + 3 * x2 + 2 * x3 ^ 2

    const char* vars[] = {"x1", "x2", "x3"};
    //  x1   x2   x3
    double mxobj_l[] = {0., 3., 0,};
    //  x1   x2   x3
    double mxobj_q[] = {
      1., 0., 0.,
      0., 0., 0.,
      0., 0., 2.,
    };
    //  x1   x2   x3   rhs
    double mxEQ[] = {
      5., -1., -3., 2.,
      3.,  1.,  2., 7.,
    };

    // mx[ # variables + # constraints , # constraints + # variables + rhs ]
    //   L1        L2       x1    x2    x3     rhs
    // -dG1/dx1  -dG2/dx1  dO/x1   0     0     rhs
    // -dG1/dx2  -dG2/dx2    0    dO/x2  0     rhs
    // -dG1/dx3  -dG2/dx3    0     0    dO/x3  rhs
    //                      C11   C12   C13    rhs
    //                      C21   C22   C23    rhs
    // where L1, L2 are lagrangian lambdas
    // solve for x1 and x2
    // L1  L2  x1 x2 x3  rhs
    double mxKKT[] = {
      // lagrange coefs ?
      -5., -3.,  2.,  0.,  0.,  0.,
      +1., -1.,  0.,  0.,  0., -3.,
      +3., -2.,  0.,  0.,  4.,  0.,
      // constraints
       0.,  0.,  5., -1., -3.,  2.,
       0.,  0.,  3.,  1.,  2.,  7.,
    };
    double mxKKT_sol[] = { 0, 0, 0, 0, 0, };
    double mxKKT_ver[] = { -0.7907, 2.2093, 1.3372, -0.4070, 1.6977 };
    gauss( mxKKT_sol, mxKKT, 5 );
    rc = tlp_mxequal(mxKKT_ver, mxKKT_sol, 1e-3, 5); CHECK;
    printf("ok\n");
  }

  printf("----- new problem\n");
  {
    // https://www.brainkart.com/article/Special-Cases-in-the-Simplex-Method_11207/
    // PROBLEM SET 3.5A, Problem #2
    // degenerate/oscillating

    const char* vars[] = { "x0", "x1" };

    //  x0   x1
    double mxobj[] = {
      3., 2.
    };

    //  x0   x1    rhs
    double mxLE[] = {
      4., 3., 12.,
      4., -1., 8.,
      4., 1., 8.,
    };

    //  x0   x1    rhs
    double mxEQ[] = {
      0., 0., 0.
    };

    //  x0    x1   value-of-obj-fxn
    double mxsol[] = { 0.,  0.,  0. };
    double mxver[] = { 0.,  4.,  6. };

    rc = tlp_setup_max(&mx2, mxobj, 2, mxLE, 3, mxEQ, 0, (const char**)&vars); CHECK;
    printf("%s\n", mx2.bMaxProblem ? "Maximize" : "Minimize");
    tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
    elavhod_setup( &mx2, &htlod, 10, 1.0E-8 );
    while( (rc = tlp_pivot( &mx2 )) == TLP_RCUNFINISHED )
    {
      printf("iteration %d: %d enters %d leaves\n", mx2.iIter, mx2.iVarEnters, mx2.iVarLeaves);
      tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
      tlp_dump_current_soln(&mx2);
      elavhod_update( &mx2, &htlod );
      //elavhod_dump_history(&mx2, &htlod);
      if( elavhod_detect( &mx2, &htlod ) ) { printf("oscillation detected\n"); rc = 0; break; }
    }
    CHECK;
    tlp_soln(&mx2, mxsol);
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
    elavhod_fin( &mx2, &htlod  );
  }

  printf("----- new problem\n");
  {
    const char* vars[] = { "x0", "x1" };

    //  x0   x1
    double mxobj[] = {
      3., 5.
    };

    //  x0   x1    rhs
    double mxLE[] = {
      2., 0., 4.,
      0., 2., 12.,
    };

    //  x0   x1    rhs
    double mxEQ[] = {
      3., 2., 18.
    };

    //  x0    x1   value-of-obj-fxn
    double mxsol[] = { 0.,  0.,  0. };
    double mxver[] = { 2.,  6.,  36. };

    rc = tlp_setup_max(&mx2, mxobj, 2, mxLE, 2, mxEQ, 1, (const char**)&vars); CHECK;
    printf("%s\n", mx2.bMaxProblem ? "Maximize" : "Minimize");
    tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
    elavhod_setup( &mx2, &htlod, 10, 1.0E-8 );
    while( (rc = tlp_pivot( &mx2 )) == TLP_RCUNFINISHED )
    {
      printf("iteration %d: %d enters %d leaves\n", mx2.iIter, mx2.iVarEnters, mx2.iVarLeaves);
      tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
      tlp_dump_current_soln(&mx2);
      elavhod_update( &mx2, &htlod );
      //elavhod_dump_history(&mx2, &htlod);
      if( elavhod_detect( &mx2, &htlod ) ) { printf("oscillation detected\n"); rc = 0; break; }
    }
    CHECK;
    tlp_soln(&mx2, mxsol);
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
    elavhod_fin( &mx2, &htlod  );
  }

  printf("----- new problem\n");
  {
    const char* vars[] = { "x0", "x1" };

    //  x0   x1
    double mxobj[] = {
      6., 5.
    };

    //  x0   x1    rhs
    double mxLE[] = {
      1.,  1.,  5.,
      3.,  2., 12.,
    };

    //  x0    x1   value-of-obj-fxn
    double mxsol[] = { 0.,  0.,  0. };
    double mxver[] = { 2.,  3.,  27. };

    rc = tlp_setup_max(&mx2, mxobj, 2, mxLE, 2, (double *) 0, 0, (const char**)&vars); CHECK;
    printf("%s\n", mx2.bMaxProblem ? "Maximize" : "Minimize");
    tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
    while( (rc = tlp_pivot( &mx2 )) == TLP_RCUNFINISHED )
    {
      printf("iteration %d: %d enters %d leaves\n", mx2.iIter, mx2.iVarEnters, mx2.iVarLeaves);
      tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
      tlp_dump_current_soln(&mx2);
    }
    CHECK;
    tlp_soln(&mx2, mxsol);
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
  }

  // https://math.libretexts.org/Bookshelves/Applied_Mathematics/Book%3A_Applied_Finite_Mathematics_(Sekhon_and_Bloom)/04%3A_Linear_Programming_The_Simplex_Method/4.03%3A_Minimization_By_The_Simplex_Method
  printf("----- new problem\n");
  {
    const char* vars[] = { "x0", "x1" };

    //  x0   x1
    double mxobj[] = {
      40., 30.
    };

    //  x0   x1    rhs
    double mxLE[] = {
      1., 1., 12.,
      2., 1., 16.,
    };

    //  x0    x1   value-of-obj-fxn
    double mxsol[] = {0., 0., 0.};
    double mxver[] = {4., 8., 400.};

    rc = tlp_setup_max(&mx2, mxobj, 2, mxLE, 2, (double *) 0, 0, (const char**)&vars); CHECK;
    printf("%s\n", mx2.bMaxProblem ? "Maximize" : "Minimize");
    tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
    while( (rc = tlp_pivot( &mx2 )) == TLP_RCUNFINISHED )
    {
      printf("iteration %d: %d enters %d leaves\n", mx2.iIter, mx2.iVarEnters, mx2.iVarLeaves);
      tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
      tlp_dump_current_soln(&mx2);
    }
    CHECK;
    tlp_soln(&mx2, mxsol);
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
  }

  printf("----- new problem\n");
  {
    const char* vars[] = { "x0", "x1" };

    //  x0   x1
    double mxobj[] = {
      12., 16.
    };

    //  x0   x1    rhs
    double mxLE[] = {
      1., 2., 40.,
      1., 1., 30.,
    };

    //  x0    x1   value-of-obj-fxn
    double mxsol[] = {0., 0., 0.};
    double mxver[] = {20., 10., 400.};

    rc = tlp_setup_max(&mx2, mxobj, 2, mxLE, 2, (double *) 0, 0, (const char**)&vars); CHECK;
    printf("%s\n", mx2.bMaxProblem ? "Maximize" : "Minimize");
    tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
    while( (rc = tlp_pivot( &mx2 )) == TLP_RCUNFINISHED )
    {
      printf("iteration %d: %d enters %d leaves\n", mx2.iIter, mx2.iVarEnters, mx2.iVarLeaves);
      tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
      tlp_dump_current_soln(&mx2);
    }
    CHECK;
    tlp_soln(&mx2, mxsol);
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
  }

  printf("----- new problem\n");
  {
    const char* vars[] = { "x0", "x1" };

    //  x0   x1
    double mxobj[] = {
      40., 30.
    };

    //  x0   x1    rhs
    double mxGE[] = {
      1., 1., 12.,
      2., 1., 16.,
    };

    //  x0    x1   value-of-obj-fxn
    double mxsol[] = {0., 0., 0.};
    double mxver[] = {4., 8., 400.};

    rc = tlp_setup_min(&mx2, mxobj, 2, mxGE, 2, (double *) 0, 0, (const char**)&vars); CHECK;
    printf("%s\n", mx2.bMaxProblem ? "Maximize" : "Minimize");
    tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
    while( (rc = tlp_pivot( &mx2 )) == TLP_RCUNFINISHED )
    {
      printf("iteration %d: %d enters %d leaves\n", mx2.iIter, mx2.iVarEnters, mx2.iVarLeaves);
      tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
      tlp_dump_current_soln(&mx2);
    }
    CHECK;
    tlp_soln(&mx2, mxsol);
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
  }

  printf("----- new problem\n");
  {
    // H&L p728
    // n variables, m constraints
    // obj = 5 * x1 + x2 - (x1 - x2) ^2
    // obj = 5 * x1 + x2 - (   x1^2 - x1 * x2 - x2 * x1 + x2^2 )
    // obj = 5 * x1 + x2 + ( - x1^2 + x1 * x2 + x2 * x1 - x2^2 )
    // obj = A1 * x1 + A2 * x2 + Q11 * x1 ^2 + Q12 * x1 * x2 + Q21 * x2 * x1 + Q22 * x2 ^2
    // st.  C1 * x1 + C2 * x2 <= 2, x1 >= 0, x2 >= 0
    // nb. obj form must be Ecx - .5 EEqxjxk, so [q] coefs must be * 2, (H&L s:18.2 p:725)

    const char* vars[] = {"x1", "x2"};
    //  x1   x2
    double mxobj_l[] = {5., 1,};
    //  x1   x2
    double mxobj_q[] = {
      +1., -1.,
      -1., +1.,
    };
    //  x1   x3   rhs
    double mxLE[] = {
      1., 1., 2.,
    };

    // mx[ #variables + #constraints , #quadratics + #lagranges + #slacks + #artificals + rhs ]
    //  x1     x2   x3    L1     y1  y2  z1  z2  rhs
    // 2*Q11  2*Q12  0   dG1/dx1  -1   0   1   0  rhs
    // 2*Q21  2*Q22  0   dG1/dx2   0  -1   0   1  rhs
    //  A1     A2    1                            rhs
    // where:
    // x3 is a slack variable for LE relation
    // y1, y2 are surplus variables for GE relation?
    // z1, z2 are artificals (and are in the obj function)
// why is the slack x3 needed?
    //
    // minimize z1 + z2
    // x1   x2   x3   L1   y1   y2    z1   z2   rhs
    double mxMin[] = {
      +2.,  -2.,  0.,  1., -1.,  0.,  1.,  0.,  5.,
      -2.,  +2.,  0.,  1.,  0., -1.,  0.,  1.,  1.,
       1.,   1.,  1.,  0.,  0.,  0.,  0.,  0.,  2.,
    };
    double mxMin_sol[] = { 0, 0, 0, };
    double mxMin_ver[] = { 3./2., 1./2., 3., }; // x1 x2 y1

    TLP_UINT activevars[5];
    struct MXInfo mx = {
      .bMaxProblem = 0,
      .iRows = 3, .iCols = 8 +1, // +1 for rhs
.pActiveVariables = activevars,
.pMatrix = mxMin,
.fMin = 1e-99, .fMinNeg = -1e-99,
.fMax = 1e99, .fMaxNeg = -1e99,
.fZero = 1e-10,
.iConstraints = 1, .iDefiningvars = 2, .iSlackvars = 3,
.szVars = vars,
.iIter = -1,
.iVarEnters = ~0, .iVarLeaves = ~0,
    };
    tlp_dump_tableau(&mx, TLP_BADINDEX, TLP_BADINDEX);
    rc = tlp_mxequal(mxMin_ver, mxMin_sol, 1e-3, 5); CHECK;
  }

  return 0;
}
