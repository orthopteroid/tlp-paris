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
      printf("%s r %d c %d\n", tlp_messages[e], r, c); \
      assert( !rc ); \
    } \
  } while(0) \

int main()
{
  struct ELAVHODInfo htlod;
  struct MXInfo mx2;
  TLP_RCCODE rc;
  TLP_UINT e, r, c;

  printf("----- QLP maximize. line %d\n", __LINE__);
  {
    // QLP maximization problem [H&L ed7 p684]
    // max F == 15*x1 + 30*x2 + 4*x1x2 - 2*x1x1 - 4*x2x2
    // st  G == 1*x1 + 2*x2 <= 30
    double mxobj[] = {     //  x1 x2 [ [ x1 x2 ] [ x1 x2 ] ]
      // linear part
      15., 30,
      //  quadratic part (fully symmetric)
      -2., +2.,  // nb ??? 4x1x2 ???
      +2., -4.,
    };
    double mxLE[] = {
    // x1  x3   rhs
      1.,  2.,  30.,
    };
    const char* vars[] = {"x1", "x2"};

    double mxsol[] = { 0, 0, 0, };
    double mxver[] = { 12., 9., 0, }; // x1 x2 rhs

    double d = determinant( &mxobj[2] /* read quadratic part only */, 2 );
    if(d<=0.) { printf("bad obj Q terms"); return -1; }

    rc = tlp_setup_max_qlp(&mx2, mxobj, 2, mxLE, 1, (const char**)&vars); CHECK;
    printf("%s\n", mx2.bMaximize ? "Maximize" : "Minimize");
    tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
    elavhod_setup( &mx2, &htlod, 10, 1.0E-8 );
    while( 1 )
    {
      rc = tlp_pivot( &mx2 );
      if( tlp_rc_decode(rc) != TLP_UNFINISHED ) break;
      printf("iteration %d: column %d leaves %d enters \n", mx2.iIter, mx2.cLeaves, mx2.cEnters);

      rc = elavhod_check( &mx2, &htlod );
      if( tlp_rc_decode(rc) == TLP_OSCILLATION ) break;

      tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
      //if( !tlp_is_augmented(&mx2) ) { printf("FEASIBLE SOLUTION\n"); tlp_dump_current_soln(&mx2); }
      tlp_dump_active_soln(&mx2); // for QP, result is always augmented
    }
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );

    //tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
    //rc = tlp_mxequal(mx_ver, mx_sol, 1e-3, 5); CHECK;
    if( tlp_rc_decode(rc) == TLP_OSCILLATION ) elavhod_dump_history(&mx2, &htlod);
    rc = tlp_soln(&mx2, mxsol);
    printf("tlp_soln %s\n", tlp_messages[ tlp_rc_decode(rc) ] );

    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double));
    printf("tlp_mxequal %s\n", tlp_messages[ tlp_rc_decode(rc) ] );

    //double objval = tlp_eval_qlp(&mx2, mxobj);

    rc = tlp_fini(&mx2); CHECK;
    elavhod_fin( &mx2, &htlod  );
  }

  printf("----- LP minimize. line %d\n", __LINE__);
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
    printf("%s\n", mx2.bMaximize ? "Maximize" : "Minimize");
    tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
    while( 1 )
    {
      rc = tlp_pivot( &mx2 );
      if( tlp_rc_decode(rc) != TLP_UNFINISHED ) break;
      printf("iteration %d: column %d leaves %d enters \n", mx2.iIter, mx2.cLeaves, mx2.cEnters);
      tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
      if( tlp_is_augmented(&mx2) ) printf("AUGMENTED\n"); else tlp_dump_current_soln(&mx2);
    }
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    if( !(rc = tlp_soln(&mx2, mxsol)) ) printf("%s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
  }

  printf("----- KKT minimize. line %d\n", __LINE__);
  {
    // https://www.youtube.com/watch?v=gCs4YKiHIhg
    // obj = 1 * x1 ^ 2 + 3 * x2 + 2 * x3 ^ 2

    const char* vars[] = {"x1", "x2", "x3"};
    //  x1 x2 x3 [ [ x1 x2 x3 ] [ x1 x2 x3 ] [ x1 x2 x3 ] ]
    double mxobj[] = {
      // linear part
      0., 3., 0,
      //  quadratic part
      1., 0., 0.,
      0., 0., 0.,
      0., 0., 2.,
    };
    //  x1   x2   x3   rhs
    double mxEQ[] = {
      5., -1., -3., 2., // eqG1
      3.,  1.,  2., 7., // eqG2
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

  printf("----- LP maximize (degenerate/oscillating). line %d\n", __LINE__);
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
    printf("%s\n", mx2.bMaximize ? "Maximize" : "Minimize");
    tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
    elavhod_setup( &mx2, &htlod, 10, 1.0E-8 );
    while( 1 )
    {
      rc = tlp_pivot( &mx2 );
      if( tlp_rc_decode(rc) != TLP_UNFINISHED ) break;
      printf("iteration %d: column %d leaves %d enters \n", mx2.iIter, mx2.cLeaves, mx2.cEnters);

      rc = elavhod_check( &mx2, &htlod );
      if( tlp_rc_decode(rc) == TLP_OSCILLATION ) break;

      tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
      if( tlp_is_augmented(&mx2) ) printf("AUGMENTED\n"); else tlp_dump_current_soln(&mx2);
    }
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    if( tlp_rc_decode(rc) == TLP_OSCILLATION ) elavhod_dump_history(&mx2, &htlod);
    if( !(rc = tlp_soln(&mx2, mxsol)) ) printf("%s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
    elavhod_fin( &mx2, &htlod  );
  }

  printf("----- LP maximize (maximize le and eq constraints). line %d\n", __LINE__);
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
    printf("%s\n", mx2.bMaximize ? "Maximize" : "Minimize");
    tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
    elavhod_setup( &mx2, &htlod, 10, 1.0E-8 );
    while( 1 )
    {
      rc = tlp_pivot( &mx2 );
      if( tlp_rc_decode(rc) != TLP_UNFINISHED ) break;
      printf("iteration %d: column %d leaves %d enters \n", mx2.iIter, mx2.cLeaves, mx2.cEnters);

      rc = elavhod_check( &mx2, &htlod );
      if( tlp_rc_decode(rc) == TLP_OSCILLATION ) break;

      tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
      if( tlp_is_augmented(&mx2) ) printf("AUGMENTED\n"); else tlp_dump_current_soln(&mx2);
    }
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    if( tlp_rc_decode(rc) == TLP_OSCILLATION ) elavhod_dump_history(&mx2, &htlod);
    if( !(rc = tlp_soln(&mx2, mxsol)) ) printf("%s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
    elavhod_fin( &mx2, &htlod  );
  }

  printf("----- LP maximize. line %d\n", __LINE__);
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
    printf("%s\n", mx2.bMaximize ? "Maximize" : "Minimize");
    tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
    while( 1 )
    {
      rc = tlp_pivot( &mx2 );
      if( tlp_rc_decode(rc) != TLP_UNFINISHED ) break;
      printf("iteration %d: column %d leaves %d enters \n", mx2.iIter, mx2.cLeaves, mx2.cEnters);
      tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
      if( tlp_is_augmented(&mx2) ) printf("AUGMENTED\n"); else tlp_dump_current_soln(&mx2);
    }
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    if( !(rc = tlp_soln(&mx2, mxsol)) ) printf("%s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
  }

  // https://math.libretexts.org/Bookshelves/Applied_Mathematics/Book%3A_Applied_Finite_Mathematics_(Sekhon_and_Bloom)/04%3A_Linear_Programming_The_Simplex_Method/4.03%3A_Minimization_By_The_Simplex_Method
  printf("----- LP maximize. line %d\n", __LINE__);
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
    printf("%s\n", mx2.bMaximize ? "Maximize" : "Minimize");
    tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
    while( 1 )
    {
      rc = tlp_pivot( &mx2 );
      if( tlp_rc_decode(rc) != TLP_UNFINISHED ) break;
      printf("iteration %d: column %d leaves %d enters \n", mx2.iIter, mx2.cLeaves, mx2.cEnters);
      tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
      if( tlp_is_augmented(&mx2) ) printf("AUGMENTED\n"); else tlp_dump_current_soln(&mx2);
    }
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    if( !(rc = tlp_soln(&mx2, mxsol)) ) printf("%s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
  }

  printf("----- LP maximize. line %d\n", __LINE__);
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
    printf("%s\n", mx2.bMaximize ? "Maximize" : "Minimize");
    tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
    while( 1 )
    {
      rc = tlp_pivot( &mx2 );
      if( tlp_rc_decode(rc) != TLP_UNFINISHED ) break;
      printf("iteration %d: column %d leaves %d enters \n", mx2.iIter, mx2.cLeaves, mx2.cEnters);
      tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
      if( tlp_is_augmented(&mx2) ) printf("AUGMENTED\n"); else tlp_dump_current_soln(&mx2);
    }
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    if( !(rc = tlp_soln(&mx2, mxsol)) ) printf("%s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
  }

  return 0;
}
