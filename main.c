// linear program solver (simplex method), orthopteroid@gmail.com, BSD2 license
// based on "Operations Research", Hiller & Lieberman (1974)

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "tinylp.h"

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
  struct MXInfo mx2;
  TLP_RCCODE rc;
  TLP_UINT e, r, c;

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

  return 0;
}
