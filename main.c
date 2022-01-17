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
  struct MXInfo mx2;
  TLP_RCCODE rc;
  TLP_UINT e, r, c;

  printf("----- SVM minimize. line %d\n", __LINE__);
  {
    // Patrick Winston mit opencourseware 6.034 2010
    // [1] Y? = sign( W dot X? + B )
    // [2] Yi ( W Xi + B ) -1 = 0, Xi in gutter constraint [???]
    // [3] L = .5 || W || ^2 - sigma( ALPHAi ( Yi ( W dot Xi + B ) - 1 ) ), using lagrange's method
    // [3.1] dL/dW = W - sigma( ALPHAi Yi Xi ), [3.1] = 0 at extreema
    // [3.2] dL/dB = sigma( Xi Yi ), [3.2] = 0 at extreema
    // [4] maximze L = sigma( ALPHAi ) - .5 sigma sigma( ALPHAi ALPHAj Yi Yj ( Xi dot Xj ) ) s.t. 0 <= ALPHAi <= C
    // [4.1] maximize L = sigma( ALPHAi ) - .5 sigma sigma( ALPHAi ALPHAj Yi Yj K( Xi, Xj ) ), using kernel K()
    // kernels: linear, quadratic/conics (parabola, circle, hyperbola), polynomial, radial basis
    // [5.1] ( Xi dot Xj + 1 ) ^ N
    // [5.2] e ^ ( - || Xi - Xj || / SIGMA ), radial basis function

    // CHRISTOPHER J.C. BURGES lucent 1998
    // https://people.csail.mit.edu/dsontag/courses/ml14/notes/burges_SVM_tutorial.pdf
    // [1] L = .5 || W || ^2 - sigma( ALPHAi Yi ( W dot Xi + B ) ) - sigma( ALPHAi ), vs [Winston 3.0]
    // [] [Winston 4.0] can use Wolfe

    // Jessica Noss mit opencourseware 6.034 2016
    // Yi = { + , - }, is the classification of vector i
    // ALPHAi >= 0, is the supportiveness of vector i to determine the boundry
    // now solve for ALPHAi...
    // [1.1] sigma( ALPHAi ) = sigma( ALPHAj ), where i are + support vectors and j are - support vectors
    // [1.2] sigma( ALPHA+ ) = sigma( ALPHA- ), where ALPHA+ and ALPHA- are coefs for the + and - support vectors
    // [1.3] sigma( ALPHAk ) = 0, when k are not support vectors (ie not in gutter)
    // [2.1] sigma( ALPHAi Yi Xi ) = W, over all support vectors i
    // [2.2] sigma( ALPHA+ X+ ) - simga( ALPHA- X- ) = W
    // use 2.1 and 2.2 to solve for ALPHAi [do you need to know W?]
    // insights about ALPHA values:
    // [3.1] since 2 / || W || = width of W, small ALPHAs are associated with larger margins
    // [3.2] in arrangements of non-equidistant support vectors, ALPHAs will be larger for closer support vectors of differing classes
    // calculation of similarity:
    // [4.1] Y? = sign( sigma( ALPHAi Yi Xi ) dot X? + B ), [sigma brackets correct?]
    // [4.2] Y? = sign( sigma( ALPHAi Yi K( Xi, X? ) ) + B ), using kernel K()

    // Zisserman ox.ac.uk C19 2015
    // QLP nuts and bolts: https://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf
    // https://www.robots.ox.ac.uk/~az/lectures/ml/lect3.pdf
    // [1.1] f(X) = ( W transpose X ) + B, is primal problem classification test
    // [1.2] f(X) = sigma( ALPHAi Yi ( Xi transpose X ) + B ), is dual problem classification test, with Xi being support vectors
    // [2] efficiency considerations
    // [2.1] since W is D dimensional, techniques based upon ALPHA as variables can be more efficient with higher dimensions
    // [2.2] techniques such as [1.1] or [1.2] use the ( X transpose X ) form that is kernel efficient
    // [3] determines ALPHAi using the learning function
    // [3.1] maximize sigma( ALPHA+ ) - .5 sigma( ALPHAj ALPHAk Yj Yk ( Xj transpose Xk ) ) s.t. 0 <= ALPHAi <= C and sigma( ALPHAi Yi ) = 0
    // [3.2] minimize sigma( ALPHAi ALPHAj Yi Yj ( Xi transpose Xj ) ) s.t. Yi sigma( ALPHAj Yj ( Xj transpose Xi ) + B ) >= 1 forall i
    // [4] test X class using [1.2]

    // libsvm: Sequential Minimal Optimization (SMO) (Platt, 1998)
    // https://www.csie.ntu.edu.tw/~cjlin/libsvm/
    // https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
    // https://www.csie.ntu.edu.tw/~cjlin/papers/quadworkset.pdf
    // SMO implementation: https://github.com/FazelYU/SVM-for-Genetic-Algorithm/blob/5f55190fb253f4a6f6c841b87cd1befbeda6ea7d/libsvm-master/svm.cpp#L375
    // https://dsmilab.github.io/Yuh-Jye-Lee/assets/file/teaching/2017_machine_learning/SMO_algorithm.pdf
    // https://www.csie.ntu.edu.tw/~cjlin/libsvm/
    // https://github.com/cjlin1/libsvm

    // gpusvm: https://home.ttic.edu/~cotter/projects/gtsvm/
    // Stochastic Batch Perceptron svm: https://home.ttic.edu/~cotter/projects/SBP/
    // https://github.com/LihO/SVMLightClassifier/blob/master/src/svm_learn.c
    // https://github.com/FazelYU/SVM-for-Genetic-Algorithm/blob/main/libsvm-master/svm-toy/qt/svm-toy.cpp
    // http://pages.cs.wisc.edu/~swright/talks/sjw-complearning.pdf
  }

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
    while( 1 )
    {
      rc = tlp_pivot( &mx2 );
      if( tlp_rc_decode(rc) != TLP_UNFINISHED ) break;
      printf("iteration %d: column %d leaves %d enters \n", mx2.iIter, mx2.cLeaves, mx2.cEnters);

      tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
      tlp_dump_current_soln(&mx2); // for QP, result is always augmented
    }
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );

    //tlp_dump_tableau(&mx2, TLP_BADINDEX, TLP_BADINDEX);
    //rc = tlp_mxequal(mx_ver, mx_sol, 1e-3, 5); CHECK;
    if( tlp_rc_decode(rc) == TLP_OSCILLATION ) tlp_dump_osc_history(&mx2);
    rc = tlp_soln(&mx2, mxsol);
    printf("tlp_soln %s\n", tlp_messages[ tlp_rc_decode(rc) ] );

    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double));
    printf("tlp_mxequal %s\n", tlp_messages[ tlp_rc_decode(rc) ] );

    //double objval = tlp_eval_qlp(&mx2, mxobj); // TODO

    rc = tlp_fini(&mx2); CHECK;
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
      tlp_dump_current_soln(&mx2);
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

    double *pmx = 0;
    double mxsol[] = { 0, 0, 0, 0, 0, };
    double mxver[] = { -0.7907, 2.2093, 1.3372, -0.4070, 1.6977 }; // L1 L2 x1 x2 x3

    rc = qeq_setup_min( &pmx, mxobj, 3, mxEQ, 2 ); CHECK;
    tlp_dump_mx( pmx, 5, 6 );
    rc = gauss( mxsol, pmx, 5 );
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    tlp_dump_mx( pmx, 5, 6 );
    tlp_dump_mx( mxsol, 1, 5 );
    rc = tlp_mxequal( mxver, mxsol, 1e-3, 5 ); CHECK;
    rc = qeq_fin(&pmx);
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
    while( 1 )
    {
      rc = tlp_pivot( &mx2 );
      if( tlp_rc_decode(rc) != TLP_UNFINISHED ) break;
      printf("iteration %d: column %d leaves %d enters \n", mx2.iIter, mx2.cLeaves, mx2.cEnters);

      tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
      tlp_dump_current_soln(&mx2);
    }
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    if( tlp_rc_decode(rc) == TLP_OSCILLATION ) tlp_dump_osc_history(&mx2);
    if( !(rc = tlp_soln(&mx2, mxsol)) ) printf("%s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
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
    while( 1 )
    {
      rc = tlp_pivot( &mx2 );
      if( tlp_rc_decode(rc) != TLP_UNFINISHED ) break;
      printf("iteration %d: column %d leaves %d enters \n", mx2.iIter, mx2.cLeaves, mx2.cEnters);

      tlp_dump_tableau( &mx2, TLP_BADINDEX, TLP_BADINDEX );
      tlp_dump_current_soln(&mx2);
    }
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    if( tlp_rc_decode(rc) == TLP_OSCILLATION ) tlp_dump_osc_history(&mx2);
    if( !(rc = tlp_soln(&mx2, mxsol)) ) printf("%s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
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
      tlp_dump_current_soln(&mx2);
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
      tlp_dump_current_soln(&mx2);
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
      tlp_dump_current_soln(&mx2);
    }
    printf("status: %s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    if( !(rc = tlp_soln(&mx2, mxsol)) ) printf("%s\n", tlp_messages[ tlp_rc_decode(rc) ] );
    rc = tlp_mxequal(mxsol, mxver, mx2.fZero, sizeof(mxsol) / sizeof(double)); CHECK;
    rc = tlp_fini(&mx2); CHECK;
  }

  return 0;
}
