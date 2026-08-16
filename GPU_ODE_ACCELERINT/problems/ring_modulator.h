// Ring modulator (Test Set for IVP Solvers II-3); Cs is swept.
// Mirrors GPU_ODE_MPGOS/problems/ring_modulator.cuh.

#ifndef PROBLEM_RING_MODULATOR_H
#define PROBLEM_RING_MODULATOR_H

#include "precision.h"

#define PROBLEM_NAME "ring_modulator"
#define PROBLEM_SD 15
#define PROBLEM_DURATION 1.0e-3
#define PROBLEM_SWEEP_MIN 2.0e-13
#define PROBLEM_SWEEP_MAX 2.0e-9
#define PROBLEM_SWEEP_LOG 1
#define PROBLEM_HAS_ANALYTIC_JACOBIAN 1

#define RM_C AC(1.6e-8)
#define RM_CP AC(1.0e-8)
#define RM_LH AC(4.45)
#define RM_LS1 AC(0.002)
#define RM_LS2 AC(5.0e-4)
#define RM_LS3 AC(5.0e-4)
#define RM_GAMMA AC(40.67286402e-9)
#define RM_R AC(25000.0)
#define RM_RP AC(50.0)
#define RM_RG1 AC(36.3)
#define RM_RG2 AC(17.3)
#define RM_RG3 AC(17.3)
#define RM_RI AC(50.0)
#define RM_RC AC(600.0)
#define RM_DELTA AC(17.7493332)
#define RM_W1 AC(6283.185307179586)
#define RM_W2 AC(62831.85307179586)

static inline void problem_rhs(const areal t, const areal cs,
                               const areal * __restrict__ X,
                               areal * __restrict__ F)
{
    const areal uin1 = AC(0.5) * sin(RM_W1 * t);
    const areal uin2 = AC(2.0) * sin(RM_W2 * t);
    const areal ud1 =  X[2] - X[4] - X[6] - uin2;
    const areal ud2 = -X[3] + X[5] - X[6] - uin2;
    const areal ud3 =  X[3] + X[4] + X[6] + uin2;
    const areal ud4 = -X[2] - X[5] + X[6] + uin2;
    const areal q1 = RM_GAMMA * (exp(RM_DELTA * ud1) - AC(1.0));
    const areal q2 = RM_GAMMA * (exp(RM_DELTA * ud2) - AC(1.0));
    const areal q3 = RM_GAMMA * (exp(RM_DELTA * ud3) - AC(1.0));
    const areal q4 = RM_GAMMA * (exp(RM_DELTA * ud4) - AC(1.0));

    F[0]  = (X[7] - AC(0.5) * X[9] + AC(0.5) * X[10] + X[13] - X[0] / RM_R) / RM_C;
    F[1]  = (X[8] - AC(0.5) * X[11] + AC(0.5) * X[12] + X[14] - X[1] / RM_R) / RM_C;
    F[2]  = (X[9] - q1 + q4) / cs;
    F[3]  = (-X[10] + q2 - q3) / cs;
    F[4]  = (X[11] + q1 - q3) / cs;
    F[5]  = (-X[12] - q2 + q4) / cs;
    F[6]  = (-X[6] / RM_RP + q1 + q2 - q3 - q4) / RM_CP;
    F[7]  = -X[0] / RM_LH;
    F[8]  = -X[1] / RM_LH;
    F[9]  = (AC(0.5) * X[0] - X[2] - RM_RG2 * X[9]) / RM_LS2;
    F[10] = (AC(-0.5) * X[0] + X[3] - RM_RG3 * X[10]) / RM_LS3;
    F[11] = (AC(0.5) * X[1] - X[4] - RM_RG2 * X[11]) / RM_LS2;
    F[12] = (AC(-0.5) * X[1] + X[5] - RM_RG3 * X[12]) / RM_LS3;
    F[13] = (-X[0] + uin1 - (RM_RI + RM_RG1) * X[13]) / RM_LS1;
    F[14] = (-X[1] - (RM_RC + RM_RG1) * X[14]) / RM_LS1;
}

//! jac[i * PROBLEM_SD + j] is d(F_j)/d(X_i).
static inline void problem_jacobian(const areal t, const areal cs,
                                    const areal * __restrict__ X,
                                    areal * __restrict__ jac)
{
    const areal uin2 = AC(2.0) * sin(RM_W2 * t);
    const areal ud1 =  X[2] - X[4] - X[6] - uin2;
    const areal ud2 = -X[3] + X[5] - X[6] - uin2;
    const areal ud3 =  X[3] + X[4] + X[6] + uin2;
    const areal ud4 = -X[2] - X[5] + X[6] + uin2;
    // d(qk)/d(udk)
    const areal e1 = RM_GAMMA * RM_DELTA * exp(RM_DELTA * ud1);
    const areal e2 = RM_GAMMA * RM_DELTA * exp(RM_DELTA * ud2);
    const areal e3 = RM_GAMMA * RM_DELTA * exp(RM_DELTA * ud3);
    const areal e4 = RM_GAMMA * RM_DELTA * exp(RM_DELTA * ud4);

    for (int i = 0; i < PROBLEM_SD * PROBLEM_SD; ++i) {
        jac[i] = AC(0.0);
    }
#define J(i, j) jac[(i) * PROBLEM_SD + (j)]

    J(0, 0)  = AC(-1.0) / (RM_R * RM_C);
    J(0, 7)  = AC(-1.0) / RM_LH;
    J(0, 9)  = AC(0.5) / RM_LS2;
    J(0, 10) = AC(-0.5) / RM_LS3;
    J(0, 13) = AC(-1.0) / RM_LS1;

    J(1, 1)  = AC(-1.0) / (RM_R * RM_C);
    J(1, 8)  = AC(-1.0) / RM_LH;
    J(1, 11) = AC(0.5) / RM_LS2;
    J(1, 12) = AC(-0.5) / RM_LS3;
    J(1, 14) = AC(-1.0) / RM_LS1;

    J(2, 2) = (-e1 - e4) / cs;
    J(2, 4) = e1 / cs;
    J(2, 5) = -e4 / cs;
    J(2, 6) = (e1 + e4) / RM_CP;
    J(2, 9) = AC(-1.0) / RM_LS2;

    J(3, 3)  = (-e2 - e3) / cs;
    J(3, 4)  = -e3 / cs;
    J(3, 5)  = e2 / cs;
    J(3, 6)  = (-e2 - e3) / RM_CP;
    J(3, 10) = AC(1.0) / RM_LS3;

    J(4, 2)  = e1 / cs;
    J(4, 3)  = -e3 / cs;
    J(4, 4)  = (-e1 - e3) / cs;
    J(4, 6)  = (-e1 - e3) / RM_CP;
    J(4, 11) = AC(-1.0) / RM_LS2;

    J(5, 2)  = -e4 / cs;
    J(5, 3)  = e2 / cs;
    J(5, 5)  = (-e2 - e4) / cs;
    J(5, 6)  = (e2 + e4) / RM_CP;
    J(5, 12) = AC(1.0) / RM_LS3;

    J(6, 2) = (e1 + e4) / cs;
    J(6, 3) = (-e2 - e3) / cs;
    J(6, 4) = (-e1 - e3) / cs;
    J(6, 5) = (e2 + e4) / cs;
    J(6, 6) = (AC(-1.0) / RM_RP - e1 - e2 - e3 - e4) / RM_CP;

    J(7, 0) = AC(1.0) / RM_C;
    J(8, 1) = AC(1.0) / RM_C;

    J(9, 0)  = AC(-0.5) / RM_C;
    J(9, 2)  = AC(1.0) / cs;
    J(9, 9)  = -RM_RG2 / RM_LS2;

    J(10, 0)  = AC(0.5) / RM_C;
    J(10, 3)  = AC(-1.0) / cs;
    J(10, 10) = -RM_RG3 / RM_LS3;

    J(11, 1)  = AC(-0.5) / RM_C;
    J(11, 4)  = AC(1.0) / cs;
    J(11, 11) = -RM_RG2 / RM_LS2;

    J(12, 1)  = AC(0.5) / RM_C;
    J(12, 5)  = AC(-1.0) / cs;
    J(12, 12) = -RM_RG3 / RM_LS3;

    J(13, 0)  = AC(1.0) / RM_C;
    J(13, 13) = -(RM_RI + RM_RG1) / RM_LS1;

    J(14, 1)  = AC(1.0) / RM_C;
    J(14, 14) = -(RM_RC + RM_RG1) / RM_LS1;

#undef J
}

static inline void problem_initial_state(areal * X)
{
    for (int i = 0; i < PROBLEM_SD; ++i) {
        X[i] = AC(0.0);
    }
}

#endif
