// Lorenz; rho is swept. Mirrors GPU_ODE_MPGOS/problems/lorenz.cuh.

#ifndef PROBLEM_LORENZ_H
#define PROBLEM_LORENZ_H

#include "precision.h"

#define PROBLEM_NAME "lorenz"
#define PROBLEM_SD 3
#define PROBLEM_DURATION 1.0
#define PROBLEM_SWEEP_MIN 0.0
#define PROBLEM_SWEEP_MAX 21.0
#define PROBLEM_SWEEP_LOG 0
#define PROBLEM_HAS_ANALYTIC_JACOBIAN 1

#define LZ_SIGMA AC(10.0)
#define LZ_BETA AC(8.0 / 3.0)

static inline void problem_rhs(const areal t, const areal rho,
                               const areal * __restrict__ y,
                               areal * __restrict__ dy)
{
    dy[0] = LZ_SIGMA * (y[1] - y[0]);
    dy[1] = y[0] * (rho - y[2]) - y[1];
    dy[2] = y[0] * y[1] - LZ_BETA * y[2];
}

//! jac[i * PROBLEM_SD + j] is d(dy_j)/d(y_i).
static inline void problem_jacobian(const areal t, const areal rho,
                                    const areal * __restrict__ y,
                                    areal * __restrict__ jac)
{
    jac[0] = -LZ_SIGMA;   jac[1] = rho - y[2];  jac[2] = y[1];
    jac[3] = LZ_SIGMA;    jac[4] = AC(-1.0);    jac[5] = y[0];
    jac[6] = AC(0.0);     jac[7] = -y[0];       jac[8] = -LZ_BETA;
}

static inline void problem_initial_state(areal * y)
{
    y[0] = AC(1.0);
    y[1] = AC(0.0);
    y[2] = AC(0.0);
}

#endif
