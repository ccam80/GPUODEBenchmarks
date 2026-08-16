// Checks the analytic Jacobian against a central difference of the RHS, in double.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "header.h"
#include "dydt.h"
#include "jacob.h"

#define SAMPLES 8

static double frand(double lo, double hi)
{
    return lo + (hi - lo) * ((double)rand() / (double)RAND_MAX);
}

int main(void)
{
    srand(12345);
    areal y[NSP], jac[NSP * NSP], fp[NSP], fm[NSP];
    double worst = 0.0, worst_fd = 0.0, worst_an = 0.0;
    int worst_i = -1, worst_j = -1;

    for (int s = 0; s < SAMPLES; ++s) {
        const areal t = (areal)frand(0.0, PROBLEM_DURATION);
        const areal p = (areal)(PROBLEM_SWEEP_LOG
            ? pow(10.0, frand(log10((double)PROBLEM_SWEEP_MIN),
                              log10((double)PROBLEM_SWEEP_MAX)))
            : frand((double)PROBLEM_SWEEP_MIN, (double)PROBLEM_SWEEP_MAX));
        for (int i = 0; i < NSP; ++i) {
            y[i] = (areal)frand(-0.2, 0.2);
        }

        eval_jacob(t, p, y, jac);

        // Entries far below the matrix norm are unresolvable by a difference
        // quotient, so the comparison is relative to the norm as well.
        double jnorm = 0.0;
        for (int i = 0; i < NSP * NSP; ++i) {
            jnorm = fmax(jnorm, fabs((double)jac[i]));
        }
        const double floor_ = 1.0e-6 * jnorm;

        for (int j = 0; j < NSP; ++j) {
            const areal y0 = y[j];
            const areal h = (areal)(1.0e-6 * (1.0 + fabs((double)y0)));
            y[j] = y0 + h;
            dydt(t, p, y, fp);
            y[j] = y0 - h;
            dydt(t, p, y, fm);
            y[j] = y0;
            for (int i = 0; i < NSP; ++i) {
                const double fd = ((double)fp[i] - (double)fm[i])
                                  / (2.0 * (double)h);
                // jac[j * NSP + i] is d(f_i)/d(y_j)
                const double an = (double)jac[j * NSP + i];
                const double scale = fmax(fmax(fabs(fd), fabs(an)), floor_);
                if (scale <= 0.0) {
                    continue;
                }
                const double rel = fabs(fd - an) / scale;
                if (rel > worst) {
                    worst = rel;
                    worst_i = i;
                    worst_j = j;
                    worst_fd = fd;
                    worst_an = an;
                }
            }
        }
    }

    printf("%s: worst relative Jacobian error %.3e", PROBLEM_NAME, worst);
    if (worst_i >= 0) {
        printf(" at d(f%d)/d(y%d): fd %.6e, analytic %.6e",
               worst_i, worst_j, worst_fd, worst_an);
    }
    printf("\n");
    return worst < 1.0e-4 ? 0 : 1;
}
