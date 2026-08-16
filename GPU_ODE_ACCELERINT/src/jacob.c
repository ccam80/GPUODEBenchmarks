// Column-major: jac[i * NSP + j] is d(dy_j)/d(y_i).

#include "header.h"
#include "jacob.h"

#ifdef PROBLEM_HAS_ANALYTIC_JACOBIAN
void eval_jacob (const areal t, const areal p,
                 const areal * __restrict__ y, areal * __restrict__ jac)
{
    problem_jacobian(t, p, y, jac);
}
#endif
