#ifndef JACOB_H
#define JACOB_H

#include "precision.h"

//! Analytic Jacobian; only built when the problem supplies one.
void eval_jacob (const areal t, const areal p,
                 const areal * __restrict__ y, areal * __restrict__ jac);

#endif
