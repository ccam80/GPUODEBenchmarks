#ifndef SPARSE_HEAD
#define SPARSE_HEAD

#include "header.h"

//! w := A * Vm, with A laid out as in eval_jacob(). Used by the exponential integrators.
void sparse_multiplier (const areal *, const areal *, areal *);

#endif
