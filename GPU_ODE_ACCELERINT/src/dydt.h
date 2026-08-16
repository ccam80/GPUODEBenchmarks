#ifndef DYDT_H
#define DYDT_H

#include "precision.h"

//! RHS for one problem instance with swept parameter p.
void dydt (const areal t, const areal p, const areal * __restrict__ y,
           areal * __restrict__ dy);

#endif
