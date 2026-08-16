// Working type, matching whichever precision accelerInt was built at.

#ifndef PRECISION_H
#define PRECISION_H

#include <tgmath.h>

#ifdef ACC_DOUBLE
typedef double areal;
#else
typedef float areal;
#endif

//! Round a literal to the working type at compile time.
#define AC(x) ((areal)(x))

#endif
