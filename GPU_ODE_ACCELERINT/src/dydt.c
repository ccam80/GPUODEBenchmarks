// y and dy are per-problem copies unpacked by intDriver().

#include "header.h"
#include "dydt.h"

void dydt (const areal t, const areal p, const areal * __restrict__ y,
           areal * __restrict__ dy)
{
    problem_rhs(t, p, y, dy);
}
