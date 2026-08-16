#include "sparse_multiplier.h"

void sparse_multiplier(const areal * A, const areal * Vm, areal * w)
{
    for (int j = 0; j < NSP; ++j) {
        areal acc = AC(0.0);
        for (int i = 0; i < NSP; ++i) {
            acc += A[i * NSP + j] * Vm[i];
        }
        w[j] = acc;
    }
}
