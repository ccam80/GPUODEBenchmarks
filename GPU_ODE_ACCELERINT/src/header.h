// System size and setup declarations, filled in by the selected problem header.

#ifndef HEADER_GUARD_H
#define HEADER_GUARD_H

#include <stdlib.h>
#include "precision.h"

#ifdef _OPENMP
 #include <omp.h>
#else
 #define omp_get_max_threads() 1
 #define omp_get_num_threads() 1
#endif

#if defined(PROBLEM_RING_MODULATOR)
 #include "ring_modulator.h"
#elif defined(PROBLEM_LORENZ)
 #include "lorenz.h"
#else
 #error "define one of PROBLEM_LORENZ, PROBLEM_RING_MODULATOR"
#endif

//! IVP system size
#define NSP (PROBLEM_SD)
//! Input vector size in read_initial_conditions
#define NN (NSP)

//! Binary file of NUM float64 sweep values, used in place of the generated grid
#define SWEEP_FILE "sweep.bin"

//! Fill y_host (column-major, NUM x NSP) with the ICs and var_host with the sweep grid.
void set_same_initial_conditions(int NUM, areal** y_host, areal** var_host);

//! No-op; the pyJac path expects these.
void apply_mask(areal* y_host);
void apply_reverse_mask(areal* y_host);

#endif
