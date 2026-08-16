// Initial conditions and the swept parameter grid.

#include <stdio.h>
#include <math.h>
#include "header.h"
#include "status.h"

//! numpy's linspace/logspace over the problem's sweep range, computed in double.
static void fill_default_sweep(int NUM, areal* p)
{
    if (NUM == 1) {
        p[0] = AC(PROBLEM_SWEEP_MIN);
        return;
    }
#if PROBLEM_SWEEP_LOG
    const double lo = log10((double)PROBLEM_SWEEP_MIN);
    const double hi = log10((double)PROBLEM_SWEEP_MAX);
#else
    const double lo = (double)PROBLEM_SWEEP_MIN;
    const double hi = (double)PROBLEM_SWEEP_MAX;
#endif
    const double step = (hi - lo) / (double)(NUM - 1);
    for (int i = 0; i < NUM; ++i) {
        const double v = lo + step * (double)i;
#if PROBLEM_SWEEP_LOG
        p[i] = (areal)pow(10.0, v);
#else
        p[i] = (areal)v;
#endif
    }
    p[NUM - 1] = AC(PROBLEM_SWEEP_MAX);
}

//! Read exactly NUM float64 sweep values from SWEEP_FILE; 1 on success.
static int read_sweep_file(int NUM, areal* p)
{
    FILE* f = fopen(SWEEP_FILE, "rb");
    if (f == NULL) {
        return 0;
    }
    double* buffer = (double*)malloc((size_t)NUM * sizeof(double));
    if (buffer == NULL) {
        fclose(f);
        return 0;
    }
    size_t got = fread(buffer, sizeof(double), (size_t)NUM, f);
    int extra = (fgetc(f) != EOF);
    fclose(f);
    if (got != (size_t)NUM || extra) {
        printf("Warning: %s holds the wrong number of values for NUM = %d;"
               " falling back to the generated grid\n", SWEEP_FILE, NUM);
        free(buffer);
        return 0;
    }
    for (int i = 0; i < NUM; ++i) {
        p[i] = (areal)buffer[i];
    }
    free(buffer);
    printf("# sweep grid: read from %s\n", SWEEP_FILE);
    return 1;
}

void set_same_initial_conditions(int NUM, areal** y_host, areal** var_host)
{
    status_init(NUM);
    (*y_host) = (areal*)malloc((size_t)NUM * NSP * sizeof(areal));
    (*var_host) = (areal*)malloc((size_t)NUM * sizeof(areal));
    if ((*y_host) == NULL || (*var_host) == NULL) {
        printf("Error: could not allocate state arrays for NUM = %d\n", NUM);
        exit(-1);
    }

    if (!read_sweep_file(NUM, *var_host)) {
        fill_default_sweep(NUM, *var_host);
        printf("# sweep grid: generated over [%g, %g], log = %d\n",
               (double)PROBLEM_SWEEP_MIN, (double)PROBLEM_SWEEP_MAX,
               PROBLEM_SWEEP_LOG);
    }

    areal y0[NSP];
    problem_initial_state(y0);
    // the global array is column major
    for (int i = 0; i < NUM; ++i) {
        for (int s = 0; s < NSP; ++s) {
            (*y_host)[i + s * NUM] = y0[s];
        }
    }
}

void apply_mask(areal* y_host) {}
void apply_reverse_mask(areal* y_host) {}
