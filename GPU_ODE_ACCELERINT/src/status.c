// Replaces accelerInt's check_error, which exits the process on the first
// failing trajectory. A trajectory that fails to converge is a result, so the
// code is recorded and the run continues.

#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "radau2a_props.h"
#include "status.h"

#define STATUS_CODES 5

int* status_codes = NULL;
static int status_num = 0;

static const char* status_name(int code)
{
    switch (code) {
        case EC_success: return "success";
        case EC_consecutive_steps: return "too many consecutive failed steps";
        case EC_max_steps_exceeded: return "max steps exceeded";
        case EC_h_plus_t_equals_h: return "stepsize underflow (t + h == t)";
        case EC_newton_max_iterations_exceeded: return "newton iteration limit";
        default: return "unknown";
    }
}

static void status_report(void)
{
    if (status_codes == NULL) {
        return;
    }
    int counts[STATUS_CODES] = {0};
    int other = 0;
    for (int i = 0; i < status_num; ++i) {
        const int code = status_codes[i];
        if (code >= 0 && code < STATUS_CODES) {
            counts[code] += 1;
        } else {
            other += 1;
        }
    }
    printf("# status: %d/%d converged\n", counts[EC_success], status_num);
    for (int code = 1; code < STATUS_CODES; ++code) {
        if (counts[code]) {
            printf("#   %d %s\n", counts[code], status_name(code));
        }
    }
    if (other) {
        printf("#   %d unknown\n", other);
    }
    FILE* f = fopen(STATUS_FILE, "wb");
    if (f != NULL) {
        fwrite(status_codes, sizeof(int), (size_t)status_num, f);
        fclose(f);
    }
}

void status_init(int NUM)
{
    status_codes = (int*)calloc((size_t)NUM, sizeof(int));
    if (status_codes == NULL) {
        printf("Error: could not allocate the status array for NUM = %d\n", NUM);
        exit(-1);
    }
    status_num = NUM;
    atexit(status_report);
}

void check_error(int tid, int code)
{
    if (status_codes != NULL && tid >= 0 && tid < status_num) {
        status_codes[tid] = code;
    }
}
