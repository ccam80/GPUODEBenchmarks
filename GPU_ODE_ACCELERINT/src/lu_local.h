#ifndef LU_LOCAL_H
#define LU_LOCAL_H

#include <complex.h>

void sgetrf_(const int*, const int*, float*, const int*, int*, int*);
void dgetrf_(const int*, const int*, double*, const int*, int*, int*);
void cgetrf_(const int*, const int*, float complex*, const int*, int*, int*);
void zgetrf_(const int*, const int*, double complex*, const int*, int*, int*);

void sgetrs_(const char*, const int*, const int*, float*, const int*, int*,
             float*, const int*, int*);
void dgetrs_(const char*, const int*, const int*, double*, const int*, int*,
             double*, const int*, int*);
void cgetrs_(const char*, const int*, const int*, float complex*, const int*,
             int*, float complex*, const int*, int*);
void zgetrs_(const char*, const int*, const int*, double complex*, const int*,
             int*, double complex*, const int*, int*);

#endif
