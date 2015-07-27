#include "omp.h"
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>


typedef unsigned int uint;
typedef double complex cdplx;


/* Find |z_{k+1}| = z_k^2 + c such that |z_{k+1}| >= lim or k >= N for each
 * pixel */
uint core(cdplx z, cdplx c, uint lim, uint cutoff) {
    uint cnt = 0;
    while (cabs(z) < lim && cnt < cutoff) {
        z = z * z + c;
        cnt += 1;
    };
    return cnt;
};


/* Compute the julia set */
void compute(uint **data, uint size, cdplx c,
             double bound, uint lim, uint cutoff) {

    uint i, j;
    double delta = 2.0 * bound / (size - 1);

    omp_set_num_threads(2);
    #pragma omp parallel for private(i, j)
    for (i=0; i<size; i++) {
        double re = -(double)bound + i * delta;
        for (j=0; j<size; j++) {
            double im = -(double)bound + j * delta;
            cdplx z = re + I * im;
            data[i][j] = core(z, c, lim, cutoff);
        };
    };
};