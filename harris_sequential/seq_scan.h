/*
 * Sequential versions of the upsweep/downsweep phases of a scan.
 */

#ifndef SEQ_SCAN_H
#define SEQ_SCAN_H

#include "log.h"
#include "utils.h"

#include <cmath>
#include <cstring>

#define PARANOID             false
#define UPSWEEP_POW2_PRINT   false
#define DOWNSWEEP_POW2_PRINT false
#define SCAN_ARB_PRINT       false

void inline upsweep_inner(int *x, int n, int d) {
  for (int k=0; k<n; k+=(int)pow((float)2,d+1)) {
    //printf("d %d k %d\n", d, k);
    int ai = k + (int)pow((float)2, d   ) - 1;
    int bi = k + (int)pow((float)2,(d+1)) -1;
    x[bi] = x[ai] + x[bi];
  }
}

void upsweep_pow2(int *x, int n) {
  int log2n = (int)log2(n);
  for (int d=0; d<log2n; d++) {
    upsweep_inner(x, n, d);
#if UPSWEEP_POW2_PRINT
    std::stringstream ss;
    ss << "UPSWEEP" << d;
    file << atos(x, n, ss.str());
#endif
  }
}

void inline downsweep_inner(int *x, int n, int d) {
  for (int k=0; k<n; k+=(int)pow((float)2,d+1)) {
    int ai = k + (int)pow((float)2, d   ) - 1;
    int bi = k + (int)pow((float)2,(d+1)) -1;
    int tmp = x[ai];
              x[ai] = x[bi];
                      x[bi] += tmp;
  }
}

void downsweep_pow2(int *x, int n) {
  x[(n-1)] = 0;
#if DOWNSWEEP_POW2_PRINT
  file << atos(x, n, "CLEAR");
#endif
  int log2n = (int)log2(n) - 1;
  for (int d=log2n; d>-1; d--) {
    downsweep_inner(x, n, d);
#if DOWNSWEEP_POW2_PRINT
    std::stringstream ss;
    ss << "DOWNSWEEP" << d;
    file << atos(x, n, ss.str());
#endif
  }
}

void scan_pow2(int *x, int n) {
  upsweep_pow2(x, n);
  downsweep_pow2(x, n);
}

void scan_pad_to_pow2(int *x, int n, int m) {
  int *xext = new int[m];
  memcpy(xext, x, sizeof(int)*n);
  memset(&xext[n], 0, sizeof(int)*(m-n));
#if PARANOID
  for (int i=0; i<m; i++) {
    if (i < n) assert(xext[i] == x[i]);
    else       assert(xext[i] == 0   );
  }
#endif
  upsweep_pow2(xext, m);
  downsweep_pow2(xext, m);
  memcpy(x, xext, sizeof(int)*n);
  delete[] xext;
}

inline void upsweep_subarrays(int *x, int k, int m) {
  for (int i=0; i<k; i++) {
    upsweep_pow2(&x[i*m], m);
  }
}

inline void get_partials(int *x, int *partials, int k, int m) {
  for (int i=0; i<k; i++) {
    partials[i] = x[(i*m)+(m-1)];
  }
}

inline void downsweep_subarrays(int *x, int k, int m) {
  for (int i=0; i<k; i++) {
    downsweep_pow2(&x[i*m], m);
  }
}

inline void add_partials(int *x, int *partials, int k, int m) {
  for (int i=0; i<k; i++) {
    for (int j=0; j<m; j++) {
      x[(i*m)+j] += partials[i];
    }
  }
}

void recursive_scan_arb(int *data, int n, int m, int callnum) {
  int k = (int) ceil((float)n/(float)m);
  if (k == 1) {
    scan_pad_to_pow2(data, n, m);
  } else {
    // extended array is composed of k subarrays
    int *x = new int[k*m];
    memcpy(x, data, sizeof(int)*n);
    memset(&x[n], 0, sizeof(int)*((k*m)-n));
#if PARANOID
    for (int i=0; i<k*m; i++) {
      if (i < n) assert(x[i] == data[i]);
      else       assert(x[i] == 0      );
    }
#endif

    // partials array of length k
    int *partials = new int[k];
    memset(partials, 0, sizeof(int)*k);

    // do subarray-wise scan
    upsweep_subarrays(x, k, m);
#if SCAN_ARB_PRINT
    file << atos(x, k*m, "SUBARRAY_UPSWEEP");
#endif

    get_partials(x, partials, k, m);
#if SCAN_ARB_PRINT
  file << atos(partials, k, "PARTIALS");
#endif

    downsweep_subarrays(x, k, m);
#if SCAN_ARB_PRINT
  file << atos(xext, k*m, "SUBARRAY_DOWNSWEEP");
#endif

    // do partials scan
    recursive_scan_arb(partials, k, m, callnum+1);

    // fold partial scan results back
    add_partials(x, partials, k, m);

    // copy result back and cleanup
    memcpy(data, x, sizeof(int)*n);
    delete[] x;
    delete[] partials;
  }
}
#endif
