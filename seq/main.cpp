#include "framework.h"
#include "seq_scan.h"

void run(int *data, int n, int num_iter, struct options &opt) {
  int *copy = new int[n];
  for (int run=0; run<num_iter; run++) {
    memcpy(copy, data, sizeof(int)*n);
    scan_arb(copy, n, (int)opt.wx);
  }
  memcpy(data, copy, sizeof(int)*n);
  delete[] copy;
}
