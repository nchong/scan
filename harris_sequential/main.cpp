#include "framework.h"
#include "seq_scan.h"

void run(int *data, int n, int num_iter, map<string,float> &/* timings unused*/) {
  int *copy = new int[n];
  for (int run=0; run<num_iter; run++) {
    memcpy(copy, data, sizeof(int)*n);
    recursive_scan_arb(copy, n, (int)opt.wx, 0);
  }
  memcpy(data, copy, sizeof(int)*n);
  delete[] copy;
}
