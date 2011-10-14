#include "clwrapper.h"
#define SEGMENTED true
#include "framework.h"
#include "segscan.h"

#include <cmath>

using namespace std;

void run(int *data, int *flag, int n, int num_iter, map<string,float> &timings) {
  // PLATFORM AND DEVICE INFO
  if (opt.verbose) {
    cout << clinfo();
  }

  // BUILD PROGRAM AND KERNELS
  CLWrapper clw(/*platform=*/0,/*device=*/0,/*profiling=*/true);
  SegmentedScan *ss = new SegmentedScan(clw, opt.wx);

  // RUN TEST
  int *x = new int[n];
  for (int run=0; run<num_iter; run++) {
    memcpy(x, data, n*sizeof(int));
    ss->scan(x, flag, n);
  }
  memcpy(data, x, n*sizeof(int));
  delete[] x;

  // INSERT TIMINGS
  ss->get_timers(timings);
}
