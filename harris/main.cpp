#include "clwrapper.h"
#include "framework.h"
#include "scan.h"

#include <cmath>
#include <cstring>

using namespace std;

void run(int *data, int n, int num_iter, map<string,float> &timings) {
  // PLATFORM AND DEVICE INFO
  if (opt.verbose) {
    cout << clinfo();
  }

  // BUILD PROGRAM AND KERNELS
  CLWrapper clw(/*platform=*/0,/*device=*/0,/*profiling=*/true);
  Scan *s = new Scan(clw, opt.wx);

  int *x = new int[n];
  for (int run=0; run<num_iter; run++) {
    memcpy(x, data, n*sizeof(int));
    s->scan(x, n);
  }
  memcpy(data, x, n*sizeof(int));
  delete[] x;

  s->get_timers(timings);
}
