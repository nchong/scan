#include "cl_common.h"
#include "scanref.h"
#include "segscan.h"
#include "utils.h"

#include "UnitTest++.h"
#include <cmath>

#define N 8

TEST(Simple) {
  CLWrapper clw(/*platform=*/0,/*device=*/0,/*profiling=*/true);
  SegmentedScan *ss = new SegmentedScan(clw, /*wx=*/4);
  int x[N]            = { 3, 1, 7, 0, 4, 1, 6, 3 };
  int f[N]            = { 1, 0, 1, 0, 0, 1, 0, 0 };
  const int result[N] = { 0, 3, 0, 7, 7, 0, 1, 7 };
  ss->scan(x, f, N);
  CHECK_ARRAY_EQUAL(result, x, N);
}

void random_test(int n, int wx) {
  CLWrapper clw(/*platform=*/0,/*device=*/0,/*profiling=*/true);
  SegmentedScan *ss = new SegmentedScan(clw, wx);
  int *x = new int[n];
  int *f = new int[n];
  int *result = new int[n];
  fill_random_data(x, n, n);
  fill_random_data(f, n, 2);
  segmented_exclusive_scan_host(result, x, f, n);
  ss->scan(x, f, n);
  CHECK_ARRAY_EQUAL(result, x, n);
  delete[] x;
  delete[] f;
  delete[] result;
}

TEST(Random_256) {
  random_test(256, 128);
}

TEST(Random_1024) {
  random_test(1024, 128);
}

int main() {
  return UnitTest::RunAllTests();
}
