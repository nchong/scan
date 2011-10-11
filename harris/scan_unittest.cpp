#include "cl_common.h"
#include "UnitTest++.h"
#include "scan.h"
#include "scan_reference.h"
#include "utils.h"

#define N 8

TEST(Simple) {
  CLWrapper clw(/*platform=*/0,/*device=*/0,/*profiling=*/true);
  Scan *s = new Scan(clw, /*wx=*/4);
  int x[N]            = { 3, 1, 7,  0,  4,  1,  6,  3 };
  const int result[N] = { 0, 3, 4, 11, 11, 15, 16, 22 };
  s->scan(x, N);
  CHECK_ARRAY_EQUAL(result, x, N);
}

void random_test(int n, int wx) {
  CLWrapper clw(/*platform=*/0,/*device=*/0,/*profiling=*/true);
  Scan *s = new Scan(clw, wx);
  int *x = new int[n];
  int *result = new int[n];
  fill_random_data(x, n, n);
  exclusive_scan_host(result, x, n);
  s->scan(x, n);
  CHECK_ARRAY_EQUAL(result, x, n);
  delete[] x;
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
