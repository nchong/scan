#include "clwrapper.h"
#include "scan.h"
#include "scanref.h"
#include "utils.h"

#include "UnitTest++.h"

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

TEST(Random_1048576) {
  random_test(1048576, 128);
}

int main() {
  return UnitTest::RunAllTests();
}
