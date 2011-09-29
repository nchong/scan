#ifndef UTILS_H
#define UTILS_H

#include <sstream>

/* 
 * Convert array [data] of length [n] to string
 * We output an array literal format for python.
 */
std::string atos(int *data, int n, std::string var="x") {
  std::stringstream ss;
  ss << var << " = [ " << data[0];
  for (int i=1; i<n; i++) {
    ss << ", " << data[i];
  }
  ss << "]" << std::endl;
  return ss.str();
}

/* 
 * Get a random integer in [0..n) 
 */
int rand_int(int n) {
  int limit = RAND_MAX - RAND_MAX % n;
  int rnd;

  do {
    rnd = random();
  } while (rnd >= limit);
  return rnd % n;
}

/* 
 * Fill an array of length [n] with random integers from rand_int()
 */
void fill_random_data(int *data, int n) {
  for (int i=0; i<n; i++) {
    data[i] = rand_int(n);
  }
}

bool ispow2(int x) {
  return (x>0 && ((x & (x - 1)) == 0));
}

#endif
