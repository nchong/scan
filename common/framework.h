#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "utils.h"

std::ofstream file;

/* 
 * Exclusive scan on array [input] of length [n]
 */
void exclusive_scan_host(int *output, int *input, int n) {
  output[0] = 0;
  for (int i=1; i<n; i++) {
    output[i] = output[i-1] + input[i-1];
  }
}

/* check two arrays for equality */
bool check_results(int *expected_result, int *result, int n) {
  for (int i=0; i<n; i++) {
    if (expected_result[i] != result[i]) {
    //printf("[FAIL ] expected_result[%d] = %d\n", i, expected_result[i]);
    //printf("[FAIL ]          result[%d] = %d\n", i, result[i]);
      return false;
    }
  }
  return true;
}

struct options {
  bool verbose;
  bool debug;
  int wx;
  long seed;
};

/*
 * Run [num_iter] iterations of the scan operation.
 */
extern void run(int *data, int n, int num_iter, struct options &opt);

void print_usage(std::string progname) {
  printf("Usage: %s [options]\n", progname.c_str());
  printf("Options:\n");
  printf("   -v         be verbose\n");
  printf("   -d         print debug information\n");
  printf("   -n arg     size of input data\n");
  printf("   -w arg     size of workgroup for OpenCL implementations\n");
  printf("   -r arg     number of runs\n");
  printf("   -s seed    set seed for generating input data\n");
}


int main(int argc, char **argv) {

  file.open ("log.txt");

  // PARSE CMDLINE
  std::string progname(argv[0]);


  // problem size
  int n = 1024;
  int num_iter = 1000;
  // optional arguments
  struct options opt;
  opt.verbose = false;
  opt.debug = false;
  opt.wx = 256;
  opt.seed = -1;

  int c;
  while ((c = getopt (argc, argv, "hdvn:r:w:s:")) != -1) {
    switch (c) {
      case 'h':
        print_usage(progname);
        return 1;
      case 'd':
        opt.debug = true;
        break;
      case 'v':
        opt.verbose = true;
        break;
      case 'n':
        n = atoi(optarg);
        break;
      case 'r':
        num_iter = atoi(optarg);
        break;
      case 'w':
        opt.wx = atoi(optarg);
        break;
      case 's':
        opt.seed = atol(optarg);
        srandom(opt.seed);
        break;
      case '?':
        if (optopt == 'n' || optopt == 'r' || optopt == 's' || optopt == 'w')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr,
              "Unknown option character `\\x%x'.\n",
              optopt);
        break;//return 1;
      default:
        abort ();
    }
  }

  if (opt.debug) {
    printf ("# Command-line parsing: verbose=%d n=%d num_iter=%d seed=%ld\n",
        opt.verbose, n, num_iter, opt.seed);
    for (int i=optind; i<argc; i++)
      printf ("# Non-option argument: %s\n", argv[i]);
  }

  if (opt.verbose) {
    printf("# Program: %s\n", progname.c_str());
    printf("# N: %d\n", n);
    printf("# Num Iterations: %d\n", num_iter);
    if (opt.seed != -1) {
      printf("# Seed: %ld\n", opt.seed);
    }
  }

  //GENERATE RANDOM DATA
  int *data = new int[n];
  fill_random_data(data, n);

  int *expected_result = new int[n];
  exclusive_scan_host(expected_result, data, n);

  if (opt.verbose) {
    file << atos(data, n, "INPUT");
    file << atos(expected_result, n, "EXPECTED");
  }

  // RUN TEST
  run(data, n, num_iter, opt);

  if (opt.verbose) {
    file << atos(data, n, "RESULT");
  }

  // CHECK RESULTS
  bool pass = check_results(expected_result, data, n);
  printf("TEST %s\n", pass ? "PASSED" : "FAILED");

  // PRINT TIMING INFORMATION
  file.flush();
  file.close();

  // CLEANUP
  delete[] data;
  delete[] expected_result;

  return 0;
}
#endif
