#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include "utils.h"
#include "scanref.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>

using namespace std;

ofstream file;

struct options {
  bool verbose;
  bool debug;
  int wx;
  long seed;
};
struct options opt;

/*
 * Run [num_iter] iterations of the scan operation.
 */
#if SEGMENTED
extern void run(int *data, int *flag, int n, int num_iter, map<string,float> &timings);
#else
extern void run(int *data, int n, int num_iter, map<string,float> &timings);
#endif

void print_usage(string progname) {
  printf("Usage: %s [options]\n", progname.c_str());
  printf("Options:\n");
  printf("   -v         be verbose\n");
  printf("   -d         print debug information\n");
  printf("   -n arg     size of input data\n");
  printf("   -w arg     size of workgroup for OpenCL implementations\n");
  printf("   -r arg     number of runs\n");
  printf("   -s seed    set seed for generating input data\n");
}

struct _max_str {
  int operator()(int max, map<string,float>::value_type &item) {
    int len = item.first.length();
    return (len > max ? len : max);
  }
} max_str_in_key;
string print_timings(map<string,float> timings, int num_iter) {
  // table widths
  int w0 = 1 + accumulate(timings.begin(), timings.end(), 7, max_str_in_key);
  int w1 = 11;
  int w2 = 15;

  stringstream ss;
  ss << left;
  ss << fixed;
  ss << setprecision(3);
  float total_in_ms = 0;
  map<string,float>::iterator i;
  ss << setw(w0) << "# TASK";
  ss << setw(w1) << "TIME (ms)";
  ss << setw(w2) << "PER-ITER (ms)";
  ss << endl;
  for (i = timings.begin(); i != timings.end(); i++) {
    string s = i->first;
    float  t = i->second;
    ss << setw(w0) << s;
    ss << setw(w1) << t;
    ss << setw(w2) << t/num_iter;
    ss << endl;
    total_in_ms += t;
  }
  ss << setw(w0) << "TOTAL";
  ss << setw(w1) << total_in_ms;
  ss << setw(w2) << total_in_ms/num_iter;
  ss << endl;
  return ss.str();
}

int main(int argc, char **argv) {

  file.open ("log.txt");

  // PARSE CMDLINE
  string progname(argv[0]);

  // problem size
  int n = 1024;
  int num_iter = 1000;
  // optional arguments
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
  fill_random_data(data, n, n);
#if SEGMENTED
  int *flag = new int[n];
  fill_random_data(flag, n, 2);
  flag[0] = 1;
#endif

  int *expected_result = new int[n];
#if SEGMENTED
  segmented_exclusive_scan_host(expected_result, data, flag, n);
#else
  exclusive_scan_host(expected_result, data, n);
#endif

  if (opt.verbose) {
    file << atos(data, n, "DATA");
#if SEGMENTED
    file << atos(flag, n, "FLAG");
#endif
    file << atos(expected_result, n, "EXPECTED");
  }

  // RUN TEST
  map<string,float> timings;
#if SEGMENTED
  run(data, flag, n, num_iter, timings);
#else
  run(data, n, num_iter, timings);
#endif

  if (opt.verbose) {
    file << atos(data, n, "RESULT");
  }

  // CHECK RESULTS
  bool pass = check_results(expected_result, data, n);
  if (!pass) {
    cout << "# ***TEST FAILED***" << endl;
  } else if (opt.verbose) {
    cout << "# ***TEST PASSED***" << endl;
  }

  // PRINT TIMING INFORMATION
  cout << print_timings(timings, num_iter);

  // FLUSH FILE OUTPUT
  file.flush();
  file.close();

  // CLEANUP
  delete[] data;
#if SEGMENTED
  delete[] flag;
#endif
  delete[] expected_result;

  return 0;
}
#endif
