#include "cl_common.h"
#include "framework.h"

#define DEBUG false

using namespace std;

float k0 = 0; float k1 = 0; float k2 = 0;
cl_kernel scan_subarrays;
cl_kernel scan_pad_to_pow2;
cl_kernel scan_inc_subarrays;
void recursive_scan(CLWrapper &clw, cl_mem d_data, int n, int callnum) {
  // KERNEL INDEX SPACE
  // Split the array into [k] subarrays each of length m = 2*wx, where wx is the size of each workgroup.
  size_t wx = (size_t) opt.wx;
  int m = (int) wx * 2;
  int k = (int) ceil((float)n/(float)m);
  // This gives a global size of [gx = k*wx] workitems
  size_t gx = k * wx;
  if (opt.verbose) {
    printf("# RECURSIVE SCAN (CALL %d)\n", callnum);
    printf("# \tLength of array (n)             = %d\n", n);
    printf("# \tWorkgroup size (wx)             = %u\n", (int)wx);
    printf("# \tLength of local subarrays (m)   = %d\n", m);
    printf("# \tNumber of workgroups (k)        = %d\n", k);
    printf("# \tGlobal number of workitems (gx) = %d\n", (int)gx);
  }

  // RECURSIVE SCAN
  // base case: do the scan in one workgroup.
  if (k == 1) {
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_pad_to_pow2, 0, sizeof(cl_mem), (void *)&d_data));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_pad_to_pow2, 1, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_pad_to_pow2, 2, sizeof(int), &n));
    k0 += clw.run_kernel_with_timing("scan_pad_to_pow2", /*dim=*/1, &wx, &wx);

  // otherwise:
  //   do a scan within each subarray gathering a partials array,
  //   perform a scan on the partials array, and
  //   increment each subarray accordingly.
  } else {
    // DEVICE MEMORY
    // [d_partial] stores the reduction of each subarray
    cl_mem d_partial = clw.dev_malloc(sizeof(int)*k);
#if DEBUG
    // [d_dbg] stores the full extended array
    int dbg_len = k*m;
    int *dbg = new int[dbg_len];
    cl_mem d_dbg = clw.dev_malloc(sizeof(int)*dbg_len);
#endif

    // KERNEL ARGS FOR SCAN WITHIN SUBARRAYS
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_subarrays, 0, sizeof(cl_mem), (void *)&d_data));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_subarrays, 1, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_subarrays, 2, sizeof(cl_mem), (void *)&d_partial));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_subarrays, 3, sizeof(int), &n));
#if DEBUG
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_subarrays, 4, sizeof(cl_mem), (void *)&d_dbg));
#endif

    // REAL WORK DONE HERE
    k1 += clw.run_kernel_with_timing("scan_subarrays", /*dim=*/1, &gx, &wx);
    recursive_scan(clw, d_partial, k, callnum+1);

    // KERNEL ARGS FOR SUBARRAY INCREMENT
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_inc_subarrays, 0, sizeof(cl_mem), (void *)&d_data));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_inc_subarrays, 1, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_inc_subarrays, 2, sizeof(cl_mem), (void *)&d_partial));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_inc_subarrays, 3, sizeof(int), &n));
#if DEBUG
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_inc_subarrays, 4, sizeof(cl_mem), (void *)&d_dbg));
#endif

    // REAL WORK DONE HERE
    k2 += clw.run_kernel_with_timing("scan_inc_subarrays", /*dim=*/1, &gx, &wx);
  }
}

void run(int *data, int n, int num_iter, map<string,float> &timings) {
  // PLATFORM AND DEVICE INFO
  if (opt.verbose) {
    cout << clinfo();
  }

  // BUILD PROGRAM AND KERNELS
  CLWrapper clw(/*platform=*/0,/*device=*/0,/*profiling=*/true);
  cl_program program = clw.compile("scan.cl", (DEBUG ? " -D DEBUG=true" : ""));
  clw.create_all_kernels(program);
  scan_subarrays = clw.kernel_of_name("scan_subarrays");
  scan_pad_to_pow2 = clw.kernel_of_name("scan_pad_to_pow2");
  scan_inc_subarrays = clw.kernel_of_name( "scan_inc_subarrays");

  // DEVICE MEMORY
  cl_mem d_data = clw.dev_malloc(sizeof(int)*n);

  int *result = new int[n];
  float m0 = 0; float m1 = 0;
  for (int run=0; run<num_iter; run++) {
    m0 += clw.memcpy_to_dev(d_data, sizeof(int)*n, data);
    recursive_scan(clw, d_data, n, 0);
    m1 += clw.memcpy_from_dev(d_data, sizeof(int)*n, result);
  }
  timings.insert(make_pair("1. memcpy to dev     ", m0));
  timings.insert(make_pair("2. scan_pad_to_pow2  ", k0));
  timings.insert(make_pair("3. scan_subarrays    ", k1));
  timings.insert(make_pair("4. scan_inc_subarrays", k2));
  timings.insert(make_pair("5. memcpy from dev   ", m1));

  clw.memcpy_from_dev(d_data, sizeof(int)*n, data);
  delete[] result;
}
