#include "cl_common.h"
#define SEGMENTED true
#include "framework.h"

#define DEBUG false

using namespace std;

cl_kernel scan_pow2;
cl_kernel scan_pad_to_pow2;
cl_kernel upsweep_subarrays;
cl_kernel downsweep_subarrays;

/*
 * Algorithm 5 as outlined in "Scan Primitives for GPU Computing" 
 *                                              (Sengupta et al).
 */
void recursive_scan(CLWrapper &clw, bool verbose, size_t wx,
                    cl_mem d_data, cl_mem d_part, cl_mem d_flag,
                    int n, int callnum) {
  int m = wx*2;
  int k = (int) ceil((float)n/(float)m);
  size_t gx = k * wx;
  if (verbose) {
    printf("# RECURSIVE SCAN (CALL %d)\n", callnum);
    printf("# \tLength of array (n)             = %d\n", n);
    printf("# \tWorkgroup size (wx)             = %u\n", (int)wx);
    printf("# \tLength of local subarrays (m)   = %d\n", m);
    printf("# \tNumber of workgroups (k)        = %d\n", k);
    printf("# \tGlobal number of workitems (gx) = %d\n", (int)gx);
  }

  if (k == 1) {
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_pad_to_pow2, 0, sizeof(cl_mem), (void *)&d_data));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_pad_to_pow2, 1, sizeof(cl_mem), (void *)&d_part));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_pad_to_pow2, 2, sizeof(cl_mem), (void *)&d_flag));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_pad_to_pow2, 3, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_pad_to_pow2, 4, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_pad_to_pow2, 5, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(scan_pad_to_pow2, 6, sizeof(int), &n));
    clw.run_kernel_with_timing("scan_pad_to_pow2", /*dim=*/1, &wx, &wx);

  } else {
    cl_mem d_data2 = clw.dev_malloc(sizeof(int)*k);
    cl_mem d_flag2 = clw.dev_malloc(sizeof(int)*k);
    cl_mem d_part2 = clw.dev_malloc(sizeof(int)*k);

    ASSERT_NO_CL_ERROR(
      clSetKernelArg(upsweep_subarrays, 0, sizeof(cl_mem), (void *)&d_data));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(upsweep_subarrays, 1, sizeof(cl_mem), (void *)&d_part));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(upsweep_subarrays, 2, sizeof(cl_mem), (void *)&d_flag));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(upsweep_subarrays, 3, sizeof(cl_mem), (void *)&d_data2));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(upsweep_subarrays, 4, sizeof(cl_mem), (void *)&d_part2));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(upsweep_subarrays, 5, sizeof(cl_mem), (void *)&d_flag2));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(upsweep_subarrays, 6, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(upsweep_subarrays, 7, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(upsweep_subarrays, 8, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(upsweep_subarrays, 9, sizeof(int), &n));
    clw.run_kernel_with_timing("upsweep_subarrays", /*dim=*/1, &gx, &wx);

    recursive_scan(clw, verbose, wx,
                   d_data2, d_part2, d_flag2,
                   k, callnum+1);

    ASSERT_NO_CL_ERROR(
      clSetKernelArg(downsweep_subarrays, 0, sizeof(cl_mem), (void *)&d_data));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(downsweep_subarrays, 1, sizeof(cl_mem), (void *)&d_part));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(downsweep_subarrays, 2, sizeof(cl_mem), (void *)&d_flag));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(downsweep_subarrays, 3, sizeof(cl_mem), (void *)&d_data2));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(downsweep_subarrays, 4, sizeof(cl_mem), (void *)&d_part2));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(downsweep_subarrays, 5, sizeof(cl_mem), (void *)&d_flag2));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(downsweep_subarrays, 6, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(downsweep_subarrays, 7, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(downsweep_subarrays, 8, sizeof(int)*m, NULL));
    ASSERT_NO_CL_ERROR(
      clSetKernelArg(downsweep_subarrays, 9, sizeof(int), &n));
    clw.run_kernel_with_timing("downsweep_subarrays", /*dim=*/1, &gx, &wx);
  }
}

void run(int *data, int *flag, int n, int num_iter, map<string,float> &timings) {
  // PLATFORM AND DEVICE INFO
  if (opt.verbose) {
    cout << clinfo();
  }

  // BUILD PROGRAM AND KERNELS
  CLWrapper clw(/*platform=*/0,/*device=*/0,/*profiling=*/true);
  cl_program program = clw.compile("scan.cl", (DEBUG ? " -D DEBUG=true" : ""));
  clw.create_all_kernels(program);
  scan_pow2 = clw.kernel_of_name("scan_pow2_wrapper");
  scan_pad_to_pow2 = clw.kernel_of_name("scan_pad_to_pow2");
  upsweep_subarrays = clw.kernel_of_name("upsweep_subarrays");
  downsweep_subarrays = clw.kernel_of_name("downsweep_subarrays");

  // DEVICE MEMORY
  int m = opt.wx*2;
  int k = (int) ceil((float)n/(float)m);
  cl_mem d_data = clw.dev_malloc(sizeof(int)*k*m);
  cl_mem d_part = clw.dev_malloc(sizeof(int)*k*m);
  cl_mem d_flag = clw.dev_malloc(sizeof(int)*k*m);

  clw.memcpy_to_dev(d_data, sizeof(int)*n, data);
  clw.memcpy_to_dev(d_part, sizeof(int)*n, flag);
  clw.memcpy_to_dev(d_flag, sizeof(int)*n, flag);

  recursive_scan(clw, opt.verbose, opt.wx,
                 d_data, d_part, d_flag,
                 n, 0);

  clw.memcpy_from_dev(d_data, sizeof(int)*n, data);
}