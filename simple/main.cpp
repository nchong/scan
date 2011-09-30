#include "cl_common.h"
#include "framework.h"

#define DEBUG false

using namespace std;

void run(int *data, int n, int num_iter, struct options &opt, map<string,float> &timings) {
  if (opt.verbose) {
    cout << clinfo();
  }

  // BUILD PROGRAM AND KERNEL
  CLWrapper clw(/*platform=*/0,/*device=*/0,/*profiling=*/true);
  cl_program program = clw.compile("scan.cl", (DEBUG ? " -D DEBUG=true" : ""));
  clw.create_all_kernels(program);
  cl_kernel scan_subarrays = clw.kernel_of_name("scan_subarrays");
  cl_kernel scan_pow2 = clw.kernel_of_name("scan_pow2");
  cl_kernel scan_inc_subarrays = clw.kernel_of_name( "scan_inc_subarrays");

  // KERNEL INDEX SPACE
  // Split the array into [k] subarrays each of length [2*wx].
  // We assign one workgroup of size [wx] per subarray.
  // Giving a global size of [gw = k*wx] workitems
  // workgroup size
  size_t wx = (size_t)opt.wx;
  // list lengths
  int m = wx * 2;
  int k = (int) ceil((float)n/(float)m);
  // global workitem size
  size_t gx = k * wx;
  if (opt.verbose) {
    printf("# PARAMETERS\n");
    printf("# \tLength of array (n)             = %d\n", n);
    printf("# \tWorkgroup size (wx)             = %u\n", (int)wx);
    printf("# CALCULATED PARAMETERS\n");
    printf("# \tLength of local subarrays (m)   = %d\n", m);
    printf("# \tNumber of workgroups (k)        = %d\n", k);
    printf("# \tGlobal number of workitems (gx) = %d\n", (int)gx);
  }
  if (k > (int)wx) {
    LOG(LOG_FATAL, "Number of partials (k) > Size of local subarray (m)");
  }

  // DEVICE MEMORY
  cl_mem d_data = clw.dev_malloc(sizeof(int)*n);
  cl_mem d_partial = clw.dev_malloc(sizeof(int)*m);

#if DEBUG
  int dbg_len = k*m;
  int *dbg = new int[dbg_len];
  cl_mem d_dbg = clw.dev_malloc(sizeof(int)*dbg_len);
#endif

  // KERNEL ARGS
  // NB: set local data array to length gx.
  // We pad redundant elements with 0 in the kernel.
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

  ASSERT_NO_CL_ERROR(
    clSetKernelArg(scan_pow2, 0, sizeof(cl_mem), (void *)&d_partial));
  ASSERT_NO_CL_ERROR(
    clSetKernelArg(scan_pow2, 1, sizeof(int)*m, NULL));
  ASSERT_NO_CL_ERROR(
    clSetKernelArg(scan_pow2, 2, sizeof(int), &m));

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

  float m0 = 0; float m1 = 0;
  float k0 = 0; float k1 = 0; float k2 = 0;
  int *result = new int[n];
  for (int run=0; run<num_iter; run++) {
    m0 += clw.memcpy_to_dev(d_data, sizeof(int)*n, data);
    k0 += clw.run_kernel_with_timing("scan_subarrays", /*dim=*/1, &gx, &wx);
    k1 += clw.run_kernel_with_timing("scan_pow2", /*dim=*/1, &wx, &wx);
    k2 += clw.run_kernel_with_timing("scan_inc_subarrays", /*dim=*/1, &gx, &wx);
    m1 += clw.memcpy_from_dev(d_data, sizeof(int)*n, result);
#if DEBUG
    if (run == 0) {
      clw.memcpy_from_dev(d_dbg, sizeof(int)*dbg_len, dbg);
      file << atos(dbg, dbg_len, "DEBUG");
    }
#endif
  }
  timings.insert(make_pair("1. memcpy to dev     ", m0));
  timings.insert(make_pair("2. scan_subarrays    ", k0));
  timings.insert(make_pair("3. scan_pow2         ", k1));
  timings.insert(make_pair("4. scan_inc_subarrays", k2));
  timings.insert(make_pair("5. memcpy from dev   ", m1));

  clw.memcpy_from_dev(d_data, sizeof(int)*n, data);
  delete[] result;
#if DEBUG
  delete[] dbg;
#endif
}
