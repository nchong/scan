#include "scan.h"

#include <cmath>

void Scan::scan(int *data, int n) {
  int k = (int) ceil((float)n/(float)m);
  cl_mem d_data = clw.dev_malloc(sizeof(int)*k*m);
  m0 += clw.memcpy_to_dev(d_data, sizeof(int)*n, data);
  recursive_scan(d_data, n);
  m1 += clw.memcpy_from_dev(d_data, sizeof(int)*n, data);
  clw.dev_free(d_data);
}

void Scan::scan(cl_mem data, int n) {
  int k = (int) ceil((float)n/(float)m);
  cl_mem d_data = clw.dev_malloc(sizeof(int)*k*m);
  c0 += clw.copy_buffer(data, d_data, sizeof(int)*n);
  recursive_scan(d_data, n);
  c1 += clw.copy_buffer(d_data, data, sizeof(int)*n);
  clw.dev_free(d_data);
}

void Scan::recursive_scan(cl_mem d_data, int n) {
  int k = (int) ceil((float)n/(float)m);
  //size of each subarray stored in local memory
  size_t bufsize = sizeof(int)*m;
  if (k == 1) {
    clw.kernel_arg(scan_pad_to_pow2,
      d_data, bufsize, n);
    k0 += clw.run_kernel_with_timing(scan_pad_to_pow2, /*dim=*/1, &wx, &wx);
  } else {
    size_t gx = k * wx;
    cl_mem d_partial = clw.dev_malloc(sizeof(int)*k);
    clw.kernel_arg(scan_subarrays,
      d_data, bufsize, d_partial, n);
    k1 += clw.run_kernel_with_timing(scan_subarrays, /*dim=*/1, &gx, &wx);
    recursive_scan(d_partial, k);
    clw.kernel_arg(scan_inc_subarrays,
      d_data, bufsize, d_partial, n);
    k2 += clw.run_kernel_with_timing(scan_inc_subarrays, /*dim=*/1, &gx, &wx);

    clw.dev_free(d_partial);
  }
}

Scan::Scan(CLWrapper &clw, size_t wx) : clw(clw), wx(wx) {
  m = wx * 2;
  clw.create_all_kernels(clw.compile("scan.cl"));
  scan_pow2 = clw.kernel_of_name("scan_pow2_wrapper");
  scan_pad_to_pow2 = clw.kernel_of_name("scan_pad_to_pow2");
  scan_subarrays = clw.kernel_of_name("scan_subarrays");
  scan_inc_subarrays = clw.kernel_of_name("scan_inc_subarrays");
}

void Scan::reset_timers() {
  c0 = c1 = 0;
  m0 = m1 = 0;
  k0 = k1 = k2 = 0;
}

void Scan::get_timers(map<string,float> &timings) {
  timings.insert(make_pair("1. memcpy to dev     ", m0));
  timings.insert(make_pair("2. scan_pad_to_pow2  ", k0));
  timings.insert(make_pair("3. scan_subarrays    ", k1));
  timings.insert(make_pair("4. scan_inc_subarrays", k2));
  timings.insert(make_pair("5. memcpy from dev   ", m1));
  timings.insert(make_pair("6. cpy from dev      ", c0));
  timings.insert(make_pair("7. cpy from dev      ", c1));
}
