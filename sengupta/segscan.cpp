#include "segscan.h"

#include <cmath>

void SegmentedScan::scan(int *data, int *flag, int n) {
  int k = (int) ceil((float)n/(float)m);
  cl_mem d_data = clw.dev_malloc(sizeof(int)*k*m);
  cl_mem d_part = clw.dev_malloc(sizeof(int)*k*m);
  cl_mem d_flag = clw.dev_malloc(sizeof(int)*k*m);
  m0 += clw.memcpy_to_dev(d_data, sizeof(int)*n, data);
  m1 += clw.memcpy_to_dev(d_part, sizeof(int)*n, flag);
  m2 += clw.memcpy_to_dev(d_flag, sizeof(int)*n, flag);
  recursive_scan(d_data, d_part, d_flag, n);
  m3 += clw.memcpy_from_dev(d_data, sizeof(int)*n, data);
  clw.dev_free(d_data);
  clw.dev_free(d_part);
  clw.dev_free(d_flag);
}

void SegmentedScan::scan(cl_mem data, cl_mem flag, int n) {
  int k = (int) ceil((float)n/(float)m);
  cl_mem d_data = clw.dev_malloc(sizeof(int)*k*m);
  cl_mem d_part = clw.dev_malloc(sizeof(int)*k*m);
  cl_mem d_flag = clw.dev_malloc(sizeof(int)*k*m);
  clw.copy_buffer(data, d_data, sizeof(int)*n);
  clw.copy_buffer(flag, d_part, sizeof(int)*n);
  clw.copy_buffer(flag, d_flag, sizeof(int)*n);
  recursive_scan(d_data, d_part, d_flag, n);
  clw.copy_buffer(d_data, data, sizeof(int)*n);
  clw.dev_free(d_data);
  clw.dev_free(d_part);
  clw.dev_free(d_flag);
}

void SegmentedScan::recursive_scan(cl_mem d_data, cl_mem d_part, cl_mem d_flag, int n) {
  int k = (int) ceil((float)n/(float)m);
  //size of each subarray stored in local memory
  size_t bufsize = sizeof(int)*m;
  if (k == 1) {
    clw.kernel_arg(scan_pad_to_pow2,
      d_data,  d_part,  d_flag,
      bufsize, bufsize, bufsize,
      n);
    k0 += clw.run_kernel_with_timing(scan_pad_to_pow2, /*dim=*/1, &wx, &wx);

  } else {
    size_t gx = k * wx;
    cl_mem d_data2 = clw.dev_malloc(sizeof(int)*k);
    cl_mem d_part2 = clw.dev_malloc(sizeof(int)*k);
    cl_mem d_flag2 = clw.dev_malloc(sizeof(int)*k);
    clw.kernel_arg(upsweep_subarrays,
      d_data,  d_part,  d_flag,
      d_data2, d_part2, d_flag2,
      bufsize, bufsize, bufsize,
      n);
    k1 += clw.run_kernel_with_timing(upsweep_subarrays, /*dim=*/1, &gx, &wx);

    recursive_scan(d_data2, d_part2, d_flag2, k);

    clw.kernel_arg(downsweep_subarrays,
      d_data,  d_part,  d_flag,
      d_data2, d_part2, d_flag2,
      bufsize, bufsize, bufsize,
      n);
    k2 += clw.run_kernel_with_timing(downsweep_subarrays, /*dim=*/1, &gx, &wx);

    clw.dev_free(d_data2);
    clw.dev_free(d_part2);
    clw.dev_free(d_flag2);
  }
}

SegmentedScan::SegmentedScan(CLWrapper &clw, size_t wx) : clw(clw), wx(wx),
  m0(0), m1(0), m2(0), m3(0),
  k0(0), k1(0), k2(0) {
  m = wx * 2;
#if EMBED_CL
  #include "segscan.cl.h"
  clw.create_all_kernels(clw.compile_from_string((char *)&segscan_cl));
#else
  clw.create_all_kernels(clw.compile("segscan.cl"));
#endif
  scan_pow2 = clw.kernel_of_name("segscan_pow2_wrapper");
  scan_pad_to_pow2 = clw.kernel_of_name("segscan_pad_to_pow2");
  upsweep_subarrays = clw.kernel_of_name("upsweep_subarrays");
  downsweep_subarrays = clw.kernel_of_name("downsweep_subarrays");
  reset_timers();
}

void SegmentedScan::reset_timers() {
  m0 = m1 = m2 = m3 = 0;
  k0 = k1 = k2 = 0;
}

void SegmentedScan::get_timers(map<string,float> &timings) {
  timings.insert(make_pair("SEGSCAN1. data_memcpy_to_dev",   m0));
  timings.insert(make_pair("SEGSCAN2. part_memcpy_to_dev",   m1));
  timings.insert(make_pair("SEGSCAN3. flag_memcpy_to_dev",   m2));
  timings.insert(make_pair("SEGSCAN4. scan_pad_to_pow2",     k0));
  timings.insert(make_pair("SEGSCAN5. upsweep_subarrays",    k1));
  timings.insert(make_pair("SEGSCAN6. downsweep_subarrays",  k2));
  timings.insert(make_pair("SEGSCAN7. data_memcpy_from_dev", m3));
}
