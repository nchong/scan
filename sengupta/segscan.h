#include "cl_common.h"
#include <cmath>

class SegmentedScan {
  private:
    CLWrapper &clw;
    cl_kernel scan_pow2;
    cl_kernel scan_pad_to_pow2;
    cl_kernel upsweep_subarrays;
    cl_kernel downsweep_subarrays;
    size_t wx; // workgroup size
    int m;     // length of each subarray ( = wx*2 )

    //timings
    float m0; float m1; float m2; float m3;
    float k0; float k1; float k2;

    void recursive_scan(cl_mem d_data, cl_mem d_part, cl_mem d_flag, int n) {
      int k = (int) ceil((float)n/(float)m);
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
        k0 += clw.run_kernel_with_timing("segscan_pad_to_pow2", /*dim=*/1, &wx, &wx);

      } else {
        size_t gx = k * wx;
        cl_mem d_data2 = clw.dev_malloc(sizeof(int)*k);
        cl_mem d_part2 = clw.dev_malloc(sizeof(int)*k);
        cl_mem d_flag2 = clw.dev_malloc(sizeof(int)*k);

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
        k1 += clw.run_kernel_with_timing("upsweep_subarrays", /*dim=*/1, &gx, &wx);

        recursive_scan(d_data2, d_part2, d_flag2, k);

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
        k2 += clw.run_kernel_with_timing("downsweep_subarrays", /*dim=*/1, &gx, &wx);

        clw.dev_free(d_data2);
        clw.dev_free(d_part2);
        clw.dev_free(d_flag2);
      }
    }

  public:
    SegmentedScan(CLWrapper &_clw, size_t _wx=256) : clw(_clw), wx(_wx) {
      m = wx * 2;
      clw.create_all_kernels(clw.compile("segscan.cl"));
      scan_pow2 = clw.kernel_of_name("segscan_pow2_wrapper");
      scan_pad_to_pow2 = clw.kernel_of_name("segscan_pad_to_pow2");
      upsweep_subarrays = clw.kernel_of_name("upsweep_subarrays");
      downsweep_subarrays = clw.kernel_of_name("downsweep_subarrays");
      reset_timers();
    }

    ~SegmentedScan() {
    }

    void reset_timers() {
      m0 = m1 = m2 = m3 = 0;
      k0 = k1 = k2 = 0;
    }

    void get_timers(map<string,float> &timings) {
      timings.insert(make_pair("1. data_memcpy_to_dev",   m0));
      timings.insert(make_pair("2. part_memcpy_to_dev",   m1));
      timings.insert(make_pair("3. flag_memcpy_to_dev",   m2));
      timings.insert(make_pair("4. scan_pad_to_pow2",     k0));
      timings.insert(make_pair("5. upsweep_subarrays",    k1));
      timings.insert(make_pair("6. downsweep_subarrays",  k2));
      timings.insert(make_pair("7. data_memcpy_from_dev", m3));
    }

    void scan(int *data, int *flag, int n) {
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

    void scan(cl_mem data, cl_mem flag, int n) {
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

};
