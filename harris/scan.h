#include "cl_common.h"
#include <cmath>

class Scan {
  private:
    CLWrapper &clw;
    cl_kernel scan_pow2;
    cl_kernel scan_pad_to_pow2;
    cl_kernel scan_subarrays;
    cl_kernel scan_inc_subarrays;
    size_t wx; // workgroup size
    int m;     // length of each subarray ( = wx*2 )

    //timings
    float m0; float m1;
    float k0; float k1; float k2;

    void recursive_scan(cl_mem d_data, int n) {
      int k = (int) ceil((float)n/(float)m);
      if (k == 1) {
        ASSERT_NO_CL_ERROR(
          clSetKernelArg(scan_pad_to_pow2, 0, sizeof(cl_mem), (void *)&d_data));
        ASSERT_NO_CL_ERROR(
          clSetKernelArg(scan_pad_to_pow2, 1, sizeof(int)*m, NULL));
        ASSERT_NO_CL_ERROR(
          clSetKernelArg(scan_pad_to_pow2, 2, sizeof(int), &n));
        k0 += clw.run_kernel_with_timing("scan_pad_to_pow2", /*dim=*/1, &wx, &wx);

      } else {
        size_t gx = k * wx;
        cl_mem d_partial = clw.dev_malloc(sizeof(int)*k);

        ASSERT_NO_CL_ERROR(
          clSetKernelArg(scan_subarrays, 0, sizeof(cl_mem), (void *)&d_data));
        ASSERT_NO_CL_ERROR(
          clSetKernelArg(scan_subarrays, 1, sizeof(int)*m, NULL));
        ASSERT_NO_CL_ERROR(
          clSetKernelArg(scan_subarrays, 2, sizeof(cl_mem), (void *)&d_partial));
        ASSERT_NO_CL_ERROR(
          clSetKernelArg(scan_subarrays, 3, sizeof(int), &n));
        k1 += clw.run_kernel_with_timing("scan_subarrays", /*dim=*/1, &gx, &wx);

        recursive_scan(d_partial, k);

        ASSERT_NO_CL_ERROR(
          clSetKernelArg(scan_inc_subarrays, 0, sizeof(cl_mem), (void *)&d_data));
        ASSERT_NO_CL_ERROR(
          clSetKernelArg(scan_inc_subarrays, 1, sizeof(int)*m, NULL));
        ASSERT_NO_CL_ERROR(
          clSetKernelArg(scan_inc_subarrays, 2, sizeof(cl_mem), (void *)&d_partial));
        ASSERT_NO_CL_ERROR(
          clSetKernelArg(scan_inc_subarrays, 3, sizeof(int), &n));
        k2 += clw.run_kernel_with_timing("scan_inc_subarrays", /*dim=*/1, &gx, &wx);

        clw.dev_free(d_partial);
      }
    }

  public:
    Scan(CLWrapper &_clw, size_t _wx=256) : clw(_clw), wx(_wx) {
      m = wx * 2;
      clw.create_all_kernels(clw.compile("scan.cl"));
      scan_pow2 = clw.kernel_of_name("scan_pow2");
      scan_pad_to_pow2 = clw.kernel_of_name("scan_pad_to_pow2");
      scan_subarrays = clw.kernel_of_name("scan_subarrays");
      scan_inc_subarrays = clw.kernel_of_name("scan_inc_subarrays");
    }

    ~Scan() {
    }

    void reset_timers() {
      m0 = m1 = 0;
      k0 = k1 = k2 = 0;
    }

    void get_timers(map<string,float> &timings) {
      timings.insert(make_pair("1. memcpy to dev     ", m0));
      timings.insert(make_pair("2. scan_pad_to_pow2  ", k0));
      timings.insert(make_pair("3. scan_subarrays    ", k1));
      timings.insert(make_pair("4. scan_inc_subarrays", k2));
      timings.insert(make_pair("5. memcpy from dev   ", m1));
    }

    void scan(int *data, int n) {
      int k = (int) ceil((float)n/(float)m);
      cl_mem d_data = clw.dev_malloc(sizeof(int)*k*m);
      m0 += clw.memcpy_to_dev(d_data, sizeof(int)*n, data);
      recursive_scan(d_data, n);
      m1 += clw.memcpy_from_dev(d_data, sizeof(int)*n, data);
      clw.dev_free(d_data);
    }

    void scan(cl_mem data, int n) {
      int k = (int) ceil((float)n/(float)m);
      cl_mem d_data = clw.dev_malloc(sizeof(int)*k*m);
      clw.copy_buffer(data, d_data, sizeof(int)*n);
      recursive_scan(d_data, n);
      clw.copy_buffer(d_data, data, sizeof(int)*n);
      clw.dev_free(d_data);
    }

};
