#ifndef SEGSCAN_H
#define SEGSCAN_H

#include "clwrapper.h"

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
    float c0; float c1;
    float m0; float m1; float m2; float m3;
    float k0; float k1; float k2;

    void recursive_scan(cl_mem d_data, cl_mem d_part, cl_mem d_flag, int n);

  public:
    SegmentedScan(CLWrapper &clw, size_t wx=256);
    void reset_timers();
    void get_timers(map<string,float> &timings);
    void scan(int *data, int *flag, int n);
    void scan(cl_mem data, cl_mem flag, int n);
};

#endif
