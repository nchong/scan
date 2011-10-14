#ifndef SCAN_H
#define SCAN_H

#include "clwrapper.h"

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
    float c0; float c1;           //copy buffers
    float m0; float m1;           //memcpy buffers
    float k0; float k1; float k2; //kernels

    void recursive_scan(cl_mem d_data, int n);

  public:
    Scan(CLWrapper &clw, size_t wx=256);
    void reset_timers();
    void get_timers(map<string,float> &timings);

    void scan(int *data, int n);
    void scan(cl_mem data, int n);
};

#endif
