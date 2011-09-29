/*
 * Perform an inplace reduce on a local array [data] of length [m].
 * NB: [m] must be a power of two.
 */
inline void reduce_pow2(__local int *data, int m) {
  int lid = get_local_id(0);
  int lane = (lid*2)+1;

  int depth = 1 + (int) log2((float)m);
  for (int d=0; d<depth; d++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int mask = (0x1 << d) - 1;
    if ((lid & mask) == mask) {
      int offset = (0x1 << d);
      data[lane] += data[lane-offset];
    }
  }
}

/*
 * Perform an inplace sweepdown on a local array [data] of length [m]
 * NB: [m] must be a power of two.
 */
inline void sweepdown_pow2(__local int *data, int m) {
  int lid = get_local_id(0);
  int lane = (lid*2)+1;

  int depth = (int) log2((float)m);
  for (int d=depth; d>-1; d--) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int mask = (0x1 << d) - 1;
    if ((lid & mask) == mask) {
      int offset = (0x1 << d);
      int ai = lane-offset;
      int tmp = data[ai];
                data[ai] = data[lane];
                           data[lane] += tmp;
    }
  }
}

/*
 * Perform an inplace scan on a global array [gdata] of length [m].
 * We load data into a local array [ldata] (also of length [m]),
 *   and use a local reduce and sweepdown.
 * NB: [m] must be a power of two and
 *     There must be exactly one workgroup of size m/2
 */
__kernel void scan_pow2(__global int *gdata, __local int *ldata, int m) {
  int gid = get_global_id(0);
  int lane0 = (gid*2);
  int lane1 = (gid*2)+1;

  ldata[lane0] = gdata[lane0];
  ldata[lane1] = gdata[lane1];

  reduce_pow2(ldata, m);
  if (lane1 == (m-1)) {
    ldata[lane1] = 0;
  }
  sweepdown_pow2(ldata, m);

  gdata[lane0] = ldata[lane0];
  gdata[lane1] = ldata[lane1];
}

/*
 * Perform the first phase of an inplace exclusive scan on a global array [gdata] of arbitrary length [n].
 *
 * We assume that we have k workgroups each of size m/2 workitems.
 * Each workgroup handles a subarray of length [m] (where m is a power of two).
 * The last subarray will be padded with 0 if necessary (n < k*m).
 * We use the primitives above to perform a scan operation within each subarray.
 * We store the intermediate reduction of each subarray (following reduce_pow2) in [gpartial].
 * These partial values can themselves be scanned and fed into [scan_inc_subarrays].
 */
__kernel void scan_subarrays(
  __global int *gdata,    //length [n]
  __local  int *ldata,    //length [m]
  __global int *gpartial, //length [m]
           int n
#if DEBUG
  , __global int *debug   //length [k*m]
#endif
) {
  // workgroup size
  int wx = get_local_size(0);
  // global identifiers and indexes
  int gid = get_global_id(0);
  int lane0 = (2*gid)  ;
  int lane1 = (2*gid)+1;
  // local identifiers and indexes
  int lid = get_local_id(0);
  int local_lane0 = (2*lid)  ;
  int local_lane1 = (2*lid)+1;
  int grpid = get_group_id(0);
  // list lengths
  int m = wx * 2;
  int k = get_num_groups(0);

  // copy into local data padding elements >= n with 0
  ldata[local_lane0] = (lane0 < n) ? gdata[lane0] : 0;
  ldata[local_lane1] = (lane1 < n) ? gdata[lane1] : 0;

  // ON EACH SUBARRAY
  // a reduce on each subarray
  reduce_pow2(ldata, m);
  // last workitem per workgroup saves last element of each subarray in [gpartial] before zeroing
  if (lid == (wx-1)) {
    gpartial[grpid] = ldata[local_lane1];
                      ldata[local_lane1] = 0;
  }
  // a sweepdown on each subarray
  sweepdown_pow2(ldata, m);

  // copy back to global data
  if (lane0 < n) {
    gdata[lane0] = ldata[local_lane0];
  }
  if (lane1 < n) {
    gdata[lane1] = ldata[local_lane1];
  }

#if DEBUG
  debug[lane0] = ldata[local_lane0];
  debug[lane1] = ldata[local_lane1];
#endif
}

/*
 * Perform the second phase of an inplace exclusive scan on a global array [gdata] of arbitrary length [n].
 *
 * We assume that we have k workgroups each of size m/2 workitems.
 * Each workgroup handles a subarray of length [m] (where m is a power of two).
 * We sum each element by the sum of the preceding subarrays taken from [gpartial].
 */
__kernel void scan_inc_subarrays(
  __global int *gdata,    //length [n]
  __local  int *ldata,    //length [m]
  __global int *gpartial, //length [m]
           int n
#if DEBUG
  , __global int *debug   //length [k*m]
#endif
) {
  // global identifiers and indexes
  int gid = get_global_id(0);
  int lane0 = (2*gid)  ;
  int lane1 = (2*gid)+1;
  // local identifiers and indexes
  int lid = get_local_id(0);
  int local_lane0 = (2*lid)  ;
  int local_lane1 = (2*lid)+1;
  int grpid = get_group_id(0);

  // copy into local data padding elements >= n with 0
  ldata[local_lane0] = (lane0 < n) ? gdata[lane0] : 0;
  ldata[local_lane1] = (lane1 < n) ? gdata[lane1] : 0;

  ldata[local_lane0] += gpartial[grpid];
  ldata[local_lane1] += gpartial[grpid];

  // copy back to global data
  if (lane0 < n) {
    gdata[lane0] = ldata[local_lane0];
  }
  if (lane1 < n) {
    gdata[lane1] = ldata[local_lane1];
  }

#if DEBUG
  debug[lane0] = ldata[local_lane0];
  debug[lane1] = ldata[local_lane1];
#endif
}
