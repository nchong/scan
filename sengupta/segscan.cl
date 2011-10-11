/*
 * Inplace upsweep (reduce) on local array [x] with partial [p].
 * [x] and [p] are of length [m].
 * NB: m must be a power of two.
 */
inline void upsweep_pow2(__local int *x, __local int *p, int m) {
  int lid = get_local_id(0);
  int bi = (lid*2)+1;

  int depth = 1 + (int) log2((float)m);
  for (int d=0; d<depth; d++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int mask = (0x1 << d) - 1;
    if ((lid & mask) == mask) {
      int offset = (0x1 << d);
      int ai = bi - offset;
      if (!p[bi]) {
        x[bi] += x[ai];
      }
      p[bi] = p[bi] | p[ai];
    }
  }
}

/*
 * Inplace downsweep on a local array [x] with partial [p] and flag [f].
 * [x], [p] and [f] are of length [m].
 * NB: m must be a power of two.
 */
inline void sweepdown_pow2(__local int *x, __local int *p, __local int *f, int m) {
  int lid = get_local_id(0);
  int bi = (lid*2)+1;

  int depth = (int) log2((float)m);
  for (int d=depth; d>-1; d--) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int mask = (0x1 << d) - 1;
    if ((lid & mask) == mask) {
      int offset = (0x1 << d);
      int ai = bi - offset;
      int tmp = x[ai];
                x[ai] = x[bi];
      if (f[ai+1]) {
        x[bi] = 0;
      } else if (p[ai]) {
        x[bi] = tmp;
      } else {
        x[bi] += tmp;
      }
      p[ai] = 0;
    }
  }
}

/*
 * Inplace scan on a local array [x] with partial [p] and flag [f].
 * [x], [p] and [f] are of length [m].
 * NB: m must be a power of two.
 */
inline void scan_pow2(__local int *x, __local int *p, __local int *f, int m) {
  int lid = get_local_id(0);
  int lane1 = (lid*2)+1;
  upsweep_pow2(x, p, m);
  if (lane1 == (m-1)) {
    x[lane1] = 0;
  }
  sweepdown_pow2(x, p, f, m);
}

/*
 * Inplace scan on a global array [data] with partial [part] and flag [flag].
 * [data], [part] and [flag] are of length [m].
 * We load data into local arrays [x], [p], [f] (also of length [m]),
 *   and use a local upsweep and downsweep.
 * NB: m must be a power of two, and
 *     there must be exactly one workgroup of size m/2
 */
__kernel void segscan_pow2_wrapper(
    __global int *data, __global int *part, __global int *flag,
    __local  int *x,    __local  int *p,     __local int *f,
    int m) {
  int gid = get_global_id(0);
  int lane0 = (gid*2);
  int lane1 = (gid*2)+1;

  // load data into local arrays
  x[lane0] = data[lane0];
  x[lane1] = data[lane1];
  p[lane0] = part[lane0];
  p[lane1] = part[lane1];
  f[lane0] = flag[lane0];
  f[lane1] = flag[lane1];

  // inplace local scan
  scan_pow2(x, p, f, m);

  // writeback data
  data[lane0] = x[lane0];
  data[lane1] = x[lane1];
}

/*
 * Inplace scan on a global array [data] with partial [part] and flag [flag].
 * [data], [part] and [flag] are of length [n].
 * We load data into local arrays [x], [p], [f] (padded to length [m]),
 *   and use a local upsweep and downsweep.
 * NB: We assume n < m,
 *     m must be a power of two, and
 *     there must be exactly one workgroup of size m/2
 */
__kernel void segscan_pad_to_pow2(
    __global int *data, __global int *part, __global int *flag,
    __local  int *x,    __local  int *p,     __local int *f,
    int n) {
  int gid = get_global_id(0);
  int lane0 = (gid*2);
  int lane1 = (gid*2)+1;
  int m = 2*get_local_size(0);

  // load data into local arrays, padding with identity element
  x[lane0] = lane0 < n ? data[lane0] : 0;
  x[lane1] = lane1 < n ? data[lane1] : 0;
  p[lane0] = lane0 < n ? part[lane0] : 0;
  p[lane1] = lane1 < n ? part[lane1] : 0;
  f[lane0] = lane0 < n ? flag[lane0] : 0;
  f[lane1] = lane1 < n ? flag[lane1] : 0;

  // inplace local scan
  scan_pow2(x, p, f, m);

  // store back to global data
  if (lane0 < n)
    data[lane0] = x[lane0];
  if (lane1 < n)
    data[lane1] = x[lane1];
}

/*
 * First stage of a multiblock segmented scan.
 *
 * Given a global array [data] with partial [part] and flag [flag].
 * We assume [data], [part] and [flag] are of length [k*m], where
 *   k is the number of workgroups    / number of subarrays
 *   m is the 2 * number of workitems / length of each subarray.
 * Only the first n elements of the input arrays must be filled.
 *
 * Each workgroup works on a separate subarray.
 * Each workgroup loads its assigned subarray into local arrays
 *   [x], [p], [f] (of length [m]) and perform a local upsweep.
 *
 * We store the last element of each subarray into global arrays
 *   [data2], [part2] and [flag2] (of length [k]).
 * These will be used for a second-level scan.
 *
 * We writeback [data] and [part] for the second stage [downsweep_subarrays].
 */
__kernel void upsweep_subarrays(
    __global int *data,  __global int *part,  __global int *flag,
    __global int *data2, __global int *part2, __global int *flag2,
    __local  int *x,     __local  int *p,     __local  int *f,
    int n) {
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

  // load into local data padding elements >= n with identity
  x[local_lane0] = (lane0 < n) ? data[lane0] : 0;
  x[local_lane1] = (lane1 < n) ? data[lane1] : 0;
  p[local_lane0] = (lane0 < n) ? part[lane0] : 0;
  p[local_lane1] = (lane1 < n) ? part[lane1] : 0;
  f[local_lane0] = (lane0 < n) ? flag[lane0] : 0;
  f[local_lane1] = (lane1 < n) ? flag[lane1] : 0;

  // upsweep on each subarray
  upsweep_pow2(x, p, m);

  // last workitem per workgroup stores 
  //   - last element of each subarray in global [data2] and [part2]
  if (lid == (wx-1)) {
    data2[grpid] = x[local_lane1];
    part2[grpid] = p[local_lane1];
  }
  // first workitem per workgroup stores 
  //   - first element of each subarray in global [flag2]
  if (lid == 0) {
    flag2[grpid] = f[local_lane0];
  }

  // store back to global data
  data[lane0] = x[local_lane0];
  data[lane1] = x[local_lane1];
  part[lane0] = p[local_lane0];
  part[lane1] = p[local_lane1];
}

/*
 * Second part of a multiblock segmented scan.
 *
 * Given a global array [data] with partial [part] and flag [flag].
 * We assume [data], [part] and [flag] are of length [k*m], where
 *   k is the number of workgroups    / number of subarrays
 *   m is the 2 * number of workitems / length of each subarray.
 * All elements have been filled by first stage [upsweep_subarrays].
 *
 * We also assume [data2], [part2], [flag2] contains the results of a scan.
 *
 * Each workgroup loads its assigned subarray into local arrays
 *   [x], [p], [f] (of length [m]).
 * We fold in results from [data2] and perform a local downsweep.
 */
__kernel void downsweep_subarrays(
    __global int *data,  __global int *part,  __global int *flag,
    __global int *data2, __global int *part2, __global int *flag2,
    __local  int *x,     __local  int *p,     __local  int *f,
    int n) {
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

  // load into local data
  x[local_lane0] = data[lane0];
  x[local_lane1] = data[lane1];
  p[local_lane0] = part[lane0];
  p[local_lane1] = part[lane1];
  f[local_lane0] = flag[lane0];
  f[local_lane1] = flag[lane1];

  // fold partial results back
  if (lid == (wx-1)) {
    x[local_lane1] = data2[grpid];
    p[local_lane1] = part2[grpid];
    //               flag2 not needed
  }
  // downsweep on each subarray
  sweepdown_pow2(x, p, f, m);

  // store back to global data
  if (lane0 < n)
    data[lane0] = x[local_lane0];
  if (lane1 < n)
    data[lane1] = x[local_lane1];
}
