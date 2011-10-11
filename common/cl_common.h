#ifndef CL_COMMON_H
#define CL_COMMON_H

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#elif __linux__
  #include <CL/cl.h>
#else
  #error Not sure where to find OpenCL header
#endif

#include "cl_error_handling.h"
#include "log.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

cl_uint query_num_platforms() {
  static cl_uint num_platforms = 0;
  if (num_platforms == 0) {
    ASSERT_NO_CL_ERROR(
      clGetPlatformIDs(/*num_entries=*/0, /*platforms=*/NULL, &num_platforms));
  }
  return num_platforms;
}

cl_platform_id *&get_platform_list() {
  cl_uint num_platforms = query_num_platforms();
  static cl_platform_id *platforms = NULL;
  if (platforms == NULL) {
    platforms = new cl_platform_id[num_platforms];
    assert(platforms);
    ASSERT_NO_CL_ERROR(
      clGetPlatformIDs(num_platforms, platforms, /*num_platforms=*/NULL));
  }
  return platforms;
}

cl_uint query_num_devices(cl_platform_id platform) {
  cl_uint num_devices;
  ASSERT_NO_CL_ERROR(
    clGetDeviceIDs(
      platform, CL_DEVICE_TYPE_ALL, /*num_entries=*/0, /*devices=*/NULL, &num_devices));
  return num_devices;
}

cl_device_id *get_device_list(cl_platform_id platform) {
  cl_uint num_devices = query_num_devices(platform);
  cl_device_id *devices = new cl_device_id[num_devices];
  ASSERT_NO_CL_ERROR(
    clGetDeviceIDs(
      platform, CL_DEVICE_TYPE_ALL, num_devices, devices, /*num_devices=*/NULL));
  return devices;
}

void context_error_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
  LOG(LOG_ERR, "context_error_callback %s", errinfo);
}

string clinfo(void) {
  stringstream ss;

  // PLATFORMS
  // query number of platforms
  cl_uint num_platforms = query_num_platforms();
  ss << "Found " << num_platforms << " OpenCL platform" << (num_platforms == 1 ?  "":"s") << "\n";
  // get platform list
  cl_platform_id *platforms = get_platform_list();

  // query devices
  char device_name[1024];
  char device_vendor[1024];
  cl_uint num_cores;
  cl_uint clk_freq;
  cl_long global_mem_size;
  cl_ulong local_mem_size;
  for (int i=0; i<(int)num_platforms; i++) {
    cl_uint num_devices = query_num_devices(platforms[i]);
    ss << "Platform " << i << " has " << num_devices << " device" << (num_devices == 1 ? "":"s") << "\n";

    // get device list
    cl_device_id *devices = get_device_list(platforms[i]);
    for (int j=0; j<(int)num_devices; j++) {
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_cores), &num_cores, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clk_freq), &clk_freq, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, /*param_value_size_ret=*/NULL));

      ss << "Device " << j << "\n";
      ss << "\tName: " << device_name << "\n";
      ss << "\tVendor: " << device_vendor << "\n";
      ss << "\tCompute units: " << num_cores << "\n";
      ss << "\tClock frequency: " << clk_freq << " MHz\n";
      ss << "\tGlobal memory: " << (global_mem_size>>30) << "GB\n";
      ss << "\tLocal memory: " <<  (local_mem_size>>10) << "KB\n";
    }
    delete[] devices;
  }

  return ss.str();
}

class CLWrapper {
  private:
    cl_uint num_platforms;
    cl_platform_id *platforms;
    int p;

    cl_uint num_devices;
    cl_device_id *devices;
    int d;

    bool profiling;
    cl_context context;
    cl_command_queue command_queue;
    vector<cl_program> programs;
    map<string,cl_kernel> kernelmap;
    vector<cl_mem> memobjs;

    float timestamp_diff_in_ms(cl_ulong start, cl_ulong end) {
      return (end-start) * 1.0e-6f;
    }

    float time_and_release_event(cl_event e) {
      cl_ulong start;
      cl_ulong end;
      ASSERT_NO_CL_ERROR(
        clWaitForEvents(/*num_events=*/1, &e));
      ASSERT_NO_CL_ERROR(
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(clReleaseEvent(e));
      return timestamp_diff_in_ms(start, end);
    }
    
    void attach_context(bool all_devices=false) {
      if (all_devices) {
        LOG(LOG_INFO, "Attaching context for all devices");
      } else {
        LOG(LOG_INFO, "Attaching context for device %d", d);
      }

      cl_uint ndev = (all_devices ? num_devices : 1);
      cl_device_id *dev = (all_devices ? devices : &devices[d]);
      cl_int ret;
      context = clCreateContext(/*properties=*/NULL, ndev, dev, context_error_callback, NULL, &ret);
      assert(context);
      ASSERT_NO_CL_ERROR(ret);
    }

    void attach_command_queue(cl_command_queue_properties properties=0) {
      LOG(LOG_INFO, "Attaching command queue%sfor device %d", profiling ? " with profiling " : " ", d);
      cl_int ret;
      command_queue = clCreateCommandQueue(context, devices[d], properties, &ret);
      assert(command_queue);
      ASSERT_NO_CL_ERROR(ret);
    }

  public:
    // Constructor
    CLWrapper(int _p=0, int _d=0, bool _profiling=false) : num_platforms(0), platforms(NULL), p(_p), num_devices(0), devices(NULL), d(_d), profiling(_profiling) {
      LOG(LOG_INFO, "Initializing context and command queue for device %d on platform %d", d, p);

      num_platforms = query_num_platforms();
      assert(p < (int)num_platforms);
      platforms = get_platform_list();      

      num_devices = query_num_devices(platforms[p]);
      devices = get_device_list(platforms[p]);
      assert(d < (int)num_devices);

      attach_context();
      attach_command_queue(profiling ? CL_QUEUE_PROFILING_ENABLE : 0);
    }

    cl_program &compile(const char *fname,
        const string &extra_flags="",
        map<string,string> substitutions=(map<string,string>()),
        bool all_devices=false) {
      if (all_devices) {
        LOG(LOG_INFO, "Compiling file <%s> for all devices", fname);
      } else {
        LOG(LOG_INFO, "Compiling file <%s> for device %d", fname, d);
      }

      // read in file to string (via buf)
      ifstream file(fname, ios::binary);
      if (!file.is_open()) {
        LOG(LOG_FATAL, "Unable to open file <%s>.", fname);
      }
      std::ostringstream buf;
      buf << file.rdbuf() << endl;
      string s = buf.str();

      // perform text substitutions s/sub/replace
      // NB: we only replace the first instance of sub
      map<string,string>::iterator i;
      for (i = substitutions.begin(); i != substitutions.end(); i++) {
        string sub = i->first;
        string replace = i->second;
        if (s.find(sub) != string::npos) {
          s.replace(s.find(sub), sub.length(), replace);
          LOG(LOG_INFO, "Substituting [%s] with [%s]", sub.c_str(), replace.c_str());
        } else {
          LOG(LOG_INFO, "Could not find [%s] to substitute with [%s]", sub.c_str(), replace.c_str());
        }
      }

      // now convert to char array
      char *program_buf = (char *) s.c_str();

      cl_int ret;
      //lengths=NULL -> program_buf is null terminated
      cl_program program = clCreateProgramWithSource(context, /*count=*/1, (const char **) &program_buf, /*lengths=*/NULL, &ret );
      assert(program);
      ASSERT_NO_CL_ERROR(ret);
      programs.push_back(program);

      stringstream flags;
      flags << extra_flags;

      // Math intrinsics options
      //flags << " -cl-single-precision-constant";
      //flags << " -cl-denorms-are-zero";

      // Optimization options
      //flags << " -cl-opt-disable";
      //flags << " -cl-strict-aliasing";
      //flags << " -cl-mad-enable";
      //flags << " -cl-no-signed-zeros";
      //flags << " -cl-unsafe-math-optimizations";
      //flags << " -cl-finite-math-only";
      //flags << " -cl-fast-relaxed-math";

      // Warnings suppress/request
      //flags << " -w";
      flags << " -Werror";

      cl_uint ndev = (all_devices ? num_devices : 1);
      cl_device_id *dev = (all_devices ? devices : &devices[d]);
      //pfn_notify=NULL -> call is blocking
      ASSERT_NO_CL_ERROR(
        clBuildProgram(program, ndev, dev, flags.str().c_str(), /*pfn_notify=*/NULL, /*user_data=*/NULL));
      return programs.back();
    }

    cl_mem &dev_malloc(size_t size, cl_mem_flags flags=CL_MEM_READ_WRITE) {
      cl_int ret;
      cl_mem m = clCreateBuffer(context, flags, size, /*host_ptr*/NULL, &ret);
      ASSERT_NO_CL_ERROR(ret);
      memobjs.push_back(m);
      return memobjs.back();
    }

    void dev_free(cl_mem m) {
      vector<cl_mem>::iterator it = find(memobjs.begin(), memobjs.end(), m);
      if (it == memobjs.end()) {
        LOG(LOG_WARN, "Freeing memory object not found in [memobjs]");
      } else {
        memobjs.erase(it);
      }
      ASSERT_NO_CL_ERROR(clReleaseMemObject(m));
    }

    void free_all_memobjs() {
      LOG(LOG_INFO, "Freeing %d memobject%s", (int)memobjs.size(), memobjs.size() == 1 ? "" : "s");
      for (int i=0; i<(int)memobjs.size(); i++) {
        dev_free(memobjs.at(i));
      }
    }

    // Destructor
    ~CLWrapper() {
      free_all_memobjs();
      if (command_queue) {
        clReleaseCommandQueue(command_queue);
      }
      if (context) {
        clReleaseContext(context);
      }
      LOG(LOG_INFO, "Releasing %d program%s", (int)programs.size(), programs.size() == 1 ? "" : "s");
      for (int i=0; i<(int)programs.size(); i++) {
        ASSERT_NO_CL_ERROR(clReleaseProgram(programs.at(i)));
      }
      programs.clear();
      LOG(LOG_INFO, "Releasing %d kernel%s", (int)kernelmap.size(), kernelmap.size() == 1 ? "" : "s");
      map<string, cl_kernel>::iterator i;
      for (i=kernelmap.begin(); i != kernelmap.end(); i++) {
        ASSERT_NO_CL_ERROR(clReleaseKernel(i->second));
      }
      kernelmap.clear();
      delete[] devices;
    }

    float memcpy_to_dev(cl_mem buffer, size_t size, const void *ptr, size_t offset=0) {
      cl_bool blocking_write = CL_TRUE;
      cl_uint num_events_in_wait_list = 0;
      cl_event *event_wait_list = NULL;
      cl_event e;
      ASSERT_NO_CL_ERROR(
        clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, &e));
      return time_and_release_event(e);
    }

    float memcpy_from_dev(cl_mem buffer, size_t size, void *ptr, size_t offset=0) {
      cl_bool blocking_read = CL_TRUE;
      cl_uint num_events_in_wait_list = 0;
      cl_event *event_wait_list = NULL;
      cl_event e;
      ASSERT_NO_CL_ERROR(
        clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, &e));
      return time_and_release_event(e);
    }

    float copy_buffer(cl_mem src, cl_mem dst, size_t cb) {
      size_t src_offset = 0;
      size_t dst_offset = 0;
      cl_uint num_events_in_wait_list = 0;
      const cl_event *event_wait_list = NULL;
      cl_event e;
      ASSERT_NO_CL_ERROR(
        clEnqueueCopyBuffer(command_queue, src, dst, src_offset, dst_offset, cb, num_events_in_wait_list, event_wait_list, &e));
      return time_and_release_event(e);
    }

    void run_kernel(cl_kernel kernel, 
      cl_uint work_dim,
      const size_t *global_work_size,
      const size_t *local_work_size,
      const size_t *global_work_offset=NULL,
      cl_uint num_events_in_wait_list=0,
      const cl_event *event_wait_list=NULL,
      cl_event *event=NULL) {
      ASSERT_NO_CL_ERROR(
        clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event));
    }

    cl_kernel &kernel_of_name(const string name) {
      map<string,cl_kernel>::iterator it = kernelmap.find(name);
      if (it == kernelmap.end()) {
        LOG(LOG_FATAL, "Could not find kernel [%s]", name.c_str());
      }
      return it->second;
    }

    void run_kernel(const string kernel_name,
      cl_uint work_dim,
      const size_t *global_work_size,
      const size_t *local_work_size,
      const size_t *global_work_offset=NULL,
      cl_uint num_events_in_wait_list=0,
      const cl_event *event_wait_list=NULL,
      cl_event *event=NULL) {
      run_kernel(kernel_of_name(kernel_name),
        work_dim,
        global_work_size,
        local_work_size,
        global_work_offset,
        num_events_in_wait_list,
        event_wait_list,
        event);
    }

    float run_kernel_with_timing(cl_kernel kernel,
      cl_uint work_dim,
      const size_t *global_work_size,
      const size_t *local_work_size,
      const size_t *global_work_offset=NULL,
      cl_uint num_events_in_wait_list=0,
      const cl_event *event_wait_list=NULL) {
      cl_event e;
      run_kernel(kernel,
        work_dim,
        global_work_size,
        local_work_size,
        global_work_offset,
        num_events_in_wait_list,
        event_wait_list,
        &e);
      return time_and_release_event(e);
    }

    float run_kernel_with_timing(const string kernel_name,
      cl_uint work_dim,
      const size_t *global_work_size,
      const size_t *local_work_size,
      const size_t *global_work_offset=NULL,
      cl_uint num_events_in_wait_list=0,
      const cl_event *event_wait_list=NULL) {
      return run_kernel_with_timing(kernel_of_name(kernel_name),
        work_dim,
        global_work_size,
        local_work_size,
        global_work_offset,
        num_events_in_wait_list,
        event_wait_list);
    }

    cl_kernel &create_kernel(cl_program program, const char*kernel_name) {
      cl_int ret;
      cl_kernel k = clCreateKernel(program, kernel_name, &ret);
      ASSERT_NO_CL_ERROR(ret);
      kernelmap.insert(make_pair(kernel_name, k));
      return kernel_of_name(kernel_name);
    }

    void create_all_kernels(cl_program program) {
      cl_uint num_kernels;
      ASSERT_NO_CL_ERROR(
        clCreateKernelsInProgram(program, /*num_kernels=*/0, /*kernels=*/NULL, &num_kernels));
      cl_kernel *kernels = new cl_kernel[num_kernels];
      assert(kernels);
      ASSERT_NO_CL_ERROR(
        clCreateKernelsInProgram(program, num_kernels, kernels, NULL));
      for (int i=0; i<(int)num_kernels; i++) {
        size_t size;
        ASSERT_NO_CL_ERROR(
          clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, /*size=*/0, /*value=*/NULL, &size));
        char *kernel_name = new char[size];
        ASSERT_NO_CL_ERROR(
          clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, size, kernel_name, NULL));
        kernelmap.insert(make_pair(kernel_name, kernels[i]));
        delete[] kernel_name;
      }
      delete[] kernels;
    }
};

#endif
