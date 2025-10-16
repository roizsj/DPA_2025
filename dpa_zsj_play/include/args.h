#pragma once
#include <stdint.h>

/* 在设备端(dpacc)编译时，__DOCA_DPA__ 通常会被定义 */
#ifdef __DOCA_DPA__
  #include <doca_dpa_dev.h>               // 只在设备端包含
  #include <doca_mmap.h>                  // 只在设备端包含
  #include <doca_dpa.h>
  #include <doca_dpa_dev_buf.h>
  typedef doca_dpa_dev_uintptr_t dpa_uaddr_t;
  #define DPA_PARAM __dpa_global__
#else
  typedef uint64_t                dpa_uaddr_t;  // Host 侧用 64 位无符号代替设备地址类型
  #define DPA_PARAM                        /* Host 侧为空 */
#endif

// typedef DPA_PARAM struct {
//     doca_dpa_dev_mmap_t handle;               // 共享内存的handle，DPA凭借这个来访问host mmap出来的内存
//     uint32_t a_offset;                        // a's offset
//     uint32_t b_offset;                        // b's offset
//     uint32_t out_offset;                      // dist(a,b)'s offset
//     uint32_t dim;                      // e.g. 32
//     uint32_t frac_bits;                // q (e.g. 16)
// }l2_single_dist_args ;
