#include <stdint.h>                          
#include <doca_dpa_dev_buf.h>

#include "../include/args.h"

typedef __dpa_global__ struct {
    doca_dpa_dev_mmap_t handle;               // 共享内存的handle，DPA凭借这个来访问host mmap出来的内存
    uint64_t a_offset;                        // a's offset
    uint64_t b_offset;                        // b's offset
    uint64_t out_offset;                      // dist(a,b)'s offset
    uint64_t dim;                      // e.g. 32
    uint64_t frac_bits;                // q (e.g. 16)
}l2_single_dist_args;

__dpa_global__ void l2_single_kernel(l2_single_dist_args args)
{
    const int32_t *a = (int32_t*)doca_dpa_dev_mmap_get_external_ptr(args.handle, args.a_offset);
    const int32_t *b = (int32_t*)doca_dpa_dev_mmap_get_external_ptr(args.handle, args.b_offset);
    uint64_t *out = (uint64_t *)doca_dpa_dev_mmap_get_external_ptr(args.handle, args.out_offset);
    uint32_t dim = args.dim;

    // (a[i]-b[i])^2 累加 => 2Q(2q)
    int64_t dist = 0;
    DOCA_DPA_DEV_LOG_INFO("2222");

    for (uint32_t i = 0; i < dim; ++i)
        dist += ((int64_t)a[i] - (int64_t)b[i]) * ((int64_t)a[i] - (int64_t)b[i]);   // 2Q(2q)

    *out = dist; // 2Q(2q)
    DOCA_DPA_DEV_LOG_INFO("End l2 single kernel\n");
}