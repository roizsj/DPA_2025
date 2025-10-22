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

typedef __dpa_global__ struct {
    doca_dpa_dev_mmap_t handle;

    uint64_t a_base;
    uint64_t b_base;
    uint64_t out_base;

    uint64_t a_stride;     // 每个向量 a 的步长（字节）
    uint64_t b_stride;     // 每个向量 b 的步长（字节）
    uint64_t out_stride;   // 每个输出 dist 的步长（字节）

    uint32_t dim;          // 向量维度（例如 32）
    uint32_t frac_bits;    // 定点位（例如 16）
    uint32_t batch_size;   // 这一批里有多少个距离要算
} l2_batch_args;

__dpa_global__ void l2_single_kernel(l2_single_dist_args args)
{
    const int32_t *a = (int32_t*)doca_dpa_dev_mmap_get_external_ptr(args.handle, args.a_offset);
    const int32_t *b = (int32_t*)doca_dpa_dev_mmap_get_external_ptr(args.handle, args.b_offset);
    uint64_t *out = (uint64_t *)doca_dpa_dev_mmap_get_external_ptr(args.handle, args.out_offset);
    uint32_t dim = args.dim;

    // (a[i]-b[i])^2 累加 => 2Q(2q)
    int64_t dist = 0;

    for (uint32_t i = 0; i < dim; ++i)
        dist += ((int64_t)a[i] - (int64_t)b[i]) * ((int64_t)a[i] - (int64_t)b[i]);   // 2Q(2q)

    *out = dist; // 2Q(2q)
    DOCA_DPA_DEV_LOG_INFO("End l2 single kernel\n");
}

__dpa_global__ void l2_batch_kernel(l2_batch_args args)
{
    // DOCA_DPA_DEV_LOG_INFO("start batch kernel\n");
    unsigned int rank = doca_dpa_dev_thread_rank() % doca_dpa_dev_num_threads();
    unsigned int num_threads = doca_dpa_dev_num_threads();

    for (uint32_t idx = rank; idx < args.batch_size; idx += num_threads) {
        const int32_t *a = (int32_t*)doca_dpa_dev_mmap_get_external_ptr(
            args.handle, args.a_base + (uint64_t)idx * args.a_stride);
        const int32_t *b = (int32_t*)doca_dpa_dev_mmap_get_external_ptr(
            args.handle, args.b_base + (uint64_t)idx * args.b_stride);
        uint64_t *out = (uint64_t*)doca_dpa_dev_mmap_get_external_ptr(
            args.handle, args.out_base + (uint64_t)idx * args.out_stride);

        int64_t dist = 0;
        #pragma unroll
        for (uint32_t i = 0; i < args.dim; ++i) {
            int64_t da = (int64_t)a[i] - (int64_t)b[i];
            dist += da * da;
        }
        *out = (uint64_t)dist;
    }
}