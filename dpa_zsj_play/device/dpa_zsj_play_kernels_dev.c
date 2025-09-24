#include <stdint.h>
#include <doca_dpa_dev.h>

__dpa_global__ typedef struct {
    doca_dpa_dev_uintptr_t a_base;     // const int32_t*  (Qq)
    doca_dpa_dev_uintptr_t B_base;     // const int32_t*  (T x dim, row-major)
    doca_dpa_dev_uintptr_t normB_base; // const uint64_t* (预存每行范数的Q(2q)值)
    doca_dpa_dev_uintptr_t out_base;   // uint64_t*       (输出平方距离Q(2q))
    uint32_t dim;                      // e.g. 32
    uint32_t T;                        // 任务数
    uint32_t frac_bits;                // q (e.g. 16)
    uint64_t sumA;                     // ||a||^2 in Q(2q)
} l2_dist_args;

__dpa_global__ void l2_batch_kernel(l2_dist_args args)
{
	DOCA_DPA_DEV_LOG_INFO("Hello from kernel\n");
    const int32_t *a   = (const int32_t *)args.a_base;
    const int32_t *B   = (const int32_t *)args.B_base;
    const uint64_t *nB = (const uint64_t *)args.normB_base;
    uint64_t *out      = (uint64_t *)args.out_base;
    const uint32_t dim = args.dim;
    const uint32_t T   = args.T;
    const uint64_t sumA = args.sumA;      // Q(2q)
    const uint32_t tid = doca_dpa_dev_thread_rank();
    const uint32_t nth = doca_dpa_dev_num_threads();


    for (uint32_t t = tid; t < T; t += nth) {
        const int32_t *b = B + (size_t)t * dim;

        // 点积：Qq * Qq 累加 => Q(2q)
        int64_t acc = 0;
        for (uint32_t i = 0; i < dim; ++i)
            acc += (int64_t)a[i] * (int64_t)b[i];   // Q(2q)

        // ||a-b||^2 = ||a||^2 + ||b||^2 - 2 * dot(a,b)
        // 注意：acc 是有符号 Q(2q)，组装为无符号输出时做有界处理
        uint64_t val;
        if (acc >= 0) {
            // sumA + nB - 2*dot
            val = sumA + nB[t] - ( (uint64_t)acc << 1 );
        } else {
            val = sumA + nB[t] + ( (uint64_t)(-acc) << 1 );
        }
        out[t] = val; // Q(2q)
    }
}