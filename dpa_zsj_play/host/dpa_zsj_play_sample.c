/*
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_dev.h>

#include "dpa_common.h"
#include "../include/utils.h"
#include "../include/args.h"

DOCA_LOG_REGISTER(ZSJ_PLAY::SAMPLE);

/* Kernel function decleration */
// extern doca_dpa_func_t l2_batch_kernel;
extern doca_dpa_func_t l2_single_kernel;
extern doca_dpa_func_t l2_batch_kernel;

struct l2_single_dist_args {
    doca_dpa_dev_mmap_t handle;               // 共享内存的handle，DPA凭借这个来访问host mmap出来的内存
    uint64_t a_offset;                        // a的存储地址
    uint64_t b_offset;                        // b的存储地址
    uint64_t out_offset;                      // 计算结果 dist(a,b)的存储地址
    uint64_t dim;                      // e.g. 32
    uint64_t frac_bits;                // q (e.g. 16)
};

struct l2_batch_args {
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
};

static inline uint64_t diff_ns(struct timespec a, struct timespec b) {
    int64_t sec  = (int64_t)b.tv_sec  - (int64_t)a.tv_sec;
    int64_t nsec = (int64_t)b.tv_nsec - (int64_t)a.tv_nsec;
    if (nsec < 0) { nsec += 1000000000LL; sec -= 1; }
    return (uint64_t)sec * 1000000000ULL + (uint64_t)nsec;
}

/*
 * Run kernel_launch sample
 *
 * @resources [in]: DOCA DPA resources that the DPA sample will use
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t kernel_launch(struct dpa_resources *resources)
{
	struct doca_sync_event *wait_event = NULL;
	struct doca_sync_event *comp_event = NULL;
	/* Wait event threshold */
	uint64_t wait_thresh = 0;
	/* Completion event val */
	uint64_t comp_event_val = 64;
	/* Number of DPA threads */
	const unsigned int num_dpa_threads = 64;
	doca_error_t result, tmp_result;
	struct timespec t0, t1;

	/* Creating DOCA sync event for DPA kernel completion */
	result = create_doca_dpa_completion_sync_event(resources->doca_dpa, resources->doca_device, &comp_event);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create DOCA sync event for DPA kernel completion: %s",
			     doca_error_get_descr(result));
		return result;
	}
	const uint32_t dim = 32, q = 16, batch_size = 1024 * 1024; // params
	
	/* allocate host memory for input and output */
	/* single vector distance calc */
	double *a_raw = malloc(dim * sizeof(double) * batch_size);
	double *b_raw = malloc(dim * sizeof(double) * batch_size);

	for(size_t i = 0; i < dim * batch_size; i++)
	{
		a_raw[i] = rand_double(-100.0, 100.0);
		b_raw[i] = rand_double(-100.0, 100.0);
	}

	// double dist_cpu = l2_distance(a_raw, b_raw, dim); // 计算一组a,b的L2 distance


	// 准备a和b的Q16.16格式，位于host memory
	int32_t *a_local = malloc(dim * sizeof(int32_t) * batch_size); // a vector(Quantized)
	int32_t *b_local = malloc(dim * sizeof(int32_t) * batch_size); // b vector(Quantized)

	double_array_to_q16_16(a_raw, a_local, dim * batch_size);
	double_array_to_q16_16(b_raw, b_local, dim * batch_size);

	// 准备用来存dist(a,b)的内存空间
	uint64_t *out_local = malloc(sizeof(uint64_t) * batch_size);
	// 置0
	memset(out_local, 0, sizeof(uint64_t));

	struct doca_mmap *mm = NULL; // mm
    doca_dpa_dev_mmap_t dpa_h = 0;  // DPA访问mmap内存时用到的handler
	// size_t total_bytes = 2 * dim * sizeof(int32_t) + sizeof(uint64_t) + sizeof(l2_single_dist_args); // 大小
	size_t total_bytes = 3 * batch_size * dim * sizeof(int32_t);

	DOCA_LOG_INFO("Start allocating mmap for DPA");
	doca_error_t st;
	// 1. mmap_create
	st = doca_mmap_create(&mm);
    if (st != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create doca_mmap");
		return st;
	}
	DOCA_LOG_INFO("mmap created");
	// 2. mmap_set_permissions
	st = doca_mmap_set_permissions(mm, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_PCI_READ_WRITE);
    if (st != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set permissions");
		doca_mmap_destroy(mm); 
		return st; 
	}
	DOCA_LOG_INFO("mmap oermission set");
	// 3. mmap_add_dev
	st = doca_mmap_add_dev(mm, resources->doca_device);
	if (st != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add dev");
		doca_mmap_destroy(mm);
		return st; 
	}
	DOCA_LOG_INFO("mmap dev added");
	// 4. allocate host memory
	void *buf = NULL;
    if (posix_memalign(&buf, 64, total_bytes) != 0) {
		DOCA_LOG_ERR("Failed to memalign");
        doca_mmap_destroy(mm);
        return DOCA_ERROR_NO_MEMORY;
    }
    memset(buf, 0, total_bytes);
	DOCA_LOG_INFO("host memory allocated");
	// 5. set memrange
    st = doca_mmap_set_memrange(mm, buf, total_bytes);
    if (st != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set memrange");
		free(buf); 
		doca_mmap_destroy(mm);
		return st; 
	}
	DOCA_LOG_INFO("memrange set");
    // 6. start mmap
    st = doca_mmap_start(mm);
    if (st != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start mmap");
        (void)doca_mmap_set_free_cb(mm, NULL, NULL);
        free(buf);
        doca_mmap_destroy(mm);
        return st;
    }
	DOCA_LOG_INFO("mmap started");
	// check 1. check mmap 属性
	uint8_t exported = 0, from_export = 0;
	void *chk_addr = NULL; size_t chk_len = 0;
	doca_error_t st2 = doca_mmap_get_memrange(mm, &chk_addr, &chk_len);
	DOCA_LOG_INFO("mmap memrange addr=%p len=%zu st=%s", chk_addr, chk_len, doca_error_get_descr(st2));

	// check 2. 是否支持 export PCI
	struct doca_devinfo *info = doca_dev_as_devinfo(resources->doca_device);
	uint8_t ok = 0;
	doca_error_t st3 = doca_mmap_cap_is_export_pci_supported(info, &ok);
	DOCA_LOG_INFO("export_pci_supported=%u (%s)", ok, doca_error_get_descr(st3));

    // 拿到 DPA 侧 mmap 句柄：传给 kernel
    st = doca_mmap_dev_get_dpa_handle(mm, resources->doca_device, &dpa_h);
    if (st != DOCA_SUCCESS) {
        doca_mmap_stop(mm);
        (void)doca_mmap_set_free_cb(mm, NULL, NULL);
        free(buf);
        doca_mmap_destroy(mm);
        return st;
    }

	// 把a,b,out拷贝到mmap出来的host memory里面
	memcpy((uint8_t*)buf, a_local, dim * batch_size * sizeof(int32_t));
	memcpy((uint8_t*)buf + dim * batch_size * sizeof(int32_t), b_local, dim * batch_size * sizeof(int32_t));
	memcpy((uint8_t*)buf + dim * batch_size * sizeof(int32_t), out_local, 8 * batch_size);

    /* single kernel args */
    // struct l2_single_dist_args args;
	// args.handle = dpa_h;
	// args.a_offset = buf;
	// args.b_offset = buf + dim * sizeof(int32_t);
	// args.out_offset = buf + 2 * dim * sizeof(int32_t);
	// args.frac_bits = 16;
	// args.dim = 32;

	/* batch kernel args */
	struct l2_batch_args args;
	args.handle = dpa_h;
	args.a_base = buf;
	args.b_base = buf + batch_size * dim * sizeof(int32_t);
	args.out_base = buf + 2 * batch_size * dim * sizeof(int32_t);
	args.a_stride = dim * sizeof(int32_t);
	args.b_stride = dim * sizeof(int32_t);
	args.out_stride = sizeof(int64_t);
	args.frac_bits = 16;
	args.dim = 32;
	args.batch_size = batch_size;

	DOCA_LOG_INFO("All DPA resources have been created\n");
	DOCA_LOG_INFO("args.handle = %u", args.handle);
	DOCA_LOG_INFO("args.dim = %u", args.dim);
	DOCA_LOG_INFO("args.frac_bits = %u", args.frac_bits);
	DOCA_LOG_INFO("args.batch_size = %u", args.batch_size);


	doca_sync_event_get(comp_event, &comp_event_val);
	printf("comp_event_val = %ld\n", comp_event_val);

	/* kernel launch */
	clock_gettime(CLOCK_MONOTONIC, &t0);
	result = doca_dpa_kernel_launch_update_set(resources->doca_dpa,
						   wait_event,
						   wait_thresh,
						   comp_event,
						   comp_event_val + 1,
						   num_dpa_threads,
						   &l2_batch_kernel,
						   args);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to launch zsj's play kernel: %s", doca_error_get_descr(result));
		goto destroy_event;
	}

	/* Wait until completion event reach completion val */
	result = doca_sync_event_wait_gt(comp_event, comp_event_val, SYNC_EVENT_MASK_FFS);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to wait for host completion event: %s", doca_error_get_descr(result));
	clock_gettime(CLOCK_MONOTONIC, &t1);

	uint64_t dpa_time_ns = diff_ns(t0, t1);
	
	clock_gettime(CLOCK_MONOTONIC, &t0);
	for(uint32_t i = 0; i < batch_size; ++i)
	{
		int32_t *a = a_local + i * dim;
		int32_t *b = b_local + i * dim;
		int64_t dist = 0;
		for (uint32_t j = 0; j < dim; ++j) {
            int64_t da = (int64_t)a[j] - (int64_t)b[j];
            dist += da * da;
        }
		out_local[i] = (uint64_t)dist;
	}
	clock_gettime(CLOCK_MONOTONIC, &t1);
	uint64_t cpu_time_ns = diff_ns(t0, t1);
	// uint64_t dist_sq_q2q = *(uint64_t *)args.out_offset;
    // double dist_dpa = sqrt((double)dist_sq_q2q) / (double)(1u << q);

    // printf("CPU L2 = %.8f, DPA L2 = %.8f, abs=%.3e, rel=%.3e\n",
    //        dist_cpu, dist_dpa, fabs(dist_dpa - dist_cpu), (dist_cpu>0 ? fabs(dist_dpa-dist_cpu)/dist_cpu : 0.0));
	printf("Kernel wall time: %.3f ms %lu ns\n", dpa_time_ns / 1e6, dpa_time_ns);
	printf("CPU wall time: %.3f ms %lu ns\n", cpu_time_ns / 1e6, cpu_time_ns);

recycle_event:
	doca_mmap_stop(mm);
	doca_mmap_destroy(mm);

destroy_event:
	/* destroy events */
	tmp_result = doca_sync_event_destroy(comp_event);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy DOCA sync event: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
	return result;
}
