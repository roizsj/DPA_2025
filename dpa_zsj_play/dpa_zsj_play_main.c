/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#include <doca_error.h>
#include <doca_log.h>

#include "dpa_common.h"

DOCA_LOG_REGISTER(ZSJ_PLAY::MAIN);

/* Sample's Logic */
doca_error_t kernel_launch(struct dpa_resources *resources);

/*
 * Sample main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int main(int argc, char **argv)
{
	struct dpa_config cfg = {{0}};
	struct dpa_resources resources = {0};
	doca_error_t result;
	struct doca_log_backend *sdk_log;
	int exit_status = EXIT_FAILURE;

	/* Set default value for device name */
	// strcpy(cfg.device_name, DEVICE_DEFAULT_NAME);
	strcpy(cfg.device_name, "mlx5_1");
	// 敢情之前测的时候这个device就是本机
	// 要改成DPA再用！
	// mlx5_1是对的 mlx5_0找不到设备，mlx5_2没有分配EU（其中原理暂时还不太懂）


	/* Register a logger backend */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	/* Register a logger backend for internal SDK errors and warnings */
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		goto sample_exit;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	DOCA_LOG_INFO("Starting the sample");

	result = doca_argp_init("doca_dpa_zsj_play", &cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		goto sample_exit;
	}

	/* Register DPA params */
	result = register_dpa_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register sample parameters: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse sample input: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	/* Allocating resources */ // resources里面有doca_dev(设备信息)和doca_dpa(context)
	result = allocate_dpa_resources(&cfg, &resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to Allocate DPA Resources: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	/* Running sample */
	result = kernel_launch(&resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("kernel_launch() encountered an error: %s", doca_error_get_descr(result));
		goto dpa_cleanup;
	}

	exit_status = EXIT_SUCCESS;

dpa_cleanup:
	/* Destroying DPA resources */
	result = destroy_dpa_resources(&resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy DOCA DPA resources: %s", doca_error_get_descr(result));
		exit_status = EXIT_FAILURE;
	}
argp_cleanup:
	doca_argp_destroy();
sample_exit:
	if (exit_status == EXIT_SUCCESS)
		DOCA_LOG_INFO("Sample finished successfully");
	else
		DOCA_LOG_INFO("Sample finished with errors");
	return exit_status;
}
