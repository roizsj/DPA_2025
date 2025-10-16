#ifndef UTIL_H
#define UTIL_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <time.h>

/* 单个 float 转 Q16.16 */
int32_t float_to_q16_16(float x);

/* 单个 double 转 Q16.16 */
int32_t double_to_q16_16(double x);

/* Q16.16 转回 float */
float q_16_16_to_float(int32_t qval);

/* Q16.16 转回 double */
double q16_16_to_double(int32_t qval);


/* 批量 float[] -> Q16.16 int32[] */
void float_array_to_q16_16(const float *src, int32_t *dst, size_t len);

/* 批量 Q16.16 int32[] -> float[] */
void q16_16_array_to_float(const int32_t *src, float *dst, size_t len);

/* 批量 double[] -> Q16.16 int32[] */
void double_array_to_q16_16(const double *src, int32_t *dst, size_t len);

/* 批量 Q16.16 int32[] -> double[] */
void q16_16_array_to_double(const int32_t *src, double *dst, size_t len);

/* 生成 [min, max) 区间的均匀随机 double */
double rand_double(double min, double max);

/* 计算欧氏距离L2 */
double l2_distance(const double *a, const double *b, size_t dim);

#endif // UTIL_H


