#include "../include/utils.h"

/* 单个 float 转 Q16.16 */
inline int32_t float_to_q16_16(float x)
{
    const int frac_bits = 16;
    const float scale = (float)(1 << frac_bits);

    double scaled = (double)x * scale;
    long long val = llround(scaled);

    if (val > INT32_MAX) val = INT32_MAX;
    if (val < INT32_MIN) val = INT32_MIN;

    return (int32_t)val;
}

/* 单个 double 转 Q16.16 */
inline int32_t double_to_q16_16(double x)
{
    const int frac_bits = 16;
    const double scale = (double)(1 << frac_bits);

    long long val = llround(x * scale);

    if (val > INT32_MAX) val = INT32_MAX;
    if (val < INT32_MIN) val = INT32_MIN;

    return (int32_t)val;
}

/* Q16.16 转回 float */
inline float q16_16_to_float(int32_t qval)
{
    const int frac_bits = 16;
    return (double)qval / (double)(1 << frac_bits);
}

/* Q16.16 转回 double */
inline double q16_16_to_double(int32_t qval)
{
    const int frac_bits = 16;
    return (double)qval / (double)(1 << frac_bits);
}


/* 批量 float[] -> Q16.16 int32[] */
void float_array_to_q16_16(const float *src, int32_t *dst, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        dst[i] = float_to_q16_16(src[i]);
    }
}

/* 批量 Q16.16 int32[] -> float[] */
void q16_16_array_to_float(const int32_t *src, float *dst, size_t len)
{
    const int frac_bits = 16;
    const float scale = (float)(1 << frac_bits);

    for (size_t i = 0; i < len; i++) {
        dst[i] = (float)src[i] / scale;
    }
}

/* 批量 double[] -> Q16.16 int32[] */
void double_array_to_q16_16(const double *src, int32_t *dst, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        dst[i] = double_to_q16_16(src[i]);
    }
}

/* 批量 Q16.16 int32[] -> double[] */
void q16_16_array_to_double(const int32_t *src, double *dst, size_t len)
{
    const int frac_bits = 16;
    const double scale = (double)(1 << frac_bits);

    for (size_t i = 0; i < len; i++) {
        dst[i] = (double)src[i] / scale;
    }
}

/* 生成 [min, max) 区间的均匀随机 double */
inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / (double)RAND_MAX);
}

/* 计算欧氏距离 L2 */
double l2_distance(const double *a, const double *b, size_t dim) {
    double acc = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double d = a[i] - b[i];
        acc += d * d;
    }
    return sqrt(acc);
}
