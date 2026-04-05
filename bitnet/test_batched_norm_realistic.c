/*
 * test_batched_norm_realistic.c — Check actual matmul output distribution
 *
 * Simulate: post-RMSNorm activations × random ternary weights
 * Measure the ratio between max and min |dot product| across rows.
 * If the ratio is small, batched normalization works.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

int main(void) {
    printf("Matmul output distribution analysis\n");
    printf("====================================\n\n");

    int n_inner = 2560;
    int n_output = 2560;
    int16_t * act = (int16_t *)malloc(n_inner * sizeof(int16_t));
    int8_t * wt = (int8_t *)malloc(n_inner * sizeof(int8_t));
    int32_t * raw = (int32_t *)malloc(n_output * sizeof(int32_t));

    int trials = 100;
    double sum_ratio = 0;
    double max_ratio = 0;
    int zero_rows_total = 0;

    for (int t = 0; t < trials; t++) {
        srand(t + 42);

        /* Simulate post-RMSNorm activations: normalized, tight range */
        for (int i = 0; i < n_inner; i++) {
            act[i] = (int16_t)((rand() % 2000) - 1000);  /* [-1000, 1000] */
        }

        /* For each output row, use different ternary weights */
        int32_t max_abs = 0;
        int32_t min_abs_nonzero = INT32_MAX;
        int zero_rows = 0;

        for (int row = 0; row < n_output; row++) {
            /* Random ternary weights with ~33% sparsity */
            int32_t dot = 0;
            for (int i = 0; i < n_inner; i++) {
                int w = (rand() % 3) - 1;  /* {-1, 0, 1} */
                dot += (int32_t)act[i] * w;
            }
            raw[row] = dot;

            int32_t a = dot > 0 ? dot : -dot;
            if (a > max_abs) max_abs = a;
            if (a > 0 && a < min_abs_nonzero) min_abs_nonzero = a;
            if (a == 0) zero_rows++;
        }

        double ratio = (min_abs_nonzero > 0 && min_abs_nonzero < INT32_MAX)
            ? (double)max_abs / (double)min_abs_nonzero : 0;
        sum_ratio += ratio;
        if (ratio > max_ratio) max_ratio = ratio;
        zero_rows_total += zero_rows;
    }

    printf("  Trials: %d, n_inner=%d, n_output=%d\n", trials, n_inner, n_output);
    printf("  Avg max/min ratio: %.1f\n", sum_ratio / trials);
    printf("  Max max/min ratio: %.1f\n", max_ratio);
    printf("  Avg zero rows per trial: %.1f\n", (double)zero_rows_total / trials);
    printf("\n  MTFP21 mantissa range: 21523360\n");
    printf("  If max/min < ~21M (mantissa range), batched norm preserves all values.\n");
    printf("  If max/min > ~21M, small values get crushed.\n");

    double threshold = 21523360.0;
    if (max_ratio < threshold) {
        printf("\n  VERDICT: max/min ratio (%.1f) << MTFP21 range (%.0f)\n", max_ratio, threshold);
        printf("  Batched normalization is SAFE for this data distribution.\n");
    } else {
        printf("\n  VERDICT: max/min ratio (%.1f) exceeds MTFP21 range\n", max_ratio);
        printf("  Batched normalization will crush small values.\n");
    }

    free(act); free(wt); free(raw);
    return 0;
}
