/*
 * demo_routing_v3.c — Shirley Phase 1: Multi-Trit Ternary Input
 *
 * Instead of quantizing 256-level pixels to 1 trit (3 states, 80% info loss),
 * encode each pixel as 5 balanced ternary trits (243 states, <1% info loss).
 *
 * Routing switches from hot-map lookup (27 routing states per block) to
 * sign_epi8 dot product matching against shape prototypes (ternary dot
 * product — the same instruction that benchmarked 6.87x faster than float32).
 *
 * Build:  gcc -O3 -mavx2 -march=native -o demo_routing_v3 demo_routing_v3.c -lm
 * Run:    ./demo_routing_v3 <path-to-mnist-data-dir>
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ================================================================
 *  Configuration
 * ================================================================ */

#define TRAIN_N       60000
#define TEST_N        10000
#define IMG_W         28
#define IMG_H         28
#define PIXELS        784
#define N_CLASSES     10
#define TRITS_PER_PX  5
#define TRIT_DIM      (PIXELS * TRITS_PER_PX)  /* 784 * 5 = 3920 trits per image */
#define TRIT_PADDED   3936                       /* next multiple of 32 */

/* ================================================================
 *  Timing
 * ================================================================ */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 *  MNIST IDX Loader
 * ================================================================ */

static uint8_t *load_idx(const char *path, uint32_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    uint32_t magic, n;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); exit(1); }
    if (fread(&n, 4, 1, f) != 1) { fclose(f); exit(1); }
    magic = __builtin_bswap32(magic);
    n = __builtin_bswap32(n);
    *count = n;
    int ndim = magic & 0xFF;
    size_t item_size = 1;
    if (ndim >= 3) {
        uint32_t rows, cols;
        if (fread(&rows, 4, 1, f) != 1) { fclose(f); exit(1); }
        if (fread(&cols, 4, 1, f) != 1) { fclose(f); exit(1); }
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);
        item_size = (size_t)rows * cols;
    }
    size_t total = (size_t)n * item_size;
    uint8_t *data = malloc(total);
    if (fread(data, 1, total, f) != total) { fclose(f); exit(1); }
    fclose(f);
    return data;
}

/* ================================================================
 *  Balanced Ternary Encoding: uint8 → 5 trits
 *
 *  uint8 (0-255) → balanced ternary with 5 trits (243 states)
 *  Encoding: value mapped to nearest balanced ternary representation
 *
 *  Balanced ternary digit weights: 81, 27, 9, 3, 1
 *  Range: -121 to +121
 *  We map uint8 [0,242] → balanced ternary [-121, +121]
 *  Values 243-255 are clamped to 242 (max representable)
 * ================================================================ */

static void uint8_to_5trits(uint8_t val, int8_t *trits) {
    /* Map 0-255 to -121..+121 range */
    int v = (int)val;
    if (v > 242) v = 242;
    v -= 121;  /* now in range [-121, +121] */

    /* Convert to balanced ternary, most significant trit first */
    static const int weights[5] = {81, 27, 9, 3, 1};
    for (int i = 0; i < 5; i++) {
        int w = weights[i];
        if (v > 0) {
            /* Round to nearest: if v >= w/2 (rounding boundary), use +1 */
            if (v * 2 >= w) {
                trits[i] = 1;
                v -= w;
            } else {
                trits[i] = 0;
            }
        } else if (v < 0) {
            if (-v * 2 >= w) {
                trits[i] = -1;
                v += w;
            } else {
                trits[i] = 0;
            }
        } else {
            trits[i] = 0;
        }
    }
}

/* Verify: convert 5 trits back to integer */
static int trits5_to_int(const int8_t *trits) {
    static const int weights[5] = {81, 27, 9, 3, 1};
    int v = 0;
    for (int i = 0; i < 5; i++)
        v += trits[i] * weights[i];
    return v;
}

/* ================================================================
 *  Encode full image: uint8[784] → int8[3920] (5 trits per pixel)
 * ================================================================ */

static void encode_image(const uint8_t *src, int8_t *dst) {
    for (int i = 0; i < PIXELS; i++)
        uint8_to_5trits(src[i], dst + i * TRITS_PER_PX);
    /* Zero-pad to TRIT_PADDED */
    memset(dst + TRIT_DIM, 0, TRIT_PADDED - TRIT_DIM);
}

/* ================================================================
 *  AVX2 Ternary Dot Product (from bench_dot.c / sstt_v2.c)
 * ================================================================ */

static inline int32_t ternary_dot(const int8_t *a, const int8_t *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < TRIT_PADDED; i += 32) {
        __m256i va = _mm256_load_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_load_si256((const __m256i *)(b + i));
        __m256i prod = _mm256_sign_epi8(va, vb);
        acc = _mm256_add_epi8(acc, prod);
    }
    /* Widen to avoid overflow: epi8 → epi16 → epi32 */
    __m256i lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc, 1));
    __m256i sum16 = _mm256_add_epi16(lo16, hi16);
    __m256i sum32 = _mm256_madd_epi16(sum16, _mm256_set1_epi16(1));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum32),
                              _mm256_extracti128_si256(sum32, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

/* ================================================================
 *  Class Prototypes: mean ternary image per class
 *
 *  For each class, compute the element-wise sign of the sum of
 *  all training images' trit vectors. This gives a ternary prototype
 *  that captures the "average" shape of each digit.
 *
 *  This IS a frozen shape — computed once from training data,
 *  never modified. The routing decision is: which prototype
 *  has the highest dot product with the input?
 * ================================================================ */

static int8_t prototypes[N_CLASSES][TRIT_PADDED] __attribute__((aligned(32)));

static void build_prototypes(const uint8_t *train_imgs,
                              const uint8_t *labels, int n) {
    /* Accumulate per-class sums */
    int32_t *sums[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) {
        sums[c] = calloc(TRIT_PADDED, sizeof(int32_t));
    }

    int8_t *trit_buf = aligned_alloc(32, TRIT_PADDED);

    for (int i = 0; i < n; i++) {
        encode_image(train_imgs + (size_t)i * PIXELS, trit_buf);
        int lbl = labels[i];
        for (int j = 0; j < TRIT_DIM; j++)
            sums[lbl][j] += trit_buf[j];
    }

    /* Sign → ternary prototype */
    for (int c = 0; c < N_CLASSES; c++) {
        for (int j = 0; j < TRIT_DIM; j++) {
            if (sums[c][j] > 0)       prototypes[c][j] = 1;
            else if (sums[c][j] < 0)  prototypes[c][j] = -1;
            else                       prototypes[c][j] = 0;
        }
        memset(prototypes[c] + TRIT_DIM, 0, TRIT_PADDED - TRIT_DIM);
        free(sums[c]);
    }
    free(trit_buf);
}

/* ================================================================
 *  k-Nearest Neighbor with ternary dot product
 *
 *  For higher accuracy: compare input against all training images
 *  using sign_epi8 dot product, take the k nearest, majority vote.
 *  This is brute-force routing — every training image is a "shape"
 *  and the routing finds the best match.
 * ================================================================ */

static int8_t *all_train_trits;  /* [TRAIN_N * TRIT_PADDED] */

static void encode_all_training(const uint8_t *train_imgs, int n) {
    all_train_trits = aligned_alloc(32, (size_t)n * TRIT_PADDED);
    for (int i = 0; i < n; i++)
        encode_image(train_imgs + (size_t)i * PIXELS,
                     all_train_trits + (size_t)i * TRIT_PADDED);
}

/* Classification by prototype matching (nearest centroid) */
static int classify_prototype(const int8_t *query) {
    int best = 0;
    int32_t best_score = ternary_dot(query, prototypes[0]);
    for (int c = 1; c < N_CLASSES; c++) {
        int32_t score = ternary_dot(query, prototypes[c]);
        if (score > best_score) {
            best_score = score;
            best = c;
        }
    }
    return best;
}

/* Classification by 1-NN (brute force, ternary dot product) */
static int classify_1nn(const int8_t *query, const uint8_t *labels) {
    int32_t best_score = -999999;
    int best_idx = 0;
    for (int i = 0; i < TRAIN_N; i++) {
        int32_t score = ternary_dot(query,
            all_train_trits + (size_t)i * TRIT_PADDED);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }
    return labels[best_idx];
}

/* Classification by k-NN (k=3, majority vote) */
static int classify_3nn(const int8_t *query, const uint8_t *labels) {
    int32_t top_scores[3] = {-999999, -999999, -999999};
    int top_idx[3] = {0, 0, 0};

    for (int i = 0; i < TRAIN_N; i++) {
        int32_t score = ternary_dot(query,
            all_train_trits + (size_t)i * TRIT_PADDED);
        /* Insert into top-3 if better than worst */
        int worst = 0;
        if (top_scores[1] < top_scores[worst]) worst = 1;
        if (top_scores[2] < top_scores[worst]) worst = 2;
        if (score > top_scores[worst]) {
            top_scores[worst] = score;
            top_idx[worst] = i;
        }
    }

    /* Majority vote */
    int votes[N_CLASSES] = {0};
    for (int i = 0; i < 3; i++)
        votes[labels[top_idx[i]]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (votes[c] > votes[best]) best = c;
    return best;
}

/* ================================================================
 *  Main
 * ================================================================ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mnist-data-dir>\n", argv[0]);
        return 1;
    }

    const char *dir = argv[1];
    char path[512];

    printf("Shirley Phase 1 — Multi-Trit Ternary Input + Dot Product Routing\n");
    printf("=================================================================\n\n");

    /* Load MNIST */
    printf("Loading MNIST...\n");
    uint32_t n;
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", dir);
    uint8_t *train_imgs = load_idx(path, &n);
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", dir);
    uint8_t *train_labels = load_idx(path, &n);
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", dir);
    uint8_t *test_imgs = load_idx(path, &n);
    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", dir);
    uint8_t *test_labels = load_idx(path, &n);

    /* Verify encoding round-trip */
    printf("\nVerifying balanced ternary encoding...\n");
    int max_err = 0;
    for (int v = 0; v <= 255; v++) {
        int8_t trits[5];
        uint8_to_5trits((uint8_t)v, trits);
        int reconstructed = trits5_to_int(trits) + 121;
        int err = abs(v - reconstructed);
        if (err > max_err) max_err = err;
    }
    printf("  Max round-trip error: %d (across all 256 uint8 values)\n", max_err);
    printf("  Trits per pixel: %d (3^%d = %d states)\n",
           TRITS_PER_PX, TRITS_PER_PX, 243);
    printf("  Trit vector length: %d trits per image (%d padded)\n",
           TRIT_DIM, TRIT_PADDED);
    printf("  Memory per image: %d bytes (trit) vs %d bytes (float32)\n",
           TRIT_PADDED, PIXELS * 4);

    /* Build prototypes */
    printf("\nBuilding class prototypes (frozen shapes)...\n");
    double t0 = now_sec();
    build_prototypes(train_imgs, train_labels, TRAIN_N);
    double t1 = now_sec();
    printf("  Built %d prototypes in %.3f sec\n", N_CLASSES, t1 - t0);

    /* Count sparsity of prototypes */
    for (int c = 0; c < N_CLASSES; c++) {
        int nonzero = 0;
        for (int j = 0; j < TRIT_DIM; j++)
            if (prototypes[c][j] != 0) nonzero++;
        printf("  Class %d: %d/%d trits active (%.1f%% sparse)\n",
               c, nonzero, TRIT_DIM, 100.0 * (1.0 - (double)nonzero / TRIT_DIM));
    }

    /* Encode all training data for kNN */
    printf("\nEncoding all training images to 5-trit ternary...\n");
    t0 = now_sec();
    encode_all_training(train_imgs, TRAIN_N);
    t1 = now_sec();
    printf("  Encoded %d images in %.3f sec (%.1f MB)\n",
           TRAIN_N, t1 - t0,
           (double)TRAIN_N * TRIT_PADDED / (1024.0 * 1024.0));

    /* Allocate query buffer */
    int8_t *query = aligned_alloc(32, TRIT_PADDED);

    /* === Prototype matching (nearest centroid) === */
    printf("\n--- Prototype matching (10 frozen shapes, sign_epi8 routing) ---\n");
    int correct = 0;
    t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        encode_image(test_imgs + (size_t)i * PIXELS, query);
        int pred = classify_prototype(query);
        if (pred == test_labels[i]) correct++;
    }
    t1 = now_sec();
    printf("  Accuracy:   %.2f%% (%d/%d)\n", 100.0 * correct / TEST_N, correct, TEST_N);
    printf("  Time:       %.1f ms (%.1f µs/image)\n",
           (t1 - t0) * 1000, (t1 - t0) * 1e6 / TEST_N);
    printf("  Throughput: %.0f images/sec\n", TEST_N / (t1 - t0));
    printf("  Shapes:     10 (one ternary prototype per class)\n");
    printf("  Routing:    sign_epi8 dot product, argmax\n");

    /* === 1-NN (brute force, every training image is a shape) === */
    printf("\n--- 1-NN (60000 shapes, sign_epi8 routing) ---\n");
    correct = 0;
    t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        encode_image(test_imgs + (size_t)i * PIXELS, query);
        int pred = classify_1nn(query, train_labels);
        if (pred == test_labels[i]) correct++;
    }
    t1 = now_sec();
    printf("  Accuracy:   %.2f%% (%d/%d)\n", 100.0 * correct / TEST_N, correct, TEST_N);
    printf("  Time:       %.1f sec (%.1f ms/image)\n",
           t1 - t0, (t1 - t0) * 1000 / TEST_N);
    printf("  Throughput: %.1f images/sec\n", TEST_N / (t1 - t0));
    printf("  Shapes:     60000 (every training image)\n");
    printf("  Routing:    sign_epi8 dot product, nearest match\n");

    /* === 3-NN (brute force, majority vote) === */
    printf("\n--- 3-NN (60000 shapes, sign_epi8 routing, majority vote) ---\n");
    correct = 0;
    t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        encode_image(test_imgs + (size_t)i * PIXELS, query);
        int pred = classify_3nn(query, train_labels);
        if (pred == test_labels[i]) correct++;
    }
    t1 = now_sec();
    printf("  Accuracy:   %.2f%% (%d/%d)\n", 100.0 * correct / TEST_N, correct, TEST_N);
    printf("  Time:       %.1f sec (%.1f ms/image)\n",
           t1 - t0, (t1 - t0) * 1000 / TEST_N);
    printf("  Throughput: %.1f images/sec\n", TEST_N / (t1 - t0));
    printf("  Shapes:     60000 (every training image)\n");
    printf("  Routing:    sign_epi8 dot product, top-3, majority vote\n");

    /* Summary */
    printf("\n=== SUMMARY ===\n\n");
    printf("  Input representation: 5 balanced ternary trits per pixel\n");
    printf("    - 243 states per pixel (vs 3 in v1/v2 pipeline)\n");
    printf("    - <1%% information loss (vs 80%% in 1-trit quantization)\n");
    printf("    - Routing via sign_epi8 dot product (6.87x faster than float32)\n\n");
    printf("  All computation uses MUL (sign_epi8), ADD (add_epi8), MAX (argmax)\n");
    printf("  Zero floating point. Zero learned parameters.\n");
    printf("  The prototypes are frozen shapes. The training images are frozen shapes.\n");
    printf("  sign_epi8 is simultaneously the routing function AND the compute.\n");

    free(query);
    free(all_train_trits);
    free(train_imgs);
    free(train_labels);
    free(test_imgs);
    free(test_labels);

    return 0;
}
