/*
 * demo_routing_v5.c — Shirley Phase 1.5: Hamming-Accelerated Routing
 *
 * Replaces brute-force 60K dot products with LCVDB-style two-stage routing:
 *   Stage 1: Hamming distance triage (XOR + POPCNT) → top-K shortlist
 *   Stage 2: sign_epi8 dot product rerank on shortlist only
 *
 * Compares against brute-force 1-NN and 3-NN from v3.
 *
 * Build:  gcc -O3 -mavx2 -march=native -mpopcnt -o demo_routing_v5 demo_routing_v5.c -lm
 * Run:    ./demo_routing_v5 <path-to-mnist-data-dir>
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

/* 5-trit encoding (winner from v3/v4 comparison) */
#define TRITS_PER_PX  5
#define TRIT_DIM      (PIXELS * TRITS_PER_PX)  /* 3920 */
#define TRIT_PADDED   3936                       /* next multiple of 32 */

/* Hamming fingerprint: collapse 5-trit values to 1-bit sign
 * 784 pixels → 784 bits → 98 bytes, pad to 128 bytes for AVX2 */
#define FP_BITS       PIXELS                     /* 784 bits */
#define FP_BYTES      128                        /* padded to 4 × 32-byte AVX2 regs */

/* Shortlist sizes to test */
#define SL_50         50
#define SL_100        100
#define SL_200        200

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
 *  5-Trit Balanced Ternary Encoding (from v3)
 * ================================================================ */

static void uint8_to_5trits(uint8_t val, int8_t *trits) {
    int v = (int)val;
    if (v > 242) v = 242;
    v -= 121;
    static const int weights[5] = {81, 27, 9, 3, 1};
    for (int i = 0; i < 5; i++) {
        int w = weights[i];
        if (v > 0) {
            if (v * 2 >= w) { trits[i] = 1; v -= w; }
            else trits[i] = 0;
        } else if (v < 0) {
            if (-v * 2 >= w) { trits[i] = -1; v += w; }
            else trits[i] = 0;
        } else {
            trits[i] = 0;
        }
    }
}

static void encode_image_trits(const uint8_t *src, int8_t *dst) {
    for (int i = 0; i < PIXELS; i++)
        uint8_to_5trits(src[i], dst + i * TRITS_PER_PX);
    memset(dst + TRIT_DIM, 0, TRIT_PADDED - TRIT_DIM);
}

/* ================================================================
 *  Hamming Fingerprint: Collapse pixel to 1-bit sign
 *
 *  For each pixel, sum its 5 trits. If sum > 0: bit=1, else bit=0.
 *  This captures the dominant polarity of each pixel position.
 *  784 pixels → 784 bits → 98 bytes → pad to 128 bytes.
 *
 *  Hamming distance between fingerprints ≈ coarse similarity.
 *  Low Hamming distance → similar images → worth reranking.
 * ================================================================ */

static void encode_fingerprint(const int8_t *trits, uint8_t *fp) {
    memset(fp, 0, FP_BYTES);
    for (int p = 0; p < PIXELS; p++) {
        /* Sum the 5 trits for this pixel */
        int sum = 0;
        for (int t = 0; t < TRITS_PER_PX; t++)
            sum += trits[p * TRITS_PER_PX + t];
        /* Set bit if sum > 0 (positive polarity) */
        if (sum > 0)
            fp[p / 8] |= (1u << (p % 8));
    }
}

/* Also build a richer 2-bit fingerprint: 3 states per pixel
 * Positive (sum > threshold), negative (sum < -threshold), neutral
 * Packed as 2 bits per pixel: 784 × 2 = 1568 bits = 196 bytes → pad to 224 */
#define FP2_BYTES     224
#define FP2_THRESHOLD 1

static void encode_fingerprint_2bit(const int8_t *trits, uint8_t *fp_pos, uint8_t *fp_neg) {
    memset(fp_pos, 0, FP2_BYTES / 2);
    memset(fp_neg, 0, FP2_BYTES / 2);
    for (int p = 0; p < PIXELS; p++) {
        int sum = 0;
        for (int t = 0; t < TRITS_PER_PX; t++)
            sum += trits[p * TRITS_PER_PX + t];
        if (sum > FP2_THRESHOLD)
            fp_pos[p / 8] |= (1u << (p % 8));
        else if (sum < -FP2_THRESHOLD)
            fp_neg[p / 8] |= (1u << (p % 8));
    }
}

/* ================================================================
 *  AVX2 Hamming Distance
 *  Adapted from LCVDB's sce_hamming192 (Mula/Harley VPSHUFB popcount)
 * ================================================================ */

static inline int32_t hamming_distance(const uint8_t *a, const uint8_t *b, int nbytes) {
    const __m256i lo_mask = _mm256_set1_epi8(0x0F);
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    __m256i acc = _mm256_setzero_si256();

    int i;
    for (i = 0; i + 32 <= nbytes; i += 32) {
        __m256i x = _mm256_xor_si256(
            _mm256_loadu_si256((const __m256i *)(a + i)),
            _mm256_loadu_si256((const __m256i *)(b + i)));

        __m256i lo = _mm256_and_si256(x, lo_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), lo_mask);

        __m256i pc = _mm256_add_epi8(
            _mm256_shuffle_epi8(lut, lo),
            _mm256_shuffle_epi8(lut, hi));

        acc = _mm256_add_epi64(acc,
            _mm256_sad_epu8(pc, _mm256_setzero_si256()));
    }

    __m128i lo128 = _mm256_castsi256_si128(acc);
    __m128i hi128 = _mm256_extracti128_si256(acc, 1);
    __m128i sum = _mm_add_epi64(lo128, hi128);
    int32_t result = (int32_t)(_mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1));

    /* Scalar tail */
    for (; i < nbytes; i++) {
        uint8_t x = a[i] ^ b[i];
        result += __builtin_popcount(x);
    }
    return result;
}

/* 2-bit Hamming: distance on (fp_pos XOR fp_pos) + (fp_neg XOR fp_neg) */
static inline int32_t hamming_distance_2bit(
    const uint8_t *a_pos, const uint8_t *a_neg,
    const uint8_t *b_pos, const uint8_t *b_neg, int nbytes_half) {
    return hamming_distance(a_pos, b_pos, nbytes_half) +
           hamming_distance(a_neg, b_neg, nbytes_half);
}

/* ================================================================
 *  AVX2 Ternary Dot Product (from v3, with overflow-safe accumulation)
 * ================================================================ */

static inline int32_t ternary_dot(const int8_t *a, const int8_t *b) {
    __m256i acc32 = _mm256_setzero_si256();
    int total_iters = TRIT_PADDED / 32;
    int chunk = 96;

    for (int start = 0; start < total_iters; start += chunk) {
        int end = start + chunk;
        if (end > total_iters) end = total_iters;

        __m256i acc8 = _mm256_setzero_si256();
        for (int i = start; i < end; i++) {
            __m256i va = _mm256_load_si256((const __m256i *)(a + i * 32));
            __m256i vb = _mm256_load_si256((const __m256i *)(b + i * 32));
            acc8 = _mm256_add_epi8(acc8, _mm256_sign_epi8(va, vb));
        }
        __m256i lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc8));
        __m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc8, 1));
        __m256i sum16 = _mm256_add_epi16(lo16, hi16);
        __m256i sum32 = _mm256_madd_epi16(sum16, _mm256_set1_epi16(1));
        acc32 = _mm256_add_epi32(acc32, sum32);
    }

    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc32),
                              _mm256_extracti128_si256(acc32, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

/* ================================================================
 *  Data Arrays
 * ================================================================ */

static int8_t  *all_train_trits;    /* [TRAIN_N × TRIT_PADDED] */
static uint8_t *all_train_fp;       /* [TRAIN_N × FP_BYTES] — 1-bit fingerprints */
static uint8_t *all_train_fp_pos;   /* [TRAIN_N × FP2_BYTES/2] — 2-bit positive */
static uint8_t *all_train_fp_neg;   /* [TRAIN_N × FP2_BYTES/2] — 2-bit negative */

static void encode_all(const uint8_t *train_imgs) {
    all_train_trits  = aligned_alloc(32, (size_t)TRAIN_N * TRIT_PADDED);
    all_train_fp     = aligned_alloc(32, (size_t)TRAIN_N * FP_BYTES);
    all_train_fp_pos = aligned_alloc(32, (size_t)TRAIN_N * (FP2_BYTES / 2));
    all_train_fp_neg = aligned_alloc(32, (size_t)TRAIN_N * (FP2_BYTES / 2));

    for (int i = 0; i < TRAIN_N; i++) {
        int8_t *trits = all_train_trits + (size_t)i * TRIT_PADDED;
        encode_image_trits(train_imgs + (size_t)i * PIXELS, trits);
        encode_fingerprint(trits, all_train_fp + (size_t)i * FP_BYTES);
        encode_fingerprint_2bit(trits,
            all_train_fp_pos + (size_t)i * (FP2_BYTES / 2),
            all_train_fp_neg + (size_t)i * (FP2_BYTES / 2));
    }
}

/* ================================================================
 *  Shortlist via Hamming distance (max-heap for top-K smallest)
 * ================================================================ */

typedef struct { int32_t dist; int idx; } candidate_t;

/* Simple insertion into a sorted shortlist (small K, not worth a heap) */
/* Max-heap on distance: sl[0] is worst (largest distance).
 * We keep the K smallest distances. */
static void shortlist_insert(candidate_t *sl, int *count, int max_k,
                              int32_t dist, int idx) {
    if (*count < max_k) {
        int i = *count;
        sl[i] = (candidate_t){dist, idx};
        (*count)++;
        /* Sift up */
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (sl[i].dist > sl[parent].dist) {
                candidate_t tmp = sl[i]; sl[i] = sl[parent]; sl[parent] = tmp;
                i = parent;
            } else break;
        }
    } else if (dist < sl[0].dist) {
        sl[0] = (candidate_t){dist, idx};
        /* Sift down */
        int i = 0;
        while (1) {
            int l = 2*i + 1, r = 2*i + 2, largest = i;
            if (l < max_k && sl[l].dist > sl[largest].dist) largest = l;
            if (r < max_k && sl[r].dist > sl[largest].dist) largest = r;
            if (largest != i) {
                candidate_t tmp = sl[i]; sl[i] = sl[largest]; sl[largest] = tmp;
                i = largest;
            } else break;
        }
    }
}

/* ================================================================
 *  Two-Stage Classifiers
 * ================================================================ */

/* Brute-force 1-NN (baseline, from v3) */
static int classify_bruteforce_1nn(const int8_t *query, const uint8_t *labels) {
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

/* Stage 1: Hamming shortlist → Stage 2: dot product rerank → 1-NN */
static int classify_hamming_1bit(const int8_t *query_trits,
                                  const uint8_t *query_fp,
                                  const uint8_t *labels, int shortlist_k) {
    /* Stage 1: Hamming scan for shortlist */
    candidate_t *sl = malloc(shortlist_k * sizeof(candidate_t));
    int sl_count = 0;

    for (int i = 0; i < TRAIN_N; i++) {
        int32_t dist = hamming_distance(query_fp,
            all_train_fp + (size_t)i * FP_BYTES, FP_BYTES);
        shortlist_insert(sl, &sl_count, shortlist_k, dist, i);
    }

    /* Stage 2: Rerank shortlist with sign_epi8 dot product */
    int32_t best_score = -999999;
    int best_idx = 0;
    for (int i = 0; i < sl_count; i++) {
        int idx = sl[i].idx;
        int32_t score = ternary_dot(query_trits,
            all_train_trits + (size_t)idx * TRIT_PADDED);
        if (score > best_score) {
            best_score = score;
            best_idx = idx;
        }
    }
    free(sl);
    return labels[best_idx];
}

/* Stage 1: 2-bit Hamming shortlist → Stage 2: dot product rerank → 1-NN */
static int classify_hamming_2bit(const int8_t *query_trits,
                                  const uint8_t *query_fp_pos,
                                  const uint8_t *query_fp_neg,
                                  const uint8_t *labels, int shortlist_k) {
    candidate_t *sl = malloc(shortlist_k * sizeof(candidate_t));
    int sl_count = 0;
    int half = FP2_BYTES / 2;

    for (int i = 0; i < TRAIN_N; i++) {
        int32_t dist = hamming_distance_2bit(
            query_fp_pos, query_fp_neg,
            all_train_fp_pos + (size_t)i * half,
            all_train_fp_neg + (size_t)i * half, half);
        shortlist_insert(sl, &sl_count, shortlist_k, dist, i);
    }

    int32_t best_score = -999999;
    int best_idx = 0;
    for (int i = 0; i < sl_count; i++) {
        int idx = sl[i].idx;
        int32_t score = ternary_dot(query_trits,
            all_train_trits + (size_t)idx * TRIT_PADDED);
        if (score > best_score) {
            best_score = score;
            best_idx = idx;
        }
    }
    free(sl);
    return labels[best_idx];
}

/* 3-NN majority vote variant */
static int classify_hamming_2bit_3nn(const int8_t *query_trits,
                                      const uint8_t *query_fp_pos,
                                      const uint8_t *query_fp_neg,
                                      const uint8_t *labels, int shortlist_k) {
    candidate_t *sl = malloc(shortlist_k * sizeof(candidate_t));
    int sl_count = 0;
    int half = FP2_BYTES / 2;

    for (int i = 0; i < TRAIN_N; i++) {
        int32_t dist = hamming_distance_2bit(
            query_fp_pos, query_fp_neg,
            all_train_fp_pos + (size_t)i * half,
            all_train_fp_neg + (size_t)i * half, half);
        shortlist_insert(sl, &sl_count, shortlist_k, dist, i);
    }

    /* Rerank and find top-3 */
    int32_t top3_scores[3] = {-999999, -999999, -999999};
    int top3_idx[3] = {0, 0, 0};
    for (int i = 0; i < sl_count; i++) {
        int idx = sl[i].idx;
        int32_t score = ternary_dot(query_trits,
            all_train_trits + (size_t)idx * TRIT_PADDED);
        int worst = 0;
        if (top3_scores[1] < top3_scores[worst]) worst = 1;
        if (top3_scores[2] < top3_scores[worst]) worst = 2;
        if (score > top3_scores[worst]) {
            top3_scores[worst] = score;
            top3_idx[worst] = idx;
        }
    }

    int votes[N_CLASSES] = {0};
    for (int i = 0; i < 3; i++)
        votes[labels[top3_idx[i]]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (votes[c] > votes[best]) best = c;
    free(sl);
    return best;
}

/* ================================================================
 *  Runner
 * ================================================================ */

typedef struct {
    const char *name;
    double accuracy;
    double time_sec;
    double per_image_ms;
    double throughput;
} result_t;

static result_t run_bruteforce(const uint8_t *test_imgs,
                                const uint8_t *train_labels, const uint8_t *test_labels) {
    int8_t *query = aligned_alloc(32, TRIT_PADDED);
    int correct = 0;
    double t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        encode_image_trits(test_imgs + (size_t)i * PIXELS, query);
        if (classify_bruteforce_1nn(query, train_labels) == test_labels[i])
            correct++;
    }
    double t1 = now_sec();
    free(query);
    return (result_t){
        "Brute-force 1-NN (60K dot products)",
        100.0 * correct / TEST_N, t1 - t0,
        (t1 - t0) * 1000.0 / TEST_N, TEST_N / (t1 - t0)
    };
}

static result_t run_hamming_1bit(const uint8_t *test_imgs,
                                  const uint8_t *train_labels, const uint8_t *test_labels,
                                  int shortlist_k) {
    int8_t *query_trits = aligned_alloc(32, TRIT_PADDED);
    uint8_t *query_fp = aligned_alloc(32, FP_BYTES);
    int correct = 0;
    double t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        encode_image_trits(test_imgs + (size_t)i * PIXELS, query_trits);
        encode_fingerprint(query_trits, query_fp);
        if (classify_hamming_1bit(query_trits, query_fp, train_labels, shortlist_k)
            == test_labels[i])
            correct++;
    }
    double t1 = now_sec();
    free(query_trits);
    free(query_fp);
    char name[128];
    snprintf(name, sizeof(name), "1-bit Hamming → sl=%d → dot rerank", shortlist_k);
    return (result_t){
        strdup(name), 100.0 * correct / TEST_N, t1 - t0,
        (t1 - t0) * 1000.0 / TEST_N, TEST_N / (t1 - t0)
    };
}

static result_t run_hamming_2bit(const uint8_t *test_imgs,
                                  const uint8_t *train_labels, const uint8_t *test_labels,
                                  int shortlist_k) {
    int8_t *query_trits = aligned_alloc(32, TRIT_PADDED);
    uint8_t *query_fp_pos = aligned_alloc(32, FP2_BYTES / 2);
    uint8_t *query_fp_neg = aligned_alloc(32, FP2_BYTES / 2);
    int correct = 0;
    double t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        encode_image_trits(test_imgs + (size_t)i * PIXELS, query_trits);
        encode_fingerprint_2bit(query_trits, query_fp_pos, query_fp_neg);
        if (classify_hamming_2bit(query_trits, query_fp_pos, query_fp_neg,
                                   train_labels, shortlist_k) == test_labels[i])
            correct++;
    }
    double t1 = now_sec();
    free(query_trits);
    free(query_fp_pos);
    free(query_fp_neg);
    char name[128];
    snprintf(name, sizeof(name), "2-bit Hamming → sl=%d → dot rerank", shortlist_k);
    return (result_t){
        strdup(name), 100.0 * correct / TEST_N, t1 - t0,
        (t1 - t0) * 1000.0 / TEST_N, TEST_N / (t1 - t0)
    };
}

static result_t run_hamming_2bit_3nn(const uint8_t *test_imgs,
                                      const uint8_t *train_labels, const uint8_t *test_labels,
                                      int shortlist_k) {
    int8_t *query_trits = aligned_alloc(32, TRIT_PADDED);
    uint8_t *query_fp_pos = aligned_alloc(32, FP2_BYTES / 2);
    uint8_t *query_fp_neg = aligned_alloc(32, FP2_BYTES / 2);
    int correct = 0;
    double t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        encode_image_trits(test_imgs + (size_t)i * PIXELS, query_trits);
        encode_fingerprint_2bit(query_trits, query_fp_pos, query_fp_neg);
        if (classify_hamming_2bit_3nn(query_trits, query_fp_pos, query_fp_neg,
                                       train_labels, shortlist_k) == test_labels[i])
            correct++;
    }
    double t1 = now_sec();
    free(query_trits);
    free(query_fp_pos);
    free(query_fp_neg);
    char name[128];
    snprintf(name, sizeof(name), "2-bit Hamming → sl=%d → 3-NN vote", shortlist_k);
    return (result_t){
        strdup(name), 100.0 * correct / TEST_N, t1 - t0,
        (t1 - t0) * 1000.0 / TEST_N, TEST_N / (t1 - t0)
    };
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

    printf("Shirley Phase 1.5 — Hamming-Accelerated Routing (LCVDB-style)\n");
    printf("===============================================================\n\n");

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

    /* Encode all training data */
    printf("Encoding training data (5-trit + fingerprints)...\n");
    double t0 = now_sec();
    encode_all(train_imgs);
    double t1 = now_sec();
    printf("  Encoded %d images in %.3f sec\n", TRAIN_N, t1 - t0);
    printf("  5-trit vectors: %.1f MB\n",
           (double)TRAIN_N * TRIT_PADDED / (1024.0 * 1024.0));
    printf("  1-bit fingerprints: %.1f MB\n",
           (double)TRAIN_N * FP_BYTES / (1024.0 * 1024.0));
    printf("  2-bit fingerprints: %.1f MB\n",
           (double)TRAIN_N * FP2_BYTES / (1024.0 * 1024.0));

    printf("\nRunning classifiers...\n\n");

    /* Collect results */
    result_t results[16];
    int nr = 0;

    /* Brute force baseline */
    printf("  Running brute-force 1-NN (this takes ~2 min)...\n");
    results[nr++] = run_bruteforce(test_imgs, train_labels, test_labels);

    /* 1-bit Hamming variants */
    printf("  Running 1-bit Hamming sl=50...\n");
    results[nr++] = run_hamming_1bit(test_imgs, train_labels, test_labels, 50);
    printf("  Running 1-bit Hamming sl=100...\n");
    results[nr++] = run_hamming_1bit(test_imgs, train_labels, test_labels, 100);
    printf("  Running 1-bit Hamming sl=200...\n");
    results[nr++] = run_hamming_1bit(test_imgs, train_labels, test_labels, 200);

    /* 2-bit Hamming variants */
    printf("  Running 2-bit Hamming sl=50...\n");
    results[nr++] = run_hamming_2bit(test_imgs, train_labels, test_labels, 50);
    printf("  Running 2-bit Hamming sl=100...\n");
    results[nr++] = run_hamming_2bit(test_imgs, train_labels, test_labels, 100);
    printf("  Running 2-bit Hamming sl=200...\n");
    results[nr++] = run_hamming_2bit(test_imgs, train_labels, test_labels, 200);

    /* Best 2-bit with 3-NN */
    printf("  Running 2-bit Hamming sl=200 → 3-NN...\n");
    results[nr++] = run_hamming_2bit_3nn(test_imgs, train_labels, test_labels, 200);

    /* Print results table */
    printf("\n=== RESULTS ===\n\n");
    printf("  %-45s  %6s  %10s  %9s  %8s\n",
           "Method", "Acc", "Total", "Per-image", "Speedup");
    printf("  ---------------------------------------------"
           "  ------  ----------  ---------  --------\n");

    double baseline_time = results[0].time_sec;
    for (int i = 0; i < nr; i++) {
        double speedup = baseline_time / results[i].time_sec;
        printf("  %-45s  %5.2f%%  %8.1f s  %7.2f ms  %6.1fx\n",
               results[i].name, results[i].accuracy,
               results[i].time_sec, results[i].per_image_ms, speedup);
    }

    printf("\n=== ARCHITECTURE ===\n\n");
    printf("  Two-stage routing (LCVDB-inspired):\n");
    printf("    Stage 1: XOR + POPCNT Hamming scan (coarse routing)\n");
    printf("    Stage 2: sign_epi8 dot product rerank (fine routing)\n\n");
    printf("  All operations are integer. No floating point.\n");
    printf("  Hamming scan: XOR (ternary in disguise) + POPCNT\n");
    printf("  Dot product: sign_epi8 (ternary MUL) + add_epi8 (ADD)\n");
    printf("  Classification: argmax (MAX prime)\n");

    free(all_train_trits);
    free(all_train_fp);
    free(all_train_fp_pos);
    free(all_train_fp_neg);
    free(train_imgs);
    free(train_labels);
    free(test_imgs);
    free(test_labels);

    return 0;
}
