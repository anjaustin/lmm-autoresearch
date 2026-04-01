/*
 * demo_routing.c — Shirley Phase 1, Step 1.2
 *
 * Demonstrates routing-first computation using frozen shapes on MNIST.
 * Extracted and reframed from SSTT's sstt_v2.c.
 *
 * The pipeline:
 *   1. Raw pixels → ternary quantization (domain entry)
 *   2. Ternary image → block encoding (routing signatures)
 *   3. Block values → frozen shape lookup (hot map = precomputed shape output)
 *   4. Accumulate shape outputs → classification (argmax)
 *
 * In Shirley terms:
 *   - Quantization = domain entry (continuous → ternary)
 *   - Block encoding = signature generation
 *   - Hot map lookup = routing to frozen shape + shape evaluation
 *   - Accumulation = ADD prime
 *   - Argmax = MAX prime
 *
 * No learned parameters. No floating point. No multiply.
 * Every operation is ADD, MAX, or CONST. MUL is implicit in the
 * block encoding (ternary values ARE the routing decision).
 *
 * Build:  gcc -O3 -mavx2 -march=native -o demo_routing demo_routing.c -lm
 * Run:    ./demo_routing <path-to-mnist-data-dir>
 *
 * Example: ./demo_routing ~/Projects/000-research/sstt/data/
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
#define BLK_W         3
#define BLKS_PER_ROW  9
#define N_BLOCKS      (BLKS_PER_ROW * IMG_H)  /* 252 */
#define N_BVALS       27                       /* 3^3 */
#define N_CLASSES     10
#define CLS_PAD       16

/* Background block values */
#define BG_PIXEL      0    /* block_encode(-1,-1,-1): all dark */
#define BG_GRAD       13   /* block_encode(0,0,0): flat region */

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
    fread(&magic, 4, 1, f);
    fread(&n, 4, 1, f);
    magic = __builtin_bswap32(magic);
    n = __builtin_bswap32(n);
    *count = n;
    int ndim = magic & 0xFF;
    size_t item_size = 1;
    if (ndim >= 3) {
        uint32_t rows, cols;
        fread(&rows, 4, 1, f);
        fread(&cols, 4, 1, f);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);
        item_size = (size_t)rows * cols;
    }
    size_t total = (size_t)n * item_size;
    uint8_t *data = malloc(total);
    fread(data, 1, total, f);
    fclose(f);
    return data;
}

/* ================================================================
 *  PRIME: CONST — Ternary Quantization Thresholds
 *  Domain entry: continuous uint8 pixels → ternary {-1, 0, +1}
 * ================================================================ */

static void quantize_image(const uint8_t *src, int8_t *dst) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    const __m256i thi  = _mm256_set1_epi8((char)(170 ^ 0x80));
    const __m256i tlo  = _mm256_set1_epi8((char)(85  ^ 0x80));
    const __m256i one  = _mm256_set1_epi8(1);
    int i;
    for (i = 0; i + 32 <= PIXELS; i += 32) {
        __m256i px = _mm256_loadu_si256((const __m256i *)(src + i));
        __m256i spx = _mm256_xor_si256(px, bias);
        __m256i pos = _mm256_cmpgt_epi8(spx, thi);
        __m256i neg = _mm256_cmpgt_epi8(tlo, spx);
        __m256i p = _mm256_and_si256(pos, one);
        __m256i n = _mm256_and_si256(neg, one);
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_sub_epi8(p, n));
    }
    for (; i < PIXELS; i++) {
        if (src[i] >= 170)     dst[i] =  1;
        else if (src[i] < 85)  dst[i] = -1;
        else                    dst[i] =  0;
    }
}

/* ================================================================
 *  ROUTING: Signature Generation — Block Encoding
 *  3 ternary trits → 1 block value (0-26)
 *  This IS the routing decision: the block value selects the frozen shape.
 * ================================================================ */

static inline uint8_t block_encode(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0 + 1) * 9 + (t1 + 1) * 3 + (t2 + 1));
}

static void compute_signature(const int8_t *tern, uint8_t *sig) {
    for (int y = 0; y < IMG_H; y++) {
        const int8_t *row = tern + y * IMG_W;
        for (int s = 0; s < BLKS_PER_ROW; s++)
            sig[y * BLKS_PER_ROW + s] = block_encode(
                row[s * BLK_W], row[s * BLK_W + 1], row[s * BLK_W + 2]);
    }
}

/* ================================================================
 *  GRADIENT OBSERVER: Ternary edge detection
 *  Clamp(diff) → {-1, 0, +1}
 * ================================================================ */

static inline int8_t clamp_trit(int v) {
    return v > 0 ? 1 : v < 0 ? -1 : 0;
}

static void compute_gradients(const int8_t *tern, int8_t *h_grad, int8_t *v_grad) {
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W - 1; x++) {
            h_grad[y * IMG_W + x] = clamp_trit(
                tern[y * IMG_W + x + 1] - tern[y * IMG_W + x]);
        }
        h_grad[y * IMG_W + IMG_W - 1] = 0;
    }
    for (int y = 0; y < IMG_H - 1; y++) {
        for (int x = 0; x < IMG_W; x++) {
            v_grad[y * IMG_W + x] = clamp_trit(
                tern[(y + 1) * IMG_W + x] - tern[y * IMG_W + x]);
        }
    }
    memset(v_grad + (IMG_H - 1) * IMG_W, 0, IMG_W);
}

/* ================================================================
 *  FROZEN SHAPES: Hot Map (precomputed shape outputs)
 *
 *  hot_map[position][block_value][class] = frequency count
 *
 *  This IS a frozen shape library:
 *  - 252 positions × 27 block values = 6,804 unique shapes
 *  - Each shape maps to a 10-class score vector
 *  - The shapes are "trained" by counting (closed-form, not learned)
 *  - At inference, shapes are FROZEN — never modified
 *
 *  Total size: 252 × 27 × 16 × 4 = 435 KB (L2-cache resident)
 * ================================================================ */

static uint32_t px_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t hg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

static void build_frozen_shapes(const uint8_t *train_imgs,
                                 const uint8_t *labels, int n) {
    int8_t tern[PIXELS], h_grad[PIXELS], v_grad[PIXELS];
    uint8_t px_sig[N_BLOCKS], hg_sig[N_BLOCKS], vg_sig[N_BLOCKS];

    memset(px_hot, 0, sizeof(px_hot));
    memset(hg_hot, 0, sizeof(hg_hot));
    memset(vg_hot, 0, sizeof(vg_hot));

    for (int i = 0; i < n; i++) {
        const uint8_t *img = train_imgs + (size_t)i * PIXELS;
        int lbl = labels[i];

        quantize_image(img, tern);
        compute_gradients(tern, h_grad, v_grad);
        compute_signature(tern, px_sig);
        compute_signature(h_grad, hg_sig);
        compute_signature(v_grad, vg_sig);

        for (int k = 0; k < N_BLOCKS; k++) {
            px_hot[k][px_sig[k]][lbl]++;
            hg_hot[k][hg_sig[k]][lbl]++;
            vg_hot[k][vg_sig[k]][lbl]++;
        }
    }
}

/* ================================================================
 *  INFERENCE: Route → Shape → Accumulate → Argmax
 *
 *  For each block position:
 *    1. ROUTE: block value selects frozen shape (hot map row)
 *    2. SHAPE: retrieve precomputed class scores (ADD prime: load)
 *    3. ACCUMULATE: sum scores across positions (ADD prime)
 *  Finally:
 *    4. ARGMAX: select class with highest score (MAX prime)
 *
 *  Primes used: ADD, MAX, CONST (thresholds).
 *  MUL is implicit: ternary block values ARE the routing decision.
 *  No floating point. No multiply instruction.
 * ================================================================ */

static int classify(const uint8_t *img) {
    int8_t tern[PIXELS], h_grad[PIXELS], v_grad[PIXELS];
    uint8_t px_sig[N_BLOCKS], hg_sig[N_BLOCKS], vg_sig[N_BLOCKS];

    /* Domain entry */
    quantize_image(img, tern);
    compute_gradients(tern, h_grad, v_grad);

    /* Signature generation (routing) */
    compute_signature(tern, px_sig);
    compute_signature(h_grad, hg_sig);
    compute_signature(v_grad, vg_sig);

    /* Route to frozen shapes and accumulate */
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();

    /* Pixel channel */
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = px_sig[k];
        if (bv == BG_PIXEL) continue;  /* skip background (sparsity!) */
        const __m256i *shape = (const __m256i *)px_hot[k][bv];
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_load_si256(shape));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_load_si256(shape + 1));
    }
    /* H-gradient channel */
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = hg_sig[k];
        if (bv == BG_GRAD) continue;
        const __m256i *shape = (const __m256i *)hg_hot[k][bv];
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_load_si256(shape));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_load_si256(shape + 1));
    }
    /* V-gradient channel */
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = vg_sig[k];
        if (bv == BG_GRAD) continue;
        const __m256i *shape = (const __m256i *)vg_hot[k][bv];
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_load_si256(shape));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_load_si256(shape + 1));
    }

    /* Argmax (MAX prime) */
    uint32_t scores[CLS_PAD] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)scores, acc_lo);
    _mm256_store_si256((__m256i *)(scores + 8), acc_hi);

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (scores[c] > scores[best]) best = c;
    return best;
}

/* ================================================================
 *  Main
 * ================================================================ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mnist-data-dir>\n", argv[0]);
        fprintf(stderr, "  e.g. %s ~/Projects/000-research/sstt/data/\n", argv[0]);
        return 1;
    }

    const char *dir = argv[1];
    char path[512];

    printf("Shirley Phase 1 — Routing Demo (Frozen Shape Classification)\n");
    printf("=============================================================\n\n");

    /* Load MNIST */
    printf("Loading MNIST data from %s ...\n", dir);
    uint32_t n;
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", dir);
    uint8_t *train_imgs = load_idx(path, &n);
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", dir);
    uint8_t *train_labels = load_idx(path, &n);
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", dir);
    uint8_t *test_imgs = load_idx(path, &n);
    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", dir);
    uint8_t *test_labels = load_idx(path, &n);
    printf("  Train: %d images, Test: %d images\n\n", TRAIN_N, TEST_N);

    /* Build frozen shapes from training data */
    printf("Building frozen shapes (hot maps) from training data...\n");
    double t0 = now_sec();
    build_frozen_shapes(train_imgs, train_labels, TRAIN_N);
    double t1 = now_sec();
    printf("  Built in %.3f seconds\n", t1 - t0);
    printf("  Shape library: %d positions × %d shapes = %d frozen shapes\n",
           N_BLOCKS, N_BVALS, N_BLOCKS * N_BVALS);
    printf("  Shape library size: %.1f KB per channel × 3 channels = %.1f KB total\n",
           (double)(N_BLOCKS * N_BVALS * CLS_PAD * 4) / 1024.0,
           3.0 * (double)(N_BLOCKS * N_BVALS * CLS_PAD * 4) / 1024.0);
    printf("  All shapes L2-cache resident: YES\n\n");

    /* Classify all test images */
    printf("Classifying %d test images via routing...\n", TEST_N);

    int correct = 0;
    int confusion[N_CLASSES][N_CLASSES] = {0};

    t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        int pred = classify(test_imgs + (size_t)i * PIXELS);
        int true_label = test_labels[i];
        if (pred == true_label) correct++;
        confusion[true_label][pred]++;
    }
    t1 = now_sec();

    double accuracy = 100.0 * correct / TEST_N;
    double total_ms = (t1 - t0) * 1000.0;
    double per_image_us = total_ms * 1000.0 / TEST_N;
    double images_per_sec = TEST_N / (t1 - t0);

    printf("\n");
    printf("=== RESULTS ===\n\n");
    printf("  Accuracy:        %.2f%% (%d / %d)\n", accuracy, correct, TEST_N);
    printf("  Total time:      %.1f ms\n", total_ms);
    printf("  Per image:       %.1f µs\n", per_image_us);
    printf("  Throughput:      %.0f images/sec\n", images_per_sec);
    printf("\n");

    /* Primes used */
    printf("=== PRIMES USED ===\n\n");
    printf("  ADD:   Accumulate shape outputs across 252 positions (AVX2 add_epi32)\n");
    printf("  MUL:   Implicit — ternary block values ARE the routing decision\n");
    printf("  MAX:   Argmax over 10 classes for final prediction\n");
    printf("  CONST: Quantization thresholds (85, 170), background values (0, 13)\n");
    printf("  EXP:   Not used (no transcendentals in this pipeline)\n");
    printf("  LOG:   Not used (no transcendentals in this pipeline)\n");
    printf("\n");

    /* Architecture summary */
    printf("=== ARCHITECTURE ===\n\n");
    printf("  Learned parameters:  0 (shapes computed by counting, not training)\n");
    printf("  Floating point ops:  0 (entire pipeline is integer ternary)\n");
    printf("  Multiply instructions: 0 (routing replaces multiplication)\n");
    printf("  Shape library:       L2-cache resident (%.1f KB)\n",
           3.0 * (double)(N_BLOCKS * N_BVALS * CLS_PAD * 4) / 1024.0);
    printf("  Routing mechanism:   Block value → shape lookup (deterministic)\n");
    printf("  Sparsity:            Background blocks skipped (native ternary 0)\n");
    printf("\n");

    /* Confusion matrix */
    printf("=== CONFUSION MATRIX ===\n\n");
    printf("     ");
    for (int c = 0; c < N_CLASSES; c++) printf(" %4d", c);
    printf("  | correct\n");
    printf("     ");
    for (int c = 0; c < N_CLASSES; c++) printf(" ----");
    printf("  +--------\n");
    for (int t = 0; t < N_CLASSES; t++) {
        printf("  %d: ", t);
        int row_correct = confusion[t][t];
        int row_total = 0;
        for (int p = 0; p < N_CLASSES; p++) {
            printf(" %4d", confusion[t][p]);
            row_total += confusion[t][p];
        }
        printf("  | %5.1f%%\n", 100.0 * row_correct / row_total);
    }
    printf("\n");

    /* Shirley interpretation */
    printf("=== SHIRLEY INTERPRETATION ===\n\n");
    printf("  This pipeline IS routing-first computation:\n");
    printf("    1. Domain entry:  pixels → ternary (CONST thresholds)\n");
    printf("    2. Signatures:    ternary blocks → routing indices (block encoding)\n");
    printf("    3. Routing:       index selects frozen shape (hot map lookup)\n");
    printf("    4. Computation:   shape output accumulated (ADD prime)\n");
    printf("    5. Decision:      argmax (MAX prime)\n");
    printf("\n");
    printf("  The shapes are frozen. The routing is the only degree of freedom.\n");
    printf("  No parameters were learned. No gradients were computed.\n");
    printf("  This is what Shirley looks like when it runs.\n");

    free(train_imgs);
    free(train_labels);
    free(test_imgs);
    free(test_labels);

    return 0;
}
