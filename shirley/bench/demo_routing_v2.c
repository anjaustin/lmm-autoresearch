/*
 * demo_routing_v2.c — Shirley Phase 1, Step 1.2b
 *
 * Enhanced routing demo: adds IG weighting and multi-probe Hamming-1
 * expansion to the baseline routing pipeline. These are pure routing
 * refinements — the frozen shapes don't change.
 *
 * Runs three configurations and compares:
 *   v1: bare routing (block value → shape lookup)
 *   v2: + IG weighting (position-dependent routing priority)
 *   v3: + IG weighting + multi-probe (soft routing via Hamming-1 neighbors)
 *
 * Build:  gcc -O3 -mavx2 -march=native -o demo_routing_v2 demo_routing_v2.c -lm
 * Run:    ./demo_routing_v2 <path-to-mnist-data-dir>
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
#define IG_SCALE      16

#define BG_PIXEL      0
#define BG_GRAD       13

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
 *  Ternary Quantization
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
 *  Block Encoding & Gradients
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

static inline int8_t clamp_trit(int v) {
    return v > 0 ? 1 : v < 0 ? -1 : 0;
}

static void compute_gradients(const int8_t *tern, int8_t *h_grad, int8_t *v_grad) {
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W - 1; x++)
            h_grad[y * IMG_W + x] = clamp_trit(
                tern[y * IMG_W + x + 1] - tern[y * IMG_W + x]);
        h_grad[y * IMG_W + IMG_W - 1] = 0;
    }
    for (int y = 0; y < IMG_H - 1; y++)
        for (int x = 0; x < IMG_W; x++)
            v_grad[y * IMG_W + x] = clamp_trit(
                tern[(y + 1) * IMG_W + x] - tern[y * IMG_W + x]);
    memset(v_grad + (IMG_H - 1) * IMG_W, 0, IMG_W);
}

/* ================================================================
 *  Frozen Shapes (Hot Maps)
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
 *  ROUTING REFINEMENT 1: Information Gain Weights
 *
 *  IG(position k) = H(class) - H(class | block_value_at_k)
 *
 *  This is a CONST prime: frozen weights derived from training
 *  statistics. Not learned — computed in closed form from Shannon
 *  entropy. Positions with high IG (digit center) get weight ~16.
 *  Positions with low IG (corners) get weight ~1.
 * ================================================================ */

static uint16_t ig_w_px[N_BLOCKS];
static uint16_t ig_w_hg[N_BLOCKS];
static uint16_t ig_w_vg[N_BLOCKS];

static void compute_ig_weights(uint32_t hot[][N_BVALS][CLS_PAD],
                                uint16_t *weights,
                                const uint8_t *labels, int n) {
    /* Class prior entropy H(class) */
    int class_counts[N_CLASSES] = {0};
    for (int i = 0; i < n; i++)
        class_counts[labels[i]]++;

    double h_class = 0.0;
    for (int c = 0; c < N_CLASSES; c++) {
        double p = (double)class_counts[c] / n;
        if (p > 0) h_class -= p * log2(p);
    }

    double raw_ig[N_BLOCKS];
    double max_ig = 0.0;

    for (int k = 0; k < N_BLOCKS; k++) {
        /* Total observations at each block value */
        int val_total[N_BVALS] = {0};
        for (int v = 0; v < N_BVALS; v++)
            for (int c = 0; c < N_CLASSES; c++)
                val_total[v] += hot[k][v][c];

        /* Conditional entropy H(class | block_k) */
        double h_cond = 0.0;
        for (int v = 0; v < N_BVALS; v++) {
            if (val_total[v] == 0) continue;
            double pv = (double)val_total[v] / n;
            double hv = 0.0;
            for (int c = 0; c < N_CLASSES; c++) {
                double pc = (double)hot[k][v][c] / val_total[v];
                if (pc > 0) hv -= pc * log2(pc);
            }
            h_cond += pv * hv;
        }

        raw_ig[k] = h_class - h_cond;
        if (raw_ig[k] > max_ig) max_ig = raw_ig[k];
    }

    /* Normalize to [1, IG_SCALE] */
    for (int k = 0; k < N_BLOCKS; k++) {
        if (max_ig > 0)
            weights[k] = (uint16_t)(raw_ig[k] / max_ig * IG_SCALE + 0.5);
        else
            weights[k] = 1;
        if (weights[k] == 0) weights[k] = 1;
    }
}

/* ================================================================
 *  ROUTING REFINEMENT 2: Multi-Probe Hamming-1 Neighbors
 *
 *  For each block value, enumerate all values that differ by exactly
 *  one trit. 3 positions × 2 alternatives = 6 neighbors always.
 *
 *  Neighbors vote at half weight — soft routing that relaxes hard
 *  boundaries between shapes.
 * ================================================================ */

static uint8_t nbr_table[N_BVALS][6];

static void build_neighbor_table(void) {
    int trit_vals[3] = {-1, 0, 1};
    for (int v = 0; v < N_BVALS; v++) {
        int t0 = (v / 9) - 1;
        int t1 = ((v / 3) % 3) - 1;
        int t2 = (v % 3) - 1;
        int orig[3] = {t0, t1, t2};
        int nc = 0;
        for (int pos = 0; pos < 3; pos++) {
            for (int alt = 0; alt < 3; alt++) {
                if (trit_vals[alt] == orig[pos]) continue;
                int mod[3] = {orig[0], orig[1], orig[2]};
                mod[pos] = trit_vals[alt];
                nbr_table[v][nc++] = block_encode(
                    (int8_t)mod[0], (int8_t)mod[1], (int8_t)mod[2]);
            }
        }
    }
}

/* ================================================================
 *  Classification — Three Modes
 * ================================================================ */

/* v1: Bare routing (baseline) */
static int classify_v1(const uint8_t *img) {
    int8_t tern[PIXELS], h_grad[PIXELS], v_grad[PIXELS];
    uint8_t px_sig[N_BLOCKS], hg_sig[N_BLOCKS], vg_sig[N_BLOCKS];

    quantize_image(img, tern);
    compute_gradients(tern, h_grad, v_grad);
    compute_signature(tern, px_sig);
    compute_signature(h_grad, hg_sig);
    compute_signature(v_grad, vg_sig);

    uint32_t scores[CLS_PAD] __attribute__((aligned(32))) = {0};

    for (int k = 0; k < N_BLOCKS; k++) {
        if (px_sig[k] != BG_PIXEL)
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += px_hot[k][px_sig[k]][c];
        if (hg_sig[k] != BG_GRAD)
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += hg_hot[k][hg_sig[k]][c];
        if (vg_sig[k] != BG_GRAD)
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += vg_hot[k][vg_sig[k]][c];
    }

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (scores[c] > scores[best]) best = c;
    return best;
}

/* v2: + IG weighting (routing priority) */
static int classify_v2(const uint8_t *img) {
    int8_t tern[PIXELS], h_grad[PIXELS], v_grad[PIXELS];
    uint8_t px_sig[N_BLOCKS], hg_sig[N_BLOCKS], vg_sig[N_BLOCKS];

    quantize_image(img, tern);
    compute_gradients(tern, h_grad, v_grad);
    compute_signature(tern, px_sig);
    compute_signature(h_grad, hg_sig);
    compute_signature(v_grad, vg_sig);

    uint64_t scores[CLS_PAD] __attribute__((aligned(32))) = {0};

    for (int k = 0; k < N_BLOCKS; k++) {
        if (px_sig[k] != BG_PIXEL) {
            uint16_t w = ig_w_px[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)px_hot[k][px_sig[k]][c] * w;
        }
        if (hg_sig[k] != BG_GRAD) {
            uint16_t w = ig_w_hg[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)hg_hot[k][hg_sig[k]][c] * w;
        }
        if (vg_sig[k] != BG_GRAD) {
            uint16_t w = ig_w_vg[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)vg_hot[k][vg_sig[k]][c] * w;
        }
    }

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (scores[c] > scores[best]) best = c;
    return best;
}

/* v3: + IG weighting + multi-probe Hamming-1 (soft routing) */
static int classify_v3(const uint8_t *img) {
    int8_t tern[PIXELS], h_grad[PIXELS], v_grad[PIXELS];
    uint8_t px_sig[N_BLOCKS], hg_sig[N_BLOCKS], vg_sig[N_BLOCKS];

    quantize_image(img, tern);
    compute_gradients(tern, h_grad, v_grad);
    compute_signature(tern, px_sig);
    compute_signature(h_grad, hg_sig);
    compute_signature(v_grad, vg_sig);

    uint64_t scores[CLS_PAD] __attribute__((aligned(32))) = {0};

    for (int k = 0; k < N_BLOCKS; k++) {
        /* Pixel channel — exact match at full weight, neighbors at half */
        if (px_sig[k] != BG_PIXEL) {
            uint16_t w = ig_w_px[k];
            uint8_t bv = px_sig[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)px_hot[k][bv][c] * w;
            /* Hamming-1 neighbors at half weight */
            uint16_t hw = w / 2;
            if (hw > 0) {
                for (int ni = 0; ni < 6; ni++) {
                    uint8_t nv = nbr_table[bv][ni];
                    for (int c = 0; c < N_CLASSES; c++)
                        scores[c] += (uint64_t)px_hot[k][nv][c] * hw;
                }
            }
        }

        /* H-gradient channel */
        if (hg_sig[k] != BG_GRAD) {
            uint16_t w = ig_w_hg[k];
            uint8_t bv = hg_sig[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)hg_hot[k][bv][c] * w;
            uint16_t hw = w / 2;
            if (hw > 0) {
                for (int ni = 0; ni < 6; ni++) {
                    uint8_t nv = nbr_table[bv][ni];
                    for (int c = 0; c < N_CLASSES; c++)
                        scores[c] += (uint64_t)hg_hot[k][nv][c] * hw;
                }
            }
        }

        /* V-gradient channel */
        if (vg_sig[k] != BG_GRAD) {
            uint16_t w = ig_w_vg[k];
            uint8_t bv = vg_sig[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)vg_hot[k][bv][c] * w;
            uint16_t hw = w / 2;
            if (hw > 0) {
                for (int ni = 0; ni < 6; ni++) {
                    uint8_t nv = nbr_table[bv][ni];
                    for (int c = 0; c < N_CLASSES; c++)
                        scores[c] += (uint64_t)vg_hot[k][nv][c] * hw;
                }
            }
        }
    }

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (scores[c] > scores[best]) best = c;
    return best;
}

/* v4: + IG weighting + IG-gated multi-probe (probe only high-IG positions) */
#define IG_PROBE_THRESHOLD 4  /* only probe neighbors where IG >= this */

static int classify_v4(const uint8_t *img) {
    int8_t tern[PIXELS], h_grad[PIXELS], v_grad[PIXELS];
    uint8_t px_sig[N_BLOCKS], hg_sig[N_BLOCKS], vg_sig[N_BLOCKS];

    quantize_image(img, tern);
    compute_gradients(tern, h_grad, v_grad);
    compute_signature(tern, px_sig);
    compute_signature(h_grad, hg_sig);
    compute_signature(v_grad, vg_sig);

    uint64_t scores[CLS_PAD] __attribute__((aligned(32))) = {0};

    for (int k = 0; k < N_BLOCKS; k++) {
        /* Pixel channel */
        if (px_sig[k] != BG_PIXEL) {
            uint16_t w = ig_w_px[k];
            uint8_t bv = px_sig[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)px_hot[k][bv][c] * w;
            /* Only probe neighbors at high-IG positions */
            if (w >= IG_PROBE_THRESHOLD) {
                uint16_t hw = w / 2;
                for (int ni = 0; ni < 6; ni++) {
                    uint8_t nv = nbr_table[bv][ni];
                    for (int c = 0; c < N_CLASSES; c++)
                        scores[c] += (uint64_t)px_hot[k][nv][c] * hw;
                }
            }
        }

        /* H-gradient channel */
        if (hg_sig[k] != BG_GRAD) {
            uint16_t w = ig_w_hg[k];
            uint8_t bv = hg_sig[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)hg_hot[k][bv][c] * w;
            if (w >= IG_PROBE_THRESHOLD) {
                uint16_t hw = w / 2;
                for (int ni = 0; ni < 6; ni++) {
                    uint8_t nv = nbr_table[bv][ni];
                    for (int c = 0; c < N_CLASSES; c++)
                        scores[c] += (uint64_t)hg_hot[k][nv][c] * hw;
                }
            }
        }

        /* V-gradient channel */
        if (vg_sig[k] != BG_GRAD) {
            uint16_t w = ig_w_vg[k];
            uint8_t bv = vg_sig[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)vg_hot[k][bv][c] * w;
            if (w >= IG_PROBE_THRESHOLD) {
                uint16_t hw = w / 2;
                for (int ni = 0; ni < 6; ni++) {
                    uint8_t nv = nbr_table[bv][ni];
                    for (int c = 0; c < N_CLASSES; c++)
                        scores[c] += (uint64_t)vg_hot[k][nv][c] * hw;
                }
            }
        }
    }

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (scores[c] > scores[best]) best = c;
    return best;
}

/* v5: IG weighting + IG-gated multi-probe + neighbor weight = IG/4 (softer) */
static int classify_v5(const uint8_t *img) {
    int8_t tern[PIXELS], h_grad[PIXELS], v_grad[PIXELS];
    uint8_t px_sig[N_BLOCKS], hg_sig[N_BLOCKS], vg_sig[N_BLOCKS];

    quantize_image(img, tern);
    compute_gradients(tern, h_grad, v_grad);
    compute_signature(tern, px_sig);
    compute_signature(h_grad, hg_sig);
    compute_signature(v_grad, vg_sig);

    uint64_t scores[CLS_PAD] __attribute__((aligned(32))) = {0};

    for (int k = 0; k < N_BLOCKS; k++) {
        if (px_sig[k] != BG_PIXEL) {
            uint16_t w = ig_w_px[k];
            uint8_t bv = px_sig[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)px_hot[k][bv][c] * w;
            if (w >= IG_PROBE_THRESHOLD) {
                uint16_t hw = w / 4;  /* quarter weight */
                if (hw > 0)
                    for (int ni = 0; ni < 6; ni++) {
                        uint8_t nv = nbr_table[bv][ni];
                        for (int c = 0; c < N_CLASSES; c++)
                            scores[c] += (uint64_t)px_hot[k][nv][c] * hw;
                    }
            }
        }
        if (hg_sig[k] != BG_GRAD) {
            uint16_t w = ig_w_hg[k];
            uint8_t bv = hg_sig[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)hg_hot[k][bv][c] * w;
            if (w >= IG_PROBE_THRESHOLD) {
                uint16_t hw = w / 4;
                if (hw > 0)
                    for (int ni = 0; ni < 6; ni++) {
                        uint8_t nv = nbr_table[bv][ni];
                        for (int c = 0; c < N_CLASSES; c++)
                            scores[c] += (uint64_t)hg_hot[k][nv][c] * hw;
                    }
            }
        }
        if (vg_sig[k] != BG_GRAD) {
            uint16_t w = ig_w_vg[k];
            uint8_t bv = vg_sig[k];
            for (int c = 0; c < N_CLASSES; c++)
                scores[c] += (uint64_t)vg_hot[k][bv][c] * w;
            if (w >= IG_PROBE_THRESHOLD) {
                uint16_t hw = w / 4;
                if (hw > 0)
                    for (int ni = 0; ni < 6; ni++) {
                        uint8_t nv = nbr_table[bv][ni];
                        for (int c = 0; c < N_CLASSES; c++)
                            scores[c] += (uint64_t)vg_hot[k][nv][c] * hw;
                    }
            }
        }
    }

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (scores[c] > scores[best]) best = c;
    return best;
}

/* ================================================================
 *  Evaluate a classifier
 * ================================================================ */

typedef int (*classify_fn)(const uint8_t *);

static void evaluate(const char *name, classify_fn fn,
                     const uint8_t *test_imgs, const uint8_t *test_labels) {
    int correct = 0;
    int per_class_correct[N_CLASSES] = {0};
    int per_class_total[N_CLASSES] = {0};

    double t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        int pred = fn(test_imgs + (size_t)i * PIXELS);
        int truth = test_labels[i];
        per_class_total[truth]++;
        if (pred == truth) {
            correct++;
            per_class_correct[truth]++;
        }
    }
    double t1 = now_sec();

    double accuracy = 100.0 * correct / TEST_N;
    double total_ms = (t1 - t0) * 1000.0;
    double per_image_us = total_ms * 1000.0 / TEST_N;
    double images_per_sec = TEST_N / (t1 - t0);

    printf("  %-42s  %6.2f%%  %8.1f µs  %9.0f img/s\n",
           name, accuracy, per_image_us, images_per_sec);
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

    printf("Shirley Phase 1 — Routing Refinement Study\n");
    printf("============================================\n\n");

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

    /* Build frozen shapes */
    printf("Building frozen shapes...\n");
    build_frozen_shapes(train_imgs, train_labels, TRAIN_N);

    /* Build routing refinements */
    printf("Computing IG weights (routing priority)...\n");
    compute_ig_weights(px_hot, ig_w_px, train_labels, TRAIN_N);
    compute_ig_weights(hg_hot, ig_w_hg, train_labels, TRAIN_N);
    compute_ig_weights(vg_hot, ig_w_vg, train_labels, TRAIN_N);

    printf("Building neighbor table (soft routing)...\n");
    build_neighbor_table();

    printf("\nClassifying %d test images...\n\n", TEST_N);

    /* Header */
    printf("  %-42s  %6s  %8s  %9s\n",
           "Configuration", "Acc", "Latency", "Throughput");
    printf("  ------------------------------------------"
           "  ------  --------  ---------\n");

    /* Run all three configurations */
    evaluate("v1: bare routing",
             classify_v1, test_imgs, test_labels);
    evaluate("v2: + IG weighting (routing priority)",
             classify_v2, test_imgs, test_labels);
    evaluate("v3: + IG + multi-probe everywhere (soft routing)",
             classify_v3, test_imgs, test_labels);
    evaluate("v4: + IG + IG-gated multi-probe (selective soft)",
             classify_v4, test_imgs, test_labels);
    evaluate("v5: + IG + IG-gated probe at 1/4 weight (gentle)",
             classify_v5, test_imgs, test_labels);

    printf("\n");
    printf("=== ANALYSIS ===\n\n");
    printf("  The frozen shapes are IDENTICAL across all five configurations.\n");
    printf("  The six primes are IDENTICAL across all five configurations.\n");
    printf("  Only ROUTING changed:\n");
    printf("    v1 → v2: Added position-dependent weights (CONST prime)\n");
    printf("    v2 → v3: Added Hamming-1 neighbors everywhere at 1/2 weight\n");
    printf("    v2 → v4: Added Hamming-1 neighbors only at high-IG positions (1/2 wt)\n");
    printf("    v2 → v5: Added Hamming-1 neighbors only at high-IG positions (1/4 wt)\n\n");
    printf("  Every accuracy change comes from routing refinement alone.\n");
    printf("  This is the experiment surface for the LEMM loop.\n");

    free(train_imgs);
    free(train_labels);
    free(test_imgs);
    free(test_labels);

    return 0;
}
