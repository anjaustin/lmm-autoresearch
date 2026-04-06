/*
 * shirley_threading.h — Lightweight thread coordination for Shirley ops
 *
 * Pattern: thread 0 does sequential prep, sets a flag, all threads
 * execute their partition of the parallel work, thread 0 waits for
 * completion, then continues to next phase.
 */

#ifndef SHIRLEY_THREADING_H
#define SHIRLEY_THREADING_H

#include <stdint.h>
#include <stdatomic.h>

/* Per-op thread coordination state */
typedef struct {
    atomic_int phase;           /* current phase (incremented by thread 0) */
    atomic_int threads_done;    /* count of threads that finished current phase */
    int nth;                    /* total threads */
} shirley_barrier_t;

static inline void shirley_barrier_init(shirley_barrier_t * b, int nth) {
    atomic_store(&b->phase, 0);
    atomic_store(&b->threads_done, 0);
    b->nth = nth;
}

/* Thread 0 calls this to signal that prep is done and parallel work can begin */
static inline void shirley_barrier_signal(shirley_barrier_t * b) {
    atomic_store(&b->threads_done, 0);
    atomic_fetch_add(&b->phase, 1);
}

/* Non-zero threads call this to wait for thread 0's signal */
static inline void shirley_barrier_wait(shirley_barrier_t * b, int expected_phase) {
    while (atomic_load(&b->phase) < expected_phase) {
        /* spin — tiny wait, threads are on the same core complex */
    }
}

/* All threads call this when done with their parallel work */
static inline void shirley_barrier_arrive(shirley_barrier_t * b) {
    atomic_fetch_add(&b->threads_done, 1);
}

/* Thread 0 calls this to wait for all threads to finish parallel work */
static inline void shirley_barrier_wait_all(shirley_barrier_t * b) {
    while (atomic_load(&b->threads_done) < b->nth) {
        /* spin */
    }
}

#endif
