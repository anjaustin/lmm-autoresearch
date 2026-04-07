/*
 * shirley_barrier.h — Thread barriers for Shirley ops
 *
 * Pure _mm_pause() spin-wait. No sched_yield(), no futex.
 *
 * _mm_pause() is the correct primitive here:
 *   - Tells CPU to deprioritize this thread's execution resources
 *   - Reduces power ~10× per spin iteration vs bare spin
 *   - Prevents speculative memory order violations (pipeline flush)
 *   - Does NOT give up the time slice (unlike sched_yield)
 *
 * On loaded systems, sched_yield() causes 1-10ms reschedule delays
 * that compound across 300+ barriers per token. _mm_pause() keeps
 * the thread on-core with minimal cache pollution.
 *
 * Monotonic phase counter: never resets within a session.
 * Overflows after ~10M tokens (INT_MAX / 210). Acceptable for
 * interactive inference. Server deployments should reset between
 * sessions (clear kv_pos resets the call count).
 */

#ifndef SHIRLEY_BARRIER_H
#define SHIRLEY_BARRIER_H

#include <stdint.h>
#include <immintrin.h>

/* ---- Phase gate: thread 0 does sequential work, others wait ---- */

static inline void shirley_phase_wait(volatile int *phase, int target) {
    while (__atomic_load_n(phase, __ATOMIC_ACQUIRE) < target) {
        _mm_pause();
    }
}

static inline void shirley_phase_set(volatile int *phase, int value) {
    __atomic_store_n(phase, value, __ATOMIC_RELEASE);
}

/* ---- Count barrier: all threads arrive, thread 0 waits + resets ---- */

static inline void shirley_barrier_arrive(volatile int *counter) {
    __atomic_fetch_add(counter, 1, __ATOMIC_ACQ_REL);
}

static inline void shirley_barrier_wait_reset(volatile int *counter, int nth) {
    while (__atomic_load_n(counter, __ATOMIC_ACQUIRE) < nth) {
        _mm_pause();
    }
    __atomic_store_n(counter, 0, __ATOMIC_RELEASE);
}

#endif /* SHIRLEY_BARRIER_H */
