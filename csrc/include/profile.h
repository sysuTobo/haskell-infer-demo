/**
 * profile.h - Opt-in per-region CUDA timing (plan F0).
 *
 * docs/plan-numeric-contract.md's F0 asks for "per-region CUDA time, launch count, host
 * synchronization and memory traffic for decode M=1 and representative prefill M=2/64/128"
 * *before* any fusion is chosen, and it is explicit about how: "Run timing with taps
 * disabled, after warm-up, and synchronize only at measurement boundaries; keep diagnostic
 * captures separate."
 *
 * That is what this is, and the shape follows from those two sentences:
 *
 *   * **off by default.** The gates and the golden captures run with it disabled, so the
 *     timed path never changes what they measure. `profile_set_enabled(1)` allocates the
 *     event pool, `profile_set_enabled(0)` frees it.
 *   * **one region invocation, one event pair.** A scope records a start and an end event on
 *     the caller's stream; nothing synchronizes while recording, so a request runs at its
 *     own pace. Nesting is supported (the dispatcher's `mixer.*` wraps the attention layer's
 *     `attn.*` scopes), and the reported total for a name is that name's own sum, not a
 *     roll-up.
 *   * **one synchronization per report.** `profile_report` is the measurement boundary: it
 *     synchronizes the stream the scopes ran on and then sums. Calling it per measured call
 *     (which is what the benchmark runner does) keeps the pool small and the arithmetic
 *     clean, and a *request*-level total is then the runner's sum over its own call records,
 *     not this facility's.
 *
 * What it does **not** measure, because it cannot without a profiler: memory traffic and
 * host-synchronization counts per region. F0's numbers are therefore time and launch count,
 * and the plan's remaining two columns stay recorded as not measured rather than estimated.
 */
#ifndef HASKELL_INFER_PROFILE_H
#define HASKELL_INFER_PROFILE_H

#ifdef __cplusplus
extern "C" {
#endif

/* One forward of a 64-layer hybrid layer walk is well under this; the pool is sized for the
 * deepest model this repository loads plus headroom, and a request that would exceed it stops
 * recording rather than dropping into a mid-request synchronization. */
#define PROFILE_MAX_SCOPES 4096

/* A region name is a static string literal at the call site. */
#define PROFILE_NAME_MAX 48

/* Enable or disable recording. Enabling allocates the event pool and clears the ring;
 * disabling frees it. Idempotent, and disabled again by process exit. */
void profile_set_enabled(int enabled);
int profile_enabled(void);

/* Clear the ring without touching the events, for a caller that reports per invocation. */
void profile_reset(void);

/* Open and close one region scope. `stream` is the CUDA stream as an opaque pointer so this
 * header stays free of CUDA types; both calls are no-ops while recording is disabled. */
void profile_scope_begin(const char *region, void *stream);
void profile_scope_end(const char *region, void *stream);

/* How many scopes the last reset's recording captured, and whether the pool overflowed. */
int profile_scope_count(void);
int profile_scope_overflowed(void);

/* Write a text report (one line per region: name, count, total_ms, mean_ms, max_ms, depth)
 * into `buf` and return the number of characters the whole report needs. Synchronizes the
 * recorded stream first. */
int profile_report(char *buf, int len);

const char *profile_last_error(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/* A scope guard, so a call site reads as one line and the early-return paths in the kernels
 * cannot leave a scope open. */
class ProfileScope {
public:
    ProfileScope(const char *region, void *stream)
        : opened_(profile_enabled() != 0), region_(region), stream_(stream) {
        if (opened_) profile_scope_begin(region_, stream_);
    }
    ~ProfileScope() {
        if (opened_) profile_scope_end(region_, stream_);
    }
    ProfileScope(const ProfileScope &) = delete;
    ProfileScope &operator=(const ProfileScope &) = delete;

private:
    bool opened_;
    const char *region_;
    void *stream_;
};

/* One line, and the name is a literal so no allocation is involved. */
#define PROFILE_SCOPE(region, stream) \
    ProfileScope profile_scope_guard_##__LINE__(region, (void *)(stream))

#endif /* __cplusplus */

#endif /* HASKELL_INFER_PROFILE_H */
