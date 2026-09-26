/**
 * profile.cu - Opt-in per-region CUDA timing (plan F0). See csrc/include/profile.h.
 *
 * The ring is fixed-size and never synchronizes while recording: a scope records two events
 * on the caller's stream and advances an index. When the ring is full, further scopes are
 * skipped and `profile_scope_overflowed` says so, which is the honest alternative to a
 * mid-request synchronization that would change what is being measured.
 */
#include "profile.h"

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>

namespace {

struct ScopeEntry {
    const char *region;   /* a static literal from the call site */
    cudaEvent_t start;
    cudaEvent_t end;
    int depth;            /* nesting depth at entry; 1 is an outermost scope */
    int device;           /* the device the events were created and recorded on */
};

/* An event belongs to a device, and `cudaEventElapsedTime` only accepts events of the
 * current device, so a pool that outlives a device switch has to be re-created on the new
 * device. A two-device layer split switches device once per layer, which is a few hundred
 * event creations per forward - acceptable for a profiler and invisible when it is off. */
cudaEvent_t g_events[2 * PROFILE_MAX_SCOPES];
/* -1 means "this slot holds no event yet"; a slot is only usable when both of its events were
 * created on the device the scope is running on. (An earlier version zero-initialised this and
 * treated a zero as device 0, so on a device-0 scope the pool looked ready while holding NULL
 * handles: the failed record poisoned the stream and aborted the forward, which is how the
 * two-device profile came back with three scopes and an "invalid resource handle".) */
int g_event_device[2 * PROFILE_MAX_SCOPES];

thread_local char g_error[256];

int g_enabled = 0;
int g_count = 0;
int g_overflowed = 0;
cudaStream_t g_stream = nullptr;  /* the last stream a scope ran on, for the report */
ScopeEntry g_scopes[PROFILE_MAX_SCOPES];
int g_stack[PROFILE_MAX_SCOPES];
int g_depth = 0;
int g_unreadable = 0;   /* scopes whose events could not be read back */

void fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
}

/* Make the two events of `slot` belong to `device`, re-creating them if they belonged to
 * another one. Returns false (with the error text set) if they could not be created. */
bool events_for_slot(int slot, int device) {
    if (g_event_device[2 * slot] == device && g_event_device[2 * slot + 1] == device) return true;
    for (int k = 0; k < 2; ++k) {
        if (g_event_device[2 * slot + k] >= 0) {
            cudaSetDevice(g_event_device[2 * slot + k]);
            cudaEventDestroy(g_events[2 * slot + k]);
        }
        cudaSetDevice(device);
        if (cudaEventCreateWithFlags(&g_events[2 * slot + k], cudaEventDefault) != cudaSuccess) {
            fail("profile: could not create an event on device %d: %s", device,
                 cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        g_event_device[2 * slot + k] = device;
    }
    return true;
}

}  // namespace

extern "C" {

void profile_set_enabled(int enabled) {
    if (enabled && !g_enabled) {
        for (int i = 0; i < 2 * PROFILE_MAX_SCOPES; ++i) g_event_device[i] = -1;
        g_enabled = 1;
        profile_reset();
    } else if (!enabled && g_enabled) {
        g_enabled = 0;
        for (int i = 0; i < 2 * PROFILE_MAX_SCOPES; ++i) {
            if (g_event_device[i] >= 0) {
                cudaSetDevice(g_event_device[i]);
                cudaEventDestroy(g_events[i]);
                g_event_device[i] = -1;
            }
        }
        g_count = 0;
        g_overflowed = 0;
        g_depth = 0;
    }
}

int profile_enabled(void) { return g_enabled; }

void profile_reset(void) {
    g_count = 0;
    g_overflowed = 0;
    g_depth = 0;
    g_unreadable = 0;
}

int profile_scope_count(void) { return g_count; }
int profile_scope_overflowed(void) { return g_overflowed; }

void profile_scope_begin(const char *region, void *stream) {
    if (!g_enabled) return;
    if (g_count >= PROFILE_MAX_SCOPES) {
        g_overflowed = 1;
        return;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    const int index = g_count;
    if (!events_for_slot(index, device)) return;
    ScopeEntry &entry = g_scopes[index];
    entry.region = region;
    entry.start = g_events[2 * index];
    entry.end = g_events[2 * index + 1];
    entry.depth = g_depth + 1;
    entry.device = device;
    ++g_count;
    g_stream = static_cast<cudaStream_t>(stream);
    cudaEventRecord(entry.start, g_stream);
    g_stack[g_depth] = index;
    ++g_depth;
}

void profile_scope_end(const char *, void *stream) {
    if (!g_enabled) return;
    if (g_depth <= 0) return;  /* an unbalanced end (a skipped begin) is a no-op */
    --g_depth;
    const int index = g_stack[g_depth];
    ScopeEntry &entry = g_scopes[index];
    cudaEventRecord(entry.end, static_cast<cudaStream_t>(stream));
}

int profile_report(char *buf, int len) {
    if (buf == nullptr || len <= 0) {
        fail("profile_report: no buffer");
        return 0;
    }
    /* The one measurement boundary: everything recorded has been enqueued, so waiting on the
     * last stream makes every elapsed_time below well defined. */
    if (g_stream != nullptr) cudaStreamSynchronize(g_stream);

    int written = 0;
    auto emit = [&](const char *fmt, ...) {
        const int room = len - written;
        va_list ap;
        va_start(ap, fmt);
        const int need = vsnprintf(buf + written, room > 0 ? (size_t)room : 0, fmt, ap);
        va_end(ap);
        if (need > 0) written += need;
    };

    /* Sum per name, in first-seen order, so the report is stable across runs. */
    const char *names[PROFILE_MAX_SCOPES];
    long long counts[PROFILE_MAX_SCOPES];
    float totals[PROFILE_MAX_SCOPES];
    float maxes[PROFILE_MAX_SCOPES];
    int depths_seen[PROFILE_MAX_SCOPES];
    int distinct = 0;
    for (int i = 0; i < g_count; ++i) {
        const ScopeEntry &entry = g_scopes[i];
        /* The events belong to the device the scope ran on, and elapsed_time only accepts
         * events of the current device. */
        cudaSetDevice(entry.device);
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, entry.start, entry.end) != cudaSuccess) {
            ms = 0.0f;
            g_unreadable += 1;
        }
        int slot = -1;
        for (int j = 0; j < distinct; ++j) {
            if (strcmp(names[j], entry.region) == 0) {
                slot = j;
                break;
            }
        }
        if (slot < 0) {
            slot = distinct++;
            names[slot] = entry.region;
            counts[slot] = 0;
            totals[slot] = 0.0f;
            maxes[slot] = 0.0f;
            depths_seen[slot] = entry.depth;
        }
        ++counts[slot];
        totals[slot] += ms;
        if (ms > maxes[slot]) maxes[slot] = ms;
    }

    emit("region count total_ms mean_ms max_ms depth\n");
    for (int j = 0; j < distinct; ++j) {
        emit("%s %lld %.6f %.6f %.6f %d\n", names[j], counts[j], (double)totals[j],
             (double)totals[j] / (double)(counts[j] > 0 ? counts[j] : 1), (double)maxes[j],
             depths_seen[j]);
    }
    if (g_overflowed) emit("# overflowed: the pool filled before the request ended\n");
    if (g_unreadable) emit("# %d scope(s) could not be read back\n", g_unreadable);
    if (g_unreadable == g_count && g_count > 0) {
        fail("profile: every scope's events were unreadable: %s",
             cudaGetErrorString(cudaGetLastError()));
    }
    return written;
}

const char *profile_last_error(void) { return g_error; }

}  // extern "C"
