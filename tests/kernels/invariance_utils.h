/*
 * invariance_utils.h - shared instrumentation for the Stage-2 experiments
 * (docs/plan-numeric-contract.md, claims B, C and D).
 *
 * These are *experiments*, so the shape is different from the Stage-1 gate:
 * every measurement is printed with the arm that produced it, and a claim's
 * headline decision is asserted at the end (bitwise for a claim that holds, a
 * documented bound for one that is falsified and quantified). A measurement that
 * came out bitwise is counted, so "how invariant is this, really" is a number and
 * not an impression.
 *
 * `differing` counts elements that are not byte-identical, `max_abs`/`rms` are in
 * the output's own units. A nonfinite element fails immediately: a NaN must never
 * be averaged into a bound.
 */
#ifndef HASKELL_INFER_INVARIANCE_UTILS_H
#define HASKELL_INFER_INVARIANCE_UTILS_H

#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace invariance {

using test::Bf16;

struct Delta {
    double max_abs = 0.0;
    double rms = 0.0;
    size_t differing = 0;
    size_t total = 0;
    bool finite = true;
    bool bitwise() const { return finite && differing == 0; }
};

template <typename T>
Delta delta(const std::vector<T> &a, const std::vector<T> &b) {
    Delta d;
    d.total = a.size();
    if (a.size() != b.size()) {
        d.finite = false;
        d.max_abs = std::numeric_limits<double>::infinity();
        return d;
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = test::value(a[i]), y = test::value(b[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            d.finite = false;
            d.differing = a.size();
            d.max_abs = std::numeric_limits<double>::infinity();
            return d;
        }
        // Compare the stored bits, not the arithmetic values: -0.0 == 0.0 but the
        // two are not the same bytes, and this is a bitwise comparison.
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) ++d.differing;
        const double error = std::abs(x - y);
        d.max_abs = std::max(d.max_abs, error);
        sum += error * error;
    }
    d.rms = a.empty() ? 0.0 : std::sqrt(sum / a.size());
    return d;
}

/* One line per measurement plus a running summary. The line is the recorded
 * evidence; the summary is what the claim's decision is made from. */
class Recorder {
public:
    explicit Recorder(const char *claim) : claim_(claim) {}

    /* `peak` is the magnitude of the reference output for this arm, when the
     * caller knows it. It turns an absolute bound into a relative one, which is
     * what makes "one bf16 ULP" or "fp32 accumulation" readable in the summary
     * instead of leaving a bare number. */
    void record(const std::string &label, const Delta &d, const char *gate, double peak = 0.0) {
        ++measurements_;
        if (d.bitwise()) ++bitwise_;
        if (!d.finite) ++nonfinite_;
        if (d.max_abs > worst_) {
            worst_ = d.max_abs;
            worst_label_ = label;
        }
        if (peak > 0.0) {
            const double rel = d.max_abs / peak;
            if (rel > worst_rel_) {
                worst_rel_ = rel;
                worst_rel_label_ = label;
            }
        }
        std::printf("invariance: %-2s %-8s %-46s max_abs=%.9g rms=%.9g differing=%zu/%zu\n",
                    claim_, gate, label.c_str(), d.max_abs, d.rms, d.differing, d.total);
    }

    int measurements() const { return measurements_; }
    int bitwise() const { return bitwise_; }
    int nonfinite() const { return nonfinite_; }
    double worst() const { return worst_; }
    double worst_rel() const { return worst_rel_; }
    const std::string &worst_label() const { return worst_label_; }

    void summary() const {
        std::printf("invariance: claim %s: %d measurements, %d bitwise, worst max_abs=%.9g (%s)\n",
                    claim_, measurements_, bitwise_, worst_,
                    worst_label_.empty() ? "none" : worst_label_.c_str());
        if (worst_rel_ > 0.0) {
            std::printf("invariance: claim %s: worst max_abs / reference peak = %.3e (%s)\n",
                        claim_, worst_rel_,
                        worst_rel_label_.empty() ? "none" : worst_rel_label_.c_str());
        }
    }

private:
    const char *claim_;
    int measurements_ = 0;
    int bitwise_ = 0;
    int nonfinite_ = 0;
    double worst_ = 0.0;
    double worst_rel_ = 0.0;
    std::string worst_label_;
    std::string worst_rel_label_;
};

}  // namespace invariance

#endif
