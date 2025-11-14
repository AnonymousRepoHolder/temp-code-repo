// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
#include "peak_memory_tracker.h"
#include <iostream>

PeakMemoryTracker::PeakMemoryTracker(int sample_interval_ms)
    : sample_interval_ms_(sample_interval_ms),
      monitoring_(false),
      baseline_mb_(0),
      peak_mb_(0) {
}

PeakMemoryTracker::~PeakMemoryTracker() {
    if (monitoring_.load()) {
        stop();
    }
}

void PeakMemoryTracker::start() {
    // Record baseline memory before starting monitoring
    auto baseline_opt = mem_monitor_.rss_mb();
    baseline_mb_ = baseline_opt.value_or(0);
    peak_mb_.store(baseline_mb_);

    // Start monitoring flag
    monitoring_.store(true);

    // Launch background monitoring thread
    monitor_thread_ = std::thread([this]() {
        MemoryMonitor local_monitor;
        while (monitoring_.load()) {
            auto current = local_monitor.rss_mb();
            if (current.has_value()) {
                // Lock-free CAS loop to update peak
                double old_peak = peak_mb_.load();
                while (*current > old_peak &&
                       !peak_mb_.compare_exchange_weak(old_peak, *current)) {
                    // CAS failed, retry with updated old_peak
                }
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(sample_interval_ms_));
        }
    });
}

void PeakMemoryTracker::stop() {
    if (monitoring_.load()) {
        monitoring_.store(false);
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }
}

double PeakMemoryTracker::get_baseline_mb() const {
    return baseline_mb_;
}

double PeakMemoryTracker::get_peak_mb() const {
    return peak_mb_.load();
}

double PeakMemoryTracker::get_overhead_mb() const {
    return peak_mb_.load() - baseline_mb_;
}
